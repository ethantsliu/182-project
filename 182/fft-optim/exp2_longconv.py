"""Experiment 2: FFT-Adam for long-convolution kernels.

kill  (CPU): system identification y = h* (*) x + noise, L = 1024, smooth h*, AR(1) inputs.
       GO if the best FFT-Adam variant reaches Adam's final excess loss in <= half the steps.
main  (GPU, only after GO or FORCE_MAIN=1): a Fu-et-al.-style long-conv model on sequential CIFAR-10
       (an in-repo re-implementation, not HazyResearch/safari), optimizer x regularizer {none, squash, smooth}.

FFT-Adam on a kernel h (..., L):  g^ = rfft(grad, norm=ortho);  m <- EMA g^;  v <- EMA |g^|^2
   u = irfft(m^ * gain, n=L, norm=ortho),  gain = 1/(sqrt(v^)+eps)            (p = 0.5)
   partial p < 0.5:  gain = 1/(v^^p * vbar^(0.5-p) + eps), vbar = per-channel mean of v^ over frequency
                     (the extra per-channel scalar keeps Adam's units, so LR grids are comparable)
   cap kappa:        gain <- min(gain, kappa * median_freq(gain))
   shared:           v averaged over channels (one spectrum per kernel tensor)
"""
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fftopt.core import nbytes_state, warmup_linear  # noqa: E402
from fftopt import pool  # noqa: E402

OUT = os.environ.get("FFT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
SMOKE = os.environ.get("SMOKE") == "1"
CIFAR = os.environ.get("CIFAR_PT", "/mnt/work/dev_patel/data/cifar10.pt")


class FFTAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, p=0.5, shared=False, cap=None, decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, p=p, shared=shared, cap=cap, decay=decay, lr_mult=1.0))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                L = p.shape[-1]
                gh = torch.fft.rfft(p.grad.float(), norm="ortho")
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(gh)
                    vshape = list(gh.shape)
                    if g["shared"]:
                        vshape[:-1] = [1] * (len(vshape) - 1)
                    st["v"] = torch.zeros(vshape, device=gh.device)
                st["t"] += 1
                t = st["t"]
                st["m"].lerp_(gh, 1 - b1)
                pw = gh.abs() ** 2
                if g["shared"]:
                    pw = pw.reshape(-1, pw.shape[-1]).mean(0).reshape(st["v"].shape)
                st["v"].mul_(b2).add_(pw, alpha=1 - b2)
                mh = st["m"] / (1 - b1 ** t)
                vh = st["v"] / (1 - b2 ** t)
                if g["p"] == 0.5:
                    gain = 1.0 / (vh.sqrt() + g["eps"])
                else:
                    vbar = vh.mean(-1, keepdim=True)
                    gain = 1.0 / (vh.pow(g["p"]) * vbar.pow(0.5 - g["p"]) + g["eps"])
                if g["cap"]:
                    med = gain.median(dim=-1, keepdim=True).values
                    gain = torch.minimum(gain, g["cap"] * med)
                u = torch.fft.irfft(mh * gain, n=L, norm="ortho")
                if g["decay"]:
                    p.mul_(1 - g["lr_mult"] * g["decay"])
                p.add_(u.to(p.dtype), alpha=-g["lr"] * g["lr_mult"])


OPT_ARMS = {
    "adam": dict(kind="adam"),
    "fftadam": dict(kind="fft", p=0.5),
    "fftadam_p0.25": dict(kind="fft", p=0.25),
    "fftadam_shared": dict(kind="fft", p=0.5, shared=True),
    "fftadam_cap3": dict(kind="fft", p=0.5, cap=3.0),
    "fftadam_cap10": dict(kind="fft", p=0.5, cap=10.0),
}


def make_kernel_opt(params, arm, lr, decay=0.0):
    spec = OPT_ARMS[arm]
    if spec["kind"] == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    return FFTAdam(params, lr=lr, p=spec.get("p", 0.5), shared=spec.get("shared", False), cap=spec.get("cap"), decay=decay)


def roughness(h, L):
    """Fraction of kernel energy above frequency L/8 (rfft bins > L/8)."""
    e = torch.fft.rfft(h.float(), norm="ortho").abs() ** 2
    return (e[..., L // 8 + 1:].sum() / e.sum().clamp_min(1e-30)).item()


def set_mult(opt, mult):
    for g in opt.param_groups:
        g.setdefault("base_lr", g["lr"])
        if "lr_mult" in g:
            g["lr_mult"] = mult
        else:
            g["lr"] = g["base_lr"] * mult


# ----------------------------------------------------------------------------- kill test (CPU)

def sysid_run(arm, lr, seed, steps, L=1024, C=16, B=8, rho=0.95, noise=0.1, eval_every=50):
    torch.set_num_threads(1)
    T = 2 * L
    gen = torch.Generator().manual_seed(1000)  # the system is fixed across seeds/arms
    tau = 30 + 270 * torch.rand(C, 1, generator=gen)
    f = 16 * torch.rand(C, 1, generator=gen)
    ph = 2 * math.pi * torch.rand(C, 1, generator=gen)
    tt = torch.arange(L).float()
    hstar = torch.exp(-tt / tau) * torch.cos(2 * math.pi * f * tt / L + ph)
    hstar = hstar / hstar.norm(dim=-1, keepdim=True)
    nfft = 2 * T

    def conv(x, h):  # x (B,C,T), h (C,L) -> valid outputs (B,C,T-L+1)
        y = torch.fft.irfft(torch.fft.rfft(x, n=nfft) * torch.fft.rfft(h, n=nfft), n=nfft)
        return y[..., L - 1:T]

    def ar1(n, g):
        e = torch.randn(n, C, T, generator=g)
        x = torch.empty_like(e)
        x[..., 0] = e[..., 0] / math.sqrt(1 - rho ** 2)
        # vectorised AR(1) via a scan in blocks would be faster; T=2048 loop is fine on CPU
        for t in range(1, T):
            x[..., t] = rho * x[..., t - 1] + e[..., t]
        return x * math.sqrt(1 - rho ** 2)

    g = torch.Generator().manual_seed(seed)
    ev_x = ar1(32, torch.Generator().manual_seed(999))
    ev_y = conv(ev_x, hstar)
    sig = noise * ev_y.std().item()
    pool_x = ar1(512, g)  # a bank of AR(1) segments; each step draws B of them with random sign/offset
    h = torch.zeros(C, L, requires_grad=True)
    opt = make_kernel_opt([h], arm, lr)
    curve = []
    t0 = time.time()
    for s in range(steps + 1):
        if s % eval_every == 0 or s == steps:
            with torch.no_grad():
                ex = F.mse_loss(conv(ev_x, h), ev_y).item()  # excess loss (noise-free target)
            curve.append([s, ex])
        if s == steps:
            break
        set_mult(opt, warmup_linear(s, steps))
        idx = torch.randint(0, pool_x.shape[0], (B,), generator=g)
        x = pool_x[idx] * (torch.randint(0, 2, (B, 1, 1), generator=g) * 2 - 1)
        y = conv(x, hstar) + sig * torch.randn(B, C, T - L + 1, generator=g)
        loss = F.mse_loss(conv(x, h), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    fin = curve[-1][1]
    return dict(arm=arm, lr=lr, seed=seed, final=fin if math.isfinite(fin) else None, curve=curve,
                ms_per_step=1000 * (time.time() - t0) / steps, opt_state_mb=nbytes_state(opt) / 2 ** 20,
                rough=roughness(h.detach(), L), rough_hstar=roughness(hstar, L))


def _sysid_job(a):
    return sysid_run(**a)


def kill():
    import multiprocessing as mp
    steps = 300 if SMOKE else 3000
    d = os.path.join(OUT, "exp2_kill")
    os.makedirs(d, exist_ok=True)
    centers = {a: 1e-3 for a in OPT_ARMS}
    seeds = (0, 1, 2)
    ctx = mp.get_context("spawn")
    res = {a: {} for a in OPT_ARMS}
    grid = {a: [centers[a] * 3 ** k for k in (-2, -1, 0, 1, 2)] for a in OPT_ARMS}
    with ctx.Pool(min(64, os.cpu_count())) as P:
        jobs = [dict(arm=a, lr=lr, seed=0, steps=steps) for a in OPT_ARMS for lr in grid[a]]
        for r in P.map(_sysid_job, jobs):
            res[r["arm"]][pool.lr_tag(r["lr"])] = r
        best = {a: max(grid[a], key=lambda lr: pool._score(res[a][pool.lr_tag(lr)], False)) for a in OPT_ARMS}
        # one edge extension
        ext = []
        for a in OPT_ARMS:
            if best[a] == min(grid[a]):
                ext.append(dict(arm=a, lr=best[a] / 3, seed=0, steps=steps))
            elif best[a] == max(grid[a]):
                ext.append(dict(arm=a, lr=best[a] * 3, seed=0, steps=steps))
        for r in P.map(_sysid_job, ext):
            res[r["arm"]][pool.lr_tag(r["lr"])] = r
            grid[r["arm"]].append(r["lr"])
        best = {a: max(grid[a], key=lambda lr: pool._score(res[a][pool.lr_tag(lr)], False)) for a in OPT_ARMS}
        seeded = P.map(_sysid_job, [dict(arm=a, lr=best[a], seed=s, steps=steps) for a in OPT_ARMS for s in seeds[1:]])
    results = {a: {"best_lr": best[a], "grid": res[a],
                   "seeds": [res[a][pool.lr_tag(best[a])]] + [r for r in seeded if r["arm"] == a]} for a in OPT_ARMS}
    adam_final = statistics.median(s["final"] for s in results["adam"]["seeds"] if s["final"] is not None)
    lines = ["## Experiment 2 kill test: system identification (CPU)", "",
             f"L=1024, 16 channels, AR(1) rho=0.95 inputs, noise 10% of output std, {steps} steps, batch 8, "
             "warmup 5% + linear decay, LR grid ratio 3 (5 points + 1 edge extension), 3 seeds at the best LR.",
             f"Metric: excess eval MSE vs the noise-free output. Adam's median final = {adam_final:.4g}.", "",
             "| arm | best lr | final excess (median of 3) | steps to Adam's final (median) | ≤ half? | roughness >L/8 | ms/step | opt MB |",
             "|---|---|---|---|---|---|---|---|"]
    go = False
    for a, r in results.items():
        fins = [s["final"] for s in r["seeds"] if s["final"] is not None]
        hits = [pool.first_hit(s["curve"], adam_final, False) for s in r["seeds"]]
        hits_ok = [h for h in hits if h is not None]
        hm = statistics.median(hits_ok) if len(hits_ok) == len(hits) else None
        half = hm is not None and hm <= steps / 2
        if a != "adam" and half:
            go = True
        lines.append(f"| {a} | {r['best_lr']:.3g} | {statistics.median(fins) if fins else float('nan'):.4g} | {hm} | "
                     f"{'yes' if half else 'no'} | {statistics.median(s['rough'] for s in r['seeds']):.3g} | "
                     f"{statistics.median(s['ms_per_step'] for s in r['seeds']):.2f} | {r['seeds'][0]['opt_state_mb']:.2f} |")
    lines.append("")
    lines.append(f"h* roughness (energy above L/8): {results['adam']['seeds'][0]['rough_hstar']:.3g}")
    lines.append("")
    lines.append(f"**Verdict: {'GO' if go else 'NO-GO'}** (GO = some FFT-Adam variant reaches Adam's final loss in ≤ {steps // 2} steps).")
    md = "\n".join(lines) + "\n"
    open(os.path.join(d, "summary.md"), "w").write(md)
    json.dump({"go": go, "results": results}, open(os.path.join(d, "results.json"), "w"))
    print(md)
    return go


# ----------------------------------------------------------------------------- main (GPU): sequential CIFAR

class LongConv(nn.Module):
    def __init__(self, d, L, reg, lam=1e-3, smooth=5):
        super().__init__()
        self.k = nn.Parameter(torch.randn(d, L) * 0.002)
        self.D = nn.Parameter(torch.randn(d))
        self.reg, self.lam, self.smooth, self.L = reg, lam, smooth, L

    def kernel(self):
        k = self.k
        if self.reg == "squash":
            k = F.relu(k.abs() - self.lam) * torch.sign(k)
        elif self.reg == "smooth":
            k = F.avg_pool1d(k[:, None], self.smooth, stride=1, padding=self.smooth // 2, count_include_pad=False)[:, 0]
        return k

    def forward(self, x):  # x (B, L, d)
        u = x.transpose(1, 2).float()
        n = 2 * self.L
        y = torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(self.kernel().float(), n=n), n=n)[..., :self.L]
        return (y + u * self.D[:, None]).transpose(1, 2).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, L, reg, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.conv = LongConv(d, L, reg)
        self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop(self.o(F.gelu(self.conv(self.n1(x)))))
        return x + self.drop(self.mlp(self.n2(x)))


class SeqNet(nn.Module):
    def __init__(self, d=128, layers=4, L=1024, reg="none"):
        super().__init__()
        self.inp = nn.Linear(3, d)
        self.blocks = nn.ModuleList(Block(d, L, reg) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x).mean(1))


def main_job(a):
    torch.manual_seed(a["seed"])
    dev = "cuda"
    data = torch.load(CIFAR)
    mean = data["train_x"].float().mean((0, 1, 2)) / 255
    std = data["train_x"].float().std((0, 1, 2)) / 255

    def prep(x):
        return ((x.to(dev).float() / 255 - mean.to(dev)) / std.to(dev)).reshape(x.shape[0], 1024, 3)

    Xtr, Ytr = prep(data["train_x"]), data["train_y"].to(dev)
    Xte, Yte = prep(data["test_x"]), data["test_y"].to(dev)
    if SMOKE:
        Xtr, Ytr, Xte, Yte = Xtr[:2048], Ytr[:2048], Xte[:1024], Yte[:1024]
    model = SeqNet(reg=a["reg"]).to(dev)
    kern = [b.conv.k for b in model.blocks]
    kid = {id(p) for p in kern}
    rest = [p for p in model.parameters() if id(p) not in kid]
    bs, epochs = 128, (1 if SMOKE else a.get("epochs", 8))
    spe = Xtr.shape[0] // bs
    total = spe * epochs
    kopt = make_kernel_opt(kern, a["arm"], a["lr"])
    ropt = torch.optim.AdamW(rest, lr=1e-3, weight_decay=0.05, fused=True)
    curve, ttrain, step = [], 0.0, 0

    def evaluate():
        model.eval()
        correct = 0
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for i in range(0, Xte.shape[0], 500):
                correct += (model(Xte[i:i + 500]).argmax(-1) == Yte[i:i + 500]).sum().item()
        model.train()
        return correct / Xte.shape[0]

    diverged = False
    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            mult = warmup_linear(step, total)
            set_mult(kopt, mult)
            set_mult(ropt, mult)
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            kopt.zero_grad(set_to_none=True)
            ropt.zero_grad(set_to_none=True)
            loss.backward()
            kopt.step()
            ropt.step()
            step += 1
        torch.cuda.synchronize()
        ttrain += time.time() - t0
        if not math.isfinite(loss.item()):
            diverged = True
            break
        curve.append([step, evaluate()])
    rough = statistics.mean(roughness(b.conv.kernel().detach(), 1024) for b in model.blocks)
    res = dict(a, final=None if diverged else curve[-1][1], curve=curve, ms_per_step=1000 * ttrain / max(step, 1),
               opt_state_mb=nbytes_state(kopt) / 2 ** 20, rough=rough, steps=step)
    json.dump(res, open(a["out"], "w"))


def main():
    d = os.path.join(OUT, "exp2_main")
    arms = {a: dict(lr=1e-3, reg="none") for a in OPT_ARMS}
    res = pool.sweep(os.path.abspath(__file__), arms, os.path.join(d, "reg_none"), higher=True)
    jobs = []
    for a in OPT_ARMS:
        for reg in ("squash", "smooth"):
            for s in (0, 1, 2):
                jobs.append(dict(arm=a, lr=res[a]["best_lr"], seed=s, reg=reg,
                                 out=os.path.join(d, f"reg_{reg}", a, f"s{s}.json")))
    regres = pool.run_jobs(os.path.abspath(__file__), jobs)
    md, _ = pool.summarize(res, True, "test acc", "Experiment 2 main: sCIFAR long-conv, reg = none (LR tuned here)")
    lines = [md, "### Optimizer × regularizer at each optimizer's reg=none best LR (test acc mean ± std over 3 seeds; kernel roughness >L/8)", "",
             "| arm | none | squash | smooth |", "|---|---|---|---|"]
    for a in OPT_ARMS:
        cells = []
        for reg in ("none", "squash", "smooth"):
            rs = res[a]["seeds"] if reg == "none" else [r for r in regres if r["arm"] == a and r["reg"] == reg]
            f = [r["final"] for r in rs if r.get("final") is not None]
            ro = [r["rough"] for r in rs if r.get("rough") is not None]
            cells.append(f"{statistics.mean(f):.4f} ± {statistics.stdev(f) if len(f) > 1 else 0:.4f} (n={len(f)}), rough {statistics.mean(ro):.3g}" if f else "failed")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(d, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "job":
        main_job(json.loads(sys.argv[2]))
    elif cmd == "kill":
        kill()
    elif cmd == "main":
        main()
    elif cmd == "all":
        go = kill()
        if go or os.environ.get("FORCE_MAIN") == "1":
            main()
        else:
            print("exp2: kill test NO-GO; main run skipped (FORCE_MAIN=1 to override)")
