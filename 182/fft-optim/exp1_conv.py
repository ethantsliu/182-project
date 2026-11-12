"""Experiment 1: per-frequency Shampoo / SOAP for conv kernels (Shampoo-C, SOAP-C) on ResNet-18 / CIFAR-10.

Views of a kernel K (Cout, Cin, kh, kw):
  freq     rfft2(K, s=(kh, kw), norm=ortho) -> one complex (Cout, Cin) matrix per stored frequency
  offsets  one real (Cout, Cin) matrix per spatial offset; with shared factors this is the tied/KFC control
           L = sum_d G_d G_d^T, R = sum_d G_d^T G_d
  flat     (Cout, Cin*kh*kw), column-blocked to <=1152 for Shampoo/SOAP (Muon uses it unblocked)
The new optimizers touch 3x3 convs only (incl. the stem); 1x1 shortcuts, BN and fc use AdamW in every run.

kill: AdamW for 5 epochs, accumulating L_w, R_w (and the tied pair) over every step (~485). At the end,
      cos(per-frequency update, tied update) on each conv using Adam's momentum. GO if the median layer's
      cos < 0.9; NO-GO if > 0.97 on every layer; otherwise GRAY (main still runs, flagged).
main: AdamW, SGD, Muon, Muon-C, Shampoo, SOAP, tied, Shampoo-C, SOAP-C on the 3x3 convs, LR grid ratio 3
      (+1 edge extension), 3 seeds at the best LR. CIFAR-10, 12 epochs, batch 512 (scaled from the plan).
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
from fftopt.core import MatrixOpt, View, blocked_flat_view, inv_root, nbytes_state, warmup_linear  # noqa: E402
from fftopt import pool  # noqa: E402

OUT = os.environ.get("FFT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
SMOKE = os.environ.get("SMOKE") == "1"
DATA = os.environ.get("FFT_DATA", "/mnt/work/dev_patel/data")
DECAY = 5e-5  # absolute per-step decoupled decay on target params (= AdamW lr 1e-3 x wd 0.05), times the schedule


# ----------------------------------------------------------------------------- model / data

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class ResNet18(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, cin = [], 64
        for cout, s in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [BasicBlock(cin, cout, s), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, nc)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def load_cifar(name, dev):
    d = torch.load(os.path.join(DATA, f"{name}.pt"))
    x = d["train_x"].to(dev).permute(0, 3, 1, 2).float() / 255
    mean, std = x.mean((0, 2, 3), keepdim=True), x.std((0, 2, 3), keepdim=True)
    xt = d["test_x"].to(dev).permute(0, 3, 1, 2).float() / 255
    return (x - mean) / std, d["train_y"].to(dev), (xt - mean) / std, d["test_y"].to(dev)


def augment(x):
    B = x.shape[0]
    p = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (B,), device=x.device)
    ox = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (oy[:, None] + ar)[:, :, None]
    cols = (ox[:, None] + ar)[:, None, :]
    out = p.permute(0, 2, 3, 1)[torch.arange(B, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def split_params(model):
    tgt = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d) and m.kernel_size[0] >= 3]
    tid = {id(p) for p in tgt}
    rest = [p for p in model.parameters() if id(p) not in tid]
    return tgt, rest


# ----------------------------------------------------------------------------- views / optimizers

def freq_view(shape):
    co, ci, kh, kw = shape
    kw2 = kw // 2 + 1

    def fwd(x):
        return torch.fft.rfft2(x.float(), s=(kh, kw), norm="ortho").permute(2, 3, 0, 1).reshape(kh * kw2, co, ci)

    def bwd(Y):
        return torch.fft.irfft2(Y.reshape(kh, kw2, co, ci).permute(2, 3, 0, 1), s=(kh, kw), norm="ortho")

    return View(fwd, bwd)


def offsets_view(shape):
    co, ci, kh, kw = shape
    return View(lambda x: x.permute(2, 3, 0, 1).reshape(kh * kw, co, ci),
                lambda Y: Y.reshape(kh, kw, co, ci).permute(2, 3, 0, 1))


def flat_view(shape, blocked):
    co = shape[0]
    if blocked:
        return blocked_flat_view(shape, 1152, lambda x: x.reshape(co, -1), lambda X: X.reshape(shape))
    return View(lambda x: x.reshape(co, -1)[None], lambda Y: Y[0].reshape(shape))


ARMS = {  # name: (kind, view, center lr)
    "adamw": ("adamw", None, 1e-3),
    "sgd": ("sgd", None, 0.1),
    "muon": ("muon", "flat", 0.02),
    "muon_c": ("muon", "freq", 0.02),
    "shampoo": ("shampoo", "flat_blocked", 1e-3),
    "soap": ("soap", "flat_blocked", 3e-3),
    "tied": ("shampoo_shared", "offsets", 1e-3),
    "shampoo_c": ("shampoo", "freq", 1e-3),
    "soap_c": ("soap", "freq", 3e-3),
}


class SGDDecay(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr, momentum=0.9, nesterov=True)
        for g in self.param_groups:
            g["base_lr"] = lr

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                p.mul_(1 - DECAY * g["lr"] / g["base_lr"])
        super().step()


def make_target_opt(arm, params, lr):
    kind, view, _ = ARMS[arm]
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=DECAY / lr, fused=True)
    if kind == "sgd":
        return SGDDecay(params, lr)
    groups = []
    for p in params:
        v = {"freq": freq_view, "offsets": offsets_view}.get(view)
        v = v(p.shape) if v else flat_view(p.shape, view == "flat_blocked")
        groups.append(dict(params=[p], view=v))
    return MatrixOpt(groups, lr=lr, mode=kind, decay=DECAY, precond_every=20)


def set_mult(opts, mult):
    for o in opts:
        for g in o.param_groups:
            g.setdefault("base_lr", g["lr"])
            if "lr_mult" in g:
                g["lr_mult"] = mult
            else:
                g["lr"] = g["base_lr"] * mult


# ----------------------------------------------------------------------------- training

def train(a, collect=None):
    torch.manual_seed(a["seed"])
    dev = "cuda"
    Xtr, Ytr, Xte, Yte = load_cifar(a.get("dataset", "cifar10"), dev)
    if SMOKE:
        Xtr, Ytr, Xte, Yte = Xtr[:4096], Ytr[:4096], Xte[:2000], Yte[:2000]
    model = ResNet18(100 if a.get("dataset") == "cifar100" else 10).to(dev).to(memory_format=torch.channels_last)
    tgt, rest = split_params(model)
    topt = make_target_opt(a["arm"], tgt, a["lr"])
    decay, no_decay = [p for p in rest if p.ndim > 1], [p for p in rest if p.ndim <= 1]
    ropt = torch.optim.AdamW([dict(params=decay, weight_decay=0.05), dict(params=no_decay, weight_decay=0.0)], lr=1e-3, fused=True)
    bs = 512
    epochs = a.get("epochs", 12) if not SMOKE else 2
    spe = Xtr.shape[0] // bs
    total, step, ttrain, curve, diverged = spe * epochs, 0, 0.0, [], False

    def evaluate():
        model.eval()
        c = 0
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for i in range(0, Xte.shape[0], 1000):
                c += (model(Xte[i:i + 1000].contiguous(memory_format=torch.channels_last)).argmax(-1) == Yte[i:i + 1000]).sum().item()
        model.train()
        return c / Xte.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            set_mult([topt, ropt], warmup_linear(step, total))
            x = augment(Xtr[idx]).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(x), Ytr[idx])
            topt.zero_grad(set_to_none=True)
            ropt.zero_grad(set_to_none=True)
            loss.backward()
            if collect is not None:
                collect(tgt)
            topt.step()
            ropt.step()
            step += 1
        torch.cuda.synchronize()
        ttrain += time.time() - t0
        if not math.isfinite(loss.item()):
            diverged = True
            break
        curve.append([step, evaluate()])
    return model, tgt, topt, dict(final=None if diverged else curve[-1][1], curve=curve, steps=step,
                                  ms_per_step=1000 * ttrain / max(step, 1), opt_state_mb=nbytes_state(topt) / 2 ** 20)


def job(a):
    _, _, _, res = train(a)
    json.dump({**a, **res}, open(a["out"], "w"))


def cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


def kill():
    d = os.path.join(OUT, "exp1_kill")
    os.makedirs(d, exist_ok=True)
    stats = {}

    def collect(tgt):
        with torch.no_grad():
            for i, p in enumerate(tgt):
                G = p.grad.float()
                Gf = freq_view(p.shape).fwd(G)
                Go = offsets_view(p.shape).fwd(G)
                s = stats.setdefault(i, dict(Lw=0, Rw=0, Lt=0, Rt=0, n=0))
                s["Lw"] = s["Lw"] + Gf @ Gf.mH
                s["Rw"] = s["Rw"] + Gf.mH @ Gf
                s["Lt"] = s["Lt"] + (Go @ Go.mT).sum(0)
                s["Rt"] = s["Rt"] + (Go.mT @ Go).sum(0)
                s["n"] += 1

    a = dict(arm="adamw", lr=1e-3, seed=0, epochs=5)
    model, tgt, topt, res = train(a, collect)
    rows = []
    for i, p in enumerate(tgt):
        s = stats[i]
        M = topt.state[p]["exp_avg"].float()
        fv = freq_view(p.shape)
        Uc = fv.bwd(inv_root(s["Lw"], 4) @ fv.fwd(M) @ inv_root(s["Rw"], 4))
        ov = offsets_view(p.shape)
        Ut = ov.bwd(inv_root(s["Lt"], 4) @ ov.fwd(M) @ inv_root(s["Rt"], 4))
        Gf = fv.fwd(M)
        dc = (Gf[0].abs() ** 2).sum() / (Gf.abs() ** 2).sum()  # energy share of the DC frequency (unweighted)
        rows.append(dict(layer=i, shape=list(p.shape), cos_c_tied=cos(Uc, Ut), cos_c_m=cos(Uc, M), cos_tied_m=cos(Ut, M),
                         dc_share=dc.item(), steps=s["n"]))
    cs = [r["cos_c_tied"] for r in rows]
    med = statistics.median(cs)
    verdict = "GO" if med < 0.9 else ("NO-GO" if min(cs) > 0.97 else "GRAY")
    lines = ["## Experiment 1 kill test: per-frequency vs tied Shampoo directions (ResNet-18 / CIFAR-10, AdamW 5 epochs)", "",
             f"Statistics summed over {rows[0]['steps']} steps; update = L^-1/4 M R^-1/4 with M = Adam's first moment at the end. "
             f"AdamW run: test acc {res['final']}.", "",
             "| layer | shape | cos(freq, tied) | cos(freq, M) | cos(tied, M) | DC share of M |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['layer']} | {r['shape']} | {r['cos_c_tied']:.4f} | {r['cos_c_m']:.4f} | {r['cos_tied_m']:.4f} | {r['dc_share']:.3f} |")
    lines += ["", f"median cos(freq, tied) = {med:.4f}, min = {min(cs):.4f}, max = {max(cs):.4f}",
              f"**Verdict: {verdict}** (GO if median < 0.9; NO-GO if every layer > 0.97)."]
    md = "\n".join(lines) + "\n"
    open(os.path.join(d, "summary.md"), "w").write(md)
    json.dump(dict(verdict=verdict, rows=rows), open(os.path.join(d, "results.json"), "w"))
    print(md)
    return verdict


def main():
    d = os.path.join(OUT, "exp1_main")
    arms = {n: dict(lr=c, dataset="cifar10") for n, (_, _, c) in ARMS.items()}
    res = pool.sweep(os.path.abspath(__file__), arms, d, higher=True, timeout=2400)
    md, _ = pool.summarize(res, True, "test acc", "Experiment 1 main: ResNet-18 / CIFAR-10, 12 epochs, optimizer on 3x3 convs only")
    open(os.path.join(d, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "job":
        job(json.loads(sys.argv[2]))
    elif cmd == "kill":
        kill()
    elif cmd == "main":
        main()
    elif cmd == "all":
        v = kill()
        if v != "NO-GO" or os.environ.get("FORCE_MAIN") == "1":
            main()
        else:
            print("exp1: kill test NO-GO; main run skipped (FORCE_MAIN=1 to override)")
