"""Experiment 4: does the DCT basis matter for Adam on transformer weights? (Trion follow-up)

rescale:  G~ = G Q;  m, v EMAs of G~;  update = (m^ / (sqrt(v^) + eps)) Q^T          (identity Q = AdamW)
truncate: same, but keep only the top-r columns of the current G~ by norm (r = n/4), Trion-style
bases:    identity, DCT-II (orthonormal), random orthogonal (QR of a Gaussian, drawn per seed)
Q acts on the input dimension of every 2D weight inside the transformer blocks; embeddings, the tied
head, norms and positions use AdamW (fixed LR) in every run.

kill (this is the whole experiment at this scale): a ~30M-param GPT-2-style model (d=384, 6 layers, 6 heads,
vocab 50304, tied) on 300M FineWeb-10B tokens (4578 steps x 65,536 tokens), rescale mode, 3 bases, LR grid
ratio 3 (+1 edge extension), 3 seeds at the best LR. If DCT and random are within seed noise (Welch |t| < 2.78,
n=3 vs 3), run truncate mode for the 3 bases x 3 seeds and stop. Scaling to 124M is flagged only if DCT
clearly beats random; it is not run here.
"""
import glob
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fftopt.core import nbytes_state, warmup_linear  # noqa: E402
from fftopt import pool  # noqa: E402

OUT = os.environ.get("FFT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
SMOKE = os.environ.get("SMOKE") == "1"
FW = os.environ.get("FINEWEB", "/home/dev_patel/optim_proj/data/fineweb10B")
DECAY = 2e-4  # absolute per-step decoupled decay on target weights (= lr 2e-3 x wd 0.1), times the schedule
T, BS, D, NL, NH, V = 1024, 64, 384, 6, 6, 50304
TOKENS = 300_000_000


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, path
    return np.fromfile(path, dtype=np.uint16, offset=256 * 4, count=int(header[2]))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D, bias=False), nn.LayerNorm(D, bias=False)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.fc = nn.Linear(D, 4 * D, bias=False)
        self.proj = nn.Linear(4 * D, D, bias=False)
        nn.init.normal_(self.o.weight, std=0.02 / math.sqrt(2 * NL))
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * NL))

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, NH, D // NH).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D)
        x = x + self.o(y)
        return x + self.proj(F.gelu(self.fc(self.n2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.wpe = nn.Embedding(T, D)
        self.blocks = nn.ModuleList(Block() for _ in range(NL))
        self.nf = nn.LayerNorm(D, bias=False)
        for b in self.blocks:
            nn.init.normal_(b.qkv.weight, std=0.02)
            nn.init.normal_(b.fc.weight, std=0.02)
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.01)

    def forward(self, idx, tgt):
        x = self.wte(idx) + self.wpe.weight[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        logits = F.linear(self.nf(x), self.wte.weight).float()
        return F.cross_entropy(logits.view(-1, V), tgt.reshape(-1))


def dct_matrix(n, dev):
    k = torch.arange(n, device=dev, dtype=torch.float64)
    C = torch.cos(math.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * math.sqrt(2 / n)
    C[0] /= math.sqrt(2)
    return C.T.float().contiguous()  # columns are the DCT basis vectors -> G Q gives DCT coefficients of rows


def basis(kind, n, seed, dev):
    if kind == "identity":
        return None
    if kind == "dct":
        return dct_matrix(n, dev)
    g = torch.Generator(device="cpu").manual_seed(10_000 + seed * 101 + n)
    Qm, R = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return (Qm * torch.sign(torch.diagonal(R))).float().to(dev)


class BasisAdam(torch.optim.Optimizer):
    def __init__(self, params_q, lr, betas=(0.9, 0.95), eps=1e-8, decay=0.0, truncate=None):
        groups = [dict(params=[p], Q=q) for p, q in params_q]
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps, decay=decay, truncate=truncate, lr_mult=1.0))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            Q = g["Q"]
            for p in g["params"]:
                G = p.grad.float()
                Gt = G if Q is None else G @ Q
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(Gt), torch.zeros_like(Gt)
                st["t"] += 1
                t = st["t"]
                st["m"].lerp_(Gt, 1 - b1)
                st["v"].mul_(b2).addcmul_(Gt, Gt, value=1 - b2)
                u = (st["m"] / (1 - b1 ** t)) / ((st["v"] / (1 - b2 ** t)).sqrt() + g["eps"])
                if g["truncate"]:
                    r = max(1, int(g["truncate"] * Gt.shape[1]))
                    keep = Gt.norm(dim=0).topk(r).indices
                    mask = torch.zeros(Gt.shape[1], device=G.device, dtype=u.dtype)
                    mask[keep] = 1
                    u = u * mask
                upd = u if Q is None else u @ Q.T
                p.mul_(1 - g["lr_mult"] * g["decay"])
                p.add_(upd.to(p.dtype), alpha=-g["lr"] * g["lr_mult"])


def job(a):
    torch.manual_seed(a["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    files = sorted(glob.glob(os.path.join(FW, "fineweb_train_*.bin")))[:3]
    train_tok = torch.from_numpy(np.concatenate([load_shard(f) for f in files]).astype(np.int32)).to(dev)
    val_tok = torch.from_numpy(load_shard(os.path.join(FW, "fineweb_val_000000.bin"))[: 4 * 2 ** 20 + 1].astype(np.int32)).to(dev)
    model = GPT().to(dev)
    tgt = [m.weight for b in model.blocks for m in (b.qkv, b.o, b.fc, b.proj)]
    tid = {id(p) for p in tgt}
    rest = [p for p in model.parameters() if id(p) not in tid]
    topt = BasisAdam([(p, basis(a["basis"], p.shape[1], a["seed"], dev)) for p in tgt], lr=a["lr"], decay=DECAY,
                     truncate=0.25 if a.get("mode") == "truncate" else None)
    ropt = torch.optim.AdamW([dict(params=[p for p in rest if p.ndim > 1], weight_decay=0.1),
                              dict(params=[p for p in rest if p.ndim <= 1], weight_decay=0.0)],
                             lr=2e-3, betas=(0.9, 0.95), fused=True)
    cmodel = torch.compile(model) if os.environ.get("NO_COMPILE") != "1" else model
    steps = 200 if SMOKE else TOKENS // (BS * T)
    g = torch.Generator(device=dev).manual_seed(a["seed"])
    ar = torch.arange(T + 1, device=dev)

    def batch():
        s = torch.randint(0, train_tok.numel() - T - 1, (BS,), device=dev, generator=g)
        x = train_tok[s[:, None] + ar]
        return x[:, :-1], x[:, 1:]

    def evaluate():
        model.eval()
        n = (val_tok.numel() - 1) // T
        n = min(n, 64 if SMOKE else n)
        tot = 0.0
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for i in range(0, n, BS):
                j = min(n, i + BS)
                xy = val_tok[i * T: j * T + 1]
                x, y = xy[:-1].view(j - i, T), xy[1:].view(j - i, T)
                tot += cmodel(x, y).item() * (j - i)
        model.train()
        return tot / n

    curve, ttrain, diverged = [], 0.0, False
    every = 50 if SMOKE else 500
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(steps):
        mult = warmup_linear(step, steps, 0.02, floor=0.1)
        for o in (topt, ropt):
            for gr in o.param_groups:
                gr.setdefault("base_lr", gr["lr"])
                if "lr_mult" in gr:
                    gr["lr_mult"] = mult
                else:
                    gr["lr"] = gr["base_lr"] * mult
        x, y = batch()
        with torch.autocast("cuda", torch.bfloat16):
            loss = cmodel(x, y)
        loss.backward()
        topt.step()
        ropt.step()
        topt.zero_grad(set_to_none=True)
        ropt.zero_grad(set_to_none=True)
        if (step + 1) % every == 0 or step + 1 == steps:
            torch.cuda.synchronize()
            ttrain += time.time() - t0
            if not math.isfinite(loss.item()):
                diverged = True
                break
            curve.append([step + 1, evaluate()])
            torch.cuda.synchronize()
            t0 = time.time()
    res = dict(a, final=None if diverged else curve[-1][1], curve=curve, steps=steps,
               ms_per_step=1000 * ttrain / steps, opt_state_mb=nbytes_state(topt) / 2 ** 20,
               n_params=sum(p.numel() for p in model.parameters()))
    json.dump(res, open(a["out"], "w"))


def welch_t(a, b):
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    return (statistics.mean(a) - statistics.mean(b)) / math.sqrt(max(va + vb, 1e-30))


def main():
    d = os.path.join(OUT, "exp4")
    arms = {b: dict(lr=2e-3, basis=b, mode="rescale") for b in ("identity", "dct", "random")}
    res = pool.sweep(os.path.abspath(__file__), arms, os.path.join(d, "rescale"), higher=False, timeout=2400)
    md, rows = pool.summarize(res, False, "val loss", "Experiment 4: rescale mode (Adam in a fixed basis), 30M GPT, 300M tokens")
    fd = [s["final"] for s in res["dct"]["seeds"] if s.get("final") is not None]
    fr = [s["final"] for s in res["random"]["seeds"] if s.get("final") is not None]
    t = welch_t(fd, fr) if len(fd) > 1 and len(fr) > 1 else float("nan")
    within = abs(t) < 2.78
    verdict = ("DCT and random are within seed noise" if within else
               ("DCT clearly beats random -> scale to 124M (not run here)" if t < 0 else "random beats DCT"))
    md += f"\nDCT - random: mean {statistics.mean(fd) - statistics.mean(fr):+.4f} nats, Welch t = {t:.2f} (|t| < 2.78 = within noise at n=3+3)\n\n**Finding: {verdict}.**\n"
    if within:
        jobs = [dict(arm=f"truncate_{b}", basis=b, mode="truncate", lr=res[b]["best_lr"], seed=s,
                     out=os.path.join(d, "truncate", b, f"s{s}.json")) for b in arms for s in (0, 1, 2)]
        tr = pool.run_jobs(os.path.abspath(__file__), jobs, timeout=2400)
        md += "\n### Truncate mode (top 25% of columns of G Q by norm), at each basis's rescale-best LR\n\n| basis | final val loss (mean ± std, n) | ms/step |\n|---|---|---|\n"
        for b in arms:
            f = [r["final"] for r in tr if r["basis"] == b and r.get("final") is not None]
            ms = statistics.median(r.get("ms_per_step", float("nan")) for r in tr if r["basis"] == b)
            md += f"| {b} | {statistics.mean(f):.4f} ± {statistics.stdev(f) if len(f) > 1 else 0:.4f} ({len(f)}) | {ms:.1f} |\n" if f else f"| {b} | failed | |\n"
    open(os.path.join(d, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "job":
        job(json.loads(sys.argv[2]))
    elif cmd in ("main", "all", "kill"):
        main()
