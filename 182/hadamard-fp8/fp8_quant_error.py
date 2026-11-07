"""Offline FP8 quantization error of the record's fp8 GEMMs, with vs without block-Hadamard rotation.

Runs on CPU, on tensors saved by FP8_DUMP_DIR (track_1_short/perf/fp8_dump.py) or on synthetic ones:

    python 182/hadamard-fp8/fp8_quant_error.py --dump /path/to/dumpdir           # every step*.pt in it
    python 182/hadamard-fp8/fp8_quant_error.py --synthetic                      # smoke test, no data
    options: --blocks 128,256  --seed 1234  --json out.json  --max-rows 4096

For every GEMM C = A @ B it reports, per rotation (none / block b along the contraction dim, random
signs from --seed, same Q as the training kernels when b = 128 and the seed matches HADAMARD_SEED):
  operand A and B: relative MSE ||Q(t) - t||^2 / ||t||^2, SNR (dB), amax / rms, the fraction saturated
  by the clamp and the fraction of nonzeros flushed to zero;
  output: relative MSE and SNR of dequant(Q(A)) @ dequant(Q(B)) against the fp64 product of the bf16
  operands, and the output rel-MSE ratio rotated / plain (< 1 means the rotation helps).

Scale rules (the record's, model/gpt.py and perf/kernels/*):
  MLP  x e4m3 static 2^-4; W1, W2 e4m3 exact amax/448; post e4m3 delayed (amax * 1.03 / 448);
       g e5m2 static 2^-6; dpre e5m2 delayed (amax * 1.25 / 57344)
  QKV  x e4m3 static 8/448; W e4m3 exact; g e4m3 static 1/448 (QKV is not rotated in training yet:
       its rotated arms are what-ifs under the MLP's rules)
Offline there is no previous step, so a delayed scale uses the tensor's own amax and headroom (the
real run's one-step lag is not modelled). Rotated operands get the HADAMARD_FP8 rules: static scales
stay static (g), the x_t of dW1 switches to delayed (amax * 1.10 / 448), weights stay exact, post and
dpre stay delayed -- each from the ROTATED amax. "dW1 xt-static" is the what-if that keeps x_t at the
static 2^-4 after rotating it along tokens: a channel with a large per-token offset concentrates into
one coefficient per block and saturates.

post / dpre are rebuilt from the dump exactly as the kernels compute them: pre = bf16(x @ W1.T),
post = bf16(relu(pre)^2), dpre = bf16(2 (g @ W2_eff.T) relu(pre)) with W2_eff = p_fold * W2. Each GEMM is
measured with exact (unquantized) inputs, so the numbers are per-GEMM, not compounded.
"""
import argparse
import glob
import json
import math
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from track_1_short.perf.hadamard_ref import FP8_MAX, rotate, signed_hadamard  # noqa: E402

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
MLP_X_SCALE, MLP_G_SCALE = 2.0 ** -4, 2.0 ** -6
POST_HEADROOM, DPRE_HEADROOM, XT_HEADROOM = 1.03, 1.25, 1.10
ATTN_X_SCALE, ATTN_G_SCALE = 8.0 / 448.0, 1.0 / 448.0


def scale_of(rule, t, fmt):
    kind, v = rule
    if kind == "static":
        return v
    headroom = 1.0 if kind == "exact" else v
    return max(float(t.abs().max()) * headroom / FP8_MAX[fmt], 1e-30)


def quant_stats(t, fmt, rule):
    s = scale_of(rule, t, fmt)
    lim = FP8_MAX[fmt]
    scaled = t / s
    q = torch.clamp(scaled, -lim, lim).to(fmt).to(torch.float64) * s
    err = q - t
    ss = float((t * t).sum())
    rel = float((err * err).sum()) / ss if ss > 0 else 0.0
    nz = t != 0
    return q, dict(
        rel_mse=rel, snr_db=(-10 * math.log10(rel) if rel > 0 else float("inf")),
        amax_over_rms=float(t.abs().max()) / max(math.sqrt(ss / t.numel()), 1e-300),
        saturated=float((scaled.abs() > lim).double().mean()),
        flushed=float(((q == 0) & nz).double().sum() / max(int(nz.sum()), 1)),
        scale=s)


def gemm_report(name, a, b, a_fmt, b_fmt, a_rule, b_rule, blocks, seed, a_rot_rule=None, b_rot_rule=None):
    """C = a @ b, contraction over a's last / b's first dim, in fp64."""
    a, b = a.double(), b.double()
    ref = a @ b
    ref_ss = float((ref * ref).sum())
    out = {"gemm": name, "shape": [a.shape[0], a.shape[1], b.shape[1]], "rot": {}}
    K = a.shape[1]
    for blk in [0] + list(blocks):
        if blk and K % blk:
            out["rot"][str(blk)] = None  # contraction dim not a multiple of the block
            continue
        if blk:
            qm = signed_hadamard(blk, seed)
            ar, br = rotate(a, 1, qm), rotate(b, 0, qm)
            ra, rb = a_rot_rule or a_rule, b_rot_rule or b_rule
        else:
            ar, br, ra, rb = a, b, a_rule, b_rule
        qa, sa = quant_stats(ar, a_fmt, ra)
        qb, sb = quant_stats(br, b_fmt, rb)
        c = qa @ qb
        rel = float(((c - ref) ** 2).sum()) / ref_ss
        out["rot"][str(blk)] = dict(A=sa, B=sb, out_rel_mse=rel, out_snr_db=-10 * math.log10(max(rel, 1e-300)))
    plain = out["rot"]["0"]["out_rel_mse"]
    for blk in blocks:
        r = out["rot"].get(str(blk))
        if r is not None:
            r["ratio_vs_plain"] = r["out_rel_mse"] / plain if plain > 0 else float("nan")
    return out


def bf16(t):
    return t.to(torch.bfloat16).to(torch.float64)


def mlp_gemms(rec, blocks, seed, max_rows):
    T = (min(rec["x"].shape[0], max_rows) // 256) * 256
    x, g = bf16(rec["x"][:T].float()), bf16(rec["g"][:T].float())
    W1, W2 = bf16(rec["W1"].float()), bf16(rec["W2"].float())
    p = rec.get("p_fold")
    W2_eff = bf16(W2 * p) if p is not None else W2
    pre = bf16(x @ W1.T)
    post = bf16(torch.relu(pre) ** 2)
    dpre = bf16(2 * (g @ W2_eff.T) * torch.relu(pre))
    st = lambda v: ("static", v)
    ex, dl = ("exact", None), lambda h: ("delayed", h)
    return [
        # forward GEMMs (not rotated by HADAMARD_FP8; listed for context)
        gemm_report("fwd x@W1.T", x, W1.T, E4, E4, st(MLP_X_SCALE), ex, blocks, seed),
        gemm_report("fwd post@W2", post, W2_eff, E4, E4, dl(POST_HEADROOM), ex, blocks, seed),
        # the four gradient GEMMs HADAMARD_FP8 rotates
        gemm_report("dW2 post.T@g", post.T, g, E4, E5, dl(POST_HEADROOM), st(MLP_G_SCALE), blocks, seed),
        gemm_report("dpost g@W2.T", g, W2_eff.T, E5, E4, st(MLP_G_SCALE), ex, blocks, seed),
        gemm_report("dW1 dpre.T@x", dpre.T, x, E5, E4, dl(DPRE_HEADROOM), st(MLP_X_SCALE), blocks, seed,
                    b_rot_rule=dl(XT_HEADROOM)),
        gemm_report("dx dpre@W1", dpre, W1, E5, E4, dl(DPRE_HEADROOM), ex, blocks, seed),
        # what-if: the rotated x_t kept at the static 2^-4 (why HADAMARD_FP8 gives x_t a delayed scale)
        gemm_report("dW1 xt-static", dpre.T, x, E5, E4, dl(DPRE_HEADROOM), st(MLP_X_SCALE), blocks, seed),
    ]


def qkv_gemms(rec, blocks, seed, max_rows):
    T = (min(rec["x"].shape[0], max_rows) // 256) * 256
    x, g, W = bf16(rec["x"][:T].float()), bf16(rec["g"][:T].float()), bf16(rec["W"].float())
    st, ex = (lambda v: ("static", v)), ("exact", None)
    # Not rotated by HADAMARD_FP8 yet; rotated arms use the MLP's rules (token-rotated x: delayed scale).
    return [
        gemm_report("qkv fwd x@W.T", x, W.T, E4, E4, st(ATTN_X_SCALE), ex, blocks, seed),
        gemm_report("qkv dW g.T@x", g.T, x, E4, E4, st(ATTN_G_SCALE), st(ATTN_X_SCALE), blocks, seed,
                    b_rot_rule=("delayed", XT_HEADROOM)),
        gemm_report("qkv dx g@W", g, W, E4, E4, st(ATTN_G_SCALE), ex, blocks, seed),
    ]


def synthetic_records(seed):
    """Plausible-shaped stand-ins (NOT evidence): rms-normed x with a few large-offset channels, heavy-tailed
    per-token gradients at a realistic small magnitude, N(0, 0.02) weights."""
    g = torch.Generator().manual_seed(seed)
    T, D, H = 1024, 768, 2816
    x = torch.randn(T, D, generator=g)
    x[:, :4] += torch.tensor([12.0, -9.0, 7.0, -15.0])  # massive-activation style channels
    x = x / x.pow(2).mean(1, keepdim=True).sqrt()
    tok = torch.exp(1.5 * torch.randn(T, 1, generator=g))
    gr = 3e-5 * tok * torch.randn(T, D, generator=g)
    W1 = 0.02 * torch.randn(H, D, generator=g)
    W2 = 0.02 * torch.randn(H, D, generator=g)
    qkv = 1536
    Wqkv = 0.02 * torch.randn(qkv, D, generator=g)
    gq = 1e-4 * tok * torch.randn(T, qkv, generator=g)
    return {"synthetic": dict(records={
        "mlp0": dict(kind="mlp", x=x.bfloat16(), g=gr.bfloat16(), W1=W1.bfloat16(), W2=W2.bfloat16(), p_fold=None),
        "qkv0": dict(kind="qkv", x=x.bfloat16(), W=Wqkv.bfloat16(), g=gq.bfloat16()),
    })}


def fmt_row(r, blocks):
    p = r["rot"]["0"]
    cells = [f"{r['gemm']:<15}", f"{p['out_snr_db']:6.1f}"]
    for blk in blocks:
        x = r["rot"].get(str(blk))
        cells.append("   n/a          " if x is None else f"{x['out_snr_db']:6.1f} x{x['ratio_vs_plain']:5.2f}")
    cells.append(f"A sat {p['A']['saturated']:.1e} flush {p['A']['flushed']:.1e} crest {p['A']['amax_over_rms']:6.1f}")
    cells.append(f"B sat {p['B']['saturated']:.1e} flush {p['B']['flushed']:.1e} crest {p['B']['amax_over_rms']:6.1f}")
    return "  ".join(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump", help="a FP8_DUMP_DIR, or one step*.pt file")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--blocks", default="128,256")
    ap.add_argument("--seed", type=int, default=1234, help="HADAMARD_SEED of the run (training default 1234)")
    ap.add_argument("--max-rows", type=int, default=4096, help="tokens per tensor (CPU time grows linearly)")
    ap.add_argument("--json", help="write every number here")
    args = ap.parse_args()
    blocks = [int(b) for b in args.blocks.split(",") if b]

    if args.synthetic:
        dumps = synthetic_records(args.seed)
    else:
        files = sorted(glob.glob(os.path.join(args.dump, "step*.pt"))) if os.path.isdir(args.dump) else [args.dump]
        assert files, f"no step*.pt in {args.dump}"
        dumps = {os.path.basename(f): torch.load(f, map_location="cpu", weights_only=False) for f in files}

    results, ratios = [], {}
    header = f"{'gemm':<15}  {'plain':>6}  " + "  ".join(f"{'H' + str(b):>6} {'ratio':>6}" for b in blocks) + "   (SNR dB; ratio = rotated/plain output rel-MSE)"
    for name, dump in dumps.items():
        for tag, rec in sorted(dump["records"].items()):
            rows = (mlp_gemms if rec["kind"] == "mlp" else qkv_gemms)(rec, blocks, args.seed, args.max_rows)
            print(f"\n== {name} {tag} (T={rows[0]['shape'][0]})")
            print(header)
            for r in rows:
                print(fmt_row(r, blocks))
                for blk in blocks:
                    x = r["rot"].get(str(blk))
                    if x is not None:
                        ratios.setdefault((r["gemm"], blk), []).append(x["ratio_vs_plain"])
            results.append(dict(dump=name, tag=tag, gemms=rows))

    print("\n== geometric-mean output rel-MSE ratio, rotated / plain (< 1: rotation helps)")
    for (gemm, blk), v in sorted(ratios.items()):
        gm = math.exp(sum(math.log(max(r, 1e-300)) for r in v) / len(v))
        print(f"  {gemm:<15} H{blk:<4} {gm:6.3f}  over {len(v)} records")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
