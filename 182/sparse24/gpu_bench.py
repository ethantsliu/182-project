"""SPARSE24_MLP microbenchmark on one GPU (about 2 minutes): does 2:4 pay at the record's MLP shapes?

    python 182/sparse24/gpu_bench.py                         # all parts, one table, the projection
    python 182/sparse24/gpu_bench.py --json sp24_bench.json --iters 30 --tokens 16384,49152

Parts (every number is a microbenchmark of one process on one GPU, warm L2, CUDA-event medians):
  (1) GEMMs at the record's shapes, per token count T: the K=768 GEMMs (fwd x @ W1.T, dpost g @ W2.T; out 2816 wide),
      the K=2816 GEMMs (fwd post @ W2, dx dpre @ W1; out 768 wide) and the dW GEMMs (contract over T), each as
      dense bf16 (torch.mm), dense fp8 (torch._scaled_mm, e4m3, the record's format), 2:4 bf16 (cuSPARSELt, alg 0
      and the best of torch._cslt_sparse_mm_search) and 2:4 fp8 (cuSPARSELt e4m3 -> bf16, if this build supports it).
  (2) One MLP call, forward + backward, at slot 0: the record's fp8 FusedLinearReLUSquareFunction (via the
      HADAMARD_FP8 sanity harness's mini state, flags off), the bf16 dense BF16LinearReLUSquareFunction (FP8_OFF=mlp)
      and Sparse24ReLUSquareFunction (cuSPARSELt). Eager; --compile also times them under torch.compile.
  (3) The per-step refresh over the 12-slot bank: the record's fp8 weight refresh vs the 2:4 mask + compress
      (greedy and optimal).
  (4) The projection over the record schedule (STAGES, 11 MLP calls per step):
        delta_s = sum_stages steps * 11 * (sparse_call_ms(T) - fp8_call_ms(T)) + 1194 * (refresh delta)
      and the loss budget it buys at 164 ms/mnat (band 120-250): budget_mnat = -delta_s / 0.164, nothing if
      delta_s >= 0. A second line projects the GEMMs alone (4 sparse bf16 + 2 dense bf16 dW vs 6 fp8), the
      ceiling if every elementwise pass were fused away.
Nothing here is a training run.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The fp8 baseline borrows GPT's real fp8 MLP state, which must not be the SPARSE24 variant.
for _k in ("SPARSE24_MLP", "HADAMARD_FP8", "FP8_OFF", "MONARCH_MLP"):
    os.environ.pop(_k, None)

STAGES = ((16384, 320), (32768, 361), (49152, 426), (40960, 67), (16384, 20))  # (per-rank tokens, steps)
TOTAL_STEPS = sum(s for _, s in STAGES)
MLP_CALLS_PER_STEP = 11
MS_PER_MNAT, MS_PER_MNAT_BAND = 164.0, (120.0, 250.0)
H, D = 2816, 768


def _hadamard_sanity():
    spec = importlib.util.spec_from_file_location("hadamard_gpu_sanity", ROOT / "182/hadamard-fp8/gpu_sanity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_ms(fn, iters, warmup=3):
    import torch
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def _try(fn):
    try:
        return fn()
    except Exception as e:  # report, do not die: an unsupported variant is a result
        return f"unsupported: {type(e).__name__}: {str(e).splitlines()[0][:120]}"


# -----------------------------------------------------------------------------------------------------
# (1) GEMMs

def gemm_table(tokens, iters):
    import torch

    from track_1_short.perf import sparse24 as S
    dev, bf16, e4 = "cuda", torch.bfloat16, torch.float8_e4m3fn
    w = (0.03 * torch.randn(H, D, device=dev)).bfloat16()
    wm = w * S.mask_tile_greedy(w)
    packed, packed_t = S._compress_pair(w, "greedy")
    one = torch.ones((), device=dev)
    out = {}
    for T in tokens:
        zd = torch.randn(T, D, device=dev).bfloat16()     # x, g:   [T, 768]
        zh = (0.1 * torch.randn(T, H, device=dev)).bfloat16()  # post, dpre: [T, 2816]
        row = {}
        # name: (dense A as [m, K] for z @ A.T, z, compressed A)
        for name, a, z, pk in (("k768", wm, zd, packed), ("k2816", wm.T.contiguous(), zh, packed_t)):
            m = a.shape[0]
            r = {"dense_bf16": time_ms(lambda: z @ a.T, iters)}
            a8, z8 = a.to(e4), z.to(e4)
            r["dense_fp8"] = _try(lambda: time_ms(
                lambda: torch._scaled_mm(z8, a8.T, scale_a=one, scale_b=one, out_dtype=bf16), iters))
            r["sparse_bf16"] = time_ms(lambda: S.cslt_linear(pk, z, m), iters)

            def searched():
                alg = torch._cslt_sparse_mm_search(pk, z.t())
                alg = alg[0] if isinstance(alg, tuple) else alg
                return {"alg": int(alg), "ms": time_ms(
                    lambda: torch._cslt_sparse_mm(pk, z.t(), transpose_result=True, alg_id=int(alg)), iters)}
            r["sparse_bf16_search"] = _try(searched)

            def fp8_sparse():
                pk8 = torch._cslt_compress(a.to(e4))
                return time_ms(lambda: torch._cslt_sparse_mm(pk8, z8.t(), alpha=one, out_dtype=bf16,
                                                             transpose_result=True), iters)
            r["sparse_fp8"] = _try(fp8_sparse)
            row[name] = r
        # dW = dpre.T @ x: [2816, T] @ [T, 768]; dense only (contracts over tokens)
        r = {"dense_bf16": time_ms(lambda: zh.T @ zd, iters)}
        zh8t, zd8 = zh.to(e4).T.contiguous(), zd.to(e4)
        r["dense_fp8"] = _try(lambda: time_ms(
            lambda: torch._scaled_mm(zh8t, zd8.T.contiguous().T, scale_a=one, scale_b=one, out_dtype=bf16), iters))
        row["dW"] = r
        out[str(T)] = row
        torch.cuda.empty_cache()
    return out


# -----------------------------------------------------------------------------------------------------
# (2) one MLP call, (3) refresh

def mlp_calls(tokens, iters, compile_too):
    import torch

    from track_1_short.model import gpt
    from track_1_short.perf import sparse24 as S
    from track_1_short.perf.kernels.mlp import BF16LinearReLUSquareFunction, FusedLinearReLUSquareFunction
    hs = _hadamard_sanity()
    dev = torch.device("cuda")
    m = hs.make_mini_state(H, D, dev, slot=0)
    for _ in range(gpt.FP8_EXACT_SCALE_CALLS):
        m._quantize_mlp_weights_and_scales()
    S.SPARSE24_BACKEND = "cusparselt"
    bufs = S.sparse24_buffers(gpt.NUM_MLP_SLOTS, H, D, dev)
    bank = m.mlp_bank.detach()
    res = {"refresh": {"fp8": time_ms(m._quantize_mlp_weights_and_scales, iters)}, "T": {}}
    for kind in ("greedy", "optimal"):
        S.SPARSE24_MASK = kind
        res["refresh"][f"sparse_{kind}"] = time_ms(lambda: S.refresh_sparse24_(bank[:, 0], bank[:, 1], bufs), iters)
    S.SPARSE24_MASK = "greedy"
    S.refresh_sparse24_(bank[:, 0], bank[:, 1], bufs)
    W1, W2 = (m.mlp_bank[0, i].detach().requires_grad_() for i in (0, 1))
    ops = S.slot_operands(bufs, 0)

    def fp8_fn(x):
        args = m._mlp_fp8_args(0, m._mlp_up_f8[0], m._quantize_mlp_input(x.detach().view(-1, D), 0), None, xt_slot=0)
        return FusedLinearReLUSquareFunction.apply(x, W1, W2, *args)

    arms = {"fp8_record": fp8_fn,
            "bf16_dense": lambda x: BF16LinearReLUSquareFunction.apply(x, W1, W2),
            "sparse24_bf16": lambda x: S.Sparse24MLP(x, W1, W2, *ops)}
    if compile_too:
        arms.update({f"{k}_compiled": torch.compile(f, fullgraph=True) for k, f in list(arms.items())})
    for T in tokens:
        torch.manual_seed(T)
        x = hs.rms_rows(torch.randn(T, D, device=dev)).view(1, T, D)
        g = (1e-2 * torch.randn(1, T, D, device=dev)).bfloat16()
        row = {}
        for name, f in arms.items():
            def call():
                xin = x.detach().requires_grad_()
                torch.autograd.grad(f(xin), (xin, W1, W2), g)
            row[name] = _try(lambda: time_ms(call, iters))
        res["T"][str(T)] = row
        torch.cuda.empty_cache()
    return res


# -----------------------------------------------------------------------------------------------------
# (4) projection

def _steps_by_T():
    by = {}
    for T, steps in STAGES:
        by[T] = by.get(T, 0) + steps
    return by


def project(calls, gemms):
    """Seconds over the run, sparse arm minus record, from whole-call times and from GEMMs alone."""
    out = {}
    by = _steps_by_T()
    if all(str(T) in calls["T"] for T in by):
        try:
            d = sum(n * MLP_CALLS_PER_STEP * (calls["T"][str(T)]["sparse24_bf16"] - calls["T"][str(T)]["fp8_record"])
                    for T, n in by.items()) / 1000.0
            d += TOTAL_STEPS * (calls["refresh"]["sparse_greedy"] - calls["refresh"]["fp8"]) / 1000.0
            out["calls_delta_s"] = d
        except TypeError:
            out["calls_delta_s"] = None  # an arm was unsupported
    if all(str(T) in gemms for T in by):
        try:
            d = 0.0
            for T, n in by.items():
                r = gemms[str(T)]
                sparse = 2 * r["k768"]["sparse_bf16"] + 2 * r["k2816"]["sparse_bf16"] + 2 * r["dW"]["dense_bf16"]
                fp8 = 2 * r["k768"]["dense_fp8"] + 2 * r["k2816"]["dense_fp8"] + 2 * r["dW"]["dense_fp8"]
                d += n * MLP_CALLS_PER_STEP * (sparse - fp8)
            out["gemm_only_delta_s"] = d / 1000.0
        except TypeError:
            out["gemm_only_delta_s"] = None
    for k in ("calls_delta_s", "gemm_only_delta_s"):
        d = out.get(k)
        if d is not None:
            out[k.replace("delta_s", "budget_mnat")] = {
                "centre": max(0.0, -d * 1000.0 / MS_PER_MNAT),
                "band": [max(0.0, -d * 1000.0 / b) for b in MS_PER_MNAT_BAND]}
    return out


def fmt(v):
    return f"{v:8.3f}" if isinstance(v, (int, float)) else (f"{v['ms']:8.3f}(a{v['alg']})" if isinstance(v, dict) else "     n/a")


def report(res):
    lines = ["(1) GEMM ms  [k768: out 2816 wide (fwd W1, dpost) | k2816: out 768 wide (fwd W2, dx) | dW]"]
    for T, row in res["gemms"].items():
        for name, r in row.items():
            lines.append(f"  T={T:>6} {name:6} " + "  ".join(f"{k}={fmt(v)}" for k, v in r.items()))
            for k, v in r.items():
                if isinstance(v, str):
                    lines.append(f"      {k}: {v}")
    lines.append("(2) one MLP call, fwd+bwd ms")
    for T, row in res["calls"]["T"].items():
        lines.append(f"  T={T:>6} " + "  ".join(f"{k}={fmt(v)}" for k, v in row.items()))
    lines.append("(3) refresh ms per step (12 slots): " + "  ".join(f"{k}={fmt(v)}" for k, v in res["calls"]["refresh"].items()))
    p = res["projection"]
    lines.append("(4) projected over the 1194-step schedule (sparse arm - record):")
    for k, label in (("calls", "whole calls + refresh"), ("gemm_only", "GEMMs alone (fusion ceiling)")):
        d = p.get(f"{k}_delta_s")
        if d is None:
            lines.append(f"  {label}: n/a")
            continue
        b = p[f"{k}_budget_mnat"]
        lines.append(f"  {label}: {d:+.2f} s; loss budget {b['centre']:.1f} mnat (band {b['band'][1]:.1f}-{b['band'][0]:.1f})"
                     + ("  -> cannot pay: slower before any loss cost" if d >= 0 else ""))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", default="16384,32768,40960,49152")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compile", action="store_true", help="also time the three MLP arms under torch.compile")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        sys.exit("gpu_bench.py needs a CUDA GPU")
    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "tokens": tokens}
    res["gemms"] = gemm_table(tokens, args.iters)
    res["calls"] = mlp_calls(tokens, args.iters, args.compile)
    res["projection"] = project(res["calls"], res["gemms"])
    print(f"{res['device']}, torch {res['torch']}")
    print(report(res))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
