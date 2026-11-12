"""GPU microbenchmark of MOE_ROUTER_FUSED: ms per call of the stock router pieces vs the fused kernels, at every
per-rank token count of training, and the projected time per run. One GPU, about 2 minutes.

    python variants/t1-gpu-stack/182/kernel-cost/bench_router.py            # on the node, inside the run venv
    python variants/t1-gpu-stack/182/kernel-cost/bench_router.py --tokens 49152 --rep 200

Stock (as RoutedFP8MoE runs them):
  fwd  F.linear(rows, W) [cuBLAS bf16, N = 8] + softmax(.float()) + max(-1)
  bwd  dx_combine(dx_pad, dest, dlogits_bf.float(), W) [Triton] + dlogits_bf.t() @ rows [cuBLAS bf16, M = 8, K = T]
Fused: router_topk (fwd) and dx_combine_drouter (bwd), perf/kernels/moe_router.py.

The stock softmax/max run eagerly here (2-3 kernels) where the trainer has inductor fuse them into one, so the fwd row
also prints the GEMM alone; the projection uses GEMM-alone + one 3 us kernel for the stock fwd, the conservative choice.
Each timing is triton.testing.do_bench (L2 flushed between calls, median). Before timing, every size checks the fused
outputs against the stock ones (dx bitwise, drouter within one bf16 step, >= 98% expert agreement) and refuses to time a
mismatch. The last lines: the effective HBM bandwidth of the two cuBLAS calls (plug into cost_model.EFF['skinny_cublas'])
and the projected delta over the 1053 steps (5 MoE layers per step).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="*", default=[16384, 32768, 40960, 49152])
    ap.add_argument("--rep", type=int, default=100, help="do_bench rep (ms of timing per measurement)")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    from triton.testing import do_bench

    import cost_model as cm
    from track_1_short.perf.kernels.moe_mlp import _dx_combine_impl, padded_rows
    from track_1_short.perf.kernels.moe_router import _dx_combine_drouter_impl, _router_topk_impl

    D, E, dev = 768, 8, "cuda"
    print(f"# {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"{'tokens':>7} {'fwd stock':>10} {'(gemm)':>8} {'fwd fused':>10} {'bwd stock':>10} {'(drouter)':>10} "
          f"{'bwd fused':>10} {'saved/layer':>12}   ms per call")
    saved_by_tokens, bw, alt_by_tokens = {}, {}, {}
    for T in a.tokens:
        g = torch.Generator(device=dev).manual_seed(T)
        rows = F.rms_norm(torch.randn(T, D, device=dev, generator=g), (D,)).bfloat16()
        W = (0.1 * torch.randn(E, D, device=dev, generator=g)).bfloat16()
        m = padded_rows(T, E)
        dx_pad = torch.randn(m, D, device=dev, generator=g).bfloat16()
        dest = torch.randperm(m, device=dev, generator=g)[:T].to(torch.int32)
        dlog = (1e-3 * torch.randn(T, E, device=dev, generator=g)).bfloat16()

        def fwd_stock():
            probs = torch.softmax(F.linear(rows, W).float(), dim=-1)
            return probs, *probs.max(dim=-1)

        def bwd_stock():
            return _dx_combine_impl(dx_pad, dest, dlog.float(), W), dlog.t() @ rows

        # correctness gate
        p_s, g_s, e_s = fwd_stock()
        p_f, g_f, e_f = _router_topk_impl(rows, W)
        agree = (e_s == e_f).float().mean().item()
        assert agree >= 0.98, f"T={T}: expert agreement {agree:.4f}"
        torch.testing.assert_close(p_f, p_s, rtol=3e-2, atol=1e-3)
        dx_s, dr_s = bwd_stock()
        dx_f, dr_f = _dx_combine_drouter_impl(dx_pad, dest, dlog, W, rows)
        assert torch.equal(dx_f, dx_s), f"T={T}: dx differs from dx_combine"
        torch.testing.assert_close(dr_f.float(), dr_s.float(), rtol=2 ** -6, atol=1e-6 * dr_s.float().abs().max().item())

        t_fs = do_bench(fwd_stock, rep=a.rep)
        t_gemm = do_bench(lambda: F.linear(rows, W), rep=a.rep)
        t_ff = do_bench(lambda: _router_topk_impl(rows, W), rep=a.rep)
        t_bs = do_bench(bwd_stock, rep=a.rep)
        t_dr = do_bench(lambda: dlog.t() @ rows, rep=a.rep)
        t_bf = do_bench(lambda: _dx_combine_drouter_impl(dx_pad, dest, dlog, W, rows), rep=a.rep)
        alt = {rpp: do_bench(lambda: _dx_combine_drouter_impl(dx_pad, dest, dlog, W, rows, rows_per_prog=rpp),
                             rep=a.rep) for rpp in (128, 256)}
        alt_by_tokens[T] = (t_bf, alt)
        stock = (t_gemm + 0.003) + t_bs
        saved = stock - (t_ff + t_bf)
        saved_by_tokens[T] = saved
        bw[T] = (T * D * 2 / (t_gemm * 1e-3) / 1e12, T * D * 2 / (t_dr * 1e-3) / 1e12)
        print(f"{T:>7} {t_fs:>10.4f} {t_gemm:>8.4f} {t_ff:>10.4f} {t_bs:>10.4f} {t_dr:>10.4f} {t_bf:>10.4f} "
              f"{saved:>12.4f}")
    print("\n# bwd fused at other rows-per-program (ms; the trainer uses DR_ROWS_PER_PROG = 512). If one is clearly faster,")
    print("# set DR_ROWS_PER_PROG in perf/kernels/moe_router.py to it before the A/B; the projection below uses 512.")
    for T, (t512, alt) in alt_by_tokens.items():
        print(f"  T={T}: 512 {t512:.4f}  " + "  ".join(f"{k} {v:.4f}" for k, v in alt.items()))
    print("\n# effective HBM read bandwidth of the cuBLAS calls (TB/s, of 3.35): router GEMM / drouter GEMM")
    for T, (b1, b2) in bw.items():
        print(f"  T={T}: {b1:.2f} ({b1 / 3.35:.0%}) / {b2:.2f} ({b2 / 3.35:.0%})")
    total, missing = 0.0, []
    for name, (r, n) in cm.regimes().items():
        if r.tokens in saved_by_tokens:
            total += saved_by_tokens[r.tokens] * n * len(cm.MOE_LAYERS)
        else:
            missing.append(name)
    print(f"\nPROJECTED router fusion saving: {total:.1f} ms per run (5 MoE layers x 1053 steps; positive = faster)"
          + (f"; regimes not measured: {missing}" if missing else ""))
    print("Kill if the projection is under ~20 ms (inside run-to-run noise); then MOE_ROUTER_FUSED is not worth an A/B.")


if __name__ == "__main__":
    main()
