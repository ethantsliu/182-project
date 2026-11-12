# kernel-cost: a per-op cost model of the t1-gpu-stack step, and MOE_ROUTER_FUSED

Nothing here has run on a GPU.

## What is here

- `cost_model.py` (stdlib only): every GEMM, attention call, MoE dispatch/gather/combine, quantize and elementwise
  pass, lookup, optimizer piece and exposed collective of one training step of #397 on one H100 SXM rank, with FLOPs
  and HBM bytes from the shapes in `track_1_short/` and a roofline time (989 TF/s bf16, 1979 TF/s fp8, 3.35 TB/s)
  scaled by an achieved fraction per op class (the `EFF` table at the top, each value commented) plus 2 us per kernel.
  It walks the real schedule (`config.py`: 1053 steps, six regimes from 16384 to 49152 tokens per rank, sampled
  softmax widths 10240 / 14336 / 24576 / 50304), prints the ranked op table of one regime, a calibration against the
  measured ms/step of a #397 log, whole-run seconds per op, and the not-yet-tuned ops. `--csv` writes the table, for
  side-by-side comparison with the node profile (`182/tools/profile_nsys.sh` + `analyze_nsys.py`).
- `perf/kernels/moe_router.py` (new, in the variant tree) behind `MOE_ROUTER_FUSED=1`, off by default: the MoE
  router's forward GEMM + softmax + top-1 as one Triton kernel, and its backward GEMM (drouter) folded into the
  existing dx_combine pass. `model/moe.py` picks the autograd op at import; unset, the module is not even imported.
- `test_cost_model.py`, `test_moe_router.py`, `compile_router.py`, `bench_router.py`: see below.

## What the cost model says, and how far to trust it

Modelled run total 30.65 s against 30.66 s measured; per regime the model/measured ratio is 0.92 to 1.08. That
agreement is not evidence that the per-op numbers are right: the largest line, "residual/norm/MUDD elementwise"
(373 bf16 [T, 768] passes, about 9.3 s per run, 31% of the step), uses assumed pass counts and is effectively the
free parameter that absorbs every other error. The ranking of the GEMM and attention lines is better grounded
(shapes are exact; only `EFF` is assumed). The profile is what settles it.

Stage-2 (49152 tokens, P = 14336) top lines, ms per step: elementwise 12.4, FA3 fwd+bwd 4.0, dense-MLP dW1+dW2+dx
3.0, qk-norm/rope 1.7, O projection 1.7, QKV dX+dW 1.6, lm_head logits and dX 1.0 each. By group: elementwise 12.4,
attention 10.5, dense MLP 6.4, MoE 4.6, lm_head 3.1, lookups 1.4, optimizer 0.9, exposed comm 0.6, CPLM 0.3.

Already hand-written in #397: the Gluon warp-specialized G3 kernels (`mlp_g3.py`: dense MLP up-projection with the
relu^2 epilogue and the dpre backward; `moe_g3.py`: the MoE grouped up-projection and dpre), the Triton grouped fp8
MoE GEMMs, dispatch, gathers, dx_combine (`moe_mlp.py`), qk-norm/rope, the fp8 attention quantizes, the nvrtc CE
kernel, the ANVIL polar cascade. Not tuned (whole-run seconds in the model): the inductor elementwise aggregate (9.3),
the router GEMM with N = 8 (0.22), the drouter GEMM with M = 8, K = T (0.22), the router softmax/top-1 and the
softmax backward in inductor (0.03, 0.06).

Why the router and not the elementwise aggregate: the aggregate is not one op but a few hundred inductor-fused
kernels spread over the whole forward and backward (norms, residual adds, MUDD mixes), inductor already fuses within
them, and its size is the model's least grounded number. Cutting it means taking codegen away from inductor across
the model, not a call-site edit, and should wait for the profile to show which passes are real. The router is the
next untuned item, memory-bound (two reads of a 75.5 MB activation per layer at stage 2 by cuBLAS calls on skinny
shapes), feasible in plain Triton 3.6, and local to one autograd op. No upstream PR (searched `router` and `MoE` in
KellerJordan/modded-nanogpt, through #402) and nothing in-tree fuses it.

## The kernels

Forward, `router_topk`: one program per 64 rows reads rows in 256-column chunks, `tl.dot` against W.T padded to 16
experts, rounds the logits to bf16 (as F.linear's bf16 output is), then fp32 softmax and first-index argmax (torch.max's
tie rule) in registers; writes probs [T, 8] fp32, gate, expert. Replaces a cuBLAS GEMM plus inductor's softmax/max.

Backward, `dx_combine_drouter`: the dx_combine arithmetic statement for statement (so dx is bitwise the stock dx),
plus, from the same 128-column slab and row block, `dlogits_bf.T @ rows` accumulated in fp32 per program over 512 rows;
a [cdiv(T, 512), 8, 768] fp32 partial (2.4 MB at stage 2) is summed and rounded to bf16 once. Deterministic, no
atomics. Replaces the cuBLAS drouter GEMM, which re-reads rows.

Numerics: not bitwise against the stock path. A logit can round to the neighbouring bf16 value (different fp32
summation order than cuBLAS), which can flip the top-1 expert only where the top two logits are within that step, the
same kind of ulp flip the step-graph self-check already documents between its two compiled graphs. drouter differs from
cuBLAS by fp32 summation order before one bf16 rounding.

## Expected effect

Per layer and step at stage 2 the stock pieces read rows twice through cuBLAS (22.5 us each at full HBM bandwidth);
the fused forward reads them once at streaming bandwidth and the backward reads them inside a pass that already runs.
The answer hinges on what fraction of HBM bandwidth cuBLAS reaches on N = 8 and M = 8, K = T shapes, which nothing
off the GPU can tell. Cost model (fused kernels at 0.8 of HBM), whole-run delta by that cuBLAS fraction:
0.25: -503 ms, 0.4: -253 ms, 0.5: -170 ms, 0.6: -115 ms, 0.7: -75 ms, 0.8: -45 ms.

Stated range: -0.25 s to +0.03 s per run, centre about -0.1 s (cuBLAS near 0.6 of HBM). The positive end is the case
where cuBLAS is already near roofline and the fused backward, which runs 576 programs of 168 registers at stage 2
instead of dx_combine's 4608, loses bandwidth to lower occupancy. Loss is unchanged by construction apart from ulp-level
routing flips, so the break-even is 0 ms rather than a loss price; at about 250 ms per millinat on this stack a +0.4
mnat shift would cancel a -0.1 s gain, and nothing here should move loss that much. A 4-seed A/B cannot resolve 0.1 s
(run-to-run sd of train time is about 0.12 s, the figure 182/tools/README.md uses); the microbenchmark is the real decider and the A/B checks that nothing breaks
and loss does not move.

## How to run

CPU (from the repo root):

    python variants/t1-gpu-stack/182/kernel-cost/cost_model.py [--regime taper] [--router-fused] [--csv ops.csv]
    python -m pytest variants/t1-gpu-stack/182/kernel-cost/test_cost_model.py -q
    bash 182/tools/triton_cpu/run.sh python -m pytest variants/t1-gpu-stack/182/kernel-cost/test_moe_router.py -q
    TRITON_INTERPRET=0 bash 182/tools/triton_cpu/run.sh python variants/t1-gpu-stack/182/kernel-cost/compile_router.py

Node, first (one GPU, about 2 minutes; prints ms per call stock vs fused at 16384 / 32768 / 40960 / 49152 tokens, the
bandwidth cuBLAS reaches, the fused backward at 128 / 256 rows per program, and a projected saving per run):

    python variants/t1-gpu-stack/182/kernel-cost/bench_router.py
    python -m pytest variants/t1-gpu-stack/182/kernel-cost/test_moe_router.py -q      # incl. the whole op vs stock on H100

If the projection is under about 20 ms, stop: the A/B is not worth node time. Otherwise (if 128 or 256 rows per
program is clearly faster, set `DR_ROWS_PER_PROG` first):

    AB_ARMS="base router" AB_RUNS=4 variants/t1-gpu-stack/182/run_node.sh ab          # 8 runs, about 45 min

or through the session worker's generic xab phase (`182/experiments.d/kernel.sh`). The log must print
`[moe] MOE_ROUTER_FUSED=1: fused router kernels`.

## Verified here, and what is not

Verified on CPU: the cost model's schedule equals `TrainingSchedule`'s boundaries and step count, the hand-computed
FLOPs, the calibration guard (`test_cost_model.py`, 7 tests); both kernels against the stock router in the Triton
interpreter at 200 and 1000 tokens, including dx bitwise equal to dx_combine and drouter within one bf16 step, at 512
and 128 rows per program (`test_moe_router.py`); both kernels compile for sm_90a with Triton 3.6 at every training
token count, no spills (`compile_router.py`). With the flag off, the variant's `test_node.py` and `test_folds.py` pass.

Not verified: any timing; the cuBLAS bandwidth on these shapes (the whole premise); the whole op against
`RoutedFP8MoE` with real fp8 experts (`test_whole_op_matches_stock_on_hopper` needs an H100); torch.compile tracing
and CUDA-graph capture of the two new custom ops in the trainer and the step-graph self-check with them; GPU
accumulation order in `tl.dot` (the interpreter runs dots in fp64); every number in `EFF` and the elementwise pass
counts.
