# fp8-attn-o: the attention output projection in FP8

Nothing here has run on a GPU. Every number below is arithmetic, a CPU / Triton-interpreter test, or an sm_90a
compile without execution. The expectation, stated up front: **this probably does not pay** (see the cost model);
it is a cheap experiment with a clean A/B.

## What it is

`FP8_ATTN_O=1` (default off) runs the attention output projection, `F.linear(y, o_scale * o_w)` in
`track_1_short/model/attention.py`, as three fp8 `torch._scaled_mm` GEMMs during training:

| GEMM | operands | accumulate | descale |
|---|---|---|---|
| forward `out = y @ O.T` | e4m3 x e4m3 | fast | `s_y * (s_w * o_scale)` |
| `dy = g @ O` | e5m2 x e4m3 | full | `s_g * (s_w * o_scale)` |
| `dW = g.T @ y` | e5m2 x e4m3 | full | `s_g * s_y`, then `d o_w = o_scale * dW` |

It is the last bf16 dense GEMM family in the training step (attention itself stays bf16 FA3). Validation stays bf16.

- **The model is unchanged in exact arithmetic.** `o_scale = sa_lambdas[1]` (times `o_gain` where the layer has one)
  never multiplies the weight; it rides the weight's descale, as the QKV path folds `sa_lambdas[0]`.
  `d o_scale = <dW, o_w>`. What does change is rounding: the record rounds `bf16(o_scale * o_w)` once, this path
  quantizes `o_w` alone to e4m3 and applies `o_scale` in fp32.
- **`o_w` caches: in the lagged refresh, not per call.** `GPT._quantize_attn_o_fp8` runs inside
  `quantize_attn_fp8`, so it is part of the attention half of the fp8 weight-cache refresh and of its CUDA graph
  (buffers named `_attn_o_*`). One launch quantizes all seven `[768, 768]` O matrices in both layouts from the
  strided `vo_bank` view; two byte copies cut the half-V layers' `[768, 384]` caches out of it. That is
  7 x 768² x (2 B read + 2 B written) = 16.5 MB of traffic per step. Quantizing per call would cost the same
  bytes but as 14 extra launches inside the compiled step, and the weights only change once per step anyway.
- **`y` and `g` are quantized per call**, each in two layouts (row for forward / dx, transposed for dW):
  - layers 1, 3, 10 (XSA and/or gate): the row quantize is aten ops that inductor fuses into the XSA / gate
    elementwise kernel that produces `y`, plus one fp8 transpose (`quantize_dual_layout_fused`, the record's
    helper);
  - layers 0, 2, 5, 8 (y straight from FA3, nothing to fuse into): one Triton pass reading `y` once and writing
    both layouts (`fp8_dual_quant`, the Hadamard dual-quantize kernel with both rotations off, behind a custom op);
  - `g`: the fused path on every layer.
- **Files.** `track_1_short/perf/kernels/attn_o.py` (Function, quantize ops, weight caches),
  `track_1_short/perf/attn_o_scales.py` (scales, derivation, pure-torch reference), `model/gpt.py` (buffers,
  refresh, per-layer args, sat-stats rows), `model/attention.py` (the call), `config.py` (knobs),
  `perf/cuda_graphs/fp8_refresh_graphs.py` (the `_attn_o_` prefix), `gpu_sanity.py` and `tests/` here.

## Knobs

| env | meaning |
|---|---|
| `FP8_ATTN_O=1` | turn it on (0 = the record) |
| `FP8_ATTN_O_SCALES=static` (default) | the power-of-two scales below |
| `FP8_ATTN_O_SCALES=delayed` | `y` and `g` scales from the previous step's amax (headroom 1.10 / 1.25), `o_w` from its current amax. Every layer then uses the Triton quantize, so it is slower than static |
| `HADAMARD_SAT_STATS=1` | analysis only: adds `o_y` / `o_g` / `o_w` rows to the `fp8 sat` lines (amax / (scale x format max) per layer). Records amaxes, so it also forces the Triton quantize; the log is tagged `INVALID_FOR_RECORD` |

Interplay: `FP8_OFF=all` together with `FP8_ATTN_O=1` fails at import (that arm is the all-bf16 upper bound). A
partial `FP8_OFF`, `HADAMARD_FP8=1` (MLP only) and `MONARCH_MLP=1` all compose with it.

## The static scales (unit scaling)

The rule is u-μP's (Blake et al. 2024; Narayan et al. 2025 use the same idea for static FP8): scale each tensor so
its typical rms lands near 1 with a power of two, then check a bound on the largest entry against the format's
max where a bound exists. fp8 is floating point, so relative error is flat across about 15 binades (e4m3) or 30
(e5m2); the scale only has to keep the bulk inside that window. In the tests, the error at these static scales
equals the error at the tensor's own amax scale (2.65% rel-L2 for e4m3, 5.3% for e5m2) for rms between about
2^-5 and 2^3 times the scale. The full derivation is the docstring of `perf/attn_o_scales.py`; in short:

- **`y`, e4m3, `2^-3`.** FA3 returns a convex combination of V rows per head, so `|y| <= max|v|` and
  `rms(y) <= rms(v)`. `v = lambda0 * x_hat @ W_v.T`, `x_hat` RMS-normed, so `|v| <= lambda0 * r * sqrt(768)` for a
  `W_v` row norm `r`; at init `lambda0 = 0.5`, `r = 0.5`: `std(v) = 0.25`, `|v| <= 6.93`. XSA at most doubles the
  norm, the gate multiplies by `G` (about 0.25 at init). Bound at init: `|y| <= 13.9 G`, against `448 * 2^-3 = 56`.
  Typical `rms(y)` is 0.025 to 0.25, i.e. 0.2 to 2 in scaled units.
- **`o_w`, e4m3, `2^-6`.** Init is uniform ±2^-5 (std 0.018, 1.15 in scaled units). ANVIL's whitened steps with
  decoupled decay give a random-walk equilibrium entry std of 0.024 to 0.031, largest entry about 0.15, against
  `448 * 2^-6 = 7`.
- **`g`, e5m2, `2^-6`.** The residual-stream gradient right after the attention add, one MLP away from the MLP's
  incoming gradient, which the record already quantizes at a static e5m2 `2^-6`. Same stream, same scale.

What the analysis does **not** bound: growth of `lambda0`, the `W_v` row norms and the gate during training (the
`y` bound holds while `lambda0 * r * (1 + |alpha|) * G` stays within 4x its init value), the last layer's `v_mudd`
term, the `o_w` equilibrium (an argument, not a bound) and `g` (borrowed evidence). That is why the delayed arm
and the `fp8 sat` rows exist: one `HADAMARD_SAT_STATS=1` run shows whether any ratio approaches 1.

## Cost model (not measured)

Record shapes: 1194 steps, per-rank tokens 16384 x 340, 32768 x 361, 49152 x 426, 40960 x 67 steps, i.e.
41.08 M token-steps per rank. Five O matrices are `[768, 768]`, two are `[768, 384]`.

| | MFLOP per token | share |
|---|---|---|
| O projection: 3 GEMMs x 2 x 768 x (5 x 768 + 2 x 384) | 21.2 | 7.4% of the MLP |
| MLP: 6 GEMMs x 2 x 768 x 2816 x 11 live layers | 285.5 | |

- **What the GEMMs cost now.** 8.72e14 FLOP per rank over the run. At an assumed 350 to 600 TFLOP/s effective
  for bf16 cuBLAS at these shapes (k = 768 or 384; an assumption, not a measurement) that is 1.5 to 2.5 s of the
  39.9 s.
- **What fp8 can save.** At an assumed 1.3x to 1.8x fp8-over-bf16 GEMM speedup: 0.33 to 1.1 s.
- **What the quantizes add.** Per token: `y` on the plain layers 4 B per element (2 read, 2 written) over 2688
  elements; `y` on the fused layers about 1 B net over 1920 elements; `g` between 1 and 5 B per element over
  5376 elements, depending on whether inductor can replace `g`'s bf16 write or must add a pass. Total 18 to
  40 KB per token, 0.74 to 1.62 TB per rank over the run. At an assumed 1 to 2 TB/s of effective memory
  throughput: 0.37 to 1.63 s. The `o_w` refresh (16.5 MB per step) and the dropped `o_scale * o_w` product are
  noise.
- **Expected Δtime: −0.7 s to +1.3 s, centre about +0.2 s (slower).** The MLP gets its quantizes for free in GEMM
  epilogues and the QKV path shares one input quantize across layers; the O projection has neither, so it has
  the worst quantize-to-GEMM ratio of any dense layer in the model. `gpu_sanity.py` replaces these assumptions
  with per-call milliseconds and prints the projected Δtime.
- **Loss break-even.** At upstream's 164 ms per millinat (band 120 to 250), fp8 here may cost at most
  `−Δtime / 0.164` mnat: 4.3 mnat (2.8 to 5.8 across the band) at the best-case −0.7 s, and nothing at all if
  Δtime is positive. fp8 can only add noise to the loss, so `net_s < 0` needs a real time win first.

## Prior attempts

No record or upstream PR we could find tried fp8 on the attention output projection.
- `records/track_1_short/*/README.md`: every fp8 mention is the MLP, the lm_head or the packed QKV projection.
  ANVIL2's "FP8 everywhere it pays" (2026-08-30) lists the MLP forward and backward, the lagged weight-cache
  quantize, the fp8 lm_head cache and the packed QKV op; the output projection is absent, with no measurement
  or reason given. Whether it was tried and dropped or never tried is not recorded.
- Upstream PR text and comments searched (2026-11-11): 344, 349, 358, 360, 363, 371, 382, 387, 393, 396, 397. None
  mentions it. The closest remark is PR 363 (FP8 MLP input gradient): its dW GEMMs "stay BF16 because they need a
  transposed FP8 copy", which is the cost that dominates here too.
- Nothing newer than 2026-11-08 upstream touches this path (the only newer PR, 402, is Track 3).

## Running it (8xH100)

```bash
python 182/fp8-attn-o/gpu_sanity.py --json fp8o_sanity.json     # ~2 min, one GPU: compile + CUDA graph, ms per call
python -m pytest 182/fp8-attn-o/tests -q -rs                    # test_gpu_attn_o.py runs; interpreter files skip
FP8_ATTN_O=1 torchrun --standalone --nproc_per_node=8 train_gpt.py                             # static arm
FP8_ATTN_O=1 FP8_ATTN_O_SCALES=delayed torchrun --standalone --nproc_per_node=8 train_gpt.py   # delayed arm
FP8_ATTN_O=1 HADAMARD_SAT_STATS=1 torchrun --standalone --nproc_per_node=8 train_gpt.py        # one run: are the static scales safe?
```

Decision rule: paired A/B, record vs `FP8_ATTN_O=1`, same seeds, fixed steps; `net_s = Δtime_s + 0.164 x Δval_mnat`
must be below 0 beyond its confidence interval. If `gpu_sanity.py` projects a positive Δtime, the A/B is only
worth running for the loss number.

CPU (this is what has been run):

```bash
python -m pytest 182/fp8-attn-o/tests -q -rs                                         # host: scales, reference, config
bash 182/tools/triton_cpu/run.sh python -m pytest 182/fp8-attn-o/tests -q -rs        # Triton interpreter: kernels, Function, GPT state
git show bfa81d4:track_1_short/model/attention.py > base_attention.py                # flags-off identity in the container
FP8_ATTN_O_BASE_ATTENTION=/repo/base_attention.py bash 182/tools/triton_cpu/run.sh python -m pytest 182/fp8-attn-o/tests/test_interp_attn_o.py -q
bash 182/tools/triton_cpu/run.sh aot                                                 # sm_90a compile; record kernels byte-identical
```

## What is verified, and what is not

Verified on CPU:
- The Function's forward, dx, dW and `d o_scale` match a pure-torch fp8-emulation reference and sit within fp8
  tolerance of the bf16 autograd result; a power-of-two `o_scale` scales every output bit-exactly; saturation
  clamps instead of producing NaN.
- The quantize kernels' bytes (both layouts, both formats, amax), the weight caches for all seven slots
  including the half-V copies, the delayed-scale refresh, the sat-stats rows and the post-warmup reset, on a
  CPU-built GPT under the Triton interpreter.
- Flags off: no `_attn_o_*` buffer exists, `attention.py` with `o_fp8=None` is bit-identical to the base commit's
  (forward, and backward on the bf16 training path), and the sm_90a PTX and SASS of the record's kernels are
  unchanged (`run.sh aot`).
- The new kernel variants compile for sm_90a.

Not verified (needs the node):
- `torch.compile` and the step's CUDA graph with the new custom ops and the buffer writes of the delayed arm
  (forward and backward). `gpu_sanity.py` part (i) and `tests/test_gpu_attn_o.py` check exactly this.
- cuBLAS fp8 `_scaled_mm` at these shapes with an e5m2 A operand and a `[768, T]` A for dW (the combinations exist
  in the record's QKV and MLP paths, but not at these exact shapes).
- The attention refresh CUDA graph with the extra launches (its self-check against the eager body runs at first
  use and would assert).
- Whether inductor really fuses the row quantize into the XSA / gate kernel and `g`'s producer.
- Every time number, the loss effect, and whether the static scales stay clip-free through training.
- Peak memory: two fp8 copies of `y` per attention layer are alive until the backward (about 75 MB per layer at
  49152 tokens), replacing the bf16 `y` the record saves.

## The butterfly question

"Did we try butterfly, or will that not work?" Two different things go by that name.

**(a) Butterfly (fast Walsh-Hadamard) factorization of the rotation in `HADAMARD_FP8`.** Not implemented, and
the numbers say it would not be cheaper. The rotation is a block dot of the tile with a ±1 matrix on the tensor
cores: 2b FLOP per element (256 at block 128, 64 at block 32). A butterfly needs log2(b) add/sub stages (7 at
128), which looks 36x cheaper in FLOP, but those are scalar fp32 operations and the dot is a `wgmma`.
`182/tools/triton_cpu/butterfly_probe.py` compiles both for sm_90a in Triton 3.6.0 (compile only, no timing):

| variant | compiles | shared memory | SASS instructions per thread per tile | barriers |
|---|---|---|---|---|
| no rotation | yes | 0 | 400 | 0 |
| dense dot, block 128 (current default) | yes | 64 KB | 520 (8 `wgmma`) | 5 |
| dense dot, block 32 | yes | 4 KB | 512 (8 `wgmma`) | 5 |
| butterfly, 7 stages (full 128) | yes | 16 KB | 1216 (448 adds, 96 shared-memory moves) | 31 |
| butterfly, 5 stages (block 32) | yes | 16 KB | 896 | 15 |
| H16 dot + 3 butterfly stages | yes | 32 KB | 864 | 39 |

So it is implementable (the earlier note that it would not fit shared memory was too pessimistic for a
standalone quantize kernel), but the full butterfly executes 816 instructions per thread on top of the plain
quantize, against 120 plus eight `wgmma` for the dense dot, and synchronizes the warps 31 times per tile instead
of 5, because Triton lowers each stage's pairing to a layout conversion through shared memory. The tensor core
does the 256 FLOP per element in roughly the time of one hundred scalar instructions per thread; seven butterfly
stages cost several hundred. Three quarters of the rotation's estimated cost is also in the MLP kernel's
epilogues, which already sit at the 227 KB shared-memory limit, where another 16 to 32 KB would cost a pipeline
stage. The cheap lever already exists: `HADAMARD_BLOCK=32`. Whether the butterfly is really slower in wall time
is unmeasured; instruction counts are the only evidence.

**(b) Butterfly-structured weight matrices (Dao et al. 2019, 2020) instead of dense MLP weights.** Not tried, on
purpose; `MONARCH_MLP` (Dao et al. 2022, `182/structured/README.md`) is the same research line's GPU-friendly
successor and is what we have. At the record's MLP (768 to 2816 and back, 4.33 M weights and MACs per token per
layer):

| | parameters per MLP | per-token work | GEMM shape |
|---|---|---|---|
| dense (record) | 4.33 M | 4.33 M MAC | `[T, 768] @ [768, 2816]`, k = 768: tensor cores, fp8 |
| Monarch, default (4 blocks, expand 2) | 1.77 M | 1.77 M MAC | batched `[T, 192] @ [192, 384]`, k = 192: tensor cores |
| butterfly, padded to 4096 (768 and 2816 are not powers of two) | about 0.2 M (2 n log2 n per matrix) | about 0.2 M MAC, in 24 sequential stages | 2 x 2 blocks, k = 2 |

Three reasons it is a poor fit here. Capacity: about 20x fewer parameters than dense and 9x fewer than Monarch,
when Monarch's 2.45x cut is already expected to cost more loss than it saves in time. Hardware: `wgmma` needs
k >= 16, so 2 x 2 blocks never touch the tensor cores; each of the 24 stages is a strided scalar pass over a
`[T, 4096]` activation (403 MB in bf16 at 49152 tokens, so about 19 GB of traffic per layer forward unless stages
are fused in registers, which runs into the layout conversions measured in (a)), while the dense fp8 forward is
4.3e11 FLOP, well under a millisecond of tensor-core time. Optimizer: ANVIL orthogonalizes each matrix, and the
polar factor of a 2 x 2 block is close to a sign update; per-block Muon on small blocks is already the known
weak point of the Monarch arm.
