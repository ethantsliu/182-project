# hadamard-fp8: how much does FP8 cost, and can block-Hadamard rotation win it back?

Folded into `main` at the repo root, on top of upstream record #92 (ANVIL2, 39.9 s). **Nothing here has run on a GPU.**
CPU-verified: the pure-torch Hadamard reference, the bf16 loss, the dump format and the offline script
(`python -m pytest 182/hadamard-fp8/tests -q`: 23 pass, the GPU file skips). Every Triton change, every
training-path change and every torch.compile / CUDA-graph interaction is **untested**.

## Decision rule

Run the bf16 upper bound first. **If the bf16-vs-fp8 val-loss gap at a fixed step count is < 0.5 millinat
(0.0005), Hadamard is pointless: drop it.** Rotation can at most recover that gap. Estimate the gap from
paired runs (same `TRAIN_SEED`, fp8 vs `FP8_OFF=1`), at least 4 seeds, and report the mean paired
difference with its standard error. Track-1 records report run-to-run std of 0.7-2 mnat, so a single
pair can't resolve 0.5 mnat. Use the 182/tools A/B runner + t-test.
- Upper end of the 95% CI below 0.5 mnat: stop.
- Gap clearly above 0.5 mnat: attribute it (`FP8_OFF=mlp`, `=qkv`, `=lm`), dump tensors, run the offline
  script, and run the `HADAMARD_FP8=1` arm only if the script shows the rotated gradient GEMMs gaining
  meaningfully (output rel-MSE ratio well below ~0.9) on the part that carries the gap.

The prior is that it is pointless. e4m3/e5m2 are float formats, so with per-tensor scales the relative
rounding error is about the same whatever the scale. Outliers only hurt through saturation (static
scales) or flush-to-zero. The synthetic smoke test shows exactly this: 60x outlier channels at an
amax scale give rotated/plain error 0.99 (`test_rotation_barely_matters_for_fp8_at_amax_scale`).
Rotation only wins big where a static scale saturates (`test_rotation_avoids_static_scale_saturation`).
The record's scales were tuned not to saturate.

## What changed (all behind env vars; defaults = the record)

| knob | effect |
|---|---|
| `FP8_OFF=1` (or `all`, or a list like `mlp,qkv,lm`) | Runs those training GEMMs in bf16 with the same math otherwise. **MLP**: `BF16LinearReLUSquareFunction` (same forward kernel as validation; dW2, dpre, dW1, dx as bf16 GEMMs; post-lambda via `scale_add`, not folded). **QKV**: `PackedBF16QKV` (bf16 projection, the same fused norm/rotary/V backward kernel with a new `FP8_OUT=False` bf16 output). **lm**: `perf/bf16_loss.py`, plain-torch slabbed softcapped MTP + prefix CE with a bf16 lm_head GEMM and the CE kernel's exact gradient convention, also on sampled-softmax steps via a new `SampledLoss.candidates` view. The fp8 caches are still refreshed, just not read. Attention itself stays bf16 FA3 either way. Slower; compare val loss at fixed steps only. |
| `HADAMARD_FP8=1` (`HADAMARD_SEED`, default 1234) | Block-128 random-sign Hadamard rotation along the contraction dim for the four MLP gradient GEMMs, fused into the existing quantize paths (`perf/kernels/hadamard.py`, `mlp.py` epilogues): dW2 (post_t from the forward epilogue, g_t along tokens), dpost (g_row, W2 row cache along 768), dW1 (dpre_t from the backward epilogue, x_t along tokens), dx (dpre row, W1 col cache along hidden). The forward GEMMs are untouched, and so are their numerics: the rotated post_t has its own delayed scale (`post_t_scale`/`post_t_amax`), the forward's post scale sees only the unrotated post. Scales: g stays static; dpre keeps one delayed scale whose amax covers both its layouts; the rotated weight caches get their own exact-then-delayed scales; x_t gets a new delayed scale (`FP8_XT_HEADROOM` 1.10). A token-rotated x at the static 2^-4 can saturate (see the `dW1 xt-static` row of the script). QKV and lm_head are **not** rotated (next step if MLP shows a gain). Requires per-rank tokens, 2816 and 768 to be multiples of 128 (asserted). |
| `HADAMARD_GEMMS=dw1,dw2,dx,dpost` (subset; default all) | Only the listed gradient GEMMs read rotated operands; an unlisted one reads the record's operands on both sides with no extra dot, kernel or weight-cache write (dW2: forward post_t epilogue + g_t; dpost: g_row + W2 row cache; dW1: backward dpre_t epilogue + x_t; dx: backward dpre row epilogue + W1 col cache). Inert without `HADAMARD_FP8=1`. |
| `HADAMARD_BLOCK=32\|64\|128` (default 128) | Rotation block. Kernels keep 128-wide tiles and rotate each block-diagonally (reshape + `[R*128/b, b] @ [b, b]` dot), so the epilogue dots cost 2b instead of 256 FLOP/element. A factored full-128 rotation (H16 ⊗ H8) was not implemented: the H8 butterflies need shared-memory layout round trips in Triton and K=8 is too small for a bf16 wgmma (see `perf/kernels/hadamard.py`). |
| `FP8_DUMP_DIR=dir` (`FP8_DUMP_STEPS=100,500,1000`, `FP8_DUMP_ROWS=8192`, `FP8_DUMP_SLOTS=0,5,10`) | Saves `dir/stepNNNNNN.pt` with bf16 x / g / W for those MLP slots and attention layers (`perf/fp8_dump.py`). At those steps, rank 0 runs one extra **eager** forward+backward of the uncompiled model: no gradient sinks, results dropped, every fp8 amax buffer restored. The full softmax is used even on sampled steps. The clock is paused around it (`t0` moves forward) and all ranks barrier. A dump run is an analysis run, not a timing run: every val line it prints ends in `INVALID_FOR_RECORD(fp8_dump)`. The hooks check `FP8_DUMP.active()`, which short-circuits on `torch.compiler.is_compiling()`, so compiled graphs are unchanged. |

`HADAMARD_SAT_STATS=1` (analysis only; val lines get `INVALID_FOR_RECORD(sat_stats)`): at every validation,
one `fp8 sat` line per MLP fp8 scale (post, dpre, w1, w2; with HADAMARD_FP8 also post_t, x_t, w1_rot,
w2_rot). Each line gives the per-slot ratio of the last step's amax to its fp8 limit (max over ranks;
`SAT[slots]` marks ratios > 1.01, i.e. clamped values), the scales, and `STUCK[slots]` for a delayed scale
still at its init value, i.e. an amax that never arrived. Ignore STUCK at step 0.

`gpu_sanity.py` (one GPU, < 3 min, `python 182/hadamard-fp8/gpu_sanity.py`): HADAMARD on vs off in two
subprocesses. Reports (i) the stage counts from prime_stage_cache and the regs/spills/smem of every
compiled variant, (ii) the compiled + CUDA-graph amax check, and (iii) per-kernel ms per MLP call at
T = 16k/32k/40k/48k, projected over the 1194-step schedule. `HADAMARD_GEMMS` / `HADAMARD_BLOCK` are
passed through.

Also added: `perf/hadamard_ref.py`, the pure-torch reference and the convention for "rotate both
operands along k". The CPU tests prove (A R)(Rᵀ B) = A B, the dW token-rotation identity, and all four
MLP gradient GEMMs in the kernels' exact layouts. `fp8_quant_error.py` is the offline study described
in its docstring. `tests/test_gpu_kernels.py` checks every new Triton path against the reference, and
the bf16 paths against autograd and the fp8 kernels.

## FA3 FP8 attention: investigated, not implemented

From upstream `hopper/flash_api.cpp` / `flash_attn_interface.py` (Dao-AILab main):
- The forward takes e4m3 q/k/v on SM90 with optional `q/k/v_descale` of shape (batch, heads_k). The
  output must be bf16. `flash_attn_varlen_func` passes the descales through. Varlen and local
  (sliding-window) attention are not blocked for fp8 at the API level (only build-time disable flags).
- **The backward rejects fp8**: `mha_bwd` checks `q_type == kFloat16 || kBFloat16`
  ("FlashAttention only support fp16 and bf16 data type"). Training would therefore need bf16 q/k/v
  for the backward anyway, with an fp8 forward whose saved LSE doesn't match the bf16 recompute.
- Head dims: the fp8 dispatch has d ∈ {64, 96, 128, 192, 256} and only one differing-V path
  (d ≤ 192, dv ≤ 128 → the 192/128 kernel). Nothing explicit covers the record's (qk 64, v 128) layers.
  Also, the record runs a patched third-party FA3 build (`devenpzak/flash-attn3-12864`) for its mixed
  widths, so an fp8 forward would need yet another patched build.

Not clearly supported for training, so per the brief there is no flag.

## Commands (8xH100, from the repo root)

```bash
pip install pytest && python -m pytest 182/hadamard-fp8/tests -q          # GPU tests run here too
HADAMARD_FP8=1 python -m pytest 182/hadamard-fp8/tests/test_gpu_kernels.py -q -s   # prints error tables

# 1. the gap: paired seeds, fp8 (record) vs bf16 upper bound, full schedule
for s in 0 1 2 3; do
  TRAIN_SEED=$s torchrun --standalone --nproc_per_node=8 train_gpt.py
  TRAIN_SEED=$s FP8_OFF=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
done
grep -h "val_loss" logs/*.txt     # last line of each run: step:N/N val_loss:...

# 2. only if the gap is >= 0.5 mnat: attribute it, dump, analyse
TRAIN_SEED=0 FP8_OFF=mlp torchrun --standalone --nproc_per_node=8 train_gpt.py   # also =qkv, =lm
TRAIN_SEED=0 FP8_DUMP_DIR=dumps/fp8 torchrun --standalone --nproc_per_node=8 train_gpt.py
python 182/hadamard-fp8/fp8_quant_error.py --dump dumps/fp8 --json dumps/fp8_err.json   # CPU is fine

# 3. only if step 2 shows a real gain for the MLP gradient GEMMs
for s in 0 1 2 3; do TRAIN_SEED=$s HADAMARD_FP8=1 torchrun --standalone --nproc_per_node=8 train_gpt.py; done
```

## Estimated HADAMARD overhead per run (8xH100, not measured)

Round-1 method: the measured +3.17 s at all@128 split by share (backward dpre epilogue 50%, forward post_t 25%,
g + x_t quantizes 15% split by FLOP, rotated weight caches 2.5%, 7.5% unattributed and assumed to be the
lost pipeline stage of the rotated variants); the dot parts scale with b/128, the cache writes do not. The low
end drops the unattributed 0.24 s. Small blocks are optimistic: part of each rotation is a layout conversion
of the bf16 tile that does not shrink with b.

| HADAMARD_GEMMS | MFLOP/token @128 | @128 | @64 | @32 |
|---|---|---|---|---|
| all (round 1) | 30.1 | 2.9-3.2 s | 1.5-1.7 s | 0.8-1.0 s |
| dw1,dw2 | 20.0 | 1.9-2.1 s | 1.0-1.2 s | 0.5-0.7 s |
| dw2 | 10.1 | 1.0-1.2 s | 0.5-0.7 s | 0.24-0.48 s |
| dw1 | 9.9 | 0.9-1.2 s | 0.5-0.7 s | 0.24-0.47 s |
| dx | 7.9 | 0.8-1.1 s | 0.4-0.7 s | 0.24-0.48 s |
| dpost | 2.2 | 0.20 s | 0.12 s | 0.08 s |

Break-even is ~0.1-0.3 s (see the decision rule), so only dpost alone, or a single GEMM at block 32, comes close.
`182/hadamard-fp8/gpu_sanity.py` measures the real numbers.

## Still needs GPU verification / risks

- Interpreter-tested (triton-interp image, `tests/test_interp_kernels.py`): every rotated kernel variant for
  blocks 32/64/128, the Function end to end for six HADAMARD_GEMMS/BLOCK settings, and byte-identical record
  kernels with the flags off. That checks arithmetic and indexing only, not layouts, smem or registers.
- None of the Triton code has been compiled. Watch for: `tl.dot(tl.trans(tile), Q)` layout conversions
  inside the persistent `tl.range(flatten=True)` epilogue; extra shared memory (the stage-stepping retry
  in `prime_stage_cache` covers OutOfResources); and register spills from the four extra 128³ dots in
  the dpre epilogue. The HADAMARD step time is unmeasured and almost certainly slower: g's row half is no
  longer fused into its producer, the weight caches are written twice, and there are the epilogue dots.
- `FP8_OFF` under torch.compile + CUDA graphs: the slabbed bf16 CE unrolls 4096-row slabs (≈12 at the
  largest batch); memory per slab is about 0.8 GB per fp32 [4096, 50304] temporary. Not measured.
- The dump's extra eager pass at batch 24 (step 1000) may need a lot of memory beside the graph pool.
  Pick earlier steps if it OOMs.
- x_t amax: `hadamard_t_quant` is now functional (returns `(t, amax)`), and `GPT._quantize_mlp_input`
  folds the amax into `_mlp_xt_amax[slot]` with aten ops. CPU-tested eager and under
  `torch.compile(fullgraph, backend="aot_eager")` (`tests/test_hadamard_state.py`, in the triton-interp
  image). Inductor + CUDA-graph replay is GPU-untested: `gpu_sanity.py` part (ii) / `tests/test_gpu_state.py`.
- post_t has its own delayed scale (`_mlp_post_t_scales` / `_mlp_post_t_amax`), so the forward's e4m3 post
  keeps the record's scale. The arg contract with `mlp.py` is the comment at `FP8_XT_HEADROOM` in
  `model/gpt.py`. `GPT.init_mlp_fp8` asserts that `mlp.py` implements it.
- Offline numbers model the delayed scales with the tensor's own amax, not last step's. The synthetic
  mode is a smoke test, not evidence.

Credits: the fp8 MLP / QKV / CE kernels and every scale rule are record #360 (ANVIL2) and #344; the
bf16 MLP kernel is by @andrewbriand and @jrauvola; QK-norm by @Grad62304977. This adds only the
flags, the rotation and the analysis.
