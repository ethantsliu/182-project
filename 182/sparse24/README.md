# SPARSE24_MLP: 2:4 structured sparsity on the root record's MLP

Nothing here has run on a GPU.

## What

`SPARSE24_MLP=1` runs every MLP of the root record (11 live slots, `relu(x @ W1.T)^2 @ W2` with `W1 = c_fc` and
`W2 = c_proj`, both `[2816, 768]`) with 2:4 sparse weights on the H100's sparse tensor cores. It is off by default.
With the flag off the trained model and the timed path do not change. The code is in
`track_1_short/perf/sparse24.py`. The flag lines are at the bottom of `track_1_short/config.py`. The hooks are in
`track_1_short/model/gpt.py`: `init_mlp`, the refresh, `_mlp` and the two `forward` lines.

- **Which GEMMs go sparse.** Sparse tensor cores only accelerate `A @ B` with `A` 2:4 along the contraction
  dimension. Each sparse GEMM is therefore written as `z @ A.T = (A @ z.T).T`.

  | GEMM | sparse operand | sparse along | sparse? |
  |---|---|---|---|
  | fwd `x @ W1.T` | `W1` [2816, 768] | 768 | yes |
  | fwd `post @ W2` | `W2.T` [768, 2816] | 2816 | yes |
  | bwd `g @ W2.T` (dpost) | `W2` | 768 | yes |
  | bwd `dpre @ W1` (dx) | `W1.T` | 2816 | yes |
  | bwd `dpre.T @ x`, `post.T @ g` (dW1, dW2) | none: they contract over tokens | | dense bf16 |

  Both `W` and `W.T` must be 2:4, which requires a **transposable mask**: every aligned 4x4 tile keeps at most
  two entries per row and two per column (Hubara et al. 2021). Four of the six MLP GEMMs go sparse. Making the dW
  GEMMs sparse would mean sparsifying the gradient (the MVUE estimator of Chmiel et al., which Hu et al. 2024
  use). That is out of scope here.
- **Gradient.** Straight-through: the dW GEMMs give the gradient with respect to the masked weights, and ANVIL
  applies it to every dense weight, pruned or not. This is torchao's `SemiSparseLinear` recipe; the PyTorch blog
  reports that the dense gradient worked better than masking it. SR-STE's masked decay is not used. Under ANVIL
  the update is orthogonalized, so an SR-STE decay term has no obvious place to go.
- **Mask schedule.** The mask is recomputed from `|W|` at every refresh, which runs after every optimizer step in
  the slot where the record writes its fp8 weight caches. The weights move every step, so the compressed
  operands have to be rebuilt every step anyway, and refreshing the mask adds nothing. Refreshing every step lets
  the mask follow the weights from step 0. A fixed mask chosen at init is what the PyTorch blog found clearly
  worse.
  - `SPARSE24_MASK=greedy` (default) is torch's `_sparse_semi_structured_tile` rule. It visits each tile's 16
    entries by `|w|` descending and keeps one while its row and column hold fewer than 2. It can stop at 7.
  - `optimal` takes the best of the 186 maximal 4x4 patterns by kept `|w|`: the 90 with exactly two per row and
    column, plus 96 seven-entry ones. Hu et al. search the 90. The seven-entry ones are included because one of
    them sometimes beats every exact pattern (`test_seven_can_beat_every_exact_pattern`).
  - On Gaussian `[2816, 768]` weights, the kept share of `sum |w|` / `sum w^2` is: plain row-wise 2:4 (not
    transposable) 0.745 / 0.868, transposable optimal 0.703 / 0.820, transposable greedy 0.692 / 0.820
    (`test_kept_energy_ordering`).
- **Dense tail.** `SPARSE24_DENSE_FROM_WS=N` makes the MLP dense bf16 from the first stage whose long window is at
  least N blocks, and in validation too. It is stage-aligned, so each compiled graph is either sparse or dense.
  - 13: the 20-step extension stage only.
  - 11: stages 2-4, so sparse for 42% of the tokens.
  - 0 (default): sparse throughout, so the evaluated model is the 2:4 model.

  The tail is dense **bf16**, not the record's fp8. The fp8 MLP's delayed activation scales are never fed while
  the MLP is sparse, so switching back to fp8 mid-run would start from a garbage scale.
- **Numeric format.** The sparse arm is bf16 2:4 and the record is fp8 dense, so the A/B compares two numeric
  formats as well as sparsity. The `FP8_OFF=mlp` arm (bf16 dense, same math) separates the two. Training runs no
  fp8 2:4: torch exposes fp8 cuSPARSELt only as the raw `_cslt_sparse_mm`, with no fused relu^2 or quantize
  epilogue. `gpu_bench.py` times it at the GEMM level to show what a custom kernel could reach.
- **Backends.**
  - `SPARSE24_BACKEND=cusparselt` (default) calls `torch._cslt_sparse_mm` behind the opaque custom op
    `sparse24::cslt_linear`, whose fake gives torch.compile the output shape. The four compressed operands per
    slot (W1, W1.T, W2, W2.T, about 2.4 MB each) are non-persistent `_mlp_sp24_*` buffers. The fp8 refresh CUDA
    graph treats them as outputs, like the fp8 caches they replace.
  - `dense` is the same masked math as dense GEMMs on masked weight copies. It is the CPU test path, and a
    loss-only emulation on a GPU.
- **Incompatible flags.** `HADAMARD_FP8` and `MONARCH_MLP` are rejected at import. `FP8_OFF` is redundant for the
  MLP and harmless.

## Prior art (checked 2026-11-12)

- In-tree: no 2:4, semi-structured or cuSPARSELt code or mention in `track_1_short/`, `variants/` or any
  `records/track_1_short/*/README.md`. The repo's "sparse" hits are the sparse bigram gradient and sparse comms.
- Upstream `KellerJordan/modded-nanogpt`: searching PRs for "sparsity" and "2:4" finds nothing on structured
  weight sparsity.
- Literature:
  - Hu et al. 2024, "Accelerating Transformer Pre-training with 2:4 Sparsity" (ICML; arXiv 2404.01847):
    transposable masks by convolution, MVUE for dW, masked decay on gradients, and a dense fine-tune at the end.
  - The PyTorch blog of 2024-11-13: torchao `SemiSparseLinear`, a runtime prune+compress kernel (now
    `torch._sparse_semi_structured_tile`), and dense straight-through gradients. It measured 1.3x on ViT-L MLP
    linears fwd+bwd on A100, but only 6% end to end. A100 GEMM 13008x4096x1024 took 538 us dense vs 387 us 2:4.
    DINOv2 ImageNet was 82.8 dense, 82.7 at 70% sparse-then-dense, and 82.3 at 100% sparse.
  - Haziza et al. 2025, "Accelerating Transformer Inference and Training with 2:4 Activation Sparsity" (arXiv
    2503.16672): squared-ReLU activations are 84-98% sparse, so they sparsify the activations instead. They
    report fp8 2:4 GEMMs 1.5-1.7x over dense fp8 on H100 "depending on the matrix shape", up to 1.3x per FFN, a
    dense first 1k steps, and loss parity at 1.5B / 63B tokens (emulated). This record's MLP is squared-ReLU too,
    so activation sparsity is the better-founded idea. It needs a custom 2:4 kernel with a sparsifying epilogue
    and is not implemented here.

## Cost model (not measured)

Record shapes: 1194 steps, per-rank token-steps 41.08 M. MLP: 6 GEMMs x 2 x 768 x 2816 x 11 slots = 285.5 MFLOP
per token, 1.17e16 FLOP per rank over the run.

- **What the record's MLP costs.** At an assumed 780-1300 TFLOP/s effective for the fused fp8 kernels, the MLP
  takes 9-15 s of the run, centre 12 s, or about 2 s per GEMM type. This is an assumption, not a measurement.
- **Sparse forward and input-gradient GEMMs (4 of 6).** These are bf16 2:4 against fp8 dense.
  - Assumptions: fp8 runs 1.3-1.8x faster than bf16 at these shapes, and 2:4 bf16 runs 1.1-1.5x faster than
    dense bf16. The 2:4 speedup is below the paper numbers because K = 768 compresses to 384, and the published
    1.3-1.7x figures are for larger shapes.
  - Result: sparse bf16 time / fp8 time = 0.87-1.64, centre 1.19. That is **-1.0 to +5.1 s, centre +1.5 s**.
- **dW GEMMs (2 of 6) drop from fp8 to bf16.** **+1.2 to +3.2 s, centre +2.2 s.**
- **Unfused elementwise.** The record computes relu^2 and the fp8 quantizes in its GEMM epilogues. Here relu^2,
  the dpre product and their bf16 round trips are separate passes. One bf16 `[T, 2816]` pass costs 2.55 TB per
  rank over the run, about 0.85-1.0 s at 2.5-3 TB/s. Assuming 1.5-3 extra passes after Inductor fusion: **+1.3 to
  +3.0 s, centre +2.0 s.**
- **Refresh.** The refresh prunes and compresses 24 matrices per step (4.3 MB each, two compressions per call),
  instead of the record's 24 fp8 quantizes. **+0.1 to +0.6 s.**
- **Expected Δtime: +1.6 s to +11.9 s, centre +6 s (slower).** It is slower in every corner of the ranges. Even
  the best case, where 2:4 bf16 beats fp8 dense by 13%, loses to the dW GEMMs moving to bf16 and to the lost
  epilogue fusion.
- **Expected Δloss: +10 to +40 mnat, centre +20 mnat, at sparse-throughout.** This is a guess. Half the MLP's
  weights are gone in every forward, and the transposable greedy mask keeps about 69% of `sum |w|` on Gaussian
  weights. The token budget is short (1194 steps), and ANVIL's orthogonalized update does not know about the mask.
  The literature's near-parity results come with long dense phases or emulated activation sparsity. The 20-step
  dense tail (`=13`) may recover a few mnat. The `=11` tail recovers more but runs dense bf16 for 58% of the
  tokens, which is slower still.
- **Net.** At 164 ms/mnat (band 120-250): **+9 s centre, +3 s to +22 s range.** There is no loss budget: Δtime is
  positive before any loss cost, so this arm cannot pay.
- **fp8 2:4, the version that could pay.** It needs a custom kernel, not built here.
  - With the 4 sparse GEMMs in fp8 2:4 at 1.1-1.6x over fp8 dense, they save 0.7-3.0 s.
  - Without a fused relu^2 and quantize epilogue, the elementwise passes give back 1.3-3.0 s, and the refresh
    adds 0.1-0.6 s. That is a Δtime of -1.6 to +2.9 s, centre about +0.5 s.
  - With the same +10-40 mnat, the net breaks even only in the single best corner. It needs a fused 2:4 fp8
    CUTLASS/Gluon kernel before it is worth a node run.

`gpu_bench.py` replaces the four central assumptions with measurements at the record's shapes: fp8 vs bf16, 2:4
vs dense, whole-call times and the refresh. It prints the projected Δs over the schedule and the loss budget it
buys.

## How to run

```bash
python 182/sparse24/gpu_bench.py --json sp24_bench.json                 # ~2 min, one GPU: the gate
python -m pytest 182/sparse24/tests/test_gpu_sparse24.py -q -rs        # cuSPARSELt path, compile + CUDA graph
SPARSE24_MLP=1 torchrun --standalone --nproc_per_node=8 train_gpt.py                              # sparse throughout
SPARSE24_MLP=1 SPARSE24_DENSE_FROM_WS=13 torchrun --standalone --nproc_per_node=8 train_gpt.py    # dense extension stage
FP8_OFF=mlp torchrun --standalone --nproc_per_node=8 train_gpt.py                                 # bf16 dense control
SPARSE24_MLP=1 SPARSE24_BACKEND=dense torchrun --standalone --nproc_per_node=8 train_gpt.py        # loss-only fallback
```

The session worker's `xab` phase reads `182/experiments.d/sparse24.sh`. It runs three arms (record, bf16 dense
control, sparse) with 4 seeds each.

Decision rule:
- Run the A/B only if `gpu_bench.py` projects `calls_delta_s` below 0. If it projects a positive Δs, as expected,
  the A/B is a loss study only. It is then worth running only to calibrate the loss cost of 2:4 for an fp8-kernel
  follow-up. The `SPARSE24_BACKEND=dense` arm answers the same loss question if cuSPARSELt misbehaves.
- With paired seeds: `net_s = Δtime_s + 0.164 x Δval_mnat` against the record. `Δval(sparse) - Δval(FP8_OFF=mlp)`
  is the cost of sparsity alone.

CPU checks: `python -m pytest 182/sparse24/tests -q` covers the masks, the Function against plain autograd, the
refresh, the fake-tensor shape, the aot_eager compile and the bench projection math. `bash
182/tools/triton_cpu/run.sh python -m pytest 182/sparse24/tests -q` adds the real-GPT state cases in the Triton
interpreter on torch 2.10.

## Unverified

- That torch 2.10's `_cslt_sparse_mm` with `transpose_result=True` returns `z @ A.T` in the layout
  `cslt_linear` assumes. `SPARSE24_CSLT_FUSE_T=0` is the fallback: transpose the result with a copy.
- That `_sparse_semi_structured_tile(use_cutlass=False)` output is valid cuSPARSELt input on sm_90. torch's own
  comment marks this "TODO".
- That cuSPARSELt calls are CUDA-graph capturable inside the compiled step graph and the fp8 refresh graph.
  `test_compiled_and_graph_captured` checks this on the node.
- Whether torch's greedy kernel breaks ties the way `mask_tile_greedy` does. `test_torch_greedy_tile_matches_emulation`
  checks this.
- Every number in the cost model, and the loss estimate.
- The end-to-end trainer run with the flag on. The CPU state cases cover the model's state and dispatch, not a
  compiled multi-GPU step.
