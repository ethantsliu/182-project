# optim-kernel: does a faster ANVIL kernel shorten the run?

Nothing here has run on a GPU. Everything is off by default; with the flags unset the root trainer is the record.

## What ANVIL's matrix work is (root tree, per rank, per step)

`anvil_cascade` (`track_1_short/optim/anvil.py`) runs once per bank on the rank's shard, batched over its matrices:
qk_bank 6 x 256x768 (wide), vo_bank 2 x 768x768, mlp_bank 3 x 2816x768 (tall), all bf16. After the twin-rail
momentum pre-pass (elementwise, inductor-fused) it does one Gram, a Frobenius normalization taken from that Gram's
trace, and six quintic maps. Each map is three GEMM-class kernels on the small side `s = min(rows, cols)`:
`A = Gram(X)` (s x s), `B = b*A + c*A@A`, `X = a*X + X@B`. That is 18 GEMM kernels per bank and 54 per step, about
135 GFLOP per rank (mlp_bank 104, vo_bank 24, qk_bank 7), in one CUDA graph per bank.

Already in the record, so not redone here:
- **Symmetric half-products.** `XXT`, `XTX` and `ba_plus_cAA` (Triton) compute one triangle of blocks and mirror it.
- **Small-side Gram.** Tall banks use `X^T X` and right-multiply, wide banks `X X^T` and left-multiply.
- **Horner form.** `X(aI + bA + cA^2)` is evaluated as `B = bA + cA^2`, then `aX + XB`: two products on A, one on X.
- **Batching.** Each bank is one batched launch per kernel. The three banks have different shapes and their
  gradients land at different times, so they cannot share a batch.
- **Trace reuse.** The norm comes from the first Gram's diagonal; X and that Gram are rescaled, not recomputed.
- **No copies.** Output buffers are swapped between rounds, and mlp_bank splits `baddbmm` to avoid its input copy.
- **Launch overhead.** One CUDA graph replay per bank; the scalars are read from one device buffer.

## The alternative: `ANVIL_KERNEL` (same update rule, Gram-space evaluation)

A map is `X <- X P` with `P = aI + B`, and P is a polynomial in the Gram A, so the next Gram is `P A P` and needs
no access to X. A group of g consecutive maps is evaluated on s x s matrices only, accumulating `Q = P_1 ... P_g`;
X is touched twice per group (one Gram, one `X @ Q`). The normalization is folded into the first Q.

| name | groups | small products | mlp_bank large kernels (record: 6 Grams + 6 GEMMs + 6 add passes) | mlp_bank GFLOP (record 104) |
|---|---|---|---|---|
| `gram222` | 2+2+2 | cuBLAS `baddbmm` | 3 Grams + 3 GEMMs | 81 |
| `gram222s` | 2+2+2 | new symmetric Triton kernel (`perf/kernels/anvil_chain.py`) | 3 + 3 | 71 |
| `gram33` | 3+3 | cuBLAS | 2 + 2 | 74 |
| `gram33s` | 3+3 | symmetric Triton | 2 + 2 | 60 |

`ANVIL_BANKS` picks the banks (default `mlp_bank`). For vo_bank (square) the Gram space is as large as X, so there
is nothing to save; qk_bank's matrices are tiny either way. The bench measures all three.

One group of all six maps would be cheapest but is not offered: the Gram is stored in bf16, its small eigenvalues are
rounding noise, and four or more maps composed without re-reading X amplify it (relative error above 1e4 on a
sigma ~ 1/i spectrum; `test_chaining_all_six_maps_is_the_documented_failure`).

**Numerical agreement** (CPU, fp32-accumulate / bf16-store emulation of the kernels, real shapes; relative Frobenius
error). The alternatives are not bit-identical to the record: they are a second bf16 evaluation of the same rule, and
differ from the record by about as much as the record differs from the rule in fp64.

| spectrum | bank | record vs fp64 rule | gram222 vs fp64 rule | gram33 vs fp64 rule | gram222 vs record (Frobenius / max element) |
|---|---|---|---|---|---|
| sigma ~ i^-0.5 | mlp_bank | 0.62% | 0.74% | 0.75% | 0.95% / 2.0% |
| sigma ~ i^-0.5 | vo_bank | 0.48% | 0.76% | 0.71% | 0.89% / 1.7% |
| sigma ~ i^-0.5 | qk_bank | 0.60% | 0.98% | 0.94% | 1.1% / 1.8% |
| sigma ~ 1/i | mlp_bank | 3.4% | 1.3% | 2.2% | 3.6% / 10% |
| sigma ~ 1/i | vo_bank | 2.6% | 1.1% | 2.0% | 2.8% / 7.2% |
| sigma ~ 1/i | qk_bank | 2.2% | 1.1% | 2.3% | 2.5% / 5.3% |

The `s` variants are within 0.3 points of their cuBLAS twins. "max element" is max |alt - record| / max |record|.

**Expected gain.** If FLOP-bound, mlp_bank saves 23-44 GFLOP and seven full-size elementwise passes per step: about
0.1-0.2 ms, 0.1-0.25 s over 1194 steps if the bank update is fully exposed. Every kernel here is under one wave of
CTAs on an H100, so the cost may be latency-bound, and the Gram-space version launches more (small) kernels: a null
or a small loss is as likely. The variant tree's fused pre-pass (`OPTIM_MOEFUSED`) was projected at 0.03-0.08 s.

## The upper bound: `ANVIL_ABLATE=skip_matrix`

Measurement only, **not a valid training run**. The Grams, polynomials and update GEMMs are skipped: the update is the
Frobenius-normalized momentum (the cascade with zero maps, so no direction moves more than in the record). Rails,
equalizer, decay, comms and graphs stay. Its train_time against base is the latency the cascade exposes: the ceiling
for any optimizer kernel. Val lines carry `INVALID_FOR_RECORD(anvil_ablate)`; ignore its val and net_s.

## On the node

```bash
python 182/optim-kernel/bench.py --json kbench.json           # 1 GPU, ~1 min: KBENCH / KBENCH_TOTAL lines
python -m pytest 182/optim-kernel/test_optim_kernel.py -q     # CPU tests + the GPU tests (Triton kernel, graph capture)
XAB_ONLY=anvilkernel bash 182/gpu_session.sh xab              # base vs gram222 vs skipmatrix, 4 seeds, ~56 min
ANVIL_KERNEL=gram222 bash run.sh                              # one run by hand (ANVIL_BANKS=qk_bank,vo_bank,mlp_bank for all)
bash 182/tools/triton_cpu/run.sh aot --base e33c471           # flags-off check (Docker): expect 93/93 identical
```

Read the bench first: `update.mlp_bank.<kernel>` is the bank update as the trainer replays it (CUDA graph), record
vs kernel; `ablate.<bank>` gives the matrix share; `KBENCH_TOTAL kind=matrix_share` is the per-step ceiling. If no
`update.mlp_bank.*` shows speedup > 1, drop the kernel arm and run only `skipmatrix` (edit the arms in
`182/experiments.d/anvilkernel.sh`). `rails_equal=0` would mean the alternative's momentum pre-pass is not
bit-identical to the record's (it is the same text, compiled separately).

Laptop: `python3 182/optim-kernel/bench.py --device cpu --smoke` and the same pytest command (GPU tests skip).

## Unverified (no GPU, no Triton, no Docker on the authoring machine)

- Every time. Only the error columns above were measured, and on a CPU emulation of the kernels' rounding.
- `anvil_aS_plus_SB_kernel` (the `s` variants) has never been compiled; it is `ba_plus_cAA_kernel` with a second
  operand. `gram222` / `gram33` use only the record's kernels and cuBLAS.
- That the flagged bank update captures and passes the optimizer graph's bitwise self-check (the GPU test does the
  same capture / replay / compare), and that `torch.compile` accepts the two small pre-pass functions.
- `aot --base` was not run here. No existing kernel file was edited (the test pins `anvil_cascade`'s and
  `polar_express.py`'s sha256), and with the flags unset the only new code on the step path is one host-side
  set-membership test per bank update, which runs at graph capture and not at replay.
- That `skip_matrix` trains stably enough to finish (it should: the updates are bounded by the record's).
