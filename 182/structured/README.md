# 182/structured — Monarch MLP + fixed-basis optimizer ablations

These are cheap falsification experiments. **We expect both to come out negative.** The point is a clean, quick answer.
Nothing here has run on a GPU. Every number below is either arithmetic or from a CPU test.

## 1. Monarch MLP on Track 1 (record #92 model)

**What changed.** The `MONARCH_MLP=1` flag (default off, so the record path is unchanged) replaces each MLP slot's dense
`c_fc`/`c_proj` with Monarch factors (block-diagonal → perfect shuffle → block-diagonal, from Dao et al. 2022):

| file | change |
| - | - |
| `track_1_short/model/monarch.py` | new: shapes, `monarch_up`/`monarch_down`/`monarch_mlp`, init, per-block lr multipliers |
| `track_1_short/model/gpt.py` | `init_monarch_mlp`; `_mlp` takes `(weights, slot)`; the MLP fp8 path is gated on `use_fp8_mlp`; the fp8 MLP refresh becomes a counted no-op |
| `track_1_short/optim/anvil.py` | if a param has `matrix_lr_mul`, it is used as the per-matrix lr (the dense bank keeps its c_proj-parity rule) |
| `train_gpt.py` | one log line with the Monarch config |

- **Shape.** Blocks are `[q, p]` with `p = 768/k`, `q = e·p`. The first factor has `k` blocks and the second has `l = e·k`, so hidden = `768·e²`.
  The down projection uses the transposed structure and stores its factors transposed (like `c_proj` is stored), so every block in the bank has the same `[q, p]` shape.
  Because the shapes match, `mlp_bank` stays a single ANVIL parameter: `[12, 2(k+l), q, p]`, viewed as `12·2(k+l)` matrices.
  Sharding, CUDA graphs, tail averaging and the gathers all keep working unchanged.
  Equal block shapes require `768/hidden` to be a rational square, so hidden is **3072, not 2816**.
- **Defaults: `MONARCH_BLOCKS=4`, `MONARCH_EXPAND=2`.** Blocks are `[384, 192]`, 24 per slot, hidden 3072.
  That is 1.77M params and MACs per token per slot, against 4.33M dense, about 2.45× fewer.
  `MONARCH_BLOCKS=8` gives 4.9× fewer.
  `MONARCH_EXPAND=3` gives hidden 6912 at 3.54M per slot, a wide MLP with about the dense parameter count.
- **Per-block Muon.** ANVIL orthogonalizes each block on its own. For a block-diagonal factor, that is exactly Muon applied to the factor: the polar factor of a block-diagonal matrix is the block-diagonal of the blocks' polars (a CPU test checks this).
  It is **not** Muon applied to the composed MLP weight. Separately, orthogonalizing each attention head on its own, instead of the whole Q/K/V matrix, is known to hurt in this speedrun. So expect some of the loss hit to come from the optimizer as well as from lost capacity.
  Lr multipliers: `MONARCH_LR_MUL` (1.0) on every block, and an extra `MONARCH_DOWN_LR_MUL` (2.0) on the down blocks, mirroring c_proj's 2×. The dead slot 7 gets 0.
- **Init.** The up projection keeps the dense `c_fc`'s output variance. Only the *last* down factor starts at zero; if both down factors were zero, neither would ever get a gradient.
- **Precision and speed.** The Monarch MLP is plain bf16 `torch.einsum` plus transposes. It **bypasses the FP8 MLP kernel**. The QKV projections and the lm_head stay fp8; attention itself is bf16 FA3 either way. The fp8 MLP buffers are still allocated but unused. `HADAMARD_FP8=1` together with `MONARCH_MLP=1` fails at import, because there is no fp8 MLP to rotate. `FP8_OFF=mlp` together with it is redundant but harmless.
  So the measured step time is *not* what a tuned Monarch kernel would give. It will probably be slower than dense FP8 even though it does fewer FLOPs.
  Use the loss side as the result.

**A/B at fixed steps** (`182/structured/ab_fixed_steps.py`). Every arm runs on the same seeds at the record's step count, interleaved
seed by seed (seed 0 of every arm, then seed 1, ...) so node drift does not land on Δtime. Logs tagged `INVALID_FOR_RECORD` are skipped.
Every arm uses the root's timing, which, like upstream master, lacks #390's `dist.barrier()` before `t0`. Never mix in timings from a tree that has it.
The script reports, against the first arm:
- `Δloss_mnat`
- `loss→s = Δloss_mnat × 0.164 s` (164 ms per millinat is upstream's normalization convention: the rate that makes
  the ANVIL2 ablation table sum to its certified win; the band measured during development is ~120-250 ms/millinat,
  records/track_1_short/2026-08-30_ANVIL2/README.md. Treat `loss→s` and `net_s` as uncertain by that factor.)
- the measured `Δtime_s`
- `net_s` = the sum of the two

To beat the baseline, an arm needs `net_s < 0`. With the current bf16 implementation, read `loss→s` as the floor that a perfect kernel would have to win back.

```bash
# 8xH100, repo root, data in data/fineweb10B. Each run is ~7 min compile + ~40 s training.
python 182/structured/ab_fixed_steps.py --seeds 0 1 2 3 \
  --arm base: \
  --arm monarch_k4:MONARCH_MLP=1 \
  --arm monarch_k8:MONARCH_MLP=1,MONARCH_BLOCKS=8 \
  --arm monarch_e3:MONARCH_MLP=1,MONARCH_EXPAND=3
# optional lr check on the default shape:
python 182/structured/ab_fixed_steps.py --seeds 0 1 --arm base: --arm lr05:MONARCH_MLP=1,MONARCH_LR_MUL=0.5 --arm lr2:MONARCH_MLP=1,MONARCH_LR_MUL=2
# re-summarize existing logs:
python 182/structured/ab_fixed_steps.py --parse base=logs/ab_structured/base_seed0.txt,... monarch_k4=...
```

**Needs GPU verification:**
- `torch.compile(fullgraph=True)` and the step-graph capture with the einsum path.
- The ANVIL Triton `XTX`/`ba_plus_cAA` kernels on `[*, 384, 192]` banks (they use their non-768 tile branch).
- Peak memory.
- The fp8-refresh "mlp" CUDA graph (`perf/cuda_graphs/fp8_refresh_graphs.py`) captures an empty body under Monarch. CUDA may warn about capturing an empty graph; that should be harmless, but it is unverified.
- Whether the bf16 einsum is acceptably fast.

None of `gpt.py`/`anvil.py` can be imported on this Mac (they need Triton), so the integration has only been `py_compile`d. The Monarch math itself is unit-tested.

## 2. Fixed-basis optimizers on Track 3

Track 3 allows only optimizer, hyperparameter and init changes (rule 1: same data, batch and architecture), so these arms change
only the optimizer. Monarch is an architecture change and is Track 1 only. These arms are ablations on the 3250-step #36 baseline.
They are **not record candidates**: the accepted record is #46 at 2690 steps, and open PRs claim 2615 (#388), 2635 (#398, H200) and
2450 pooled / 2455 prospective (#395). A record needs fewer steps than the record with `(3.28 - avg)·√n >= 0.004`, i.e. a mean at or
below 3.2780 for n=4, 3.27859 for n=8, 3.2790 for n=16. It should also clear the pairwise rule against the prior record.

`182/structured/train_gpt_rotated.py` is `records/track_3_optimization/train_gpt_simple.py` (the result #36 baseline) with:
- the setup and training code wrapped in `main()`, so tests can import it
- `TRAIN_STEPS`
- inline optimizers (Track 3 rule 4 forbids importing third-party optimizers)

With no env vars set, it runs the baseline exactly: fused AdamW + Muon, 3250 steps.

- **(a) `ADAM_BASIS`** applies to the Adam-managed params (embed, lm head, 1-D params). Adam's moments live in a fixed orthogonal basis of the feature dim: `g' = g Q`, `u = adam(g') Qᵀ`. Options:
  - `identity`: the inline AdamW without rotation, an A/A control
  - `dct`: DCT-II
  - `hadamard`: Sylvester for `2^a`; Paley-12 ⊗ Sylvester for `12·2^a`, which covers 768 and 3072
  - `random`: seeded Haar orthogonal

  `ADAM_BASIS_PARAMS=all|2d|1d` picks which params rotate. A dim with no construction, or wider than 4096 (the 50304 vocab bias), stays in the standard basis.
- **(b) `HIDDEN_OPT=soap_fixed`** is SOAP with its Shampoo eigenbases replaced by a *fixed* `HIDDEN_BASIS` (default hadamard): `u = Q_L adam(Q_Lᵀ G Q_R) Q_Rᵀ`. It replaces Muon on the hidden matrices.
  `HIDDEN_BASIS=identity` makes it plain AdamW on the hidden matrices, which separates "Adam vs Muon" from "basis vs no basis".
  Defaults (untuned, SOAP-paper style): `HIDDEN_LR=0.003 HIDDEN_BETAS=0.95,0.95 HIDDEN_EPS=1e-8 HIDDEN_WD=0.1 HIDDEN_WARMUP_STEPS=250`.
- **Why there is no "Muon in a Hadamard basis" arm.** `polar(QLᵀ G QR) = QLᵀ polar(G) QR`, so Muon does not depend on the basis (a CPU test checks the NS iteration). Only elementwise preconditioners can see the basis.
- **Significance.** The trainer prints `(3.28 - avg)·√n >= 0.004: PASS/FAIL` over its `num_trials`.
  `182/structured/t3_significance.py` applies that rule per arm. Against the first arm, it also applies the README's pairwise rule: `((L_b - L_a) + ΔS/100·0.0045) / √(1/n_b + 1/n_a)`. A value ≤ −0.004 means the arm is significantly *worse*, which is the falsification outcome.

```bash
# 8xH100 (or 1/2/4 GPUs), repo root, once:  python data/cached_fineweb10B.py 20
T=182/structured/train_gpt_rotated.py; R="torchrun --standalone --nproc_per_node=8"
$R $T 4                                                   # baseline, n=4
ADAM_BASIS=identity $R $T 4                               # A/A: should match the baseline
ADAM_BASIS=hadamard $R $T 4
ADAM_BASIS=dct $R $T 4
HIDDEN_OPT=soap_fixed HIDDEN_BASIS=hadamard $R $T 4
HIDDEN_OPT=soap_fixed HIDDEN_BASIS=identity $R $T 4       # Adam on hidden, no basis
# each run prints its logs/<uuid>.txt; then:
python 182/structured/t3_significance.py base=logs/A.txt aa=logs/B.txt adam_h=logs/C.txt adam_dct=logs/D.txt soap_h=logs/E.txt adam_hidden=logs/F.txt
```

If `soap_fixed` loses badly at its default lr, sweep `HIDDEN_LR` over {0.0015, 0.003, 0.006} with n=1 before concluding.
SOAPFixedBasis computes every update on every rank (dense fp32 rotations, up to 3072²). That is slower in wall-clock time, which Track 3 does not score.

## Tests (CPU)

```bash
TORCH_COMPILE_DISABLE=1 python -m pytest -q 182/structured/test_structured.py   # 22 tests, ~50 s on an M-series Mac
```
On Apple Silicon, run it under the arm64 Python (`arch -arm64 python3 ...`). An x86 `timeout` wrapper will load the wrong-architecture torch.

**Monarch:**
- The dense equivalent matches an independently built `L_bd · P · R_bd`.
- Every output mixes every input.
- Down equals upᵀ when the factors are equal.
- bf16 forward/backward works, with zero output at init and a nonzero gradient into the zero factor.
- Init variance is correct.
- lr multipliers are correct.
- ANVIL can shard the bank 8 ways for every k ∈ {2,4,8,16}, e ∈ {1,2,3}.
- Per-block polar equals the polar of the block-diagonal factor.

**Bases:**
- Paley-12 is Hadamard.
- Hadamard, DCT and random bases are orthonormal for 2…3072, and inverting a rotation recovers the input.
- Unsupported sizes return None.
- Muon commutes with rotations.

**Optimizers:**
- Inline AdamW with the identity basis matches `torch.optim.AdamW`.
- Rotated Adam equals Adam on the rotated parameter, and the same holds for two-sided SOAPFixedBasis.
- The significance rule matches the README's examples.

**Smoke run:** 40 steps of the trainer's GPT at toy size (2 layers, dim 256, vocab 128) under each of 10 optimizer configs. Loss falls from 3.7 to about 1.48 in every config, and every parameter moves.
The CPU-only substitutions are an fp32 embedding and an fp32 Newton-Schulz, because bf16 matmuls are a slow reference path on arm64.
