# optim-cost: what the ANVIL cascade costs per step on t1-gpu-stack, and three flags that could shave it

Nothing here has run on a GPU.

## What the optimizer already does

The orthogonalizing step is ANVIL's cascade (`track_1_short/optim/anvil.py`): a Frobenius normalization, then four
quintic maps (Polar Express family). Each map is one Gram (`X^T X` or `X X^T`), one polynomial `b*A + c*A@A`, and one
update GEMM `a*X + X@B`. Four banks run it: qk_bank, vo_bank, mlp_bank, and moe_bank (the experts). Before looking
for gains, here is what earlier PRs already did:

- **Batched per bank.** Each rank updates one contiguous shard of each bank. Every kernel of the cascade runs over the
  whole shard as a batch (the leading dim): 6 qk matrices, 2 vo, 2 mlp, 10 moe. There are no per-matrix launches, so
  idea (a), batching, is already done within each bank. Batching across banks would merge different shapes (256, 768,
  1024 and 2816 rows) whose gradients land at different times (separate reduce-scatters, in `work_order`). The
  merged batch would have to wait for the last of them. Not done.
- **One CUDA graph per bank update** (`perf/cuda_graphs/optimizer_graphs.py`, record #360's "tgr", measured −0.92 s).
  Launch overhead is gone.
- **Symmetric kernels** (`perf/kernels/polar_express.py`) compute only the upper-triangle blocks of the Gram and the
  polynomial. **Hand-fused tail** (`perf/kernels/anvil_opt.py`, #393): the rails, Nesterov blend and bf16 cast run as
  one pass, and the Gram and copy kernels are tuned. All banks use it except moe_bank. Its margin is 1.1 and the
  fused normalization bakes 1.05, so it runs the inductor `anvil_cascade`.
- **Six maps cut to four** (#393), at equal val: −0.19 s over the run. This is idea (b) done once already, and it is
  the only measured number for the cascade's cost.
- **Overlap.** The bank updates run in `work_order` on the compute stream, each right after its own reduce-scatter
  lands. qk and vo go first, while the n-gram exchange and the big reduce-scatters are still in flight. The gathers
  are deferred to the next step's fp8 refresh (`perf/deferred_gathers.py`). What can still be exposed is the end of
  the chain: moe_bank's update after its reduce-scatter, then mlp_bank's. Idea (c), moving that work into the next
  forward, does not fit. The forward's first MLP layer reads mlp_bank's fp8 cache, which the refresh builds from the
  gathered update.

## The numbers (`cost_model.py`)

Per rank per step, four maps:

| bank | shard | shape | path | cascade GFLOP executed | Gram CTAs (132 SMs) |
|---|---|---|---|---|---|
| qk_bank | 6 | 256×768 | wide | 4.4 | 60 |
| vo_bank | 2 | 768×768 | wide | 15.7 | 84 |
| mlp_bank | 2 | 2816×768 | tall | 46.3 | 84 |
| moe_bank | 10 | 1024×768 | tall | 97.6 | 210 |

That totals 164 GFLOP of GEMM work (after the symmetric-block savings), plus about 800 MB of elementwise traffic and
a dependency chain of about 80 kernels. Assuming 250-450 TFLOP/s on these mid-size batched bf16 GEMMs, 2.7 TB/s
streaming, and 2-4 µs per dependent kernel, the bank updates take **0.8-1.3 ms of GPU time per step**. That is
**2.8-4.3% of the 29.4 ms step** (30.957 s / 1053 steps). The cascade alone is calibrated by #393's six-to-four cut:
0.19 s / 2 maps / 1053 steps = **~90 µs per map round**, about 0.36 ms per step for four maps. That is **~0.38 s per
run, 1.2% of the step**. The model predicts 91-164 µs per round, so the measured rounds look nearly fully exposed.
moe_bank does about 60% of the cascade work. It runs on the unfused path and sits at the end of the chain.

## The flags (all off by default; `track_1_short/optim/cascade_flags.py`)

| flag | arm | what | Δtime expected | judged by |
|---|---|---|---|---|
| `OPTIM_PROFILE=1` | `optprof` | CUDA events around each label's reduce-scatter wait and update inside `AnvilAndAdam.step`. Prints `[optim-profile] rank=.. tail=.. \| moe_bank w=.. u=..` every 100 steps (window mean[run mean], ms) | ~0 (about 40 event records per step, host side) | the log lines |
| `OPTIM_MOEFUSED=1` | `optmoefused` | moe_bank moves onto the fused tail (`pre_fused` + `polar_flagged` at margin 1.1) | −0.03 to −0.08 s | exact A/B, 4 seeds |
| `OPTIM_MAPS=3` | `optmaps3` | three maps `ANVIL_MAPS3`: one Gram/poly/update round less per bank | −0.07 to −0.12 s at equal steps; break-even +0.38 mnat (250 ms/mnat), +0.58 (164) | step sweep |
| `OPTIM_NORM=s4` | `opts4`, `opts4maps3` | divide by the Schatten-4 norm `‖XᵀX‖_F^½` instead of `‖X‖_F` | ~0 (one 768² reduction) | step sweep |

`OPTIM_MAPS=3` and `OPTIM_NORM=s4` send every bank through `polar_flagged`, so they include `OPTIM_MOEFUSED`'s path.
Their time effect should be read against `optmoefused`, not `base`.

**OPTIM_MOEFUSED** is meant to be exact. The kernels are the ones `anvil_opt.py` already showed byte-equal to the
stock chain for the other banks. The tall 1024-row non-split case uses `torch.baddbmm` in both paths, and the margin
enters as the same fp32 constant. The saving comes from the stock pre-pass: about 346 MB for moe_bank's rails against
about 157 MB fused. The 346 MB figure is our assumption, read off anvil_opt's note of three inductor kernels plus a
scratch copy of the rails. At 2.7 TB/s the difference is ~70 µs per step, plus two or three fewer kernels. Range
−0.03 to −0.08 s, if moe_bank's update is on the critical path. If `optprof` shows moe_bank waiting on its
reduce-scatter (large `w`), expect null.

**OPTIM_MAPS=3.** `ANVIL_MAPS3` was designed like `ANVIL_MAPS`: minimax of max|p(σ)−1| over normalized σ, bounded up
to σ = 1.02 (fit in `cascade_flags.py`'s comment; envelopes in `test_map_envelopes`). The shipped four maps hold
|p−1| ≤ 0.2388 from σ = 3e-3. Three maps hold the same only from σ = 1.134e-2. The small-σ gain falls from 276 to 73.
On synthetic spectra (`cascade_sim.py`, bf16), three maps match four on flat, Gaussian-tall, σ ∝ i^-0.5 and low-rank
spectra. On a heavy tail (σ ∝ i^-1) they leave **87% of the singular values below 0.5, against 48%**. The Schatten-4
scale does not rescue that (83%), because ‖X‖_F/‖X‖_S4 is only 1.26 on that spectrum. So the update changes wherever
the real velocity spectrum is heavy-tailed. We do not know the real spectrum: no trained velocity exists off the node.
**Expectation: null to negative.** A 0.1 s saving has to cost less than 0.4 millinats, and the six-to-four cut needed
new maps to hold val. Run it only if `optprof` shows moe_bank and mlp_bank with `w ≈ 0` (the cascade is exposed).

**OPTIM_NORM=s4** is not a speed lever by itself. ‖X^T X‖_F^½ = (Σσ⁴)^¼ still bounds σ_max, so the maps' range is
unchanged. It is tighter than ‖X‖_F, which lifts the small singular values by ‖X‖_F/‖X‖_S4 (1.3-5.3 on the synthetic
spectra). It exists as the companion that might let three maps work. Alone (`opts4`) it whitens more of the spectrum
at no cost: a loss-only change of unknown sign.

Orthogonality error ‖out − UVᵀ‖_F/‖UVᵀ‖_F after k maps (1024×768, σ ∝ i^-0.5, bf16 simulation). The intermediate maps
overshoot by design, so only the last column counts:

| schedule | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| F4 (shipped) | 0.819 | 0.455 | 0.479 | 0.161 |
| F3 | 0.821 | 0.492 | 0.138 | |
| S4+4 | 0.622 | 0.428 | 0.409 | 0.167 |
| S4+3 | 0.624 | 0.302 | 0.167 | |

The full table, all four bank shapes × five spectra × four schedules, comes from `python 182/optim-cost/cascade_sim.py`.
The ±0.23 ripple of the maps sets an error floor of ~0.14-0.22 even where every σ is covered.

## Not done, and why

- **(a) Cross-bank batching:** the banks are ready at different times, and each is already one batch.
- **Split-K for the under-one-wave Grams** (qk 60, vo 84 and mlp 84 CTAs on 132 SMs): at most ~20 µs per step
  (≤ 0.02 s per run), below what 4 seeds can resolve (~0.05 s).
- **Skipping mlp_bank's frozen padding matrices** (ranks 5-7 hold only padding): every rank still does the same work
  as ranks 0-4, which hold live matrices. The step waits for the slowest rank, so nothing is saved.
- **(c) Splitting moe_bank's reduce-scatter in two,** so the first half's cascade overlaps the second half's transfer
  (a strided copy of 63 MB, ~40 µs, plus a second collective). Worth up to ~0.15 ms per step, but only if moe_bank's
  `w` in `optprof` is large. Build it after that measurement, not before.
- **fp8 GEMMs in the early maps:** the cascade corrects singular values, not directions, so the rounding error of an
  fp8 round stays in the update.

## How to run

```bash
variants/t1-gpu-stack/182/run_node.sh check
AB_ARMS="base optprof optmoefused" variants/t1-gpu-stack/182/run_node.sh ab          # 12 runs, ~60 min (= xab optim)
grep '\[optim-profile\] rank=0 steps=1000' variants/t1-gpu-stack/logs/node/ab/optprof/*.txt
# only if moe_bank / mlp_bank show w ~ 0 and u >= 0.1 ms:
AB_ARMS="optmoefused optmaps3 opts4maps3" AB_STEPS="951 966 981" variants/t1-gpu-stack/182/run_node.sh ab   # 36 runs
python -m pytest variants/t1-gpu-stack/182/optim-cost/test_optim_cost.py -q          # on the node: the GPU bitwise test
```

The generic session phase reads `182/experiments.d/optim.sh`, which declares the same three exact arms.

## Tests here

`python -m pytest 182/optim-cost/test_optim_cost.py -q` on the host. The CPU tests check flag parsing and defaults, the
two schedules' envelopes, the cost model's shapes and tile configs against the source, the simulator's ordering, and
`StepProfile`'s accounting on fake events. Under `bash 182/tools/triton_cpu/run.sh` (Triton interpreter, repo root),
three more checks run: `polar_flagged` at the stock settings equals `polar` bit for bit; the fused moe path is close
to `anvil_cascade`; the flagged schedules match the simulator to bf16 tolerance. The moe check is bitwise only on a
GPU, where the reference is the compiled `anvil_cascade`. Under the interpreter the reference runs as eager ATen,
whose lerp rounds differently from the inductor lowering that `pre_fused` copies. There the rails differ by at most
0.39 ulp of their largest value, and the cascade output by 0.40% relative (1.6e-5 of the bf16 inputs flip one bit,
and the four maps amplify that). The interpreter also computes dots in fp64, so its bitwise check covers logic and
indexing only. The GPU run of the same test is the real exactness check.

## Unverified

- Every time number. The cascade's ~90 µs per round is scaled from #393's −0.19 s. The moe pre-pass bytes are
  estimated, not profiled.
- That `OPTIM_MOEFUSED` is bitwise equal to `anvil_cascade` on an H100 (the GPU test).
- That the bank graphs capture and self-check under the flags. They should: same kernels and buffers, and the
  normalization is a new torch.compile call site with four shape specializations.
- The loss effect of `OPTIM_MAPS=3` and `OPTIM_NORM=s4` on real velocity spectra.
- With the flags off, the trained model and the GPU work are unchanged. `step()` gains three
  `if prof is not None` tests per label (host only), and `anvil_bank_update` gains one module-constant test.
- The trainer's own banner still says "ANVIL 4-map cascade" under `OPTIM_MAPS=3`. The `[optim]` line printed by rank 0
  gives the true setting.
