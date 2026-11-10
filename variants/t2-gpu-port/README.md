# variants/t2-gpu-port — Track 2 (GPT-2 Medium, val ≤ 2.92), GPU only

This is the rule-clean hedge for Track 2: no CPU retrieval, only small-track features ported into
`train_gpt_medium.py`. It starts from **PR #261** (@Lisennlp, MUDD connections, 14.34 min), which
already includes #252 (@JOD-code: bigram hash, fused FP8 CE, unified NorMuonAndAdam).

**Nothing here has been run on a GPU.** Every flag defaults to off, and with all defaults the
numerics are #261's. The only default-path changes are a no-op prefix-target gather plus two
kernel arguments, and the run log now also includes `triton_kernels.py` and `medium_extras.py`.

## What changed (all behind env vars)

| Flag | What it does | Ported from |
|---|---|---|
| `CANON_MASK=1` | At the final validation, masks the (prev, cur) token pairs GPT-2's tokenizer never emits. This can only lower the loss. Rank 0 builds the 316 MB bit mask in a forked child **on the clock** (6.5 s of CPU on an M-series Mac), then it is waited on, copied and broadcast before the clock stops. Intermediate validations run unmasked. | Track 1 #91, PR #350 @jvarho; background build from #92, PR #360 @DevenPzak (`medium_extras.py`, verbatim) |
| `VAL_CE_CHUNKS=N` | Number of row chunks in the validation CE (default 4, as #261). Per-token losses are identical. Use 8 with `CANON_MASK`, which adds a mask gather to every logit block. | — |
| `PREFIX_WEIGHT=w` | Prefix-token prediction aux loss: an extra CE term toward the longest proper-prefix token of the next token, added inside the fused CE kernel (`triton_kernels.py`). Schedule: `w` until `PREFIX_HOLD_FRAC` (0.2854), then 0.8w, 0.6w, 0.4w, 0.2w in equal steps until `PREFIX_END_FRAC` (0.6063), then 0. That is track 1's 0.25 → 0.20/0.15/0.10/0.05 → 0 at its stage fractions. The table is built on the clock, one bucket per rank, then `all_reduce(MAX)`. | Track 1 #88, PR #337 @jvarho; bucketed build from #92 |
| `XSA=1`, `XSA_LAYERS=all\|i,j,..` | Learnable per-(layer, head) exclusive self-attention: `y -= tanh(α)·(⟨y,v⟩/‖v‖²)·v`, with α starting at zero, Adam (0.9, 0.95), lr_mul 1, no wd. Medium has no paired layers, so `all` means all 16. | Track 1 #82, PR #264 @djdumpling; algebraic rewrite #86, PR #317 @aryavohra |
| `ATTN_GATE=0` | Removes the sparse attention gate (`attn_gate_bank`), as #86 did once XSA was in. | Track 1 #86, PR #317 |
| `FUSED_MLP=1` | Runs the MLP through `triton_kernels.FusedLinearReLUSquareFunction` (already in this tree, unused by #261): `x @ W1.T` with the relu² epilogue in one kernel, and `2·(g @ W2.T)·relu(pre)` fused in the backward. Medium forked from track 1 (#188) before this kernel landed, so it still runs three inline ops. Same math in bf16. | Track 1 #59, PR #197 @andrewbriand @jrauvola (1.820 → 1.781 min there, ~2%) |
| `COORD_DESCENT_TUNING=1` | `torch._inductor.config.coordinate_descent_tuning`. The README explicitly allows it on the medium track. #252 measured ~5 s; #261 shipped with it commented out. It adds a lot of compile time (the code comment says 30+ min). | #252 |
| `NUM_SCHEDULED_ITERATIONS` (4125), `NUM_EXTENSION_ITERATIONS` (25), `COOLDOWN_FRAC` (0.70), `MUON_LR` (0.015), `ADAM_LR` (0.004), `VAL_LOSS_EVERY` (250) | Step and LR retune knobs. | — |

The kernel change copies master's `track_1_short/perf/kernels/cross_entropy.py` prefix logic into
medium's older kernel, which keeps its bf16 sigmoid cache and `grad_scale`. The S_w term, the
prefix thread `n_predict` and the term2 collision handling are the same as master's. The prefix
arguments go at the end of `FusedSoftcappedCrossEntropy.apply`, so `train_gpt.py`'s existing
8-argument call still works.

### What was verified (CPU only, `python 182/t2-gpu-port/test_cpu.py`, ~17 s)

1. **Canonical mask soundness.** 228k tokens of real text (repo READMEs and .py files, plus
   unicode and contraction edge cases) contain **0** masked adjacent pairs. 2.80% of all pairs are
   masked. EOT/BOS is never masked. The torch bit-unpack used in eval matches numpy's `bitorder="little"`.
2. **Prefix table.** The 8 per-rank buckets max-reduce to exactly the full table. Spot checks
   confirm each entry is the longest proper prefix.
3. **CE kernel logic.** A Python transliteration of the edited kernel matches autograd for loss
   and dlogits under MTP plus prefix: the dense pass, then sparse overwrites for duplicate MTP
   targets and for a prefix target that collides with an MTP target.
4. **Autograd.** Surplus `None` gradients are accepted, so the 13-gradient backward is fine for
   8-argument callers.
5. **Prefix schedule shape.**
6. **Model smoke test.** The real `GPT` and `TrainingManager` source is exec'd on CPU at d=128,
   with flash-attn and the fused CE replaced by torch references. Forward and backward run under
   every flag and the combined stack, and every parameter gets a gradient. Optimizer
   `param_table`, `scatter_order` and `work_order` cover the parameter labels exactly. XSA at
   α=0 is bit-identical to base, and α≠0 changes the output. An all-zero mask is a no-op, and a
   sound random mask lowers the eval loss. Chunking does not change the loss.
7. **`FUSED_MLP`.** The real `FusedLinearReLUSquareFunction` autograd code (exec'd from
   `triton_kernels.py`, with a torch stand-in for the Triton kernel that asserts its 2D contiguous bf16
   contract) matches the inline MLP's loss and every parameter gradient up to bf16 rounding.
8. **`check_canon_val.py`** reads a fineweb `.bin` shard and finds an injected masked pair. On the box,
   `DATA_PATH=../.. python 182/t2-gpu-port/check_canon_val.py` counts the masked adjacent pairs in the
   real first 10,485,760 val tokens (~10 s, no GPU). It must print 0 before any `canon` run.

### What is NOT verified (needs the 8xH100 box)

- The edited CUDA kernel has not been compiled (no nvcc here). The first GPU run is the compile test,
  and the default path uses the edited kernel too. Its prefix logic is line-for-line master's
  `track_1_short/perf/kernels/cross_entropy.py`, which compiles and runs on H100.
- `FUSED_MLP`'s Triton kernel has never run at d=1024/4096 (track 1 ran it at 768/3072). The block
  sizes divide both shapes and shared memory does not depend on K, but only a GPU run proves it.
- Whether the `canon_mask` gather/shift/`masked_fill` fuses under Inductor in the eval graph.
  Without fusion, a 65k-row chunk materializes ~13 GB of fp32 logits plus a bool mask. #261's eval
  peak is unknown and its training peak is 60 GB. Watch for OOM at the final validation, and raise
  `VAL_CE_CHUNKS` if needed.
- Fork-on-clock of the mask builder alongside medium's data-loader threads. Master does the same
  thing, but not with this loader.
- All loss and speed effects. No speedup is claimed.

## Ablation plan (8xH100)

```bash
cd variants/t2-gpu-port
pip install -r requirements.txt && (cd ../.. && python data/cached_fineweb10B.py 25)  # ~1.9B tokens (+BOS-alignment waste)
python 182/t2-gpu-port/test_cpu.py                                   # sanity check on the box
DATA_PATH=../.. python 182/t2-gpu-port/check_canon_val.py            # must print "0 masked"
182/t2-gpu-port/ablate.sh base 1      # first: compiles the edited kernel; confirm val ≈ 2.917, ~860 s
```

`ablate.sh` cd's into this folder and defaults `DATA_PATH` to `../..`. It sets `VAL_LOSS_EVERY=0` for
every arm (#261 used 250). Use the default for a record submission.

Each run takes ~14.3 min of training plus ~7 min of compile/warmup, so about 22 min (much more with
`COORD_DESCENT_TUNING=1`). The plan is ordered by expected value per GPU-hour. The step estimates
come from #261's own logs: the end-of-run slope is 7.4e-5 loss/step at 207 ms/step, so 0.001 of
loss is worth ~13.5 steps, about 2.8 s.

| # | Arm | Runs | Expectation from track 1 | What decides it |
|---|---|---|---|---|
| 0 | `base` | 2 | #261: 2.9171 ± 0.0007, 860.5 s | Node reproduces the record. |
| 1 | `canon` | 2 | −0.0017 loss at 3.28, so maybe −15 to −25 steps (~3–5 s) | Final-val delta. It is eval-only, so the masked run is the same run as base except at the last validation. Check the eval doesn't OOM. |
| 2 | `cdt` | 1 | ~−5 s (#252) | Step time only. Costs the compile. |
| 3 | `prefix` vs `prefix_s` | 2 each | −15 steps of 1375 at 3.28, ~1% | Medium's MTP only runs for the first 1/6 of training, so also try a shorter prefix window (`prefix_s`). |
| 3b | `fmlp` | 2 | #59: ~2% of step time at d=768, so a guess of 5–15 s here (at d=1024 the GEMMs grow faster than the elementwise traffic the fusion saves) | Step time only, at the same loss. |
| 4 | `xsa`, `xsa_nog` | 2 each | #82: −30 steps of 1440. #86: dropping the gate made each step faster at about the same loss | Loss delta and step time (XSA adds elementwise work). |
| 5 | `stack` | 3 | Sum of what survived 1–4 | Should stay additive. |
| 6 | `stack_steps` | 5–8 | Spend the gain: `NUM_SCHEDULED_ITERATIONS=4125 − Δsteps`, then confirm p < 0.01 | #261 already used up most of #252's budget (2.9171 vs 2.92). With sd 0.0007 and n=6, p < 0.01 needs a mean ≤ ~2.919, so only ~0.002 of slack is left (~25 steps) before any new feature. |

Summarize with `python 182/t2-gpu-port/summarize.py runs/`. It prints mean, sd, p(<2.92), time,
and the steps the loss gain buys versus `base`. For the LR retune (priority 6 in the brief), sweep
`MUON_LR` ∈ {0.013, 0.015, 0.017} and `COOLDOWN_FRAC` ∈ {0.65, 0.70, 0.75} on the `stack`
config with 1 run each, only after step 6 settles the feature set. `182/tools` has the
t-test/nsys tooling.

## Not ported (yet), in priority order, with notes

- **FP8 MLP up/down/backward and packed FP8 QKV (#84, #89, #90).** Medium hard-disables FP8 in
  `CastedLinear` ("requires tuning of scales") and runs the MLP as inline bf16 `F.linear`. The
  port needs master's `perf/kernels/mlp.py`, its delayed-scaling amax, and new
  d=1024 scale constants. Master's static x scale of 2^-4 is justified by max|x| ≤ √768 < 448·2^-4 = 28
  after RMS norm. At d=1024, √1024 = 32 > 28, so it needs 2^-3 here. It is the largest likely speed win (d=1024 MLPs are more compute-bound
  than d=768), but it cannot be checked without a GPU, so it was skipped rather than shipped broken.
  `182/hadamard-fp8` covers the FP8 numerics question.
- **Paired-head attention (#58).** Needs paired YaRN tables and an interleaved q/k layout. With
  medium's 8 heads × 128 it is a real architecture change, so do it after XSA results come in.
- **Medium-native features dropped by #188:** Snoo and EMA optimizer wrappers (#128/#129/#137),
  Smear-MTP (#151), the cubic window schedule (record 8), and sharded mixed-precision Muon
  (record 7). Snoo/EMA would wrap the unified `NorMuonAndAdam`, which is not a small diff.
  The cubic window schedule is cheap, but every new window size is another FA3 compile.
- **Hashed n-gram table and sampled softmax (#92).** At d=1024, 84.6M rows is 173 GB in bf16, or
  21.7 GB per rank. With #261's 60 GB peak, that does not fit in 80 GB. It would need about half
  the rows (~11 GB per rank) or a 768-wide table projected up. This is a stretch goal and was not started.
