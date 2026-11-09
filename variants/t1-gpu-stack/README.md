# variants/t1-gpu-stack: #397 + open GPU-only PRs, each behind an env var

Base: upstream PR #397 (@atveit), 30.67 s, n=12: #393 (MoE + Gluon, on #391 CPLM + n-gram normalization) plus #396's exact levers.
**With no new env var set, the code path is #397's.** The one exception is the attention fp8 quantize kernel, which
now takes an optional gain argument that is off, so the arithmetic is unchanged. Nothing here has run on a GPU.
The code is #397's submitted tree (= #396's, `EXACT_LEVERS=all` by default). #397's 12 timed runs also carried the
author's off-by-default timing hooks, which this tree lacks. Nobody has re-timed the stripped tree, so time #397 on your node first.

| env var | PR (author) | what it does | changes the model? |
|---|---|---|---|
| `HOST_PREP=batchplan,cands` (= `all`), `ngramhash` only by name | #387 (@nikhil-lalgudi) | exact host-side rewrites: the document plan by cumulative sums and searchsorted (`data.py`), the candidate gather by integer indexing (`sampled_softmax.py`), the n-gram row hash as one CPU-compiled expression, primed before the clock (`ngram_table.py`) | no (same tokens, same rows) |
| `DOC_COPY=1` | #378 (@josusanmartin) | document-local copy feature: the next token after the longest earlier exact match in the document is fed in as an extra input (`doc_copy.py`, unchanged) | yes |
| `LAYER_ROPE=1` | #385 (@shikhar-srivastava) | depth-conditioned rotary gains on the 4 branch sites. Attention: folded into the QKV columns (fp8 cache) and O rows, as in #385. MLP: **applied as activation scales** (normed input × gain, output × gain) on the dense and MoE layers (`model/layer_rope.py`) | yes |
| `COPY_MIX=1` (`COPY_MIX_FIT=consumed`, the default, or `shard`) | #376 (@cghane), fit source of #382 (@krishaab) | eval-time copy mixture `(1-λ_b)P + λ_b[x=c]` with 31 bucket weights, fitted **on the clock** by maximum likelihood on training tokens: the last ones consumed (#382) or the head of the last train shard (#376). Its eval graph is compiled in the warmup (`copy_mixture.py`) | eval only |
| (none) | #390 (@atveit) | `dist.barrier()` before `t0` | already in #397 |

What is not here: #387's FA3 batch-axis change, which is #396's `fa3copy` lever and already on. #382's other pieces:
its compact n-gram scatter overlaps #396's `prescatter` and its sampled head is #393's, and it is a monolithic trainer.
#367/#380/#381/#399 (CPU retrieval) are out by design.

## Overlaps worth knowing before you spend GPU time
- **CPLM already copies.** #397's CPLM (#379) mixes in a learned pointer over the previous 2048 (train) / 8192 (val) document
  tokens. #376 found 79% of its gain from matches more than 2560 tokens back, and most of those sit inside CPLM's 8192 band.
  So expect `COPY_MIX` to gain much less than its 0.0031. Its fit costs about 0.3–0.45 s on the clock (#376: 0.44 s on a
  slow 8×H100). It pays only if `copy_gain` × (steps per nat) × (s per step) beats `copy_fit_ms`. Each run prints both
  losses from the same forward, so one run measures the gain with no seed noise. `DOC_COPY` overlaps CPLM in the same way.
- **LayerRoPE is a partial port.** #385 folded the MLP gains into the dense bank's fp8 caches, via a custom autograd fold.
  That conflicts with `bankinplace`, and #385's base had no MoE layers. The activation version is mathematically the same
  model, except that the MoE router also sees the gained input. Its costs: extra elementwise ops, and the MLP input's
  static fp8 scale (2⁻⁴, saturating at |x| > 28) now applies after the gain. The gains start at RMS 1. Check the
  `layer_rope ... gain rms` line printed at the end of each run.
- **`HOST_PREP=ngramhash`** adds a new `torch.compile` call site for a CPU function (no inductor flags). Track 1's compile
  rule may not allow it, so `all` leaves it out, and the stack is not rule-clean with it on. `batchplan` and `cands` are
  plain numpy. All three run in the main thread, which the GPU's CUDA-graph replay normally runs ahead of. They save
  wall time only where the host is the bottleneck, and #387 has no H100 timing.
- **`COPY_MIX_FIT=consumed` never sees a match more than 3071 tokens back.** The aligned training loader cuts documents
  at 3072 tokens in the last stages, so the >2560 distance buckets (79% of #376's gain) are fitted on a thin slice. `shard`
  sees full documents, but it reads training tokens the run never consumes (#376), which #382 avoided for rule 1.
  One millinat is about 6 scheduled steps on this stack (#393: 999 → 981 steps cost about 3.0 millinats and saved about
  0.75 s), or about 0.25 s, so a 0.15–0.45 s fit needs `copy_gain` of about 0.0006–0.0018 just to break even.
- **#385's runs had outliers.** In development, 2 of 98 runs finished at 3.3244 and 3.3420. One such run sinks a
  p < 0.01 pool. The activation-scale MLP gains also add per-token elementwise passes (forward and backward) on every
  MLP layer, which #385's weight fold avoids. The step time has to be measured.

## Run (8×H100, repo root, the repo README's stack: torch 2.10 cu128, `pip install -r requirements.txt`, `python data/cached_fineweb10B.py 9`)
```bash
python 182/t1-gpu-stack/test_folds.py                       # CPU checks (the GPU box too: also compiles the n-gram hash)
torchrun --standalone --nproc_per_node=8 train_gpt.py       # = #397 (check: "[folds] HOST_PREP=none DOC_COPY=0 ...")
HOST_PREP=all LAYER_ROPE=1 DOC_COPY=1 COPY_MIX=1 torchrun --standalone --nproc_per_node=8 train_gpt.py   # everything
```
`[final] val_loss:… train_time_ms:…` is the full-precision result line; `COPY_MIX` adds
`val_loss_no_copy / val_loss_copy / copy_gain / copy_fit_ms / lam`.

## What to A/B (one flag at a time; interleave arms on one node; ≥4 runs/arm; #397 on this node first, since nodes differ by ~0.6 s)
Every arm is judged at equal loss against #397's own step reduction, swept over the same counts:
```bash
python 182/t1-gpu-stack/sweep_steps.py run --steps 951 966 981 --runs 4 --out sweeps/base --tag base   # no flags
HOST_PREP=all python 182/t1-gpu-stack/sweep_steps.py run --steps 951 966 981 --runs 4 --out sweeps/hostprep --tag hostprep
python 182/t1-gpu-stack/sweep_steps.py compare sweeps/base sweeps/hostprep sweeps/copymix sweeps/lrope --target 3.2785
```
`compare` gives each arm's train time where its loss fit crosses the target, minus the baseline's, with a 90% bootstrap
interval. Keep an arm only if the whole interval is below zero. The order below is cheapest first.
1. **Smoke:** one run of each flag alone at the default 981 steps. It must start (graph self-checks, the kernel
   asserts), finish, and print sane losses. `COPY_MIX` must print `[copy-mix] … compiled in the warmup` and no
   recompile at the final validation.
2. **`HOST_PREP=all` (batchplan,cands) vs none:** time only. Losses should match #397 to within run-to-run noise.
3. **`COPY_MIX=1`:** read `copy_gain` and `copy_fit_ms` off one or two runs and decide. With `copy_gain` below about
   0.0006, or below `copy_fit_ms` / 250 ms per millinat, it cannot pay for the fit; drop it.
4. **`LAYER_ROPE=1`** (only if the smoke run's step time and its `layer_rope ... gain rms` line are sane), **then
   `DOC_COPY=1`:** each with a step sweep. Each changes the loss per step, so compare at equal loss.
   ```bash
   LAYER_ROPE=1 python 182/t1-gpu-stack/sweep_steps.py run --steps 951 966 981 --runs 4 --out sweeps/lrope --tag lrope
   ```
5. **The combined stack** (whatever survived 2–4), swept to the target, then a pool at the chosen count:
   ```bash
   HOST_PREP=all LAYER_ROPE=1 DOC_COPY=1 python 182/t1-gpu-stack/sweep_steps.py run --steps 931 951 971 --runs 3 --out sweeps/stack --tag stack
   python 182/t1-gpu-stack/sweep_steps.py fit sweeps/stack --target 3.2785
   ```
   `sweep_steps.py` fits loss linearly in `NUM_SCHEDULED_ITERATIONS` over all runs and solves for the target mean (3.2785).
   It then reports the predicted time there and roughly how many runs p < 0.01 against 3.28 needs. On the #397 record
   logs it reads n=12, mean 3.27743, sd 0.00116, 30.675 s. That is ~0.001 nats of headroom to 3.2785, so #397 alone should
   lose some steps. Pick the count, then run a fresh pool of ≥12 at it. The sweep runs are not the record evidence.

## Tested here (arm64 Mac, CPU torch 2.12, no GPU), and what is not
- `py_compile` on every changed file.
- `test_folds.py`: the batch plan equals #397's walk (31 batches × 8 ranks, through the end of the shard). The candidate
  gather equals `np.take`. The functional n-gram hash equals `ngram_row_ids`. `copy_candidates` and #378's
  `doc_copy_features` equal brute force. The mixture sums to one, and `copy_mix_loss` equals the explicit −log P′.
  `copy_fit` equals a grid argmax. The consumed-token window is right. The CPU-compiled hash also matches, and on this
  Mac it runs 0.34 ms vs 1.24 ms at 49152 tokens but 0.40 vs 0.19 ms at 16384; measure it on the box's host CPU. The LayerRoPE gains equal #385's formula, and
  the fold gradient identities hold.
- **Untested (needs 8×H100):** everything compiled or CUDA. That covers the model with each flag, the QKV `col_scale`
  backward, the fp8 quantize gain path, the CUDA-graph capture and self-checks with the new parameters and ops, the
  warmup precompile of the eval graph (and that the final validation hits it without a recompile), the on-clock fit's
  cost, the replicated-Adam handling of the new parameters, and every speed or loss effect. No speedup is claimed.
