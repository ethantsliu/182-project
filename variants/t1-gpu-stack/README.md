# variants/t1-gpu-stack: upstream PR #397, plus open GPU-only PRs behind flags that are off

Base: upstream PR #397 (atveit; open, head `11699e31`, unchanged since 2026-10-31): #393 (MoE + Gluon kernels on
#391's CPLM and n-gram normalization) plus #396's four exact levers. The author reports 30.6746 s, n=12, mean val
loss 3.27743 on a RunPod 8xH100 node. **Nothing in this folder has run on a GPU here.** The only number from our
node so far is the root record at 40.93 s (n=1) against its official 39.9 s, so expect #397 to land a little above
30.67 s there too.

With no flag exported the code path is #397's. Every change of ours sits behind `HOST_PREP`, `DOC_COPY`,
`LAYER_ROPE` or `COPY_MIX` (`track_1_short/folds.py`), with three exceptions that do not change a run: the trainer
prints a `[folds]` line and a full-precision `[final]` line, `train_step` returns the batch it consumed, and the
attention fp8 quantize kernels take a gain argument that is compiled out when it is off (`HAS_GAIN` is a
constexpr). That was checked by reading `git diff` against the PR head, not by a run.

Two things differ from the author's timed runs. They carried the author's own off-by-default timing hooks, which
#397's submitted tree (and so this one) does not have; nobody has timed the stripped tree. And they used
`TRAIN_SEED` 3-7 and 101-107, while a plain run draws a random init.

## Run on the node

```bash
variants/t1-gpu-stack/182/run_node.sh check   # < 1 min, no GPU work: PASS / WARN / FAIL lines, each FAIL with its fix
variants/t1-gpu-stack/182/run_node.sh prep    # 9 data shards to the repo root, the pinned FA3 binary, the tokenizer file
variants/t1-gpu-stack/182/run_node.sh run     # one timed run of #397; ends in a RESULT line
```

The script works from any directory. It activates the repo root's `.venv-run` when no venv is active; that venv
comes from `bash 182/tools/node_setup.sh --mode venv` at the repo root, which `check` names as the fix when the
stack is missing. This tree needs the same stack as the root record: torch 2.10.0+cu128, triton 3.6.0 (with
`triton.experimental.gluon`), `kernels==0.16.1`, `libcudart.so.13`, and `CUDA_HOME` with `include/cuda_bf16.h`.
There is nothing to build.

`run` honours `SEED` (sets `TRAIN_SEED`), `STEPS` (sets `NUM_SCHEDULED_ITERATIONS`, default 981), `EXTRA_ENV="K=V K=V"`
or plainly exported flags, `LOG_DIR` (default `logs/node`), `RUN_TIMEOUT_S` (1800) and `DRY_RUN=1` (prints the
command and exits). `NPROC` must be 8: the trainer refuses any other world size. The first run compiles for about
7 minutes before its clock starts. The last line of stdout is

```
RESULT variant=t1-gpu-stack val_loss=3.278512 train_time_s=30.660 steps=1053 target=3.28 hit=1 log=<path>
RESULT variant=t1-gpu-stack status=failed reason=<one line>          # with a nonzero exit
```

`steps` is the steps trained (981 scheduled + 52 growth + 20 extension). `log` is a copy of the trainer's own log
(source, environment, every line); the console capture sits next to it. Every knob is listed at the top of
`182/t1-gpu-stack/node.py`.

## What the flags are

| flag | PR (author) | what it does | changes the model? |
|---|---|---|---|
| `HOST_PREP=all` (= `batchplan,cands`) | #387 (nikhil-lalgudi) | the document plan by cumulative sums and searchsorted instead of a per-document Python walk (`data.py`); the sampled-softmax candidate gather by integer indexing (`sampled_softmax.py`) | no: same batches, same arrays |
| `HOST_PREP=ngramhash` (by name only) | #387 | the host n-gram row hash as one CPU-compiled expression, primed before the clock (`ngram_table.py`) | no; adds a `torch.compile` call site |
| `COPY_MIX=1`, `COPY_MIX_FIT=consumed` (default) or `shard` | #376 (cyrusghane), fit source of #382 (krishaab) | at the final validation, `P' = (1-λ_b)P + λ_b[x=c]` with 31 bucket weights fitted on the clock on training tokens (`copy_mixture.py`) | eval only |
| `LAYER_ROPE=attn` | #385 (shikhar-srivastava) | depth-conditioned rotary gains on the attention input and residual, folded into the QKV columns (fp8 cache) and the O rows | yes |
| `LAYER_ROPE=1` | #385 | the same, plus the MLP input and residual gains as activation scales on the dense and MoE layers | yes |
| `DOC_COPY=1` | #378 (josusanmartin) | the next token after the longest earlier exact match in the document, fed in as an extra input (`doc_copy.py`) | yes |

Not here: #387's FA3 batch-axis change (it is #396's `fa3copy` lever, already on), the rest of #382 (its n-gram
scatter overlaps `prescatter`, its sampled head is #393's), #390 (the pre-clock barrier, already in #397), and the
CPU retrieval line (#367/#380/#381/#399), which lives in `variants/t1-cpu-ngram`.

## What to run first, and what each result means

```bash
variants/t1-gpu-stack/182/run_node.sh ab      # 12 runs: base, hostprep, copymix x 4 seeds, about an hour
```

`ab` runs `AB_ARMS` (default `base hostprep copymix`) at `AB_STEPS` (default `981`) for `AB_RUNS` seeds (default
4, seeds `AB_SEED_BASE`.. = 1..4). The arms of one seed run back to back and the order rotates each seed. An arm
is exactly its own flag: whatever flags the shell exports are cleared. Finished runs are kept, so the same command
resumes; `DRY_RUN=1` prints the plan. It ends with a table and one `VERDICT` line per arm.

1. **`base`: #397 itself, n=4.** This is the number that matters most, since nobody has timed this tree. Confirms:
   every run starts (the kernel asserts and graph self-checks pass), mean time within a few percent of 30.67 s,
   mean val loss near 3.2774. Kills: a kernel assert at startup (the stack is not the record's; see `check`), or a
   mean loss above 3.28.
2. **`hostprep`: time only.** The lever is exact, so the losses should match the base within seed noise and the
   verdict is the paired time difference with its 95% interval. KEEP only if the whole interval is below zero.
   #387 has no H100 timing. Both rewrites run on the host's main thread, which the CUDA-graph replay lets run
   ahead of the GPU, so the saving is real only where the host is the bottleneck; a null result is likely and is
   an answer. The author's node had a run-to-run sd of 0.017 s, so four pairs resolve about 0.05 s.
3. **`copymix`: read `copy_gain` and `copy_fit_ms`.** Each run prints the loss with and without the mixture from
   the same forward, so the gain has no seed noise. The fit is on the clock. The verdict prices the gain at 250 ms
   per millinat (#393: 999 to 981 steps cost about 3.0 millinats and saved about 0.75 s) and says KEEP only if
   that beats the fit. Expect little: CPLM already mixes in a learned pointer over the previous 8192 document
   tokens at validation, and #376 found 79% of its 3.1 millinats in matches more than 2560 tokens back, most of
   them inside that band. With `consumed`, the fit never sees a match more than 3071 tokens back (training
   documents are cut at 3072), so the long-distance buckets are thin. `AB_ARMS="base copymixshard"` fits on the
   head of the last train shard instead, as #376 did; those are training tokens the run never consumed, which #382
   avoided on purpose.
4. **`lropeattn`, then only if it helps `lrope` and `doccopy`: at equal loss.** These change the loss per step, so
   one step count cannot judge them:
   ```bash
   AB_ARMS="base lropeattn" AB_STEPS="951 966 981" variants/t1-gpu-stack/182/run_node.sh ab   # 24 runs
   ```
   With several step counts `ab` ends in `sweep_steps.py compare`: each arm's train time where its fitted loss
   crosses 3.2785, minus the base's, with a 90% bootstrap interval. KEEP only if the whole interval is below
   zero. The base half of this sweep is useful by itself: #397's mean is 3.27743, about 1 millinat under 3.2785,
   so the base alone should be able to drop a few steps.

Why `lropeattn` before `lrope`: #385 folded all four gains into weights, which costs nothing per token. Here the
MLP gains cannot be folded (five MLP layers are MoE, and the dense bank's gradient is written in place by
`bankinplace`), so `LAYER_ROPE=1` multiplies every MLP layer's input and output by a per-channel gain, forward
and backward, on up to 49152 x 768 activations per rank. Our estimate is a few milliseconds per step, which is
more than the 17 steps (about 0.5 s) #385 saved on #360. It is an estimate; nothing was measured. `attn` keeps
only the two sites that are still weight folds. How much of #385's gain those two sites carry is unknown: the PR
reports no per-site ablation. #385 also had 2 of 98 development runs end near 3.33, so read every run's loss, and
the `layer_rope ... gain rms` lines at the end of each log.

`doccopy` goes last: #378 measured -1.17 s on #360, before CPLM existed, and CPLM copies from the same matches.
`HOST_PREP=ngramhash` is in no arm: it adds a compile call site that the Track 1 rules may not allow, and on this
Mac it was faster at 49152 tokens and slower at 16384.

After the screen, a record attempt is a fresh pool at one step count, not the sweep runs:

```bash
python variants/t1-gpu-stack/182/t1-gpu-stack/sweep_steps.py fit variants/t1-gpu-stack/logs/node/ab/base --target 3.2785
```

`fit` gives the count whose predicted mean is the target and roughly how many runs p < 0.01 against 3.28 needs.

## Tested here, and what is not

On an arm64 Mac with CPU torch 2.12 and no GPU:

- `python 182/t1-gpu-stack/test_folds.py`: the batch plan equals #397's walk, the candidate gather equals
  `np.take`, the functional and compiled n-gram hashes equal `ngram_row_ids`, the copy matchers equal brute force,
  the mixture loss and fit equal their definitions, the LayerRoPE gain formula and fold gradients hold, and the
  equal-loss comparison recovers a known difference.
- `python 182/t1-gpu-stack/test_node.py`: the log parser on a real #397 record log (seed 3: 3.2785, 30.660 s, 1053
  steps), on truncated, crashed and NaN logs; `check`'s line format and exit code; `prep` and `run` dry runs;
  `run` end to end against a fake `torchrun` (success, crash, timeout with the process group killed); the `ab`
  plan, resume, flag clearing and verdicts; the `LAYER_ROPE` values.

Not verifiable without the node:

- that #397 starts and reaches its time on this stack and host (the Gluon kernels, the nvrtc CE kernels, the FA3
  binary, CUDA-graph capture);
- every flag's compiled or CUDA path: the QKV column-scale backward, the fp8 quantize gain path, graph capture
  and self-checks with the new parameters, the replicated-Adam handling of them, `LAYER_ROPE=attn` leaving the
  MLP gain rows without a gradient, the warmup precompile of the copy mixture's eval graph (the log must say
  `[copy-mix] final-validation eval graph compiled in the warmup` and show no recompile at the final validation);
- every time and loss effect. No speedup is claimed for any flag;
- `check` on Linux beyond its logic: the `nvidia-smi` query, the `libcudart.so.13` search and the HF cache layout
  that `kernels` 0.16.1 reads were written from the root's `node_setup.sh`, not run;
- that 9 shards are enough is taken from the root record, which trains more steps on the same stage sizes.
