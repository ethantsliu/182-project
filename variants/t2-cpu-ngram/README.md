# t2-cpu-ngram: PR #380's CPU-retrieval stack on Track 2 (GPT-2 Medium, val loss ≤ 2.92)

`variants/t2-cpu-ngram/` is a full runnable tree based on upstream PR #380. Credits: the exact-match retrieval is
PR #367 (hermabr). The exact-count chain, the smaller transformer and the extra hints are PR #380 (devenpzak). The
Track 2 sizes are ldmberman's adaptation, posted in the #380 thread on 2026-10-04: **val loss 2.8769 in 221 s**
(`step_avg` 38.1 ms, 5800 steps) on 8xH100. That is their run, not ours, and it is **one run** (seed 1337; they
report 2.8991 for the model before the exact-count chain). For scale, the official Track 2 record is 17.35 min and
the best open GPU PR (#261) is 14.34 min. **We have not run this tree on a GPU.**

## Run on the node
From the repository root, after `bash 182/tools/node_setup.sh` has made the root's `.venv-run` (this tree needs the
same stack: torch 2.10.0+cu128, triton 3.6.0, the pinned FA3 kernel, `libcudart.so.13`, CUDA headers):

```bash
bash variants/t2-cpu-ngram/182/run_node.sh check   # what is missing, one PASS|WARN|FAIL line each (no GPU work)
bash variants/t2-cpu-ngram/182/run_node.sh prep    # 103 shards (~20 GiB), the Rust extension, FA3 + tokenizer caches
bash variants/t2-cpu-ngram/182/run_node.sh run     # one timed run; the last line is RESULT variant=t2-cpu-ngram ...
```

- `check` exits 1 on any FAIL and each FAIL line carries its fix. A torch other than 2.10.0+cu128 or a triton
  other than 3.6.0 is a FAIL (fix: `bash 182/tools/node_setup.sh --mode venv`, which makes `.venv-run`). The node
  that ran torch 2.11 trained all 5800 steps and then ran out of memory, so `run` checks the stack even with
  `SKIP_CHECK=1`, and the trainer stops before touching a GPU on another stack. `ALLOW_STACK=1` overrides all
  three (unsupported). Most are fixed by `prep`. The ones `prep` cannot
  fix are the host's: fewer than 8 GPUs, no CUDA 13 runtime or headers, under ~390 GiB of available RAM, or a
  `/dev/shm` with under 144 GiB free (`sudo mount -o remount,size=200G /dev/shm`; in docker `--ipc=host`).
- `prep` is idempotent and resumable. It needs `cargo`; `INSTALL_RUST=1` lets it install rustup. Limits:
  `PREP_TIMEOUT_S` (14400, the download) and `BUILD_TIMEOUT_S` (3600).
- `run` repeats the check, then runs `train_gpt_medium_cpu.py` on 8 GPUs with `DATA_PATH=../..`. Expect roughly
  20 minutes: compile and CUDA-graph capture (PR #380 says ~7 min at the Track 1 size), prefaulting ~250 GiB of
  index memory, then the timed part. On success the last line is
  `RESULT variant=t2-cpu-ngram val_loss=<f> train_time_s=<f> steps=<n> target=2.92 hit=<0|1> torch=<v> log=<path>`;
  on failure it is `RESULT variant=t2-cpu-ngram status=failed torch=<v> reason=<one line>` with a nonzero exit.
  `torch=` is what the trainer printed on its `stack: torch=<v> triton=<v> cuda=<v>` line (else the version
  installed for that Python); the run log's header has the same versions.
- Settings, all optional: `NPROC` (8; other counts are untested), `SEED` (unset means a random init, as in a record
  run), `STEPS` (see below), `EXTRA_ENV="K=V ..."` or plain exported variables, `LOG_DIR`, `DRY_RUN=1` (print the
  command and exit), `RUN_TIMEOUT_S` (5400), `SKIP_CHECK=1` (the stack is still checked), `ALLOW_STACK=1`. The full list is at the top of
  `182/t2-cpu-ngram/node.py`. The Python used is `$PY`, else the active virtualenv, else `.venv-run`, else `python3`.
- Files of a run go to `LOG_DIR` (default `182/t2-cpu-ngram/results/node/<time>-run/`): `stdout.txt`, the run log
  (source, environment, every step line) and `summary.txt`. Every finished run is also appended to
  `182/t2-cpu-ngram/results/sweep.jsonl`, which the sweep fit reads.

### What one run tells you
The trainer cannot validate mid-run: the validation index is looked up once, after training, and the chain is
fitted at the end. So the run reports as much as it can from the one validation, off the clock:
- `endpoint_result:` the final val loss and train time; `val_loss_parts:` the model's own loss and what the
  exact-count chain takes off (ldmberman reports 2.8991 and 2.8769).
- Before the clock: the host line (CPU model, CPUs, RAM, `/dev/shm`), the retrieval's memory plan and the GPU memory
  budget. After it: peak GPU memory, rank 0's peak RSS and the node's memory in use.
- `summary.txt` puts these together: the margin to 2.92 in millinats, ms per step, GPU headroom, index build times.

### The step count: 5790 or 5800?
Both. `NUM_SCHEDULED_ITERATIONS=5790` is the length of the four main stages, and the schedule always adds a
10-step extension stage, so the run trains **5800 steps**. Log lines (`step:5800/5800`) and the `steps=` field
count all 5800. `STEPS=5200` sets the scheduled part, giving 5210 steps. Our default is ldmberman's setting.

## The first experiment: a shorter run
The one lever with a predictable payoff is the step count. ldmberman's run is 43 millinats under the target, and
train time is proportional to steps. Nothing else we added changes the timed run.

```bash
bash variants/t2-cpu-ngram/182/run_node.sh ab                 # default vs 5200 scheduled steps, 4 seed pairs
AB_RUNS=1 bash variants/t2-cpu-ngram/182/run_node.sh ab       # the 40-minute version: one pair
```

`ab` runs the default and `AB_ENV` (default `NUM_SCHEDULED_ITERATIONS=5200`) with the same seed in each pair,
alternating which arm goes first. It prints the paired differences, a two-point fit of loss against ln(steps) and
one `AB variant=t2-cpu-ngram ...` line. Each run is a fresh process, about 20 minutes, so four pairs take about
2.7 hours.

- If the default lands near 2.877 and the 5210-step arm is at or under ~2.915, the lever is confirmed: about 10%
  off the train time. Follow the fit with `python 182/t2-cpu-ngram/sweep_steps.py --steps <N>` and then
  `significance.py` (a record needs p < 0.01 that the mean is ≤ 2.92).
- If the 5210-step arm is over 2.92, loss falls faster with steps than we assumed (more than 0.4 per ln-step).
  The margin is then worth under 10% and the step count should stay above about 5500.
- If the default itself misses 2.877 by more than ~5 millinats, stop and compare the log's `model_configuration:`,
  stage boundaries and shard count with the numbers in "Verified here". Fix the reproduction before sweeping.
- The slope is not known in advance. The 5200 default assumes roughly 0.3 per ln-step, which would put that arm
  near 2.909. That figure is a guess from other speedrun sweeps, not a measurement of this model.

`COORD_DESCENT_TUNING=1` (allowed on Track 2 only) is the second candidate. It adds a long untimed compile and
most hot paths here are hand-written Triton or CUDA-graph replays, so we expect little. Test it only after the
step count is settled: `AB_ENV="COORD_DESCENT_TUNING=1" AB_RUNS=2 bash .../run_node.sh ab`. Under 1% of train time
means drop it.

## Smaller hosts
`check` computes the retrieval's host memory with the same code the trainer uses at startup (`track_1_short/host.py`).
- **RAM.** The plan is 144 GiB of tables in `/dev/shm`, ~148 GiB of build arenas and a 12 GiB allowance per rank:
  about 390 GiB. `VAL_BUILD_THREADS=16` (new; unset keeps PR #380's 64/32/16 by vCPUs per GPU) cuts 36 GiB of arena
  at 64 threads. The cost is a slower index build beside training. The hints only start at 50% and 85% of the run,
  about 110 s and 190 s in, so a build several times slower than PR #380's should still finish in time, but
  that is unmeasured. `check` suggests the setting when it is what makes the host fit.
- **/dev/shm.** No fallback: the three tables are fixed-size (constants in `exact_match/src/slot.rs`), and
  changing them changes the hints. Remount it larger.
- **CPUs.** Under 24 vCPUs per GPU the code takes PR #380's small-host path (fewer build threads, unpinned corpus
  reads). `check` warns; nobody has reported a Track 2 time on that path.
- `HOST_CHECK=0` turns the trainer's own memory check into a report. A host that is really too small then dies
  mid-build with a SIGBUS and can leave a table in `/dev/shm` (`rm /dev/shm/exact-match-*`).

## What changed on top of PR #380
With no variable set, `train_gpt.py` here is PR #380's Track 1 run plus extra log lines; the items marked
"reported only" never touch the timed region.

| Change | Where | Effect with defaults |
|---|---|---|
| Track 2 entry point: `MODEL_DIM=1024 MODEL_LAYERS=16 MLP_HIDDEN_DIM=4096 LR_COOLDOWN_FRAC=0.70 LR_FLOOR=0.10 VAL_BATCH_SIZE=1048576 NUM_SCHEDULED_ITERATIONS=5790 TARGET_LOSS=2.92`, each overridable | `train_gpt_medium_cpu.py` (new) | Track 2 only |
| The same names as env knobs with Track 1 defaults; ldmberman's depth generalization (`LAST_LAYER`, plain layers 10–14, a 20-slot MLP bank with dead slots frozen) | `config.py`, `schedule.py`, `model/gpt.py` | none (tested: same parameters, buffers, topology) |
| `COORD_DESCENT_TUNING=1` | `train_gpt_medium_cpu.py` | off |
| `VAL_BUILD_THREADS` | `host.py`, `retrieval.py` | unset: PR #380's rule (tested) |
| Host report, memory plan and a pre-clock check (`HOST_CHECK`), peak RSS after the run | `host.py` (new), `retrieval.py` | reported; a host that cannot fit fails before the clock |
| GPU memory budget, `model_configuration:`, measured allocation, `endpoint_result:`, `val_loss_parts:` | `memory_budget.py` (new), `train_gpt.py` | reported only |
| Failure paths: a short corpus read raises instead of looping; a failed index build raises on every rank instead of being queried half-built; a failed table create removes its `/dev/shm` file | `exact_counts.py`, `retrieval.py` | none unless something fails |
| The exact-count kernels widen to int64 before multiplying the block index | `exact_counts.py` | none at 8 GPUs (a rank's corpus share is under 2³¹ tokens); matters with 4 or fewer |
| `PRINT_EVERY`; the run log also holds `train_gpt.py` when the entry point is the wrapper | `train_gpt.py`, `run_log.py` | 25, as before |
| `FINAL_VAL_RELEASE` (1): after the clock stops, drop what the final validation never reads before its forward (see "Memory at d=1024") | `final_val.py` (new), `train_gpt.py`, `exact_counts.py`, `step_graphs.py`, `ngram_table.py` | off the clock; no token's loss changes |
| `EVAL_CE_SLAB_ROWS` (32768, as before): rows per validation loss slab. On a CUDA out-of-memory error the final validation retries the batch with half the slab, down to 2048 | `model/gpt.py`, `final_val.py` | only after an OOM, which used to end the run |
| Stack guard: another torch/triton is a `check` FAIL, also under `SKIP_CHECK=1`, and the trainer exits before any GPU work (`ALLOW_STACK=1` overrides); `stack:` line, `torch=` in the RESULT line | `node.py`, `train_gpt_medium_cpu.py` | none on the pinned stack |
| Node runner, sweep, significance test, CPU tests | `182/` | not part of the run |

Not ported: the hierarchical and adaptive output heads in ldmberman's attachment (~770 lines); the 2.8769 run
used `variant: sampled`, PR #380's own head. As in PR #380, every attention layer after layer 7 reads the same
input, norm(cache[7]). Removed in this round: upstream's `train_gpt_medium.py` (the GPU Track 2 trainer, unrelated
to this stack; it is at the repository root) and the `ALLOW_PARTIAL_SHARDS` switch (the retrieval asserts all 103
shards, so it could never run).

## Memory at d=1024 (estimate, printed at startup)
`python -m track_1_short.memory_budget` prints it offline.
- **GPU, per rank: ~59.8 GiB peak allocated of 79.6 GiB** (ldmberman's log: 61242 MiB allocated, 70578 MiB
  reserved). The n-gram table shard is 20.3 GiB because its width is `MODEL_DIM`. Exact counts take ~6.3 GiB.
  Activations and CUDA-graph pools are ~28.6 GiB, a constant fitted to that log. Checked on the Track 1 log, which
  it was not fitted to: 47.6 GiB predicted, 47.5 logged. That log does not say whether its peak was reached in
  training or in the final validation.
- **Host:** see "Smaller hosts". Building the model is a transient ~12.6 GiB across the 8 ranks.

### The final validation
On our node (torch 2.11, not the pinned 2.10) the run trained all 5800 steps and then died with a CUDA
out-of-memory error inside a model forward, with 73 GiB in use and 5.7 GiB of that in CUDA-graph pools. The only
model forward after the last step is the final validation's (`train_gpt.py`, the `eval_forward` call in the
`with torch.no_grad()` loop). The training steps are graph replays and nothing else runs a forward unless
`run_evals` is set, which it is not. That forward runs on 131072 tokens per rank, 8 times a training step, and
until now it ran with everything training had left behind still allocated. From shapes (`memory_budget.py`,
printed at startup):

| At the final validation forward, per rank | GiB |
|---|---|
| The chain's fit steps: hidden states kept per fit step plus their concatenation (90 steps x 16384 tokens x 1024, bf16, twice) | 5.62 |
| The exact counts: corpus share and table | 6.25 |
| The exact counts' queries and `out` | 1.66 |
| The training step graphs' private pool (the node's OOM message) | 5.7 |
| The forward: the layer cache and MLP hidden at 131072 tokens | 6.25 |
| The forward: one 32768-row loss slab, fused (bf16 matmul output + fp32 logits) | 9.21 |
| The same slab unfused (every step materialized) | 18.42 |

The first four are what `FINAL_VAL_RELEASE=1` (the default) now drops after the clock stops and before the
forward: about 13.5 GiB plus the step graphs' pool (the graphs are reset, `p.grad` is cleared, then
`torch.cuda.empty_cache()`). The validation reads none of them, so the loss is unchanged. The trainer prints
the measured allocation before and after the release (`memory before the final validation`, `final validation
memory release: ... GiB freed`). If the forward still runs out of memory, it retries that batch with half the
loss slab (32768, then 16384, down to 2048 rows). Each slab is row-local: the same tokens, the same per-token
formula, so the same mean up to float rounding (`182/t2-cpu-ngram/tests/test_final_val.py`). Measured on CPU:
eager is bit for bit at every slab size of 2 rows and up. Through inductor, half a slab moved one token's loss by at
most 3.8e-6 nats and the batch mean by 2e-8, because a different row count can change how a reduction is split.
On the GPU, inductor's kernel choice and cuBLAS's GEMM algorithm can also depend on the row count, so expect the
same rounding-level effect, unmeasured. Each retry prints one line and recompiles the eval graph, off the clock.
A retry only happens where the run used to die.

Not known without the node's log: how much of the 73 GiB was the forward itself, and why torch 2.11 needed more
than ldmberman's 61.2 GiB peak. The pinned stack is the first fix. ldmberman's 70.6 GiB reserved left ~9 GiB of headroom,
which is why `VAL_BATCH_SIZE` is halved to 1048576. Do not raise it.

## Track 2 rules notes
- Same rules as Track 1 at 2.92: p < 0.01 that the mean is ≤ 2.92, train and val streams unchanged. The README sets
  no model size for Track 2, so 528.5M parameters plus the n-gram table are within the rules as written.
- The whole CPU-retrieval line (#367, #380, #381) is **unmerged**. The maintainer wrote on 2026-10-03 (#367) that
  CPU/RAM rules are coming so that "we don't see run-away CPU farms". As of 2026-11-11 there is no such rule and
  no new commit or comment on #380. A cap on vCPUs or host RAM could slow this run or rule it out; the log's host
  lines are the disclosure.
- `coordinate_descent_tuning` is allowed on the medium track only. `train_gpt.py` never sets it.
- All 103 train shards must be on disk. The exact-count corpus is every shard the loader cannot reach (12 are
  excluded at 5800 steps), so the shard count changes the hints and with them the loss.

## After the first runs
```bash
cd variants/t2-cpu-ngram
python 182/t2-cpu-ngram/sweep_steps.py --fit-only                        # fit of everything run so far
python 182/t2-cpu-ngram/sweep_steps.py --steps 4900 5000 --seeds 0 1     # more points near the fitted count
python 182/t2-cpu-ngram/significance.py --steps <N> --runs 6             # exit 0 only if p < 0.01
```
The fit is `val_loss = a + b·ln(steps)` over every run without other settings changed, plus ldmberman's point
unless `--no-seed-point`. It solves for the mean that gives p < 0.01 with 6 runs, 2.918 at a seed std of 0.0014
(Track 1's spread; ours is unmeasured).

## Verified here (CPU only)
- `python -m pytest -q 182/t2-cpu-ngram/tests` from this folder. Triton, FA3 and the Rust extension are stubbed, so
  the model is built on the meta device, not run.
- At the Track 2 settings the model has exactly the logged 528,536,207 parameters, bank order
  `{0,2,5,10–14,1,8,3,15}`, stage boundaries `[(0,1653),(1653,3513),(3513,5443),(5443,5790),(5790,5800)]`, the
  sampled-softmax switches and the 0.1 lr floor of ldmberman's log. With defaults its parameters, buffers and
  topology constants equal PR #380's.
- The node runner: the parser on a real PR #380 record log and on truncated, crashed and NaN logs; the RESULT
  line; `DRY_RUN=1`; `check` (it fails cleanly on a machine without GPUs); the host fallback; a full
  run/timeout/failure cycle against a stand-in trainer.
- The failure paths above (a two-rank gloo test for the index build), and the t-test against scipy.
- The final validation's loss slabs (`tests/test_final_val.py`): per-token losses bit for bit across slab sizes in
  eager mode, the out-of-memory retry, the release, the stack guard and the RESULT line. The CPU inductor
  comparison is opt-in (`RUN_INDUCTOR_PROBE=1`): 3.8e-6 nats at most per token at 45 rows, bit for bit at 12 rows.

## Not verified without an 8xH100
- That this tree trains at all at d=1024: compile, CUDA-graph capture and the kernels have only run for PR #380
  (Track 1 size) and for ldmberman's own copy. Ours matches theirs in config, schedule and model shape, not by
  execution.
- That it reproduces ~2.877 in ~221 s, the seed std, and the real peak memory on our node.
- `prep` on Linux: the 103-shard download, the Rust build, the FA3 prefetch. The command lines are tested, not
  their execution.
- The `val_loss_parts:` line (two lines in the validation loop, off the clock).
- The final-validation release on a GPU: that `CUDAGraph.reset()` plus `empty_cache()` really returns the step
  graphs' pool (the trainer prints how much was freed), and the OOM retry's recompile under CUDA. Both are tested
  on CPU with stand-ins only.
- `VAL_BUILD_THREADS` below the default, the small-host path, and any `NPROC` other than 8.
- The slope of loss against steps, and any effect of `COORD_DESCENT_TUNING`.
