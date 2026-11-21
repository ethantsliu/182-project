# EECS 182 project: Do Faster Optimizer Kernels Reduce End-to-End Training Time?

This is the map of the project. The repository is a fork of
[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT), one branch, `main`.
The root `README.md` is upstream's, with a short pointer to this file at the top.

**Results and report-ready tables: [`182/results/README.md`](results/README.md).** The report draft is in [`182/paper/`](paper/) (`main.tex`, `main.pdf`, figures, `overleaf.zip`).

**Prior art.** The Gram-space kernel is not a new method. It is the Polar Express appendix's fast iteration for rectangular matrices (arXiv 2505.16932) and Gram Newton-Schulz (Zhang, Amsel, Chen, Dao, 2026), which we found after writing it. Ours are the port into the ANVIL2 trainer and the measurements (the 0.7% ceiling and why the gain is small). See "What is ours" in the results file.
If a number here or in a subdirectory README disagrees with it, trust the results file. Many subdirectory READMEs
were written before any GPU run and still say "nothing here has run on a GPU"; read that as "at the time of writing".

## What the project asks

The upstream speedrun trains a GPT-2-small-sized model to 3.28 validation loss on FineWeb on 8 H100 GPUs. Its current
optimizer (ANVIL) spends part of every step on matrix work: Gram matrices, polynomials and GEMMs that whiten the
update of each weight bank. The question is whether a cheaper kernel for that matrix work makes the whole run
shorter, or whether the saving is hidden by the rest of the step.

We test it in three ways, all on the root tree, all behind flags that are off by default:

1. An alternative kernel with the same update rule (`ANVIL_KERNEL`), compared with the record in paired runs.
2. An upper bound (`ANVIL_ABLATE=skip_matrix`): skip the matrix work entirely and see how much time that frees.
   This is a measurement, not a valid training run.
3. A one-GPU microbenchmark of the kernels alone (`182/optim-kernel/bench.py`).

With every project flag unset, the root tree (`train_gpt.py`, `track_1_short/`) runs the upstream record.

## Where the alternative optimizer kernel lives

| Path | What |
|---|---|
| `track_1_short/perf/anvil_kernel.py` | The Gram-space cascade (`ANVIL_KERNEL`) and the ablation (`ANVIL_ABLATE`) |
| `track_1_short/perf/kernels/anvil_chain.py` | The new symmetric Triton kernel used by the `s` variants (`gram222s`, `gram33s`) |
| `track_1_short/config.py` | Where the flags are read and checked (bottom of the file; `RUN_METRICS` / `VAL_CURVE` are mid-file) |
| `track_1_short/optim/anvil.py` | The one hook in the bank update that switches to the alternative for the chosen banks |
| `train_gpt.py` | Log lines for the flags, the `RUN_METRICS` clock readings, the `VAL_CURVE` validations |
| `track_1_short/run_metrics.py` | Formats the `METRIC` lines |

| Flag | Values | Meaning |
|---|---|---|
| `ANVIL_KERNEL` | `gram222`, `gram222s`, `gram33`, `gram33s` | Alternative kernel, same update rule. Not bit-identical to the record |
| `ANVIL_BANKS` | comma list of `qk_bank`, `vo_bank`, `mlp_bank` | Which banks the kernel or ablation applies to. Default: `mlp_bank` for the kernel, all banks for the ablation |
| `ANVIL_ABLATE` | `skip_matrix` | Measurement only: no matrix work. Validation lines are tagged `INVALID_FOR_RECORD(anvil_ablate)` |
| `RUN_METRICS` | `1` | Prints `METRIC` lines: step latency, peak memory, validation loss to 6 digits |
| `VAL_CURVE` | step interval `N` | Also validates every N steps. A curve run, not a timing run |

`ANVIL_KERNEL` and `ANVIL_ABLATE` cannot be set together. The design, the error table and the expected gain are in
[`optim-kernel/README.md`](optim-kernel/README.md).

## Run the kernel benchmark and tests

From the repo root. On a laptop (CPU; the GPU tests skip):

```bash
python3 182/optim-kernel/bench.py --device cpu --smoke
python3 -m pytest 182/optim-kernel/test_optim_kernel.py -q
```

On a GPU node:

```bash
python 182/optim-kernel/bench.py --json kbench.json           # 1 GPU, about 1 min
python -m pytest 182/optim-kernel/test_optim_kernel.py -q     # CPU tests plus the GPU tests
ANVIL_KERNEL=gram222 bash run.sh                              # one 8-GPU run by hand
XAB_ONLY=anvilkernel bash 182/gpu_session.sh xab              # base vs gram222 vs skip_matrix, 4 seeds
```

How to read the benchmark output is in [`optim-kernel/README.md`](optim-kernel/README.md). The paired timing and
curve runs with `ab_run.py` and `compare_runs.py` are under [Run metrics](#run-metrics) below.

## Launch GPU jobs

We rent an 8xH100 node per job on Lor. One job claims a node, sets it up, runs one command and releases the node.
Push first: the job clones the pushed commit.

```bash
182/lor/submit.sh NAME EST_MIN MAX_MIN '<command, run from the repo root on the node>'
182/lor/status.sh          # what is holding GPUs, and the spend so far
182/lor/status.sh stop     # stop everything that is still live
```

Logs and small output files land in `~/.lor-182/jobs/NAME-<time>/` on the machine that submitted. The job scripts
that were used are in `182/lor/jobs/`. Setup, cost and known pitfalls: [`lor/README.md`](lor/README.md).

## Directory map

Status words: **used** = GPU measurements exist for it; **exploratory** = written and tested on CPU, no GPU result
that we rely on; **dead end** = measured or modelled and dropped. The results file is the authority on what the
report uses.

### `182/`

| Path | What it is | Status |
|---|---|---|
| `182/optim-kernel/` | The project's kernel work: microbenchmark (`bench.py`), tests, design notes | Used. The core of the project |
| `182/lor/` | Scripts to rent a Lor node, run one command, collect the log (`submit.sh`, `status.sh`, `collect.sh`, `entry.sh`) and the job scripts (`jobs/`) | Used for the GPU runs |
| `182/gpu_session.sh` | Node entry point: runs phases (base record, variants, profile, A/Bs) and bundles the output. Reference below | Used |
| `182/tools/` | Shared harness: node setup, A/B runners (`ab_run.py`, `xab_run.py`), statistics, log parser, `compare_runs.py`, session summary, nsys profiling, CPU harness for Triton kernels | Used |
| `182/experiments.d/` | One file per paired A/B for the session's `xab` phase. `anvilkernel.sh` is the project's A/B. `sparse24.sh`, `optim.sh`, `kernel.sh`, `hostcpu.sh` belong to the exploratory work | `anvilkernel.sh` used; the rest exploratory |
| `182/profile/` | torch.profiler hook and trace analyzer: where the step time goes, per category | Used (a root-tree profile ran on Lor) |
| `182/results/` | Results and report-ready tables. Also the default output directory of the session on a node | Written separately |
| `182/hadamard-fp8/` | Earlier idea: measure what FP8 costs in loss (`FP8_OFF`), and a block-Hadamard rotation to win it back (`HADAMARD_FP8`) | Dead end. One node run was 3.17 s slower than the record |
| `182/fp8-attn-o/` | Earlier idea: the attention output projection in FP8 (`FP8_ATTN_O`) | Exploratory. Its own cost model expects about +0.2 s, so no gain |
| `182/sparse24/` | Earlier idea: 2:4 sparse MLP weights (`SPARSE24_MLP`) | Exploratory. Its own cost model says it cannot pay |
| `182/structured/` | Earlier idea: Monarch MLP (`MONARCH_MLP`) and fixed-basis Adam/SOAP for Track 3 | Exploratory. Written as falsification tests, expected negative |

### `variants/`

Each variant is a full, self-contained copy of an open upstream pull request plus our flags, off by default. They
rewrite the same files in different ways, so they are folders and not merged into the root. They are not part of the
kernel question; they are the faster trees we compared against and tuned. Each has its own README and
`182/run_node.sh check|prep|run`.

| Path | What it is | Status |
|---|---|---|
| `variants/t1-cpu-ngram/` | Track 1 with CPU n-gram retrieval: upstream PR #381 (on #380 and #367). Defaults to 668 scheduled steps, our trim; `NUM_SCHEDULED_ITERATIONS=675` reproduces #381 | Used. Ran on Lor; the 675 vs 668 comparison is in its README |
| `variants/t1-gpu-stack/` | Track 1, GPU only: upstream PR #397 (MoE + Gluon kernels) plus our flags. Holds three sub-studies under `182/`: `optim-cost/` (cascade cost model and `OPTIM_*` flags), `kernel-cost/` (per-op cost model, `MOE_ROUTER_FUSED`), `hostcpu/` (`HOST_ASYNC`) | Base tree ran on Lor. The three sub-studies are exploratory |
| `variants/t2-cpu-ngram/` | Track 2 (GPT-2 Medium, target 2.92): the CPU retrieval chain of PR #380 at Track 2 sizes | One run on Lor |
| `variants/t2-gpu-port/` | Track 2, GPU only: upstream PR #261 (MUDD) plus our flags | Exploratory. We found no GPU run of it |

### Root (upstream, plus hooks)

| Path | What |
|---|---|
| `train_gpt.py`, `track_1_short/` | Upstream record #92 (ANVIL2) plus the project flags and the earlier flags, all off by default |
| `train_gpt_medium.py`, `records/`, `data/`, `evals/`, `img/`, `Dockerfile`, `run.sh` | Upstream, unchanged |

## Tests that run on a laptop

```bash
python3 -m pytest 182/optim-kernel/test_optim_kernel.py -q   # the kernel
python3 -m pytest 182/tools/tests -q                         # the harness (stdlib only)
```

The other directories list their own test commands in their READMEs.

## Earlier single runs (before the Lor jobs)

Kept for the record. Two single runs on a borrowed 8xH100 node; neither log is in this repo.

- The root record: 40.931 s, val loss 3.2780, 1194 steps. The official record it reproduces is 39.9 s.
- `variants/t1-gpu-stack`: 30.957 s, val loss 3.276481, 1053 steps. Its author reported 30.67 s over 12 runs.
- One `HADAMARD_FP8` run: 3.17 s slower than the record.

One run cannot show that a tree meets its target. The rule is the mean under the target with p < 0.01.

## GPU session reference: phases

The "best reported" times in this table are the ones each tree's author reported upstream, on their nodes. Our own
measurements are in `182/results/README.md`.

| Phase | Tree | What it is | Best reported time (not ours) | Target | Official record | One command |
|---|---|---|---|---|---|---|
| `base` | repo root | Upstream record #92 (ANVIL2), unchanged | 39.9 s (the record; our node, one run: 40.931 s, 3.2780) | val loss <= 3.28 | 39.9 s | `bash 182/gpu_session.sh base` |
| `t1gpu` | `variants/t1-gpu-stack` | Track 1, GPU only: open PR #397 (MoE + Gluon + exact levers) plus our flags | 30.67 s, n=12, mean loss 3.27743 (our node, one run: 30.957 s, 3.276481) | <= 3.28 | 39.9 s | `bash 182/gpu_session.sh t1gpu` |
| `t1cpu` | `variants/t1-cpu-ngram` | Track 1, CPU retrieval: the tree of open PR #381 (exact-count chain of #380) | 9.653 s, n=16 (PR #380; #381 itself has no 8xH100 time) | <= 3.28 | 39.9 s | `bash 182/gpu_session.sh t1cpu` |
| `t2cpu` | `variants/t2-cpu-ngram` | Track 2 (GPT-2 Medium): that CPU retrieval chain | 221 s, one run, 5800 steps, loss 2.8769 | <= 2.92 | 17.35 min | `bash 182/gpu_session.sh t2cpu` |
| `t2gpu` | `variants/t2-gpu-port` | Track 2, GPU only: open PR #261 (MUDD) plus our flags | 14.34 min (860.5 s), loss 2.9171 ± 0.0007 | <= 2.92 | 17.35 min | `bash 182/gpu_session.sh t2gpu` |
| `profile` | `t1-gpu-stack`, repo root | torch.profiler over 20 timed-loop steps on all ranks: time per category, exposed NCCL, idle gaps, top kernels (`182/profile/README.md`); not timing runs | - | - | - | `bash 182/gpu_session.sh profile` |
| `nccl` | `t1-gpu-stack`, repo root | A/B: NCCL defaults vs `NCCL_NVLS_ENABLE=0` vs forced NVLS (4 seed pairs); skipped if `nvls_check.py` finds NVLS unavailable | - | - | - | `bash 182/gpu_session.sh nccl` |
| `xab` | per file | Every `182/experiments.d/*.sh` A/B (format: `182/experiments.d/README.md`): Δval mnat and Δtime with 95% CIs, net seconds | - | - | - | `bash 182/gpu_session.sh xab` |
| `gate` | repo root | A/B: the record vs `FP8_OFF=1` (what does fp8 cost in val loss?) | - | - | - | `bash 182/gpu_session.sh gate` |
| `attno` | repo root | A/B: the record vs `FP8_ATTN_O=1` (4 seed pairs; cost model expects about +0.2 s, range -0.7 to +1.3 s) | - | - | - | `bash 182/gpu_session.sh attno` |
| `hadamard` | repo root | A/B: the record vs `HADAMARD_FP8=1` variants; runs only if the gate finds a gap | - | - | - | `bash 182/gpu_session.sh gate hadamard` |

Each variant is a full, self-contained tree of an open upstream PR plus our changes, behind flags that are off by
default. The variants rewrite the same files in different ways, so they live in folders instead of being merged;
each has its own README and its own node script, `variants/<name>/182/run_node.sh check|prep|run`, which the
session drives. The root flags (`FP8_OFF`, `HADAMARD_FP8` / `_SEED` / `_BLOCK` / `_GEMMS`,
`HADAMARD_SAT_STATS`, `FP8_DUMP_*`, `MONARCH_*`) are listed, with where each is read, in `track_1_short/config.py`.

## GPU session reference: on the node

```bash
tmux new -s gpu                # first: an ssh drop kills a plain shell and the session with it (reattach: tmux a -t gpu)
bash 182/tools/node_setup.sh --mode venv && source .venv-run/bin/activate   # the root record's stack
DRY_RUN=1 bash 182/gpu_session.sh   # the plan, the estimates, each variant's check, every command; runs nothing
bash 182/gpu_session.sh             # everything, most promising first (about 4 h if nothing is skipped)
QUICK=1 bash 182/gpu_session.sh     # instead, if time is short: env sanity base t1gpu profile bundle (under 45 min)
```

Docker instead of a venv: `bash 182/tools/node_setup.sh --mode docker`, then put `SESSION_DOCKER=modded-nanogpt` in
front of the same commands.

It ends with `SEND THIS FILE BACK: 182/results/session-<date>-<sha>.tar.gz`. Send that one file; read it with
`python 182/tools/summarize_session.py <bundle>`. The summary starts with one table (method, runs, mean val loss,
mean time, best reported, official record, target hits) and says plainly when a row is a single run.

### Recovering from a logging-permission failure

Node runners now put each trainer's source/environment log under the selected `LOG_DIR`, in a `trainer/`
subdirectory. Root A/B and profile runs use their own output directories too. They do not need to write to a
legacy `logs/` directory left owned by a previous container user. For a direct trainer invocation, use
`TRAIN_LOG_DIR=/absolute/writable/path`; the default remains `logs/`. An unwritable selected destination fails
before a node runner starts GPU workers and reports the actual filesystem error. No ownership changes are needed.

After pulling a fix, explicitly reuse the **existing session directory**, since the default includes the current
Git SHA and would create a new session. Find it with `find 182/results -path '*/status/base' -type f`: use the
parent of `status/`, not a tarball or the exported `final-results/` report directory. In the same pinned environment
and container used for the successful baseline:

```bash
export SESSION_DIR=/absolute/path/to/the/existing/session
bash 182/gpu_session.sh env sanity base t1gpu profile bundle
bash 182/gpu_session.sh t1cpu t2cpu t2gpu bundle
```

Do not set `FORCE=1`: PASS phases and successful runs are retained; failed variant attempts are archived and
retried. Run both commands even if the first reports a failed phase; its nonzero exit is a summary of failures,
not a reason to abandon the independent phases. Each invocation still runs its selected later phases and bundles
the available evidence. Missing runtime metadata on a startup failure means unknown, not a stack mismatch;
the environment and variant checks establish stack compatibility. The single-GPU gates and A/B commands above
can follow once their prerequisites succeed.

The order is `env tests sanity base t1gpu profile nccl xab t1cpu t2cpu t2gpu gate attno hadamard bundle`:

- `base` first (2 runs): this node's own reference and its spread, with logs, which the one earlier run did not leave.
- Then the four variants, Track 1 before Track 2, GPU-only before CPU retrieval: 4, 4, 1 and 1 runs. For each, the
  session runs the variant's `check`; a `FAIL` line skips that variant with the reason (it is not a session
  failure, and `FORCE_CHECK=1` preps and runs it anyway). The CPU retrieval trees need a host with far more RAM, `/dev/shm`
  and vCPUs than the record does, so expect them to be skipped on a plain 8xH100 box. `prep` (data downloads,
  extension builds; Rust needs rustc >= 1.80, and `INSTALL_RUST=1` lets prep install it) runs off the clock, for all selected variants in one low-priority background job that starts
  with the session; a variant waits for that job before it times anything.
  The runs use each tree's default configuration, with two things to know: `t2gpu` runs with `CANON_MASK=1`, the
  first run its README recommends (`T2GPU_RUN_ENV=` gives plain PR #261), and `t1cpu` runs configuration 381 with this
  tree's 668 scheduled steps (`T1CPU_RUN_ENV=NUM_SCHEDULED_ITERATIONS=675` gives upstream #381), which has no published
  8xH100 time; for the one that has (9.653 s), use a session directory of its own:
  `SESSION_DIR=182/results/t1cpu-380 T1CPU_RUN_ENV=CONFIG=380 bash 182/gpu_session.sh t1cpu bundle`.
- Then the root fp8 experiments, which need 4 seed pairs each to say anything: `gate`, `attno`, and `hadamard` last
  (one measurement on the node had it 3.17 s slower, and it needs a 3-4 millinat fp8 gap to pay).

Rules that hold for every phase: a failed or skipped phase never stops the ones after it; re-running the same
command resumes (finished phases and finished runs are kept, failed runs are retried); after a ctrl-C,
`bash 182/gpu_session.sh bundle` packs what exists. `env` fails unless the stack is the root record's (torch
2.10.0+cu128, triton 3.6.0, the venv's torchrun, the pinned FA3 loading); that skips the root phases only, since
each variant checks its own stack, and `ALLOW_STACK=1` runs the root phases anyway. Every knob (run counts, seeds,
timeouts) is listed at the top of `182/gpu_session.sh`.

A single method by hand, without the session:

```bash
bash variants/t1-gpu-stack/182/run_node.sh check   # what this host lacks, with the fix for each line
bash variants/t1-gpu-stack/182/run_node.sh prep    # data and builds
bash variants/t1-gpu-stack/182/run_node.sh run     # one timed run; the last line is RESULT variant=... val_loss=... train_time_s=...
torchrun --standalone --nproc_per_node=8 train_gpt.py   # the root record, from the repo root
```

### Run metrics for the kernel comparison (root trainer)

<a id="run-metrics"></a>

Two flags, off by default (`track_1_short/config.py`; flags off is the upstream run), print greppable `METRIC` lines
(`track_1_short/run_metrics.py`) to the console and the run log:

- `RUN_METRICS=1`, for both arms of a timing A/B: `METRIC step_latency scope=all|stageN ... mean_ms median_ms p95_ms`
  (one host clock reading per step, no synchronize; validation excluded), `METRIC peak_mem rank=R|max allocated_mib
  reserved_mib warmup_*` (every rank, timed loop and before the clock separately, read after the loop) and
  `METRIC val step= train_time_s= val_loss= final=` (6 digits).
- `VAL_CURVE=N`: also validates every N steps with the clock stopped. This is a curve run, not a timing run (each
  validation leaves reads and pulls on the clock; val lines are tagged `INVALID_FOR_RECORD(val_curve)`).

```bash
W=182/results/kernel-work                      # shared worktrees and warm compile caches
python 182/tools/ab_run.py plan --out 182/results/kernel-timing --work-root $W --arm base=ref:HEAD --arm kern=ref:HEAD \
  --common-env RUN_METRICS=1 --env kern:ANVIL_KERNEL=<value> --n 4 --seeds 1,2,3,4 --warmup 1 --cache warm --go
python 182/tools/ab_run.py plan --out 182/results/kernel-curve --work-root $W --arm base=ref:HEAD --arm kern=ref:HEAD \
  --common-env RUN_METRICS=1 --common-env VAL_CURVE=100 --env kern:ANVIL_KERNEL=<value> --n 2 --seeds 1,2 --cache warm --go
python 182/tools/compare_runs.py --names base,kernel --csv 182/results/kernel-curves.csv \
  --a 182/results/kernel-timing@base 182/results/kernel-curve@base \
  --b 182/results/kernel-timing@kern 182/results/kernel-curve@kern
```

`compare_runs.py` (sources: logs, directories, globs or `ABDIR@ARM`) prints, per arm, n, mean ± sd and the 95% interval of
train time, step latency, final val loss and peak memory, the paired B - A differences by seed, and the time to 3.28
interpolated from the curve runs, and writes the curves as CSV. This section was written before the GPU runs; what
ran, and the numbers, are in `182/results/README.md`.
