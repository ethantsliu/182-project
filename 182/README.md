# 182-project

Private working fork of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT) for
8xH100 speedrun attempts. One branch, `main`. We have no GPUs of our own: everything here is checked on CPU, and
a friend runs it on an 8xH100 node.

## What has been measured, and what has not

Two single runs on the friend's node, and nothing else. Neither log is in this repo.

- The root record: **40.931 s, val loss 3.2780**, 1194 steps (one run). The official record it reproduces is 39.9 s.
- `variants/t1-gpu-stack`: **30.957 s, val loss 3.276481**, 1053 steps, 981 of them scheduled (one run). Its
  author reported 30.67 s over 12 runs.
- Not measured by us: the other three variants, every flag of ours except one `HADAMARD_FP8` run (3.17 s slower than
  the record, no log here), and the run-to-run spread of anything on that node. One run cannot show that a tree meets its target: the rule is the mean under the target with p < 0.01.

The "best reported" times in the table below are the ones each tree's author reported upstream, on their nodes.

## Methods

| Phase | Tree | What it is | Best reported time (not ours) | Target | Official record | One command |
|---|---|---|---|---|---|---|
| `base` | repo root | Upstream record #92 (ANVIL2), unchanged | 39.9 s (the record; our node, one run: 40.931 s, 3.2780) | val loss <= 3.28 | 39.9 s | `bash 182/gpu_session.sh base` |
| `t1gpu` | `variants/t1-gpu-stack` | Track 1, GPU only: open PR #397 (MoE + Gluon + exact levers) plus our flags | 30.67 s, n=12, mean loss 3.27743 (our node, one run: 30.957 s, 3.276481) | <= 3.28 | 39.9 s | `bash 182/gpu_session.sh t1gpu` |
| `t1cpu` | `variants/t1-cpu-ngram` | Track 1, CPU retrieval: the tree of open PR #381 (exact-count chain of #380) | 9.653 s, n=16 (PR #380; #381 itself has no 8xH100 time) | <= 3.28 | 39.9 s | `bash 182/gpu_session.sh t1cpu` |
| `t2cpu` | `variants/t2-cpu-ngram` | Track 2 (GPT-2 Medium): that CPU retrieval chain | 221 s, one run, 5800 steps, loss 2.8769 | <= 2.92 | 17.35 min | `bash 182/gpu_session.sh t2cpu` |
| `t2gpu` | `variants/t2-gpu-port` | Track 2, GPU only: open PR #261 (MUDD) plus our flags | 14.34 min (860.5 s), loss 2.9171 ± 0.0007 | <= 2.92 | 17.35 min | `bash 182/gpu_session.sh t2gpu` |
| `gate` | repo root | A/B: the record vs `FP8_OFF=1` (what does fp8 cost in val loss?) | - | - | - | `bash 182/gpu_session.sh gate` |
| `attno` | repo root | A/B: the record vs `FP8_ATTN_O=1` (4 seed pairs; cost model expects about +0.2 s, range -0.7 to +1.3 s) | - | - | - | `bash 182/gpu_session.sh attno` |
| `hadamard` | repo root | A/B: the record vs `HADAMARD_FP8=1` variants; runs only if the gate finds a gap | - | - | - | `bash 182/gpu_session.sh gate hadamard` |

Each variant is a full, self-contained tree of an open upstream PR plus our changes, behind flags that are off by
default. The variants rewrite the same files in different ways, so they live in folders instead of being merged;
each has its own README and its own node script, `variants/<name>/182/run_node.sh check|prep|run`, which the
session drives. The root flags (`FP8_OFF`, `HADAMARD_FP8` / `_SEED` / `_BLOCK` / `_GEMMS`,
`HADAMARD_SAT_STATS`, `FP8_DUMP_*`, `MONARCH_*`) are listed, with where each is read, in `track_1_short/config.py`.

## On the node

```bash
tmux new -s gpu                # first: an ssh drop kills a plain shell and the session with it (reattach: tmux a -t gpu)
bash 182/tools/node_setup.sh --mode venv && source .venv-run/bin/activate   # the root record's stack
DRY_RUN=1 bash 182/gpu_session.sh   # the plan, the estimates, each variant's check, every command; runs nothing
bash 182/gpu_session.sh             # everything, most promising first (about 4 h if nothing is skipped)
QUICK=1 bash 182/gpu_session.sh     # instead, if time is short: env sanity base t1gpu bundle (under 45 min)
```

Docker instead of a venv: `bash 182/tools/node_setup.sh --mode docker`, then put `SESSION_DOCKER=modded-nanogpt` in
front of the same commands.

It ends with `SEND THIS FILE BACK: 182/results/session-<date>-<sha>.tar.gz`. Send that one file; read it with
`python 182/tools/summarize_session.py <bundle>`. The summary starts with one table (method, runs, mean val loss,
mean time, best reported, official record, target hits) and says plainly when a row is a single run.

The order is `env tests sanity base t1gpu t1cpu t2cpu t2gpu gate attno hadamard bundle`:

- `base` first (2 runs): this node's own reference and its spread, with logs, which the one earlier run did not leave.
- Then the four variants, Track 1 before Track 2, GPU-only before CPU retrieval: 4, 4, 1 and 1 runs. For each, the
  session runs the variant's `check`; a `FAIL` line skips that variant with the reason (it is not a session
  failure, and `FORCE_CHECK=1` preps and runs it anyway). The CPU retrieval trees need a host with far more RAM, `/dev/shm`
  and vCPUs than the record does, so expect them to be skipped on a plain 8xH100 box. `prep` (data downloads,
  extension builds; Rust needs rustc >= 1.80, and `INSTALL_RUST=1` lets prep install it) runs off the clock, for all selected variants in one low-priority background job that starts
  with the session; a variant waits for that job before it times anything.
  The runs use each tree's default configuration, with two things to know: `t2gpu` runs with `CANON_MASK=1`, the
  first run its README recommends (`T2GPU_RUN_ENV=` gives plain PR #261), and `t1cpu` runs configuration 381, which
  has no published 8xH100 time; for the one that has (9.653 s), use a session directory of its own:
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

## Layout

| Path | What |
|---|---|
| repo root (`train_gpt.py`, `track_1_short/`) | Upstream record #92 plus env-flag experiments that are off by default |
| `182/gpu_session.sh` | The node entry point described above |
| `182/tools/` | Node setup, A/B runner, statistics, log parser, session summary, nsys profiling (see its README) |
| `182/hadamard-fp8/`, `182/structured/` | Tests and analysis scripts for the root flags; Track 3 fixed-basis Adam/SOAP |
| `variants/*/` | The four trees of the table; data is shared from the repo root (`DATA_PATH=../..`) |
| `182/results/` | Session output on the node (gitignored) |
