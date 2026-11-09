# 182-project

Private working fork of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT) for 8xH100 speedrun attempts.
One branch, `main`. **Nothing here has been run on GPUs yet.**

## Layout

| Path | What | Run from |
|---|---|---|
| repo root (`train_gpt.py`, `track_1_short/`) | Upstream record #92 (ANVIL2, 39.9 s), plus env-flag experiments that are off by default: `FP8_OFF`, `HADAMARD_FP8`, `FP8_DUMP_DIR` (`182/hadamard-fp8`) and `MONARCH_MLP` (`182/structured`) | repo root |
| `182/tools/` | node setup, A/B runner, t-tests, nsys profiling | repo root |
| `182/hadamard-fp8/`, `182/structured/` | tests and analysis scripts for the root-level flags; Track 3 fixed-basis Adam/SOAP | repo root |
| `variants/t1-gpu-stack/` | Track 1 GPU only: open PR #397 (30.67 s) + #378/#376/#382/#385/#387 behind flags | that folder |
| `variants/t1-cpu-ngram/` | Track 1 CPU retrieval: open PRs #380/#381 (~9.6 s) + #399 table vote, host checks | that folder |
| `variants/t2-cpu-ngram/` | Track 2 (medium, 2.92) port of the CPU retrieval line | that folder |
| `variants/t2-gpu-port/` | Track 2 GPU only: open PR #261 (14.34 min) + canonical mask / prefix loss / XSA | that folder |

Each variant is a full, self-contained tree of an open upstream PR plus our changes. The variants rewrite the same
files in different ways, so they live in folders instead of being merged. Each one has its own README.

Data lives at the repo root (`data/cached_fineweb10B.py`); point a variant at it with `DATA_PATH=../..`:

```bash
cd variants/t1-gpu-stack && DATA_PATH=../.. torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## First steps on the 8xH100 box

```bash
bash 182/tools/node_setup.sh --mode venv && source .venv-run/bin/activate   # docker: --mode docker, then SESSION_DOCKER=modded-nanogpt bash 182/gpu_session.sh
bash 182/gpu_session.sh        # ~1.5-2 h; resumable (re-run the same command); DRY_RUN=1 prints the plan
```

It runs `env tests sanity gate hadamard t2 bundle` (or just the phases you name), prints a GPU-minute estimate
before each, and ends with `SEND THIS FILE BACK: 182/results/session-<date>-<sha>.tar.gz`. Send that one file.
The gate is the `182/hadamard-fp8` decision rule (record vs `FP8_OFF=1`, 4 paired seeds); the Hadamard A/B runs
only if its 95% CI reaches 0.5 mnat; t2 is skipped (with the reason) on a host without ~390 GiB RAM / 24 vCPU per GPU.
Knobs are listed at the top of the script. Read a bundle with `python 182/tools/summarize_session.py <bundle>`.
