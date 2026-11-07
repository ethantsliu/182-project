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

1. `bash 182/tools/node_setup.sh --mode docker`, then `python variants/t1-cpu-ngram/182/t1-cpu-ngram/node_check.py` (RAM/CPU for the CPU-retrieval variants).
2. `182/tools/profile_nsys.sh` on the root record, then a root-vs-root A/B to confirm the harness shows no difference.
3. Reproduce `variants/t2-cpu-ngram` (~2.877 at 5800 steps), then cut steps.
4. Cheap negative-result checks: `FP8_OFF=mlp,qkv,lm` and `MONARCH_MLP=1` A/Bs at the root.
