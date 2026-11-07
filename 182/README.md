# 182-project

Private working fork of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT) for 8xH100 speedrun attempts.

- `master` = upstream master (record #92, ANVIL2, 39.9 s).
- `pr/NNN` = snapshots of open upstream PRs (2026-11-06), credited to their authors.
- `ws/*` = our workstreams:
  - `ws/t1-gpu-stack`   Track 1 GPU-only: #397 + #378/#376/#382/#385/#390, step retune
  - `ws/t1-cpu-ngram`   Track 1 CPU-retrieval line: #380/#381 + #399, host tuning
  - `ws/t2-cpu-ngram`   Track 2 (medium, 2.92) CPU-retrieval port of #380
  - `ws/t2-gpu-port`    Track 2 GPU-only: #261 + 2026 small-track features
  - `ws/hadamard-fp8`   FP8-vs-bf16 gap test + block-Hadamard rotated FP8 quantization
  - `ws/structured`     Monarch MLP + rotated-basis optimizer ablations (expected negative)
  - `ws/harness`        A/B runner, t-test, nsys profiling + analysis, node setup

Nothing here has been run on GPUs yet; see each workstream's `182/<ws>/README.md`.
