# Pinned reruns and FNO follow-ups — 2026-11-12

This round first reruns nanoGPT on the records' PyTorch stack, then tests the strongest result from the FFT experiments: per-mode SOAP for Fourier neural operators. The old tables remain in `runs/SUMMARY.md`. New jobs never overwrite those results.

## Checkouts and stack

- `/home/dev_patel/182-project` stays on **main**, pulled with `git pull --ff-only origin main` at `4c43aac0139b`. This is newer than the requested `3715aa6` and includes additional validation-memory fixes. `git stash push` found no local changes.
- `/home/dev_patel/182-project-dev` is a worktree of **dev**. Main was merged into dev, preserving the FFT experiments. Main's deletion of the obsolete `variants/t2-gpu-port/train_gpt.py` won the only merge conflict. The obsolete committed torch-2.11 FA3 loader patch in the dev Track-2 trainer is also removed in this change.
- Ran `bash 182/tools/node_setup.sh --mode venv`. The resulting `.venv-run` imports **torch 2.10.0+cu128 and Triton 3.6.0**. The old `~/venv` is retained for the existing CUDA 13 runtime library; its torch is not used.
- The host has glibc older than the pinned FA3 binary requires. Execution therefore uses the existing Ubuntu 24.04 `nanogpt-run:gcc` image, extended with Git as `nanogpt-pinned:20261112`. The pinned host venv is mounted into the container. No host driver or main trainer was changed.
- Expose `/usr/local/cuda-13.0/compat` and `~/venv/lib/python3.12/site-packages/nvidia/cu13/lib` on `LD_LIBRARY_PATH`. The driver is still **550.90.07**; logs record it, and timings should not be described as reproducing the records' driver-580 environment.
- The full dry run passed in the corrected container; its plan is logged at `../results/pinned-followup-20261112/dry-run.log` in the main checkout.

The initial attempts are retained, including the missing CUDA runtime path, host glibc mismatch, missing container Git, root-owned log/cache directories, and shared-extension preflight failures. The generated log directories were returned to the host user's ownership. Preps now run **inline (`PREP_BACKGROUND=0`)**, so t1/t2's different packages named `exact_match` are built and used sequentially rather than overwriting each other before the first run. The controller puts Inductor/Triton caches on writable `/mnt/work` paths.

## NanoGPT order

1. Run the three requested variants with the pinned venv. CPU phases use all 103 existing FineWeb training shards, linked into the root data directory. The GPU phase uses a separate view of the first 25 shards plus the same validation shard: its preflight assumes every training shard is full-sized and incorrectly rejects FineWeb's valid shorter final shard. No tokens are modified or padded.
2. For t2-gpu-port only, expose `kernels==0.11.7` through a separate `PYTHONPATH`. Its unmodified loader calls the older `get_kernel(repo)` API; the root's pinned `kernels==0.16.1` requires an explicit version/revision and uses a different Hub namespace. The torch-2.10/cu128 build was prefetched from model repo `varunneal/flash-attention-3` at `de87b9b5af06dd9984df595bef90b2eba44b181a`. Root/CPU phases retain kernels 0.16.1. The compatible loader import was checked in the container. This is an environment adjustment, not a source loader patch.
3. Only if a failed Track-2 phase logs CUDA OOM: retry t2gpu with `T2GPU_RUN_ENV="CANON_MASK=1 VAL_CE_CHUNKS=16"`, or t2cpu with `T2CPU_RUN_ENV="VAL_BATCH_SIZE=524288"`. Failed attempts remain alongside the retry records.
4. If successful t1cpu runs still average 3.29–3.32 (the operational meaning of “about 3.30”), run `T1CPU_RUN_ENV=CONFIG=380` in the separate `182/results/t1cpu-380` session.
5. Resume `bash 182/gpu_session.sh`, including the attno A/B and the existing gate for Hadamard. Already successful phases are reused by the session harness. Then run FNO. A failed main phase is recorded and does not prevent the independent FNO experiments from running.

The first literal multi-variant command was attempted and logged. The continuation uses sequential preps and a separately invoked GPU phase because of the native-extension and shard-check issues above. It preserves the requested training configurations and conditional retry policy.

The durable continuation is `182/tools/queue_pinned_fno.py`. It waits for the retained initial container before launching any subsequent GPU job, logs every command/exit code, makes a copy of each session bundle before a later bundle can replace it, and finishes with `completion.json`. It checks the conditional follow-ups both before and after the full session, running each fallback at most once. Missing/failed required phases, failed config-380 runs, and failed required invocations cannot be marked as overall success. It runs in a detached tmux session so closing the client does not stop the queue. It does not commit, push, or send messages.

Main session directory:

```
/home/dev_patel/182-project/182/results/session-4c43aac0139b-pinned/
```

Controller logs, status ledger, setup logs, runtime-image Dockerfile/build log, FA3 revision, and copied bundles:

```
/home/dev_patel/182-project/182/results/pinned-followup-20261112/
```

The session harness writes `182/results/session-YYYY-MM-DD-4c43aac0139b.tar.gz`. The controller preserves labeled copies for initial runs, retries, config 380, and the final full session. **A bundle existing is not evidence that its training phases passed**: check `summary.txt`, `status/`, and the per-run result records.

## FNO experiments

Default queue order:

| Stage | Runs | Question |
|---|---|---|
| `controls-darcy` | AdamW; AdamW with betas (.95,.95); radial complex Adam with those betas; shared-factor SOAP; mode SOAP; equally sized mixed-mode SOAP blocks | Are the gains explained by betas, complex second moments, factor sharing, or generic small blocks? |
| `ns-boundary` | Mode SOAP centered at LR .027, initially testing .009/.027/.081; independently tuned AdamW | Does the old NS result improve beyond its LR boundary? |
| `refresh-darcy` | Mode SOAP every 10/25/50/100 steps, plus AdamW and flat SOAP | Can less frequent eigenbasis updates improve time to accuracy? |
| `refresh-ns` | Same refresh intervals and references, centered on the NS LR range | Does the benefit survive on NS? |

`controls-burgers` is available through `FNO_STAGES` but is not in the default queue. Model-width/mode-count scaling and public-dataset replication are deferred until the controls resolve the mechanism.

Every arm/refresh setting receives an independent ratio-3 LR search on seed **100**, initially three points, with up to two edge extensions. A remaining optimum at a boundary is explicitly labeled. Confirmation uses paired fresh training seeds **101, 102, 103** at the selected LR. Deterministic successful jobs can be reused across stages with identical configurations. Failed jobs and their consoles are retained under `.failed-...` names before a resume retries them.

The shared SOAP arm shares channel covariance/eigenbasis factors while retaining separate rotated Adam moments for each mode. The mixed-block control preserves input-channel rows and uses a fixed permutation of mode/output columns, with the same number and sizes of matrices as mode SOAP. The complex-Adam control uses one real `|g|²` moment per complex entry, whereas PyTorch AdamW treats real and imaginary coordinates separately.

SOAP variants retain the existing `fftopt.core` math, including its fresh-eigendecomposition/second-moment convention. This round isolates the proposed controls; it is not a claim of exact parity with the authors' QR-based SOAP implementation. Original optimizer code is unchanged.

## Data and metrics

- Deterministically split the old **training** set into **800 train / 200 validation**, with split seed 182. Training normalization uses only those 800 examples.
- Never tune on or evaluate the old 200 test examples. Generate independent 200-example test sets using seed **20261112** and the same PDE generators/settings. Generator seed arguments preserve the original defaults for old experiments.
- LR-search jobs evaluate validation only. Confirmation jobs evaluate the new test set once, after the final training step. Do not use the reported test scores for another tuning round.
- The NS problem remains the locally generated **T=2** map, not a long-horizon public benchmark. New results have fewer training examples and a different stack from the old exploration and should not be spliced into the old table.
- Record validation error versus steps, accumulated training seconds, and elapsed seconds; final test relative L2; train/total time; spectral/all-optimizer state; and peak allocated/reserved GPU memory.
- Report mean ± sample standard deviation, successful/planned seeds, and reached/planned targets. Fixed exploratory validation targets are Burgers .001563, Darcy .01051, NS .006132. Unreached seeds stay in the denominator; conditional time-to-target summaries are explicitly labeled “reached only.” Raw curves support comparison at common elapsed times.
- Data hashes, source hashes, stack, manifest, commands, per-job consoles, and failures are logged. A source/data/torch change refuses to reuse an existing prepared output directory.

## Run and monitor

Inside the compatible container with the pinned venv activated, from the dev worktree:

```bash
DRY_RUN=1 bash 182/fft-optim/queue_fno.sh
bash 182/fft-optim/queue_fno.sh
# Optional extra stage, using the same compatible stack:
FNO_STAGES=controls-burgers bash 182/fft-optim/queue_fno.sh
```

`FNO_OUT` overrides the result directory; the default is `182/results/fno-followup-YYYYMMDD-<dev-sha>/`. `FFT_DATA` overrides `/mnt/work/dev_patel/data`. `NGPU` controls worker GPU slots (default 8). The queue writes `queue.log`, per-stage logs, `status.tsv`, per-job JSON/consoles, `SUMMARY.md`, and a sibling `.tar.gz` bundle. A running queue has an output-directory lock. No result is committed automatically.

Monitor the deployed continuation and initial runs:

```bash
tmux attach -t pinned-fno
docker logs --tail 30 pinned-reruns-20261112
tail -n 30 /home/dev_patel/182-project/182/results/pinned-followup-20261112/controller.log
cat /home/dev_patel/182-project/182/results/pinned-followup-20261112/status.jsonl
```

Tests cover invertible block views, radial Adam, shared-factor shapes, train/validation isolation, validation-only LR selection, independent seeds, CPU training without test access, conditional retry decisions, and durable failure logging. They run without GPUs; full experiment outcomes are determined by the queued jobs, not by these tests.

Verification before launch: 7 FNO tests and 7 controller tests passed; the existing tools suite ran 121 tests with 7 optional skips and no failures. The FNO dry run, shell syntax check, Python compilation, and whitespace check passed. An independent read-only review identified the controller failure-accounting and post-full-session retry gaps; regression tests now cover both. GPU experiment results were still pending when this queue was prepared.
