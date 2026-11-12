# variants/t2-gpu-port — Track 2 (GPT-2 Medium, val loss ≤ 2.92), GPU only

The rule-clean Track 2 tree: no CPU retrieval, only GPU-side changes to `train_gpt_medium.py`. It is
**PR #261** (Lisennlp, MUDD connections; reported 14.34 min = 860.5 s, val loss 2.9171 ± 0.0007, n=6, on
torch 2.10.0+cu128 / triton 3.6.0) plus a few flags that are all **off by default**. The official Track 2
record is 17.35 min.

**Nothing here has been run on a GPU by us.** With every flag off the model, the kernels, the schedule and
the timed region are #261's (how that was checked is below).

## Run on the node

Use the repo root's environment (torch 2.10.0+cu128, triton 3.6.0: the stack #261 was measured on).

```bash
variants/t2-gpu-port/182/run_node.sh prep     # once: 25 data shards (the root record needs 9), FA3 wheel, tokenizer, mask check
variants/t2-gpu-port/182/run_node.sh check    # preflight, no GPU work: PASS|WARN|FAIL lines with the fix, exit 1 on FAIL
variants/t2-gpu-port/182/run_node.sh run      # one timed run of PR #261's configuration (~22 min wall)
```

`run` repeats the preflight, runs `torchrun --standalone --nproc_per_node=8 train_gpt_medium.py` in its own
process group with a timeout (90 min; 3 h with `COORD_DESCENT_TUNING=1`), tees stdout to `runs/<time>-run.out`
and ends with exactly one of

```
RESULT variant=t2-gpu-port val_loss=2.9161 train_time_s=860.467 steps=4150 target=2.92 hit=1 log=<path>
RESULT variant=t2-gpu-port status=failed reason=<one line>          (exit code != 0)
```

A run is about 14.4 min of training plus about 7 min of compile and warmup.

| Variable | Meaning |
|---|---|
| any flag from the table below, or `EXTRA_ENV="K=V K=V"` | turns that flag on for the run |
| `NPROC` (8) | GPUs; must divide 8 (the trainer accumulates 8/NPROC batches) |
| `DATA_PATH` (repo root) | folder that holds `data/fineweb10B/` |
| `LOG_DIR` (`runs/` in this folder) | where the captured stdout goes; the trainer's own log is `logs/<uuid>.txt` |
| `DRY_RUN=1` | print the exact command and exit 0; nothing is created |
| `RUN_TIMEOUT`, `SKIP_CHECK=1`, `PREP_TIMEOUT` (3600), `PREP_SHARDS` (25), `PYTHON` | timeouts, skip the preflight, interpreter |
| `SEED` | ignored with a note: #261 does not seed its RNGs |

What `check` looks at: Python ≥ 3.10; torch is a CUDA build that `varunneal/flash-attention-3` has a wheel
for (2.10 + CUDA 12.8 expected); triton 3.6.0; `kernels`; numpy; the tiktoken GPT-2 vocabulary; CUDA headers
(`cuda_bf16.h`) and a real NVRTC compile of both cross-entropy kernels for sm_90 (CPU only, ~2 s); GPU count,
model and memory; the val shard and at least 20 train shards (one run reads 1.905 B tokens, 25 shards
recommended); RAM, `/dev/shm`, vCPUs per rank, free disk.

## What to run first, and what each result means

**1. `CANON_MASK=1 variants/t2-gpu-port/182/run_node.sh run` (one run, ~22 min).** This is the base run and
the canonical-mask measurement at once. The mask only changes the final validation, so the trained weights
are a normal #261 run's. After the timed result the trainer scores the same weights once more with the mask
cleared (off the clock) and logs both; `run_node.sh` prints them just above the RESULT line:

```
CANON variant=t2-gpu-port val_loss_masked=<x> val_loss_unmasked=<y> delta=<y-x> canon_ms=<ms>
```

- `val_loss_unmasked` near 2.917 and `train_time_s` near 860 says the node reproduces #261.
  (`train_time_s` includes `canon_ms`, the wait + copy + broadcast of the mask at the end, and the mask
  build that runs in a forked process on rank 0 during the first seconds.)
- `delta` is the mask's gain on identical weights, so one run measures it without run-to-run noise. It
  cannot be negative. Track 1 measured about 0.0017 at loss 3.28 (PR #350). #261's own logs lose
  7.4e-5 of loss per step at the end (207 ms per step), so 0.0017 would be worth about 23 steps, about 4.8 s.
- Keep the mask if `delta` ≥ 0.0005 and `canon_ms` is under about 1000. Drop it if the final validation
  runs out of memory even with `VAL_CE_CHUNKS=16`, or if `delta` is smaller than that.

**2. `variants/t2-gpu-port/182/run_node.sh ab` (4 runs, ~1.5 h).** Interleaved A B B A:
A = `CANON_MASK=1`, B = `CANON_MASK=1 FUSED_MLP=1`, both with `VAL_LOSS_EVERY=0` (final validation only).
`FUSED_MLP` is the pure-speed lever: the same MLP arithmetic in bf16 through one fused Triton kernel.
The summary prints per-arm means, the paired `B-A` time and loss differences, and the mask delta of all runs.

- #261's six runs have a time standard deviation of 0.28 s, so a 1 % change (8.6 s) is far outside noise
  even with two pairs.
- Keep `FUSED_MLP` if B is faster in both pairs by at least 4 s (0.5 %) and the unmasked loss difference is
  within ±0.0015 (loss sd is 0.0007 per run). Drop it if the gain is under 1 s, if it is slower (at d=1024
  the GEMMs dominate more than at Track 1's d=768, where PR #197 measured about 2 %), or if it fails to
  launch (its kernel has never run at these shapes; forward shared memory is 229408 B of the H100's 232448).
- `AB_RUNS` is the number of pairs (default 2, not 4, because each pair costs about 45 min);
  `AB_A` / `AB_B` choose other flag sets, e.g. `AB_B="CANON_MASK=1 XSA=1"`.

**3. Spend what survived.** If the mask (and possibly `FUSED_MLP`) hold, lower
`NUM_SCHEDULED_ITERATIONS` from 4125 by about `delta / 7.4e-5` steps and confirm the mean stays ≤ 2.92 with
enough runs for p < 0.01. #261 sits at 2.9171, so there are only about 0.003 of slack before any change.

The other flags (prefix loss, XSA, coordinate-descent tuning) are second tier: each needs loss runs with
noise 0.0007 and so several runs per arm. Try them only after 1 and 2.

## What changed relative to PR #261 (all behind environment variables)

| Flag | What it does | Ported from |
|---|---|---|
| `CANON_MASK=1` | At the final validation, masks the (prev, cur) token pairs GPT-2's tokenizer never emits. Can only lower the loss. Rank 0 builds the 316 MB bit mask in a forked child on the clock; it is waited on, copied and broadcast before the clock stops. Intermediate validations are unmasked. Logs the unmasked loss of the same weights too (off the clock). | Track 1 #91 (PR #350, jvarho); background build from #92 (PR #360, DevenPzak), `medium_extras.py` |
| `VAL_CE_CHUNKS=N` | Row chunks of the validation cross-entropy: 4 as #261, 8 when `CANON_MASK=1` (the mask adds a gather to every logit block). Per-token losses are the same. | — |
| `FUSED_MLP=1` | Runs the MLP through `triton_kernels.FusedLinearReLUSquareFunction`, which #261's file already contains but Medium never calls. | Track 1 #59 (PR #197, andrewbriand, jrauvola) |
| `PREFIX_WEIGHT=w` | Prefix-token prediction auxiliary loss inside the fused CE kernel: `w` until `PREFIX_HOLD_FRAC` (0.2854), then 0.8w, 0.6w, 0.4w, 0.2w until `PREFIX_END_FRAC` (0.6063), then 0. Uses a separate kernel compiled only when the flag is on. The table is built on the clock, one bucket per rank. | Track 1 #88 (PR #337, jvarho); bucketed build from #92 |
| `XSA=1`, `XSA_LAYERS=all\|i,j,..` | Learnable per-(layer, head) exclusive self-attention, `y -= tanh(α)·(⟨y,v⟩/‖v‖²)·v`, α starts at 0. | Track 1 #82 (PR #264, djdumpling), rewrite of #86 (PR #317, aryavohra) |
| `ATTN_GATE=0` | Removes the sparse attention gate, as #86 did once XSA was in. | Track 1 #86 |
| `COORD_DESCENT_TUNING=1` | `torch._inductor.config.coordinate_descent_tuning`, allowed on this track. #252 measured about 5 s; adds 30+ min of compile. | #252 |
| `NUM_SCHEDULED_ITERATIONS` (4125), `NUM_EXTENSION_ITERATIONS` (25), `COOLDOWN_FRAC` (0.70), `MUON_LR` (0.015), `ADAM_LR` (0.004), `VAL_LOSS_EVERY` (250) | Step and learning-rate knobs, defaults are #261's values. | — |

### What is the same as #261 with every flag off, and how that was checked

- `triton_kernels.py` up to its last section is #261's file byte for byte (a test compares the SHA-256). The
  prefix cross-entropy is a separate kernel, custom op and autograd Function appended after it, built from
  #261's kernel source by six textual substitutions and compiled only for `PREFIX_WEIGHT > 0`.
- The model calls the base cross-entropy with #261's eight arguments and has #261's parameters and buffers
  (tested on CPU with the real model source at a small width).
- The diff of `train_gpt_medium.py` against #261 was read line by line. Outside flag branches it contains:
  the run log also includes `triton_kernels.py` and `medium_extras.py`; one `flags:` line is logged;
  hyperparameters are read from the environment with #261's values as defaults; a zero prefix-weight tensor
  rides along in the schedule config unused; `import medium_extras` loads the tiktoken vocabulary at start.
  None of these is inside the timed region.
- #261's `train_gpt.py`, `run.sh` and `Dockerfile` are not in this folder: they are the Track 1 trainer of
  that PR's base, a launcher for it, and an image with a torch nightly this tree was not measured on.

### Verified on CPU

`python -m pytest 182 -q` from this folder (38 tests, about 90 s):

1. Canonical mask: 0 masked adjacent pairs in about 228k tokens of real text; 2.80 % of all pairs are
   masked; EOT is never masked; the torch bit-unpack matches numpy's.
2. Prefix table: the 8 per-rank buckets max-reduce to the full table; entries are the longest proper prefix.
3. Kernels file is #261's plus the prefix section; each of the six kernel edits applies exactly once.
4. Prefix kernel arithmetic: a Python transliteration matches autograd for loss and dlogits, including
   duplicate targets, a prefix target that collides with another target, and an out-of-range prefix target.
5. Model smoke test of every flag and the combined stack with the real `GPT` / `TrainingManager` source:
   every parameter gets a gradient, optimizer tables cover the parameters, XSA at α=0 is bit-identical,
   an all-zero mask is a no-op and a sound mask lowers the loss, `FUSED_MLP` matches the inline MLP up to
   bf16 rounding.
6. `run_node.sh`: parsing of a real #261 record log and of truncated, NaN, timed-out and failed runs;
   the exact RESULT line; dry runs; the preflight on this machine (fails with reasons) and on fake nodes
   (complete, too few shards, too few or too small GPUs, wrong torch, missing wheel, NVRTC failure); a full
   `run` and `ab` against a stand-in interpreter, including the timeout kill.

`bash 182/compile_check.sh` (docker, `triton-cpu:3.6.0`: torch 2.10.0, triton 3.6.0, NVRTC 12.8.93)
**compiles** every kernel for sm_90a without a GPU: the base CE kernel and the prefix CE kernel through
torch's own NVRTC call and `ptxas`, and the `FUSED_MLP` Triton kernel forward and backward at Medium's
shapes (K=1024, N=4096, per-rank token counts 16384 to 65536 and the 262144-row validation batch).

On the node, `prep` also runs `182/check_canon_val.py`, which counts masked adjacent pairs in the real
10,485,760 validation tokens. It must print 0 before any `CANON_MASK` result is trusted.

### Not verified (needs the 8xH100 node)

- Any loss or time. No speedup is claimed.
- That the kernels compute the right thing on the GPU: the compile check executes nothing. The prefix
  kernel's arithmetic is only checked through the Python transliteration.
- `FUSED_MLP` at d=1024/4096 under `torch.compile` and its speed.
- `CANON_MASK` final validation memory at 8 chunks (training peaks at 59977 MiB allocated in #261's logs;
  the validation should need far less, but it was never run), the fork alongside Medium's data-loader
  threads, and the second (unmasked) validation pass.
- XSA, the prefix loss and `ATTN_GATE=0` under `torch.compile` (no graph breaks expected, not tried).
- `check`'s NVRTC compile and the FA3 wheel resolution on the node's real stack. `kernels` 0.16.1 was
  only inspected: it accepts the `torch210-cxx11-cu128-x86_64-linux` build the FA3 repository publishes.
- `prep`'s downloads (network), and whether 25 shards cover the tokens BOS alignment skips (20 is the
  arithmetic minimum; the loader stops with an error if it runs out).

## Upstream status (checked 2026-11-11)

PR #261 is open with a single commit (cdddeda8, 2026-04-27) and no maintainer decision; its only comment
(2026-07-08) asks whether Medium records are still being merged. No Track 2 PR was opened or updated after
2026-11-08, and there is no new rule decision for this track. Nothing was ported this round.

## Not ported, in priority order

- **FP8 MLP and packed FP8 QKV (Track 1 #84, #89, #90).** Medium disables FP8 in `CastedLinear` and runs
  the MLP in bf16. The port needs master's MLP kernels, delayed-scaling amax and new d=1024 scale constants
  (master's static x scale 2^-4 assumes max|x| ≤ √768; at d=1024 it needs 2^-3). Probably the largest speed
  win, and not checkable without a GPU.
- **Paired-head attention (#58).** A real architecture change with 8 heads × 128.
- **Medium-native features dropped by #188:** Snoo / EMA wrappers, Smear-MTP, the cubic window schedule.
- **Hashed n-gram table and sampled softmax (#92).** At d=1024 the table does not fit next to #261's 60 GB peak.
