# t2-cpu-ngram: PR #380's CPU-retrieval stack on Track 2 (GPT-2 Medium, val ≤ 2.92)

`variants/t2-cpu-ngram/` (a full runnable tree), based on upstream PR #380. Credits: the exact-match retrieval is PR #367 (@hermabr). The
exact-count chain, smaller transformer and extra hints are PR #380 (@devenpzak). The Track 2 sizes come from
@ldmberman's adaptation, posted in the #380 thread on 2026-10-04: **2.8769 in 221 s** (`step_avg` 38.1 ms,
5800 steps) on 8xH100. That is their run, not ours, and it is **one run** (seed 1337; the neural+copy loss alone was
2.8991 before the exact-count chain). For scale, the official Track 2 record is 17.35 min and
the best open GPU PR (#261, @Lisennlp) is 14.34 min. We have not run anything on GPU.

## What changed
- `train_gpt_medium_cpu.py` (new): the Track 2 entry point. It sets these defaults, each overridable by the env
  var of the same name: `MODEL_DIM=1024 MODEL_LAYERS=16 MLP_HIDDEN_DIM=4096 LR_COOLDOWN_FRAC=0.70 LR_FLOOR=0.10
  VAL_BATCH_SIZE=1048576 NUM_SCHEDULED_ITERATIONS=5790 TARGET_LOSS=2.92`. Then it runs `train_gpt.main()`.
  The max train batch stays 131072 tokens. The sampled-softmax switches (10240/14336/24576/full at steps
  0/3513/4800/5443) follow from the stretched stage boundaries; PR #380's code needed no change for that.
- The package changes are env knobs. **Their defaults reproduce Track 1 exactly**, so `train_gpt.py` runs as before:
  `config.py` (dims, layers, val batch, cooldown, target), `schedule.py` (`LR_FLOOR`), and `model/gpt.py`. The
  gpt.py change is @ldmberman's depth generalization: `LAST_LAYER`, layers 10–14 as plain attention+MLP layers,
  an MLP bank of 20 slots with dead slots frozen. Those files (config.py, schedule.py, model/gpt.py) match their
  attachment apart from comments and the defaults (their `training.py` also has a `cluster_logits` branch, used
  only by the unported heads). Not every package file does: `exact_counts.py`, `retrieval.py` and `run_log.py`
  were changed here (fail-loudly paths, host preflight, logging), and some modules here are not in the
  attachment at all. Note that, as in PR #380, every attention layer after layer 7 (so 8 and 10–15) reads the same input,
  norm(cache[7]).
- `train_gpt.py`: `num_layers=MODEL_LAYERS`, `PRINT_EVERY` env, and three new log lines: the memory budget,
  `model_configuration:`, and the measured allocation after setup. It also prints one `endpoint_result:` line,
  after the clock has stopped. `run_log.py` also logs `train_gpt.py` when the entry point is the wrapper.
- `track_1_short/memory_budget.py` (new): the startup memory estimate (see below).
- `track_1_short/host.py` (new, ported from `variants/t1-cpu-ngram`): before the clock, the log gets the CPU model,
  allowed CPUs, RAM, /dev/shm and THP, and the retrieval's host-memory estimate. Unless `HOST_CHECK=0`, the run
  fails there if /dev/shm or the available RAM (clamped to the cgroup `memory.max`) cannot hold it. After the
  run, the log gets the peak host RSS.
- Failure paths in PR #380's code (no change when nothing fails): an exact-count corpus read error or short
  read now raises instead of hanging, and a failed validation-index build now raises on every rank instead of
  being queried half-built or leaving the peers at a barrier. A failed table create removes its /dev/shm file.
- Not ported: the hierarchical and adaptive output-head variants (~770 lines) in @ldmberman's attachment. The
  2.8769 run used `variant: sampled`, i.e. PR #380's own head.

## Memory at d=1024 (estimate, printed at startup)
`python -m track_1_short.memory_budget` prints the estimate offline. The defaults are Track 2's, and you
can pass `--model-dim/--layers/--mlp-hidden/--val-batch/--steps`.
- **GPU, per rank: ~59.8 GiB peak allocated of 79.6 GiB** (logged: 61242 MiB alloc / 70578 MiB reserved). The
  n-gram table shard grows with d: NGRAM_DIM = MODEL_DIM, so it goes from 10.2 to **20.3 GiB**. Exact counts
  take ~6.3 GiB (corpus share + 4 GiB hash table, independent of d). Activations and the CUDA-graph pools are
  ~28.6 GiB, and change little with d. That last term is a constant calibrated on the Track 2 log. Checked on
  the Track 1 log, which it was not fitted to: predicted 47.6 GiB vs 47.5 logged.
- Headroom is ~9 GiB reserved. That is why VAL_BATCH_SIZE stays at 1048576. A wider model (d ≥ 1152) adds
  ~2.5 GiB of table per 128 channels.
- **Host:** model construction is fp32 on the host for all 8 ranks at once, a transient ~12.6 GiB. The
  canonical-mask child takes 2.4 GiB. Everything else (Rust indexes, corpus reads) does not depend on d.
  The retrieval needs **144 GiB of /dev/shm** for its three tables, plus ~148 GiB of build arenas, about 390 GiB
  of host RAM in all with 12 GiB per rank (`host.py`'s estimate; the startup check prints it). It still needs
  ≥ 24 vCPUs/GPU for `LARGE_HOST`.

## Track 2 rules notes
- Same rules as Track 1 at 2.92: p < 0.01 that the mean is ≤ 2.92; train/val streams unchanged. The README sets no
  model size for Track 2 ("All other rules remain the same"), so the 528.5M parameters (plus the n-gram table) are fine.
- The whole CPU-retrieval line (#367, #380, #381) is **unmerged**. @ClassicLarry (2026-10-03, #367) said CPU/RAM
  rules are coming, so that "we don't see run-away CPU farms". A cap on vCPUs or host RAM could slow this run or
  rule it out. The log's host lines are the disclosure.
- `torch._inductor.config.coordinate_descent_tuning` is **allowed on the medium track only** (README, Rules
  note). `COORD_DESCENT_TUNING=1` turns it on in `train_gpt_medium_cpu.py`. It is never set by `train_gpt.py`
  (still banned on Track 1). It costs a long untimed compile, so use it only for the final significance runs.
  Its gain on this stack is unmeasured: most hot paths are hand-written Triton plus CUDA-graph replays.
- Download **all 103 train shards** (`python data/cached_fineweb10B.py 103`). The exact-count corpus is every
  shard the loader cannot reach: 12 are excluded at 5800 steps, fewer at shorter runs. So the shard count on
  disk changes the hints, and with them the loss.

## Commands on 8xH100
```bash
# from variants/t2-cpu-ngram/; the shards live at the repository root
pip install -r requirements.txt && pip install ./exact_match && python ../../data/cached_fineweb10B.py 103
python -m track_1_short.memory_budget                                         # offline budget
DATA_PATH=../.. torchrun --standalone --nproc_per_node=8 train_gpt_medium_cpu.py   # 1) reproduce ~2.877 @ 5800
python 182/t2-cpu-ngram/sweep_steps.py --steps 4000 4600 5200 --seeds 0       # 2) coarse step sweep
python 182/t2-cpu-ngram/sweep_steps.py --steps <N-50> <N+50> --seeds 0 1      #    refine near the fitted N
COORD_DESCENT_TUNING=1 python 182/t2-cpu-ngram/significance.py --steps <N> --runs 6 \
    --markdown records/track_2_medium/<date>_CPURetrieval/statistics.md        # 3) record runs, p<0.01
python 182/t2-cpu-ngram/sweep_steps.py --fit-only                             # refit any time
```
- **Sweep fit:** `sweep_steps.py` fits `val_loss = a + b·ln(steps)` over every run. @ldmberman's 5800-step
  point is included unless `--no-seed-point`. It solves for the target mean, `2.92 + t₀.₀₁,₅·s/√6`, which is
  2.918 at s ≈ 0.0014, using the measured seed std once there are repeats.
- **Data:** the sweep and significance scripts use `DATA_PATH=../..` when it is unset and the variant folder has no
  shards. They refuse to launch without all 103 train shards (`ALLOW_PARTIAL_SHARDS=1` overrides), because the
  exact-count corpus, and so the loss, depends on how many are on disk. The entry point prints a warning in that case.
- **Results and resume:** results append to `182/t2-cpu-ngram/results/sweep.jsonl` (git-ignored), so the sweep
  can be resumed.
- **Significance:** `significance.py` exits 0 only if p < 0.01.
- **Run cost:** each run is a fresh torchrun: ~7 min compile/capture + ~3–4 min training.

## Verified here (CPU, no GPU)
- `python -m pytest -q 182/t2-cpu-ngram/tests` (21 tests). Triton, FA3 and the Rust extension are stubbed, so the
  model is built on the meta device, not run. At the Track 2 settings it has exactly the logged 528,536,207
  parameters and bank order. With the defaults its parameters, buffers and topology constants equal PR #380's.
  The tests also check the schedules at 3000–5790 steps, the run helpers, and the failure paths above (a two-rank
  gloo test for the build).
- A forward/backward on CPU is not possible: FA3 and the Triton/NVRTC kernels are CUDA-only.
- `py_compile` on every changed file.
- Track 1 defaults: the gpt.py topology constants equal HEAD's, and the schedule reproduces the Track 1
  record's boundaries `[(0,204)…(715,725)]`.
- Track 2 env reproduces @ldmberman's logged boundaries `[(0,1653),(1653,3513),(3513,5443),(5443,5790),(5790,5800)]`,
  the P switches, LR floor 0.1, and the bank order `{0,2,5,10–14,1,8,3,15}`. The built model has exactly
  the logged 528,536,207 params.
- The t-test matches scipy to 1e-15, and reproduces the PR #380 record logs' mean (3.27588). The parser reads
  @ldmberman's log and our new `endpoint_result` line.

## Still needs an 8xH100
Every training claim is still unverified on GPU:
- that this variant reproduces ~2.877 at 5800 steps
- the loss-vs-steps curve and the step count that gives mean ≈ 2.918
- the seed std at d=1024
- the effect of coordinate_descent_tuning
- the real peak memory
