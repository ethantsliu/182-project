# t1-cpu-ngram: the CPU n-gram retrieval line (Track 1)

A full runnable tree of upstream PR #381 (Daniel Monroe, "Improve input hints at test time", 725 -> 685 steps), which
is stacked on #380 (devenpzak, the exact-count chain) and #367 (hermabr, exact-match retrieval), plus our changes
behind environment variables. With every variable at its default the run does what #381 does, plus a host report
and a host-memory check before the clock. None of these PRs is merged upstream.

**Nothing in this folder has run on a GPU.** The numbers below are other people's, from their PRs.

| | time | val loss | where |
| --- | --- | --- | --- |
| official record (#92 / #360) | 39.9 s | <= 3.28 | 8xH100 |
| #380, 725 steps | 9.653 s (n=16, std 0.041) | 3.2759 (std 0.0016) | 8xH100, 2x Xeon 8462Y+ (128 vCPU), 2 TB RAM |
| #381, 685 steps (this tree's default) | 31.76 s vs #380's 33.62 s (n=3 each) | 3.2725 vs 3.2734 | **4x RTX PRO 6000 only** |

#381 has no 8xH100 time. Scaling its -5.5% to #380's node gives about 9.1 s, which is an estimate and not a
measurement. #367's own head is at 6.702 s (n=16), but it trains and validates on FineWeb-100B paths, a different
validation stream, so it is not adopted here.

**Upstream status, checked 2026-11-11:** no new commits on #380, #381, #367 or #399 since 2026-11-08 (#381's last
change is a title edit on 2026-11-07), no competing retrieval PR, and no ruling on CPU or RAM caps. The maintainer's
only statement is still the 2026-10-03 comment on #367 that rules will be set "so that we don't see run-away CPU
farms". The host report in every log (`host:`, `retrieval host plan:`, `host memory:` lines) is there so a result
from this tree can be judged against whatever cap arrives.

## The first node result: 3.3008, on another stack

The first node session ran `run` with the default `CONFIG=381`: 685 of 685 steps, val_loss **3.3008** (the target is
3.28), peak GPU memory 54 GiB, time not reported. That was not this tree's run on the records' stack. The node had
torch 2.11, not the pinned 2.10.0+cu128 with triton 3.6.0, and the agent running it had patched the FA3 loader in
`track_1_short/model/attention.py`. Neither change was committed, and the logs did not come back.

**What it tells us:** the tree trains to the end and validates on 8 GPUs, and its GPU memory matches #380's logs
(54,618 MiB reserved).

**What it does not tell us:** whether this tree reaches 3.28 on the pinned stack. The miss is not seed noise. On
8xH100 the expected value is about 3.2750, not #381's 3.2725 (see below). 3.3008 is 0.0249 above that, about 15
standard deviations of #380's 16 runs (0.0016), and 0.0212 above the worst of those runs (3.2796).

**What reading and testing the code found:**

- *The expectation.* #381's 3.2725 comes from 4x RTX PRO 6000 only, with whatever local changes it took to run there:
  the committed code asserts exactly 8 ranks (`track_1_short/distributed.py`). On that box #380's configuration gave
  3.2734; on 8xH100, #380's 16 runs averaged 3.2759. So #381 on 8xH100 should land near 3.2759 - 0.0009 = 3.2750.
  This is an estimate. Nobody has run #381's code on 8 GPUs.
- *World size.* Batch sizes in `config.py` are global (131,072 tokens per step). Every per-rank size divides them by
  the world size, and the step schedule does not depend on it. The only world-size-dependent part of #381's new code
  is its validation hint rows: a per-rank sample cap of `SCAN_HINT_CAP // world` (12 on 8 ranks, 24 on 4), the sample
  exchange, the row kernel's `WORLD * W` tally, and the reordering of the marks from rank order to stream order.
  `182/test_hint_rows_world.py` runs that code on 8 and on 4 gloo ranks under the Triton interpreter. The rows match a
  brute force at both sizes, with every context marked and with random marks. With the cap wrongly fixed at `96 // 4`,
  the test fails at 8 ranks. No port bug was found.
- *CPU count.* `VAL_BUILD_THREADS` and `LARGE_HOST` (`retrieval.py`) depend on the vCPUs per GPU. At 24 or more,
  rank 0 builds with 64 threads and the exact-count corpus is read through pinned buffers, a mode never measured
  upstream. The thread count only changes which entries a full table bucket keeps, so it affects the validation hint
  rows, not training. The log's `host:` and `retrieval host plan:` lines record which mode ran.
- *The attention kernel.* `model/attention.py` has no fallback. `get_kernel` either loads the pinned FA3 build or
  raises, and the binary's sha256 is asserted. The pinned repo ships one variant, `torch-stable-abi29-cu128-x86_64-linux`.
  kernels 0.16.1 accepts it only when torch was built for CUDA 12.8 or a later 12.x, so a torch 2.11 built for CUDA 13
  is refused. That is the likely reason the loader was patched. What the patch loaded instead is unknown. Forcing the
  same binary would run the same kernel (it already links `libcudart.so.13`). `kernels-community/flash-attn3` pads the
  (64, 128) heads: the same math, slower. Anything else (SDPA, or flex attention without the sliding windows and
  document masks) changes the model and could explain a miss this large. The code cannot decide which.

**What would settle the 3.3008 run, if its files survive on the node:** the trainer log
(`variants/t1-cpu-ngram/logs/<uuid>.txt`, or its copy in `logs/node/*.log.txt`). It starts with the full source as
launched, so its `track_1_short/model/attention.py` section is the patched loader verbatim. Its
`Running PyTorch ... compiled for CUDA ...` line names the torch build, and its `host:` and
`retrieval host plan:` lines give the vCPUs per GPU and the build threads. Also the output of
`pip list 2>/dev/null | grep -iE "torch|triton|kernels|flash"` from that environment.

**What makes the next run decisive** (all CPU-tested):

- `check` FAILs on a different torch or triton (`ALLOW_STACK=1` downgrades that) and now also when the trainer's
  source (`train_gpt.py`, `track_1_short/`, `exact_match/src`, `requirements.txt`) differs from the commit
  (`ALLOW_DIRTY=1` downgrades that).
- The log names the attention kernel the model calls: `attention kernel: pinned=1 ...` only for the pinned binary.
- After the clock stops, a `val diagnostics:` line gives the network's own loss before the chain and the share of
  validation positions whose hint row came from the 6-token index, the 3/4-token rows, the 2-token rows, or none.
  The final val_loss and the timed path are unchanged.
- `RESULT` carries `config= torch= triton= fa3= tree=` (torch and triton as the trainer logged them). `run` prints
  `WARN this run's ...` for an off-stack, unpinned or modified run, and a `reference:` line: the loss against the
  config's expectation in units of #380's std, with the network-alone loss and the hint-row shares.

**Rerun order:**

1. On the pinned stack, from a clean checkout: `bash 182/tools/node_setup.sh --mode venv && source .venv-run/bin/activate`,
   then `bash variants/t1-cpu-ngram/182/run_node.sh prep` and `SEED=0 bash variants/t1-cpu-ngram/182/run_node.sh run`.
   The RESULT line must say `torch=2.10.0+cu128 triton=3.6.0 fa3=pinned` and a `tree=` with no `-dirty`. At or below
   3.28 (about 3.275 expected), the stack or the patch caused the 3.3008.
2. If it is still above 3.28: `CONFIG=380 SEED=0 bash variants/t1-cpu-ngram/182/run_node.sh run`, #380's configuration,
   which has a published 8xH100 result (3.2759, range 3.2738-3.2796, n=16).
   - If 380 misses by a similar amount, the cause is in the part both share (training, the node, the stack), not in
     #381's hints. Compare the two runs' `network` losses.
   - If 380 lands in its range, the cause is #381's hint path on this node. Compare the two runs' hint-row shares,
     then run `CONFIG=381 EXTRA_ENV="SCAN_HINTS_SKIP=0"`. If the `host:` line shows 24 or more vCPUs per GPU, also run
     with `VAL_BUILD_THREADS=32`, the records' thread count.

## Run on the node

From the repo root, in the environment that has torch 2.10.0+cu128 (the root's `182/tools/node_setup.sh` makes it):

```bash
bash variants/t1-cpu-ngram/182/run_node.sh prep     # build the Rust extension, fetch all 103 shards (~20 GB), then check
bash variants/t1-cpu-ngram/182/run_node.sh check    # preflight: PASS/WARN/FAIL lines with the fix for each; exit 1 on any FAIL
bash variants/t1-cpu-ngram/182/run_node.sh run      # one timed run; the last line is RESULT variant=t1-cpu-ngram ...
```

`prep` is idempotent and resumable (`PREP_TIMEOUT_S`, default 7200). It needs rustc >= 1.80; Ubuntu 24.04's apt
package is 1.75, so `INSTALL_RUST=1 ... prep` installs rustup into `~/.cargo` if you allow it. `check` needs no GPU
and no torch. It prints the memory plan against the host, and it FAILs on: not Linux, a missing or different
torch/triton (`ALLOW_STACK=1` downgrades that), a trainer source that differs from the commit (`ALLOW_DIRTY=1`
downgrades that), fewer GPUs than `NPROC`, no `libcudart.so.13` or CUDA headers, the
extension not built from this source, fewer than 104 whole shards, RAM or `/dev/shm` below the plan, fewer than 4
cores per GPU. `run` runs `check` first and stops on a FAIL (`FORCE=1` runs anyway).

What `run` uses by default, and why: **`CONFIG=381`, this tree's own defaults (685 steps).** It is the base PR's
behaviour, it is the faster configuration on the one machine where both were measured, and its loss margin is not
smaller than #380's there. `CONFIG=380` runs #380's configuration on this code (725 steps, `SCAN_HINTS=0`,
`VAL_TOPK=2`): the only one with a published 8xH100 time, so it is the run to compare against 9.653 s if the default
looks off. It is not byte-identical to #380's code (the validation lookups also use the build cores).

Other knobs: `NPROC` (8), `SEED` (sets `TRAIN_SEED`; unset means a random init), `EXTRA_ENV="K=V ..."` or exported
trainer knobs, `DATA_PATH` (the directory holding `data/fineweb10B`; default the repo root), `LOG_DIR` (default
`logs/node` here), `DRY_RUN=1` (print the command, touch nothing), `RUN_TIMEOUT_S` (2400). The first run compiles for
about 7 minutes before the clock starts. Each run leaves `<LOG_DIR>/<stamp>.log.txt` (the trainer's log) and
`.console.txt` (stdout with a wall-clock stamp per line), and prints three things the log alone does not give:

- `timeline:` when each index build finished on the run's clock against the last training step, i.e. whether the CPU
  build is on the critical path. Nobody has published this for the line;
- `host memory:` rank 0's peak RSS and the node's memory in use;
- with `TABLE_VOTE=1`, the table vote's effect on this run's own weights (below).

The data sits on a disk by default, and the 6-token build reads 20 GB of it on the clock, so `run` reads every shard
once before launching to put it in the page cache. The records kept the data in `/dev/shm`:
`DATA_PATH=/dev/shm bash .../run_node.sh prep` copies it there (+20 GiB of RAM), then pass the same `DATA_PATH` to `run`.

### Host memory

Estimated host RAM on fineweb10B, in GiB (`track_1_short/host.py`; the trainer checks the same model before the
clock). The 12 GiB per rank is an allowance, not a measurement.

| build threads (vCPUs per GPU) | scan arenas | tables in `/dev/shm` | 8 ranks x 12 | total | with `SLOT_SHARED_ARENA=1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 (< 16) | 119 | 144 | 96 | 359 | 321 |
| 32 (16-23, the records' box) | 131 | 144 | 96 | 371 | 325 |
| 64 (>= 24) | 155 | 144 | 96 | 396 | 333 |

`/dev/shm` must have 144 GiB free (164 with the data there). Most distributions size it at half of RAM, so a 256 GiB
node fails on both counts and a 512 GiB node passes. The published runs had 16 vCPUs per GPU (32 build threads); the
64-thread mode was never measured in their logs.

## What to run first, and what each result means

1. **`run` (default config).** Confirms the tree works and gives the three lines above. The loss target is 3.28; #381's
   4-GPU mean was 3.2725, and about 3.2750 is expected on 8xH100 (the section on the first node result has the
   reasoning and what to run if it misses). If the `timeline:` line says the builds finished before the last training step, host-side
   build work cannot change the time on this node and the remaining host ideas at the bottom are dead here. If it
   says AFTER, the build is the bottleneck and `VAL_BUILD_THREADS` is the first thing to vary.
2. **`ab` (default: base vs `TABLE_VOTE=1`, 4 pairs, about an hour).** The only lever here that could shorten the run.
   One B run already answers the loss question exactly, because it reports the loss with and without the vote on the
   same weights; the pairs are for the time cost. #381 bought 40 steps with 11.3 mnat, about 0.28 mnat per step, and
   a step is about 12.6 ms. So:
   - **confirmed** if the within-run delta is at or below -0.0010 (about 4 steps, 45 ms) and the paired
     time cost is below that. Next: `EXTRA_ENV="TABLE_VOTE=1 NUM_SCHEDULED_ITERATIONS=671"` and re-check the loss;
   - **killed** if the delta is above -0.0005, or if the time cost exceeds the steps it buys.
3. **`AB_B="SLOT_SHARED_ARENA=1" ... ab`** only if RAM matters (a cap is ruled, or the node is under 400 GiB).
   Confirmed if `host memory:` drops by about 46 GiB (32 threads) or 62 GiB (64) with time and loss inside the A
   arm's spread; killed if the time CI excludes zero on the slow side.

## Our result: 668 scheduled steps (the default here)

#381 schedules 675 steps; this tree defaults to 668. `NUM_SCHEDULED_ITERATIONS=675` reproduces #381. Measured on one
8xH100 node (Lor, pinned stack), 675 and 668 interleaved with the same seed per pair, after a warm-up run:

| | 675 (upstream) | 668 (ours) |
| --- | --- | --- |
| mean time, 8 interleaved runs | 9.959 s | 9.828 s |
| mean val loss | about 3.2751 | 3.27688 (11 seeds, sd 0.00146) |

- Paired time difference: -0.131 s (se 0.039, 95% interval +/-0.093, t = -3.33); 668 was faster in 7 of 8 pairs.
  Without the first pair of each job (warm-up drift): -0.077 s (+/-0.061).
- Loss rule: the 11-seed mean is under 3.28 with t = 7.09 (p < 0.01). The margin is thin; the worst run was 3.2800.
- 662 steps was too few: 3.2795, 3.2771, 3.2784 over three seeds.

Jobs: `182/lor/jobs/ngram_steps.sh`, `ngram_confirm.sh`, `ngram_final.sh` at the repo root.

## What changed on top of #381

| knob (default) | what it does | tested on CPU |
| --- | --- | --- |
| *(always)* host report and check | `track_1_short/host.py`: CPU model, CPUs/cores/NUMA, vCPUs per GPU, RAM (clamped to a container's cgroup headroom), `/dev/shm`, THP, AVX-512 into the log; the estimated plan; the arenas and tables actually allocated; peak RSS at the end. Raises before the clock if the tables do not fit `/dev/shm` or the total does not fit the available RAM. `HOST_CHECK=0` only reports; `HOST_RANK_GIB` (12) is the per-rank allowance. | pytest |
| *(always; failure paths only)* | Three hangs in #381 raise instead: a failed or short corpus load left `launch()` waiting forever; a failed index build was still queried; a rank-0 build error never reached the other ranks. A successful run takes the same path as #381. | pytest (7 of these tests fail on #381's code) |
| `SLOT_SHARED_ARENA=0` | The 2- and 3-token builds reuse the 6-token index's scan arena instead of owning one each (the builds are sequential anyway; a lock enforces it). Saves the RAM in the table above, two prefaults before the clock and two ~34 GB unmaps during training. Tables are byte-identical single-threaded; multi-threaded, which entries a full bucket keeps depends on thread timing in #381 too. | cargo test |
| `TABLE_VOTE=0` | #399's table vote, ported as three more chain links: the 6-token table's candidate and hit counts at match lengths 6/10/20 (`SlotIndex.query_counts`), counted on the clock after the exact-count scan's launch, with weights fit on the fit steps like every other link. The run then reports, off the clock and on the same weights, the loss without the links (the chain refit without them) and with #399's fixed-k vote instead. The chain already holds exact order-2..8 counts over the same shards, so only the 10- and 20-token matches are new evidence: expect much less than #399's -14 steps. | cargo test (counts vs brute force); pytest (chain bookkeeping, #399's formula) |
| `VAL_BUILD_THREADS` (auto 64/32/16), `STREAM_THREADS=4` | Overrides for rank 0's build threads and each rank's online-index threads. The online index's rows are identical for any thread count. | cargo test |
| *(build)* `exact_match/src/sys.rs` | x86 intrinsics and Linux madvise/affinity behind scalar or no-op fallbacks, so the crate builds and its tests run elsewhere. AVX-512 is detected at run time on x86-64; no build flag is needed. | cargo test |
| *(always; reports)* attribution | The log's `attention kernel: pinned=0/1 module=... sha256=...` line (`run_log.py`, before the clock; the pinned binary's digest is the one the loader already checked) and, after the clock stops, its `val diagnostics:` line (`train_gpt.py`: the network's loss before the chain, the hint rows' sources). Neither changes the trained model, the final val_loss or the timed path. | pytest (the kernel line); the diagnostics' arithmetic only by hand on CPU tensors |
| `182/` | `run_node.sh` + `node.py` (the node contract: RESULT's `config= torch= triton= fa3= tree=`, the `trainer source` check, the `reference:` line), `test_node.py`, `test_failure_paths.py`, `test_hint_rows_world.py` (#381's hint rows on 8 and 4 gloo ranks under the Triton interpreter). | pytest |

Removed in this round: `TABLE_VOTE=fixed` as a run mode (its constants were fit for another stack, and the chain
form contains it; it survives as a line of the vote report), the old `182/t1-cpu-ngram/node_check.py` (now `check`),
and upstream's `train_gpt_medium.py` (Track 2; it never belonged to this variant).

How "flags off = #381" was checked: `DRY_RUN=1 run` sets nothing but `DATA_PATH` (a test asserts it); the diff of
`train_gpt.py`, `retrieval.py` and `exact_counts.py` against `upstream/pr/381` was read line by line, and every
added statement is under `if TABLE_VOTE`, in a failure branch, an argument that is inert with the flags off, or a
print outside the clock, except two: the host
check before the clock, and the build-time line on stdout, which now names its table. The `attention kernel:` line
(before the clock) and the `val diagnostics:` accumulation (in the untimed validation loop) came later and are reports
only. Not checked by running it.

## `EXACT_BACKEND=sa`: the exact counts from suffix arrays (correct, and slower)

**What.** `EXACT_BACKEND=sa` (default `gpu`) computes the chain's exact order-2..8 counts from a suffix array of
each rank's corpus share on the host, infini-gram style, instead of the GPU scan in `exact_counts.py`. The code is
`exact_match/src/sa.rs` (cargo feature `sa`, libsais 2.10.4 through `libsais-sys = "=0.2.0"`), and
`track_1_short/exact_counts_sa.py`, which has ExactCounts' interface. The hook is one line in `train_gpt.py`. Off by
default, and the default build does not compile the crate's new code.

**How the counts are made now.** The exact counts do not live in the big host tables. Those tables (the 144 GiB in
`/dev/shm` and the scan arenas, about 390 GiB in all) belong to the 6-, 3- and 2-token SlotIndexes that give the hint
rows. After training, each GPU hashes the run's 12,189,696 queries into a table: the 10,485,760 validation positions
plus 8 x 212,992 fit positions, 8 grams each. Then it scans its share of the 101 counted shards (about 1.3B tokens in
GPU memory, loaded during training) and reads off the counts, and the counts are all-reduced. Per GPU that is 2.6 GB
of corpus plus a 4.0 GiB table, and no host RAM. On the GPU this takes an estimated 0.1-0.3 s, which nobody has
measured.

**Same counts.** Each rank sorts the same share as the GPU (its files concatenated, so grams cross file junctions
but not shares) and answers each query with binary searches. That gives N for context lengths 2..8 and C for
context + target. Every query's start is cut at its segment, as on the GPU. The GPU reads a context's count from
its query table, so it returns 0 for a context that no query inserted, even when the context occurs in the corpus.
The SA backend copies that rule. The GPU's 64-bit hash collisions are the only possible difference. `182/sufarr/`
checks both against one exact reference: the suffix arrays (cargo tests, and pytest through real shard files and
the trainer's query layout), and the GPU kernels themselves under the Triton interpreter. The cases are random
BOS-led corpora (including broken targets) and a real fineweb10B sample (6 x 12,000 training tokens, 6,000
validation tokens).

**Memory and time on the record's corpus (10.1B tokens, 8 ranks, 128 vCPUs).**

| | GPU scan (default) | `EXACT_BACKEND=sa` |
| --- | --- | --- |
| host RAM | ~0 | 6 bytes/token (text 2 + 32-bit SA 4): 7.8 GB on a 1.3B-token rank, ~61 GB for all 8, plus 0.7 GB of counts per rank |
| `/dev/shm` | 0 | 0 |
| GPU memory | 2.6 GB corpus + 4.0 GiB table per GPU | 0.7 GB (the counts) |
| build, on the clock | the corpus copy, under training | the suffix sort of 1.3B tokens per rank, all 8 at once |
| after training | the scan, ~0.1-0.3 s (estimated) | 12.19M queries per rank on its threads |

Measured here (x86-64 build under Rosetta, Apple silicon, so only indicative): libsais16 on real fineweb tokens sorts
24 M tokens/s single-threaded at 4M tokens and 20 M/s at 32M. Queries run at 0.44 M/s per thread against a 32M-token
array (1.4 M/s at 4M). Scaled to a 1.3B-token share: the sort takes 90-130 s single-threaded, or an assumed 15-45 s
with OpenMP on 16 threads, while 8 ranks share the memory bandwidth and the trainer's own CPU work. The queries take
about 2.5-5 s (0.15-0.3 M/s per thread on 16 threads). Matching the GPU scan's ~0.2 s would need about 60 M
queries/s per rank, 10-25 times this. The whole run is about 9 s, and the build cannot start before the clock
without becoming a pre-built index of training data.

**Expected effect: +15 to +60 s per run, with no change in loss** (the counts are equal, so the chain and val_loss
are too). There is no break-even: at 164 ms per mnat, even +15 s would need about 90 mnat. A suffix array would only
pay if a RAM cap is ruled and the build is allowed to run longer than this run takes. The memory it could save is in
the SlotIndexes, not here: one 40-bit array over all 10.1B tokens needs 7 bytes/token (71 GB), against the ~263-299 GiB
of tables and arenas, but a build peak of 15 bytes/token and minutes of 64-128 threads. That is not implemented.
**Rule risk:** upstream has no ruling on CPU or RAM. This backend adds about 128 threads for tens of seconds and
~61 GB of host RAM on the clock, which is the "run-away CPU farm" the maintainer's #367 comment warns about.

**Limits.** The backend does not take SCAN_HINTS samples (the 381 config's default), so it raises unless
`SCAN_HINTS=0`, i.e. `CONFIG=380`. `SA_THREADS` sets the per-rank threads (default CPUs / world). The sort is
single-threaded unless the build has `sa-openmp`. `SuffixCounts.longest_into` (∞-gram longest match) exists and is
tested in Rust, but nothing calls it. The host check does not count the backend's RAM.

**On the node** (memory and time, no timed A/B; there is no `182/experiments.d/` file because the expected sign is
known):

```bash
bash variants/t1-cpu-ngram/182/run_node.sh prep                 # shards + default build
bash variants/t1-cpu-ngram/182/sufarr/build.sh                  # rebuild with --features sa-openmp (else sa); rustc >= 1.85
python variants/t1-cpu-ngram/182/sufarr/bench.py --data <DATA_PATH> --check-gpu             # 8 ranks at once, ~5-10 min
python variants/t1-cpu-ngram/182/sufarr/bench.py --data <DATA_PATH> --ranks 0               # one share alone: the floor
CONFIG=380 SEED=0 EXTRA_ENV="EXACT_BACKEND=sa" bash variants/t1-cpu-ngram/182/run_node.sh run   # optional: the cost on the clock
```

`bench.py` prints, per rank, the build time, the query time, the index size and the peak RSS. With `--check-gpu`
it also prints the GPU scan's time on the same queries and `mismatch=<rows>`, which should be 0. It ends with a
`RESULT sufarr build_s= query_s= index_gib= rss_gib=` line. The estimate above is confirmed if build_s is over about
10 s. It is wrong (and worth a timed run) only if build_s + query_s comes in under about 1 s. A nonzero mismatch is a
bug, or a hash collision in the scan if only a few rows differ. Tests: `(cd exact_match && cargo test --release
--no-default-features --features sa)` (12 tests); `python -m pytest -q 182/sufarr/test_sufarr_counts.py` after
build.sh (9; skipped without the feature); and from the repo root,
`bash 182/tools/triton_cpu/run.sh python -m pytest variants/t1-cpu-ngram/182/sufarr/test_sufarr_kernels.py -q` (5).

**Not verified:** any time on the node (the build and query rates above are scaled from a Mac under Rosetta);
libsais's OpenMP speedup and the `sa-openmp` build on Linux; `exact_counts_sa.py` inside a real training run (it was
tested on the CPU as `train_gpt.py` drives it); whether rustc 1.80 still builds the default crate now that the lock
file lists an edition-2024 optional dependency (cargo should not fetch it unless `sa` is on). Nothing here has run on
a GPU.

## Tests

```bash
cd variants/t1-cpu-ngram
python3 -m pytest -q 182                                           # 37 tests (6 more skip), no GPU, no built extension needed
(cd exact_match && cargo test --release --no-default-features)     # 6 tests on small tables
```

The six skipped tests (`182/test_hint_rows_world.py`) need real Triton under its interpreter; from the repo root:
`bash 182/tools/triton_cpu/run.sh python -m pytest variants/t1-cpu-ngram/182/test_hint_rows_world.py -q` (about a minute).

## Not verified (needs the node)

- Why the first node run gave 3.3008 (see the section above): the stack and the patched loader are the leading
  suspects, but nothing here can show it.
- The `val diagnostics:` line on the GPU: its accumulation was checked on CPU tensors, not in a run.

- Any run time or loss. No speedup is claimed.
- That the default config reaches 3.28 on 8 GPUs: #381 measured it on 4 (3.2725, n=3).
- `prep`'s download, the attention-kernel prefetch, and the Rust build on Linux with AVX-512. The build and the
  tests ran here on x86-64 macOS under Rosetta, which has no AVX-512, so those paths compiled but never executed.
- The `TABLE_VOTE=1` report's GPU code path (a second chain fit after the clock stops, then one extra chain per
  validation batch). Its bookkeeping is tested on CPU. An error in the per-batch part prints `nan` for the report
  and keeps the run's result; an error in the refit itself would lose that run's result line.
- `SLOT_SHARED_ARENA=1` end to end, and the host check's thresholds on a real box.
- The `timeline:` line's accuracy: it assumes rank 0's stdout reaches the runner within a few ms.

## Host-side ideas not implemented

1. The online index re-hashes and re-sorts the whole stream every phase on every rank; merging only each phase's new
   entries would remove most of that work.
2. A GPU index build (#367's blog estimates about 3.1 s). Only worth anything if the `timeline:` line shows the CPU
   build on the critical path.
