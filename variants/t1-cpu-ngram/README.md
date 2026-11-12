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
torch/triton (`ALLOW_STACK=1` downgrades that), fewer GPUs than `NPROC`, no `libcudart.so.13` or CUDA headers, the
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
   4-GPU mean was 3.2725. If the `timeline:` line says the builds finished before the last training step, host-side
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

## What changed on top of #381

| knob (default) | what it does | tested on CPU |
| --- | --- | --- |
| *(always)* host report and check | `track_1_short/host.py`: CPU model, CPUs/cores/NUMA, vCPUs per GPU, RAM (clamped to a container's cgroup headroom), `/dev/shm`, THP, AVX-512 into the log; the estimated plan; the arenas and tables actually allocated; peak RSS at the end. Raises before the clock if the tables do not fit `/dev/shm` or the total does not fit the available RAM. `HOST_CHECK=0` only reports; `HOST_RANK_GIB` (12) is the per-rank allowance. | pytest |
| *(always; failure paths only)* | Three hangs in #381 raise instead: a failed or short corpus load left `launch()` waiting forever; a failed index build was still queried; a rank-0 build error never reached the other ranks. A successful run takes the same path as #381. | pytest (7 of these tests fail on #381's code) |
| `SLOT_SHARED_ARENA=0` | The 2- and 3-token builds reuse the 6-token index's scan arena instead of owning one each (the builds are sequential anyway; a lock enforces it). Saves the RAM in the table above, two prefaults before the clock and two ~34 GB unmaps during training. Tables are byte-identical single-threaded; multi-threaded, which entries a full bucket keeps depends on thread timing in #381 too. | cargo test |
| `TABLE_VOTE=0` | #399's table vote, ported as three more chain links: the 6-token table's candidate and hit counts at match lengths 6/10/20 (`SlotIndex.query_counts`), counted on the clock after the exact-count scan's launch, with weights fit on the fit steps like every other link. The run then reports, off the clock and on the same weights, the loss without the links (the chain refit without them) and with #399's fixed-k vote instead. The chain already holds exact order-2..8 counts over the same shards, so only the 10- and 20-token matches are new evidence: expect much less than #399's -14 steps. | cargo test (counts vs brute force); pytest (chain bookkeeping, #399's formula) |
| `VAL_BUILD_THREADS` (auto 64/32/16), `STREAM_THREADS=4` | Overrides for rank 0's build threads and each rank's online-index threads. The online index's rows are identical for any thread count. | cargo test |
| *(build)* `exact_match/src/sys.rs` | x86 intrinsics and Linux madvise/affinity behind scalar or no-op fallbacks, so the crate builds and its tests run elsewhere. AVX-512 is detected at run time on x86-64; no build flag is needed. | cargo test |
| `182/` | `run_node.sh` + `node.py` (the node contract), `test_node.py`, `test_failure_paths.py`. | pytest |

Removed in this round: `TABLE_VOTE=fixed` as a run mode (its constants were fit for another stack, and the chain
form contains it; it survives as a line of the vote report), the old `182/t1-cpu-ngram/node_check.py` (now `check`),
and upstream's `train_gpt_medium.py` (Track 2; it never belonged to this variant).

How "flags off = #381" was checked: `DRY_RUN=1 run` sets nothing but `DATA_PATH` (a test asserts it); the diff of
`train_gpt.py`, `retrieval.py` and `exact_counts.py` against `upstream/pr/381` was read line by line, and every
added statement is under `if TABLE_VOTE`, in a failure branch, an argument that is inert with the flags off, or a
print outside the clock, except two: the host
check before the clock, and the build-time line on stdout, which now names its table. Not checked by running it.

## Tests

```bash
cd variants/t1-cpu-ngram
python3 -m pytest -q 182                                           # 31 tests, no GPU, no built extension needed
(cd exact_match && cargo test --release --no-default-features)     # 6 tests on small tables
```

## Not verified (needs the node)

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
