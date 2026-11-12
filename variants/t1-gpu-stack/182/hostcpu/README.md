# HOST_ASYNC: the training loader's host work on a producer thread (t1-gpu-stack)

Nothing here has run on a GPU.

## What

`HOST_ASYNC=1` (off by default, `track_1_short/folds.py`) makes the training loader an `AsyncBatchLoader`
(`track_1_short/data.py`). The loader body is split in two:

- `host_batch_generator`: the document plan (`Shard.next_batch`), the token gather, the targets shift, the
  cu_seqlens table and the host n-gram row hash. No CUDA call.
- `_upload_batch`: the copy into the next pinned slot and the H2D launch.

With the flag on, a daemon thread replays `host_batch_generator` up to `HOST_ASYNC_DEPTH` (default 4) batches
ahead of the consumer. The upload stays on the main thread, so no CUDA call, stream or event changes thread, and
the pinned ring is used in the same order. `ScheduledBatches` already knows every step's geometry, so it hands the
loader the full list of `.send()` arguments up front (`plan()`). The producer feeds the generator exactly that
sequence, so the batches come out identical and in the same order; `send()` asserts each argument against the
plan. The thread starts on the first fetch, not when the loader is built, so the first shard is still read on
the clock. It exits by itself after the last planned batch. `close()` stops it early, and an exception in the
producer is re-raised on the consumer's next fetch. Validation (`align_to_bos=False`) always stays synchronous.
With the flag off, the same generator runs with the same calls in the same order, on the main thread.

## Why, and what to expect

Per step, the host does the following. Medians are from `bench_host.py` on an Apple M1 Pro with one torch thread
and synthetic FineWeb-like tokens. The machine was shared with other jobs at the time, so read these as rough.

| host op | thread today | ms per step (stage 0 .. 2) | whole 1053-step run |
|---|---|---|---|
| document plan, #397's walk | main | 0.13-0.39 | 304 ms |
| document plan, `HOST_PREP=batchplan` | main | 0.036 | 38 ms |
| token gather + cum_lengths + segments | main | 0.12-0.35 | 255 ms |
| `_prepare_batch` (cu_seqlens table, casts, n-gram hash) | main | 0.21-0.57 | 404 ms |
| copy into the pinned slot | main (stays) | 0.02-0.04 | 30 ms |
| `RowOwners.request` host part (event steps) | main (stays) | 0.03-0.33 | 44 ms |
| sampled-softmax candidates | prep thread already | 0.3-0.9 | 585 ms |
| cycle want lists (event steps) | prep thread already | 10-35 | 7.1 s |

So the loader puts about 0.9 ms per step on the main thread with #397's walk, and about 0.66 ms with
`HOST_PREP=all`. The GPU step is about 29.4 ms (30.957 s over 1053 steps on the node). A server core may be
1.5-2x slower than an M1 core, which gives 1-2 ms per step.

**Is the host ever on the critical path?** Probably not, so the expected gain is about zero. The forward and
backward are CUDA-graph replays and the optimizer is graph-captured, so the host enqueues each step in far less
than 29 ms and then runs ahead of the GPU. How far ahead is capped by the pinned rings (16 batch slots, 4
candidate slots, 4 want-list slots, 8 scalar slots). The timed loop has no device sync: `val_loss_every` is 0, so
only the final validation syncs. Inside that loop, host time costs wall time only if the host falls behind the
GPU, i.e. if the whole per-step host time (graph dispatch, optimizer launches, loader) gets close to 29 ms, or if
a burst is longer than the run-ahead slack. The largest loader burst is at an event step, when the row prefetch
peeks up to 4 batches at once: about 2.6-3.6 ms here. That is under one GPU step of slack. Outside the loop, the
clock start (step 0's shard read, first batch and cold row fill) is unchanged by design: the producer starts on
the first fetch. The final validation's loader stays synchronous.

**The GIL limits what can move.** The producer runs Python. #397's document walk is pure Python and holds the
GIL for its whole run. The gather, hash and numpy parts release it in their C calls. So `HOST_ASYNC=1` takes off
the main thread only the GIL-free part of the build. The rest just moves the stall: the main thread waits for the
GIL while a build runs, for at most the length of that build. Use it with `HOST_PREP=all`, which takes the walk
out. `bench_host.py` measures this with a main-thread loop of small torch calls, timed while a producer builds
stage-2 batches (2 s per mode, M1, shared machine, so noisy):

```
 idle: mean   76.2  p50    56.2  p99   419.9  p99.9  1851.8  max  10553.7  (us)
 plan: mean  223.1  p50    57.7  p99  2805.7  p99.9  7696.2  max  32390.3  producer nonstop, HOST_PREP=all
 walk: mean  262.8  p50    57.7  p99  3498.6  p99.9  9067.2  max  13647.2  producer nonstop, #397 walk
paced: mean  194.3  p50    56.5  p99  1897.2  p99.9 10642.8  max 149619.0  one batch per 29.4 ms
```

The median does not move, but the tail does: a main-thread step can be delayed by about a build's length (1-3
ms) when the two threads collide. That is harmless while the host is ahead of the GPU, which is the same
condition under which the move cannot gain anything.

**Expected Δtime:** about 0 for both `HOST_PREP=all` and `HOST_PREP=all HOST_ASYNC=1`, with a range of -0.02 to
+0.01 s. That is below what 4 seeds can resolve (the author's sd of 0.017 s resolves about 0.05 s with 4 pairs).
The most the main thread could save is the loader's whole share, 0.7-1 s per run, and only if the host were the
bottleneck on every step, which the graph replay argues against. Both levers are exact, so the loss and the
164-250 ms/mnat exchange rate do not come into it: any time saved would be pure gain. A paired 95% CI below 0
would show that the host was on the critical path after all. In that case the next step is
`182/tools/profile_nsys.sh` on base: look for GPU idle gaps between steps that line up with host work.

## Prior art

- In tree: #387's `HOST_PREP=batchplan` (the document plan by prefix sums) and `HOST_PREP=ngramhash` (fused hash)
  shorten the same work rather than moving it. #397 already builds the sampled-softmax candidates and the cycle
  want lists on prep threads, and #127 (merged upstream, in the tree) loads the next shard and scans its BOS
  index on background threads.
- Upstream #231 (open): H2D copies on a separate copy stream. Its author reports that putting loader work in an
  optimizer hook caused NCCL errors with threads. Here the producer makes no CUDA or NCCL call, and the H2D stays
  where it was. #28 (closed): pinned memory and async H2D, both already in the tree.
- No upstream PR moves the per-step host build to a thread.

## How to run

CPU, from `variants/t1-gpu-stack`:

```bash
nice -n 10 env OMP_NUM_THREADS=2 python3 182/hostcpu/test_host_async.py   # ~7 s; also runs under pytest
nice -n 10 env OMP_NUM_THREADS=1 python3 182/hostcpu/bench_host.py [--quick] [--shard data/fineweb10B/fineweb_train_000001.bin]
```

`test_host_async.py` uses synthetic 5M-token shards in a temp dir. It checks that every field of every Batch
matches the synchronous loader's, bit for bit and in order: over the full step sequence of a shortened schedule
(all 4 stage boundaries) and over a warmup-like list of non-consecutive steps, for 3 seed/rank pairs, with
`HOST_PREP=batchplan` both off and on, across several shard switches. It also checks that uploads stay on the
consumer thread, the producer never runs more than the depth ahead, `close()` stops it, producer errors reach
the consumer, the plan is enforced, and validation stays synchronous.

On the node, in the A/B phase: `182/experiments.d/hostcpu.sh` (arms `base`, `HOST_PREP=all`,
`HOST_PREP=all HOST_ASYNC=1`). One-off:

```bash
SEED=1 EXTRA_ENV="HOST_PREP=all HOST_ASYNC=1" variants/t1-gpu-stack/182/run_node.sh run
AB_ARMS="base hostprep hostasync" variants/t1-gpu-stack/182/run_node.sh ab
```

The run log's `[folds]` line ends in `HOST_ASYNC=1 depth=4` when the flag is on.

## Unverified

- Everything on CUDA: pinned memory, H2D, events, and the real 8-rank run. The test fakes the staging and
  `torch.distributed`.
- Whether the host is ever behind the GPU. The argument above is from reading the code, not from a trace.
- The GIL numbers come from a loaded Mac with a synthetic dispatch loop, not the trainer's real Python.
- The partial-to-full BOS index switch depends on timing in #397 too. A shard's batches use its first 6M tokens'
  BOS index until the background scan of the whole shard finishes (about 0.43 s per 100M-token shard here). If
  the loader ran off the end of the partial index first, it would skip to the next shard. In the timed run that
  is about 45 stage-0 steps (about 1.3 s) away from the start. The producer gets there up to `HOST_ASYNC_DEPTH`
  batches (about 0.12 s) sooner than the synchronous loader would, so the margin is a little smaller. The test
  uses 5M-token shards, where the partial index is already the whole shard, so it does not cover this race.
- `HOST_ASYNC_DEPTH` other than 4 and 3 is untested beyond the plan asserts.
