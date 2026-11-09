# t1-cpu-ngram: the CPU n-gram retrieval line (Track 1)

`variants/t1-cpu-ngram/` (a full runnable tree), based on upstream PR #381 (Daniel Monroe, "Use all shards for input hints", 725 -> 685 steps),
which is devenpzak's #380 exact-count chain (9.653 s mean over 16 runs in its logs), which builds on hermabr's #367
exact-match retrieval. #399 (romeerpillay, table vote) is ported here. **None of these are merged upstream, and the
CPU/RAM rules are still open** (caps were promised). Everything new is behind an env var. With every flag at its
default the run does what #381 does, plus a host report and a host-memory check before the clock.

**Where this line stands (as of 2026-11-08):** the official record is #92 at 39.9 s. #380 measured 9.653 s (n=16, 8xH100,
2x Xeon 8462Y+, 2 TB RAM). **#381 has no 8xH100 time.** Its only measured times are on 4x RTX PRO 6000: 33.62 s
(#380) -> 31.76 s (#381), -5.5%, n=3 per arm, at val loss 3.2734 -> 3.2725. Scaling that to #380's node gives roughly
9.1-9.2 s, an estimate, not a measurement. The fastest open Track-1 PR is #367's current head at 6.702 s (n=16). That
is a later #367 than the one #380 builds on. It trains and validates on FineWeb-100B paths (a different val stream),
so it is not adopted here, and its comparability to the 10B line is for the maintainers to rule on.

## The pipeline, end to end (as in #381)

- **Before the clock.** Each rank pins itself to its share of the physical cores local to its GPU (`pin_cpus`).
  Rank 0 creates three `SlotIndex` tables as files in `/dev/shm`, all over the training shards the loader never
  reads (`exclude` = 2 at ~700 steps):
  - 6-token key: 101 shards, about 10.1B tokens, a 96 GiB table;
  - 3-token and 2-token keys: 20 shards each, 24 GiB tables.

  It also faults in each table's private huge-page scan arena (MADV_HUGEPAGE, DONTFORK, alternate VMAs DONTDUMP,
  first-touch per pinned build thread). The other ranks mmap the tables. Each rank also builds a `StreamIndex` for
  its training rows (4 threads, the shard files mmapped as the corpus).
- **On the clock, during training.**
  - `StreamIndex.build` plans every step's documents, then indexes and resolves rows in phases (steps 8/32/128/256/end).
    Each phase is a rolling hash of the stream, a scatter into 4096 partitions, and radix sorts, strictly causal.
  - From step 25, rank 0 builds the 2-, 3- and then 6-token tables one after another on `VAL_BUILD_THREADS` pinned
    threads (64 if a GPU has >= 24 vCPUs, 32 if >= 16, else 16):
    1. scan: pread, AVX-512 hashing, and write-combined non-temporal scatter into 2048-entry blocks, with no count pass;
    2. finish: per partition, place entries into b1, else b2, else drop, then stream to the table and MADV_DONTNEED.
  - Then each rank loads its corpus share to its GPU for the exact-count scan.
- **On the clock, after training.**
  - Each rank looks up its validation chunks in place in the mmapped tables (prefetching the next bucket), and builds
    the marks for the GPU exact-count scan, which it then launches.
  - The GPU counts orders 2..8 exactly over the same shards (Triton), with SCAN_HINTS samples for the 2-, 3- and
    4-token rows.
  - The chain weights are fit (EM plus a hidden-state gate) on the fit steps. The clock stops.
  - The validation forward applies the chain, using the corpus orders, the hint rows and the in-document cache.

**Note on the records' box:** 2x Xeon 8462Y+ with 128 vCPU, i.e. 16 per GPU. So the published runs used 32 build
threads with `LARGE_HOST` off. The ">= 24 vCPUs per GPU" mode was never measured in those logs.

## What changed

| knob (default) | what it does | tested here |
| --- | --- | --- |
| *(always)* `track_1_short/host.py`, `host_plan` | Prints the host to the log: CPU model, CPUs/cores/NUMA, per-GPU vCPUs, RAM, `/dev/shm`, THP, AVX-512, kernel. Prints the retrieval's estimated host memory, then the arenas and tables actually allocated (`SlotIndex.memory`). At the end, prints rank 0's peak RSS and the node's memory in use. **Fails loudly before the clock** if the tables don't fit `/dev/shm` or the total doesn't fit MemAvailable. | py_compile; smoke test on CPU |
| `HOST_CHECK=1` | `0`: only report; never fail | — |
| `HOST_RANK_GIB=12` | Per-rank non-retrieval allowance in the estimate. Not measured, deliberately generous; can false-fail a tight box, so use `HOST_CHECK=0` there. | — |
| `VAL_BUILD_THREADS` (auto 64/32/16) | Overrides rank 0's build thread count | — |
| `STREAM_THREADS=4` | Online-index threads per rank. Rows are identical for any count (cargo test). | cargo test |
| `SLOT_SHARED_ARENA=0` | **The host-side improvement.** The 2- and 3-token builds reuse the 6-token index's already-faulted scan arena instead of owning one each. The builds are already sequential, and an arena lock now enforces that. The same tables are produced (cargo test: byte-identical single-threaded, including three builds started at once on one arena; identical kept-entry multisets multi-threaded at low load). At the real load, a hot key's bucket keeps the first ~96 of its entries in scan order, and the scan hands pieces to threads dynamically. So even two #381 builds differ in which entries they keep, and bit-identical tables are not expected, shared or not. Saves about **62 GiB** of host RAM on fineweb10B at 64 threads (396 -> 333 GiB estimated in total), about 46 GiB at the records' 32 threads (371 -> 325 GiB). It also skips two arena prefaults before the clock and two ~34 GB unmaps during training (each unmap takes the mmap lock the training threads also take). | cargo test (x86_64 under Rosetta, and native arm64) |
| `TABLE_VOTE=0` (`chain` / `fixed`) | #399's table vote, ported: `SlotIndex.query_counts` gives candidates and hits at match lengths 6/10/20 from the 6-token table. Counted **on the clock** in the lookup thread, after the exact-count scan's launch. (#399 itself read the table in the off-clock eval loop; this port is the conservative choice.) `chain`: the counts become three more chain links, q = hits/n with λ by (length, log2 n). They are fit on the fit steps with the other links; those positions are disjoint from the table's shards. This is the vote's form, (h + kP)/(n + k) = (1 − λ)P + λ·h/n, with λ fit rather than fixed. `fixed`: #399's vote on the chain's final probability, with `VOTE_K` (default #399's `706.6,43.39,2.694`). **Those constants were fit for #399's network without the chain, so they must be refit here.** #399 is stacked on #367's later head, not on #380: FineWeb-100B, a 36.1B-token table whose buckets hold ~4x the entries of this 48-slot table, and no exact-count chain. It ran on 1 H100 simulating 8 ranks, quality only (build time excluded), with the vote scored offline. Here the chain already holds exact order-6..8 counts over the same shards, so only the 10/20-token matches add new evidence. Expect much less than #399's -14 steps. | counts vs brute force (≥99% of positions exact); vote math vs #399's formula |
| *(build)* `exact_match/src/sys.rs` | Portability shims: x86 intrinsics and Linux madvise/affinity, with scalar/no-op fallbacks elsewhere. Unchanged on x86-64 Linux. `cargo test --release --no-default-features` runs 6 tests on small tables (`cfg(test)` constants). | both arches |
| *(always; failure paths only)* | Three hangs in #381 now raise instead. (1) The corpus loader thread (`exact_counts.load_corpus`): an exception, or a shard that shrank, used to leave `launch()` waiting forever, and a 0-byte `preadv` spun forever. (2) A failed index build: `built.done()` was true, so the online rows queried the partial table. (3) A rank-0 build error never reached the peers, who waited in the post-build barrier. That barrier is now an all-reduce of an error flag. A successful run does exactly what #381 does. The host check also clamps MemAvailable to a container's memory-cgroup headroom (v2/v1). | `182/t1-cpu-ngram/test_failure_paths.py` (14 tests; 7 fail on #381's code) |
| `182/t1-cpu-ngram/node_check.py` | Stdlib-only readiness check: FAIL/WARN/PASS lines, exit 1 on any FAIL. Hard floor: Linux, ≥ 64 CPUs, ≥ 16 vCPU/GPU, `/dev/shm` ≥ tables, RAM ≥ plan, 103 shards, and the extension or cargo. Recommended: ≥ 700 GiB, ≥ 24 vCPU/GPU, AVX-512, THP, kernel ≥ 5.14, data in `/dev/shm`. | ran here (correctly NOT READY) |

The estimated host memory needed on fineweb10B is about 396 GiB (333 GiB with the shared arena), so the requested
~700 GiB is headroom rather than need. The breakdown at 64 build threads:

| | GiB |
| --- | ---: |
| scan arenas | 155 (93 shared) |
| tables in `/dev/shm` | 144 |
| 8 ranks × 12 GiB allowance | 96 |
| data shards in `/dev/shm`, if there | +20 |

### Other host-side ideas (not implemented, not measured)
1. The online index re-hashes and re-sorts the whole stream every phase, about 1.6× the stream in total, on every
   rank: 8× duplicated work. Merging only each phase's new entries into the sorted partitions would remove most of
   that. A count-free block scatter (as in SlotIndex) would drop the separate `keys` count pass.
2. Fuse the 2- and 3-token scans: one pread/hash pass over their 20 shared shards. The scatter traffic dominates,
   so the expected gain is small.
3. Fewer `threads × partitions` partial blocks. That is 17 GB of slack per arena at 64 threads; only 17 GB remain
   with sharing on.
4. A GPU index build (#367's blog estimates ~3.1 s): **not attempted.** The CPU builds run concurrently with
   training, and nobody has shown whether they are on the critical path. That needs a timeline from a real run
   first (the `validation index built in X s` lines go to stdout, not the log).

## Run on 8xH100

```bash
pip install ./exact_match                       # Rust toolchain; or: cd exact_match && cargo test --release --no-default-features
python3 182/t1-cpu-ngram/node_check.py --data "$DATA_PATH"   # exit 0 = ready
clean() { pkill -f train_gpt.py; rm -f /dev/shm/exact-match-*; }
# A (= #381 behaviour) and B legs interleaved, TRAIN_SEED = leg index, data in /dev/shm
clean; TRAIN_SEED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
clean; TRAIN_SEED=0 SLOT_SHARED_ARENA=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
clean; TRAIN_SEED=0 TABLE_VOTE=chain torchrun --standalone --nproc_per_node=8 train_gpt.py
clean; TRAIN_SEED=0 TABLE_VOTE=fixed torchrun --standalone --nproc_per_node=8 train_gpt.py
clean; TRAIN_SEED=0 STREAM_THREADS=8 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
Compare `train_time` and `val_loss`. The `host:`, `retrieval host plan/memory:` and `host memory:` log lines are the
disclosure. If `TABLE_VOTE=chain` lowers val_loss, re-tune `NUM_SCHEDULED_ITERATIONS` down: #399 traded the vote for
14 steps on its own base (see above; the gain on this stack is unknown).

## Needs GPU verification (nothing here ran on a GPU)
- Every run-time claim. No speedup has been measured, and none is claimed.
- `SLOT_SHARED_ARENA=1` end to end: val_loss within #381's run-to-run spread (not bit-identical: see the build-order note above), unchanged `train_time`, and the RAM drop in the `host memory` line.
- `TABLE_VOTE`: its val_loss effect is unknown. The chain may already absorb the 6-gram evidence through its corpus
  orders. Also check the counts' on-clock cost on the lookup thread (1.3M val + ~0.2M fit positions per rank).
- The host check's thresholds on the real box, and that `HOST_RANK_GIB=12` is not too tight or too loose.
- The AVX-512 paths were compiled here (x86_64 under Rosetta) but not executed. Rosetta has no AVX-512, so the
  tests ran the AVX2/scalar paths, plus the native arm64 fallbacks.
