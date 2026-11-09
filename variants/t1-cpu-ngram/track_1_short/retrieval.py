"""Exact-match retrieval: each position sees the next tokens that followed its context before.

Training rows come from a StreamIndex, strictly causal in the token stream: at the start of training it plans every
step's documents and resolves each step's rows, from the documents of earlier steps, in phases ahead of training.
Validation rows come from a SlotIndex over all training shards, built on the clock by rank 0 while training runs and
looked up by every rank after training, when the validation tokens are read.

A row is a cell (the match length bucket, and bins of the matches' total count and of the top token's share of it)
and the top two next tokens with their counts (token | count << 16, 0 if none); all zero where nothing matched.
"""
import glob
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import torch
import torch.distributed as dist
from exact_match import CELLS as RET_CELLS, SlotIndex, StreamIndex
from track_1_short import host

per_gpu = len(os.sched_getaffinity(0)) // int(os.environ.get("LOCAL_WORLD_SIZE", "1"))  # vCPUs per GPU, before any pinning
# rank 0's validation index build, on cores every rank sets aside (VAL_BUILD_THREADS overrides the choice by host size)
VAL_BUILD_THREADS = int(os.environ.get("VAL_BUILD_THREADS", 0)) or (64 if per_gpu >= 24 else 32 if per_gpu >= 16 else 16)
LARGE_HOST = per_gpu >= 24  # >= 24 vCPUs per GPU
# STREAM_THREADS (default 4): each rank's online (training-row) index threads; its rows do not depend on the count
STREAM_THREADS = int(os.environ.get("STREAM_THREADS", "4"))
# SLOT_SHARED_ARENA (default 0): the 3- and 2-token indexes, built before the 6-token one, reuse its scan arena (one
# arena for the three builds instead of three: ~67 GB less host memory on fineweb10B, the same tables)
SLOT_SHARED_ARENA = os.environ.get("SLOT_SHARED_ARENA", "0") == "1"
# TABLE_VOTE (default 0; PR #399's table vote, romeerpillay): the 6-token index's candidates at match lengths 6, 10
# and 20 and those followed by the target, counted on the clock beside the validation lookups. "chain": three more
# links of the chain (q = hits / candidates, lam by (length, log2 candidates)), fit with the others on the fit
# steps; "fixed": #399's vote on the chain's probability, P = (hits + k P) / (candidates + k), k = VOTE_K per length.
TABLE_VOTE = os.environ.get("TABLE_VOTE", "0")
assert TABLE_VOTE in ("0", "chain", "fixed"), f"TABLE_VOTE={TABLE_VOTE}: 0, chain or fixed"
# #399's pseudo-counts, fit there on the network alone (no chain): to be refit for this stack
VOTE_K = tuple(float(k) for k in os.environ.get("VOTE_K", "706.6,43.39,2.694").split(","))
VOTE_BINS = 25  # the chain links' groups per length: log2 of the candidate count
# VAL_TOPK: the candidates the model reads per row at validation (default 5; training reads 2)
VAL_WIDTH = 1 + int(os.environ.get("VAL_TOPK", "5"))

@dataclass(frozen=True)
class CpuPlan:
    main: set[int]          # the main thread's core, its own from step 1
    rest: set[int]          # every other thread of this rank from step 1 (NCCL, gloo, loaders, the online index)
    val_build: list[int]    # this rank's share of the cores rank 0's validation index build runs on
    val_lookup: list[int]   # this rank's validation lookups, after training


def pin_cpus(local_rank: int, world_size: int) -> CpuPlan:
    """Each rank gets an equal share of the physical cores local to its GPU (within the allowed set: Slurm hands out
    arbitrary CPU ids), and is pinned to it before the process group starts, so every thread inherits it. Its last
    cores run rank 0's validation index build, one thread per core; after training both hyperthreads of every core but
    the first run this rank's validation lookups, those of its build cores too (the builds are over by then). From step
    1 the main thread has the first core to itself."""
    cpulist = lambda path: {c for r in open(path).read().split(",") for a, _, b in [r.partition("-")] for c in range(int(a), int(b or a) + 1)}
    allowed = os.sched_getaffinity(0)
    cores = lambda cpus: sorted({frozenset(cpulist(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")) & allowed
                                 for c in cpus}, key=min)
    local = [frozenset(cpulist("/sys/bus/pci/devices/{0.pci_domain_id:04x}:{0.pci_bus_id:02x}:{0.pci_device_id:02x}.0/local_cpulist"
                               .format(torch.cuda.get_device_properties(i)))) & allowed for i in range(torch.cuda.device_count())]
    if min(len(cores(cpus)) / local.count(cpus) for cpus in local) < 2:  # too few allowed cores near some GPU: ignore locality
        local = [frozenset(allowed)] * len(local)
    near = cores(local[local_rank])
    k, n = local[:local_rank].count(local[local_rank]), local.count(local[local_rank])
    mine = near[k * len(near) // n:(k + 1) * len(near) // n]
    val_lookup = sorted(set().union(*mine[1:]))
    spare = -(-VAL_BUILD_THREADS // world_size)
    build = mine[-spare:] if len(mine) - spare >= 2 else []
    val_build, mine = [min(c) for c in build], mine[:len(mine) - len(build)]
    val_lookup = (sorted(set().union(*mine[1:])) or sorted(mine[0])) + sorted(set().union(*build))
    os.sched_setaffinity(0, set().union(*mine))
    return CpuPlan(main=set(mine[0]), rest=set().union(*mine[1:]) or set(mine[0]), val_build=val_build, val_lookup=val_lookup)


def pin_threads(plan: CpuPlan, child: int | None):
    """At step 1: the main thread to its own core, every other thread but the validation index's pools (pinned
    already) and the forked `child` process, if any, to the rest."""
    for tid in map(int, os.listdir("/proc/self/task")):
        try:
            if not open(f"/proc/self/task/{tid}/comm").read().startswith("slotindex-"):
                os.sched_setaffinity(tid, plan.main if tid == threading.get_native_id() else plan.rest)
        except (ProcessLookupError, FileNotFoundError):
            pass
    if child:
        try:
            os.sched_setaffinity(child, plan.rest)
        except ProcessLookupError:
            pass


class OnlineCache:
    """Training rows, resolved on the clock ahead of training into a pinned ring allocated before the clock (a
    cudaHostAlloc on the clock holds the driver lock). A step's job is submitted when the loader fetches its batch, at
    most `lookahead` steps before the step trains, so a ring deeper than that never rewrites a slot whose step is still
    to train; a slot is rewritten only after its previous H2D retired (one event per slot)."""
    def __init__(self, files, schedule, rank, world, device, lookahead):
        self.index = StreamIndex(sorted(glob.glob(files)), schedule, rank, world, threads=STREAM_THREADS)
        self.builder = ThreadPoolExecutor(max_workers=1)
        self.worker = ThreadPoolExecutor(max_workers=1, initializer=torch.cuda.set_device, initargs=(device,))
        self.ring = torch.zeros((lookahead + 2, max(n for n, _ in schedule), 3), dtype=torch.int32, pin_memory=True)
        self.uploaded = [torch.cuda.Event() for _ in range(len(self.ring))]
        self.device = device
        self.jobs = {}  # step -> its rows' future, from fetch until the step trains
        self.step0 = None
        self.hint_from = (round(0.5 * len(schedule)), round(0.85 * len(schedule)))  # 3/2-, 6-token

    def start(self):
        self.ready = self.builder.submit(self.index.build)

    def submit(self, step, batch):
        """ScheduledBatches' fetch hook: `batch.docs` is the loader's (shard tokens, every rank's document starts, ends)."""
        self.jobs[step] = self.worker.submit(self._rows, step, *batch.docs, batch.inputs_cpu)

    def _rows(self, step, tokens, starts, ends, inputs):
        x = inputs.astype(np.uint16)  # per position the longest match of the 6-, 3- and 2-token indexes
        # a finished build's result() re-raises its error: a failed table is never queried (the step's job fails)
        lower = [(source, cache) for source, cache in self.lower if step >= self.hint_from[0] and cache.built.done()
                 and cache.built.result() is None]
        if self.full is not None and step >= self.hint_from[1] and self.full.built.done() and self.full.built.result() is None:
            rows = np.zeros((len(x), 9), dtype=np.int32)
            self.full.index.query_into(x, len(x), rows, 0)
        else:
            rows = self.index.rows(step, len(tokens), starts, ends)
        for source, cache in lower:
            cache.index.query_into(x, len(x), rows, RET_CELLS * source)
        if self.mix is not None:
            self.mix.put(step, rows)
        i = step % len(self.ring)
        self.uploaded[i].synchronize()
        slot = self.ring[i, :len(rows)]
        slot.numpy()[:] = rows[:, :3]
        return slot

    def rows(self, step, n):
        """This step's rows on the device. Zero during warmup (no jobs) and at step 0: nothing was trained before it, so
        its rows are only checked to be empty, once step 1's are taken."""
        job = self.jobs.pop(step, None)
        if job is None or step == 0:
            self.step0 = job
            return torch.zeros((n, 3), dtype=torch.int32, device=self.device)
        if step == 1:
            assert not self.step0.result().any(), "step 0 has retrieval matches"
        rows = job.result().to(self.device, non_blocking=True)  # the job also checks the loader's documents against the plan
        self.uploaded[step % len(self.ring)].record()
        return rows

class ValidationCache:
    """Validation rows from one SlotIndex over all training shards, a table in /dev/shm shared by the ranks.

    Before the clock rank 0 creates the table and faults in its build memory, and the other ranks open it. On the
    clock, while training runs, rank 0 builds it on threads pinned to the `build_cpus` every rank sets aside. After
    training each rank reads and looks up just the `chunk`-token validation chunks it evaluates, on its `lookup_cpus`.
    """
    def __init__(self, train_files, val_files, chunk, total, group, build_cpus, lookup_cpus, exclude=0, key=6, shards=None, share=None):
        files = sorted(glob.glob(train_files))
        assert len(files) == 103, f"the validation index covers all 103 training shards, found {len(files)}"
        files = files[exclude:][:shards]
        self.val_file, self.chunk, self.group = sorted(glob.glob(val_files))[0], chunk, group
        self.rank, world = dist.get_rank(group), dist.get_world_size(group)
        self.offsets = range(self.rank * chunk, total, world * chunk)
        path = [f"/dev/shm/exact-match-{uuid.uuid4().hex}"]
        dist.broadcast_object_list(path, src=0, group=group)
        cpus = [None] * world
        dist.all_gather_object(cpus, sorted(build_cpus), group=group)
        if self.rank == 0:  # share (SLOT_SHARED_ARENA): an index built after this one, whose scan arena this one reuses
            self.index = SlotIndex.create(path[0], files, VAL_BUILD_THREADS, sum(cpus, [])[:VAL_BUILD_THREADS], lookup_cpus, key,
                                          share=share.index if share is not None and SLOT_SHARED_ARENA else None)
        dist.barrier(group)
        if self.rank != 0:
            self.index = SlotIndex.attach(path[0], lookup_cpus, key)
        dist.barrier(group)
        if self.rank == 0:
            os.unlink(path[0])
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.worker.submit(lambda: None).result()  # its thread starts before the clock

    def start(self, after):
        """During training, once `after` (a future) is done: rank 0 builds the table."""
        self.built = self.worker.submit(self._build, after)

    def _build(self, after):
        failed = None
        try:
            if after is not None:
                after.result()
            if self.rank == 0:
                t = time.perf_counter()
                self.index.build()
                print(f"validation index built in {time.perf_counter() - t:.2f} s", flush=True)
        except Exception as e:
            failed = e
        # the barrier, carrying whether any rank failed: every rank's build future then fails together (no rank waits
        # on a peer that has given up, and none queries a partial table)
        status = torch.tensor([failed is not None], dtype=torch.int32)
        dist.all_reduce(status, group=self.group)
        if failed is not None:
            raise failed
        if status.item():
            raise RuntimeError("the validation index build failed on another rank")
        if self.rank == 0:
            self.index.release()

    def lookup(self, lower=(), marks=False, then=None, fit=None):
        """After training: read this rank's validation chunks and look them up, beside the main thread.
        marks (SCAN_HINTS_SKIP; the mark of all the scan's contexts): then self.marks, every validation position's mark
        for the exact-count scan, and then(self.marks) on this thread (the scan's launch: the main thread does not wait
        for the marks). TABLE_VOTE: then self.vote, the table's counts at each validation chunk's positions, and (fit:
        the kept fit steps' host (inputs, targets)) self.fit_vote, at the fit positions."""
        def rows():
            # TABLE_VOTE: one token past each chunk, its last position's target
            read = [np.fromfile(self.val_file, "<u2", self.chunk + (TABLE_VOTE != "0"), offset=1024 + 2 * o) for o in self.offsets]
            tokens = np.concatenate([r[:self.chunk] for r in read])
            out = np.zeros((len(tokens), 9), dtype=np.int32)
            self.index.query_into(tokens, self.chunk, out, 0)
            empty = out[:, 0] == 0 if marks else None
            for source, cache in lower:  # then the 3- and 2-token indexes, into the rows still empty
                cache.built.result()
                cache.index.query_into(tokens, self.chunk, out, RET_CELLS * source)
            if marks:
                self.marks = self._marks(out[:, 0], empty, lower[0][0], marks)
                if then is not None:
                    then(self.marks)
            if TABLE_VOTE != "0":  # after the scan's launch, under it
                targets = np.concatenate([r[1:self.chunk + 1] for r in read])
                self.vote = np.split(self.index.query_counts(tokens, targets, self.chunk).reshape(-1, 6), len(self.offsets))
                if fit:
                    self.fit_vote = np.concatenate([self.index.query_counts(x, y, len(x)).reshape(-1, 6) for x, y in fit])
            return np.split(out, len(self.offsets))
        self.looked_up = self.worker.submit(rows)

    def _marks(self, cell, empty, tag3, every=3):
        """The contexts whose samples the exact-count scan is to take (exact_counts.py), per validation position: bit 1:
        its 3-token context, bit 0: its 2-token one. A position the 6-token index left `empty` marks its 3-token context,
        and its 2-token one if it has no 3-token row (tag3: those rows' tag) -- except the context of a row that already
        holds a large sample: a match at the key's own length (bucket 1) in the top count bin (at least 33 matches),
        which the position keeps. The longer contexts in `every` (bit 2: the 4-token one), of which the indexes have no
        rows, are marked at each `empty` position. This rank's positions here; every rank's, in the validation stream's order, returned."""
        is3, large = cell // RET_CELLS == tag3, cell % RET_CELLS // 6 == 13  # cell = (7 bucket + count bin) 6 + purity bin
        mine = torch.from_numpy((empty * ((every & ~3) + 2 * ~(is3 & large) + (~is3 & ~large))).astype(np.int8))
        parts = [torch.empty_like(mine) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(parts, mine, group=self.group)
        return torch.stack(parts).view(len(parts), -1, self.chunk).transpose(0, 1).reshape(-1)  # chunk c is rank c % world's

    def rows(self):
        """Wait for the build and the lookup: the rows of each of this rank's validation chunks."""
        self.built.result()
        return self.looked_up.result()


def table_part(counts, device):  # TABLE_VOTE=chain: the table's links, one per match length: q = hits / candidates
    counts = torch.as_tensor(counts.astype(np.float64), device=device)
    n, h = counts[:, :3], counts[:, 3:]
    bins = torch.clamp(torch.floor(torch.log2(n.clamp_min(1))), 0, VOTE_BINS - 1).long()
    return [(torch.where(n[:, k] > 0, h[:, k] / n[:, k].clamp_min(1), 0.0), torch.where(n[:, k] > 0, bins[:, k], -1), VOTE_BINS)
            for k in range(3)]


def table_vote(p, counts):  # TABLE_VOTE=fixed (PR #399): the table votes on the probability p, shortest length first
    counts = torch.as_tensor(counts.astype(np.float64), device=p.device)
    p = p.double()
    for k, kk in enumerate(VOTE_K):
        if kk >= 0:  # VOTE_K < 0: that length does not vote; no candidates: p unchanged
            n = counts[:, k]
            p = torch.where(n > 0, (counts[:, 3 + k] + kk * p) / (n + kk).clamp_min(1e-30), p)
    return p


def host_plan(train_files, exclude, world, print0, caches=()):
    """Before the retrieval allocates (local rank 0): this node's CPUs and RAM against the retrieval's estimated host
    memory, into the log; fails (HOST_CHECK=1, the default) if the tables or the total cannot fit. With `caches` (the
    ValidationCaches, rank 0, once created): the arenas and tables they actually hold."""
    files = sorted(glob.glob(train_files))[exclude:]
    tokens = [host.shard_tokens(f) for f in files]
    need = host.plan(sum(tokens), sum(tokens[:20]), VAL_BUILD_THREADS, world, shared=SLOT_SHARED_ARENA)
    if caches:
        arenas, tables, shared = zip(*(c.index.memory() for c in caches))
        print0(f"retrieval host memory: arenas {sum(arenas) / host.GiB:.1f} GiB (shared: {list(shared)}) tables "
               f"{sum(tables) / host.GiB:.1f} GiB", console=True)
        return need
    h = host.detect()
    print0(host.describe(h), console=True)
    print0(f"retrieval host plan: {VAL_BUILD_THREADS} build threads, {STREAM_THREADS} online-index threads per rank; "
           f"6-token index {len(files)} shards ({sum(tokens) / 1e9:.2f}B tokens); estimated arenas {need.arenas / host.GiB:.0f} GiB + "
           f"tables {need.tables / host.GiB:.0f} GiB (/dev/shm) + {world} ranks x {host.RANK_GIB:.0f} GiB = {need.total / host.GiB:.0f} GiB"
           f"{' (shared arena)' if SLOT_SHARED_ARENA else ''}", console=True)
    problems = host.check(h, need)
    for msg in problems:
        print0(f"HOST CHECK: {msg}", console=True)
    if problems and host.HOST_CHECK:
        raise RuntimeError("this node cannot hold the exact-match retrieval: " + "; ".join(problems) + " (HOST_CHECK=0 to try anyway)")
    return need


def host_usage(print0):  # after the run, off the clock: this rank's peak RSS and the node's memory in use
    peak, used, shm = host.usage()
    print0(f"host memory: rank 0 peak RSS {peak:.1f} GiB, node in use {used:.0f} GiB (incl. /dev/shm {shm:.0f} GiB)", console=True)


def mix_q(rows, targets):  # the target's share of the row's candidate counts
    tok, cnt = rows[:, 1:] & 0xFFFF, (rows[:, 1:] >> 16).float()
    return (cnt * (tok == targets[:, None]).float()).sum(1) / cnt.sum(1).clamp_min(1.0)


def rows_part(cell, q):  # the hint rows' link of the chain: q, weighted by the row's tagged cell
    return [(q, torch.where(cell % RET_CELLS > 0, cell, torch.full_like(cell, -1)), 4 * RET_CELLS)]


class FitRows:  # the kept fit steps' rows (pinned as they are looked up), then their cells and target shares
    def __init__(self, steps, max_tokens, device):
        self.steps, self.device, self.n, self.jobs = steps, device, 0, {}
        self.cell = torch.zeros(len(steps) * max_tokens, dtype=torch.int64, device=device)
        self.q = torch.zeros(len(steps) * max_tokens, device=device)
        self.pinned = torch.zeros((len(steps), max_tokens, 9), dtype=torch.int32, pin_memory=True)

    def put(self, step, rows):
        if step in self.steps:
            slot = self.pinned[self.steps.index(step), :len(rows)]
            slot.numpy()[:] = rows
            self.jobs[step] = slot

    def collect(self, step, targets):
        rows = self.jobs.pop(step).to(self.device, non_blocking=True)
        n = len(rows)
        self.cell[self.n:self.n + n] = rows[:, 0].long()
        self.q[self.n:self.n + n] = mix_q(rows, targets)
        self.n += n
