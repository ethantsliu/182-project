"""Data loading: FineWeb .bin shards -> BOS-aligned varlen batches, plus the n-gram table row ids."""
import glob
import os
import queue
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import numpy as np

import torch
import torch.distributed as dist
from torch import Tensor

from track_1_short.config import VIRTUAL_SEQ_CAP
from track_1_short import folds
from track_1_short.folds import HOST_BATCH_PLAN, HOST_NGRAM_HASH
from track_1_short.model.layers import next_multiple_of_n
from track_1_short.ngram_table import ngram_row_ids, ngram_row_ids_fused
from track_1_short.perf.pinned_batches import PinnedBatchStaging
from track_1_short.schedule import TrainingSchedule

HEADER_BYTES = 256 * 4
# Parallel pread threads for a shard read. The first training shard and every validation shard are read
# on the clock; one thread's readinto leaves the page cache / disk bandwidth underused (record #360).
SHARD_READ_THREADS = 4


def pread_parallel(fd: int, out: np.ndarray, offset: int, nbytes: int) -> int:
    """Read nbytes at `offset` of fd into `out` (a contiguous array of exactly nbytes), split
    SHARD_READ_THREADS ways over disjoint byte ranges. Returns the bytes read (short on EOF)."""
    view = memoryview(out).cast("B")
    assert len(view) == nbytes, f"buffer holds {len(view)} bytes, reading {nbytes}"

    def read_range(lo: int, hi: int) -> int:
        start = lo
        while lo < hi:
            n = os.preadv(fd, [view[lo:hi]], offset + lo)
            if n == 0:
                break  # EOF
            lo += n
        return lo - start

    edges = [(nbytes * i) // SHARD_READ_THREADS for i in range(SHARD_READ_THREADS + 1)]
    with ThreadPoolExecutor(SHARD_READ_THREADS) as pool:
        return sum(pool.map(read_range, edges, edges[1:]))


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        # straight into the array: avoids a bytes->array copy (@YouJiacheng)
        nbytes = pread_parallel(f.fileno(), tokens.numpy(), HEADER_BYTES, 2 * num_tokens)
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256


class Batch(NamedTuple):
    inputs: Tensor         # [T] int32, device
    targets: Tensor        # [T] int64, device
    cum_seqlens: Tensor    # [max_num_docs] int32, device
    ngram_ids: Tensor      # [2T] int32 n-gram table rows (ngram_row_ids), device: the forward's cache lookup
    ngram_ids_cpu: np.ndarray  # the same, host: the row-pull want lists are built from it
    targets_cpu: Tensor    # host copy: the sampled-softmax candidate build reads targets without a D2H
    inputs_cpu: np.ndarray  # host copy: the value_embeds row-pull want lists are built from it

# Rows of the packed cu_seqlens table per local batch size (tokens per rank); 40960 is the batch-20 taper.
TRAIN_MAX_NUM_DOCS = {16384: 64, 32768: 96, 40960: 128, 49152: 128}


def cu_seqlens_rows(tokens_per_rank: int) -> int:
    """Length of the packed cu_seqlens table for a batch of this many tokens per rank (a fixed shape,
    so the compiled model and its CUDA graphs see one shape per batch size)."""
    return TRAIN_MAX_NUM_DOCS.get(tokens_per_rank, next_multiple_of_n(tokens_per_rank // 300, n=128))

class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0
        # HOST_PREP=batchplan (#387): the cumulative capped document lengths, cached per (BOS index, cap).
        self._plan_key = None
        self._plan_prefix = None

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        if HOST_BATCH_PLAN:
            return self._next_batch_planned(num_tokens_local, max_seq_len)
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                idx += 1
                end = min(self.bos_idx[idx] if idx < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur

            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    def _next_batch_planned(self, num_tokens_local: int, max_seq_len: int):
        """next_batch's starts and ends without the per-document walk (PR #387, nikhil-lalgudi).

        A document contributes min(its length, max_seq_len) tokens until the cap or the BOS index changes,
        so one cumulative sum of those lengths serves every rank: rank r's documents run up to the first
        prefix >= its goal (num_tokens_local + 1 tokens), and only the last one is cut to the exact budget.
        The same starts and ends as the walk (182/t1-gpu-stack/test_folds.py checks it), as int64 numpy arrays.
        """
        bos = self.bos_idx
        if self._plan_key is None or self._plan_key[0] is not bos or self._plan_key[1] != max_seq_len:
            lengths = np.minimum(np.diff(bos, append=self.size), max_seq_len)
            prefix = np.empty(len(bos) + 1, dtype=np.int64)
            prefix[0] = 0
            np.cumsum(lengths, out=prefix[1:])
            self._plan_key, self._plan_prefix = (bos, max_seq_len), prefix
        prefix = self._plan_prefix
        starts, ends = [], []
        idx = self.i
        for _ in range(self.world_size):
            goal = prefix[idx] + num_tokens_local + 1
            stop = int(np.searchsorted(prefix, goal, side="left"))
            if stop > len(bos):
                raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
            start = bos[idx:stop].copy()
            end = start + np.diff(prefix[idx:stop + 1])
            end[-1] -= prefix[stop] - goal  # truncate only the final document to the exact budget
            starts.append(start)
            ends.append(end)
            idx = stop
        # A failure on any rank consumes no documents, as in the walk.
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

def split_attention_segments(cum_lengths: Tensor, cap: int) -> Tensor:
    """Document ends (cumulative) -> attention-segment ends, with a boundary every `cap` tokens inside
    each document. Only attention sees the extra boundaries; the tokens are unchanged."""
    ends = [end for start, stop in pairwise([0, *cum_lengths.tolist()])
            for end in (*range(start + cap, stop, cap), stop)]
    return torch.tensor(ends, dtype=cum_lengths.dtype)

class HostBatch(NamedTuple):
    """A batch's host tensors, everything but the H2D copies (HOST_ASYNC=1 builds these on its producer thread)."""
    inputs: Tensor       # [T] int32
    targets: Tensor      # [T] int64
    cum_seqlens: Tensor  # [max_num_docs] int32
    ngram_ids: Tensor    # [2T] int32


def _prepare_batch(_inputs: Tensor, _targets: Tensor, cum_lengths: Tensor, num_tokens_local: int,
                   max_num_docs: int) -> HostBatch:
    """Host tokens and document ends -> the padded cu_seqlens table and the n-gram rows. Host only, no CUDA call."""
    _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
    _cum_lengths[0] = 0
    _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

    # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
    _inputs = _inputs.to(dtype=torch.int32)
    _targets = _targets.to(dtype=torch.int64)
    _cum_lengths = _cum_lengths.to(dtype=torch.int32)
    _ngram_ids = ngram_row_ids_fused(_inputs) if HOST_NGRAM_HASH else ngram_row_ids(_inputs)
    return HostBatch(_inputs, _targets, _cum_lengths, _ngram_ids)


def _upload_batch(host: HostBatch, staging: PinnedBatchStaging) -> Batch:
    """A HostBatch -> a Batch: the pinned-slot copies and their H2D launches (the caller's thread and stream)."""
    inputs, targets, cum_seqlens, ngram_ids = staging.upload(host.inputs, host.targets, host.cum_seqlens, host.ngram_ids)

    return Batch(
        inputs=inputs,
        targets=targets,
        cum_seqlens=cum_seqlens,
        ngram_ids=ngram_ids,
        ngram_ids_cpu=host.ngram_ids.numpy(),
        targets_cpu=host.targets,
        inputs_cpu=host.inputs.numpy(),
    )


def _package_batch(_inputs: Tensor, _targets: Tensor, cum_lengths: Tensor, num_tokens_local: int, max_num_docs: int,
                   staging: PinnedBatchStaging) -> Batch:
    """Host tokens and document ends -> a Batch: the padded cu_seqlens table, the n-gram rows, the H2D copies."""
    return _upload_batch(_prepare_batch(_inputs, _targets, cum_lengths, num_tokens_local, max_num_docs), staging)


def eval_batch_from_tokens(inputs: Tensor, targets: Tensor, staging: PinnedBatchStaging) -> Batch:
    """A validation-shaped batch (the unaligned loader's: documents split only at BOS, no length cap, the
    same cu_seqlens table) from [T] host inputs and their targets. The copy mixture's fit batches
    (copy_mixture.py) use it, so they run the validation's compiled graph."""
    assert inputs.ndim == 1 and inputs.shape == targets.shape
    num_tokens_local = inputs.numel()
    max_num_docs = cu_seqlens_rows(num_tokens_local)
    cum_lengths = torch.nonzero(inputs == BOS_ID)[:, 0]
    assert len(cum_lengths) < max_num_docs, f"{len(cum_lengths)} documents overflow the {max_num_docs}-row cu_seqlens table"
    return _package_batch(inputs, targets, cum_lengths, num_tokens_local, max_num_docs, staging)


def read_shard_head(file: str | Path, num_tokens: int) -> Tensor:
    """The first `num_tokens` tokens of a .bin shard (uint16), without reading the rest of it."""
    file = Path(file)
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    assert int(header[2]) >= num_tokens, f"{file} holds {int(header[2])} tokens, {num_tokens} wanted"
    tokens = torch.empty(num_tokens, dtype=torch.uint16)
    with file.open("rb", buffering=0) as f:
        assert pread_parallel(f.fileno(), tokens.numpy(), HEADER_BYTES, 2 * num_tokens) == 2 * num_tokens
    return tokens


def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, staging: PinnedBatchStaging,
                               align_to_bos: bool = True):
    """Batches of this rank, with new (num_tokens, max_seq_len) taken via .send(). align_to_bos: each sequence begins
    with Beginning of Sequence token, sequences truncated to max_seq_len. staging: the pinned slots every batch's H2D
    copies go through (perf/pinned_batches.py). With HOST_ASYNC=1 the aligned (training) loader is an
    AsyncBatchLoader, which ScheduledBatches drives; validation keeps the plain generator."""
    host_batches = host_batch_generator(filename_pattern, num_tokens, max_seq_len, align_to_bos)
    if folds.HOST_ASYNC and align_to_bos:
        return AsyncBatchLoader(host_batches, staging, depth=folds.HOST_ASYNC_DEPTH)
    return _uploaded(host_batches, staging)


def _uploaded(host_batches, staging: PinnedBatchStaging):
    """The synchronous loader: each batch built and uploaded when it is asked for, on the caller's thread."""
    new_params = None
    while True:
        new_params = yield _upload_batch(host_batches.send(new_params), staging)


def host_batch_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, align_to_bos: bool = True):
    """distributed_data_generator's HostBatches (no CUDA call), with new (num_tokens, max_seq_len) via .send()."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert num_tokens % world_size == 0, "Batch size must be divisible by world size"

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        shard = Shard(tokens, world_size)
        next_shard_getter = Shard.load_async(next(file_iter), world_size)
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = cu_seqlens_rows(num_tokens_local)

        if align_to_bos:
            try:
                seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                shard = next_shard_getter()
                tokens = shard.tokens
                try:
                    next_shard_getter = Shard.load_async(next(file_iter), world_size)
                except StopIteration:
                    next_shard_getter = None  # no more shards to preload
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
            if max_seq_len > VIRTUAL_SEQ_CAP:
                cum_lengths = split_attention_segments(cum_lengths, VIRTUAL_SEQ_CAP)
                assert len(cum_lengths) < max_num_docs, f"{len(cum_lengths)} attention segments overflow the {max_num_docs}-row cu_seqlens table"

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        new_params = yield _prepare_batch(_inputs, _targets, cum_lengths, num_tokens_local, max_num_docs)

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len) via .send()
            num_tokens, max_seq_len = new_params
            assert num_tokens % world_size == 0, "Num tokens must be divisible by world size"


class _ProducerFailed(NamedTuple):
    error: BaseException


class AsyncBatchLoader:
    """HOST_ASYNC=1: the training loader with its host work on a producer thread (182/hostcpu/README.md).

    What moves: host_batch_generator's whole body -- the document plan (Shard.next_batch), the token gather, the
    cu_seqlens table and the n-gram row hashes -- runs on one daemon thread, up to `depth` batches ahead of the
    consumer. What stays: send() copies each finished batch into the next pinned slot and launches its H2D on the
    calling (main) thread, exactly as the synchronous loader does, so no CUDA call, stream or event moves to
    another thread and the pinned ring's order is unchanged.

    Exactness: the producer runs the same generator with the same sequence of .send() arguments, which plan()
    fixes up front (ScheduledBatches knows every step's geometry), so it yields the same HostBatches in the same
    order; send() asserts each argument against the plan. The generator's state (the shard, its cursor, the shard
    files) is touched by the producer only.

    Lifetime: the thread starts on the first send() -- not at construction -- so the first shard is still read on
    the clock, as in the synchronous loader. It ends by itself after the planned number of batches; close()
    stops it early (it drains the queue so a blocked put returns). A producer exception is re-raised by the next
    send().
    """

    def __init__(self, host_batches, staging: PinnedBatchStaging, depth: int):
        assert depth >= 1
        self.host_batches = host_batches
        self.staging = staging
        self.depth = depth
        self.plan_params: list | None = None
        self.taken = 0
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def plan(self, send_params: Sequence) -> None:
        """The .send() argument of every batch the consumer will take, in order (None = geometry unchanged)."""
        assert self.plan_params is None and self.thread is None, "plan() once, before the first send()"
        self.plan_params = list(send_params)

    def _produce(self):
        try:
            for params in self.plan_params:
                if self.stop.is_set():
                    return
                self.queue.put(self.host_batches.send(params))
        except BaseException as error:  # noqa: BLE001 -- handed to the consumer
            self.queue.put(_ProducerFailed(error))

    def send(self, new_params) -> Batch:
        assert self.plan_params is not None, "AsyncBatchLoader needs plan() (ScheduledBatches calls it)"
        assert self.taken < len(self.plan_params), f"only {len(self.plan_params)} batches were planned"
        assert new_params == self.plan_params[self.taken], \
            f"batch {self.taken}: send({new_params}) but the plan says {self.plan_params[self.taken]}"
        if self.thread is None:
            self.thread = threading.Thread(target=self._produce, name="host-async-loader", daemon=True)
            self.thread.start()
        item = self.queue.get()
        if isinstance(item, _ProducerFailed):
            self.close()
            raise item.error
        self.taken += 1
        return _upload_batch(item, self.staging)

    def __next__(self) -> Batch:
        return self.send(None)

    def __iter__(self):
        return self

    def close(self) -> None:
        """Stop the producer and wait for it (it is done already if every planned batch was taken)."""
        self.stop.set()
        while self.thread is not None and self.thread.is_alive():
            try:
                while True:
                    self.queue.get_nowait()
            except queue.Empty:
                pass
            self.thread.join(timeout=0.05)


def scheduled_send_params(schedule: TrainingSchedule, steps: Sequence[int]) -> list:
    """The loader .send() argument of each step in `steps` (fetched in order): the step's (batch_size,
    train_max_seq_len) when it differs from the previous fetch's, else None. The loader starts at stage 0's."""
    stage, _ = schedule.lookup(0)
    current = (stage.batch_size, stage.train_max_seq_len)  # what the loader was created with
    params = []
    for step in steps:
        stage, _ = schedule.lookup(step)
        geometry = (stage.batch_size, stage.train_max_seq_len)
        params.append(geometry if geometry != current else None)
        current = geometry
    return params


class ScheduledBatches:
    """Training batches by step, over the sequence of steps a run trains (one microbatch per step).

    The batch geometry (tokens, max sequence length) of each step comes from the schedule and is sent to
    the loader when it changes. Batches are fetched in step order and held until taken, so peek() can
    look ahead: the sampled-softmax candidate build peeks one step ahead, the row prefetch up to
    MAX_CYCLE_STEPS + PREP_SLACK_STEPS (perf/row_prefetch.py). `steps` is range(total_steps) for the
    timed run and the sampled, non-consecutive steps for warmup.
    """
    def __init__(self, loader, schedule: TrainingSchedule, steps: Sequence[int]):
        self.loader = loader
        self.schedule = schedule
        self.steps = list(steps)
        self.send_params = scheduled_send_params(schedule, self.steps)
        if isinstance(loader, AsyncBatchLoader):  # HOST_ASYNC=1: the producer replays this exact sequence
            loader.plan(self.send_params)
        self.next_index = 0  # position in `steps` of the next batch to fetch
        self.ahead: dict[int, Batch] = {}  # step -> batch, fetched and not yet taken, in step order

    def _fetch_next(self):
        step = self.steps[self.next_index]
        self.ahead[step] = self.loader.send(self.send_params[self.next_index])
        self.next_index += 1

    def peek(self, step: int) -> Batch:
        while step not in self.ahead:
            assert self.next_index < len(self.steps) and self.steps[self.next_index] <= step, f"step {step} is not ahead"
            self._fetch_next()
        return self.ahead[step]

    def take(self, step: int) -> Batch:
        batch = self.peek(step)
        assert next(iter(self.ahead)) == step, f"step {step} taken out of order"
        del self.ahead[step]
        return batch
