"""CPU checks of HOST_ASYNC=1 (data.AsyncBatchLoader): the batch sequence equals the synchronous loader's, bit for bit.

    python 182/hostcpu/test_host_async.py      # from the variant directory (or anywhere); also runs under pytest

What it covers, on synthetic .bin shards written to a temp dir (small, so a run crosses several shard boundaries):
  - the timed run's step sequence (range(total_steps)) over every stage boundary of a shortened schedule, and a
    warmup-like non-consecutive step list, for several data seeds and ranks, with HOST_PREP unset and =batchplan:
    every field of every Batch (device-side tensors as the fake staging returns them, and the host copies) is
    identical to the synchronous loader's, in the same order;
  - the uploads (the only CUDA work per batch) happen on the consumer's thread, in fetch order, one per batch;
  - the producer never runs more than HOST_ASYNC_DEPTH batches ahead of the consumer and stops by itself after
    the planned batches; close() mid-run stops it; a producer error (no shard files) reaches the consumer.
What it does not cover: CUDA (pinned memory, H2D, events), timing, or the real 8-rank run.
"""
import os
import sys
import tempfile
import threading
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, VARIANT)


def _stub_triton():
    """Enough of triton for the kernel modules to import on a machine without it (as test_folds.py does)."""
    class _Any(types.ModuleType):
        def __getattr__(self, name):
            return _Any(name)

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

        def __getitem__(self, item):
            return self

    triton = _Any("triton")
    sys.modules.setdefault("triton", triton)
    sys.modules.setdefault("triton.language", triton.language)


try:
    import triton  # noqa: F401
except ImportError:
    _stub_triton()
try:
    import kernels  # noqa: F401
except ImportError:
    _kernels = types.ModuleType("kernels")
    _kernels.get_kernel = lambda *args, **kwargs: sys.modules["triton"]
    sys.modules["kernels"] = _kernels

from track_1_short import data, folds  # noqa: E402
from track_1_short.config import TRAINING_STAGES  # noqa: E402
from track_1_short.schedule import TrainingSchedule  # noqa: E402

WORLD = 8
SHARD_TOKENS = 5_000_000   # < 6M: the partial BOS index is already the whole shard, so shard switches are deterministic


class FakeStaging:
    """PinnedBatchStaging's interface on the CPU: 'device' copies are clones; records the uploading thread."""

    def __init__(self):
        self.threads = []

    def upload(self, *host):
        self.threads.append(threading.get_ident())
        return tuple(t.clone() for t in host)


def write_shards(directory: str, seed: int, count: int) -> str:
    rng = np.random.default_rng(seed)
    for k in range(count):
        # Geometric document lengths around FineWeb's ~690 tokens, plus some long ones (cut by the seq caps).
        toks = rng.integers(0, 50257, SHARD_TOKENS).astype(np.uint16)
        toks[rng.random(SHARD_TOKENS) < 1.0 / 690] = data.BOS_ID
        long_docs = rng.integers(0, SHARD_TOKENS - 9000, 40)
        for start in long_docs:
            toks[start + 1:start + 8000][toks[start + 1:start + 8000] == data.BOS_ID] = 1
        toks[0] = data.BOS_ID
        header = np.zeros(256, dtype=np.int32)
        header[:3] = (20240520, 1, SHARD_TOKENS)
        with open(os.path.join(directory, f"fineweb_train_{k:06d}.bin"), "wb") as f:
            f.write(header.tobytes())
            f.write(toks.tobytes())
    return os.path.join(directory, "fineweb_train_*.bin")


def schedule(scheduled: int = 30) -> TrainingSchedule:
    """The record's five stages over a short run (scheduled + 52 growth + 20 extension steps): every boundary."""
    return TrainingSchedule(TRAINING_STAGES, scheduled, 20, torch.device("cpu"), cooldown_frac=0.8,
                            split_embed_stage=4, ws_post_yarn_ext=20)


def _load_unpinned(file):
    """data._load_data_shard without pin_memory=True, which on macOS torch >= 2.11 yields an MPS tensor."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520 and header[1] == 1
    num_tokens = int(header[2])
    tokens = torch.empty(num_tokens, dtype=torch.uint16)
    with open(file, "rb", buffering=0) as f:
        assert data.pread_parallel(f.fileno(), tokens.numpy(), data.HEADER_BYTES, 2 * num_tokens) == 2 * num_tokens
    return tokens


if torch.empty(1, pin_memory=True).device.type != "cpu":
    data._load_data_shard = _load_unpinned


def shards_needed(sched: TrainingSchedule) -> int:
    """Enough shards for every step's global batch, with room for the documents the length caps cut short."""
    tokens = sum(sched.lookup(step)[0].batch_size for step in range(sched.total_steps))
    return int(1.5 * tokens / SHARD_TOKENS) + 2


def fake_dist(rank: int):
    data.dist = types.SimpleNamespace(get_rank=lambda: rank, get_world_size=lambda: WORLD)


def run(pattern: str, sched: TrainingSchedule, steps, host_async: bool, depth: int = 4):
    """Every batch of `steps` through ScheduledBatches with the run's access pattern (peek ahead, take in order)."""
    folds.HOST_ASYNC, folds.HOST_ASYNC_DEPTH = host_async, depth
    staging = FakeStaging()
    stage0 = TRAINING_STAGES[0]
    loader = data.distributed_data_generator(pattern, stage0.batch_size, stage0.train_max_seq_len, staging)
    assert isinstance(loader, data.AsyncBatchLoader) == host_async
    batches = data.ScheduledBatches(loader, sched, steps)
    steps = list(steps)
    out = []
    for i, step in enumerate(steps):
        for ahead in steps[i:i + 8]:   # the row prefetch peeks up to MAX_CYCLE_STEPS + PREP_SLACK_STEPS ahead
            batches.peek(ahead)
        out.append(batches.take(step))
    if host_async:
        loader.thread.join(timeout=10)
        assert not loader.thread.is_alive(), "the producer did not stop after the planned batches"
        loader.close()
    return out, staging


def assert_same(a: list, b: list, what: str):
    assert len(a) == len(b), what
    for k, (x, y) in enumerate(zip(a, b)):
        for name in data.Batch._fields:
            u, v = getattr(x, name), getattr(y, name)
            u, v = (torch.from_numpy(u) if isinstance(u, np.ndarray) else u), (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
            assert u.dtype == v.dtype and u.shape == v.shape and torch.equal(u, v), f"{what}: batch {k} field {name}"


def check_bit_identical():
    sched = schedule()
    total = sched.total_steps
    boundaries = [start for start, _ in sched.boundaries[1:]]
    warmup_like = sorted({0, 1, 2, 5, *[b + d for b in boundaries for d in (-1, 0, 1)], total - 1})
    tokens = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed, rank in ((0, 0), (1, 3), (2, 7)):
            directory = os.path.join(tmp, f"seed{seed}")
            os.makedirs(directory)
            pattern = write_shards(directory, seed, count=shards_needed(sched))
            fake_dist(rank)
            for plan in (False, True):
                data.HOST_BATCH_PLAN = plan
                for steps in (range(total), warmup_like):
                    ref, ref_staging = run(pattern, sched, steps, host_async=False)
                    got, staging = run(pattern, sched, steps, host_async=True)
                    assert_same(ref, got, f"seed {seed} rank {rank} batchplan={plan} steps={len(steps)}")
                    me = threading.get_ident()
                    assert staging.threads == [me] * len(steps), "an upload ran off the consumer thread"
                    tokens += sum(b.inputs.numel() for b in got)
            for name in os.listdir(directory):
                os.remove(os.path.join(directory, name))
    data.HOST_BATCH_PLAN = False
    print(f"ok  HOST_ASYNC=1 == synchronous: every Batch field, {total}-step runs over all {len(boundaries)} stage "
          f"boundaries and a {len(warmup_like)}-step warmup-like list, 3 seeds/ranks x batchplan on/off "
          f"({tokens / 1e6:.1f}M tokens per rank, shards of {SHARD_TOKENS / 1e6:.0f}M: several shard switches)")


def check_depth_shutdown_errors():
    sched = schedule()
    with tempfile.TemporaryDirectory() as tmp:
        pattern = write_shards(tmp, 5, count=shards_needed(sched))
        fake_dist(2)
        folds.HOST_ASYNC, folds.HOST_ASYNC_DEPTH = True, 3
        stage0 = TRAINING_STAGES[0]
        loader = data.distributed_data_generator(pattern, stage0.batch_size, stage0.train_max_seq_len, FakeStaging())
        assert loader.thread is None
        batches = data.ScheduledBatches(loader, sched, range(sched.total_steps))
        assert loader.thread is None, "the producer must not start before the first fetch (first shard on the clock)"
        produced = []
        original = loader.host_batches
        def counting():  # wraps the generator to count what the producer built
            item = original.send(None)
            while True:
                produced.append(1)
                item = original.send((yield item))
        loader.host_batches = counting()
        batches.take(0)
        for _ in range(200):  # bounded: until the producer has filled the queue and blocks on one more
            if len(produced) >= 5:
                break
            loader.thread.join(timeout=0.01)
        loader.thread.join(timeout=0.2)
        # taken 1, queued 3 (the depth), 1 built and blocked in put: never more
        assert loader.thread.is_alive() and len(produced) == 1 + 3 + 1, f"producer built {len(produced)} at depth 3"
        batches.take(1)
        loader.close()
        assert not loader.thread.is_alive(), "close() did not stop the producer"
        # A producer error reaches the consumer.
        empty = os.path.join(tmp, "none_*.bin")
        bad = data.distributed_data_generator(empty, stage0.batch_size, stage0.train_max_seq_len, FakeStaging())
        data.ScheduledBatches(bad, sched, range(4))
        try:
            next(bad)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("the producer's FileNotFoundError was swallowed")
        assert not bad.thread.is_alive()
        # A send() off the plan is refused.
        off = data.distributed_data_generator(pattern, stage0.batch_size, stage0.train_max_seq_len, FakeStaging())
        off.plan([None, None])
        try:
            off.send((1, 2))
        except AssertionError:
            pass
        else:
            raise AssertionError("send() off the plan was accepted")
        # Validation (align_to_bos=False) never goes async.
        assert not isinstance(data.distributed_data_generator(pattern, 8 * 4096, -1, FakeStaging(), align_to_bos=False),
                              data.AsyncBatchLoader)
    folds.HOST_ASYNC, folds.HOST_ASYNC_DEPTH = False, 4
    print("ok  HOST_ASYNC=1 lifetime: lazy start, stays depth ahead, close() stops it, errors reach the consumer, "
          "plan enforced, validation stays synchronous")


def test_bit_identical():
    check_bit_identical()


def test_depth_shutdown_errors():
    check_depth_shutdown_errors()


if __name__ == "__main__":
    import time
    for check in (check_bit_identical, check_depth_shutdown_errors):
        t0 = time.perf_counter()
        check()
        print(f"    ({time.perf_counter() - t0:.1f} s)", flush=True)
    print("2 HOST_ASYNC checks passed")
