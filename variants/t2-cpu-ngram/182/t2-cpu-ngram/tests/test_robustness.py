"""CPU tests for the failure paths of the retrieval and exact-count code (no GPU, no Rust extension).

  cd variants/t2-cpu-ngram && python -m pytest -q 182/t2-cpu-ngram/tests
"""
import os
import socket
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
VARIANT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import cpu_stubs  # noqa: E402

cpu_stubs.install_all(VARIANT)
if not hasattr(os, "sched_getaffinity"):  # macOS: retrieval.py reads it at import
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count() or 1))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from track_1_short import exact_counts, host, retrieval  # noqa: E402


# ---- exact_counts ----------------------------------------------------------------------------------------

def test_read_exact_reads_and_raises_at_eof(tmp_path):
    f = tmp_path / "shard.bin"
    f.write_bytes(b"h" * 1024 + bytes(range(256)) * 4)
    buf = bytearray(1024)
    with open(f, "rb", buffering=0) as fh:
        assert exact_counts.read_exact(fh.fileno(), memoryview(buf), 1024) == 1024
    assert bytes(buf) == bytes(range(256)) * 4
    big = bytearray(1025)  # one byte past the end: the old loop retried preadv's 0 forever
    with open(f, "rb", buffering=0) as fh, pytest.raises(EOFError):
        exact_counts.read_exact(fh.fileno(), memoryview(big), 1024)


def test_load_corpus_error_reaches_launch():
    ec = exact_counts.ExactCounts.__new__(exact_counts.ExactCounts)
    ec.device, ec.files, ec.bufs, ec.n, ec.loaded = torch.device("cpu"), ["/nonexistent/shard.bin"], None, 1, threading.Event()
    ec.load_corpus()  # torch.cuda.set_device fails on this machine: any error must reach launch()
    assert ec.loaded.wait(30), "the loader thread never signalled (launch() would hang)"
    with pytest.raises(RuntimeError, match="exact-count corpus"):
        ec.launch()


# ---- retrieval ---------------------------------------------------------------------------------------------

def test_built_raises_on_failed_build():
    pending, ok, bad = Future(), Future(), Future()
    ok.set_result(None)
    bad.set_exception(OSError("SlotIndex.build failed"))
    assert retrieval.built(pending) is False and retrieval.built(ok) is True
    with pytest.raises(OSError):
        retrieval.built(bad)  # done() is True here: the half-built table must not be queried


LINGER = 15.0  # seconds rank 0 stays up after its build


class _Index:
    def __init__(self, fail):
        self.fail, self.released = fail, False

    def build(self):
        if self.fail:
            raise OSError("build failed on rank 0")

    def release(self):
        self.released = True


def _build_worker(rank, port, fail, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=__import__("datetime").timedelta(seconds=60))
    cache = retrieval.ValidationCache.__new__(retrieval.ValidationCache)
    cache.rank, cache.group, cache.index = rank, dist.group.WORLD, _Index(fail and rank == 0)
    t0 = time.monotonic()
    try:
        cache._build(None)
        results[rank] = "ok" + ("+released" if cache.index.released else "")
    except Exception as e:  # noqa: BLE001
        results[rank] = f"raised {type(e).__name__}"
    results[f"t{rank}"] = time.monotonic() - t0
    if rank == 0:
        time.sleep(LINGER)  # as in a run: rank 0 keeps training after its build thread failed
    dist.barrier()
    dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("fail", [False, True])
def test_build_failure_reaches_every_rank(fail):
    with mp.Manager() as m:
        results = m.dict()
        ctx = mp.spawn(_build_worker, args=(_free_port(), fail, results), nprocs=2, join=False)
        for p in ctx.processes:  # (ProcessContext.join returns as soon as one process ends)
            p.join(timeout=90)
        hung = [p for p in ctx.processes if p.is_alive()]
        for p in hung:
            p.kill()
        assert not hung, "a rank hung waiting for the failed build"
        got = dict(results)
    assert got.pop("t1") < LINGER / 2, "the peer waited for rank 0 instead of learning of the failure"
    got.pop("t0")
    if fail:  # rank 0 re-raises its own error, the peer is told instead of waiting at a barrier
        assert got == {0: "raised OSError", 1: "raised RuntimeError"}
    else:
        assert got == {0: "ok+released", 1: "ok"}


# ---- host --------------------------------------------------------------------------------------------------

def test_track2_host_plan_numbers():
    # PR #380 at 5800 steps: 12 shards excluded, so the 6-token index covers 91 shards; the 3-/2-token ones 20 each
    need = host.plan(91 * 100_000_000, 20 * 100_000_000, 64, 8)
    assert need.tables == (48 << 28) * 8 + 2 * (48 << 26) * 8 == 144 * host.GiB  # slot.rs: SLOTS << (PBITS + nbits)
    assert 140 * host.GiB < need.arenas < 155 * host.GiB


def test_cgroup_memory(tmp_path):
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "memory.max").write_text("214748364800\n")
    (v2 / "memory.current").write_text("10737418240\n")
    assert host.cgroup_memory(str(v2)) == (200 * host.GiB, 10 * host.GiB)
    (v2 / "memory.max").write_text("max\n")
    assert host.cgroup_memory(str(v2)) == (0, 0)
    v1 = tmp_path / "v1" / "memory"
    v1.mkdir(parents=True)
    (v1 / "memory.limit_in_bytes").write_text(str(1 << 62))  # v1's "unlimited"
    assert host.cgroup_memory(str(tmp_path / "v1")) == (0, 0)
    (v1 / "memory.limit_in_bytes").write_text(str(300 * host.GiB))
    (v1 / "memory.usage_in_bytes").write_text("0")
    assert host.cgroup_memory(str(tmp_path / "v1")) == (300 * host.GiB, 0)


def test_check_flags_small_shm_and_cgroup():
    need = host.plan(91 * 100_000_000, 20 * 100_000_000, 64, 8)
    h = host.Host(cpu_model="x", allowed=192, cores=96, numa_nodes=2, mem_total=2048 * host.GiB, mem_available=200 * host.GiB,
                  shm_total=64 * host.GiB, shm_free=64 * host.GiB, thp="always", avx512=True, kernel="k", gpus=8,
                  cgroup_limit=200 * host.GiB, stale_tables=40 * host.GiB)
    problems = host.check(h, need)
    assert len(problems) == 2
    assert "/dev/shm has 64 GiB free" in problems[0] and "40 GiB of stale" in problems[0]
    assert "cgroup memory limit is 200 GiB" in problems[1]
    big = host.Host(**{**h.__dict__, "mem_available": 1500 * host.GiB, "shm_total": 1000 * host.GiB,
                       "shm_free": 1000 * host.GiB, "cgroup_limit": 0, "stale_tables": 0})
    assert host.check(big, need) == []


def test_stale_tables(tmp_path):
    (tmp_path / "exact-match-dead").write_bytes(b"x" * 8192)
    (tmp_path / "other").write_bytes(b"x" * 8192)
    assert 8192 <= host.stale_table_bytes(str(tmp_path)) < 16384
