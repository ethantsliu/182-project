"""CPU tests of the retrieval's failure paths, the host check and the TABLE_VOTE report's chain bookkeeping (no GPU, no Triton, no built extension needed).

    cd variants/t1-cpu-ngram && python3 -m pytest -q 182

Triton and the exact_match extension are stubbed where a module imports them; only pure-Python code paths run.
"""
import concurrent.futures
import contextlib
import os
import sys
import threading
import types

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # variants/t1-cpu-ngram
sys.path.insert(0, ROOT)

from track_1_short import host  # noqa: E402

GiB = host.GiB


def _stub(name, **attrs):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod


def _identity_jit(fn=None, **_):
    return fn if fn is not None else (lambda f: f)


@pytest.fixture(scope="module")
def exact_counts():
    _stub("triton", jit=_identity_jit, cdiv=lambda a, b: -(-a // b))
    _stub("triton.language", constexpr=int)
    sys.modules["triton"].language = sys.modules["triton.language"]
    from track_1_short import exact_counts
    return exact_counts


@pytest.fixture(scope="module")
def retrieval():
    _stub("exact_match", CELLS=4, SlotIndex=object, StreamIndex=object)
    if not hasattr(os, "sched_getaffinity"):  # not Linux: retrieval reads the CPU set at import
        os.sched_getaffinity = lambda pid: set(range(os.cpu_count() or 1))
    from track_1_short import retrieval
    return retrieval


class _NoCuda:  # torch.cuda's stream/event calls, as no-ops on CPU
    def __init__(self, *a, **k): pass
    def synchronize(self): pass
    def record(self, *a): pass


def _shard(path, n_tokens):
    with open(path, "wb") as f:
        f.write(b"\0" * 1024)
        f.write(np.arange(n_tokens, dtype=np.int16).tobytes())
    return str(path)


def _counts(exact_counts, files, pinned):
    ec = exact_counts.ExactCounts.__new__(exact_counts.ExactCounts)
    ec.device, ec.files = "cpu", files
    ec.n = sum((os.path.getsize(f) - 1024) // 2 for f in files)
    ec.bufs = [torch.empty(max((os.path.getsize(f) - 1024) // 2 for f in files), dtype=torch.int16) for _ in range(2)] if pinned else None
    ec.corpus = torch.zeros(ec.n, dtype=torch.int16)
    ec.loaded = threading.Event()
    ec._count = lambda *a, **k: None
    ec.out = None
    ec.q_tok = ec.q_tgt = ec.q_chunk = torch.zeros(16, dtype=torch.int32)
    return ec


@contextlib.contextmanager
def _cpu_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)
    monkeypatch.setattr(torch.cuda, "Stream", _NoCuda)
    monkeypatch.setattr(torch.cuda, "Event", _NoCuda)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    yield


@pytest.mark.parametrize("pinned", [False, True])
def test_corpus_load_reads_every_token(exact_counts, tmp_path, monkeypatch, pinned):
    files = [_shard(tmp_path / f"s{i}.bin", 3000 + 7 * i) for i in range(3)]
    ec = _counts(exact_counts, files, pinned)
    with _cpu_cuda(monkeypatch):
        ec.load_corpus()
        assert ec.loaded.wait(10)
    assert ec.load_error is None
    want = np.concatenate([np.fromfile(f, dtype=np.int16, offset=1024) for f in files])
    assert np.array_equal(ec.corpus.numpy(), want)
    ec.launch()  # no error: proceeds to the count


@pytest.mark.parametrize("pinned", [False, True])
def test_truncated_shard_fails_launch_instead_of_hanging(exact_counts, tmp_path, monkeypatch, pinned):
    files = [_shard(tmp_path / f"s{i}.bin", 4000) for i in range(2)]
    ec = _counts(exact_counts, files, pinned)  # sizes planned at construction
    with open(files[1], "r+b") as f:  # then a shard shrinks before the load
        f.truncate(1024 + 2 * 1000)
    if pinned:  # the pinned path sizes each read by the file's current size; a short read past it must not spin
        real = os.preadv
        calls = []
        def short(fd, bufs, off):
            calls.append(off)
            assert len(calls) < 1000, "preadv loop does not terminate"
            return real(fd, bufs, off)
        monkeypatch.setattr(os, "preadv", short)
        monkeypatch.setattr(os.path, "getsize", lambda f, _g=os.path.getsize: _g(f) + (6000 if f == files[1] else 0))
    with _cpu_cuda(monkeypatch):
        ec.load_corpus()
        assert ec.loaded.wait(10), "a failed load must still release launch()"
    assert isinstance(ec.load_error, EOFError)
    with pytest.raises(RuntimeError, match="corpus load failed"):
        ec.launch()


def test_loader_exception_releases_launch(exact_counts, tmp_path, monkeypatch):
    ec = _counts(exact_counts, [_shard(tmp_path / "s.bin", 100)], False)
    def boom(d):
        raise MemoryError("no room")
    monkeypatch.setattr(torch.cuda, "set_device", boom)
    ec.load_corpus()
    assert ec.loaded.wait(10)
    with pytest.raises(RuntimeError) as err:
        ec.launch()
    assert isinstance(err.value.__cause__, MemoryError)


def test_failed_build_is_never_queried(retrieval):
    oc = retrieval.OnlineCache.__new__(retrieval.OnlineCache)
    failed = concurrent.futures.Future()
    failed.set_exception(OSError("scan arena is full"))
    queried = []
    cache = types.SimpleNamespace(built=failed, index=types.SimpleNamespace(query_into=lambda *a: queried.append(a)))
    oc.lower, oc.full, oc.mix, oc.hint_from = [(2, cache)], None, None, (0, 0)
    oc.index = types.SimpleNamespace(rows=lambda *a: np.zeros((8, 9), dtype=np.int32))
    with pytest.raises(OSError, match="arena is full"):
        oc._rows(5, np.zeros(8), None, None, np.zeros(8, dtype=np.int64))
    assert not queried


def test_build_failure_reaches_every_rank(retrieval, monkeypatch):
    """Rank 0's build raises: its status goes through the post-build collective, so a peer's build future fails too."""
    reduced = []
    def all_reduce(t, group=None):
        reduced.append(int(t.item()))
        t.fill_(1)  # as if rank 0 reported a failure
    monkeypatch.setattr(retrieval.dist, "all_reduce", all_reduce)
    peer = retrieval.ValidationCache.__new__(retrieval.ValidationCache)
    peer.rank, peer.group = 1, None
    with pytest.raises(RuntimeError, match="another rank"):
        peer._build(None)
    assert reduced == [0]
    zero = retrieval.ValidationCache.__new__(retrieval.ValidationCache)
    zero.rank, zero.group = 0, None
    def fail():
        raise OSError("scan arena is full")
    zero.index = types.SimpleNamespace(build=fail)
    with pytest.raises(OSError):
        zero._build(None)
    assert reduced == [0, 1]


# host.check on the records' boxes: it must never abort a box the published runs used
TOK6, TOK20 = 101 * 100_000_000, 20 * 100_000_000

def _box(ram_gib, avail_gib, shm_gib, cpus=128, cgroup=0):
    return host.Host("Xeon", cpus, cpus // 2, 2, ram_gib * GiB, avail_gib * GiB, shm_gib * GiB, shm_gib * GiB, "madvise", True, "6.8", 8,
                     cgroup_limit=cgroup)

@pytest.mark.parametrize("shared", [False, True])
def test_record_boxes_pass(shared):
    for box in (_box(2048, 1900, 1024), _box(1575, 1450, 787)):  # #380's node (2 TB), #367's (1575 GiB)
        assert host.check(box, host.plan(TOK6, TOK20, 32, 8, shared=shared)) == []
        assert host.check(box, host.plan(TOK6, TOK20, 64, 8, shared=shared)) == []


def test_small_box_fails():
    problems = host.check(_box(256, 240, 128), host.plan(TOK6, TOK20, 32, 8))
    assert len(problems) == 2  # /dev/shm and RAM


def _cg(root, rel, files):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (d / k).write_text(v)


def test_cgroup_v2_limit(tmp_path):
    _cg(tmp_path, "kubepods/pod1", {"memory.max": str(300 * GiB), "memory.current": str(100 * GiB),
                                    "memory.stat": f"anon {40 * GiB}\nfile {70 * GiB}\nshmem {10 * GiB}\n"})
    proc = tmp_path / "cgroup"
    proc.write_text("0::/kubepods/pod1\n")
    limit, headroom = host.cgroup_memory(str(tmp_path), str(proc))
    assert limit == 300 * GiB
    assert headroom == (300 - (100 - (70 - 10))) * GiB  # page cache is reclaimable, shmem (/dev/shm) is not


def test_cgroup_unlimited(tmp_path):
    _cg(tmp_path, "", {"memory.max": "max", "memory.current": "0"})
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n")
    assert host.cgroup_memory(str(tmp_path), str(proc)) is None
    _cg(tmp_path, "memory", {"memory.limit_in_bytes": str((1 << 63) - 4096)})
    proc.write_text("5:memory:/\n")
    assert host.cgroup_memory(str(tmp_path), str(proc)) is None


def test_cgroup_v1_limit(tmp_path):
    _cg(tmp_path, "memory/docker/abc", {"memory.limit_in_bytes": str(200 * GiB), "memory.usage_in_bytes": str(50 * GiB),
                                        "memory.stat": f"cache {30 * GiB}\nshmem {5 * GiB}\ntotal_cache {30 * GiB}\ntotal_shmem {5 * GiB}\n"})
    proc = tmp_path / "cgroup"
    proc.write_text("7:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n")
    assert host.cgroup_memory(str(tmp_path), str(proc)) == (200 * GiB, (200 - (50 - 25)) * GiB)


def test_detect_runs_here():
    h = host.detect(gpus=8)
    assert h.allowed >= 1 and h.mem_total > 0
    assert "host:" in host.describe(h)


# TABLE_VOTE: the run reports the loss of the chain refit without the table's links (train_gpt.py: vote_base_prob)
def _vote_case(exact_counts, retrieval, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    J = len(exact_counts.ORDERS)
    N = torch.randint(0, 50, (n, J), generator=g).float()
    C = (N * torch.rand(n, J, generator=g)).floor()
    cell = torch.randint(0, 4 * retrieval.RET_CELLS, (n,), generator=g)
    hints = retrieval.rows_part(cell, torch.rand(n, generator=g))
    doc = [(torch.rand(n, generator=g), torch.randint(-1, 7, (n,), generator=g), 7) for _ in range(3)]
    counts = np.zeros((n, 6), dtype=np.uint32)
    counts[:, :3] = torch.randint(0, 40, (n, 3), generator=g).numpy()
    counts[:, 3:] = (counts[:, :3] * torch.rand(n, 3, generator=g).numpy()).astype(np.uint32)
    return (N, C), hints, doc, counts, torch.rand(n, generator=g) * 0.5, torch.randn(n, 8, generator=g), g


def test_vote_report_base_chain_is_the_chain_without_the_table_links(exact_counts, retrieval):
    ec = exact_counts
    counts_nc, hints, doc, votes, p, hidden, g = _vote_case(ec, retrieval)
    J, B = len(ec.ORDERS), retrieval.VOTE_BINS
    base_groups = J * ec.NBINS + 4 * retrieval.RET_CELLS + 3 * 7
    K0 = J + 1 + 3
    theta, W, b = torch.randn(base_groups, generator=g).double(), torch.randn(8, K0, generator=g) * 0.1, torch.randn(K0, generator=g) * 0.1
    base = ec.vote_base_prob(p, counts_nc, hints + doc, hidden, (theta, W, b), base_groups)
    # by hand: the chain over the same links
    Q, G, total = ec.components(*counts_nc, hints + doc)
    assert total == base_groups
    assert torch.equal(base, ec.chain_prob(p, Q, ec.gate_L(hidden, G, theta, W, b, base_groups)))
    assert ((base > 0) & (base < 1)).all()
    # the run's chain (the table's links between the hint rows' and the document's) with those links' weights at zero
    cut, kcut = J * ec.NBINS + 4 * retrieval.RET_CELLS, J + 1
    theta_f = torch.cat([theta[:cut], torch.full((3 * B,), -700.0, dtype=torch.float64), theta[cut:]])
    W_f = torch.cat([W[:, :kcut], torch.zeros(8, 3), W[:, kcut:]], 1)
    b_f = torch.cat([b[:kcut], torch.zeros(3), b[kcut:]])
    Qf, Gf, total_f = ec.components(*counts_nc, hints + retrieval.table_part(votes, "cpu") + doc)
    assert total_f == base_groups + 3 * B and Qf.shape[1] == K0 + 3
    full = ec.chain_prob(p, Qf, ec.gate_L(hidden, Gf, theta_f, W_f, b_f, total_f))
    assert torch.allclose(full, base, rtol=0, atol=1e-12)
    with pytest.raises(AssertionError, match="groups"):
        ec.vote_base_prob(p, counts_nc, hints + retrieval.table_part(votes, "cpu") + doc, hidden, (theta, W, b), base_groups)


def test_fixed_vote_is_pr399s_formula(retrieval):
    p = torch.tensor([0.2, 0.5, 0.9])
    counts = np.array([[0, 0, 0, 0, 0, 0], [10, 0, 0, 4, 0, 0], [10, 5, 2, 10, 5, 2]], dtype=np.uint32)
    out = retrieval.table_vote(p, counts)
    k6, k10, k20 = retrieval.VOTE_K
    assert out[0] == pytest.approx(0.2)                       # no candidates: unchanged
    assert out[1].item() == pytest.approx((4 + k6 * 0.5) / (10 + k6))
    want = (10 + k6 * 0.9) / (10 + k6)
    want = (5 + k10 * want) / (5 + k10)
    assert out[2].item() == pytest.approx((2 + k20 * want) / (2 + k20))
    links = retrieval.table_part(counts, "cpu")
    assert [n for _, _, n in links] == [retrieval.VOTE_BINS] * 3
    assert links[0][1].tolist() == [-1, 3, 3] and links[0][0].tolist() == pytest.approx([0.0, 0.4, 1.0])


def test_table_vote_flag_values(retrieval):
    import subprocess
    code = ("import sys, types, os; m = types.ModuleType('exact_match'); m.CELLS = 4; m.SlotIndex = m.StreamIndex = object; "
            "sys.modules['exact_match'] = m; os.sched_getaffinity = getattr(os, 'sched_getaffinity', lambda p: {0}); "
            "from track_1_short import retrieval as r; print(r.TABLE_VOTE, r.SLOT_SHARED_ARENA, r.STREAM_THREADS)")
    def flag(value):
        env = {k: v for k, v in os.environ.items() if k not in ("TABLE_VOTE", "SLOT_SHARED_ARENA", "STREAM_THREADS")}
        if value is not None:
            env["TABLE_VOTE"] = value
        return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert flag(None).stdout.split() == ["False", "False", "4"]  # every flag off by default
    assert flag("0").stdout.split()[0] == "False" and flag("1").stdout.split()[0] == "True" and flag("chain").stdout.split()[0] == "True"
    assert flag("fixed").returncode != 0 and "TABLE_VOTE=fixed" in flag("fixed").stderr
