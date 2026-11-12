"""PR #381's validation hint rows (SCAN_HINTS, exact_counts.py) on 8 ranks and on 4, against a brute force.

#381 was measured on 4 GPUs only, while this tree (and its distributed.py) runs 8. Every world-size-dependent part
of its new code runs here, for real, on gloo ranks under the Triton interpreter: the per-rank sample cap
(SCAN_HINT_CAP // world), the scan's sampling, the context ids, the sample-row exchange (all_gather_into_tensor,
all_to_all_single) and the row kernel's WORLD * W tally. The rows must equal a brute force over each rank's shard
share: the longest marked 4-/3-/2-token context with a match, its samples (each rank's first cap occurrences in
corpus order), the top 8 next tokens by count (ties: lower token), the cell and the tag.

Needs real Triton with TRITON_INTERPRET=1 (skipped otherwise). From the repo root:

    bash 182/tools/triton_cpu/run.sh python -m pytest variants/t1-cpu-ngram/182/test_hint_rows_world.py -q
"""
import os
import socket
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # variants/t1-cpu-ngram

try:
    import triton  # noqa: F401
    import triton.language  # noqa: F401
    REAL_TRITON = hasattr(triton, "__version__")
except ImportError:
    REAL_TRITON = False
pytestmark = pytest.mark.skipif(not REAL_TRITON or os.environ.get("TRITON_INTERPRET") != "1",
                                reason="needs real Triton under TRITON_INTERPRET=1 (182/tools/triton_cpu/run.sh)")

SHARDS, SHARD_TOKENS, VOCAB = 8, 1500, 24
V, CHUNK, F = 512, 32, 32        # validation positions, chunk (per-rank interleave), fit positions per rank
TAGS = (300, 200)                # the 2-token rows' tag, the 3-token rows' (the 4-token rows take the 3-token tag)
CELLS = 168                      # exact_match.CELLS: 4 match-length buckets x 7 count bins x 6 purity bins
HB_TEST = 15                     # a small query table (the run's 2^27 buckets are 4 GiB)


def _data(tmp):
    rng = np.random.default_rng(0)
    p = 1.0 / np.arange(1, VOCAB + 1) ** 1.1  # skewed: some contexts exceed the 8-rank cap of 12 per rank
    p /= p.sum()
    files = []
    for s in range(SHARDS):
        path = os.path.join(tmp, f"fineweb_train_{s:06d}.bin")
        with open(path, "wb") as f:
            f.write(b"\0" * 1024)
            f.write(rng.choice(VOCAB, SHARD_TOKENS, p=p).astype(np.int16).tobytes())
        files.append(path)
    val = rng.choice(VOCAB, V + 1, p=p).astype(np.int32)
    return files, val


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _marks(seed):  # None: every context marked (SCAN_HINTS_SKIP=0); else random marks, the same on every rank
    return None if seed is None else np.random.default_rng(seed).integers(0, 8, V).astype(np.int8)


def _rank(rank, world, port, files, val, out_dir, mark_seed=None):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    sys.path.insert(0, ROOT)
    import torch.distributed as dist
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.set_device = lambda *a, **k: None

    class _NoCuda:
        def __init__(self, *a, **k): pass
        def synchronize(self): pass
        def record(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    torch.cuda.Stream = torch.cuda.Event = _NoCuda
    torch.cuda.stream = lambda s: _NoCuda()
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from track_1_short import exact_counts as ec_mod
    ec_mod.HB = HB_TEST
    i = torch.arange(V + world * F)
    q_chunk = torch.where(i < V, i // CHUNK * CHUNK, V + (i - V) // F * F)
    ec = ec_mod.ExactCounts(files, 0, rank, world, "cpu", q_chunk, pinned=False, n_val=V)
    ec.load_corpus()
    ec.loaded.wait()
    assert ec.load_error is None, ec.load_error
    fit = torch.as_tensor(np.random.default_rng(1).integers(0, VOCAB, world * F + 1), dtype=torch.int32)
    ec.q_tok[:V], ec.q_tgt[:V] = torch.as_tensor(val[:V]), torch.as_tensor(val[1:])
    ec.q_tok[V:], ec.q_tgt[V:] = fit[:-1], fit[1:]
    ec.queries.set()
    if mark_seed is not None:  # as launch_marked does with ValidationCache._marks' result
        ec.marks.copy_(torch.as_tensor(_marks(mark_seed)))
    ec.launch()
    dist.all_reduce(ec.out)
    own = torch.cat([torch.arange(o, o + CHUNK) for o in range(rank * CHUNK, V, world * CHUNK)])
    rows = ec.hint_rows(own, ec.marks[own], TAGS)
    np.save(os.path.join(out_dir, f"rows{rank}.npy"), rows.numpy())
    np.save(os.path.join(out_dir, f"own{rank}.npy"), own.numpy())
    np.save(os.path.join(out_dir, f"out{rank}.npy"), ec.out[own].numpy())
    dist.destroy_process_group()


def _cell(total, top):
    count_bin = sum(total >= t for t in (2, 3, 5, 9, 17, 33))
    purity = top / max(total, 1)
    purity_bin = sum(purity >= t for t in (0.3, 0.5, 0.7, 0.85, 0.95))
    return (7 + count_bin) * 6 + purity_bin


def _brute(files, val, world, marks=None):
    """Per validation position: (row [9], the counts of its 2-/3-/4-token contexts) as the code should make them."""
    cap = 96 // world
    widths = [w for w in (2, 8) if w < cap] + [cap]
    shares = [np.concatenate([np.fromfile(f, np.int16, offset=1024).astype(np.int64) for f in files[r::world]]) for r in range(world)]
    occ = {}  # context -> per rank, the next tokens of its occurrences in corpus order (None: the share's last token)
    for r, s in enumerate(shares):
        for L in (2, 3, 4):
            for p in range(L - 1, len(s)):
                occ.setdefault(tuple(s[p - L + 1:p + 1]), [[] for _ in range(world)])[r].append(int(s[p + 1]) if p + 1 < len(s) else None)
    rows, counts = np.zeros((V, 9), np.int64), np.zeros((V, 3), np.int64)
    for q in range(V):
        start = q // CHUNK * CHUNK
        best = None
        for k, L in ((0, 2), (1, 3), (2, 4)):
            if q - L + 1 < start:
                continue
            per = occ.get(tuple(val[q - L + 1:q + 1]), [[] for _ in range(world)])
            counts[q, k] = sum(map(len, per))
            # marked by this position (and the query before it exists, in the same segment) and matched
            if q >= 1 and counts[q, k] > 0 and (marks is None or marks[q] >> k & 1):
                best = (k, per)
        if best is None:
            continue
        k, per = best
        n_all = sum(map(len, per))
        w = widths[sum(n_all > x for x in widths[:-1])]  # each rank sends its first w samples (w <= cap)
        samples = [t for r in range(world) for t in per[r][:w] if t is not None]
        tally = {}
        for t in samples:
            tally[t] = tally.get(t, 0) + 1
        ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ranked[0][1] if ranked else 0
        rows[q, 0] = _cell(len(samples), top) + TAGS[min(k, 1)] if samples else 0
        for i, (t, c) in enumerate(ranked[:8]):
            rows[q, 1 + i] = t | (c << 16)
    return rows, counts


@pytest.mark.parametrize("world,mark_seed", [(8, None), (4, None), (8, 7), (4, 7)])
def test_hint_rows_match_brute_force(world, mark_seed):
    with tempfile.TemporaryDirectory() as tmp:
        files, val = _data(tmp)
        mp.start_processes(_rank, args=(world, _free_port(), files, val, tmp, mark_seed), nprocs=world, join=True, start_method="spawn")
        got = np.zeros((V, 9), np.int64)
        cnt = np.zeros((V, 3), np.int64)
        for r in range(world):
            own = np.load(os.path.join(tmp, f"own{r}.npy"))
            got[own] = np.load(os.path.join(tmp, f"rows{r}.npy"))
            cnt[own] = np.load(os.path.join(tmp, f"out{r}.npy"))[:, :3]
        want, want_cnt = _brute(files, val, world, _marks(mark_seed))
    # the exact counts of each position's 2-/3-/4-token context, summed over the ranks
    for k, L in enumerate((2, 3, 4)):
        ok = np.arange(V) - L + 1 >= np.arange(V) // CHUNK * CHUNK
        np.testing.assert_array_equal(cnt[ok, k], want_cnt[ok, k], err_msg=f"{L}-token context counts")
    assert (want[:, 0] > 0).sum() > V // (2 if mark_seed is None else 3), "the test data gives most positions a row"
    assert (cnt[:, 0] > 96).any(), "some context exceeds the cap (the capped path is exercised)"
    bad = np.flatnonzero((got != want).any(1))
    assert not len(bad), f"{len(bad)} of {V} rows differ at world={world}; first {bad[0]}: got {got[bad[0]]}, want {want[bad[0]]}"


def _marks_rank(rank, world, port, out_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    sys.path.insert(0, ROOT)
    import types
    import torch.distributed as dist
    sys.modules["exact_match"] = types.SimpleNamespace(CELLS=CELLS, SlotIndex=object, StreamIndex=object)
    if not hasattr(os, "sched_getaffinity"):
        os.sched_getaffinity = lambda pid: {0}
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from track_1_short import retrieval
    vc = retrieval.ValidationCache.__new__(retrieval.ValidationCache)
    vc.chunk, vc.group = CHUNK, None
    offsets = range(rank * CHUNK, V, world * CHUNK)  # as ValidationCache.__init__
    n = len(offsets) * CHUNK
    rng = np.random.default_rng(100 + rank)
    # this rank's rows, in its chunks' order: 6-token cells (tag 0), 3-token rows (tag 2) large or not, 2-token rows (tag 3)
    tag = rng.choice([0, 2, 3], n)
    cell = tag * CELLS + rng.choice([13 * 6 + 1, 8 * 6 + 2], n)
    empty = rng.random(n) < 0.6
    marks = vc._marks(cell, empty, 2, 7)
    large = cell % CELLS // 6 == 13  # bucket 1 (the key's own length), count bin 6 (>= 33 matches)
    mine = (empty * (4 + 2 * ~((tag == 2) & large) + (~(tag == 2) & ~large))).astype(np.int8)
    np.save(os.path.join(out_dir, f"marks{rank}.npy"), marks.numpy())
    np.save(os.path.join(out_dir, f"mine{rank}.npy"), mine)
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [8, 4])
def test_marks_land_in_validation_order(world):
    """ValidationCache._marks: each rank's marks, gathered, at its own chunks' places in the validation stream."""
    with tempfile.TemporaryDirectory() as tmp:
        mp.start_processes(_marks_rank, args=(world, _free_port(), tmp), nprocs=world, join=True, start_method="spawn")
        got = [np.load(os.path.join(tmp, f"marks{r}.npy")) for r in range(world)]
        mine = [np.load(os.path.join(tmp, f"mine{r}.npy")) for r in range(world)]
    for r in range(world):
        np.testing.assert_array_equal(got[r], got[0], err_msg="every rank holds the same marks")
    want = np.zeros(V, np.int8)
    for r in range(world):
        for j, o in enumerate(range(r * CHUNK, V, world * CHUNK)):
            want[o:o + CHUNK] = mine[r][j * CHUNK:(j + 1) * CHUNK]
    np.testing.assert_array_equal(got[0], want)
    assert len(set(want.tolist())) > 2
