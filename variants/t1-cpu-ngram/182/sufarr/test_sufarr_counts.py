"""EXACT_BACKEND=sa: exact_match.SuffixCounts and track_1_short/exact_counts_sa.py against the exact reference
(sufarr_ref.py), on random corpora and a real fineweb10B sample, through real shard files (headers, several files per
share, grams across file junctions but not across shares) and the trainer's query layout (chunks, fit segments,
document starts). test_sufarr_kernels.py checks the GPU scan's kernels against the same reference.

Needs the extension built with the `sa` feature (skipped otherwise): bash 182/sufarr/build.sh, then
    python -m pytest -q 182/sufarr/test_sufarr_counts.py"""
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # variants/t1-cpu-ngram
sys.path.insert(0, HERE)
from sufarr_ref import COLS, cases, reference, segment_starts, write_shard  # noqa: E402

try:
    import exact_match
    HAVE_SA = hasattr(exact_match, "SuffixCounts")
    if not HAVE_SA:  # e.g. the crate's directory taken for a namespace package: not left for the other tests' stubs
        del sys.modules["exact_match"]
except ImportError:
    HAVE_SA = False
pytestmark = pytest.mark.skipif(not HAVE_SA, reason="exact_match without the `sa` feature (bash 182/sufarr/build.sh)")
CASES = cases()


def _files(tmp, files):  # [rank][file] token arrays -> [rank][path], named so that sorted()[rank::world] gives them back
    world = len(files)
    paths = [[None] * len(share) for share in files]
    for r, share in enumerate(files):
        for k, toks in enumerate(share):
            paths[r][k] = os.path.join(tmp, f"fineweb_train_{k * world + r:06d}.bin")
            write_shard(paths[r][k], toks)
    return paths


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_counts_match_reference(case, tmp_path):
    name, world, files, tok, tgt, q_chunk = case
    start = segment_starts(tok, q_chunk)
    want = reference([np.concatenate(share) for share in files], tok, tgt, start)
    got = np.zeros_like(want)
    for r, paths in enumerate(_files(str(tmp_path), files)):
        sc = exact_match.SuffixCounts(paths, 2)
        assert sc.tokens == sum(map(len, files[r])) and not sc.packed
        out = np.zeros((len(tok), COLS), np.int32)
        sc.counts_into(tok, tgt, start, out)
        got += out
    bad = np.flatnonzero((got != want).any(1))
    assert not len(bad), f"{name}: {len(bad)} of {len(tok)} rows differ; first {bad[0]}: got {got[bad[0]]}, want {want[bad[0]]}"
    # the case means something: counts at every order, a context counted 0 although it occurs (the table rule), and
    # positions whose context stops at a document or chunk start
    assert (want[:, COLS - 1] > 0).sum() >= 20, f"{name}: too few 9-gram matches"
    assert ((want[:, :7] > 0).sum(0) > 0).all() and (start > 0).any()


def test_packed_suffix_array_matches():  # the 40-bit entries (shares of 2^31 tokens or more), forced on a small one
    name, world, files, tok, tgt, q_chunk = CASES[-1]
    start = segment_starts(tok, q_chunk)
    for r in range(world):
        share = np.concatenate(files[r])
        a, b = (exact_match.SuffixCounts([], 2, tokens=share, force_p40=p) for p in (False, True))
        assert b.packed and not a.packed and b.memory == a.memory + len(share)
        oa, ob = (np.zeros((len(tok), COLS), np.int32) for _ in range(2))
        a.counts_into(tok, tgt, start, oa)
        b.counts_into(tok, tgt, start, ob)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(oa, reference([share], tok, tgt, start))


def test_bad_queries_raise():
    sc = exact_match.SuffixCounts([], 1, tokens=np.arange(100, dtype=np.uint16))
    tok = np.array([1, 2, 3], np.int32)
    with pytest.raises(ValueError):  # a start after its position
        sc.counts_into(tok, tok, np.array([0, 2, 3], np.int64)[::-1].copy(), np.zeros((3, COLS), np.int32))
    with pytest.raises(ValueError):  # the wrong output shape
        sc.counts_into(tok, tok, np.zeros(3, np.int64), np.zeros((3, COLS - 1), np.int32))


@pytest.fixture
def backend(monkeypatch):
    """(exact_counts, exact_counts_sa): launch() uses exact_counts.segment_starts, and that module imports triton at
    the top, so without triton a stand-in that makes its kernel decorators inert is put in its place. The modules
    track_1_short modules imported here are dropped afterwards, so no other test sees one built on the stand-in."""
    monkeypatch.syspath_prepend(ROOT)
    try:
        import triton  # noqa: F401
    except ImportError:
        tl = types.SimpleNamespace(constexpr=lambda x: x)
        jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)  # noqa: E731
        monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(jit=jit, language=tl, cdiv=lambda a, b: -(-a // b)))
        monkeypatch.setitem(sys.modules, "triton.language", tl)
    before = set(sys.modules)
    from track_1_short import exact_counts, exact_counts_sa
    yield exact_counts, exact_counts_sa
    for name in set(sys.modules) - before:
        if name.startswith("track_1_short"):
            del sys.modules[name]


def test_backend_class_matches_reference(backend, tmp_path):
    """SuffixExactCounts as train_gpt.py drives it (load_corpus, then the queries, launch, and the all-reduce's sum),
    on the CPU, with the trainer's own segment_starts."""
    import torch
    exact_counts, ecs = backend
    name, world, files, tok, tgt, q_chunk = CASES[-2]
    paths = sorted(p for share in _files(str(tmp_path), files) for p in share)
    with pytest.raises(ValueError, match="SCAN_HINTS"):
        ecs.SuffixExactCounts(paths, 0, 0, world, "cpu", torch.as_tensor(q_chunk), n_val=len(tok))
    total = torch.zeros(len(tok), COLS, dtype=torch.int32)
    for r in range(world):
        ec = ecs.SuffixExactCounts(paths, 0, r, world, "cpu", torch.as_tensor(q_chunk))
        assert ec.n == sum(map(len, files[r]))
        ec.load_corpus()
        ec.q_tok.copy_(torch.as_tensor(tok))
        ec.q_tgt.copy_(torch.as_tensor(tgt))
        ec.queries.set()
        ec.launch()
        assert ec.loaded.is_set() and ec.load_error is None and ec.build_s is not None
        total += ec.out
    start = exact_counts.segment_starts(torch.as_tensor(tok), torch.as_tensor(q_chunk)).numpy()
    np.testing.assert_array_equal(start, segment_starts(tok, q_chunk))
    want = reference([np.concatenate(share) for share in files], tok, tgt, start)
    np.testing.assert_array_equal(total.numpy(), want)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the build thread re-raises, as ExactCounts'
def test_backend_build_error_reaches_launch(backend, tmp_path):
    import torch
    _, ecs = backend
    path = os.path.join(str(tmp_path), "fineweb_train_000000.bin")
    write_shard(path, np.arange(50, dtype=np.uint16))
    ec = ecs.SuffixExactCounts([path], 0, 0, 1, "cpu", torch.zeros(4, dtype=torch.int64))
    with open(path, "r+b") as f:  # the header no longer agrees with the size
        f.truncate(1024 + 60)
    ec.load_corpus()
    with pytest.raises(RuntimeError, match="suffix array build failed"):
        ec.launch()
