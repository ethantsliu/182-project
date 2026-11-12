"""CPU tests of the 2:4 masks in track_1_short/perf/sparse24.py (pure torch, no GPU, no Triton).

    python -m pytest 182/sparse24/tests/test_sparse24_masks.py -q
"""
import itertools
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from track_1_short.perf import sparse24 as S  # noqa: E402


def tiles(mask):
    return S._to_tiles(mask).reshape(-1, 4, 4)


def test_tile_roundtrip():
    w = torch.arange(8 * 12).reshape(8, 12)
    t = S._to_tiles(w)
    assert t.shape == (2 * 3, 16)
    assert t[0].tolist() == [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39]  # rows 0-3, cols 0-3
    assert t[1].tolist()[:4] == [4, 5, 6, 7]                                              # next tile along K
    assert torch.equal(S._from_tiles(t, w.shape), w)
    wb = torch.randn(3, 8, 12)
    assert torch.equal(S._from_tiles(S._to_tiles(wb), wb.shape), wb)


def test_maximal_patterns():
    p = S.maximal_patterns().reshape(186, 4, 4)
    assert len({tuple(x.flatten().tolist()) for x in p}) == 186
    assert bool((p.sum(1) <= 2).all() and (p.sum(2) <= 2).all())
    sizes = p.sum((1, 2))
    assert int((sizes == 8).sum()) == 90 and int((sizes == 7).sum()) == 96
    # independent count of the 90 exact ones: 4x4 0/1 matrices with all row and column sums 2
    count = sum(1 for bits in itertools.product((0, 1), repeat=16)
                if all(sum(bits[4 * r:4 * r + 4]) == 2 for r in range(4))
                and all(sum(bits[c::4]) == 2 for c in range(4)))
    assert count == 90
    # maximal: adding any zero entry breaks a row or column limit
    for x in p:
        for r, c in (x == 0).nonzero().tolist():
            assert x[r].sum() == 2 or x[:, c].sum() == 2


@pytest.mark.parametrize("kind", ["greedy", "optimal"])
@pytest.mark.parametrize("shape", [(64, 48), (2, 32, 64), (2816 // 8, 768 // 8)])
def test_masks_are_transposable_24(kind, shape):
    torch.manual_seed(0)
    w = torch.randn(shape)
    m = S.transposable_mask(w, kind)
    assert m.shape == w.shape and m.dtype == torch.bool
    assert S.is_24(m, -1) and S.is_24(m, -2)                     # W and W.T both 2:4
    per_tile = tiles(m).sum((1, 2))
    assert bool((per_tile >= 7).all() and (per_tile <= 8).all())  # a maximal pattern: 8, or 7 when one is stranded


def _greedy_tile_ref(tile):
    """Literal per-tile greedy: 16 entries by |w| descending (ties: lower index first)."""
    a = tile.abs().flatten().tolist()
    order = sorted(range(16), key=lambda i: (-a[i], i))
    rows, cols, keep = [0] * 4, [0] * 4, [False] * 16
    for i in order:
        r, c = divmod(i, 4)
        if rows[r] < 2 and cols[c] < 2:
            keep[i], rows[r], cols[c] = True, rows[r] + 1, cols[c] + 1
    return torch.tensor(keep).reshape(4, 4)


def test_greedy_matches_literal_reference():
    torch.manual_seed(1)
    w = torch.randn(32, 32)
    m = tiles(S.mask_tile_greedy(w))
    for i, t in enumerate(tiles(w)):
        assert torch.equal(m[i], _greedy_tile_ref(t)), i


def test_seven_can_beat_every_exact_pattern():
    # Greedy keeps 9, 8, 7, 6, 5, 4, then row 3 / column 3 hold the only free slot: 7 kept, sum 40. The best
    # exact (8-entry) pattern must take a zero instead of the 4 and reaches only 36. Optimal covers both.
    t = torch.tensor([[9., 8., 0., 0.],
                      [7., 0., 6., 0.],
                      [0., 5., 4., 0.],
                      [0., 0., 0., 1.]])
    g, o = S.mask_tile_greedy(t), S.mask_tile_optimal(t)
    assert int(g.sum()) == 7 and float((t * g).sum()) == 40.0
    exact = S.maximal_patterns()[:90].float()
    assert float((S._to_tiles(t) @ exact.T).max()) == 36.0
    assert float((t * o).sum()) == 40.0


def test_optimal_never_below_greedy():
    torch.manual_seed(5)
    w = torch.randn(3, 64, 64)
    a = (w.abs() * S.mask_tile_optimal(w)).flatten(1).sum(1)
    b = (w.abs() * S.mask_tile_greedy(w)).flatten(1).sum(1)
    assert bool((a >= b).all())
    tg = S._to_tiles(w.abs() * S.mask_tile_greedy(w)).sum(-1)
    to = S._to_tiles(w.abs() * S.mask_tile_optimal(w)).sum(-1)
    assert bool((to >= tg - 1e-6).all())  # per tile, too


def test_optimal_is_exhaustive_best():
    torch.manual_seed(2)
    pats = S.maximal_patterns().float()
    w = torch.randn(16, 16)
    m = tiles(S.mask_tile_optimal(w)).reshape(-1, 16).float()
    t = S._to_tiles(w.abs())
    best = (t @ pats.T).max(dim=1).values
    assert torch.allclose((t * m).sum(1), best)


def test_kept_energy_ordering():
    """Rowwise 2:4 (non-transposable) >= transposable optimal >= transposable greedy (L1), on Gaussian weights.
    The printed numbers are quoted in 182/sparse24/README.md."""
    torch.manual_seed(3)
    w = torch.randn(2816, 768) * 0.02
    fr = {k: (S.kept_fraction(w, m, 1), S.kept_fraction(w, m, 2))
          for k, m in (("rowwise", S.mask_rowwise(w)), ("optimal", S.mask_tile_optimal(w)),
                       ("greedy", S.mask_tile_greedy(w)))}
    print("kept |w| / w^2: " + ", ".join(f"{k} {a:.4f} / {b:.4f}" for k, (a, b) in fr.items()))
    assert fr["rowwise"][0] >= fr["optimal"][0] >= fr["greedy"][0]
    assert 0.5 < fr["greedy"][0] < fr["rowwise"][0] < 1.0


def test_rowwise_is_24_along_rows_only():
    torch.manual_seed(4)
    w = torch.randn(64, 64)
    m = S.mask_rowwise(w)
    assert S.is_24(m, -1)
    assert int(m.sum()) == w.numel() // 2
    assert not S.is_24(m, -2)  # generic data: the transpose is not 2:4 (why the tile masks exist)


def test_zero_matrix_still_gets_a_valid_mask():
    # c_proj is zero at init: every |w| ties. The mask must still be a valid transposable 2:4 pattern.
    for kind in ("greedy", "optimal"):
        m = S.transposable_mask(torch.zeros(8, 8), kind)
        assert S.is_24(m, -1) and S.is_24(m, -2) and int(m.sum()) >= 14
