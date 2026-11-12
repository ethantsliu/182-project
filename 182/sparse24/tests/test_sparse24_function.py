"""CPU tests of Sparse24ReLUSquareFunction and the per-slot refresh (track_1_short/perf/sparse24.py), on the dense
backend: the 2:4 GEMMs are emulated as dense GEMMs on masked weights, so everything checks against plain autograd.

    python -m pytest 182/sparse24/tests/test_sparse24_function.py -q
"""
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from track_1_short.perf import sparse24 as S  # noqa: E402


@pytest.fixture
def dense_backend(monkeypatch):
    monkeypatch.setattr(S, "SPARSE24_BACKEND", "dense")


def reference(x, W1m, W2m):
    """The masked MLP in plain autograd, w.r.t. the masked weights."""
    pre = x.reshape(-1, x.shape[-1]) @ W1m.T
    return (torch.relu(pre) ** 2 @ W2m).reshape(x.shape)


@pytest.mark.parametrize("kind", ["greedy", "optimal"])
def test_function_matches_masked_reference_fp64(dense_backend, kind):
    torch.manual_seed(0)
    T, h, d = 40, 64, 32
    x = torch.randn(1, T, d, dtype=torch.float64, requires_grad=True)
    W1 = (0.2 * torch.randn(h, d, dtype=torch.float64)).requires_grad_()
    W2 = (0.2 * torch.randn(h, d, dtype=torch.float64)).requires_grad_()
    m1, m2 = S.transposable_mask(W1, kind), S.transposable_mask(W2, kind)
    W1m, W2m = (W1 * m1).detach(), (W2 * m2).detach()
    out = S.Sparse24MLP(x, W1, W2, W1m, W1m.T, W2m, W2m.T)
    g = torch.randn_like(out)
    dx, dW1, dW2 = torch.autograd.grad(out, (x, W1, W2), g)

    xr = x.detach().clone().requires_grad_()
    W1r, W2r = W1m.clone().requires_grad_(), W2m.clone().requires_grad_()
    ref = reference(xr, W1r, W2r)
    rdx, rdW1, rdW2 = torch.autograd.grad(ref, (xr, W1r, W2r), g)
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    assert torch.allclose(dx, rdx, rtol=1e-12, atol=1e-12)
    # straight-through: the dense weights get the masked weights' gradient on every entry, pruned ones too
    assert torch.allclose(dW1, rdW1, rtol=1e-12, atol=1e-12)
    assert torch.allclose(dW2, rdW2, rtol=1e-12, atol=1e-12)
    assert (dW1[~m1] != 0).float().mean() > 0.5 and (dW2[~m2] != 0).float().mean() > 0.5


def test_function_bf16_close_to_fp32_reference(dense_backend):
    torch.manual_seed(1)
    T, h, d = 64, 128, 64
    x32 = torch.randn(1, T, d)
    x32 = x32 / x32.pow(2).mean(-1, keepdim=True).sqrt()
    W1_32, W2_32 = 0.5 * d ** -0.5 * torch.randn(h, d), 0.02 * torch.randn(h, d)
    W1m32, W2m32 = W1_32 * S.mask_tile_greedy(W1_32), W2_32 * S.mask_tile_greedy(W2_32)
    x = x32.bfloat16().requires_grad_()
    W1, W2 = W1_32.bfloat16().requires_grad_(), W2_32.bfloat16().requires_grad_()
    W1m, W2m = W1m32.bfloat16(), W2m32.bfloat16()
    out = S.Sparse24MLP(x, W1, W2, W1m, W1m.T, W2m, W2m.T)
    assert out.dtype == torch.bfloat16
    g = torch.randn(out.shape).bfloat16()
    dx, dW1, dW2 = torch.autograd.grad(out, (x, W1, W2), g)
    xr = x32.clone().requires_grad_()
    W1r, W2r = W1m32.clone().requires_grad_(), W2m32.clone().requires_grad_()
    ref = reference(xr, W1r, W2r)
    rdx, rdW1, rdW2 = torch.autograd.grad(ref, (xr, W1r, W2r), g.float())

    def rel(a, b):
        return float((a.detach().float() - b.detach()).norm() / b.detach().norm())
    assert rel(out, ref) < 2e-2 and rel(dx, rdx) < 2e-2 and rel(dW1, rdW1) < 2e-2 and rel(dW2, rdW2) < 2e-2


@pytest.mark.parametrize("kind", ["greedy", "optimal"])
def test_refresh_writes_masked_bank(dense_backend, monkeypatch, kind):
    monkeypatch.setattr(S, "SPARSE24_MASK", kind)
    torch.manual_seed(2)
    n, h, d = 3, 32, 16
    bank = torch.randn(n, 2, h, d).bfloat16()
    bufs = S.sparse24_buffers(n, h, d, torch.device("cpu"))
    assert set(bufs) == {"_mlp_sp24_w1", "_mlp_sp24_w2"}
    S.refresh_sparse24_(bank[:, 0], bank[:, 1], bufs)
    for s in range(n):
        a1, a1t, a2, a2t = S.slot_operands(bufs, s)
        for w, a in ((bank[s, 0], a1), (bank[s, 1], a2)):
            assert torch.equal(a, w * S.transposable_mask(w, kind))
        assert torch.equal(a1t, a1.T) and torch.equal(a2t, a2.T)
        assert S.is_24(a1 != 0, -1) and S.is_24(a1 != 0, -2)
    # addresses stay put across refreshes (CUDA graphs bake them)
    ptrs = {k: v.data_ptr() for k, v in bufs.items()}
    bank.mul_(-1.5)
    S.refresh_sparse24_(bank[:, 0], bank[:, 1], bufs)
    assert {k: v.data_ptr() for k, v in bufs.items()} == ptrs
    assert torch.equal(bufs["_mlp_sp24_w1"][0], bank[0, 0] * S.transposable_mask(bank[0, 0], kind))


def test_mask_follows_the_weights(dense_backend):
    """The mask is recomputed from |W| at every refresh: moving the weights moves the mask."""
    torch.manual_seed(3)
    bank = torch.randn(1, 2, 16, 16).bfloat16()
    bufs = S.sparse24_buffers(1, 16, 16, torch.device("cpu"))
    S.refresh_sparse24_(bank[:, 0], bank[:, 1], bufs)
    m0 = bufs["_mlp_sp24_w1"][0] != 0
    bank[0, 0][~m0] *= 100  # the pruned entries become the largest
    S.refresh_sparse24_(bank[:, 0], bank[:, 1], bufs)
    m1 = bufs["_mlp_sp24_w1"][0] != 0
    assert (m0 & m1).float().mean() < 0.1


def test_cslt_fake_shape():
    """The opaque cuSPARSELt op's fake (what torch.compile traces) gives z @ A.T's shape and dtype."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        packed = torch.empty(2816 * 768 * 9 // 16, dtype=torch.bfloat16)
        z = torch.empty(4096, 768, dtype=torch.bfloat16)
        y = torch.ops.sparse24.cslt_linear(packed, z, 2816)
        assert y.shape == (4096, 2816) and y.dtype == torch.bfloat16


def test_compiles_fullgraph_aot_eager(dense_backend):
    """Dynamo traces the Function (forward and backward) in one graph; aot_eager, so no Inductor/C++ compile."""
    torch.manual_seed(4)
    T, h, d = 32, 64, 32
    W1, W2 = (0.1 * torch.randn(h, d)).requires_grad_(), (0.1 * torch.randn(h, d)).requires_grad_()
    W1m, W2m = W1.detach() * S.mask_tile_greedy(W1), W2.detach() * S.mask_tile_greedy(W2)

    def f(x, W1, W2):
        return S.Sparse24MLP(x, W1, W2, W1m, W1m.T, W2m, W2m.T).square().sum()

    x = torch.randn(1, T, d, requires_grad=True)
    cf = torch.compile(f, backend="aot_eager", fullgraph=True)
    got = torch.autograd.grad(cf(x, W1, W2), (x, W1, W2))
    want = torch.autograd.grad(f(x, W1, W2), (x, W1, W2))
    for a, b in zip(got, want):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_sparse24_active_by_stage(monkeypatch):
    from track_1_short.config import BLOCK_SIZE, TRAINING_STAGES, WS_POST_YARN_EXT
    ws = [s.window_sizes[1] * BLOCK_SIZE for s in TRAINING_STAGES] + [WS_POST_YARN_EXT * BLOCK_SIZE]  # + validation
    monkeypatch.setattr(S, "SPARSE24_MLP", False)
    assert [S.sparse24_active(w) for w in ws] == [False] * 6
    monkeypatch.setattr(S, "SPARSE24_MLP", True)
    want = {0: [True] * 6,
            13: [True, True, True, True, False, False],    # dense extension stage + validation
            11: [True, True, False, False, False, False],  # dense from stage 2
            7: [True, False, False, False, False, False]}
    for n, row in want.items():
        monkeypatch.setattr(S, "SPARSE24_DENSE_FROM_WS", n)
        assert [S.sparse24_active(w) for w in ws] == row, n
