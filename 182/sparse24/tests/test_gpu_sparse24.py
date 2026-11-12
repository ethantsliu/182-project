"""GPU tests of the cuSPARSELt path of SPARSE24_MLP (track_1_short/perf/sparse24.py). Skip without a CUDA GPU with
cuSPARSELt (every test here needs the sparse tensor cores); on the node:

    python -m pytest 182/sparse24/tests/test_gpu_sparse24.py -q -rs

What they pin down, none of which the CPU tests can: torch's greedy tile kernel picks the mask mask_tile_greedy
emulates; each of the four compressed operands (W1, W1.T, W2, W2.T) multiplies as z @ A.T in the layout
cslt_linear assumes (transpose_result); the cuSPARSELt Function matches the dense emulation forward and backward;
the refresh and the Function run under torch.compile(fullgraph=True) and inside a captured CUDA graph, as in training.
"""
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _cusparselt_ok():
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    if torch.cuda.get_device_capability() < (8, 0):
        return False, "2:4 sparse tensor cores need sm_80+"
    ok = getattr(torch.backends, "cusparselt", None) is not None and torch.backends.cusparselt.is_available()
    return ok, "torch built without cuSPARSELt"


_OK, _WHY = _cusparselt_ok()
pytestmark = pytest.mark.skipif(not _OK, reason=_WHY)

from track_1_short.perf import sparse24 as S  # noqa: E402

H, D = 2816, 768
DEV = "cuda"


def rel(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def rand_w(*shape, std=0.03, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (std * torch.randn(*shape, device=DEV, generator=g)).bfloat16()


def rms_rows(t):
    return (t / t.float().pow(2).mean(-1, keepdim=True).sqrt()).bfloat16()


def test_torch_greedy_tile_matches_emulation():
    from torch.sparse import SparseSemiStructuredTensorCUTLASS
    w = rand_w(H, D, seed=1)
    pruned = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(w).to_dense()
    got = pruned != 0
    want = S.mask_tile_greedy(w)
    agree = float((got == want).float().mean())
    print(f"torch tile kernel vs mask_tile_greedy: {agree:.6f} of entries agree")
    assert S.is_24(got, -1) and S.is_24(got, -2)
    assert agree > 0.9999  # bf16 ties are the only allowed difference


@pytest.mark.parametrize("kind", ["greedy", "optimal"])
def test_compressed_operands_decode_to_masked_weights(monkeypatch, kind):
    """cslt_linear(packed, I, m) = I @ A.T recovers A.T exactly: each buffer holds the masked W (or W.T)."""
    monkeypatch.setattr(S, "SPARSE24_BACKEND", "cusparselt")
    monkeypatch.setattr(S, "SPARSE24_MASK", kind)
    w = rand_w(H, D, seed=2)
    wm = w * S.transposable_mask(w, kind)
    packed, packed_t = S._compress_pair(w, kind)
    a = S.cslt_linear(packed, torch.eye(D, device=DEV, dtype=torch.bfloat16), H).T     # A = W  [H, D]
    at = S.cslt_linear(packed_t, torch.eye(H, device=DEV, dtype=torch.bfloat16), D).T  # A = W.T [D, H]
    assert torch.equal(a, wm) and torch.equal(at, wm.T)


@pytest.mark.parametrize("T", [4096, 16384])
def test_cslt_linear_all_four_layouts(monkeypatch, T):
    monkeypatch.setattr(S, "SPARSE24_BACKEND", "cusparselt")
    w = rand_w(H, D, seed=3)
    wm = (w * S.mask_tile_greedy(w)).float()
    packed, packed_t = S._compress_pair(w, "greedy")
    zd = rms_rows(torch.randn(T, D, device=DEV))
    zh = (0.1 * torch.randn(T, H, device=DEV)).bfloat16()
    for got, want in ((S.cslt_linear(packed, zd, H), zd.float() @ wm.T),
                      (S.cslt_linear(packed_t, zh, D), zh.float() @ wm)):
        assert got.shape == want.shape and got.dtype == torch.bfloat16 and got.is_contiguous()
        assert rel(got, want) < 1e-2


def _bufs_for(bank, backend, monkeypatch):
    monkeypatch.setattr(S, "SPARSE24_BACKEND", backend)
    bufs = S.sparse24_buffers(bank.shape[0], H, D, torch.device(DEV))
    S.refresh_sparse24_(bank[:, 0], bank[:, 1], bufs)
    return bufs


def _run(backend, bank, x, g, slot, monkeypatch):
    bufs = _bufs_for(bank, backend, monkeypatch)
    W1, W2 = (bank[slot, i].detach().clone().requires_grad_() for i in (0, 1))
    xin = x.detach().clone().requires_grad_()
    out = S.Sparse24MLP(xin, W1, W2, *S.slot_operands(bufs, slot))
    return (out, *torch.autograd.grad(out, (xin, W1, W2), g))


def test_function_cusparselt_matches_dense_emulation(monkeypatch):
    bank = torch.stack([torch.stack([rand_w(H, D, std=0.5 * D ** -0.5, seed=10 + s), rand_w(H, D, seed=20 + s)])
                        for s in range(2)])
    T = 8192
    x = rms_rows(torch.randn(1, T, D, device=DEV))
    g = (1e-2 * torch.randn(1, T, D, device=DEV)).bfloat16()
    sparse = _run("cusparselt", bank, x, g, 1, monkeypatch)
    dense = _run("dense", bank, x, g, 1, monkeypatch)
    for name, a, b in zip(("out", "dx", "dW1", "dW2"), sparse, dense):
        r = rel(a, b)
        print(f"cusparselt vs dense emulation {name}: rel {r:.2e}")
        assert r < 1e-2, name


def test_compiled_and_graph_captured(monkeypatch):
    """Refresh + forward + backward inside torch.compile(fullgraph=True) and a captured CUDA graph, replayed after
    the bank moves: must match eager on the moved bank."""
    monkeypatch.setattr(S, "SPARSE24_BACKEND", "cusparselt")
    n, slot, T = 2, 1, 4096
    bank = torch.nn.Parameter(torch.stack([torch.stack([rand_w(H, D, seed=30 + s), rand_w(H, D, seed=40 + s)])
                                           for s in range(n)]))
    bufs = S.sparse24_buffers(n, H, D, torch.device(DEV))
    x = rms_rows(torch.randn(1, T, D, device=DEV)).requires_grad_()
    g = (1e-2 * torch.randn(1, T, D, device=DEV)).bfloat16()

    def mlp(x):
        return S.Sparse24MLP(x, bank[slot, 0], bank[slot, 1], *S.slot_operands(bufs, slot))

    cmlp = torch.compile(mlp, fullgraph=True)
    static = {}

    def step():
        S.refresh_sparse24_(bank.detach()[:, 0], bank.detach()[:, 1], bufs)
        out = cmlp(x)
        static["out"] = out
        static["grads"] = torch.autograd.grad(out, (x, bank), g)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # compile + warm up outside capture
            step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    with torch.no_grad():
        bank.mul_(-1.3).add_(0.01)
    graph.replay()
    torch.cuda.synchronize()
    S.refresh_sparse24_(bank.detach()[:, 0], bank.detach()[:, 1], bufs)
    out = mlp(x)
    dx, dbank = torch.autograd.grad(out, (x, bank), g)
    assert rel(static["out"], out) < 1e-3
    assert rel(static["grads"][0], dx) < 1e-3 and rel(static["grads"][1], dbank) < 1e-3
