"""FP8_ATTN_O on a GPU (skipped without CUDA): the compiled kernels and cuBLAS fp8 GEMMs at record shapes against
the fp8-emulation reference, the o_w cache bytes, and the torch.compile + CUDA-graph replay against eager.
182/gpu_session.sh's tests phase runs this file on the H100s (the interpreter files skip there):

    python -m pytest 182/fp8-attn-o/tests -q -rs
"""
import pathlib
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU (H100: 182/gpu_session.sh tests phase)", allow_module_level=True)
pytest.importorskip("triton")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpu_sanity  # noqa: E402  (182/fp8-attn-o/gpu_sanity.py)
from track_1_short.perf import attn_o_scales as S  # noqa: E402
from track_1_short.perf.kernels import attn_o  # noqa: E402
from track_1_short.perf.kernels.fp8_attention_quant import AMAX_PARTS  # noqa: E402

DEV = torch.device("cuda")
E4, E5 = S.E4M3, S.E5M2


def bytes_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("T", [16384, 49152])
@pytest.mark.parametrize("K", [768, 384])
@pytest.mark.parametrize("fmt", [E4, E5])
def test_dual_quant_bytes_and_amax(T, K, fmt):
    torch.manual_seed(T + K)
    x = (0.3 * torch.randn(T, K, device=DEV)).bfloat16()
    s = torch.tensor(S.ATTN_O_Y_SCALE if fmt == E4 else S.ATTN_O_G_SCALE, device=DEV)
    amax = torch.zeros(1, device=DEV)
    row, t = attn_o.fp8_dual_quant(x, s, fmt, amax)
    want = S.quant(x, s.item(), fmt)
    assert bytes_equal(row, want) and bytes_equal(t, want.T)
    assert amax.item() == x.float().abs().max().item()


@pytest.mark.parametrize("exact", [False, True])
def test_weight_cache_bytes(exact):
    torch.manual_seed(1)
    L = 7
    vo = (0.03 * torch.randn(16, 768, 768, device=DEV)).bfloat16()
    scales = torch.full((L,), S.ATTN_O_W_SCALE, device=DEV)
    pamax = torch.zeros(L, AMAX_PARTS, device=DEV)
    row = torch.empty(L, 768, 768, device=DEV, dtype=E4)
    col = torch.empty_like(row)
    attn_o.quantize_attn_o_weights(vo[1:2 * L:2], scales, pamax, row, col, exact=exact, record_amax=True)
    for j in range(L):
        w = vo[2 * j + 1]
        s = (w.float().abs().max() / 448).item() if exact else S.ATTN_O_W_SCALE
        assert scales[j].item() == pytest.approx(s, rel=1e-6)
        want = torch.clamp(w.float() / scales[j], -448, 448).to(E4)
        assert bytes_equal(row[j], want) and bytes_equal(col[j], want.T)


@pytest.mark.parametrize("T,K", [(16384, 768), (49152, 384)])
@pytest.mark.parametrize("fuse", [True, False])
def test_function_matches_reference_at_record_shapes(T, K, fuse):
    torch.manual_seed(T)
    y = (0.2 * torch.randn(T, K, device=DEV)).bfloat16()
    w = ((torch.rand(768, K, device=DEV) * 2 - 1) * S.INIT_BOUND).bfloat16()
    g = (1e-3 * torch.randn(T, 768, device=DEV)).bfloat16()
    o = torch.tensor(0.9, device=DEV)
    wq = S.quant(w, S.ATTN_O_W_SCALE, E4)
    t = lambda v: torch.tensor(float(v), device=DEV)
    yr, wr, orr = y.clone().requires_grad_(), w.clone().requires_grad_(), o.clone().requires_grad_()
    out = attn_o.AttnOFP8(yr, wr, orr, wq, wq.T.contiguous(), t(S.ATTN_O_W_SCALE), t(S.ATTN_O_Y_SCALE),
                          t(S.ATTN_O_G_SCALE), None, None, fuse, fuse)
    out.backward(g)
    ref = S.attn_o_reference(y.cpu(), w.cpu(), o.cpu(), g.cpu())
    bf = S.attn_o_bf16_reference(y, w, o, g)
    for k, got in (("out", out), ("dy", yr.grad), ("dw", wr.grad)):
        # cuBLAS fp8: fast accumulation on the forward, a different fp32 summation order than the CPU reference
        assert S.rel_err(got.cpu(), ref[k]) < 5e-3, k
        assert S.rel_err(got, bf[k]) < (0.045 if k == "out" else 0.07), k


@pytest.mark.parametrize("mode", gpu_sanity.MODES)
def test_compiled_cuda_graph_matches_eager(mode):
    res = gpu_sanity.check_compiled_graph(mode)
    bad = {k: v for k, v in res.items() if not v[0]}
    assert not bad, bad
