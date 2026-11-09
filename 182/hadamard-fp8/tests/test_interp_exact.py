"""The record's MLP / Hadamard Triton kernels under the Triton interpreter, byte for byte against a torch
emulation of each kernel's exact arithmetic (interpreter-only; CPU):

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/hadamard-fp8/tests/test_interp_exact.py -q

test_interp_kernels.py checks the same kernels against perf/hadamard_ref.py with fp8-level tolerances;
this file pins the arithmetic. The emulation follows the kernels' rounding points: GEMM accumulation in
fp32 per BLOCK_K tile, the bf16 rounding of pre / dpre, fp32 post (triton 3.6's tl.maximum promotes bf16
to fp32; rotate_last casts it back to bf16 for the dot), x * (1 / scale) in fp32, round-to-nearest-even
fp8. The interpreter shim (182/tools/triton_cpu/shim.py) accumulates each dot in fp64 and rounds once, as
the emulation does, so a mismatch is an indexing / layout / flag bug, not rounding -- except a handful of
bytes where two fp64 summation orders round to different fp32 values (allowed: < 1e-4 of bytes, each one
fp8 ulp). GPU differences the interpreter cannot show: tl.sqrt is approximate on the GPU, and tensor-core
accumulation order differs.

The kernels are launched directly or through linear_relu_square: hadamard_t_quant registers a torch
CPU kernel (hadamard.py), so on CPU tensors the custom op runs that reference, not Triton.
"""
import math
import os
import pathlib
import sys

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("interpreter-only (182/tools/triton_cpu/run.sh)", allow_module_level=True)
pytest.importorskip("triton")

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from track_1_short.perf.hadamard_ref import signed_hadamard  # noqa: E402
from track_1_short.perf.kernels import hadamard as hd  # noqa: E402
from track_1_short.perf.kernels import mlp  # noqa: E402

E4, E5, BF = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16
LIM = {E4: 448.0, E5: 57344.0}
HBS = (32, 64, 128)
QS = {hb: signed_hadamard(hb, seed=1234) for hb in HBS}


def inv(scale):
    return torch.tensor(1.0, dtype=torch.float32) / scale.float().reshape(())


def quant(v, inv_scale, fmt):
    return torch.clamp(v.float() * inv_scale, -LIM[fmt], LIM[fmt]).to(fmt)


def rot(t, dim, hb):
    """The kernels' rotation of a bf16 tensor along dim in hb-blocks: fp64 dot with Q, rounded once to fp32,
    times the fp32 constant 1/sqrt(hb)."""
    b = t.to(BF).movedim(dim, -1).double()
    out = (b.reshape(*b.shape[:-1], -1, hb) @ QS[hb].double()).reshape(b.shape).float()
    return (out * torch.tensor(1.0 / math.sqrt(hb), dtype=torch.float32)).movedim(-1, dim)


def assert_bytes(got, want, max_frac=1e-4):
    assert got.shape == want.shape and got.dtype == want.dtype, (got.shape, want.shape, got.dtype, want.dtype)
    g, w = got.contiguous().view(torch.uint8).int(), want.contiguous().view(torch.uint8).int()
    bad = g != w
    frac = bad.float().mean().item()
    assert frac <= max_frac, f"{frac:.2e} of bytes differ ({int(bad.sum())})"
    if bad.any():  # one ulp: adjacent codes of the same sign
        assert ((g[bad] - w[bad]).abs() == 1).all(), "a differing byte is more than one fp8 ulp away"


# ---- perf/kernels/hadamard.py ---------------------------------------------------------------------------

@pytest.mark.parametrize("hb", HBS)
@pytest.mark.parametrize("emit_row,rot_row", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("rot_t", [False, True])
@pytest.mark.parametrize("record_amax", [False, True])
@pytest.mark.parametrize("e5m2", [False, True])
def test_hadamard_quant_dual_kernel_flags(hb, emit_row, rot_row, rot_t, record_amax, e5m2):
    torch.manual_seed(10)
    M, N, pad = 256, 384, 64
    fmt = E5 if e5m2 else E4
    src = (0.5 * torch.randn(M, N + pad)).to(BF)[:, :N]  # a strided source (stride_m = N + pad)
    row_s, t_s = torch.tensor([2.0 ** -6]), torch.tensor([0.0123])  # one power of 2, one not
    row, t = torch.empty(M, N, dtype=fmt), torch.empty(N, M, dtype=fmt)
    amax = torch.zeros(1)
    hd._hadamard_quant_dual_kernel[(M // 128, N // 128)](
        src, QS[hb], row, t, row_s, t_s, amax, M, N, src.stride(0), EMIT_ROW=emit_row, ROT_ROW=rot_row, ROT_T=rot_t,
        RECORD_T_AMAX=record_amax, E5M2=e5m2, BLOCK=128, HB=hb)
    if emit_row:
        assert_bytes(row, quant(rot(src, 1, hb) if rot_row else src.float(), inv(row_s), fmt))
    yt = rot(src, 0, hb).T if rot_t else src.float().T
    assert_bytes(t, quant(yt, inv(t_s), fmt))
    assert amax.item() == (yt.abs().max().item() if record_amax else 0.0)


def test_hadamard_wrappers_match_kernel():
    """hadamard_dual_quant (Triton) at the configured block, and hadamard_t_quant's CPU reference kernel
    against the Triton kernel it stands in for (they must agree for the CPU tests to mean anything)."""
    hb = hd.HADAMARD_BLOCK
    q = QS[hb]
    torch.manual_seed(11)
    g = (1e-3 * torch.randn(512, 256)).to(BF)
    sc = torch.tensor(2.0 ** -6)
    row, t = hd.hadamard_dual_quant(g, q, sc, True)
    assert_bytes(row, quant(rot(g, 1, hb), inv(sc), E5))
    assert_bytes(t, quant(rot(g, 0, hb).T, inv(sc), E5))
    x = torch.randn(512, 256).to(BF)
    xs = torch.tensor(0.0071)
    t_ref, amax_ref = hd.hadamard_t_quant(x, q, xs)  # CPU tensors: the registered torch reference
    t_k, amax_k = torch.empty(256, 512, dtype=E4), torch.zeros(1)
    hd._hadamard_quant_dual_kernel[(4, 2)](x, q, t_k, t_k, xs, xs, amax_k, 512, 256, 256, EMIT_ROW=False,
                                           ROT_ROW=False, ROT_T=True, RECORD_T_AMAX=True, E5M2=False, BLOCK=128, HB=hb)
    assert_bytes(t_k, quant(rot(x, 0, hb).T, inv(xs), E4))
    # the reference divides by the scale where the kernel multiplies by its reciprocal: rare 1-ulp ties
    assert_bytes(t_ref, t_k, max_frac=1e-3)
    assert torch.allclose(amax_ref, amax_k, rtol=1e-6)


@pytest.mark.parametrize("emit_row,rot_row,rot_t", [(True, True, False), (False, False, True)])
def test_hadamard_g_quant(emit_row, rot_row, rot_t):
    """The one-GEMM-rotated gradient quantize (mlp.hadamard_g_quant, HADAMARD_GEMMS=dpost or dw2 alone)."""
    hb = hd.HADAMARD_BLOCK
    torch.manual_seed(14)
    g = (1e-3 * torch.randn(256, 384)).to(BF)
    sc = torch.tensor(2.0 ** -6)
    row, t = mlp.hadamard_g_quant(g, QS[hb], sc, emit_row, rot_row, rot_t)
    if emit_row:
        assert_bytes(row, quant(rot(g, 1, hb) if rot_row else g.float(), inv(sc), E5))
    else:
        assert row.numel() == 0
    assert_bytes(t, quant(rot(g, 0, hb).T if rot_t else g.float().T, inv(sc), E5))


@pytest.mark.parametrize("hb", HBS)
@pytest.mark.parametrize("along_h", [True, False])
def test_quantize_weights_rotated_kernel(hb, along_h):
    torch.manual_seed(12)
    L, H, D = 2, 512, 256
    bank = (0.02 * torch.randn(L, 2, H, D)).to(BF)[:, 1]
    y = rot(bank, 1 if along_h else 2, hb)
    scales = torch.tensor([0.0003, 0.00021])
    nt = hd.quantize_weights_rotated_ntiles(H, D)
    pam = torch.zeros(L, nt)
    out = torch.empty((L, D, H) if along_h else (L, H, D), dtype=E4)
    hd._quantize_weights_rotated_kernel[(L, nt)](
        bank, QS[hb], out, scales, pam, bank.stride(0), bank.stride(1), bank.stride(2), H=H, D=D,
        num_tiles_d=D // 128, num_tiles=nt, ALONG_H=along_h, BLOCK=128, HB=hb)
    want = y.transpose(1, 2) if along_h else y
    for layer in range(L):
        assert_bytes(out[layer], quant(want[layer], inv(scales[layer]), E4))
    tiles = y.reshape(L, H // 128, 128, D // 128, 128).abs().amax((2, 4)).reshape(L, -1)  # h-major tile order
    assert torch.equal(pam, tiles)
    if hb == hd.HADAMARD_BLOCK:  # the wrapper: refresh the scales from the tile amaxes, then requantize
        hd.quantize_weights_rotated(bank, QS[hb], scales, pam, out, along_h=along_h, update_scales=True)
        assert torch.allclose(scales, tiles.amax(1) * (hd.ROTATED_WEIGHT_SCALE_HEADROOM / 448.0))


# ---- perf/kernels/mlp.py: linear_relu_square --------------------------------------------------------

BK = 128


def gemm_fp32_ktiled(a, b, bk=BK):
    """acc = sum over BLOCK_K tiles of (a_k @ b_k.T), each tile's dot exact then rounded into the fp32 acc."""
    acc = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32)
    for k in range(0, a.shape[1], bk):
        acc = (a[:, k:k + bk].double() @ b[:, k:k + bk].double().T + acc.double()).float()
    return acc


def mlp_case(T=256, D=256, H=512, seed=13):
    torch.manual_seed(seed)
    x = torch.randn(T, D)
    x = (x / x.pow(2).mean(1, keepdim=True).sqrt()).to(BF)
    W1, W2 = (0.05 * torch.randn(H, D)).to(BF), (0.05 * torch.randn(H, D)).to(BF)
    g = (1e-2 * torch.randn(T, D)).to(BF)
    x_f8 = quant(x, inv(torch.tensor(2.0 ** -4)), E4)
    w1_s, w2_s = W1.float().abs().max() / 448, W2.float().abs().max() / 448
    return x, W1, W2, g, x_f8, quant(W1, inv(w1_s), E4), w1_s, quant(W2, inv(w2_s), E4), w2_s


def emulate_forward(x_f8, w1_f8, dq, post_scale, post_t_scale, hb):
    pre = (gemm_fp32_ktiled(x_f8, w1_f8) * dq.float()).to(BF)
    post = torch.clamp(pre.float(), min=0.0) ** 2      # fp32: tl.maximum promotes the bf16 pre
    post_f8 = torch.clamp(post * inv(post_scale), max=448.0).to(E4)
    if hb:
        r = rot(post, 0, hb)                            # rotate_last casts post to bf16 first
        return post_f8, quant(r, inv(post_t_scale), E4).T.contiguous(), post.max(), r.abs().max()
    return post_f8, post_f8.T.contiguous(), post.max(), None


def _forward(x, W1, x_f8, w1_f8, dq, post_scale, hb, post_t_scale=None):
    amax, t_amax = torch.zeros(3), torch.zeros(3)
    kw = dict(hadamard_q=QS[hb], rot_t=True, post_t_scale=post_t_scale, post_t_amax=t_amax) if hb else {}
    _, _, post_f8, post_t = mlp.linear_relu_square(x, W1, a_f8=x_f8, b_f8=w1_f8, dequant_scale_ptr=dq, emit_f8=True,
                                                   emit_t=True, post_scale=post_scale, post_amax=amax,
                                                   store_pre=False, store_post_bf=False, **kw)
    return post_f8, post_t, amax, t_amax


def test_lrs_bf16_forward_record_path():
    x, W1, *_ = mlp_case()
    pre, post, _, _ = mlp.linear_relu_square(x, W1)
    want_pre = gemm_fp32_ktiled(x, W1, bk=64).to(BF)  # BLOCK_K = 64 on the bf16 path
    assert torch.equal(pre, want_pre)
    assert torch.equal(post, (torch.clamp(want_pre.float(), min=0) ** 2).to(BF))


@pytest.mark.parametrize("hb", (0,) + HBS)
def test_lrs_fp8_forward(hb):
    """hb=0: the record's fp8 forward epilogue (post row + transposed copy + amax). hb>0: post_t rotated along
    tokens at its own scale into its own amax; post_f8 and the post amax must stay the record's."""
    x, W1, W2, g, x_f8, w1_f8, w1_s, *_ = mlp_case()
    dq = (w1_s * 2.0 ** -4).reshape(1)
    post_scale = torch.tensor([0.004])
    _, _, _, ra = emulate_forward(x_f8, w1_f8, dq, post_scale, torch.tensor([1.0]), hb or 128)
    post_t_scale = (ra * 1.03 / 448).reshape(1)  # the delayed-scale rule from this step's amax: no saturation
    post_f8, post_t, amax, t_amax = _forward(x, W1, x_f8, w1_f8, dq, post_scale, hb, post_t_scale)
    want_f8, want_t, want_amax, want_t_amax = emulate_forward(x_f8, w1_f8, dq, post_scale, post_t_scale, hb)
    assert_bytes(post_f8, want_f8)
    assert_bytes(post_t, want_t)
    assert amax[0].item() == want_amax.item() and amax[1:].eq(0).all()
    if hb:
        assert t_amax[0].item() == want_t_amax.item() and t_amax[1:].eq(0).all()
    else:
        assert torch.equal(post_t.view(torch.uint8), post_f8.T.contiguous().view(torch.uint8))
        assert t_amax.eq(0).all()


def emulate_backward(g_f8, w2_f8, dq, post_f8, post_scale, dpre_scale, hb, rot_row, rot_t):
    acc = gemm_fp32_ktiled(g_f8, w2_f8) * dq.float()
    c = (2.0 * acc * torch.sqrt(post_f8.float() * post_scale.float())).to(BF)
    idp = inv(dpre_scale)
    plain = quant(c.float(), idp, E5)
    amaxes = [] if (rot_row and rot_t) else [c.float().abs().max()]
    if rot_row:
        r = rot(c, 1, hb)
        row = quant(r, idp, E5)
        amaxes.append(r.abs().max())
    else:
        row = plain
    if rot_t:
        t = rot(c, 0, hb)
        tt = quant(t, idp, E5).T.contiguous()
        amaxes.append(t.abs().max())
    else:
        tt = plain.T.contiguous()
    return row, tt, max(a.item() for a in amaxes)


@pytest.mark.parametrize("hb,rot_row,rot_t", [(0, False, False)] + [(hb, r, t) for hb in HBS
                                                                     for r, t in ((1, 1), (1, 0), (0, 1))])
def test_lrs_fp8_backward(hb, rot_row, rot_t):
    """The dpre epilogue (both e5m2 layouts + amax): the record's (hb=0) and every rotation subset."""
    x, W1, W2, g, x_f8, w1_f8, w1_s, w2_f8, w2_s = mlp_case()
    post_scale = torch.tensor([0.004])
    post_f8, *_ = emulate_forward(x_f8, w1_f8, (w1_s * 2.0 ** -4).reshape(1), post_scale, None, 0)
    g_f8 = quant(g, inv(torch.tensor(2.0 ** -6)), E5)
    dq = (w2_s * 2.0 ** -6).reshape(())
    dpre_scale = torch.tensor(3e-7)
    amax = torch.zeros(3)
    kw = dict(hadamard_q=QS[hb], rot_row=bool(rot_row), rot_t=bool(rot_t)) if hb else {}
    dpre_f8, dpre_t = mlp.linear_relu_square(g, W2, aux=post_f8, a_f8=g_f8, b_f8=w2_f8, dequant_scale_ptr=dq,
                                             post_scale=post_scale, dpre_scale=dpre_scale, dpre_amax=amax, **kw)
    want_row, want_t, want_amax = emulate_backward(g_f8, w2_f8, dq, post_f8, post_scale, dpre_scale, hb,
                                                   rot_row, rot_t)
    assert_bytes(dpre_f8, want_row)
    assert_bytes(dpre_t, want_t)
    assert amax[0].item() == want_amax and amax[1:].eq(0).all()
