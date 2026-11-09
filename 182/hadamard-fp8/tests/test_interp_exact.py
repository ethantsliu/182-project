"""The record's MLP / Hadamard Triton kernels under the Triton interpreter, byte for byte against a torch
emulation of each kernel's exact arithmetic (interpreter-only; CPU):

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/hadamard-fp8/tests/test_interp_exact.py -q

The emulation follows the kernels' rounding points: GEMM accumulation in fp32 per BLOCK_K tile, the bf16
rounding of pre / dpre, fp32 post (triton 3.6's tl.maximum promotes bf16 to fp32), x * (1 / scale) in
fp32, round-to-nearest-even fp8. The interpreter shim (182/tools/triton_cpu/shim.py) accumulates each
dot in fp64 and rounds once, as the emulation does, so a mismatch is an indexing / layout / flag bug, not
rounding -- except a handful of bytes where two fp64 summation orders round to different fp32 values
(allowed: < 1e-4 of bytes, each one fp8 ulp). GPU differences the interpreter cannot show: tl.sqrt is
approximate on the GPU, and tensor-core accumulation order differs.
"""
import importlib.util
import math
import os
import pathlib
import sys
import tempfile

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("interpreter-only (182/tools/triton_cpu/run.sh)", allow_module_level=True)

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from track_1_short.perf.hadamard_ref import rotate, signed_hadamard  # noqa: E402
from track_1_short.perf.kernels import hadamard as hd  # noqa: E402
from track_1_short.perf.kernels import mlp  # noqa: E402

E4, E5, BF = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16
LIM = {E4: 448.0, E5: 57344.0}
S = torch.tensor(1.0 / math.sqrt(128), dtype=torch.float32)  # the kernels' fp32 HADAMARD_SCALE
Q = signed_hadamard(128, seed=1234)


def inv(scale):
    return torch.tensor(1.0, dtype=torch.float32) / scale.float().reshape(())


def quant(v, inv_scale, fmt):
    return torch.clamp(v.float() * inv_scale, -LIM[fmt], LIM[fmt]).to(fmt)


def rot(t, dim):
    """The kernels' rotation of a bf16 tensor: fp64 dot with Q, rounded once to fp32, times fp32 1/sqrt(128)."""
    b = t.movedim(dim, -1).double()
    out = (b.reshape(*b.shape[:-1], -1, 128) @ Q.double()).reshape(b.shape).float() * S
    return out.movedim(-1, dim)


def assert_bytes(got, want, max_frac=1e-4):
    assert got.shape == want.shape and got.dtype == want.dtype, (got.shape, want.shape, got.dtype, want.dtype)
    g, w = got.contiguous().view(torch.uint8).int(), want.contiguous().view(torch.uint8).int()
    bad = g != w
    frac = bad.float().mean().item()
    assert frac <= max_frac, f"{frac:.2e} of bytes differ ({int(bad.sum())})"
    if bad.any():  # one ulp: adjacent codes of the same sign
        assert ((g[bad] - w[bad]).abs() == 1).all(), "a differing byte is more than one fp8 ulp away"


# ---- perf/kernels/hadamard.py ---------------------------------------------------------------------------

@pytest.mark.parametrize("emit_row", [False, True])
@pytest.mark.parametrize("rot_row", [False, True])
@pytest.mark.parametrize("rot_t", [False, True])
@pytest.mark.parametrize("record_amax", [False, True])
@pytest.mark.parametrize("e5m2", [False, True])
def test_hadamard_quant_dual_kernel_flags(emit_row, rot_row, rot_t, record_amax, e5m2):
    if rot_row and not emit_row:
        pytest.skip("ROT_ROW without EMIT_ROW is dead code")
    torch.manual_seed(10)
    M, N, pad = 256, 384, 64
    fmt = E5 if e5m2 else E4
    src_full = (0.5 * torch.randn(M, N + pad)).to(BF)
    src = src_full[:, :N]  # a strided source (stride_m = N + pad)
    row_s, t_s = torch.tensor([2.0 ** -6]), torch.tensor([0.0123])  # one power of 2, one not
    row, t = torch.empty(M, N, dtype=fmt), torch.empty(N, M, dtype=fmt)
    amax = torch.zeros(1)
    hd._hadamard_quant_dual_kernel[(M // 128, N // 128)](
        src, Q, row, t, row_s, t_s, amax, M, N, src.stride(0), EMIT_ROW=emit_row, ROT_ROW=rot_row, ROT_T=rot_t,
        RECORD_T_AMAX=record_amax, E5M2=e5m2, BLOCK=128)
    if emit_row:
        y = rot(src, 1) if rot_row else src.float()
        assert_bytes(row, quant(y, inv(row_s), fmt))
    yt = rot(src, 0).T if rot_t else src.float().T
    assert_bytes(t, quant(yt, inv(t_s), fmt))
    if record_amax:
        assert amax.item() == yt.abs().max().item()
    else:
        assert amax.item() == 0.0


def test_hadamard_dual_quant_and_t_quant_wrappers():
    torch.manual_seed(11)
    g = (1e-3 * torch.randn(512, 256)).to(BF)
    sc = torch.tensor(2.0 ** -6)
    row, t = hd.hadamard_dual_quant(g, Q, sc, True)
    assert_bytes(row, quant(rot(g, 1), inv(sc), E5))
    assert_bytes(t, quant(rot(g, 0).T, inv(sc), E5))
    x = torch.randn(512, 256).to(BF)
    amax = torch.full((1,), 0.5)  # a stale smaller amax is overwritten; atomic_max keeps the larger
    xt = hd.hadamard_t_quant(x, Q, torch.tensor(2.0 ** -4), amax)
    assert_bytes(xt, quant(rot(x, 0).T, inv(torch.tensor(2.0 ** -4)), E4))
    assert amax.item() == rot(x, 0).abs().max().item()


@pytest.mark.parametrize("along_h", [True, False])
def test_quantize_weights_rotated_kernel(along_h):
    torch.manual_seed(12)
    L, H, D = 2, 512, 256
    bank = (0.02 * torch.randn(L, 2, H, D)).to(BF)[:, 1]
    y = rot(bank, 1 if along_h else 2)
    scales = torch.tensor([0.0003, 0.00021])
    nt = hd.quantize_weights_rotated_ntiles(H, D)
    pam = torch.zeros(L, nt)
    out = torch.empty((L, D, H) if along_h else (L, H, D), dtype=E4)
    hd.quantize_weights_rotated(bank, Q, scales, pam, out, along_h=along_h, update_scales=False)
    want = y.transpose(1, 2) if along_h else y
    for layer in range(L):
        assert_bytes(out[layer], quant(want[layer], inv(scales[layer]), E4))
    # each 128x128 tile's amax, in tile order (h-major), then update_scales derives the next scale from them
    tiles = y.reshape(L, H // 128, 128, D // 128, 128).abs().amax((2, 4)).reshape(L, -1)
    assert torch.equal(pam, tiles)
    hd.quantize_weights_rotated(bank, Q, scales, pam, out, along_h=along_h, update_scales=True)
    assert torch.allclose(scales, tiles.amax(1) * (hd.ROTATED_WEIGHT_SCALE_HEADROOM / 448.0))


# ---- perf/kernels/mlp.py: linear_relu_square --------------------------------------------------------

BM, BN, BK = 128, 256, 128


def gemm_fp32_ktiled(a_f8, b_f8):
    """acc = sum over BLOCK_K tiles of (a_k @ b_k.T), each tile's dot exact then rounded into the fp32 acc."""
    acc = torch.zeros(a_f8.shape[0], b_f8.shape[0], dtype=torch.float32)
    for k in range(0, a_f8.shape[1], BK):
        acc = (a_f8[:, k:k + BK].double() @ b_f8[:, k:k + BK].double().T + acc.double()).float()
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


def emulate_forward(x_f8, w1_f8, dq, post_scale, hadamard):
    pre = (gemm_fp32_ktiled(x_f8, w1_f8) * dq.float()).to(BF)
    post = torch.clamp(pre.float(), min=0.0) ** 2      # fp32: tl.maximum promotes the bf16 pre
    ip = inv(post_scale)
    post_f8 = torch.clamp(post * ip, max=448.0).to(E4)
    amax = post.max()
    if hadamard:  # the proposed fix: rotate the bf16-rounded post
        r = rot(post.to(BF), 0)                         # [T, H], rotated along tokens
        post_t = quant(r, ip, E4).T.contiguous()
        amax = torch.maximum(amax, r.abs().max())
    else:
        post_t = post_f8.T.contiguous()
    return post_f8, post_t, amax


def emulate_backward(g_f8, w2_f8, dq, post_f8, post_scale, dpre_scale, hadamard):
    acc = gemm_fp32_ktiled(g_f8, w2_f8) * dq.float()
    c = (2.0 * acc * torch.sqrt(post_f8.float() * post_scale.float())).to(BF)
    idp = inv(dpre_scale)
    if hadamard:
        r, t = rot(c, 1), rot(c, 0)
        return quant(r, idp, E5), quant(t, idp, E5).T.contiguous(), torch.maximum(r.abs().max(), t.abs().max())
    q = quant(c.float(), idp, E5)
    return q, q.T.contiguous(), c.float().abs().max()


def _forward(mod, x, W1, x_f8, w1_f8, dq, post_scale, hq):
    amax = torch.zeros(3)
    _, _, post_f8, post_t = mod.linear_relu_square(x, W1, a_f8=x_f8, b_f8=w1_f8, dequant_scale_ptr=dq, emit_f8=True,
                                                   emit_t=True, post_scale=post_scale, post_amax=amax,
                                                   store_pre=False, store_post_bf=False, hadamard_q=hq)
    return post_f8, post_t, amax


def test_lrs_bf16_forward_record_path():
    x, W1, *_ = mlp_case()
    pre, post, _, _ = mlp.linear_relu_square(x, W1)
    want_pre = torch.zeros(x.shape[0], W1.shape[0])
    for k in range(0, x.shape[1], 64):  # BLOCK_K = 64 on the bf16 path
        want_pre = (x[:, k:k + 64].double() @ W1[:, k:k + 64].double().T + want_pre.double()).float()
    want_pre = want_pre.to(BF)
    assert torch.equal(pre, want_pre)
    assert torch.equal(post, (torch.clamp(want_pre.float(), min=0) ** 2).to(BF))


def test_lrs_fp8_forward_record_path():
    """HADAMARD off: the record's fp8 forward epilogue (post row + transposed copy + amax)."""
    x, W1, W2, g, x_f8, w1_f8, w1_s, *_ = mlp_case()
    dq = (w1_s * 2.0 ** -4).reshape(1)
    post_scale = torch.tensor([0.004])
    post_f8, post_t, amax = _forward(mlp, x, W1, x_f8, w1_f8, dq, post_scale, None)
    want_f8, want_t, want_amax = emulate_forward(x_f8, w1_f8, dq, post_scale, False)
    assert_bytes(post_f8, want_f8)
    assert torch.equal(post_t.view(torch.uint8), post_f8.T.contiguous().view(torch.uint8))
    assert amax[0].item() == want_amax.item() and amax[1:].eq(0).all()


@pytest.mark.parametrize("hadamard", [False, True])
def test_lrs_fp8_backward(hadamard):
    """The dpre epilogue (both e5m2 layouts + amax), plain (record) and rotated."""
    x, W1, W2, g, x_f8, w1_f8, w1_s, w2_f8, w2_s = mlp_case()
    post_scale = torch.tensor([0.004])
    post_f8, _, _ = emulate_forward(x_f8, w1_f8, (w1_s * 2.0 ** -4).reshape(1), post_scale, False)
    g_f8 = quant(g, inv(torch.tensor(2.0 ** -6)), E5)
    w2_row = w2_f8  # the operand layout is all the kernel sees; rotation of W2 is the caller's business
    dq = (w2_s * 2.0 ** -6).reshape(())
    dpre_scale = torch.tensor(3e-7)
    amax = torch.zeros(3)
    dpre_f8, dpre_t = mlp.linear_relu_square(g, W2, aux=post_f8, a_f8=g_f8, b_f8=w2_row, dequant_scale_ptr=dq,
                                             post_scale=post_scale, dpre_scale=dpre_scale, dpre_amax=amax,
                                             hadamard_q=Q if hadamard else None)
    want_row, want_t, want_amax = emulate_backward(g_f8, w2_row, dq, post_f8, post_scale, dpre_scale, hadamard)
    assert_bytes(dpre_f8, want_row)
    assert_bytes(dpre_t, want_t)
    assert amax[0].item() == want_amax.item()


def test_lrs_fp8_forward_hadamard_compiles():
    """HADAMARD forward as written: tl.dot(tl.trans(c0_post), Q) with c0_post fp32 (tl.maximum promotes the
    bf16 pre) and Q bf16 -- triton 3.5.x-3.8.x reject mixed dtypes at compile time, on the GPU too."""
    x, W1, W2, g, x_f8, w1_f8, w1_s, *_ = mlp_case()
    _forward(mlp, x, W1, x_f8, w1_f8, (w1_s * 2.0 ** -4).reshape(1), torch.tensor([0.004]), Q)


def _patched_mlp():
    """mlp.py with the one-line fix (rotate the bf16-rounded post), imported from a temp copy."""
    src = (ROOT / "track_1_short/perf/kernels/mlp.py").read_text()
    fixed = src.replace("tl.dot(tl.trans(c0_post), ", "tl.dot(tl.trans(c0_post.to(dtype)), ").replace(
        "tl.dot(tl.trans(c1_post), ", "tl.dot(tl.trans(c1_post.to(dtype)), ")
    assert fixed.count(".to(dtype)), ") == 2, "the forward rotation lines moved; update the patch"
    d = tempfile.mkdtemp()
    path = os.path.join(d, "mlp_hadamard_fix.py")
    with open(path, "w") as f:
        f.write(fixed)
    spec = importlib.util.spec_from_file_location("mlp_hadamard_fix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lrs_fp8_forward_hadamard_with_fix():
    """With the fix the rest of the rotated forward epilogue is right: row post unchanged (plain), post_t
    rotated along tokens, and the shared post amax covering both (finding (b) of round 1: the rotated amax
    sets the forward row's scale too)."""
    fixed = _patched_mlp()
    x, W1, W2, g, x_f8, w1_f8, w1_s, *_ = mlp_case()
    dq = (w1_s * 2.0 ** -4).reshape(1)
    _, _, a = emulate_forward(x_f8, w1_f8, dq, torch.tensor([1.0]), True)
    post_scale = (a * 1.03 / 448).reshape(1)  # the delayed-scale rule, from this step's amax: no saturation
    post_f8, post_t, amax = _forward(fixed, x, W1, x_f8, w1_f8, dq, post_scale, Q)
    want_f8, want_t, want_amax = emulate_forward(x_f8, w1_f8, dq, post_scale, True)
    assert_bytes(post_f8, want_f8)
    assert_bytes(post_t, want_t)
    assert amax[0].item() == want_amax.item()
    # against the pure-torch reference (perf/hadamard_ref.rotate) from the same fp8 operands: only the
    # e4m3 rounding of post_t is left (~2-3% relative)
    pre = (x_f8.float() * 2.0 ** -4) @ (w1_f8.float() * w1_s).T
    ref = rotate(torch.clamp(pre, min=0) ** 2, 0, Q).T
    assert ((post_t.float() * post_scale - ref).norm() / ref.norm()).item() < 0.04


def test_fused_mlp_hadamard_grads_with_fix(monkeypatch):
    """End to end with the fix: test_gpu_kernels.test_fused_mlp_fp8_grads (fp8 Function vs the fp32 torch MLP,
    T=1024, D=768, H=1024) with the patched Function, HADAMARD on, then off for comparison."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import test_gpu_kernels as tg
    fixed = _patched_mlp()
    errs = {}
    for hadamard in (True, False):
        x, W1, W2, g = tg._mlp_case()
        ref_out, ref_dx, ref_dW1, ref_dW2 = tg._torch_mlp_grads(x, W1, W2, g)
        args = tg._fp8_mlp_args(x, W1, W2, hadamard)
        for _ in range(2):
            xr, W1r, W2r = (t.detach().clone().requires_grad_() for t in (x, W1, W2))
            out = fixed.FusedLinearReLUSquareFunction.apply(xr, W1r, W2r, *args)
            out.backward(g)
            args[14] = (args[15].amax() * 1.25 / 57344).clamp(min=1e-12)
            args[15] = torch.zeros(1)
        rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
        errs[hadamard] = dict(out=rel(out, ref_out), dx=rel(xr.grad, ref_dx), dW1=rel(W1r.grad, ref_dW1),
                              dW2=rel(W2r.grad, ref_dW2))
    print("hadamard", errs[True], "plain", errs[False])
    assert all(e < 0.1 for e in errs[True].values()), errs
    assert errs[True]["out"] == errs[False]["out"] or abs(errs[True]["out"] - errs[False]["out"]) < 0.01
