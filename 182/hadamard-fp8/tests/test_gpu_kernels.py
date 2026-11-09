"""GPU tests (H100; skipped without CUDA): the HADAMARD_FP8 Triton kernels against perf/hadamard_ref.py,
and the FP8_OFF bf16 paths against autograd references.

    HADAMARD_FP8=1 python -m pytest 182/hadamard-fp8/tests/test_gpu_kernels.py -q -x

(HADAMARD_FP8 / HADAMARD_GEMMS / HADAMARD_BLOCK are read at import: the Function rotates the GEMMs
HADAMARD_GEMMS selects, with HADAMARD_BLOCK-sized Q. Run once per setting you care about.)
Importing the record's kernels needs the CUDA toolchain (the CE kernel compiles at import).
"""
import pathlib
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from track_1_short.config import HADAMARD_BLOCK, HADAMARD_GEMMS  # noqa: E402
from track_1_short.perf.hadamard_ref import FP8_MAX, rotate, signed_hadamard  # noqa: E402
from track_1_short.perf.kernels.hadamard import (  # noqa: E402
    hadamard_dual_quant, hadamard_t_quant, quantize_weights_rotated, quantize_weights_rotated_ntiles)
from track_1_short.perf.kernels.mlp import (  # noqa: E402
    BF16LinearReLUSquareFunction, FusedLinearReLUSquareFunction, linear_relu_square, quantize_dual_layout_fused,
    quantize_mlp_weights_dual, quantize_weights_dual_ntiles)

DEV = torch.device("cuda")
E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
Q = signed_hadamard(HADAMARD_BLOCK, seed=1234, device=DEV)


def ref_quant(t, scale, fmt):
    lim = FP8_MAX[fmt]
    return torch.clamp(t.float() / scale, -lim, lim).to(fmt)


def assert_fp8_close(got, want, scale, max_mismatch=2e-3):
    """fp8 bytes may differ by one ulp where the fp32 accumulation order rounds across a tie."""
    assert got.shape == want.shape and got.dtype == want.dtype
    g, w = got.float() * scale, want.float() * scale
    mismatch = (got.view(torch.uint8) != want.view(torch.uint8)).float().mean().item()
    assert mismatch < max_mismatch, f"{mismatch:.2e} of bytes differ"
    rel = ((g - w).norm() / w.norm().clamp(min=1e-30)).item()
    assert rel < 2e-2, rel


def test_hadamard_dual_quant_matches_reference():
    torch.manual_seed(0)
    g = (1e-3 * torch.randn(1024, 768, device=DEV)).bfloat16()
    scale = torch.tensor(2.0 ** -6, device=DEV)
    row, t = hadamard_dual_quant(g, Q, scale, True)
    assert_fp8_close(row, ref_quant(rotate(g, 1, Q), 2.0 ** -6, E5), 2.0 ** -6)
    assert_fp8_close(t, ref_quant(rotate(g, 0, Q).T.contiguous(), 2.0 ** -6, E5), 2.0 ** -6)


def test_hadamard_t_quant_matches_reference_and_records_amax():
    torch.manual_seed(1)
    x = torch.randn(1024, 768, device=DEV).bfloat16()
    scale = torch.tensor(2.0 ** -4, device=DEV)
    amax = torch.zeros(1, device=DEV)
    t = hadamard_t_quant(x, Q, scale, amax)
    ref = rotate(x, 0, Q)
    assert_fp8_close(t, ref_quant(ref.T.contiguous(), 2.0 ** -4, E4), 2.0 ** -4)
    assert abs(amax.item() - ref.abs().max().item()) <= 1e-3 * ref.abs().max().item()


@pytest.mark.parametrize("along_h", [True, False])
def test_quantize_weights_rotated(along_h):
    torch.manual_seed(2)
    L, H, D = 3, 1024, 768
    bank = (0.02 * torch.randn(L, 2, H, D, device=DEV)).bfloat16()[:, 1]  # a strided bank slice, as in the model
    ref = rotate(bank, 1 if along_h else 2, Q)
    scales = (ref.reshape(L, -1).abs().amax(1) / 448).float()
    pam = torch.zeros(L, quantize_weights_rotated_ntiles(H, D), device=DEV)
    out = torch.empty((L, D, H) if along_h else (L, H, D), device=DEV, dtype=E4)
    quantize_weights_rotated(bank, Q, scales, pam, out, along_h=along_h, update_scales=False)
    want = ref.transpose(1, 2).contiguous() if along_h else ref
    for layer in range(L):
        assert_fp8_close(out[layer], ref_quant(want[layer], scales[layer], E4), scales[layer].item())
    assert torch.allclose(pam.amax(1), ref.reshape(L, -1).abs().amax(1), rtol=1e-3)


def _mlp_case(T=1024, D=768, H=1024, seed=3):
    torch.manual_seed(seed)
    x = torch.randn(T, D, device=DEV)
    x = (x / x.pow(2).mean(1, keepdim=True).sqrt()).bfloat16()
    W1 = (0.03 * torch.randn(H, D, device=DEV)).bfloat16()
    W2 = (0.03 * torch.randn(H, D, device=DEV)).bfloat16()
    g = (1e-2 * torch.randn(T, D, device=DEV)).bfloat16()
    return x, W1, W2, g


def _torch_mlp_grads(x, W1, W2, g):
    xr, W1r, W2r = (t.detach().float().requires_grad_() for t in (x, W1, W2))
    out = torch.relu(xr @ W1r.T).square() @ W2r
    out.backward(g.float())
    return out.detach(), xr.grad, W1r.grad, W2r.grad


def _fp8_mlp_args(x, W1, W2, hadamard):
    """The Function's 17 fp8 args (+ p_fold=None, post_t_scale, post_t_amax under hadamard), built the way
    GPT._mlp_fp8_args / quantize_mlp_fp8 build them for HADAMARD_GEMMS."""
    gm = HADAMARD_GEMMS if hadamard else frozenset()
    H, D = W1.shape
    e4m3 = E4
    bank = torch.stack((W1, W2))[None]  # [1, 2, H, D]
    up, down = bank[:, 0], bank[:, 1]
    col_major = lambda: torch.empty(1, D, H, dtype=e4m3, device=DEV).transpose(1, 2)
    up_row, up_col, down_col = torch.empty(1, H, D, dtype=e4m3, device=DEV), col_major(), col_major()
    down_row = torch.empty(1, H, D, dtype=e4m3, device=DEV)
    up_s = (W1.abs().max() / 448).float().reshape(1)
    down_s = (W2.abs().max() / 448).float().reshape(1)
    nt = quantize_weights_dual_ntiles(H, D)
    quantize_mlp_weights_dual(up, up_s, torch.zeros(1, nt, device=DEV), up_row, up_col.transpose(1, 2), False)
    quantize_mlp_weights_dual(down, down_s, torch.zeros(1, nt, device=DEV), down_row, down_col.transpose(1, 2), False)
    x_scale, g_scale = torch.tensor(2.0 ** -4, device=DEV), torch.tensor(2.0 ** -6, device=DEV)
    w1_col_s, w2_row_s = up_s[0], down_s[0]
    nr = quantize_weights_rotated_ntiles(H, D)
    if "dx" in gm:
        w1_col_s = (rotate(up, 1, Q).abs().max() / 448).float().reshape(1)
        quantize_weights_rotated(up, Q, w1_col_s, torch.zeros(1, nr, device=DEV), up_col.transpose(1, 2), True, False)
        w1_col_s = w1_col_s[0]
    if "dpost" in gm:
        w2_row_s = (rotate(down, 2, Q).abs().max() / 448).float().reshape(1)
        quantize_weights_rotated(down, Q, w2_row_s, torch.zeros(1, nr, device=DEV), down_row, False, False)
        w2_row_s = w2_row_s[0]
    if "dw1" in gm:
        lim = 448.0
        x_row = torch.clamp(x.float() / x_scale, -lim, lim).to(e4m3)
        xt_scale = (rotate(x, 0, Q).abs().max() * 1.1 / 448).float()
        x_t = hadamard_t_quant(x, Q, xt_scale, torch.zeros(1, device=DEV))
    else:
        x_row, x_t = quantize_dual_layout_fused(x, x_scale, E4)
        xt_scale = x_scale
    pre = (x.float() @ W1.float().T).bfloat16()
    post = torch.relu(pre).float().square()
    post_s = (post.abs().max() * 1.03 / 448).reshape(1)
    dpre_s = torch.tensor(1e-3 * 1.25 / 57344, device=DEV)  # refined below by one priming call
    args = [up_row[0], (up_s * x_scale).reshape(1), x_row, down_col[0], down_s[0], post_s[0],
            torch.zeros(1, device=DEV), down_row[0], (w2_row_s * g_scale), x_t, xt_scale, up_col[0], w1_col_s,
            g_scale, dpre_s, torch.zeros(1, device=DEV), Q if hadamard else None]
    if hadamard:
        post_t_s = (rotate(post.bfloat16(), 0, Q).abs().max() * 1.03 / 448).reshape(1)[0]
        args += [None, post_t_s, torch.zeros(1, device=DEV)]
    return args


@pytest.mark.parametrize("hadamard", [False, True])
def test_fused_mlp_fp8_grads(hadamard):
    """The fp8 Function (with or without rotation) against the fp32 torch MLP: the rotated path must be at
    least about as accurate as the record's."""
    x, W1, W2, g = _mlp_case()
    ref_out, ref_dx, ref_dW1, ref_dW2 = _torch_mlp_grads(x, W1, W2, g)
    args = _fp8_mlp_args(x, W1, W2, hadamard)
    for _ in range(2):  # the first call records the dpre amax; set the delayed scale from it, as a refresh would
        xr, W1r, W2r = (t.detach().clone().requires_grad_() for t in (x, W1, W2))
        out = FusedLinearReLUSquareFunction.apply(xr, W1r, W2r, *args)
        out.backward(g)
        args[14] = (args[15].amax() * 1.25 / 57344).clamp(min=1e-12)
        args[15] = torch.zeros(1, device=DEV)
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
    errs = dict(out=rel(out, ref_out), dx=rel(xr.grad, ref_dx), dW1=rel(W1r.grad, ref_dW1), dW2=rel(W2r.grad, ref_dW2))
    print("hadamard" if hadamard else "plain", errs)
    assert all(e < 0.1 for e in errs.values()), errs


def test_linear_relu_square_hadamard_epilogues():
    """The forward's rotated post_t against the reference, at its own scale; post_f8 and the post amax stay
    bit-identical to the unrotated kernel's (risk b)."""
    x, W1, W2, g = _mlp_case(seed=4)
    args = _fp8_mlp_args(x, W1, W2, hadamard=False)
    post_scale = args[5]
    post = torch.relu((x.float() @ W1.float().T).bfloat16()).float().square().bfloat16()
    post_t_scale = (rotate(post, 0, Q).abs().max() * 1.03 / 448).reshape(1)
    outs = []
    for rot in (False, True):
        amax, t_amax = torch.zeros(1, device=DEV), torch.zeros(1, device=DEV)
        kw = dict(hadamard_q=Q, rot_t=True, post_t_scale=post_t_scale, post_t_amax=t_amax) if rot else {}
        _, _, post_f8, post_t = linear_relu_square(x, W1, a_f8=args[2], b_f8=args[0], dequant_scale_ptr=args[1],
                                                   emit_f8=True, emit_t=True, post_scale=post_scale,
                                                   post_amax=amax, store_pre=False, store_post_bf=False, **kw)
        outs.append((post_f8, amax, post_t, t_amax))
    assert torch.equal(outs[0][0].view(torch.uint8), outs[1][0].view(torch.uint8))
    assert torch.equal(outs[0][1], outs[1][1])
    # the fp8 forward GEMM's own rounding makes post differ slightly: compare dequantized, loosely
    got = outs[1][2].float() * post_t_scale
    want = rotate(post, 0, Q).T
    assert ((got - want).norm() / want.norm()).item() < 0.08
    assert abs(outs[1][3].item() - want.abs().max().item()) < 0.02 * want.abs().max().item()


def test_bf16_mlp_matches_torch():
    x, W1, W2, g = _mlp_case(seed=5)
    ref_out, ref_dx, ref_dW1, ref_dW2 = _torch_mlp_grads(x, W1, W2, g)
    xr, W1r, W2r = (t.detach().clone().requires_grad_() for t in (x, W1, W2))
    out = BF16LinearReLUSquareFunction.apply(xr, W1r, W2r)
    out.backward(g)
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
    for a, b in ((out, ref_out), (xr.grad, ref_dx), (W1r.grad, ref_dW1), (W2r.grad, ref_dW2)):
        assert rel(a, b) < 2e-2


def test_bf16_qkv_matches_fp8_qkv():
    """FP8_OFF=qkv wiring: PackedBF16QKV and PackedFP8QKV agree to fp8 accuracy (non-paired, no key offset)."""
    from track_1_short.perf.kernels.fp8_attention_quant import quantize_dual_layout_packed_batched
    from track_1_short.perf.kernels.qkv_rope import PackedBF16QKV, PackedFP8QKV
    torch.manual_seed(6)
    T, D, Hh, dq, dv = 1024, 768, 6, 64, 128
    x = torch.randn(T, D, device=DEV)
    x = (x / x.pow(2).mean(1, keepdim=True).sqrt()).bfloat16()
    qk_w = (0.03 * torch.randn(2 * Hh * dq, D, device=DEV)).bfloat16()
    v_w = (0.03 * torch.randn(Hh * dv, D, device=DEV)).bfloat16()
    lam = torch.tensor(0.9, device=DEV)
    theta = torch.arange(T, device=DEV)[:, None] * torch.linspace(0, 1, dq, device=DEV)[None]
    f1, f2 = theta.cos().bfloat16(), theta.sin().bfloat16()
    f2[:, 1::2] *= -1
    rows, cols = 2 * Hh * dq + Hh * dv, D
    row = torch.empty(1, rows, cols, dtype=E4, device=DEV)
    col = torch.empty(1, cols, rows, dtype=E4, device=DEV)
    scales, pam = torch.ones(1, device=DEV), torch.zeros(1, 32, device=DEV)
    quantize_dual_layout_packed_batched(qk_w[None], v_w[None], scales, pam, row, col)
    xs, gs = torch.tensor([8.0 / 448], device=DEV), torch.tensor([1.0 / 448], device=DEV)
    x_f8, x_f8_t = quantize_dual_layout_fused(x, xs, E4)
    gq = (1e-1 * torch.randn(T, Hh, dq, device=DEV)).bfloat16()
    gk = (1e-1 * torch.randn(T, Hh, dq, device=DEV)).bfloat16()
    gv = (1e-1 * torch.randn(T, Hh, dv, device=DEV)).bfloat16()
    outs = []
    for fp8 in (True, False):
        xr, qr, vr, lr = (t.detach().clone().requires_grad_() for t in (x, qk_w, v_w, lam))
        if fp8:
            q, k, v = PackedFP8QKV(xr, qr, vr, row[0], col[0], scales[0:1], lr, xs, gs, f1, f2, Hh, False, False, 64,
                                   x_f8, x_f8_t)
        else:
            q, k, v = PackedBF16QKV(xr, qr, vr, lr, f1, f2, Hh, False, False, 64)
        torch.autograd.backward((q, k, v), (gq, gk, gv))
        outs.append([q, k, v, xr.grad, qr.grad, vr.grad, lr.grad])
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()
    for a, b in zip(*outs):
        assert rel(a, b) < 0.1


def test_bf16_loss_close_to_fp8_kernel():
    from track_1_short.perf.bf16_loss import bf16_training_loss
    from track_1_short.perf.kernels.cross_entropy import FusedSoftcappedCrossEntropy, quantize_lm_head_dual, \
        lm_head_inverse_scale
    torch.manual_seed(7)
    n, D, V = 2048, 768, 50304
    x = torch.randn(n, D, device=DEV)
    x = (x / x.pow(2).mean(1, keepdim=True).sqrt()).bfloat16()
    W = (0.02 * torch.randn(D, V, device=DEV)).bfloat16()
    targets = torch.randint(0, 50257, (n,), device=DEV)
    prefix = torch.full((n,), -1, device=DEV, dtype=torch.int64)
    mtp, pw = torch.tensor([1.0, 0.5, 0.25], device=DEV), torch.tensor(0.0, device=DEV)
    x_s, w_s, grad_s = 100 / 448, 2.0 / 448, (0.75 / 8) / 448
    row = torch.empty(D, V, dtype=E4, device=DEV)
    col = torch.empty_strided((D, V), (1, D), dtype=E4, device=DEV)
    quantize_lm_head_dual(W, row, col, lm_head_inverse_scale(w_s, DEV))
    res = []
    for fp8 in (True, False):
        xr, Wr = x.clone().requires_grad_(), W.clone().requires_grad_()
        if fp8:
            loss = FusedSoftcappedCrossEntropy.apply(xr, targets, mtp, prefix, pw, Wr, col, row, x_s, w_s, grad_s)
        else:
            loss = bf16_training_loss(xr, Wr, targets, mtp, prefix, pw)
        loss.sum().backward()
        res.append((loss.detach(), xr.grad.float(), Wr.grad.float()))
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    assert rel(res[1][0], res[0][0]) < 1e-2
    assert rel(res[1][1], res[0][1]) < 0.1 and rel(res[1][2], res[0][2]) < 0.1
