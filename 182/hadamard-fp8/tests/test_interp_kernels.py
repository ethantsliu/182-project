"""Triton-interpreter tests (CPU, no GPU): the HADAMARD_FP8 kernels and the fp8 MLP Function under
HADAMARD_GEMMS / HADAMARD_BLOCK, against perf/hadamard_ref.py. Skipped unless triton is importable and
TRITON_INTERPRET=1, e.g. in the triton-interp docker image:

    docker run --rm -e TRITON_INTERPRET=1 -v $PWD:/repo -w /repo triton-interp \
        python -m pytest 182/hadamard-fp8/tests/test_interp_kernels.py -q

Interpreter-tested is not GPU-tested: no layouts, shared memory, register pressure or wgmma shape rules
are exercised, only the kernels' arithmetic and indexing. Triton 3.5.1 interpreter bugs patched here:
bf16 tl.dot multiplies the raw uint16 bits; tl.range over a program_id scalar fails under numpy 2; its own
fp32 -> fp8 / bf16 and fp8 -> fp32 conversions are replaced by torch's (it truncates bf16, and drops fp8 values
just below a power of two by a binade).
fp8 bytes are still compared loosely: a different fp32 summation order can flip a rounding tie.
"""
import itertools
import os
import pathlib
import subprocess
import sys
import types

import pytest

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs TRITON_INTERPRET=1 (the triton-interp docker image)", allow_module_level=True)
triton = pytest.importorskip("triton")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def patch_interpreter():
    import triton.language as tl
    import triton.runtime.interpreter as ti
    if getattr(ti, "_hd_patched", False):
        return
    orig_dot = ti.InterpreterBuilder.create_dot

    torch_fp8 = {tl.float8e4nv: torch.float8_e4m3fn, tl.float8e5: torch.float8_e5m2}

    def up(h):
        if h.dtype == tl.bfloat16:
            return (h.data.astype(np.uint32) << 16).view(np.float32)
        if h.dtype in torch_fp8:
            return torch.from_numpy(np.ascontiguousarray(h.data)).view(torch_fp8[h.dtype]).float().numpy()
        return h.data.astype(np.float32)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if a.dtype == tl.bfloat16 or b.dtype == tl.bfloat16 or a.dtype in torch_fp8 or b.dtype in torch_fp8:
            return ti.TensorHandle(np.matmul(up(a), up(b), dtype=np.float32) + d.data, d.dtype.scalar)
        return orig_dot(self, a, b, d, input_precision, max_num_imprecise_acc)

    orig_patch = ti._patch_lang_tensor

    def patch_lang_tensor(tensor):
        orig_patch(tensor)
        tensor.__index__ = lambda self: int(np.asarray(self.handle.data).reshape(-1)[0])

    orig_fp_to_fp = ti.InterpreterBuilder.create_fp_to_fp

    def create_fp_to_fp(self, src, dst_type, rounding_mode):
        # fp32 -> fp8 through torch (round to nearest even, as the GPU cvt.rn): the interpreter's own
        # conversion drops values just below a power of two by a whole binade (0.47 -> 0.25 in e5m2).
        if src.dtype.scalar == tl.float32 and dst_type.scalar in torch_fp8:
            data = torch.from_numpy(np.ascontiguousarray(src.data)).to(torch_fp8[dst_type.scalar]).view(torch.uint8).numpy()
            return ti.TensorHandle(data, dst_type.scalar)
        return orig_fp_to_fp(self, src, dst_type, rounding_mode)

    orig_cast = ti.InterpreterBuilder.cast_impl

    def cast_impl(self, src, dst_type):
        # fp32 -> bf16 through torch (round to nearest even); the interpreter's conversion truncates.
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            data = torch.from_numpy(np.ascontiguousarray(src.data)).bfloat16().view(torch.int16).numpy().view(np.uint16)
            return ti.TensorHandle(data, dst_type.scalar)
        return orig_cast(self, src, dst_type)

    ti.InterpreterBuilder.cast_impl = cast_impl
    ti.InterpreterBuilder.create_dot = create_dot
    ti.InterpreterBuilder.create_fp_to_fp = create_fp_to_fp
    ti._patch_lang_tensor = patch_lang_tensor
    ti._hd_patched = True
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(multi_processor_count=2)


patch_interpreter()

from track_1_short.perf.hadamard_ref import FP8_MAX, rotate, signed_hadamard  # noqa: E402
from track_1_short.perf.kernels import hadamard as hk  # noqa: E402
from track_1_short.perf.kernels.mlp import linear_relu_square  # noqa: E402

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2


def ref_quant(t, scale, fmt):
    lim = FP8_MAX[fmt]
    return torch.clamp(t.float() / scale, -lim, lim).to(fmt)


def assert_fp8_close(got, want, scale, max_mismatch=0.02):
    assert got.shape == want.shape and got.dtype == want.dtype
    mismatch = (got.view(torch.uint8) != want.view(torch.uint8)).float().mean().item()
    g, w = got.float() * scale, want.float() * scale
    rel = ((g - w).norm() / w.norm().clamp(min=1e-30)).item()
    assert mismatch < max_mismatch and rel < 1e-2, (mismatch, rel)


# -----------------------------------------------------------------------------
# hadamard.py kernels

@pytest.mark.parametrize("hb", [32, 64, 128])
@pytest.mark.parametrize("emit_row,rot_row,rot_t", [(True, True, True), (True, True, False), (False, False, True),
                                                    (True, False, True)])
def test_dual_quant_kernel(hb, emit_row, rot_row, rot_t):
    torch.manual_seed(0)
    M, N = 256, 128
    src = (1e-2 * torch.randn(M, N)).bfloat16()
    q = signed_hadamard(hb, 1234)
    scale = torch.tensor([2.0 ** -6])
    amax = torch.zeros(1)
    row, t = torch.empty(M, N, dtype=E5), torch.empty(N, M, dtype=E5)
    hk._hadamard_quant_dual_kernel[(M // 128, N // 128)](
        src, q, row, t, scale, scale, amax, M, N, src.stride(0), EMIT_ROW=emit_row, ROT_ROW=rot_row, ROT_T=rot_t,
        RECORD_T_AMAX=True, E5M2=True, BLOCK=128, HB=hb)
    want_t = rotate(src, 0, q) if rot_t else src.float()
    assert_fp8_close(t, ref_quant(want_t.T.contiguous(), 2.0 ** -6, E5), 2.0 ** -6)
    assert abs(amax.item() - want_t.abs().max().item()) <= 1e-5 * want_t.abs().max().item()
    if emit_row:
        want_row = rotate(src, 1, q) if rot_row else src.float()
        assert_fp8_close(row, ref_quant(want_row, 2.0 ** -6, E5), 2.0 ** -6)


@pytest.mark.parametrize("hb", [32, 128])
@pytest.mark.parametrize("along_h", [True, False])
def test_quantize_weights_rotated_kernel(hb, along_h):
    torch.manual_seed(1)
    L, H, D = 2, 256, 128
    bank = (0.02 * torch.randn(L, 2, H, D)).bfloat16()[:, 1]
    q = signed_hadamard(hb, 1234)
    ref = rotate(bank, 1 if along_h else 2, q)
    scales = (ref.reshape(L, -1).abs().amax(1) / 448).float()
    nt = (H // 128) * (D // 128)
    pam = torch.zeros(L, nt)
    out = torch.empty((L, D, H) if along_h else (L, H, D), dtype=E4)
    hk._quantize_weights_rotated_kernel[(L, nt)](
        bank, q, out, scales, pam, bank.stride(0), bank.stride(1), bank.stride(2),
        H=H, D=D, num_tiles_d=D // 128, num_tiles=nt, ALONG_H=along_h, BLOCK=128, HB=hb)
    want = ref.transpose(1, 2).contiguous() if along_h else ref
    for layer in range(L):
        assert_fp8_close(out[layer], ref_quant(want[layer], scales[layer], E4), scales[layer].item())
    assert torch.allclose(pam.amax(1), ref.reshape(L, -1).abs().amax(1), rtol=1e-5)


# -----------------------------------------------------------------------------
# mlp.py epilogues

def _operands(seed=3, T=256, D=256, H=256):
    torch.manual_seed(seed)
    x = torch.randn(T, D)
    x = (x / x.pow(2).mean(1, keepdim=True).sqrt()).bfloat16()
    W1, W2 = (0.05 * torch.randn(H, D)).bfloat16(), (0.05 * torch.randn(H, D)).bfloat16()
    g = (1e-2 * torch.randn(T, D)).bfloat16()
    return x, W1, W2, g


def _forward(x, W1, hq=None, post_t_scale=None):
    x_s = torch.tensor([2.0 ** -4])
    w_s = W1.float().abs().max() / 448
    xq, wq = ref_quant(x, x_s, E4), ref_quant(W1, w_s, E4)
    dq = (x_s * w_s).reshape(1)
    pre = ((xq.float() @ wq.float().T) * dq).bfloat16().float()
    post = torch.relu(pre) ** 2
    ps = (post.max() * 1.03 / 448).reshape(1)
    pa, pta = torch.zeros(1), torch.zeros(1)
    kw = {} if hq is None else dict(hadamard_q=hq, rot_t=True, post_t_scale=post_t_scale, post_t_amax=pta)
    _, _, post_f8, post_t = linear_relu_square(x, W1, a_f8=xq, b_f8=wq, dequant_scale_ptr=dq, emit_f8=True,
                                               emit_t=True, post_scale=ps, post_amax=pa, store_pre=False,
                                               store_post_bf=False, **kw)
    return post, post_f8, post_t, ps, pa, pta


@pytest.mark.parametrize("hb", [32, 128])
def test_forward_epilogue_keeps_record_post_and_separates_post_t(hb):
    """Risk (b): post_f8 and the post amax are bit-identical to the record kernel's; post_t is the rotated
    post at its own scale, and its amax goes to post_t_amax."""
    x, W1, _, _ = _operands()
    q = signed_hadamard(hb, 1234)
    post, rec_f8, rec_t, ps, rec_amax, _ = _forward(x, W1)
    post_rot = rotate(post.bfloat16(), 0, q)
    pts = (post_rot.abs().max() * 1.03 / 448).reshape(1)
    _, post_f8, post_t, _, amax, t_amax = _forward(x, W1, q, pts)
    assert torch.equal(post_f8.view(torch.uint8), rec_f8.view(torch.uint8))
    assert torch.equal(amax, rec_amax)
    assert abs(rec_amax.item() - post.max().item()) <= 1e-2 * post.max().item()
    assert abs(t_amax.item() - post_rot.abs().max().item()) <= 1e-2 * post_rot.abs().max().item()
    assert_fp8_close(post_t, ref_quant(post_rot.T.contiguous(), pts, E4), pts.item(), max_mismatch=0.05)
    assert_fp8_close(rec_t, ref_quant(post.T.contiguous(), ps, E4), ps.item(), max_mismatch=0.05)


def _backward(g, W2, post_f8, ps, hq=None, rot_row=None, rot_t=None):
    g_s = torch.tensor([2.0 ** -6])
    w_s = W2.float().abs().max() / 448
    gq, wq = ref_quant(g, g_s, E5), ref_quant(W2, w_s, E4)
    dq = (g_s * w_s).reshape(1)
    dpre = (2.0 * ((gq.float() @ wq.float().T) * dq) * torch.sqrt(post_f8.float() * ps)).bfloat16()
    ds = (dpre.float().abs().max() * 4 / 57344).reshape(1)  # headroom for the rotated layouts
    da = torch.zeros(1)
    kw = {} if hq is None else dict(hadamard_q=hq, rot_row=rot_row, rot_t=rot_t)
    d_f8, d_t = linear_relu_square(g, W2, aux=post_f8, a_f8=gq, b_f8=wq, dequant_scale_ptr=dq, post_scale=ps,
                                   dpre_scale=ds, dpre_amax=da, **kw)
    return dpre, d_f8, d_t, ds, da


@pytest.mark.parametrize("hb", [32, 64, 128])
@pytest.mark.parametrize("rot_row,rot_t", [(True, True), (True, False), (False, True)])
def test_backward_epilogue_layouts(hb, rot_row, rot_t):
    """Each dpre layout is rotated iff selected; an unselected layout is byte-identical to the record
    kernel's; the amax covers exactly the stored layouts."""
    x, W1, W2, g = _operands(4)
    q = signed_hadamard(hb, 1234)
    _, post_f8, _, ps, _, _ = _forward(x, W1)
    dpre, rec_f8, rec_t, ds, rec_amax = _backward(g, W2, post_f8, ps)
    _, d_f8, d_t, _, amax = _backward(g, W2, post_f8, ps, q, rot_row, rot_t)
    want_row = rotate(dpre, 1, q) if rot_row else dpre.float()
    want_t = rotate(dpre, 0, q) if rot_t else dpre.float()
    if rot_row:
        assert_fp8_close(d_f8, ref_quant(want_row, ds, E5), ds.item(), max_mismatch=0.05)
    else:
        assert torch.equal(d_f8.view(torch.uint8), rec_f8.view(torch.uint8))
    if rot_t:
        assert_fp8_close(d_t, ref_quant(want_t.T.contiguous(), ds, E5), ds.item(), max_mismatch=0.05)
    else:
        assert torch.equal(d_t.view(torch.uint8), rec_t.view(torch.uint8))
    want_amax = max(want_row.abs().max().item(), want_t.abs().max().item())
    assert abs(amax.item() - want_amax) <= 2e-2 * want_amax
    assert abs(rec_amax.item() - dpre.float().abs().max().item()) <= 2e-2 * rec_amax.item()


# -----------------------------------------------------------------------------
# The whole Function per HADAMARD_GEMMS / HADAMARD_BLOCK (config constants: one subprocess per setting)

CASES = [("", "128"), ("all", "128"), ("all", "32"), ("dw1,dw2", "64"), ("dpost", "128"), ("dx", "32")]


@pytest.mark.parametrize("gemms,block", CASES)
def test_function_matches_reference(gemms, block):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HADAMARD")}
    if gemms:
        env.update(HADAMARD_FP8="1", HADAMARD_GEMMS=gemms, HADAMARD_BLOCK=block)
    res = subprocess.run([sys.executable, __file__, "--function-case"], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "OK" in res.stdout, res.stdout


def _function_case():
    """Build the Function's args the way gpt.py does for the configured HADAMARD_GEMMS, run forward and
    backward, and compare with mlp_fp8_reference at the same scales."""
    from track_1_short.config import HADAMARD_BLOCK, HADAMARD_FP8, HADAMARD_GEMMS
    from track_1_short.perf.hadamard_ref import amax_scale, mlp_fp8_reference
    from track_1_short.perf.kernels.mlp import FusedLinearReLUSquareFunction
    x, W1, W2, g = _operands(5)
    q = signed_hadamard(HADAMARD_BLOCK, 1234)
    gm = HADAMARD_GEMMS
    pre = (x.float() @ W1.float().T).bfloat16().float()
    post = torch.relu(pre) ** 2
    dpre = 2.0 * (g.float() @ W2.float().T) * torch.relu(pre)
    s = dict(x=2.0 ** -4, g=2.0 ** -6, w1=amax_scale(W1, E4), w2=amax_scale(W2, E4), post=amax_scale(post, E4, 1.03),
             post_t=amax_scale(rotate(post.bfloat16(), 0, q), E4, 1.03), dpre=amax_scale(dpre, E5, 4.0),
             xt=amax_scale(rotate(x, 0, q), E4, 1.10), w1_col=amax_scale(rotate(W1, 0, q), E4),
             w2_row=amax_scale(rotate(W2, 1, q), E4))
    t = lambda v: torch.tensor(float(v))
    w1_row = ref_quant(W1, s["w1"], E4)
    w1_col = ref_quant(rotate(W1, 0, q) if "dx" in gm else W1, s["w1_col"] if "dx" in gm else s["w1"], E4).T.contiguous().T
    w2_col = ref_quant(W2, s["w2"], E4).T.contiguous().T
    w2_row = ref_quant(rotate(W2, 1, q) if "dpost" in gm else W2, s["w2_row"] if "dpost" in gm else s["w2"], E4)
    x_row = ref_quant(x, s["x"], E4)
    if "dw1" in gm:
        x_t, xt_s = ref_quant(rotate(x, 0, q).T.contiguous(), s["xt"], E4), s["xt"]
    else:
        x_t, xt_s = x_row.T.contiguous(), s["x"]
    post_amax, dpre_amax, post_t_amax = torch.zeros(1), torch.zeros(1), torch.zeros(1)
    args = [w1_row, t(s["w1"] * s["x"]).reshape(1), x_row, w2_col, t(s["w2"]), t(s["post"]), post_amax, w2_row,
            t((s["w2_row"] if "dpost" in gm else s["w2"]) * s["g"]), x_t, t(xt_s), w1_col,
            t(s["w1_col"] if "dx" in gm else s["w1"]), t(s["g"]), t(s["dpre"]), dpre_amax, q if HADAMARD_FP8 else None]
    if HADAMARD_FP8:
        args += [None, t(s["post_t"]), post_t_amax]
    xr, W1r, W2r = (v.detach().clone().requires_grad_() for v in (x, W1, W2))
    out = FusedLinearReLUSquareFunction.apply(xr, W1r, W2r, *args)
    out.backward(g)
    ref = mlp_fp8_reference(x, W1, W2, g, q, gm, s)
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
    errs = dict(out=rel(out, ref["out"]), dx=rel(xr.grad, ref["dx"]), dW1=rel(W1r.grad, ref["dW1"]),
                dW2=rel(W2r.grad, ref["dW2"]))
    print(sorted(gm), HADAMARD_BLOCK, errs)
    assert all(e < 2e-2 for e in errs.values()), errs
    assert abs(post_amax.item() - ref["post_amax"].item()) <= 1e-2 * ref["post_amax"].item()
    if "dw2" in gm:
        assert abs(post_t_amax.item() - ref["post_t_amax"].item()) <= 1e-2 * ref["post_t_amax"].item()
    else:
        assert post_t_amax.item() == 0.0
    assert abs(dpre_amax.item() - ref["dpre_amax"].item()) <= 2e-2 * ref["dpre_amax"].item()
    print("OK")


if __name__ == "__main__" and "--function-case" in sys.argv:
    _function_case()
