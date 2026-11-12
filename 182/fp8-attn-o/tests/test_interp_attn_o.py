"""FP8_ATTN_O on the Triton interpreter (CPU, no GPU): the dual-layout quantize, the o_w weight caches, the autograd
Function against the fp8-emulation reference (perf/attn_o_scales.py) and against bf16, the o_scale folding, and
the flags-off attention forward against the base commit's. Interpreter-only:

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/fp8-attn-o/tests -q -rs

Interpreter-tested is not GPU-tested: arithmetic and indexing only, no layouts, smem, registers, cuBLAS fp8
accumulation (torch._scaled_mm runs its CPU reference here) or torch.compile / CUDA graphs (test_gpu_attn_o.py).
"""
import os
import pathlib
import subprocess
import sys
import types

import pytest

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("interpreter-only (182/tools/triton_cpu/run.sh)", allow_module_level=True)
pytest.importorskip("triton")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from track_1_short.perf import attn_o_scales as ref  # noqa: E402
from track_1_short.perf.kernels import attn_o  # noqa: E402
from track_1_short.perf.kernels.fp8_attention_quant import AMAX_PARTS  # noqa: E402
from triton.runtime.interpreter import InterpretedFunction  # noqa: E402

if not isinstance(attn_o._hadamard_quant_dual_kernel, InterpretedFunction):
    pytest.skip("the kernels were imported compiled earlier in this session (TRITON_INTERPRET was not 1 then)",
                allow_module_level=True)

E4, E5, BF = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16
# The commit this branch forked from: the flags-off attention forward must match its, bit for bit.
BASE_COMMIT = "bfa81d4"


def bytes_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def operands(T=256, K=256, N=128, seed=0, y_std=0.2, w_std=0.02, g_std=1e-3):
    g = torch.Generator().manual_seed(seed)
    y = (y_std * torch.randn(T, K, generator=g)).to(BF)
    w = (w_std * torch.randn(N, K, generator=g)).to(BF)
    gr = (g_std * torch.randn(T, N, generator=g)).to(BF)
    return y, w, gr


# ---------------------------------------------------------------------------- the dual-layout quantize

@pytest.mark.parametrize("fmt,scale", [(E4, ref.ATTN_O_Y_SCALE), (E5, ref.ATTN_O_G_SCALE), (E4, 0.0123)])
@pytest.mark.parametrize("with_amax", [False, True])
def test_fp8_dual_quant_bytes(fmt, scale, with_amax):
    y, _, _ = operands(T=256, K=384)
    y[3, 5] = 1e4  # saturates: the clamp must hold
    s = torch.tensor(scale)
    amax = torch.zeros(1) if with_amax else None
    row, t = attn_o.fp8_dual_quant(y, s, fmt, amax)
    want = torch.clamp(y.float() * (1.0 / s), -ref.FP8_MAX[fmt], ref.FP8_MAX[fmt]).to(fmt)
    assert bytes_equal(row, want)
    assert bytes_equal(t, want.T.contiguous())
    if with_amax:
        assert amax.item() == y.float().abs().max().item()


def test_fp8_dual_quant_accumulates_amax_as_max():
    y, _, _ = operands()
    amax = torch.tensor([1e3])
    attn_o.fp8_dual_quant(y, torch.tensor(1.0), E4, amax)
    assert amax.item() == 1e3  # a max, never an overwrite


def test_fp8_dual_quant_rejects_partial_tiles():
    with pytest.raises(AssertionError):
        attn_o.fp8_dual_quant(torch.zeros(100, 128, dtype=BF), torch.tensor(1.0), E4)


# ---------------------------------------------------------------------------- weight caches

@pytest.mark.parametrize("exact", [False, True])
def test_quantize_attn_o_weights(exact):
    L, D = 3, 256
    g = torch.Generator().manual_seed(3)
    bank = (0.03 * torch.randn(2 * L + 2, D, D, generator=g)).to(BF)
    o_all = bank[1:2 * L:2]  # the strided vo_bank view gpt.py passes
    scales = torch.full((L,), ref.ATTN_O_W_SCALE)
    pamax = torch.zeros(L, AMAX_PARTS)
    row, col = torch.empty(L, D, D, dtype=E4), torch.empty(L, D, D, dtype=E4)
    attn_o.quantize_attn_o_weights(o_all, scales, pamax, row, col, exact=exact, record_amax=True)
    for j in range(L):
        amax = o_all[j].float().abs().max()
        assert pamax[j].max().item() == amax.item()
        s = (amax.clamp(min=1e-12) * (1.0 / 448.0)).item() if exact else ref.ATTN_O_W_SCALE
        assert scales[j].item() == pytest.approx(s, rel=1e-7)
        want = torch.clamp(o_all[j].float() / scales[j], -448, 448).to(E4)
        assert bytes_equal(row[j], want)
        assert bytes_equal(col[j], want.T.contiguous())


def test_static_weight_quantize_skips_the_amax_pass():
    L, D = 2, 128
    bank = torch.randn(2 * L, D, D).to(BF)
    pamax = torch.full((L, AMAX_PARTS), -1.0)
    scales = torch.full((L,), ref.ATTN_O_W_SCALE)
    attn_o.quantize_attn_o_weights(bank[1::2], scales, pamax, torch.empty(L, D, D, dtype=E4),
                                   torch.empty(L, D, D, dtype=E4), exact=False, record_amax=False)
    assert (pamax == -1).all() and (scales == ref.ATTN_O_W_SCALE).all()


# ---------------------------------------------------------------------------- the Function

def run_function(y, w, o_scale, g, fuse=True, amax=False, s_y=ref.ATTN_O_Y_SCALE, s_w=ref.ATTN_O_W_SCALE,
                 s_g=ref.ATTN_O_G_SCALE):
    wq = ref.quant(w, s_w, E4)
    t = lambda v: torch.tensor(float(v))
    y_amax, g_amax = (torch.zeros(1), torch.zeros(1)) if amax else (None, None)
    yr, wr = y.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    sr = torch.tensor(float(o_scale)).requires_grad_()
    out = attn_o.AttnOFP8(yr, wr, sr, wq, wq.T.contiguous(), t(s_w), t(s_y), t(s_g), y_amax, g_amax, fuse, fuse)
    out.backward(g)
    return dict(out=out.detach(), dy=yr.grad, dw=wr.grad, do_scale=sr.grad, y_amax=y_amax, g_amax=g_amax)


@pytest.mark.parametrize("fuse,amax", [(True, False), (False, False), (False, True), (True, True)])
@pytest.mark.parametrize("K", [256, 128])  # H * v_dim: 768 / 384 at record width, scaled down
def test_function_matches_fp8_reference_and_bf16(fuse, amax, K):
    y, w, g = operands(K=K)
    o_scale = 0.87
    got = run_function(y, w, o_scale, g, fuse=fuse, amax=amax)
    want = ref.attn_o_reference(y, w, torch.tensor(o_scale), g)
    bf = ref.attn_o_bf16_reference(y, w, torch.tensor(o_scale), g)
    for k in ("out", "dy", "dw"):
        assert got[k].dtype == BF
        # the same quantized operands; only the fp32 summation order of the GEMMs differs
        assert ref.rel_err(got[k], want[k]) < 2e-3, (k, ref.rel_err(got[k], want[k]))
        # fp8 vs the record's bf16 math: e4m3 / e5m2 rounding (~2^-4 relative per element, averaged down)
        assert ref.rel_err(got[k], bf[k]) < 6e-2, (k, ref.rel_err(got[k], bf[k]))
    assert got["do_scale"].dtype == torch.float32
    assert got["do_scale"].item() == pytest.approx(want["do_scale"].item(), rel=2e-3)
    # <dW_eff, o_w> cancels across signs: bound the error by the size of its terms, not by its (small) value
    terms = (bf["dw"].float().abs() / 0.87 * w.float().abs()).sum().item()
    assert abs(got["do_scale"].item() - bf["do_scale"].item()) < 3e-2 * terms
    if amax:
        assert got["y_amax"].item() == y.float().abs().max().item()
        assert got["g_amax"].item() == g.float().abs().max().item()


@pytest.mark.parametrize("k", [-3, 0, 2])
def test_o_scale_folding_is_exact_for_powers_of_two(k):
    """o_scale only enters the descale: a power-of-two o_scale scales every output bit-exactly (no bf16 product
    with O), and d o_w carries the same factor."""
    y, w, g = operands(seed=5)
    a, b = run_function(y, w, 1.0, g), run_function(y, w, 2.0 ** k, g)
    assert torch.equal(b["out"], a["out"] * 2.0 ** k)
    assert torch.equal(b["dy"], a["dy"] * 2.0 ** k)
    assert torch.equal(b["dw"], a["dw"] * 2.0 ** k)
    assert torch.equal(b["do_scale"], a["do_scale"])  # <dW_eff, o_w> does not depend on o_scale


def test_o_scale_folding_matches_exact_arithmetic():
    """In exact arithmetic out = y @ (o_scale * o_w).T: the fold is the same product as scaling the dequantized
    weight in fp64."""
    y, w, g = operands(seed=6)
    o = 0.731
    got = run_function(y, w, o, g)
    yq = ref.quant(y, ref.ATTN_O_Y_SCALE, E4).double() * ref.ATTN_O_Y_SCALE
    wq = ref.quant(w, ref.ATTN_O_W_SCALE, E4).double() * ref.ATTN_O_W_SCALE
    exact = yq @ (o * wq).T
    assert ref.rel_err(got["out"].double(), exact) < 4e-3  # = the bf16 output rounding


def test_function_saturation_is_clamped_not_nan():
    y, w, g = operands()
    y[0, 0] = 3e3   # > 448 * 2^-4
    g[1, 1] = 5e4   # > 57344 * 2^-6
    got = run_function(y, w, 1.0, g)
    assert all(torch.isfinite(got[k]).all() for k in ("out", "dy", "dw", "do_scale"))


# ---------------------------------------------------------------------------- the attention module

def _load_attention(source: str, name: str):
    """model/attention.py with FA3 replaced by a torch varlen sliding-window attention (CPU)."""
    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal,
                               softmax_scale, window_size):
        out = torch.zeros(q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype)
        cu = cu_seqlens_q.tolist()
        for a, b in zip(cu, cu[1:]):
            qs, ks, vs = (t[a:b].float().transpose(0, 1) for t in (q, k, v))
            s = (qs @ ks.transpose(1, 2)) * softmax_scale
            i = torch.arange(b - a)
            allow = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] <= window_size[0])
            p = torch.softmax(s.masked_fill(~allow, float("-inf")), dim=-1)
            out[a:b] = (p @ vs).transpose(0, 1).to(q.dtype)
        return out
    marker = "flash_attn_interface = load_flash_attn3()"
    assert source.count(marker) == 1
    mod = types.ModuleType(name)
    sys.modules.setdefault("kernels", types.SimpleNamespace(get_kernel=None))
    mod.__dict__["_FA"] = types.SimpleNamespace(flash_attn_varlen_func=flash_attn_varlen_func)
    exec(compile(source.replace(marker, "flash_attn_interface = _FA"), name, "exec"), mod.__dict__)
    return mod


def _base_source():
    """The base commit's attention.py: from git, or from the file FP8_ATTN_O_BASE_ATTENTION names (the triton-cpu image
    has no git and a worktree's .git is not mounted: `git show bfa81d4:track_1_short/model/attention.py >
    base_attention.py` in the repo root, then FP8_ATTN_O_BASE_ATTENTION=/repo/base_attention.py through run.sh)."""
    path = os.environ.get("FP8_ATTN_O_BASE_ATTENTION")
    if path:
        return pathlib.Path(path).read_text()
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:track_1_short/model/attention.py"],
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"git show {BASE_COMMIT} unavailable here ({type(e).__name__}); a worktree's .git is not mounted "
                    "into the container")


def _attn_case(mod, training, o_fp8_args=None, seed=0):
    torch.manual_seed(seed)
    H, hd, dim, T = 6, 128, 768, 128
    m = mod.CausalSelfAttention(H, hd, qk_dim=64, v_dim=128, val_max_seq_len=T, paired=False)
    m.train(training)
    yarn = mod.Yarn(64, T, attn_scale=0.13, device=torch.device("cpu"))
    seqlens = torch.tensor([0, 64, T], dtype=torch.int32)
    lam = torch.tensor([0.5, 0.9], requires_grad=True)
    gate = torch.randn(1, T, H) * 0.25
    args = mod.AttnArgs(sa_lambdas=lam, seqlens=seqlens, bm_size=48, yarn=yarn, key_offset=False, attn_gate_w=gate,
                        aux_v=None, xsa_alpha=torch.randn(1, T, H) * 0.3, train_max_seq_len=T,
                        o_gain=torch.tensor(1.1, dtype=BF))
    x = F.rms_norm(torch.randn(1, T, dim), (dim,)).to(BF).requires_grad_(training)
    bound = 0.5 * dim ** -0.5 * 3 ** 0.5
    qk_w = ((torch.rand(2 * H * 64, dim) * 2 - 1) * bound).to(BF).requires_grad_(training)
    v_w = ((torch.rand(H * 128, dim) * 2 - 1) * bound).to(BF).requires_grad_(training)
    o_w = ((torch.rand(dim, H * 128) * 2 - 1) * bound).to(BF).requires_grad_(training)
    kw = {} if o_fp8_args is None else {"o_fp8": o_fp8_args(o_w)}
    with torch.set_grad_enabled(training):
        out = m(x, args, qk_w, v_w, o_w, None, **kw)
        if not training:
            return dict(out=out)
        g = torch.randn_like(out) * 1e-3
        out.backward(g)
    return dict(out=out.detach(), dx=x.grad, dqk=qk_w.grad, dv=v_w.grad, do=o_w.grad, dlam=lam.grad)


@pytest.mark.parametrize("training", [False, True])
def test_flags_off_attention_matches_base_commit(training):
    """o_fp8=None (FP8_ATTN_O off, and every validation): the forward (and, in training with qkv_fp8=None, i.e. the
    FP8_OFF=qkv path, the backward) is bit-identical to the base commit's attention.py."""
    base = _load_attention(_base_source(), "attention_base")
    new = _load_attention((ROOT / "track_1_short/model/attention.py").read_text(), "attention_new")
    a, b = _attn_case(base, training), _attn_case(new, training)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_fp8_attention_module_close_to_bf16():
    new = _load_attention((ROOT / "track_1_short/model/attention.py").read_text(), "attention_new2")
    t = lambda v: torch.tensor(float(v))

    def o_fp8(o_w):
        wq = ref.quant(o_w.detach(), ref.ATTN_O_W_SCALE, E4)
        return (wq, wq.T.contiguous(), t(ref.ATTN_O_W_SCALE), t(ref.ATTN_O_Y_SCALE), t(ref.ATTN_O_G_SCALE), None, None)
    a, b = _attn_case(new, True), _attn_case(new, True, o_fp8_args=o_fp8)
    for k in a:
        if k == "dlam":  # 2 scalars: d sa_lambdas, through <dW_eff, o_w> (cancelling) and the QKV path
            assert ref.rel_err(b[k], a[k]) < 0.25, (k, b[k], a[k])
            continue
        # out and d o_w see one fp8 GEMM; dx / dqk / dv see dy's fp8 error through the softmax backward (at this
        # random init, many small, cancelling contributions per entry)
        tol = 6e-2 if k in ("out", "do") else 0.1
        assert ref.rel_err(b[k], a[k]) < tol, (k, ref.rel_err(b[k], a[k]))
