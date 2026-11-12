"""FP8_ATTN_O (182/fp8-attn-o): the attention output projection in fp8 during training, forward and backward.

What it replaces: `F.linear(y, o_scale * o_w.type_as(y))` (model/attention.py), a bf16 GEMM forward and two
bf16 GEMMs backward (dx, dW), plus the bf16 product o_scale * o_w.
How: three per-tensor-scaled `torch._scaled_mm` GEMMs on fp8 operands, the QKV pattern (qkv_rope.PackedFP8QKV):
  forward  out    = q(y) @ q(o_w).T     e4m3 x e4m3, fast accumulate, descale s_y * (s_w * o_scale)
  dx       dy     = q(g) @ q(o_w)       e5m2 x e4m3, descale s_g * (s_w * o_scale)
  dW       dW_eff = q(g).T @ q(y)       e5m2 x e4m3, descale s_g * s_y;  d o_w = o_scale * dW_eff
o_scale = sa_lambdas[1] (* o_gain) never touches the weight: it is folded into scale_b (the math is in
perf/attn_o_scales.py, with the scales and their derivation), so the fp8 weight cache stays valid whatever
o_scale does, and d o_scale = <dW_eff, o_w> as in PackedFP8QKV.

Operand layouts (_scaled_mm wants A row-major, B column-major, both K-major on Hopper):
  o_w   both layouts from the lagged weight-cache refresh (GPT._quantize_attn_o_fp8, in the attention refresh
        graph): row [768, H*v] for the forward, col = transposed [H*v, 768] for dx.
  y     row [T, H*v] (forward) and transposed [H*v, T] (dW), quantized here:
          gated / XSA layers (the static arm): quantize_dual_layout_fused, whose row half inductor fuses into
            the gate / XSA elementwise kernel that produces y (the bf16 y is then never written), plus one fp8
            transpose pass;
          plain layers (y straight from FA3, no elementwise pass to fuse into), and every layer when an amax is
            recorded: fp8_dual_quant, one Triton pass reading y once and writing both layouts.
  g     the same two choices for the incoming gradient (e5m2): fused into its producer, or fp8_dual_quant.
fp8_dual_quant is hadamard._hadamard_quant_dual_kernel with both rotations off (no Q load, no dot): a plain
dual-layout quantize at a static scale, optionally atomic-maxing the tensor's amax into a delayed-scale slot.

Invariants: `use_fast_accum=False` on both gradient GEMMs (as every gradient GEMM in this repo); an explicit
clamp before every fp8 cast; bf16 outputs. Validation never comes here (no_grad: bf16 F.linear).
"""
import torch
import triton

from track_1_short.perf.kernels.fp8_attention_quant import (AMAX_PARTS, _packed_batched_matrix_amax_partial_kernel,
                                                            _quantize_dual_layout_packed_batched_kernel)
from track_1_short.perf.kernels.hadamard import HADAMARD_TILE, _hadamard_quant_dual_kernel
from track_1_short.perf.kernels.mlp import _f8_mm_op, quantize_dual_layout_fused

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2

# The three GEMMs, each its own custom op (opaque to inductor, named in traces), as mlp.py's.
attn_o_fwd_f8_op = _f8_mm_op("attn_o_fwd_f8", transpose_b=True, fast_accum=True)   # out    = y_row @ w_row.T
attn_o_dx_f8_op = _f8_mm_op("attn_o_dx_f8", transpose_b=True, fast_accum=False)    # dy     = g_row @ w_col.T
attn_o_dw_f8_op = _f8_mm_op("attn_o_dw_f8", transpose_b=True, fast_accum=False)    # dW_eff = g_t @ y_t.T


def fp8_dual_quant(src: torch.Tensor, scale: torch.Tensor, fmt: torch.dtype, amax: torch.Tensor | None = None):
    """(row [M, N], t [N, M]) = fmt copies of src / scale (clamped), from one read of the bf16 src. With `amax`
    (a (1,) f32 slot of a delayed-scale buffer), the kernel also atomic-maxes max|src| into it. M and N must be
    multiples of 128 (the kernel's tile edge). Launched directly (not a custom op), like the MLP kernels: under
    torch.compile the amax write is a user-Triton-kernel mutation, as linear_relu_square's post/dpre amax."""
    assert src.ndim == 2 and src.stride(1) == 1 and src.dtype == torch.bfloat16
    M, N = src.shape
    assert M % HADAMARD_TILE == 0 and N % HADAMARD_TILE == 0, f"{tuple(src.shape)} not a multiple of {HADAMARD_TILE}"
    row = torch.empty((M, N), device=src.device, dtype=fmt)
    t = torch.empty((N, M), device=src.device, dtype=fmt)
    # q_ptr is never loaded with both rotations off (any tensor will do); HB is then only a constexpr.
    _hadamard_quant_dual_kernel[(M // HADAMARD_TILE, N // HADAMARD_TILE)](
        src, src, row, t, scale, scale, amax if amax is not None else scale, M, N, src.stride(0),
        EMIT_ROW=True, ROT_ROW=False, ROT_T=False, RECORD_T_AMAX=amax is not None, E5M2=fmt == E5M2,
        BLOCK=HADAMARD_TILE, HB=HADAMARD_TILE, num_warps=8)
    return row, t


def _dual(src, scale, fmt, fused: bool, amax):
    """Both layouts of an activation / gradient: fused into its producer (static arm, gated/XSA y and g) or one
    Triton pass (plain layers; and whenever an amax is recorded). A shape that is not a multiple of 128 (no
    training shape; a hypothetical caller) falls back to the fused path, without an amax."""
    if not fused and src.shape[0] % HADAMARD_TILE == 0 and src.shape[1] % HADAMARD_TILE == 0:
        return fp8_dual_quant(src, scale, fmt, amax)
    return quantize_dual_layout_fused(src, scale, fmt)


class AttnOFP8Function(torch.autograd.Function):
    """out = y @ (o_scale * o_w).T with every GEMM in fp8 (see the module docstring).

    y [T, H*v] bf16 (contiguous), o_w [768, H*v] bf16 (a view of vo_bank), o_scale 0-d fp32;
    w_row / w_col: o_w's e4m3 caches ([768, H*v] and [H*v, 768], contiguous) at w_scale (0-d f32);
    y_scale / g_scale: 0-d f32 (static or delayed); y_amax / g_amax: (1,) f32 slots or None (no amax recorded);
    fuse_y / fuse_g: quantize y / g with the producer-fused aten path instead of the Triton pass."""
    @staticmethod
    def forward(ctx, y, o_w, o_scale, w_row, w_col, w_scale, y_scale, g_scale, y_amax, g_amax, fuse_y, fuse_g):
        y_row, y_t = _dual(y, y_scale, E4M3, fuse_y and y_amax is None, y_amax)
        scale_b = w_scale * o_scale
        out = torch.ops.nanogpt.attn_o_fwd_f8(y_row, w_row, y_scale, scale_b)
        ctx.save_for_backward(y_t, o_w, o_scale, scale_b)
        # Persistent caches and scales by name (as FusedLinearReLUSquareFunction keeps them).
        ctx.w_col, ctx.y_scale, ctx.g_scale, ctx.g_amax, ctx.fuse_g = w_col, y_scale, g_scale, g_amax, fuse_g
        return out

    @staticmethod
    def backward(ctx, grad_out):
        st = ctx.saved_tensors
        y_t, o_w, o_scale, scale_b = st[0], st[1], st[2], st[3]
        g = grad_out.reshape(-1, grad_out.shape[-1])
        g_row, g_t = _dual(g, ctx.g_scale, E5M2, ctx.fuse_g and ctx.g_amax is None, ctx.g_amax)
        dy = torch.ops.nanogpt.attn_o_dx_f8(g_row, ctx.w_col, ctx.g_scale, scale_b)
        dw_eff = torch.ops.nanogpt.attn_o_dw_f8(g_t, y_t, ctx.g_scale, ctx.y_scale)
        # o_scale is an fp32 scalar while the weights are bf16; autograd checks each gradient's dtype.
        grad_o_scale = (dw_eff.float() * o_w.float()).sum().to(o_scale.dtype)
        # One slot per forward argument; y, o_w and o_scale carry gradients.
        return dy, dw_eff * o_scale, grad_o_scale, None, None, None, None, None, None, None, None, None


AttnOFP8 = AttnOFP8Function.apply


# -----------------------------------------------------------------------------
# The weight caches (GPT._quantize_attn_o_fp8, inside the attention half of the lagged fp8 refresh).

def quantize_attn_o_weights(o_all: torch.Tensor, scales: torch.Tensor, partial_amax: torch.Tensor,
                            row: torch.Tensor, col: torch.Tensor, exact: bool, record_amax: bool):
    """e4m3 copies of every attention slot's full-width O, o_all [L, 768, 768] (a strided view of vo_bank), in both
    layouts: row [L, 768, 768] and col [L, 768, 768] (col[j] = row[j].T), at scales [L] (one launch of
    fp8_attention_quant's packed dual-layout kernel, with no second source).
      exact        scales = this call's amax / 448 (FP8_ATTN_O_SCALES=delayed; as the QKV caches)
      static       scales are left alone (filled once with ATTN_O_W_SCALE); the amax pass runs only with
                   record_amax (HADAMARD_SAT_STATS), for the o_w saturation ratio.
    The half-V layers' packed caches are gathered from these by the caller."""
    L, M, N = o_all.shape
    assert o_all.stride(1) == N and o_all.stride(2) == 1
    assert row.shape == (L, M, N) and col.shape == (L, N, M) and row.is_contiguous() and col.is_contiguous()
    assert scales.shape == (L,) and partial_amax.shape == (L, AMAX_PARTS)
    if exact or record_amax:
        _packed_batched_matrix_amax_partial_kernel[(L, AMAX_PARTS)](
            o_all, o_all, partial_amax, FIRST_ELEMENTS=M * N, SECOND_ELEMENTS=0,
            FIRST_STRIDE_B=o_all.stride(0), SECOND_STRIDE_B=o_all.stride(0),
            NUM_PARTS=AMAX_PARTS, BLOCK_SIZE=2048, num_stages=1, num_warps=8)
    if exact:
        torch.mul(partial_amax.amax(dim=1).clamp_(min=1.0e-12), 1.0 / 448.0, out=scales)
    block_m, block_n = 64, 128
    _quantize_dual_layout_packed_batched_kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n), L)](
        o_all, o_all, row, col, scales, M, N, FIRST_ROWS=M,
        FIRST_STRIDE_B=o_all.stride(0), SECOND_STRIDE_B=o_all.stride(0),
        ROW_STRIDE_B=row.stride(0), TRANSPOSED_STRIDE_B=col.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, num_stages=2, num_warps=8)
