"""MOE_ROUTER_FUSED=1: the MoE router's two GEMMs as fused Triton kernels (off by default; model/moe.py picks the op).

What it replaces, inside RoutedFP8MoE (perf/kernels/moe_mlp.py), per MoE layer and step:
  forward   logits = F.linear(rows, router_w)          a cuBLAS bf16 GEMM with N = E = 8 (one read of rows [T, 768])
            probs = softmax(logits.float()); gate, expert = probs.max(-1)       inductor kernels over [T, 8]
  backward  dx = dx_combine(dx_pad, dest, dlogits_bf.float(), router_w)          one Triton pass (reads dx_pad, writes dx)
            drouter = dlogits_bf.t() @ rows             a cuBLAS bf16 GEMM with M = 8, K = T (a second read of rows)

With the flag:
  forward   router_topk: ONE kernel reads rows once and writes probs, gate and expert (logits rounded to bf16 first,
            exactly as F.linear's bf16 output is, then the fp32 softmax and a first-index argmax, as torch's max).
  backward  dx_combine_drouter: the dx_combine pass also reads the same 128-column slab of rows and accumulates
            dlogits_bf.T @ rows for its row block into an fp32 partial [E, D]; one small sum over the row blocks gives
            drouter. dx is computed statement for statement as dx_combine computes it (same loads, same fp32 order).

Why it can be faster: the two skinny GEMMs are memory-bound reads of rows (75.5 MB at 49152 tokens per rank) that
cuBLAS runs with tile shapes meant for square problems (N = 8 / M = 8); the fused forward reads rows at streaming
bandwidth with the softmax in registers, and the fused backward reads rows inside a pass that is already running.
Nothing here has run on a GPU: whether cuBLAS is already near the bandwidth roofline on these shapes is exactly what
182/kernel-cost/bench_router.py measures. If it is, the gain is ~0.

Numerics (not bitwise against the stock path, by design of the fusion):
  * logits: fp32 accumulation of bf16 products in a different order than cuBLAS, then the same bf16 rounding. A logit
    can land on the other side of a bf16 rounding boundary (rare), which moves probs by one bf16 ulp of a logit and can
    flip the expert only when the top two logits are within that ulp.
  * softmax: tl.exp (fast exp2-based) vs torch's expf: ~1e-7 relative on probs.
  * dx: bitwise equal to dx_combine (checked in the interpreter; the GPU test checks it on the node).
  * drouter: fp32 accumulation over row blocks of 512 then one fp32 sum, rounded to bf16 once; cuBLAS also accumulates
    in fp32 and rounds once, in another order. Deterministic (no atomics).
"""
import os

import torch
import triton
import triton.language as tl
from torch import Tensor

from track_1_short.perf.kernels.moe_mlp import _experts_backward, _experts_forward, _keep_for_backward, moe_dispatch

MOE_ROUTER_FUSED = os.environ.get("MOE_ROUTER_FUSED", "0") == "1"

# Forward tiles: BM rows per program, BD columns per dot step. E is padded to E_PAD >= 16 for tl.dot.
RF_BLOCK_M, RF_BLOCK_D = 64, 256
# Backward tiles: dx_combine's own (64 x 128), and each program walks DR_ROWS_PER_PROG rows so the drouter
# partials stay small ([cdiv(T, 512), E, D] fp32 = 2.4 MB at T = 49152).
DR_BLOCK_M, DR_BLOCK_D, DR_ROWS_PER_PROG = 64, 128, 512


def _e_pad(E: int) -> int:
    return max(16, triton.next_power_of_2(E))


@triton.jit
def _router_topk_kernel(x_ptr, w_ptr, probs_ptr, gate_ptr, expert_ptr, n,
                        D: tl.constexpr, E: tl.constexpr, E_PAD: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr):
    """probs[t] = softmax(bf16(x[t] @ W.T)) in fp32, gate[t] = max_e probs[t, e], expert[t] = its first argmax."""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    valid = rows < n
    e = tl.arange(0, E_PAD)
    e_ok = e < E
    acc = tl.zeros((BM, E_PAD), dtype=tl.float32)
    for d0 in tl.static_range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        x = tl.load(x_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], mask=valid[:, None], other=0.0)
        w = tl.load(w_ptr + e[None, :] * D + cols[:, None], mask=e_ok[None, :], other=0.0)     # [BD, E_PAD] = W.T
        acc = tl.dot(x, w, acc)
    logits = acc.to(tl.bfloat16).to(tl.float32)                     # F.linear returns bf16; the op upcasts it
    logits = tl.where(e_ok[None, :], logits, float("-inf"))
    m = tl.max(logits, axis=1)
    p = tl.exp(logits - m[:, None])
    probs = p / tl.sum(p, axis=1)[:, None]
    gate = tl.max(probs, axis=1)
    expert = tl.min(tl.where((probs == gate[:, None]) & e_ok[None, :], e[None, :], E_PAD), axis=1)
    tl.store(probs_ptr + rows.to(tl.int64)[:, None] * E + e[None, :], probs, mask=valid[:, None] & e_ok[None, :])
    tl.store(gate_ptr + rows, gate, mask=valid)
    tl.store(expert_ptr + rows, expert.to(tl.int64), mask=valid)


def _router_topk_impl(rows: Tensor, router_w: Tensor):
    """(probs [T, E] fp32, gate [T] fp32, expert [T] int64) of the top-1 softmax router; rows [T, D], router_w [E, D]."""
    rows, router_w = rows.contiguous(), router_w.contiguous()
    n, D = rows.shape
    E = router_w.size(0)
    assert rows.dtype == torch.bfloat16 and router_w.dtype == torch.bfloat16 and router_w.shape == (E, D)
    assert D % RF_BLOCK_D == 0
    probs = torch.empty((n, E), device=rows.device, dtype=torch.float32)
    gate = torch.empty(n, device=rows.device, dtype=torch.float32)
    expert = torch.empty(n, device=rows.device, dtype=torch.int64)
    _router_topk_kernel[(triton.cdiv(n, RF_BLOCK_M),)](
        rows, router_w, probs, gate, expert, n, D=D, E=E, E_PAD=_e_pad(E), BM=RF_BLOCK_M, BD=RF_BLOCK_D,
        num_warps=4, num_stages=3)
    return probs, gate, expert


@triton.jit
def _dx_combine_drouter_kernel(dxp_ptr, dest_ptr, dlog_ptr, r_ptr, x_ptr, out_ptr, part_ptr, n,
                               D: tl.constexpr, E: tl.constexpr, E_PAD: tl.constexpr, BM: tl.constexpr,
                               BD: tl.constexpr, RPP: tl.constexpr):
    """out[t] = dx_pad[dest[t]] + sum_e dlog[t, e] * R[e] (moe_mlp._dx_combine_kernel's arithmetic), and
    part[pm, e, cols] = sum over this program's rows t of dlog[t, e] * x[t, cols] (fp32), from the same pass."""
    pm = tl.program_id(0)
    pd = tl.program_id(1)
    cols = pd * BD + tl.arange(0, BD)
    e = tl.arange(0, E_PAD)
    e_ok = e < E
    dr = tl.zeros((E_PAD, BD), dtype=tl.float32)
    for r0 in tl.static_range(0, RPP, BM):
        rows = pm * RPP + r0 + tl.arange(0, BM)
        valid = rows < n
        d = tl.load(dest_ptr + rows, mask=valid, other=0)
        acc = tl.load(dxp_ptr + d.to(tl.int64)[:, None] * D + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        for k in tl.static_range(E):
            w = tl.load(dlog_ptr + rows * E + k, mask=valid, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + k * D + cols).to(tl.float32)
            acc += w[:, None] * r[None, :]
        tl.store(out_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], acc.to(tl.bfloat16), mask=valid[:, None])
        g = tl.load(dlog_ptr + rows[:, None] * E + e[None, :], mask=valid[:, None] & e_ok[None, :], other=0.0)
        xr = tl.load(x_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], mask=valid[:, None], other=0.0)
        dr = tl.dot(tl.trans(g), xr, dr)
    tl.store(part_ptr + pm * (E * D) + e[:, None] * D + cols[None, :], dr, mask=e_ok[:, None])


def _dx_combine_drouter_impl(dx_pad: Tensor, dest: Tensor, dlogits_bf: Tensor, router_w: Tensor, rows: Tensor,
                             rows_per_prog: int = DR_ROWS_PER_PROG):
    """(dx [n, D] bf16 = dx_pad[dest] + dlogits_bf @ router_w, drouter [E, D] bf16 = dlogits_bf.T @ rows).
    dlogits_bf [n, E] bf16 (the op's bf16-rounded softmax gradient), dest [n] int32, rows [n, D] bf16.
    rows_per_prog (a multiple of DR_BLOCK_M): rows each program walks; the trainer uses the default, bench_router.py
    also times 128 / 256 (more programs, larger partials) since occupancy vs partial size is not known off the GPU."""
    dx_pad, dest, dlogits_bf = dx_pad.contiguous(), dest.contiguous(), dlogits_bf.contiguous()
    router_w, rows = router_w.contiguous(), rows.contiguous()
    n, E = dlogits_bf.shape
    D = dx_pad.size(1)
    assert dlogits_bf.dtype == torch.bfloat16 and rows.dtype == torch.bfloat16
    assert router_w.shape == (E, D) and dest.dtype == torch.int32 and dest.shape == (n,)
    assert rows.shape == (n, D) and D % DR_BLOCK_D == 0 and rows_per_prog % DR_BLOCK_M == 0
    num_pm = triton.cdiv(n, rows_per_prog)
    out = torch.empty((n, D), device=dx_pad.device, dtype=torch.bfloat16)
    part = torch.empty((num_pm, E, D), device=dx_pad.device, dtype=torch.float32)
    _dx_combine_drouter_kernel[(num_pm, D // DR_BLOCK_D)](
        dx_pad, dest, dlogits_bf, router_w, rows, out, part, n, D=D, E=E, E_PAD=_e_pad(E), BM=DR_BLOCK_M,
        BD=DR_BLOCK_D, RPP=rows_per_prog, num_warps=4)
    return out, part.sum(0).to(router_w.dtype)


# Custom ops (opaque to inductor), as every launch in moe_mlp.py is.
@torch.library.custom_op("nanogpt_moe_router::topk", mutates_args=())
def _router_topk_op(rows: Tensor, router_w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return _router_topk_impl(rows, router_w)


@_router_topk_op.register_fake
def _(rows, router_w):
    n, E = rows.size(0), router_w.size(0)
    return (rows.new_empty((n, E), dtype=torch.float32), rows.new_empty(n, dtype=torch.float32),
            rows.new_empty(n, dtype=torch.int64))


@torch.library.custom_op("nanogpt_moe_router::dx_combine_drouter", mutates_args=())
def _dx_combine_drouter_op(dx_pad: Tensor, dest: Tensor, dlogits_bf: Tensor, router_w: Tensor,
                           rows: Tensor) -> tuple[Tensor, Tensor]:
    return _dx_combine_drouter_impl(dx_pad, dest, dlogits_bf, router_w, rows)


@_dx_combine_drouter_op.register_fake
def _(dx_pad, dest, dlogits_bf, router_w, rows):
    return rows.new_empty(rows.shape, dtype=torch.bfloat16), router_w.new_empty(router_w.shape)


def router_topk(rows, router_w):
    return torch.ops.nanogpt_moe_router.topk(rows, router_w)


def dx_combine_drouter(dx_pad, dest, dlogits_bf, router_w, rows):
    return torch.ops.nanogpt_moe_router.dx_combine_drouter(dx_pad, dest, dlogits_bf, router_w, rows)


class RoutedFP8MoEFusedRouter(torch.autograd.Function):
    """moe_mlp.RoutedFP8MoE with the router's forward GEMM + softmax + top-1 and its backward GEMM fused (see the module
    docstring). Same arguments, same returns, same experts' kernels; only the router lines differ."""
    @staticmethod
    def forward(ctx, rows, router_w, post, w_fc, w_proj, up_row, up_col, down_row, down_col, w1_scales, w2_scales,
                x_scale, g_scale, g_amax, post_scale, post_amax, dpre_scale, dpre_amax, bank_grad, layer, acc_flag,
                topk=1):
        E = w_fc.shape[0]
        assert topk == 1, "the MoE kernels route top-1 only"
        assert bank_grad is not None, "the experts' dW is written straight into the bank's gradient"
        assert router_w.shape == (E, rows.size(1)) and post.numel() == 1
        probs, gate, expert = router_topk(rows, router_w)
        topi = expert[:, None]
        dest, src, offs, tile_expert, counts = moe_dispatch(expert, E)
        post_f = post.float().reshape(())
        scale = gate * post_f
        y_pad, post_f8, post_t, x_t = _experts_forward(rows, src, tile_expert, up_row, down_col, w1_scales,
                                                       w2_scales, x_scale, post_scale, post_amax)
        ctx.save_for_backward(y_pad, post_f8, post_t, x_t, scale, src, dest, offs, tile_expert, rows, router_w, probs,
                              topi, gate, post_f, bank_grad)
        _keep_for_backward(ctx, w_fc, up_col, down_row, w1_scales, w2_scales, x_scale, g_scale, post_scale,
                           dpre_scale, dpre_amax, layer)
        ctx.post_dtype, ctx.post_shape = post.dtype, post.shape
        ctx.acc_flag = acc_flag
        ctx.mark_non_differentiable(counts)
        out = y_pad.index_select(0, dest) * scale[:, None].to(torch.bfloat16)
        return out, probs, counts

    @staticmethod
    def backward(ctx, grad, dprobs, dcounts):
        st = ctx.saved_tensors
        y_pad, post_f8, post_t, x_t, scale, src, dest, offs, tile_expert = (
            st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[8])
        rows, router_w, probs, topi, gate, post_f, bank_grad = st[9], st[10], st[11], st[12], st[13], st[14], st[15]
        dx_pad, ds_pad = _experts_backward(ctx, grad, y_pad, post_f8, post_t, x_t, scale, src, offs, tile_expert,
                                           bank_grad)
        ds = ds_pad.index_select(0, dest)
        dpost = (ds * gate).sum().to(ctx.post_dtype).reshape(ctx.post_shape) if ctx.needs_input_grad[2] else None
        dp = torch.zeros_like(probs) if dprobs is None else dprobs.float()
        dp = dp.scatter_add(1, topi, (ds * post_f).view(1, -1).t())
        dlogits = probs * (dp - (probs * dp).sum(-1, keepdim=True))
        dlogits_bf = dlogits.to(torch.bfloat16).contiguous()
        dx, drouter = dx_combine_drouter(dx_pad, dest, dlogits_bf, router_w, rows)
        return (dx, drouter, dpost, None, None) + (None,) * 17
