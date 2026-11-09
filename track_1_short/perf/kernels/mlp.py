"""The fused MLP, relu(x @ W1.T)^2 @ W2.T, with every GEMM in FP8 during training.

What it replaces: a bf16 up-projection kernel, a bf16 `post @ W2`, and four bf16 backward GEMMs.
Why it is faster: each GEMM runs on fp8 operands, and every fp8 operand a later GEMM needs -- in
both the row-major and the transposed layout `torch._scaled_mm` wants -- is written by the kernel
epilogue that already holds the tile in registers, never by a separate quantize/transpose pass
(a standalone transpose pass, even a bandwidth-optimal one, measured +10.7 s per run).
Invariants: `use_fast_accum=False` on every gradient GEMM (NaN otherwise), and an explicit clamp
before every fp8 cast (e4m3fn overflow is NaN, not inf).

Scales, all per-tensor fp32 (the caller in model/gpt.py owns and refreshes them):
  x      static 2^-4 (post-RMS-norm rows have max|x| <= sqrt(768) < 448 * 2^-4)
  W1, W2 per bank slot, from the weights (exact for the first calls, then one step delayed)
  post   delayed: the forward epilogue atomic-maxes this step's amax, read next step
  grad   static 2^-6 (e5m2) for the incoming gradient
  dpre   delayed, like post, from the backward epilogue

Provenance: bf16 kernel by @andrewbriand, @jrauvola; full-fp8 MLP from record #360 (ANVIL2).
"""
import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources
from triton.tools.tensor_descriptor import TensorDescriptor

from track_1_short.config import (HADAMARD_DPOST, HADAMARD_DW1, HADAMARD_DW2, HADAMARD_DX, HADAMARD_FP8,
                                  HADAMARD_SEED)
from track_1_short.perf.hadamard_ref import signed_hadamard
from track_1_short.perf.kernels.hadamard import (HADAMARD_BLOCK, HADAMARD_TILE, _hadamard_quant_dual_kernel,
                                                 hadamard_dual_quant, rotate_last)
from track_1_short.perf.fp8_dump import FP8_DUMP
from track_1_short.perf.kernels.transpose import transpose_copy

# Weight scales lag a step (see quantize_mlp_weights_dual), so they keep 12% headroom.
WEIGHT_SCALE_HEADROOM = 1.12

# -----------------------------------------------------------------------------
# The GEMM kernel. Forward: pre = x @ W1.T, post = relu(pre)^2. Backward (fp8 only, training):
# dpre = 2 * (g @ W2) * relu(pre), with relu(pre) = sqrt(post) rebuilt from the e4m3 post, emitted as
# e5m2 in both layouts (fp8 dx and dW1) with its amax recorded for the next step's scale.
# The constexpr flags select the forward epilogue and are dead-code-eliminated when off. Each
# transposed store is a register-tile tl.trans, so it costs writes only.
#   STORE_PRE, STORE_POST_BF  bf16 pre / post (the bf16 forward: validation)
#   EMIT_F8, EMIT_T           e4m3 post, row-major and transposed (fp8 down projection and dW2)

# HADAMARD_FP8 (perf/kernels/hadamard.py): the rotated epilogues multiply each bf16 [128, 128] half tile,
# block-diagonally in HB-blocks (rotate_last), by Q = diag(signs) @ H_HB (exact +-1) in fp32 accumulation,
# then by 1 / sqrt(HB). Three constexprs, each off unless its GEMM is in HADAMARD_GEMMS, so an unselected
# GEMM's operand takes the record's code exactly:
#   forward   ROT_T    post_t rotated along tokens (dW2), at its OWN delayed scale post_t_scale, amax into
#                      post_t_amax. post_f8 and post_amax stay the record's: the forward is never rotated.
#   backward  ROT_ROW  dpre row rotated along features (dx);  ROT_T  dpre_t rotated along tokens (dW1).
#                      One dpre scale serves both layouts (both are gradient-only), its amax covers both.
# Q is loaded once per tile, after the mainloop (L2-resident, HB^2 * 2 bytes), and reused by every dot.
#
# Register budget of the backward epilogue (8 warps, 256 threads, <= 255 regs/thread at 1 CTA/SM): the
# [128, 256] fp32 accumulator is 128 regs/thread. c0/c1 are converted to bf16 first (32 regs each), which
# ends the accumulator's live range; then each rotation (64 regs fp32) is quantized and stored before the
# next one starts, in the order r0, r1 (row), t0, t1 (transposed). Peak ~ c0 + c1 + one rotation + its fp8
# copy ~ 64 + 64 + 16 = 144 regs plus addressing, instead of r0 r1 t0 t1 all live (256 regs, spills).
#
# Shared-memory budget (H100: 227 KB per CTA), estimated, not measured -- prime_stage_cache resolves
# the real stage count per variant and returns it. fp8 mainloop: one stage = A [128, 128] + B [256, 128]
# = 16 + 32 = 48 KB; 3 stages = 144 KB (the default), 2 = 96 KB. Record epilogue: the TMA-store staging
# of the [128, 128] fp8 boxes (16 KB each) and the backward's [128, 256] e4m3 post box (32 KB). A rotation
# adds the wgmma B operand Q (HB^2 * 2 B: 32 KB at 128, 8 KB at 64, 2 KB at 32) and up to one bf16
# [128, 128] dot-operand / reshape conversion buffer (<= 32 KB, independent of HB):
#   variant            mainloop@3   rotation extra      expectation
#   record fwd / bwd   144 KB       0                   3 stages (the record's)
#   rot @128           144 KB       32 + <=32 KB        may exceed 227 KB -> 2 stages (round 1: ~240 KB)
#   rot @64            144 KB        8 + <=32 KB        borderline
#   rot @32            144 KB        2 + <=32 KB        best chance of keeping 3 stages

@triton.jit
def _hadamard_q(q_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    return tl.load(q_ptr + offs[:, None] * BLOCK + offs[None, :])

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                              dequant_scale_ptr,
                              M, N, K,
                              aux_f8_desc, post_scale_ptr, post_amax_ptr, aux_t_desc,
                              c_f8_desc, c_t_desc, dpre_scale_ptr, dpre_amax_ptr, aux_w_desc,
                              hadamard_q_ptr, post_t_scale_ptr, post_t_amax_ptr,
                              BLOCK_SIZE_M: tl.constexpr,
                              BLOCK_SIZE_N: tl.constexpr,
                              BLOCK_SIZE_K: tl.constexpr,
                              NUM_SMS: tl.constexpr,
                              FORWARD: tl.constexpr,
                              USE_FP8: tl.constexpr,
                              EMIT_F8: tl.constexpr, EMIT_T: tl.constexpr, STORE_PRE: tl.constexpr,
                              STORE_POST_BF: tl.constexpr,
                              ROT_ROW: tl.constexpr, ROT_T: tl.constexpr, HB: tl.constexpr,
                              ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # Epilogue scalars load once per CTA, above the tile loop: no store target ever aliases them.
    if USE_FP8:
        dequant_scale = tl.load(dequant_scale_ptr)
    else:
        dequant_scale = 1.0
    if EMIT_F8:
        inv_post_scale = 1.0 / tl.load(post_scale_ptr)
    else:
        inv_post_scale = 1.0
    if FORWARD:
        post_scale = 1.0
        inv_dpre_scale = 1.0
    else:
        post_scale = tl.load(post_scale_ptr)
        inv_dpre_scale = 1.0 / tl.load(dpre_scale_ptr)
    if FORWARD and ROT_T:
        inv_post_t_scale = 1.0 / tl.load(post_t_scale_ptr)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        if USE_FP8:
            accumulator *= dequant_scale

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        if FORWARD:
            c0 = acc0.to(dtype)                      # pre (bf16-rounded)
            if STORE_PRE:
                c_desc.store([offs_am, offs_bn], c0)
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            if STORE_POST_BF:
                aux_desc.store([offs_am, offs_bn], c0_post)
            if EMIT_F8:
                q0 = tl.minimum(c0_post.to(tl.float32) * inv_post_scale, 448.0).to(tl.float8e4nv)
                aux_f8_desc.store([offs_am, offs_bn], q0)
                if EMIT_T:
                    if ROT_T:
                        # post_t feeds dW2 (k = tokens): rotate the tile along its BLOCK_SIZE_M tokens.
                        hq = _hadamard_q(hadamard_q_ptr, HB)
                        r0 = rotate_last(tl.trans(c0_post), hq, BLOCK_SIZE_N // 2, BLOCK_SIZE_M, HB)
                        rot_max = tl.max(tl.max(tl.abs(r0), axis=1), axis=0)
                        aux_t_desc.store([offs_bn, offs_am], tl.maximum(tl.minimum(r0 * inv_post_t_scale, 448.0), -448.0).to(tl.float8e4nv))
                    else:
                        aux_t_desc.store([offs_bn, offs_am], tl.trans(q0))
            c1 = acc1.to(dtype)
            if STORE_PRE:
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            if STORE_POST_BF:
                aux_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1_post)
            if EMIT_F8:
                q1 = tl.minimum(c1_post.to(tl.float32) * inv_post_scale, 448.0).to(tl.float8e4nv)
                aux_f8_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], q1)
                if EMIT_T:
                    if ROT_T:
                        r1 = rotate_last(tl.trans(c1_post), hq, BLOCK_SIZE_N // 2, BLOCK_SIZE_M, HB)
                        rot_max = tl.maximum(rot_max, tl.max(tl.max(tl.abs(r1), axis=1), axis=0))
                        aux_t_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], tl.maximum(tl.minimum(r1 * inv_post_t_scale, 448.0), -448.0).to(tl.float8e4nv))
                    else:
                        aux_t_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], tl.trans(q1))
                # atomic_max, not a loop-carried scalar: tl.range(flatten=True) does not guarantee those.
                tile_max = tl.max(tl.max(tl.maximum(c0_post.to(tl.float32), c1_post.to(tl.float32)), axis=1), axis=0)
                tl.atomic_max(post_amax_ptr, tile_max)
                if ROT_T and EMIT_T:
                    # The rotated post_t has its own scale: the record's post amax (and so post_f8, the
                    # forward's operand) never sees the rotation.
                    tl.atomic_max(post_t_amax_ptr, rot_max)
        else:
            # aux is the e4m3 post: one [BM, BN] TMA box, re-split exactly like the accumulator.
            aux_tile = aux_w_desc.load([offs_am, offs_bn])
            aux_tile = tl.reshape(aux_tile, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            aux_tile = tl.permute(aux_tile, (0, 2, 1))
            a0_raw, a1_raw = tl.split(aux_tile)
            c0 = (2.0 * acc0 * tl.sqrt(a0_raw.to(tl.float32) * post_scale)).to(dtype)
            c1 = (2.0 * acc1 * tl.sqrt(a1_raw.to(tl.float32) * post_scale)).to(dtype)
            if ROT_ROW or ROT_T:
                # dpre row feeds dx (k = hidden): rotated along each half's BLOCK_SIZE_N // 2 features.
                # dpre_t feeds dW1 (k = tokens): rotated along the BLOCK_SIZE_M tokens, already transposed.
                # An unselected layout is the record's plain quantize. Each rotation is quantized and
                # stored before the next is computed (register budget above).
                hq = _hadamard_q(hadamard_q_ptr, HB)
                if ROT_ROW and ROT_T:
                    tile_max = 0.0
                else:
                    c0f = c0.to(tl.float32)
                    c1f = c1.to(tl.float32)
                    q0 = tl.maximum(tl.minimum(c0f * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5)
                    q1 = tl.maximum(tl.minimum(c1f * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5)
                    tile_max = tl.max(tl.max(tl.maximum(tl.abs(c0f), tl.abs(c1f)), axis=1), axis=0)
                if ROT_ROW:
                    r0 = rotate_last(c0, hq, BLOCK_SIZE_M, BLOCK_SIZE_N // 2, HB)
                    c_f8_desc.store([offs_am, offs_bn], tl.maximum(tl.minimum(r0 * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5))
                    tile_max = tl.maximum(tile_max, tl.max(tl.max(tl.abs(r0), axis=1), axis=0))
                    r1 = rotate_last(c1, hq, BLOCK_SIZE_M, BLOCK_SIZE_N // 2, HB)
                    c_f8_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], tl.maximum(tl.minimum(r1 * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5))
                    tile_max = tl.maximum(tile_max, tl.max(tl.max(tl.abs(r1), axis=1), axis=0))
                else:
                    c_f8_desc.store([offs_am, offs_bn], q0)
                    c_f8_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], q1)
                if ROT_T:
                    t0 = rotate_last(tl.trans(c0), hq, BLOCK_SIZE_N // 2, BLOCK_SIZE_M, HB)
                    c_t_desc.store([offs_bn, offs_am], tl.maximum(tl.minimum(t0 * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5))
                    tile_max = tl.maximum(tile_max, tl.max(tl.max(tl.abs(t0), axis=1), axis=0))
                    t1 = rotate_last(tl.trans(c1), hq, BLOCK_SIZE_N // 2, BLOCK_SIZE_M, HB)
                    c_t_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], tl.maximum(tl.minimum(t1 * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5))
                    tile_max = tl.maximum(tile_max, tl.max(tl.max(tl.abs(t1), axis=1), axis=0))
                else:
                    c_t_desc.store([offs_bn, offs_am], tl.trans(q0))
                    c_t_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], tl.trans(q1))
            else:
                c0f = c0.to(tl.float32)
                c1f = c1.to(tl.float32)
                q0 = tl.maximum(tl.minimum(c0f * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5)
                q1 = tl.maximum(tl.minimum(c1f * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5)
                c_f8_desc.store([offs_am, offs_bn], q0)
                c_f8_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], q1)
                c_t_desc.store([offs_bn, offs_am], tl.trans(q0))
                c_t_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], tl.trans(q1))
                tile_max = tl.max(tl.max(tl.maximum(tl.abs(c0f), tl.abs(c1f)), axis=1), axis=0)
            tl.atomic_max(dpre_amax_ptr, tile_max)


# -----------------------------------------------------------------------------
# Activation quantize for the MLP input and the incoming gradient. The cost is passes over the
# [T, 768] residual stream, not arithmetic: the row-major half is plain ATen, which inductor fuses
# into the producer's epilogue; the transposed half is one coalesced fp8 -> fp8 pass.

@torch.library.custom_op("nanogpt::glue_fp8_t", mutates_args=())
def glue_fp8_t_op(src: torch.Tensor) -> torch.Tensor:
    """dst [N, M] = src [M, N].T for a row-major 1-byte tensor. A custom op so inductor keeps it one
    coalesced tiled pass instead of a strided pointwise copy; byte-exact."""
    assert src.ndim == 2 and src.stride(1) == 1 and src.element_size() == 1
    dst = torch.empty((src.shape[1], src.shape[0]), device=src.device, dtype=src.dtype)
    transpose_copy(src.view(torch.uint8), dst.view(torch.uint8))
    return dst

@glue_fp8_t_op.register_fake
def _(src):
    return src.new_empty((src.shape[1], src.shape[0]), dtype=src.dtype)

def quantize_dual_layout_fused(src: torch.Tensor, scale: torch.Tensor, fmt: torch.dtype):
    """(row [M, N], transposed [N, M]) fp8 copies of src / scale, saturated to fmt's range.

    Inductor fuses the row-major half into src's producer and rounds that kernel's fp32
    registers, not the stored bf16: one rounding instead of two, so never bit-identical to a
    standalone quantize, but never farther from the exact value."""
    assert src.ndim == 2
    lim = 448.0 if fmt == torch.float8_e4m3fn else 57344.0
    row = torch.clamp(src.to(torch.float32) * (1.0 / scale), -lim, lim).to(fmt)
    return row, torch.ops.nanogpt.glue_fp8_t(row)


# -----------------------------------------------------------------------------
# Weight caches: one read of a [L, H, D] bank writes both e4m3 layouts and each tile's amax.

@triton.jit
def _quantize_weights_dual_kernel(w_ptr, row_ptr, col_ptr, scale_ptr, partial_amax_ptr, w_stride_l, w_stride_h,
                                  w_stride_d, H: tl.constexpr, D: tl.constexpr, num_tiles_d: tl.constexpr,
                                  num_tiles: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
    layer = tl.program_id(0)
    tile = tl.program_id(1)
    tile_h = tile // num_tiles_d
    tile_d = tile % num_tiles_d
    offs_h = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tile_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_h[:, None] < H) & (offs_d[None, :] < D)
    v = tl.load(w_ptr + layer * w_stride_l + offs_h[:, None] * w_stride_h + offs_d[None, :] * w_stride_d, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + layer)
    q = tl.maximum(tl.minimum(v / s, 448.0), -448.0).to(tl.float8e4nv)
    tl.store(row_ptr + layer * H * D + offs_h[:, None] * D + offs_d[None, :], q, mask=mask)
    mask_t = (offs_d[:, None] < D) & (offs_h[None, :] < H)
    tl.store(col_ptr + layer * H * D + offs_d[:, None] * H + offs_h[None, :], tl.trans(q), mask=mask_t)
    tl.store(partial_amax_ptr + layer * num_tiles + tile, tl.max(tl.max(tl.abs(v), axis=1), axis=0))

def quantize_weights_dual_ntiles(H: int, D: int) -> int:
    """Partial-amax slots per layer: one per tile of _quantize_weights_dual_kernel's 64x64 grid."""
    return triton.cdiv(H, 64) * triton.cdiv(D, 64)

def quantize_mlp_weights_dual(bank: torch.Tensor, scales: torch.Tensor, partial_amax: torch.Tensor,
                              row: torch.Tensor, col_t: torch.Tensor, update_scales: bool):
    """Quantize a weight bank [L, H, D] (bf16) into both fp8 caches from one read.

    `row` is [L, H, D] e4m3 contiguous; `col_t` is [L, D, H] contiguous, i.e. the transpose(1, 2)
    view of a column-major [L, H, D] cache. With update_scales, `scales` [L] are first refreshed
    from the previous call's per-tile amaxes: the scale lags a step, which takes the reduction off
    the critical path and is loss-neutral because the weights move <<1% per step."""
    L, H, D = bank.shape
    BLOCK_H = BLOCK_D = 64
    num_tiles_d = triton.cdiv(D, BLOCK_D)
    num_tiles = triton.cdiv(H, BLOCK_H) * num_tiles_d
    assert partial_amax.shape == (L, num_tiles)
    if update_scales:
        torch.clamp(partial_amax.amax(dim=1) * (WEIGHT_SCALE_HEADROOM / 448.0), min=1e-12, out=scales)
    _quantize_weights_dual_kernel[(L, num_tiles)](
        bank, row, col_t, scales, partial_amax, bank.stride(0), bank.stride(1), bank.stride(2), H=H, D=D,
        num_tiles_d=num_tiles_d, num_tiles=num_tiles, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2)


# -----------------------------------------------------------------------------
# Launcher.
#
# num_stages per kernel variant. Emit variants start at 3 stages and step down if one exceeds
# H100 shared memory. This is the one module-level mutable in the package, and it has to be:
# linear_relu_square runs inside torch.compile's autograd-Function subgraph, which rejects any
# Python-state mutation and cannot retry a failed launch. So the dict is read-only under compile
# and written only by eager calls, i.e. by prime_stage_cache() before compiling.
_lrs_stage_cache = {}

def linear_relu_square(a, b, aux=None, a_f8=None, b_f8=None, dequant_scale_ptr=None,
                       emit_f8=False, post_scale=None, post_amax=None,
                       emit_t=False, store_pre=True, store_post_bf=True,
                       dpre_scale=None, dpre_amax=None, hadamard_q=None, rot_row=None, rot_t=None,
                       post_t_scale=None, post_t_amax=None):
    """Forward (aux is None) returns (pre, post, post_f8, post_t), None where not requested. Backward
    (aux = the e4m3 post; fp8 operands only) returns (dpre_f8, dpre_t), the e5m2 dpre in both layouts.

    HADAMARD_FP8: with hadamard_q (Q, [HB, HB]) the forward rotates post_t (rot_t, default on) at
    post_t_scale and records its amax in post_t_amax; the backward rotates the dpre row (rot_row) and/or
    dpre_t (rot_t), both default on. Without hadamard_q (or with both flags False) it is the record's kernel."""
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    use_fp8 = b_f8 is not None
    FORWARD = aux is None
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # One tile shape for both passes: the backward measured 0.3824 ms at 128x256, 0.4102 ms transposed.
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 128 if use_fp8 else 64
    num_warps = 8
    # The transposed post only exists for the fp8 dW2, when the bf16 post is dead.
    assert emit_f8 or not emit_t
    assert not (emit_t and store_post_bf)
    rot_row = hadamard_q is not None and (not FORWARD) and (True if rot_row is None else rot_row)
    rot_t = hadamard_q is not None and (True if rot_t is None else rot_t)
    hb = hadamard_q.shape[0] if (rot_row or rot_t) else 0
    if rot_row or rot_t:
        # The rotation blocks tile one M tile (tokens) and one half N tile (features).
        assert use_fp8 and BLOCK_SIZE_M == BLOCK_SIZE_N // 2 == HADAMARD_TILE
        assert hadamard_q.shape == (hb, hb) and hb in (32, 64, 128) and hadamard_q.dtype == torch.bfloat16
        assert M % BLOCK_SIZE_M == 0 and N % (BLOCK_SIZE_N // 2) == 0, "partial Hadamard blocks"
    if FORWARD and rot_t:
        assert emit_t, "the forward only rotates the transposed post"
        assert post_t_scale is not None and post_t_amax is not None, "post_t has its own scale and amax"
    if not FORWARD:
        assert use_fp8 and aux.dtype == torch.float8_e4m3fn
        assert post_scale is not None and dpre_scale is not None and dpre_amax is not None
    e4m3, e5m2 = torch.float8_e4m3fn, torch.float8_e5m2
    c, aux_out, aux_f8, aux_t, c_f8, c_t = [
        torch.empty((rows, cols), device=a.device, dtype=dt) if wanted else None
        for wanted, rows, cols, dt in (
            (FORWARD and store_pre,     M, N, dtype),   # c       bf16 pre
            (FORWARD and store_post_bf, M, N, dtype),   # aux     bf16 post
            (FORWARD and emit_f8,       M, N, e4m3),    # aux_f8  e4m3 post, row
            (FORWARD and emit_t,        N, M, e4m3),    # aux_t   e4m3 post, .T
            (not FORWARD,               M, N, e5m2),    # c_f8    e5m2 dpre, row
            (not FORWARD,               N, M, e5m2))]   # c_t     e5m2 dpre, .T
    if not FORWARD:
        aux_out = aux
    # A disabled epilogue still needs a descriptor: point it at a small dummy tile rather than at a
    # live output buffer, and the wide [BM, BN] box exists exactly when the backward reads it.
    dummy_tile = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), device=a.device, dtype=dtype) if c is None or aux_out is None else None
    box, t_box = [BLOCK_SIZE_M, BLOCK_SIZE_N // 2], [BLOCK_SIZE_N // 2, BLOCK_SIZE_M]
    a_desc = TensorDescriptor.from_tensor(a_f8 if use_fp8 else a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b_f8 if use_fp8 else b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c if c is not None else dummy_tile, box)
    aux_desc = TensorDescriptor.from_tensor(aux_out if aux_out is not None else dummy_tile, box)
    aux_f8_desc = TensorDescriptor.from_tensor(aux_f8, box) if aux_f8 is not None else aux_desc
    aux_t_desc = TensorDescriptor.from_tensor(aux_t, t_box) if aux_t is not None else aux_desc
    c_f8_desc = TensorDescriptor.from_tensor(c_f8, box) if c_f8 is not None else c_desc
    c_t_desc = TensorDescriptor.from_tensor(c_t, t_box) if c_t is not None else c_desc
    aux_w_desc = aux_desc if FORWARD else TensorDescriptor.from_tensor(aux_out, [BLOCK_SIZE_M, BLOCK_SIZE_N])
    # The Triton signature wants a pointer even where the kernel never loads it, so an unused scalar
    # arg gets a per-call scratch element (never a cached global: see _lrs_stage_cache).
    assert use_fp8 == (dequant_scale_ptr is not None)
    assert post_scale is not None or not (FORWARD and emit_f8)
    unused_ptr = torch.empty(1, dtype=torch.float32, device=a.device)
    if not (FORWARD and rot_t):
        post_t_scale = post_t_amax = None
    dequant_scale_ptr, post_scale, post_amax, dpre_scale, dpre_amax, hadamard_q_ptr, post_t_scale, post_t_amax = [
        unused_ptr if p is None else p
        for p in (dequant_scale_ptr, post_scale, post_amax, dpre_scale, dpre_amax,
                  hadamard_q if hb else None, post_t_scale, post_t_amax)]

    key = (FORWARD, use_fp8, emit_f8, emit_t, store_pre, store_post_bf, rot_row, rot_t, hb,
           BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps)
    num_stages = _lrs_stage_cache.get(key, 4 if (FORWARD and not (emit_f8 or emit_t)) else 3)
    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)), )

    def launch(stages):
        linear_relu_square_kernel[grid](
            a_desc, b_desc, c_desc, aux_desc, dequant_scale_ptr, M, N, K,
            aux_f8_desc, post_scale, post_amax, aux_t_desc,
            c_f8_desc, c_t_desc, dpre_scale, dpre_amax, aux_w_desc, hadamard_q_ptr, post_t_scale, post_t_amax,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K, NUM_SMS=NUM_SMS,
            FORWARD=FORWARD, USE_FP8=use_fp8,
            EMIT_F8=aux_f8 is not None, EMIT_T=aux_t is not None,
            STORE_PRE=FORWARD and store_pre, STORE_POST_BF=FORWARD and store_post_bf,
            ROT_ROW=rot_row, ROT_T=rot_t, HB=max(hb, 16),
            num_stages=stages, num_warps=num_warps,
        )

    if torch.compiler.is_compiling():
        launch(num_stages)  # the retry below cannot run under compile: unprimed variants take the default
    else:
        while True:
            try:
                launch(num_stages)
                break
            except OutOfResources:  # this variant overflows shared memory: shorten the pipeline
                if num_stages <= 1:
                    raise
                num_stages -= 1
        _lrs_stage_cache[key] = num_stages
    return (c, aux_out, aux_f8, aux_t) if FORWARD else (c_f8, c_t)

def prime_stage_cache():
    """Resolve num_stages for every linear_relu_square variant training reaches, by launching each
    once on one-tile inputs (shared-memory pressure depends only on the variant, not on M/N/K).
    Call once, before torch.compile traces the model. Returns {variant key: num_stages}; under
    HADAMARD_FP8 the record's fp8 variants are primed too, so a rotated variant that lost a pipeline
    stage shows up next to the record's count."""
    dev = torch.device("cuda")
    M, N, K = (128, 256, 128)  # one (BLOCK_M, BLOCK_N) tile, one fp8 K-block
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    zeros = lambda fmt, *shape: torch.zeros(*shape, device=dev, dtype=fmt)
    a, b = zeros(torch.bfloat16, M, K), zeros(torch.bfloat16, N, K)
    a_e4, b_e4 = zeros(torch.float8_e4m3fn, M, K), zeros(torch.float8_e4m3fn, N, K)
    g_f8, aux_e4 = zeros(torch.float8_e5m2, M, K), zeros(torch.float8_e4m3fn, M, N)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    amax = torch.zeros(num_sms, dtype=torch.float32, device=dev)
    # Each row mirrors one caller: the bf16 forward (validation, and the FP8_OFF=mlp training forward)
    # and the fp8 forward and backward (training), with the HADAMARD_FP8 rotations HADAMARD_GEMMS selects
    # (forward_rotation / backward_rotation, as FusedLinearReLUSquareFunction calls them).
    fwd = {"a_f8": a_e4, "b_f8": b_e4, "dequant_scale_ptr": one, "emit_f8": True, "emit_t": True,
           "post_scale": one, "post_amax": amax, "store_pre": False, "store_post_bf": False}
    bwd = {"aux": aux_e4, "a_f8": g_f8, "b_f8": b_e4, "dequant_scale_ptr": one,
           "post_scale": one, "dpre_scale": one, "dpre_amax": amax}
    variants = [{}, fwd, bwd]
    if HADAMARD_FP8:
        hq = signed_hadamard(HADAMARD_BLOCK, HADAMARD_SEED, device=dev)
        variants += [dict(fwd, **forward_rotation(hq), post_t_scale=one, post_t_amax=amax),
                     dict(bwd, **backward_rotation(hq))]
    for kwargs in variants:
        linear_relu_square(a, b, **kwargs)
    torch.cuda.synchronize(dev)
    return dict(_lrs_stage_cache)


# -----------------------------------------------------------------------------
# HADAMARD_GEMMS: which operands the Function rotates (config.py). A GEMM not selected reads the record's
# unrotated operands on both sides: no rotation dot, no extra kernel, no extra weight-cache write.
#   dW2   post_t (forward epilogue ROT_T) and g_t      dpost  g_row and the W2 row cache (gpt.py)
#   dW1   dpre_t (backward ROT_T) and x_t (gpt.py)     dx     dpre row (backward ROT_ROW) and the W1 col cache

def forward_rotation(hadamard_q):
    """linear_relu_square kwargs of the fp8 forward: rotate post_t only when dW2 is selected."""
    return {"hadamard_q": hadamard_q, "rot_t": True} if (hadamard_q is not None and HADAMARD_DW2) else {}

def backward_rotation(hadamard_q):
    """linear_relu_square kwargs of the fp8 backward: dpre row for dx, dpre_t for dW1."""
    if hadamard_q is None or not (HADAMARD_DX or HADAMARD_DW1):
        return {}
    return {"hadamard_q": hadamard_q, "rot_row": HADAMARD_DX, "rot_t": HADAMARD_DW1}

@torch.library.custom_op("nanogpt::hadamard_g_quant", mutates_args=())
def hadamard_g_quant(src: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, emit_row: bool, rot_row: bool,
                     rot_t: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The incoming gradient when only one of its two GEMMs is rotated, at the static e5m2 `scale`, one
    pass over the bf16 g: (row [M, N], rotated along N iff rot_row; or an empty tensor without emit_row)
    and t [N, M], rotated along M iff rot_t."""
    assert src.ndim == 2 and src.stride(1) == 1 and src.dtype == torch.bfloat16
    assert q.shape == (HADAMARD_BLOCK, HADAMARD_BLOCK) and q.dtype == torch.bfloat16 and q.is_contiguous()
    M, N = src.shape
    assert M % HADAMARD_TILE == 0 and N % HADAMARD_TILE == 0, f"{tuple(src.shape)} not a multiple of {HADAMARD_TILE}"
    t = torch.empty((N, M), device=src.device, dtype=torch.float8_e5m2)
    row = torch.empty((M, N) if emit_row else (0,), device=src.device, dtype=torch.float8_e5m2)
    _hadamard_quant_dual_kernel[(M // HADAMARD_TILE, N // HADAMARD_TILE)](
        src, q, row if emit_row else t, t, scale, scale, scale, M, N, src.stride(0),
        EMIT_ROW=emit_row, ROT_ROW=rot_row, ROT_T=rot_t, RECORD_T_AMAX=False, E5M2=True,
        BLOCK=HADAMARD_TILE, HB=HADAMARD_BLOCK, num_warps=8)
    return row, t

@hadamard_g_quant.register_fake
def _(src, q, scale, emit_row, rot_row, rot_t):
    fmt = torch.float8_e5m2
    return src.new_empty(src.shape if emit_row else (0,), dtype=fmt), src.new_empty((src.shape[1], src.shape[0]), dtype=fmt)

def quantize_grad(g, hadamard_q, g_scale):
    """(g_row for the dpost GEMM, g_t for dW2) in e5m2 at the static g_scale, each rotated iff its GEMM is.
      neither   the record's quantize_dual_layout_fused (row fused into g's producer, fp8 transpose)
      both      hadamard_dual_quant (one pass, both rotated)
      dW2 only  the record's fused row, and g_t rotated from the bf16 g (one kernel, no row written)
      dpost     g_row rotated and g_t plain, one pass over the bf16 g (the plain t rounds the stored bf16
                g, where the record rounds the producer's fp32: at most one rounding apart)."""
    rot_row = hadamard_q is not None and HADAMARD_DPOST
    rot_t = hadamard_q is not None and HADAMARD_DW2
    if not (rot_row or rot_t):
        return quantize_dual_layout_fused(g, g_scale, fmt=torch.float8_e5m2)
    if rot_row and rot_t:
        return hadamard_dual_quant(g, hadamard_q, g_scale, True)
    if rot_t:
        row = torch.clamp(g.to(torch.float32) * (1.0 / g_scale), -57344.0, 57344.0).to(torch.float8_e5m2)
        return row, hadamard_g_quant(g, hadamard_q, g_scale, False, False, True)[1]
    return hadamard_g_quant(g, hadamard_q, g_scale, True, True, False)


# -----------------------------------------------------------------------------
# The four fp8 _scaled_mm GEMMs, each its own custom op so it stays opaque to inductor and shows up
# by name in traces. They differ only in which operand is transposed and in use_fast_accum (False
# on every gradient GEMM). Output is bf16 to match the bank grad dtype; fp32 would double the
# reduce_scatter volume.

def _f8_mm_op(name, transpose_b, fast_accum):
    def _fn(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
        return torch._scaled_mm(a, b.T if transpose_b else b, out_dtype=torch.bfloat16, scale_a=a_scale,
                                scale_b=b_scale, use_fast_accum=fast_accum)
    _fn.__name__ = name + "_op"
    op = torch.library.custom_op("nanogpt::" + name, _fn, mutates_args=())
    op.register_fake(lambda a, b, a_scale, b_scale:
                     a.new_empty((a.shape[0], b.shape[0] if transpose_b else b.shape[1]), dtype=torch.bfloat16))
    return op

dp_f8_op = _f8_mm_op("dp_f8", transpose_b=False, fast_accum=True)     # x3  = post_f8 @ W2_f8 (down projection)
wg2_f8_op = _f8_mm_op("wg2_f8", transpose_b=True, fast_accum=False)   # dW2 = post_t_f8 @ g_t_f8.T
wg1_f8_op = _f8_mm_op("wg1_f8", transpose_b=True, fast_accum=False)   # dW1 = dpre_t_f8 @ x_t_f8.T
dx_f8_op = _f8_mm_op("dx_f8", transpose_b=False, fast_accum=False)    # dx  = dpre_f8 @ w1_f8_col


class FusedLinearReLUSquareFunction(torch.autograd.Function):
    """relu(x @ W1.T)^2 @ W2 as one autograd op. x is the [T, 768] bf16 normed residual; W1/W2 are
    [H, 768] bf16 bank slices. Called with only (x, W1, W2) it runs the bf16 forward (validation, under
    no_grad: it has no backward). Training passes the fp8 caches and scales too (see GPT._mlp_fp8_args),
    all of them:
      W1_f8 / w1_f8_col   W1 in e4m3, row-major and column-major
      W2_f8 / w2_f8_row   W2 in e4m3, column-major and row-major
      x_f8 / x_f8_t       the input in e4m3 and its transpose
      dequant_scale       x_scale * w1_scale;  dq_bwd = w2_scale * g_scale
      post_scale/amax, dpre_scale/amax  the delayed activation scales and their amax slots
      hadamard_q          HADAMARD_FP8 only (else None): the bf16 Q of perf/kernels/hadamard.py. The gradient
                          GEMMs in HADAMARD_GEMMS then read rotated operands; the caller passes the rotated
                          weight caches (w2_f8_row iff dpost, w1_f8_col iff dx), their scales, and the
                          rotated x_t with its scale (iff dw1).
      p_fold              the residual post-lambda folded into the down-projection scale (w2_scale
                          and dq_bwd), or None. It rides along only so its gradient can be returned.
      post_t_scale/amax   HADAMARD_FP8 only (gpt.py passes p_fold positionally, possibly None, then these):
                          the rotated post_t's own delayed scale and amax slot, used iff dw2 is selected.
                          post_scale/post_amax stay the unrotated post's, as in the record.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, W1_f8=None, dequant_scale=None, x_f8=None, W2_f8=None, w2_scale=None,
                post_scale=None, post_amax=None, w2_f8_row=None, dq_bwd=None, x_f8_t=None, x_scale=None,
                w1_f8_col=None, w1_scale=None, g_scale=None, dpre_scale=None, dpre_amax=None, hadamard_q=None,
                p_fold=None, post_t_scale=None, post_t_amax=None):
        x_flat = x.view((-1, x.shape[-1]))
        if W1_f8 is None:  # validation: nothing saved, the backward below is fp8-only
            _, post, _, _ = linear_relu_square(x_flat, W1)
            return (post @ W2).view(x.shape)
        # The fp8 forward emits the e4m3 post in both layouts (the fp8 down projection and dW2) and
        # no bf16 pre or post.
        rot_fwd = forward_rotation(hadamard_q)
        if rot_fwd:
            rot_fwd.update(post_t_scale=post_t_scale, post_t_amax=post_t_amax)
        _, _, post_f8, post_t = linear_relu_square(
            x_flat, W1, a_f8=x_f8.view((-1, x_f8.shape[-1])), b_f8=W1_f8, emit_f8=True, emit_t=True,
            dequant_scale_ptr=dequant_scale, post_scale=post_scale, post_amax=post_amax,
            store_pre=False, store_post_bf=False, **rot_fwd)
        x3 = torch.ops.nanogpt.dp_f8(post_f8, W2_f8, post_scale, w2_scale)
        # Intermediates go through save_for_backward: stashing them on ctx inside the compiled
        # subgraph pins them past their last use (+3.3 MB). The optional p_fold is trace-time constant.
        ctx.save_for_backward(*((x, W1, W2) + ((p_fold,) if p_fold is not None else ())))
        # Persistent caches and scales are kept by name.
        ctx.fold_p = p_fold is not None
        ctx.post_f8, ctx.post_t, ctx.post_scale = post_f8, post_t, post_scale
        ctx.post_t_scale = post_t_scale if rot_fwd else post_scale  # dW2's dequant of post_t
        ctx.w2_f8_row, ctx.dq_bwd, ctx.x_f8_t, ctx.x_scale = w2_f8_row, dq_bwd, x_f8_t, x_scale
        ctx.w1_f8_col, ctx.w1_scale, ctx.g_scale = w1_f8_col, w1_scale, g_scale
        ctx.dpre_scale, ctx.dpre_amax = dpre_scale, dpre_amax
        ctx.hadamard_q = hadamard_q
        if FP8_DUMP.active():
            ctx.dump_tag = FP8_DUMP.tag
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        # No star-unpack of saved_tensors and no bool-as-index: keeps dynamo happy.
        st = ctx.saved_tensors
        x, W2 = st[0], st[2]
        p_fold = st[3] if ctx.fold_p else None
        g_flat = grad_output.view((-1, grad_output.shape[-1]))

        if FP8_DUMP.active() and getattr(ctx, "dump_tag", None) is not None:
            FP8_DUMP.save_mlp(ctx.dump_tag, x, st[1], W2, g_flat, p_fold, x_scale=ctx.x_scale, g_scale=ctx.g_scale,
                              post_scale=ctx.post_scale, dpre_scale=ctx.dpre_scale)
        # 1. Quantize the incoming gradient (both layouts), then dW2 = post^T @ g. Under HADAMARD_FP8 a
        # layout is rotated iff its GEMM is selected: g_f8 along 768 (dpost), g_f8_t along tokens (dW2).
        g_f8, g_f8_t = quantize_grad(g_flat, ctx.hadamard_q, ctx.g_scale)
        dW2 = torch.ops.nanogpt.wg2_f8(ctx.post_t, g_f8_t, ctx.post_t_scale, ctx.g_scale)

        # 2. Post-lambda folded into the down-projection scale: the forward computed out = post @ (p * W2),
        # so dW2 above is d/d(p*W2). Then dL/dW2 = p * dW2 and dL/dp = <dW2, W2>, an [H, 768] dot that
        # replaces two [T, 768] passes at the residual add.
        grad_p = None
        if ctx.fold_p:
            grad_p = (dW2.float() * W2.float()).sum().to(p_fold.dtype)
            dW2 = dW2 * p_fold

        # 3. dpre in e5m2 (both layouts, from the kernel epilogue), then dW1 and dx.
        dpre_f8, dpre_t = linear_relu_square(
            g_flat, W2, aux=ctx.post_f8, a_f8=g_f8, b_f8=ctx.w2_f8_row,
            dequant_scale_ptr=ctx.dq_bwd, post_scale=ctx.post_scale,
            dpre_scale=ctx.dpre_scale, dpre_amax=ctx.dpre_amax, **backward_rotation(ctx.hadamard_q))
        dW1 = torch.ops.nanogpt.wg1_f8(dpre_t, ctx.x_f8_t, ctx.dpre_scale, ctx.x_scale)
        dx = torch.ops.nanogpt.dx_f8(dpre_f8, ctx.w1_f8_col, ctx.dpre_scale, ctx.w1_scale)

        # One slot per forward input: (x, W1, W2), 16 fp8 args, hadamard_q, p_fold, post_t_scale/amax
        # (autograd drops trailing None slots of inputs that were not passed).
        return (dx.view(x.shape), dW1, dW2) + (None,) * 17 + (grad_p, None, None)


class BF16LinearReLUSquareFunction(torch.autograd.Function):
    """FP8_OFF=mlp: relu(x @ W1.T)^2 @ W2 for training with every GEMM in bf16 (fp32 accumulate).

    The upper-bound arm of the bf16-vs-fp8 test (182/hadamard-fp8): the same forward kernel validation
    runs, and the backward the fp8 path computes, with bf16 operands instead of fp8 ones:
      dW2 = post.T @ g,  dpre = 2 (g @ W2.T) relu(pre),  dW1 = dpre.T @ x,  dx = dpre @ W1.
    The post-lambda is not folded (the caller applies it, as validation does). Speed is not a goal.
    """
    @staticmethod
    def forward(ctx, x, W1, W2):
        x_flat = x.view((-1, x.shape[-1]))
        pre, post, _, _ = linear_relu_square(x_flat, W1)
        ctx.save_for_backward(x, W1, W2, pre)
        if FP8_DUMP.active():
            ctx.dump_tag = FP8_DUMP.tag
        return (post @ W2).view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        st = ctx.saved_tensors
        x, W1, W2, pre = st[0], st[1], st[2], st[3]
        x_flat = x.view((-1, x.shape[-1]))
        g = grad_output.reshape((-1, grad_output.shape[-1]))
        if FP8_DUMP.active() and getattr(ctx, "dump_tag", None) is not None:
            FP8_DUMP.save_mlp(ctx.dump_tag, x, W1, W2, g, None)
        relu_pre = torch.relu(pre)
        dW2 = (relu_pre * relu_pre).T @ g
        dpre = (2.0 * (g @ W2.T).float() * relu_pre.float()).to(pre.dtype)
        dW1 = dpre.T @ x_flat
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2
