"""Block-Hadamard-rotated fp8 quantize kernels for the MLP gradient GEMMs (HADAMARD_FP8=1).

What it adds: the activation / gradient quantizes and the weight-cache quantizes of the fp8 MLP, with
the operand rotated along the contraction dim of the GEMM that reads it (perf/hadamard_ref.py has the
math, the convention and the pure-torch reference these kernels must match). The rotation is
block-diagonal in HB = HADAMARD_BLOCK blocks (see "Block size" below): a tl.dot of the bf16 tile with
Q = diag(signs) @ H_HB (exactly +-1 in bf16) in fp32 accumulation, times 1 / sqrt(HB), done on the tile
the quantize already holds -- one read, the same writes as the unrotated path.

Which operand is rotated along which dim (MLP: pre = x @ W1.T, post = relu(pre)^2, out = post @ W2):
  dW2  = post.T @ g     k = tokens  post_t (forward epilogue, mlp.py), g_t   (hadamard_dual_quant)
  dpost = g @ W2.T      k = 768     g_row  (hadamard_dual_quant),      W2 row cache (quantize_weights_rotated)
  dW1  = dpre.T @ x     k = tokens  dpre_t (backward epilogue),        x_t   (hadamard_t_quant)
  dx   = dpre @ W1      k = hidden  dpre row (backward epilogue),      W1 col cache (quantize_weights_rotated)
The forward GEMMs (x @ W1.T, post @ W2) and their operands (x row, W1 row, post row, W2 col) are
untouched.

Scales: g stays at the static e5m2 2^-6 (a rotation preserves each block's L2 norm, and e5m2 has the
range); x_t gets its own delayed scale from the rotated amax (a static 2^-4 is NOT safe once a
channel's tokens are summed into one coefficient); the rotated post_t gets its own delayed scale
(post_t_scale / post_t_amax, model/gpt.py) so the forward's unrotated post keeps the record's scale;
dpre keeps one delayed scale whose amax covers both rotated layouts; the rotated weight caches get
their own exact-then-delayed scales.

Block size (HADAMARD_BLOCK in config.py, 32 | 64 | 128): every kernel keeps HADAMARD_TILE = 128 wide tiles
and rotates each tile block-diagonally: the tile is reshaped so each length-b block is one row, and one
tl.dot with the b x b Q (K = b >= 16, as wgmma needs for bf16) costs 2b FLOP per element instead of 256.
That is exactly hadamard_ref.rotate with the b x b Q, since the 128-tiles start at multiples of b.
Why not a factored full-128 rotation (H_128 = H_16 (x) H_8: one K=16 dot plus 3 butterfly stages): a
butterfly pairs elements 8, 4, 2 apart along the rotated dim, which in Triton is reshape -> permute ->
tl.split -> add/sub -> tl.join -> permute per stage. On an MMA-layout register tile those permutes lower to
convert_layout round trips through shared memory (a 128 x 128 fp32 tile is 64 KB, in an epilogue that is
already at the 227 KB limit), and the H_8 factor cannot be a dot instead (K = 8 < 16). Expected cost is
above the dense H_16 dot it saves, so only the block-diagonal variants exist.

Invariants: token counts, the hidden size and the model dim are multiples of HADAMARD_TILE (a partial
tile would rotate padding into dropped lanes); every cast is preceded by an explicit clamp.
"""
import torch
import triton
import triton.language as tl

from track_1_short.config import HADAMARD_BLOCK  # the rotation block, Q is [HADAMARD_BLOCK, HADAMARD_BLOCK]

HADAMARD_TILE = 128  # every kernel's tile edge; the rotation is block-diagonal inside it
# The rotated weight caches keep the unrotated caches' one-step-delayed scale headroom (mlp.py).
ROTATED_WEIGHT_SCALE_HEADROOM = 1.12


@triton.jit
def _load_q(q_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    return tl.load(q_ptr + offs[:, None] * BLOCK + offs[None, :])


@triton.jit
def _inv_sqrt(HB: tl.constexpr):
    if HB == 32:
        return 0.17677669529663687
    elif HB == 64:
        return 0.125
    else:
        return 0.08838834764831845


@triton.jit
def rotate_last(x, q, R: tl.constexpr, C: tl.constexpr, HB: tl.constexpr):
    """x [R, C] (bf16) rotated along its last dim in blocks of HB: each length-HB slice v -> v @ Q / sqrt(HB),
    fp32. HB < C reshapes so each block is one row of a [R * C / HB, HB] dot operand (row-major order is
    kept, so block j of row r is row r * C / HB + j). Shared by mlp.py's epilogues.
    x is cast to Q's bf16 first: tl.maximum promotes bf16 to fp32, so the forward's relu(pre)^2 arrives in
    fp32, and tl.dot asserts equal operand dtypes (a no-op cast where x already is bf16)."""
    x = x.to(q.dtype)
    if HB == C:
        y = tl.dot(x, q)
    else:
        y = tl.reshape(tl.dot(tl.reshape(x, (R * C // HB, HB)), q), (R, C))
    return y * _inv_sqrt(HB)


@triton.jit
def _quant(v, inv_scale, E5M2: tl.constexpr):
    if E5M2:
        return tl.maximum(tl.minimum(v * inv_scale, 57344.0), -57344.0).to(tl.float8e5)
    else:
        return tl.maximum(tl.minimum(v * inv_scale, 448.0), -448.0).to(tl.float8e4nv)


@triton.jit
def _hadamard_quant_dual_kernel(src_ptr, q_ptr, row_ptr, t_ptr, row_scale_ptr, t_scale_ptr, t_amax_ptr,
                                M, N, stride_m,
                                EMIT_ROW: tl.constexpr, ROT_ROW: tl.constexpr, ROT_T: tl.constexpr,
                                RECORD_T_AMAX: tl.constexpr, E5M2: tl.constexpr, BLOCK: tl.constexpr,
                                HB: tl.constexpr):
    """src [M, N] bf16 -> row [M, N] (rotated along N if ROT_ROW) and t [N, M] (rotated along M if ROT_T).
    BLOCK is the tile edge, HB the rotation block (Q is [HB, HB])."""
    offs_m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tile = tl.load(src_ptr + offs_m[:, None].to(tl.int64) * stride_m + offs_n[None, :])
    if ROT_ROW or ROT_T:
        q = _load_q(q_ptr, HB)
    if EMIT_ROW:
        if ROT_ROW:
            y = rotate_last(tile, q, BLOCK, BLOCK, HB)
        else:
            y = tile.to(tl.float32)
        tl.store(row_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :],
                 _quant(y, 1.0 / tl.load(row_scale_ptr), E5M2))
    if ROT_T:
        yt = rotate_last(tl.trans(tile), q, BLOCK, BLOCK, HB)  # [BLOCK n, BLOCK m]
    else:
        yt = tl.trans(tile.to(tl.float32))
    tl.store(t_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], _quant(yt, 1.0 / tl.load(t_scale_ptr), E5M2))
    if RECORD_T_AMAX:
        tl.atomic_max(t_amax_ptr, tl.max(tl.max(tl.abs(yt), axis=1), axis=0))


def _check_2d(src: torch.Tensor, q: torch.Tensor):
    assert src.ndim == 2 and src.stride(1) == 1 and src.dtype == torch.bfloat16
    assert q.shape == (HADAMARD_BLOCK, HADAMARD_BLOCK) and q.dtype == torch.bfloat16 and q.is_contiguous()
    M, N = src.shape
    assert M % HADAMARD_TILE == 0 and N % HADAMARD_TILE == 0, f"{tuple(src.shape)} not a multiple of {HADAMARD_TILE}"
    return M, N


@torch.library.custom_op("nanogpt::hadamard_dual_quant", mutates_args=())
def hadamard_dual_quant(src: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, e5m2: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(row [M, N] rotated along N, t [N, M] rotated along M), both at the one static `scale`. The MLP's
    incoming gradient: g_row feeds the dpre GEMM (k = 768), g_t feeds dW2 (k = tokens)."""
    M, N = _check_2d(src, q)
    fmt = torch.float8_e5m2 if e5m2 else torch.float8_e4m3fn
    row = torch.empty((M, N), device=src.device, dtype=fmt)
    t = torch.empty((N, M), device=src.device, dtype=fmt)
    _hadamard_quant_dual_kernel[(M // HADAMARD_TILE, N // HADAMARD_TILE)](
        src, q, row, t, scale, scale, scale, M, N, src.stride(0),
        EMIT_ROW=True, ROT_ROW=True, ROT_T=True, RECORD_T_AMAX=False, E5M2=e5m2,
        BLOCK=HADAMARD_TILE, HB=HADAMARD_BLOCK, num_warps=8)
    return row, t

@hadamard_dual_quant.register_fake
def _(src, q, scale, e5m2):
    fmt = torch.float8_e5m2 if e5m2 else torch.float8_e4m3fn
    return src.new_empty(src.shape, dtype=fmt), src.new_empty((src.shape[1], src.shape[0]), dtype=fmt)


@torch.library.custom_op("nanogpt::hadamard_t_quant", mutates_args=())
def hadamard_t_quant(src: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(t [N, M] e4m3 = quant(src.T rotated along M) at the delayed `scale`, amax [1] f32 = max |rotated|).
    The MLP input's transposed copy, read only by dW1.

    Functional: the amax comes back as a fresh tensor (atomic-maxed by the kernel into its own zeroed
    element) instead of being written through a view of the caller's buffer. The caller folds it into
    its delayed-amax buffer with aten ops (GPT._quantize_mlp_input), so torch.compile sees an ordinary
    buffer mutation, which functionalization and the captured step graphs handle like any other."""
    M, N = _check_2d(src, q)
    t = torch.empty((N, M), device=src.device, dtype=torch.float8_e4m3fn)
    amax = torch.zeros(1, device=src.device, dtype=torch.float32)
    _hadamard_quant_dual_kernel[(M // HADAMARD_TILE, N // HADAMARD_TILE)](
        src, q, t, t, scale, scale, amax, M, N, src.stride(0),
        EMIT_ROW=False, ROT_ROW=False, ROT_T=True, RECORD_T_AMAX=True, E5M2=False,
        BLOCK=HADAMARD_TILE, HB=HADAMARD_BLOCK, num_warps=8)
    return t, amax

@hadamard_t_quant.register_fake
def _(src, q, scale):
    return src.new_empty((src.shape[1], src.shape[0]), dtype=torch.float8_e4m3fn), src.new_empty((1,), dtype=torch.float32)

@hadamard_t_quant.register_kernel("cpu")
def _(src, q, scale):
    """CPU reference (tests only; training is CUDA): the same math in torch, perf/hadamard_ref.py."""
    from track_1_short.perf.hadamard_ref import FP8_MAX, rotate
    _check_2d(src, q)
    r = rotate(src.float(), 0, q.float())
    lim = FP8_MAX[torch.float8_e4m3fn]
    t = torch.clamp(r.T / scale, -lim, lim).to(torch.float8_e4m3fn).contiguous()
    return t, r.abs().amax().reshape(1)


# -----------------------------------------------------------------------------
# Rotated weight caches: one read of a [L, H, D] bank slice writes the one rotated layout a gradient
# GEMM reads, and each 128 x 128 tile's rotated amax (the next call's delayed scale). gpt.py runs it only
# for the caches of selected GEMMs (HADAMARD_GEMMS: W1 col for dx, W2 row for dpost).

@triton.jit
def _quantize_weights_rotated_kernel(w_ptr, q_ptr, out_ptr, scale_ptr, partial_amax_ptr,
                                     w_stride_l, w_stride_h, w_stride_d,
                                     H: tl.constexpr, D: tl.constexpr, num_tiles_d: tl.constexpr,
                                     num_tiles: tl.constexpr, ALONG_H: tl.constexpr, BLOCK: tl.constexpr,
                                     HB: tl.constexpr):
    layer = tl.program_id(0)
    tile = tl.program_id(1)
    offs_h = (tile // num_tiles_d) * BLOCK + tl.arange(0, BLOCK)
    offs_d = (tile % num_tiles_d) * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(w_ptr + layer * w_stride_l + offs_h[:, None] * w_stride_h + offs_d[None, :] * w_stride_d)
    q = _load_q(q_ptr, HB)
    inv_s = 1.0 / tl.load(scale_ptr + layer)
    if ALONG_H:
        # W1 for dx = dpre @ W1 (k = hidden): out [L, D, H] contiguous, i.e. the column-major cache.
        y = rotate_last(tl.trans(w), q, BLOCK, BLOCK, HB)  # [BLOCK d, BLOCK h]
        tl.store(out_ptr + layer * H * D + offs_d[:, None] * H + offs_h[None, :], _quant(y, inv_s, False))
    else:
        # W2 for dpost = g @ W2.T (k = model dim): out [L, H, D] row-major.
        y = rotate_last(w, q, BLOCK, BLOCK, HB)            # [BLOCK h, BLOCK d]
        tl.store(out_ptr + layer * H * D + offs_h[:, None] * D + offs_d[None, :], _quant(y, inv_s, False))
    tl.store(partial_amax_ptr + layer * num_tiles + tile, tl.max(tl.max(tl.abs(y), axis=1), axis=0))


def quantize_weights_rotated_ntiles(H: int, D: int) -> int:
    return triton.cdiv(H, HADAMARD_TILE) * triton.cdiv(D, HADAMARD_TILE)


def quantize_weights_rotated(bank: torch.Tensor, q: torch.Tensor, scales: torch.Tensor, partial_amax: torch.Tensor,
                             out: torch.Tensor, along_h: bool, update_scales: bool):
    """Quantize bank [L, H, D] (bf16) rotated along H (out = [L, D, H] contiguous) or along D (out =
    [L, H, D] contiguous). With update_scales, `scales` [L] are first refreshed from the previous call's
    per-tile rotated amaxes, as quantize_mlp_weights_dual does for the unrotated caches."""
    L, H, D = bank.shape
    assert H % HADAMARD_TILE == 0 and D % HADAMARD_TILE == 0
    assert q.shape == (HADAMARD_BLOCK, HADAMARD_BLOCK)
    assert out.is_contiguous() and out.dtype == torch.float8_e4m3fn
    assert out.shape == ((L, D, H) if along_h else (L, H, D))
    num_tiles_d = D // HADAMARD_TILE
    num_tiles = (H // HADAMARD_TILE) * num_tiles_d
    assert partial_amax.shape == (L, num_tiles)
    if update_scales:
        torch.clamp(partial_amax.amax(dim=1) * (ROTATED_WEIGHT_SCALE_HEADROOM / 448.0), min=1e-12, out=scales)
    _quantize_weights_rotated_kernel[(L, num_tiles)](
        bank, q, out, scales, partial_amax, bank.stride(0), bank.stride(1), bank.stride(2),
        H=H, D=D, num_tiles_d=num_tiles_d, num_tiles=num_tiles, ALONG_H=along_h, BLOCK=HADAMARD_TILE,
        HB=HADAMARD_BLOCK, num_warps=8, num_stages=1)
