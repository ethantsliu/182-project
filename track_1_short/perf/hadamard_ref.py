"""Block-Hadamard rotation for fp8 GEMM operands: the pure-torch reference (no Triton, runs on CPU).

A GEMM C = A @ B contracts over k. For any orthogonal R (R @ R.T = I), (A @ R) @ (R.T @ B) = A @ B,
so both operands may be quantized in a rotated basis without changing the exact product. With R a
block-diagonal of b x b blocks D_s H_b / sqrt(b) (H_b the Sylvester Hadamard matrix, D_s a fixed
random +-1 diagonal), the rotation spreads each block's outliers over b entries, which shrinks the
amax relative to the rms and so the fp8 rounding error at a fixed per-tensor scale.

Convention used everywhere (this file, perf/kernels/hadamard.py, the fp8 MLP): every operand is
rotated by treating its contraction-dim vectors as ROW vectors and right-multiplying by R. For A [M, K]
that is A @ R; for B [K, N] it is (B.T @ R).T = R.T @ B. Both operands of one GEMM therefore apply the
same transform, `rotate(t, dim=<contraction dim>)`, and the k-blocks start at index 0 on both.

Q = D_s @ H_b holds exactly +-1, so it is exact in bf16: the Triton kernels feed bf16 Q to tl.dot with
fp32 accumulation and multiply by 1 / sqrt(b) after, which `rotate` reproduces with the same Q.
"""
import math

import torch

FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}


def hadamard_matrix(n: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """The n x n Sylvester Hadamard matrix (+-1 entries, H @ H.T = n I, symmetric). n a power of 2."""
    assert n >= 1 and n & (n - 1) == 0, f"Hadamard block {n} is not a power of 2"
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h


def random_signs(n: int, seed: int, device=None) -> torch.Tensor:
    """A fixed +-1 vector of length n from `seed` (CPU generator, so it is the same on every rank)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float32).to(device)


def signed_hadamard(block: int, seed: int, dtype=torch.bfloat16, device=None) -> torch.Tensor:
    """Q = diag(signs) @ H_block, entries exactly +-1. The rotation block is Q / sqrt(block)."""
    return (random_signs(block, seed)[:, None] * hadamard_matrix(block)).to(dtype=dtype, device=device)


def rotation_block(block: int, seed: int, dtype=torch.float64) -> torch.Tensor:
    """R_b = Q / sqrt(block), orthogonal."""
    return signed_hadamard(block, seed, dtype=dtype) / math.sqrt(block)


def rotate(t: torch.Tensor, dim: int, q: torch.Tensor) -> torch.Tensor:
    """Rotate `t` along `dim` in blocks of q.shape[0]: each length-b slice v becomes v @ Q / sqrt(b).

    Computes in fp32 (fp64 if t is fp64); t.shape[dim] must be a multiple of the block.
    """
    b = q.shape[0]
    dim = dim % t.ndim
    n = t.shape[dim]
    assert n % b == 0, f"dim {dim} of size {n} is not a multiple of the Hadamard block {b}"
    work = torch.float64 if t.dtype == torch.float64 else torch.float32
    moved = t.movedim(dim, -1).to(work)
    blocks = moved.reshape(*moved.shape[:-1], n // b, b)
    out = (blocks @ q.to(work)) * (1.0 / math.sqrt(b))
    return out.reshape(moved.shape).movedim(-1, dim)


def unrotate(t: torch.Tensor, dim: int, q: torch.Tensor) -> torch.Tensor:
    """Inverse of `rotate` (R is orthogonal: v = (v @ R) @ R.T)."""
    b = q.shape[0]
    dim = dim % t.ndim
    work = torch.float64 if t.dtype == torch.float64 else torch.float32
    moved = t.movedim(dim, -1).to(work)
    blocks = moved.reshape(*moved.shape[:-1], moved.shape[-1] // b, b)
    out = (blocks @ q.to(work).T) * (1.0 / math.sqrt(b))
    return out.reshape(moved.shape).movedim(-1, dim)


def rotated_matmul(a: torch.Tensor, b: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(A @ R) @ (R.T @ B) for A [M, K], B [K, N]: equals A @ B up to rounding."""
    return rotate(a, -1, q) @ rotate(b, 0, q)


# -----------------------------------------------------------------------------
# fp8 quantization with the repo's per-tensor scale conventions: value = fp8 * scale, the cast is
# preceded by an explicit saturating clamp (e4m3fn overflow is NaN, not inf).

def quantize(t: torch.Tensor, scale: float | torch.Tensor, fmt: torch.dtype) -> torch.Tensor:
    lim = FP8_MAX[fmt]
    return torch.clamp(t.to(torch.float32) / scale, -lim, lim).to(fmt)


def dequantize(q: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def fake_quant(t: torch.Tensor, scale, fmt: torch.dtype) -> torch.Tensor:
    return dequantize(quantize(t, scale, fmt), scale)


def amax_scale(t: torch.Tensor, fmt: torch.dtype, headroom: float = 1.0, floor: float = 1e-12) -> float:
    """amax * headroom / fmt_max: the repo's exact (headroom 1) and delayed (headroom > 1, from last
    step's amax) rules. Offline there is no previous step, so 'delayed' uses this tensor's own amax."""
    return max(float(t.abs().max()) * headroom / FP8_MAX[fmt], floor)


def quantized_rotated_matmul(a, b, q, a_scale, b_scale, a_fmt, b_fmt):
    """The fp8 GEMM the HADAMARD_FP8 path runs, in fp32: dequant(quant(A R)) @ dequant(quant(R.T B))."""
    return fake_quant(rotate(a, -1, q), a_scale, a_fmt) @ fake_quant(rotate(b, 0, q), b_scale, b_fmt)


def rotate_in_tiles(t: torch.Tensor, q: torch.Tensor, tile: int = 128) -> torch.Tensor:
    """What the kernels' rotate_last does to a [R, tile] register tile: reshape so each length-b block is a
    row, one [R * tile / b, b] @ [b, b] product, reshape back. Equals rotate(t, -1, q) for every b | tile."""
    R, C = t.shape
    b = q.shape[0]
    assert C == tile and tile % b == 0
    work = torch.float64 if t.dtype == torch.float64 else torch.float32
    return ((t.to(work).reshape(R * C // b, b) @ q.to(work)) * (1.0 / math.sqrt(b))).reshape(R, C)


# -----------------------------------------------------------------------------
# The fp8 MLP (perf/kernels/mlp.py FusedLinearReLUSquareFunction) under HADAMARD_GEMMS, in fp32: every
# operand fake-quantized the way its kernel quantizes it. GEMMs not in `gemms` read the record's operands.

MLP_GEMMS = frozenset({"dw1", "dw2", "dx", "dpost"})


def mlp_fp8_reference(x, W1, W2, g, q, gemms, s):
    """x [T, D], W1 / W2 [H, D] (out = relu(x W1^T)^2 W2), g = dL/dout [T, D]; q the b x b Q; s the scales:
    x, w1, w2 (the plain caches), post, post_t, g, dpre, xt, w1_col, w2_row. Returns a dict with out, dx,
    dW1, dW2 and the amaxes the kernels record for the next step's delayed scales:
      post_amax    max relu(pre)^2 over the UNROTATED post only (the record's: the forward is never rotated)
      post_t_amax  max |rotated post_t| (None unless dw2)
      dpre_amax    max over the dpre layouts actually stored (rotated or not; one dpre scale serves both)."""
    assert gemms <= MLP_GEMMS
    e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
    f = lambda t: t.float()
    xq, w1q = fake_quant(x, s["x"], e4), fake_quant(W1, s["w1"], e4)
    pre = (xq @ w1q.T).bfloat16().float()          # the kernel rounds the dequantized accumulator to bf16
    post = torch.relu(pre) ** 2                       # fp32 (tl.maximum promotes the bf16 pre)
    post_f8 = fake_quant(post, s["post"], e4)
    out = post_f8 @ fake_quant(W2, s["w2"], e4)
    res = dict(out=out, post_amax=post.max(), post_t_amax=None)
    # dW2 = post^T g, k = tokens
    if "dw2" in gemms:
        post_rot = rotate(post.bfloat16(), 0, q)      # the epilogue rotates the bf16-cast tile
        res["post_t_amax"] = post_rot.abs().max()
        post_t, g_t = fake_quant(post_rot, s["post_t"], e4), fake_quant(rotate(g, 0, q), s["g"], e5)
    else:
        post_t, g_t = post_f8, fake_quant(g, s["g"], e5)
    res["dW2"] = post_t.T @ g_t
    # dpost = g W2^T, k = model dim; dpre = 2 dpost relu(pre), relu(pre) rebuilt from the e4m3 post
    if "dpost" in gemms:
        g_row, w2_row = fake_quant(rotate(g, 1, q), s["g"], e5), fake_quant(rotate(W2, 1, q), s["w2_row"], e4)
    else:
        g_row, w2_row = fake_quant(g, s["g"], e5), fake_quant(W2, s["w2"], e4)
    dpre = (2.0 * (g_row @ w2_row.T) * torch.sqrt(post_f8)).bfloat16().float()
    # dW1 = dpre^T x (k = tokens), dx = dpre W1 (k = hidden)
    dpre_t = rotate(dpre.bfloat16(), 0, q) if "dw1" in gemms else dpre
    dpre_row = rotate(dpre.bfloat16(), 1, q) if "dx" in gemms else dpre
    res["dpre_amax"] = torch.maximum(dpre_t.abs().max(), dpre_row.abs().max())
    x_t = fake_quant(rotate(x, 0, q), s["xt"], e4) if "dw1" in gemms else xq
    res["dW1"] = fake_quant(dpre_t, s["dpre"], e5).T @ x_t
    w1_col = fake_quant(rotate(W1, 0, q), s["w1_col"], e4) if "dx" in gemms else w1q
    res["dx"] = fake_quant(dpre_row, s["dpre"], e5) @ w1_col
    res.update(post_f8=post_f8, post_t=post_t, dpre=dpre)
    return {k: (f(v) if isinstance(v, torch.Tensor) else v) for k, v in res.items()}


def mlp_reference_scales(x, W1, W2, g, q, gemms):
    """Exact (own-amax) scales for mlp_fp8_reference, with the repo's headrooms; g and x keep their static
    scales, as in training."""
    e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
    pre = (x.float() @ W1.float().T).bfloat16().float()
    post = torch.relu(pre) ** 2
    dpre = 2.0 * (g.float() @ W2.float().T) * torch.relu(pre)
    rot = lambda t, d: rotate(t, d, q)
    return dict(
        x=2.0 ** -4, g=2.0 ** -6,
        w1=amax_scale(W1, e4), w2=amax_scale(W2, e4),
        post=amax_scale(post, e4, 1.03), post_t=amax_scale(rot(post.bfloat16(), 0), e4, 1.03),
        dpre=amax_scale(torch.cat([rot(dpre, 0).flatten(), rot(dpre, 1).flatten(), dpre.flatten()]), e5, 1.25),
        xt=amax_scale(rot(x, 0), e4, 1.10), w1_col=amax_scale(rot(W1, 0), e4), w2_row=amax_scale(rot(W2, 1), e4))
