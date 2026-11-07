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
