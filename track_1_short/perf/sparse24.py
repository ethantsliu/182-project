"""SPARSE24_MLP=1: 2:4 structured weight sparsity on the MLP (182/sparse24/README.md). Off by default.

The MLP is out = relu(x @ W1.T)^2 @ W2, with W1 = c_fc and W2 = c_proj both stored [h, d] (h = 2816, d = 768).
Its six GEMMs, written as y = z @ A.T so the sparse operand A is always the left operand of A @ z.T (the only
side NVIDIA's sparse tensor cores accelerate, sparse along the contraction dim):

    fwd   pre   = x    @ W1.T      A = W1   [h, d]  2:4 along d
    fwd   out   = post @ W2        A = W2.T [d, h]  2:4 along h
    bwd   dpost = g    @ W2.T      A = W2   [h, d]  2:4 along d
    bwd   dx    = dpre @ W1        A = W1.T [d, h]  2:4 along h
    bwd   dW1   = dpre.T @ x       dense (contracts over tokens)
    bwd   dW2   = post.T @ g       dense

W1 and W1.T (and W2 and W2.T) are both 2:4 only under a transposable mask: every aligned 4x4 tile keeps at most
two entries per row and two per column (Hubara et al. 2021). With one, four of the six GEMMs go sparse; the
weight gradients stay dense and update every dense weight, pruned or not (straight-through estimator, as in the
torchao SemiSparseLinear training recipe). The mask is recomputed from |W| at every refresh, i.e. after every
optimizer step, where the record writes its fp8 weight caches: the weights move every step, so the compressed
operands must be rebuilt every step anyway, and a fresh mask costs nothing extra.

Masks (SPARSE24_MASK):
  greedy   per 4x4 tile, visit the 16 entries by |w| descending and keep one while its row and column hold
           fewer than 2 (what torch._sparse_semi_structured_tile does; it can stop at 7 kept).
  optimal  the best of the 186 maximal 4x4 patterns (at most two per row and column) by kept |w|: the L1
           criterion of Hu et al. 2024, who compute it as a convolution over the 90 exact patterns; the 96
           seven-entry ones are added because one of them sometimes beats every exact pattern.

Backends (SPARSE24_BACKEND):
  cusparselt  torch._cslt_sparse_mm on the compressed operands, behind the opaque custom op sparse24::cslt_linear
              (torch.compile sees a fake with the right shape). bf16 in, bf16 out, fp32 accumulate. GPU only.
  dense       the same masked math with dense GEMMs on masked weight copies: the CPU path the tests run, and a
              loss-only emulation on GPU (no speedup).

Pure torch: no Triton kernels here. Everything except the cuSPARSELt calls runs on CPU.
"""
import itertools
import os

import torch
from torch import Tensor

from track_1_short.config import BLOCK_SIZE, SPARSE24_BACKEND, SPARSE24_DENSE_FROM_WS, SPARSE24_MASK, SPARSE24_MLP

# Every sparse operand and its dense emulation copy is bf16 (the bank's training dtype).
SP24_DTYPE = torch.bfloat16
# cuSPARSELt writes A @ B transposed in its epilogue (transpose_result) so each call yields z @ A.T directly.
# 0 = take the [m, T] result and transpose it with a copy (the fallback if the fused layout ever misbehaves).
CSLT_FUSE_TRANSPOSE = os.environ.get("SPARSE24_CSLT_FUSE_T", "1") == "1"
# cuSPARSELt algorithm id (torch._cslt_sparse_mm_search can find a faster one; 182/sparse24/gpu_bench.py reports it).
CSLT_ALG_ID = int(os.environ.get("SPARSE24_CSLT_ALG_ID", "0"))


def sparse24_active(ws_long_tokens: int) -> bool:
    """Whether a forward with this long window (in tokens) runs the sparse MLP. A trace-time constant: ws_long
    only changes at stage boundaries (and for the final validation), each of which is its own compiled graph."""
    if not SPARSE24_MLP:
        return False
    return SPARSE24_DENSE_FROM_WS == 0 or ws_long_tokens < SPARSE24_DENSE_FROM_WS * BLOCK_SIZE


# -----------------------------------------------------------------------------
# Masks (pure torch, any device). Weights are [..., M, K] with M and K multiples of 4.

def _to_tiles(w: Tensor) -> Tensor:
    """[..., M, K] -> [..., M/4 * K/4, 16]; tile entry 4 * r + c is row r, column c of its tile."""
    *lead, m, k = w.shape
    assert m % 4 == 0 and k % 4 == 0, f"2:4 tiles need both dims divisible by 4, got {tuple(w.shape)}"
    t = w.reshape(*lead, m // 4, 4, k // 4, 4).transpose(-3, -2)
    return t.reshape(*lead, (m // 4) * (k // 4), 16)


def _from_tiles(t: Tensor, shape) -> Tensor:
    *lead, m, k = shape
    return t.reshape(*lead, m // 4, k // 4, 4, 4).transpose(-3, -2).reshape(shape)


def maximal_patterns() -> Tensor:
    """The 186 maximal 4x4 0/1 patterns with at most two ones per row and per column (no entry can be added):
    90 with exactly two per row and column, and 96 with seven ones (one row and one column hold a single
    entry that cannot be paired, e.g. what greedy can end on). [186, 16] bool, row-major within the tile."""
    valid = {b for b in itertools.product((0, 1), repeat=16)
             if all(sum(b[4 * r:4 * r + 4]) <= 2 for r in range(4)) and all(sum(b[c::4]) <= 2 for c in range(4))}
    maximal = sorted((b for b in valid
                      if not any(b[i] == 0 and b[:i] + (1,) + b[i + 1:] in valid for i in range(16))),
                     key=lambda b: (-sum(b), b))
    out = torch.tensor(maximal, dtype=torch.bool)
    assert out.shape == (186, 16)
    return out


_PATTERNS: dict[torch.device, Tensor] = {}


def mask_tile_optimal(w: Tensor) -> Tensor:
    """Transposable mask maximizing the kept sum of |w| per 4x4 tile: the best maximal pattern (7 or 8 kept).
    Every greedy mask lies inside some maximal pattern, so this keeps at least what greedy keeps."""
    pats = _PATTERNS.get(w.device)
    if pats is None:
        pats = _PATTERNS[w.device] = maximal_patterns().to(w.device)
    if w.dim() > 2:  # one matrix at a time: the [tiles, 186] score matrix of a whole bank would be GBs
        return torch.stack([mask_tile_optimal(m) for m in w.reshape(-1, *w.shape[-2:])]).reshape(w.shape)
    t = _to_tiles(w.detach().abs().float())
    best = (t @ pats.T.float()).argmax(dim=-1)           # [tiles]; first max wins on ties (8-entry patterns first)
    return _from_tiles(pats[best], w.shape)


def mask_tile_greedy(w: Tensor) -> Tensor:
    """Transposable mask by the greedy 4x4 rule of torch._sparse_semi_structured_tile: entries by |w| descending
    (stable, so ties go to the lower tile index), each kept while its row and column hold fewer than 2."""
    t = _to_tiles(w.detach().abs().float())
    order = t.argsort(dim=-1, descending=True, stable=True)
    rows, cols = order // 4, order % 4
    keep = torch.zeros_like(t, dtype=torch.bool)
    row_n = torch.zeros(*t.shape[:-1], 4, dtype=torch.int32, device=t.device)
    col_n = torch.zeros_like(row_n)
    for j in range(16):
        r, c = rows[..., j:j + 1], cols[..., j:j + 1]
        ok = (row_n.gather(-1, r) < 2) & (col_n.gather(-1, c) < 2)
        keep.scatter_(-1, order[..., j:j + 1], ok)
        row_n.scatter_add_(-1, r, ok.int())
        col_n.scatter_add_(-1, c, ok.int())
    return _from_tiles(keep, w.shape)


def mask_rowwise(w: Tensor) -> Tensor:
    """Plain (non-transposable) 2:4 along the last dim: the top 2 of every 4. For comparison only: it makes the
    forward GEMM sparse but not the transposed one."""
    g = w.detach().abs().float().reshape(*w.shape[:-1], w.shape[-1] // 4, 4)
    top = g.topk(2, dim=-1).indices
    return torch.zeros_like(g, dtype=torch.bool).scatter_(-1, top, True).reshape(w.shape)


def transposable_mask(w: Tensor, kind: str | None = None) -> Tensor:
    """SPARSE24_MASK's mask (or `kind`'s)."""
    return mask_tile_optimal(w) if (kind or SPARSE24_MASK) == "optimal" else mask_tile_greedy(w)


def is_24(mask: Tensor, dim: int = -1) -> bool:
    """At most 2 kept in every aligned group of 4 along `dim`."""
    m = mask.movedim(dim, -1)
    return bool((m.reshape(*m.shape[:-1], m.shape[-1] // 4, 4).sum(-1) <= 2).all())


def kept_fraction(w: Tensor, mask: Tensor, p: int = 2) -> float:
    """Share of ||w||_p^p the mask keeps."""
    a = w.detach().abs().float() ** p
    return float((a * mask).sum() / a.sum().clamp(min=1e-30))


# -----------------------------------------------------------------------------
# Sparse GEMM: y = z @ A.T with A 2:4 along its last dim.

@torch.library.custom_op("sparse24::cslt_linear", mutates_args=())
def cslt_linear(packed: Tensor, z: Tensor, out_features: int) -> Tensor:
    """z [T, K] @ A.T for A [out_features, K] given as its cuSPARSELt compression `packed`."""
    if CSLT_FUSE_TRANSPOSE:
        return torch._cslt_sparse_mm(packed, z.t(), transpose_result=True, alg_id=CSLT_ALG_ID)
    return torch._cslt_sparse_mm(packed, z.t(), alg_id=CSLT_ALG_ID).t().contiguous()


@cslt_linear.register_fake
def _cslt_linear_fake(packed: Tensor, z: Tensor, out_features: int) -> Tensor:
    return z.new_empty(z.shape[0], out_features)


def sp_linear(z: Tensor, a: Tensor, out_features: int) -> Tensor:
    """z @ A.T: `a` is A itself (masked, dense backend; possibly a transposed view) or A's compression."""
    if SPARSE24_BACKEND == "dense":
        return z @ a.T
    return cslt_linear(a, z, out_features)


class Sparse24ReLUSquareFunction(torch.autograd.Function):
    """relu(x @ W1m.T)^2 @ W2m with W1m, W2m the masked weights; dense straight-through weight gradients.

    a1, a1t, a2, a2t: the slot's sparse operands for W1, W1.T, W2, W2.T (slot_operands). The dense W1 and W2
    are inputs only so autograd hands their gradients to the bank: dW1 = dpre.T @ x and dW2 = post.T @ g are
    the gradients with respect to the masked weights, applied to every entry. Same conventions as
    BF16LinearReLUSquareFunction (perf/kernels/mlp.py): fp32 dpre product, bf16 GEMM operands, the
    post-lambda applied by the caller.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, a1, a1t, a2, a2t):
        h, d = W1.shape
        x_flat = x.view(-1, x.shape[-1])
        pre = sp_linear(x_flat, a1, h)
        relu_pre = torch.relu(pre)
        out = sp_linear(relu_pre * relu_pre, a2t, d)
        ctx.save_for_backward(x, pre, a1t, a2)
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, pre, a1t, a2 = ctx.saved_tensors
        h, d = pre.shape[-1], x.shape[-1]
        x_flat = x.view(-1, d)
        g = grad_output.reshape(-1, d)
        relu_pre = torch.relu(pre)
        dpost = sp_linear(g, a2, h)
        acc = torch.promote_types(pre.dtype, torch.float32)  # fp32 for bf16, as BF16LinearReLUSquareFunction
        dpre = (2.0 * dpost.to(acc) * relu_pre.to(acc)).to(pre.dtype)
        dW2 = (relu_pre * relu_pre).T @ g
        dW1 = dpre.T @ x_flat
        dx = sp_linear(dpre, a1t, d)
        return dx.view(x.shape), dW1, dW2, None, None, None, None


Sparse24MLP = Sparse24ReLUSquareFunction.apply


# -----------------------------------------------------------------------------
# Per-slot state: the compressed (or masked) operands, refreshed after every optimizer step.
# Buffer names start with "_mlp_" so the fp8 refresh CUDA graph (perf/cuda_graphs/fp8_refresh_graphs.py)
# treats them as its outputs, like the fp8 MLP caches they stand in for.

def _compress_pair(w: Tensor, kind: str) -> tuple[Tensor, Tensor]:
    """(compression of W masked, compression of W.T masked), flat. W [M, K] bf16 on CUDA."""
    if kind == "greedy":
        packed, _, packed_t, _, _ = torch._sparse_semi_structured_tile(w, algorithm="", use_cutlass=False)
    else:
        wm = w * mask_tile_optimal(w)
        packed, packed_t = torch._cslt_compress(wm), torch._cslt_compress(wm.t().contiguous())
    return packed.reshape(-1), packed_t.reshape(-1)


# The operand buffers: masked dense copies of W1 and W2 (dense backend), or the four compressions (W1, W1.T, W2, W2.T).
SPARSE24_BUFFER_NAMES = (("_mlp_sp24_w1", "_mlp_sp24_w2") if SPARSE24_BACKEND == "dense" else
                         ("_mlp_sp24_w1", "_mlp_sp24_w1t", "_mlp_sp24_w2", "_mlp_sp24_w2t"))


def sparse24_buffers(num_slots: int, hidden: int, dim: int, device: torch.device) -> dict[str, Tensor]:
    """Allocate the operand buffers once (their addresses are baked into the CUDA graphs)."""
    if SPARSE24_BACKEND == "dense":
        return {"_mlp_sp24_w1": torch.zeros(num_slots, hidden, dim, dtype=SP24_DTYPE, device=device),
                "_mlp_sp24_w2": torch.zeros(num_slots, hidden, dim, dtype=SP24_DTYPE, device=device)}
    assert torch.device(device).type == "cuda", "SPARSE24_BACKEND=cusparselt needs CUDA (use dense to emulate)"
    probe = torch.zeros(hidden, dim, dtype=SP24_DTYPE, device=device)
    packed, packed_t = _compress_pair(probe, SPARSE24_MASK)
    return {name: torch.zeros(num_slots, n, dtype=SP24_DTYPE, device=device)
            for name, n in (("_mlp_sp24_w1", packed.numel()), ("_mlp_sp24_w1t", packed_t.numel()),
                            ("_mlp_sp24_w2", packed.numel()), ("_mlp_sp24_w2t", packed_t.numel()))}


@torch.no_grad()
def refresh_sparse24_(up: Tensor, down: Tensor, bufs: dict[str, Tensor]):
    """Re-mask and re-compress every slot from the current bank. up = c_fc, down = c_proj, both [n, h, d]."""
    if SPARSE24_BACKEND == "dense":
        for w, name in ((up, "_mlp_sp24_w1"), (down, "_mlp_sp24_w2")):
            torch.mul(w, transposable_mask(w), out=bufs[name])
        return
    for slot in range(up.shape[0]):
        for w, name in ((up[slot], "_mlp_sp24_w1"), (down[slot], "_mlp_sp24_w2")):
            packed, packed_t = _compress_pair(w.to(SP24_DTYPE).contiguous(), SPARSE24_MASK)
            bufs[name][slot].copy_(packed)
            bufs[name + "t"][slot].copy_(packed_t)


def slot_operands(bufs: dict[str, Tensor], slot: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(a1, a1t, a2, a2t) of Sparse24ReLUSquareFunction for one slot."""
    if SPARSE24_BACKEND == "dense":
        w1, w2 = bufs["_mlp_sp24_w1"][slot], bufs["_mlp_sp24_w2"][slot]
        return w1, w1.T, w2, w2.T
    return (bufs["_mlp_sp24_w1"][slot], bufs["_mlp_sp24_w1t"][slot],
            bufs["_mlp_sp24_w2"][slot], bufs["_mlp_sp24_w2t"][slot])


def masked_weights(bufs: dict[str, Tensor], slot: int) -> tuple[Tensor, Tensor]:
    """The dense masked (W1m, W2m) a slot's operands encode (dense backend only; for tests and analysis)."""
    assert SPARSE24_BACKEND == "dense"
    return bufs["_mlp_sp24_w1"][slot], bufs["_mlp_sp24_w2"][slot]
