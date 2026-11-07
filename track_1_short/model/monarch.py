"""Monarch (block-diagonal + perfect-shuffle) MLP projections: an opt-in falsification experiment.

MONARCH_MLP=1 replaces each dense MLP slot of mlp_bank (c_fc [h, d], c_proj [h, d]) with two Monarch
factors per projection, and runs the MLP in bf16 torch ops. Off by default: the record path is untouched.

Structure (Dao et al. 2022, https://arxiv.org/abs/2204.00595; the rectangular form of their
blockdiag_butterfly_multiply). With d = model_dim, k blocks, expand e (hidden = d * e^2), p = d / k,
q = e * p, l = e * k:

    up   (d -> h):  x [T, d] -> [T, k, p] --R (k blocks, q x p)--> [T, k, q]
                    -> transpose [T, q, k] -> view [T, l, p] --L (l blocks, q x p)--> [T, l, q] = [T, h]
    down (h -> d):  the exact transpose of that structure, its two factors stored like c_proj (transposed),
                    so every block of the whole bank has the same [q, p] shape.

Every block of the bank being [q, p] is what lets the bank stay ONE ANVIL parameter: ANVIL runs one
cascade per bank matrix, so here each Monarch block is orthogonalized on its own ("per-block Muon").
For a block-diagonal factor that is exactly Muon of the factor (the polar factor of a block-diagonal
matrix is the block-diagonal of the blocks' polar factors), but it is NOT Muon of the composed MLP
weight, and orthogonalizing per attention head (instead of the whole Q/K/V matrix) has been found to
hurt in this speedrun. Expect a loss hit from the optimizer side as well as the capacity side.

Equal block shapes need d / h to be a rational square, so the hidden size is d * e^2 (3072 at e = 2),
not the record's 2816.
"""
import math
import os
from dataclasses import dataclass

import torch
from torch import Tensor

MONARCH_MLP = os.environ.get("MONARCH_MLP", "0") == "1"
MONARCH_BLOCKS = int(os.environ.get("MONARCH_BLOCKS", "4"))   # k: blocks of the first factor
MONARCH_EXPAND = int(os.environ.get("MONARCH_EXPAND", "2"))   # e: hidden = model_dim * e^2
# ANVIL lr multipliers: every Monarch block gets MONARCH_LR_MUL, the down-projection blocks a further
# MONARCH_DOWN_LR_MUL (the dense bank gives c_proj 2x).
MONARCH_LR_MUL = float(os.environ.get("MONARCH_LR_MUL", "1.0"))
MONARCH_DOWN_LR_MUL = float(os.environ.get("MONARCH_DOWN_LR_MUL", "2.0"))


@dataclass(frozen=True)
class MonarchShape:
    dim: int
    blocks: int  # k
    expand: int  # e

    def __post_init__(self):
        assert self.dim % self.blocks == 0, f"model_dim {self.dim} must divide into {self.blocks} blocks"

    @property
    def p(self) -> int:  # input width of every block
        return self.dim // self.blocks

    @property
    def q(self) -> int:  # output width of every block
        return self.expand * self.p

    @property
    def l(self) -> int:  # blocks of the second factor
        return self.expand * self.blocks

    @property
    def hidden(self) -> int:
        return self.l * self.q  # == dim * expand ** 2

    @property
    def blocks_per_slot(self) -> int:
        """[R_up (k), L_up (l), L_down (l), R_down (k)]."""
        return 2 * (self.blocks + self.l)

    def slot_slices(self) -> tuple[slice, slice, slice, slice]:
        k, l = self.blocks, self.l
        return slice(0, k), slice(k, k + l), slice(k + l, k + 2 * l), slice(k + 2 * l, 2 * k + 2 * l)

    def params_per_slot(self) -> int:
        return self.blocks_per_slot * self.q * self.p

    def macs_per_token(self) -> int:
        """Multiply-accumulates per token of one slot (up + down), the same count as its parameters."""
        return self.params_per_slot()


def monarch_up(x: Tensor, r: Tensor, l: Tensor, shape: MonarchShape) -> Tensor:
    """x [..., d] -> [..., h]. r: [k, q, p], l: [l, q, p] (nn.Linear layout, out x in)."""
    lead = x.shape[:-1]
    x = x.reshape(-1, shape.blocks, shape.p)                        # [T, k, p]
    y = torch.einsum("tkp,kqp->tkq", x, r)                          # [T, k, q]
    y = y.transpose(1, 2).reshape(-1, shape.l, shape.p)             # [T, q, k] -> [T, l, p]
    z = torch.einsum("tlp,lqp->tlq", y, l)                          # [T, l, q]
    return z.reshape(*lead, shape.hidden)


def monarch_down(h: Tensor, l: Tensor, r: Tensor, shape: MonarchShape) -> Tensor:
    """h [..., h] -> [..., d]: the transpose of monarch_up's structure. l: [l, q, p], r: [k, q, p],
    stored transposed (in x out), as c_proj is."""
    lead = h.shape[:-1]
    h = h.reshape(-1, shape.l, shape.q)                             # [T, l, q]
    y = torch.einsum("tlq,lqp->tlp", h, l)                          # [T, l, p]
    y = y.reshape(-1, shape.q, shape.blocks).transpose(1, 2)        # [T, q, k] -> [T, k, q]
    z = torch.einsum("tkq,kqp->tkp", y, r)                          # [T, k, p]
    return z.reshape(*lead, shape.dim)


def monarch_mlp(x: Tensor, slot_blocks: Tensor, shape: MonarchShape) -> Tensor:
    """relu(up(x))^2 -> down, for one slot's [blocks_per_slot, q, p] blocks."""
    r_up, l_up, l_down, r_down = (slot_blocks[s] for s in shape.slot_slices())
    hidden = monarch_up(x, r_up, l_up, shape).relu().square()
    return monarch_down(hidden, l_down, r_down, shape)


@torch.no_grad()
def init_monarch_bank_(bank: Tensor, shape: MonarchShape):
    """bank: [slots, blocks_per_slot, q, p]. Up: the composed map has the dense c_fc's output variance
    (0.25 * Var(x): c_fc's std is 0.5 * d^-0.5); R_up is variance-preserving, L_up carries the 0.5.
    Down: zero output at init like c_proj, but only R_down (the last factor) is zero -- with both
    factors zero neither would ever get a gradient. Uniform, like the dense bank."""
    r_up, l_up, l_down, r_down = shape.slot_slices()

    def uniform_(t, std):
        bound = math.sqrt(3) * std
        t.uniform_(-bound, bound)

    uniform_(bank[:, r_up], shape.p ** -0.5)        # fan-in p
    uniform_(bank[:, l_up], 0.5 * shape.p ** -0.5)  # fan-in p
    uniform_(bank[:, l_down], shape.q ** -0.5)      # fan-in q (used as h @ L)
    bank[:, r_down].zero_()


def monarch_matrix_lr_muls(shape: MonarchShape, num_slots: int, frozen_slots) -> tuple[float, ...]:
    """Per-matrix ANVIL lr multipliers over the flattened [slots * blocks_per_slot] bank."""
    r_up, l_up, l_down, r_down = shape.slot_slices()
    per_slot = [0.0] * shape.blocks_per_slot
    for s, mul in ((r_up, 1.0), (l_up, 1.0), (l_down, MONARCH_DOWN_LR_MUL), (r_down, MONARCH_DOWN_LR_MUL)):
        for j in range(s.start, s.stop):
            per_slot[j] = MONARCH_LR_MUL * mul
    out = []
    for slot in range(num_slots):
        out.extend([0.0] * shape.blocks_per_slot if slot in frozen_slots else per_slot)
    return tuple(out)


def dense_equivalents(slot_blocks: Tensor, shape: MonarchShape) -> tuple[Tensor, Tensor]:
    """(W_up [h, d], W_down [d, h]) such that up(x) = x @ W_up.T and down(h) = h @ W_down.T (for tests)."""
    r_up, l_up, l_down, r_down = (slot_blocks[s] for s in shape.slot_slices())
    eye_d = torch.eye(shape.dim, dtype=slot_blocks.dtype)
    eye_h = torch.eye(shape.hidden, dtype=slot_blocks.dtype)
    return monarch_up(eye_d, r_up, l_up, shape).T, monarch_down(eye_h, l_down, r_down, shape).T
