"""LAYER_ROPE (PR #385, @shikhar-srivastava): depth-conditioned gains on both stages of both branches. Per layer l,
u = log(l + 1), and per site s (0 attn input, 1 mlp input, 2 attn residual, 3 mlp residual):
  gamma_eff[s, l] = exp(alpha_s + beta_s * u) * ComplexRot(gamma_s, exp(alpha_rot_s + beta_rot_s * u) * freq_j)
rotating channel pairs (2j, 2j + 1), freq_j = base ** (-2j / model_dim). The four gamma vectors (init ones) and
16 gate scalars are shared by every layer. The attention gains are folded into weights as in #385 (the input gain
into the QKV columns, the fp8 cache holding the folded weight; the residual gain into O's rows). The MLP gains
multiply the normed MLP input and the MLP output: #385 folds them into the dense MLP bank's fp8 caches, but here
five of the nine MLP layers are MoE layers, and the dense bank's gradient is written in place (bankinplace).
"""
import torch
from torch import Tensor

LAYER_ROPE_GATES_INIT = (0.0, 0.0, 0.0, 0.0)  # (alpha, beta, alpha_rot, beta_rot), on every site
LAYER_ROPE_BASE_FREQ = 100.0


def layer_rope_gains(gates: Tensor, gammas: Tensor, freqs: Tensor, u: Tensor) -> Tensor:
    """All effective LayerRoPE gains, (4 sites, num_layers, model_dim) bf16, in fp32 from the parameters alone."""
    g = gates.float()                                                                 # (4, 4)
    mag = g[:, 0:1] + g[:, 1:2] * u                                                   # (4, L)
    theta = torch.exp(g[:, 2:3] + g[:, 3:4] * u)[..., None] * freqs                   # (4, L, D/2)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    gam = gammas.float()
    gu, gv = gam[:, None, 0::2], gam[:, None, 1::2]                                   # (4, 1, D/2)
    r = torch.exp(mag)[..., None]                                                     # (4, L, 1)
    out = torch.stack([r * (gu * cos_t - gv * sin_t), r * (gu * sin_t + gv * cos_t)], dim=-1).flatten(-2)
    return out.to(torch.bfloat16)
