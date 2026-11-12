"""FP8_ATTN_O (182/fp8-attn-o): the static fp8 scales of the attention output projection, how they are derived,
and the pure-torch fp8-emulation reference the kernels and tests compare against. Pure torch (no triton), so it
imports on any CPU.

The GEMM:  out [T, 768] = y [T, H*v] @ (o_scale * o_w).T,  o_w [768, H*v],  o_scale = sa_lambdas[1] (* o_gain).
fp8 form:  out = _scaled_mm(q(y / s_y), q(o_w / s_w).T) * (s_y * (s_w * o_scale))      (e4m3 x e4m3)
           dy  = _scaled_mm(q(g / s_g), q(o_w / s_w))   * (s_g * (s_w * o_scale))      (e5m2 x e4m3)
           dW_eff = _scaled_mm(q(g / s_g).T, q(y / s_y)) * (s_g * s_y);  d o_w = o_scale * dW_eff,
           d o_scale = <dW_eff, o_w>
o_scale is folded into the weight's descale (scale_b), the way PackedFP8QKV folds sa_lambdas[0]: in exact
arithmetic the model is unchanged. The rounding is not: the record rounds bf16(o_scale * o_w) once, this path
quantizes o_w alone to e4m3 and applies o_scale in fp32 inside the GEMM epilogue.

Static scales (unit scaling: Blake et al. 2024, u-muP; Narayan et al. 2025, muNit Scaling). e4m3 has normals in
[2^-6, 448] (~15 binades, subnormals to 2^-9) and e5m2 in [2^-14, 57344] (~30 binades) at a fixed relative
precision, so a per-tensor scale only has to put the tensor's bulk well inside that window. The rule here is
u-muP's: scale so the tensor's typical rms lands near 1 (0.2..2 in scaled units), with a power of two (exact
multiplies), then check an analytic bound on its largest entry against the format's max where one exists.
Measured on Gaussian / Laplace / Student-t(4) samples (tests/test_attn_o_scales.py), the quantization error at
such a scale equals the error at the tensor's own amax scale (2.65% e4m3, 5.3% e5m2 rel-L2) over a window of
about 2^-5 .. 2^3 times the scale in rms; it grows only past the subnormal edge or at saturation. Per tensor:

  y (e4m3), S_Y = 2^-3.  FA3 returns, per head, y_t = sum_j p_tj v_j with p a probability vector, so every
      coordinate |y| <= max_j |v_j| and rms(y) <= rms(v) (convexity). v = lambda0 * x_hat @ W_v.T (+ aux_v), x_hat
      RMS-normed (||x_hat||_2 = sqrt(768)), so |v| <= lambda0 * r * sqrt(768) for a W_v row norm r (Cauchy-Schwarz)
      and v has std ~ lambda0 * r for directions not aligned with x_hat. At init lambda0 = 0.5 and r = 0.5
      (rows uniform +-2^-5: r^2 = 768 * 2^-10 / 3), so std(v) = 0.25 and |v| <= 6.93. Gated XSA subtracts
      alpha * (y . v_hat) v_hat with |alpha| < 1: y = y_perp + (1 - alpha) y_par, so ||y'|| <= 2 ||y||. The
      per-head gate multiplies by G (the MUDD lane, ~0.25 at init, unbounded in principle). Bound:
      |y| <= 2 G (lambda0 r sqrt(768) + max|aux_v|) = 13.9 G at init  <=  448 * 2^-3 = 56 (4x margin).
      Typical: rms(y) ~ std(v) / sqrt(n_eff) in [0.025, 0.25] (n_eff = effective tokens attended, 1..100), i.e.
      0.2..2 in scaled units; Student-t(4) tails do not saturate up to rms 0.4 (1.6x the top of that range),
      and the error stays at the amax-scale level down to rms 0.01. What the analysis does NOT bound: growth of lambda0,
      of the W_v row norms and of the gate during training, and the last layer's v_mudd (a learned mix of
      residual-stream snapshots added into V). The bound holds while lambda0 * r * (1 + |alpha|) * G <= 2,
      4x its init value 0.5.
  o_w (e4m3), S_W = 2^-6.  Entries start uniform in +-2^-5 (std 0.018). ANVIL steps are whitened (singular
      values ~1, ||U||_F = sqrt(768)) at lr 0.023 * lr_mul, with sign-aligned decoupled decay wd * lr * lr_mul
      * lr = 1.2e-3 * lr_mul per step on about half the entries: a random-walk equilibrium
      ||W||_F ~ ||dW||_F / sqrt(2 decay) ~ 18-24, i.e. entry std 0.024-0.031, largest entry ~5 std = 0.15.
      448 * 2^-6 = 7 covers 45x that; the init std sits at 1.15 in scaled units (unit-scaled). The only hard
      bound, |w_ij| <= ||W||_2 <= 1 + sum_t lr_t (~40 over the 1194 steps), is looser than 7: NOT bounded by
      analysis, only by the equilibrium argument.
  g (e5m2), S_G = 2^-6.  g = dL/d(attn out) is the residual-stream gradient right after the attention add
      (x = resid * x + attn_out), one MLP away from the MLP's incoming gradient, which the record quantizes at
      the static e5m2 2^-6 clip-free ("statically clip-free scales", ANVIL2): g_attn,i = r * g_mlp,i +
      J_mlp,i^T g_mlp,i. Same stream, same scale. e5m2 at 2^-6 spans normals 2^-20 .. 896, so a mismatch of
      several binades either way is still clip-free and above the subnormal edge. Borrowed evidence, not a bound.

FP8_ATTN_O_SCALES=delayed is the arm for when this derivation is wrong: y and g from the previous step's amax
(headroom ATTN_O_Y_HEADROOM / ATTN_O_G_HEADROOM, like the MLP's post / dpre), o_w from its current amax (exact,
like the QKV cache). HADAMARD_SAT_STATS=1 prints o_y / o_g / o_w "fp8 sat" lines with either arm.
"""
import math

import torch

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
FP8_MAX = {E4M3: 448.0, E5M2: 57344.0}

ATTN_O_Y_SCALE = 2.0 ** -3   # e4m3, attention output y (after XSA / gate)
ATTN_O_W_SCALE = 2.0 ** -6   # e4m3, o_w (before o_scale, which rides the descale)
ATTN_O_G_SCALE = 2.0 ** -6   # e5m2, the incoming gradient dL/d(attn out); = FP8_GRAD_SCALE of the MLP
# FP8_ATTN_O_SCALES=delayed: scale = last step's amax * headroom / fmt_max (as the MLP's post / dpre scales).
ATTN_O_Y_HEADROOM = 1.10
ATTN_O_G_HEADROOM = 1.25

# The quantities the derivation above uses (model/gpt.py, model/attention.py, training.py, optim/anvil.py).
MODEL_DIM = 768
INIT_LAMBDA0 = 0.5                       # sa_lambdas[0] at init (GPT.init_misc)
INIT_BOUND = 0.5 * MODEL_DIM ** -0.5 * 3 ** 0.5   # uniform init bound of the banks = 2^-5 (GPT.init_attn)
ANVIL_LR, ANVIL_WD = 0.023, 2.25         # training.py anvil_defaults


def init_row_norm(dim: int = MODEL_DIM) -> float:
    """Expected L2 norm of a uniform(+-INIT_BOUND) row of length dim: 0.5 at 768."""
    return math.sqrt(dim * INIT_BOUND ** 2 / 3)


def y_bound(lambda0: float, row_norm: float, gate: float = 1.0, xsa: bool = True, aux_max: float = 0.0,
            dim: int = MODEL_DIM) -> float:
    """Upper bound on |y| entries: convex combination of V rows, (1 + |alpha|) < 2 for XSA, times the gate."""
    v_max = lambda0 * row_norm * math.sqrt(dim) + aux_max
    return (2.0 if xsa else 1.0) * gate * v_max


def weight_equilibrium_std(lr_mul: float = 1.0, dim: int = MODEL_DIM, aligned_frac: float = 0.5) -> float:
    """Entry std of a whitened-update / decoupled-decay random walk at equilibrium (ANVIL, square [dim, dim])."""
    step_fro = ANVIL_LR * lr_mul * math.sqrt(dim)
    decay = ANVIL_WD * ANVIL_LR * ANVIL_LR * lr_mul * aligned_frac
    return step_fro / math.sqrt(2 * decay) / dim


def quant(t: torch.Tensor, scale, fmt: torch.dtype) -> torch.Tensor:
    """fp8 copy of t / scale, clamped to fmt's range first (e4m3fn overflow is NaN, not inf)."""
    lim = FP8_MAX[fmt]
    return torch.clamp(t.float() * (1.0 / float(scale)), -lim, lim).to(fmt)


def attn_o_reference(y, o_w, o_scale, g, s_y=ATTN_O_Y_SCALE, s_w=ATTN_O_W_SCALE, s_g=ATTN_O_G_SCALE):
    """The fp8 path in plain torch: {out, dy, dw, do_scale}, with the kernels' roundings (fp32 accumulate, bf16
    GEMM outputs, d o_w = bf16(dW_eff) * o_scale in bf16, d o_scale = <dW_eff, o_w> in fp32).
    y [T, K] bf16, o_w [N, K] bf16, o_scale 0-d fp32, g [T, N] bf16."""
    yq, wq, gq = quant(y, s_y, E4M3).float(), quant(o_w, s_w, E4M3).float(), quant(g, s_g, E5M2).float()
    o = float(o_scale)
    out = ((yq @ wq.T) * (s_y * (s_w * o))).bfloat16()
    dy = ((gq @ wq) * (s_g * (s_w * o))).bfloat16()
    dw_eff = ((gq.T @ yq) * (s_g * s_y)).bfloat16()
    # A 0-d fp32 o_scale times the bf16 matrix stays bf16, as in the Function.
    return dict(out=out, dy=dy, dw=dw_eff * torch.as_tensor(o_scale, dtype=torch.float32),
                do_scale=(dw_eff.float() * o_w.float()).sum(), dw_eff=dw_eff)


def attn_o_bf16_reference(y, o_w, o_scale, g):
    """The record's math (autograd through F.linear(y, o_scale * o_w)): {out, dy, dw, do_scale}."""
    y = y.detach().clone().requires_grad_()
    w = o_w.detach().clone().requires_grad_()
    s = torch.as_tensor(o_scale).detach().clone().requires_grad_()
    out = torch.nn.functional.linear(y, s * w)
    out.backward(g)
    return dict(out=out.detach(), dy=y.grad, dw=w.grad, do_scale=s.grad)


def rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    want = want.float()
    return ((got.float() - want).norm() / want.norm().clamp(min=1e-30)).item()


def fp8_stats(t: torch.Tensor, scale: float, fmt: torch.dtype) -> dict:
    """Saturated / subnormal / flushed fractions and the quantization rel-error of t at `scale`."""
    lim = FP8_MAX[fmt]
    min_normal = 2.0 ** -6 if fmt == E4M3 else 2.0 ** -14
    a = t.float().abs() / scale
    q = quant(t, scale, fmt).float() * scale
    return dict(saturated=(a > lim).float().mean().item(), subnormal=((a < min_normal) & (a > 0)).float().mean().item(),
                zero=((q == 0) & (t != 0)).float().mean().item(), rel_err=rel_err(q, t), amax_ratio=a.max().item() / lim)


def amax_scale(t: torch.Tensor, fmt: torch.dtype, headroom: float = 1.0) -> float:
    """The exact per-tensor scale: amax * headroom / fmt_max."""
    return max(t.float().abs().max().item(), 1e-30) * headroom / FP8_MAX[fmt]
