"""FP8_ATTN_O on any CPU (pure torch, no triton): the static scales and their derivation (perf/attn_o_scales.py),
the fp8-emulation reference against the record's bf16 math, and the config knobs.

    python -m pytest 182/fp8-attn-o/tests -q -rs
"""
import math
import os
import pathlib
import re
import subprocess
import sys
import zlib

import pytest
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from track_1_short.perf import attn_o_scales as S  # noqa: E402

E4, E5 = S.E4M3, S.E5M2
N = 1 << 20


def samples(family: str, n: int, gen: torch.Generator) -> torch.Tensor:
    """Unit-variance samples: Gaussian, Laplace, or Student-t with 4 dof (heavy tails)."""
    if family == "gauss":
        return torch.randn(n, generator=gen)
    u = torch.rand(n, generator=gen, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
    if family == "laplace":
        return -torch.sign(u - 0.5) * torch.log1p(-2 * (u - 0.5).abs()) / math.sqrt(2)
    z = torch.randn(n, generator=gen)
    chi2 = (torch.randn(4, n, generator=gen) ** 2).sum(0)
    return z / torch.sqrt(chi2 / 4) / math.sqrt(2)  # t(4) has variance 2


# ---------------------------------------------------------------------------- the scales themselves

def test_scales_are_powers_of_two():
    for s in (S.ATTN_O_Y_SCALE, S.ATTN_O_W_SCALE, S.ATTN_O_G_SCALE):
        assert math.log2(s) == round(math.log2(s))


def test_g_scale_is_the_mlp_grad_scale():
    """The derivation borrows the record's static e5m2 MLP gradient scale (same residual stream)."""
    m = re.search(r"^FP8_GRAD_SCALE = 2 \*\* (-\d+)", (ROOT / "track_1_short/model/gpt.py").read_text(), re.M)
    assert m and S.ATTN_O_G_SCALE == 2.0 ** int(m.group(1))


def test_init_constants_match_the_model():
    gpt = (ROOT / "track_1_short/model/gpt.py").read_text()
    assert "torch.tensor([0.5, 1.0]) for _ in range(num_layers)" in gpt   # sa_lambdas init: lambda0 = 0.5
    assert "std = 0.5 * model_dim ** -0.5" in gpt and "bound = (3 ** 0.5) * std" in gpt
    assert S.INIT_BOUND == pytest.approx(2.0 ** -5) and S.init_row_norm() == pytest.approx(0.5)
    tr = (ROOT / "track_1_short/training.py").read_text()
    assert re.search(r"anvil_defaults = dict\(\s*lr=0\.023,", tr) and "weight_decay=2.25" in tr


# ---------------------------------------------------------------------------- y: the analytic bound

def test_y_bound_at_init_fits_with_margin():
    b = S.y_bound(S.INIT_LAMBDA0, S.init_row_norm(), gate=1.0, xsa=True)
    assert b == pytest.approx(13.86, abs=0.01)
    assert b * 4 <= 448 * S.ATTN_O_Y_SCALE  # 4x margin
    # where the static scale stops being a bound: lambda0 * r * (1 + |alpha|) * G past 2 (4x its init value 0.5)
    assert S.y_bound(2.0, 0.5, gate=1.0) <= 448 * S.ATTN_O_Y_SCALE < S.y_bound(2.1, 0.5, gate=1.0)


def simulate_y(T=512, H=6, v=128, dim=768, window=64, gate=1.0, lam=S.INIT_LAMBDA0, seed=0, sharp=4.0):
    """The attention output at init: x_hat (RMS-normed rows with a few 20x outlier channels), V = lam * x_hat @ W_v.T
    with W_v uniform(+-2^-5), causal sliding-window softmax over random logits (temperature `sharp`), gated XSA
    with alpha ~ U(-1, 1), times the gate."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, dim, generator=g)
    x[:, :4] *= 20.0  # massive-activation channels
    x = F.rms_norm(x, (dim,))
    wv = (torch.rand(H * v, dim, generator=g) * 2 - 1) * S.INIT_BOUND
    vv = (lam * x @ wv.T).view(T, H, v)
    logits = sharp * torch.randn(H, T, T, generator=g)
    i = torch.arange(T)
    allow = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < window)
    p = torch.softmax(logits.masked_fill(~allow, float("-inf")), dim=-1)
    y = torch.einsum("hts,shd->thd", p, vv)
    alpha = torch.rand(T, H, 1, generator=g) * 2 - 1
    vh = vv / vv.norm(dim=-1, keepdim=True)
    y = y - alpha * (y * vh).sum(-1, keepdim=True) * vh
    return gate * y, vv


@pytest.mark.parametrize("sharp", [0.5, 4.0, 30.0])  # diffuse, typical, near one-hot attention
def test_simulated_y_respects_the_bound_and_the_scale(sharp):
    y, v = simulate_y(sharp=sharp)
    bound = S.y_bound(S.INIT_LAMBDA0, S.init_row_norm() * 1.05)  # sampled row norms scatter ~ +-2%
    assert y.abs().max().item() <= bound
    assert y.abs().max().item() <= 2 * v.abs().max().item() + 1e-6  # convex combination, XSA factor < 2
    st = S.fp8_stats(y.bfloat16(), S.ATTN_O_Y_SCALE, E4)
    assert st["saturated"] == 0.0 and st["amax_ratio"] < 0.25
    amax = S.fp8_stats(y.bfloat16(), S.amax_scale(y, E4), E4)
    assert st["rel_err"] <= amax["rel_err"] * 1.05 + 1e-4, (st, amax)
    rms = y.pow(2).mean().sqrt().item() / S.ATTN_O_Y_SCALE
    assert 0.05 < rms < 4, rms  # near unit in scaled units, as the derivation says


# ---------------------------------------------------------------------------- property tests: realistic ranges

# (tensor, scale, fmt, rms range the derivation claims). Each family x each rms in the range must not saturate,
# must keep the error at the amax-scale level, and must not flush more than a sliver to zero.
RANGES = [
    ("y", S.ATTN_O_Y_SCALE, E4, (0.01, 0.4)),
    ("w", S.ATTN_O_W_SCALE, E4, (0.005, 0.1)),
    ("g", S.ATTN_O_G_SCALE, E5, (2.0 ** -14, 4.0)),
]


@pytest.mark.parametrize("name,scale,fmt,rng", RANGES, ids=[r[0] for r in RANGES])
@pytest.mark.parametrize("family", ["gauss", "laplace", "t4"])
def test_static_scale_matches_amax_scale_over_the_claimed_range(name, scale, fmt, rng, family):
    gen = torch.Generator().manual_seed(zlib.crc32(f"{name}:{family}".encode()))  # deterministic, unlike hash()
    lo, hi = rng
    for k in range(9):
        rms = lo * (hi / lo) ** (k / 8)
        x = (samples(family, N, gen) * rms).bfloat16()
        st, amax = S.fp8_stats(x, scale, fmt), S.fp8_stats(x, S.amax_scale(x, fmt), fmt)
        assert st["saturated"] == 0.0, (name, family, rms, st)
        assert st["rel_err"] <= amax["rel_err"] * 1.06 + 1e-4, (name, family, rms, st, amax)
        assert st["zero"] < 0.02, (name, family, rms, st)


@pytest.mark.parametrize("name,scale,fmt,rms", [("y", S.ATTN_O_Y_SCALE, E4, 4.0), ("w", S.ATTN_O_W_SCALE, E4, 0.6)])
def test_static_scale_fails_beyond_the_range(name, scale, fmt, rms):
    """Past the range the derivation does not cover, heavy tails saturate: what FP8_ATTN_O_SCALES=delayed and the
    HADAMARD_SAT_STATS o_y / o_w rows are for."""
    gen = torch.Generator().manual_seed(7)
    x = (samples("t4", N, gen) * rms).bfloat16()
    st = S.fp8_stats(x, scale, fmt)
    assert st["saturated"] > 0 and st["amax_ratio"] > 1


def test_weight_equilibrium_is_inside_the_static_range():
    """The ANVIL random-walk equilibrium of the entry std: inside the w range above, its ~5-std max >= 40x
    below the e4m3 max at S_W."""
    for lr_mul in (1.0, 1.52, 1.73):
        std = S.weight_equilibrium_std(lr_mul)
        assert 0.015 < std < 0.04, (lr_mul, std)
        assert 5 * std * 40 < 448 * S.ATTN_O_W_SCALE
    assert S.INIT_BOUND < 448 * S.ATTN_O_W_SCALE / 100


# ---------------------------------------------------------------------------- the reference

def operands(T=512, K=768, Nout=768, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (0.2 * torch.randn(T, K, generator=g)).bfloat16()
    w = ((torch.rand(Nout, K, generator=g) * 2 - 1) * S.INIT_BOUND).bfloat16()
    gr = (1e-3 * torch.randn(T, Nout, generator=g)).bfloat16()
    return y, w, gr


@pytest.mark.parametrize("K", [768, 384])  # full and half-V layers
def test_reference_matches_bf16_within_fp8_tolerance(K):
    y, w, g = operands(K=K)
    o = torch.tensor(0.93)
    ref, bf = S.attn_o_reference(y, w, o, g), S.attn_o_bf16_reference(y, w, o, g)
    for k in ("out", "dy", "dw"):
        e = S.rel_err(ref[k], bf[k])
        # independent operand errors add in quadrature: out sqrt(2.65^2 + 2.65^2) = 3.7%, the gradients
        # sqrt(5.3^2 + 2.65^2) = 5.9% (e5m2 g)
        assert e < (0.045 if k == "out" else 0.07), (k, e)
    terms = (bf["dw"].float().abs() / 0.93 * w.float().abs()).sum().item()
    assert abs(ref["do_scale"].item() - bf["do_scale"].item()) < 0.02 * terms


def test_reference_folds_o_scale_exactly():
    y, w, g = operands(seed=1)
    a, b = S.attn_o_reference(y, w, torch.tensor(1.0), g), S.attn_o_reference(y, w, torch.tensor(0.25), g)
    for k in ("out", "dy", "dw"):
        assert torch.equal(b[k], a[k] * 0.25), k
    assert torch.equal(a["do_scale"], b["do_scale"])


def test_reference_dw_is_d_o_w_of_the_folded_product():
    """d o_w = o_scale * dW_eff and d o_scale = <dW_eff, o_w>: the chain rule of out = y @ (o_scale * o_w).T, checked
    with autograd on the dequantized operands in fp64."""
    y, w, g = operands(T=64, K=128, Nout=128, seed=2)
    o = 0.6
    yq = (S.quant(y, S.ATTN_O_Y_SCALE, E4).double() * S.ATTN_O_Y_SCALE)
    wq = (S.quant(w, S.ATTN_O_W_SCALE, E4).double() * S.ATTN_O_W_SCALE).requires_grad_()
    gq = S.quant(g, S.ATTN_O_G_SCALE, E5).double() * S.ATTN_O_G_SCALE
    s = torch.tensor(o, dtype=torch.float64, requires_grad=True)
    (yq @ (s * wq).T).backward(gq)
    ref = S.attn_o_reference(y, w, torch.tensor(o), g)
    assert S.rel_err(ref["dw"].double(), wq.grad) < 1e-2
    # <dW_eff, o_w> with the bf16 o_w vs <dW_eff, dequantized o_w>: differ by o_w's own e4m3 rounding
    assert ref["do_scale"].item() == pytest.approx(s.grad.item(), rel=0.1, abs=1e-6)


# ---------------------------------------------------------------------------- config (subprocess: env at import)

def config_run(env_extra: dict, expr: str = "c.FP8_ATTN_O, c.FP8_ATTN_O_SCALES, c.FP8_ATTN_O_AMAX"):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FP8_", "HADAMARD"))}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", f"import track_1_short.config as c; print(repr(({expr})))"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("env,want", [
    ({}, "(False, 'static', False)"),
    ({"FP8_ATTN_O": "1"}, "(True, 'static', False)"),
    ({"FP8_ATTN_O": "1", "FP8_ATTN_O_SCALES": "delayed"}, "(True, 'delayed', True)"),
    ({"FP8_ATTN_O": "1", "HADAMARD_SAT_STATS": "1"}, "(True, 'static', True)"),
    ({"FP8_ATTN_O": "0", "HADAMARD_SAT_STATS": "1"}, "(False, 'static', False)"),
    ({"FP8_ATTN_O": "1", "FP8_OFF": "mlp,lm"}, "(True, 'static', False)"),
    ({"FP8_ATTN_O": "1", "HADAMARD_FP8": "1"}, "(True, 'static', False)"),
])
def test_config_values(env, want):
    r = config_run(env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == want


@pytest.mark.parametrize("env,msg", [
    ({"FP8_ATTN_O": "1", "FP8_OFF": "1"}, "FP8_OFF=all is the bf16 upper bound"),
    ({"FP8_ATTN_O": "1", "FP8_OFF": "mlp,qkv,lm"}, "FP8_OFF=all is the bf16 upper bound"),
    ({"FP8_ATTN_O": "yes"}, "use 0 or 1"),
    ({"FP8_ATTN_O_SCALES": "amax"}, "use static or delayed"),
])
def test_config_rejects(env, msg):
    r = config_run(env)
    assert r.returncode != 0 and msg in r.stderr, r.stderr[-500:]


def test_knobs_are_documented_where_read():
    doc = (ROOT / "track_1_short/config.py").read_text().split('"""')[1]
    assert "FP8_ATTN_O" in doc and "FP8_ATTN_O_SCALES" in doc
