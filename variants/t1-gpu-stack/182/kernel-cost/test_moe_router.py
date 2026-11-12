"""Numerics of MOE_ROUTER_FUSED (track_1_short/perf/kernels/moe_router.py) against the stock router path.

Runs the real Triton kernels: on CPU tensors under the Triton interpreter (the triton-cpu docker image), or on a GPU.
Without either, every test skips. From the repo root:

    bash 182/tools/triton_cpu/run.sh python -m pytest variants/t1-gpu-stack/182/kernel-cost/test_moe_router.py -q
    python -m pytest variants/t1-gpu-stack/182/kernel-cost/test_moe_router.py -q        # on the node (GPU)

On the GPU the kernel tests also run at the record's stage-2 shape (49152 tokens per rank) and the whole op is compared
with moe_mlp.RoutedFP8MoE on Hopper (test_whole_op_matches_stock_on_hopper).
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
CUDA = torch.cuda.is_available() and not INTERP
pytestmark = pytest.mark.skipif(not (INTERP or CUDA), reason="needs the Triton interpreter (TRITON_INTERPRET=1) or a GPU")
DEV = "cuda" if CUDA else "cpu"
D, E = 768, 8
# The interpreter is slow: small token counts that still exercise the row masks (not multiples of 64 / 512).
SIZES = (1000, 49152) if CUDA else (200, 1000)


def _rows(n, gen):
    x = torch.randn(n, D, generator=gen)
    return F.rms_norm(x, (D,)).bfloat16().to(DEV)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("wscale", (0.01, 0.1))   # the init scale (many near-ties in bf16) and a trained-like one
def test_router_topk(n, wscale):
    from track_1_short.perf.kernels.moe_router import router_topk
    gen = torch.Generator().manual_seed(n + int(wscale * 1000))
    rows = _rows(n, gen)
    w = (wscale * torch.randn(E, D, generator=gen)).bfloat16().to(DEV)
    probs, gate, expert = router_topk(rows, w)
    assert probs.shape == (n, E) and probs.dtype == torch.float32
    assert gate.shape == (n,) and expert.shape == (n,) and expert.dtype == torch.int64

    # Near-exact reference for the kernel's own arithmetic: exact dot -> fp32 -> bf16 logits, fp32 softmax. The kernel
    # sums its three 256-column chunks in fp32, so a logit sitting on a bf16 rounding boundary may round the other way:
    # allow that on a handful of rows (each then within one bf16 step of a logit), everything else to fp32 precision.
    logits_exact = (rows.double() @ w.double().T).float().bfloat16().float()
    probs_exact = torch.softmax(logits_exact, dim=-1)
    off = ~torch.isclose(probs, probs_exact, rtol=2e-6, atol=2e-7)
    assert off.any(-1).float().mean().item() <= 0.005, off.sum()
    torch.testing.assert_close(probs, probs_exact, rtol=2e-2, atol=1e-4)
    torch.testing.assert_close(gate, probs.max(-1).values, rtol=0, atol=0)
    # expert is the first index of the max of the kernel's own probs (torch.max's tie rule)
    first = (probs == gate[:, None]).int().argmax(-1)
    assert torch.equal(expert, first)

    # The stock path: probs = softmax(F.linear(rows, w).float()), (gate, expert) = probs.max(-1). Its GEMM accumulates
    # in another order, so a logit may round to the neighbouring bf16 value: the experts agree except where the stock
    # top-two probabilities are within such a step of each other.
    probs_ref = torch.softmax(F.linear(rows, w).float(), dim=-1)
    gate_ref, expert_ref = probs_ref.max(-1)
    torch.testing.assert_close(probs, probs_ref, rtol=3e-2, atol=1e-3)
    differ = expert != expert_ref
    if differ.any():
        top2 = probs_ref.topk(2, dim=-1).values
        gap = (top2[:, 0] - top2[:, 1])[differ]
        assert (gap <= 1e-2 * top2[differ, 0]).all(), gap.max()
    assert differ.float().mean().item() < 0.02


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("rpp", (512, 128))      # the trainer's rows per program, and one bench_router.py alternative
def test_dx_combine_drouter(n, rpp):
    from track_1_short.perf.kernels.moe_mlp import _dx_combine_impl, padded_rows
    from track_1_short.perf.kernels.moe_router import _dx_combine_drouter_impl, dx_combine_drouter

    def call(*a):
        return dx_combine_drouter(*a) if rpp == 512 else _dx_combine_drouter_impl(*a, rows_per_prog=rpp)
    gen = torch.Generator().manual_seed(7 + n)
    m = padded_rows(n, E)
    dx_pad = torch.randn(m, D, generator=gen).bfloat16().to(DEV)
    dest = torch.randperm(m, generator=gen)[:n].to(torch.int32).to(DEV)
    dlogits_bf = (1e-3 * torch.randn(n, E, generator=gen)).bfloat16().to(DEV)
    router_w = (0.05 * torch.randn(E, D, generator=gen)).bfloat16().to(DEV)
    rows = _rows(n, gen)

    dx, drouter = call(dx_pad, dest, dlogits_bf, router_w, rows)
    # dx: bitwise the stock dx_combine (which the op feeds dlogits_bf.float())
    dx_ref = _dx_combine_impl(dx_pad, dest, dlogits_bf.float(), router_w)
    assert dx.dtype == torch.bfloat16 and torch.equal(dx, dx_ref)
    # drouter: dlogits_bf.T @ rows, fp32 accumulation, one bf16 rounding -> within one bf16 step of the exact value
    exact = (dlogits_bf.double().T @ rows.double())
    assert drouter.dtype == torch.bfloat16 and drouter.shape == (E, D)
    torch.testing.assert_close(drouter.double(), exact, rtol=2 ** -7, atol=1e-6 * exact.abs().max().item())
    stock = dlogits_bf.t() @ rows
    torch.testing.assert_close(drouter.float(), stock.float(), rtol=2 ** -6, atol=1e-6 * exact.abs().max().item())
    # deterministic: a second call gives the same bits
    dx2, drouter2 = call(dx_pad, dest, dlogits_bf, router_w, rows)
    assert torch.equal(dx2, dx) and torch.equal(drouter2, drouter)


def _hopper():
    return CUDA and torch.cuda.get_device_capability() == (9, 0)


@pytest.mark.skipif(not _hopper(), reason="the experts' fp8 grouped kernels need an H100 (sm_90)")
def test_whole_op_matches_stock_on_hopper():
    """RoutedFP8MoEFusedRouter vs moe_mlp.RoutedFP8MoE, forward and backward, same inputs and caches, record shapes
    (T = 16384 tokens, E = 8 experts of hidden 1024). A single near-tie routed differently changes bank_grad for two
    experts, so the comparison needs identical routing: it redraws the input (up to 5 times) until both routers pick
    the same expert for every token, then checks every output and gradient to fp8-GEMM tolerance."""
    from track_1_short.perf.kernels import moe_mlp
    from track_1_short.perf.kernels.moe_mlp import RoutedFP8MoE
    from track_1_short.perf.kernels.moe_router import RoutedFP8MoEFusedRouter
    moe_mlp.prime_grouped_stage_cache(E)
    T, H = 16384, 1024
    e4m3 = torch.float8_e4m3fn
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)

    def quant(w):       # [E, h, d] bf16 -> (row [E*h, d], col (E, h, d) column-major, scales [E])
        s = (w.float().abs().amax(dim=(1, 2)) / 448.0).clamp(min=1e-12)
        q = (w.float() / s[:, None, None]).clamp(-448, 448).to(e4m3)
        col = torch.empty(E, D, H, dtype=e4m3, device=dev).transpose(1, 2)
        col.copy_(q)
        return q.reshape(E * H, D).contiguous(), col, s

    w_fc = (0.03 * torch.randn(E, H, D, device=dev, generator=gen)).bfloat16()
    w_proj = (0.03 * torch.randn(E, H, D, device=dev, generator=gen)).bfloat16()
    up_row, up_col, s1 = quant(w_fc)
    down_row, down_col, s2 = quant(w_proj)
    router = (0.1 * torch.randn(E, D, device=dev, generator=gen)).bfloat16()
    f32 = dict(dtype=torch.float32, device=dev)

    def run(fn, x):
        rows = x.clone().requires_grad_(True)
        rw = router.clone().requires_grad_(True)
        post = torch.ones((), dtype=torch.bfloat16, device=dev, requires_grad=True)
        bank_grad = torch.zeros(2 * E, H, D, dtype=torch.bfloat16, device=dev)
        post_amax, dpre_amax = torch.zeros(1, **f32), torch.zeros(1, **f32)
        out, probs, counts = fn.apply(
            rows, rw, post, w_fc, w_proj, up_row, up_col, down_row, down_col, s1, s2,
            torch.tensor(2.0 ** -4, **f32), torch.tensor(2.0 ** -6, **f32), torch.zeros(1, **f32),
            torch.full((1,), 4.0, **f32), post_amax, torch.full((1,), 2.0 ** -6, **f32), dpre_amax,
            bank_grad, 0, torch.zeros(1, dtype=torch.int32, device=dev), 1)
        g = torch.randn(out.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(1)).bfloat16()
        aux = E * (counts.float() / T * probs.mean(0)).sum()
        ((out.float() * g.float()).sum() + 0.01 * aux).backward()
        torch.cuda.synchronize()
        return out, probs, counts, rows.grad, rw.grad, post.grad, bank_grad

    for seed in range(5):
        x = F.rms_norm(torch.randn(T, D, device=dev, generator=torch.Generator(device=dev).manual_seed(10 + seed)),
                       (D,)).bfloat16()
        ref = run(RoutedFP8MoE, x)
        new = run(RoutedFP8MoEFusedRouter, x)
        if torch.equal(ref[2], new[2]) and torch.equal(ref[1].argmax(-1), new[1].argmax(-1)):
            break
    else:
        pytest.fail("the fused and stock routers disagreed on the routing for 5 draws in a row")
    out_r, probs_r, counts_r, dx_r, drw_r, dpost_r, bank_r = ref
    out_n, probs_n, counts_n, dx_n, drw_n, dpost_n, bank_n = new
    torch.testing.assert_close(probs_n, probs_r, rtol=3e-2, atol=1e-3)
    torch.testing.assert_close(out_n.float(), out_r.float(), rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(bank_n.float(), bank_r.float(), rtol=3e-2, atol=1e-2 * bank_r.float().abs().max().item())
    torch.testing.assert_close(dx_n.float(), dx_r.float(), rtol=3e-2, atol=1e-2 * dx_r.float().abs().max().item())
    torch.testing.assert_close(drw_n.float(), drw_r.float(), rtol=3e-2, atol=1e-2 * drw_r.float().abs().max().item())
    torch.testing.assert_close(dpost_n.float(), dpost_r.float(), rtol=3e-2, atol=1e-3)
