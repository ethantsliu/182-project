"""CPU tests: the block-Hadamard reference (perf/hadamard_ref.py) and the FP8_OFF=lm bf16 loss.

    python -m pytest 182/hadamard-fp8/tests -q
"""
import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from track_1_short.perf.bf16_loss import (  # noqa: E402
    SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, BF16SoftcappedCrossEntropy, bf16_training_loss)
from track_1_short.perf.hadamard_ref import (  # noqa: E402
    fake_quant, hadamard_matrix, quantized_rotated_matmul, random_signs, rotate, rotated_matmul,
    rotation_block, signed_hadamard, unrotate)

BLOCKS = (128, 256)


def block_diag_rotation(n, block, seed=0):
    r = rotation_block(block, seed)
    return torch.block_diag(*([r] * (n // block)))


@pytest.mark.parametrize("n", [1, 2, 4, 64, 128, 256])
def test_hadamard_matrix_orthogonal(n):
    h = hadamard_matrix(n, dtype=torch.float64)
    assert torch.equal(h, h.T)
    assert torch.equal(h @ h.T, n * torch.eye(n, dtype=torch.float64))


@pytest.mark.parametrize("block", BLOCKS)
def test_signed_hadamard_exact_in_bf16_and_orthogonal(block):
    q = signed_hadamard(block, seed=7)
    assert q.dtype == torch.bfloat16 and set(q.float().unique().tolist()) == {-1.0, 1.0}
    r = rotation_block(block, seed=7)
    assert torch.allclose(r @ r.T, torch.eye(block, dtype=torch.float64), atol=1e-12)
    s = random_signs(block, 7)
    assert torch.equal(signed_hadamard(block, 7).double(), s.double()[:, None] * hadamard_matrix(block, torch.float64))


@pytest.mark.parametrize("block", BLOCKS)
def test_rotate_is_block_diagonal_right_multiply(block):
    torch.manual_seed(0)
    n = 3 * block
    x = torch.randn(5, n, dtype=torch.float64)
    q = signed_hadamard(block, seed=0)
    assert torch.allclose(rotate(x, -1, q), x @ block_diag_rotation(n, block), atol=1e-12)
    # along dim 0 it is the same transform applied to each column vector
    y = torch.randn(n, 4, dtype=torch.float64)
    assert torch.allclose(rotate(y, 0, q), (y.T @ block_diag_rotation(n, block)).T, atol=1e-12)
    assert torch.allclose(unrotate(rotate(x, -1, q), -1, q), x, atol=1e-12)


@pytest.mark.parametrize("block", BLOCKS)
def test_rotated_gemm_equals_gemm(block):
    """(A H)(H^T B) == A B: the contraction dim K is rotated on both operands."""
    torch.manual_seed(1)
    M, K, N = 96, 4 * block, 80
    a = torch.randn(M, K, dtype=torch.float64)
    b = torch.randn(K, N, dtype=torch.float64)
    q = signed_hadamard(block, seed=3)
    assert torch.allclose(rotated_matmul(a, b, q), a @ b, atol=1e-10)
    R = block_diag_rotation(K, block, seed=3)
    assert torch.allclose((a @ R) @ (R.T @ b), a @ b, atol=1e-10)


@pytest.mark.parametrize("block", BLOCKS)
def test_weight_gradient_token_rotation(block):
    """dW = A.T @ B with k = tokens (the MLP's dW1 / dW2): both [T, *] operands rotated along tokens."""
    torch.manual_seed(2)
    T = 2 * block
    a = torch.randn(T, 48, dtype=torch.float64)  # e.g. dpre or post
    b = torch.randn(T, 32, dtype=torch.float64)  # e.g. x or g
    q = signed_hadamard(block, seed=5)
    assert torch.allclose(rotate(a, 0, q).T @ rotate(b, 0, q), a.T @ b, atol=1e-10)


def test_mlp_gradient_gemms_with_kernel_layouts():
    """The four MLP gradient GEMMs exactly as perf/kernels/hadamard.py + mlp.py lay them out, in fp64:
    each rotated pair reproduces the unrotated product."""
    torch.manual_seed(3)
    T, D, H, b = 256, 256, 384, 128
    q = signed_hadamard(b, seed=11)
    x, g = torch.randn(T, D, dtype=torch.float64), torch.randn(T, D, dtype=torch.float64)
    W1, W2 = torch.randn(H, D, dtype=torch.float64), torch.randn(H, D, dtype=torch.float64)
    pre = x @ W1.T
    post = torch.relu(pre) ** 2
    dpre = 2 * (g @ W2.T) * torch.relu(pre)
    # dW2 = post.T @ g: post_t [H, T] and g_t [D, T], both rotated along T (their last dim)
    post_t, g_t = rotate(post.T, 1, q), rotate(g.T, 1, q)
    assert torch.allclose(post_t @ g_t.T, post.T @ g, atol=1e-8)
    # dpost = g @ W2.T: g row rotated along D, W2 row cache [H, D] rotated along D
    assert torch.allclose(rotate(g, 1, q) @ rotate(W2, 1, q).T, g @ W2.T, atol=1e-8)
    # dW1 = dpre.T @ x: dpre_t [H, T], x_t [D, T], rotated along T
    assert torch.allclose(rotate(dpre.T, 1, q) @ rotate(x.T, 1, q).T, dpre.T @ x, atol=1e-8)
    # dx = dpre @ W1: dpre row rotated along H; W1 column cache stored [D, H] rotated along H
    w1_col_t = rotate(W1.T, 1, q)  # what quantize_weights_rotated(along_h=True) writes, before quantizing
    assert torch.allclose(rotate(dpre, 1, q) @ w1_col_t.T, dpre @ W1, atol=1e-8)


def test_bf16_kernel_arithmetic_matches_reference():
    """The kernels compute tl.dot(bf16 tile, bf16 Q) in fp32, times 1/sqrt(b). Q is exact in bf16, so this
    matches the fp64 rotation of the bf16 tile to fp32 accumulation error."""
    torch.manual_seed(4)
    tile = torch.randn(128, 128).bfloat16()
    q = signed_hadamard(128, seed=1)
    kernel_like = (tile.float() @ q.float()) * (1 / math.sqrt(128))
    ref = rotate(tile.double(), 1, q)
    assert torch.allclose(kernel_like.double(), ref, rtol=0, atol=1e-5)
    assert torch.allclose(rotate(tile, 1, q).double(), ref, atol=1e-5)


def _outlier_gemm():
    torch.manual_seed(5)
    T, K, N = 512, 512, 64
    a = torch.randn(T, K)
    a[:, ::97] *= 60.0  # outlier channels along K
    b = torch.randn(K, N)
    ref = a.double() @ b.double()
    err = lambda c: float(((c.double() - ref) ** 2).sum() / (ref ** 2).sum())
    return a, b, err


def test_rotation_barely_matters_for_fp8_at_amax_scale():
    """e4m3 is a float format: at an amax per-tensor scale its relative rounding error is ~scale-free, so
    outlier channels cost little and the rotation buys little (the prior behind the README's rule)."""
    a, b, err = _outlier_gemm()
    q, e4 = signed_hadamard(128, seed=2), torch.float8_e4m3fn
    plain = fake_quant(a, a.abs().max() / 448, e4) @ fake_quant(b, b.abs().max() / 448, e4)
    ra, rb = rotate(a, 1, q), rotate(b, 0, q)
    rot = quantized_rotated_matmul(a, b, q, ra.abs().max() / 448, rb.abs().max() / 448, e4, e4)
    assert 0.7 < err(rot) / err(plain) < 1.3


def test_rotation_avoids_static_scale_saturation():
    """Where it does matter: a static scale that the outliers saturate. Rotation spreads each outlier
    over its block, the amax drops below the static limit, and the error collapses."""
    a, b, err = _outlier_gemm()
    q, e4 = signed_hadamard(128, seed=2), torch.float8_e4m3fn
    static = 32.0 / 448  # one static scale for both: |a| reaches ~250, the rotated |a R| stays below 32
    assert float(a.abs().max()) > 64 and float(rotate(a, 1, q).abs().max()) < 32
    plain = fake_quant(a, static, e4) @ fake_quant(b, b.abs().max() / 448, e4)
    rot = quantized_rotated_matmul(a, b, q, static, rotate(b, 0, q).abs().max() / 448, e4, e4)
    assert err(rot) < 0.1 * err(plain)


# -----------------------------------------------------------------------------
# FP8_OFF=lm bf16 loss: forward matches the CE kernel's per-row loss, backward matches the kernel's
# gradient convention (softmax term scaled by S_w = sum(mtp_weights) + prefix weight).

def _kernel_reference(x, W, targets, mtp_w, prefix, pw):
    """Per-row loss (the kernel's) and a surrogate whose autograd gradient is the kernel's gradient."""
    n, V = x.shape[0], W.shape[1]
    z = SOFTCAP_A * torch.sigmoid(((x @ W).float() + SOFTCAP_B) / SOFTCAP_C)
    lse = torch.logsumexp(z, 1)
    loss = torch.zeros(n)
    surrogate = (mtp_w.sum() + pw * (prefix >= 0).float()) * lse
    for k in range(mtp_w.numel()):
        for i in range(n):
            if i + k < n and 0 <= targets[i + k] < V:
                loss[i] += mtp_w[k] * (lse[i] - z[i, targets[i + k]])
                surrogate[i] = surrogate[i] - mtp_w[k] * z[i, targets[i + k]]
    for i in range(n):
        if prefix[i] >= 0:
            loss[i] += pw * (lse[i] - z[i, prefix[i]])
            surrogate[i] = surrogate[i] - pw * z[i, prefix[i]]
    return loss, surrogate


@pytest.mark.parametrize("n_predict", [1, 3])
def test_bf16_loss_matches_kernel_convention(n_predict, monkeypatch):
    import track_1_short.perf.bf16_loss as bl
    monkeypatch.setattr(bl, "SLAB_ROWS", 7)  # several slabs, one partial
    torch.manual_seed(6)
    n, D, V = 20, 16, 40
    x = torch.randn(n, D).bfloat16().requires_grad_()
    W = (0.5 * torch.randn(D, V)).bfloat16().requires_grad_()
    targets = torch.randint(0, V, (n,))
    prefix = torch.where(torch.rand(n) < 0.5, torch.randint(0, V, (n,)), torch.full((n,), -1))
    mtp_w = torch.tensor([1.0, 0.5, 0.25][:n_predict])
    pw = torch.tensor(0.2)
    loss = BF16SoftcappedCrossEntropy.apply(x, W, targets, mtp_w, prefix, pw)
    loss.sum().backward()
    xr = x.detach().clone().requires_grad_()
    Wr = W.detach().clone().requires_grad_()
    ref_loss, surrogate = _kernel_reference(xr, Wr, targets, mtp_w, prefix, pw)
    surrogate.sum().backward()
    assert torch.allclose(loss, ref_loss, atol=1e-4)
    # bf16 GEMM / logit-gradient rounding on our side; the reference's matmul is bf16 too.
    assert torch.allclose(x.grad.float(), xr.grad.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(W.grad.float(), Wr.grad.float(), atol=2e-2, rtol=2e-2)


def test_bf16_sampled_loss_dense_gradient():
    torch.manual_seed(7)
    n, D, V, P = 12, 8, 30, 10
    x = torch.randn(n, D).bfloat16()
    W = torch.randn(D, V).bfloat16().requires_grad_()
    cand = torch.sort(torch.randperm(V)[:P]).values
    target_pos = torch.randint(0, P, (n,))
    loss = bf16_training_loss(x, W, target_pos, torch.tensor([1.0]), torch.full((n,), -1), torch.tensor(0.0),
                              candidates=cand)
    loss.sum().backward()
    others = torch.ones(V, dtype=torch.bool)
    others[cand] = False
    assert W.grad.shape == (D, V) and torch.all(W.grad[:, others] == 0) and torch.any(W.grad[:, cand] != 0)


# -----------------------------------------------------------------------------
# The dump format round trip and the offline script (CPU, tiny shapes).

def test_dump_roundtrip_and_script(tmp_path):
    import subprocess
    from track_1_short.perf.fp8_dump import Fp8Dump
    torch.manual_seed(8)
    T, D, H = 512, 256, 384
    dump = Fp8Dump()
    dump.begin(rows=300)  # not a multiple of 256: the script trims to 256
    x, g = torch.randn(T, D).bfloat16(), (1e-4 * torch.randn(T, D)).bfloat16()
    W1, W2 = (0.02 * torch.randn(H, D)).bfloat16(), (0.02 * torch.randn(H, D)).bfloat16()
    dump.save_mlp("mlp0", x, W1, W2, g, torch.tensor(0.9), x_scale=torch.tensor(2.0 ** -4))
    dump.save_qkv("qkv0", x, (0.02 * torch.randn(768, D)).bfloat16(), (1e-4 * torch.randn(T, 768)).bfloat16())
    assert dump.records["mlp0"]["x"].shape == (300, D) and dump.records["mlp0"]["p_fold"] == pytest.approx(0.9)
    path = tmp_path / "step000100.pt"
    assert dump.end(str(path), dict(step=100)) == 2 and not dump.active()
    out = tmp_path / "r.json"
    script = ROOT / "182" / "hadamard-fp8" / "fp8_quant_error.py"
    res = subprocess.run([sys.executable, str(script), "--dump", str(tmp_path), "--json", str(out)],
                         capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "dW1 dpre.T@x" in res.stdout and out.exists()


def test_script_synthetic_smoke():
    import subprocess
    script = ROOT / "182" / "hadamard-fp8" / "fp8_quant_error.py"
    res = subprocess.run([sys.executable, str(script), "--synthetic", "--max-rows", "512"],
                         capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "geometric-mean" in res.stdout
