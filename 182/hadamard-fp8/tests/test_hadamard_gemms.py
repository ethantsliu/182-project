"""CPU tests: HADAMARD_BLOCK (block-diagonal rotation inside 128 tiles) and HADAMARD_GEMMS (per-GEMM
selection), against perf/hadamard_ref.mlp_fp8_reference, the pure-torch model of the fp8 MLP kernels.

    python -m pytest 182/hadamard-fp8/tests -q
"""
import itertools
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from track_1_short.perf.hadamard_ref import (  # noqa: E402
    MLP_GEMMS, hadamard_matrix, mlp_fp8_reference, mlp_reference_scales, rotate, rotate_in_tiles, rotation_block,
    signed_hadamard)

SUBSETS = [frozenset(c) for n in range(5) for c in itertools.combinations(sorted(MLP_GEMMS), n)]


@pytest.mark.parametrize("block", [32, 64, 128])
def test_in_tile_rotation_is_block_diagonal_rotate(block):
    """The kernels' reshape-and-dot on a 128-wide tile is exactly rotate() with the b x b Q, for rows of the
    tile and (through a transpose) for its columns."""
    torch.manual_seed(0)
    q = signed_hadamard(block, seed=9)
    tile = torch.randn(128, 128, dtype=torch.float64)
    assert torch.allclose(rotate_in_tiles(tile, q), rotate(tile, 1, q), atol=1e-12)
    assert torch.allclose(rotate_in_tiles(tile.T, q), rotate(tile, 0, q).T, atol=1e-12)
    r = torch.block_diag(*([rotation_block(block, 9)] * (128 // block)))
    assert torch.allclose(r @ r.T, torch.eye(128, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(rotate_in_tiles(tile, q), tile @ r, atol=1e-12)


def test_h128_factors_as_kronecker():
    """The factored alternative considered in perf/kernels/hadamard.py: Sylvester H_128 = H_16 (x) H_8."""
    h16, h8 = hadamard_matrix(16, torch.float64), hadamard_matrix(8, torch.float64)
    assert torch.equal(torch.kron(h16, h8), hadamard_matrix(128, torch.float64))


def _case(seed=0, T=256, D=256, H=384):
    torch.manual_seed(seed)
    x = torch.randn(T, D)
    x = (x / x.pow(2).mean(1, keepdim=True).sqrt()).bfloat16()
    W1, W2 = (0.03 * torch.randn(H, D)).bfloat16(), (0.03 * torch.randn(H, D)).bfloat16()
    g = (1e-2 * torch.randn(T, D)).bfloat16()
    return x, W1, W2, g


def _exact_grads(x, W1, W2, g):
    xr, W1r, W2r = (t.double().requires_grad_() for t in (x, W1, W2))
    (torch.relu(xr @ W1r.T).square() @ W2r).backward(g.double())
    return dict(dx=xr.grad, dW1=W1r.grad, dW2=W2r.grad)


@pytest.mark.parametrize("block", [32, 128])
def test_forward_and_post_amax_never_see_the_rotation(block):
    """Risk (b): under any HADAMARD_GEMMS the forward output, post_f8 and the post amax that sets the next
    step's post scale are bit-identical to the record's; the rotated post_t has its own amax."""
    x, W1, W2, g = _case()
    q = signed_hadamard(block, 1234)
    s = mlp_reference_scales(x, W1, W2, g, q, MLP_GEMMS)
    rec = mlp_fp8_reference(x, W1, W2, g, q, frozenset(), s)
    for gemms in SUBSETS:
        r = mlp_fp8_reference(x, W1, W2, g, q, gemms, s)
        assert torch.equal(r["out"], rec["out"]) and torch.equal(r["post_f8"], rec["post_f8"])
        assert torch.equal(r["post_amax"], rec["post_amax"])
        assert (r["post_t_amax"] is not None) == ("dw2" in gemms)
    r = mlp_fp8_reference(x, W1, W2, g, q, frozenset({"dw2"}), s)
    assert not torch.equal(r["post_t_amax"], rec["post_amax"])  # the two scales really are separate


def test_unselected_gemms_read_the_record_operands():
    """A GEMM outside HADAMARD_GEMMS gives the record's result exactly. dW2 depends only on dw2; dpre on
    dpost; dW1 on dpost and dw1; dx on dpost and dx."""
    x, W1, W2, g = _case(1)
    q = signed_hadamard(64, 1234)
    s = mlp_reference_scales(x, W1, W2, g, q, MLP_GEMMS)
    rec = mlp_fp8_reference(x, W1, W2, g, q, frozenset(), s)
    deps = dict(dW2={"dw2"}, dpre={"dpost"}, dW1={"dpost", "dw1"}, dx={"dpost", "dx"})
    for gemms in SUBSETS:
        r = mlp_fp8_reference(x, W1, W2, g, q, gemms, s)
        for name, needs in deps.items():
            assert torch.equal(r[name], rec[name]) == (not (needs & gemms)), (sorted(gemms), name)


@pytest.mark.parametrize("block", [32, 64, 128])
@pytest.mark.parametrize("gemms", [frozenset(), MLP_GEMMS, frozenset({"dw1", "dw2"}), frozenset({"dpost"})],
                         ids=["none", "all", "dw", "dpost"])
def test_every_variant_is_as_accurate_as_the_record(block, gemms):
    x, W1, W2, g = _case(2)
    q = signed_hadamard(block, 1234)
    exact = _exact_grads(x, W1, W2, g)
    s = mlp_reference_scales(x, W1, W2, g, q, gemms)
    rec = mlp_fp8_reference(x, W1, W2, g, q, frozenset(), s)
    r = mlp_fp8_reference(x, W1, W2, g, q, gemms, s)
    rel = lambda a, b: ((a.double() - b).norm() / b.norm()).item()
    for name in ("dx", "dW1", "dW2"):
        assert rel(r[name], exact[name]) < 0.12
        assert rel(r[name], exact[name]) < 1.1 * rel(rec[name], exact[name]) + 1e-3


def _config(env):
    code = ("import track_1_short.config as c; print(c.HADAMARD_BLOCK, sorted(c.HADAMARD_GEMMS), "
            "c.HADAMARD_DW1, c.HADAMARD_DW2, c.HADAMARD_DX, c.HADAMARD_DPOST)")
    full = {k: v for k, v in os.environ.items() if not k.startswith("HADAMARD")} | env
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=full, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("env,want", [
    ({}, "128 [] False False False False"),
    ({"HADAMARD_FP8": "1"}, "128 ['dpost', 'dw1', 'dw2', 'dx'] True True True True"),
    ({"HADAMARD_FP8": "1", "HADAMARD_GEMMS": "dw1, DW2", "HADAMARD_BLOCK": "32"}, "32 ['dw1', 'dw2'] True True False False"),
    ({"HADAMARD_GEMMS": "dx"}, "128 [] False False False False"),  # inert without HADAMARD_FP8
])
def test_config_knobs(env, want):
    res = _config(env)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == want


@pytest.mark.parametrize("env", [{"HADAMARD_BLOCK": "16"}, {"HADAMARD_FP8": "1", "HADAMARD_GEMMS": "dw3"}])
def test_config_rejects_bad_values(env):
    assert _config(env).returncode != 0


def test_gpu_sanity_stage_labels_are_distinct():
    """gpu_sanity (i) keys its stage table by describe_stage_key: every linear_relu_square variant
    prime_stage_cache can reach must get its own label, or a rotated variant's stage count overwrites the
    record's (the old 7-field reading labelled the rotated forward 'fwd-fp8-emit', like the record's)."""
    sys.path.insert(0, str(ROOT / "182" / "hadamard-fp8"))
    import gpu_sanity
    tail = (128, 256, 128, 8)
    keys = [(True, False, False, False, True, True, False, False, 0) + (128, 256, 64, 8),   # bf16 forward
            (True, True, True, True, False, False, False, False, 0) + tail,                 # fp8 forward
            (False, True, False, False, True, True, False, False, 0) + tail]                # fp8 backward
    for hb in (32, 64, 128):
        keys.append((True, True, True, True, False, False, False, True, hb) + tail)
        keys += [(False, True, False, False, True, True, rr, rt, hb) + tail for rr, rt in ((1, 1), (1, 0), (0, 1))]
    labels = [gpu_sanity.describe_stage_key(k) for k in keys]
    assert len(set(labels)) == len(labels), labels
    assert labels[1] == "fwd-fp8-emit" and labels[3] == "fwd-fp8-emit-ROT_T-HB32"
