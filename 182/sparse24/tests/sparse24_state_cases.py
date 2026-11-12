"""CPU cases of the SPARSE24_MLP state in model/gpt.py, on a real (CPU-built) GPT: the operand buffers, the refresh
that replaces the fp8 MLP refresh, the per-slot dispatch in GPT._mlp (sparse in training and validation, the dense
bf16 tail), the refresh-graph prefix; and, with the flag off, that none of it exists.

Not collected directly: test_sparse24_state.py runs this file in subprocesses with TRITON_INTERPRET=1, CUDA hidden
and SPARSE24_* set per case (config.py reads them at import). SPARSE24_BACKEND=dense in every case: the
cuSPARSELt path needs a GPU (tests/test_gpu_sparse24.py).

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/sparse24/tests/test_sparse24_state.py -q

gpt.py is imported with its CUDA-only modules stubbed, as 182/fp8-attn-o/tests/attn_o_state_cases.py does.
"""
import os
import pathlib
import sys
from unittest import mock

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") != "1" or "SPARSE24_CASE" not in os.environ:
    pytest.skip("run through test_sparse24_state.py (needs TRITON_INTERPRET=1 and the case env at import)",
                allow_module_level=True)
pytest.importorskip("triton")

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

if not torch.cuda.is_available():
    import types

    import track_1_short.perf.bf16_loss as _b
    for _n in ("track_1_short.perf.kernels.cross_entropy", "track_1_short.perf.kernels.sampled_cross_entropy"):
        sys.modules.setdefault(_n, mock.MagicMock(name=_n))
    _c = sys.modules["track_1_short.perf.kernels.cross_entropy"]
    _c.SOFTCAP_A, _c.SOFTCAP_B, _c.SOFTCAP_C = _b.SOFTCAP_A, _b.SOFTCAP_B, _b.SOFTCAP_C
    _c.lm_head_inverse_scale = lambda w_s, device: torch.tensor(1.0 / w_s, device=device)
    # model/attention.py as it is, minus the FA3 load at import (nothing here runs attention)
    _src = (ROOT / "track_1_short/model/attention.py").read_text()
    _marker = "flash_attn_interface = load_flash_attn3()"
    assert _src.count(_marker) == 1
    sys.modules.setdefault("kernels", types.SimpleNamespace(get_kernel=None))
    _attn = types.ModuleType("track_1_short.model.attention")
    exec(compile(_src.replace(_marker, "flash_attn_interface = None"), "attention.py", "exec"), _attn.__dict__)
    sys.modules["track_1_short.model.attention"] = _attn

from track_1_short import config  # noqa: E402
from track_1_short.config import BLOCK_SIZE, TRAINING_STAGES, WS_POST_YARN_EXT  # noqa: E402
from track_1_short.model import gpt  # noqa: E402
from track_1_short.perf import sparse24 as S  # noqa: E402
from track_1_short.perf.cuda_graphs.fp8_refresh_graphs import Fp8RefreshGraphs  # noqa: E402

CASE = os.environ["SPARSE24_CASE"]  # off | greedy | optimal | tail13
ON = CASE != "off"
KIND = "optimal" if CASE == "optimal" else "greedy"
NAMES = {"_mlp_sp24_w1", "_mlp_sp24_w2"}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = gpt.GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768, max_seq_len=2048,
                ngram_dim=64, world_size=8, device=torch.device("cpu"))
    m.cast_matrix_weights_bf16()
    with torch.no_grad():  # trained-looking c_proj (zero at init)
        m.mlp_bank[:, 1].normal_(0, 0.02)
    return m


def test_config_case():
    assert config.SPARSE24_MLP == ON and gpt.SPARSE24_MLP == ON
    assert config.SPARSE24_BACKEND == "dense"
    assert config.SPARSE24_MASK == KIND
    assert config.SPARSE24_DENSE_FROM_WS == (13 if CASE == "tail13" else 0)


def test_buffers_exist_iff_flag(model):
    names = {n for n, _ in model.named_buffers() if n.startswith("_mlp_sp24_")}
    if not ON:
        assert not names
        return
    assert names == NAMES == set(S.SPARSE24_BUFFER_NAMES)
    assert names.isdisjoint(model.state_dict())  # not persistent: rebuilt by every refresh
    assert model._mlp_sp24_w1.shape == (gpt.NUM_MLP_SLOTS, gpt.MLP_HIDDEN_DIM, 768)
    assert model._mlp_sp24_w1.dtype == torch.bfloat16


def test_refresh_graph_captures_the_operands(model):
    graphs = Fp8RefreshGraphs(model)
    got = {n for n in graphs.mlp_outputs if n.startswith("_mlp_sp24_")}
    assert got == (NAMES if ON else set())


def test_refresh_writes_masked_bank_and_skips_fp8(model):
    up_f8 = model._mlp_up_f8.clone()
    calls = model.mlp_quantize_calls
    model.quantize_mlp_fp8(refresh_lm=False)
    assert model.mlp_quantize_calls == calls + 1
    if not ON:
        assert not torch.equal(model._mlp_up_f8.view(torch.uint8), up_f8.view(torch.uint8))  # the record's refresh ran
        return
    assert torch.equal(model._mlp_up_f8.view(torch.uint8), up_f8.view(torch.uint8))  # fp8 MLP refresh skipped
    for slot in (0, 5, gpt.PARALLEL_MLP_SLOT):
        for j, name in ((0, "_mlp_sp24_w1"), (1, "_mlp_sp24_w2")):
            w = model.mlp_bank[slot, j].detach()
            got = getattr(model, name)[slot]
            assert torch.equal(got, w * S.transposable_mask(w, KIND)), (slot, name)
            assert S.is_24(got != 0, -1) and S.is_24(got != 0, -2)
    assert gpt._fp8_sat_dead_slots("post") == set(range(gpt.NUM_MLP_SLOTS))


def _slot_io(model, slot, T=48):
    torch.manual_seed(slot)
    x = torch.randn(1, T, 768)
    x = (x / x.pow(2).mean(-1, keepdim=True).sqrt()).bfloat16().requires_grad_()
    mlp_all = model.mlp_bank.flatten(0, 1).unbind(0)
    return x, (mlp_all[0::2], mlp_all[1::2])


@pytest.mark.parametrize("training", [True, False])
def test_mlp_dispatch_sparse(model, training):
    if not ON:
        pytest.skip("flag off")
    model.quantize_mlp_fp8(refresh_lm=False)
    model.train(training)
    slot = 3
    x, weights = _slot_io(model, slot)
    out = model._mlp(x, weights, slot, None, sparse24=True)
    W1m, W2m = model._mlp_sp24_w1[slot].float(), model._mlp_sp24_w2[slot].float()
    ref = (torch.relu(x.detach().float().view(-1, 768) @ W1m.T) ** 2 @ W2m).view(x.shape)
    assert out.dtype == torch.bfloat16
    assert float((out.detach().float() - ref).norm() / ref.norm()) < 2e-2
    if training:
        model.mlp_bank.grad = None
        out.float().square().sum().backward()
        g = model.mlp_bank.grad
        assert g is not None and g[slot].abs().sum() > 0
        assert g[slot, 0][model._mlp_sp24_w1[slot] == 0].abs().sum() > 0  # straight-through: pruned entries too
        others = [s for s in range(gpt.NUM_MLP_SLOTS) if s != slot]
        assert g[others].abs().sum() == 0
        model.mlp_bank.grad = None
    model.train(True)


def test_mlp_dispatch_dense_paths_unchanged(model):
    """sparse24=False keeps the existing dispatch: bf16 dense in training (no fp8 args under the flag), the
    validation kernel in eval. With the flag off this is the record's _mlp, called without the new kwarg."""
    slot = 2
    x, weights = _slot_io(model, slot)
    c_fc, c_proj = weights[0][slot], weights[1][slot]
    with mock.patch.object(gpt, "ReLUSqrdMLPBF16", side_effect=lambda *a: ("bf16", len(a))) as bf16, \
         mock.patch.object(gpt, "ReLUSqrdMLP", side_effect=lambda *a: ("fp8-or-val", len(a))) as rec:
        model.train(True)
        assert model._mlp(x, weights, slot, None) == ("bf16", 3)
        model.train(False)
        assert model._mlp(x, weights, slot, None) == ("fp8-or-val", 3)
        assert model._mlp(x, weights, slot, None, sparse24=False) == ("fp8-or-val", 3)
        model.train(True)
    assert bf16.call_count == 1 and rec.call_count == 2


def test_sparse_by_stage_and_validation():
    ws = [s.window_sizes[1] * BLOCK_SIZE for s in TRAINING_STAGES] + [WS_POST_YARN_EXT * BLOCK_SIZE]
    want = {"off": [False] * 6, "greedy": [True] * 6, "optimal": [True] * 6,
            "tail13": [True, True, True, True, False, False]}[CASE]
    assert [S.sparse24_active(w) for w in ws] == want


def test_rearm_then_refresh_rebuilds(model):
    if not ON:
        pytest.skip("flag off")
    with torch.no_grad():
        model._mlp_sp24_w1.zero_()
    model.rearm_fp8_bootstrap()
    model.quantize_mlp_fp8(refresh_lm=True)
    w = model.mlp_bank[0, 0].detach()
    assert torch.equal(model._mlp_sp24_w1[0], w * S.transposable_mask(w, KIND))
