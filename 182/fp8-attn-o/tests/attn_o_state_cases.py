"""CPU cases of the FP8_ATTN_O state in model/gpt.py, on a real (CPU-built) GPT: the o_w caches the attention refresh
writes, the per-layer Function args, the delayed scales, the HADAMARD_SAT_STATS rows, the post-warmup rearm and
the refresh-graph prefix; and, with the flag off, that none of it exists.

Not collected directly: test_attn_o_state.py runs this file in subprocesses with TRITON_INTERPRET=1, CUDA hidden
and FP8_ATTN_O / FP8_ATTN_O_SCALES / HADAMARD_SAT_STATS set per case (config.py reads them at import):

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/fp8-attn-o/tests/test_attn_o_state.py -q

gpt.py is imported with its CUDA-only modules stubbed, as 182/hadamard-fp8/tests/hadamard_state_cases.py does.
"""
import os
import pathlib
import sys
from unittest import mock

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") != "1" or "FP8_ATTN_O_CASE" not in os.environ:
    pytest.skip("run through test_attn_o_state.py (needs TRITON_INTERPRET=1 and the case env at import)",
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
from track_1_short.model import gpt  # noqa: E402
from track_1_short.perf import attn_o_scales as S  # noqa: E402
from track_1_short.perf.cuda_graphs.fp8_refresh_graphs import Fp8RefreshGraphs  # noqa: E402

CASE = os.environ["FP8_ATTN_O_CASE"]  # off | static | static_sat | delayed
E4 = torch.float8_e4m3fn


def bytes_equal(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = gpt.GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768, max_seq_len=2048,
                ngram_dim=64, world_size=8, device=torch.device("cpu"))
    m.cast_matrix_weights_bf16()
    with torch.no_grad():  # trained-looking O: a little bigger than init, one outlier
        m.vo_bank.mul_(1.7)
        m.vo_bank[3, 5, 7] = 0.5
        m.scalars[3] = 0.8  # sa_lambdas[1] of layer 1
    return m


def test_config_case():
    assert config.FP8_ATTN_O == (CASE != "off")
    assert config.FP8_ATTN_O_SCALES == ("delayed" if CASE == "delayed" else "static")
    assert config.FP8_ATTN_O_AMAX == (CASE in ("static_sat", "delayed"))


def test_buffers_exist_iff_flag(model):
    names = {n for n, _ in model.named_buffers() if n.startswith("_attn_o_")}
    if CASE == "off":
        assert not names
        assert not hasattr(model, "_attn_o_half_slots")
        assert gpt._attn_o_sat_rows() == []
        return
    assert names == {"_attn_o_f8_row", "_attn_o_f8_col", "_attn_o_f8_half_row", "_attn_o_f8_half_col",
                     "_attn_o_w_scales", "_attn_o_w_partial_amax", "_attn_o_y_scales", "_attn_o_g_scales",
                     "_attn_o_y_amax", "_attn_o_g_amax"}
    assert not any(b.requires_grad for n, b in model.named_buffers() if n in names)
    assert names.isdisjoint(model.state_dict())  # none persistent


def test_refresh_graph_captures_the_caches(model):
    graphs = Fp8RefreshGraphs(model)
    o = {n for n in graphs.attn_outputs if n.startswith("_attn_o_")}
    if CASE == "off":
        assert not o
    else:
        assert "_attn_o_f8_row" in o and "_attn_o_y_scales" in o and "_attn_o_g_amax" in o
        assert not {n for n in graphs.mlp_outputs if n.startswith("_attn_o_")}


def test_caches_hold_each_layers_o_w(model):
    if CASE == "off":
        pytest.skip("flag off")
    model.quantize_attn_fp8()
    weights = model._attn_weights()
    for layer in gpt.ATTN_BANK_ORDER:
        o_w = weights[layer][2].detach()  # exactly what the layer's forward reads, cut to its v_dim
        w_row, w_col, w_scale, y_scale, g_scale, y_amax, g_amax = model._attn_o_fp8_args(layer)
        j = gpt.ATTN_BANK_ORDER.index(layer)
        full = model.vo_bank[2 * j + 1].float()
        s = full.abs().max().item() / 448 if CASE == "delayed" else S.ATTN_O_W_SCALE
        assert w_scale.item() == pytest.approx(s, rel=1e-6)
        want = torch.clamp(o_w.float() / w_scale, -448, 448).to(E4)
        assert w_row.shape == o_w.shape and w_col.shape == o_w.T.shape
        assert bytes_equal(w_row, want), layer
        assert bytes_equal(w_col, want.T), layer
        assert w_row.is_contiguous() and w_col.is_contiguous()
        assert y_scale.dim() == g_scale.dim() == w_scale.dim() == 0
        if config.FP8_ATTN_O_AMAX:
            assert y_amax.shape == (1,) and y_amax.data_ptr() == model._attn_o_y_amax[j].data_ptr()
            assert g_amax.data_ptr() == model._attn_o_g_amax[j].data_ptr()
        else:
            assert y_amax is None and g_amax is None
        if config.FP8_ATTN_O_AMAX:
            assert model._attn_o_w_partial_amax[j].max().item() == full.abs().max().item()


def test_half_v_layers_read_packed_copies(model):
    if CASE == "off":
        pytest.skip("flag off")
    for layer in gpt.HALF_V_LAYERS:
        w_row, w_col = model._attn_o_fp8_args(layer)[:2]
        assert w_row.shape == (768, 6 * 64) and w_col.shape == (6 * 64, 768)


def test_delayed_scales_refresh_from_amax(model):
    if CASE == "off":
        pytest.skip("flag off")
    with torch.no_grad():
        model._attn_o_y_amax[2, 0] = 3.0
        model._attn_o_g_amax[5, 0] = 0.07
    y0, g0 = model._attn_o_y_scales.clone(), model._attn_o_g_scales.clone()
    model.quantize_attn_fp8()
    if CASE == "delayed":
        assert model._attn_o_y_scales[2].item() == pytest.approx(3.0 * S.ATTN_O_Y_HEADROOM / 448, rel=1e-6)
        assert model._attn_o_g_scales[5].item() == pytest.approx(0.07 * S.ATTN_O_G_HEADROOM / 57344, rel=1e-6)
        # slots with no amax keep their scale (not floored)
        assert model._attn_o_y_scales[0].item() == y0[0].item()
    else:
        assert torch.equal(model._attn_o_y_scales, y0) and torch.equal(model._attn_o_g_scales, g0)
        assert (model._attn_o_y_scales == S.ATTN_O_Y_SCALE).all() and (model._attn_o_g_scales == S.ATTN_O_G_SCALE).all()
    if config.FP8_ATTN_O_AMAX:
        assert (model._attn_o_y_amax == 0).all() and (model._attn_o_g_amax == 0).all()
    else:  # nothing reads or clears them
        assert model._attn_o_y_amax[2, 0].item() == 3.0


def test_sat_snapshot_rows(model):
    model.quantize_attn_fp8()
    names, snap = model.fp8_sat_snapshot()
    if CASE == "off":
        assert names == ["post", "dpre", "w1", "w2"]
        return
    assert names[-3:] == ["o_y", "o_g", "o_w"]
    assert snap.shape == (len(names), 3, gpt.NUM_MLP_SLOTS)
    ow = snap[names.index("o_w")]
    for layer in range(gpt.NUM_MLP_SLOTS):
        if layer not in gpt.ATTN_BANK_ORDER:
            assert ow[2, layer].item() == -1.0  # dead column
            continue
        j = gpt.ATTN_BANK_ORDER.index(layer)
        assert ow[0, layer].item() == model._attn_o_w_scales[j].item()
        if config.FP8_ATTN_O_AMAX:
            want = model.vo_bank[2 * j + 1].float().abs().max().item() / (model._attn_o_w_scales[j].item() * 448)
            assert ow[1, layer].item() == pytest.approx(want, rel=1e-6)
    lines = gpt.format_fp8_sat(10, names, snap)
    o_line = next(ln for ln in lines if " o_w " in ln)
    assert "ratio:[" in o_line and " - " in o_line  # layers 4, 6, 7, 9 and slot 11 print "-"
    if CASE == "delayed":  # y / g at their init value with no amax recorded yet: STUCK on every attention layer
        assert "STUCK" in next(ln for ln in lines if " o_y " in ln)


def test_rearm_resets(model):
    if CASE == "off":
        pytest.skip("flag off")
    with torch.no_grad():
        model._attn_o_y_amax.fill_(1.0)
        model._attn_o_y_scales.fill_(0.5)
        model._attn_o_g_scales.fill_(0.5)
    model.rearm_fp8_bootstrap()
    assert (model._attn_o_y_amax == 0).all() and (model._attn_o_g_amax == 0).all()
    assert (model._attn_o_y_scales == S.ATTN_O_Y_SCALE).all() and (model._attn_o_g_scales == S.ATTN_O_G_SCALE).all()
