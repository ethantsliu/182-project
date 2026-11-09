"""CPU tests of the HADAMARD_FP8 fp8 MLP state in model/gpt.py: the functional x_t amax (risk a), the
post_t scale contract (risk b) and the HADAMARD_SAT_STATS snapshot (risk c).

Needs triton importable. Host python3 has none, so these tests skip there. Run them in the triton-interp
image (torch CPU, TRITON_INTERPRET=1):

    docker run --rm -v $PWD:/repo -w /repo triton-interp python -m pytest 182/hadamard-fp8/tests -q

gpt.py is imported with three CUDA-only modules stubbed (FA3 attention and the two CE kernels, which compile
at import). The fp8 state methods run on a small module built by gpu_sanity.make_mini_state. That module
borrows GPT's methods unchanged.
"""
import os
import pathlib
import subprocess
import sys
from unittest import mock

import pytest
import torch

os.environ.setdefault("TRITON_INTERPRET", "1")
pytest.importorskip("triton")

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STUBS = """
import sys
from unittest import mock
import torch
if not torch.cuda.is_available():
    import track_1_short.perf.bf16_loss as _b
    for _n in ("track_1_short.model.attention", "track_1_short.perf.kernels.cross_entropy",
               "track_1_short.perf.kernels.sampled_cross_entropy"):
        sys.modules.setdefault(_n, mock.MagicMock(name=_n))
    _c = sys.modules["track_1_short.perf.kernels.cross_entropy"]
    _c.SOFTCAP_A, _c.SOFTCAP_B, _c.SOFTCAP_C = _b.SOFTCAP_A, _b.SOFTCAP_B, _b.SOFTCAP_C
"""

if "track_1_short.config" in sys.modules and not sys.modules["track_1_short.config"].HADAMARD_FP8:
    pytest.skip("track_1_short.config was imported with HADAMARD_FP8=0 earlier in this session", allow_module_level=True)
os.environ["HADAMARD_FP8"] = "1"
exec(STUBS)

from track_1_short.model import gpt  # noqa: E402
from track_1_short.perf.hadamard_ref import FP8_MAX, rotate, signed_hadamard  # noqa: E402
from track_1_short.perf.kernels.hadamard import hadamard_t_quant  # noqa: E402

import gpu_sanity  # noqa: E402

H, D, SLOT = 256, 128, 5
E4 = torch.float8_e4m3fn


@pytest.fixture
def state(monkeypatch):
    # Until mlp.py implements the post_t contract, the init-time check would refuse HADAMARD_FP8.
    monkeypatch.setattr(gpt, "check_mlp_takes_post_t", lambda: None)
    torch.manual_seed(0)
    return gpu_sanity.make_mini_state(H, D, torch.device("cpu"), slot=SLOT, full=False)


def rms_x(T=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return gpu_sanity.rms_rows(torch.randn(T, D, generator=g))


# ---- risk (a): hadamard_t_quant returns its amax; gpt.py folds it into the buffer ----

def test_t_quant_returns_amax_functionally():
    x, q = rms_x(), signed_hadamard(128, 1234)
    scale = torch.tensor(0.05)
    t, amax = hadamard_t_quant(x, q, scale)
    ref = rotate(x.float(), 0, q.float())
    assert t.shape == (D, 512) and t.dtype == E4 and amax.shape == (1,) and amax.dtype == torch.float32
    assert torch.equal(t.view(torch.uint8), torch.clamp(ref.T / scale, -448, 448).to(E4).view(torch.uint8))
    assert torch.allclose(amax, ref.abs().amax().reshape(1))
    schema = torch.ops.nanogpt.hadamard_t_quant.default._schema
    assert not any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments), schema


def test_t_quant_fake_matches_real():
    from torch._subclasses.fake_tensor import FakeTensorMode
    x, q = rms_x(), signed_hadamard(128, 1234)
    with FakeTensorMode() as mode:
        t, amax = hadamard_t_quant(mode.from_tensor(x), mode.from_tensor(q), mode.from_tensor(torch.tensor(0.05)))
    assert (tuple(t.shape), t.dtype, tuple(amax.shape), amax.dtype) == ((D, 512), E4, (1,), torch.float32)


@pytest.mark.parametrize("compiled", [False, True])
def test_xt_amax_lands_in_target_slot_only(state, compiled):
    """Eager and torch.compile(fullgraph) (aot_eager: dynamo + functionalization + AOTAutograd, the part
    that decides whether a buffer mutation survives; inductor is GPU-tested in test_gpu_state.py)."""
    run = torch.compile(state, fullgraph=True, dynamic=False, backend="aot_eager") if compiled else state
    x1, x2 = rms_x(seed=1), 0.25 * rms_x(seed=2).float()
    run(x1, None)
    a1 = rotate(x1.float(), 0, state._hadamard_q.float()).abs().max()
    amax = state._mlp_xt_amax[:, 0]
    assert torch.allclose(amax[SLOT], a1) and amax[SLOT] > 0
    assert (torch.cat((amax[:SLOT], amax[SLOT + 1:])) == 0).all()
    run(x2.bfloat16(), None)  # smaller input: the buffer keeps the max over forwards
    assert torch.allclose(state._mlp_xt_amax[SLOT, 0], a1)


def test_refresh_moves_xt_and_post_t_scales_then_clears(state):
    state(rms_x(seed=3), None)
    with torch.no_grad():
        state._mlp_post_t_amax[2, 0] = 300.0
    xt_amax = state._mlp_xt_amax[SLOT, 0].item()
    state.mlp_quantize_calls = gpt.FP8_EXACT_SCALE_CALLS  # the captured (delayed-scale) refresh body
    state._quantize_mlp_weights_and_scales()
    xs, ps = state._mlp_xt_scales, state._mlp_post_t_scales
    assert xs[SLOT].item() == pytest.approx(xt_amax * gpt.FP8_XT_HEADROOM / 448, rel=1e-6)
    assert (torch.cat((xs[:SLOT], xs[SLOT + 1:])) == gpt.FP8_MLP_X_SCALE).all()
    assert ps[2].item() == pytest.approx(300.0 * gpt.FP8_POST_HEADROOM / 448, rel=1e-6)
    assert (torch.cat((ps[:2], ps[3:])) == 1.0).all()
    assert (state._mlp_xt_amax == 0).all() and (state._mlp_post_t_amax == 0).all()
    # post's own scale never sees post_t's amax
    assert (state._mlp_post_scales == 1.0).all()


def test_rearm_resets_xt_and_post_t(state):
    with torch.no_grad():
        state._mlp_post_t_amax.fill_(5.0)
        state._mlp_xt_amax.fill_(5.0)
        state._mlp_post_t_scales.fill_(0.3)
        state._mlp_xt_scales.fill_(0.3)
    state.mlp_quantize_calls = 99
    state.rearm_fp8_bootstrap()
    assert state.mlp_quantize_calls == 0
    assert (state._mlp_post_t_amax == 0).all() and (state._mlp_xt_amax == 0).all()
    assert (state._mlp_post_t_scales == 1.0).all() and (state._mlp_xt_scales == gpt.FP8_MLP_X_SCALE).all()


def test_new_buffers_reach_dump_restore_and_refresh_graph_snapshot(state):
    names = dict(state.named_buffers())
    # perf/fp8_dump.py restores every buffer with "amax" in its name; fp8_refresh_graphs.py snapshots and
    # self-checks every "_mlp_" buffer.
    for n in ("_mlp_xt_amax", "_mlp_post_t_amax", "_mlp_post_t_scales", "_mlp_xt_scales"):
        assert n in names and n.startswith("_mlp_")
    assert {"_mlp_xt_amax", "_mlp_post_t_amax"} <= {n for n in names if "amax" in n}


# ---- risk (b): the post_t args contract with mlp.py ----

@pytest.mark.parametrize("with_fold", [False, True])
def test_mlp_fp8_args_append_post_t(state, with_fold):
    fold = torch.tensor(0.9, dtype=torch.bfloat16) if with_fold else None
    x_f8 = (torch.zeros(512, D, dtype=E4), torch.zeros(D, 512, dtype=E4))
    args = state._mlp_fp8_args(3, state._mlp_up_f8[3], x_f8, fold, xt_slot=SLOT)
    assert len(args) == 20
    assert args[16] is state._hadamard_q and args[17] is fold
    for got, buf in ((args[18], state._mlp_post_t_scales), (args[19], state._mlp_post_t_amax),
                     (args[5], state._mlp_post_scales), (args[6], state._mlp_post_amax)):
        want = buf[3]
        assert got.data_ptr() == want.data_ptr() and got.shape == want.shape, "must be a view of slot 3"
    assert args[5].shape == () and args[6].shape == (1,)
    assert args[10].data_ptr() == state._mlp_xt_scales[SLOT].data_ptr()  # x_t scale of the input's slot


def test_contract_check(monkeypatch):
    class Old:
        @staticmethod
        def forward(ctx, x, hadamard_q=None, p_fold=None):
            pass

    class New:
        @staticmethod
        def forward(ctx, x, hadamard_q=None, p_fold=None, post_t_scale=None, post_t_amax=None):
            pass

    monkeypatch.setattr(gpt, "FusedLinearReLUSquareFunction", Old)
    with pytest.raises(AssertionError, match="post_t"):
        gpt.check_mlp_takes_post_t()
    monkeypatch.setattr(gpt, "FusedLinearReLUSquareFunction", New)
    gpt.check_mlp_takes_post_t()


def test_flags_off_args_and_buffers_unchanged():
    """HADAMARD_FP8 is import-time, so the record configuration is checked in a fresh interpreter."""
    code = STUBS + f"""
import sys; sys.path[:0] = [{str(ROOT)!r}, {str(HERE)!r}]
import torch, gpu_sanity
from track_1_short.model import gpt
assert not gpt.HADAMARD_FP8
m = gpu_sanity.make_mini_state({H}, {D}, torch.device("cpu"), full=False)
names = {{n for n, _ in m.named_buffers()}}
assert not any("post_t" in n or "xt" in n or "rot" in n for n in names), names
x_f8 = m._quantize_mlp_input(gpu_sanity.rms_rows(torch.randn(256, {D})), 0)
a = m._mlp_fp8_args(0, m._mlp_up_f8[0], x_f8, None, xt_slot=0)
b = m._mlp_fp8_args(0, m._mlp_up_f8[0], x_f8, torch.tensor(1.0), xt_slot=0)
assert len(a) == 17 and a[16] is None and len(b) == 18
names, snap = m.fp8_sat_snapshot()
assert names == ["post", "dpre", "w1", "w2"], names
print("OK")
"""
    env = dict(os.environ, HADAMARD_FP8="0", TRITON_INTERPRET="1")
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0 and "OK" in p.stdout, p.stdout[-2000:] + p.stderr[-3000:]


# ---- risk (c): HADAMARD_SAT_STATS ----

def test_sat_snapshot_flags_saturated_and_stuck_slots(state):
    with torch.no_grad():
        state._mlp_post_scales.fill_(0.01)
        state._mlp_post_amax[:, 0] = 0.01 * 448 * 0.97      # every slot under its limit ...
        state._mlp_post_amax[2, 0] = 0.01 * 448 * 1.5       # ... except slot 2
        state._mlp_xt_scales.fill_(0.02)
        state._mlp_xt_scales[7] = gpt.FP8_MLP_X_SCALE       # slot 7 never got an amax
    names, snap = state.fp8_sat_snapshot()
    assert names[:2] == ["post", "dpre"] and {"post_t", "x_t", "w1_rot", "w2_rot"} <= set(names)
    assert snap.shape == (len(names), 3, gpt.NUM_MLP_SLOTS)
    lines = dict(zip(names, gpt.format_fp8_sat(100, names, snap)))
    assert "SAT[2]" in lines["post"] and "max_ratio:1.500" in lines["post"] and "STUCK" not in lines["post"]
    assert "STUCK[7]" in lines["x_t"] and "SAT" not in lines["x_t"]
    assert all(line.startswith("fp8 sat step:100 ") for line in lines.values())
