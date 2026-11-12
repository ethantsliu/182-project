"""Tests for the OPTIM_* flags (optim/cascade_flags.py) and the cost tools in this directory.

    python -m pytest 182/optim-cost/test_optim_cost.py -q                                        # host: CPU tests
    bash ../../182/tools/triton_cpu/run.sh python -m pytest \\
        variants/t1-gpu-stack/182/optim-cost/test_optim_cost.py -q                               # + Triton interpreter
    python -m pytest 182/optim-cost/test_optim_cost.py -q                                        # on the node: + GPU

Host (no triton): the flags parse and default off, the map schedules' envelopes, the cost model's shapes against the
source, the simulator's ordering claims, StepProfile's accounting (fake CUDA events).
Interpreter (TRITON_INTERPRET=1, docker) or GPU: polar_flagged at the stock settings equals polar() bit for bit; the
expert bank's fused path (OPTIM_MOEFUSED) equals anvil_cascade bit for bit on the GPU (the real exactness check;
under the interpreter the reference is eager ATen, so there it is a closeness check); polar_flagged with
OPTIM_MAPS=3 / OPTIM_NORM=s4 matches the CPU simulator to bf16 tolerance. Under the interpreter the torch.compile'd
helpers run as their eager bodies (no CPU inductor compile).
"""
import ast
import importlib
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, VARIANT)
sys.path.insert(0, HERE)

import cascade_sim  # noqa: E402
def _load_by_path(name, path):
    # Both optim-cost/ and kernel-cost/ have a cost_model.py; load each by path under its own name so one pytest
    # session collecting both directories does not hand one test the other's module from sys.modules.
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cost_model = _load_by_path("optim_cost_model", os.path.join(HERE, "cost_model.py"))  # noqa: E402
from track_1_short.optim import cascade_flags  # noqa: E402


def _src(*parts):
    with open(os.path.join(VARIANT, *parts)) as f:
        return f.read()


def _comp(maps, s):
    s = np.asarray(s, dtype=np.float64)
    for a, b, c in maps:
        s = a * s + b * s ** 3 + c * s ** 5
    return s


# ----------------------------------------------------------------------------------------------------------- host tests

def test_flags_default_off_and_strict():
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPTIM_")}
    code = ("import sys; sys.path.insert(0, %r); from track_1_short.optim import cascade_flags as c; "
            "print(c.OPTIM_MOEFUSED, c.OPTIM_MAPS, c.OPTIM_NORM_S4, c.OPTIM_PROFILE, c.OPTIM_FLAGGED_UPDATE)" % VARIANT)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "4", "False", "False", "False"]
    for name, value, flagged in (("OPTIM_MAPS", "3", "True"), ("OPTIM_NORM", "s4", "True"), ("OPTIM_MOEFUSED", "1", "True"),
                                 ("OPTIM_PROFILE", "1", "False"), ("OPTIM_MAPS", "4", "False")):
        out = subprocess.run([sys.executable, "-c", code], env={**env, name: value}, capture_output=True, text=True,
                             check=True).stdout.split()
        assert out[-1] == flagged, (name, value, out)
    bad = subprocess.run([sys.executable, "-c", code], env={**env, "OPTIM_MAPS": "2"}, capture_output=True, text=True)
    assert bad.returncode != 0 and "OPTIM_MAPS" in bad.stderr


def test_map_envelopes():
    """The shipped four maps hold max|p - 1| = 0.2388 from 3e-3; MAPS3 holds the same from 1.134e-2 (not from 3e-3), and
    both stay bounded past the top of the normalized range."""
    u = 1 / 1.05
    ship = cascade_sim.ANVIL_MAPS
    m3 = cascade_flags.ANVIL_MAPS3
    s4 = np.logspace(np.log10(3e-3), np.log10(u), 4000)
    s3 = np.logspace(np.log10(1.134e-2), np.log10(u), 4000)
    e4 = np.abs(_comp(ship, s4) - 1).max()
    e3 = np.abs(_comp(m3, s3) - 1).max()
    assert abs(e4 - 0.2388) < 2e-3
    assert e3 <= e4 + 2e-3
    assert np.abs(_comp(m3, s4) - 1).max() > 0.5            # three maps do NOT cover 3e-3
    top = np.linspace(u, 1.02, 400)
    assert np.abs(_comp(m3, top)).max() < 5 and np.abs(_comp(ship, top)).max() < 5
    # the small-sigma gain of the whole cascade: 276 (four maps) vs ~73 (three)
    assert abs(math.prod(a for a, _, _ in ship) - 276.09) < 0.1
    assert 60 < math.prod(a for a, _, _ in m3) < 90


def test_cost_model_matches_source():
    gpt = _src("track_1_short", "model", "gpt.py")
    moe = _src("track_1_short", "model", "moe.py")
    tg = _src("train_gpt.py")
    assert re.search(r"num_heads=6,", tg) and re.search(r"head_dim=128,", tg) and re.search(r"model_dim=768,", tg)
    assert "MLP_HIDDEN_DIM = 2816" in gpt and "MOE_HIDDEN = 1024" in moe and "MOE_EXPERTS = 8" in moe
    assert "MOE_LAYERS = (1, 3, 5, 6, 8)" in moe
    banks = {b.name: b for b in cost_model.BANKS}
    assert (banks["qk_bank"].rows, banks["qk_bank"].cols, banks["qk_bank"].matrices) == (2 * 128, 768, 48)
    assert (banks["vo_bank"].rows, banks["vo_bank"].matrices) == (768, 16)
    assert (banks["mlp_bank"].rows, banks["mlp_bank"].matrices) == (2816, 16)
    assert (banks["moe_bank"].rows, banks["moe_bank"].matrices) == (1024, 5 * 2 * 8)
    # the fused banks' tiles are anvil_opt._GRAM_CFG's
    tree = ast.parse(_src("track_1_short", "perf", "kernels", "anvil_opt.py"))
    cfg = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_GRAM_CFG")
    assert banks["qk_bank"].gram_tile == cfg[("XXT", 256)][:2] and banks["qk_bank"].poly_tile == cfg[("baAA", 256)][:2]
    assert banks["vo_bank"].gram_tile == cfg[("XXT", 768)][:2] and banks["mlp_bank"].gram_tile == cfg[("XTX", 768)][:2]
    rows = {r["bank"]: r for r in (cost_model.bank_cost(b) for b in cost_model.BANKS)}
    assert [rows[k]["chunk"] for k in ("qk_bank", "vo_bank", "mlp_bank", "moe_bank")] == [6, 2, 2, 10]
    # moe_bank carries more cascade work per rank than the other three together
    assert rows["moe_bank"]["gflop_cascade_eff"] > sum(rows[k]["gflop_cascade_eff"] for k in ("qk_bank", "vo_bank", "mlp_bank"))
    assert cost_model.live_tiles(768, 64, 128) == 42 and cost_model.live_tiles(256, 64, 64) == 10


def test_simulator_orders_schedules():
    """On a heavy-tailed spectrum three maps leave more singular values short of 1 than four, and the Schatten-4 scale
    keeps sigma_max <= 1/margin (it bounds the spectral norm) while lifting the small ones."""
    gen = torch.Generator().manual_seed(0)
    X, P = cascade_sim.make_matrix(1024, 768, "pow1", gen)
    f4 = cascade_sim.score(cascade_sim.cascade(X.float(), cascade_sim.ANVIL_MAPS), P)
    f3 = cascade_sim.score(cascade_sim.cascade(X.float(), cascade_flags.ANVIL_MAPS3), P)
    s44 = cascade_sim.score(cascade_sim.cascade(X.float(), cascade_sim.ANVIL_MAPS, s4=True), P)
    assert f3["below"] > f4["below"] + 0.2 and f3["rel"] > f4["rel"]
    assert s44["below"] < f4["below"]
    sv = torch.linalg.svdvals(X)
    s4n = sv.pow(4).sum().pow(0.25)
    assert sv.max() / (1.05 * s4n) <= 1 / 1.05 + 1e-9 and torch.linalg.norm(X) / s4n > 1.2


def test_step_profile_accounting(monkeypatch):
    """StepProfile with fake events: wait = ready - issue, update = done - ready, tail = end - begin; a record is only
    read once its last event completed, and nothing is recorded during a capture."""
    from track_1_short.optim import step_profile

    clock = {"t": 0.0, "done_until": 1e9}

    class FakeEvent:
        def __init__(self, enable_timing=False):
            self.t = None

        def record(self):
            clock["t"] += 1.0
            self.t = clock["t"]

        def query(self):
            return self.t <= clock["done_until"]

        def elapsed_time(self, other):
            return other.t - self.t

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    capturing = {"on": False}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing["on"])
    monkeypatch.setattr(step_profile, "REPORT_EVERY", 2)
    lines = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: lines.append(" ".join(map(str, a))))
    prof = step_profile.StepProfile(rank=3)
    for _ in range(2):
        rec = prof.begin()                 # t=1 begin
        rec.mark("qk_bank.issue")          # 2
        rec.mark("qk_bank.ready")          # 3
        clock["t"] += 4.0                  # the update takes 5 ms of fake time
        rec.mark("qk_bank.done")           # 8
        rec.end()                          # 9
        clock["t"] = 0.0
    clock["done_until"] = 0.0              # the second record is not finished yet
    capturing["on"] = True
    assert prof.begin() is None            # capture: no record, no drain
    capturing["on"] = False
    clock["done_until"] = 1e9
    prof._drain()
    assert prof.steps == 2 and len(lines) == 1
    assert "rank=3" in lines[0] and "qk_bank w=1.000[1.000] u=5.000[5.000]" in lines[0] and "tail=8.000" in lines[0]


# ------------------------------------------------------------------------------------- interpreter (docker) / GPU tests

def _device():
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        try:
            import triton  # noqa: F401
        except ImportError:
            return None
        return "cpu"
    return None


DEVICE = _device()
needs_kernels = pytest.mark.skipif(DEVICE is None, reason="needs CUDA, or TRITON_INTERPRET=1 with triton (182/tools/triton_cpu)")


def _eager(fn):
    return getattr(fn, "_torchdynamo_orig_callable", fn)


@pytest.fixture
def ok(monkeypatch):
    """anvil_opt; on CPU its torch.compile'd helpers are swapped for their eager bodies."""
    mod = importlib.import_module("track_1_short.perf.kernels.anvil_opt")
    if DEVICE == "cpu":
        for name in ("gram_norm", "gram_norm_dup", "gram_norm_flagged"):
            monkeypatch.setattr(mod, name, _eager(getattr(mod, name)))
    return mod


def _x(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * 0.02).bfloat16().to(DEVICE)


@needs_kernels
@pytest.mark.parametrize("shape", [(2, 256, 768), (1, 768, 768), (2, 1024, 768)])
def test_polar_flagged_default_is_polar(ok, shape):
    maps = cascade_sim.ANVIL_MAPS
    x = _x(shape, 1)
    ref = ok.polar(x.clone(), False, maps)
    got = ok.polar_flagged(x.clone(), False, maps, 1.05, False)
    assert torch.equal(ref.view(torch.int16), got.view(torch.int16))


@needs_kernels
@pytest.mark.parametrize("maps_name,s4", [("ANVIL_MAPS3", False), ("ANVIL_MAPS", True), ("ANVIL_MAPS3", True)])
@pytest.mark.parametrize("shape,margin", [((2, 256, 768), 1.05), ((1, 1024, 768), 1.1)])
def test_polar_flagged_matches_simulator(ok, maps_name, s4, shape, margin):
    maps = cascade_flags.ANVIL_MAPS3 if maps_name == "ANVIL_MAPS3" else cascade_sim.ANVIL_MAPS
    x = _x(shape, 2)
    got = ok.polar_flagged(x.clone(), False, maps, margin, s4).float().cpu()
    ref = cascade_sim.cascade(x.float().cpu(), maps, margin, s4).float()
    rel = (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()
    assert rel < 3e-2, rel  # bf16 rounding of each round's operands, compounded over the maps


@needs_kernels
def test_moe_fused_equals_anvil_cascade(monkeypatch):
    """OPTIM_MOEFUSED's expert-bank path (pre_fused + polar_flagged at margin 1.1) against anvil_cascade, from the same
    gradient and rails: the output and the updated rails, bit for bit on the GPU; rails within 2 ulp of their scale and output within 1%
    under the interpreter (see below)."""
    ok_mod = importlib.import_module("track_1_short.perf.kernels.anvil_opt")
    anvil = importlib.import_module("track_1_short.optim.anvil")
    cascade = anvil.anvil_cascade
    if DEVICE == "cpu":
        monkeypatch.setattr(ok_mod, "gram_norm_flagged", _eager(ok_mod.gram_norm_flagged))
        cascade = _eager(cascade)
    shape = (2, 1024, 768)
    g = torch.Generator().manual_seed(3)
    grad = (torch.randn(shape, generator=g) * 1e-3).bfloat16().to(DEVICE)
    vel = (torch.randn((2, *shape), generator=g) * 1e-3).to(DEVICE)
    scal = lambda v: torch.tensor(v, dtype=torch.float32, device=DEVICE)  # noqa: E731
    mom, fb, fw = scal(0.93), scal(0.85), scal(0.4385)
    v_ref = vel.clone()
    ref = cascade(grad.clone(), v_ref, mom, False, fb, fw, margin=anvil.MOE_ANVIL_MARGIN)
    v_got = vel.clone()
    x = torch.empty_like(grad)
    ok_mod.pre_fused(grad, v_got, x, mom, fb, fw)
    got = ok_mod.polar_flagged(x, False, anvil.ANVIL_MAPS, anvil.MOE_ANVIL_MARGIN, False)
    if DEVICE == "cuda":  # the reference is the compiled anvil_cascade, as in the run
        assert torch.equal(v_ref.view(torch.int32), v_got.view(torch.int32)), "rails differ"
        assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), "cascade output differs"
    else:
        # Interpreter: anvil_cascade runs as eager ATen, whose lerp rounds differently from the inductor lowering that
        # pre_fused reproduces, so only closeness is meaningful here. Rails: ~10% of elements differ by a rounding;
        # a per-element ulp count is meaningless where the lerp cancels to ~1e-9 (observed 51080 ulp there), so the
        # bound is in ulps of the rail's largest value (observed max |diff| = 0.39 of them).
        err = (v_ref - v_got).abs().max().item()
        assert err <= 2 * torch.finfo(torch.float32).eps * v_ref.abs().max().item(), err
        rel = (torch.linalg.norm(ref.float() - got.float()) / torch.linalg.norm(ref.float())).item()
        assert rel < 1e-2, rel
