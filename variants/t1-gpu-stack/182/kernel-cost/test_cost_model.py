"""CPU tests of cost_model.py: the schedule it walks is the trainer's, the shapes give the hand-computed FLOPs, and the
printed report holds together. `python -m pytest variants/t1-gpu-stack/182/kernel-cost/test_cost_model.py -q`."""
import contextlib
import io
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

def _load_by_path(name, path):
    # Both optim-cost/ and kernel-cost/ have a cost_model.py; load each by path under its own name so one pytest
    # session collecting both directories does not hand one test the other's module from sys.modules.
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load_by_path("kernel_cost_model", os.path.join(HERE, "cost_model.py"))  # noqa: E402


def test_stage_ends_match_training_schedule():
    torch = pytest.importorskip("torch")
    from track_1_short.config import (LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT,
                                      Hyperparameters)
    from track_1_short.schedule import TrainingSchedule
    args = Hyperparameters()
    s = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations,
                         torch.device("cpu"), LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, WS_POST_YARN_EXT)
    assert cm.stage_ends() == s.boundaries
    assert s.total_steps == 1053


def test_candidate_counts():
    # sampled_softmax.py's docstring: P = 10240 on steps 0-594, 14336 on 595-847, 24576 on 848-973, full from 974
    for step, P in ((0, 10240), (594, 10240), (595, 14336), (847, 14336), (848, 24576), (973, 24576),
                    (974, cm.VOCAB), (1052, cm.VOCAB)):
        assert cm.regime_at(step).vocab == P, step
    try:
        from track_1_short.sampled_softmax import CANDIDATES_BY_STAGE
    except Exception:   # needs torch/numpy; the literal above is the check that always runs
        return
    assert CANDIDATES_BY_STAGE == cm.CANDIDATES_BY_STAGE


def test_regime_step_counts():
    regs = cm.regimes()
    assert sum(n for _, n in regs.values()) == 1053
    assert {k: n for k, (_, n) in regs.items()} == {
        "stage0": 280, "stage1": 315, "stage2_p14336": 253, "stage2_p24576": 126, "taper": 59, "extension": 20}
    assert regs["stage2_p14336"][0].tokens == 49152 and regs["taper"][0].tokens == 40960


def test_mean_keys_limits():
    assert cm.mean_keys(1, 3072) == pytest.approx(1.0)
    big = cm.mean_keys(10 ** 6, 896)
    small = cm.mean_keys(128, 896)
    assert 1 < small < 128 < big
    assert cm.mean_keys(640, 3072, paired=True) >= cm.mean_keys(640, 3072)


def test_hand_computed_flops():
    r = cm.regimes()["stage2_p14336"][0]
    ops = {}
    for o in cm.step_ops(r):
        ops.setdefault(o.name, []).append(o)
    up = ops["dense MLP x5: up fwd relu^2 epilogue (G3)"]
    assert len(up) == 5 and up[0].flops == 2 * 49152 * 768 * 2816
    assert ops["lm_head logits GEMM (fp8)"][0].flops == 2 * 49152 * 768 * 14336
    qkv = {o.flops for o in ops["attn x7: QKV GEMM fwd (fp8)"]}
    assert qkv == {2 * 49152 * 768 * n for n in (6 * 256, 6 * 192, 6 * 384)}
    assert len(ops["MoE x5: router GEMM N=8 (cuBLAS bf16)"]) == 5
    # the sampled dW only from 32768 rows (sampled_head.MIN_ROWS): stage 0 runs it dense
    names0 = {o.name for o in cm.step_ops(cm.regimes()["stage0"][0])}
    assert "lm_head dW GEMM (fp8)" in names0 and "lm_head dW GEMM (fp8, sampled 1/4 rows)" not in names0


def test_router_fused_is_cheaper_and_monotone():
    r = cm.regimes()["stage2_p14336"][0]
    assert cm.step_seconds(r, router_fused=True) < cm.step_seconds(r)
    assert cm._router_delta_at(0.25) < cm._router_delta_at(0.4) < cm._router_delta_at(0.8) < 0


def test_report_prints(tmp_path):
    out = io.StringIO()
    csv = tmp_path / "ops.csv"
    with contextlib.redirect_stdout(out):
        assert cm.main(["--top", "10", "--csv", str(csv)]) == 0
    txt = out.getvalue()
    assert "# calibration against #397" in txt and "MOE_ROUTER_FUSED modelled delta" in txt
    for name in cm.MEASURED_MS:
        line = next(ln for ln in txt.splitlines() if ln.startswith(name + " "))
        ratio = float(line.split()[-1])
        assert 0.7 < ratio < 1.3, line          # a gross-breakage guard, not a fit
    assert csv.read_text().count("\n") > 30
