"""CPU test of gpu_bench.py's projection and report on synthetic timings (the GPU parts need the node).

    python -m pytest 182/sparse24/tests/test_bench_projection.py -q
"""
import importlib.util
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sp24_gpu_bench", HERE.parent / "gpu_bench.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def synthetic(sparse_ms, fp8_ms, refresh_sparse=0.5, refresh_fp8=0.2, gemm_sparse=1.0, gemm_fp8=1.0):
    Ts = sorted({T for T, _ in B.STAGES})
    calls = {"T": {str(T): {"fp8_record": fp8_ms * T / 16384, "bf16_dense": 2 * fp8_ms * T / 16384,
                            "sparse24_bf16": sparse_ms * T / 16384} for T in Ts},
             "refresh": {"fp8": refresh_fp8, "sparse_greedy": refresh_sparse, "sparse_optimal": 2 * refresh_sparse}}
    gemms = {str(T): {"k768": {"dense_bf16": 2.0, "dense_fp8": gemm_fp8, "sparse_bf16": gemm_sparse},
                      "k2816": {"dense_bf16": 2.0, "dense_fp8": gemm_fp8, "sparse_bf16": gemm_sparse,
                                "sparse_fp8": "unsupported: RuntimeError: x"},
                      "dW": {"dense_bf16": 2.0, "dense_fp8": gemm_fp8}} for T in Ts}
    return calls, gemms


def test_schedule_matches_the_record():
    assert B.TOTAL_STEPS == 1194
    assert sum(T * n for T, n in B.STAGES) == 41_082_880  # per-rank token-steps, as the fp8-attn-o README uses


def test_projection_slower_arm_has_no_budget():
    calls, gemms = synthetic(sparse_ms=1.2, fp8_ms=1.0)
    p = B.project(calls, gemms)
    tok = sum(T * n for T, n in B.STAGES) / 16384
    want = (tok * 11 * 0.2 + 1194 * 0.3) / 1000
    assert p["calls_delta_s"] == pytest.approx(want)
    assert p["calls_budget_mnat"]["centre"] == 0.0
    # GEMMs: 4 sparse at 1.0 + 2 bf16 dW at 2.0 vs 6 fp8 at 1.0 -> +2 ms per call
    assert p["gemm_only_delta_s"] == pytest.approx(1194 * 11 * 2.0 / 1000)
    text = B.report({"gemms": gemms, "calls": calls, "projection": p})
    assert "cannot pay" in text and "unsupported: RuntimeError" in text


def test_projection_faster_arm_buys_loss():
    calls, gemms = synthetic(sparse_ms=0.8, fp8_ms=1.0, refresh_sparse=0.2, gemm_sparse=0.5)
    p = B.project(calls, gemms)
    d = p["calls_delta_s"]
    assert d < 0
    assert p["calls_budget_mnat"]["centre"] == pytest.approx(-d * 1000 / 164)
    lo, hi = p["calls_budget_mnat"]["band"]
    assert lo > p["calls_budget_mnat"]["centre"] > hi  # 120 ms/mnat buys more mnat than 250


def test_unsupported_arm_gives_na():
    calls, gemms = synthetic(sparse_ms=0.8, fp8_ms=1.0)
    for row in calls["T"].values():
        row["sparse24_bf16"] = "unsupported: RuntimeError: no cuSPARSELt"
    p = B.project(calls, gemms)
    assert p["calls_delta_s"] is None
    assert "n/a" in B.report({"gemms": gemms, "calls": calls, "projection": p})
