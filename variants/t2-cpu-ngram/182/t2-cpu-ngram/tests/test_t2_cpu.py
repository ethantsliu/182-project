"""CPU checks for the Track 2 port (no GPU): model shapes, schedules, env defaults, run helpers.

  cd variants/t2-cpu-ngram && python -m pytest -q 182/t2-cpu-ngram/tests

The reference numbers come from @ldmberman's Track 2 run log (PR #380 thread, 2026-10-04) and from PR #380's
Track 1 code (git fetch upstream pull/380/head; built the same way, it gives the same numbers).
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
VARIANT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import runlib  # noqa: E402

KNOBS = ("MODEL_DIM", "MODEL_LAYERS", "MLP_HIDDEN_DIM", "LR_COOLDOWN_FRAC", "LR_FLOOR", "VAL_BATCH_SIZE",
         "NUM_SCHEDULED_ITERATIONS", "TARGET_LOSS")
TRACK2 = dict(MODEL_DIM="1024", MODEL_LAYERS="16", MLP_HIDDEN_DIM="4096", LR_COOLDOWN_FRAC="0.70", LR_FLOOR="0.10",
              VAL_BATCH_SIZE=str(2 * 64 * 1024 * 8), NUM_SCHEDULED_ITERATIONS="5790", TARGET_LOSS="2.92")


def probe(**env):
    base = {k: v for k, v in os.environ.items() if k not in KNOBS}
    out = subprocess.run([sys.executable, str(HERE / "build_probe.py")], env={**base, **env}, cwd=VARIANT,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_wrapper_defaults_match_ldmberman_settings():
    src = (VARIANT / "train_gpt_medium_cpu.py").read_text()
    ns = {}
    exec(src.split("for key, value in TRACK2_DEFAULTS.items():")[0].split("import sys", 1)[1], ns)
    assert ns["TRACK2_DEFAULTS"] == TRACK2


def test_track2_matches_ldmberman_log():
    r = probe(**TRACK2)
    # model_configuration line of the log: parameters=528536207
    assert r["parameters"] == 528536207
    # "head widths by layer, in bank order={0: .., 2: .., 5: .., 10: .., 11: .., 12: .., 13: .., 14: .., 1: .., 8: .., 3: .., 15: ..}"
    assert r["attn_bank_order"] == [0, 2, 5, 10, 11, 12, 13, 14, 1, 8, 3, 15]
    assert r["long_window"] == [3, 15] and r["value_embed_layers"] == [1, 2, 8, 15]
    assert r["num_mlp_slots"] == 20 and r["parallel_mlp_slot"] == 16
    assert r["frozen"] == [14, 15, 34, 35, 36, 37, 38, 39]  # dead slot 7 and padding slots 17-19
    assert r["mlp_bank"] == [20, 2, 4096, 1024] and r["ngram_dim"] == 1024
    assert r["boundaries"] == [[0, 1653], [1653, 3513], [3513, 5443], [5443, 5790], [5790, 5800]]
    # "sampled softmax: P by step change [(0, 10240), (3513, 14336), (4800, 24576), (5443, 0)]"
    assert r["switches"] == [[0, 10240], [3513, 14336], [4800, 24576], [5443, 0]]
    assert r["lr_end"] == pytest.approx(0.1) and r["val_batch"] == 1048576 and r["cooldown"] == 0.7


def test_defaults_reproduce_pr380_track1():
    r = probe()
    assert r["parameters"] == 185626563  # PR #380's GPT, built the same way
    assert r["attn_bank_order"] == [0, 2, 5, 1, 8, 3, 10]
    assert r["long_window"] == [3, 10] and r["value_embed_layers"] == [1, 2, 8, 10]
    assert r["num_mlp_slots"] == 12 and r["parallel_mlp_slot"] == 11 and r["frozen"] == [14, 15]
    assert r["mlp_bank"] == [12, 2, 1792, 512]
    assert r["boundaries"][-1] == [715, 725] and r["val_batch"] == 4 * 64 * 1024 * 8
    assert r["lr_floor"] == 0.30 and r["cooldown"] == 0.65


@pytest.mark.parametrize("steps", [3000, 4000, 4600, 5200])
def test_sweep_step_counts_build(steps):
    r = probe(**{**TRACK2, "NUM_SCHEDULED_ITERATIONS": str(steps)})
    b = r["boundaries"]
    assert b[0][0] == 0 and b[-2][1] == steps and b[-1] == [steps, steps + 10]
    assert all(x[1] == y[0] for x, y in zip(b, b[1:]))
    s = r["switches"]
    assert [p for _, p in s] == [10240, 14336, 24576, 0] and s[-1][0] == b[3][0]  # full softmax from the taper


# ---- run helpers ---------------------------------------------------------------------------------------

LDM_TAIL = """step:5800/5800 val_loss:2.8769 train_time:221003ms step_avg:38.10ms
endpoint_result:{"variant": "sampled", "steps": 5800, "seed": 1337, "val_loss": 2.876861095428467, "train_time_ms": 221002.89688599968}
"""
OUR_TAIL = """logs/0f8e2a6c-1111-2222-3333-444455556666.txt
step:4610/4610 val_loss:2.9171 train_time:176000ms step_avg:38.18ms
endpoint_result: steps=4610 seed=None val_loss=2.917123 train_time_ms=176000 target_loss=2.92 target_reached=True
"""


def test_parse_result_both_formats():
    a = runlib.parse_result(LDM_TAIL)
    assert a["steps"] == 5800 and a["val_loss"] == pytest.approx(2.876861) and a["seed"] == 1337
    b = runlib.parse_result(OUR_TAIL)
    assert b == dict(steps=4610, val_loss=2.917123, train_time_ms=176000.0, seed=None,
                     log="logs/0f8e2a6c-1111-2222-3333-444455556666.txt")
    assert runlib.parse_result("step:100/4610 train_time:1ms") is None


def test_t_test_matches_scipy():
    stats = pytest.importorskip("scipy.stats", reason="scipy not installed (optional cross-check)")
    rng = random.Random(0)
    for n in (2, 3, 6, 12):
        xs = [2.918 + rng.gauss(0, 0.0015) for _ in range(n)]
        assert runlib.one_sided_p(xs)[1] == pytest.approx(stats.ttest_1samp(xs, 2.92, alternative="less").pvalue,
                                                          rel=1e-9)
    assert runlib.required_mean(0.0014) == pytest.approx(2.92 + stats.t.ppf(0.01, 5) * 0.0014 / 6 ** 0.5, abs=1e-12)


def _shards(root: Path, n_train: int, val=True):
    d = root / "data" / "fineweb10B"
    d.mkdir(parents=True)
    for i in range(1, n_train + 1):
        (d / f"fineweb_train_{i:06d}.bin").write_bytes(b"")
    if val:
        (d / "fineweb_val_000000.bin").write_bytes(b"")


def test_data_path_defaults_to_repo_root(tmp_path):
    repo = tmp_path / "variants" / "t2"
    repo.mkdir(parents=True)
    _shards(tmp_path, 103)
    assert runlib.resolve_data_path({}, repo) == "../.."  # the README layout: shards at the repo root
    assert runlib.resolve_data_path({"DATA_PATH": "/x"}, repo) == "/x"
    assert runlib.check_shards("../..", repo) == (103, None)


def test_shard_preflight(tmp_path):
    _shards(tmp_path, 12)
    n, problem = runlib.check_shards(".", tmp_path)
    assert n == 12 and "not 103" in problem
    assert runlib.check_shards(".", tmp_path / "nowhere")[1].startswith("no FineWeb")
    empty = tmp_path / "noval"
    _shards(empty, 103, val=False)
    assert runlib.check_shards(".", empty)[1].startswith("no FineWeb")


def test_launch_refuses_partial_download(tmp_path, monkeypatch):
    _shards(tmp_path, 12)
    monkeypatch.setattr(runlib, "REPO", tmp_path)
    monkeypatch.setattr(runlib, "resolve_data_path", lambda env, repo=tmp_path: ".")
    monkeypatch.setattr(runlib, "check_shards", lambda dp, repo=tmp_path: (12, "12 train shards, not 103"))
    monkeypatch.delenv("ALLOW_PARTIAL_SHARDS", raising=False)
    with pytest.raises(SystemExit):
        runlib.launch(4000, 0, tmp_path / "out.txt")
