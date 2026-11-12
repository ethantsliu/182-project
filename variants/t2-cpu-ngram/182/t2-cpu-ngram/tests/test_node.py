"""CPU tests of the node runner (182/run_node.sh -> 182/t2-cpu-ngram/node.py): the log parser on a real upstream
log and on failed ones, the RESULT line, the dry run, the preflight's output format, and the host fallbacks.

fixtures/pr380_record_002bc57f_tail.txt is the part after the source dump of a PR #380 record log
(records/track_1_short/2026-10-03_ExactCountChain/this_pr/002bc57f-....txt at upstream pull/380/head): the same
trainer and line formats as this tree, at the Track 1 size.
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
VARIANT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import node  # noqa: E402
import runlib  # noqa: E402
import sweep_steps  # noqa: E402

RUN_NODE = VARIANT / "182" / "run_node.sh"
REAL = (HERE / "fixtures" / "pr380_record_002bc57f_tail.txt").read_text()
OURS = """logs/0f8e2a6c-1111-2222-3333-444455556666.txt
validation index built in 2.10 s
validation index built in 2.40 s
validation index built in 11.70 s
step:5800/5800 val_loss:2.8770 train_time:221500ms step_avg:38.19ms
endpoint_result: steps=5800 seed=3 val_loss=2.877012 train_time_ms=221500 target_loss=2.92 target_reached=True
val_loss_parts: model_only=2.899100 with_chain=2.877012 chain_gain=0.022088
peak memory allocated: 61242 MiB reserved: 70578 MiB
host memory: rank 0 peak RSS 31.5 GiB, node in use 402 GiB (incl. /dev/shm 0 GiB)
"""
RESULT_OK = re.compile(r"^RESULT variant=t2-cpu-ngram val_loss=\d+\.\d+ train_time_s=\d+\.\d+ steps=\d+ target=2\.92 hit=[01] torch=\S+ log=\S+$")
RESULT_FAILED = re.compile(r"^RESULT variant=t2-cpu-ngram status=failed torch=\S+ reason=\S.*$")
CLEAN = ("NPROC", "SEED", "STEPS", "EXTRA_ENV", "LOG_DIR", "DRY_RUN", "DATA_PATH", "AB_ENV", "AB_RUNS", "AB_SEEDS", "PY",
         "NUM_SCHEDULED_ITERATIONS", "TRAIN_SEED", "SKIP_CHECK", "VAL_BUILD_THREADS", "HOST_CHECK", "ALLOW_STACK")


def sh(mode, **env):
    base = {k: v for k, v in os.environ.items() if k not in CLEAN}
    return subprocess.run(["bash", str(RUN_NODE), *([mode] if mode else [])], env={**base, "PY": sys.executable, **env},
                          capture_output=True, text=True, timeout=300, cwd="/")


# ---- parser ---------------------------------------------------------------------------------------------

def test_parser_reads_a_real_upstream_log():
    r = runlib.parse_result(REAL)
    assert r == dict(steps=725, val_loss=3.2783, train_time_ms=9667.0)
    assert runlib.parse_details(REAL) == dict(peak_alloc_mib=48643, peak_reserved_mib=54618)
    line = node.result_line(r, "logs/x.txt")
    assert RESULT_OK.match(line) and " hit=0 " in line and "val_loss=3.278300 train_time_s=9.667 steps=725" in line


def test_parser_reads_this_trees_lines():
    r = runlib.parse_result(OURS)
    assert r["val_loss"] == pytest.approx(2.877012) and r["seed"] == 3 and r["log"].startswith("logs/0f8e")
    d = runlib.parse_details(OURS)
    assert d["model_only"] == pytest.approx(2.8991) and d["index_build_s"] == [2.1, 2.4, 11.7]
    assert d["rank0_rss_gib"] == 31.5 and d["host_used_gib"] == 402
    assert " hit=1 " in node.result_line(r, "x")
    text = "\n".join(node.summarize(r, {**d, "gpu_total_mib": 81559}))
    assert "43.0 millinats under" in text and "model alone 2.8991" in text and "10.7 GiB headroom" in text


def test_truncated_and_failed_logs_have_no_result():
    cut = REAL[:REAL.index("step:350/725")] + "step:350/725 train_time:4347ms step_avg:12.42ms\n"
    assert runlib.parse_result(cut) is None
    assert runlib.failure_reason(cut, -9) == "exit -9, no error line (last step line: 350/725)"
    oom = cut + "Traceback (most recent call last):\n  File \"train_gpt.py\", line 1\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
    assert "CUDA out of memory" in runlib.failure_reason(oom, 1)
    host_fail = "HOST CHECK: /dev/shm has 32 GiB free, the index tables need 144 GiB\n"
    assert "/dev/shm has 32 GiB" in runlib.failure_reason(host_fail, 1)
    nan = "step:5800/5800 val_loss:nan train_time:1ms step_avg:1ms\nendpoint_result: steps=5800 seed=None val_loss=nan train_time_ms=1\n"
    assert runlib.parse_result(nan) is None
    line = node.result_line(None, None, "exit 1: two\nlines")
    assert RESULT_FAILED.match(line) and "\n" not in line


# ---- run_node.sh ----------------------------------------------------------------------------------------

def test_usage_and_unknown_mode():
    for mode in ("", "bogus"):
        r = sh(mode)
        assert r.returncode == 2 and "check" in r.stderr and "RESULT" not in r.stdout


def test_dry_run_prints_the_command_and_runs_nothing(tmp_path):
    r = sh("run", DRY_RUN="1", LOG_DIR=str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    line = r.stdout.strip().splitlines()[-1]
    assert f"cd {VARIANT} && DATA_PATH=../.." in line
    assert "-m torch.distributed.run --standalone --nproc_per_node=8 train_gpt_medium_cpu.py" in line
    assert "TRAIN_SEED" not in line and "NUM_SCHEDULED_ITERATIONS" not in line  # the default run sets neither
    assert not (tmp_path / "out").exists()
    r = sh("run", DRY_RUN="1", NPROC="4", SEED="7", STEPS="5200", EXTRA_ENV="COORD_DESCENT_TUNING=1 VAL_BUILD_THREADS=16", DATA_PATH="/d")
    line = r.stdout.strip().splitlines()[-1]
    for part in ("--nproc_per_node=4", "TRAIN_SEED=7", "NUM_SCHEDULED_ITERATIONS=5200", "COORD_DESCENT_TUNING=1", "VAL_BUILD_THREADS=16", "DATA_PATH=/d"):
        assert part in line


def test_dry_run_of_ab_and_prep():
    r = sh("ab", DRY_RUN="1", AB_RUNS="2")
    lines = [l for l in r.stdout.splitlines() if l.startswith("+ ")]
    arms = ["NUM_SCHEDULED_ITERATIONS=5200" in l for l in lines]
    assert r.returncode == 0 and arms == [False, True, True, False]  # interleaved: base alt alt base
    assert ["TRAIN_SEED=0" in l for l in lines] == [True, True, False, False] and "TRAIN_SEED=1" in lines[2]
    r = sh("prep", DRY_RUN="1")
    assert r.returncode == 0 and "cached_fineweb10B.py 103" in r.stdout and "prep: dry run" in r.stdout


def test_bad_python_fails_with_the_contract_line():
    r = sh("run", PY="/nonexistent/python")
    assert r.returncode == 1 and RESULT_FAILED.match(r.stdout.strip().splitlines()[-1])
    r = sh("check", PY="/nonexistent/python")
    assert r.returncode == 1 and r.stdout.startswith("FAIL python:")


def test_check_prints_only_contract_lines():
    r = sh("check")
    lines = r.stdout.strip().splitlines()
    assert r.returncode in (0, 1) and not r.stderr.strip(), r.stderr
    assert all(re.match(r"^(PASS|WARN|FAIL) [^:]+: \S", l) for l in lines[:-1]), lines
    fails = sum(l.startswith("FAIL") for l in lines)
    assert lines[-1].startswith("check: ") and f"{fails} fail" in lines[-1] and r.returncode == (1 if fails else 0)
    assert {"torch", "gpus", "data", "host memory", "exact_match extension"} <= {l.split(" ", 1)[1].split(":")[0] for l in lines[:-1]}


def test_check_without_a_gpu_fails_cleanly(monkeypatch):
    monkeypatch.setattr(node, "gpu_query", lambda: None)
    out = []
    assert node.run_check({"NPROC": "8"}, out.append) >= 1
    assert any(l.startswith("FAIL gpus: nvidia-smi found no GPU") for l in out)
    monkeypatch.setattr(node.host, "detect", lambda gpus=None: 1 / 0)  # a check that raises is one FAIL line
    out = []
    node.run_check({}, out.append)
    assert any(l.startswith("FAIL host: the check itself failed: ZeroDivisionError") for l in out)


# ---- one run, end to end with a stand-in trainer --------------------------------------------------------

def fake_run(monkeypatch, tmp_path, script):
    monkeypatch.setattr(node, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(node, "build_run", lambda overrides=None, environ=None: ([sys.executable, "-c", script], dict(os.environ), dict(overrides or {})))
    monkeypatch.setattr(node, "gpu_query", lambda: None)
    return node.run_once({}, tmp_path / "log")


def test_run_once_success_failure_timeout(monkeypatch, tmp_path, capsys):
    row, line = fake_run(monkeypatch, tmp_path, f"print({OURS!r})")
    assert RESULT_OK.match(line) and " hit=1 " in line and row["scheduled_iterations"] == 5790
    assert (tmp_path / "log" / "summary.txt").exists() and len(runlib.load_jsonl(tmp_path / "results" / "sweep.jsonl")) == 1
    row, line = fake_run(monkeypatch, tmp_path, "import sys; print('step:25/5800 train_time:1ms step_avg:1ms'); print('RuntimeError: boom'); sys.exit(3)")
    assert row is None and RESULT_FAILED.match(line) and "exit 3: RuntimeError: boom" in line
    row, line = fake_run(monkeypatch, tmp_path, f"import sys; print({OURS!r}); sys.exit(1)")  # a result but a failed exit
    assert row is None and RESULT_FAILED.match(line)
    monkeypatch.setenv("RUN_TIMEOUT_S", "1")
    row, line = fake_run(monkeypatch, tmp_path, "import time; print('started', flush=True); time.sleep(60)")
    assert row is None and "timeout after 1s" in line
    assert len(runlib.load_jsonl(tmp_path / "results" / "sweep.jsonl")) == 1  # failures are not recorded as results


def test_failed_preflight_stops_the_run(monkeypatch, capsys):
    monkeypatch.setattr(node, "run_check", lambda: 2)
    monkeypatch.setattr(node, "run_once", lambda *a, **k: pytest.fail("must not launch"))
    monkeypatch.delenv("SKIP_CHECK", raising=False)
    assert node.mode_run(dry=False) == 1
    assert RESULT_FAILED.match(capsys.readouterr().out.strip().splitlines()[-1])


def test_extra_env_parsing():
    assert node.parse_extra_env("A=1 B='x y' C=") == {"A": "1", "B": "x y", "C": ""}
    for bad in ("A", "1A=2", "=3"):
        with pytest.raises(SystemExit):
            node.parse_extra_env(bad)
    cmd, env, setv = node.build_run({"TRAIN_SEED": "5"}, {"SEED": "1", "STEPS": "5000", "EXTRA_ENV": "LR_FLOOR=0.2", "NPROC": "2"})
    assert setv["TRAIN_SEED"] == "5" and setv["NUM_SCHEDULED_ITERATIONS"] == "5000" and env["LR_FLOOR"] == "0.2"  # the arm wins
    assert cmd[-2:] == ["--nproc_per_node=2", "train_gpt_medium_cpu.py"] and setv["DATA_PATH"] == "../.."


# ---- A/B report and the sweep fit ------------------------------------------------------------------------

def test_ab_report_and_fit_filter():
    run = lambda steps, loss, ms, seed: dict(steps=steps, val_loss=loss, train_time_ms=ms, seed=seed)
    pairs = [(run(5800, 2.8770, 221000, 0), run(5210, 2.9080, 199000, 0)), (run(5800, 2.8790, 222000, 1), run(5210, 2.9100, 198000, 1))]
    lines = node.ab_report(pairs, {"NUM_SCHEDULED_ITERATIONS": "5200"})
    last = lines[-1]
    assert last.startswith("AB variant=t2-cpu-ngram arm=NUM_SCHEDULED_ITERATIONS=5200 n=2 ") and "d_val_loss=+0.031000" in last and "alt_hit=2/2" in last
    assert any("two-point fit: -0.28" in l for l in lines)
    rows = [dict(steps=5210, val_loss=2.908, train_time_ms=1, seed=0), dict(steps=5800, val_loss=2.5, train_time_ms=1, knobs={"LR_FLOOR": "0.2"})]
    assert sweep_steps.fit([r for r in rows if not r.get("knobs")] + [sweep_steps.SEED_POINT])[1] == pytest.approx(-0.290, abs=0.01)


# ---- preflight pieces -------------------------------------------------------------------------------------

def test_build_threads_rule_and_override():
    h = node.host
    assert [h.build_threads(n, {}) for n in (26, 24, 23, 16, 15, 4)] == [64, 64, 32, 32, 16, 16]  # PR #380's rule
    assert h.build_threads(26, {"VAL_BUILD_THREADS": "16"}) == 16 and h.build_threads(26, {"VAL_BUILD_THREADS": ""}) == 64
    with pytest.raises(ValueError):
        h.build_threads(26, {"VAL_BUILD_THREADS": "0"})
    big, small = (node.retrieval_plan(Path("/nonexistent"), 5800, t, 8) for t in (64, 16))
    assert big.tables == small.tables == 144 * node.GiB  # the tables do not shrink
    assert round((big.total - small.total) / node.GiB) == 36 and round(big.total / node.GiB) == 388  # at a nominal 100M tokens per shard


def test_host_fallback_hint(monkeypatch):
    h = node.host
    def fake(avail):
        return h.Host("cpu", 208, 104, 2, 512 * node.GiB, avail * node.GiB, 256 * node.GiB, 256 * node.GiB, "always [madvise] never", True, "6.8.0", 8)
    monkeypatch.setattr(node, "gpu_query", lambda: None)
    for avail, level, hint in ((400, "PASS", False), (370, "FAIL", True), (300, "FAIL", False)):
        monkeypatch.setattr(h, "detect", lambda gpus=None, a=avail: fake(a))
        out = []
        node.run_check({"NPROC": "8"}, out.append)
        line = next(l for l in out if " host memory: " in l)
        assert line.startswith(level) and ("VAL_BUILD_THREADS=16 fits" in line) == hint, line
    out = []
    node.run_check({"NPROC": "8", "VAL_BUILD_THREADS": "16"}, out.append)  # 300 GiB: still short at 16 threads
    assert next(l for l in out if " host memory: " in l).startswith("FAIL")
    out = []
    node.run_check({"NPROC": "8", "HOST_CHECK": "0"}, out.append)
    assert next(l for l in out if " host memory: " in l).startswith("WARN")


def test_shard_size_check(tmp_path):
    def shard(name, tokens, size=None):
        head = (20240520).to_bytes(4, "little") + (1).to_bytes(4, "little") + tokens.to_bytes(4, "little")
        (tmp_path / name).write_bytes((head + bytes(1012) + bytes(2 * tokens))[:size])
    shard("fineweb_train_000001.bin", 50)
    shard("fineweb_train_000002.bin", 50, size=1024 + 60)  # an interrupted download
    shard("fineweb_val_000000.bin", 10)
    (tmp_path / "fineweb_train_000003.bin").write_bytes(b"")
    assert node.shard_problems(tmp_path) == ["fineweb_train_000002.bin", "fineweb_train_000003.bin"]


def test_fa3_cache_check(tmp_path):
    pin = dict(FA3_REPO="a/b", FA3_REVISION="r" * 40, FA3_BINARY_SHA256=hashlib.sha256(b"bin").hexdigest())
    assert node.fa3_cached(tmp_path, pin) == (False, f"a/b@rrrrrrrr not downloaded in {tmp_path}")
    (tmp_path / "models--a--b").mkdir()
    assert "only under models--" in node.fa3_cached(tmp_path, pin)[1]
    d = tmp_path / "kernels--a--b" / "snapshots" / pin["FA3_REVISION"] / "build" / "torch-stable-abi29-cu128-x86_64-linux"
    d.mkdir(parents=True)
    (d / "_flash_attn3_cuda_x.so").write_bytes(b"other")
    assert "has sha256" in node.fa3_cached(tmp_path, pin)[1]
    (d / "_flash_attn3_cuda_x.so").write_bytes(b"bin")
    assert node.fa3_cached(tmp_path, pin)[0]
    assert node.fa3_pin()["FA3_REPO"] == "devenpzak/flash-attn3-12864"  # read from model/attention.py
    assert node.hf_hub_dir({"HF_HOME": "/h"}) == Path("/h/hub") and node.hf_hub_dir({"HF_HUB_CACHE": "/c"}) == Path("/c")
