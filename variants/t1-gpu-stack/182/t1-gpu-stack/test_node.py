"""CPU tests of the node contract (182/run_node.sh, node.py, runlib.py). No GPU, no torch, no data, no network.

    python 182/t1-gpu-stack/test_node.py          # plain asserts, one line per test (pytest collects them too)

The log fixture is real: fixtures/pr397_seed3_3012f81e.tail.txt is everything after the source dump of
records/track_1_short/2026-10-31_ExactSpeedups/this_pr/3012f81e-....txt on PR #397's head (seed 3, 30.660 s,
val loss 3.2785). The end-to-end tests put a fake `torchrun` on PATH that replays it, so launching, the timeout,
the log copy and the RESULT line are exercised without a GPU. Nothing here changes os.environ.
"""
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import tempfile
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import node  # noqa: E402
import runlib  # noqa: E402

RUN_NODE = os.path.join(HERE, "..", "run_node.sh")
FIXTURE = os.path.join(HERE, "fixtures", "pr397_seed3_3012f81e.tail.txt")
RESULT_OK = re.compile(r"^RESULT variant=t1-gpu-stack val_loss=\d+\.\d+ train_time_s=\d+\.\d+ steps=\d+ target=3\.28 "
                       r"hit=[01] (?:torch=\S+ )?(?:triton=\S+ )?log=\S+$")
RESULT_FAILED = re.compile(r"^RESULT variant=t1-gpu-stack status=failed (?:torch=\S+ )?(?:triton=\S+ )?reason=\S.*$")
CHECK_LINE = re.compile(r"^(PASS|WARN|FAIL) [^:]+: \S.*$")

FAKE_TORCHRUN = """#!/usr/bin/env bash
# A stand-in for torchrun: records its environment, then behaves as FAKE_MODE says.
env | grep -E '^(DATA_PATH|TRAIN_SEED|NUM_SCHEDULED_ITERATIONS|HOST_PREP|DOC_COPY|LAYER_ROPE|COPY_MIX|COPY_MIX_FIT)=' | sort >> "$FAKE_ENV_LOG"
echo "args: $*" >> "$FAKE_ENV_LOG"
case "${FAKE_MODE:-ok}" in
  ok) cat "$FAKE_LOG";;
  crash) grep -v 'val_loss' "$FAKE_LOG"; echo "AssertionError: [kernels] the dense G3 (Gluon) kernel is not active: priming fell back to Triton"; exit 1;;
  hang) echo "step:25/1053 train_time:700ms step_avg:28.00ms"; sleep 600;;
  runlog) "$FAKE_PYTHON" - "$FAKE_RUN_LOG_MODULE" "$FAKE_LOG" <<'PY'
import os, pathlib, runpy, sys, types
from unittest import mock
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['triton'] = types.ModuleType('triton')
logger = runpy.run_path(sys.argv[1])
legacy = pathlib.Path('logs').resolve()
original = os.makedirs
def guarded(path, *args, **kwargs):
    if pathlib.Path(path).resolve() == legacy:
        raise PermissionError('simulated root-owned legacy logs directory')
    return original(path, *args, **kwargs)
with mock.patch('os.makedirs', side_effect=guarded):
    print0, flush = logger['start_run_log'](True, 'aabbccdd-1234')
    text = pathlib.Path(sys.argv[2]).read_text()
    print0(text, console=True)
    flush()
PY
  ;;
esac
"""


def fixture_text() -> str:
    return open(FIXTURE).read()


def full_log_text() -> str:
    """The fixture behind this tree's own source, as a trainer log starts: the source holds the format strings
    of every line the parser looks for, which must not be taken for results."""
    source = open(os.path.join(runlib.VARIANT_DIR, "train_gpt.py")).read()
    return source + "\n" + fixture_text()


def shell(mode: str, env: dict | None = None, cwd: str | None = None, timeout: float = 120):
    """run_node.sh <mode> with the caller's environment plus `env`, never the repo's venv (the tests' python)."""
    full = dict(os.environ, RUN_NODE_NO_VENV="1", PYTHON=sys.executable)
    for key in runlib.FLAG_KEYS + runlib.BASE_KEYS + ("SEED", "STEPS", "EXTRA_ENV", "NPROC", "DRY_RUN", "LOG_DIR",
                                                      "TRAIN_LOG_DIR", "DATA_PATH", "AB_RUNS", "AB_ARMS", "AB_STEPS", "AB_SEED_BASE"):
        full.pop(key, None)
    full.update(env or {})
    p = subprocess.run(["bash", RUN_NODE, mode], capture_output=True, text=True, env=full, cwd=cwd, timeout=timeout)
    return p.returncode, [line for line in p.stdout.splitlines() if line.strip()], p.stderr


def write_shard(path: str, tokens: int = 16, cut: int = 0):
    header = struct.pack("<3i", node.SHARD_MAGIC, 1, tokens) + bytes(node.SHARD_HEADER_BYTES - 12)
    with open(path, "wb") as f:
        f.write((header + bytes(2 * tokens))[:node.SHARD_HEADER_BYTES + 2 * tokens - cut])


def fake_node(tmp: str, mode: str = "ok", log_text: str | None = None) -> dict:
    """A temp dir that passes `run`'s preflight: two train shards and the validation shard, and a fake torchrun
    first on PATH. Returns the environment to run with."""
    data = os.path.join(tmp, "data", "fineweb10B")
    os.makedirs(data)
    for name in ("fineweb_val_000000.bin", "fineweb_train_000001.bin", "fineweb_train_000002.bin"):
        write_shard(os.path.join(data, name))
    bin_dir = os.path.join(tmp, "bin")
    os.makedirs(bin_dir)
    fake = os.path.join(bin_dir, "torchrun")
    with open(fake, "w") as f:
        f.write(FAKE_TORCHRUN)
    os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
    log = os.path.join(tmp, "fake_log.txt")
    with open(log, "w") as f:
        f.write(log_text if log_text is not None else fixture_text())
    return dict(PATH=bin_dir + os.pathsep + os.environ["PATH"], DATA_PATH=tmp, FAKE_LOG=log, FAKE_MODE=mode,
                FAKE_ENV_LOG=os.path.join(tmp, "env.txt"), LOG_DIR=os.path.join(tmp, "out"), FAKE_PYTHON=sys.executable,
                FAKE_RUN_LOG_MODULE=os.path.join(runlib.VARIANT_DIR, "track_1_short", "run_log.py"))


# ------------------------------------------------------------------------------------------- the parser
def test_parse_real_pr397_log():
    for text in (fixture_text(), full_log_text()):
        r = runlib.parse_text(text)
        assert r is not None and (r.scheduled, r.steps) == (981, 1053), r
        assert r.val_loss == 3.2785 and r.train_time_s == 30.660 and r.hit and r.copy is None, r


def test_parse_truncated_and_crashed_logs():
    lines = fixture_text().splitlines(keepends=True)
    cut = "".join(line for line in lines if "val_loss" not in line and "cplm_diag" not in line and "peak memory" not in line)
    assert runlib.parse_text(cut) is None
    assert runlib.failure_reason(cut, 1) == "exit code 1; no final validation line (last step 1053/1053)"
    early = cut[:cut.index("step:500/1053")]
    assert runlib.parse_text(early) is None and "last step 475/1053" in runlib.failure_reason(early, None)
    crashed = early + "Traceback (most recent call last):\n  File \"train_gpt.py\", line 279, in main\n" \
                      "AssertionError: [kernels] the dense G3 (Gluon) kernel is not active\n"
    reason = runlib.failure_reason(crashed, 1)
    assert "AssertionError: [kernels] the dense G3" in reason and "\n" not in reason, reason
    assert runlib.parse_text("") is None and "no training step logged" in runlib.failure_reason("", 1)
    # An intermediate validation is not a result, and neither is a NaN.
    assert runlib.parse_text("step:250/1053 val_loss:4.1000 train_time:7000ms step_avg:28.00ms\n") is None
    nan = "step:1053/1053 val_loss:nan train_time:30660ms step_avg:29.12ms\n"
    assert runlib.parse_text(nan) is None and runlib.failure_reason(nan, 0) == "val_loss is not finite"


def test_parse_our_final_and_copy_lines():
    text = ("[config] steps 966 scheduled + 52 growth + 20 extension = 1038; seed 4; train files x\n"
            "step:1038/1038 val_loss:3.2791 train_time:30410ms step_avg:29.30ms\n"
            "[final] val_loss:3.279112 train_time_ms:30410 steps:1038 scheduled:966\n"
            "step:1038/1038 val_loss_no_copy:3.279912 val_loss_copy:3.279112 copy_gain:0.000800 copy_fit_ms:212 "
            "fit:consumed lam:[0.0, 0.1]\n")
    r = runlib.parse_text(text)
    assert (r.scheduled, r.steps, r.val_loss, r.train_time_s) == (966, 1038, 3.279112, 30.41) and r.hit
    assert r.copy == dict(val_loss_no_copy=3.279912, copy_gain=0.0008, copy_fit_ms=212, fit="consumed")
    assert not runlib.parse_text(text.replace("3.279112", "3.280112")).hit
    negative = runlib.parse_text(text.replace("copy_gain:0.000800", "copy_gain:-0.000100"))
    assert negative.copy["copy_gain"] == -0.0001


def test_sweep_steps_reads_the_same_logs():
    import sweep_steps
    assert sweep_steps.parse_log(FIXTURE) == (981, 3.2785, 30.66)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "no_config.txt")
        with open(path, "w") as f:
            f.write("step:1053/1053 val_loss:3.2785 train_time:30660ms step_avg:29.12ms\n")
        assert runlib.parse_log(path).scheduled is None and sweep_steps.parse_log(path) is None  # no step count: no fit point


# ------------------------------------------------------------------------------------ check / prep helpers
def test_shard_problem():
    with tempfile.TemporaryDirectory() as tmp:
        good, cut, junk = (os.path.join(tmp, n) for n in ("good.bin", "cut.bin", "junk.bin"))
        write_shard(good)
        write_shard(cut, cut=8)
        with open(junk, "wb") as f:
            f.write(b"<html>rate limited</html>")
        assert node.shard_problem(good) is None
        assert "partial download" in node.shard_problem(cut)
        assert "bad header" in node.shard_problem(junk)
        assert node.shard_problem(os.path.join(tmp, "absent.bin")) is not None


def test_fa3_pin_matches_the_trainer():
    pin = node.fa3_pin()
    assert pin["REPO"] == "devenpzak/flash-attn3-12864" and len(pin["REVISION"]) == 40 and len(pin["BINARY_SHA256"]) == 64


def test_extra_env_parsing():
    assert node.parse_extra_env("HOST_PREP=all  COPY_MIX=1") == {"HOST_PREP": "all", "COPY_MIX": "1"}
    assert node.parse_extra_env("") == {} and node.parse_extra_env("A='b c'") == {"A": "b c"}
    for bad in ("HOST_PREP", "1X=2", "A-B=1"):
        try:
            node.parse_extra_env(bad)
        except ValueError:
            continue
        raise AssertionError(f"EXTRA_ENV {bad!r} must be refused")


def test_check_prints_contract_lines_and_fails_cleanly():
    code, lines, err = shell("check")
    assert "Traceback" not in err, err
    body, summary = lines[:-1], lines[-1]
    assert body and all(CHECK_LINE.match(line) for line in body), [line for line in body if not CHECK_LINE.match(line)]
    fails = sum(line.startswith("FAIL ") for line in body)
    assert summary.startswith(f"check: {fails} FAIL") and code == (1 if fails else 0), (code, summary)
    whats = {line.split(" ", 1)[1].split(":")[0] for line in body}
    assert {"python", "torch", "triton", "kernels", "gpus", "data", "vcpus", "ram", "extensions", "flags"} <= whats, whats
    for line in body:  # every failure names its fix
        if line.startswith("FAIL ") and not line.startswith(("FAIL gpus", "FAIL libcudart", "FAIL gpu memory", "FAIL flags")):
            assert "fix:" in line or "182/tools/node_setup.sh" in line, line


def test_check_refuses_a_bad_flag():
    code, lines, _ = shell("check", dict(LAYER_ROPE="sideways"))
    flag_line, = [line for line in lines if line.startswith(("PASS flags", "FAIL flags"))]
    assert code == 1 and flag_line.startswith("FAIL flags") and "LAYER_ROPE" in flag_line, flag_line


def test_prep_dry_run_lists_every_step():
    with tempfile.TemporaryDirectory() as tmp:
        code, lines, _ = shell("prep", dict(DRY_RUN="1", SHARDS="3", HF_HOME=tmp))
    text = "\n".join(lines)
    assert code == 0 and "cached_fineweb10B.py 3" in text and "prep fa3:" in text and "prep tokenizer:" in text, text
    code, lines, _ = shell("prep", dict(DRY_RUN="1", SKIP_DATA="1", SKIP_FA3="1"))
    assert code == 0 and "cached_fineweb10B" not in "\n".join(lines)


# -------------------------------------------------------------------------------------------------- run
def test_dry_run_prints_the_exact_command_from_any_directory():
    with tempfile.TemporaryDirectory() as elsewhere:
        code, lines, _ = shell("run", dict(DRY_RUN="1"), cwd=elsewhere)
        assert code == 0 and lines == [
            f"cd {runlib.VARIANT_DIR} && env DATA_PATH=../.. torchrun --standalone --nproc_per_node=8 train_gpt.py"], lines
        code, lines, _ = shell("run", dict(DRY_RUN="1", SEED="3", STEPS="966", EXTRA_ENV="HOST_PREP=all", COPY_MIX="1"),
                               cwd=elsewhere)
        assert code == 0 and lines[-1].endswith(
            "env COPY_MIX=1 DATA_PATH=../.. TRAIN_SEED=3 NUM_SCHEDULED_ITERATIONS=966 HOST_PREP=all "
            "torchrun --standalone --nproc_per_node=8 train_gpt.py"), lines


def test_run_failures_end_in_a_failed_result_line():
    cases = [dict(NPROC="4"), dict(EXTRA_ENV="HOST_PREP"), dict(SEED="three"), dict(NPROC="8", DATA_PATH="/nonexistent")]
    for env in cases:
        code, lines, err = shell("run", env)
        assert code != 0 and RESULT_FAILED.match(lines[-1]) and "Traceback" not in err, (env, lines[-1:], err)
    code, lines, _ = shell("bogus")
    assert code == 2 and "check|prep|run|ab" in "\n".join(lines) or "run_node.sh check" in "\n".join(lines)


def test_run_end_to_end_with_a_fake_torchrun():
    with tempfile.TemporaryDirectory() as tmp:
        env = fake_node(tmp)
        code, lines, err = shell("run", dict(env, SEED="3", HOST_PREP="all"))
        assert code == 0 and RESULT_OK.match(lines[-1]), (lines[-3:], err)
        fields = dict(f.split("=", 1) for f in lines[-1].split()[1:])
        assert (fields["val_loss"], fields["train_time_s"], fields["steps"], fields["hit"]) == ("3.278500", "30.660", "1053", "1")
        assert fields["torch"] == "2.10.0+cu128" and fields["triton"] == "3.6.0"
        assert os.path.dirname(fields["log"]) == env["LOG_DIR"] and runlib.parse_log(fields["log"]).steps == 1053
        launched = open(env["FAKE_ENV_LOG"]).read().splitlines()
        assert launched == [f"DATA_PATH={tmp}", "HOST_PREP=all", "TRAIN_SEED=3",
                            "args: --standalone --nproc_per_node=8 train_gpt.py"], launched


def test_report_reads_stack_from_file_only_environment_headers():
    with tempfile.TemporaryDirectory() as tmp:
        source_log = Path(tmp, "trainer-source.txt")
        source_log.write_text("Running PyTorch 2.10.0+cu128 compiled for CUDA 12.8\n"
                              "Running Triton version 3.6.0\n")
        console = Path(tmp, "console.txt")
        console.write_text(f"TRAIN_LOG_PATH={source_log}\n"
                           "[final] val_loss:3.277000 train_time_ms:30660 steps:1053 scheduled:981\n")
        with mock.patch("builtins.print") as output:
            assert node.report_run(str(console), 0, "", tmp, "copied") == 0
        result = output.call_args_list[-1].args[0]
        assert "torch=2.10.0+cu128 triton=3.6.0" in result


def test_permission_cause_survives_torchrun_child_failure_wrapper():
    text = ("[rank0]: PermissionError: [Errno 13] Permission denied: 'logs/run.txt'\n"
            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n")
    reason = runlib.failure_reason(text, 1)
    assert "PermissionError" in reason and "logs/run.txt" in reason
    assert "ChildFailedError" not in reason


def test_run_reports_a_crash_and_a_timeout():
    with tempfile.TemporaryDirectory() as tmp:
        env = fake_node(tmp, mode="crash")
        code, lines, _ = shell("run", env)
        assert code == 1 and RESULT_FAILED.match(lines[-1]), lines[-2:]
        assert "exit code 1; AssertionError: [kernels] the dense G3" in lines[-1] and "log=" in lines[-1], lines[-1]
        assert "torch=2.10.0+cu128 triton=3.6.0" in lines[-1]
    with tempfile.TemporaryDirectory() as tmp:
        env = fake_node(tmp, mode="hang")
        code, lines, _ = shell("run", dict(env, RUN_TIMEOUT_S="2"), timeout=60)
        assert code == 1 and RESULT_FAILED.match(lines[-1]) and "timeout after 2 s" in lines[-1], lines[-2:]
        left = subprocess.run(["pgrep", "-f", os.path.join(tmp, "bin", "torchrun")], capture_output=True, text=True).stdout
        assert not left.strip(), f"the fake torchrun survived its timeout: {left}"


def test_run_uses_session_trainer_logs_when_legacy_logs_are_unwritable():
    with tempfile.TemporaryDirectory(prefix="session logs ") as tmp:
        env = fake_node(tmp, mode="runlog")
        code, lines, err = shell("run", env)
        assert code == 0, (lines, err)
        trainer = Path(env["LOG_DIR"]) / "trainer" / "aabbccdd-1234.txt"
        assert trainer.exists() and runlib.parse_log(str(trainer)).steps == 1053
        marker = f"TRAIN_LOG_PATH={trainer}"
        assert marker in lines
        assert runlib.trainer_log_path("\n".join(lines)) == str(trainer)
        # report_run must preserve the source log even when the destination contains spaces.
        copies = list(Path(env["LOG_DIR"]).glob("t1-gpu-stack_*.txt"))
        assert any(p.read_text() == trainer.read_text() for p in copies)


def test_log_directory_failure_is_actionable_and_happens_before_launch():
    with tempfile.TemporaryDirectory() as tmp:
        env = fake_node(tmp)
        # An ordinary file deterministically blocks directory creation, even in root-run CI.
        blocked = Path(tmp) / "blocked"
        blocked.write_text("keep this file")
        for variable in ("LOG_DIR", "TRAIN_LOG_DIR"):
            code, lines, err = shell("run", dict(env, **{variable: str(blocked)}))
            assert code == 1 and RESULT_FAILED.match(lines[-1]), (lines, err)
            assert "logging:" in lines[-1] and "writable directories" in lines[-1]
            assert "Traceback" not in err
            assert not Path(env["FAKE_ENV_LOG"]).exists(), "torchrun must not launch"
            assert blocked.read_text() == "keep this file"


def test_launch_probes_trainer_log_permissions_before_spawning():
    with tempfile.TemporaryDirectory() as tmp:
        console = str(Path(tmp) / "console.txt")
        # A mocked permission error also covers CI that runs as root and ignores chmod bits.
        with mock.patch.object(runlib.tempfile, "TemporaryFile", side_effect=PermissionError("root-owned logs")), \
             mock.patch.object(runlib.subprocess, "Popen") as popen:
            try:
                runlib.launch({}, console)
            except PermissionError as exc:
                assert "root-owned" in str(exc)
            else:
                raise AssertionError("an unwritable trainer directory must fail before torchrun")
            popen.assert_not_called()


def test_trainer_log_parser_keeps_legacy_paths():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "logs" / "aabbccdd-1234.txt"
        path.parent.mkdir()
        path.write_text(fixture_text())
        with mock.patch.object(runlib, "VARIANT_DIR", tmp):
            assert runlib.trainer_log_path("logs/aabbccdd-1234.txt\n") == str(path)


# --------------------------------------------------------------------------------------------------- ab
def test_ab_plan_interleaves_and_rotates():
    plan = node.ab_plan(["base", "hostprep", "copymix"], [981], runs=4, seed_base=1, out_dir="/x")
    assert [p["arm"] for p in plan] == ["base", "hostprep", "copymix", "hostprep", "copymix", "base",
                                        "copymix", "base", "hostprep", "base", "hostprep", "copymix"]
    assert [p["seed"] for p in plan] == [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3
    assert plan[1]["path"] == "/x/hostprep/hostprep_s981_r0.txt"
    sweep = node.ab_plan(["base", "lropeattn"], [951, 981], runs=2, seed_base=5, out_dir="/x")
    assert len(sweep) == 8 and {(p["arm"], p["steps"], p["seed"]) for p in sweep} == {
        (a, s, k) for a in ("base", "lropeattn") for s in (951, 981) for k in (5, 6)}
    assert node.arm_env("lropeattn", 951, 5)["LAYER_ROPE"] == "attn" and "LAYER_ROPE" not in node.arm_env("base", 981, 1)


def test_ab_dry_run_clears_exported_flags():
    code, lines, _ = shell("ab", dict(DRY_RUN="1", AB_RUNS="2", LAYER_ROPE="1", TRAIN_SEED="99"))
    runs = [line for line in lines if line.startswith("[")]
    assert code == 0 and len(runs) == 6, lines
    assert not any("LAYER_ROPE" in line or "TRAIN_SEED=99" in line for line in runs), runs
    assert sum("HOST_PREP=all" in line for line in runs) == 2 and sum("COPY_MIX=1" in line for line in runs) == 2
    base = [line for line in runs if "] base " in line]
    assert all(line.endswith("env DATA_PATH=../.. TRAIN_SEED=%d NUM_SCHEDULED_ITERATIONS=981 torchrun --standalone "
                             "--nproc_per_node=8 train_gpt.py" % seed) for line, seed in zip(base, (1, 2))), base
    code, lines, _ = shell("ab", dict(DRY_RUN="1", AB_ARMS="hostprep nonsense"))
    assert code == 2 and "unknown arm" in lines[-1]
    code, lines, _ = shell("ab", dict(DRY_RUN="1", AB_ARMS="lropeattn", AB_STEPS="951 981", AB_RUNS="1"))
    assert code == 0 and len([line for line in lines if line.startswith("[")]) == 4  # base is added


def test_ab_summary_verdicts():
    arms, steps = ["base", "hostprep", "copymix", "doccopy"], [981]
    plan = node.ab_plan(arms, steps, runs=4, seed_base=1, out_dir="/x")
    results = {}
    for p in plan:
        k = p["round"]
        base_time, base_loss = 31.20 + 0.01 * k, 3.2770 + 0.0004 * k
        if p["arm"] == "base":
            run = runlib.Run(981, 1053, base_loss, base_time)
        elif p["arm"] == "hostprep":
            run = runlib.Run(981, 1053, base_loss + 0.0001 * (-1) ** k, base_time - 0.10 - 0.004 * k)
        elif p["arm"] == "copymix":
            run = runlib.Run(981, 1053, base_loss - 0.0004, base_time + 0.30,
                             copy=dict(val_loss_no_copy=base_loss, copy_gain=0.0004, copy_fit_ms=290, fit="consumed"))
        else:
            run = runlib.Run(981, 1053, base_loss - 0.0010, base_time + 0.50)
        results[p["path"]] = run
    lines = node.ab_summary(plan, results, arms, steps)
    verdict = {line.split()[1].rstrip(":"): line for line in lines if line.startswith("VERDICT")}
    assert "31.215 s" in verdict["base"] and "n=4" in verdict["base"], verdict["base"]
    assert "time -0.106 s" in verdict["hostprep"] and verdict["hostprep"].endswith("KEEP (the whole interval is below 0)")
    assert "copy_gain 0.40 mnat" in verdict["copymix"] and "290 ms fit" in verdict["copymix"] and verdict["copymix"].endswith("DROP")
    assert "net +250 ms" in verdict["doccopy"] and "AB_STEPS" in verdict["doccopy"], verdict["doccopy"]
    # A missing arm and a single pair are said, not computed.
    one = {p["path"]: results[p["path"]] for p in plan if p["round"] == 0 and p["arm"] in ("base", "hostprep")}
    lines = node.ab_summary(plan, one, arms, steps)
    assert any("one pair only" in line for line in lines) and any("VERDICT copymix: no seed finished" in line for line in lines)


def test_ab_end_to_end_resumes():
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(fake_node(tmp), AB_RUNS="2", AB_ARMS="base hostprep")
        code, lines, err = shell("ab", env)
        text = "\n".join(lines)
        assert code == 0 and "VERDICT base: #397 on this node 30.660 s" in text and "VERDICT hostprep: n=2 pairs" in text, (text, err)
        launches = open(env["FAKE_ENV_LOG"]).read().count("args:")
        assert launches == 4
        code, lines, _ = shell("ab", env)  # everything is on disk: nothing is launched again
        assert code == 0 and open(env["FAKE_ENV_LOG"]).read().count("args:") == 4 and sum(": have " in line for line in lines) == 4


# ------------------------------------------------------------------------------------------------ flags
def test_layer_rope_flag_values():
    sys.path.insert(0, runlib.VARIANT_DIR)
    from track_1_short import folds
    assert [folds._layer_rope(v) for v in ("0", "", "off", "1", "on", "all", "attn", " ATTN ")] == \
        ["", "", "", "all", "all", "all", "attn", "attn"]
    try:
        folds._layer_rope("mlp")
    except AssertionError:
        pass
    else:
        raise AssertionError("an unknown LAYER_ROPE value must be refused")
    # As the trainer reads them, in a child process (the module reads os.environ at import).
    for value, want in (("attn", "LAYER_ROPE=attn"), ("1", "LAYER_ROPE=1"), (None, "LAYER_ROPE=0")):
        env = {k: v for k, v in os.environ.items() if k not in runlib.FLAG_KEYS}
        if value is not None:
            env["LAYER_ROPE"] = value
        out = subprocess.run([sys.executable, "-c", "from track_1_short import folds; print(folds.describe(), "
                              "int(folds.LAYER_ROPE), int(folds.LAYER_ROPE_MLP))"], cwd=runlib.VARIANT_DIR, env=env,
                             capture_output=True, text=True).stdout.strip()
        assert want in out and out.endswith({"attn": "1 0", "1": "1 1", None: "0 0"}[value]), out


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    chosen = [t for t in TESTS if not sys.argv[1:] or t.__name__ in sys.argv[1:]]
    for test in chosen:
        test()
        print(f"ok  {test.__name__}", flush=True)
    print(f"{len(chosen)} node tests passed")
