"""CPU tests of the node contract (182/run_node.sh, 182/node.py): log parsing on a real upstream log and on failed
ones, the settings, the dry run, the check, and a whole `run` against a stand-in torchrun.

    cd variants/t1-cpu-ngram && python3 -m pytest -q 182
"""
import gzip
import os
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import node  # noqa: E402

RUN_NODE = os.path.join(HERE, "run_node.sh")
TAGS = r"(?: [a-z_0-9]+=\S+)*"  # config=, torch=, triton=, fa3=, tree=: before log= / reason=, which stay last
RESULT_OK = re.compile(rf"^RESULT variant=t1-cpu-ngram val_loss=\d+\.\d{{4}} train_time_s=\d+\.\d{{3}} steps=\d+ target=3\.28 hit=[01]{TAGS} log=\S+$")
RESULT_FAILED = re.compile(rf"^RESULT variant=t1-cpu-ngram status=failed{TAGS} reason=\S.*$")


@pytest.fixture(scope="module")
def upstream_log():
    """PR #380's record log 54de7700 (seed 0: 725 steps, val_loss 3.2765, 9.660 s), whole: source header and all."""
    with gzip.open(os.path.join(HERE, "testdata", "pr380_54de7700.txt.gz"), "rt") as f:
        return f.read()


def test_real_upstream_log(upstream_log):
    p = node.parse_log(upstream_log)
    assert p["completed"] and p["steps"] == p["total_steps"] == 725
    assert p["val_loss"] == 3.2765 and p["train_time_s"] == pytest.approx(9.660)
    assert p["vote"] is None
    line = node.result_line(p, "/x/log.txt")
    assert RESULT_OK.match(line) and " val_loss=3.2765 train_time_s=9.660 steps=725 " in line and " hit=1 " in line


def test_truncated_log(upstream_log):
    cut = upstream_log[:upstream_log.index("step:425/725")]
    p = node.parse_log(cut)
    assert not p["completed"] and p["val_loss"] is None and p["last_train_step"] == 400
    assert "step 400/725" in node.failure_reason(cut, p)
    head = upstream_log[:upstream_log.index("step:25/725")]  # died in the warmup: the source's f-strings are not step lines
    p = node.parse_log(head)
    assert not p["completed"] and p["last_train_step"] is None
    assert "before the timed run" in node.failure_reason(head, p)


def test_failed_log(upstream_log):
    text = upstream_log[:upstream_log.index("step:425/725")] + (
        "Traceback (most recent call last):\n  File \"train_gpt.py\", line 1, in <module>\n"
        "RuntimeError: this node cannot hold the exact-match retrieval: 100 GiB of RAM available\n")
    p = node.parse_log(text)
    reason = node.failure_reason(text, p)
    assert reason.startswith("RuntimeError: this node cannot hold")
    assert RESULT_FAILED.match(node.result_line(reason=reason + "\n second line"))
    nan = "step:685/685 val_loss:nan train_time:9000ms step_avg:13.00ms\n"
    assert not node.parse_log(nan)["completed"] and "nan" in node.failure_reason(nan, node.parse_log(nan))


def test_vote_and_host_lines():
    text = ("step:685/685 val_loss:3.2741 train_time:9120ms step_avg:13.31ms\n"
            "table vote: val_loss:3.27412 with the table's links, 3.27530 without them, 3.27611 with PR #399's fixed vote instead (k=706.6,43.39,2.694)\n"
            "host memory: rank 0 peak RSS 181.2 GiB, node in use 402 GiB (incl. /dev/shm 164 GiB)\n")
    p = node.parse_log(text)
    assert p["completed"] and p["vote"] == (3.27412, 3.2753, 3.27611) and p["host_memory"].startswith("host memory: rank 0")
    assert node.result_line(p, "l").endswith("hit=1 log=l")
    assert " hit=0 " in node.result_line(dict(p, val_loss=3.2801), "l")


def test_timeline():
    console = "\n".join([
        "[1000.000] logs/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.txt",
        "[1100.500] step:25/685 train_time:500ms step_avg:20.00ms",
        "[1101.000] validation index built in 0.90 s (2-token key)",
        "[1102.000] validation index built in 1.00 s (3-token key)",
        "[1107.500] validation index built in 5.40 s (6-token key)",
        "[1108.700] step:685/685 train_time:8700ms step_avg:12.70ms",
        "[1109.300] step:685/685 val_loss:3.2741 train_time:9250ms step_avg:13.50ms"])
    tl = node.timeline(console)
    assert [k for k, _, _ in tl["builds"]] == ["2", "3", "6"]
    assert tl["builds"][2][1] == pytest.approx(7.5) and tl["last_train_s"] == 8.7 and tl["clock_s"] == 9.25
    assert "1.20 s before it (not on the critical path)" in node.describe_timeline(tl)
    late = console.replace("[1107.500]", "[1109.000]")
    assert "AFTER it" in node.describe_timeline(node.timeline(late))
    assert node.timeline("step:25/685 train_time:500ms") is None  # no stamps


def test_settings_and_knobs():
    s = node.Settings({})
    assert s.config == "381" and s.nproc == 8 and s.data == node.REPO and s.command()[-1] == "train_gpt.py"
    assert s.knobs() == {"DATA_PATH": node.REPO}  # the default run sets nothing but the data path: PR #381's behaviour
    s = node.Settings({"CONFIG": "380", "SEED": "3", "NPROC": "4", "EXTRA_ENV": "SLOT_SHARED_ARENA=1 VOTE_K='1,2,3'", "TABLE_VOTE": "1",
                       "DATA_PATH": "/dev/shm", "UNRELATED": "x"})
    k = s.knobs({"STREAM_THREADS": "8"})
    assert k == {"NUM_SCHEDULED_ITERATIONS": "715", "SCAN_HINTS": "0", "VAL_TOPK": "2", "TABLE_VOTE": "1", "SLOT_SHARED_ARENA": "1",
                 "VOTE_K": "1,2,3", "STREAM_THREADS": "8", "TRAIN_SEED": "3", "DATA_PATH": "/dev/shm"}
    assert "--nproc_per_node=4" in s.shell_line() and "TRAIN_SEED=3" in s.shell_line()
    for bad in ({"CONFIG": "399"}, {"SEED": "x"}, {"EXTRA_ENV": "novalue"}, {"EXTRA_ENV": "1A=2"}):
        with pytest.raises(ValueError):
            node.Settings(bad)


def _shard(path, n, magic=node.SHARD_MAGIC, short=0):
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", magic, 1, n).ljust(1024, b"\0") + b"\0" * (2 * n - short))


def test_shards(tmp_path):
    names = node.shard_names()
    assert len(names) == 104 and names[0] == "fineweb_val_000000.bin" and names[-1] == "fineweb_train_000103.bin"
    for n in names:
        _shard(tmp_path / n, 10)
    assert node.missing_shards(str(tmp_path)) == []
    _shard(tmp_path / names[5], 10, short=2)       # a partial download
    _shard(tmp_path / names[7], 10, magic=1)       # not a shard
    os.remove(tmp_path / names[9])
    assert node.missing_shards(str(tmp_path)) == [names[5], names[7], names[9]]


def test_is_trainer_and_stats():
    assert node.is_trainer(["/venv/bin/python3.12", "-u", "train_gpt.py"])
    assert node.is_trainer(["/venv/bin/torchrun", "--standalone", "--nproc_per_node=8", "train_gpt_medium.py"])
    assert not node.is_trainer(["/bin/zsh", "-c", "grep -n x train_gpt.py"])
    assert not node.is_trainer(["vim", "train_gpt.py"]) and not node.is_trainer([""])
    mean, half = node.paired([0.1, 0.2, 0.3, 0.2])
    assert mean == pytest.approx(0.2) and half == pytest.approx(3.182 * 0.0816496581 / 2)
    assert node.paired([0.5]) == (0.5, None)
    a = dict(train_time_s=9.6, val_loss=3.2760, vote=None)
    b = dict(train_time_s=9.7, val_loss=3.2750, vote=(3.2750, 3.2761, 3.2770))
    lines = node.ab_summary([(a, b), (a, dict(b, train_time_s=9.8))], "", "TABLE_VOTE=1")
    assert lines[0].startswith("AB n=2 pairs, B - A: train_time +0.150 s") and "links - none -0.00110" in lines[2]
    assert node.ab_summary([], "", "x") == ["AB no completed pair"]
    assert len(node.source_hash()) == 64 and node.source_hash() == node.source_hash()


# ------------------------------------------------------------------- the script itself, as the session calls it

def _sh(mode, env=None, cwd="/"):
    full = {k: v for k, v in os.environ.items() if k not in ("CONFIG", "SEED", "EXTRA_ENV", "DRY_RUN", "FORCE", "NPROC", "DATA_PATH", "LOG_DIR") + node.KNOBS}
    full.update(env or {})
    return subprocess.run(["bash", RUN_NODE, mode], capture_output=True, text=True, cwd=cwd, env=full, timeout=300)


def test_dry_run_prints_the_command_and_changes_nothing(tmp_path):
    before = sorted(os.listdir(node.VDIR))
    r = _sh("run", {"DRY_RUN": "1", "LOG_DIR": str(tmp_path / "logs")}, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    last = r.stdout.strip().splitlines()[-1]
    assert last.endswith("torchrun --standalone --nproc_per_node=8 train_gpt.py") and f"cd {node.VDIR} && " in last
    assert "TABLE_VOTE" not in last and "SLOT_SHARED_ARENA" not in last and "NUM_SCHEDULED" not in last
    assert not (tmp_path / "logs").exists() and sorted(os.listdir(node.VDIR)) == before
    r = _sh("run", {"DRY_RUN": "1", "CONFIG": "380", "SEED": "7"})
    assert "NUM_SCHEDULED_ITERATIONS=715 SCAN_HINTS=0 VAL_TOPK=2 TRAIN_SEED=7" in r.stdout
    r = _sh("ab", {"DRY_RUN": "1", "AB_RUNS": "3"})
    lines = [l for l in r.stdout.splitlines() if l.startswith("cd ")]
    assert r.returncode == 0 and len(lines) == 6 and sum("TABLE_VOTE=1" in l for l in lines) == 3
    assert ["TABLE_VOTE=1" in l for l in lines] == [False, True, True, False, False, True]  # interleaved
    assert _sh("prep", {"DRY_RUN": "1"}).returncode == 0


def test_bad_settings_fail_with_a_result_line():
    r = _sh("run", {"CONFIG": "nope"})
    assert r.returncode != 0 and RESULT_FAILED.match(r.stdout.strip().splitlines()[-1])
    assert _sh("frobnicate").returncode == 2


def test_check_reports_and_never_crashes(tmp_path):
    r = _sh("check", {"DATA_PATH": str(tmp_path)})  # no data there: at least that FAILs, on any box
    assert "Traceback" not in r.stdout + r.stderr, r.stdout + r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0].startswith("host: ") and lines[1].startswith("plan: ")
    rows = lines[2:-1]
    assert rows and all(re.match(r"^(PASS|WARN|FAIL) [^:]+: \S", l) for l in rows), rows
    fails = sum(l.startswith("FAIL") for l in rows)
    assert fails >= 1 and r.returncode == 1 and lines[-1].startswith(f"NOT READY: {fails} FAIL")
    assert any(l.startswith("FAIL data: 0/104") and "run_node.sh prep" in l for l in rows)
    assert not any("; fix:" in l for l in rows if l.startswith("PASS"))


FAKE_TORCHRUN = """#!/usr/bin/env bash
# a stand-in for torchrun: writes a trainer log and prints what rank 0 prints
case "$FAKE_MODE" in
  hang) sleep 600;;
  crash) echo "logs/11111111-2222-3333-4444-555555555555.txt"; echo "step:25/685 train_time:500ms step_avg:20.00ms"
         echo "RuntimeError: CUDA error: an illegal memory access was encountered" >&2; exit 1;;
esac
mkdir -p "$TRAIN_LOG_DIR"
log="$TRAIN_LOG_DIR/11111111-2222-3333-4444-555555555555.txt"
{
  echo 'print0(f"step:{step}/{n} val_loss:{val_loss:.4f} train_time:{t:.0f}ms")'
  echo "Running PyTorch ${FAKE_TORCH:-2.10.0+cu128} compiled for CUDA 12.8"
  echo "Running Triton version 3.6.0"
  echo "attention kernel: pinned=${FAKE_PINNED:-1} module=flash_attn3.flash_attn_interface binary=_flash_attn3_cuda_8a730d9.abi3.so sha256=x repo=r@1 kernels=0.16.1"
  echo "retrieval host plan: 32 build threads, 4 online-index threads per rank; x"
  echo "step:25/685 train_time:500ms step_avg:20.00ms"
  echo "step:685/685 train_time:8700ms step_avg:12.70ms"
  echo "step:685/685 val_loss:3.2741 train_time:9250ms step_avg:13.50ms"
  [ -n "$TABLE_VOTE" ] && echo "table vote: val_loss:3.27412 with the table's links, 3.27530 without them, 3.27611 with PR #399's fixed vote instead (k=1)"
  echo "val diagnostics: network val_loss:3.3120 (before the chain; the chain adds -0.0379); hint rows: 6-token 0.4100, 3/4-token 0.3500, 2-token 0.2000, none 0.0400 of 10485760"
  echo "host memory: rank 0 peak RSS 181.2 GiB, node in use 402 GiB (incl. /dev/shm 164 GiB)"
} > "$log"
echo "$log"
echo "TRAIN_LOG_PATH=$log"
echo "seen: args=$* seed=${TRAIN_SEED:-none} data=$DATA_PATH steps=${NUM_SCHEDULED_ITERATIONS:-default}"
echo "validation index built in 5.40 s (6-token key)"
grep -v print0 "$log"
"""


@pytest.fixture
def fake_torchrun(tmp_path):
    path = tmp_path / "bin" / "torchrun"
    path.parent.mkdir()
    path.write_text(FAKE_TORCHRUN)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    env = {"PATH": f"{path.parent}{os.pathsep}{os.environ['PATH']}", "FORCE": "1", "WARM_DATA": "0", "LOG_DIR": str(tmp_path / "out"),
           "DATA_PATH": str(tmp_path)}
    yield env
    leftover = os.path.join(node.VDIR, "logs", "11111111-2222-3333-4444-555555555555.txt")
    if os.path.exists(leftover):
        os.remove(leftover)
        if not os.listdir(os.path.dirname(leftover)):
            os.rmdir(os.path.dirname(leftover))


def test_run_end_to_end_with_a_stand_in_torchrun(fake_torchrun, tmp_path):
    r = _sh("run", dict(fake_torchrun, SEED="5", CONFIG="380"), cwd=str(tmp_path))
    lines = r.stdout.strip().splitlines()
    assert r.returncode == 0, r.stdout + r.stderr
    assert RESULT_OK.match(lines[-1]) and " val_loss=3.2741 train_time_s=9.250 steps=685 " in lines[-1]
    assert "seen: args=--standalone --nproc_per_node=8 train_gpt.py seed=5 data=" + str(tmp_path) + " steps=715" in r.stdout
    log = lines[-1].rsplit("log=", 1)[1]
    assert os.path.isfile(log) and log.startswith(str(tmp_path / "out")) and "val_loss:3.2741" in open(log).read()
    assert any(l.startswith("timeline: index builds finished 6-token at") for l in lines)
    assert any(l.startswith("host memory: rank 0 peak RSS") for l in lines)
    consoles = [f for f in os.listdir(tmp_path / "out") if f.endswith(".console.txt")]
    assert len(consoles) == 1 and re.match(r"^\[\d+\.\d{3}\] ", open(tmp_path / "out" / consoles[0]).readline())


def test_run_failures_end_with_a_failed_result(fake_torchrun):
    r = _sh("run", dict(fake_torchrun, FAKE_MODE="crash"))
    last = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 1 and RESULT_FAILED.match(last) and "exit 1: RuntimeError: CUDA error" in last
    r = _sh("run", dict(fake_torchrun, FAKE_MODE="hang", RUN_TIMEOUT_S="2"))
    last = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 1 and RESULT_FAILED.match(last) and "timed out after 2 s" in last
    r = _sh("run", dict(fake_torchrun, FORCE="0"))  # the preflight stops a run that cannot work (no data in DATA_PATH)
    last = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 1 and RESULT_FAILED.match(last) and "preflight failed" in last and "seen: args" not in r.stdout


def test_ab_end_to_end_with_a_stand_in_torchrun(fake_torchrun):
    r = _sh("ab", dict(fake_torchrun, AB_RUNS="2"))
    assert r.returncode == 0, r.stdout + r.stderr
    runs = [l for l in r.stdout.splitlines() if l.startswith("AB_RUN ")]
    assert [l.split()[1:3] for l in runs] == [["arm=A", "seed=0"], ["arm=B", "seed=0"], ["arm=B", "seed=1"], ["arm=A", "seed=1"]]
    assert "AB n=2 pairs, B - A: train_time +0.000 s" in r.stdout
    assert "AB table vote within B's runs (same weights, n=2): links - none -0.00118" in r.stdout


def test_upstream_log_stack_fields(upstream_log):
    p = node.parse_log(upstream_log)
    assert (p["torch"], p["cuda"], p["triton"]) == ("2.10.0+cu128", "12.8", "3.6.0")
    assert p["fa3"] is None and p["diag"] is None and p["build_threads"] is None  # #380's logs predate these lines
    tags = dict(config="380", torch=p["torch"], triton=p["triton"], fa3="unknown", tree="abc-dirty")
    line = node.result_line(p, "/x/log.txt", tags=tags)
    assert RESULT_OK.match(line) and line.endswith(" hit=1 config=380 torch=2.10.0+cu128 triton=3.6.0 fa3=unknown tree=abc-dirty log=/x/log.txt")
    failed = node.result_line(reason="exit 1: boom", tags=dict(tags, torch="2.11 x"))
    assert RESULT_FAILED.match(failed) and " torch=2.11_x " in failed and failed.endswith(" reason=exit 1: boom")
    ref = node.reference_line("380", dict(p, val_loss=3.3008))
    assert ref.startswith("reference: val_loss 3.3008 vs 3.2759") and "+0.0249, +15.6 std" in ref


def test_result_line_parses_in_the_session_summary(upstream_log):
    """The root's 182/tools/summarize_session.py reads the tagged RESULT lines as before."""
    import importlib.util
    path = os.path.join(node.REPO, "182", "tools", "summarize_session.py")
    spec = importlib.util.spec_from_file_location("summarize_session", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tags = dict(config="381", torch="2.10.0+cu128", triton="3.6.0", fa3="pinned", tree="0123456789ab")
    r = mod.parse_result(node.result_line(node.parse_log(upstream_log), "/x/l.txt", tags=tags))
    assert r["ok"] and r["val_loss"] == 3.2765 and r["log"] == "/x/l.txt" and r["variant"] == "t1-cpu-ngram"
    r = mod.parse_result(node.result_line(reason="preflight failed: x", tags=tags))
    assert not r["ok"] and r["reason"] == "preflight failed: x"


def test_tree_state(tmp_path):
    assert node.tree_state(str(tmp_path)) == (None, [])  # not a checkout
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "track_1_short").mkdir()
    (tmp_path / "track_1_short" / "attention.py").write_text("x = 1\n")
    (tmp_path / "train_gpt.py").write_text("y = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    commit, dirty = node.tree_state(str(tmp_path))
    assert re.fullmatch(r"[0-9a-f]{12}", commit) and dirty == []
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.txt").write_text("not the trainer")  # outside TREE_PATHS
    assert node.tree_state(str(tmp_path)) == (commit, [])
    (tmp_path / "track_1_short" / "attention.py").write_text("import sys\nx = 2\n")  # a patched loader
    (tmp_path / "track_1_short" / "fallback.py").write_text("z = 1\n")
    assert node.tree_state(str(tmp_path)) == (commit, ["track_1_short/attention.py", "track_1_short/fallback.py"])


def test_check_fails_on_a_changed_trainer(monkeypatch, tmp_path):
    rows = {}
    def run_check(env, tree):
        monkeypatch.setattr(node, "tree_state", lambda vdir=None: tree)
        out = []
        node.check(node.Settings(dict(env, DATA_PATH=str(tmp_path))), out=out.append)
        return next(l for l in out if " trainer source: " in l)
    assert run_check({}, ("0123456789ab", [])).startswith("PASS trainer source: as committed (0123456789ab)")
    row = run_check({}, ("0123456789ab", ["track_1_short/model/attention.py"]))
    assert row.startswith("FAIL trainer source: 1 file(s) differ from commit 0123456789ab: track_1_short/model/attention.py") and "ALLOW_DIRTY=1" in row
    assert run_check({"ALLOW_DIRTY": "1"}, ("0123456789ab", ["x"])).startswith("WARN trainer source: ")
    assert run_check({}, (None, [])).startswith("WARN trainer source: not a git checkout")


def test_run_attributes_an_off_stack_result(fake_torchrun, tmp_path):
    r = _sh("run", dict(fake_torchrun, FAKE_TORCH="2.11.0+cu130", FAKE_PINNED="0"), cwd=str(tmp_path))
    lines = r.stdout.strip().splitlines()
    assert r.returncode == 0, r.stdout + r.stderr
    assert RESULT_OK.match(lines[-1]) and " config=381 torch=2.11.0+cu130 triton=3.6.0 fa3=other tree=" in lines[-1]
    assert any(l.startswith("WARN this run's stack: torch 2.11.0+cu130 / triton 3.6.0, not the records' 2.10.0+cu128") for l in lines)
    assert any(l.startswith("WARN this run's attention kernel: other") for l in lines)
    ref = next(l for l in lines if l.startswith("reference: "))
    assert "val_loss 3.2741 vs 3.2750 (estimate" in ref and "network alone 3.3120; hint rows 6-token 0.410, 3/4-token 0.350, 2-token 0.200, none 0.040" in ref
    r = _sh("run", fake_torchrun, cwd=str(tmp_path))  # the records' stack and the pinned kernel: no stack warnings
    assert r.returncode == 0 and " torch=2.10.0+cu128 triton=3.6.0 fa3=pinned " in r.stdout.strip().splitlines()[-1]
    assert "WARN this run's stack" not in r.stdout and "WARN this run's attention kernel" not in r.stdout


def test_run_uses_selected_trainer_log_directory(fake_torchrun, tmp_path):
    selected = tmp_path / "source logs"
    r = _sh("run", dict(fake_torchrun, TRAIN_LOG_DIR=str(selected)))
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"TRAIN_LOG_PATH={selected}/11111111-2222-3333-4444-555555555555.txt" in r.stdout
    copied = r.stdout.strip().splitlines()[-1].rsplit("log=", 1)[1]
    assert Path(copied).read_text().startswith('print0(')


def test_bad_trainer_log_destination_fails_before_launch(fake_torchrun, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("keep")
    r = _sh("run", dict(fake_torchrun, TRAIN_LOG_DIR=str(blocker)))
    assert r.returncode == 1
    assert "cannot write run logs" in r.stdout and "set LOG_DIR and TRAIN_LOG_DIR" in r.stdout
    assert "seen: args" not in r.stdout and blocker.read_text() == "keep"
