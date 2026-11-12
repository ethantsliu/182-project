"""CPU tests for 182/run_node.sh and node.py: log parsing, RESULT lines, dry runs, the preflight, and a
whole `run` / `ab` against a stand-in for the Python interpreter (no torch, no GPU).

    python -m pytest 182 -q        # from variants/t2-gpu-port

fixtures/pr261_0f0f8a50.txt is one of PR #261's six record logs with the source dump and most per-step
lines removed.
"""
import json
import os
import re
import stat
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import node  # noqa: E402

RUN_NODE = os.path.join(HERE, "run_node.sh")
FIXTURE = os.path.join(HERE, "fixtures", "pr261_0f0f8a50.txt")
RESULT_OK = re.compile(r"^RESULT variant=t2-gpu-port val_loss=\d+\.\d{4} train_time_s=\d+\.\d{3} steps=\d+ target=2\.92 hit=[01] torch=\S+ log=\S+$")
RESULT_FAIL = re.compile(r"^RESULT variant=t2-gpu-port status=failed torch=\S+ reason=\S.*$")

with open(FIXTURE) as _f:
    REAL_LOG = _f.read()
CANON_TAIL = "canon_mask: val_loss_masked:2.914312 val_loss_unmasked:2.916101 delta:0.001789 canon_ms:412\n"


def clean_env(**extra):
    """The caller's environment without any knob of this tree or of the node contract."""
    drop = set(node.FLAGS) | {"NPROC", "SEED", "EXTRA_ENV", "LOG_DIR", "DRY_RUN", "DATA_PATH", "RUN_TIMEOUT", "SKIP_CHECK", "ALLOW_STACK",
                              "AB_RUNS", "AB_A", "AB_B", "PYTHON", "CUDA_HOME", "CUDA_PATH", "PREP_TIMEOUT", "PREP_SHARDS"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def sh(mode, env, timeout=120):
    return subprocess.run(["bash", RUN_NODE, mode], capture_output=True, text=True, env=env, timeout=timeout)


# ----------------------------------------------------------------------------- parsing

def test_parse_real_pr261_log():
    p = node.parse_log(REAL_LOG)
    assert p["completed"] and p["val_loss"] == 2.9161 and p["train_time_s"] == 860.467
    assert p["steps"] == p["total_steps"] == 4150 and p["peak_mem_mib"] == 59977 and p["canon"] is None
    assert node.failure_reason(p) is None
    line = node.result_line(p, "/x/logs/a.txt", None)
    assert line == ("RESULT variant=t2-gpu-port val_loss=2.9161 train_time_s=860.467 steps=4150 target=2.92 hit=1 "
                    "torch=2.10.0+cu128 log=/x/logs/a.txt")
    assert RESULT_OK.match(line)


def test_parse_truncated_log():
    cut = REAL_LOG[:REAL_LOG.index("step:2600/4150 train_time")]
    p = node.parse_log(cut + "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.17 GiB\n")
    assert not p["completed"] and p["val_loss"] == pytest.approx(3.0, abs=0.2) and p["steps"] == 2500
    reason = node.failure_reason(p, rc=1)
    assert "stopped at training step 2550 of 4150" in reason and "exit code 1" in reason and "out of memory" in reason
    line = node.result_line(p, "x", reason)
    assert RESULT_FAIL.match(line) and "\n" not in line and "val_loss=" not in line


OOM_TAIL = """mem rank=3 at=warmup_end alloc_gib=9.10 peak_alloc_gib=58.40 reserved_gib=60.00 peak_reserved_gib=60.10 free_gib=19.0 total_gib=79.19
OOM rank=3 phase=final_val step=4150 tokens=262144 max_seq_len=262144 VAL_CE_CHUNKS=64 alloc_gib=72.02 peak_alloc_gib=72.02 reserved_gib=73.10 peak_reserved_gib=73.10 free_gib=0.50 total_gib=79.19
OOM rank=3 | |===========================================================================|
OOM rank=3 | |                  PyTorch CUDA memory summary, device ID 3                 |
Traceback (most recent call last):
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.14 GiB. GPU 3 has a total capacity of 79.19 GiB of which 512.00 MiB is free.
"""


def test_parse_oom_report_goes_into_the_reason():
    """The trainer's OOM report (phase, step, tokens, allocator counters) lands in the one-line reason."""
    cut = REAL_LOG[:REAL_LOG.index("step:4150/4150 val_loss")]
    p = node.parse_log(cut + OOM_TAIL)
    reason = node.failure_reason(p, rc=1)
    assert "stopped at training step 4150 of 4150" in reason and "Tried to allocate 6.14 GiB" in reason
    assert "OOM rank=3 phase=final_val step=4150 tokens=262144 max_seq_len=262144 VAL_CE_CHUNKS=64 alloc_gib=72.02" in reason
    line = node.result_line(p, "x", reason)
    assert RESULT_FAIL.match(line) and "\n" not in line and " torch=2.10.0+cu128 reason=" in line
    # no report (e.g. an OOM inside NCCL): the last memory line stands in
    p = node.parse_log(cut + OOM_TAIL.replace("OOM rank=3 phase=", "XOOM rank=3 phase="))
    assert "last memory line: mem rank=3 at=warmup_end" in node.failure_reason(p, rc=1)
    # a validation that recovered is noted, and is not a failure
    rec = REAL_LOG.replace("step:4150/4150 val_loss", "OOM rank=0 recovered: final_val is repeated with VAL_CE_CHUNKS=16 "
                                                      "(same per-token losses, smaller logit blocks)\nstep:4150/4150 val_loss")
    p = node.parse_log(rec)
    assert node.failure_reason(p) is None and p["oom_recovered"][0].startswith("OOM rank=0 recovered: final_val")


def test_parse_no_steps_nan_and_timeout():
    p = node.parse_log("logs/x.txt\nTraceback (most recent call last):\nModuleNotFoundError: No module named 'kernels'\n")
    assert "no training step was logged" in node.failure_reason(p, rc=1) and "No module named 'kernels'" in node.failure_reason(p, rc=1)
    nan = REAL_LOG.replace("val_loss:2.9161", "val_loss:nan")
    p = node.parse_log(nan)
    assert not p["completed"] and "val_loss is nan" in node.failure_reason(p)
    assert "timed out" in node.failure_reason(node.parse_log(REAL_LOG), rc=-15, timed_out=True)
    # a complete log from a process that still exited non-zero is a failure, and says so
    assert "exit code 9" in node.failure_reason(node.parse_log(REAL_LOG), rc=9)
    # a missed target is a result, not a failure
    miss = node.parse_log(REAL_LOG.replace("val_loss:2.9161", "val_loss:2.9213"))
    assert node.result_line(miss, "l", node.failure_reason(miss)).split()[5:8] == ["target=2.92", "hit=0", "torch=2.10.0+cu128"]
    # the source dump's own print statement is not a result line
    assert node.parse_log('print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{t:.0f}ms")')["val_loss"] is None


def test_parse_canon_line():
    p = node.parse_log(REAL_LOG + CANON_TAIL)
    assert p["canon"] == dict(masked=2.914312, unmasked=2.916101, delta=0.001789, canon_ms=412.0)
    assert node.canon_line(p) == ("CANON variant=t2-gpu-port val_loss_masked=2.914312 val_loss_unmasked=2.916101 "
                                  "delta=0.001789 canon_ms=412")


def test_trainer_log_lines_match_the_parser():
    """The f-strings in train_gpt_medium.py that node.py parses still have the parsed shape."""
    with open(os.path.join(node.VARIANT_DIR, node.TRAINER)) as f:
        src = f.read()
    assert 'print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms' in src
    assert 'print0(f"canon_mask: val_loss_masked:{val_loss:.6f} val_loss_unmasked:{unmasked_loss:.6f} "' in src
    assert 'f"delta:{unmasked_loss - val_loss:.6f} canon_ms:{canon_ms:.0f}"' in src
    assert 'print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}", console=True)' in src
    assert 'head = (f"OOM rank={rank} phase={PHASE[\'phase\']} step={PHASE[\'step\']} tokens={PHASE[\'tokens\']} "' in src
    assert 'emit(f"OOM rank={rank} recovered: {phase} is repeated with VAL_CE_CHUNKS={VAL_CE_CHUNKS} "' in src
    assert 'f"mem rank={rank} at=stage_end step={s} bs={get_bs(s - 1)} ws={get_ws(s - 1)[1]} "' in src
    for flag in node.FLAGS:  # every flag node.py forwards is read by the trainer, with the same name
        assert re.search(rf'_env_(int|float)\("{flag}"|os\.environ\.get\("{flag}"', src), flag


def test_extra_env_parsing():
    assert node.parse_extra_env("CANON_MASK=1  XSA_LAYERS='3,15'") == {"CANON_MASK": "1", "XSA_LAYERS": "3,15"}
    assert node.parse_extra_env("") == {}
    for bad in ("FUSED_MLP", "1X=2", "A B=1"):
        with pytest.raises(ValueError):
            node.parse_extra_env(bad)


def test_data_budget():
    # 4125 scheduled steps over the batch schedule + 25 extension steps of 524288 tokens
    assert node.TRAIN_TOKENS == 1_905_459_200 and node.MIN_SHARDS == 20 and node.WANT_SHARDS == 25
    with open(os.path.join(node.VARIANT_DIR, node.TRAINER)) as f:
        src = f.read()
    assert "train_bs_schedule: tuple = (131072, 262144, 393216, 524288," in src
    assert '_env_int("NUM_SCHEDULED_ITERATIONS", 4125)' in src and '_env_int("NUM_EXTENSION_ITERATIONS", 25)' in src


def test_summarize_ab():
    rows = [dict(arm="A", pair=0, val_loss=2.9150, train_time_s=861.0, unmasked=2.9168, canon_ms=400.0),
            dict(arm="B", pair=0, val_loss=2.9152, train_time_s=850.0, unmasked=2.9170, canon_ms=420.0),
            dict(arm="B", pair=1, val_loss=2.9148, train_time_s=851.0, unmasked=2.9166, canon_ms=410.0),
            dict(arm="A", pair=1, val_loss=2.9156, train_time_s=860.0, unmasked=2.9174, canon_ms=410.0)]
    out = "\n".join(node.summarize_ab(rows, {"A": "CANON_MASK=1", "B": "CANON_MASK=1 FUSED_MLP=1"}))
    assert "AB arm=A flags=[CANON_MASK=1] n=2 val_loss=2.9153" in out and "train_time_s=860.5" in out
    assert "AB paired B-A n=2 train_time_s=-10.00 (-1.16%)" in out and "pairs_faster=2/2" in out
    assert "val_loss=-0.0003" in out  # mean of the unmasked differences (+0.0002, -0.0008)
    assert "AB canon_mask n=4 unmasked-masked=0.001800" in out and "canon_ms=410" in out
    # one arm only (the run after it failed): no paired line, no crash
    assert len(node.summarize_ab(rows[:1], {"A": "x", "B": "y"})) == 2


# ----------------------------------------------------------------------------- run_node.sh, dry

def test_dry_run_prints_the_command_and_touches_nothing(tmp_path):
    log_dir = tmp_path / "logs"
    r = sh("run", clean_env(DRY_RUN=1, LOG_DIR=log_dir, NPROC=4, CANON_MASK=1, EXTRA_ENV="FUSED_MLP=1 BOGUS=2", SEED=7))
    assert r.returncode == 0, r.stdout + r.stderr
    cmd = r.stdout.strip().splitlines()[-1]
    assert cmd.startswith(f"cd {node.VARIANT_DIR} && env DATA_PATH={node.REPO_ROOT} ")
    assert cmd.endswith("-m torch.distributed.run --standalone --nproc_per_node=4 train_gpt_medium.py")
    assert " CANON_MASK=1 " in cmd and " FUSED_MLP=1 " in cmd and " BOGUS=2 " in cmd
    assert "NOTE SEED is ignored" in r.stdout and "NOTE EXTRA_ENV sets BOGUS" in r.stdout
    assert not log_dir.exists() and "RESULT" not in r.stdout


def test_dry_run_default_is_base_config():
    r = sh("run", clean_env(DRY_RUN=1))
    assert r.returncode == 0
    (cmd,) = r.stdout.strip().splitlines()
    assert not any(f" {flag}=" in cmd for flag in node.FLAGS), cmd  # no flag set: PR #261's configuration
    assert "--nproc_per_node=8 " in cmd


def test_dry_run_ab_and_prep():
    r = sh("ab", clean_env(DRY_RUN=1, AB_RUNS=2))
    assert r.returncode == 0
    lines = r.stdout.strip().splitlines()
    assert "order A B B A" in lines[0] and len(lines) == 5
    assert all("CANON_MASK=1" in ln and "VAL_LOSS_EVERY=0" in ln for ln in lines[1:])
    assert ["FUSED_MLP=1" in ln for ln in lines[1:]] == [False, True, True, False]
    r = sh("prep", clean_env(DRY_RUN=1))
    assert r.returncode == 0 and "cached_fineweb10B.py 25" in r.stdout and "check_canon_val.py" in r.stdout
    assert "varunneal/flash-attention-3" in r.stdout and "nothing done" in r.stdout


def test_bad_inputs_fail_cleanly():
    r = sh("run", clean_env(DRY_RUN=1, EXTRA_ENV="FUSED_MLP"))
    assert r.returncode == 2 and RESULT_FAIL.match(r.stdout.strip().splitlines()[-1]) and "Traceback" not in r.stderr
    r = sh("nonsense", clean_env())
    assert r.returncode == 2 and "usage:" in r.stderr


def test_check_on_this_machine_reports_reasons():
    """No GPU here: check must exit 1 with FAIL lines that carry a fix, and never a traceback."""
    r = sh("check", clean_env(PATH=os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep)
                                                    if not os.path.exists(os.path.join(p, "nvidia-smi")))))
    lines = r.stdout.strip().splitlines()
    assert "Traceback" not in r.stdout + r.stderr, r.stderr
    assert all(re.match(r"^(PASS|WARN|FAIL) [^:]+: \S", ln) for ln in lines[:-1]), r.stdout
    assert re.match(r"^check: \d+ PASS, \d+ WARN, \d+ FAIL$", lines[-1])
    fails = [ln for ln in lines if ln.startswith("FAIL")]
    assert r.returncode == 1 and any(ln.startswith("FAIL GPUs: nvidia-smi not found") for ln in fails)
    assert all(ln.count("\n") == 0 and len(ln) < 600 for ln in lines)
    if not os.path.isdir(os.path.join(node.REPO_ROOT, "data", "fineweb10B")):
        assert any(ln.startswith("FAIL train shards: 0 in") and "run_node.sh prep" in ln for ln in fails)


# ----------------------------------------------------------------------------- a stand-in interpreter

GOOD_PROBE = dict(python="3.12.3", torch=dict(version="2.10.0+cu128", cuda="12.8", gpus=8), triton="3.6.0",
                  kernels=dict(version="0.16.1", hub="1.29.0", fa3_cached=True), numpy="2.2.6", tiktoken=50256,
                  nvrtc="ce_fwd_bwd_kernel (160285 B PTX), ce_fwd_bwd_prefix_kernel (162354 B PTX)")

FAKE_PY = r'''#!/usr/bin/env bash
# Stand-in for the node's python: answers node.py's import probe and "runs" the trainer from a canned log.
case " $* " in
  *" torch.distributed.run "*)
    echo "$FLAG_DUMP_PREFIX CANON_MASK=${CANON_MASK:-} FUSED_MLP=${FUSED_MLP:-} DATA_PATH=${DATA_PATH:-} args=$*" >> "$FAKE_CALLS"
    [ -n "${FAKE_SLEEP:-}" ] && { echo "step:10/4150 train_time:900ms step_avg:90.00ms"; sleep "$FAKE_SLEEP"; }
    if [ "${FUSED_MLP:-0}" = 1 ] && [ -n "${FAKE_LOG_FUSED:-}" ]; then cat "$FAKE_LOG_FUSED"; else cat "$FAKE_LOG"; fi
    exit "${FAKE_RC:-0}" ;;
  *PROBE*) echo "PROBE $FAKE_PROBE" ;;
  *) echo "/nonexistent" ;;
esac
'''


@pytest.fixture
def fake(tmp_path):
    """Environment in which PYTHON is the stand-in and LOG_DIR / the call record live under tmp_path."""
    py = tmp_path / "fakepython"
    py.write_text(FAKE_PY)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "trainer_stdout.txt"
    log.write_text(REAL_LOG)
    cuda = tmp_path / "cuda" / "include"
    cuda.mkdir(parents=True)
    (cuda / "cuda_bf16.h").write_text("")
    env = clean_env(PYTHON=py, FAKE_LOG=log, FAKE_CALLS=tmp_path / "calls.txt", LOG_DIR=tmp_path / "out",
                    FAKE_PROBE=json.dumps(GOOD_PROBE), CUDA_HOME=tmp_path / "cuda", SKIP_CHECK=1, FLAG_DUMP_PREFIX="call")
    return env, tmp_path


def node_py(mode, env, timeout=120):
    """node.py under the real interpreter, with PYTHON (what it launches) pointing at the stand-in."""
    return subprocess.run([sys.executable, os.path.join(HERE, "node.py"), mode], capture_output=True, text=True, env=env, timeout=timeout)


def test_run_success_last_line_is_result(fake):
    env, tmp = fake
    r = node_py("run", env)
    assert r.returncode == 0, r.stdout + r.stderr
    last = r.stdout.rstrip("\n").splitlines()[-1]
    m = RESULT_OK.match(last)
    assert m and " val_loss=2.9161 train_time_s=860.467 steps=4150 target=2.92 hit=1 " in last
    out_files = list((tmp / "out").glob("*-run.out"))
    assert len(out_files) == 1 and last.endswith(f"log={out_files[0]}")  # no logs/<uuid>.txt from the stand-in
    assert "step:4150/4150 val_loss:2.9161" in out_files[0].read_text()
    call = (tmp / "calls.txt").read_text()
    assert f"DATA_PATH={node.REPO_ROOT} " in call and "--nproc_per_node=8 train_gpt_medium.py" in call
    assert "CANON_MASK= FUSED_MLP= " in call  # default run sets none of this tree's flags


def test_run_with_canon_mask_reports_both_losses(fake):
    env, tmp = fake
    (tmp / "trainer_stdout.txt").write_text(REAL_LOG + CANON_TAIL)
    r = node_py("run", dict(env, CANON_MASK="1"))
    lines = r.stdout.rstrip("\n").splitlines()
    assert r.returncode == 0 and RESULT_OK.match(lines[-1])
    assert lines[-2].startswith("CANON variant=t2-gpu-port val_loss_masked=2.914312 val_loss_unmasked=2.916101 ")
    assert "CANON_MASK=1 " in (tmp / "calls.txt").read_text()


def test_run_failure_and_truncated_log(fake):
    env, tmp = fake
    (tmp / "trainer_stdout.txt").write_text(REAL_LOG[:REAL_LOG.index("step:1300/4150 train_time")] + "RuntimeError: NCCL error\n")
    r = node_py("run", dict(env, FAKE_RC="1"))
    last = r.stdout.rstrip("\n").splitlines()[-1]
    assert r.returncode == 1 and RESULT_FAIL.match(last)
    assert "stopped at training step 1250 of 4150" in last and "exit code 1" in last and "NCCL error" in last


def test_run_timeout_kills_the_group(fake):
    env, tmp = fake
    r = node_py("run", dict(env, FAKE_SLEEP="60", RUN_TIMEOUT="2"), timeout=60)
    last = r.stdout.rstrip("\n").splitlines()[-1]
    assert r.returncode == 1 and RESULT_FAIL.match(last) and "timed out (last training step 10 of 4150)" in last


def test_run_preflight_blocks_a_doomed_run(fake):
    env, tmp = fake
    probe = dict(GOOD_PROBE, torch=dict(version="2.12.0", cuda=None, gpus=0))
    r = node_py("run", dict(env, SKIP_CHECK="0", FAKE_PROBE=json.dumps(probe), DATA_PATH=tmp))
    last = r.stdout.rstrip("\n").splitlines()[-1]
    assert r.returncode == 1 and RESULT_FAIL.match(last) and "preflight" in last and " torch=2.12.0 " in last
    assert not (tmp / "calls.txt").exists()  # the trainer was never started


def test_run_preflight_blocks_an_unpinned_stack(fake):
    """torch 2.11 (what the node ran): check FAILs with the fix, run never starts; ALLOW_STACK=1 lets it run."""
    env, tmp = fake
    bindir = make_node(tmp)
    probe = dict(GOOD_PROBE, torch=dict(version="2.11.0+cu128", cuda="12.8", gpus=8))
    env = dict(env, SKIP_CHECK="0", FAKE_PROBE=json.dumps(probe), DATA_PATH=tmp, PATH=bindir + os.pathsep + env["PATH"])
    r = node_py("run", env)
    last = r.stdout.rstrip("\n").splitlines()[-1]
    assert r.returncode == 1 and RESULT_FAIL.match(last) and " torch=2.11.0+cu128 " in last
    assert "preflight: 1 FAIL, first: torch: 2.11.0+cu128" in last and "node_setup.sh --mode venv" in last
    assert not (tmp / "calls.txt").exists()
    r = node_py("run", dict(env, ALLOW_STACK="1"))
    last = r.stdout.rstrip("\n").splitlines()[-1]
    assert r.returncode == 0 and RESULT_OK.match(last), r.stdout
    assert "WARN torch: 2.11.0+cu128" in r.stdout and "running anyway (ALLOW_STACK=1)" in r.stdout
    # the version the trainer logged wins over the probe's (the stand-in replays #261's log)
    assert " torch=2.10.0+cu128 " in last


def make_node(tmp, n_train=25, gpus=8, name="NVIDIA H100 80GB HBM3", mem=81559):
    """A fake node: nvidia-smi on PATH and sparse shards of the real size under tmp/data/fineweb10B."""
    bindir = tmp / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\n" + "".join(f"echo '{name}, {mem}'\n" for _ in range(gpus)))
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    d = tmp / "data" / "fineweb10B"
    d.mkdir(parents=True)
    for fn in ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, n_train + 1)]:
        with open(d / fn, "wb") as f:
            f.truncate(1024 + 2 * 100_000_000)
    return str(bindir)


def test_check_passes_on_a_complete_fake_node(fake):
    env, tmp = fake
    bindir = make_node(tmp)
    r = node_py("check", dict(env, PATH=bindir + os.pathsep + env["PATH"], DATA_PATH=tmp))
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAIL")]
    assert r.returncode == 0 and not fails, r.stdout
    for want in ("PASS torch: 2.10.0+cu128", "PASS triton: 3.6.0", "PASS GPUs: 8 x NVIDIA H100 80GB HBM3",
                 "PASS train shards: 25 in", "PASS CE kernel compile: NVRTC sm_90", "PASS CUDA headers:", "PASS flash-attention wheel"):
        assert want in r.stdout, (want, r.stdout)


@pytest.mark.parametrize("change,expect", [
    (dict(n_train=9), "FAIL train shards: 9 in"),
    (dict(n_train=22), "WARN train shards: 22 in"),
    (dict(gpus=4), "FAIL GPUs: 4 visible, NPROC=8; fix: NPROC=4"),
    (dict(name="NVIDIA A100-SXM4-40GB", mem=40960), "FAIL GPUs: 8 x NVIDIA A100-SXM4-40GB; 8 have < 80 GB"),
])
def test_check_node_problems(fake, change, expect):
    env, tmp = fake
    bindir = make_node(tmp, **change)
    r = node_py("check", dict(env, PATH=bindir + os.pathsep + env["PATH"], DATA_PATH=tmp))
    assert expect in r.stdout, r.stdout
    assert r.returncode == (1 if expect.startswith("FAIL") else 0)


@pytest.mark.parametrize("probe_change,expect", [
    (dict(torch=dict(version="2.7.1+cu126", cuda="12.6", gpus=8)), "FAIL torch: 2.7.1+cu126 (CUDA 12.6): varunneal/flash-attention-3 has no build"),
    (dict(torch=dict(version="2.11.0+cu128", cuda="12.8", gpus=8)), "FAIL torch: 2.11.0+cu128 (CUDA 12.8); PR #261 was measured on 2.10.0+cu128"),
    (dict(torch=dict(version="2.10.1+cu128", cuda="12.8", gpus=8)), "FAIL torch: 2.10.1+cu128"),
    (dict(torch=dict(version="2.10.0+cu126", cuda="12.6", gpus=8)), "FAIL torch: 2.10.0+cu126"),
    (dict(triton="3.5.1"), "FAIL triton: 3.5.1; PR #261 ran 3.6.0"),
    (dict(nvrtc=None, nvrtc_error="OSError: CUDA_HOME environment variable is not set."), "FAIL CE kernel compile: NVRTC: OSError: CUDA_HOME"),
    (dict(kernels=None, kernels_error="ModuleNotFoundError: No module named 'kernels'"), "FAIL kernels: ModuleNotFoundError"),
    (dict(kernels=dict(version="0.16.1", hub="1.29.0", fa3_cached=False)), "WARN flash-attention wheel"),
])
def test_check_software_problems(fake, probe_change, expect):
    env, tmp = fake
    bindir = make_node(tmp)
    probe = {k: v for k, v in dict(GOOD_PROBE, **probe_change).items() if v is not None}
    r = node_py("check", dict(env, PATH=bindir + os.pathsep + env["PATH"], DATA_PATH=tmp, FAKE_PROBE=json.dumps(probe)))
    assert expect in r.stdout, r.stdout
    assert r.returncode == (1 if expect.startswith("FAIL") else 0), r.stdout
    if expect.startswith(("FAIL torch: 2.1", "FAIL triton")):  # the pin, not a missing build: fix command + opt-out
        assert "bash 182/tools/node_setup.sh --mode venv" in r.stdout and "ALLOW_STACK=1" in r.stdout
        r = node_py("check", dict(env, PATH=bindir + os.pathsep + env["PATH"], DATA_PATH=tmp, FAKE_PROBE=json.dumps(probe), ALLOW_STACK="1"))
        assert expect.replace("FAIL", "WARN", 1).split(";")[0] in r.stdout and r.returncode == 0, r.stdout


def test_ab_end_to_end(fake):
    env, tmp = fake
    (tmp / "trainer_stdout.txt").write_text(REAL_LOG + CANON_TAIL)
    fused = tmp / "fused.txt"
    fused.write_text(REAL_LOG.replace("train_time:860467ms", "train_time:851467ms") + CANON_TAIL)
    r = node_py("ab", dict(env, AB_RUNS="1", FAKE_LOG_FUSED=fused))
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert len(re.findall(r"^RESULT variant=t2-gpu-port val_loss=", out, re.M)) == 2
    assert "AB paired B-A n=1 train_time_s=-9.00 (-1.05%)" in out and "pairs_faster=1/1" in out
    assert "AB canon_mask n=2 unmasked-masked=0.001789" in out and out.rstrip().endswith("AB status=ok runs=2")
    calls = (tmp / "calls.txt").read_text().splitlines()
    assert ["FUSED_MLP=1" in c for c in calls] == [False, True] and all("CANON_MASK=1" in c for c in calls)


def test_ab_stops_at_the_first_failed_run(fake):
    env, tmp = fake
    r = node_py("ab", dict(env, AB_RUNS="2", FAKE_RC="3"))
    assert r.returncode == 1 and "AB status=failed reason=pair 1 arm A:" in r.stdout
    assert len((tmp / "calls.txt").read_text().splitlines()) == 1
