"""Shared by node.py (182/run_node.sh) and sweep_steps.py: where the tree is, how a run is launched, how its log
is read. Standard library only, so it imports on a machine with no torch.

The log format is this tree's own (train_gpt.py):
    step:1053/1053 val_loss:3.2785 train_time:30660ms step_avg:29.12ms       every validation (upstream's line)
    [final] val_loss:3.278512 train_time_ms:30660 steps:1053 scheduled:981    ours, full precision, final only
    step:1053/1053 val_loss_no_copy:... val_loss_copy:... copy_gain:... copy_fit_ms:... fit:...   COPY_MIX=1 only
A log from #397 itself (records/track_1_short/2026-10-31_ExactSpeedups) has only the first line, so the parser
falls back to it and takes the scheduled count from the "[config] steps N scheduled" line.
"""
from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

VARIANT_NAME = "t1-gpu-stack"
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR = os.path.abspath(os.path.join(TOOLS_DIR, "..", ".."))
REPO_DIR = os.path.abspath(os.path.join(VARIANT_DIR, "..", ".."))
TARGET_LOSS = 3.28
WORLD_SIZE = 8  # track_1_short/distributed.py asserts it

# Our flags (track_1_short/folds.py) and the base PR's own knobs. A/B arms clear every one of them first.
FLAG_KEYS = ("HOST_PREP", "DOC_COPY", "LAYER_ROPE", "COPY_MIX", "COPY_MIX_FIT",
             # r6-optim: track_1_short/optim/cascade_flags.py
             "OPTIM_MOEFUSED", "OPTIM_MAPS", "OPTIM_NORM", "OPTIM_PROFILE",
             # r6-kernel: track_1_short/perf/kernels/moe_router.py
             "MOE_ROUTER_FUSED",
             # r6-hostcpu: track_1_short/data.py
             "HOST_ASYNC", "HOST_ASYNC_DEPTH")
BASE_KEYS = ("EXACT_LEVERS", "NUM_SCHEDULED_ITERATIONS", "TRAIN_SEED")

_NUM = r"(?:[0-9]+(?:\.[0-9]*)?(?:e-?[0-9]+)?|nan|inf)"
FINAL_RE = re.compile(rf"^\[final\] val_loss:({_NUM}) train_time_ms:([0-9]+) steps:([0-9]+) scheduled:([0-9]+)", re.M)
VAL_RE = re.compile(rf"^step:(\d+)/(\d+) val_loss:({_NUM}) train_time:([0-9]+)ms", re.M)
TRAIN_RE = re.compile(r"^step:(\d+)/(\d+) train_time:([0-9]+)ms", re.M)
CONFIG_RE = re.compile(r"^\[config\] steps (\d+) scheduled", re.M)
COPY_RE = re.compile(rf"^step:\d+/\d+ val_loss_no_copy:({_NUM}) val_loss_copy:({_NUM}) copy_gain:(-?{_NUM}) "
                     rf"copy_fit_ms:([0-9]+) fit:(\w+)", re.M)
TRAINER_LOG_RE = re.compile(r"^(logs/[0-9a-f-]+\.txt)\s*$", re.M)
TRAINER_LOG_PATH_RE = re.compile(r"^TRAIN_LOG_PATH=(.+)$", re.M)
ERROR_RE = re.compile(r"^(?:\[rank\d+\]:\s*)?\S*(?:Error|Exception|AssertionError)\b.*$|^.*CUDA out of memory.*$|^.*Killed.*$", re.M)


@dataclass
class Run:
    scheduled: int | None     # NUM_SCHEDULED_ITERATIONS (None if the log does not say)
    steps: int                # steps trained (scheduled + growth + extension)
    val_loss: float           # the scored loss (the copy mixture's with COPY_MIX=1)
    train_time_s: float
    copy: dict | None = None  # COPY_MIX=1: val_loss_no_copy, copy_gain, copy_fit_ms, fit

    @property
    def hit(self) -> bool:
        return self.val_loss <= TARGET_LOSS


def parse_text(text: str) -> Run | None:
    """The finished run in a console capture or a trainer log, or None (crashed, truncated, NaN)."""
    final = FINAL_RE.findall(text)
    if final:
        loss, ms, steps, sched = final[-1]
        run = Run(int(sched), int(steps), float(loss), int(ms) / 1000)
    else:
        vals = [v for v in VAL_RE.findall(text) if v[0] == v[1]]  # the validation at the last step only
        if not vals:
            return None
        step, _, loss, ms = vals[-1]
        cfg = CONFIG_RE.findall(text)
        run = Run(int(cfg[-1]) if cfg else None, int(step), float(loss), int(ms) / 1000)
    if not math.isfinite(run.val_loss):
        return None
    copy = COPY_RE.findall(text)
    if copy:
        no_copy, _, gain, fit_ms, fit = copy[-1]
        run.copy = dict(val_loss_no_copy=float(no_copy), copy_gain=float(gain), copy_fit_ms=int(fit_ms), fit=fit)
    return run


def parse_log(path: str) -> Run | None:
    try:
        with open(path, errors="replace") as f:
            return parse_text(f.read())
    except OSError:
        return None


def failure_reason(text: str, exit_code: int | None) -> str:
    """One line saying why a log has no result: the last error line it holds, and how far training got."""
    if re.search(r"^(?:step:\d+/\d+|\[final\]) val_loss:(?:nan|inf)", text, re.M):
        return "val_loss is not finite"
    errors = [m.group(0).strip() for m in ERROR_RE.finditer(text)]
    # torchrun appends its generic child-failure exception after the rank's actual traceback.
    causes = [line for line in errors if "ChildFailedError" not in line]
    errors = causes or errors
    progress = TRAIN_RE.findall(text)
    where = f"last step {progress[-1][0]}/{progress[-1][1]}" if progress else "no training step logged"
    code = "" if exit_code in (None, 0) else f"exit code {exit_code}; "
    if errors:
        return f"{code}{errors[-1][:200]} ({where})".replace("\n", " ")
    return f"{code}no final validation line ({where})"


def trainer_log_path(console_text: str) -> str | None:
    """The trainer's own log (source + environment + every line), which it names on its first stdout line."""
    m = TRAINER_LOG_PATH_RE.search(console_text) or TRAINER_LOG_RE.search(console_text)
    path = os.path.join(VARIANT_DIR, m.group(1).rstrip("\r")) if m else None
    return path if path and os.path.exists(path) else None


def data_path() -> str:
    """Where data/fineweb10B lives: DATA_PATH if exported (relative to the variant dir), else the repo root."""
    return os.environ.get("DATA_PATH") or os.path.join("..", "..")


def torchrun_cmd(nproc: int) -> list[str]:
    return ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "train_gpt.py"]


def format_cmd(env_set: dict, cmd: list[str]) -> str:
    env = " ".join(f"{k}={v}" for k, v in env_set.items())
    return f"cd {VARIANT_DIR} && env {env} {' '.join(cmd)}"


def launch(env_set: dict, console_path: str, nproc: int = WORLD_SIZE, timeout_s: float = 1800,
           clear: tuple[str, ...] = (), echo: bool = False) -> tuple[int | None, str]:
    """One `torchrun ... train_gpt.py` from the variant dir, console to `console_path` (and to stdout with echo).
    The environment is the caller's minus `clear` plus `env_set`. Returns (exit code, note); the code is None on
    a timeout, after which the whole process group (torchrun and its 8 ranks) is killed."""
    env = {k: v for k, v in os.environ.items() if k not in clear}
    env.update({k: str(v) for k, v in env_set.items()})
    console_dir = os.path.dirname(os.path.abspath(console_path))
    os.makedirs(console_dir, exist_ok=True)
    trainer_dir = env.get("TRAIN_LOG_DIR") or os.path.join(console_dir, "trainer")
    trainer_dir = os.path.abspath(os.path.join(VARIANT_DIR, trainer_dir))
    env["TRAIN_LOG_DIR"] = trainer_dir
    # Check this before torchrun launches eight ranks. Never repair permissions or silently redirect
    # an explicitly selected directory: report the path that needs to be changed instead.
    os.makedirs(trainer_dir, exist_ok=True)
    with tempfile.TemporaryFile(dir=trainer_dir):
        pass
    deadline = time.monotonic() + timeout_s
    with open(console_path, "w") as out:
        try:
            proc = subprocess.Popen(torchrun_cmd(nproc), cwd=VARIANT_DIR, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
        except OSError as e:
            out.write(f"could not start torchrun: {e}\n")
            return 127, f"could not start torchrun: {e}"

        def kill_group(sig=signal.SIGTERM):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

        def on_signal(signum, _frame):
            kill_group()
            raise KeyboardInterrupt(f"signal {signum}")

        old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        note = ""
        try:
            import threading

            def watchdog():
                while proc.poll() is None:
                    if time.monotonic() > deadline:
                        kill_group()
                        time.sleep(10)
                        kill_group(signal.SIGKILL)
                        return
                    time.sleep(1)

            threading.Thread(target=watchdog, daemon=True).start()
            for line in proc.stdout:
                out.write(line)
                if echo:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            code = proc.wait()
            if time.monotonic() > deadline and code != 0:
                code, note = None, f"timeout after {timeout_s:.0f} s (RUN_TIMEOUT_S)"
        except KeyboardInterrupt as e:
            kill_group()
            proc.wait()
            code, note = 130, f"interrupted ({e})"
        finally:
            for s, h in old.items():
                signal.signal(s, h)
            kill_group()  # stragglers of a crashed torchrun
    return code, note
