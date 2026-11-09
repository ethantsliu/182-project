"""Shared helpers for sweep_steps.py and significance.py: launch one run, parse its result, one-sided t-test.

Standard library only (scipy is used to cross-check the t-test when it is installed).
"""
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENTRY = "train_gpt_medium_cpu.py"
TARGET = 2.92

_ENDPOINT_KV = re.compile(r"^endpoint_result: (.*)$", re.M)
_ENDPOINT_JSON = re.compile(r"^endpoint_result:(\{.*\})$", re.M)  # @ldmberman's attachment format
_VAL_LINE = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms", re.M)
_LOG_PATH = re.compile(r"^(logs/[0-9a-f-]+\.txt)$", re.M)


def parse_result(text: str) -> dict | None:
    """The final validation of one run's stdout or log file: steps, val_loss, train_time_ms (+ seed, log)."""
    out = None
    m = _VAL_LINE.findall(text)
    final = [x for x in m if x[0] == x[1]]
    if final:
        step, _, loss, ms = final[-1]
        out = dict(steps=int(step), val_loss=float(loss), train_time_ms=float(ms))
    if (m := _ENDPOINT_KV.findall(text)):
        kv = dict(item.split("=", 1) for item in m[-1].split())
        out = dict(steps=int(kv["steps"]), val_loss=float(kv["val_loss"]), train_time_ms=float(kv["train_time_ms"]),
                   seed=None if kv.get("seed") in (None, "None") else int(kv["seed"]))
    elif (m := _ENDPOINT_JSON.findall(text)):
        j = json.loads(m[-1])
        out = dict(steps=j["steps"], val_loss=j["val_loss"], train_time_ms=j["train_time_ms"], seed=j.get("seed"))
    if out is not None and (p := _LOG_PATH.search(text)):
        out["log"] = p.group(1)
    return out


FULL_TRAIN_SHARDS = 103  # data/cached_fineweb10B.py's default: all of fineweb10B


def resolve_data_path(env: dict, repo: Path = REPO) -> str:
    """The DATA_PATH the run will use (config.py: DATA_PATH, default ".", resolved against the run's cwd, the
    variant folder). The shards live at the repository root, two levels up, so when DATA_PATH is unset
    and the variant folder holds no shards, ../.. is used instead of letting the run start without data."""
    if "DATA_PATH" in env:
        return env["DATA_PATH"]
    here = list((repo / "data" / "fineweb10B").glob("fineweb_train_*.bin"))
    root = list((repo / ".." / ".." / "data" / "fineweb10B").glob("fineweb_train_*.bin"))
    return "../.." if not here and root else "."


def check_shards(data_path: str, repo: Path = REPO) -> tuple[int, str | None]:
    """(train shard count, problem). The exact-count corpus is every train shard on disk that the loader
    cannot reach, so a partial download silently changes the hints, and with them the loss: anything but
    the full 103 shards is reported. No validation shard or no train shard is a hard problem."""
    base = (repo / data_path / "data" / "fineweb10B").resolve()
    n_train = len(list(base.glob("fineweb_train_*.bin")))
    if n_train == 0 or not list(base.glob("fineweb_val_*.bin")):
        return n_train, f"no FineWeb train/val shards under {base} (set DATA_PATH; README: Commands)"
    if n_train != FULL_TRAIN_SHARDS:
        return n_train, (f"{n_train} train shards under {base}, not {FULL_TRAIN_SHARDS}: the exact-count corpus, "
                         f"and so the val loss, will differ from @ldmberman's run")
    return n_train, None


def launch(scheduled_iterations: int, seed: int | None, out_file: Path, nproc: int = 8, extra_env: dict | None = None,
           dry_run: bool = False) -> dict | None:
    """One fresh torchrun of the Track 2 entry point (PR #380's chain is fitted at the endpoint, so every
    sweep point is its own run). stdout+stderr go to out_file; returns the parsed result."""
    env = dict(os.environ, NUM_SCHEDULED_ITERATIONS=str(scheduled_iterations), **(extra_env or {}))
    env["DATA_PATH"] = resolve_data_path(env)
    n_train, problem = check_shards(env["DATA_PATH"])
    if problem and not dry_run:
        if n_train == 0 or os.environ.get("ALLOW_PARTIAL_SHARDS", "0") != "1":
            raise SystemExit(f"[run] {problem}" + ("" if n_train == 0 else "; ALLOW_PARTIAL_SHARDS=1 to run anyway"))
        print(f"[run] WARNING: {problem}", file=sys.stderr)
    if seed is not None:
        env["TRAIN_SEED"] = str(seed)
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", ENTRY]
    shown = " ".join(f"{k}={env[k]}" for k in ("DATA_PATH", "NUM_SCHEDULED_ITERATIONS", "TRAIN_SEED", *(extra_env or {})) if k in env)
    print(f"[run] {shown} {' '.join(cmd)}  > {out_file}", flush=True)
    if dry_run:
        return None
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    text = out_file.read_text()
    result = parse_result(text)
    if proc.returncode != 0 or result is None:
        print(f"[run] FAILED (exit {proc.returncode}); see {out_file}", file=sys.stderr)
        return None
    result.update(scheduled_iterations=scheduled_iterations, seed=seed, out=str(out_file))
    return result


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------------------------------------
# Student t without scipy: the CDF via the regularized incomplete beta (Numerical Recipes' continued fraction)

def _betacf(a, b, x, iters=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c
            c = c if abs(c) > 1e-300 else 1e-300
            h *= d * c
        if abs(d * c - 1) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0 or x >= 1:
        return float(x >= 1)
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: int) -> float:
    tail = 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))
    return 1 - tail if t > 0 else tail


def t_ppf(q: float, df: int) -> float:
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return mu, sd


def one_sided_p(losses, target=TARGET) -> tuple[float, float]:
    """(t, p) for H1: mean < target, i.e. scipy.stats.ttest_1samp(losses, target, alternative="less")."""
    n = len(losses)
    mu, sd = mean_std(losses)
    t = (mu - target) / (sd / math.sqrt(n))
    return t, t_cdf(t, n - 1)


def required_mean(sd: float, n: int = 6, alpha: float = 0.01, target: float = TARGET) -> float:
    """The largest sample mean that still gives p < alpha with n runs at sample std sd."""
    return target + t_ppf(alpha, n - 1) * sd / math.sqrt(n)
