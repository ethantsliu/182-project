"""Shared helpers for node.py, sweep_steps.py and significance.py: parse a run's output, the data preflight, one-sided t-test.

Standard library only (scipy is used to cross-check the t-test when it is installed).
"""
import json
import math
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = 2.92

_ENDPOINT_KV = re.compile(r"^endpoint_result: (.*)$", re.M)
_ENDPOINT_JSON = re.compile(r"^endpoint_result:(\{.*\})$", re.M)  # ldmberman's attachment format
_VAL_LINE = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms", re.M)
LOG_PATH = re.compile(r"^(logs/[0-9a-f-]+\.txt)$", re.M)
_STEP = re.compile(r"^step:(\d+)/(\d+) ", re.M)
_PEAK = re.compile(r"^peak memory allocated: (\d+) MiB reserved: (\d+) MiB", re.M)
_HOST = re.compile(r"^host memory: rank 0 peak RSS ([\d.]+) GiB, node in use ([\d.]+) GiB", re.M)
_PARTS = re.compile(r"^val_loss_parts: model_only=([\d.]+)", re.M)
_BUILT = re.compile(r"^validation index built in ([\d.]+) s", re.M)


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
    if out is not None and not math.isfinite(out["val_loss"]):
        return None  # a diverged run has no result
    if out is not None and (p := LOG_PATH.search(text)):
        out["log"] = p.group(1)
    return out


FULL_TRAIN_SHARDS = 103  # data/cached_fineweb10B.py's default: all of fineweb10B


def resolve_data_path(env: dict, repo: Path = REPO) -> str:
    """The DATA_PATH the run will use (config.py resolves it against the run's cwd, the variant folder). Unset,
    it is ../..: the shards live at the repository root, two levels up, where run_node.sh prep downloads them.
    Only a variant folder that holds shards of its own keeps config.py's default, "."."""
    if "DATA_PATH" in env:
        return env["DATA_PATH"]
    return "." if list((repo / "data" / "fineweb10B").glob("fineweb_train_*.bin")) else "../.."


def check_shards(data_path: str, repo: Path = REPO) -> tuple[int, str | None]:
    """(train shard count, problem). The exact-count corpus is every train shard on disk that the loader
    cannot reach, so a partial download silently changes the hints, and with them the loss: anything but
    the full 103 shards is reported. No validation shard or no train shard is a hard problem."""
    base = (repo / data_path / "data" / "fineweb10B").resolve()
    n_train = len(list(base.glob("fineweb_train_*.bin")))
    if n_train == 0 or not list(base.glob("fineweb_val_*.bin")):
        return n_train, f"no FineWeb train/val shards under {base} (bash 182/run_node.sh prep downloads them to the repository root)"
    if n_train != FULL_TRAIN_SHARDS:
        return n_train, (f"{n_train} train shards under {base}, not {FULL_TRAIN_SHARDS}: the exact-count corpus, "
                         f"and so the val loss, will differ from ldmberman's run")
    return n_train, None


def parse_details(text: str) -> dict:
    """The rest of what one run's output says: memory peaks, the loss before the chain, index build times."""
    out = {}
    if (m := _PEAK.search(text)):
        out.update(peak_alloc_mib=int(m.group(1)), peak_reserved_mib=int(m.group(2)))
    if (m := _HOST.search(text)):
        out.update(rank0_rss_gib=float(m.group(1)), host_used_gib=float(m.group(2)))
    if (m := _PARTS.search(text)):
        out["model_only"] = float(m.group(1))
    builds = [float(x) for x in _BUILT.findall(text)]
    if builds:
        out["index_build_s"] = builds
    return out


_ERROR = re.compile(r"(Error|Exception|HOST CHECK|wrong stack|out of memory|Killed|Segmentation|SIGBUS|SIGKILL|Signal \d+|exitcode)", re.I)


def failure_reason(text: str, returncode: int) -> str:
    """One line for a failed run: its last error-looking output line (torchrun's own wrap-up lines aside)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        if _ERROR.search(line) and "ChildFailedError" not in line and not line.startswith(("Traceback", "File ", "raise ")):
            return f"exit {returncode}: {line[:240]}"
    last = runlib_last_step(text)
    return f"exit {returncode}, no error line" + (f" (last step line: {last})" if last else "")


def runlib_last_step(text: str) -> str | None:
    steps = _STEP.findall(text)
    return f"{steps[-1][0]}/{steps[-1][1]}" if steps else None


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
