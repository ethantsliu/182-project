"""Parse modded-nanogpt run logs (Track 1 short, Track 2 medium, Track 3 optimization).

Every trainer in this repo prints the same validation line, in one of two time units:

    Track 1 / Track 2 (train_gpt.py, train_gpt_medium.py, every records/track_1_short and
    records/track_2_medium log):
        step:1194/1194 val_loss:3.2758 train_time:40020ms step_avg:33.52ms
    Track 3 (records/track_3_optimization/train_gpt_simple.py and its results/):
        step:3350/3350 val_loss:3.27839 train_time:492.485s step_avg:147.03ms

Training-progress lines drop the val_loss field. The log file starts with the trainer's own source
(which contains these f-strings, never matching lines with literal numbers), then a "=====" line.
Surveyed on the records/ tree at master: 18056 ms-format and 19321 s-format val lines, nothing else.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field, asdict

_NUM = r"(?:[0-9]+(?:\.[0-9]*)?|nan|inf)"
VAL_RE = re.compile(
    rf"^step:(\d+)/(\d+) val_loss:({_NUM}) train_time:([0-9.]+)(ms|s)(?: step_avg:([0-9.]+)ms)?",
    re.MULTILINE,
)
TRAIN_RE = re.compile(r"^step:(\d+)/(\d+) train_time:([0-9.]+)(ms|s)(?: step_avg:([0-9.]+)ms)?", re.MULTILINE)
PEAK_RE = re.compile(r"^peak memory allocated: (\d+) MiB reserved: (\d+) MiB", re.MULTILINE)
TORCH_RE = re.compile(r"^Running PyTorch (\S+)", re.MULTILINE)
GPU_RE = re.compile(r"NVIDIA (H100|A100|H200|B200|GH200)[^|\n]*", re.MULTILINE)
DRIVER_RE = re.compile(r"Driver Version: ([0-9.]+)")
LOGPATH_RE = re.compile(r"^(logs/[0-9A-Za-z_.\-]+\.txt)\s*$", re.MULTILINE)


def _seconds(value: str, unit: str) -> float:
    return float(value) / 1000.0 if unit == "ms" else float(value)


@dataclass
class RunLog:
    path: str | None
    fmt: str | None                # "ms" (Track 1/2) or "s" (Track 3); None if no val line
    total_steps: int | None
    final_step: int | None         # step of the last val line
    completed: bool                # the last val line is at step == total_steps
    val_loss: float | None         # val loss of the last val line
    train_time_s: float | None     # train_time of the last val line (the timed region)
    step_avg_ms: float | None
    last_train_step: int | None    # last progress line, for crashed runs
    val_curve: list = field(default_factory=list)  # [(step, val_loss, train_time_s)]
    peak_mem_mib: int | None = None
    torch_version: str | None = None
    gpu: str | None = None
    driver: str | None = None

    @property
    def is_nan(self) -> bool:
        return self.val_loss is not None and self.val_loss != self.val_loss

    def val_at(self, step: int) -> float | None:
        """Val loss printed at exactly `step` (Track 3 fixed-step readout), else None."""
        for s, v, _ in self.val_curve:
            if s == step:
                return v
        return None

    def first_step_below(self, target: float) -> int | None:
        for s, v, _ in self.val_curve:
            if v <= target:
                return s
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_nan"] = self.is_nan
        return d


def parse_text(text: str, path: str | None = None) -> RunLog:
    # Only look after the source dump when there is one: the "=====" separator line.
    sep = text.find("\n" + "=" * 100 + "\n")
    body = text[sep:] if sep >= 0 else text
    curve = []
    fmt = total = step_avg = None
    for m in VAL_RE.finditer(body):
        step, tot, val, t, unit, avg = m.groups()
        curve.append((int(step), float(val), _seconds(t, unit)))
        fmt, total, step_avg = unit, int(tot), (float(avg) if avg else None)
    last_train = None
    for m in TRAIN_RE.finditer(body):
        last_train = int(m.group(1))
        if total is None:
            total = int(m.group(2))
    peak = PEAK_RE.search(body)
    torch_v = TORCH_RE.search(body)
    gpu = GPU_RE.search(body)
    drv = DRIVER_RE.search(body)
    final = curve[-1] if curve else None
    return RunLog(
        path=path,
        fmt=fmt,
        total_steps=total,
        final_step=final[0] if final else None,
        completed=bool(final and total is not None and final[0] == total),
        val_loss=final[1] if final else None,
        train_time_s=final[2] if final else None,
        step_avg_ms=step_avg,
        last_train_step=last_train,
        val_curve=curve,
        peak_mem_mib=int(peak.group(1)) if peak else None,
        torch_version=torch_v.group(1) if torch_v else None,
        gpu=gpu.group(0).strip() if gpu else None,
        driver=drv.group(1) if drv else None,
    )


def parse_file(path: str) -> RunLog:
    with open(path, "r", errors="replace") as f:
        return parse_text(f.read(), path)


def find_log_path(stdout: str) -> str | None:
    """The trainers print their log path ("logs/<uuid>.txt") as the first stdout line on rank 0."""
    m = LOGPATH_RE.search(stdout)
    return m.group(1) if m else None


def expand(patterns: list[str]) -> list[str]:
    """Files, directories (all *.txt inside, non-recursive) and globs -> sorted run-log paths.

    Skips README/statistics/metadata files that are not run logs.
    """
    out = []
    for p in patterns:
        if os.path.isdir(p):
            cands = glob.glob(os.path.join(p, "*.txt"))
        else:
            cands = glob.glob(p) or ([p] if os.path.exists(p) else [])
        for c in sorted(cands):
            base = os.path.basename(c).lower()
            if base.startswith(("readme", "statistics", "metadata", "requirements", "stderr", "remote_manifest")):
                continue
            out.append(c)
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Print the parsed summary of run logs, one JSON per line.")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--curve", action="store_true", help="include the full val curve")
    a = ap.parse_args()
    for p in expand(a.logs):
        d = parse_file(p).to_dict()
        if not a.curve:
            d.pop("val_curve")
        print(json.dumps(d))
