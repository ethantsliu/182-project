#!/usr/bin/env python3
"""Two arms of root-trainer runs -> the comparison table of the course project, plus a CSV of the val curves.

    python 182/tools/compare_runs.py --a SRC [SRC ...] --b SRC [SRC ...] [--names base,kernel]
                                     [--target 3.28] [--csv curves.csv] [--json out.json]

SRC is a run log, a directory of *.txt logs, a glob, or DIR@ARM: the ok measured runs of arm ARM in an ab_run.py
results directory (plan.json + ledger.jsonl; the session's root A/Bs and `XAB_TREE=.` experiments). Timing runs
and curve runs (VAL_CURVE=N) may be mixed in one arm: a run is a curve run if it says so (METRIC run_config), carries
the INVALID_FOR_RECORD(val_curve) tag or has more than one validation; curve runs never enter the time,
latency and memory rows.

Rows, per arm: n, mean ± sd and the 95% t-interval of the mean for train time, step latency (mean / median / p95),
final val loss and peak memory; hits of the target. Between arms (B - A): paired by seed (or by ab_run rep) when
the keys match one-to-one, with the 95% t-interval of the mean difference, else the Welch interval.
Time to target: in each curve run, the first validation at or under the target, linearly interpolated from the
validation before it (in loss, over both wall-clock and steps). Two caveats the output repeats: a curve run's clock
carries each validation's on-clock part, so the crossing is also given rescaled to the arm's mean timing-run time;
and the final validation evaluates the tail-averaged weights with the window extension and canonical mask, which
no intermediate validation does, so a crossing inside the last interval interpolates across that jump.

Sources of each number: METRIC lines (RUN_METRICS=1 / VAL_CURVE=N, track_1_short/run_metrics.py) when the log has
them; otherwise the val line (loss to 4 digits), the 25-step progress lines (latency per window, marked
"windows") and the legacy rank-0 whole-process peak-memory line (marked "rank0"). Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logparse  # noqa: E402
import stats  # noqa: E402
import summarize_session as ss  # noqa: E402

NAN = float("nan")


# ----------------------------------------------------------------------------- one run

def _f(d: dict, k: str):
    try:
        return float(d[k])
    except (KeyError, ValueError):
        return None


def _pct(xs, q):
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def window_latency(text: str) -> dict | None:
    """Step latency from the progress lines alone: ms/step of each window between consecutive lines."""
    pts = [p for p in ss.progress_points(text) if p[0] > 0]
    wins = [(t1 - t0) * 1000 / (s1 - s0) for (s0, t0), (s1, t1) in zip(pts, pts[1:]) if s1 > s0]
    if not wins:
        return None
    (s0, t0), (s1, t1) = pts[0], pts[-1]
    return dict(mean_ms=(t1 - t0) * 1000 / (s1 - s0), median_ms=_pct(wins, 0.5), p95_ms=_pct(wins, 0.95), src="windows")


def parse_run(text: str, path: str | None = None, key: str | None = None) -> dict:
    """One run -> dict(path, key, seed, kind, completed, steps, train_time_s, val_loss, curve, lat, mem, ...)."""
    log = logparse.parse_text(text, path)
    metrics = logparse.parse_metrics(text)
    cfg = next((d for n, d in metrics if n == "run_config"), {})
    seed = logparse.parse_seed(text) or (cfg.get("seed") if cfg.get("seed") not in (None, "random") else None)
    vals = [d for n, d in metrics if n == "val"]
    if vals:  # 6-digit losses
        curve = [(int(d["step"]), float(d["train_time_s"]), float(d["val_loss"])) for d in vals]
    else:
        curve = [(s, t, v) for s, v, t in log.val_curve]
    final = curve[-1] if curve and log.completed else None
    is_curve = (int(cfg.get("VAL_CURVE", "0") or 0) > 0 or "INVALID_FOR_RECORD(val_curve)" in logparse._body(text)
                or len(curve) > 1)
    lat = next((dict(mean_ms=_f(d, "mean_ms"), median_ms=_f(d, "median_ms"), p95_ms=_f(d, "p95_ms"), src="steps")
                for n, d in metrics if n == "step_latency" and d.get("scope") == "all"), None)
    stage_lat = {d["scope"]: _f(d, "median_ms") for n, d in metrics
                 if n == "step_latency" and d.get("scope", "").startswith("stage")}
    peak = next((d for n, d in metrics if n == "peak_mem" and d.get("rank") == "max"), None)
    if peak:
        mem = dict(allocated_mib=_f(peak, "allocated_mib"), reserved_mib=_f(peak, "reserved_mib"),
                   warmup_allocated_mib=_f(peak, "warmup_allocated_mib"), src="all-ranks")
    elif log.peak_mem_mib is not None:
        mem = dict(allocated_mib=float(log.peak_mem_mib), reserved_mib=None, warmup_allocated_mib=None, src="rank0")
    else:
        mem = None
    return dict(path=path, seed=seed, key=key if key is not None else (f"seed{seed}" if seed is not None else None),
                kind="curve" if is_curve else "timing", completed=log.completed, steps=log.final_step,
                train_time_s=final[1] if final else None, val_loss=final[2] if final else None,
                curve=curve, lat=lat or window_latency(text), stage_lat=stage_lat, mem=mem)


def load_source(src: str) -> list[dict]:
    """A SRC argument -> parsed runs (see the module docstring)."""
    d, at, arm = src.rpartition("@")
    if at and os.path.isfile(os.path.join(d, "plan.json")):
        ab = ss.load_ab(d)
        if arm not in ab["groups"]:
            raise SystemExit(f"{src}: no arm {arm!r} in {d} (arms: {', '.join(ab['arms'])})")
        runs = []
        for r in ab["groups"][arm]:
            p = ss._remap(r.get("log"), d, "logs") or ss._remap(r.get("stdout"), d, "stdout")
            if r.get("status") != "ok" or not p:
                continue
            with open(p, errors="replace") as f:
                runs.append(parse_run(f.read(), p, key=f"seed{r['seed']}" if r.get("seed") is not None else f"rep{r['rep']}"))
        return runs
    paths = logparse.expand([src])
    if not paths:
        raise SystemExit(f"{src}: no run logs (a log file, a directory of *.txt, a glob, or ABDIR@ARM)")
    runs = []
    for p in paths:
        with open(p, errors="replace") as f:
            runs.append(parse_run(f.read(), p))
    return runs


# ----------------------------------------------------------------------------- statistics

def summary(xs) -> dict:
    xs = [x for x in xs if x is not None and x == x]
    m, se, lo, hi, n = ss.ci95(xs)
    return dict(n=n, mean=m, sd=stats.mean_sd(xs)[1], lo=lo, hi=hi)


def crossing(curve, target: float) -> dict | None:
    """First validation at or under target, interpolated from the one before: dict(time_s, step, last_interval,
    censored). None if the run never gets there. censored: the very first validation was already under, so the
    crossing is only known to be earlier than it."""
    for i, (step, t, v) in enumerate(curve):
        if v <= target:
            if i == 0:
                return dict(time_s=t, step=float(step), last_interval=len(curve) == 1, censored=True)
            s0, t0, v0 = curve[i - 1]
            w = (v0 - target) / (v0 - v) if v0 != v else 1.0
            return dict(time_s=t0 + w * (t - t0), step=s0 + w * (step - s0), last_interval=i == len(curve) - 1,
                        censored=False)
    return None


def arm_stats(runs: list[dict], target: float) -> dict:
    timing = [r for r in runs if r["kind"] == "timing" and r["completed"]]
    curves = [r for r in runs if r["kind"] == "curve" and r["completed"]]
    lat = [r["lat"] for r in timing if r["lat"]]
    mem = [r["mem"] for r in timing if r["mem"]]
    time_mean = summary([r["train_time_s"] for r in timing])["mean"]
    cross, censored = [], 0
    for r in curves:
        c = crossing(r["curve"], target)
        if c is not None and c["censored"]:
            censored += 1
        elif c is not None:
            scale = time_mean / r["train_time_s"] if time_mean == time_mean and r["train_time_s"] else NAN
            cross.append(dict(c, rescaled_s=c["time_s"] * scale, path=r["path"]))
    return dict(
        n_runs=len(runs), n_incomplete=sum(1 for r in runs if not r["completed"]), n_timing=len(timing), n_curve=len(curves),
        time=summary([r["train_time_s"] for r in timing]),
        val=summary([r["val_loss"] for r in timing]),
        hits=sum(1 for r in timing if r["val_loss"] is not None and r["val_loss"] <= target),
        steps=sorted({r["steps"] for r in timing}),
        lat={k: summary([x[k] for x in lat]) for k in ("mean_ms", "median_ms", "p95_ms")},
        lat_src=sorted({x["src"] for x in lat}),
        stage_lat={k: summary([r["stage_lat"].get(k) for r in timing])
                   for k in sorted({k for r in timing for k in r["stage_lat"]})},
        mem={k: summary([x[k] for x in mem]) for k in ("allocated_mib", "reserved_mib", "warmup_allocated_mib")},
        mem_src=sorted({x["src"] for x in mem}),
        curve_val=summary([r["val_loss"] for r in curves]),
        curve_time=summary([r["train_time_s"] for r in curves]),
        cross_n=len(cross), cross_last=sum(1 for c in cross if c["last_interval"]), cross_censored=censored,
        cross_time=summary([c["time_s"] for c in cross]), cross_rescaled=summary([c["rescaled_s"] for c in cross]),
        cross_step=summary([c["step"] for c in cross]),
    )


def difference(a_runs: list[dict], b_runs: list[dict], field) -> dict:
    """B - A of field(run) over completed timing runs: paired when both arms' keys are unique and overlap."""
    a = [r for r in a_runs if r["kind"] == "timing" and r["completed"] and field(r) is not None]
    b = [r for r in b_runs if r["kind"] == "timing" and r["completed"] and field(r) is not None]
    ka, kb = [r["key"] for r in a], [r["key"] for r in b]
    if None not in ka and None not in kb and len(set(ka)) == len(ka) and len(set(kb)) == len(kb) and set(ka) & set(kb):
        by = {r["key"]: r for r in a}
        diffs = [field(r) - field(by[r["key"]]) for r in b if r["key"] in by]
        m, se, lo, hi, n = ss.ci95(diffs)
        return dict(kind="paired", n=n, mean=m, lo=lo, hi=hi, unpaired=len(a) + len(b) - 2 * n)
    if not a or not b:
        return dict(kind="welch", n=0, mean=NAN, lo=NAN, hi=NAN, unpaired=0)
    w = stats.welch_test([field(r) for r in b], [field(r) for r in a], alternative="two-sided")
    lo, hi = w.get("ci95", (NAN, NAN))
    return dict(kind="welch", n=min(len(a), len(b)), mean=w["delta"], lo=lo, hi=hi, unpaired=0)


def compare(a_runs, b_runs, target: float = 3.28) -> dict:
    def lat_mean(r):
        return r["lat"]["mean_ms"] if r["lat"] else None

    def mem_alloc(r):
        return r["mem"]["allocated_mib"] if r["mem"] else None

    return dict(target=target, a=arm_stats(a_runs, target), b=arm_stats(b_runs, target),
                d_time=difference(a_runs, b_runs, lambda r: r["train_time_s"]),
                d_val=difference(a_runs, b_runs, lambda r: r["val_loss"]),
                d_lat=difference(a_runs, b_runs, lat_mean), d_mem=difference(a_runs, b_runs, mem_alloc))


# ----------------------------------------------------------------------------- output

def _cell(s: dict, fmt: str) -> str:
    if not s["n"]:
        return "-"
    out = fmt.format(s["mean"])
    if s["n"] >= 2:
        out += f" ± {fmt.format(s['sd'])} [{fmt.format(s['lo'])}, {fmt.format(s['hi'])}]"
    return out + f" (n={s['n']})"


def _diff(d: dict, fmt: str) -> str:
    if d["mean"] != d["mean"]:
        return "-"
    sign = "{:+" + fmt[2:]
    out = sign.format(d["mean"])
    if d["lo"] == d["lo"]:
        out += f" [{sign.format(d['lo'])}, {sign.format(d['hi'])}]"
    return out + f" ({d['kind']}, n={d['n']}" + (f"; {d['unpaired']} runs without a partner" if d["unpaired"] else "") + ")"


def table(C: dict, names=("A", "B")) -> list[str]:
    A, B, tgt = C["a"], C["b"], C["target"]

    def src(k):
        return "/".join(sorted(set(A[k]) | set(B[k]))) or "-"

    def both(k, fmt, sub=None):
        return [_cell(x[k][sub] if sub else x[k], fmt) for x in (A, B)]

    rows = [
        ("runs: timing / curve / incomplete", *[f"{x['n_timing']} / {x['n_curve']} / {x['n_incomplete']}" for x in (A, B)], ""),
        ("steps", *[",".join(map(str, x["steps"])) or "-" for x in (A, B)], ""),
        ("train time (s)", *both("time", "{:.3f}"), _diff(C["d_time"], "{:.3f}")),
        (f"step latency mean (ms) [{src('lat_src')}]", *both("lat", "{:.3f}", "mean_ms"), _diff(C["d_lat"], "{:.3f}")),
        ("step latency median (ms)", *both("lat", "{:.3f}", "median_ms"), ""),
        ("step latency p95 (ms)", *both("lat", "{:.3f}", "p95_ms"), ""),
    ]
    for k in sorted(set(A["stage_lat"]) | set(B["stage_lat"])):
        rows.append((f"  {k} median (ms)", *[_cell(x["stage_lat"][k], "{:.3f}") if k in x["stage_lat"] else "-" for x in (A, B)], ""))
    rows += [
        ("final val loss", *both("val", "{:.5f}"), _diff(C["d_val"], "{:.5f}")),
        (f"runs at or under {tgt:.2f}", f"{A['hits']}/{A['n_timing']}", f"{B['hits']}/{B['n_timing']}", ""),
        (f"peak allocated, max rank (MiB) [{src('mem_src')}]", *both("mem", "{:.0f}", "allocated_mib"), _diff(C["d_mem"], "{:.0f}")),
        ("peak reserved, max rank (MiB)", *both("mem", "{:.0f}", "reserved_mib"), ""),
        ("peak allocated before the clock (MiB)", *both("mem", "{:.0f}", "warmup_allocated_mib"), ""),
        ("curve runs: final val loss", *both("curve_val", "{:.5f}"), ""),
        ("curve runs: train time incl. validation overhead (s)", *both("curve_time", "{:.3f}"), ""),
        (f"time to {tgt:.2f}, curve clock (s)", *both("cross_time", "{:.3f}"), ""),
        (f"time to {tgt:.2f}, rescaled to timing runs (s)", *both("cross_rescaled", "{:.3f}"), ""),
        (f"steps to {tgt:.2f}", *both("cross_step", "{:.1f}"), ""),
        ("  crossings: found / curve runs / in the last interval",
         *[f"{x['cross_n']} / {x['n_curve']} / {x['cross_last']}" for x in (A, B)], ""),
    ]
    head = ("metric", f"A: {names[0]}", f"B: {names[1]}", "B - A")
    w = [max(len(r[i]) for r in [head, *rows]) for i in range(4)]

    def line(r):
        return "  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip()

    out = [line(head), line(tuple("-" * x for x in w)), *map(line, rows), "",
           "Cells: mean ± sd [95% t-interval of the mean] (n). B - A: mean difference [95% interval]; paired by seed/rep "
           "when the keys match, else Welch. Negative time = B faster.",
           "Timing rows use timing runs only. In a timing run the time to the target is the train time if the final "
           f"val loss is at or under {tgt:.2f} (the 'runs at or under' row)."]
    if A["n_curve"] or B["n_curve"]:
        out.append("Time to target is interpolated between validations of curve runs (VAL_CURVE=N): their clock includes each "
                   "validation's on-clock reads/pulls (the rescaled row scales it by timing-run time / curve-run time), and "
                   "the final validation alone uses the tail-averaged weights, window extension and canonical mask, so a "
                   "crossing in the last interval interpolates across that jump.")
    if A["cross_censored"] or B["cross_censored"]:
        out.append(f"Left out of the time-to-target rows: {A['cross_censored']} (A) / {B['cross_censored']} (B) curve runs whose "
                   "first validation was already at or under the target (use a smaller VAL_CURVE).")
    if "windows" in src("lat_src"):
        out.append("Latency marked 'windows' comes from the 25-step progress lines (no per-step data; p95 is over windows): "
                   "run with RUN_METRICS=1 for per-step values.")
    if "rank0" in src("mem_src"):
        out.append("Memory marked 'rank0' is the legacy line: rank 0 only, whole process (warmup included).")
    return out


def write_csv(path: str, arms: dict) -> int:
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "run", "seed", "kind", "step", "train_time_s", "val_loss"])
        for arm, runs in arms.items():
            for r in runs:
                for step, t, v in r["curve"]:
                    w.writerow([arm, os.path.basename(r["path"] or ""), r["seed"] or "", r["kind"], step, f"{t:.4f}", f"{v:.6f}"])
                    n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True, metavar="SRC", help="arm A (baseline): logs, directories, globs or ABDIR@ARM")
    ap.add_argument("--b", nargs="+", required=True, metavar="SRC", help="arm B")
    ap.add_argument("--names", default="A,B", help="two labels, comma-separated")
    ap.add_argument("--target", type=float, default=3.28)
    ap.add_argument("--csv", default="compare_runs_curves.csv", help="val curves of every run (default %(default)s; '' to skip)")
    ap.add_argument("--json", help="also write the statistics as JSON")
    a = ap.parse_args(argv)
    names = tuple(a.names.split(","))
    if len(names) != 2:
        ap.error("--names takes two labels")
    a_runs = [r for s in a.a for r in load_source(s)]
    b_runs = [r for s in a.b for r in load_source(s)]
    C = compare(a_runs, b_runs, a.target)
    print("\n".join(table(C, names)))
    if a.csv:
        n = write_csv(a.csv, {names[0]: a_runs, names[1]: b_runs})
        print(f"curves: {n} rows -> {a.csv}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(C, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
