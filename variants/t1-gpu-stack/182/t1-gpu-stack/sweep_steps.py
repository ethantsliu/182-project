"""Step-count sweep: find the NUM_SCHEDULED_ITERATIONS whose mean final val loss sits at a target (default 3.2785).

Run on the 8xH100 box from the repo root. Every run is a plain `torchrun ... train_gpt.py` with the current
environment (so EXACT_LEVERS / HOST_PREP / DOC_COPY / LAYER_ROPE / COPY_MIX pass through) plus
NUM_SCHEDULED_ITERATIONS; the console goes to <out>/<tag>_s<steps>_r<k>.txt, and the analysis reads those files.

  # 3 step counts x 4 runs, interleaved (one run of each count, then the next round), then the fit:
  python 182/t1-gpu-stack/sweep_steps.py run --steps 951 966 981 --runs 4 --out sweeps/stack --tag stack
  # Analysis only (also works on logs from any earlier sweep or record folder with [final] / val_loss lines):
  python 182/t1-gpu-stack/sweep_steps.py fit sweeps/stack --target 3.2785

The fit: mean val loss is close to linear in the step count over a few dozen steps, so it fits
loss = a + b * scheduled_steps by least squares on all runs (not on the group means), solves for the target,
and reports the residual sd, each count's mean and one-sided p (mean < 3.28), the predicted train time at the
suggested count (linear in steps too), and how many runs a record claim at that count would need. A suggestion
outside the swept range is an extrapolation and says so.
"""
import argparse
import glob
import math
import os
import re
import statistics
import subprocess
import sys

FINAL_RE = re.compile(r"\[final\] val_loss:([0-9.]+) train_time_ms:([0-9]+) steps:([0-9]+) scheduled:([0-9]+)")
# Logs without the [final] line (other branches): the last step:N/N val_loss line and the [config] steps line.
STEP_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+) train_time:([0-9]+)ms")
CONFIG_RE = re.compile(r"\[config\] steps (\d+) scheduled")
RECORD_BAR = 3.28


def parse_log(path: str):
    """(scheduled steps, val loss, train time s) of one finished run, or None."""
    text = open(path, errors="replace").read()
    m = FINAL_RE.findall(text)
    if m:
        loss, ms, _, sched = m[-1]
        return int(sched), float(loss), int(ms) / 1000
    steps = [s for s in STEP_RE.findall(text) if s[0] == s[1]]
    cfg = CONFIG_RE.findall(text)
    if steps and cfg:
        _, _, loss, ms = steps[-1]
        return int(cfg[-1]), float(loss), int(ms) / 1000
    return None


def one_sided_p_below(values, bar=RECORD_BAR):
    """p-value of H0: mean >= bar against mean < bar (one-sample t; normal tail if scipy is missing)."""
    n = len(values)
    if n < 2:
        return float("nan")
    t = (statistics.mean(values) - bar) / (statistics.stdev(values) / math.sqrt(n))
    try:
        from scipy import stats
        return float(stats.t.cdf(t, df=n - 1))
    except ImportError:
        return 0.5 * math.erfc(-t / math.sqrt(2))


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    return a, b, sd


def fit(log_dirs, target: float):
    runs = []
    for d in log_dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)):
            r = parse_log(path)
            if r is None:
                print(f"skip (unfinished or unparsed): {path}")
            else:
                runs.append(r)
    if not runs:
        sys.exit("no finished runs found")
    by_steps = {}
    for sched, loss, secs in runs:
        by_steps.setdefault(sched, []).append((loss, secs))
    print(f"{'scheduled':>9} {'n':>3} {'mean val':>10} {'sd':>8} {'p(<3.28)':>9} {'mean time s':>11}")
    for sched in sorted(by_steps):
        losses = [l for l, _ in by_steps[sched]]
        secs = [s for _, s in by_steps[sched]]
        sd = statistics.stdev(losses) if len(losses) > 1 else float("nan")
        print(f"{sched:>9} {len(losses):>3} {statistics.mean(losses):>10.5f} {sd:>8.5f} "
              f"{one_sided_p_below(losses):>9.2g} {statistics.mean(secs):>11.3f}")
    if len(by_steps) < 2:
        print("one step count only: no slope; sweep at least two counts")
        return
    xs = [r[0] for r in runs]
    a, b, sd = linfit(xs, [r[1] for r in runs])
    ta, tb, tsd = linfit(xs, [r[2] for r in runs])
    print(f"\nfit: val = {a:.5f} + ({b * 1e3:.4f}e-3) * scheduled   (residual sd {sd:.5f}; {b * 10:+.5f} per 10 steps)")
    print(f"fit: time = {ta:.3f} + {tb * 1e3:.3f} ms * scheduled   (residual sd {tsd:.3f} s)")
    if b >= 0:
        print("the loss does not fall with steps over this range: widen the sweep or add runs")
        return
    x_star = (target - a) / b
    suggested = math.ceil(x_star)
    lo, hi = min(xs), max(xs)
    where = "inside" if lo <= x_star <= hi else "OUTSIDE (extrapolated)"
    print(f"\ntarget mean {target}: NUM_SCHEDULED_ITERATIONS = {x_star:.1f} -> {suggested}, {where} the swept {lo}-{hi}")
    print(f"predicted at {suggested}: val {a + b * suggested:.5f}, train time {ta + tb * suggested:.3f} s")
    # Runs a record claim at the target needs for p < 0.01 against 3.28 (one-sided, z approximation).
    margin = RECORD_BAR - target
    if margin > 0 and sd > 0:
        need = math.ceil(((2.326 * sd) / margin) ** 2)
        print(f"with run-to-run sd {sd:.5f}, p < 0.01 against {RECORD_BAR} needs about {max(need, 3)} runs at that count")


def run(args):
    os.makedirs(args.out, exist_ok=True)
    env_extra = {k: os.environ[k] for k in ("EXACT_LEVERS", "HOST_PREP", "DOC_COPY", "LAYER_ROPE", "COPY_MIX",
                                            "COPY_MIX_FIT", "DATA_PATH") if k in os.environ}
    print(f"sweep {args.steps} x {args.runs} runs, env {env_extra or '(defaults)'} -> {args.out}")
    for k in range(args.runs):
        for steps in args.steps:  # interleaved: drift on the node hits every count alike
            path = os.path.join(args.out, f"{args.tag}_s{steps}_r{k}.txt")
            if os.path.exists(path) and parse_log(path) is not None:
                print(f"have {path}")
                continue
            env = dict(os.environ, NUM_SCHEDULED_ITERATIONS=str(steps))
            cmd = ["torchrun", "--standalone", f"--nproc_per_node={args.nproc}", "train_gpt.py"]
            print(f"run {path}: NUM_SCHEDULED_ITERATIONS={steps} {' '.join(cmd)}", flush=True)
            with open(path, "w") as f:
                code = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            r = parse_log(path)
            print(f"  exit {code}: {r if r else 'no final validation in the log'}", flush=True)
    fit([args.out], args.target)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--steps", type=int, nargs="+", required=True, help="NUM_SCHEDULED_ITERATIONS values")
    r.add_argument("--runs", type=int, default=3, help="runs per step count")
    r.add_argument("--out", required=True)
    r.add_argument("--tag", default="sweep")
    r.add_argument("--nproc", type=int, default=8)
    r.add_argument("--target", type=float, default=3.2785)
    f = sub.add_parser("fit")
    f.add_argument("dirs", nargs="+")
    f.add_argument("--target", type=float, default=3.2785)
    args = p.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        fit(args.dirs, args.target)


if __name__ == "__main__":
    main()
