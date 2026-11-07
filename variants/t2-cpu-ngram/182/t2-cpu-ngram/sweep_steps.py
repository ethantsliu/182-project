"""Step-count sweep for the Track 2 CPU-retrieval run: cut NUM_SCHEDULED_ITERATIONS until the mean lands near 2.918.

@ldmberman's 5790 + 10 steps reached 2.8769, ~43 millinats under the 2.92 target. Every point is a fresh
torchrun of train_gpt_medium_cpu.py (the stage boundaries, sampled-softmax switches, lr cooldown and the
exact-count corpus split all follow NUM_SCHEDULED_ITERATIONS). Results append to a JSONL, so the sweep
can be stopped, resumed and refitted. The fit is val_loss = a + b * ln(total_steps) by least squares
over every run (seeds pooled), solved for the target mean.

The target mean: with n significance runs at per-seed std s, p < 0.01 needs mean <= 2.92 + t_0.01,n-1 * s/sqrt(n)
(2.92 - 1.374 s for n = 6). --target defaults to that at the measured std when there are repeats at
one step count, else to 2.918 (s ~ 0.0014, Track 1's PR #380 spread).

  python 182/t2-cpu-ngram/sweep_steps.py --steps 4000 4600 5200 --seeds 0      # coarse pass, 3 runs
  python 182/t2-cpu-ngram/sweep_steps.py --steps 4700 4800 --seeds 0 1           # refine near the solution
  python 182/t2-cpu-ngram/sweep_steps.py --fit-only                              # refit what is in the JSONL
  python 182/t2-cpu-ngram/sweep_steps.py --ingest logs/*.txt                     # add finished run logs
"""
import argparse
import math
import sys
from pathlib import Path

from runlib import TARGET, append_jsonl, launch, load_jsonl, mean_std, parse_result, required_mean

HERE = Path(__file__).resolve().parent
EXTENSION_STEPS = 10  # config.Hyperparameters.num_extension_iterations
# @ldmberman's run (PR #380 thread, 2026-10-04), the one point known before any sweep. Not our measurement.
SEED_POINT = dict(scheduled_iterations=5790, steps=5800, seed=1337, val_loss=2.876861, train_time_ms=221003,
                  out="ldmberman pr380-adapted-to-track-2.txt")


def fit(rows):
    xs = [math.log(r["steps"]) for r in rows]
    ys = [r["val_loss"] for r in rows]
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    return a, b, rmse


def pooled_seed_std(rows):
    """Pooled within-step-count std over step counts with >= 2 runs (None if there are none)."""
    groups = {}
    for r in rows:
        groups.setdefault(r["steps"], []).append(r["val_loss"])
    ss, dof = 0.0, 0
    for g in groups.values():
        if len(g) >= 2:
            mu = sum(g) / len(g)
            ss += sum((x - mu) ** 2 for x in g)
            dof += len(g) - 1
    return math.sqrt(ss / dof) if dof else None


def report(rows, target_arg, n_sig, include_seed_point):
    data = rows + ([SEED_POINT] if include_seed_point else [])
    print(f"\n{'steps':>6} {'seed':>6} {'val_loss':>9} {'train_s':>8}  source")
    for r in sorted(data, key=lambda r: (r["steps"], str(r.get("seed")))):
        print(f"{r['steps']:>6} {str(r.get('seed')):>6} {r['val_loss']:>9.4f} {r['train_time_ms'] / 1000:>8.1f}  {r.get('out', '')}")
    sd = pooled_seed_std(data)
    target = target_arg if target_arg is not None else (required_mean(sd, n_sig) if sd else 2.918)
    print(f"\nper-seed std: {'%.5f (pooled)' % sd if sd else 'unknown (no repeats yet), assuming ~0.0014'}; "
          f"target mean {target:.4f} (p<0.01 with n={n_sig} needs mean <= {TARGET} + t*s/sqrt(n))")
    f = fit(data)
    if f is None:
        print("need runs at >= 2 step counts to fit")
        return
    a, b, rmse = f
    total = math.exp((target - a) / b)
    sched = math.ceil(total - EXTENSION_STEPS)
    ms_per_step = sum(r["train_time_ms"] / r["steps"] for r in data) / len(data)
    print(f"fit: val_loss = {a:.4f} + {b:.4f} ln(steps), rmse {rmse:.4f} over {len(data)} runs")
    print(f"-> NUM_SCHEDULED_ITERATIONS ~ {sched} (total {sched + EXTENSION_STEPS}); predicted train time "
          f"~{(sched + EXTENSION_STEPS) * ms_per_step / 1000:.0f} s at {ms_per_step:.2f} ms/step")
    lo, hi = [math.ceil(math.exp((target - a + k * rmse) / b) - EXTENSION_STEPS) for k in (1, -1)]
    print(f"   (+-1 rmse in loss: {min(lo, hi)}..{max(lo, hi)}). Confirm with 2 seeds there, then significance.py.")
    if b >= 0:
        print("   WARNING: loss does not fall with steps in this range; the fit is meaningless")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, nargs="*", default=[], help="NUM_SCHEDULED_ITERATIONS values to run")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0], help="TRAIN_SEED values per step count")
    ap.add_argument("--results", type=Path, default=HERE / "results" / "sweep.jsonl")
    ap.add_argument("--target", type=float, default=None, help="target mean val loss (default: from the seed std)")
    ap.add_argument("--n-significance", type=int, default=6)
    ap.add_argument("--no-seed-point", action="store_true", help="leave @ldmberman's 5800-step point out of the fit")
    ap.add_argument("--ingest", type=Path, nargs="*", default=[], help="finished run logs/stdout files to add")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the torchrun commands only")
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()

    rows = load_jsonl(a.results)
    for path in a.ingest:
        r = parse_result(path.read_text())
        if r is None:
            print(f"[ingest] no final val_loss in {path}", file=sys.stderr)
            continue
        r.update(scheduled_iterations=r["steps"] - EXTENSION_STEPS, out=str(path))
        rows.append(r)
        append_jsonl(a.results, r)
    if not a.fit_only:
        done = {(r["scheduled_iterations"], r.get("seed")) for r in rows}
        for s in a.steps:
            for seed in a.seeds:
                if (s, seed) in done:
                    print(f"[skip] {s} seed {seed}: already in {a.results}")
                    continue
                r = launch(s, seed, HERE / "results" / f"steps{s}_seed{seed}.out", nproc=a.nproc, dry_run=a.dry_run)
                if r is not None:
                    rows.append(r)
                    append_jsonl(a.results, r)
                    print(f"[done] steps={r['steps']} val_loss={r['val_loss']:.4f} train_time={r['train_time_ms'] / 1000:.1f}s")
    report(rows, a.target, a.n_significance, include_seed_point=not a.no_seed_point)


if __name__ == "__main__":
    main()
