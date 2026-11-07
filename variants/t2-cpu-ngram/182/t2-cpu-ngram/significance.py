"""Significance runs for a Track 2 record: n fresh runs at one step count, one-sided t-test of mean val loss < 2.92.

The README's rule: enough runs for p < 0.01 that the mean val loss is <= 2.92, tested as in the Track 1
records' statistics.md (scipy.stats.ttest_1samp(losses, 2.92, alternative="less")). Prints a
statistics.md-style table; exit status 0 only if p < 0.01.

  python 182/t2-cpu-ngram/significance.py --steps 4750 --runs 6                    # launch 6 runs
  COORD_DESCENT_TUNING=1 python 182/t2-cpu-ngram/significance.py --steps 4750 --runs 6   # final record runs
  python 182/t2-cpu-ngram/significance.py --logs logs/*.txt                       # test finished logs only

Seeds: --seeds, else unseeded (random init, as in a record run; each log then records seed=random).
"""
import argparse
import sys
from pathlib import Path

from runlib import TARGET, launch, mean_std, one_sided_p, parse_result

HERE = Path(__file__).resolve().parent


def table(rows, target, alpha):
    losses = [r["val_loss"] for r in rows]
    times = [r["train_time_ms"] for r in rows]
    n = len(rows)
    mu, sd = mean_std(losses)
    tmu, tsd = mean_std(times)
    t, p = one_sided_p(losses, target)
    lines = ["# Run statistics", "", f"{n} runs, step {rows[0]['steps']}/{rows[0]['steps']}.", "",
             "| Run | val_loss | train_time |", "|---|---|---|"]
    lines += [f"| {Path(r.get('log') or r.get('out', '?')).stem[:8]} | {r['val_loss']:.4f} | {r['train_time_ms']:.0f}ms |" for r in rows]
    lines += ["", "## Summary", "", "|  | mean | std | min | max |", "|---|---|---|---|---|",
              f"| val_loss | {mu:.5f} | {sd:.5f} | {min(losses):.4f} | {max(losses):.4f} |",
              f"| train_time (ms) | {tmu:.1f} | {tsd:.1f} | {min(times):.0f} | {max(times):.0f} |", "",
              f"p(val_loss < {target}) = **{p:.4f}**  ({'PASS' if p < alpha else 'FAIL'} at p < {alpha})", "",
              f"One-sided one-sample t-test, n = {n}: `scipy.stats.ttest_1samp(losses, {target}, alternative=\"less\").pvalue`.",
              f"Verifier output: `n={n} val_loss={mu:.4f}±{sd:.4f} t={t:.3f} p(mean<{target})={p:.6f} time={tmu / 1000:.2f}±{tsd / 1000:.2f}s`."]
    try:
        from scipy import stats
        lines.append(f"(scipy cross-check: p = {stats.ttest_1samp(losses, target, alternative='less').pvalue:.6f})")
    except ImportError:
        pass
    return "\n".join(lines), p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, help="NUM_SCHEDULED_ITERATIONS for the launched runs")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--logs", type=Path, nargs="*", default=[], help="finished run logs/stdout files (no launch)")
    ap.add_argument("--target", type=float, default=TARGET)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--markdown", type=Path, help="also write the table here (e.g. the record folder's statistics.md)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()

    rows = []
    for path in a.logs:
        r = parse_result(path.read_text())
        if r is None:
            print(f"no final val_loss in {path}", file=sys.stderr)
            continue
        r.setdefault("out", str(path))
        rows.append(r)
    if a.steps is not None:
        seeds = a.seeds if a.seeds is not None else [None] * a.runs
        for i, seed in enumerate(seeds):
            tag = f"seed{seed}" if seed is not None else f"run{i}"
            r = launch(a.steps, seed, HERE / "results" / f"sig_steps{a.steps}_{tag}.out", nproc=a.nproc, dry_run=a.dry_run)
            if r is not None:
                rows.append(r)
                print(f"[done] {len(rows)}: val_loss={r['val_loss']:.4f} train_time={r['train_time_ms'] / 1000:.1f}s")
    if len(rows) < 2:
        print("need at least 2 finished runs")
        return 2
    if len({r["steps"] for r in rows}) > 1:
        print("WARNING: the runs have different step counts", file=sys.stderr)
    md, p = table(rows, a.target, a.alpha)
    print(md)
    if a.markdown:
        a.markdown.write_text(md + "\n")
    return 0 if p < a.alpha else 1


if __name__ == "__main__":
    sys.exit(main())
