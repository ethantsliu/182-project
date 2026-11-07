"""Summarize ablate.sh runs: final val loss / train time per arm, one-sided t-test vs 2.92,
and an iso-loss step estimate from the end-of-run slope of PR #261's six record logs.

    python 182/t2-gpu-port/summarize.py runs/
"""
import glob
import os
import re
import statistics
import sys

TARGET = 2.92
# PR #261 logs (records/track_2_medium/2026-0427-MuddFormer): mean val 2.9282 at step 4000 and 2.9171 at
# 4150, i.e. ~7.4e-5 loss/step over the last 150 steps, at ~207 ms/step.
LOSS_PER_STEP = (2.92820 - 2.91707) / 150
MS_PER_STEP = 207.3

FINAL = re.compile(r"^step:(\d+)/(\d+) val_loss:([0-9.]+) train_time:(\d+)ms", re.M)


def main(root):
    arms = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    base_loss = None
    print(f"{'arm':14s} {'n':>2s} {'steps':>5s} {'loss mean':>9s} {'sd':>7s} {'p(<2.92)':>9s} {'time s':>8s} {'sd':>5s} {'Δloss→steps':>11s}")
    for d in arms:
        losses, times, steps = [], [], set()
        for f in sorted(glob.glob(os.path.join(d, "*.out"))):
            m = [x for x in FINAL.finditer(open(f, errors="ignore").read()) if x.group(1) == x.group(2)]
            if m:
                s, _, loss, t = m[-1].groups()
                losses.append(float(loss)); times.append(int(t) / 1000); steps.add(int(s))
        if not losses:
            continue
        mean = statistics.fmean(losses)
        sd = statistics.stdev(losses) if len(losses) > 1 else float("nan")
        try:
            from scipy.stats import ttest_1samp
            p = ttest_1samp(losses, TARGET, alternative="less").pvalue if len(losses) > 1 else float("nan")
        except ImportError:
            p = float("nan")
        name = os.path.basename(d)
        if name == "base":
            base_loss = mean
        delta = "" if base_loss is None or name == "base" else f"{(base_loss - mean) / LOSS_PER_STEP:+.0f} st"
        tsd = statistics.stdev(times) if len(times) > 1 else float("nan")
        print(f"{name:14s} {len(losses):2d} {','.join(map(str, sorted(steps))):>5s} {mean:9.4f} {sd:7.4f} {p:9.4f} "
              f"{statistics.fmean(times):8.1f} {tsd:5.1f} {delta:>11s}")
    print(f"\nΔloss→steps: steps the loss gain vs 'base' would buy at {LOSS_PER_STEP:.2e} loss/step "
          f"(~{MS_PER_STEP * 1e-3:.2f} s/step); a rough guide only, the cooldown shrinks with the step count.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
