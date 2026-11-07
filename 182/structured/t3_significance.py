"""Track 3 significance for train_gpt_rotated.py logs (records/track_3_optimization/README.md, Rules).

    python 182/structured/t3_significance.py base=logs/a.txt,logs/b.txt hadamard=logs/c.txt

Each log may hold several trials (the trainer's num_trials argument); every `step:T/T val_loss:` line
is one trial's final loss. Per arm it prints the target rule, (3.28 - avg) * sqrt(n) >= 0.004, and
against the first arm the pairwise rule

    ((L_base - L_arm) + (S_base - S_arm) / 100 * 0.0045) / sqrt(1/n_base + 1/n_arm) >= 0.004

(0.0045 val loss per 100 steps, the README's extrapolation). A pairwise value <= -0.004 means the arm is
significantly WORSE than the base: that is the falsification outcome these experiments expect.
"""
import math
import re
import sys
from pathlib import Path

FINAL_RE = re.compile(r"^step:(\d+)/(\d+) val_loss:([0-9.]+)", re.M)
LOSS_PER_100_STEPS = 0.0045
THRESH = 0.004


def final_losses(paths):
    out = []  # (steps, loss)
    for p in paths:
        for step, total, loss in FINAL_RE.findall(Path(p).read_text()):
            if step == total:
                out.append((int(total), float(loss)))
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    arms = {}
    for spec in argv:
        name, _, paths = spec.partition("=")
        arms[name] = final_losses(paths.split(","))
    base = next(iter(arms))
    print(f"{'arm':<16} {'n':>3} {'steps':>6} {'avg':>8} {'target':>8}  {'pairwise vs ' + base:>20}")
    b = arms[base]
    for name, runs in arms.items():
        if not runs:
            print(f"{name:<16} no finished trials")
            continue
        steps = {s for s, _ in runs}
        assert len(steps) == 1, f"{name}: mixed step counts {steps}"
        (s,) = steps
        n, avg = len(runs), sum(l for _, l in runs) / len(runs)
        target = (3.28 - avg) * math.sqrt(n)
        line = f"{name:<16} {n:>3} {s:>6} {avg:>8.5f} {target:>+8.5f}{'*' if target >= THRESH else ' '}"
        if name != base and b:
            sb, lb, nb = b[0][0], sum(l for _, l in b) / len(b), len(b)
            pair = ((lb - avg) + (sb - s) / 100 * LOSS_PER_100_STEPS) / math.sqrt(1 / nb + 1 / n)
            verdict = "better" if pair >= THRESH else "WORSE" if pair <= -THRESH else "n.s."
            line += f" {pair:>+12.5f} ({verdict})"
        print(line)
    print("* = passes the 3.28 target rule")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
