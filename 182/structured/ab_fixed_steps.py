"""Fixed-step A/B for Track 1 env-flag experiments (e.g. MONARCH_MLP=1): run each arm on the same seeds at
the same step count, then turn the final val-loss difference into seconds.

    python 182/structured/ab_fixed_steps.py --seeds 0 1 2 3 \\
        --arm base: --arm monarch:MONARCH_MLP=1 --arm monarch_k8:MONARCH_MLP=1,MONARCH_BLOCKS=8

    python 182/structured/ab_fixed_steps.py --parse base=logs/a.txt,logs/b.txt monarch=logs/c.txt  # no runs

Conversion: near the record's step count, 1 millinat (0.001) of final val loss is worth ~SEC_PER_MNAT
seconds of training (~164 ms/millinat; override with --sec-per-mnat). 164 is upstream's normalization
convention (the rate that makes the ANVIL2 ablation table sum to its certified win); the band measured
during development is ~120-250 ms/millinat (records/track_1_short/2026-08-30_ANVIL2/README.md), so the
loss-side seconds carry that uncertainty. An arm's net cost against the first arm is

    net_s = (train_time_arm - train_time_base) + (loss_arm - loss_base) / 0.001 * sec_per_mnat

and net_s < 0 means the arm would be a faster record at equal loss. The conversion is only meaningful
at (close to) the full schedule: shortened --steps runs give a Δloss, not a record-time estimate.
Only the loss-side number is trustworthy when an arm is not perf-tuned (the Monarch MLP is plain bf16
torch.einsum, not a fused kernel), so both terms are printed separately.
"""
import argparse
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

SEC_PER_MNAT = 0.164
VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+) train_time:(\d+)ms")
REPO = Path(__file__).resolve().parents[2]


def final_val(text: str):
    """(val_loss, train_time_s) of the last validation line, or None."""
    hits = VAL_RE.findall(text)
    if not hits:
        return None
    if "INVALID_FOR_RECORD" in text:  # e.g. an FP8_DUMP_DIR analysis run: its train_time is not a timing
        print("warning: log is tagged INVALID_FOR_RECORD; skipped", file=sys.stderr)
        return None
    step, total, loss, ms = hits[-1]
    if step != total:
        print(f"warning: last val line is step {step}/{total} (run did not finish?)", file=sys.stderr)
    return float(loss), int(ms) / 1000.0


def parse_arm(spec: str) -> tuple[str, dict[str, str]]:
    name, _, envs = spec.partition(":")
    env = {}
    for kv in filter(None, envs.split(",")):
        k, _, v = kv.partition("=")
        env[k] = v
    return name, env


def run_one(name, env_extra, seed, steps, nproc, log_dir: Path):
    """One run of one arm; (val_loss, train_time_s) or None."""
    env = {**os.environ, **env_extra, "TRAIN_SEED": str(seed)}
    if steps is not None:
        env["NUM_SCHEDULED_ITERATIONS"] = str(steps)
    out = log_dir / f"{name}_seed{seed}.txt"
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "train_gpt.py"]
    print(f"[{name} seed={seed}] {' '.join(f'{k}={v}' for k, v in env_extra.items())} {' '.join(cmd)}", flush=True)
    with out.open("w") as f:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    r = final_val(out.read_text())
    if proc.returncode != 0 or r is None:
        print(f"  FAILED (rc={proc.returncode}); see {out}", flush=True)
        return None
    print(f"  val_loss={r[0]:.4f} train_time={r[1]:.2f}s", flush=True)
    return r


def run_interleaved(arm_specs, seeds, steps, nproc, log_dir: Path):
    """Seed-major order (seed 0: every arm, seed 1: every arm, ...), so node drift over the session lands
    on every arm alike instead of on Δtime."""
    arms = {name: [] for name, _ in arm_specs}
    for seed in seeds:
        for name, env in arm_specs:
            if (r := run_one(name, env, seed, steps, nproc, log_dir)) is not None:
                arms[name].append(r)
    return arms


def mean_sem(xs):
    m = statistics.fmean(xs)
    sem = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")
    return m, sem


def summarize(arms: dict[str, list[tuple[float, float]]], sec_per_mnat: float):
    names = [n for n, r in arms.items() if r]
    if not names:
        print("no results")
        return
    base = names[0]
    bl, bl_sem = mean_sem([r[0] for r in arms[base]])
    bt, _ = mean_sem([r[1] for r in arms[base]])
    print(f"\n{'arm':<14} {'n':>2} {'val_loss':>9} {'±sem':>7} {'time_s':>8} {'Δloss_mnat':>11} {'±sem':>6} "
          f"{'loss→s':>7} {'Δtime_s':>8} {'net_s':>7}")
    for name in names:
        losses, times = [r[0] for r in arms[name]], [r[1] for r in arms[name]]
        lm, ls = mean_sem(losses)
        tm, _ = mean_sem(times)
        d_mnat = (lm - bl) * 1000
        d_sem = math.sqrt(ls ** 2 + bl_sem ** 2) * 1000 if name != base else float("nan")
        loss_s = d_mnat * sec_per_mnat
        print(f"{name:<14} {len(losses):>2} {lm:>9.4f} {ls:>7.4f} {tm:>8.2f} {d_mnat:>+11.2f} {d_sem:>6.2f} "
              f"{loss_s:>+7.2f} {tm - bt:>+8.2f} {loss_s + tm - bt:>+7.2f}")
    print(f"\nloss→s = Δloss_mnat * {sec_per_mnat} s/mnat; net_s = loss→s + Δtime_s (negative = better than '{base}').")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], help="name:K=V,K=V (first arm is the baseline)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=None, help="NUM_SCHEDULED_ITERATIONS for every arm (default: the record's)")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--log-dir", default="logs/ab_structured")
    ap.add_argument("--sec-per-mnat", type=float, default=SEC_PER_MNAT)
    ap.add_argument("--parse", nargs="+", default=None, metavar="NAME=LOG[,LOG...]",
                    help="summarize existing logs instead of running")
    args = ap.parse_args()

    arms: dict[str, list] = {}
    if args.parse:
        for spec in args.parse:
            name, _, paths = spec.partition("=")
            arms[name] = [r for p in paths.split(",") if (r := final_val(Path(p).read_text())) is not None]
    else:
        if not args.arm:
            ap.error("give at least one --arm (or --parse)")
        log_dir = (REPO / args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        arms = run_interleaved([parse_arm(spec) for spec in args.arm], args.seeds, args.steps, args.nproc, log_dir)
    summarize(arms, args.sec_per_mnat)


if __name__ == "__main__":
    main()
