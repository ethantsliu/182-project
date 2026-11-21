"""Figures for the report, from the archived logs in ../results/raw. Run: python3 make_figs.py"""
import re, statistics as st, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = pathlib.Path(__file__).parent / "../results/raw/bringup-1119"
ARMS = {"record": ["base.log", "base-s2.log", "base-s3.log"],
        "Gram-space (ours)": ["gram222s-s1.log", "gram222s-s2.log", "gram222s-s3.log"],
        "no matrix work (ceiling)": ["skipmatrix.log", "skipmatrix-s2.log", "skipmatrix-s3.log"]}

def read(f):
    t = ms = n = None
    for l in open(RAW / f, errors="ignore"):
        if "METRIC val " in l and "final=1" in l:
            t = float(re.search(r"train_time_s=([\d.]+)", l).group(1))
        if "METRIC step_latency scope=all" in l:
            ms = float(re.search(r"mean_ms=([\d.]+)", l).group(1)); n = int(re.search(r" n=(\d+)", l).group(1))
    loop = ms * n / 1000
    return loop, t - loop

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
colors = ["#444444", "#c0392b", "#2980b9"]
for ax, idx, title in ((axes[0], 0, "step loop (1193 steps)"), (axes[1], 1, "rest of the clock")):
    for i, (arm, files) in enumerate(ARMS.items()):
        v = [read(f)[idx] for f in files]
        ax.scatter([i] * 3, v, color=colors[i], s=18, zorder=3)
        ax.errorbar(i + 0.22, st.mean(v), yerr=st.stdev(v), fmt="_", color=colors[i], capsize=3, ms=10)
        print(title, arm, [round(x, 3) for x in v], "mean %.3f sd %.3f" % (st.mean(v), st.stdev(v)))
    ax.set_xticks(range(3)); ax.set_xticklabels(["record", "ours", "ceiling"]); ax.set_xlim(-0.5, 2.7)
    ax.set_ylabel("seconds"); ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(pathlib.Path(__file__).parent / "fig_steploop.png", dpi=250)
