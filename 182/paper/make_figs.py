"""Figures for the report, drawn from the archived logs in ../results.

Style follows the conventions of ChenLiu-1996/figures4papers (scientific-figure-making: Helvetica-family sans,
top/right spines off, thick axes, semantic palette with blue for the proposed method and red for the baseline,
black bar edges, values printed above bars, tightened y-limits, 300 dpi). No code is copied from that repository.

Run: python3 make_figs.py   (writes fig_*.png next to this file)
"""
import pathlib, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
RES = HERE / "../results"
RAW = RES / "raw/bringup-1119"
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B", "red_2": "#E9A6A1",
           "red_strong": "#B64342", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E"}
C_RECORD, C_OURS, C_CEIL = PALETTE["red_2"], PALETTE["blue_main"], PALETTE["neutral"]
C_RECORD_DARK = PALETTE["red_strong"]
plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"], "font.size": 16,
                     "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2,
                     "legend.frameon": False, "xtick.major.width": 2, "ytick.major.width": 2})
ERR = {"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"}


def save(fig, name, pad=1):
    fig.tight_layout(pad=pad)
    fig.savefig(HERE / name, dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. singular values through the six maps
MAPS = [(3.923798038567, -6.095026865488, 3.905234618423), (3.278126713798, -3.328923386476, 0.989127286973),
        (3.505298394150, -5.137358782410, 1.968325560615), (2.815058591845, -3.685181239622, 1.417196497642),
        (2.245503932403, -2.443826979899, 0.963091710461), (2.256537145403, -2.166840097229, 0.929501253245)]

def fig_maps():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s0, c in ((0.01, PALETTE["blue_main"]), (0.05, PALETTE["teal"]), (0.2, PALETTE["violet"]), (0.6, PALETTE["red_strong"])):
        t = [s0]
        for a, b, cc in MAPS:
            t.append(t[-1] * (a + b * t[-1] ** 2 + cc * t[-1] ** 4))
        ax.plot(range(7), t, color=c, lw=2.5, marker="o", ms=6, label=rf"$\sigma_0={s0}$")
    ax.axhline(1, color="black", lw=1.5, ls="--", alpha=0.6)
    ax.set_xlabel("after map"); ax.set_ylabel(r"singular value $\sigma$")
    ax.set_xticks(range(7)); ax.set_ylim(0, 1.6)
    ax.legend(loc="lower right", fontsize=13, ncol=2)
    save(fig, "fig_maps.png")


# ---------------------------------------------------------------- 2. kernel microbenchmark, per weight group
def bench_rows():
    rows = {}
    for l in open(RAW / "bench.log", errors="ignore"):
        m = re.match(r"KBENCH name=(matrix|update)\.(\w+_bank)\.gram222s ", l)
        if m:
            g = lambda k: float(re.search(rf"{k}=([\d.eE+-]+)", l).group(1))
            rows[m.groups()] = {k: g(k) for k in ("base_ms", "alt_ms", "base_p10", "base_p90", "alt_p10", "alt_p90")}
    return rows

def fig_bench():
    rows = bench_rows()
    groups = [("qk_bank", r"$6\times256\times768$"), ("vo_bank", r"$2\times768\times768$"), ("mlp_bank", r"$3\times2816\times768$")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, kind, title in ((axes[0], "matrix", "matrix sequence"), (axes[1], "update", "full update")):
        x = np.arange(3); w = 0.38
        for off, key, col, lab in ((-w / 2, "base", C_RECORD, "record"), (w / 2, "alt", C_OURS, "Gram-space")):
            v = [rows[(kind, g)][f"{key}_ms"] for g, _ in groups]
            lo = [rows[(kind, g)][f"{key}_ms"] - rows[(kind, g)][f"{key}_p10"] for g, _ in groups]
            hi = [rows[(kind, g)][f"{key}_p90"] - rows[(kind, g)][f"{key}_ms"] for g, _ in groups]
            ax.bar(x + off, v, w, yerr=[lo, hi], color=col, edgecolor="black", linewidth=1.5, capsize=4, error_kw=ERR, label=lab)
            for xi, vi in zip(x + off, v):
                ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels([s for _, s in groups], fontsize=12)
        ax.set_title(title, fontsize=16); ax.set_ylim(0, 0.62)
    axes[0].set_ylabel("time per update (ms)"); axes[0].legend(loc="upper left", fontsize=13)
    save(fig, "fig_bench.png")
    for k, v in sorted(rows.items()):
        print(" bench", k, v["base_ms"], v["alt_ms"], "speedup %.3f" % (v["base_ms"] / v["alt_ms"]))


# ---------------------------------------------------------------- 3. end to end: step loop and remainder
ARMS = [("record", ["base.log", "base-s2.log", "base-s3.log"], C_RECORD),
        ("Gram-space", ["gram222s-s1.log", "gram222s-s2.log", "gram222s-s3.log"], C_OURS),
        ("no matrix work (ablation)", ["skipmatrix.log", "skipmatrix-s2.log", "skipmatrix-s3.log"], C_CEIL)]

def read_run(f):
    t = ms = n = None
    for l in open(RAW / f, errors="ignore"):
        if "METRIC val " in l and "final=1" in l:
            t = float(re.search(r"train_time_s=([\d.]+)", l).group(1))
        if "METRIC step_latency scope=all" in l:
            ms = float(re.search(r"mean_ms=([\d.]+)", l).group(1)); n = int(re.search(r" n=(\d+)", l).group(1))
    loop = ms * n / 1000
    return loop, t - loop

def fig_steploop():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, idx, title, fmt, pad in ((axes[0], 0, "step loop (1193 steps)", "{:.3f}", 0.012), (axes[1], 1, "rest of the clock", "{:.3f}", 0.008)):
        allv = []
        for i, (arm, files, col) in enumerate(ARMS):
            v = [read_run(f)[idx] for f in files]; allv += v
            ax.bar(i, st.mean(v), 0.62, yerr=st.stdev(v), color=col, edgecolor="black", linewidth=1.5, capsize=8, error_kw=ERR, label=arm)
            ax.scatter(np.full(3, i) + np.array([-0.16, 0, 0.16]), v, color="black", s=22, zorder=4)
            ax.text(i, max(v) + pad, fmt.format(st.mean(v)), ha="center", va="bottom", fontsize=13)
            print(" e2e", title, arm, [round(x, 3) for x in v], "mean %.3f sd %.3f" % (st.mean(v), st.stdev(v)))
        lo, hi = min(allv), max(allv); r = hi - lo
        ax.set_ylim(lo - 0.6 * r, hi + 0.45 * r)
        ax.set_xticks([]); ax.set_xlim(-0.6, 2.6); ax.set_ylabel("seconds"); ax.set_title(title, fontsize=16)
    axes[0].legend(loc="upper right", fontsize=12, bbox_to_anchor=(1.04, 1.02))
    save(fig, "fig_steploop.png")


# ---------------------------------------------------------------- 4. where a step's GPU time goes (profile)
def fig_profile():
    txt = (RES / "README.md").read_text()
    sec = txt[txt.index("### 3.6"):]
    rows = re.findall(r"^\| ([^|]+?) \| ([\d.]+) \| ([\d.]+)% \|$", sec, flags=re.M)
    rows = sorted(((n.split(" (")[0], float(ms), float(p)) for n, ms, p in rows), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cols = [C_OURS if n.startswith("optimizer") else PALETTE["neutral"] for n, _, _ in rows]
    ax.barh(range(len(rows)), [r[1] for r in rows], color=cols, edgecolor="black", linewidth=1.5)
    for i, (_, ms, p) in enumerate(rows):
        ax.text(ms + 0.12, i, f"{p:.1f}%", va="center", fontsize=12)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlabel("GPU kernel time per step (ms)"); ax.set_xlim(0, 9.8)
    save(fig, "fig_profile.png")
    print(" profile", rows)


# ---------------------------------------------------------------- 5. paired step-loop differences, both sessions
RAW2 = RES / "raw/jobs/kernel-ab2-1121-152359/out"
PAIRS = {  # seed: (base, gram222s, skip_matrix or None); seeds 1-3 bring-up node, 6-7 second node (order swapped)
    1: (RAW / "base.log", RAW / "gram222s-s1.log", RAW / "skipmatrix.log"),
    2: (RAW / "base-s2.log", RAW / "gram222s-s2.log", RAW / "skipmatrix-s2.log"),
    3: (RAW / "base-s3.log", RAW / "gram222s-s3.log", RAW / "skipmatrix-s3.log"),
    6: (RAW2 / "root-base-s6.log", RAW2 / "root-gram222s-s6.log", RAW2 / "root-skipmatrix-s6.log"),
    7: (RAW2 / "root-base-s7.log", RAW2 / "root-gram222s-s7.log", None),
}

def step_ms(path):
    for l in open(path, errors="ignore"):
        if "METRIC step_latency scope=all" in l:
            return float(re.search(r"mean_ms=([\d.]+)", l).group(1))

def final(path):
    for l in open(path, errors="ignore"):
        m = re.search(r"METRIC val step=\d+ total=\d+ train_time_s=([\d.]+) val_loss=([\d.]+) final=1", l)
        if m:
            return float(m.group(1)), float(m.group(2))

def fig_paired():
    dg = {s: step_ms(g) - step_ms(b) for s, (b, g, k) in PAIRS.items()}
    dk = {s: step_ms(k) - step_ms(b) for s, (b, g, k) in PAIRS.items() if k is not None}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, (d, col, lab) in enumerate(((dg, C_OURS, "Gram-space"), (dk, C_CEIL, "no matrix work (ablation)"))):
        v = list(d.values()); mu = st.mean(v); sd = st.stdev(v)
        ax.bar(i, mu, 0.6, yerr=sd, color=col, edgecolor="black", linewidth=1.5, capsize=8, error_kw=ERR, label=lab)
        ax.scatter(np.linspace(i - 0.2, i + 0.2, len(v)), v, color="black", s=24, zorder=4)
        ax.text(i, min(v) - 0.022, f"{mu:.3f}", ha="center", va="top", fontsize=13)
        se = sd / len(v) ** 0.5
        print(f" paired {lab}: n={len(v)} per-seed {[round(x, 3) for x in v]} mean {mu:.4f} sd {sd:.4f} se {se:.4f} t {mu / se:.2f} -> {mu * 1.193:.3f} s/run")
    ax.axhline(0, color="black", lw=2)
    ax.set_xticks([]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.31, 0.03)
    ax.set_ylabel("step time vs. record (ms)")
    ax.legend(loc="lower left", fontsize=12)
    save(fig, "fig_paired.png")
    for name, idx in (("record", 0), ("gram222s", 1)):
        loss = [final(v[idx])[1] for v in PAIRS.values()] + [final(RAW / ("base-s4.log" if idx == 0 else "../jobs/fastest-1-1119-094724/out/root-gram222s-s4.log"))[1]]
        print(f" loss {name}: n={len(loss)} mean {st.mean(loss):.6f} sd {st.stdev(loss):.6f}")


# ---------------------------------------------------------------- 6. validation loss during training
def fig_curve():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for f, col, lab, ls in ((RAW2 / "root-curve-base-s5.log", C_RECORD_DARK, "record", "-"), (RAW2 / "root-curve-gram222s-s5.log", C_OURS, "Gram-space", "--")):
        pts = [(int(a), float(b), float(c)) for a, b, c in re.findall(r"METRIC val step=(\d+) total=\d+ train_time_s=([\d.]+) val_loss=([\d.]+)", open(f, errors="ignore").read())]
        # the clock of a curve run carries the validations' own overhead and, for the first run on the node, a cold
        # first step: plot time relative to the first validation
        t0 = pts[0][1]
        axes[0].plot([p[0] for p in pts], [p[2] for p in pts], color=col, lw=2.5, ls=ls, marker="o", ms=6, label=lab)
        axes[1].plot([p[1] - t0 for p in pts], [p[2] for p in pts], color=col, lw=2.5, ls=ls, marker="o", ms=6, label=lab)
        print(" curve", lab, pts)
    for ax, xl in ((axes[0], "training step"), (axes[1], "training time since step 150 (s)")):
        ax.axhline(3.28, color="black", lw=1.5, ls=":", alpha=0.7)
        ax.set_xlabel(xl); ax.set_ylabel("validation loss")
    axes[0].legend(loc="upper right", fontsize=13)
    save(fig, "fig_curve.png")


if __name__ == "__main__":
    fig_maps(); fig_bench(); fig_steploop(); fig_profile(); fig_paired(); fig_curve()
