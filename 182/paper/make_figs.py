"""Figures for the report, drawn from the archived logs in ../results.

Style follows the conventions of ChenLiu-1996/figures4papers (scientific-figure-making: Helvetica-family sans,
top/right spines off, thick axes, semantic palette with blue for the proposed method and red for the baseline,
black bar edges, values printed above bars, tightened y-limits, 300 dpi). No code is copied from that repository.

Run: python3 make_figs.py   (writes fig_*.png next to this file)

fig_paired, fig_curve, fig_sweep and fig_profile_late take their runs, pairs and normalizations from
../results/make_tables.py (imported below), so they show the same numbers as tables.md sections B2c, C2 and D2.
"""
import pathlib, re, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
RES = HERE / "../results"
RAW = RES / "raw/bringup-1119"
sys.path.insert(0, str(RES.resolve()))
import make_tables as MT                         # stdlib only: the log parsers, pair definitions and statistics
ROWS = MT.collect_runs()
# Font sizes, in one place. The figures are drawn 6-11 in wide and printed at 0.4-0.8 of a 5.5 in text block,
# so these are large on purpose: FS = axis labels and default tick labels, FS_S = legends, value labels and
# crowded tick labels, FS_T = panel titles.
FS, FS_S, FS_T = 19, 15, 19
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B", "red_2": "#E9A6A1",
           "red_strong": "#B64342", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E"}
C_RECORD, C_OURS, C_CEIL = PALETTE["red_2"], PALETTE["blue_main"], PALETTE["neutral"]
C_RECORD_DARK = PALETTE["red_strong"]
plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"], "font.size": FS,
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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=FS_S, ncol=4, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
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
    groups = [("qk_bank", "6\u00d7256\u00d7768"), ("vo_bank", "2\u00d7768\u00d7768"), ("mlp_bank", "3\u00d72816\u00d7768")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, kind, title in ((axes[0], "matrix", "matrix sequence"), (axes[1], "update", "full update")):
        x = np.arange(3); w = 0.38
        for off, key, col, lab in ((-w / 2, "base", C_RECORD, "record"), (w / 2, "alt", C_OURS, "Gram-space")):
            v = [rows[(kind, g)][f"{key}_ms"] for g, _ in groups]
            lo = [rows[(kind, g)][f"{key}_ms"] - rows[(kind, g)][f"{key}_p10"] for g, _ in groups]
            hi = [rows[(kind, g)][f"{key}_p90"] - rows[(kind, g)][f"{key}_ms"] for g, _ in groups]
            ax.bar(x + off, v, w, yerr=[lo, hi], color=col, edgecolor="black", linewidth=1.5, capsize=4, error_kw=ERR, label=lab)
            for xi, vi in zip(x + off, v):
                ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=FS_S)
        ax.set_xticks(x); ax.set_xticklabels([s for _, s in groups], fontsize=FS_S)
        ax.set_title(title, fontsize=FS_T); ax.set_ylim(0, 0.62)
    axes[0].set_ylabel("time per update (ms)"); axes[0].legend(loc="upper left", fontsize=FS_S)
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
            ax.text(i, max(v) + pad, fmt.format(st.mean(v)), ha="center", va="bottom", fontsize=FS_S)
            print(" e2e", title, arm, [round(x, 3) for x in v], "mean %.3f sd %.3f" % (st.mean(v), st.stdev(v)))
        lo, hi = min(allv), max(allv); r = hi - lo
        ax.set_ylim(lo - 0.6 * r, hi + 0.45 * r)
        ax.set_xticks([]); ax.set_xlim(-0.6, 2.6); ax.set_ylabel("seconds"); ax.set_title(title, fontsize=FS_T)
    axes[0].legend(loc="upper right", fontsize=FS_S, bbox_to_anchor=(1.04, 1.02))
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
        ax.text(ms + 0.12, i, f"{p:.1f}%", va="center", fontsize=FS_S)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=FS_S)
    ax.set_xlabel("GPU kernel time per step (ms)"); ax.set_xlim(0, 9.8)
    save(fig, "fig_profile.png")
    print(" profile", rows)


# ---------------------------------------------------------------- 5. paired step-time savings, all three sessions
def fig_paired():
    # savings: the record's mean step time minus the arm's at the same seed, so a faster arm is positive (the report's
    # convention). Both arms use the same ten seeds (MT.COMMON_SEEDS); marker shape = session.
    runs = MT.timing_runs(ROWS); seeds = MT.COMMON_SEEDS
    session = [MT.SESSION[runs[("base", s)]["experiment"]] for s in seeds]
    marks = {1: ("o", "session 1"), 2: ("s", "session 2"), 3: ("^", "session 3")}
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for i, (arm, col, lab) in enumerate((("gram222s", C_OURS, "Gram-space"), ("skip_matrix", C_CEIL, "no matrix work (ablation)"))):
        v = [MT.saving(runs, arm, s, "all") for s in seeds]; mu = st.mean(v); sd = st.stdev(v)
        ax.bar(i, mu, 0.66, yerr=sd, color=col, edgecolor="black", linewidth=1.5, capsize=8, error_kw=ERR)
        for x, y, ses in zip(np.linspace(i - 0.25, i + 0.25, len(v)), v, session):
            ax.scatter(x, y, marker=marks[ses][0], s=58, facecolor="white", edgecolor="black", linewidth=1.5, zorder=4)
        ax.text(i, max(max(v), mu + sd) + 0.008, f"{mu:.3f}", ha="center", va="bottom", fontsize=FS_S)
        se = sd / len(v) ** 0.5
        print(f" paired {lab}: n={len(v)} seeds {seeds} per-seed {[round(x, 3) for x in v]} mean {mu:.4f} sd {sd:.4f} se {se:.4f} t {mu / se:.2f} -> {mu * 1.193:.3f} s/run")
    for m, lab in marks.values():
        ax.scatter([], [], marker=m, s=58, facecolor="white", edgecolor="black", linewidth=1.5, label=lab)
    ax.axhline(0, color="black", lw=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Gram-space", "no matrix work\n(ablation)"]); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.31)
    ax.set_ylabel("step time saved (ms)")
    ax.legend(loc="upper left", fontsize=FS_S, handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig_paired.png")


# ---------------------------------------------------------------- 6. training objective and validation loss during training
STAGE_STARTS = (320, 681, 1107, 1174)     # first step of stages 1-4: the batch size and the loss weights change here

def fig_curve():
    # Third session's curve runs, seeds 5-7 of the record and the kernel: mean line, band = +- s.d. across the seeds.
    # Top left has a LOG y axis: the per-token objective falls from 17 to 3.2 (its auxiliary weights are annealed away),
    # and on a linear axis the last 500 steps, where it moves by 0.2, would be a flat line. The two arms coincide at this
    # scale in both top panels, so the bottom row shows the paired difference (Gram-space minus record, same seed).
    cr = MT.curve_runs(ROWS); seeds = ["5", "6", "7"]
    stage_last = MT.timing_runs(ROWS)[("base", "8")]["stage_last"]
    def series(arm, kind):
        if kind == "train": d = [dict(MT.per_token(cr[(arm, s)], stage_last)) for s in seeds]
        else: d = [{p[0]: p[1] for p in cr[(arm, s)]["val_pts"]} for s in seeds]
        steps = sorted(d[0])
        return np.array(steps), np.array([[x[k] for k in steps] for x in d])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex="col", gridspec_kw={"height_ratios": [2.1, 1]})
    for j, (kind, ylab) in enumerate((("train", "training objective per token"), ("val", "validation loss"))):
        top, bot = axes[0, j], axes[1, j]
        data = {}
        for arm, col, lab, ls in (("base", C_RECORD_DARK, "record", "-"), ("gram222s", C_OURS, "Gram-space", "--")):
            x, y = series(arm, kind); data[arm] = y
            mu, sd = y.mean(0), y.std(0, ddof=1)
            top.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.25, lw=0)
            top.plot(x, mu, color=col, lw=2.5, ls=ls, label=lab, **({"marker": "o", "ms": 6} if kind == "val" else {}))
            print(f" curve {kind} {lab}: n={len(seeds)} steps {x[0]}..{x[-1]} ({len(x)} points) first {mu[0]:.4f} last {mu[-1]:.4f} max sd {sd.max():.4f} last sd {sd[-1]:.5f}")
        d = data["gram222s"] - data["base"]; dm, ds = d.mean(0), d.std(0, ddof=1)
        bot.fill_between(x, dm - ds, dm + ds, color=PALETTE["neutral"], alpha=0.8, lw=0)
        bot.plot(x, dm, color="black", lw=2, **({"marker": "o", "ms": 5} if kind == "val" else {}))
        bot.axhline(0, color="black", lw=1.2, ls=":")
        print(f" curve {kind} difference (Gram-space - record): max |mean| {abs(dm).max():.4f} at step {x[abs(dm).argmax()]}, last {dm[-1]:+.5f} +- {ds[-1]:.5f}")
        top.set_ylabel(ylab); bot.set_xlabel("training step"); bot.set_xlim(-20, 1214); bot.set_xticks([0, 300, 600, 900, 1194])
        lim = 1.25 * max(abs(dm - ds).max(), abs(dm + ds).max()); bot.set_ylim(-lim, lim)
        if kind == "train":
            top.set_yscale("log"); top.set_yticks([3, 4, 6, 8, 12, 16]); top.set_yticklabels(["3", "4", "6", "8", "12", "16"])
            top.minorticks_off(); top.set_ylim(2.9, 19)
            for a in (top, bot):
                for b in STAGE_STARTS: a.axvline(b, color=PALETTE["neutral"], lw=1.2, zorder=0)
        else:
            top.axhline(3.28, color="black", lw=1.5, ls=":", alpha=0.7)
    axes[1, 0].set_ylabel("difference"); axes[0, 0].legend(loc="upper right", fontsize=FS_S)
    fig.align_ylabels(axes[:, 0])
    save(fig, "fig_curve.png")


# ---------------------------------------------------------------- 7. microbenchmark over matrix sizes
def fig_sweep():
    k = MT.parse_kbench(MT.rglob("jobs/qlabs-*/sweep.log")[0])
    R = {r["name"]: float(r["speedup"]) for r in k["rows"]}       # measured rows only (the failed cases have none)
    shapes = list(dict.fromkeys(r["shape"] for r in k["rows"] if r["name"].startswith("matrix.")))
    dims = lambda sh: tuple(int(v) for v in sh.split("x"))
    thin = sorted((sh for sh in shapes if dims(sh)[2] == 768), key=lambda sh: dims(sh)[1])
    wide = [sh for sh in shapes if dims(sh)[2] != 768]
    styles = (("matrix", "gram222s", C_OURS, "o", "-", "Gram-space, two maps per restart"),
              ("matrix", "gram33s", PALETTE["teal"], "s", "-", "Gram-space, three maps per restart"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True, gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for kind, kern, col, m, ls, lab in styles:
        x = [dims(sh)[1] / 768 for sh in thin]; y = [R[f"{kind}.sweep_{sh}.{kern}"] for sh in thin]
        ax.plot(x, y, color=col, lw=2.5, ls=ls, marker=m, ms=8, mec="white", mew=1, label=lab)
        print(f" sweep {kind} {kern} at 768 columns:", [(round(a, 2), round(b, 3)) for a, b in zip(x, y)])
    ax.axvline(2816 / 768, color=PALETTE["neutral"], lw=1.5, zorder=0)
    ax.text(2816 / 768 * 1.06, 0.78, "MLP group", fontsize=FS_S, va="center")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"]); ax.minorticks_off()
    ax.set_xlabel("rows / columns (768 columns)"); ax.set_ylabel("speedup over record")
    ax.legend(loc="upper left", fontsize=FS_S, borderaxespad=0.1, handlelength=1.6)
    ax = axes[1]
    for kind, kern, col, m, ls, lab in styles:
        y = [R[f"{kind}.sweep_{sh}.{kern}"] for sh in wide]
        ax.plot(range(len(wide)), y, color=col, ls="none", marker=m, ms=10, mec="white", mew=1)
        print(f" sweep {kind} {kern} larger shapes:", [(sh, round(b, 3)) for sh, b in zip(wide, y)])
    ax.set_xticks(range(len(wide))); ax.set_xticklabels([f"{dims(sh)[1]}\n$\\times${dims(sh)[2]}" for sh in wide], fontsize=FS_S)
    ax.set_xlim(-0.5, len(wide) - 0.5); ax.set_xlabel("rows $\\times$ columns")
    for a in axes: a.axhline(1, color="black", lw=1.5, ls="--")
    axes[0].set_ylim(0.6, 3.2)
    save(fig, "fig_sweep.png")


# ---------------------------------------------------------------- 8. GPU time per step in the late window (step 900)
def fig_profile_late():
    rel = MT.rglob("jobs/qlabs-*/profile-late/analysis.txt")[0]
    head, cats, _ = MT.parse_profile(rel)
    rows = sorted(((n.split(" (")[0], ms, p) for n, ms, p in cats), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cols = [C_OURS if n.startswith("optimizer") else PALETTE["neutral"] for n, _, _ in rows]
    ax.barh(range(len(rows)), [r[1] for r in rows], color=cols, edgecolor="black", linewidth=1.5)
    top = max(r[1] for r in rows)
    for i, (_, ms, p) in enumerate(rows):
        ax.text(ms + 0.013 * top, i, f"{p:.1f}%", va="center", fontsize=FS_S)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=FS_S)
    ax.set_xlabel("GPU kernel time per step (ms)"); ax.set_xlim(0, top * 1.165)
    save(fig, "fig_profile_late.png")
    print(" profile late:", head.split(";")[0], rows)


def losses():
    lr = MT.timing_runs(ROWS, with_seed4=True)
    seeds = sorted({s for a, s in lr if a == "gram222s" and ("base", s) in lr}, key=int)
    for name, arm in (("record", "base"), ("gram222s", "gram222s")):
        loss = [MT.V(lr[(arm, s)]) for s in seeds]
        print(f" loss {name}: n={len(loss)} seeds {seeds} mean {st.mean(loss):.6f} sd {st.stdev(loss):.6f}")


if __name__ == "__main__":
    fig_maps(); fig_bench(); fig_steploop(); fig_profile(); fig_paired(); fig_curve(); fig_sweep(); fig_profile_late(); losses()
