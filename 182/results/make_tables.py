#!/usr/bin/env python3
"""Rebuild runs.csv, tables.md and figures/*.png from the raw logs in raw/.

    python3 182/results/make_tables.py            # stdlib only; figures need matplotlib (skipped without it)

Every number in README.md comes from tables.md, and every number in tables.md is parsed here from a file
under raw/ (the `log` column of runs.csv names the file for each run). Nothing is typed in by hand: the only
constants below are which folder holds which experiment, the 3.28 / 2.92 targets, and the year of the runs.
"""
import csv, glob, json, math, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
FIG = os.path.join(HERE, "figures")
YEAR = 2026                       # the root trainer's torchrun line carries month, day and time only
TARGET = {"1": 3.28, "2": 2.92}   # speedrun val-loss targets (Track 1, Track 2)
EXT_STEPS = 10                    # t1-cpu-ngram: total steps = NUM_SCHEDULED_ITERATIONS + 10 (685 = 675 + 10)
# Kernel A/B arms by the flag line the trainer prints, and the three sessions that ran them.
ARM = {"": "base", "ANVIL_KERNEL=gram222s": "gram222s", "ANVIL_ABLATE=skip_matrix": "skip_matrix",
       "ANVIL_KERNEL=gram33s": "gram33s", "ANVIL_KERNEL=gram42s": "gram42s"}
SESSION = {"kernel_ab": 1, "kernel_ab2": 2, "kernel_ab3": 3}
# Seeds with a same-session record, kernel AND ablation run: the paired timing analysis uses these for both arms.
# Seed 4 is out (its arms ran on different nodes); seed 7 is out (no ablation run), seed 5 and 99 are not timing runs.
COMMON_SEEDS = ["1", "2", "3", "6", "8", "9", "10", "11", "12", "13"]
# Tokens per rank per step in each schedule stage = TRAINING_STAGES[i].batch_size / 8 ranks (track_1_short/config.py).
# `METRIC train loss=` is rank 0's per-token training objective SUMMED over these tokens (step_graphs.py: `.sum()`).
TOKENS_PER_RANK = (16384, 32768, 49152, 40960, 16384)

# ---------------------------------------------------------------- statistics (stdlib)

def mean(xs): return sum(xs) / len(xs)
def sd(xs):
    if len(xs) < 2: return float("nan")
    m = mean(xs); return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def t_cdf(t, df):
    """Student t CDF by Simpson integration of the density (accurate to ~1e-8 for the sizes used here)."""
    if math.isnan(t): return float("nan")
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    a, n = abs(t), 4000
    h = a / n
    s = f(0) + f(a) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))
    half = s * h / 3
    return 0.5 + half if t >= 0 else 0.5 - half

def t_crit(df, p=0.975):
    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def summary(xs):
    """n, mean, sd, se, t (mean / se), 95% CI half-width, two-sided p for mean = 0."""
    n = len(xs); m = mean(xs)
    if n < 2: return dict(n=n, mean=m, sd=float("nan"), se=float("nan"), t=float("nan"), ci=float("nan"), p=float("nan"))
    s = sd(xs); se = s / math.sqrt(n)
    t = m / se if se > 0 else float("nan")
    return dict(n=n, mean=m, sd=s, se=se, t=t, ci=t_crit(n - 1) * se, p=2 * (1 - t_cdf(abs(t), n - 1)) if se > 0 else float("nan"))

# ---------------------------------------------------------------- log parsing

TS = r"(?:\[\d+\.\d+\] )?"        # the variants' console files prefix each line with [unix time]
RE_STEP = re.compile(rf"^{TS}step:(\d+)/(\d+) train_time:(\d+)ms", re.M)
RE_VALLINE = re.compile(rf"^{TS}step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms", re.M)
RE_METRIC_VAL = re.compile(r"^METRIC val step=(\d+) total=(\d+) train_time_s=([\d.]+) val_loss=([\d.]+) final=1", re.M)
RE_FINAL_GPU = re.compile(r"\[final\] val_loss:([\d.]+) train_time_ms:(\d+) steps:(\d+)")
RE_ENDPOINT = re.compile(r"endpoint_result: steps=(\d+) seed=(\S+) val_loss=([\d.]+) train_time_ms=(\d+)")
RE_PEAK = re.compile(rf"^{TS}peak memory allocated: (\d+) MiB reserved: (\d+) MiB", re.M)
RE_PEAK_METRIC = re.compile(r"^METRIC peak_mem rank=max allocated_mib=(\d+) reserved_mib=(\d+)", re.M)
RE_LAT = re.compile(r"^METRIC step_latency scope=(\w+) steps=\S+ n=(\d+) mean_ms=([\d.]+) median_ms=([\d.]+) p95_ms=([\d.]+)", re.M)
RE_CFG = re.compile(r"^METRIC run_config .*seed=(\S+) world_size=(\d+) total_steps=(\d+)", re.M)
RE_ANVIL = re.compile(r"\[anvil-kernel\] (ANVIL_\w+=\w+) banks=(\[[^\]]*\])")
RE_ROOT_START = re.compile(r"^W(\d\d)(\d\d) (\d\d:\d\d:\d\d)\.\d+ \d+ torch/distributed/run.py", re.M)
RE_VOTE = re.compile(r"table vote: val_loss:([\d.]+) with the table's links, ([\d.]+) without them, ([\d.]+) with PR #399's fixed vote")
RE_CLOCK = re.compile(r"^METRIC clock_split step0_ms=([\d.]+) loop_end_ms=([\d.]+) tail_ms=([\d.]+) total_ms=([\d.]+)", re.M)
RE_VAL_ANY = re.compile(r"^METRIC val step=(\d+) total=(\d+) train_time_s=([\d.]+) val_loss=(\S+) final=([01])", re.M)
RE_TRAIN = re.compile(r"^METRIC train step=(\d+) total=\d+ loss=(\S+) rank=0", re.M)
RE_STAGE = re.compile(r"^METRIC step_latency scope=stage(\d) steps=(\d+)-(\d+)", re.M)
RE_ASSERT = re.compile(r"^\[rank\d+\]: (AssertionError: .*)$", re.M)
RE_NET = re.compile(r"network val_loss:([\d.]+) \(before the chain; the chain adds ([-+\d.]+)\)")

def read(rel):
    with open(os.path.join(RAW, rel), errors="replace") as f:
        return f.read()

def rglob(pattern):
    return sorted(os.path.relpath(p, RAW) for p in glob.glob(os.path.join(RAW, pattern), recursive=True))

def parse_log(rel):
    """Fields every trainer log has in some form. Losses are kept as the string the log printed."""
    txt = read(rel)
    r = dict(log="raw/" + rel, steps="", train_time_s="", val_loss="", peak_alloc_mib="", peak_reserved_mib="",
             peak_mem_source="", mean_step_ms="", loop_s="", loop_ms_per_step="", seed="", flags="")
    m = RE_METRIC_VAL.search(txt)                       # root tree with RUN_METRICS=1: 6 decimals
    g = RE_FINAL_GPU.search(txt)                        # t1-gpu-stack: 6 decimals
    e = RE_ENDPOINT.search(txt)                         # t2-cpu-ngram: 6 decimals
    v = RE_VALLINE.search(txt)                          # every tree: 4 decimals
    if m:   r.update(steps=int(m[2]), train_time_s=m[3], val_loss=m[4])
    elif g: r.update(steps=int(g[3]), train_time_s=f"{int(g[2]) / 1000:.3f}", val_loss=g[1])
    elif e: r.update(steps=int(e[1]), train_time_s=f"{int(e[4]) / 1000:.3f}", val_loss=e[3])
    elif v: r.update(steps=int(v[2]), train_time_s=f"{int(v[4]) / 1000:.3f}", val_loss=v[3])
    pm, p = RE_PEAK_METRIC.search(txt), RE_PEAK.search(txt)
    if pm:  r.update(peak_alloc_mib=int(pm[1]), peak_reserved_mib=int(pm[2]), peak_mem_source="METRIC peak_mem rank=max (timed loop)")
    elif p: r.update(peak_alloc_mib=int(p[1]), peak_reserved_mib=int(p[2]), peak_mem_source="'peak memory allocated' line (rank 0, whole process)")
    r["stages"] = {}
    for s in RE_LAT.finditer(txt):
        r["stages"][s[1]] = dict(n=int(s[2]), mean=float(s[3]), median=float(s[4]), p95=float(s[5]))
    if "all" in r["stages"]:
        r["mean_step_ms"] = f"{r['stages']['all']['mean']:.3f}"
    r["curve"] = [(int(a), int(c) / 1000) for a, b, c in RE_STEP.findall(txt)]   # (step, train time s) console lines
    if r["curve"] and r["steps"] and r["curve"][-1][0] == r["steps"]:
        r["loop_s"] = f"{r['curve'][-1][1]:.3f}"                                # clock at the last step, before the final validation
        r["loop_ms_per_step"] = f"{1000 * r['curve'][-1][1] / r['steps']:.3f}"
    c = RE_CFG.search(txt)
    if c: r["seed"] = c[1]
    a = RE_ANVIL.search(txt)
    if a: r["flags"] = a[1]
    st = RE_ROOT_START.search(txt)
    r["start_utc"] = f"{YEAR}-{st[1]}-{st[2]}T{st[3]}" if st else ""
    vt = RE_VOTE.search(txt)
    r["vote"] = vt.groups() if vt else None
    nt = RE_NET.search(txt)
    r["net_val"] = nt.groups() if nt else None
    ck = RE_CLOCK.search(txt)                           # newer root logs: the clock split into step 0, the loop, the tail
    r["clock"] = dict(zip(("step0", "loop_end", "tail", "total"), map(float, ck.groups()))) if ck else None
    r["val_pts"] = [(int(a), float(d), int(e)) for a, _, _, d, e in RE_VAL_ANY.findall(txt)]   # (step, val loss, final)
    r["train_pts"] = [(int(a), float(b)) for a, b in RE_TRAIN.findall(txt)]                    # (step, summed loss)
    r["stage_last"] = {int(k): int(b) for k, _, b in RE_STAGE.findall(txt)}                    # stage -> its last step
    asserts = RE_ASSERT.findall(txt)
    r["assertion"] = asserts[0] if asserts else ""
    return r

def name_time(rel):
    m = re.search(r"(\d{4})(\d\d)(\d\d)[-T](\d\d)(\d\d)(\d\d)", os.path.basename(rel))
    return f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}" if m else ""

def launches(job):
    """console file name -> env assignments of its `launch:` line (the wrapper prints launch, console, RESULT)."""
    out = {}
    for rel in rglob(f"jobs/{job}/out/**/*.log") + rglob(f"jobs/{job}/out/**/runs/*.out"):
        pending = None
        for line in read(rel).splitlines():
            if line.startswith("launch:"):
                env = line.split("&&", 1)[1].split("torchrun")[0].split()
                pending = [kv for kv in env if "=" in kv]
            elif pending is not None:
                m = re.match(r"console: (\S+)", line) or re.search(r"RESULT .* log=(\S+)", line)
                if m:
                    out[os.path.basename(m[1]).replace(".log.txt", ".console.txt")] = pending
                    if line.startswith("console:"): continue
                    pending = None
    return out

def lor_time(job, what):
    """UTC time of a `LOR HH:MM:SS <what>` line of the job log; the date comes from the job's creation time."""
    m = re.search(rf"^.*LOR (\d\d:\d\d:\d\d) {re.escape(what)}", read(f"jobs/{job}/log.txt"), re.M)
    if not m: return ""
    created = json.load(open(os.path.join(RAW, "jobs", job, "job.json")))["created_at"]
    day = created[:10]
    if m[1] < created[11:19]:           # past midnight UTC
        import datetime
        day = str(datetime.date.fromisoformat(day) + datetime.timedelta(days=1))
    return f"{day}T{m[1]}"

def ql_time(job, what):
    """UTC time of a `QL HH:MM:SS <what>` line of a Q Labs job's slurm.out; the date comes from job.json's created_at."""
    m = re.search(rf"^QL (\d\d:\d\d:\d\d) {re.escape(what)}", read(f"jobs/{job}/slurm.out"), re.M)
    if not m: return ""
    created = json.load(open(os.path.join(RAW, "jobs", job, "job.json")))["created_at"]
    return f"{created[:10]}T{m[1]}"       # the job did not cross midnight UTC

def first_nan(pts): return next((p[0] for p in pts if math.isnan(p[1])), None)
def last_finite(pts): return max((p[0] for p in pts if not math.isnan(p[1]) and p[0] < (first_nan(pts) or 1e9)), default=None)

def phase_time(job, phase):
    for rel in rglob(f"jobs/{job}/out/session/session.log"):
        m = re.search(rf"^== {phase}: est\..* -- (\d{{4}}-\d\d-\d\d) (\d\d:\d\d:\d\d)", read(rel), re.M)
        if m: return f"{m[1]}T{m[2]}"
    return ""

def collect_runs():
    rows = []
    def add(rel, **kw):
        r = parse_log(rel); r.update({k: v for k, v in kw.items() if v not in (None, "")}); rows.append(r); return r

    # root tree: the ANVIL kernel A/B (interactive bring-up node, then two more runs in the fastest-1 job)
    for rel in rglob("bringup-1119/*.log") + rglob("jobs/fastest-1-*/out/root-*.log"):
        base = os.path.basename(rel)
        if not re.match(r"(root-)?(base|gram222s|skipmatrix)", base): continue
        job = "bringup-1119" if rel.startswith("bringup") else rel.split("/")[1]
        r = add(rel, job=job, tree="root (record #92)", track="1", experiment="kernel_ab", role="ab")
        if not r.get("start_utc"):      # no torchrun banner in this log: take the job script's "LOR <time> <name> start" line
            r["start_utc"] = lor_time(job, base[len("root-"):-len(".log")] + " start")
        r["env"] = " ".join(x for x in ["RUN_METRICS=1", f"TRAIN_SEED={r['seed']}", r["flags"]] if x)
    # root tree, second session (job kernel-ab2, another node): two VAL_CURVE runs, then base / gram222s with the order
    # swapped per seed and one skip_matrix run. Kept as its own experiments so the first session's tables stay as they were.
    for rel in rglob("jobs/kernel-ab2-*/out/root-*.log"):
        base = os.path.basename(rel)[len("root-"):-len(".log")]
        job = rel.split("/")[1]
        curve = base.startswith("curve-")
        r = add(rel, job=job, tree="root (record #92)", track="1", experiment="val_curve" if curve else "kernel_ab2",
                role="curve" if curve else "ab",
                note="VAL_CURVE=150: validates every 150 steps; not a timing run" if curve else "")
        if not r.get("start_utc"):
            r["start_utc"] = lor_time(job, base + " start")
        r["env"] = " ".join(x for x in ["RUN_METRICS=1", "VAL_CURVE=150" if curve else "", f"TRAIN_SEED={r['seed']}", r["flags"]] if x)
    # root tree, third session (Slurm job qlabs-13354, one node; logs sit flat in the job folder): a discarded warm-up,
    # then base / gram222s / skip_matrix on seeds 8-13 with the arm order rotating through all six permutations
    # (kernel_ab3), and VAL_CURVE=150 TRAIN_LOSS_EVERY=25 curve runs of base, gram222s, gram33s, gram42s (curve3).
    for rel in rglob("jobs/qlabs-*/root-*.log"):
        kind, arm, _ = os.path.basename(rel)[len("root-"):-len(".log")].split("-")
        job = rel.split("/")[1]
        curve, warm = kind == "curve", arm == "warmup"
        r = add(rel, job=job, tree="root (record #92)", track="1", experiment="curve3" if curve else "kernel_ab3",
                role="curve" if curve else "warmup" if warm else "ab",
                note="VAL_CURVE=150 TRAIN_LOSS_EVERY=25: validates every 150 steps and reads the training loss every 25; not a timing run" if curve
                else "discarded warm-up run before the timing runs: excluded from all statistics" if warm else "")
        if not r.get("start_utc"):
            r["start_utc"] = ql_time(job, os.path.basename(rel)[len("root-"):-len(".log")] + " start")
        r["env"] = " ".join(x for x in ["RUN_METRICS=1", "VAL_CURVE=150 TRAIN_LOSS_EVERY=25" if curve else "", f"TRAIN_SEED={r['seed']}", r["flags"]] if x)
        final = [v for v in r["val_pts"] if v[2] == 1]
        if not final:                   # no final validation: the run died (the fallback regex must not pick up anything)
            r.update(steps="", train_time_s="", val_loss="", loop_s="", loop_ms_per_step="")
            r["fail"] = "FAILED before training" + (f" ({r['assertion']})" if r["assertion"] else " (no result in the log)")
        elif math.isnan(final[0][1]):   # ran to the end with a NaN loss: the fallback regex would otherwise report the step-150 validation
            m = re.search(r"^METRIC val step=(\d+) total=(\d+) train_time_s=([\d.]+) val_loss=nan final=1", read(rel), re.M)
            r.update(steps=int(m[2]), train_time_s=m[3], val_loss="nan")
            r["fail"] = (f"DIVERGED: training loss is NaN from the step-{first_nan(r['train_pts'])} read on (last finite read: step "
                         f"{last_finite(r['train_pts'])}), validation loss is NaN from step {first_nan(r['val_pts'])} to the end")
    for rel in rglob("jobs/fastest-1-*/out/session/profile/root/train.out"):
        job = rel.split("/")[1]
        add(rel, job=job, tree="root (record #92)", track="1", experiment="profile", role="profile", seed="random",
            start_utc=phase_time(job, "profile"), env="(torch.profiler attached)",
            note="profiled run: not a timing run")
    for rel in rglob("jobs/qlabs-*/profile-late/train.out"):
        job = rel.split("/")[1]
        add(rel, job=job, tree="root (record #92)", track="1", experiment="profile", role="profile", seed="",
            start_utc=ql_time(job, "late-stage profile start"), env="(torch.profiler attached, 20 steps from step 900; seed not logged)",
            note="profiled run: not a timing run")
    # variants/t1-gpu-stack
    for rel in rglob("jobs/*/out/session/t1gpu/logs/*.console.txt"):
        add(rel, job=rel.split("/")[1], tree="variants/t1-gpu-stack", track="1", experiment="repro", role="session",
            seed="random", start_utc=name_time(rel), env="(defaults)")
    # variants/t1-cpu-ngram
    for job_dir in sorted(glob.glob(os.path.join(RAW, "jobs", "*"))):
        job = os.path.basename(job_dir)
        L = launches(job)
        for rel in rglob(f"jobs/{job}/out/**/*-c381*.console.txt"):
            base = os.path.basename(rel)
            sub = rel.split("/out/")[1].split("/")[0]
            env = L.get(base)
            sm = re.search(r"-s(\d+)\.console", base)
            arm = (re.search(r"-ab([AB])-", base) or [None, ""])[1]
            if sub == "warm":   exp, role = "warmup", "warmup"
            elif job.startswith("ngram-ab"): exp, role = "table_vote_ab", "ab"
            elif job.startswith("ngram-steps"): exp, role = "step_sweep", "sweep"
            elif sub == "ab":   exp, role = "steps_pairs", "ab"
            else:               exp, role = "repro", "session"
            flags = " ".join(kv for kv in (env or []) if not re.match(r"(TRAIN_SEED|DATA_PATH|PYTHONUNBUFFERED)=", kv))
            r = add(rel, job=job, tree="variants/t1-cpu-ngram", track="1", experiment=exp, role=role, arm=arm,
                    seed=sm[1] if sm else "random", start_utc=name_time(rel), flags=flags,
                    env=" ".join(env) if env else "(launch line not captured; seed from the file name)")
            if r["steps"]:
                r["sched_steps"] = r["steps"] - EXT_STEPS
    # variants/t2-cpu-ngram
    for rel in rglob("jobs/*/out/session/t2cpu/logs/stdout.txt"):
        job = rel.split("/")[1]
        add(rel, job=job, tree="variants/t2-cpu-ngram", track="2", experiment="track2", role="session", seed="random",
            start_utc=phase_time(job, "t2cpu"), env="(defaults)")

    # order within the job, first run of a tree on a fresh node, validity
    by_job = defaultdict(list)
    for r in rows: by_job[r["job"]].append(r)
    for job, rs in by_job.items():
        rs.sort(key=lambda r: r.get("start_utc", ""))
        seen = defaultdict(int)
        for i, r in enumerate(rs, 1):
            seen[r["tree"]] += 1
            r["order_in_job"], r["order_in_tree"] = i, seen[r["tree"]]
    for r in rows:
        notes = [r.get("note", "")]
        r["valid"] = 1
        if r.get("fail"):
            r["valid"] = 0; notes.append(r["fail"])
        elif not r["val_loss"]:
            r["valid"] = 0; notes.append("FAILED before training (no result in the log)")
        if "ANVIL_ABLATE" in r["flags"]:
            notes.append("measurement only: the optimizer's matrix work is skipped, the loss is not a valid result")
        if r["order_in_tree"] == 1 and r["valid"]:
            notes.append("first run of this tree on a fresh node (compiles first; runs slower)")
        r["note"] = "; ".join(n for n in notes if n)
        r["hit_target"] = "" if not r["val_loss"] else int(float(r["val_loss"]) <= TARGET[r["track"]])   # nan -> 0
    rows.sort(key=lambda r: (r.get("start_utc", ""), r["job"]))
    return rows

COLS = ["job", "order_in_job", "order_in_tree", "start_utc", "tree", "track", "experiment", "role", "arm", "flags", "seed",
        "sched_steps", "steps", "train_time_s", "val_loss", "hit_target", "peak_alloc_mib", "peak_reserved_mib",
        "peak_mem_source", "mean_step_ms", "loop_s", "loop_ms_per_step", "valid", "env", "note", "log"]

# ---------------------------------------------------------------- other parsed inputs

def parse_bench():
    rows, totals, info = [], [], ""
    for line in read("bringup-1119/bench.log").splitlines():
        if line.startswith("KBENCH_INFO"): info = line
        kv = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
        if line.startswith("KBENCH name="): rows.append(kv)
        elif line.startswith("KBENCH_TOTAL"): totals.append(kv)
    return info, rows, totals

def parse_profile(rel=None):
    rels = [rel] if rel else rglob("jobs/fastest-1-*/out/session/profile/root/analysis.txt")
    if not rels: return "", [], ""
    lines = read(rels[0]).splitlines()
    cats, started = [], False
    for line in lines:
        if line.startswith("category"): started = True; continue
        if not started: continue
        m = re.match(r"(.+?)\s+([\d.]+)\s+([\d.]+)%$", line)
        if not m: break
        cats.append((m[1].strip(), float(m[2]), float(m[3])))
    nccl = next((l for l in lines if l.startswith("NCCL:")), "")
    return lines[0], cats, nccl

def parse_jobs():
    out = []
    for p in sorted(glob.glob(os.path.join(RAW, "jobs", "*", "job.json"))):
        d = json.load(open(p))
        sp = d.get("spend") or {}
        out.append((os.path.basename(os.path.dirname(p)), d.get("provider", "Lor"), d.get("state", "").replace("COMPUTE_RUN_STATE_", ""),
                    d["created_at"], d["observed_at"], sp.get("gpu_hours"), sp.get("amount")))   # amount None: not billed in money
    return out

# ---------------------------------------------------------------- tables

def f(x, nd=3): return "-" if x == "" or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.{nd}f}"
def T(r): return float(r["train_time_s"])
def V(r): return float(r["val_loss"])

def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"

def diff_row(label, d_time, d_loss=None):
    s = summary(d_time)
    row = [label, s["n"], f"{s['mean']:+.3f}", f(s["sd"]), f(s["se"]), f(s["t"], 2), f"±{f(s['ci'])}", f(s["p"], 4)]
    if d_loss is not None:
        l = summary(d_loss)
        row += [f"{l['mean']:+.5f}", f(l["se"], 5), f(l["t"], 2), f"±{f(l['ci'], 5)}", f(l["p"], 4)]
    return row

DIFF_HEAD = ["difference", "pairs", "mean Δtime (s)", "sd", "se", "t", "95% CI", "p (two-sided)"]
LOSS_HEAD = ["mean Δval loss", "se", "t", "95% CI", "p (two-sided)"]

# ---------------------------------------------------------------- third session; analysis over all sessions
# (182/paper/make_figs.py imports these helpers, so the report's figures and tables.md use the same pairs.)

def fp(p):
    if isinstance(p, float) and math.isnan(p): return "-"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}" if p >= 1e-6 else "<1e-6"      # t_cdf is good to about 1e-8

def loop_s(r):
    """The 1193 timed steps after step 0 (s): mean step latency x steps."""
    a = r["stages"]["all"]; return a["mean"] * a["n"] / 1000

def out_of_loop_s(r):
    """Clock outside the timed steps (s): step 0 plus the tail after the last step."""
    if r["clock"]: return (r["clock"]["step0"] + r["clock"]["tail"]) / 1000
    return T(r) - loop_s(r)             # first session: no clock_split line; the same quantity by subtraction

def timing_runs(rows, with_seed4=False):
    """(arm, seed) -> valid timing run, every session. Seed 4 has no same-session record run, so it is for losses only."""
    return {(ARM[r["flags"]], r["seed"]): r for r in rows
            if r["valid"] and r["experiment"] in SESSION and r["role"] == "ab" and (with_seed4 or r["seed"] != "4")}

def saving(runs, arm, seed, key):
    """Record minus arm at the same seed. key: a step_latency scope (ms), 'time' (whole-run s) or 'out' (out-of-loop s)."""
    get = T if key == "time" else out_of_loop_s if key == "out" else (lambda r: r["stages"][key]["mean"])
    return get(runs[("base", seed)]) - get(runs[(arm, seed)])

def per_token(r, stage_last):
    """[(step, training objective per token)]: the logged sum over rank 0's tokens / tokens per rank in that step's stage."""
    stage = lambda s: next(k for k in sorted(stage_last) if s <= stage_last[k])
    return [(s, v / TOKENS_PER_RANK[stage(s)]) for s, v in r["train_pts"]]

def curve_runs(rows):
    """(arm, seed) -> third-session curve run (valid or not)."""
    return {(ARM[r["flags"]], r["seed"]): r for r in rows if r["experiment"] == "curve3"}

def parse_kbench(rel):
    """KBENCH rows / totals / errors of a bench.py log, plus torch's recompile-limit message if it is there."""
    txt = read(rel); rows, totals, errors, info, done = [], [], [], "", ""
    for line in txt.splitlines():
        kv = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
        if line.startswith("KBENCH_INFO"): info = line
        elif line.startswith("KBENCH name="): rows.append(kv)
        elif line.startswith("KBENCH_TOTAL"): totals.append(kv)
        elif line.startswith("KBENCH_ERROR"):
            m = re.match(r"KBENCH_ERROR name=(\S+) err=(\w+)", line); errors.append((m[1], m[2]))
        elif line.startswith("KBENCH_DONE"): done = line
    lim = re.search(r"torch\._dynamo hit config\.recompile_limit \((\d+)\)\n.*?function: '(\w+)' \(\S*?(track_1_short/\S+?)\)\n.*?last reason: (.*)", txt)
    return dict(info=info, rows=rows, totals=totals, errors=errors, done=done, limit=lim.groups() if lim else None)

def pm(xs, nd=3): return f"{mean(xs):.{nd}f} ± {sd(xs):.{nd}f}"

def stat_cells(xs, nd=4):
    s = summary(xs)
    return [s["n"], f"{s['mean']:+.{nd}f}", f(s["sd"], nd), f(s["se"], nd), f(s["t"], 2), fp(s["p"])]

STAT_HEAD = ["n", "mean", "sd", "se", "t", "p (two-sided)"]
SCOPES = ["all", "stage0", "stage1", "stage2", "stage3", "stage4"]

def section_b2c(rows):
    k3 = [r for r in rows if r["experiment"] == "kernel_ab3"]
    if not k3: return []
    ab = sorted([r for r in k3 if r["role"] == "ab" and r["valid"]], key=lambda r: r["order_in_job"])
    runs = timing_runs(rows)
    seeds3 = sorted({r["seed"] for r in ab}, key=int)
    arms, alts = ["base", "gram222s", "skip_matrix"], ["gram222s", "skip_matrix"]
    nsteps = {s: runs[("base", seeds3[0])]["stages"][s]["n"] for s in SCOPES}
    job = k3[0]["job"]
    o = [f"### B2c. Third session (Slurm job {job}, one node; arm order rotated over seeds 8-13) and all sessions together", "",
         "Convention in B2c: a **saving** is the record (base) minus the arm at the same seed in the same session, so a faster arm is",
         "positive (B2b and B3 print arm - base, the opposite sign). Loss differences are arm - base. `mean step` is",
         "`METRIC step_latency ... mean_ms` (the 1193 steps after step 0; printed to 0.001 ms). Out-of-loop time = step 0 + the tail",
         "after the last step (`METRIC clock_split step0_ms + tail_ms`; for the first session, which has no clock_split line,",
         "train time - mean step x 1193). Whole-run train time = step loop + out-of-loop time.", ""]
    # per seed
    tab = []
    for sd_ in seeds3:
        rs = {a: runs[(a, sd_)] for a in arms}
        order = " > ".join(sorted(arms, key=lambda a: rs[a]["order_in_job"]))
        tab.append([sd_, order, " / ".join(rs[a]["train_time_s"] for a in arms), " / ".join(rs[a]["mean_step_ms"] for a in arms),
                    " / ".join(f"{out_of_loop_s(rs[a]):.3f}" for a in arms), " / ".join(rs[a]["val_loss"] for a in arms)])
    o += ["Per seed; each cell is base / gram222s / skip_matrix:", "",
          md_table(["seed", "order the arms ran in", "train time (s)", "mean step (ms)", "out-of-loop time (s)", "val loss"], tab)]
    for w in [r for r in k3 if r["role"] == "warmup"]:
        o += [f"Warm-up run before these (seed {w['seed']}, base, order {w['order_in_job']} in the job): train time {w['train_time_s']} s, mean step "
              f"{w['mean_step_ms']} ms, val loss {w['val_loss']}. It is in runs.csv with role=warmup and is not used in any statistic.", ""]
    # unpaired
    tab = []
    for a in arms:
        rs = [runs[(a, s)] for s in seeds3]; oo = [out_of_loop_s(r) for r in rs]
        tab.append([a, len(rs), pm([T(r) for r in rs], 4), pm([loop_s(r) for r in rs], 4), pm([r["stages"]["all"]["mean"] for r in rs], 4),
                    pm(oo, 4), f"{min(oo):.3f} - {max(oo):.3f}", pm([V(r) for r in rs], 6)])
    oo_all = [out_of_loop_s(runs[(a, s)]) for a in arms for s in seeds3]
    st0 = [runs[(a, s)]["clock"]["step0"] for a in arms for s in seeds3]; tl = [runs[(a, s)]["clock"]["tail"] for a in arms for s in seeds3]
    o += [f"Unpaired, by arm (mean ± sd over seeds {seeds3[0]}-{seeds3[-1]}):", "",
          md_table(["arm", "n", "train time (s)", "step loop (s)", "mean step (ms)", "out-of-loop time (s)", "out-of-loop range (s)", "val loss"], tab),
          f"Out-of-loop time over all {len(oo_all)} runs: {min(oo_all):.3f} to {max(oo_all):.3f} s (mean {mean(oo_all):.3f}, sd {sd(oo_all):.3f}); "
          f"step 0 {min(st0):.1f} to {max(st0):.1f} ms (mean {mean(st0):.1f}), tail {min(tl):.1f} to {max(tl):.1f} ms (mean {mean(tl):.1f}).", ""]
    # paired, this session
    quantities = [(f"mean step, {s} ({nsteps[s]} steps) (ms)" if s != "all" else "mean step, all 1193 steps (ms)", s) for s in SCOPES] + \
                 [("train time, whole run (s)", "time"), ("out-of-loop time (s)", "out")]
    tab = []
    for a in alts:
        for label, key in quantities:
            xs = [saving(runs, a, s, key) for s in seeds3]
            tab.append([a, label, " / ".join(f"{x:+.3f}" for x in xs)] + stat_cells(xs))
    o += [f"Paired savings against the record, this session only (seeds {', '.join(seeds3)}):", "",
          md_table(["arm", "saving (base - arm)", "by seed"] + STAT_HEAD, tab)]
    tab = []
    for a in alts:
        xs = [V(runs[(a, s)]) - V(runs[("base", s)]) for s in seeds3]
        tab.append([f"{a} - base", " / ".join(f"{x:+.6f}" for x in xs)] + stat_cells(xs, 6))
    o += ["Paired difference in final val loss, this session (the skip_matrix arm is not a valid training run):", "",
          md_table(["difference", "by seed"] + STAT_HEAD, tab)]

    # all sessions, the same ten seeds for both arms
    C = COMMON_SEEDS
    rec = [T(runs[("base", s)]) for s in C]; rec_mean = mean(rec)
    o += [f"#### All sessions, the same {len(C)} seeds for both arms ({', '.join(C)})", "",
          "Seed 4 is left out because its arms ran on different nodes; seed 7 because it has no ablation run (its kernel pair is shown",
          "separately below); seed 5 has curve runs only. Sessions: 1 = bring-up node (seeds 1-3, record first), 2 = job kernel-ab2 (seed 6),",
          f"3 = job {job} (seeds 8-13, order rotated). `s per run` = saving per step x that scope's steps; `% of run` is against the",
          f"record's mean train time over these seeds, {rec_mean:.4f} s (sd {sd(rec):.4f}).", ""]
    tab = []
    for a in alts:
        for label, key in quantities:
            xs = [saving(runs, a, s, key) for s in C]
            per_run = mean(xs) * nsteps[key] / 1000 if key in nsteps else mean(xs)
            tab.append([a, label] + stat_cells(xs) + [f"{per_run:+.4f}", f"{100 * per_run / rec_mean:+.3f}%"])
    o += [md_table(["arm", "saving (base - arm)"] + STAT_HEAD + ["s per run", "% of run"], tab)]
    tab = []
    sessions = [("1 (seeds 1-3)", ["1", "2", "3"]), ("2 (seed 6)", ["6"]), ("3 (seeds 8-13)", seeds3), (f"all three ({len(C)} seeds)", C)]
    for a in alts:
        for label, ss in sessions + ([("all three + seed 7 (kernel only; not the common-seed set)", C + ["7"])] if a == "gram222s" else []):
            xs = [saving(runs, a, s, "all") for s in ss]; ts = [saving(runs, a, s, "time") for s in ss]
            tab.append([a, label] + stat_cells(xs) + [f"{mean(xs) * 1.193:+.4f}"] + stat_cells(ts)[1:])
    o += ["By session (one pair has no sd):", "",
          md_table(["arm", "session"] + [f"step: {h}" for h in STAT_HEAD] + ["step: s per run (x 1193)"] + [f"train time (s): {h}" for h in STAT_HEAD[1:]], tab)]
    o += ["The pairs behind these rows (savings, base - arm):", "",
          md_table(["seed", "session", "gram222s: mean step (ms)", "skip_matrix: mean step (ms)", "gram222s: train time (s)", "skip_matrix: train time (s)",
                    "gram222s: out-of-loop (s)", "skip_matrix: out-of-loop (s)"],
                   [[s, SESSION[runs[("base", s)]["experiment"]]] + [f"{saving(runs, a, s, k_):+.3f}" for k_ in ("all", "time", "out") for a in alts] for s in C])]
    # validation loss, record vs kernel
    lr = timing_runs(rows, with_seed4=True)
    tab = []
    all_l = sorted({s for a, s in lr if a == "gram222s" and ("base", s) in lr}, key=int)
    for label, ss in [(f"all valid timing seeds ({', '.join(all_l)})", all_l), (f"the {len(C)} common seeds", C), (f"third session ({', '.join(seeds3)})", seeds3)]:
        b = [V(lr[("base", s)]) for s in ss]; g = [V(lr[("gram222s", s)]) for s in ss]; d = [y - x for x, y in zip(b, g)]
        tab.append([label, len(ss), pm(b, 6), pm(g, 6)] + stat_cells(d, 6)[1:] + [f"{sum(x < 0 for x in d)} of {len(d)}", f"{max(b):.6f} / {max(g):.6f}"])
    o += ["#### Final validation loss, record vs kernel (timing runs; seed 4 is included here, its loss does not depend on the node's speed)", "",
          md_table(["seeds", "n", "record (mean ± sd)", "gram222s (mean ± sd)"] + [f"gram222s - record: {h}" for h in STAT_HEAD[1:]] +
                   ["pairs where the kernel is lower", "worst run, record / gram222s"], tab)]

    # curve runs
    cr = curve_runs(rows)
    if cr:
        stage_last = runs[("base", seeds3[0])]["stage_last"]
        o += [f"#### Curve runs of the third session (experiment `curve3`; VAL_CURVE=150, TRAIN_LOSS_EVERY=25; never used for timing)", "",
              md_table(["order", "arm", "seed", "final val loss", "outcome"],
                       [[r["order_in_job"], a, s, r["val_loss"] or "-", "finished" + (", at or under 3.28" if r["hit_target"] else ", ABOVE 3.28") if r["valid"] else r["fail"]]
                        for (a, s), r in sorted(cr.items(), key=lambda kv: kv[1]["order_in_job"])])]
        cs = sorted({s for a, s in cr if a == "gram222s" and cr[(a, s)]["valid"] and cr.get(("base", s), {}).get("valid")}, key=int)
        steps = [p[0] for p in cr[("base", cs[0])]["val_pts"]]
        val = lambda a, s: dict((p[0], p[1]) for p in cr[(a, s)]["val_pts"])
        g33 = sorted((s for a, s in cr if a == "gram33s"), key=int)
        tab = []
        for st in steps:
            b = [val("base", s)[st] for s in cs]; g = [val("gram222s", s)[st] for s in cs]; d = [y - x for x, y in zip(b, g)]
            tab.append([st, pm(b, 5), pm(g, 5), f"{mean(d):+.5f} ± {sd(d):.5f}"] + [f"{val('gram33s', s)[st]:.6f}" for s in g33])
        o += [f"Validation loss by step, seeds {', '.join(cs)} (mean ± sd over the {len(cs)} seeds; the difference is paired by seed):", "",
              md_table(["step", "record", "gram222s", "gram222s - record"] + [f"gram33s seed {s}" for s in g33], tab)]
        fin = [V(cr[("gram222s", s)]) - V(cr[("base", s)]) for s in cs]
        o += [f"Final val loss of the curve runs, gram222s - record by seed ({', '.join(cs)}): {' / '.join(f'{x:+.6f}' for x in fin)}; "
              f"mean {mean(fin):+.6f}, se {sd(fin) / math.sqrt(len(fin)):.6f}, n = {len(fin)}.", ""]
        pt = {(a, s): dict(per_token(cr[(a, s)], stage_last)) for a in ("base", "gram222s") for s in cs}
        tsteps = sorted(pt[("base", cs[0])]); stage = lambda s: next(k for k in sorted(stage_last) if s <= stage_last[k])
        tab = []
        for st in tsteps:
            b = [pt[("base", s)][st] for s in cs]; g = [pt[("gram222s", s)][st] for s in cs]; d = [y - x for x, y in zip(b, g)]
            tab.append([st, stage(st), TOKENS_PER_RANK[stage(st)], pm(b, 4), pm(g, 4), f"{mean(d):+.4f} ± {sd(d):.4f}"])
        o += ["Training objective per token by step (same seeds, mean ± sd). `METRIC train loss=` is rank 0's training objective summed over",
              "its tokens of that step (`step_graphs.py`: the per-token loss `.sum()`); it is divided here by the tokens per rank of the step's",
              "stage (`config.py` batch_size / 8 ranks). The per-token objective is the weighted sum of the next-token CE, the multi-token",
              "CE terms and the prefix-token CE, over a sampled candidate set of 10240 / 14336 / 24576 classes until step 1106 and the full",
              "vocabulary after. Its weights and candidate count change with the schedule, so it is not a plain next-token cross-entropy and",
              "is not comparable with the validation loss.", "",
              md_table(["step", "stage", "tokens per rank", "record", "gram222s", "gram222s - record"], tab)]
    return o

def section_sweep():
    rel = rglob("jobs/qlabs-*/sweep.log")
    if not rel: return []
    k = parse_kbench(rel[0]); R = {r["name"]: r for r in k["rows"]}
    shapes = list(dict.fromkeys(r["shape"] for r in k["rows"] if r["name"].startswith("matrix.")))
    ratio = lambda sh: int(sh.split("x")[1]) / int(sh.split("x")[2])
    sp = lambda kind, sh, kern: R.get(f"{kind}.sweep_{sh}.{kern}")
    kerns = ["gram222s", "gram33s", "gram42s"]
    o = [f"### C2. Size sweep of the microbenchmark (third session, one H100, raw/{rel[0]})", "", f"`{k['info']}`", "",
         "Speedup = record's time / kernel's time on the same shape (above 1: the Gram-space order is faster). Shapes are N x rows x cols.", ""]
    tab = []
    for sh in shapes:
        m = sp("matrix", sh, "gram222s")
        tab.append([sh, f"{ratio(sh):.2f}", m["base_ms"], m["alt_ms"]] + [sp("matrix", sh, kn)["speedup"] for kn in kerns] +
                   [sp("update", sh, kn)["speedup"] if sp("update", sh, kn) else "failed" for kn in kerns] +
                   [f"{100 * float(m['base_exact_err']):.2f}%"] + [f"{100 * float(sp('matrix', sh, kn)['alt_exact_err']):.2f}%" for kn in kerns])
    o += [md_table(["shape", "rows / cols", "matrix seq.: record ms", "gram222s ms"] + [f"matrix seq. speedup, {kn}" for kn in kerns] +
                   [f"full update speedup, {kn}" for kn in kerns] + ["record err vs exact"] + [f"{kn} err vs exact" for kn in kerns], tab)]
    c768 = sorted([sh for sh in shapes if sh.endswith("x768")], key=ratio)
    for kind, label in (("matrix", "matrix sequence"), ("update", "full update")):
        for kn in ("gram222s", "gram33s"):
            pts = [(ratio(sh), float(sp(kind, sh, kn)["speedup"])) for sh in c768 if sp(kind, sh, kn)]
            lo = max((p for p in pts if p[1] < 1), default=None); hi = next((p for p in pts if p[1] >= 1), None)
            if lo and hi and lo[0] < hi[0]:
                x = lo[0] + (1 - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1])
                o += [f"- 768 columns, {label}, {kn}: {lo[1]:.3f}x at rows/cols = {lo[0]:.2f}, {hi[1]:.3f}x at {hi[0]:.2f}; "
                      f"break-even near rows/cols = {x:.1f} (linear interpolation between those two shapes)."]
    o += [""]
    if k["errors"]:
        by = defaultdict(list)
        for name, err in k["errors"]: by[err].append(name)
        o += [f"`{k['done'].split(' failed_names=')[0]}`: {len(k['rows'])} rows were measured and {len(k['errors'])} cases raised an error and have no row "
              f"(so {len(k['rows']) + len(k['errors'])} cases were attempted). Every failure is a `full update` or `ablate` case of the last "
              f"{len({n.split('sweep_')[1].split('.')[0] for n, _ in k['errors']})} shapes:", ""]
        o += [f"- `{err}` ({len(ns)} cases): {', '.join(ns)}" for err, ns in by.items()] + [""]
        if k["limit"]:
            n, fn, path, reason = k["limit"]
            o += [f"Cause, from the log: torch.compile's recompile limit ({n}). The full-update cases call the trainer's compiled function `{fn}` ({path}),",
                  f"which is compiled with fullgraph=True and is recompiled for every new matrix shape; after {n} shapes the next compilation raises",
                  f"(first such message: \"{reason.strip()}\"). It is a limit of the benchmark harness, not a numerical failure;",
                  "the `matrix` cases do not go through that function and have rows for all shapes. No measured row is dropped.", ""]
    b = rglob("jobs/qlabs-*/bench.log")
    if b:
        kb = parse_kbench(b[0]); B = {r["name"]: r for r in kb["rows"]}
        tab = [[kind, bank, kn, B[f"{kind}.{bank}.{kn}"]["shape"], B[f"{kind}.{bank}.{kn}"]["base_ms"], B[f"{kind}.{bank}.{kn}"]["alt_ms"], B[f"{kind}.{bank}.{kn}"]["speedup"],
                f"{100 * float(B[f'{kind}.{bank}.{kn}']['alt_exact_err']):.2f}%" if "alt_exact_err" in B[f"{kind}.{bank}.{kn}"] else "-"]
               for kind in ("matrix", "update") for bank in ("qk_bank", "vo_bank", "mlp_bank") for kn in kerns if f"{kind}.{bank}.{kn}" in B]
        o += [f"### C3. The standard microbenchmark on the third session's node (raw/{b[0]})", "", f"`{kb['info']}`", "",
              md_table(["what", "bank", "kernel", "shape", "record ms", "alt ms", "speedup", "alt err vs exact"], tab)]
        share = next((t for t in kb["totals"] if t.get("kind") == "matrix_share"), None)
        if share:
            o += [f"All three banks on this node: update {share['update_ms_per_step']} ms/step, without the matrix work {share['skip_matrix_ms_per_step']} ms/step, "
                  f"matrix work {share['matrix_ms_per_step']} ms/step (section C, first node: see above).", ""]
    return o

def build_tables(rows):
    o = ["# Summary tables", "",
         "Generated by `make_tables.py` from `raw/`. Do not edit: rerun the script. Times are the trainers' own",
         "`train_time` (seconds). Val losses are printed as logged; means of losses are computed from those values.",
         "sd is the sample standard deviation (n-1), se = sd/sqrt(n), CIs and p-values use Student's t.", ""]
    ok = [r for r in rows if r["valid"]]

    # --- A. per tree
    o += ["## A. Each tree on our node (all valid runs, default flags)", ""]
    tab = []
    def tree_row(label, rs):
        ts, vs = [T(r) for r in rs], [V(r) for r in rs]
        peaks = sorted({r["peak_alloc_mib"] for r in rs if r["peak_alloc_mib"] != ""})
        lp = [float(r["loop_ms_per_step"]) for r in rs if r["loop_ms_per_step"]]
        ms = [float(r["mean_step_ms"]) for r in rs if r["mean_step_ms"]]
        return [label, len(rs), f(mean(ts)), f(sd(ts)), f(min(ts)), f(max(ts)), f(mean(vs), 5), f(sd(vs), 5), max(rs, key=V)["val_loss"],
                sum(r["hit_target"] for r in rs), f"{peaks[0]}" if len(peaks) == 1 else f"{peaks[0]}-{peaks[-1]}",
                f(mean(lp)) if lp else "-", f(mean(ms)) if ms else "-"]
    groups = [
        ("root record #92, baseline", [r for r in ok if r["experiment"] == "kernel_ab" and not r["flags"]]),
        ("root, ANVIL_KERNEL=gram222s (ours)", [r for r in ok if r["experiment"] == "kernel_ab" and r["flags"] == "ANVIL_KERNEL=gram222s"]),
        ("t1-gpu-stack (upstream PR #397 tree)", [r for r in ok if r["tree"].endswith("t1-gpu-stack")]),
        ("t1-cpu-ngram, 675 scheduled steps (upstream PR #381), all runs", [r for r in ok if r.get("sched_steps") == 675 and "TABLE_VOTE" not in r["flags"]]),
        ("t1-cpu-ngram, 675, without each job's first run", [r for r in ok if r.get("sched_steps") == 675 and "TABLE_VOTE" not in r["flags"] and r["order_in_tree"] > 1]),
        ("t1-cpu-ngram, 668 scheduled steps (our trim), all runs", [r for r in ok if r.get("sched_steps") == 668]),
        ("t1-cpu-ngram, 668, interleaved-pair runs only", [r for r in ok if r.get("sched_steps") == 668 and r["experiment"] == "steps_pairs"]),
        ("t2-cpu-ngram (Track 2, target 2.92)", [r for r in ok if r["track"] == "2"]),
    ]
    for label, rs in groups:
        if rs: tab.append(tree_row(label, rs))
    o += [md_table(["tree / config", "runs", "mean time (s)", "sd", "min", "max", "mean val loss", "sd", "worst val loss",
                    "runs at or under target", "peak GPU mem allocated (MiB)", "loop ms/step", "METRIC mean step (ms)"], tab),
          "`loop ms/step` = training clock at the last step (before the final validation) / steps, available for every tree.",
          "`METRIC mean step` = mean of per-step latencies from `METRIC step_latency scope=all` (root tree with RUN_METRICS=1 only;",
          "it leaves out step 0). Peak memory: root rows are",
          "`METRIC peak_mem rank=max` (max over the 8 GPUs, timed loop); variant rows are the trainer's last line (rank 0).", ""]

    # --- B. kernel A/B
    k = [r for r in ok if r["experiment"] == "kernel_ab"]
    arm_of = lambda r: ARM[r["flags"]]
    o += ["## B. Optimizer kernel A/B on the root record (1194 steps)", "", "### B1. Every run, in the order it ran", ""]
    o += [md_table(["job", "order", "start (UTC)", "arm", "seed", "train time (s)", "val loss", "mean step (ms)", "p95 step (ms)", "peak alloc (MiB)", "peak reserved (MiB)"],
                   [[r["job"], r["order_in_job"], r["start_utc"][5:], arm_of(r), r["seed"], r["train_time_s"], r["val_loss"], r["mean_step_ms"],
                     f(r["stages"]["all"]["p95"]), r["peak_alloc_mib"], r["peak_reserved_mib"]] for r in k])]
    o += ["### B2. By arm", ""]
    tab = []
    for arm in ["base", "gram222s", "skip_matrix"]:
        rs = sorted([r for r in k if arm_of(r) == arm], key=lambda r: int(r["seed"]))
        ts, vs, ms = [T(r) for r in rs], [V(r) for r in rs], [float(r["mean_step_ms"]) for r in rs]
        tab.append([arm, len(rs), " / ".join(r["train_time_s"] for r in rs), f(mean(ts), 4), f(sd(ts), 4), " / ".join(r["val_loss"] for r in rs),
                    f(mean(vs), 6), f(sd(vs), 6), f(mean(ms))])
    o += [md_table(["arm", "n", "train time by seed 1-4 (s)", "mean (s)", "sd", "val loss by seed 1-4", "mean val loss", "sd", "mean step (ms)"], tab)]
    # second session and the step-time pairs over both sessions (seed 4 is left out: its arms ran on different nodes)
    k2 = [r for r in ok if r["experiment"] == "kernel_ab2"]
    if k2:
        o += ["### B2b. Second session (job kernel-ab2, another node; order swapped between seeds)", ""]
        o += [md_table(["order", "arm", "seed", "train time (s)", "val loss", "mean step (ms)"],
                       [[r["order_in_job"], arm_of(r), r["seed"], r["train_time_s"], r["val_loss"], r["mean_step_ms"]]
                        for r in sorted(k2, key=lambda r: int(r["order_in_job"]))]), ""]
        both = {(arm_of(r), r["seed"]): r for r in k + k2 if r["seed"] != "4"}
        tab = []
        for arm in ["gram222s", "skip_matrix"]:
            ds = [(sd_, float(both[(arm, sd_)]["mean_step_ms"]) - float(both[("base", sd_)]["mean_step_ms"]))
                  for sd_ in "12367" if (arm, sd_) in both and ("base", sd_) in both]
            xs = [d for _, d in ds]; m_ = mean(xs); s_ = sd(xs); se = s_ / math.sqrt(len(xs))
            tab.append([f"{arm} - base", len(xs), " / ".join(f"{d:+.3f} (s{q})" for q, d in ds), f(m_, 4), f(s_, 4), f(se, 4), f(m_ / se, 2),
                        f(m_ * 1.193, 3)])
        o += ["Paired difference in mean step time (ms), same seed, both sessions:", "",
              md_table(["difference", "pairs", "by seed", "mean (ms)", "sd", "se", "t", "s per run (x 1193 steps)"], tab), ""]
    o += section_b2c(rows)
    o += ["### B3. Paired by seed (arm - base)", ""]
    bys = {(arm_of(r), r["seed"]): r for r in k}
    tab = []
    for arm in ["gram222s", "skip_matrix"]:
        for label, seeds in [("seeds 1-4", "1234"), ("seeds 1-3 (all on the bring-up node)", "123"), ("seeds 2-4 (drops the node's first run)", "234"), ("seeds 2-3 (same node, no first run)", "23")]:
            pairs = [(bys[(arm, s)], bys[("base", s)]) for s in seeds if (arm, s) in bys and ("base", s) in bys]
            tab.append(diff_row(f"{arm} - base, {label}", [T(a) - T(b) for a, b in pairs], [V(a) - V(b) for a, b in pairs]))
    o += [md_table(DIFF_HEAD + LOSS_HEAD, tab),
          "Seed 4 of gram222s and skip_matrix ran in the `fastest-1` job, about 10 hours after base seed 4 and on a newly",
          "claimed node, so the seed-4 pair is not a same-session pair. Base seed 1 was the first training run on the bring-up node",
          "(first runs are slower: section H), and it is the slowest base run. The skip_matrix loss difference is not meaningful (invalid arm).", ""]
    o += ["### B4. Step latency by schedule stage (mean of the per-run `METRIC step_latency` means, ms)", ""]
    scopes = ["all", "stage0", "stage1", "stage2", "stage3", "stage4"]
    tab = []
    for arm in ["base", "gram222s", "skip_matrix"]:
        rs = [r for r in k if arm_of(r) == arm]
        tab.append([arm] + [f(mean([r["stages"][s]["mean"] for r in rs])) for s in scopes])
    steps_n = [str(k[0]["stages"][s]["n"]) for s in scopes]
    tab.append(["steps counted"] + steps_n)
    o += [md_table(["arm"] + scopes, tab)]

    # --- C. bench
    info, brow, btot = parse_bench()
    o += ["## C. Kernel microbenchmark (`182/optim-kernel/bench.py`, one H100, raw/bringup-1119/bench.log)", "", f"`{info}`", ""]
    tab = []
    for b in brow:
        kind, bank, *rest = b["name"].split(".")
        if kind in ("matrix", "update"):
            tab.append([kind, bank, rest[0], b["shape"], b["base_ms"], b["alt_ms"], b["speedup"],
                        f"{100 * float(b['base_exact_err']):.2f}%" if "base_exact_err" in b else "-",
                        f"{100 * float(b['alt_exact_err']):.2f}%" if "alt_exact_err" in b else "-",
                        f"{100 * float(b['fro_rel_err']):.2f}%"])
    o += [md_table(["what", "bank", "kernel", "shape", "record ms", "alt ms", "speedup", "record err vs exact", "alt err vs exact", "alt vs record (Frobenius)"], tab),
          "`matrix` = the matrix part alone (Grams, polynomials, update GEMMs). `update` = the whole per-bank optimizer update.",
          "Errors are relative Frobenius errors; \"exact\" is the bench's exact reference of the same update rule.", ""]
    tab = [[b["name"].split(".")[1], b["base_ms"], b["alt_ms"], b["speedup"]] for b in brow if b["name"].startswith("ablate.")]
    o += [md_table(["bank", "full update ms", "update with the matrix work skipped ms", "ratio"], tab)]
    share = next((b for b in btot if b.get("kind") == "matrix_share"), None)
    if share:
        o += [f"All three banks: update {share['update_ms_per_step']} ms/step, without the matrix work {share['skip_matrix_ms_per_step']} ms/step, "
              f"so the matrix work is {share['matrix_ms_per_step']} ms/step = {share['matrix_s_per_run_if_exposed']} s over {share['steps']} steps "
              f"if none of it overlaps other work (per bank: {share['per_bank']}).", ""]
    tab = [[b["kind"], b["kernel"], b["banks"], b["base_ms_per_step"], b["alt_ms_per_step"], b["saved_ms_per_step"], b["saved_s_per_run_if_exposed"]]
           for b in btot if b.get("kernel") and b["banks"] == "mlp_bank"]
    o += [md_table(["what", "kernel", "banks switched", "record ms/step (3 banks)", "alt ms/step", "saved ms/step", "saved s/run if fully exposed"], tab)]

    o += section_sweep()
    # --- D. profile
    head, cats, nccl = parse_profile()
    o += ["## D. Where the time goes (torch.profiler on the root record, raw/jobs/fastest-1-*/out/session/profile/root/analysis.txt)", "",
          head, "", md_table(["category", "ms/step", "% of step"], [[c, f(ms), f"{pc:.1f}%"] for c, ms, pc in cats]), nccl, "",
          "Categories overlap across CUDA streams, so they can sum past 100%.", ""]
    late = rglob("jobs/qlabs-*/profile-late/analysis.txt")
    if late:
        head2, cats2, nccl2 = parse_profile(late[0])
        meta = json.load(open(os.path.join(RAW, os.path.dirname(late[0]), "meta_rank0.json")))
        o += [f"### D2. Late window: {meta['count']} steps from step {meta['start']} (stage 2), third session (raw/{late[0]})", "",
              head2, "", md_table(["category", "ms/step", "% of step"], [[c, f(ms), f"{pc:.1f}%"] for c, ms, pc in cats2]), nccl2, "",
              "Same analysis script and categories as D; a different node and a different run from D, so compare shares, not absolute times.", ""]

    # --- E. TABLE_VOTE
    tv = [r for r in ok if r["experiment"] == "table_vote_ab"]
    o += ["## E. TABLE_VOTE=1 on t1-cpu-ngram (675 scheduled steps)", ""]
    o += [md_table(["order", "arm", "seed", "flags", "train time (s)", "val loss"],
                   [[r["order_in_job"], r["arm"], r["seed"], r["flags"] or "(base)", r["train_time_s"], r["val_loss"]] for r in tv])]
    A = {r["seed"]: r for r in tv if r["arm"] == "A"}; B = {r["seed"]: r for r in tv if r["arm"] == "B"}
    seeds = sorted(set(A) & set(B))
    o += [md_table(DIFF_HEAD + LOSS_HEAD, [diff_row("TABLE_VOTE=1 - base", [T(B[s]) - T(A[s]) for s in seeds], [V(B[s]) - V(A[s]) for s in seeds])])]
    votes = [(r["seed"], *r["vote"]) for r in tv if r["vote"]]
    if votes:
        o += ["Same trained weights, three ways to score the validation set (printed by each TABLE_VOTE=1 run):", "",
              md_table(["seed", "val loss with the table's links", "without them", "with PR #399's fixed vote"], votes)]
        d1 = summary([float(a) - float(b) for _, a, b, _ in votes]); d2 = summary([float(c) - float(b) for _, _, b, c in votes])
        o += [f"links - none: {d1['mean']:+.5f} (95% CI ±{d1['ci']:.5f}); fixed vote - none: {d2['mean']:+.5f} (95% CI ±{d2['ci']:.5f}); n={d1['n']}.", ""]

    # --- F. steps
    cpu = [r for r in ok if r["tree"].endswith("t1-cpu-ngram") and "TABLE_VOTE" not in r["flags"]]
    o += ["## F. Fewer scheduled steps on t1-cpu-ngram", "", "### F1. The step sweep job (`ngram-steps`), in run order", ""]
    sw = [r for r in ok if r["experiment"] == "step_sweep"]
    o += [md_table(["order", "scheduled steps", "total steps", "seed", "train time (s)", "val loss", "note"],
                   [[r["order_in_job"], r["sched_steps"], r["steps"], r["seed"], r["train_time_s"], r["val_loss"], r["note"]] for r in sw])]
    o += ["### F2. Val loss by scheduled steps (every valid run of that step count, all jobs)", ""]
    tab = []
    for n in sorted({r["sched_steps"] for r in cpu}, reverse=True):
        rs = [r for r in cpu if r["sched_steps"] == n]
        vs = [V(r) for r in rs]; s = summary([3.28 - v for v in vs])
        p1 = 1 - t_cdf(s["t"], s["n"] - 1) if s["n"] > 1 else float("nan")
        tab.append([n, n + EXT_STEPS, len(rs), len({r["seed"] for r in rs}), f(mean(vs), 5), f(sd(vs), 5), f(min(vs), 4), f(max(vs), 4),
                    f(s["mean"], 5), f(s["t"], 2), f"{p1:.2g}", sum(r["hit_target"] for r in rs)])
    o += [md_table(["scheduled steps", "total steps", "runs", "distinct seeds", "mean val loss", "sd", "best", "worst", "margin under 3.28",
                    "t (margin / se)", "p (one-sided, mean < 3.28)", "runs at or under 3.28"], tab),
          "This tree logs the val loss to 4 decimals only. The one-sided test treats runs as independent draws.", ""]

    # --- G. pairs
    pr = [r for r in ok if r["experiment"] == "steps_pairs"]
    o += ["## G. 675 vs 668 scheduled steps, interleaved pairs (same seed, back to back, order alternating)", ""]
    pairs = []
    for job in sorted({r["job"] for r in pr}, key=lambda j: min(r["start_utc"] for r in pr if r["job"] == j)):
        for seed in sorted({r["seed"] for r in pr if r["job"] == job}, key=int):
            a = next(r for r in pr if r["job"] == job and r["seed"] == seed and r["sched_steps"] == 675)
            b = next(r for r in pr if r["job"] == job and r["seed"] == seed and r["sched_steps"] == 668)
            pairs.append((job, seed, a, b))
    o += [md_table(["job", "seed", "ran first", "675: time (s)", "675: val loss", "668: time (s)", "668: val loss", "Δtime 668-675 (s)", "Δval loss 668-675"],
                   [[j, s, "675" if a["order_in_job"] < b["order_in_job"] else "668", a["train_time_s"], a["val_loss"], b["train_time_s"], b["val_loss"],
                     f"{T(b) - T(a):+.3f}", f"{V(b) - V(a):+.4f}"] for j, s, a, b in pairs])]
    tab = []
    for job in dict.fromkeys(j for j, *_ in pairs):
        ps = [(a, b) for j, s, a, b in pairs if j == job]
        tab.append(diff_row(f"668 - 675, {job}", [T(b) - T(a) for a, b in ps], [V(b) - V(a) for a, b in ps]))
    tab.append(diff_row("668 - 675, all pairs", [T(b) - T(a) for _, _, a, b in pairs], [V(b) - V(a) for _, _, a, b in pairs]))
    firsts = {min((a["order_in_job"], s) for j, s, a, b in pairs if j == job)[1] + "@" + job for job in dict.fromkeys(j for j, *_ in pairs)}
    rest = [(a, b) for j, s, a, b in pairs if s + "@" + j not in firsts]
    tab.append(diff_row("668 - 675, without each job's first pair (job positions 2-3)", [T(b) - T(a) for a, b in rest], [V(b) - V(a) for a, b in rest]))
    o += [md_table(DIFF_HEAD + LOSS_HEAD, tab)]
    lp = mean([float(x["loop_ms_per_step"]) for _, _, a, b in pairs for x in (a, b)])
    o += [f"Expected from the step count alone: 7 fewer steps x {lp:.3f} ms/step (mean loop ms/step of the pair runs) = {7 * lp / 1000:.3f} s.", ""]
    first675 = [T(b) - T(a) for _, _, a, b in pairs if a["order_in_job"] < b["order_in_job"]]
    first668 = [T(b) - T(a) for _, _, a, b in pairs if a["order_in_job"] > b["order_in_job"]]
    if first675 and first668:
        o += [f"By order: 675 ran first in {len(first675)} pairs (mean Δtime {mean(first675):+.3f} s), 668 ran first in {len(first668)} pairs "
              f"(mean Δtime {mean(first668):+.3f} s).", ""]
    a675 = [T(a) for _, _, a, b in pairs]; b668 = [T(b) for _, _, a, b in pairs]
    lp675 = [float(a["loop_ms_per_step"]) for _, _, a, b in pairs]; lp668 = [float(b["loop_ms_per_step"]) for _, _, a, b in pairs]
    o += [f"In the pairs: 675 mean {mean(a675):.3f} s (sd {sd(a675):.3f}), 668 mean {mean(b668):.3f} s (sd {sd(b668):.3f}); "
          f"loop ms/step {mean(lp675):.3f} vs {mean(lp668):.3f}.", ""]
    inv = [r for r in rows if not r["valid"] and r["experiment"] != "curve3"]   # the curve3 failures are listed in B2c
    if inv:
        o += ["Invalid / failed runs (kept in runs.csv with valid=0):", ""] + [f"- `{r['job']}` {r['log']}: {r['note']}" for r in inv] + [""]

    # --- H. run-order drift
    o += ["## H. Run-to-run drift inside a job (t1-cpu-ngram, 675-step runs by position in the job)", ""]
    pos = defaultdict(list)
    for r in cpu:
        if r["sched_steps"] == 675: pos[r["order_in_tree"]].append(T(r))
    o += [md_table(["position of the run in its job", "runs", "mean time (s)", "min", "max"],
                   [[p, len(ts), f(mean(ts)), f(min(ts)), f(max(ts))] for p, ts in sorted(pos.items())])]

    # --- I. track 2
    t2 = [r for r in ok if r["track"] == "2"]
    o += ["## I. Track 2 (variants/t2-cpu-ngram)", "",
          md_table(["job", "steps", "train time (s)", "val loss", "target", "loop ms/step", "peak alloc (MiB)", "peak reserved (MiB)"],
                   [[r["job"], r["steps"], r["train_time_s"], r["val_loss"], TARGET["2"], r["loop_ms_per_step"], r["peak_alloc_mib"], r["peak_reserved_mib"]] for r in t2])]

    # --- J. jobs
    jobs = parse_jobs()
    paid = [j for j in jobs if j[6] is not None]; free = [j for j in jobs if j[6] is None]
    o += ["## J. Jobs (raw/jobs/*/job.json)", "",
          md_table(["job", "provider", "state", "created (UTC)", "last observed (UTC)", "GPU-hours", "cost (USD)"],
                   [[*j[:5], j[5] if j[5] is not None else "-", j[6] if j[6] is not None else "- (not billed in money)"] for j in jobs]),
          f"Total: {sum(float(j[5]) for j in paid):.2f} GPU-hours, ${sum(float(j[6]) for j in paid):.2f} (the interactive bring-up node is not in this list).", ""]
    if free:
        o += [f"Jobs without a monetary spend ({', '.join(j[0] for j in free)}): {sum(float(j[5]) for j in free if j[5] is not None):.2f} GPU-hours, "
              "computed from the job's own start and end lines (see the job.json note); they are not in the total above.", ""]
    return "\n".join(o), pairs

# ---------------------------------------------------------------- figures

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3 (fixed order)

def figures(rows, pairs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"figures skipped (matplotlib not available: {e})"); return []
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
    made = []
    def save(fig, name):
        fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=170); plt.close(fig); made.append(name)
    ok = [r for r in rows if r["valid"]]

    # 1. kernel A/B: time per run by arm
    k = [r for r in ok if r["experiment"] == "kernel_ab"]
    arms = [("", "base (record)", C1), ("ANVIL_KERNEL=gram222s", "gram222s (ours)", C2), ("ANVIL_ABLATE=skip_matrix", "skip_matrix\n(ceiling, not a valid run)", C3)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (flag, label, col) in enumerate(arms):
        rs = [r for r in k if r["flags"] == flag]
        for r in rs:
            same = r["job"].startswith("bringup")
            ax.plot(i + (int(r["seed"]) - 2.5) * 0.09, T(r), "o", ms=8, mfc=col if same else SURF, mec=col, mew=1.6, zorder=3)
            ax.annotate(r["seed"], (i + (int(r["seed"]) - 2.5) * 0.09, T(r)), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK2)
        m = mean([T(r) for r in rs])
        ax.hlines(m, i - 0.3, i + 0.3, color=INK, lw=1.4, zorder=2)
        ax.annotate(f"mean {m:.3f} s", (i + 0.32, m), va="center", fontsize=9, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([a[1] for a in arms]); ax.set_xlim(-0.5, 2.95)
    ax.set_ylabel("train time (s)"); ax.grid(axis="x", visible=False)
    ax.set_title("Root record (1194 steps): train time per run, by kernel arm")
    ax.text(0.0, -0.2, "Numbers above the dots are seeds. Hollow dots ran in a later job on a newly claimed node.", transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "kernel_ab_time.png")

    # 2. kernel bench
    _, brow, _ = parse_bench()
    banks = ["qk_bank", "vo_bank", "mlp_bank"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, bank in enumerate(banks):
        b = next(x for x in brow if x["name"] == f"matrix.{bank}.gram222s")
        for j, (key, col) in enumerate([("base_ms", C1), ("alt_ms", C2)]):
            v = float(b[key]); x = i + (j - 0.5) * 0.36
            ax.bar(x, v, width=0.32, color=col, zorder=3)
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK)
        ax.annotate(f"{float(b['speedup']):.2f}x", (i, max(float(b["base_ms"]), float(b["alt_ms"]))), textcoords="offset points", xytext=(0, 16), ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.bar(0, 0, color=C1, label="record kernel"); ax.bar(0, 0, color=C2, label="gram222s (ours)")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{b}\n{next(x for x in brow if x['name'] == f'matrix.{b}.gram222s')['shape']}" for b in banks])
    ax.set_ylabel("matrix step, one bank (ms)"); ax.set_ylim(0, 0.5); ax.grid(axis="x", visible=False); ax.legend(loc="upper left")
    ax.set_title("Optimizer matrix step in isolation (one H100, 300 reps)")
    ax.text(0.0, -0.2, "Bold number = record time / our time (above 1 means ours is faster).", transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "kernel_bench.png")

    # 3. paired differences 668 - 675
    if pairs:
        d = [T(b) - T(a) for _, _, a, b in pairs]; s = summary(d)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        jobs = list(dict.fromkeys(j for j, *_ in pairs))
        for i, (j, seed, a, b) in enumerate(pairs):
            first675 = a["order_in_job"] < b["order_in_job"]
            ax.plot(i, d[i], "o", ms=8, color=C1 if jobs.index(j) == 0 else C2, mfc=None if first675 else SURF, mew=1.6, zorder=3)
        ax.axhline(0, color=INK2, lw=1)
        ax.fill_between([-0.6, len(d) - 0.5], s["mean"] - s["ci"], s["mean"] + s["ci"], color=GRID, alpha=0.7, lw=0, zorder=1)
        ax.hlines(s["mean"], -0.6, len(d) - 0.5, color=INK, lw=1.4, zorder=2)
        ax.annotate(f"mean {s['mean']:+.3f} s\n95% CI ±{s['ci']:.3f}\n(grey band)", (len(d) - 0.4, s["mean"]), va="center", fontsize=9)
        ax.set_xticks(range(len(d))); ax.set_xticklabels([f"seed {s_}" for _, s_, _, _ in pairs], fontsize=8); ax.set_xlim(-0.6, len(d) + 1.5)
        for ji, j in enumerate(jobs):
            ax.plot([], [], "o", color=[C1, C2][ji], label=j.rsplit("-", 2)[0])
        ax.plot([], [], "o", mfc=SURF, mec=INK2, mew=1.6, label="hollow: 668 ran first")
        ax.legend(loc="upper left", fontsize=8, ncol=3); ax.grid(axis="x", visible=False)
        lo, hi = min(d + [0]), max(d + [0]); ax.set_ylim(lo - 0.05, hi + 0.12)
        ax.set_ylabel("train time, 668 minus 675 steps (s)")
        ax.set_title("t1-cpu-ngram: 668 vs 675 scheduled steps, paired differences")
        save(fig, "steps_pairs_diff.png")

    # 4. val loss by scheduled steps
    cpu = [r for r in ok if r["tree"].endswith("t1-cpu-ngram") and "TABLE_VOTE" not in r["flags"]]
    ns = sorted({r["sched_steps"] for r in cpu})
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, n in enumerate(ns):
        vs = sorted(V(r) for r in cpu if r["sched_steps"] == n)
        cnt = defaultdict(int)
        for v in vs:
            ax.plot(i - 0.12 + 0.05 * cnt[v], v, "o", ms=6, color=C1, mec=SURF, mew=0.8, zorder=3); cnt[v] += 1
        ax.hlines(mean(vs), i - 0.25, i + 0.2, color=INK, lw=1.4)
        ax.annotate(f"mean {mean(vs):.5f}\nn={len(vs)}", (i + 0.23, mean(vs)), va="center", fontsize=9)
    ax.axhline(3.28, color=INK2, lw=1, ls="--"); ax.annotate("target 3.28", (-0.48, 3.28), textcoords="offset points", xytext=(0, 4), fontsize=9, color=INK2)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"{n} scheduled\n({n + EXT_STEPS} total)" + ("\nupstream" if n == 675 else "") for n in ns])
    ax.set_xlim(-0.5, len(ns) - 0.2); ax.set_ylabel("final val loss (4 decimals as logged)"); ax.grid(axis="x", visible=False)
    ax.set_title("t1-cpu-ngram: final val loss by step count (each dot is one run)")
    save(fig, "val_loss_by_steps.png")

    # 5. run-order drift
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for n, col in [(675, C1), (668, C2), (662, C3)]:
        rs = [r for r in cpu if r["sched_steps"] == n]
        ax.plot([r["order_in_tree"] for r in rs], [T(r) for r in rs], "o", ms=7, color=col, mec=SURF, mew=1, label=f"{n} scheduled steps", zorder=3)
    ax.set_xlabel("position of the run in its job (1 = first run on a fresh node)"); ax.set_ylabel("train time (s)")
    ax.legend(loc="upper right"); ax.set_title("t1-cpu-ngram: the first runs of a job are slower")
    save(fig, "time_by_run_order.png")

    # 6. profile
    head, cats, _ = parse_profile()
    if cats:
        cats2 = sorted(cats, key=lambda c: c[1])
        fig, ax = plt.subplots(figsize=(8.6, 3.9))
        ax.barh(range(len(cats2)), [c[1] for c in cats2], height=0.62, color=C1, zorder=3)
        for i, c in enumerate(cats2):
            ax.annotate(f"{c[1]:.2f} ms  ({c[2]:.1f}%)", (c[1], i), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8)
        ax.set_yticks(range(len(cats2))); ax.set_yticklabels([c[0] for c in cats2], fontsize=8)
        ax.set_xlim(0, max(c[1] for c in cats2) * 1.3); ax.set_xlabel("GPU kernel time per step (ms). Categories overlap across streams.")
        ax.grid(axis="y", visible=False)
        w = re.search(r"GPU window ([\d.]+) ms/step", head)
        ax.set_title(f"Root record, steps 300-319 ({w[1]} ms per step)" if w else "Root record: where a step goes")
        save(fig, "profile_breakdown.png")

    # 7. training clock vs step, Track 1 trees
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pick = [("root (record #92)", lambda r: r["experiment"] == "kernel_ab" and not r["flags"], C1, "root record #92"),
            ("variants/t1-gpu-stack", lambda r: True, C2, "t1-gpu-stack"),
            ("variants/t1-cpu-ngram", lambda r: r.get("sched_steps") == 675 and r["experiment"] == "steps_pairs", C3, "t1-cpu-ngram (675)")]
    for tree, cond, col, label in pick:
        rs = [r for r in ok if r["tree"] == tree and cond(r) and r["curve"]]
        if not rs: continue
        r = rs[-1]
        xs, ys = [0] + [c[0] for c in r["curve"]], [0] + [c[1] for c in r["curve"]]
        ax.plot(xs, ys, color=col, lw=2)
        ax.plot(r["steps"], T(r), "o", ms=8, color=col, mec=SURF, mew=1.5, zorder=3)
        ax.annotate(f"{label}\n{r['steps']} steps, {T(r):.2f} s, val {r['val_loss']}", (r["steps"], T(r)), textcoords="offset points",
                    xytext=(8, -4), fontsize=8, va="center")
    ax.set_xlim(0, 1700); ax.set_xlabel("step"); ax.set_ylabel("training clock (s)")
    ax.set_title("Training clock vs step, one run per Track 1 tree")
    ax.text(0.0, -0.2, "Dot = the final validation, the only validation these runs do (no mid-run loss was logged).", transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "train_time_vs_step.png")

    # 8. step latency by stage
    scopes = ["stage0", "stage1", "stage2", "stage3", "stage4"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for j, (flag, label, col) in enumerate(arms):
        rs = [r for r in k if r["flags"] == flag]
        ax.bar([i + (j - 1) * 0.27 for i in range(len(scopes))], [mean([r["stages"][s]["mean"] for r in rs]) for s in scopes],
               width=0.24, color=col, label=label.split("\n")[0], zorder=3)
    n = k[0]["stages"]
    ax.set_xticks(range(len(scopes))); ax.set_xticklabels([f"{s}\n{n[s]['n']} steps" for s in scopes])
    ax.set_ylabel("mean step latency (ms)"); ax.legend(loc="upper left", fontsize=8); ax.grid(axis="x", visible=False)
    ax.set_title("Root record: step latency by schedule stage (4 runs per arm)")
    save(fig, "step_latency_by_stage.png")
    return made

# ---------------------------------------------------------------- main

def main():
    if not os.path.isdir(RAW): sys.exit("raw/ is missing")
    rows = collect_runs()
    with open(os.path.join(HERE, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow({c: r.get(c, "") for c in COLS})
    text, pairs = build_tables(rows)
    with open(os.path.join(HERE, "tables.md"), "w") as fh: fh.write(text)
    made = figures(rows, pairs)
    print(f"runs.csv: {len(rows)} runs ({sum(1 for r in rows if not r['valid'])} invalid); tables.md written; figures: {', '.join(made) or 'none'}")

if __name__ == "__main__":
    main()
