#!/usr/bin/env python3
"""Per-step GPU timeline analysis of an Nsight Systems trace, from its sqlite export.

    nsys export --type sqlite -o rank0.sqlite rank0.nsys-rep
    python 182/tools/analyze_nsys.py rank0.sqlite [--gemm-shapes shapes.json] [--json out.json]

Reports, per profiled step and averaged:
  - kernel count (and how many came from CUDA graphs);
  - GPU busy time (union of kernel intervals on the device) and idle time = window - busy, with the
    largest gaps and the kernels on either side of them;
  - NCCL time and exposed NCCL time: NCCL kernel time not overlapped by any non-NCCL kernel;
  - top kernels by total time;
  - achieved TFLOP/s for GEMM kernels whose shapes are recoverable (see below).

Steps. With profile_nsys.sh's hook, every profiled training step is an NVTX range "step N". Kernels
are assigned to the step during which their launch API call (cudaLaunchKernel, cudaGraphLaunch, ...)
was made -- the usual NVTX-to-GPU projection, which works for CUDA-graph replays because graph
kernels carry the graph launch's correlation id. Step i's GPU window runs from its first kernel's
start to the next step's first kernel's start (for the last step: to the first kernel launched after
its NVTX range, else its last kernel's end), so the windows tile the timeline and inter-step bubbles
count as idle. Without NVTX steps, --step-marker REGEX starts a new step at each kernel whose name
matches (e.g. the first kernel of the forward).

GEMM shapes. Kernel records carry no problem sizes. They are recovered (1) from enclosing NVTX ranges
of the form "aten::mm, ..., sizes = [[M, K], [K, N]]" (torch.autograd.profiler.emit_nvtx(record_shapes=True);
eager trainers only -- CUDA-graph replays have no CPU ops), or (2) from --gemm-shapes, a JSON list of
{"match": REGEX, "mnk": [M, N, K], "batch": 1} or {"match": REGEX, "flops": F} applied per launch,
first match wins. --list-gemms prints the GEMM kernels with grid/block dims to help write that file.
GEMMs with no recoverable shape are reported with time only.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import json
import re
import sqlite3
import sys
from collections import defaultdict

NVTX_PUSHPOP, NVTX_STARTEND = 59, 60
GEMM_RE = r"(?i)gemm|xmma|nvjet|cutlass.*(gemm|kernel_?mma)|matmul|_scaled_mm|triton_tem_fused_.*mm|\bmm_|cublas"
GEMM_EXCLUDE_RE = r"(?i)flash|attn|attention|nccl|softmax|reduce"
NCCL_RE = r"(?i)nccl"
STEP_RE = r"^step[ :_#]?(\d+)$"
ATEN_MM_RE = re.compile(r"^aten::(mm|addmm|bmm|baddbmm|matmul|linear|_scaled_mm|_scaled_grouped_mm)\b")


# ----------------------------------------------------------------------------- intervals

def merge(iv):
    """Union of [start, end) intervals -> sorted, disjoint list."""
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def length(merged):
    return sum(e - s for s, e in merged)


def clip(merged, lo, hi):
    return [[max(s, lo), min(e, hi)] for s, e in merged if e > lo and s < hi]


def intersect_len(a, b):
    """Total overlap of two merged interval lists."""
    i = j = 0
    tot = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def gaps(merged, lo, hi):
    """Idle gaps of a merged busy list inside [lo, hi): [(start, end)]."""
    out, cur = [], lo
    for s, e in merged:
        if s > cur:
            out.append((cur, min(s, hi)))
        cur = max(cur, e)
        if cur >= hi:
            break
    if cur < hi:
        out.append((cur, hi))
    return [g for g in out if g[1] > g[0]]


# ----------------------------------------------------------------------------- loading

def columns(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def has_table(con, table):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def load_strings(con):
    if not has_table(con, "StringIds"):
        return {}
    return dict(con.execute("SELECT id, value FROM StringIds"))


def load_kernels(con, strings):
    cols = columns(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    want = ["start", "end", "deviceId", "streamId", "correlationId", "globalPid", "demangledName", "shortName",
            "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "graphNodeId"]
    sel = ", ".join(c if c in cols else f"NULL AS {c}" for c in want)
    ks = []
    for row in con.execute(f"SELECT {sel} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
        d = dict(zip(want, row))
        d["name"] = strings.get(d["demangledName"]) or strings.get(d["shortName"]) or str(d["demangledName"])
        d["short"] = strings.get(d["shortName"]) or d["name"]
        ks.append(d)
    return ks


def load_runtime(con, strings):
    if not has_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return {}
    out = {}
    for start, end, corr, tid, name_id in con.execute(
            "SELECT start, end, correlationId, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        pid = (tid or 0) & ~0xFFFFFF
        out[(pid, corr)] = (start, end, tid, strings.get(name_id))
        out.setdefault((None, corr), (start, end, tid, strings.get(name_id)))
    return out


def load_nvtx(con, strings):
    if not has_table(con, "NVTX_EVENTS"):
        return []
    cols = columns(con, "NVTX_EVENTS")
    text = "text" if "text" in cols else "NULL"
    text_id = "textId" if "textId" in cols else "NULL"
    out = []
    for start, end, et, t, tid_, gtid in con.execute(
            f"SELECT start, end, eventType, {text}, {text_id}, globalTid FROM NVTX_EVENTS "
            f"WHERE eventType IN ({NVTX_PUSHPOP}, {NVTX_STARTEND}) AND end IS NOT NULL ORDER BY start"):
        label = t if t is not None else strings.get(tid_)
        if label is not None:
            out.append(dict(start=start, end=end, text=label, globalTid=gtid))
    return out


def load_memcpy(con):
    if not has_table(con, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return []
    return [dict(start=s, end=e, deviceId=d, bytes=b, copyKind=k) for s, e, d, b, k in con.execute(
        "SELECT start, end, deviceId, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY")]


# ----------------------------------------------------------------------------- GEMM flops

def flops_from_sizes(op: str, sizes) -> float | None:
    try:
        if op in ("mm", "_scaled_mm"):
            (m, k), (_, n) = sizes[0], sizes[1]
            return 2.0 * m * k * n
        if op == "addmm":
            (m, k), (_, n) = sizes[1], sizes[2]
            return 2.0 * m * k * n
        if op in ("bmm",):
            (b, m, k), (_, _, n) = sizes[0], sizes[1]
            return 2.0 * b * m * k * n
        if op == "baddbmm":
            (b, m, k), (_, _, n) = sizes[1], sizes[2]
            return 2.0 * b * m * k * n
        if op == "linear":
            x, w = sizes[0], sizes[1]
            rows = 1
            for d in x[:-1]:
                rows *= d
            return 2.0 * rows * x[-1] * w[0]
        if op == "matmul":
            a, b = sizes[0], sizes[1]
            rows = 1
            for d in a[:-1]:
                rows *= d
            return 2.0 * rows * a[-1] * b[-1]
    except (ValueError, IndexError, TypeError):
        return None
    return None


def parse_aten_nvtx(text: str):
    m = ATEN_MM_RE.match(text)
    if not m:
        return None
    sm = re.search(r"sizes = (\[.*?\]\])", text)
    if not sm:
        return None
    try:
        sizes = ast.literal_eval(sm.group(1))
    except (ValueError, SyntaxError):
        return None
    return flops_from_sizes(m.group(1), sizes)


def shape_rule_flops(rules, name):
    for r in rules:
        if re.search(r["match"], name):
            if "flops" in r:
                return float(r["flops"])
            m, n, k = r["mnk"]
            return 2.0 * m * n * k * r.get("batch", 1)
    return None


# ----------------------------------------------------------------------------- analysis

def analyze(path: str, device=None, step_regex=STEP_RE, step_marker=None, gemm_regex=GEMM_RE,
            gemm_exclude=GEMM_EXCLUDE_RE, nccl_regex=NCCL_RE, gemm_rules=None, gap_threshold_us=2.0,
            top=25, top_gaps=10, peak_tflops=None) -> dict:
    con = sqlite3.connect(path)
    strings = load_strings(con)
    kernels = load_kernels(con, strings)
    if not kernels:
        raise SystemExit(f"{path}: no CUPTI_ACTIVITY_KIND_KERNEL rows (was CUDA tracing on?)")
    runtime = load_runtime(con, strings)
    nvtx = load_nvtx(con, strings)
    memcpy = load_memcpy(con)
    con.close()

    by_dev = defaultdict(int)
    for k in kernels:
        by_dev[k["deviceId"]] += 1
    dev = device if device is not None else max(by_dev, key=by_dev.get)
    ks = [k for k in kernels if k["deviceId"] == dev]
    for k in ks:
        rt = runtime.get((k["globalPid"], k["correlationId"])) or runtime.get((None, k["correlationId"]))
        k["launch"] = rt[0] if rt else None
        k["launch_tid"] = rt[2] if rt else None
        k["api"] = rt[3] if rt else None

    # --- steps
    step_re = re.compile(step_regex)
    step_ranges = [r for r in nvtx if step_re.match(r["text"])]
    steps = []
    if step_ranges and not step_marker:
        mode = "nvtx"
        for r in step_ranges:
            members = [k for k in ks if k["launch"] is not None and r["start"] <= k["launch"] < r["end"]]
            if members:
                steps.append(dict(label=r["text"], cpu_start=r["start"], cpu_end=r["end"], kernels=members))
        for i, st in enumerate(steps):
            st["gpu_start"] = min(k["start"] for k in st["kernels"])
            if i + 1 < len(steps):
                st["gpu_end"] = min(k["start"] for k in steps[i + 1]["kernels"])
            else:
                later = [k["start"] for k in ks if k["launch"] is not None and k["launch"] >= st["cpu_end"]]
                st["gpu_end"] = min(later) if later else max(k["end"] for k in st["kernels"])
    elif step_marker:
        mode = "marker"
        mre = re.compile(step_marker)
        starts = [k["start"] for k in ks if mre.search(k["name"])]
        if len(starts) < 2:
            raise SystemExit(f"--step-marker {step_marker!r} matched {len(starts)} kernels; need >= 2")
        for i in range(len(starts) - 1):
            lo, hi = starts[i], starts[i + 1]
            steps.append(dict(label=f"marker step {i}", gpu_start=lo, gpu_end=hi,
                              kernels=[k for k in ks if lo <= k["start"] < hi]))
    else:
        mode = "whole"
        lo, hi = min(k["start"] for k in ks), max(k["end"] for k in ks)
        steps.append(dict(label="whole trace", gpu_start=lo, gpu_end=hi, kernels=list(ks)))

    gemm_re, gemm_ex, nccl_re = re.compile(gemm_regex), re.compile(gemm_exclude), re.compile(nccl_regex)
    for k in ks:
        k["is_nccl"] = bool(nccl_re.search(k["name"]))
        k["is_gemm"] = (not k["is_nccl"]) and bool(gemm_re.search(k["name"])) and not gemm_ex.search(k["name"])

    # --- GEMM flops: NVTX aten ranges on the launching thread, innermost first; then the rule file
    aten = [dict(r, flops=parse_aten_nvtx(r["text"])) for r in nvtx if ATEN_MM_RE.match(r["text"])]
    aten = [r for r in aten if r["flops"]]
    aten_by_tid = defaultdict(list)
    for r in aten:
        aten_by_tid[r["globalTid"]].append(r)
    for k in ks:
        k["flops"], k["flops_src"] = None, None
        if not k["is_gemm"]:
            continue
        if k["launch"] is not None:
            enclosing = [r for r in aten_by_tid.get(k["launch_tid"], []) if r["start"] <= k["launch"] < r["end"]]
            if enclosing:
                k["flops"] = min(enclosing, key=lambda r: r["end"] - r["start"])["flops"]
                k["flops_src"] = "nvtx"
        if k["flops"] is None and gemm_rules:
            f = shape_rule_flops(gemm_rules, k["name"])
            if f is not None:
                k["flops"], k["flops_src"] = f, "rules"

    # --- per step
    all_busy = merge([(k["start"], k["end"]) for k in ks])
    nccl_busy = merge([(k["start"], k["end"]) for k in ks if k["is_nccl"]])
    comp_busy = merge([(k["start"], k["end"]) for k in ks if not k["is_nccl"]])
    by_start = sorted(ks, key=lambda k: k["start"])
    starts = [k["start"] for k in by_start]
    by_end = sorted(ks, key=lambda k: k["end"])
    ends = [k["end"] for k in by_end]

    def kernel_before(t):  # the kernel that ended last at or before t
        i = bisect.bisect_right(ends, t)
        return by_end[i - 1] if i else None

    def kernel_after(t):  # the first kernel starting at or after t
        i = bisect.bisect_left(starts, t)
        return by_start[i] if i < len(by_start) else None

    per_step, gap_list = [], []
    for st in steps:
        lo, hi = st["gpu_start"], st["gpu_end"]
        win = hi - lo
        busy = clip(all_busy, lo, hi)
        g = gaps(busy, lo, hi)
        n_cl = clip(nccl_busy, lo, hi)
        c_cl = clip(comp_busy, lo, hi)
        nccl_t = length(n_cl)
        exposed = nccl_t - intersect_len(n_cl, c_cl)
        members = st["kernels"]
        gemm_t = sum(k["end"] - k["start"] for k in members if k["is_gemm"])
        cps = [m for m in memcpy if m["deviceId"] == dev and lo <= m["start"] < hi]
        big = [x for x in g if (x[1] - x[0]) >= gap_threshold_us * 1000]
        per_step.append(dict(
            label=st["label"], window_us=win / 1e3, kernels=len(members),
            graph_kernels=sum(1 for k in members if k["graphNodeId"] is not None),
            busy_us=length(busy) / 1e3, idle_us=(win - length(busy)) / 1e3,
            gaps_over_threshold=len(big), idle_in_gaps_over_threshold_us=sum(b - a for a, b in big) / 1e3,
            largest_gap_us=max((b - a for a, b in g), default=0) / 1e3,
            nccl_us=nccl_t / 1e3, exposed_nccl_us=exposed / 1e3, gemm_us=gemm_t / 1e3,
            memcpy=len(cps), memcpy_us=sum(m["end"] - m["start"] for m in cps) / 1e3,
            memcpy_bytes=sum(m["bytes"] or 0 for m in cps)))
        for a, b in g:
            prev, nxt = kernel_before(a), kernel_after(b)
            gap_list.append(dict(step=st["label"], us=(b - a) / 1e3, start_ns=a,
                                 after=prev["short"] if prev else None, before=nxt["short"] if nxt else None))

    n = len(per_step)
    keys = [k for k in per_step[0] if k != "label"] if per_step else []
    mean = {k: sum(s[k] for s in per_step) / n for k in keys} if n else {}

    # --- top kernels over the profiled steps
    step_kernels = [k for st in steps for k in st["kernels"]]
    total_kernel_ns = sum(k["end"] - k["start"] for k in step_kernels)  # overlap counted per kernel
    agg = defaultdict(lambda: dict(count=0, ns=0, gemm=False, nccl=False, flops=0.0, flops_ns=0))
    for k in step_kernels:
        a = agg[k["short"]]
        a["count"] += 1
        a["ns"] += k["end"] - k["start"]
        a["gemm"] |= k["is_gemm"]
        a["nccl"] |= k["is_nccl"]
        if k["flops"]:
            a["flops"] += k["flops"]
            a["flops_ns"] += k["end"] - k["start"]
    top_k = sorted(agg.items(), key=lambda kv: -kv[1]["ns"])[:top]
    top_rows = [dict(name=nm, count=a["count"], per_step=a["count"] / max(n, 1), total_us=a["ns"] / 1e3,
                     avg_us=a["ns"] / a["count"] / 1e3, pct_time=100.0 * a["ns"] / total_kernel_ns if total_kernel_ns else 0.0,
                     kind="nccl" if a["nccl"] else ("gemm" if a["gemm"] else ""),
                     tflops=(a["flops"] / a["flops_ns"] / 1e3) if a["flops_ns"] else None) for nm, a in top_k]

    gemms = [k for k in step_kernels if k["is_gemm"]]
    shaped = [k for k in gemms if k["flops"]]
    gemm_rows = []
    gagg = defaultdict(lambda: dict(count=0, ns=0, flops=0.0, src=set()))
    for k in shaped:
        a = gagg[k["short"]]
        a["count"] += 1
        a["ns"] += k["end"] - k["start"]
        a["flops"] += k["flops"]
        a["src"].add(k["flops_src"])
    for nm, a in sorted(gagg.items(), key=lambda kv: -kv[1]["ns"]):
        tf = a["flops"] / a["ns"] / 1e3
        gemm_rows.append(dict(name=nm, count=a["count"], total_us=a["ns"] / 1e3, tflops=tf,
                              pct_peak=100.0 * tf / peak_tflops if peak_tflops else None, source=",".join(sorted(a["src"]))))
    shaped_ns = sum(k["end"] - k["start"] for k in shaped)
    gemm_summary = dict(
        gemm_kernels=len(gemms), with_shapes=len(shaped),
        gemm_us=sum(k["end"] - k["start"] for k in gemms) / 1e3, shaped_us=shaped_ns / 1e3,
        achieved_tflops=(sum(k["flops"] for k in shaped) / shaped_ns / 1e3) if shaped_ns else None)

    return dict(file=path, device=dev, devices={str(d): c for d, c in by_dev.items()}, step_mode=mode, steps=per_step,
                mean=mean, top_kernels=top_rows, top_gaps=sorted(gap_list, key=lambda g: -g["us"])[:top_gaps],
                gemm=gemm_summary, gemm_table=gemm_rows)


def list_gemms(path, device=None, gemm_regex=GEMM_RE, gemm_exclude=GEMM_EXCLUDE_RE):
    con = sqlite3.connect(path)
    strings = load_strings(con)
    ks = load_kernels(con, strings)
    con.close()
    g, ex = re.compile(gemm_regex), re.compile(gemm_exclude)
    rows = defaultdict(lambda: [0, 0])
    for k in ks:
        if (device is None or k["deviceId"] == device) and g.search(k["name"]) and not ex.search(k["name"]) \
                and not re.search(NCCL_RE, k["name"]):
            key = (k["name"], (k["gridX"], k["gridY"], k["gridZ"]), (k["blockX"], k["blockY"], k["blockZ"]))
            rows[key][0] += 1
            rows[key][1] += k["end"] - k["start"]
    return [dict(name=n, grid=gr, block=bl, count=c, avg_us=t / c / 1e3) for (n, gr, bl), (c, t) in
            sorted(rows.items(), key=lambda kv: -kv[1][1])]


# ----------------------------------------------------------------------------- printing

def render(res: dict) -> str:
    L = [f"# nsys analysis: {res['file']}", "",
         f"device {res['device']} (kernels per device: {res['devices']}); steps from {res['step_mode']}; "
         f"{len(res['steps'])} step(s)", ""]
    L += ["| step | window ms | kernels (graph) | busy ms | idle ms | gaps>thr (idle ms) | largest gap us | NCCL ms | exposed NCCL ms | GEMM ms | memcpy (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = res["steps"] + ([dict(label="**mean**", **res["mean"])] if len(res["steps"]) > 1 else [])
    for s in rows:
        L.append(f"| {s['label']} | {s['window_us']/1e3:.3f} | {s['kernels']:.0f} ({s['graph_kernels']:.0f}) | "
                 f"{s['busy_us']/1e3:.3f} | {s['idle_us']/1e3:.3f} | {s['gaps_over_threshold']:.1f} ({s['idle_in_gaps_over_threshold_us']/1e3:.3f}) | "
                 f"{s['largest_gap_us']:.1f} | {s['nccl_us']/1e3:.3f} | {s['exposed_nccl_us']/1e3:.3f} | {s['gemm_us']/1e3:.3f} | "
                 f"{s['memcpy']:.0f} ({s['memcpy_us']/1e3:.3f}) |")
    L += ["", "## Top kernels (profiled steps)", "", "| kernel | kind | count | /step | total ms | avg us | % kernel time | TFLOP/s |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["top_kernels"]:
        tf = f"{r['tflops']:.1f}" if r["tflops"] else ""
        L.append(f"| `{r['name'][:90]}` | {r['kind']} | {r['count']} | {r['per_step']:.1f} | {r['total_us']/1e3:.3f} | "
                 f"{r['avg_us']:.1f} | {r['pct_time']:.1f} | {tf} |")
    g = res["gemm"]
    L += ["", "## GEMMs", "",
          f"{g['gemm_kernels']} GEMM launches, {g['gemm_us']/1e3:.3f} ms; shapes recovered for {g['with_shapes']} "
          f"({g['shaped_us']/1e3:.3f} ms)" + (f"; achieved {g['achieved_tflops']:.1f} TFLOP/s over those" if g["achieved_tflops"] else
                                              "; no shapes (pass --gemm-shapes, see --list-gemms)"), ""]
    if res["gemm_table"]:
        L += ["| kernel | count | total ms | TFLOP/s | % peak | shape source |", "|---|---|---|---|---|---|"]
        for r in res["gemm_table"]:
            pk = f"{r['pct_peak']:.0f}" if r["pct_peak"] is not None else ""
            L.append(f"| `{r['name'][:90]}` | {r['count']} | {r['total_us']/1e3:.3f} | {r['tflops']:.1f} | {pk} | {r['source']} |")
    L += ["", "## Largest idle gaps", "", "| step | gap us | after kernel | before kernel |", "|---|---|---|---|"]
    for x in res["top_gaps"]:
        L.append(f"| {x['step']} | {x['us']:.1f} | `{(x['after'] or '-')[:60]}` | `{(x['before'] or '-')[:60]}` |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--device", type=int, help="CUDA device id (default: the one with the most kernels)")
    ap.add_argument("--step-regex", default=STEP_RE, help="NVTX range text that marks a step (default %(default)s)")
    ap.add_argument("--step-marker", help="kernel-name regex starting each step, when there are no NVTX steps")
    ap.add_argument("--gemm-regex", default=GEMM_RE)
    ap.add_argument("--gemm-exclude", default=GEMM_EXCLUDE_RE)
    ap.add_argument("--nccl-regex", default=NCCL_RE)
    ap.add_argument("--gemm-shapes", help="JSON list of {match, mnk|flops[, batch]} rules")
    ap.add_argument("--peak-tflops", type=float, help="for %% of peak (H100 SXM dense: 989 bf16, 1979 fp8)")
    ap.add_argument("--gap-threshold-us", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--list-gemms", action="store_true", help="list GEMM kernels with grid/block dims and exit")
    ap.add_argument("--json", help="also write the full result as JSON here")
    a = ap.parse_args(argv)
    if a.list_gemms:
        for r in list_gemms(a.sqlite, a.device, a.gemm_regex, a.gemm_exclude):
            print(json.dumps(r))
        return None
    rules = None
    if a.gemm_shapes:
        with open(a.gemm_shapes) as f:
            rules = json.load(f)
    res = analyze(a.sqlite, a.device, a.step_regex, a.step_marker, a.gemm_regex, a.gemm_exclude, a.nccl_regex,
                  rules, a.gap_threshold_us, a.top, peak_tflops=a.peak_tflops)
    print(render(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    sys.exit(0 if main() is not False else 1)
