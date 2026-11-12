#!/usr/bin/env python3
"""Where a training step's GPU time goes, from a torch.profiler (Kineto) Chrome trace written by hook.py.

    python 182/profile/analyze.py PROFILE_DIR                 # trace_rank0.json.gz + every rank<R>_summary.json
    python 182/profile/analyze.py trace.json[.gz] [--json out.json] [--top 25] [--categories cats.json]

Steps are the CPU `prof_step N` ranges hook.py puts around each profiled call. A kernel belongs to the step in
which its launch API call ran (its `correlation` id; a CUDA-graph replay's kernels share the id of their
cudaGraphLaunch), so kernels that run after the CPU has moved on are still counted with their own step. A step's
GPU window runs from its first kernel's start to the next step's first kernel's start (the last step: to its last
kernel's end), so idle time includes the bubbles between steps.

Reported per step (mean over the profiled steps, one device: the trace's own GPU):
  * per category: summed kernel time (ms/step) and as % of the GPU window. Kernels on different streams overlap,
    so the categories can add up to more than 100%; `busy` is the union of all kernels, `idle` the window minus it.
  * NCCL: kernel time, and exposed time = NCCL time not covered by any non-NCCL kernel (what overlap did not hide).
  * the largest idle gaps (with the kernels on either side), the top kernels by total time.
  * graph coverage: cudaGraphLaunch calls in the steps vs kernels traced from them. Zero kernels from graphs
    means CUPTI did not trace the replays (see hook.py) and the breakdown covers eager kernels only.
Categories are regexes on the kernel name, first match wins (DEFAULT_CATEGORIES; --categories replaces them with a
JSON list of [key, label, regex]). Memcpy/memset are their own category. With a directory, every rank's
summary (written by hook.py from its own trace) is tabulated too: a rank that is slower than the others makes
every other rank wait in NCCL.

Stdlib only.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

# (key, label, regex on the kernel name). Order matters: first match wins.
DEFAULT_CATEGORIES = [
    ("nccl", "NCCL communication", r"nccl"),
    ("attn_bwd", "attention backward", r"flash.*bwd|fmha.*bwd|FlashAttn.*Bwd|bwd.*(flash|fmha)|qkv_norm_rope.*backward|_qkv_backward"),
    ("attn_fwd", "attention forward", r"flash.*fwd|fmha.*fwd|FlashAttn.*Fwd|fwd.*(flash|fmha)|qk_norm_rope"),
    ("optim", "optimizer (polar/Newton-Schulz, Adam)",
     r"polar|newton|schulz|\bXXT|\bXTX|XXT_kernel|XTX_kernel|ba_plus_cAA|gram_(norm|tall|wide)|poly_gram|(^|[^a-z])adam|lerp_upcast|anvil|muon"),
    ("gemm_moe", "MoE GEMMs (grouped)", r"grouped_(mm|dw|lrs|linear)|_experts_(forward|backward)|moe.*(gemm|mm)"),
    ("gemm", "dense GEMMs (MLP, projections, head)",
     r"gemm|nvjet|xmma|cutlass|cublas|sm90_|_mm_|mm_kernel|_f8_mm|scaled_mm|matmul|linear_relu_square|wg_f8|dw_into_bank|"
     r"triton_tem_fused_.*mm"),
    ("fp8_quant", "fp8 quantize / scales", r"quant|amax|hadamard|f8_t|fp8|e4m3|e5m2|inverse_scale"),
    ("loss_head", "loss / head (CE, sampled softmax, CPLM)", r"(^|_)ce_|cross_entropy|softmax|cplm|copy_sink|sampled_|_copy_(fwd|bwd)|logsumexp"),
    ("rows", "row gather/scatter (n-gram, value-embed, MoE dispatch)",
     r"ngram|value_embed|embedding|_rows|rows_|scatter|gather|densify|dispatch|dx_combine|index_select|embedding_dense"),
]
MEMCPY = ("memcpy", "memcpy / memset")
OTHER = ("other", "elementwise / other")
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver", "runtime", "driver"}
STEP_RE = re.compile(r"^prof_step (\d+)$")


def load_trace(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", []) if isinstance(data, dict) else data


def compile_categories(cats=None):
    return [(k, lab, re.compile(rx, re.I)) for k, lab, rx in (cats or DEFAULT_CATEGORIES)]


def categorize(name: str, cat: str, cats) -> str:
    if cat in ("gpu_memcpy", "gpu_memset"):
        return MEMCPY[0]
    for k, _, rx in cats:
        if rx.search(name):
            return k
    return OTHER[0]


def _union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _clip(iv, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in iv if b > lo and a < hi]


def _length(iv):
    return sum(b - a for a, b in iv)


def _subtract(a_iv, b_union):
    """Length of the union of a_iv not covered by b_union (both lists of intervals; b_union already merged)."""
    total = 0.0
    for a, b in _union(a_iv):
        cur = a
        i = bisect.bisect_left([x[1] for x in b_union], a)
        while cur < b and i < len(b_union):
            s, e = b_union[i]
            if s >= b:
                break
            if s > cur:
                total += s - cur
            cur = max(cur, e)
            i += 1
        if cur < b:
            total += b - cur
    return total


def summarize(events: list[dict], steps_hint: int | None = None, categories=None, top: int = 25) -> dict:
    cats = compile_categories(categories)
    labels = dict([(k, lab) for k, lab, _ in cats] + [MEMCPY, OTHER])
    order = [k for k, _, _ in cats] + [MEMCPY[0], OTHER[0]]
    xs = [e for e in events if e.get("ph") == "X" and "ts" in e]
    gpu = [e for e in xs if e.get("cat") in GPU_CATS]
    warnings = []
    # one device: the trace's own GPU (the most common device id among kernels)
    devs = defaultdict(int)
    for e in gpu:
        devs[(e.get("args") or {}).get("device", e.get("pid"))] += 1
    dev = max(devs, key=devs.get) if devs else None
    if len(devs) > 1:
        warnings.append(f"kernels from {len(devs)} devices in one trace; using device {dev} only")
    gpu = [e for e in gpu if (e.get("args") or {}).get("device", e.get("pid")) == dev]
    launches = {}
    for e in xs:
        if e.get("cat") in LAUNCH_CATS:
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                launches[c] = e
    spans = []  # (step, cpu start, cpu end) of hook.py's record_function ranges (not their GPU-side mirrors)
    for e in xs:
        m = STEP_RE.match(str(e.get("name", "")))
        if m and e.get("cat") != "gpu_user_annotation":
            spans.append((int(m.group(1)), e["ts"], e["ts"] + e.get("dur", 0)))
    spans.sort(key=lambda s: s[1])
    if not spans:
        warnings.append("no `prof_step N` ranges in the trace: the whole trace is treated as one window"
                        + (f" of {steps_hint} steps" if steps_hint else ""))
        spans = [(-1, -float("inf"), float("inf"))]
    starts = [s[1] for s in spans]
    per_step = defaultdict(list)
    unattributed = 0
    graph_kernels = 0
    for e in gpu:
        c = (e.get("args") or {}).get("correlation")
        L = launches.get(c)
        if L is not None:
            t = L["ts"]
            if str(L.get("name", "")).startswith("cudaGraphLaunch"):
                graph_kernels += 1
        else:
            t = e["ts"]
            unattributed += 1
        i = bisect.bisect_right(starts, t) - 1
        if i < 0:
            continue  # launched before the window
        per_step[i].append(e)
    graph_launches = sum(1 for L in launches.values() if str(L.get("name", "")).startswith("cudaGraphLaunch")
                         and spans[0][1] <= L["ts"] <= spans[-1][2])
    if graph_launches and not graph_kernels:
        warnings.append(f"{graph_launches} cudaGraphLaunch calls in the steps but no kernel traced from them: CUPTI did "
                        "not record the graph replays, so this breakdown covers only the eager kernels. Use "
                        "182/tools/profile_nsys.sh (--cuda-graph-trace=node) for the graphs")
    if unattributed:
        warnings.append(f"{unattributed} GPU events without a matching launch call (assigned by their own start time)")
    nsteps = len(spans)
    first = [min((e["ts"] for e in per_step[i]), default=None) for i in range(nsteps)]
    steps, cat_tot, kern = [], defaultdict(float), {}
    gaps = []
    for i in range(nsteps):
        ev = sorted(per_step[i], key=lambda e: e["ts"])
        if not ev:
            steps.append(dict(step=spans[i][0], cpu_ms=(spans[i][2] - spans[i][1]) / 1000, kernels=0))
            continue
        lo = first[i]
        nxt = next((first[j] for j in range(i + 1, nsteps) if first[j] is not None), None)
        hi = nxt if nxt is not None else max(e["ts"] + e.get("dur", 0) for e in ev)
        # busy/idle over everything that ran in the window (a late kernel of the previous step counts as busy)
        allwin = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in gpu if e["ts"] < hi and e["ts"] + e.get("dur", 0) > lo]
        busy_u = _union(_clip(allwin, lo, hi))
        nccl_iv, comp_iv = [], []
        for e in ev:
            k = categorize(str(e.get("name", "")), e.get("cat"), cats)
            d = e.get("dur", 0)
            cat_tot[k] += d
            nm = str(e.get("name", ""))
            r = kern.setdefault(nm, dict(name=nm, cat=k, n=0, us=0.0))
            r["n"] += 1
            r["us"] += d
            (nccl_iv if k == "nccl" else comp_iv).append((e["ts"], e["ts"] + d))
        comp_all = _union(_clip([(e["ts"], e["ts"] + e.get("dur", 0)) for e in gpu
                                 if categorize(str(e.get("name", "")), e.get("cat"), cats) != "nccl"
                                 and e["ts"] < hi and e["ts"] + e.get("dur", 0) > lo], lo, hi))
        exposed = _subtract(_clip(nccl_iv, lo, hi), comp_all)
        for (_, b), (c2, _) in zip(busy_u, busy_u[1:] + [[hi, hi]]):
            if c2 > b:  # between kernels, and the bubble from the last one to the next step's first
                gaps.append((c2 - b, spans[i][0], b))
        win = hi - lo
        steps.append(dict(step=spans[i][0], cpu_ms=(spans[i][2] - spans[i][1]) / 1000 if spans[i][0] >= 0 else None,
                          window_ms=win / 1000, busy_ms=_length(busy_u) / 1000, idle_ms=(win - _length(busy_u)) / 1000,
                          nccl_ms=sum(b - a for a, b in nccl_iv) / 1000, nccl_exposed_ms=exposed / 1000, kernels=len(ev)))
    good = [s for s in steps if s.get("kernels")]
    n = len(good) or 1
    if spans[0][0] == -1 and steps_hint:
        n = steps_hint

    def mean(key):
        return sum(s[key] for s in good) / n if good else 0.0
    window = mean("window_ms")
    by_t = sorted(gpu, key=lambda e: e["ts"])
    t_starts = [e["ts"] for e in by_t]

    def neighbours(t):
        j = bisect.bisect_right(t_starts, t) - 1
        before = max((e for e in by_t[max(0, j - 50):j + 1] if e["ts"] + e.get("dur", 0) <= t + 1e-6),
                     key=lambda e: e["ts"] + e.get("dur", 0), default=None)
        after = by_t[j + 1] if j + 1 < len(by_t) else None
        return (str(before.get("name", ""))[:60] if before else "-"), (str(after.get("name", ""))[:60] if after else "-")
    gaps.sort(reverse=True)
    out = dict(
        steps_profiled=len(good) if spans[0][0] != -1 else n, device=dev, mean_window_ms=window,
        mean_busy_ms=mean("busy_ms"), mean_idle_ms=mean("idle_ms"), mean_cpu_ms=mean("cpu_ms") if good and good[0].get("cpu_ms") is not None else None,
        nccl_ms=mean("nccl_ms"), nccl_exposed_ms=mean("nccl_exposed_ms"),
        categories=[dict(key=k, label=labels[k], ms_per_step=cat_tot[k] / 1000 / n,
                         pct=(100 * cat_tot[k] / 1000 / n / window) if window else 0.0) for k in order],
        top_kernels=[dict(r, ms_per_step=r["us"] / 1000 / n, calls_per_step=r["n"] / n, avg_us=r["us"] / r["n"],
                          pct=(100 * r["us"] / 1000 / n / window) if window else 0.0)
                     for r in sorted(kern.values(), key=lambda r: -r["us"])[:top]],
        gaps=[dict(us=g, step=s, after=neighbours(t)[0], before=neighbours(t)[1]) for g, s, t in gaps[:10]],
        graph_launches=graph_launches, graph_kernels=graph_kernels, kernels=sum(len(v) for v in per_step.values()),
        steps=steps, warnings=warnings)
    return out


def report(s: dict, ranks: list[dict] | None = None, top: int = 25) -> list[str]:
    w = s["mean_window_ms"]
    L = [f"steps profiled: {s['steps_profiled']} (device {s['device']}); GPU window {w:.2f} ms/step"
         + (f", CPU step {s['mean_cpu_ms']:.2f} ms" if s.get("mean_cpu_ms") is not None else "")
         + f"; busy {s['mean_busy_ms']:.2f} ms, idle {s['mean_idle_ms']:.2f} ms ({(100 * s['mean_idle_ms'] / w) if w else 0:.1f}%)",
         f"graph coverage: {s['graph_launches']} cudaGraphLaunch call(s), {s['graph_kernels']} of {s['kernels']} kernels traced from graph replays"]
    L += [f"!! {x}" for x in s["warnings"]]
    L += ["", f"{'category':<52} {'ms/step':>8} {'% step':>7}"]
    for c in s["categories"]:
        if c["ms_per_step"] > 0:
            L.append(f"{c['label']:<52} {c['ms_per_step']:8.3f} {c['pct']:6.1f}%")
    L.append(f"{'idle (window minus union of all kernels)':<52} {s['mean_idle_ms']:8.3f} {(100 * s['mean_idle_ms'] / w) if w else 0:6.1f}%")
    ov = s["nccl_ms"] - s["nccl_exposed_ms"]
    L.append(f"NCCL: {s['nccl_ms']:.3f} ms/step of kernels, {s['nccl_exposed_ms']:.3f} ms exposed (not under any compute kernel); "
             f"{(100 * ov / s['nccl_ms']) if s['nccl_ms'] else 0:.0f}% overlapped")
    L += ["(categories overlap across streams, so they may sum past 100%)", "", f"top {top} kernels by total time:",
          f"{'ms/step':>8} {'% step':>7} {'calls':>6} {'avg us':>8}  {'category':<10} kernel"]
    for k in s["top_kernels"][:top]:
        L.append(f"{k['ms_per_step']:8.3f} {k['pct']:6.1f}% {k['calls_per_step']:6.1f} {k['avg_us']:8.1f}  {k['cat']:<10} {k['name'][:110]}")
    if s["gaps"]:
        L += ["", "largest idle gaps:"]
        L += [f"  {g['us']:8.1f} us in step {g['step']}: after {g['after']} -> before {g['before']}" for g in s["gaps"]]
    if ranks:
        L += ["", "per rank (each rank's own trace):",
              f"{'rank':>4} {'window ms':>9} {'busy ms':>8} {'idle ms':>8} {'nccl ms':>8} {'exposed':>8} {'graph k':>8}  status"]
        for r in ranks:
            m = r.get("meta", {})
            L.append(f"{m.get('rank', '?'):>4} {r['mean_window_ms']:9.2f} {r['mean_busy_ms']:8.2f} {r['mean_idle_ms']:8.2f} "
                     f"{r['nccl_ms']:8.2f} {r['nccl_exposed_ms']:8.2f} {r['graph_kernels']:8d}  {m.get('status', '?')}"
                     + (f" window {m['window_ms'] / m['count']:.2f} ms/step" if m.get("window_ms") and m.get("count") else ""))
        ws = [r["mean_window_ms"] for r in ranks if r["mean_window_ms"]]
        if len(ws) > 1:
            L.append(f"  spread of the GPU window across ranks: {min(ws):.2f}-{max(ws):.2f} ms")
    return L


def load_rank_summaries(d: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(d, "rank*_summary.json")),
                    key=lambda p: int(re.search(r"rank(\d+)_", os.path.basename(p)).group(1))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="hook.py's PROFILE_DIR, or a Chrome trace .json / .json.gz")
    ap.add_argument("--json", help="write the summary as JSON here")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--categories", help="JSON file: [[key, label, regex], ...] replacing the default categories")
    a = ap.parse_args(argv)
    cats = None
    if a.categories:
        with open(a.categories) as f:
            cats = [tuple(x) for x in json.load(f)]
    ranks = []
    path = a.input
    if os.path.isdir(path):
        ranks = load_rank_summaries(path)
        path = os.path.join(a.input, "trace_rank0.json.gz")
        if not os.path.isfile(path):
            print(f"no trace_rank0.json.gz in {a.input}" + (f" ({len(ranks)} rank summaries found)" if ranks else ""))
            for p in sorted(glob.glob(os.path.join(a.input, "meta_rank*.json")))[:1]:
                with open(p) as f:
                    m = json.load(f)
                print(f"rank meta: status={m.get('status')} export_error={m.get('export_error', '-')}")
            return 1
    s = summarize(load_trace(path), categories=cats, top=a.top)
    print("\n".join(report(s, ranks, a.top)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(s, ranks=ranks), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
