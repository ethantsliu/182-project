"""GPU job pool, LR sweep (log grid, ratio 3, edge extension, 3 seeds at the best LR) and summaries.

A job is `python <script> job '<json>'`; it must write a result JSON to args["out"]:
  {"final": float, "curve": [[step, metric], ...], "ms_per_step": float, "opt_state_mb": float, ...}
Jobs whose result file exists are skipped, so a restarted queue resumes.
"""
import json
import math
import os
import statistics
import subprocess
import sys
import time

PY = sys.executable
ENV_LD = "/usr/local/cuda-13.0/compat:/home/dev_patel/venv/lib/python3.12/site-packages/nvidia/cu13/lib:" \
         "/home/dev_patel/venv/lib/python3.12/site-packages/torch/lib"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def run_jobs(script, jobs, gpus=None, per_gpu=1, timeout=3600, cpu_threads=8):
    gpus = list(range(int(os.environ.get("NGPU", 8)))) if gpus is None else gpus
    pending = [j for j in jobs if not os.path.exists(j["out"])]
    log(f"{os.path.basename(script)}: {len(pending)} jobs to run ({len(jobs) - len(pending)} cached) on {len(gpus)}x{per_gpu} slots")
    slots = [g for g in gpus for _ in range(per_gpu)]
    free = list(range(len(slots)))
    running = {}
    while pending or running:
        while pending and free:
            s = free.pop(0)
            j = pending.pop(0)
            os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slots[s]), OMP_NUM_THREADS=str(cpu_threads),
                       LD_LIBRARY_PATH=ENV_LD)
            logf = open(j["out"] + ".log", "w")
            proc = subprocess.Popen([PY, script, "job", json.dumps(j)], stdout=logf, stderr=subprocess.STDOUT, env=env)
            running[s] = (proc, j, time.time(), logf)
        time.sleep(1)
        for s, (proc, j, t0, logf) in list(running.items()):
            rc = proc.poll()
            if rc is None and time.time() - t0 > timeout:
                proc.kill()
                rc = "timeout"
            if rc is not None:
                logf.close()
                if not os.path.exists(j["out"]):
                    log(f"FAILED ({rc}): {j['out']}  (see .log)")
                    with open(j["out"], "w") as f:
                        json.dump({**j, "failed": str(rc), "final": None, "curve": []}, f)
                del running[s]
                free.append(s)
    return [json.load(open(j["out"])) for j in jobs]


def _score(r, higher):
    v = r.get("final")
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return -math.inf
    return v if higher else -v


def lr_tag(lr):
    return f"{lr:.3g}"


def sweep(script, arms, out_dir, higher, seeds=(0, 1, 2), extend=True, **pool_kw):
    """arms: {name: {"lr": center, **job_args}}. Returns {name: {"best_lr", "grid": {lr: result}, "seeds": [...]}}."""
    def job(name, a, lr, seed):
        return {**a, "arm": name, "lr": lr, "seed": seed,
                "out": os.path.join(out_dir, name, f"lr{lr_tag(lr)}_s{seed}.json")}

    grid = {n: [a["lr"] / 3, a["lr"], a["lr"] * 3] for n, a in arms.items()}
    res = {n: {} for n in arms}
    rounds = 2 if extend else 1
    for rnd in range(rounds):
        jobs = [job(n, a, lr, seeds[0]) for n, a in arms.items() for lr in grid[n] if lr_tag(lr) not in res[n]]
        for r in run_jobs(script, jobs, **pool_kw):
            res[r["arm"]][lr_tag(r["lr"])] = r
        if rnd == rounds - 1:
            break
        for n in arms:
            lrs = sorted(grid[n])
            best = max(lrs, key=lambda lr: _score(res[n][lr_tag(lr)], higher))
            if best == lrs[0]:
                grid[n].append(lrs[0] / 3)
            elif best == lrs[-1]:
                grid[n].append(lrs[-1] * 3)
    out = {}
    jobs = []
    for n, a in arms.items():
        best = max(grid[n], key=lambda lr: _score(res[n][lr_tag(lr)], higher))
        out[n] = {"best_lr": best, "grid": res[n]}
        jobs += [job(n, a, best, s) for s in seeds[1:]]
    seeded = run_jobs(script, jobs, **pool_kw)
    for n in arms:
        out[n]["seeds"] = [out[n]["grid"][lr_tag(out[n]["best_lr"])]] + [r for r in seeded if r["arm"] == n]
    return out


def first_hit(curve, target, higher):
    for step, v in curve:
        if v is not None and ((v >= target) if higher else (v <= target)):
            return step
    return None


def summarize(results, higher, metric_name, title):
    """results: {arm: {"best_lr", "grid", "seeds"}} -> markdown table + dict."""
    rows = {}
    for n, r in results.items():
        finals = [s["final"] for s in r["seeds"] if s.get("final") is not None and math.isfinite(s["final"])]
        rows[n] = dict(best_lr=r["best_lr"], finals=finals,
                       med=statistics.median(finals) if finals else None,
                       mean=statistics.mean(finals) if finals else None,
                       std=statistics.stdev(finals) if len(finals) > 1 else 0.0,
                       ms=statistics.median([s.get("ms_per_step", float("nan")) for s in r["seeds"]]),
                       mem=r["seeds"][0].get("opt_state_mb"),
                       nfail=sum(1 for s in r["seeds"] if s.get("final") is None))
    meds = [v["med"] for v in rows.values() if v["med"] is not None]
    target = (min(meds) if higher else max(meds)) if meds else None
    lines = [f"### {title}", "",
             f"target for steps-to-target = worst arm's median final {metric_name} = {target:.4g}" if target is not None else "no finished arm",
             "", f"| arm | best lr | final {metric_name} (mean ± std, n) | median | steps-to-target (median) | ms/step | opt state MB |",
             "|---|---|---|---|---|---|---|"]
    for n, v in rows.items():
        hits = [first_hit(s["curve"], target, higher) for s in results[n]["seeds"]] if target is not None else []
        hits = [h for h in hits if h is not None]
        v["steps_to_target"] = statistics.median(hits) if hits else None
        fin = f"{v['mean']:.4g} ± {v['std']:.2g} ({len(v['finals'])})" if v["finals"] else "failed"
        lines.append(f"| {n} | {v['best_lr']:.3g} | {fin} | {v['med'] if v['med'] is None else format(v['med'], '.4g')} | "
                     f"{v['steps_to_target']} | {v['ms']:.1f} | {v['mem'] if v['mem'] is None else format(v['mem'], '.1f')} |")
    lines.append("")
    lines.append("LR grids (seed 0): " + "; ".join(
        f"{n}: " + ", ".join(f"{lr}->{(g.get('final') if g.get('final') is None else format(g['final'], '.4g'))}"
                             for lr, g in sorted(results[n]["grid"].items(), key=lambda kv: float(kv[0])))
        for n in results))
    return "\n".join(lines) + "\n", rows
