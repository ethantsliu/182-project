"""Confirmatory FNO controls. Use queue_fno.sh; `plan` never runs a GPU job.

Old Exp 3 results remain exploratory: this round reserves validation from the
original training set, regenerates test samples, and uses new confirmation seeds.
SOAP math deliberately matches fftopt.core, including its eigenbasis refresh rule.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from exp3_fno import CFG, DECAY, FNO, flat, rel_l2, set_mult
from fftopt.core import MatrixOpt, View, eigvecs, nbytes_state, warmup_linear
from fftopt import pool
import prep_data

HERE = Path(__file__).resolve().parent
TUNE_SEED = 100
CONFIRM_SEEDS = (101, 102, 103)
SPLIT_SEED = 182
TEST_SEED = 20261112
TARGETS = {"burgers": 0.001563, "darcy": 0.01051, "ns": 0.006132}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash():
    paths = [Path(__file__), HERE / "exp3_fno.py", HERE / "prep_data.py", HERE / "fftopt/core.py"]
    return hashlib.sha256("".join(sha_file(p) for p in paths).encode()).hexdigest()


def mixed_view(shape, device):
    """Random column blocks with exactly the same shapes/state sizes as mode SOAP."""
    k, ci, co = shape
    perm = torch.randperm(k * co, generator=torch.Generator().manual_seed(1729)).to(device)
    inv = torch.argsort(perm)

    def fwd(x):
        x = x.permute(1, 0, 2).reshape(ci, k * co)[:, perm]
        return x.reshape(ci, k, co).permute(1, 0, 2)

    def bwd(x):
        x = x.permute(1, 0, 2).reshape(ci, k * co)[:, inv]
        return x.reshape(ci, k, co).permute(1, 0, 2)

    return View(fwd, bwd)


class ComplexAdam(torch.optim.Optimizer):
    """Adam with a real radial second moment per complex entry and SOAP's betas."""
    def __init__(self, params, lr, decay=DECAY):
        super().__init__(params, dict(lr=lr, lr_mult=1., decay=decay, betas=(0.95, 0.95)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if not st:
                    st.update(t=0, m=torch.zeros_like(g), v=torch.zeros_like(g.real))
                st["t"] += 1
                t = st["t"]
                st["m"].lerp_(g, 1 - b1)
                st["v"].lerp_(g.abs().square(), 1 - b2)
                u = (st["m"] / (1 - b1 ** t)) / ((st["v"] / (1 - b2 ** t)).sqrt() + 1e-8)
                p.mul_(1 - group["lr_mult"] * group["decay"])
                p.add_(u, alpha=-group["lr"] * group["lr_mult"])


class SharedSOAP(MatrixOpt):
    """Shared channel eigenbases, separate rotated Adam statistics for every mode."""
    def _update_factors(self, st, gv, group, t):
        left = (gv @ gv.mH).sum(0, keepdim=True)
        right = (gv.mH @ gv).sum(0, keepdim=True)
        if "L" not in st:
            st["L"], st["R"] = torch.zeros_like(left), torch.zeros_like(right)
        st["L"].lerp_(left, 1 - group["shampoo_beta"])
        st["R"].lerp_(right, 1 - group["shampoo_beta"])
        if t == 1 or t % group["precond_every"] == 0:
            st["QL"] = eigvecs(st["L"], group["rel_eps"])
            st["QR"] = eigvecs(st["R"], group["rel_eps"])


def make_optimizer(arm, params, lr, refresh=10):
    if arm in ("adamw", "adamw_matched"):
        betas = (0.9, 0.999) if arm == "adamw" else (0.95, 0.95)
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=DECAY / lr)
    if arm == "complex_adam":
        return ComplexAdam(params, lr=lr)
    groups = []
    for p in params:
        view = (mixed_view(p.shape, p.device) if arm == "soap_mixed" else
                flat(p.shape, True) if arm == "soap_flat" else View(lambda x: x, lambda x: x))
        groups.append(dict(params=[p], view=view))
    cls = SharedSOAP if arm == "soap_shared" else MatrixOpt
    return cls(groups, lr=lr, mode="soap", decay=DECAY, precond_every=refresh)


def split_training(data):
    n = len(data["x_train"])
    nval = n // 5
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SPLIT_SEED))
    tr, va = idx[nval:], idx[:nval]
    return ((data["x_train"][tr], data["y_train"][tr]),
            (data["x_train"][va], data["y_train"][va]))


def select_best(records):
    valid = [r for r in records if r.get("final") is not None and math.isfinite(r["final"])]
    if not valid:
        raise ValueError("all learning rates failed; inspect job logs")
    return min(valid, key=lambda r: (r["final"], r["lr"]))


def stage_specs():
    def arm(name, center=.003, refresh=10):
        return dict(arm=name, center_lr=center, refresh=refresh)
    controls = [arm(n) for n in ("adamw", "adamw_matched", "complex_adam", "soap_shared", "mode_soap", "soap_mixed")]
    refresh = [arm("adamw"), arm("soap_flat", .009)] + [arm("mode_soap", .009, r) for r in (10, 25, 50, 100)]
    return {
        "controls-darcy": dict(dataset="darcy", arms=controls),
        "ns-boundary": dict(dataset="ns", arms=[arm("mode_soap", .027), arm("adamw")]),
        "refresh-darcy": dict(dataset="darcy", arms=refresh),
        "refresh-ns": dict(dataset="ns", arms=[{**a, "center_lr": .027} if a["arm"] == "mode_soap" else a for a in refresh]),
        "controls-burgers": dict(dataset="burgers", arms=controls),
    }


def prepare(out, data_root, datasets):
    out = Path(out)
    sources, tests = {}, {}
    for ds in datasets:
        source = Path(data_root) / "fno" / f"{ds}.pt"
        if not source.is_file():
            raise FileNotFoundError(f"missing {source}; run prep_data.py pde first")
        sources[ds] = dict(path=str(source.resolve()), sha256=sha_file(source))
        path = out / "data" / f"{ds}-fresh-test.pt"
        if not path.exists():
            print(f"Generating independent {ds} test samples (seed {TEST_SEED})", flush=True)
            x, y = getattr(prep_data, ds)(200, seed=TEST_SEED)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save(dict(x_test=x, y_test=y, seed=TEST_SEED), tmp)
            tmp.replace(path)
        tests[ds] = dict(path=str(path.resolve()), sha256=sha_file(path))
    meta = dict(protocol=1, source_hash=source_hash(), torch=torch.__version__,
                split_seed=SPLIT_SEED, tune_seed=TUNE_SEED, confirm_seeds=CONFIRM_SEEDS,
                test_seed=TEST_SEED, sources=sources, tests=tests)
    p = out / "data-manifest.json"
    if p.exists() and json.loads(p.read_text()) != json.loads(json.dumps(meta)):
        raise ValueError("data/code/stack changed: choose a new FNO_OUT instead of mixing results")
    atomic_json(p, meta)


def train(job):
    started = time.perf_counter()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    torch.manual_seed(job["seed"])
    dev = job.get("device", "cuda")
    source = torch.load(job["source"], map_location="cpu", weights_only=True)
    (xtr, ytr), (xv, yv) = split_training(source)
    dim, width, modes, epochs, bs = CFG[job["dataset"]]
    width, modes = job.get("width", width), job.get("modes", modes)
    epochs = job.get("epochs", epochs)
    if job.get("smoke"):
        xtr, ytr, xv, yv = xtr[:4], ytr[:4], xv[:4], yv[:4]
        width, modes, epochs, bs = 4, 2, 1, 2
    xtr, ytr, xv, yv = [t.to(dev)[:, None] for t in (xtr, ytr, xv, yv)]
    mu, sd, ymu, ysd = xtr.mean(), xtr.std(), ytr.mean(), ytr.std()
    xtr, xv = (xtr - mu) / sd, (xv - mu) / sd
    model = FNO(dim, 1, width, modes).to(dev)
    spec = [s.w for s in model.spec]
    ids = {id(p) for p in spec}
    rest = [p for p in model.parameters() if id(p) not in ids]
    opt = make_optimizer(job["arm"], spec, job["lr"], job["refresh"])
    other = torch.optim.AdamW(rest, lr=1e-3, weight_decay=1e-4)

    def sync():
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def evaluate(x, y):
        model.eval()
        errors = [rel_l2(model(x[i:i + 20]) * ysd + ymu, y[i:i + 20]) for i in range(0, len(x), 20)]
        val = torch.cat(errors).mean().item()
        model.train()
        return val

    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    spe = len(xtr) // bs
    total, step, train_s, curve = spe * epochs, 0, 0., []
    loop_start = time.perf_counter()
    diverged = False
    for ep in range(epochs):
        perm = torch.randperm(len(xtr), device=dev)
        sync()
        t0 = time.perf_counter()
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            set_mult([opt, other], warmup_linear(step, total))
            loss = rel_l2(model(xtr[idx]) * ysd + ymu, ytr[idx]).mean()
            opt.zero_grad(set_to_none=True)
            other.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            other.step()
            step += 1
        sync()
        train_s += time.perf_counter() - t0
        if not math.isfinite(loss.item()):
            diverged = True
            break
        if (ep + 1) % max(1, epochs // 20) == 0 or ep == epochs - 1:
            value = evaluate(xv, yv)
            sync()
            curve.append(dict(step=step, val=value if math.isfinite(value) else None,
                              train_s=train_s, elapsed_s=time.perf_counter() - loop_start))
    final = curve[-1]["val"] if curve and not diverged else None
    test_final = None
    if job["evaluate_test"] and final is not None:
        test = torch.load(job["fresh_test"], map_location="cpu", weights_only=True)
        xt, yt = test["x_test"].to(dev)[:, None], test["y_test"].to(dev)[:, None]
        test_final = evaluate((xt - mu) / sd, yt)
        if not math.isfinite(test_final):
            test_final = None
    sync()
    return dict(final=final, test_final=test_final, curve=curve, steps=step,
                train_seconds=train_s, total_seconds=time.perf_counter() - started,
                ms_per_step=1000 * train_s / max(step, 1), opt_state_mb=nbytes_state(opt) / 2 ** 20,
                all_opt_state_mb=(nbytes_state(opt) + nbytes_state(other)) / 2 ** 20,
                peak_allocated_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if dev.startswith("cuda") else None,
                peak_reserved_mb=torch.cuda.max_memory_reserved() / 2 ** 20 if dev.startswith("cuda") else None,
                torch=torch.__version__, source_hash=source_hash())


def job_spec(out, meta, ds, arm, lr, seed, evaluate_test):
    job = dict(dataset=ds, arm=arm["arm"], refresh=arm["refresh"], lr=lr, seed=seed,
               evaluate_test=evaluate_test, source=meta["sources"][ds]["path"],
               fresh_test=meta["tests"][ds]["path"], protocol_hash=hashlib.sha256(
                   json.dumps(meta, sort_keys=True).encode()).hexdigest())
    tag = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
    job["out"] = str(Path(out) / "jobs" / f"{ds}-{arm['arm']}-r{arm['refresh']}-s{seed}-{tag}.json")
    return job


def execute_jobs(jobs):
    # The original pool used a hardcoded torch 2.11 library path. Inherit the pinned environment instead.
    pool.ENV_LD = os.environ.get("LD_LIBRARY_PATH", "")
    for job in jobs:
        path = Path(job["out"])
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("final") is None or (job["evaluate_test"] and record.get("test_final") is None):
                # Keep the failure and its console, but allow a deliberate queue resume to retry it.
                suffix = f".failed-{time.time_ns()}"
                path.rename(str(path) + suffix)
                log = Path(str(path) + ".log")
                if log.exists():
                    log.rename(str(log) + suffix)
    records = pool.run_jobs(str(Path(__file__).resolve()), jobs, timeout=3600)
    return records


def run_stage(out, name):
    specs = stage_specs()
    stage = specs[name]
    ds = stage["dataset"]
    meta = json.loads((Path(out) / "data-manifest.json").read_text())
    if meta["source_hash"] != source_hash() or meta["torch"] != torch.__version__:
        raise ValueError("source/stack changed since preparation; choose a new FNO_OUT")
    summaries = []
    for arm in stage["arms"]:
        center = arm["center_lr"]
        grid = [center / 3, center, center * 3]
        records = []
        boundary = False
        for extension in range(3):
            jobs = [job_spec(out, meta, ds, arm, lr, TUNE_SEED, False) for lr in sorted(set(grid))]
            records = execute_jobs(jobs)
            best = select_best(records)
            lrs = sorted(set(grid))
            boundary = best["lr"] in (lrs[0], lrs[-1])
            if not boundary or extension == 2:
                break
            grid.append(lrs[0] / 3 if best["lr"] == lrs[0] else lrs[-1] * 3)
        confirm = execute_jobs([job_spec(out, meta, ds, arm, best["lr"], seed, True) for seed in CONFIRM_SEEDS])
        summaries.append(dict(arm=arm, best_lr=best["lr"], boundary=boundary,
                              tuning=records, confirmation=confirm))
        atomic_json(Path(out) / f"{name}.json", dict(stage=stage, results=summaries))
    write_report(out)
    return all(r.get("test_final") is not None for s in summaries for r in s["confirmation"])


def write_report(out):
    out = Path(out)
    lines = ["# FNO follow-up results", "", "LR selection: held-out validation only; final test: independent PDE samples.",
             "Confirmation seeds: 101, 102, 103. Original 200-sample test sets are unused.",
             "Times exclude compilation (these FNO runs are eager); train time excludes evaluation.",
             "Targets below are fixed exploratory thresholds, not selected from this round.", ""]

    def fmt(values):
        return f"{statistics.mean(values):.5g} ± {statistics.stdev(values):.2g}" if len(values) > 1 else f"{values[0]:.5g}" if values else "—"

    for name in stage_specs():
        path = out / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        target = TARGETS[data["stage"]["dataset"]]
        lines += [f"## {name}", "", f"Fixed validation error target: {target}.", "",
                  "| arm | refresh | best LR | test rel L2 mean ± std | successful / planned | train s | ms/step | spectral state MB | peak allocated MB | target reached / planned | train s to target (reached only) |",
                  "|---|---:|---:|---|---|---|---|---|---|---|---|"]
        for item in data["results"]:
            records = item["confirmation"]
            good = [r for r in records if r.get("test_final") is not None]
            hits = [next((p["train_s"] for p in r["curve"] if p["val"] is not None and p["val"] <= target), None) for r in records]
            hitvals = [h for h in hits if h is not None]
            boundary = " (grid edge)" if item["boundary"] else ""
            metrics = [fmt([r[k] for r in good if r.get(k) is not None]) for k in
                       ("test_final", "train_seconds", "ms_per_step", "opt_state_mb", "peak_allocated_mb")]
            lines.append(f"| {item['arm']['arm']} | {item['arm']['refresh']} | {item['best_lr']:.5g}{boundary} | {metrics[0]} | {len(good)}/3 | "
                         f"{metrics[1]} | {metrics[2]} | {metrics[3]} | {metrics[4]} | {len(hitvals)}/3 | {fmt(hitvals)} |")
        lines += ["", "All tuning points, validation/time curves, failure records and per-job consoles are retained in JSON/log files.", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines))


def main():
    # pool.run_jobs uses the original worker protocol.
    if len(sys.argv) > 1 and sys.argv[1] == "job":
        job = json.loads(sys.argv[2])
        atomic_json(job["out"], {**job, **train(job)})
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "prepare", "stage", "report"))
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default=os.environ.get("FFT_DATA", "/mnt/work/dev_patel/data"))
    p.add_argument("--stage", choices=tuple(stage_specs()))
    a = p.parse_args()
    if a.command == "plan":
        atomic_json(a.out, dict(stages=stage_specs(), tune_seed=TUNE_SEED, confirmation_seeds=CONFIRM_SEEDS,
                                lr_ratio=3, max_edge_extensions=2, source_hash=source_hash()))
    elif a.command == "prepare":
        prepare(a.out, a.data_root, ("darcy", "ns", "burgers"))
    elif a.command == "stage":
        if not a.stage:
            p.error("stage requires --stage")
        if not run_stage(a.out, a.stage):
            raise SystemExit("some confirmation runs failed; inspect logs and rerun to resume")
    else:
        write_report(a.out)


if __name__ == "__main__":
    main()
