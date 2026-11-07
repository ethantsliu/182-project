#!/usr/bin/env python3
"""Interleaved, pre-registered A/B runner for modded-nanogpt trainers.

Three steps, each its own subcommand, all keyed by one results directory:

  plan    Pre-register the experiment: arms (a git ref -> pinned commit + its own worktree, or an
          existing directory, pinned to its HEAD and a hash of its uncommitted diff), per-arm env vars,
          n runs per arm, warmup runs, the interleaved order, the seeds, the compile-cache policy, the
          target. Writes <out>/plan.json and prints its sha256. Nothing about the plan changes later.
  run     Execute the plan in order. Resumable: an interrupted harness picks up at the first slot
          without an "end" record (a slot interrupted mid-run is logged as such and re-run; that is a
          harness failure, not a trainer result). Every trainer outcome -- ok, crash, NaN, timeout,
          incomplete -- is recorded in <out>/ledger.jsonl and counted; nothing is retried or dropped.
  report  Tables of train time and val loss per arm, the one-sided gate test, Welch tests vs the
          baseline arm and a paired (per-rep) test. Writes report.md and runs.csv.
  parse   The same tables straight from existing log files: --group NAME=GLOB ...

Log formats: logparse.py (Track 1 short / Track 2 medium "train_time:Nms", Track 3 "train_time:N.Ns").

Example (8xH100 box, repo at ~/182-project):
  python 182/tools/ab_run.py plan --out ab/fp8 --repo . \
      --arm base=ref:master --arm new=ref:ws/hadamard-fp8 --env new:HADAMARD_FP8=1 \
      --n 8 --warmup 1 --cache warm
  python 182/tools/ab_run.py run ab/fp8
  python 182/tools/ab_run.py report ab/fp8
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logparse  # noqa: E402
import stats  # noqa: E402

DEFAULT_CMD = "torchrun --standalone --nproc_per_node={nproc} {script}"
DEFAULT_SCRIPT = "train_gpt.py"


# ----------------------------------------------------------------------------- git helpers

def git(repo: str, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {r.stderr.strip()}")
    return r.stdout.strip()


def is_git_dir(path: str) -> bool:
    return subprocess.run(["git", "-C", path, "rev-parse", "--git-dir"], capture_output=True).returncode == 0


def dir_fingerprint(path: str) -> dict:
    """HEAD sha plus a hash of uncommitted changes (tracked diff and untracked file list + contents)."""
    if not is_git_dir(path):
        return dict(sha=None, dirty=None, diff_sha256=None)
    sha = git(path, "rev-parse", "HEAD")
    diff = git(path, "diff", "HEAD", "--binary", check=False)
    untracked = git(path, "ls-files", "--others", "--exclude-standard", check=False).splitlines()
    h = hashlib.sha256(diff.encode())
    for u in sorted(untracked):
        h.update(u.encode())
        fp = os.path.join(path, u)
        if os.path.isfile(fp) and os.path.getsize(fp) < 50_000_000:
            with open(fp, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
    dirty = bool(diff.strip() or untracked)
    return dict(sha=sha, dirty=dirty, diff_sha256=h.hexdigest() if dirty else None)


# ----------------------------------------------------------------------------- plan

def parse_arm(spec: str, repo: str) -> dict:
    if "=" not in spec:
        raise SystemExit(f"--arm wants NAME=ref:<gitref> or NAME=path:<dir>, got {spec!r}")
    name, src = spec.split("=", 1)
    if src.startswith("ref:"):
        ref = src[4:]
        sha = git(repo, "rev-parse", "--verify", ref + "^{commit}")
        return dict(name=name, kind="ref", ref=ref, sha=sha, env={})
    path = src[5:] if src.startswith("path:") else src
    if not os.path.isdir(path):
        # bare value that is not a directory: treat as a git ref
        sha = git(repo, "rev-parse", "--verify", path + "^{commit}")
        return dict(name=name, kind="ref", ref=path, sha=sha, env={})
    path = os.path.abspath(path)
    return dict(name=name, kind="path", path=path, env={}, **dir_fingerprint(path))


def make_order(arm_names: list[str], n: int, warmup: int, order: str, order_seed: int, seeds: list) -> list[dict]:
    runs = []
    for w in range(warmup):
        for a in arm_names:
            runs.append(dict(rep=-1 - w, arm=a, seed=None, kind="warmup"))
    for rep in range(n):
        if order == "blocked":
            arms = list(arm_names)
            random.Random(order_seed * 1_000_003 + rep).shuffle(arms)
        elif order == "abba":
            arms = list(arm_names) if rep % 2 == 0 else list(reversed(arm_names))
        elif order == "alternate":
            arms = list(arm_names)
        else:
            raise SystemExit(f"unknown order {order}")
        for a in arms:
            runs.append(dict(rep=rep, arm=a, seed=seeds[rep] if seeds else None, kind="measure"))
    for i, r in enumerate(runs):
        r["idx"] = i
    return runs


def parse_seeds(spec: str, n: int, order_seed: int):
    if spec in ("none", "", None):
        return None
    if spec == "auto":
        rng = random.Random(order_seed + 7919)
        return [rng.randrange(1, 2**31 - 1) for _ in range(n)]
    seeds = [int(s) for s in spec.split(",")]
    if len(seeds) != n:
        raise SystemExit(f"--seeds lists {len(seeds)} seeds for --n {n}")
    return seeds


def plan_sha(plan: dict) -> str:
    body = {k: v for k, v in plan.items() if k != "created"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def cmd_plan(a) -> str:
    out = os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, "plan.json")):
        raise SystemExit(f"{out}/plan.json exists: a plan is never overwritten (choose a new --out)")
    repo = os.path.abspath(a.repo)
    arms = [parse_arm(s, repo) for s in a.arm]
    names = [x["name"] for x in arms]
    if len(set(names)) != len(names):
        raise SystemExit("arm names must be unique")
    by_name = {x["name"]: x for x in arms}
    for spec in a.env or []:
        name, kv = spec.split(":", 1)
        k, v = kv.split("=", 1)
        by_name[name]["env"][k] = v
    for spec in a.arm_cmd or []:
        name, c = spec.split(":", 1)
        by_name[name]["cmd"] = c
    for spec in a.arm_script or []:
        name, s = spec.split(":", 1)
        by_name[name]["script"] = s
    for x in arms:
        x.setdefault("cmd", a.cmd)
        x.setdefault("script", a.script)
        for k, v in (kv.split("=", 1) for kv in a.common_env or []):
            x["env"].setdefault(k, v)
    order_seed = a.order_seed if a.order_seed is not None else random.SystemRandom().randrange(2**31)
    seeds = parse_seeds(a.seeds, a.n, order_seed)
    baseline = a.baseline or names[0]
    if baseline not in by_name:
        raise SystemExit(f"--baseline {baseline} is not an arm")
    plan = dict(
        version=1,
        created=dt.datetime.now().isoformat(timespec="seconds"),
        repo=repo,
        work_root=os.path.abspath(a.work_root or os.path.join(out, "work")),
        data_dir=os.path.abspath(a.data_dir or os.path.join(repo, "data", "fineweb10B")),
        arms=arms,
        baseline=baseline,
        n=a.n,
        warmup=a.warmup,
        order=a.order,
        order_seed=order_seed,
        seed_env=a.seed_env,
        seeds=seeds,
        cache=a.cache,
        nproc=a.nproc,
        timeout_s=a.timeout_s,
        cooldown_s=a.cooldown_s,
        target=a.target,
        at_step=a.at_step,
        runs=make_order(names, a.n, a.warmup, a.order, order_seed, seeds),
    )
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    sha = plan_sha(plan)
    with open(os.path.join(out, "plan.sha256"), "w") as f:
        f.write(sha + "\n")
    print(f"pre-registered {len(plan['runs'])} runs ({a.warmup} warmup + {a.n} measured per arm, "
          f"order={a.order} seed={order_seed}) in {out}/plan.json  sha256={sha}")
    for x in arms:
        where = f"{x['ref']} @ {x['sha'][:12]}" if x["kind"] == "ref" else \
            f"{x['path']} @ {(x['sha'] or 'no-git')[:12]}{' +dirty ' + x['diff_sha256'][:12] if x.get('dirty') else ''}"
        print(f"  arm {x['name']:<12} {where}  env={x['env']}")
    print("  order: " + " ".join(f"{r['arm']}{'(w)' if r['kind'] == 'warmup' else ''}" for r in plan["runs"]))
    return out


def load_plan(out: str) -> dict:
    with open(os.path.join(out, "plan.json")) as f:
        plan = json.load(f)
    with open(os.path.join(out, "plan.sha256")) as f:
        want = f.read().strip()
    if plan_sha(plan) != want:
        raise SystemExit(f"{out}/plan.json no longer matches plan.sha256: the pre-registration was edited")
    return plan


# ----------------------------------------------------------------------------- run

def read_ledger(out: str) -> list[dict]:
    p = os.path.join(out, "ledger.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_ledger(out: str, rec: dict) -> None:
    with open(os.path.join(out, "ledger.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def prepare_arm(plan: dict, arm: dict, allow_changed: bool) -> str:
    """The directory the arm runs in: a pinned detached worktree for refs; the given dir for paths."""
    if arm["kind"] == "ref":
        wt = os.path.join(plan["work_root"], "wt", f"{arm['name']}-{arm['sha'][:12]}")
        if os.path.isdir(wt):
            head = git(wt, "rev-parse", "HEAD")
            if head != arm["sha"]:
                raise SystemExit(f"worktree {wt} is at {head}, plan pins {arm['sha']}")
            fp = dir_fingerprint(wt)
            if fp["dirty"] and not allow_changed:
                raise SystemExit(f"worktree {wt} has local changes; the plan pins a clean {arm['sha'][:12]}")
        else:
            os.makedirs(os.path.dirname(wt), exist_ok=True)
            git(plan["repo"], "worktree", "add", "--detach", wt, arm["sha"])
        path = wt
    else:
        path = arm["path"]
        fp = dir_fingerprint(path)
        if (fp["sha"], fp["diff_sha256"]) != (arm["sha"], arm["diff_sha256"]):
            msg = (f"arm {arm['name']}: {path} changed since the plan (HEAD {fp['sha']}, diff {fp['diff_sha256']}; "
                   f"planned {arm['sha']}, {arm['diff_sha256']})")
            if not allow_changed:
                raise SystemExit(msg + "; re-plan, or pass --allow-changed")
            print("WARNING: " + msg, file=sys.stderr)
    # The trainers read data/fineweb10B relative to cwd: link the shared shards in when absent.
    data = os.path.join(path, "data", "fineweb10B")
    if not os.path.exists(data) and os.path.isdir(plan["data_dir"]):
        os.makedirs(os.path.dirname(data), exist_ok=True)
        os.symlink(plan["data_dir"], data)
    return path


def gpu_snapshot() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def build_cmd(arm: dict, plan: dict, wt: str, run_dir: str, env_extra: dict) -> list[str]:
    env_flags = " ".join(f"-e {shlex.quote(k + '=' + v)}" for k, v in sorted(env_extra.items()))
    text = arm["cmd"].format(nproc=plan["nproc"], script=arm["script"], worktree=wt, run_dir=run_dir,
                             env_flags=env_flags, data_dir=plan["data_dir"])
    return shlex.split(text)


def classify(rc: int | None, timed_out: bool, log: logparse.RunLog) -> str:
    if timed_out:
        return "timeout"
    if log.is_nan:
        return "nan"
    if rc != 0:
        return "failed"
    if log.val_loss is None or not log.completed:
        return "incomplete"
    return "ok"


def execute(plan: dict, out: str, slot: dict, wt: str, arm: dict) -> dict:
    idx, name = slot["idx"], slot["arm"]
    tag = f"{idx:03d}_{name}"
    os.makedirs(os.path.join(out, "stdout"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    env_extra = dict(arm["env"])
    if slot["seed"] is not None and plan["seed_env"]:
        env_extra[plan["seed_env"]] = str(slot["seed"])
    if plan["cache"] == "cold":
        cache = os.path.join(plan["work_root"], "cache_cold", tag)
        shutil.rmtree(cache, ignore_errors=True)
    elif plan["cache"] == "warm":
        cache = os.path.join(plan["work_root"], "cache_warm", name)
    else:  # "inherit": whatever the environment already uses
        cache = None
    if cache:
        os.makedirs(cache, exist_ok=True)
        env_extra["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache, "inductor")
        env_extra["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
    env = dict(os.environ, **env_extra)
    cmd = build_cmd(arm, plan, wt, out, env_extra)
    stdout_path = os.path.join(out, "stdout", tag + ".out")
    rec = dict(event="end", idx=idx, rep=slot["rep"], arm=name, kind=slot["kind"], seed=slot["seed"],
               cmd=cmd, env=env_extra, cwd=wt, gpu_before=gpu_snapshot())
    append_ledger(out, dict(event="start", idx=idx, arm=name, time=time.time()))
    t0 = time.time()
    timed_out = False
    with open(stdout_path, "w") as so:
        proc = subprocess.Popen(cmd, cwd=wt, env=env, stdout=so, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=plan["timeout_s"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc = proc.returncode
    wall = time.time() - t0
    with open(stdout_path, errors="replace") as f:
        stdout = f.read()
    rel = logparse.find_log_path(stdout)
    log_copy = None
    if rel and os.path.isfile(os.path.join(wt, rel)):
        log_copy = os.path.join(out, "logs", f"{tag}_{os.path.basename(rel)}")
        shutil.copyfile(os.path.join(wt, rel), log_copy)
        log = logparse.parse_file(log_copy)
    else:
        log = logparse.parse_text(stdout, stdout_path)  # console=True lines carry the val line too
    status = classify(rc, timed_out, log)
    rec.update(time=time.time(), returncode=rc, status=status, wall_s=wall, log=log_copy, stdout=stdout_path,
               val_loss=log.val_loss, train_time_s=log.train_time_s, final_step=log.final_step,
               total_steps=log.total_steps, val_curve=log.val_curve, peak_mem_mib=log.peak_mem_mib,
               torch_version=log.torch_version, gpu_after=gpu_snapshot())
    if plan["cache"] == "cold" and cache:
        shutil.rmtree(cache, ignore_errors=True)
    append_ledger(out, rec)
    return rec


def cmd_run(a) -> None:
    out = os.path.abspath(a.out)
    plan = load_plan(out)
    arms = {x["name"]: x for x in plan["arms"]}
    ledger = read_ledger(out)
    done = {r["idx"] for r in ledger if r["event"] == "end"}
    started = {r["idx"] for r in ledger if r["event"] == "start"}
    for idx in sorted(started - done):
        print(f"slot {idx} was interrupted (harness died mid-run); recording and re-running it")
        append_ledger(out, dict(event="interrupted", idx=idx, time=time.time()))
    wts = {}
    todo = [s for s in plan["runs"] if s["idx"] not in done]
    if a.limit is not None:
        todo = todo[: a.limit]
    for name, arm in arms.items():
        if any(s["arm"] == name for s in todo):
            wts[name] = prepare_arm(plan, arm, a.allow_changed)
    total = len(plan["runs"])
    for k, slot in enumerate(todo):
        if k and plan["cooldown_s"]:
            time.sleep(plan["cooldown_s"])
        arm = arms[slot["arm"]]
        print(f"[{slot['idx'] + 1}/{total}] {slot['kind']:<7} arm={slot['arm']} rep={slot['rep']} seed={slot['seed']} ...",
              end="", flush=True)
        rec = execute(plan, out, slot, wts[slot["arm"]], arm)
        val = f"{rec['val_loss']:.4f}" if rec["val_loss"] is not None else "-"
        tt = f"{rec['train_time_s']:.3f}s" if rec["train_time_s"] is not None else "-"
        print(f" {rec['status']} val={val} train_time={tt} wall={rec['wall_s']:.0f}s")
    if not a.no_report:
        cmd_report(argparse.Namespace(out=out, at_step=None, quiet=False))


# ----------------------------------------------------------------------------- report

def final_records(out: str) -> list[dict]:
    """The last "end" record of each slot (an interrupted slot that was re-run has exactly one)."""
    last = {}
    for r in read_ledger(out):
        if r["event"] == "end":
            last[r["idx"]] = r
    return [last[i] for i in sorted(last)]


def _val(rec: dict, at_step):
    if at_step is None:
        return rec.get("val_loss")
    for s, v, _ in rec.get("val_curve") or []:
        if s == at_step:
            return v
    return None


def _time(rec: dict, at_step):
    if at_step is None:
        return rec.get("train_time_s")
    for s, _, t in rec.get("val_curve") or []:
        if s == at_step:
            return t
    return None


def summarize(groups: dict, baseline: str, target: float, at_step=None, planned: dict | None = None,
              pairs: dict | None = None) -> str:
    """groups: arm -> list of records (dicts with status, val_loss, train_time_s, val_curve, rep)."""
    lines = []
    hdr = "| arm | runs ok/total | train time (s) mean ± sd | val loss mean ± sd | gate p (val ≤ %.2f) | Δtime vs %s (Welch p, faster) | Δval vs %s (Welch p, 2-sided) |" % (
        target, baseline, baseline)
    lines += [hdr, "|" + "---|" * 7]
    base = groups.get(baseline, [])
    bt = [_time(r, at_step) for r in base if r["status"] == "ok" and _time(r, at_step) is not None]
    bv = [_val(r, at_step) for r in base if r["status"] == "ok" and _val(r, at_step) is not None]
    warnings = []
    for arm, recs in groups.items():
        ok = [r for r in recs if r["status"] == "ok"]
        ts = [_time(r, at_step) for r in ok if _time(r, at_step) is not None]
        vs = [_val(r, at_step) for r in ok if _val(r, at_step) is not None]
        total = len(recs)
        plan_n = (planned or {}).get(arm)
        tot_s = f"{len(ok)}/{total}" + (f" (planned {plan_n})" if plan_n is not None and plan_n != total else "")
        tm, tsd = stats.mean_sd(ts)
        vm, vsd = stats.mean_sd(vs)
        g = stats.gate_test(vs, target)
        cell_t = f"{tm:.3f} ± {tsd:.3f}" if ts else "-"
        cell_v = f"{vm:.5f} ± {vsd:.5f}" if vs else "-"
        cell_g = f"{g['p']:.2e} {'PASS' if g['passes'] else 'fail'}" if g["p"] == g["p"] else "-"
        if arm == baseline:
            dt_cell = dv_cell = "(baseline)"
        else:
            wt = stats.welch_test(ts, bt, "less")
            wv = stats.welch_test(vs, bv, "two-sided")
            dt_cell = f"{wt['delta']:+.3f} s ({wt['p']:.2g})" if wt["p"] == wt["p"] else "-"
            dv_cell = f"{wv['delta']*1000:+.2f} mnat ({wv['p']:.2g})" if wv["p"] == wv["p"] else "-"
        lines.append(f"| {arm} | {tot_s} | {cell_t} | {cell_v} | {cell_g} | {dt_cell} | {dv_cell} |")
        bad = [r for r in recs if r["status"] != "ok"]
        if bad:
            warnings.append(f"- **{arm}: {len(bad)} non-ok run(s)** "
                            + ", ".join(f"#{r.get('idx', '?')} {r['status']}" for r in bad)
                            + " -- counted in the run total; a certification must report them, never drop them.")
    if pairs:
        lines += ["", "Paired by rep (same seed / same block; diff = arm - baseline, two-sided t on the diffs):", "",
                  "| arm | pairs | Δtime mean ± sd (s) | p | Δval mean ± sd (mnat) | p |", "|---|---|---|---|---|---|"]
        for arm, (dts, dvs) in pairs.items():
            pt = stats.gate_test([-d for d in dts], 0.0) if len(dts) >= 2 else None
            pv = stats.gate_test([-d for d in dvs], 0.0) if len(dvs) >= 2 else None
            def two(p):
                return "-" if p is None or p["p"] != p["p"] else f"{min(1.0, 2 * min(p['p'], 1 - p['p'])):.2g}"
            mt, st = stats.mean_sd(dts)
            mv, sv = stats.mean_sd([d * 1000 for d in dvs])
            lines.append(f"| {arm} | {len(dts)} | {mt:+.3f} ± {st:.3f} | {two(pt)} | {mv:+.2f} ± {sv:.2f} | {two(pv)} |")
    if warnings:
        lines += [""] + warnings
    return "\n".join(lines)


def paired_diffs(groups: dict, baseline: str, at_step=None) -> dict:
    base = {r["rep"]: r for r in groups.get(baseline, []) if r["status"] == "ok"}
    out = {}
    for arm, recs in groups.items():
        if arm == baseline:
            continue
        dts, dvs = [], []
        for r in recs:
            b = base.get(r["rep"])
            if r["status"] != "ok" or b is None:
                continue
            if _time(r, at_step) is not None and _time(b, at_step) is not None:
                dts.append(_time(r, at_step) - _time(b, at_step))
            if _val(r, at_step) is not None and _val(b, at_step) is not None:
                dvs.append(_val(r, at_step) - _val(b, at_step))
        out[arm] = (dts, dvs)
    return out


def per_run_table(recs: list[dict], at_step=None) -> str:
    lines = ["| # | kind | arm | rep | seed | status | train time (s) | val loss | wall (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        v, t = _val(r, at_step), _time(r, at_step)
        lines.append(f"| {r['idx']} | {r['kind']} | {r['arm']} | {r['rep']} | {r['seed']} | {r['status']} | "
                     f"{'-' if t is None else f'{t:.3f}'} | {'-' if v is None else f'{v:.4f}'} | {r.get('wall_s', 0):.0f} |")
    return "\n".join(lines)


def cmd_report(a) -> str:
    out = os.path.abspath(a.out)
    plan = load_plan(out)
    at_step = a.at_step if getattr(a, "at_step", None) is not None else plan.get("at_step")
    recs = final_records(out)
    measured = [r for r in recs if r["kind"] == "measure"]
    names = [x["name"] for x in plan["arms"]]
    groups = {n: [r for r in measured if r["arm"] == n] for n in names}
    planned = {n: plan["n"] for n in names}
    n_planned = sum(1 for s in plan["runs"] if s["kind"] == "measure")
    interrupted = sum(1 for r in read_ledger(out) if r["event"] == "interrupted")
    head = [f"# A/B report: {os.path.basename(out)}", "",
            f"plan sha256 `{plan_sha(plan)}`; {len(measured)}/{n_planned} measured runs finished "
            f"({sum(1 for r in recs if r['kind'] == 'warmup')} warmup runs, excluded by pre-registration); "
            f"order={plan['order']} cache={plan['cache']} seeds={'none (unseeded)' if not plan['seeds'] else plan['seeds']}"
            + (f"; {interrupted} harness interruption(s) re-run" if interrupted else ""), ""]
    if at_step is not None:
        head += [f"Readout at fixed step {at_step} (same for every run).", ""]
    for x in plan["arms"]:
        where = f"`{x['ref']}` @ `{x['sha'][:12]}`" if x["kind"] == "ref" else f"`{x['path']}` @ `{(x['sha'] or 'no-git')[:12]}`"
        head.append(f"- **{x['name']}**: {where} env `{x['env']}`")
    body = summarize(groups, plan["baseline"], plan["target"], at_step, planned, paired_diffs(groups, plan["baseline"], at_step))
    text = "\n".join(head) + "\n\n" + body + "\n\n## All runs, in execution order\n\n" + per_run_table(recs, at_step) + "\n"
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(text)
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "kind", "arm", "rep", "seed", "status", "returncode", "train_time_s", "val_loss",
                    "final_step", "total_steps", "wall_s", "log"])
        for r in recs:
            w.writerow([r["idx"], r["kind"], r["arm"], r["rep"], r["seed"], r["status"], r.get("returncode"),
                        _time(r, at_step), _val(r, at_step), r.get("final_step"), r.get("total_steps"),
                        r.get("wall_s"), r.get("log")])
    if not getattr(a, "quiet", False):
        print(text)
    return text


def cmd_parse(a) -> str:
    groups, order = {}, []
    for spec in a.group:
        name, pat = spec.split("=", 1)
        if name not in groups:
            groups[name] = []
            order.append(name)
        for i, p in enumerate(logparse.expand([pat])):
            log = logparse.parse_file(p)
            status = "nan" if log.is_nan else ("ok" if log.completed else ("incomplete" if log.val_loss is not None else "no-val"))
            groups[name].append(dict(idx=os.path.basename(p), rep=i, status=status, val_loss=log.val_loss,
                                     train_time_s=log.train_time_s, val_curve=log.val_curve))
    text = summarize({n: groups[n] for n in order}, a.baseline or order[0], a.target, a.at_step)
    print(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="sub", required=True)

    p = sub.add_parser("plan", help="pre-register an experiment")
    p.add_argument("--out", required=True, help="results directory (must not hold a plan yet)")
    p.add_argument("--repo", default=".", help="git repo the refs resolve in (default .)")
    p.add_argument("--arm", action="append", required=True, help="NAME=ref:<gitref> | NAME=path:<dir>")
    p.add_argument("--env", action="append", help="NAME:KEY=VALUE, per-arm env var (repeatable)")
    p.add_argument("--common-env", action="append", help="KEY=VALUE for every arm")
    p.add_argument("--arm-cmd", action="append", help="NAME:COMMAND to override --cmd for one arm")
    p.add_argument("--arm-script", action="append", help="NAME:SCRIPT to override --script for one arm")
    p.add_argument("--cmd", default=DEFAULT_CMD,
                   help="command template, run in the arm's directory; placeholders {nproc} {script} {worktree} "
                        "{run_dir} {data_dir} {env_flags} (for docker -e). Default: %(default)s")
    p.add_argument("--script", default=DEFAULT_SCRIPT, help="trainer script (default %(default)s; "
                   "train_gpt_medium.py for Track 2, records/track_3_optimization/train_gpt_simple.py for Track 3)")
    p.add_argument("--n", type=int, required=True, help="measured runs per arm")
    p.add_argument("--warmup", type=int, default=0, help="untimed warmup runs per arm, run first, excluded from stats")
    p.add_argument("--order", choices=["blocked", "abba", "alternate"], default="blocked",
                   help="blocked: every rep runs each arm once in a seeded random order (default)")
    p.add_argument("--order-seed", type=int)
    p.add_argument("--seeds", default="none", help="'none' (unseeded, as certifications are), 'auto', or a comma list of n seeds; "
                   "the same seed goes to every arm within a rep")
    p.add_argument("--seed-env", default="TRAIN_SEED", help="env var the trainer reads its seed from (default %(default)s)")
    p.add_argument("--cache", choices=["cold", "warm", "inherit"], default="cold",
                   help="cold: fresh TORCHINDUCTOR/TRITON cache per run (certification setting); warm: one cache per arm, "
                        "kept across its runs (pair with --warmup 1); inherit: leave the env alone")
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--timeout-s", type=float, default=3600)
    p.add_argument("--cooldown-s", type=float, default=0)
    p.add_argument("--target", type=float, default=3.28, help="3.28 (Track 1, 3) or 2.92 (Track 2)")
    p.add_argument("--at-step", type=int, help="Track 3: read val/time at this fixed step instead of the final one")
    p.add_argument("--baseline", help="baseline arm name (default: the first --arm)")
    p.add_argument("--data-dir", help="shared data/fineweb10B to link into worktrees (default <repo>/data/fineweb10B)")
    p.add_argument("--work-root", help="where worktrees and caches go (default <out>/work)")
    p.add_argument("--go", action="store_true", help="run the plan right away")

    r = sub.add_parser("run", help="execute (or resume) a plan")
    r.add_argument("out")
    r.add_argument("--limit", type=int, help="run at most this many pending slots")
    r.add_argument("--allow-changed", action="store_true", help="run a path arm whose code changed since the plan")
    r.add_argument("--no-report", action="store_true")

    rp = sub.add_parser("report", help="tables from a results directory")
    rp.add_argument("out")
    rp.add_argument("--at-step", type=int)
    rp.add_argument("--quiet", action="store_true")

    pa = sub.add_parser("parse", help="tables from existing logs")
    pa.add_argument("--group", action="append", required=True, help="NAME=GLOB_OR_DIR (repeatable; same NAME appends)")
    pa.add_argument("--baseline")
    pa.add_argument("--target", type=float, default=3.28)
    pa.add_argument("--at-step", type=int)

    a = ap.parse_args(argv)
    if a.sub == "plan":
        out = cmd_plan(a)
        if a.go:
            cmd_run(argparse.Namespace(out=out, limit=None, allow_changed=False, no_report=False))
    elif a.sub == "run":
        cmd_run(a)
    elif a.sub == "report":
        cmd_report(a)
    else:
        cmd_parse(a)


if __name__ == "__main__":
    main()
