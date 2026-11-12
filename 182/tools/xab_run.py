#!/usr/bin/env python3
"""Paired A/B experiments declared in 182/experiments.d/*.sh (and the session's nccl phase), for 182/gpu_session.sh.

    xab_run.py spec 182/experiments.d/foo.sh          # shell assignments XS_* (eval'd by the session), validated
    xab_run.py run --out DIR --tree variants/t1-gpu-stack --arm base: --arm on:FOO=1 --seeds 1,2,3,4
    xab_run.py report DIR [--json F]                   # DIR: an xab_run directory or an ab_run.py directory
    xab_run.py check DIR                               # exit 0 complete+usable, 3 pending, 4 unusable, 5 dead arm, 2 no plan

A tree with 182/run_node.sh runs here: seed by seed, every arm once per seed (the arm order rotates by one each
seed so no arm always runs first), each run `bash TREE/182/run_node.sh run` with SEED, NPROC, LOG_DIR and the
arm's variables in EXTRA_ENV (every run_node.sh in variants/ reads it), under a wall-clock limit that kills the
whole process group. The run's result is its last `RESULT ...` line (summarize_session.parse_run). Resumable: a
slot with an ok result is never re-run; a failed slot is set aside as <slot>.failK.* and retried on the next
invocation. A run stops after --stop-after-failures consecutive failures; an arm whose first --arm-failures runs
all failed is abandoned (its other slots are not attempted). The root tree (".") goes through ab_run.py instead
(the session plans it); `report` and `check` read both kinds of directory.

The report: per arm ok/planned, val_loss and train_time mean ± sd; per arm against the baseline (paired by seed):
Δval in millinats ± SE with its 95% t-CI, Δtrain_time ± SE with its CI, and net seconds = Δtime + Δval x price
(SEC_PER_MNAT 164 ms/mnat; the 120-250 band at both ends) with the net's CI and a verdict. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_session as ss  # noqa: E402
import stats  # noqa: E402

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
ARM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,31}$")
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ------------------------------------------------------------------------------------------------ spec
def parse_arm(spec: str) -> tuple[str, dict]:
    """'name:K=V K2=V2' -> (name, {K: V}); values may be shell-quoted."""
    name, sep, rest = spec.partition(":")
    if not sep or not ARM_RE.match(name):
        raise ValueError(f"arm {spec!r} is not NAME:ENV (NAME: letters, digits, _; ENV: K=V words, may be empty)")
    env = {}
    for item in shlex.split(rest):
        k, eq, v = item.partition("=")
        if not eq or not KEY_RE.match(k):
            raise ValueError(f"arm {name}: {item!r} is not KEY=VALUE")
        env[k] = v
    return name, env


def load_spec(path: str, repo: str) -> dict:
    """Source the file in bash (the format is bash) and validate it. Returns dict(ok, error, name, tree, ...)."""
    code = ('set +u; XAB_NAME=; XAB_TREE=; XAB_ARMS=(); XAB_SEEDS=; XAB_EST_MIN=; XAB_NOTE=; '
            'source "$1" >/dev/null 2>&1 || { echo "__SOURCE_FAILED__"; exit 0; }; '
            'printf "%s\\0" "$XAB_NAME" "$XAB_TREE" "$XAB_SEEDS" "$XAB_EST_MIN" "$XAB_NOTE"; printf "%s\\0" "${XAB_ARMS[@]}"')
    out = dict(ok=False, file=path, name=os.path.splitext(os.path.basename(path))[0])
    try:
        r = subprocess.run(["bash", "-c", code, "_", path], capture_output=True, text=True, timeout=30, cwd=repo)
    except subprocess.TimeoutExpired:
        return dict(out, error="sourcing it took > 30 s")
    if "__SOURCE_FAILED__" in r.stdout or r.returncode != 0:
        return dict(out, error="bash could not source it (syntax error?)")
    f = r.stdout.split("\0")
    if len(f) < 6:
        return dict(out, error="it set no XAB_* variables")
    name, tree, seeds, est, note = f[:5]
    arms_raw = [a for a in f[5:] if a != ""]
    out.update(name=name or out["name"], tree=tree, note=note)
    try:
        if not NAME_RE.match(name):
            raise ValueError(f"XAB_NAME={name!r}: letters, digits, _ and - only (it names a directory)")
        if not tree or os.path.isabs(tree) or ".." in tree.split("/"):
            raise ValueError(f"XAB_TREE={tree!r}: a repo-relative directory (\".\" for the root)")
        tdir = os.path.join(repo, tree)
        if not os.path.isdir(tdir):
            raise ValueError(f"XAB_TREE={tree}: no such directory in this checkout")
        if tree.strip("/") not in (".", "") and not os.path.isfile(os.path.join(tdir, "182/run_node.sh")):
            raise ValueError(f"XAB_TREE={tree}: has no 182/run_node.sh")
        if tree.strip("/") in (".", "") and not os.path.isfile(os.path.join(repo, "train_gpt.py")):
            raise ValueError("XAB_TREE=.: no train_gpt.py at the repo root")
        n = int(seeds or "4")
        if not 1 <= n <= 32:
            raise ValueError(f"XAB_SEEDS={seeds}: a count of seeds, 1-32")
        e = int(est or "0")
        if e < 0:
            raise ValueError(f"XAB_EST_MIN={est}: node-minutes, >= 0")
        arms = [parse_arm(a) for a in arms_raw]
        if len(arms) < 2:
            raise ValueError(f"XAB_ARMS has {len(arms)} arm(s): a baseline and at least one more")
        names = [a for a, _ in arms]
        if len(set(names)) != len(names):
            raise ValueError(f"XAB_ARMS: duplicate arm names {names}")
    except ValueError as ex:
        return dict(out, error=str(ex))
    return dict(out, ok=True, error="", tree=tree.rstrip("/") or ".", seeds=",".join(str(i) for i in range(1, n + 1)),
                n=n, est=e, arms=arms_raw, root=tree.strip("/") in (".", ""))


def spec_shell(s: dict) -> str:
    q = shlex.quote
    lines = [f"XS_OK={int(s['ok'])}", f"XS_ERR={q(s.get('error', ''))}", f"XS_NAME={q(s.get('name', ''))}",
             f"XS_TREE={q(s.get('tree', ''))}", f"XS_SEEDS={q(s.get('seeds', ''))}", f"XS_N={s.get('n', 0)}",
             f"XS_EST={s.get('est', 0)}", f"XS_NOTE={q(s.get('note', ''))}", f"XS_ROOT={int(bool(s.get('root')))}",
             "XS_ARMS=(" + " ".join(q(a) for a in s.get("arms", [])) + ")"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------- run
def slot_order(arms: list[str], seeds: list[str]) -> list[tuple[str, str]]:
    out = []
    for i, seed in enumerate(seeds):
        k = i % len(arms)
        out += [(seed, a) for a in arms[k:] + arms[:k]]
    return out


def _tag(seed, arm):
    return f"s{seed}_{arm}"


def load_results(out: str) -> dict:
    """(seed, arm) -> latest result dict, plus failed attempts set aside."""
    d = os.path.join(out, "runs")
    res = {}
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        m = re.fullmatch(r"s([^_]+)_(.+)\.result", n)
        if not m or ".fail" in n:
            continue
        try:
            with open(os.path.join(d, n)) as f:
                res[(m.group(1), m.group(2))] = json.load(f)
        except ValueError:
            res[(m.group(1), m.group(2))] = dict(ok=False, reason="unreadable result (killed while writing)")
    return res


def dead_arms_x(plan: dict, res: dict, arm_failures: int | None = None) -> list[str]:
    k = arm_failures or plan.get("arm_failures", 2)
    dead = []
    for a in plan["arm_names"]:
        rs = [res[(s, a)] for s in plan["seeds"] if (s, a) in res]
        fails = sum(1 for d in os.listdir(os.path.join(plan["out"], "runs")) if re.fullmatch(rf"s[^_]+_{a}\.fail\d+\.result", d)) \
            if os.path.isdir(os.path.join(plan["out"], "runs")) else 0
        if rs and not any(r.get("ok") for r in rs) and len(rs) + fails >= k:
            dead.append(a)
    return dead


def run_one(cmd, env, cwd, out_path, timeout_s, grace_s) -> tuple[int, int]:
    """(rc, wall_s); 124 on timeout (the process group gets TERM, then KILL after grace_s)."""
    t0 = time.time()
    with open(out_path, "w") as fo:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True, bufsize=1)
        deadline = t0 + timeout_s
        timed_out = False
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(p.stdout, selectors.EVENT_READ)
        while True:
            left = deadline - time.time()
            if left <= 0:
                timed_out = True
                break
            if sel.select(timeout=min(left, 1.0)):
                line = p.stdout.readline()
                if line == "":
                    break
                fo.write(line)
                fo.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        if timed_out:
            msg = f"xab_run: run exceeded {timeout_s:.0f} s: stopping its process group\n"
            fo.write(msg)
            sys.stdout.write(msg)
            for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10)):
                try:
                    os.killpg(p.pid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                try:
                    p.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
            rc = 124
        else:
            rc = p.wait()
        try:
            os.killpg(p.pid, signal.SIGKILL)  # leftovers of a finished run
        except (ProcessLookupError, PermissionError):
            pass
    return rc, int(time.time() - t0)


def cmd_run(a) -> int:
    out = os.path.abspath(a.out)
    tree = os.path.abspath(a.tree)
    node = os.path.join(tree, "182/run_node.sh")
    if not os.path.isfile(node):
        print(f"xab_run: {node} does not exist (the root tree goes through ab_run.py)")
        return 2
    arms = [parse_arm(s) for s in a.arm]
    names = [n for n, _ in arms]
    if len(set(names)) != len(names):
        print(f"xab_run: duplicate arm names {names}")
        return 2
    seeds = [s for s in a.seeds.split(",") if s]
    plan = dict(kind="xab", name=a.name or os.path.basename(out), tree=a.tree, out=out, arm_names=names,
                arms=[dict(name=n, env=e) for n, e in arms], baseline=names[0], seeds=seeds,
                order=slot_order(names, seeds), nproc=a.nproc, arm_failures=a.arm_failures)
    pj = os.path.join(out, "plan.json")
    if os.path.isfile(pj):
        with open(pj) as f:
            old = json.load(f)
        same = all(old.get(k) == plan[k] for k in ("tree", "arm_names", "arms", "seeds"))
        if not same:
            print(f"xab_run: {pj} was planned with other arms/seeds/tree; move the directory aside (FORCE=1) to re-plan")
            return 2
        plan = dict(old, out=out, nproc=a.nproc, arm_failures=a.arm_failures)
    if a.dry_run:
        for seed, arm in plan["order"]:
            env = dict(plan["arms"][names.index(arm)]["env"])
            extra = " ".join(shlex.quote(f"{k}={v}") for k, v in env.items())
            print(f"+ SEED={seed} NPROC={a.nproc} LOG_DIR={out}/logs EXTRA_ENV={shlex.quote(extra)} bash {a.tree}/182/run_node.sh run"
                  f"   ({arm})")
        return 0
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    with open(pj, "w") as f:
        json.dump(plan, f, indent=1)
    res = load_results(out)
    consec, attempted = 0, set()
    for seed, arm in plan["order"]:
        r = res.get((seed, arm))
        if r and r.get("ok"):
            continue
        if arm in dead_arms_x(plan, res):
            continue
        tag = _tag(seed, arm)
        base = os.path.join(out, "runs", tag)
        if r is not None:  # set the failed attempt aside, retry
            k = 1
            while os.path.exists(f"{base}.fail{k}.result"):
                k += 1
            os.replace(base + ".result", f"{base}.fail{k}.result")
            if os.path.exists(base + ".out"):
                os.replace(base + ".out", f"{base}.fail{k}.out")
        env_arm = plan["arms"][names.index(arm)]["env"]
        extra = " ".join(shlex.quote(f"{k}={v}") for k, v in env_arm.items())
        env = dict(os.environ, SEED=str(seed), NPROC=str(a.nproc), LOG_DIR=os.path.join(out, "logs"))
        env.pop("EXTRA_ENV", None)
        if extra:
            env["EXTRA_ENV"] = extra
        print(f"-- xab {plan['name']}: seed {seed} arm {arm}{' (' + extra + ')' if extra else ''} -- {time.strftime('%T')}", flush=True)
        rc, wall = run_one(["bash", node, "run"], env, tree, base + ".out", a.timeout_s, a.kill_grace_s)
        with open(base + ".out", errors="replace") as f:
            r = ss.parse_run(f.read(), rc, wall, str(seed), None, extra)
        r.update(arm=arm, seed=str(seed))
        with open(base + ".result", "w") as f:
            json.dump(r, f)
        res[(seed, arm)] = r
        attempted.add((seed, arm))
        if r["ok"]:
            consec = 0
        else:
            consec += 1
            print(f"run seed {seed} arm {arm} FAILED (rc {rc}): {r.get('reason', '?')}", flush=True)
            if arm in dead_arms_x(plan, res):
                print(f"abandoning arm {arm}: its first {plan['arm_failures']} run(s) all failed", flush=True)
            if consec >= a.stop_after_failures:
                print(f"stopping: {consec} consecutive failed runs (fix the cause and re-run to resume)", flush=True)
                break
    code, why = check_dir(out)
    print(why)
    return 0 if code == 0 else code


# ---------------------------------------------------------------------------------------------- report
def records(d: str) -> dict | None:
    """Either kind of directory -> dict(kind, name, tree, arms, baseline, env, recs, pending, dead, planned_n)."""
    pj = os.path.join(d, "plan.json")
    if not os.path.isfile(pj):
        return None
    with open(pj) as f:
        plan = json.load(f)
    if plan.get("kind") == "xab":
        plan["out"] = d
        res = load_results(d)
        dead = dead_arms_x(plan, res)
        recs = [dict(arm=a, key=s, ok=bool(r.get("ok")), val=r.get("val_loss"), time=r.get("train_time_s"),
                     reason=r.get("reason")) for (s, a), r in res.items()]
        pending = sum(1 for s, a in plan["order"] if not (res.get((s, a)) or {}).get("ok") and a not in dead)
        return dict(kind="xab", name=plan["name"], tree=plan["tree"], arms=plan["arm_names"], baseline=plan["baseline"],
                    env={x["name"]: x["env"] for x in plan["arms"]}, recs=recs, pending=pending, dead=dead,
                    planned=len(plan["seeds"]), seeds=",".join(plan["seeds"]))
    ab = ss.load_ab(d)
    recs = [dict(arm=r["arm"], key=r["rep"], ok=r["status"] == "ok", val=r.get("val_loss"), time=r.get("train_time_s"),
                 reason=None if r["status"] == "ok" else f"{r['status']}: {ss.ab_run.record_cause(r, ab.get('dir'))}")
            for g in ab["groups"].values() for r in g]
    return dict(kind="ab", name=os.path.basename(d.rstrip("/")), tree=".", arms=ab["arms"], baseline=ab["baseline"],
                env=ab["env"], recs=recs, pending=ab["pending"], dead=ss.dead_arms(ab), planned=ab["plan"]["n"],
                seeds=",".join(str(s) for s in (ab["plan"].get("seeds") or [])) or "none")


def paired_stats(R: dict, arm: str, price: float = ss.SEC_PER_MNAT) -> dict:
    base = {r["key"]: r for r in R["recs"] if r["arm"] == R["baseline"] and r["ok"]}
    dv, dt = [], []
    for r in R["recs"]:
        b = base.get(r["key"])
        if r["arm"] != arm or not r["ok"] or b is None or None in (r["val"], b["val"], r["time"], b["time"]):
            continue
        dv.append((r["val"] - b["val"]) * 1000)
        dt.append(r["time"] - b["time"])
    mv, sev, lov, hiv, n = ss.ci95(dv)
    mt, set_, lot, hit, _ = ss.ci95(dt)
    mn, sen, lon, hin, _ = ss.ci95([t + v * price for v, t in zip(dv, dt)])
    band = sorted(ss.ci95([t + v * p for v, t in zip(dv, dt)])[0] for p in ss.SEC_PER_MNAT_BAND)
    return dict(n=n, dval=mv, dval_se=sev, dval_lo=lov, dval_hi=hiv, dtime=mt, dtime_se=set_, dtime_lo=lot, dtime_hi=hit,
                net=mn, net_lo=lon, net_hi=hin, net_band=band, price=price, dv=dv, dt=dt)


def report_lines(d: str, label: str | None = None) -> list[str]:
    R = records(d)
    if R is None:
        return [f"== {label or d}: no plan"]
    L = [f"== {label or R['name']}: tree {R['tree']}, baseline {R['baseline']}, seeds {R['seeds']}"
         + (f"  [{R['pending']} slot(s) not done]" if R["pending"] else "")
         + (f"  [abandoned: {', '.join(R['dead'])}]" if R["dead"] else "")]
    L.append(f"{'arm':<16} {'ok/plan':>7} {'val_loss mean±sd':>18} {'train_s mean±sd':>16}  env")
    for a in R["arms"]:
        ok = [r for r in R["recs"] if r["arm"] == a and r["ok"]]
        env = " ".join(f"{k}={v}" for k, v in sorted(R["env"].get(a, {}).items()) if k not in ss.HIDDEN_ENV) or "-"
        L.append(f"{a:<16} {len(ok):>3}/{R['planned']:<3} {ss._ms([r['val'] for r in ok], '{:.5f}'):>18} "
                 f"{ss._ms([r['time'] for r in ok]):>16}  {env}")
    for a in R["arms"]:
        if a == R["baseline"]:
            continue
        p = paired_stats(R, a)
        if p["n"] == 0:
            L.append(f"  {a} - {R['baseline']}: no complete seed pair")
            continue
        if p["n"] < 2:
            L.append(f"  {a} - {R['baseline']} (n=1 pair, no CI, not a result): Δval {p['dval']:+.2f} mnat, "
                     f"Δtime {p['dtime']:+.3f} s, net {p['net']:+.3f} s")
            continue
        verdict = ("net CI below 0: pays on this node" if p["net_hi"] < 0 else
                   "net CI above 0: costs time on this node" if p["net_lo"] > 0 else
                   f"net CI includes 0: n={p['n']} pairs do not settle it")
        loss = ("Δval CI includes 0 (loss unchanged within noise)" if p["dval_lo"] <= 0 <= p["dval_hi"] else
                "Δval CI excludes 0 (the loss moved)")
        L += [f"  {a} - {R['baseline']} (n={p['n']} seed pairs; + = {a} worse/slower):",
              f"    Δval  {p['dval']:+.2f} ± {p['dval_se']:.2f} mnat (SE), 95% CI [{p['dval_lo']:+.2f}, {p['dval_hi']:+.2f}]  -> {loss}",
              f"    Δtime {p['dtime']:+.3f} ± {p['dtime_se']:.3f} s (SE), 95% CI [{p['dtime_lo']:+.3f}, {p['dtime_hi']:+.3f}]",
              f"    net_s = Δtime + Δval x {p['price'] * 1000:.0f} ms/mnat = {p['net']:+.3f} s, 95% CI [{p['net_lo']:+.3f}, "
              f"{p['net_hi']:+.3f}]; at {ss.SEC_PER_MNAT_BAND[0] * 1000:.0f}-{ss.SEC_PER_MNAT_BAND[1] * 1000:.0f} ms/mnat: "
              f"{p['net_band'][0]:+.3f} to {p['net_band'][1]:+.3f} s  -> {verdict}"]
    bad = [r for r in R["recs"] if not r["ok"]]
    for r in bad[:12]:
        L.append(f"  failed: arm {r['arm']} seed/rep {r['key']}: {(r.get('reason') or '?')[:200]}")
    return L


def check_dir(d: str) -> tuple[int, str]:
    R = records(d)
    if R is None:
        return 2, f"no plan.json in {d}"
    if R["kind"] == "ab":
        code, why = ss.ab_check(ss.load_ab(d))
        if code != 0:
            return code, why
    elif R["pending"]:
        return 3, f"{R['pending']} slot(s) not done"
    elif R["dead"]:
        return 5, f"arm(s) {', '.join(R['dead'])} failed every run"
    thin = [a for a in R["arms"] if a != R["baseline"] and paired_stats(R, a)["n"] < 2]
    if thin:
        return 4, f"fewer than 2 seed pairs for {', '.join(thin)}"
    return 0, f"complete: {len(R['arms'])} arms x {R['planned']} seeds"


def summary_json(d: str) -> dict:
    R = records(d)
    if R is None:
        return {}
    return dict(name=R["name"], tree=R["tree"], baseline=R["baseline"], pending=R["pending"], dead=R["dead"],
                arms={a: {k: v for k, v in paired_stats(R, a).items()} for a in R["arms"] if a != R["baseline"]})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("spec")
    p.add_argument("file")
    p.add_argument("--repo", default=".")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--tree", required=True)
    r.add_argument("--arm", action="append", required=True, help="NAME:K=V K2=V2 (the first is the baseline)")
    r.add_argument("--seeds", required=True, help="comma list")
    r.add_argument("--name")
    r.add_argument("--nproc", type=int, default=8)
    r.add_argument("--timeout-s", type=float, default=1500)
    r.add_argument("--kill-grace-s", type=float, default=60)
    r.add_argument("--stop-after-failures", type=int, default=3)
    r.add_argument("--arm-failures", type=int, default=2)
    r.add_argument("--dry-run", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("dir")
    rp.add_argument("--label")
    rp.add_argument("--json")
    c = sub.add_parser("check")
    c.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        sys.stdout.write(spec_shell(load_spec(a.file, os.path.abspath(a.repo))))
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "report":
        print("\n".join(report_lines(a.dir, a.label)))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(summary_json(a.dir), f, indent=1)
        return 0
    code, why = check_dir(a.dir)
    print(why)
    return code


if __name__ == "__main__":
    sys.exit(main())
