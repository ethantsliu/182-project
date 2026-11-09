#!/usr/bin/env python3
"""One compact summary of a GPU session (182/gpu_session.sh): a bundle .tar.gz or its unpacked directory.

    python 182/tools/summarize_session.py 182/results/session-2026-11-10-abc1234.tar.gz
    python 182/tools/summarize_session.py 182/results/session-abc1234              # the live directory
    python 182/tools/summarize_session.py --gate DIR --write-decision DIR/decision.env   # gate phase only

Per A/B phase (gate, hadamard; each an ab_run.py results directory): per arm n ok/total, final val_loss
mean/sd, train_time mean/sd, step-sum (train_time of the last training-progress line, i.e. the summed step
time without the final validation), ms/step per schedule stage, and the arm's flags; then the paired
(same-seed) deltas against the baseline arm. The gate gets the README decision rule: the paired gap
val(fp8) - val(FP8_OFF=1) in mnat, mean +- SE, 95% t-CI against the 0.5 mnat threshold. The t2 phase
is compared with @ldmberman's single run (2.8769 in 221 s, 38.1 ms/step).

Stdlib only (reuses logparse/stats/ab_run from this folder), so it runs anywhere.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_run  # noqa: E402
import logparse  # noqa: E402
import stats  # noqa: E402

THRESHOLD_MNAT = 0.5          # 182/hadamard-fp8/README.md decision rule
SEC_PER_MNAT = 0.164          # #360 near its step count (182/structured/ab_fixed_steps.py)
# The hadamard phase only pays if the gap can buy back its cost: the cheapest default variant (dw1,dw2@32)
# is estimated at ~0.5-0.7 s, i.e. ~3-4 mnat at SEC_PER_MNAT, so a mean gap under ~2 mnat cannot win and
# its ~34 GPU-node-minutes are better spent elsewhere. FORCE_HADAMARD=1 in gpu_session.sh overrides.
HADAMARD_MIN_GAP_MNAT = float(os.environ.get("HADAMARD_MIN_GAP_MNAT", "2.0"))
T2_REF = dict(val_loss=2.8769, train_time_s=221.0, step_avg_ms=38.1, target=2.877)
HIDDEN_ENV = {"DATA_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TRAIN_SEED"}

STAGE_BOUNDS_RE = re.compile(r"stage boundaries=(\[[0-9(), ]*\])")
STAGE_ENDS_RE = re.compile(r"\[schedule\] stage ends=(\[[0-9, ]*\])")
FP8_FLAGS_RE = re.compile(r"^fp8 experiment: (.*)$", re.M)
ENDPOINT_RE = re.compile(r"^endpoint_result: (.*)$", re.M)


# ----------------------------------------------------------------------------- per-log details

def stage_ends(text: str) -> list[int] | None:
    """Schedule stage ends [0, e1, ..., total] from the trainer's startup line (either format)."""
    if m := STAGE_ENDS_RE.search(text):
        return list(ast.literal_eval(m.group(1)))
    if m := STAGE_BOUNDS_RE.search(text):
        pairs = ast.literal_eval(m.group(1))
        if pairs:
            return [pairs[0][0]] + [b for _, b in pairs]
    return None


def progress_points(text: str) -> list[tuple[int, float]]:
    """(step, cumulative train_time s) of the training-progress lines (no val lines), with (0, 0)."""
    pts = {0: 0.0}
    for m in logparse.TRAIN_RE.finditer(logparse._body(text)):
        pts[int(m.group(1))] = logparse._seconds(m.group(3), m.group(4))
    return sorted(pts.items())


def _interp(pts, step):
    for (s0, t0), (s1, t1) in zip(pts, pts[1:]):
        if s0 <= step <= s1:
            return t0 + (t1 - t0) * (step - s0) / (s1 - s0) if s1 != s0 else t0
    return None


def log_details(text: str) -> dict:
    """step_sum_s and per-stage ms/step (interpolated between progress lines, which come every ~25 steps)."""
    pts = progress_points(text)
    run = logparse.parse_text(text)
    total = run.total_steps
    step_sum = next((t for s, t in reversed(pts) if total is not None and s == total), None)
    ends = stage_ends(text)
    per_stage = None
    if ends and len(pts) > 1:
        per_stage = []
        for a, b in zip(ends, ends[1:]):
            ta, tb = _interp(pts, a), _interp(pts, b)
            per_stage.append(None if ta is None or tb is None or b == a else (tb - ta) * 1000 / (b - a))
    fl = FP8_FLAGS_RE.findall(text)
    return dict(step_sum_s=step_sum, stage_ms=per_stage, stage_ends=ends, fp8_line=fl[-1] if fl else None)


# ----------------------------------------------------------------------------- A/B phases

def _remap(path: str | None, ab_dir: str, sub: str) -> str | None:
    """Ledger paths are absolute on the GPU box; in a bundle the file sits in <ab_dir>/<sub>/<basename>."""
    if not path:
        return None
    if os.path.isfile(path):
        return path
    p = os.path.join(ab_dir, sub, os.path.basename(path))
    return p if os.path.isfile(p) else None


def load_ab(ab_dir: str) -> dict | None:
    pj = os.path.join(ab_dir, "plan.json")
    if not os.path.isfile(pj):
        return None
    with open(pj) as f:
        plan = json.load(f)
    recs = [r for r in ab_run.final_records(ab_dir) if r.get("kind") == "measure"]
    for r in recs:
        p = _remap(r.get("log"), ab_dir, "logs") or _remap(r.get("stdout"), ab_dir, "stdout")
        r["details"] = {}
        if p:
            with open(p, errors="replace") as f:
                r["details"] = log_details(f.read())
    arms = [a["name"] for a in plan["arms"]]
    done = {r["idx"] for r in ab_run.final_records(ab_dir)}
    return dict(plan=plan, arms=arms, baseline=plan["baseline"], pending=sum(1 for s in plan["runs"] if s["idx"] not in done),
                groups={a: [r for r in recs if r["arm"] == a] for a in arms},
                planned={a: plan["n"] for a in arms}, env={a["name"]: a.get("env", {}) for a in plan["arms"]})


def _ms(xs, fmt="{:.3f}", sd=True):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    m, s = stats.mean_sd(xs)
    return (fmt.format(m) + ("±" + fmt.format(s) if sd and s == s else "")) if xs else "-"


def paired(groups, baseline, arm, key):
    base = {r["rep"]: r for r in groups.get(baseline, []) if r["status"] == "ok"}
    out = []
    for r in groups.get(arm, []):
        b = base.get(r["rep"])
        if r["status"] != "ok" or b is None:
            continue
        va, vb = key(r), key(b)
        if va is not None and vb is not None:
            out.append(va - vb)
    return out


def ci95(diffs):
    """(mean, se, lo, hi, n) of a list of paired differences; NaN where undefined."""
    n = len(diffs)
    m, sd = stats.mean_sd(diffs)
    if n < 2:
        return m, float("nan"), float("nan"), float("nan"), n
    se = sd / math.sqrt(n)
    h = stats.t_ppf(0.975, n - 1) * se
    return m, se, m - h, m + h, n


def gate_decision(ab: dict) -> dict:
    """Gap = val(fp8 arm) - val(bf16 arm, FP8_OFF set) per seed, in mnat. Decision per the README."""
    bf16 = [a for a in ab["arms"] if ab["env"][a].get("FP8_OFF")]
    fp8 = [a for a in ab["arms"] if not ab["env"][a].get("FP8_OFF")]
    if len(bf16) != 1 or len(fp8) != 1:
        return dict(decision="NODATA", reason=f"gate needs one FP8_OFF arm and one fp8 arm, got {ab['arms']}")
    diffs = [-d * 1000 for d in paired(ab["groups"], fp8[0], bf16[0], lambda r: r.get("val_loss"))]
    m, se, lo, hi, n = ci95(diffs)
    res = dict(fp8_arm=fp8[0], bf16_arm=bf16[0], n=n, mean_mnat=m, se_mnat=se, lo_mnat=lo, hi_mnat=hi,
               threshold_mnat=THRESHOLD_MNAT, diffs_mnat=diffs)
    if n < 2:
        res.update(decision="NODATA", reason=f"{n} complete seed pair(s); need >= 2 (plan: {ab['plan']['n']})")
    elif hi < THRESHOLD_MNAT:
        res.update(decision="STOP", reason="95% CI upper < 0.5 mnat: Hadamard is pointless, drop it")
    elif lo > THRESHOLD_MNAT:
        res.update(decision="GAP", reason="95% CI lower > 0.5 mnat: attribute it (FP8_OFF=mlp/qkv/lm), dump, offline script")
    else:
        res.update(decision="INCONCLUSIVE", reason="95% CI straddles 0.5 mnat: more seeds needed")
    res["min_gap_mnat"] = HADAMARD_MIN_GAP_MNAT
    res["run_hadamard"] = res["decision"] in ("GAP", "INCONCLUSIVE") and m >= HADAMARD_MIN_GAP_MNAT
    if n >= 2 and n < ab["plan"]["n"]:
        res["reason"] += f" (only {n}/{ab['plan']['n']} seed pairs complete)"
    return res


def ab_check(ab: dict, gate: bool = False) -> tuple[int, str]:
    """(code, reason): 0 complete and usable; 3 slots still pending (resume them); 4 complete but unusable
    (the gate has < 2 seed pairs, or some arm has no ok run): re-plan, the slots will not be re-run."""
    if ab["pending"]:
        return 3, f"{ab['pending']} slot(s) pending"
    if gate:
        g = gate_decision(ab)
        if g["decision"] == "NODATA":
            return 4, g["reason"]
    empty = [a for a in ab["arms"] if not any(r["status"] == "ok" for r in ab["groups"][a])]
    if empty:
        return 4, "no ok run in arm(s) " + ", ".join(empty)
    return 0, "complete"


def ab_table(name: str, ab: dict) -> list[str]:
    plan, groups, base = ab["plan"], ab["groups"], ab["baseline"]
    lines = [f"== {name}: baseline={base} n/arm={plan['n']} seeds={plan['seeds'] or 'none'} cache={plan['cache']} "
             f"order={plan['order']}" + (f"  [{ab['pending']} slot(s) never ran]" if ab.get("pending") else "")]
    nst = max((len(r["details"].get("stage_ms") or []) for g in groups.values() for r in g), default=0)
    hdr = f"{'arm':<16} {'ok/plan':>7} {'val_loss mean±sd':>18} {'train_s mean±sd':>16} {'step_sum_s':>11}  {'ms/step by stage':<{max(16, 7 * nst)}}  flags"
    lines.append(hdr)
    for arm in ab["arms"]:
        recs = groups[arm]
        ok = [r for r in recs if r["status"] == "ok"]
        stage_cols = []
        for i in range(nst):
            xs = [r["details"]["stage_ms"][i] for r in ok
                  if r["details"].get("stage_ms") and len(r["details"]["stage_ms"]) > i and r["details"]["stage_ms"][i] is not None]
            stage_cols.append(f"{sum(xs) / len(xs):6.2f}" if xs else "     -")
        flags = " ".join(f"{k}={v}" for k, v in sorted(ab["env"][arm].items()) if k not in HIDDEN_ENV) or "-"
        bad = [r["status"] for r in recs if r["status"] != "ok"]
        lines.append(f"{arm:<16} {len(ok):>3}/{ab['planned'][arm]:<3} {_ms([r.get('val_loss') for r in ok], '{:.5f}'):>18} "
                     f"{_ms([r.get('train_time_s') for r in ok]):>16} "
                     f"{_ms([r['details'].get('step_sum_s') for r in ok], '{:.3f}', sd=False):>11}  "
                     f"{' '.join(stage_cols) or '-':<{max(16, 7 * nst)}}  {flags}"
                     + (f"  [non-ok: {','.join(bad)}]" if bad else ""))
    for arm in ab["arms"]:
        if arm == base:
            continue
        dv = [d * 1000 for d in paired(groups, base, arm, lambda r: r.get("val_loss"))]
        dt = paired(groups, base, arm, lambda r: r["details"].get("step_sum_s"))
        dtt = paired(groups, base, arm, lambda r: r.get("train_time_s"))
        mv, sev, lov, hiv, n = ci95(dv)
        mt, set_, _, _, _ = ci95(dt)
        mtt, _, _, _, _ = ci95(dtt)
        net = (mt + mv * SEC_PER_MNAT) if dt and dv else float("nan")
        lines.append(f"  paired {arm}-{base} (n={n}): Δval {mv:+.2f}±{sev:.2f} mnat (SE) CI[{lov:+.2f},{hiv:+.2f}] | "
                     f"Δstep_sum {mt:+.3f}±{set_:.3f} s | Δtrain_time {mtt:+.3f} s | "
                     f"net {net:+.2f} s (Δstep_sum + Δval x {SEC_PER_MNAT} s/mnat; + = {arm} slower to the same loss)")
    return lines


def gate_lines(g: dict) -> list[str]:
    if g["decision"] == "NODATA":
        return [f"GATE: NODATA -- {g['reason']}"]
    return [f"GATE: gap val({g['fp8_arm']}) - val({g['bf16_arm']}) = {g['mean_mnat']:+.2f} ± {g['se_mnat']:.2f} mnat "
            f"(SE, n={g['n']} seed pairs), 95% CI [{g['lo_mnat']:+.2f}, {g['hi_mnat']:+.2f}] vs {THRESHOLD_MNAT} mnat "
            f"-> {g['decision']}: {g['reason']}",
            "      per-seed gaps (mnat): " + " ".join(f"{d:+.1f}" for d in g["diffs_mnat"])]


# ----------------------------------------------------------------------------- t2

def t2_result(t2_dir: str) -> dict | None:
    cands = [os.path.join(t2_dir, f) for f in sorted(os.listdir(t2_dir)) if f.endswith(".txt")] if os.path.isdir(t2_dir) else []
    best = None
    for p in cands:
        with open(p, errors="replace") as f:
            text = f.read()
        r = logparse.parse_text(text, p)
        ep = ENDPOINT_RE.findall(text)
        if ep:
            kv = dict(item.split("=", 1) for item in ep[-1].split() if "=" in item)
            r_val = float(kv.get("val_loss", "nan"))
            r_time = float(kv["train_time_ms"]) / 1000 if "train_time_ms" in kv else r.train_time_s
            cand = dict(path=os.path.basename(p), val_loss=r_val, train_time_s=r_time, steps=int(kv.get("steps", 0)) or r.final_step,
                        step_avg_ms=r.step_avg_ms, completed=True)
        elif r.val_loss is not None:
            cand = dict(path=os.path.basename(p), val_loss=r.val_loss, train_time_s=r.train_time_s, steps=r.final_step,
                        step_avg_ms=r.step_avg_ms, completed=r.completed)
        else:
            continue
        if best is None or (cand["completed"] and not best["completed"]):
            best = cand
    return best


def t2_lines(t2_dir: str) -> list[str]:
    r = t2_result(t2_dir)
    if r is None:
        return ["== t2: no result"]
    dv = (r["val_loss"] - T2_REF["val_loss"]) * 1000
    tt = r["train_time_s"] if r["train_time_s"] is not None else float("nan")
    dt = tt - T2_REF["train_time_s"]
    ok = r["val_loss"] <= T2_REF["target"] + 0.003
    return [f"== t2 ({r['path']}): steps={r['steps']}{'' if r['completed'] else ' (INCOMPLETE)'} val_loss={r['val_loss']:.4f} "
            f"train_time={tt:.1f} s step_avg={r['step_avg_ms'] or float('nan'):.2f} ms | "
            f"ref 2.8769 in 221 s (38.1 ms): Δval {dv:+.1f} mnat, Δtime {dt:+.1f} s -> "
            f"{'reproduces (≤ 2.880)' if ok else 'does NOT reproduce ~2.877'}"]


# ----------------------------------------------------------------------------- whole session

def statuses(root: str) -> list[str]:
    d = os.path.join(root, "status")
    if not os.path.isdir(d):
        return []
    order = ["env", "tests", "sanity", "gate", "hadamard", "t2", "bundle"]
    names = sorted(os.listdir(d), key=lambda n: (order.index(n) if n in order else 99, n))
    out = []
    for n in names:
        with open(os.path.join(d, n)) as f:
            out.append(f"{n}: {f.read().strip()}")
    return out


def summarize(root: str) -> str:
    lines = []
    meta = os.path.join(root, "meta.txt")
    if os.path.isfile(meta):
        with open(meta) as f:
            lines += [ln.rstrip() for ln in f if ln.strip()]
    st = statuses(root)
    if st:
        lines += ["phases: " + " | ".join(st)]
    attempts = sorted(n for n in os.listdir(root) if re.fullmatch(r"(gate|hadamard)\.attempt\d+", n))
    for phase in attempts + ["gate", "hadamard"]:  # earlier attempts (re-planned after failing) first
        ab = load_ab(os.path.join(root, phase))
        if ab is None:
            continue
        lines += [""] + ab_table(phase, ab)
        if phase.startswith("gate"):
            lines += gate_lines(gate_decision(ab))
    if os.path.isdir(os.path.join(root, "t2")):
        lines += [""] + t2_lines(os.path.join(root, "t2"))
    return "\n".join(lines) + "\n"


def open_input(path: str) -> tuple[str, tempfile.TemporaryDirectory | None]:
    if os.path.isdir(path):
        return path, None
    tmp = tempfile.TemporaryDirectory()
    with tarfile.open(path) as tf:
        members = [m for m in tf.getmembers() if (m.isfile() or m.isdir()) and not m.name.startswith("/")
                   and ".." not in m.name.split("/") and not os.path.basename(m.name).startswith("._")]  # macOS AppleDouble
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(tmp.name, members=members, **kw)
    tops = [t for t in os.listdir(tmp.name) if not t.startswith("._")]
    root = os.path.join(tmp.name, tops[0]) if len(tops) == 1 and os.path.isdir(os.path.join(tmp.name, tops[0])) else tmp.name
    return root, tmp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="session bundle .tar.gz or session directory")
    ap.add_argument("--gate", help="an ab_run gate directory: print its table and decision only")
    ap.add_argument("--write-decision", help="with --gate: write GATE_* shell variables to this file")
    ap.add_argument("--check", metavar="DIR", help="an ab_run directory: print its state; exit 0 complete and usable, "
                    "3 slots pending, 4 complete but unusable, 2 no plan")
    ap.add_argument("--check-gate", action="store_true", help="with --check: also require a gate decision (>= 2 pairs)")
    a = ap.parse_args(argv)
    if a.check:
        ab = load_ab(a.check)
        if ab is None:
            print(f"no plan.json in {a.check}")
            return 2
        code, why = ab_check(ab, a.check_gate)
        print(why)
        return code
    if a.gate:
        ab = load_ab(a.gate)
        if ab is None:
            raise SystemExit(f"no plan.json in {a.gate}")
        g = gate_decision(ab)
        print("\n".join(ab_table("gate", ab) + gate_lines(g)))
        if a.write_decision:
            def num(x):
                return "nan" if x is None or x != x else f"{x:.4f}"
            with open(a.write_decision, "w") as f:
                f.write(f"GATE_DECISION={g['decision']}\nGATE_N={g.get('n', 0)}\nGATE_MEAN_MNAT={num(g.get('mean_mnat'))}\n"
                        f"GATE_SE_MNAT={num(g.get('se_mnat'))}\nGATE_LO_MNAT={num(g.get('lo_mnat'))}\n"
                        f"GATE_HI_MNAT={num(g.get('hi_mnat'))}\nGATE_RUN_HADAMARD={int(bool(g.get('run_hadamard')))}\n")
        return 0
    if not a.input:
        ap.error("give a bundle/directory, or --gate DIR")
    root, tmp = open_input(a.input)
    try:
        sys.stdout.write(summarize(root))
    finally:
        if tmp:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
