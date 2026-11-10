#!/usr/bin/env python3
"""One compact summary of a GPU session (182/gpu_session.sh): a bundle .tar.gz or its unpacked directory.

    python 182/tools/summarize_session.py 182/results/session-2026-11-10-abc1234.tar.gz
    python 182/tools/summarize_session.py 182/results/session-abc1234              # the live directory
    python 182/tools/summarize_session.py --gate DIR --write-decision DIR/decision.env   # gate phase only

Per A/B phase (gate, hadamard; each an ab_run.py results directory): per arm n ok/total, final val_loss
mean/sd, train_time mean/sd, step-sum (train_time of the last training-progress line, i.e. the summed step
time without the final validation), ms/step per schedule stage, and the arm's flags; then the paired
(same-seed) deltas against the baseline arm, and each failed run's cause. The gate gets the first step of
182/hadamard-fp8/README.md's workflow: the paired gap val(fp8) - val(FP8_OFF=1) in mnat (one sign convention
throughout: + = fp8 worse), mean +- SE, 95% t-CI against the 0.5 mnat threshold, and only once every gate
slot has finished. The hadamard phase compares each variant's step-time cost with the most the rotation could
buy back (gate gap x SEC_PER_MNAT). A header shows the stack the session ran on (git sha/dirty, torch, triton,
CUDA, driver, FA3, SLOW GPUs) with mismatches marked. The t2 phase is compared with @ldmberman's single run
(2.8769 in 221 s, 38.1 ms/step).

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
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_run  # noqa: E402
import logparse  # noqa: E402
import stats  # noqa: E402

THRESHOLD_MNAT = 0.5          # 182/hadamard-fp8/README.md decision rule
SEC_PER_MNAT = 0.164          # upstream's convention (#360 near its step count; 182/structured/ab_fixed_steps.py)
SEC_PER_MNAT_BAND = (0.120, 0.250)  # what was measured across step counts
# A loose prescreen for the DEFAULT HADAMARD_VARIANTS (all@128, dw1,dw2@32; ~0.5-0.7 s each, i.e. ~3-4.3 mnat at
# SEC_PER_MNAT): under a 2 mnat mean gap they cannot pay, and the phase's ~34 node-minutes go elsewhere. It does NOT
# exclude cheaper variants: dpost@32 (~0.08 s) needs ~0.5 mnat, a single GEMM @32 ~1.5-3. FORCE_HADAMARD=1 overrides.
HADAMARD_MIN_GAP_MNAT = float(os.environ.get("HADAMARD_MIN_GAP_MNAT", "2.0"))
WANT_STACK = dict(torch="2.10.0+cu128", triton="3.6.0")
MIN_DRIVER = 580
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
    """Ledger paths are absolute on the GPU box; the file sits in <ab_dir>/<sub>/<basename> (in a bundle, and in a
    re-planned <phase>.attemptN, whose absolute paths the new attempt reuses), so that copy wins."""
    if not path:
        return None
    p = os.path.join(ab_dir, sub, os.path.basename(path))
    if os.path.isfile(p):
        return p
    return path if os.path.isfile(path) else None


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
    return dict(dir=ab_dir, plan=plan, arms=arms, baseline=plan["baseline"], pending=sum(1 for s in plan["runs"] if s["idx"] not in done),
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
    pending = ab.get("pending", 0)
    res["complete"] = not pending
    if pending:
        res.update(decision="INCOMPLETE", reason=f"{pending} gate slot(s) never ran: no decision until they finish "
                   "(re-run the same command to resume)")
    elif n < 2:
        res.update(decision="NODATA", reason=f"{n} complete seed pair(s); need >= 2 (plan: {ab['plan']['n']})")
    elif hi < THRESHOLD_MNAT:
        res.update(decision="STOP", reason="95% CI upper < 0.5 mnat: the fp8 recipe costs no meaningful loss; drop Hadamard")
    elif lo > THRESHOLD_MNAT:
        res.update(decision="GAP", reason="95% CI lower > 0.5 mnat: attribute it (FP8_OFF=mlp/qkv/lm), dump, offline script")
    else:
        sd = se * math.sqrt(n)
        need = pairs_needed(m, sd)
        res.update(decision="INCONCLUSIVE",
                   reason=f"95% CI straddles 0.5 mnat (paired sd {sd:.2f} mnat); "
                   + (f"calling it either way would take n~{need} pairs if the true gap is the observed mean"
                      if need else "the observed mean sits so close to 0.5 that no affordable n resolves it"))
    res["min_gap_mnat"] = HADAMARD_MIN_GAP_MNAT
    res["run_hadamard"] = bool(res["complete"] and res["decision"] in ("GAP", "INCONCLUSIVE") and m >= HADAMARD_MIN_GAP_MNAT)
    if not pending and 2 <= n < ab["plan"]["n"]:
        res["reason"] += f" (only {n}/{ab['plan']['n']} seed pairs ok)"
    return res


def pairs_needed(mean: float, sd: float, thr: float = THRESHOLD_MNAT, cap: int = 400) -> int | None:
    """Smallest n whose 95% t-CI (half-width t*sd/sqrt(n)) clears thr, if the true paired gap is `mean`."""
    d = abs(mean - thr)
    if not (sd == sd and sd > 0) or d == 0:
        return None
    for n in range(2, cap + 1):
        if stats.t_ppf(0.975, n - 1) * sd / math.sqrt(n) < d:
            return n
    return None


def dead_arms(ab: dict) -> list[str]:
    """Arms that finished at least one run and none ok: a systematic failure (crash at import, hang, ...)."""
    return [a for a in ab["arms"] if ab["groups"][a] and not any(r["status"] == "ok" for r in ab["groups"][a])]


def ab_check(ab: dict, gate: bool = False) -> tuple[int, str]:
    """(code, reason): 0 complete and usable; 3 slots still pending (resume them); 4 complete but unusable
    (the gate has < 2 seed pairs from scattered failures): re-plan, the slots will not be re-run; 5 complete
    but some arm failed every run: systematic, re-planning would burn the same runs again (fix it, FORCE=1)."""
    dead = dead_arms(ab)
    if ab["pending"]:
        note = f" (arm(s) {', '.join(dead)}: every finished run failed)" if dead else ""
        return 3, f"{ab['pending']} slot(s) pending{note}"
    if dead:
        causes = "; ".join(f"{a}: {first_cause(ab, a)}" for a in dead)
        return 5, f"arm(s) {', '.join(dead)} failed every run ({causes})"
    if gate:
        g = gate_decision(ab)
        if g["decision"] == "NODATA":
            return 4, g["reason"]
    empty = [a for a in ab["arms"] if not any(r["status"] == "ok" for r in ab["groups"][a])]
    if empty:
        return 4, "no ok run in arm(s) " + ", ".join(empty)
    return 0, "complete"


def first_cause(ab: dict, arm: str) -> str:
    bad = [r for r in ab["groups"][arm] if r["status"] != "ok"]
    return ab_run.record_cause(bad[0], ab.get("dir")) if bad else "-"


def failure_lines(ab: dict) -> list[str]:
    """One line per distinct failure cause per arm (the stdout tail a crash left)."""
    lines = []
    for arm in ab["arms"]:
        seen = {}
        for r in ab["groups"][arm]:
            if r["status"] == "ok":
                continue
            c = ab_run.record_cause(r, ab.get("dir"))
            seen.setdefault(c, []).append(f"#{r['idx']} {r['status']}" + (f" rc={r['returncode']}" if r.get("returncode") not in (None, 0) else ""))
        for c, runs in seen.items():
            lines.append(f"  {arm} {', '.join(runs)}: {c}")
    return (["  failed runs (cause = last error line of the run's stdout):"] + lines) if lines else []


def ab_table(name: str, ab: dict, gate: bool = False, buyback_mnat: float | None = None) -> list[str]:
    """gate: the paired val gap is gate_lines' job (one sign convention), so only Δstep_sum is shown here.
    buyback_mnat: the gate gap, against which each arm's step-time cost is compared (the hadamard phase)."""
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
        if gate:
            lines.append(f"  paired {arm}-{base} (n={n}): Δstep_sum {mt:+.3f}±{set_:.3f} s | Δtrain_time {mtt:+.3f} s "
                         f"(the step-time cost of {arm}'s flags; the val gap is the GATE line)")
            continue
        lines.append(f"  paired {arm}-{base} (n={n}): Δval {mv:+.2f}±{sev:.2f} mnat (SE) CI[{lov:+.2f},{hiv:+.2f}] | "
                     f"Δstep_sum {mt:+.3f}±{set_:.3f} s | Δtrain_time {mtt:+.3f} s  (+ = {arm} worse/slower)")
        if buyback_mnat is not None and dt:
            gap = max(buyback_mnat, 0.0)
            b, lo_b, hi_b = gap * SEC_PER_MNAT, gap * SEC_PER_MNAT_BAND[0], gap * SEC_PER_MNAT_BAND[1]
            verdict = ("cannot pay even if it recovers the whole gap" if mt > hi_b else
                       "pays only if it recovers most of the gap" if mt > lo_b else
                       "could pay if it recovers a fair share of the gap")
            lines.append(f"    buy-back bound: gate gap {gap:.2f} mnat x {SEC_PER_MNAT} s/mnat (upstream's convention; measured "
                         f"{SEC_PER_MNAT_BAND[0] * 1000:.0f}-{SEC_PER_MNAT_BAND[1] * 1000:.0f} ms/mnat) = {b:.2f} s "
                         f"[{lo_b:.2f}-{hi_b:.2f}] vs Δstep_sum {mt:+.3f} s -> {verdict}"
                         + (f"; Δval at n={n} is noise-level (CI ±{(hiv - lov) / 2:.1f} mnat)" if n == n and n < 4 and hiv == hiv else ""))
    return lines + failure_lines(ab)


def gate_lines(g: dict) -> list[str]:
    if g["decision"] == "NODATA" or (g["decision"] == "INCOMPLETE" and g.get("n", 0) < 2) or "mean_mnat" not in g:
        return [f"GATE: {g['decision']} -- {g['reason']}"]
    gap = (f"gap val({g['fp8_arm']}) - val({g['bf16_arm']}) = {g['mean_mnat']:+.2f} ± {g['se_mnat']:.2f} mnat "
           f"(SE, n={g['n']} seed pairs, + = fp8 worse), 95% CI [{g['lo_mnat']:+.2f}, {g['hi_mnat']:+.2f}]")
    out = [f"GATE: {g['decision']} -- {g['reason']}" if g["decision"] == "INCOMPLETE" else
           f"GATE: {gap} vs {THRESHOLD_MNAT} mnat -> {g['decision']}: {g['reason']}"]
    if g["decision"] == "INCOMPLETE":
        out.append(f"      provisional, not a decision: {gap}")
    out.append("      per-seed gaps (mnat): " + " ".join(f"{d:+.1f}" for d in g["diffs_mnat"]))
    if g["decision"] in ("GAP", "INCONCLUSIVE"):
        if g["mean_mnat"] < g["min_gap_mnat"]:
            out.append(f"      hadamard: mean gap {g['mean_mnat']:.2f} < {g['min_gap_mnat']:.1f} mnat, the loose prescreen for the default "
                       f"HADAMARD_VARIANTS (~0.5-0.7 s each = ~3-4.3 mnat at {SEC_PER_MNAT} s/mnat): they cannot pay. It does not "
                       "exclude cheaper variants (dpost@32 ~0.08 s needs ~0.5 mnat).")
        elif g["complete"]:
            out.append(f"      hadamard: mean gap {g['mean_mnat']:.2f} >= {g['min_gap_mnat']:.1f} mnat: the variants are worth timing")
    return out


# ----------------------------------------------------------------------------- t2

def tests_lines(root: str) -> list[str]:
    d = os.path.join(root, "tests")
    out = []
    for name, rcf, label in (("pytest.txt", "rc_plain", "plain"), ("pytest_hadamard.txt", "rc_hadamard", "HADAMARD_FP8=1")):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        rc = _read(os.path.join(d, rcf)).strip() or "?"
        lines = _read(p).splitlines()
        final = next((ln.strip(" =") for ln in reversed(lines) if re.search(r"\d+ (passed|failed|error|skipped)|no tests ran", ln)), "(no pytest summary line)")
        out.append(f"== tests ({label}): rc={rc}  {final}")
        bad = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))]
        skips = [ln for ln in lines if ln.startswith("SKIPPED")]
        out += ["  " + ln[:200] for ln in bad[:12]]
        if rc not in ("0", "?") and not bad:
            out += ["  " + ln[:200] for ln in lines[-8:]]
        out += ["  " + ln[:200] for ln in skips[:6]] + ([f"  ... {len(skips) - 6} more SKIPPED"] if len(skips) > 6 else [])
    return out


def _read(p: str) -> str:
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def stack_lines(root: str) -> list[str]:
    """The header: what the session ran on, with every mismatch against the record's stack marked."""
    d = os.path.join(root, "env")
    st = {}
    try:
        with open(os.path.join(d, "stack.json")) as f:
            st = json.load(f)
    except (OSError, ValueError):
        text = _read(os.path.join(d, "env.txt"))
        for key, rx in (("torch", r"^torch (\S+) cuda (\S+)"), ("triton", r"^triton (\S+)"), ("git_sha", r"^git ([0-9a-f]{7,40})")):
            if m := re.search(rx, text, re.M):
                st[key] = m.group(1)
                if key == "torch":
                    st["torch_cuda"] = m.group(2)
        if m := re.search(r"Driver Version: (\S+)", text):
            st["driver"] = m.group(1)
    if not st:
        return ["STACK: unknown (no env/stack.json or env.txt in this session)"]
    flag = []
    dirty = st.get("git_dirty")
    line = (f"STACK: git {str(st.get('git_sha', '?'))[:12]}{' DIRTY (' + str(len(st.get('dirty_files') or [])) + ' modified tracked file(s))' if dirty else (' clean' if dirty is False else '')}"
            f" | torch {st.get('torch', '?')} (CUDA {st.get('torch_cuda', '?')}) | triton {st.get('triton', '?')}"
            f" | driver {st.get('driver', '?')} | python {st.get('python', '?')}"
            + (f" | FA3 {st['fa3']}" if st.get("fa3") else ""))
    for k, want in WANT_STACK.items():
        if st.get(k) and st.get(k) != want:
            flag.append(f"{k} {st.get(k)} != {want} (the record's stack)")
    drv = str(st.get("driver") or "")
    if drv[:3].isdigit() and int(drv.split(".")[0]) < MIN_DRIVER:
        flag.append(f"driver {drv} < {MIN_DRIVER}: ~35% slower (ANVIL2 README); absolute times and the Hadamard Δstep_sum are skewed")
    if dirty:
        flag.append("working tree has uncommitted changes to tracked files: " + " ".join((st.get("dirty_files") or [])[:6])
                    + " (the A/B arms ran the committed HEAD)")
    for prob in st.get("problems") or []:
        if prob not in " ".join(flag):
            flag.append(prob)
    nc = _read(os.path.join(d, "node_check.txt"))
    slow = [ln.strip() for ln in nc.splitlines() if "SLOW" in ln]
    flag += [f"node_check: {ln}" for ln in slow]
    if st.get("allow_stack"):
        flag.append("ALLOW_STACK=1 was set: the stack check was overridden")
    return [line] + [f"  !! {f}" for f in flag]


def phase_tail_lines(root: str, phase: str, files: list[str], status: str) -> list[str]:
    if not status.startswith("FAIL"):
        return []
    for f in files:
        txt = _read(os.path.join(root, phase, f))
        if txt.strip():
            tail = [ln for ln in txt.splitlines() if ln.strip()][-4:]
            return [f"== {phase}: {status}; tail of {phase}/{f}:"] + ["  " + ln[:200] for ln in tail]
    return []


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

def status_map(root: str) -> dict:
    d = os.path.join(root, "status")
    if not os.path.isdir(d):
        return {}
    order = ["env", "tests", "sanity", "gate", "hadamard", "t2", "bundle"]
    names = sorted(os.listdir(d), key=lambda n: (order.index(n) if n in order else 99, n))
    return {n: _read(os.path.join(d, n)).strip() for n in names}


def statuses(root: str) -> list[str]:
    return [f"{n}: {v}" for n, v in status_map(root).items()]


def gate_gap(root: str) -> float | None:
    ab = load_ab(os.path.join(root, "gate"))
    if ab is None:
        return None
    g = gate_decision(ab)
    return g["mean_mnat"] if g.get("n", 0) >= 2 else None


def hadamard_lines(root: str, name: str = "hadamard") -> list[str]:
    ab = load_ab(os.path.join(root, name))
    return [] if ab is None else ab_table(name, ab, buyback_mnat=gate_gap(root))


def summarize(root: str) -> str:
    lines = []
    meta = os.path.join(root, "meta.txt")
    if os.path.isfile(meta):
        with open(meta) as f:
            lines += [ln.rstrip() for ln in f if ln.strip()]
    lines += stack_lines(root)
    smap = status_map(root)
    if smap:
        lines += ["phases: " + " | ".join(f"{n}: {v}" for n, v in smap.items())]
    lines += phase_tail_lines(root, "env", ["env.txt"], smap.get("env", ""))
    t = tests_lines(root)
    if t:
        lines += [""] + t
    lines += phase_tail_lines(root, "sanity", ["sanity.txt"], smap.get("sanity", ""))
    attempts = sorted(n for n in os.listdir(root) if re.fullmatch(r"(gate|hadamard)\.attempt\d+", n))
    for phase in attempts + ["gate", "hadamard"]:  # earlier attempts (re-planned after failing) first
        if phase.startswith("gate"):
            ab = load_ab(os.path.join(root, phase))
            if ab is not None:
                lines += [""] + ab_table(phase, ab, gate=True) + gate_lines(gate_decision(ab))
        else:
            h = hadamard_lines(root, phase)
            lines += [""] + h if h else []
    t2s = smap.get("t2", "")
    if os.path.isdir(os.path.join(root, "t2")) and not t2s.startswith("SKIP"):
        lines += [""] + t2_lines(os.path.join(root, "t2"))
        lines += phase_tail_lines(root, "t2", ["stdout.txt", "prefetch.txt"], t2s)
    return "\n".join(lines) + "\n"


def open_input(path: str) -> tuple[str, tempfile.TemporaryDirectory | None, str | None]:
    """(session root, tempdir to clean up, warning). A truncated bundle (a cut-off scp) yields what it holds."""
    if os.path.isdir(path):
        return path, None, None
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: no such bundle or directory")
    tmp = tempfile.TemporaryDirectory()
    kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    n, warn = 0, None
    try:
        with tarfile.open(path) as tf:
            for m in tf:
                if (m.isfile() or m.isdir()) and not m.name.startswith("/") and ".." not in m.name.split("/") \
                        and not os.path.basename(m.name).startswith("._"):  # macOS AppleDouble
                    tf.extract(m, tmp.name, **kw)
                    n += 1
    except (EOFError, tarfile.TarError, OSError, zlib.error) as e:
        if n == 0:
            tmp.cleanup()
            raise SystemExit(f"{path}: not a readable .tar.gz ({type(e).__name__}: {e}). It is probably truncated: "
                             "copy it again (compare `sha256sum` on both ends).")
        warn = (f"WARNING: {path} is truncated or damaged after {n} entries ({type(e).__name__}); this summary covers "
                "only the readable part -- copy the bundle again")
    tops = [t for t in os.listdir(tmp.name) if not t.startswith("._")]
    root = os.path.join(tmp.name, tops[0]) if len(tops) == 1 and os.path.isdir(os.path.join(tmp.name, tops[0])) else tmp.name
    return root, tmp, warn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="session bundle .tar.gz or session directory")
    ap.add_argument("--gate", help="an ab_run gate directory: print its table and decision only")
    ap.add_argument("--write-decision", help="with --gate: write GATE_* shell variables to this file")
    ap.add_argument("--hadamard", metavar="SESSION_DIR", help="print the hadamard phase's table against the gate's buy-back bound")
    ap.add_argument("--check", metavar="DIR", help="an ab_run directory: print its state; exit 0 complete and usable, "
                    "3 slots pending, 4 complete but unusable (re-plan), 5 an arm failed every run (systematic: do not re-plan), 2 no plan")
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
        print("\n".join(ab_table("gate", ab, gate=True) + gate_lines(g)))
        if a.write_decision:
            def num(x):
                return "nan" if x is None or x != x else f"{x:.4f}"
            with open(a.write_decision, "w") as f:
                f.write(f"GATE_DECISION={g['decision']}\nGATE_COMPLETE={int(bool(g.get('complete')))}\nGATE_N={g.get('n', 0)}\n"
                        f"GATE_MEAN_MNAT={num(g.get('mean_mnat'))}\n"
                        f"GATE_SE_MNAT={num(g.get('se_mnat'))}\nGATE_LO_MNAT={num(g.get('lo_mnat'))}\n"
                        f"GATE_HI_MNAT={num(g.get('hi_mnat'))}\nGATE_RUN_HADAMARD={int(bool(g.get('run_hadamard')))}\n")
        return 0
    if a.hadamard:
        print("\n".join(hadamard_lines(a.hadamard)))
        return 0
    if not a.input:
        ap.error("give a bundle/directory, or --gate DIR")
    root, tmp, warn = open_input(a.input)
    try:
        sys.stdout.write((warn + "\n" if warn else "") + summarize(root))
    finally:
        if tmp:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
