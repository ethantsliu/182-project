#!/usr/bin/env python3
"""Speedrun statistics, dependency-free (stdlib math only; scipy is used by the tests to cross-check).

Rules implemented (README.md "Rules" and records/track_3_optimization/README.md "Rules"):

  gate     Track 1 / Track 2: one-sided one-sample t-test that the mean val loss is <= target,
           p < 0.01 required. H0: mu >= target; t = (target - mean) / (sd / sqrt(n)), df = n - 1.
  welch    Welch two-sample t-test (unequal variances) for a train-time (or val) delta vs a baseline,
           e.g. "rule 4: faster than the prior record on the same hardware".
  z3       Track 3 single-result rule: (target - mean) * sqrt(n) >= 0.004
           (a one-sided z-test with sigma = 0.0013 at p < .001: 3.09 * 0.0013 ~= 0.004).
  pair3    Track 3 pairwise rule: (final_loss_diff + exp_stepbased_loss_diff) / sqrt(1/n1 + 1/n2) >= 0.004,
           exp_stepbased_loss_diff = (steps_old - steps_new) / 100 * 0.0045.
  n-gate / n-welch / n-z3   required number of runs from sigma and margin, at a given power.

Values come from run logs (parsed with logparse.py: files, dirs or globs) and/or --values.

Examples:
  python 182/tools/stats.py gate records/track_1_short/2026-08-30_ANVIL2/this_pr --extra 3.2775
  python 182/tools/stats.py welch --metric time --a runs/new/*.txt --b runs/base/*.txt
  python 182/tools/stats.py n-gate --sigma 0.0013 --margin 0.0015
  python 182/tools/stats.py n-welch --sigma 0.12 --delta 0.15
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TARGET = 3.28
T3_THRESHOLD = 0.004
T3_SIGMA = 0.0013
T3_LOSS_PER_100_STEPS = 0.0045


# ----------------------------------------------------------------------------- distributions

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz; Numerical Recipes 6.4)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom (df may be fractional, as in Welch)."""
    if math.isinf(df):
        return norm_sf(t)
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    # tail = I_x(df/2, 1/2) / 2 with x = df/(df+t^2); for small |t|, x ~ 1 and the complement
    # 1 - I_{1-x}(1/2, df/2) with 1-x = t^2/(df+t^2) computed directly keeps full precision.
    x, xc = df / (df + t * t), t * t / (df + t * t)
    tail = 0.5 * (betainc(df / 2.0, 0.5, x) if x < 0.5 else 1.0 - betainc(0.5, df / 2.0, xc))
    return tail if t > 0 else 1.0 - tail


def t_cdf(t: float, df: float) -> float:
    return t_sf(-t, df)


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_cdf(z: float) -> float:
    return norm_sf(-z)


def _bisect(f, lo: float, hi: float, iters: int = 200) -> float:
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def norm_ppf(p: float) -> float:
    return _bisect(lambda z: norm_cdf(z) - p, -40.0, 40.0)


def t_ppf(p: float, df: float) -> float:
    return _bisect(lambda t: t_cdf(t, df) - p, -1e4, 1e4)


# ----------------------------------------------------------------------------- tests

def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def gate_test(values, target: float = TARGET, alpha: float = 0.01) -> dict:
    """One-sided t-test of H1: mean <= target. p small = evidence the mean is below the target."""
    vals = list(values)
    n = len(vals)
    m, sd = mean_sd(vals)
    out = dict(test="gate", n=n, mean=m, sd=sd, target=target, alpha=alpha)
    if n < 2 or not sd > 0:
        out.update(t=float("nan"), df=n - 1, p=float("nan"), passes=False,
                   note="need n >= 2 with nonzero spread")
        return out
    se = sd / math.sqrt(n)
    t = (target - m) / se
    p = t_sf(t, n - 1)
    out.update(t=t, df=n - 1, se=se, p=p, passes=p < alpha, margin=target - m)
    return out


def welch_test(a, b, alternative: str = "less", alpha: float = 0.01) -> dict:
    """Welch t-test on delta = mean(a) - mean(b).

    alternative="less": H1 mean(a) < mean(b) (a is faster / lower), "greater", or "two-sided".
    """
    a, b = list(a), list(b)
    na, nb = len(a), len(b)
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    out = dict(test="welch", alternative=alternative, n_a=na, n_b=nb, mean_a=ma, sd_a=sa, mean_b=mb, sd_b=sb,
               delta=ma - mb, alpha=alpha)
    if na < 2 or nb < 2:
        out.update(t=float("nan"), df=float("nan"), p=float("nan"), passes=False, note="need n >= 2 per arm")
        return out
    va, vb = sa * sa / na, sb * sb / nb
    se = math.sqrt(va + vb)
    if se == 0:
        out.update(t=float("nan"), df=float("nan"), p=float("nan"), passes=False, note="zero variance")
        return out
    t = (ma - mb) / se
    df = (va + vb) ** 2 / (va * va / (na - 1) + vb * vb / (nb - 1))
    if alternative == "less":
        p = t_cdf(t, df)
    elif alternative == "greater":
        p = t_sf(t, df)
    else:
        p = 2 * t_sf(abs(t), df)
    q = t_ppf(0.975, df)
    out.update(t=t, df=df, se=se, p=p, passes=p < alpha, ci95=(ma - mb - q * se, ma - mb + q * se),
               rel_delta=(ma - mb) / mb if mb else float("nan"))
    return out


def z3_rule(values, target: float = TARGET, threshold: float = T3_THRESHOLD) -> dict:
    vals = list(values)
    n = len(vals)
    m, sd = mean_sd(vals)
    score = (target - m) * math.sqrt(n) if n else float("nan")
    return dict(test="z3", n=n, mean=m, sd=sd, target=target, score=score, threshold=threshold,
                passes=bool(n) and all(math.isfinite(v) for v in vals) and score >= threshold,
                max_mean_to_pass=target - threshold / math.sqrt(n) if n else float("nan"),
                z_equiv=score / T3_SIGMA if n else float("nan"))


def pair3_rule(mean_new: float, n_new: int, steps_new: int, mean_old: float, n_old: int, steps_old: int,
               loss_per_100_steps: float = T3_LOSS_PER_100_STEPS, threshold: float = T3_THRESHOLD) -> dict:
    final_diff = mean_old - mean_new
    step_diff = (steps_old - steps_new) / 100.0 * loss_per_100_steps
    lhs = (final_diff + step_diff) / math.sqrt(1.0 / n_new + 1.0 / n_old)
    return dict(test="pair3", final_loss_diff=final_diff, exp_stepbased_loss_diff=step_diff, lhs=lhs,
                threshold=threshold, passes=lhs >= threshold)


# ----------------------------------------------------------------------------- sample size

def required_n_gate(sigma: float, margin: float, alpha: float = 0.01, power: float = 0.8, n_max: int = 100000) -> dict:
    """Smallest n for the one-sided t gate to pass with probability ~power when the true mean sits
    `margin` below the target and the run-to-run sd is sigma.

    Uses the standard approximation n >= ((t_{1-alpha,n-1} + t_{power,n-1}) * sigma / margin)^2, solved
    by iteration (it slightly understates n for very small n; the tests check it against a simulation).
    """
    if margin <= 0:
        return dict(n=None, note="margin must be > 0 (true mean below target)")
    for n in range(2, n_max):
        df = n - 1
        need = ((t_ppf(1 - alpha, df) + t_ppf(power, df)) * sigma / margin) ** 2
        if n >= need:
            return dict(n=n, sigma=sigma, margin=margin, alpha=alpha, power=power)
    return dict(n=None, note=f"> {n_max}")


def required_n_welch(sigma: float, delta: float, alpha: float = 0.01, power: float = 0.8,
                     sigma_b: float | None = None, n_max: int = 100000) -> dict:
    """Per-arm n (equal arms) for a one-sided Welch test to detect a true mean difference `delta`."""
    sb = sigma if sigma_b is None else sigma_b
    if delta <= 0:
        return dict(n=None, note="delta must be > 0")
    for n in range(2, n_max):
        va, vb = sigma * sigma / n, sb * sb / n
        df = (va + vb) ** 2 / (va * va / (n - 1) + vb * vb / (n - 1))
        need = (t_ppf(1 - alpha, df) + t_ppf(power, df)) ** 2 * (sigma * sigma + sb * sb) / delta ** 2
        if n >= need:
            return dict(n_per_arm=n, n=n, sigma_a=sigma, sigma_b=sb, delta=delta, alpha=alpha, power=power)
    return dict(n=None, note=f"> {n_max}")


def required_n_z3(expected_mean: float, sigma: float = T3_SIGMA, target: float = TARGET,
                  threshold: float = T3_THRESHOLD, power: float = 0.8) -> dict:
    """Track 3: smallest n with P((target - mean_n) * sqrt(n) >= threshold) >= power when single runs
    are N(expected_mean, sigma^2). P = Phi(((target - mu) * sqrt(n) - threshold) / sigma)."""
    margin = target - expected_mean
    if margin <= 0:
        return dict(n=None, note="expected mean must be below target")
    n = math.ceil(((threshold + norm_ppf(power) * sigma) / margin) ** 2)
    n = max(n, 1)
    p_pass = norm_cdf((margin * math.sqrt(n) - threshold) / sigma)
    return dict(n=n, expected_mean=expected_mean, sigma=sigma, power=power, p_pass_at_n=p_pass)


# ----------------------------------------------------------------------------- CLI

def _collect(patterns, extra, metric: str, at_step: int | None, allow_incomplete: bool):
    from logparse import expand, parse_file_runs

    vals, notes = [], []
    for p, r in ((p, r) for p in expand(patterns or []) for r in parse_file_runs(p)):
        if r.val_loss is None:
            notes.append(f"no val line: {p}")
            continue
        if not r.completed and not allow_incomplete and at_step is None:
            notes.append(f"incomplete run (last val step {r.final_step}/{r.total_steps}): {p}")
            continue
        if metric == "time":
            if at_step is not None:
                t = next((t for s, _, t in r.val_curve if s == at_step), None)
                if t is None:
                    notes.append(f"no val at step {at_step}: {p}")
                    continue
                vals.append(t)
            else:
                vals.append(r.train_time_s)
        else:
            v = r.val_at(at_step) if at_step is not None else r.val_loss
            if v is None:
                notes.append(f"no val at step {at_step}: {p}")
                continue
            vals.append(v)
    vals += [float(x) for x in (extra or [])]
    return vals, notes


def _fmt(d: dict) -> str:
    def f(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        if isinstance(v, tuple):
            return "(" + ", ".join(f(x) for x in v) + ")"
        return str(v)
    return "  ".join(f"{k}={f(v)}" for k, v in d.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print JSON instead of key=value")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_src(p, name="logs", flag=None):
        if flag:
            p.add_argument(flag, nargs="*", default=[], help="run logs: files, dirs or globs")
        else:
            p.add_argument(name, nargs="*", help="run logs: files, dirs or globs")

    g = sub.add_parser("gate", help="one-sided t-test mean val <= target")
    add_src(g)
    g.add_argument("--extra", "--values", dest="extra", nargs="*", type=float, help="additional values")
    g.add_argument("--target", type=float, default=TARGET)
    g.add_argument("--alpha", type=float, default=0.01)
    g.add_argument("--at-step", type=int, help="use the val printed at this step instead of the final one")
    g.add_argument("--allow-incomplete", action="store_true")

    w = sub.add_parser("welch", help="Welch test of mean(a) - mean(b)")
    add_src(w, flag="--a")
    add_src(w, flag="--b")
    w.add_argument("--a-values", nargs="*", type=float)
    w.add_argument("--b-values", nargs="*", type=float)
    w.add_argument("--metric", choices=["time", "val"], default="time")
    w.add_argument("--alternative", choices=["less", "greater", "two-sided"], default="less")
    w.add_argument("--alpha", type=float, default=0.01)
    w.add_argument("--at-step", type=int)

    z = sub.add_parser("z3", help="Track 3 rule (target - mean) * sqrt(n) >= 0.004")
    add_src(z)
    z.add_argument("--extra", "--values", dest="extra", nargs="*", type=float)
    z.add_argument("--target", type=float, default=TARGET)
    z.add_argument("--at-step", type=int, help="fixed readout step (same for every run)")

    p3 = sub.add_parser("pair3", help="Track 3 pairwise rule vs a previous result")
    for k in ["mean-new", "mean-old"]:
        p3.add_argument(f"--{k}", type=float, required=True)
    for k in ["n-new", "steps-new", "n-old", "steps-old"]:
        p3.add_argument(f"--{k}", type=int, required=True)
    p3.add_argument("--loss-per-100-steps", type=float, default=T3_LOSS_PER_100_STEPS)

    ng = sub.add_parser("n-gate", help="required n for the t gate")
    ng.add_argument("--sigma", type=float, required=True)
    ng.add_argument("--margin", type=float, help="target - expected mean")
    ng.add_argument("--expected-mean", type=float)
    ng.add_argument("--target", type=float, default=TARGET)
    ng.add_argument("--alpha", type=float, default=0.01)
    ng.add_argument("--power", type=float, default=0.8)

    nw = sub.add_parser("n-welch", help="required n per arm for a Welch time test")
    nw.add_argument("--sigma", type=float, required=True)
    nw.add_argument("--sigma-b", type=float)
    nw.add_argument("--delta", type=float, required=True, help="true difference to detect (same units)")
    nw.add_argument("--alpha", type=float, default=0.01)
    nw.add_argument("--power", type=float, default=0.8)

    nz = sub.add_parser("n-z3", help="required n for the Track 3 rule")
    nz.add_argument("--expected-mean", type=float, required=True)
    nz.add_argument("--sigma", type=float, default=T3_SIGMA)
    nz.add_argument("--power", type=float, default=0.8)

    a = ap.parse_args(argv)
    notes = []
    if a.cmd == "gate":
        vals, notes = _collect(a.logs, a.extra, "val", a.at_step, a.allow_incomplete)
        res = gate_test(vals, a.target, a.alpha)
    elif a.cmd == "welch":
        va, na = _collect(a.a, a.a_values, a.metric, a.at_step, False)
        vb, nb = _collect(a.b, a.b_values, a.metric, a.at_step, False)
        notes = na + nb
        res = welch_test(va, vb, a.alternative, a.alpha)
        res["metric"] = a.metric
    elif a.cmd == "z3":
        vals, notes = _collect(a.logs, a.extra, "val", a.at_step, a.at_step is not None)
        res = z3_rule(vals, a.target)
    elif a.cmd == "pair3":
        res = pair3_rule(a.mean_new, a.n_new, a.steps_new, a.mean_old, a.n_old, a.steps_old, a.loss_per_100_steps)
    elif a.cmd == "n-gate":
        margin = a.margin if a.margin is not None else (a.target - a.expected_mean if a.expected_mean is not None else None)
        if margin is None:
            ap.error("n-gate needs --margin or --expected-mean")
        res = required_n_gate(a.sigma, margin, a.alpha, a.power)
    elif a.cmd == "n-welch":
        res = required_n_welch(a.sigma, a.delta, a.alpha, a.power, a.sigma_b)
    else:
        res = required_n_z3(a.expected_mean, a.sigma, power=a.power)
    for n in notes:
        print(f"# skipped {n}", file=sys.stderr)
    print(json.dumps(res, default=str) if a.json else _fmt(res))
    return res


if __name__ == "__main__":
    main()
