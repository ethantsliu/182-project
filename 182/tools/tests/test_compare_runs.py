"""compare_runs.py and the root trainer's METRIC lines (track_1_short/run_metrics.py) on CPU: synthetic logs written
with the trainer's own formatters, hand-computed statistics, the legacy-line fallbacks, an ab_run directory as a
source, and that RUN_METRICS / VAL_CURVE are off by default and every use of them in train_gpt.py is guarded."""
import ast
import csv
import io
import json
import math
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from _paths import REPO
import compare_runs as cr
import logparse

if REPO not in sys.path:
    sys.path.insert(0, REPO)
from track_1_short import run_metrics as rm  # noqa: E402  (stdlib only)

TOTAL = 100
SEP = "=" * 100
BOUNDS = [(0, 40), (40, 100)]


def make_log(seed, time_s, val, *, metrics=True, curve=None, step_ms=None, peak=(1000, 1200, 1500, 1600), ranks=2):
    """A run log in the root trainer's format. curve: [(step, train_time_s, val_loss)] intermediate validations."""
    out = ["print('METRIC val step=1 total=1 train_time_s=1 val_loss=9 final=1')  # source dump: never parsed", SEP,
           "Running PyTorch 2.10.0+cu128 compiled for CUDA 12.8", SEP,
           f"attention heads=6 (qk, v) head widths by layer, in bank order={{0: (128, 128)}} seed={seed} steps={TOTAL} "
           f"stage boundaries={BOUNDS}"]
    tag = " INVALID_FOR_RECORD(val_curve)" if curve else ""
    if metrics:
        out.append(rm.run_config_line(True, 25 if curve else 0, None if seed == "random" else seed, ranks, TOTAL))
    points = dict((s, (t, v)) for s, t, v in curve or [])
    for s in range(25, TOTAL + 1, 25):
        if s in points and s < TOTAL:
            t, v = points[s]
            out.append(f"step:{s}/{TOTAL} val_loss:{v:.4f} train_time:{t * 1000:.0f}ms step_avg:{t * 1000 / s:.2f}ms{tag}")
            if metrics:
                out.append(rm.val_line(s, TOTAL, t * 1000, v))
        tt = time_s * s / TOTAL
        out.append(f"step:{s}/{TOTAL} train_time:{tt * 1000:.0f}ms step_avg:{tt * 1000 / s:.2f}ms")
    out.append(f"step:{TOTAL}/{TOTAL} val_loss:{val:.4f} train_time:{time_s * 1000:.0f}ms step_avg:{time_s * 10:.2f}ms{tag}")
    if metrics:
        out.append(rm.val_line(TOTAL, TOTAL, time_s * 1000, val))
        ms = step_ms if step_ms is not None else time_s * 1000 / TOTAL
        clock = [ms * (s + 1) for s in range(TOTAL)]
        out += rm.step_latency_lines(clock, [s for s in points if s < TOTAL] + [TOTAL], BOUNDS)
        out += rm.peak_mem_lines([[x * rm.MIB + r * rm.MIB for x in peak] for r in range(ranks)])
    out.append("peak memory allocated: 777 MiB reserved: 888 MiB")
    return "\n".join(out) + "\n"


class TestRunMetrics(unittest.TestCase):
    def test_step_latency_excludes_step0_and_validated_steps(self):
        clock = [50.0, 60.0, 70.0, 95.0, 105.0, 115.0]   # step 0 holds the table build; step 3 began with a validation
        self.assertEqual(rm.step_latencies(clock, [3, 6]), [(1, 10.0), (2, 10.0), (4, 10.0), (5, 10.0)])
        lines = rm.step_latency_lines(clock, [3, 6], [(0, 2), (2, 6)])
        got = logparse.parse_metrics("\n".join(lines))
        self.assertEqual([d["scope"] for _, d in got], ["all", "stage0", "stage1"])
        self.assertEqual(got[0][1], dict(scope="all", steps="1-5", n="4", mean_ms="10.000", median_ms="10.000",
                                         p95_ms="10.000", max_ms="10.000", clock="host_nosync"))
        self.assertEqual((got[1][1]["steps"], got[1][1]["n"], got[2][1]["steps"], got[2][1]["n"]), ("1-1", "1", "2-5", "3"))
        self.assertEqual(rm.step_latency_lines([5.0], [], BOUNDS), [])

    def test_percentile(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(rm.percentile(xs, 0.5), 3.0)
        self.assertAlmostEqual(rm.percentile(xs, 0.95), 4.8)
        self.assertEqual(rm.percentile([7.0], 0.95), 7.0)

    def test_lines_round_trip(self):
        text = "\n".join([rm.run_config_line(True, 250, None, 8, 1194), rm.val_line(250, 1194, 8432.1, 4.1234567),
                          rm.val_line(1194, 1194, 40020.0, 3.27801), *rm.peak_mem_lines([[3 * rm.MIB, 4 * rm.MIB, 5 * rm.MIB, 6 * rm.MIB],
                                                                                       [7 * rm.MIB, 2 * rm.MIB, 1 * rm.MIB, 9 * rm.MIB]])])
        m = logparse.parse_metrics(text)
        self.assertEqual(m[0], ("run_config", dict(timing="curve_run", RUN_METRICS="1", VAL_CURVE="250", seed="random",
                                                   world_size="8", total_steps="1194")))
        self.assertEqual(m[1][1], dict(step="250", total="1194", train_time_s="8.4321", val_loss="4.123457", final="0"))
        self.assertEqual(m[2][1]["final"], "1")
        self.assertEqual([d["rank"] for n, d in m if n == "peak_mem"], ["0", "1", "max"])
        self.assertEqual(m[-1][1], dict(rank="max", allocated_mib="7", reserved_mib="4", warmup_allocated_mib="5",
                                        warmup_reserved_mib="9"))
        self.assertIn("timing=timing_run", rm.run_config_line(True, 0, 3, 8, 10))


class TestParseRun(unittest.TestCase):
    def test_metric_log(self):
        r = cr.parse_run(make_log(3, 40.0, 3.2781234, step_ms=390.0), "x.txt")
        self.assertEqual((r["seed"], r["key"], r["kind"], r["completed"], r["steps"]), ("3", "seed3", "timing", True, TOTAL))
        self.assertAlmostEqual(r["val_loss"], 3.278123, places=6)       # the 6-digit METRIC value, not the val line's 3.2781
        self.assertAlmostEqual(r["train_time_s"], 40.0)
        self.assertEqual(r["lat"], dict(mean_ms=390.0, median_ms=390.0, p95_ms=390.0, src="steps"))
        self.assertEqual(sorted(r["stage_lat"]), ["stage0", "stage1"])
        self.assertEqual(r["mem"], dict(allocated_mib=1001.0, reserved_mib=1201.0, warmup_allocated_mib=1501.0, src="all-ranks"))
        self.assertEqual(len(r["curve"]), 1)   # the METRIC line in the source dump is not read

    def test_legacy_log_falls_back(self):
        r = cr.parse_run(make_log("random", 40.0, 3.2781234, metrics=False))
        self.assertEqual((r["seed"], r["key"], r["kind"]), (None, None, "timing"))
        self.assertAlmostEqual(r["val_loss"], 3.2781)
        self.assertEqual(r["lat"]["src"], "windows")
        self.assertAlmostEqual(r["lat"]["mean_ms"], 400.0)
        self.assertEqual(r["mem"], dict(allocated_mib=777.0, reserved_mib=None, warmup_allocated_mib=None, src="rank0"))

    def test_curve_run_detected(self):
        curve = [(25, 10.5, 5.0), (50, 21.0, 4.0), (75, 31.5, 3.5)]
        for metrics in (True, False):
            r = cr.parse_run(make_log(1, 42.0, 3.27, curve=curve, metrics=metrics))
            self.assertEqual(r["kind"], "curve")
            self.assertEqual([c[0] for c in r["curve"]], [25, 50, 75, 100])
            self.assertAlmostEqual(r["curve"][1][1], 21.0)

    def test_incomplete(self):
        text = make_log(1, 40.0, 3.27).split(f"step:{TOTAL}/{TOTAL} val_loss")[0]
        r = cr.parse_run(text)
        self.assertFalse(r["completed"])
        self.assertIsNone(r["train_time_s"])


class TestCrossing(unittest.TestCase):
    def test_interpolated(self):
        c = cr.crossing([(25, 10.0, 5.0), (50, 20.0, 3.30), (75, 30.0, 3.26), (100, 40.0, 3.20)], 3.28)
        self.assertAlmostEqual(c["time_s"], 25.0)
        self.assertAlmostEqual(c["step"], 62.5)
        self.assertEqual((c["last_interval"], c["censored"]), (False, False))

    def test_edges(self):
        self.assertIsNone(cr.crossing([(25, 10.0, 5.0), (100, 40.0, 3.29)], 3.28))
        self.assertTrue(cr.crossing([(25, 10.0, 3.0), (100, 40.0, 2.9)], 3.28)["censored"])
        c = cr.crossing([(75, 30.0, 3.38), (100, 40.0, 3.28)], 3.28)   # reached exactly at the final validation
        self.assertAlmostEqual(c["time_s"], 40.0)
        self.assertTrue(c["last_interval"])


class TestCompare(unittest.TestCase):
    A_TIMES = {1: 40.0, 2: 40.4, 3: 39.8, 4: 40.2}
    B_DT = {1: -0.5, 2: -0.7, 3: -0.4, 4: -0.8}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for arm in ("a", "b"):
            (self.root / arm).mkdir()
        for seed, t in self.A_TIMES.items():
            (self.root / "a" / f"t{seed}.txt").write_text(make_log(seed, t, 3.2780 + seed * 1e-4))
            (self.root / "b" / f"t{seed}.txt").write_text(
                make_log(seed, t + self.B_DT[seed], 3.2790 + seed * 1e-4, peak=(900, 1200, 1500, 1600)))
        # Curve runs: A crosses 3.28 between steps 50 and 75, B only in the last interval.
        (self.root / "a" / "c1.txt").write_text(
            make_log(1, 42.0, 3.27, curve=[(25, 10.5, 5.0), (50, 21.0, 3.30), (75, 31.5, 3.26)]))
        (self.root / "b" / "c1.txt").write_text(
            make_log(1, 41.0, 3.27, curve=[(25, 10.25, 5.0), (50, 20.5, 3.40), (75, 30.75, 3.29)]))

    def _compare(self):
        return cr.compare(cr.load_source(str(self.root / "a")), cr.load_source(str(self.root / "b")))

    def test_arm_statistics(self):
        C = self._compare()
        A, B = C["a"], C["b"]
        self.assertEqual((A["n_timing"], A["n_curve"], A["n_incomplete"], A["hits"], B["hits"]), (4, 1, 0, 4, 4))
        ts = list(self.A_TIMES.values())
        mean = sum(ts) / 4
        sd = math.sqrt(sum((t - mean) ** 2 for t in ts) / 3)
        self.assertAlmostEqual(A["time"]["mean"], mean)
        self.assertAlmostEqual(A["time"]["sd"], sd)
        half = 3.182446305 * sd / 2          # t(0.975, df=3)
        self.assertAlmostEqual(A["time"]["lo"], mean - half, places=5)
        self.assertAlmostEqual(A["time"]["hi"], mean + half, places=5)
        self.assertAlmostEqual(A["val"]["mean"], 3.27825, places=6)
        self.assertAlmostEqual(A["lat"]["mean_ms"]["mean"], mean * 10)
        self.assertEqual(A["lat_src"], ["steps"])
        self.assertAlmostEqual(A["mem"]["allocated_mib"]["mean"], 1001.0)
        self.assertAlmostEqual(B["mem"]["allocated_mib"]["mean"], 901.0)
        self.assertAlmostEqual(A["curve_time"]["mean"], 42.0)   # the curve run is not in the timing rows

    def test_paired_difference(self):
        C = self._compare()
        d = list(self.B_DT.values())
        mean = sum(d) / 4
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / 3)
        half = 3.182446305 * sd / 2
        self.assertEqual((C["d_time"]["kind"], C["d_time"]["n"], C["d_time"]["unpaired"]), ("paired", 4, 0))
        self.assertAlmostEqual(C["d_time"]["mean"], mean)
        self.assertAlmostEqual(C["d_time"]["lo"], mean - half, places=5)
        self.assertAlmostEqual(C["d_time"]["hi"], mean + half, places=5)
        self.assertAlmostEqual(C["d_val"]["mean"], 0.001, places=6)
        self.assertAlmostEqual(C["d_mem"]["mean"], -100.0)

    def test_time_to_target(self):
        C = self._compare()
        A, B = C["a"], C["b"]
        # A: 3.30 at 21.0 s / step 50, 3.26 at 31.5 s / step 75 -> halfway.
        self.assertAlmostEqual(A["cross_time"]["mean"], 26.25)
        self.assertAlmostEqual(A["cross_step"]["mean"], 62.5)
        self.assertEqual((A["cross_n"], A["cross_last"]), (1, 0))
        self.assertAlmostEqual(A["cross_rescaled"]["mean"], 26.25 * A["time"]["mean"] / 42.0)
        # B: 3.29 at 30.75 s, 3.27 at 41.0 s -> halfway through the last interval.
        self.assertAlmostEqual(B["cross_time"]["mean"], 35.875)
        self.assertEqual((B["cross_n"], B["cross_last"]), (1, 1))

    def test_welch_when_unseeded(self):
        a = [cr.parse_run(make_log("random", t, 3.278)) for t in (40.0, 40.4, 39.8)]
        b = [cr.parse_run(make_log("random", t, 3.278)) for t in (39.0, 39.4, 38.8)]
        d = cr.compare(a, b)["d_time"]
        self.assertEqual(d["kind"], "welch")
        self.assertAlmostEqual(d["mean"], -1.0)
        self.assertTrue(d["lo"] < -1.0 < d["hi"] < 0)

    def test_cli_table_and_csv(self):
        out_csv, out_json = self.root / "curves.csv", self.root / "c.json"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cr.main(["--a", str(self.root / "a"), "--b", str(self.root / "b" / "*.txt"), "--names", "base,kernel",
                          "--csv", str(out_csv), "--json", str(out_json)])
        text = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("A: base", text)
        self.assertIn("3.27825 ± 0.00013 [3.27804, 3.27846] (n=4)", text)      # losses in full
        self.assertNotIn("mnat", text)
        self.assertIn("-0.600 [", text)
        self.assertIn("(paired, n=4)", text)
        self.assertIn("time to 3.28, curve clock (s)", text)
        self.assertIn("crossing in the last interval", text)
        rows = list(csv.DictReader(open(out_csv)))
        self.assertEqual(len(rows), 2 * (4 + 4))    # per arm: 4 timing finals + a 4-point curve
        curve = [r for r in rows if r["arm"] == "kernel" and r["kind"] == "curve"]
        self.assertEqual([(r["step"], r["train_time_s"], r["val_loss"]) for r in curve][:2],
                         [("25", "10.2500", "5.000000"), ("50", "20.5000", "3.400000")])
        self.assertEqual(json.load(open(out_json))["d_time"]["kind"], "paired")

    def test_ab_run_directory_source(self):
        ab = self.root / "ab"
        (ab / "logs").mkdir(parents=True)
        runs, ledger = [], []
        for rep, seed in enumerate((11, 12)):
            for arm, t in (("base", 40.0 + rep), ("kern", 39.0 + rep)):
                idx = len(runs)
                log = ab / "logs" / f"{idx:03d}_{arm}_x.txt"
                log.write_text(make_log("random", t, 3.278))        # the key comes from the ledger's seed
                runs.append(dict(idx=idx, rep=rep, arm=arm, seed=seed, kind="measure"))
                ledger.append(dict(event="end", idx=idx, rep=rep, arm=arm, seed=seed, kind="measure", status="ok",
                                   log=f"/gone/on/the/node/{log.name}", val_loss=3.278, train_time_s=t))
        ledger[-1]["status"] = "failed"
        (ab / "plan.json").write_text(json.dumps(dict(arms=[dict(name="base", env={}), dict(name="kern", env={})],
                                                      baseline="base", n=2, runs=runs)))
        (ab / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
        a, b = cr.load_source(f"{ab}@base"), cr.load_source(f"{ab}@kern")
        self.assertEqual(([r["key"] for r in a], [r["key"] for r in b]), (["seed11", "seed12"], ["seed11"]))
        d = cr.compare(a, b)["d_time"]
        self.assertEqual((d["kind"], d["n"], d["unpaired"]), ("paired", 1, 1))
        self.assertAlmostEqual(d["mean"], -1.0)
        with self.assertRaises(SystemExit):
            cr.load_source(f"{ab}@nope")


class TestFlagsOff(unittest.TestCase):
    GUARDS = {"RUN_METRICS", "VAL_CURVE", "step_clock"}
    GUARDED = {"val_line", "run_config_line", "step_latency_lines", "peak_mem_lines", "reset_peak_memory_stats",
               "warmup_peak", "timed_peak", "peaks"}

    def _config(self, **env):
        clean = {k: v for k, v in os.environ.items() if k not in ("RUN_METRICS", "VAL_CURVE")}
        code = "from track_1_short import config as c; print(int(c.RUN_METRICS), c.VAL_CURVE, c.Hyperparameters().val_loss_every)"
        return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=dict(clean, **env), capture_output=True, text=True)

    def test_defaults_are_off(self):
        r = self._config()
        self.assertEqual((r.returncode, r.stdout.split()), (0, ["0", "0", "0"]), r.stderr)
        self.assertEqual(self._config(RUN_METRICS="1", VAL_CURVE="250").stdout.split(), ["1", "250", "0"])
        self.assertNotEqual(self._config(RUN_METRICS="yes").returncode, 0)
        self.assertNotEqual(self._config(VAL_CURVE="-1").returncode, 0)

    def test_every_metric_statement_in_the_trainer_is_guarded(self):
        """Flags off: nothing of run_metrics runs. Each statement of train_gpt.py that touches a metric name sits under
        an `if` (or is a conditional expression) that tests a flag; the imports and two inert initializers aside."""
        tree = ast.parse((Path(REPO) / "train_gpt.py").read_text())
        unguarded = []

        def names(node):
            return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}

        def walk(stmts, guarded):
            for stmt in stmts:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    continue
                if isinstance(stmt, ast.If):
                    walk(stmt.body, guarded or bool(names(stmt.test) & self.GUARDS))
                    walk(stmt.orelse, guarded)
                elif hasattr(stmt, "body"):   # def, for, while, with, try
                    for part in ("body", "orelse", "finalbody"):
                        walk(getattr(stmt, part, []), guarded)
                    for handler in getattr(stmt, "handlers", []):
                        walk(handler.body, guarded)
                elif not guarded and names(stmt) & self.GUARDED:
                    unguarded.append(ast.unparse(stmt)[:80])

        walk(tree.body, False)
        self.assertEqual(unguarded, [])
        src = (Path(REPO) / "train_gpt.py").read_text()
        self.assertIn("step_clock = [0.0] * training_schedule.total_steps if RUN_METRICS else None", src)
        self.assertIn("(VAL_CURVE > 0 and step > 0 and step % VAL_CURVE == 0)", src)
        self.assertIn('(" INVALID_FOR_RECORD(val_curve)" if VAL_CURVE else "")', src)


if __name__ == "__main__":
    unittest.main()
