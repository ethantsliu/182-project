"""End-to-end test of ab_run.py on CPU: a throwaway git repo whose train_gpt.py is fake_trainer.py,
two branches (the second commits arm_offset.txt, shifting its val), git-ref arms -> pinned worktrees."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from _paths import ANVIL2, TESTS
import ab_run

PY = sys.executable


def sh(cwd, *cmd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


class TestAbRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="abrun_")
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(os.path.join(self.repo, "data", "fineweb10B"))
        shutil.copy(os.path.join(TESTS, "fake_trainer.py"), os.path.join(self.repo, "train_gpt.py"))
        with open(os.path.join(self.repo, ".gitignore"), "w") as f:
            f.write("logs/\ndata/\n")
        sh(self.repo, "git", "init", "-q", "-b", "master")
        sh(self.repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
        sh(self.repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
        sh(self.repo, "git", "checkout", "-qb", "feature")
        with open(os.path.join(self.repo, "arm_offset.txt"), "w") as f:
            f.write("-0.002\n")
        sh(self.repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
        sh(self.repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "feature")
        sh(self.repo, "git", "checkout", "-q", "master")
        self.out = os.path.join(self.tmp, "ab")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def quiet(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            ab_run.main(argv)
        return buf.getvalue()

    def plan(self, *extra, n=4):
        return self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "base=ref:master",
                           "--arm", "new=ref:feature", "--env", "new:FAKE_TIME=39.0", "--cmd", f"{PY} {{script}}",
                           "--n", str(n), "--warmup", "1", "--seeds", "auto", "--order-seed", "3", *extra])

    def ledger_ends(self):
        return [r for r in ab_run.read_ledger(self.out) if r["event"] == "end"]

    def test_plan_run_report(self):
        self.plan("--cache", "cold")
        plan = ab_run.load_plan(self.out)
        self.assertEqual(len(plan["runs"]), 2 + 2 * 4)
        measured = [r for r in plan["runs"] if r["kind"] == "measure"]
        for rep in range(4):  # blocked: each rep has both arms, with the same pre-registered seed
            arms = [r for r in measured if r["rep"] == rep]
            self.assertEqual(sorted(r["arm"] for r in arms), ["base", "new"])
            self.assertEqual(len({r["seed"] for r in arms}), 1)
        # resume: run 3 slots, then the rest
        self.quiet(["run", self.out, "--limit", "3", "--no-report"])
        self.assertEqual(len(self.ledger_ends()), 3)
        self.quiet(["run", self.out, "--no-report"])
        ends = self.ledger_ends()
        self.assertEqual(sorted(r["idx"] for r in ends), list(range(10)))
        self.assertTrue(all(r["status"] == "ok" for r in ends))
        # pinned worktrees at the planned commits, with the data dir linked in
        for arm in plan["arms"]:
            wt = os.path.join(plan["work_root"], "wt", f"{arm['name']}-{arm['sha'][:12]}")
            self.assertEqual(ab_run.git(wt, "rev-parse", "HEAD"), arm["sha"])
            self.assertTrue(os.path.islink(os.path.join(wt, "data", "fineweb10B")))
        # cold cache: a distinct inductor dir per run, removed afterwards
        caches = [r["env"]["TORCHINDUCTOR_CACHE_DIR"] for r in ends]
        self.assertEqual(len(set(caches)), len(caches))
        self.assertFalse(any(os.path.exists(c) for c in caches))
        # seeds reach the trainer: same seed -> same noise draw, so paired val diffs are exactly the offset
        by = {(r["rep"], r["arm"]): r for r in ends if r["kind"] == "measure"}
        for rep in range(4):
            self.assertEqual(by[(rep, "base")]["env"]["TRAIN_SEED"], by[(rep, "new")]["env"]["TRAIN_SEED"])
            self.assertAlmostEqual(by[(rep, "new")]["val_loss"] - by[(rep, "base")]["val_loss"], -0.002, places=4)
            self.assertAlmostEqual(by[(rep, "new")]["train_time_s"] - by[(rep, "base")]["train_time_s"], -1.0, places=2)
        text = self.quiet(["report", self.out])
        self.assertIn("| base | 4/4 |", text)
        self.assertIn("| new | 4/4 |", text)
        self.assertIn("-1.000 s", text)
        with open(os.path.join(self.out, "runs.csv")) as f:
            self.assertEqual(len(f.read().strip().splitlines()), 1 + 10)

    def test_failures_are_counted_not_dropped(self):
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "base=ref:master", "--arm", "crash=ref:master",
                    "--arm", "nan=ref:feature", "--env", "crash:FAKE_FAIL=1", "--env", "nan:FAKE_NAN=1",
                    "--cmd", f"{PY} {{script}}", "--n", "2", "--cache", "warm", "--order", "alternate"])
        self.quiet(["run", self.out, "--no-report"])
        st = {(r["arm"], r["rep"]): r["status"] for r in self.ledger_ends()}
        self.assertEqual(st[("crash", 0)], "failed")
        self.assertEqual(st[("nan", 1)], "nan")
        self.assertEqual(st[("base", 0)], "ok")
        text = self.quiet(["report", self.out])
        self.assertIn("| crash | 0/2 |", text)
        self.assertIn("**crash: 2 non-ok run(s)**", text)
        self.assertIn("**nan: 2 non-ok run(s)**", text)
        # warm cache: one dir per arm, kept
        caches = {r["arm"]: r["env"]["TORCHINDUCTOR_CACHE_DIR"] for r in self.ledger_ends()}
        self.assertEqual(len(set(caches.values())), 3)

    def test_interrupted_slot_is_logged_and_rerun(self):
        self.plan(n=1)
        ab_run.append_ledger(self.out, dict(event="start", idx=0, arm="base", time=0))
        self.quiet(["run", self.out, "--no-report"])
        led = ab_run.read_ledger(self.out)
        self.assertEqual([r["idx"] for r in led if r["event"] == "interrupted"], [0])
        self.assertEqual(sorted(r["idx"] for r in led if r["event"] == "end"), [0, 1, 2, 3])
        self.assertIn("1 harness interruption(s) re-run", self.quiet(["report", self.out]))

    def test_plan_is_tamper_evident_and_never_overwritten(self):
        self.plan(n=1)
        with self.assertRaises(SystemExit):
            self.plan(n=1)
        p = os.path.join(self.out, "plan.json")
        with open(p) as f:
            plan = json.load(f)
        plan["n"] = 2
        with open(p, "w") as f:
            json.dump(plan, f)
        with self.assertRaises(SystemExit):
            ab_run.load_plan(self.out)

    def test_path_arm_change_detected(self):
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", f"here=path:{self.repo}",
                    "--cmd", f"{PY} {{script}}", "--n", "1"])
        with open(os.path.join(self.repo, "train_gpt.py"), "a") as f:
            f.write("\n# edited after the plan\n")
        with self.assertRaises(SystemExit):
            self.quiet(["run", self.out, "--no-report"])
        self.quiet(["run", self.out, "--no-report", "--allow-changed"])
        self.assertEqual(self.ledger_ends()[0]["status"], "ok")

    def test_main_env_arms_and_variant_use_shared_data(self):
        sh(self.repo, "git", "branch", "-m", "main")
        variant = os.path.join(self.repo, "variants", "new")
        os.makedirs(variant)
        shutil.copy(os.path.join(TESTS, "fake_trainer.py"), os.path.join(variant, "train_gpt.py"))
        sh(self.repo, "git", "add", "variants")
        sh(self.repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "variant")
        with open(os.path.join(self.repo, "data", "fineweb10B", "sentinel.bin"), "w") as f:
            f.write("shared data")
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "base=ref:main",
                    "--arm", "new=ref:main", "--env", "new:FAKE_TIME=39", "--arm", "variant=path:variants/new",
                    "--env", f"variant:FAKE_EXPECT_CWD={variant}", "--common-env", "FAKE_REQUIRE_DATA=1",
                    "--cmd", f"{PY} {{script}}", "--n", "2", "--seeds", "1,2"])
        with mock.patch.dict(os.environ, {"DATA_PATH": "/invalid/inherited/data"}):
            self.quiet(["run", self.out, "--limit", "3", "--no-report"])
            # Path-arm links/logs do not invalidate its pinned fingerprint on resume.
            self.quiet(["run", self.out, "--no-report"])
        ends = self.ledger_ends()
        self.assertEqual(len(ends), 6)
        self.assertTrue(all(r["status"] == "ok" for r in ends))
        vr = [r for r in ends if r["arm"] == "variant"]
        self.assertEqual(vr[0]["cwd"], variant)
        self.assertEqual(os.path.realpath(os.path.join(vr[0]["env"]["DATA_PATH"], "data/fineweb10B")),
                         os.path.realpath(os.path.join(self.repo, "data/fineweb10B")))
        times = {(r["arm"], r["rep"]): r["train_time_s"] for r in ends}
        self.assertAlmostEqual(times[("new", 0)] - times[("base", 0)], -1)

    def test_variant_fingerprint_ignores_other_tracked_directories(self):
        variant = os.path.join(self.repo, "variants", "new")
        os.makedirs(variant)
        with open(os.path.join(variant, "code.py"), "w") as f:
            f.write("x = 1\n")
        sh(self.repo, "git", "add", "variants")
        sh(self.repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "variant")
        before = ab_run.dir_fingerprint(variant)
        with open(os.path.join(self.repo, "train_gpt.py"), "a") as f:
            f.write("\n# another arm's edit\n")
        self.assertEqual(before, ab_run.dir_fingerprint(variant))
        with open(os.path.join(variant, "code.py"), "a") as f:
            f.write("x = 2\n")
        self.assertNotEqual(before, ab_run.dir_fingerprint(variant))

    def test_stale_shared_data_link_is_rejected(self):
        self.plan(n=1)
        plan = ab_run.load_plan(self.out)
        arm = plan["arms"][0]
        wt = ab_run.prepare_arm(plan, arm, False)
        link = os.path.join(wt, "data/fineweb10B")
        os.unlink(link)
        wrong = os.path.join(self.tmp, "wrong-data")
        os.mkdir(wrong)
        os.symlink(wrong, link)
        with self.assertRaisesRegex(SystemExit, "shared data"):
            ab_run.prepare_arm(plan, arm, False)

    def test_stop_after_failures_leaves_slots_pending_and_resumes(self):
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "a=ref:master", "--arm", "b=ref:master",
                    "--common-env", "FAKE_FAIL=1", "--cmd", f"{PY} {{script}}", "--n", "3"])
        out = self.quiet(["run", self.out, "--no-report", "--stop-after-failures", "2"])
        self.assertIn("stopping: 2 consecutive non-ok runs", out)
        self.assertEqual(len(self.ledger_ends()), 2)  # 4 slots stay pending, nothing is recorded for them
        self.quiet(["run", self.out, "--no-report"])
        self.assertEqual(len(self.ledger_ends()), 6)
        self.assertEqual({r["status"] for r in self.ledger_ends()}, {"failed"})

    def test_one_arm_failing_every_run_stops_early(self):
        """The arms interleave, so one always-crashing arm never makes K consecutive failures: --arm-failures does."""
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "a=ref:master", "--arm", "b=ref:master",
                    "--env", "b:FAKE_FAIL=1", "--cmd", f"{PY} {{script}}", "--n", "4"])
        out = self.quiet(["run", self.out, "--no-report", "--stop-after-failures", "3", "--arm-failures", "2"])
        self.assertIn("stopping: arm b failed all of its 2 run(s) so far", out)
        ends = self.ledger_ends()
        self.assertEqual(sum(r["arm"] == "b" for r in ends), 2)
        self.assertLessEqual(len(ends), 4)          # at least 4 slots stay pending
        # a resume probes the dead arm once more, then stops again
        out = self.quiet(["run", self.out, "--no-report", "--arm-failures", "2"])
        self.assertIn("stopping: arm b failed all of its 3 run(s)", out)
        self.assertEqual(sum(r["arm"] == "b" for r in self.ledger_ends()), 3)

    def test_dead_variant_arm_is_abandoned_and_the_others_finish(self):
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "a=ref:master", "--arm", "b=ref:master",
                    "--arm", "c=ref:master", "--env", "b:FAKE_FAIL=1", "--cmd", f"{PY} {{script}}", "--n", "3"])
        out = self.quiet(["run", self.out, "--no-report", "--arm-failures", "2", "--dead-arm", "abandon"])
        self.assertIn("abandoned: arm b failed all of its first 2 run(s)", out)
        ends = self.ledger_ends()
        self.assertEqual(len(ends), 9)
        st = {}
        for r in ends:
            st.setdefault(r["arm"], []).append(r["status"])
        self.assertEqual(st["a"], ["ok"] * 3)
        self.assertEqual(st["c"], ["ok"] * 3)
        self.assertEqual(sorted(st["b"]), ["abandoned", "failed", "failed"])
        self.assertIn("cause:", out)
        self.assertIn("non-ok", self.quiet(["report", self.out]))

    def test_dead_baseline_stops_even_with_abandon(self):
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "a=ref:master", "--arm", "b=ref:master",
                    "--env", "a:FAKE_FAIL=1", "--cmd", f"{PY} {{script}}", "--n", "3"])
        out = self.quiet(["run", self.out, "--no-report", "--arm-failures", "2", "--dead-arm", "abandon"])
        self.assertIn("stopping: arm a failed all of its 2 run(s)", out)

    def test_failure_cause(self):
        self.assertEqual(ab_run.failure_cause("step 1\nTraceback (most recent call last):\n  File x\nRuntimeError: CUDA oops\n\n"),
                         "RuntimeError: CUDA oops")
        self.assertEqual(ab_run.failure_cause("a\nlast words\n"), "last words")
        self.assertEqual(ab_run.failure_cause(""), "(no output)")

    def test_interrupting_the_harness_stops_the_trainer(self):
        import signal
        import time
        pidf = os.path.join(self.tmp, "trainer.pid")
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "a=ref:master", "--n", "1",
                    "--cmd", f"{PY} -c \"import os,time; open('{pidf}','w').write(str(os.getpid())); time.sleep(120)\""])
        for sig in (signal.SIGINT, signal.SIGTERM):
            if os.path.exists(pidf):
                os.remove(pidf)
            p = subprocess.Popen([PY, ab_run.__file__, "run", self.out, "--no-report"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(200):
                if os.path.exists(pidf) and open(pidf).read():
                    break
                time.sleep(0.05)
            pid = int(open(pidf).read())
            p.send_signal(sig)
            p.wait(timeout=30)
            time.sleep(0.2)
            with self.assertRaises(ProcessLookupError, msg=f"trainer survived {sig!r} of the harness"):
                os.kill(pid, 0)
        self.assertEqual(self.ledger_ends(), [])  # the slot is still pending

    def test_missing_executable_counts_as_failure_and_is_not_retried(self):
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "bad=ref:master",
                    "--cmd", "/nonexistent/trainer", "--n", "1"])
        self.quiet(["run", self.out, "--no-report"])
        self.quiet(["run", self.out, "--no-report"])
        self.assertEqual(len(self.ledger_ends()), 1)
        self.assertEqual(self.ledger_ends()[0]["status"], "failed")
        self.assertIn("launch_error", self.ledger_ends()[0])

    def test_torn_ledger_tail_recovers_without_losing_finished_slots(self):
        self.plan(n=1)
        self.quiet(["run", self.out, "--limit", "1", "--no-report"])
        ab_run.append_ledger(self.out, dict(event="start", idx=1, arm="new", time=0))
        with open(os.path.join(self.out, "ledger.jsonl"), "a") as f:
            f.write('{"event":"end", "idx":1, "status":')
        self.quiet(["run", self.out, "--no-report"])
        self.assertEqual([r["idx"] for r in self.ledger_ends()], [0, 1, 2, 3])
        self.assertEqual(len([r for r in ab_run.read_ledger(self.out) if r["event"] == "interrupted"]), 1)

    def test_valid_ledger_tail_without_newline_is_preserved(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "ledger.jsonl"), "w") as f:
            f.write('{"event":"end", "idx":0}')
        ab_run.append_ledger(self.out, dict(event="end", idx=1))
        self.assertEqual([r["idx"] for r in ab_run.read_ledger(self.out)], [0, 1])

    def test_timeout_counts_and_timing_is_not_wall_time(self):
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "slow=ref:master",
                    "--cmd", f'{PY} -c "import time; time.sleep(5)"', "--timeout-s", "0.05", "--n", "1"])
        self.quiet(["run", self.out, "--no-report"])
        self.assertEqual(self.ledger_ends()[0]["status"], "timeout")
        self.assertIsNone(self.ledger_ends()[0]["train_time_s"])

    def test_track3_seconds_format_and_at_step(self):
        self.quiet(["plan", "--out", self.out, "--repo", self.repo, "--arm", "a=ref:master", "--common-env", "FAKE_FORMAT=s",
                    "--cmd", f"{PY} {{script}}", "--n", "2", "--seeds", "1,2"])
        self.quiet(["run", self.out, "--no-report"])
        ends = self.ledger_ends()
        self.assertTrue(all(r["status"] == "ok" and 39 < r["train_time_s"] < 41 for r in ends))
        self.assertEqual([r["seed"] for r in ends], [1, 2])

    def test_parse_existing_logs(self):
        text = self.quiet(["parse", "--group", f"base={ANVIL2}/baseline", "--group", f"anvil2={ANVIL2}/this_pr"])
        self.assertIn("| base | 9/9 | 73.889 ± 0.137 |", text)
        self.assertIn("| anvil2 | 17/17 |", text)


if __name__ == "__main__":
    unittest.main()
