"""182/gpu_session.sh and summarize_session.py on CPU: a fake trainer in a throwaway git repo, driven through the
real script and the real ab_run.py (no torch, no GPU)."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

from _paths import REPO, TOOLS
import summarize_session as ss
import stats

FAKE_TRAINER = r'''
import os, random, sys, uuid
if os.environ.get("FAKE_CRASH") == "1":
    sys.exit(3)
seed = os.environ.get("TRAIN_SEED")
rng = random.Random(int(seed) if seed is not None else None)
val = 3.278 + rng.gauss(0, 0.001)              # same seed -> same noise in every arm
t = 40.0
if os.environ.get("FP8_OFF"):
    val -= float(os.environ.get("FAKE_GAP", "0"))
    t += 10.0
if os.environ.get("HADAMARD_FP8") == "1":
    t += 3.0 if os.environ.get("HADAMARD_BLOCK", "128") == "128" else 1.0
total = 100
os.makedirs("logs", exist_ok=True)
path = f"logs/{uuid.uuid4()}.txt"
print(path, flush=True)
with open(path, "w") as f:
    f.write("source\n" + "=" * 100 + "\nRunning PyTorch 2.10.0+cu128 compiled for CUDA 12.8\n" + "=" * 100 + "\n")
    f.write(f"model: seed={seed} steps={total} stage boundaries=[(0, 40), (40, 100)]\n")
    f.write(f"fp8 experiment: FP8_OFF={os.environ.get('FP8_OFF') or 'none'} HADAMARD_FP8={os.environ.get('HADAMARD_FP8', '0')}\n")
    # stage 1 at 0.3 s/step, stage 2 at the rest
    def cum(s):
        return 0.3 * s if s <= 40 else 12.0 + (t - 12.0) * (s - 40) / 60
    for s in range(20, total + 1, 20):  # 40 is a progress point: stage 1 is exactly 300 ms/step
        f.write(f"step:{s}/{total} train_time:{cum(s)*1000:.0f}ms step_avg:{cum(s)*1000/s:.2f}ms\n")
    line = f"step:{total}/{total} val_loss:{val:.4f} train_time:{(t + 0.2)*1000:.0f}ms step_avg:{(t + 0.2)*10:.2f}ms"
    f.write(line + "\npeak memory allocated: 100 MiB reserved: 200 MiB\n")
print(line)
'''


class SessionRepo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        (self.repo / "182/tools").mkdir(parents=True)
        shutil.copy(Path(REPO) / "182/gpu_session.sh", self.repo / "182/gpu_session.sh")
        for f in ("ab_run.py", "logparse.py", "stats.py", "summarize_session.py", "node_setup.sh"):
            shutil.copy(Path(TOOLS) / f, self.repo / "182/tools" / f)
        (self.repo / "train_gpt.py").write_text(FAKE_TRAINER)
        (self.repo / "track_1_short").mkdir()
        (self.repo / "track_1_short/config.py").write_text("HADAMARD_FP8 = 0\n")
        (self.repo / ".gitignore").write_text("182/results/\nlogs/\nnode_report.txt\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.marker = self.root / "torchrun_called"
        self.stub("torchrun", f'touch "{self.marker}"; exit 7\n')
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
        self.sd = self.root / "sess"
        self.env = dict(os.environ, PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}", PY=sys.executable,
                        SESSION_DIR=str(self.sd), RUN_CMD=f"{sys.executable} {{script}}", SKIP_DATA="1",
                        TMPDIR=str(self.root))
        for k in ("DRY_RUN", "FORCE", "FORCE_HADAMARD", "GATE_SEEDS", "HADAMARD_VARIANTS", "HADAMARD_SEEDS", "SESSION_DOCKER"):
            self.env.pop(k, None)

    def tearDown(self):
        self.tmp.cleanup()

    def stub(self, name, body):
        p = self.bin / name
        p.write_text("#!/usr/bin/env bash\n" + body)
        p.chmod(0o755)

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def run_session(self, *phases, **env):
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh"), *phases], cwd=self.repo,
                           env=dict(self.env, **env), capture_output=True, text=True, timeout=600)
        return r.returncode, r.stdout + r.stderr

    def status(self, phase):
        return (self.sd / "status" / phase).read_text().strip()

    def ends(self, ab):
        return [json.loads(l) for l in (self.sd / ab / "ledger.jsonl").read_text().splitlines() if '"end"' in l]


class TestGpuSession(SessionRepo):
    def test_unknown_phase(self):
        rc, out = self.run_session("gate", "bogus")
        self.assertEqual(rc, 2, out)
        self.assertIn("unknown phase 'bogus'", out)

    def test_dry_run_all_prints_and_launches_nothing(self):
        env = {k: v for k, v in self.env.items() if k != "SESSION_DIR"}
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh")], cwd=self.repo, env=dict(env, DRY_RUN="1"),
                           capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertFalse(self.marker.exists(), "DRY_RUN launched torchrun")
        pos = [out.index(f"== {p}: est. ") for p in ("env", "tests", "sanity", "gate", "hadamard", "t2", "bundle")]
        self.assertEqual(pos, sorted(pos))
        self.assertIn("node_check.py", out)
        self.assertRegex(out, r"\+ .*-m pytest 182/hadamard-fp8/tests")
        self.assertRegex(out, r"\+ .*ab_run\.py run .*/gate --no-report")
        self.assertIn("GATE: NODATA", out)
        self.assertIn("would skip: gate decision NODATA", out)
        self.assertIn("SKIP sanity: 182/hadamard-fp8/gpu_sanity.py is not in this checkout", out)
        self.assertIn("SEND THIS FILE BACK: ", out)
        self.assertFalse((self.repo / "182/results").exists(), "DRY_RUN wrote into the repo")
        self.assertFalse((self.repo / "node_report.txt").exists())

    def test_gate_stop_skips_hadamard_bundles_and_resumes(self):
        rc, out = self.run_session("gate", "hadamard", "bundle", FAKE_GAP="0")
        self.assertEqual(rc, 0, out)
        self.assertFalse(self.marker.exists())
        self.assertEqual(len(self.ends("gate")), 8)
        self.assertEqual({e["status"] for e in self.ends("gate")}, {"ok"})
        # the same seed in both arms within a rep, four distinct seeds
        seeds = {(e["arm"], e["seed"]) for e in self.ends("gate")}
        self.assertEqual(seeds, {(a, s) for a in ("fp8", "bf16") for s in (1, 2, 3, 4)})
        dec = (self.sd / "gate/decision.env").read_text()
        self.assertIn("GATE_DECISION=STOP", dec)
        self.assertIn("GATE_RUN_HADAMARD=0", dec)
        self.assertTrue(self.status("hadamard").startswith("SKIP gate decision STOP"), self.status("hadamard"))
        bundle = Path((self.sd / "bundle_path").read_text().strip())
        self.assertTrue(bundle.name.startswith("session-") and bundle.name.endswith(".tar.gz"), bundle)
        self.assertIn(f"SEND THIS FILE BACK: {bundle}", out)
        with tarfile.open(bundle) as tf:
            names = tf.getnames()
        self.assertFalse([n for n in names if "/work/" in n or n.endswith("/work")], "worktrees/caches bundled")
        self.assertTrue(any(n.endswith("gate/ledger.jsonl") for n in names))
        self.assertTrue(any(n.endswith("summary.txt") for n in names))
        # the bundle alone reproduces the summary (ledger paths are remapped into the bundle)
        r = subprocess.run([sys.executable, str(Path(TOOLS) / "summarize_session.py"), str(bundle)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("-> STOP", r.stdout)
        self.assertRegex(r.stdout, r"fp8 +4/4 +3\.\d{5}±")
        self.assertRegex(r.stdout, r"fp8 .* 300\.00 466\.67 ")   # ms/step per stage: 0.3 s, then 28 s / 60 steps
        self.assertRegex(r.stdout, r"bf16 .* 300\.00 633\.33 .*FP8_OFF=1")
        # resume: gate is PASS and is not re-run
        rc, out = self.run_session("gate", FAKE_GAP="0")
        self.assertEqual(rc, 0, out)
        self.assertIn("== gate: already PASS", out)
        self.assertEqual(len(self.ends("gate")), 8)

    def test_gate_gap_runs_hadamard_variants(self):
        (self.repo / "track_1_short/config.py").write_text("HADAMARD_GEMMS = 'all'\nHADAMARD_BLOCK = 128\n")
        self.git("commit", "-qam", "knobs")
        rc, out = self.run_session("gate", "hadamard", FAKE_GAP="0.005")
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE_DECISION=GAP", (self.sd / "gate/decision.env").read_text())
        plan = json.loads((self.sd / "hadamard/plan.json").read_text())
        arms = {a["name"]: a["env"] for a in plan["arms"]}
        self.assertEqual(arms, {"fp8": {}, "h_all_b128": {"HADAMARD_FP8": "1", "HADAMARD_GEMMS": "all", "HADAMARD_BLOCK": "128"},
                                "h_dw1_dw2_b32": {"HADAMARD_FP8": "1", "HADAMARD_GEMMS": "dw1,dw2", "HADAMARD_BLOCK": "32"}})
        self.assertEqual(len(self.ends("hadamard")), 6)
        # the hadamard baseline reuses the gate's fp8 worktree and warm cache
        self.assertEqual(len(list((self.sd / "work/wt").iterdir())), 4)
        self.assertRegex(out, r"paired h_all_b128-fp8 \(n=2\): Δval \+0\.00±0\.00 mnat .*Δstep_sum \+3\.000")
        self.assertRegex(out, r"paired h_dw1_dw2_b32-fp8 \(n=2\): .*Δstep_sum \+1\.000")

    def test_hadamard_needs_gate_unless_forced(self):
        rc, out = self.run_session("hadamard")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.status("hadamard").startswith("SKIP the gate has not produced a decision"))
        rc, out = self.run_session("hadamard", FORCE_HADAMARD="1", HADAMARD_SEEDS="5")
        self.assertEqual(rc, 0, out)
        self.assertIn("no HADAMARD_GEMMS/HADAMARD_BLOCK knobs", out)
        self.assertEqual(self.status("hadamard"), "PASS")
        self.assertEqual({e["arm"] for e in self.ends("hadamard")}, {"fp8", "h_default"})

    def test_failed_hadamard_tests_block_hadamard(self):
        (self.sd / "gate").mkdir(parents=True)
        (self.sd / "gate/decision.env").write_text("GATE_DECISION=GAP\nGATE_RUN_HADAMARD=1\nGATE_HI_MNAT=3\n")
        (self.sd / "tests").mkdir()
        (self.sd / "tests/rc_hadamard").write_text("1\n")
        rc, out = self.run_session("hadamard")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.status("hadamard").startswith("SKIP the HADAMARD_FP8=1 tests failed"), self.status("hadamard"))

    def test_sanity_failure_is_recorded_and_session_continues(self):
        rc, out = self.run_session("sanity", "bundle", SANITY_CMD="false")
        self.assertEqual(rc, 1, out)
        self.assertTrue(self.status("sanity").startswith("FAIL"))
        self.assertEqual(self.status("bundle"), "PASS")
        self.assertIn("failed phases: sanity", out)

    def test_crashing_gate_stops_early_resumes_then_replans(self):
        rc, out = self.run_session("gate", FAKE_CRASH="1")
        self.assertEqual(rc, 1, out)
        self.assertTrue(self.status("gate").startswith("FAIL"), self.status("gate"))
        self.assertIn("stopping: 3 consecutive non-ok runs", out)
        self.assertEqual(len(self.ends("gate")), 3)          # 5 slots left pending, not burned
        self.assertIn("GATE_DECISION=NODATA", (self.sd / "gate/decision.env").read_text())
        rc, out = self.run_session("gate", FAKE_CRASH="1", STOP_AFTER_FAILURES="100")
        self.assertEqual(rc, 1, out)
        self.assertIn("resuming: 5 slot(s) pending", out)
        self.assertEqual(len(self.ends("gate")), 8)
        # every slot finished, none usable: the next re-run sets the attempt aside and plans afresh
        rc, out = self.run_session("gate", "bundle", FAKE_GAP="0")
        self.assertEqual(rc, 0, out)
        self.assertIn("re-planning: previous attempt moved to", out)
        self.assertEqual(self.status("gate"), "PASS")
        self.assertIn("GATE_DECISION=STOP", (self.sd / "gate/decision.env").read_text())
        self.assertEqual(len(self.ends("gate.attempt1")), 8)
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("== gate.attempt1:", summary)
        self.assertIn("GATE: NODATA", summary)
        self.assertIn("-> STOP", summary)
        # FORCE=1 re-plans a finished phase instead of silently resuming nothing
        rc, out = self.run_session("gate", FORCE="1", GATE_SEEDS="7,8")
        self.assertEqual(rc, 0, out)
        self.assertTrue((self.sd / "gate.attempt2/plan.json").exists())
        self.assertEqual({e["seed"] for e in self.ends("gate")}, {7, 8})

    def test_bad_hadamard_variant_fails_before_planning(self):
        rc, out = self.run_session("hadamard", FORCE_HADAMARD="1", HADAMARD_VARIANTS="all@128 dx")
        self.assertEqual(rc, 1, out)
        self.assertIn("HADAMARD_VARIANTS entry 'dx' is not GEMMS@BLOCK", out)
        self.assertFalse((self.sd / "hadamard/plan.json").exists())

    def test_quick_mode_phases(self):
        env = {k: v for k, v in self.env.items() if k != "SESSION_DIR"}
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh")], cwd=self.repo,
                           env=dict(env, DRY_RUN="1", QUICK="1"), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("phases: env sanity gate bundle ", r.stdout)
        self.assertNotIn("== tests:", r.stdout)
        self.assertNotIn("t2 prefetch", r.stdout)

    def test_hung_command_is_timed_out(self):
        if subprocess.run(["timeout", "--foreground", "1", "true"], capture_output=True).returncode != 0:
            self.skipTest("no coreutils timeout")
        rc, out = self.run_session("sanity", SANITY_CMD="sleep 60", SANITY_TIMEOUT_S="1")
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.status("sanity"), "FAIL rc=124")

    def test_t2_skips_with_reason_on_small_host(self):
        rc, out = self.run_session("t2", T2_MIN_RAM_GIB="100000")
        self.assertEqual(rc, 0, out)
        self.assertRegex(self.status("t2"), r"^SKIP host cannot run t2: RAM available \d+ GiB < 100000 GiB")
        self.assertFalse(self.marker.exists())


class TestSummarize(unittest.TestCase):
    def test_log_details_stages_and_step_sum(self):
        text = ("x\n" + "=" * 100 + "\n[schedule] stage ends=[0, 50, 100]\n"
                "step:25/100 train_time:500ms step_avg:20ms\nstep:50/100 train_time:1000ms step_avg:20ms\n"
                "step:75/100 train_time:2000ms step_avg:26ms\nstep:100/100 train_time:3000ms step_avg:30ms\n"
                "step:100/100 val_loss:3.2800 train_time:3100ms step_avg:31ms\n")
        d = ss.log_details(text)
        self.assertEqual(d["stage_ends"], [0, 50, 100])
        self.assertAlmostEqual(d["stage_ms"][0], 20.0)
        self.assertAlmostEqual(d["stage_ms"][1], 40.0)
        self.assertAlmostEqual(d["step_sum_s"], 3.0)
        # the trainer's other format, interpolating inside a 25-step interval
        d = ss.log_details(text.replace("[schedule] stage ends=[0, 50, 100]", "stage boundaries=[(0, 60), (60, 100)]"))
        self.assertAlmostEqual(d["stage_ms"][0], (1000 + 0.4 * 1000) / 60)

    def test_real_record_log(self):
        p = os.path.join(REPO, "records/track_1_short/2026-08-30_ANVIL2/this_pr")
        logs = sorted(f for f in os.listdir(p) if f.endswith(".txt"))
        with open(os.path.join(p, logs[0])) as f:
            d = ss.log_details(f.read())
        self.assertEqual(d["stage_ends"][-1], 1194)
        self.assertEqual(len(d["stage_ms"]), len(d["stage_ends"]) - 1)
        self.assertTrue(all(10 < x < 60 for x in d["stage_ms"]), d["stage_ms"])
        self.assertTrue(35 < d["step_sum_s"] < 42)

    def test_gate_ci_matches_hand_computation(self):
        diffs = [1.0, 0.2, 0.6, -0.2]  # fp8 - bf16, mnat
        groups = {"fp8": [], "bf16": []}
        for rep, dd in enumerate(diffs):
            groups["fp8"].append(dict(rep=rep, status="ok", val_loss=3.278 + dd / 1000, details={}))
            groups["bf16"].append(dict(rep=rep, status="ok", val_loss=3.278, details={}))
        ab = dict(arms=["fp8", "bf16"], env={"fp8": {}, "bf16": {"FP8_OFF": "1"}}, groups=groups, plan={"n": 4})
        g = ss.gate_decision(ab)
        m, sd = stats.mean_sd(diffs)
        se = sd / 2
        h = 3.182446305284263 * se  # t_0.975,3
        self.assertAlmostEqual(g["mean_mnat"], m, places=6)
        self.assertAlmostEqual(g["se_mnat"], se, places=6)
        self.assertAlmostEqual(g["hi_mnat"], m + h, places=4)
        self.assertEqual(g["decision"], "INCONCLUSIVE")
        self.assertTrue(g["run_hadamard"])
        for rep in range(4):
            groups["fp8"][rep]["val_loss"] = 3.278 + [0.1, 0.0, 0.2, 0.1][rep] / 1000
        self.assertEqual(ss.gate_decision(ab)["decision"], "STOP")
        for rep in range(4):
            groups["fp8"][rep]["val_loss"] = 3.278 + [2.1, 2.0, 2.2, 2.1][rep] / 1000
        self.assertEqual(ss.gate_decision(ab)["decision"], "GAP")
        groups["fp8"] = groups["fp8"][:1]
        self.assertEqual(ss.gate_decision(ab)["decision"], "NODATA")

    def test_real_record_logs_with_failed_runs(self):
        """An ab_run dir as a bundle has it: real ANVIL2 logs for the fp8 arm, and a crash, a timeout and a
        NaN (with the console stdout only, no log copy) next to one ok run in the FP8_OFF arm."""
        src = os.path.join(REPO, "records/track_1_short/2026-08-30_ANVIL2/this_pr")
        logs = sorted(f for f in os.listdir(src) if f.endswith(".txt"))[:4]
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "logs")); os.makedirs(os.path.join(d, "stdout"))
            runs, recs = [], []
            for rep in range(4):
                for arm in ("fp8", "bf16"):
                    idx = len(runs)
                    runs.append(dict(idx=idx, rep=rep, arm=arm, seed=rep + 1, kind="measure"))
                    so = os.path.join(d, "stdout", f"{idx:03d}_{arm}.out")
                    rec = dict(event="end", idx=idx, rep=rep, arm=arm, kind="measure", seed=rep + 1,
                               stdout="/gpu/box/" + os.path.basename(so), log=None, val_loss=None, train_time_s=None)
                    if arm == "fp8" or rep == 3:
                        shutil.copy(os.path.join(src, logs[rep]), os.path.join(d, "logs", f"{idx:03d}_{logs[rep]}"))
                        from logparse import parse_file
                        lg = parse_file(os.path.join(d, "logs", f"{idx:03d}_{logs[rep]}"))
                        rec.update(status="ok", log="/gpu/box/" + f"{idx:03d}_{logs[rep]}", val_loss=lg.val_loss,
                                   train_time_s=lg.train_time_s)
                        if arm == "bf16":
                            rec["val_loss"] -= 0.001
                        Path(so).write_text("ok\n")
                    else:
                        status, text = [("failed", "Traceback (most recent call last):\nRuntimeError: CUDA error\n"),
                                        ("timeout", "logs/x.txt\nstep:300/1194 train_time:12000ms step_avg:40.00ms\n"),
                                        ("nan", "step:1194/1194 val_loss:nan train_time:39000ms step_avg:32ms\n")][rep]
                        rec["status"] = status
                        Path(so).write_text(text)
                    recs.append(rec)
            plan = dict(arms=[dict(name="fp8", env={}), dict(name="bf16", env={"FP8_OFF": "1"})], baseline="fp8",
                        n=4, seeds=[1, 2, 3, 4], cache="warm", order="blocked", runs=runs)
            Path(d, "plan.json").write_text(json.dumps(plan))
            Path(d, "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            ab = ss.load_ab(d)
            text = "\n".join(ss.ab_table("gate", ab))
            self.assertRegex(text, r"fp8 +4/4 +3\.2\d{4}±")
            self.assertRegex(text, r"bf16 +1/4 .*FP8_OFF=1 +\[non-ok: failed,timeout,nan\]")
            self.assertTrue(all(5 < x < 60 for x in ab["groups"]["fp8"][0]["details"]["stage_ms"]))
            self.assertIn("paired bf16-fp8 (n=1): Δval -1.00±nan mnat", text)
            g = ss.gate_decision(ab)
            self.assertEqual((g["decision"], g["n"]), ("NODATA", 1))
            self.assertEqual(ss.ab_check(ab, gate=True)[0], 4)
            self.assertEqual(ss.ab_check(ab)[0], 0)  # each arm has an ok run: fine for a non-gate phase
            os.remove(os.path.join(d, "ledger.jsonl"))
            ab = ss.load_ab(d)
            self.assertEqual(ss.ab_check(ab, gate=True), (3, "8 slot(s) pending"))
            self.assertIn("[8 slot(s) never ran]", ss.ab_table("gate", ab)[0])
            self.assertIn("GATE: NODATA", "\n".join(ss.gate_lines(ss.gate_decision(ab))))

    def test_t2_without_train_time(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "stdout.txt").write_text("endpoint_result: steps=5800 val_loss=2.95\n")
            out = "\n".join(ss.t2_lines(d))
        self.assertIn("train_time=nan s", out)
        self.assertIn("does NOT reproduce", out)

    def test_t2_vs_reference(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "stdout.txt").write_text(
                "logs/ab.txt\nstep:5800/5800 val_loss:2.8790 train_time:225000ms step_avg:38.79ms\n"
                "endpoint_result: steps=5800 seed=None val_loss=2.879012 train_time_ms=225000\n")
            out = "\n".join(ss.t2_lines(d))
        self.assertIn("val_loss=2.8790", out)
        self.assertIn("Δval +2.1 mnat", out)
        self.assertIn("Δtime +4.0 s", out)
        self.assertIn("reproduces", out)
        self.assertNotIn("NOT", out)


if __name__ == "__main__":
    unittest.main()
