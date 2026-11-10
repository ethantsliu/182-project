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
seed = os.environ.get("TRAIN_SEED")
arm = "bf16" if os.environ.get("FP8_OFF") else ("had" if os.environ.get("HADAMARD_FP8") == "1" else "fp8")
if os.environ.get("FAKE_CRASH") == "1" or f"{arm}:{seed}" in os.environ.get("FAKE_CRASH_RUNS", "").split(",") \
        or arm in os.environ.get("FAKE_CRASH_ARMS", "").split(","):
    print("Traceback (most recent call last):\nRuntimeError: fake crash in " + arm, flush=True)
    sys.exit(3)
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
        self.assertIn("GATE: INCOMPLETE", out)
        self.assertIn("would skip: gate decision INCOMPLETE", out)
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
        self.assertTrue(self.status("hadamard").startswith("SKIP the gate phase did not complete (status: none)"), self.status("hadamard"))
        rc, out = self.run_session("hadamard", FORCE_HADAMARD="1", HADAMARD_SEEDS="5")
        self.assertEqual(rc, 0, out)
        self.assertIn("no HADAMARD_GEMMS/HADAMARD_BLOCK knobs", out)
        self.assertEqual(self.status("hadamard"), "PASS")
        self.assertEqual({e["arm"] for e in self.ends("hadamard")}, {"fp8", "h_default"})

    def test_failed_hadamard_tests_block_hadamard(self):
        (self.sd / "gate").mkdir(parents=True)
        (self.sd / "gate/decision.env").write_text("GATE_DECISION=GAP\nGATE_COMPLETE=1\nGATE_RUN_HADAMARD=1\nGATE_HI_MNAT=3\n")
        (self.sd / "status").mkdir()
        (self.sd / "status/gate").write_text("PASS\n")
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

    def test_crashing_gate_stops_early_and_is_not_replanned_until_forced(self):
        rc, out = self.run_session("gate", FAKE_CRASH="1")
        self.assertEqual(rc, 1, out)
        self.assertTrue(self.status("gate").startswith("FAIL"), self.status("gate"))
        self.assertIn("stopping: 3 consecutive non-ok runs", out)
        self.assertIn("cause: RuntimeError: fake crash in", out)
        self.assertEqual(len(self.ends("gate")), 3)          # 5 slots left pending, not burned
        dec = (self.sd / "gate/decision.env").read_text()
        self.assertIn("GATE_DECISION=INCOMPLETE", dec)
        self.assertIn("GATE_RUN_HADAMARD=0", dec)
        self.assertIn("FAIL: the gate stopped with slots pending", out)
        self.assertNotIn("-> GAP", out)
        # resume: the per-arm check stops after one more probe of an arm that has never succeeded
        rc, out = self.run_session("gate", FAKE_CRASH="1", STOP_AFTER_FAILURES="100")
        self.assertEqual(rc, 1, out)
        self.assertIn("resuming: 5 slot(s) pending", out)
        self.assertIn("every finished run failed", out)
        self.assertRegex(out, r"stopping: arm (fp8|bf16) failed all of its 2 run\(s\)")
        self.assertEqual(len(self.ends("gate")), 4)
        # burn the rest: every slot finished and both arms failed every run -> code 5, not re-planned
        rc, out = self.run_session("gate", FAKE_CRASH="1", STOP_AFTER_FAILURES="100", ARM_FAILURES="0")
        self.assertEqual(rc, 1, out)
        self.assertEqual(len(self.ends("gate")), 8)
        self.assertIn("FAIL: one gate arm failed every run", out)
        rc, out = self.run_session("gate", FAKE_GAP="0")
        self.assertEqual(rc, 1, out)
        self.assertIn("NOT re-planning gate", out)
        self.assertIn("FORCE=1 bash 182/gpu_session.sh gate", out)
        self.assertFalse((self.sd / "gate.attempt1").exists())
        # the cause is fixed: FORCE=1 re-plans
        rc, out = self.run_session("gate", "bundle", FAKE_GAP="0", FORCE="1")
        self.assertEqual(rc, 0, out)
        self.assertIn("re-planning: previous attempt moved to", out)
        self.assertEqual(self.status("gate"), "PASS")
        self.assertIn("GATE_DECISION=STOP", (self.sd / "gate/decision.env").read_text())
        self.assertEqual(len(self.ends("gate.attempt1")), 8)
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("== gate.attempt1:", summary)
        self.assertIn("GATE: NODATA", summary)
        self.assertIn("fake crash in bf16", summary)       # the failure cause is in the summary
        self.assertIn("-> STOP", summary)
        rc, out = self.run_session("gate", FORCE="1", GATE_SEEDS="7,8")
        self.assertEqual(rc, 0, out)
        self.assertTrue((self.sd / "gate.attempt2/plan.json").exists())
        self.assertEqual({e["seed"] for e in self.ends("gate")}, {7, 8})

    def test_scattered_failures_replan_the_gate(self):
        """fp8 crashes on seeds 1,3 and bf16 on 2,4: no complete pair, but no arm is dead -> re-plan (code 4)."""
        crash = "fp8:1,bf16:2,fp8:3,bf16:4"
        rc, out = self.run_session("gate", FAKE_CRASH_RUNS=crash)
        self.assertEqual(rc, 1, out)
        self.assertEqual(len(self.ends("gate")), 8)
        self.assertIn("GATE_DECISION=NODATA", (self.sd / "gate/decision.env").read_text())
        rc, out = self.run_session("gate", FAKE_GAP="0")
        self.assertEqual(rc, 0, out)
        self.assertIn("re-planning: previous attempt moved to", out)
        self.assertIn("GATE_DECISION=STOP", (self.sd / "gate/decision.env").read_text())

    def test_one_arm_always_failing_stops_after_two_of_its_runs(self):
        rc, out = self.run_session("gate", "hadamard", "bundle", FAKE_CRASH_ARMS="bf16")
        self.assertEqual(rc, 1, out)
        ends = self.ends("gate")
        self.assertEqual(sum(e["arm"] == "bf16" for e in ends), 2)
        self.assertLessEqual(len(ends), 4)
        self.assertIn("stopping: arm bf16 failed all of its 2 run(s)", out)
        self.assertTrue(self.status("hadamard").startswith("SKIP the gate phase did not complete"), self.status("hadamard"))
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("GATE: INCOMPLETE", summary)
        self.assertIn("bf16 #", summary)
        self.assertIn("RuntimeError: fake crash in bf16", summary)

    def test_dead_hadamard_variant_is_abandoned_not_replanned(self):
        (self.repo / "track_1_short/config.py").write_text("HADAMARD_GEMMS = 'all'\nHADAMARD_BLOCK = 128\n")
        self.git("commit", "-qam", "knobs")
        rc, out = self.run_session("hadamard", FORCE_HADAMARD="1", FAKE_CRASH_ARMS="had", HADAMARD_VARIANTS="all@128",
                                   HADAMARD_SEEDS="1,2,3")
        self.assertEqual(rc, 1, out)
        self.assertIn("abandoned: arm h_all_b128 failed all of its first 2 run(s)", out)
        st = [e["status"] for e in self.ends("hadamard") if e["arm"] == "h_all_b128"]
        self.assertEqual(sorted(st), ["abandoned", "failed", "failed"])
        self.assertEqual([e["status"] for e in self.ends("hadamard") if e["arm"] == "fp8"], ["ok"] * 3)
        rc, out = self.run_session("hadamard", FORCE_HADAMARD="1", HADAMARD_VARIANTS="all@128", HADAMARD_SEEDS="1,2,3")
        self.assertEqual(rc, 1, out)
        self.assertIn("NOT re-planning hadamard", out)

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
        rc, out = self.run_session("sanity", SANITY_CMD="sleep 60", SANITY_TIMEOUT_S="1")
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.status("sanity"), "FAIL rc=124")

    def test_t2_skips_with_reason_on_small_host(self):
        rc, out = self.run_session("t2", T2_MIN_RAM_GIB="100000")
        self.assertEqual(rc, 0, out)
        self.assertRegex(self.status("t2"), r"^SKIP host cannot run t2: RAM available \d+ GiB < 100000 GiB")
        self.assertFalse(self.marker.exists())


class TestSessionFlow(SessionRepo):
    """The session-flow fixes: docker env forwarding, read-only DRY_RUN, the stack gate, timeouts, t2 results."""

    def fake_stack(self, torch="2.10.0+cu128", triton="3.6.0"):
        pf = self.root / "pyfake"
        for mod, ver in (("torch", torch), ("triton", triton)):
            (pf / mod).mkdir(parents=True, exist_ok=True)
            body = f"__version__ = {ver!r}\n"
            if mod == "torch":
                body += "class version:\n    cuda = '12.8'\nclass cuda:\n    @staticmethod\n    def device_count(): return 8\n"
            (pf / mod / "__init__.py").write_text(body)
        # torchrun as a console script of the session interpreter (what the stack check looks for)
        (self.bin / "torchrun").write_text(f"#!{sys.executable}\nimport sys, pathlib\npathlib.Path({str(self.marker)!r}).touch()\nsys.exit(0)\n")
        (self.bin / "torchrun").chmod(0o755)
        self.stub("nvidia-smi", 'case "$*" in *driver_version*) echo 580.65;; -L) echo "GPU 0: NVIDIA H100";; *) echo "Driver Version: 580.65";; esac\n')
        return {"PYTHONPATH": str(pf)}

    def test_docker_forwards_only_user_set_knobs(self):
        env = {k: v for k, v in self.env.items() if k not in ("RUN_CMD", "SANITY_CMD", "PY")}
        self.stub("docker", "exit 0\n")
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh"), "sanity", "gate"], cwd=self.repo,
                           env=dict(env, DRY_RUN="1", SESSION_DOCKER="img", PY=sys.executable, T2_MIN_RAM_GIB="10",
                                    HADAMARD_MIN_GAP_MNAT="1.5"),
                           capture_output=True, text=True, timeout=60)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("--init", out)
        self.assertIn("TINI_KILL_PROCESS_GROUP=1", out)
        self.assertIn("T2_MIN_RAM_GIB=10", out)
        self.assertIn("HADAMARD_MIN_GAP_MNAT=1.5", out)
        self.assertNotIn("SANITY_CMD", out)          # the default is computed inside the container, with its python
        self.assertNotIn("RUN_CMD", out)
        self.assertNotIn(" PY=", out)
        self.assertNotIn("-e PY", out)
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh"), "sanity"], cwd=self.repo,
                           env=dict(env, DRY_RUN="1", SESSION_DOCKER="img", SANITY_CMD="python x.py"),
                           capture_output=True, text=True, timeout=60)
        self.assertIn("SANITY_CMD=python\\ x.py", r.stdout)

    def test_default_sanity_cmd_gets_json_whatever_py_is(self):
        (self.repo / "182/hadamard-fp8").mkdir(parents=True)
        (self.repo / "182/hadamard-fp8/gpu_sanity.py").write_text("import sys; print('sanity args', sys.argv[1:])\n")
        rc, out = self.run_session("sanity", PY=sys.executable)
        self.assertEqual(rc, 0, out)
        self.assertIn("sanity args ['--json'", out)

    def test_dry_run_with_session_dir_changes_nothing(self):
        rc, out = self.run_session("gate", FAKE_CRASH_ARMS="bf16")
        self.assertEqual(rc, 1, out)
        before = {str(p.relative_to(self.sd)): p.read_bytes() for p in self.sd.rglob("*") if p.is_file() and "/work/" not in str(p)}
        shutil.rmtree(self.repo / "182/tools/__pycache__", ignore_errors=True)
        rc, out = self.run_session("env", "gate", "hadamard", "bundle", DRY_RUN="1", FORCE="1")
        self.assertEqual(rc, 0, out)
        self.assertIn("previewing a resume of", out)
        self.assertIn("re-planning: previous attempt moved to", out)   # what FORCE=1 would do, on the copy
        after = {str(p.relative_to(self.sd)): p.read_bytes() for p in self.sd.rglob("*") if p.is_file() and "/work/" not in str(p)}
        self.assertEqual(before, after)
        self.assertFalse((self.sd.parent / "sess.attempt1").exists())
        self.assertFalse((self.repo / "node_report.txt").exists())
        self.assertFalse(list(self.root.glob("session-*.tar.gz")))
        self.assertFalse(list((self.repo / "182/tools").glob("__pycache__")))

    def test_dry_run_shows_hadamard_env_prefix(self):
        rc, out = self.run_session("tests", DRY_RUN="1")
        self.assertEqual(rc, 0, out)
        self.assertRegex(out, r"\+ tmo 1800 env HADAMARD_FP8=1 .*-m pytest 182/hadamard-fp8/tests .*-rfEs")

    def test_wrong_stack_fails_env_and_skips_gpu_phases(self):
        env = self.fake_stack(torch="2.9.1+cu126")
        rc, out = self.run_session("env", "gate", "bundle", **env)
        self.assertEqual(rc, 1, out)
        self.assertTrue(self.status("env").startswith("FAIL"), self.status("env"))
        self.assertIn("STACK PROBLEM: torch 2.9.1+cu126 != 2.10.0+cu128", out)
        self.assertTrue(self.status("gate").startswith("SKIP env failed"), self.status("gate"))
        self.assertEqual(self.status("bundle"), "PASS")
        self.assertFalse((self.sd / "gate/plan.json").exists())
        st = json.loads((self.sd / "env/stack.json").read_text())
        self.assertEqual(st["torch"], "2.9.1+cu126")
        self.assertIs(st["git_dirty"], False)
        self.assertTrue((self.sd / "env/pip_freeze.txt").stat().st_size > 0)
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("!! torch 2.9.1+cu126 != 2.10.0+cu128", summary)
        # ALLOW_STACK=1: a warning, and the session goes on
        rc, out = self.run_session("env", **env, ALLOW_STACK="1")
        self.assertEqual(rc, 0, out)
        self.assertIn("WARN: ALLOW_STACK=1", out)

    def test_right_stack_passes_and_untracked_files_are_not_dirty(self):
        env = self.fake_stack()
        (self.repo / "scratch.txt").write_text("untracked\n")
        rc, out = self.run_session("env", **env)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("STACK PROBLEM", out)
        self.assertNotIn("modified tracked files. The gate", out)
        st = json.loads((self.sd / "env/stack.json").read_text())
        self.assertEqual((st["torch"], st["triton"], st["git_dirty"]), ("2.10.0+cu128", "3.6.0", False))
        self.assertIn("?? scratch.txt", (self.sd / "env/env.txt").read_text())
        (self.repo / "train_gpt.py").write_text(FAKE_TRAINER + "# edit\n")
        rc, out = self.run_session("env", FORCE="1", **env)
        self.assertIn("WARN: modified tracked files", out)
        self.assertEqual(json.loads((self.sd / "env/stack.json").read_text())["dirty_files"], ["train_gpt.py"])

    def test_torchrun_from_another_environment_fails_env(self):
        env = self.fake_stack()
        other = self.root / "otherpy"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(other)], check=True)
        (self.bin / "torchrun").write_text(f"#!{other}/bin/python\nimport sys\nsys.exit(0)\n")
        rc, out = self.run_session("env", **env)
        self.assertEqual(rc, 1, out)
        self.assertIn("not the session interpreter", out)

    def test_timeout_kills_a_term_ignoring_grandchild(self):
        pidf = self.root / "grandchild.pid"
        script = self.root / "hang.sh"
        script.write_text(f"{sys.executable} -c 'import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                          f"open(\"{pidf}\", \"w\").write(str(os.getpid())); time.sleep(300)' &\nwait\n")
        cmd = f"bash {script}"
        rc, out = self.run_session("sanity", SANITY_CMD=cmd, SANITY_TIMEOUT_S="2")
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.status("sanity"), "FAIL rc=124")
        self.assertIn("exceeded 2 s", out)
        pid = int(pidf.read_text())
        import time
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_t2_exit_0_without_result_fails(self):
        t2 = self.repo / "variants/t2-cpu-ngram"
        t2.mkdir(parents=True)
        (t2 / "exact_match.py").write_text("")
        (t2 / "train_gpt_medium_cpu.py").write_text("print('started, then nothing')\n")
        d = self.repo / "data/fineweb10B"
        d.mkdir(parents=True)
        for i in range(1, 104):
            (d / f"fineweb_train_{i:06d}.bin").touch()
        (d / "fineweb_val_000000.bin").touch()
        self.stub("torchrun", 'shift 2; exec "$PY" "$@"\n')
        (self.repo / "data/cached_fineweb10B.py").write_text("print('have them')\n")
        rc, out = self.run_session("t2", "bundle", T2_MIN_RAM_GIB="0", T2_MIN_SHM_GIB="0", T2_MIN_CPU_PER_GPU="0")
        self.assertEqual(rc, 1, out)
        self.assertIn("printed no final result", out)
        self.assertTrue(self.status("t2").startswith("FAIL"))
        self.assertIn("== t2: FAIL", (self.sd / "summary.txt").read_text())

    def test_skipped_t2_prints_no_result_line(self):
        rc, out = self.run_session("t2", "bundle", T2_MIN_RAM_GIB="100000")
        self.assertEqual(rc, 0, out)
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("t2: SKIP host cannot run t2", summary)
        self.assertNotIn("== t2: no result", summary)


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
        # mean 0.4 mnat < HADAMARD_MIN_GAP_MNAT (2.0): no Hadamard variant can pay for itself, skip it.
        self.assertFalse(g["run_hadamard"])
        big = {k: [dict(r) for r in v] for k, v in groups.items()}
        for rep, dd in enumerate([3.0, 1.0, 4.0, 2.4]):
            big["fp8"][rep]["val_loss"] = 3.278 + dd / 1000
        gb = ss.gate_decision(dict(ab, groups=big))
        self.assertGreaterEqual(gb["mean_mnat"], 2.0)
        self.assertTrue(gb["run_hadamard"])
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
            g = ss.gate_decision(ab)
            self.assertEqual((g["decision"], g["run_hadamard"]), ("INCOMPLETE", False))
            self.assertIn("GATE: INCOMPLETE", "\n".join(ss.gate_lines(g)))

    @staticmethod
    def gate_ab(diffs, pending=0, bf16_status="ok"):
        groups = {"fp8": [], "bf16": []}
        for rep, dd in enumerate(diffs):
            groups["fp8"].append(dict(idx=2 * rep, rep=rep, status="ok", val_loss=3.278 + dd / 1000, details={"step_sum_s": 40.0}))
            groups["bf16"].append(dict(idx=2 * rep + 1, rep=rep, status=bf16_status, val_loss=3.278 if bf16_status == "ok" else None,
                                       details={"step_sum_s": 50.0}))
        return dict(arms=["fp8", "bf16"], baseline="fp8", env={"fp8": {}, "bf16": {"FP8_OFF": "1"}}, groups=groups,
                    plan={"n": len(diffs) + pending, "seeds": None, "cache": "warm", "order": "blocked"}, pending=pending,
                    planned={"fp8": 4, "bf16": 4})

    def test_partial_gate_is_no_decision_and_never_runs_hadamard(self):
        """3 of 4 pairs with a big gap, one slot pending (the gate phase FAILed): not GAP, no hadamard."""
        g = ss.gate_decision(self.gate_ab([3.0, 3.2, 2.9], pending=2))
        self.assertEqual(g["decision"], "INCOMPLETE")
        self.assertFalse(g["run_hadamard"])
        text = "\n".join(ss.gate_lines(g))
        self.assertNotIn("-> GAP", text)
        self.assertIn("provisional, not a decision", text)
        self.assertTrue(ss.gate_decision(self.gate_ab([3.0, 3.2, 2.9, 3.1]))["run_hadamard"])

    def test_gate_table_has_one_sign_convention(self):
        ab = self.gate_ab([1.0, 0.2, 0.6, -0.2])
        text = "\n".join(ss.ab_table("gate", ab, gate=True) + ss.gate_lines(ss.gate_decision(ab)))
        self.assertNotIn("Δval", text)            # the gap appears once, as val(fp8) - val(bf16)
        self.assertNotIn("net ", text)
        self.assertIn("gap val(fp8) - val(bf16) = +0.40", text)
        self.assertIn("+ = fp8 worse", text)
        self.assertIn("Δstep_sum +10.000", text)

    def test_inconclusive_states_what_is_supported(self):
        g = ss.gate_decision(self.gate_ab([1.0, 0.2, 0.6, -0.2]))
        self.assertEqual(g["decision"], "INCONCLUSIVE")
        self.assertNotIn("more seeds needed", g["reason"])
        self.assertRegex(g["reason"], r"paired sd 0\.52 mnat\); .*n~\d+ pairs")
        text = "\n".join(ss.gate_lines(g))
        self.assertIn("loose prescreen for the default HADAMARD_VARIANTS", text)
        self.assertIn("does not exclude cheaper variants", text)
        # n~ is the smallest n whose CI clears 0.5 at the observed mean and sd
        n = ss.pairs_needed(0.4, 0.5164)
        self.assertLess(stats.t_ppf(0.975, n - 1) * 0.5164 / math.sqrt(n), 0.1)
        self.assertGreaterEqual(stats.t_ppf(0.975, n - 2) * 0.5164 / math.sqrt(n - 1), 0.1)
        self.assertIsNone(ss.pairs_needed(0.5, 1.0))

    def test_dead_arm_is_code_5_not_replan(self):
        ab = self.gate_ab([0.0] * 4, bf16_status="failed")
        code, why = ss.ab_check(ab, gate=True)
        self.assertEqual(code, 5, why)
        self.assertIn("bf16", why)
        self.assertEqual(ss.ab_check(dict(ab, pending=2), gate=True)[0], 3)

    def test_hadamard_buyback_line(self):
        ab = self.gate_ab([0.0, 0.0])
        ab["groups"]["h"] = [dict(idx=10 + r, rep=r, status="ok", val_loss=3.278, details={"step_sum_s": 40.6}) for r in range(2)]
        ab["groups"]["bf16"] = []
        ab.update(arms=["fp8", "h"], env={"fp8": {}, "h": {"HADAMARD_FP8": "1"}}, planned={"fp8": 2, "h": 2})
        text = "\n".join(ss.ab_table("hadamard", ab, buyback_mnat=3.0))
        self.assertIn("buy-back bound: gate gap 3.00 mnat x 0.164 s/mnat (upstream's convention; measured 120-250 ms/mnat) = 0.49 s", text)
        self.assertIn("cannot pay even if it recovers the whole gap", ss.ab_table("hadamard", ab, buyback_mnat=2.0)[-1])
        self.assertIn("could pay", "\n".join(ss.ab_table("hadamard", ab, buyback_mnat=10.0)))

    def test_failed_run_causes_are_listed(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "stdout"))
            Path(d, "stdout", "001_bf16.out").write_text("logs/x.txt\nTraceback (most recent call last):\nImportError: libcudart.so.13: cannot open\n")
            ab = self.gate_ab([0.0, 0.0], bf16_status="failed")
            ab["dir"] = d
            for r in ab["groups"]["bf16"]:
                r.update(stdout="/gpu/box/001_bf16.out", returncode=1)
            text = "\n".join(ss.ab_table("gate", ab, gate=True))
        self.assertIn("bf16 #1 failed rc=1, #3 failed rc=1: ImportError: libcudart.so.13: cannot open", text)

    def test_stack_header_marks_mismatches(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "env"))
            Path(d, "env/stack.json").write_text(json.dumps(dict(
                git_sha="abc123def4567890", git_dirty=True, dirty_files=["train_gpt.py"], torch="2.9.1+cu126",
                torch_cuda="12.6", triton="3.6.0", driver="575.57.08", python="3.12.3", fa3="ok", problems=[])))
            Path(d, "env/node_check.txt").write_text("gpu 3 H100: bf16 GEMM 600 TFLOP/s  (x)  <-- SLOW\n")
            text = "\n".join(ss.stack_lines(d))
        self.assertIn("STACK: git abc123def456 DIRTY (1 modified tracked file(s)) | torch 2.9.1+cu126", text)
        self.assertIn("!! torch 2.9.1+cu126 != 2.10.0+cu128", text)
        self.assertNotIn("!! triton", text)
        self.assertIn("!! driver 575.57.08 < 580", text)
        self.assertIn("!! node_check: gpu 3 H100", text)
        self.assertIn("train_gpt.py", text)

    def test_tests_tail_names_the_failing_test(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "tests"))
            Path(d, "tests/pytest.txt").write_text("..F.s\nFAILED 182/hadamard-fp8/tests/test_x.py::test_y - AssertionError\n"
                                                   "SKIPPED [1] test_z.py:3: needs H100\n1 failed, 3 passed, 1 skipped in 2.0s\n")
            Path(d, "tests/rc_plain").write_text("1\n")
            text = "\n".join(ss.tests_lines(d))
        self.assertIn("== tests (plain): rc=1  1 failed, 3 passed, 1 skipped in 2.0s", text)
        self.assertIn("FAILED 182/hadamard-fp8/tests/test_x.py::test_y", text)
        self.assertIn("SKIPPED [1] test_z.py:3: needs H100", text)

    def test_truncated_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d, "session-x")
            (sd / "status").mkdir(parents=True)
            (sd / "status/env").write_text("PASS\n")
            (sd / "meta.txt").write_text("session_date 2026-11-10\n")
            (sd / "big.bin").write_bytes(os.urandom(300_000))
            tgz = Path(d, "b.tar.gz")
            with tarfile.open(tgz, "w:gz") as tf:
                tf.add(sd / "meta.txt", "session-x/meta.txt")
                tf.add(sd / "status", "session-x/status")
                tf.add(sd / "big.bin", "session-x/big.bin")
            data = tgz.read_bytes()
            cut = Path(d, "cut.tar.gz")
            cut.write_bytes(data[: len(data) // 2])
            r = subprocess.run([sys.executable, str(Path(TOOLS) / "summarize_session.py"), str(cut)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("WARNING:", r.stdout)
            self.assertIn("truncated", r.stdout)
            self.assertIn("env: PASS", r.stdout)
            Path(d, "junk.tar.gz").write_bytes(data[:20])
            r = subprocess.run([sys.executable, str(Path(TOOLS) / "summarize_session.py"), str(Path(d, "junk.tar.gz"))],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("probably truncated", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

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
