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

from _paths import FIXTURES, REPO, TOOLS
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
if os.environ.get("FP8_ATTN_O") == "1":
    val += float(os.environ.get("FAKE_ATTNO_DVAL", "0"))
    t -= 0.5
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
        for f in ("ab_run.py", "logparse.py", "stats.py", "summarize_session.py", "node_setup.sh", "xab_run.py",
                  "nvls_check.py"):
            shutil.copy(Path(TOOLS) / f, self.repo / "182/tools" / f)
        shutil.copytree(Path(REPO) / "182/profile", self.repo / "182/profile",
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))
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
        self.state = self.root / "fakestate"
        self.env.update(FAKE_STATE=str(self.state), PREP_POLL_S="1")
        for k in list(self.env):
            if k in ("DRY_RUN", "FORCE", "FORCE_HADAMARD", "FORCE_CHECK", "GATE_SEEDS", "ATTNO_SEEDS", "HADAMARD_VARIANTS",
                     "HADAMARD_SEEDS", "SESSION_DOCKER", "QUICK", "BASE_RUNS", "PREP_BACKGROUND", "ALLOW_STACK") \
                    or k.startswith(("T1GPU_", "T1CPU_", "T2CPU_", "T2GPU_")):
                self.env.pop(k)

    VARIANTS = dict(t1gpu="t1-gpu-stack", t1cpu="t1-cpu-ngram", t2cpu="t2-cpu-ngram", t2gpu="t2-gpu-port")

    def add_variants(self, *phases):
        """The fake node script as variants/<name>/182/run_node.sh (untracked, like a fresh checkout's would be tracked)."""
        for ph in phases or self.VARIANTS:
            d = self.repo / "variants" / self.VARIANTS[ph] / "182"
            d.mkdir(parents=True)
            shutil.copy(Path(FIXTURES) / "fake_run_node.sh", d / "run_node.sh")

    def calls(self, phase):
        f = self.state / self.VARIANTS[phase] / "calls"
        return f.read_text().split("\n")[:-1] if f.exists() else []

    def result(self, phase, name):
        return json.loads((self.sd / phase / "runs" / name).read_text())

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
        phases = ("env", "tests", "sanity", "base", "t1gpu", "profile", "nccl", "xab", "t1cpu", "t2cpu", "t2gpu", "gate",
                  "attno", "hadamard", "bundle")
        pos = [out.index(f"== {p}: est. ") for p in phases]
        self.assertEqual(pos, sorted(pos))
        self.assertIn("phases: " + " ".join(phases), out)
        # the estimates come first, one per phase, with a total: base 9+4, t1gpu 9+3*4, t2cpu 20, gate 2*9+2*3*4
        head = out[:out.index("== env: est. ")]
        self.assertRegex(head, r"\n  base +13\n")
        self.assertRegex(head, r"\n  t1gpu +21  \(4 run\(s\) of variants/t1-gpu-stack; skipped if its check FAILs\)")
        self.assertRegex(head, r"\n  t2cpu +20 ")
        self.assertRegex(head, r"\n  gate +42\n")
        self.assertRegex(head, r"\n  total +\d+ node-min upper bound")
        self.assertRegex(out, r"\+ .*ab_run\.py run .*/base --no-report")
        self.assertIn("SKIP t1gpu: variants/t1-gpu-stack/182/run_node.sh is not in this checkout", out)
        self.assertIn("SKIP attno: track_1_short/config.py at HEAD does not read FP8_ATTN_O", out)
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
        self.assertIn("phases: env sanity base t1gpu profile bundle\n", r.stdout)
        self.assertNotIn("== tests:", r.stdout)
        self.assertNotIn("== gate:", r.stdout)
        self.assertRegex(r.stdout, r"\n  t1gpu +13  \(2 run")      # QUICK: 2 runs of t1gpu unless T1GPU_RUNS is set
        total = int(__import__("re").search(r"total +(\d+) node-min", r.stdout).group(1))
        self.assertLess(total, 45)
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh")], cwd=self.repo,
                           env=dict(env, DRY_RUN="1", QUICK="1", T1GPU_RUNS="3"), capture_output=True, text=True, timeout=600)
        self.assertRegex(r.stdout, r"\n  t1gpu +17  \(3 run")

    def test_hung_command_is_timed_out(self):
        rc, out = self.run_session("sanity", SANITY_CMD="sleep 60", SANITY_TIMEOUT_S="1")
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.status("sanity"), "FAIL rc=124")

    def test_base_runs_the_record_and_the_ab_phases_reuse_its_worktree(self):
        rc, out = self.run_session("base", "gate", "bundle", FAKE_GAP="0", BASE_RUNS="3")
        self.assertEqual(rc, 0, out)
        ends = self.ends("base")
        self.assertEqual([(e["arm"], e["status"], e["seed"]) for e in ends], [("fp8", "ok", None)] * 3)   # unseeded, as certifications are
        self.assertEqual(sorted(p.name.split("-")[0] for p in (self.sd / "work/wt").iterdir()), ["bf16", "fp8"])
        summary = (self.sd / "summary.txt").read_text()
        self.assertRegex(summary, r"base: root record #92 ANVIL2 +\| 3/3 ok +\| 3\.27\d\d ± 0\.00\d\d +\| 40\.20 ± 0\.00 +\| 39\.9 s \(official\) +\| 39\.9 s, <= 3\.28 +\| \d/3 runs")
        self.assertIn("== base: baseline=fp8 n/arm=3 seeds=none", summary)
        self.assertRegex(summary, r"t1gpu: t1-gpu-stack \(PR #397\) +\| not run +\| - ")
        rc, out = self.run_session("base", BASE_RUNS="3")
        self.assertIn("== base: already PASS", out)

    def test_attno_skips_until_the_flag_is_committed_then_reports_net_seconds(self):
        rc, out = self.run_session("attno")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.status("attno").startswith("SKIP track_1_short/config.py at HEAD does not read FP8_ATTN_O"), self.status("attno"))
        self.assertFalse((self.sd / "attno/plan.json").exists())
        # in the working tree only: still skipped (the arms run the committed HEAD)
        (self.repo / "track_1_short/config.py").write_text("FP8_ATTN_O = 0\n")
        rc, out = self.run_session("attno")
        self.assertTrue(self.status("attno").startswith("SKIP"), self.status("attno"))
        self.git("commit", "-qam", "attn_o flag")
        rc, out = self.run_session("attno", "bundle", FAKE_ATTNO_DVAL="0.001")   # no gate needed
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.status("attno"), "PASS")
        plan = json.loads((self.sd / "attno/plan.json").read_text())
        self.assertEqual({a["name"]: a["env"] for a in plan["arms"]}, {"fp8": {}, "attn_o": {"FP8_ATTN_O": "1"}})
        self.assertEqual({(e["arm"], e["seed"]) for e in self.ends("attno")}, {(a, s) for a in ("fp8", "attn_o") for s in (1, 2, 3, 4)})
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("ATTN_O (attn_o - fp8, n=4 seed pairs; + = FP8_ATTN_O=1 worse / slower):", summary)
        self.assertIn("val delta        +1.00 ± 0.00 millinats (SE), 95% CI [+1.00, +1.00]", summary)
        self.assertIn("train_time delta -0.500 ± 0.000 s (SE)", summary)
        # net = -0.500 + 1.00 * 0.164 = -0.336 s; band: -0.500 + 0.120 .. -0.500 + 0.250
        self.assertIn("164 ms per millinat = -0.336 s", summary)
        self.assertIn("120-250 ms per millinat band: -0.380 to -0.250 s", summary)
        self.assertIn("the flag pays on this node", summary)
        self.assertIn(summary[summary.index("ATTN_O (attn_o"):summary.index("-> the 95%")], out)


class TestSessionFlow(SessionRepo):
    """The session flow: docker env forwarding, read-only DRY_RUN, the root stack gate, timeouts, and the variant
    phases against fake run_node.sh scripts (check fail => skipped; prep; run failures; RESULT parsing; resume)."""

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
                           env=dict(env, DRY_RUN="1", SESSION_DOCKER="img", PY=sys.executable, T2CPU_RUNS="2",
                                    HADAMARD_MIN_GAP_MNAT="1.5"),
                           capture_output=True, text=True, timeout=60)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("--init", out)
        self.assertIn("TINI_KILL_PROCESS_GROUP=1", out)
        self.assertIn("T2CPU_RUNS=2", out)
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

    def test_tests_phase_without_pytest_says_why(self):
        bare = self.root / "barepy"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(bare)], check=True)
        rc, out = self.run_session("tests", "bundle", PY=str(bare / "bin/python"))
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL: pytest is not importable", out)
        self.assertIn("pytest is not importable", (self.sd / "summary.txt").read_text())

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

    def test_variant_runs_results_resume_and_summary(self):
        self.add_variants("t1gpu")
        fail = dict(FAKE_RUN_FAIL="t1-gpu-stack:2", T1GPU_RUNS="3", T1GPU_SEEDS="5,6,7", NPROC="4")
        rc, out = self.run_session("t1gpu", "bundle", **fail)
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.status("t1gpu"), "FAIL rc=1")
        self.assertIn("failed phases: t1gpu", out)
        self.assertEqual(self.calls("t1gpu")[:3], ["check", "prep", "check"])       # check, prep, then the check that decides
        logs = self.sd / "t1gpu/logs"
        self.assertEqual(self.calls("t1gpu")[3:], [f"run NPROC=4 SEED={s} LOG_DIR={logs}" for s in (5, 6, 7)])
        r1, r2, r3 = (self.result("t1gpu", f"0{i}.result") for i in (1, 2, 3))
        self.assertEqual((r1["ok"], r1["val_loss"], r1["train_time_s"], r1["steps"], r1["target"], r1["hit"], r1["seed"]),
                         (True, 3.271, 30.1, 1100, 3.28, 1, "5"))
        self.assertEqual((r2["ok"], r2["rc"], r2["reason"]), (False, 1, "CUDA out of memory on rank 3"))
        self.assertTrue(r3["ok"])
        self.assertIn("run 2/3 FAILED (rc 1): CUDA out of memory on rank 3", out)
        self.assertEqual(sorted(p.name for p in logs.iterdir()), ["run1.txt", "run3.txt"])
        summary = (self.sd / "summary.txt").read_text()
        self.assertRegex(summary, r"t1gpu: t1-gpu-stack \(PR #397\) +\| 2/3 ok +\| 3\.2720 ± 0\.0014 +\| 30\.20 ± 0\.14 +\| 30\.67 s, n=12 +\| 39\.9 s, <= 3\.28 +\| 2/2 runs")
        self.assertIn("== t1gpu: t1-gpu-stack (PR #397), 2/3 run(s) ok  [FAIL rc=1]", summary)
        self.assertIn("run 1: val_loss=3.2710 train_time_s=30.100 steps=1100 target=3.28 hit=1 seed=5", summary)
        self.assertIn("run 2: FAILED (rc 1): CUDA out of memory on rank 3", summary)
        # resume: only the failed run is retried (the 4th invocation of the fake), with its seed; the attempt is kept
        rc, out = self.run_session("t1gpu", "bundle", T1GPU_RUNS="3", T1GPU_SEEDS="5,6,7")
        self.assertEqual(rc, 0, out)
        self.assertIn("run 1/3: already done", out)
        self.assertIn("run 3/3: already done", out)
        self.assertEqual(self.status("t1gpu"), "PASS")
        self.assertEqual([c for c in self.calls("t1gpu") if c.startswith("run")][3:], [f"run NPROC=8 SEED=6 LOG_DIR={logs}"])
        self.assertEqual(self.result("t1gpu", "02.result")["val_loss"], 3.274)
        self.assertFalse(self.result("t1gpu", "02.fail1.result")["ok"])
        self.assertTrue((self.sd / "t1gpu/runs/02.fail1.out").exists())
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("3/3 ok", summary)
        self.assertIn("1 earlier failed attempt(s) were retried", summary)
        with tarfile.open((self.sd / "bundle_path").read_text().strip()) as tf:
            names = tf.getnames()
        self.assertTrue(any(n.endswith("t1gpu/logs/run4.txt") for n in names))
        self.assertTrue(any(n.endswith("t1gpu/check.txt") for n in names))
        # PASS: not re-run; FORCE=1 redoes it and keeps the old runs
        rc, out = self.run_session("t1gpu", T1GPU_RUNS="3")
        self.assertIn("== t1gpu: already PASS", out)
        rc, out = self.run_session("t1gpu", T1GPU_RUNS="1", FORCE="1")
        self.assertEqual(rc, 0, out)
        self.assertTrue((self.sd / "t1gpu/runs.attempt1/03.result").exists())
        self.assertEqual(sorted(p.name for p in (self.sd / "t1gpu/runs").iterdir()), ["01.out", "01.result"])

    def test_variant_check_fail_is_a_skip_and_later_phases_run(self):
        self.add_variants()
        rc, out = self.run_session("t1gpu", "t1cpu", "t2", "t2gpu", "bundle", FAKE_CHECK_FAIL="t1-cpu-ngram,t2-cpu-ngram",
                                   FAKE_CHECK_WARN="t2-gpu-port", T1GPU_RUNS="1", T1CPU_RUNS="1")
        self.assertEqual(rc, 0, out)                      # a skipped variant is not a session failure
        self.assertIn("phases: t1gpu t1cpu t2cpu t2gpu bundle", out)   # t2 is the old name of t2cpu
        want = "SKIP t1-cpu-ngram check: FAIL ram: 64 GiB available, needs 390 GiB (use a larger host) (FORCE_CHECK=1 runs anyway)"
        self.assertEqual(self.status("t1cpu"), want)
        self.assertTrue(self.status("t2cpu").startswith("SKIP t2-cpu-ngram check: FAIL ram"))
        self.assertEqual((self.status("t1gpu"), self.status("t2gpu")), ("PASS", "PASS"))
        self.assertNotIn("prep", self.calls("t1cpu"))     # a FAIL that prep cannot fix: no download for a host that cannot run it
        self.assertFalse([c for c in self.calls("t1cpu") if c.startswith("run")])
        self.assertIn("prep", self.calls("t2gpu"))
        self.assertTrue(self.calls("t2gpu")[-1].endswith(" CANON_MASK=1"), self.calls("t2gpu"))   # t2gpu's default run env
        self.assertFalse([c for c in self.calls("t1gpu") if "CANON" in c])
        summary = (self.sd / "summary.txt").read_text()
        self.assertRegex(summary, r"t1cpu: t1-cpu-ngram \(PR #381\) +\| SKIPPED +\| - ")
        self.assertRegex(summary, r"t2gpu: t2-gpu-port \(PR #261\) \[CANON_MASK=1\] +\| 1/1 ok \(ONE run\) +\| 2\.9171 +\| 860\.50 +\| 860\.5 s \(14\.34 min\) +\| 17\.35 min \(1041 s\), <= 2\.92 +\| 1/1 runs")
        self.assertIn("  t1cpu " + want, summary)
        self.assertIn("check: WARN vcpu: 12 per GPU, 24 recommended", summary)
        self.assertNotIn("== t1cpu:", summary)
        self.assertIn("earlier single runs on our node (n=1 each, logs not in the repo): root record 40.931 s, val_loss 3.2780", summary)
        self.assertIn("  run env: CANON_MASK=1", summary)
        # FORCE_CHECK=1 preps and runs it anyway, here with an extra run env
        rc, out = self.run_session("t1cpu", "t2gpu", FAKE_CHECK_FAIL="t1-cpu-ngram", T1CPU_RUNS="1", FORCE_CHECK="1", FORCE="1",
                                   T1CPU_RUN_ENV="CONFIG=380", T2GPU_RUN_ENV="")
        self.assertEqual(rc, 0, out)
        self.assertIn("prep", self.calls("t1cpu"))
        self.assertTrue(self.calls("t1cpu")[-1].endswith(" CONFIG=380"), self.calls("t1cpu"))
        self.assertNotIn("CANON", self.calls("t2gpu")[-1])
        self.assertEqual(self.result("t1cpu", "01.result")["run_env"], "CONFIG=380")
        self.assertIn("WARN: FORCE_CHECK=1, running t1-cpu-ngram despite: FAIL ram", out)
        self.assertEqual(self.status("t1cpu"), "PASS")

    def test_prep_fixes_a_check_that_names_prep_background_and_inline(self):
        self.add_variants("t1gpu", "t2gpu")
        need = dict(FAKE_NEEDS_PREP="t1-gpu-stack,t2-gpu-port", T1GPU_RUNS="1")
        rc, out = self.run_session("t1gpu", PREP_BACKGROUND="0", **need)
        self.assertEqual(rc, 0, out)
        self.assertIn("t1gpu: check + prep (inline", out)
        self.assertEqual(self.calls("t1gpu")[:3], ["check", "prep", "check"])
        self.assertNotIn("FAIL data", (self.sd / "t1gpu/check.txt").read_text())
        self.assertIn("FAIL data: 0 of 9 shards", (self.sd / "t1gpu/check_pre.txt").read_text())
        # background: one job preps both; the first variant phase waits for the whole job before it times anything
        rc, out = self.run_session("t1gpu", "t2gpu", FORCE="1", FAKE_PREP_SLEEP="2", **need)
        self.assertEqual(rc, 0, out)
        self.assertIn("prep: check + prep of t1gpu t2gpu started in the background", out)
        self.assertIn("t1gpu: waiting for the background prep job", out)
        self.assertEqual(self.calls("t2gpu")[:2], ["check", "prep"])
        self.assertEqual((self.status("t1gpu"), self.status("t2gpu")), ("PASS", "PASS"))
        self.assertFalse((self.sd / "prep_overlap.txt").exists())
        self.assertEqual((self.sd / "prep_chain.rc").read_text().strip(), "0")

    def test_prep_failure_fails_the_phase_with_its_tail(self):
        self.add_variants("t1gpu", "t2gpu")
        rc, out = self.run_session("t1gpu", "t2gpu", "bundle", FAKE_PREP_FAIL="t1-gpu-stack", T1GPU_RUNS="1")
        self.assertEqual(rc, 1, out)
        self.assertIn("prep` exited 5", out)
        self.assertIn("download failed: 401", out)
        self.assertEqual((self.status("t1gpu"), self.status("t2gpu")), ("FAIL rc=1", "PASS"))   # independent phases go on
        self.assertFalse(self.state.joinpath("t1-gpu-stack/runs").exists())
        self.assertIn("download failed: 401", (self.sd / "summary.txt").read_text())

    def test_two_consecutive_failed_runs_stop_a_variant(self):
        self.add_variants("t1gpu")
        rc, out = self.run_session("t1gpu", "bundle", FAKE_RUN_FAIL="t1-gpu-stack:all")
        self.assertEqual(rc, 1, out)
        self.assertIn("stopping t1gpu: 2 consecutive failed runs; 2 run(s) not attempted", out)
        self.assertEqual(self.state.joinpath("t1-gpu-stack/runs").read_text().strip(), "2")
        summary = (self.sd / "summary.txt").read_text()
        self.assertRegex(summary, r"t1gpu: t1-gpu-stack \(PR #397\) +\| 0/4 ok +\| - ")
        self.assertIn("2 planned run(s) never started", summary)

    def test_run_without_result_line_or_timed_out_is_failed(self):
        self.add_variants("t1gpu", "t1cpu")
        rc, out = self.run_session("t1gpu", "t1cpu", FAKE_RUN_NORESULT="t1-gpu-stack", T1GPU_RUNS="1",
                                   T1CPU_RUNS="1", FAKE_RUN_SLEEP="30", T1CPU_TIMEOUT_S="1", T1GPU_TIMEOUT_S="60")
        self.assertEqual(rc, 1, out)
        r = self.result("t1gpu", "01.result")
        self.assertEqual((r["ok"], r["rc"]), (False, 0))
        self.assertIn("no RESULT line; last output line: started, then nothing", r["reason"])
        r = self.result("t1cpu", "01.result")
        self.assertEqual((r["ok"], r["rc"]), (False, 124))
        self.assertTrue(r["reason"].startswith("timed out"), r["reason"])

    def test_broken_check_fails_and_missing_script_skips(self):
        self.add_variants("t1gpu")
        rc, out = self.run_session("t1gpu", "t2gpu", FAKE_CHECK_CRASH="t1-gpu-stack")
        self.assertEqual(rc, 1, out)
        self.assertIn("exited 2 without a FAIL line (a broken check, not a host verdict)", out)
        self.assertEqual(self.status("t1gpu"), "FAIL rc=1")
        self.assertTrue(self.status("t2gpu").startswith("SKIP variants/t2-gpu-port/182/run_node.sh is not in this checkout"))

    def test_wrong_root_stack_skips_root_phases_but_variants_run(self):
        self.add_variants("t1gpu")
        env = self.fake_stack(torch="2.9.1+cu126")
        rc, out = self.run_session("env", "base", "t1gpu", "attno", "bundle", T1GPU_RUNS="1", **env)
        self.assertEqual(rc, 1, out)
        self.assertTrue(self.status("env").startswith("FAIL"))
        self.assertTrue(self.status("base").startswith("SKIP env failed"), self.status("base"))
        self.assertTrue(self.status("attno").startswith("SKIP env failed"), self.status("attno"))
        self.assertEqual(self.status("t1gpu"), "PASS")
        self.assertIn("The variant phases still run", out)
        self.assertEqual(out[out.index("failed phases:"):].strip(), "failed phases: env")

    def test_dry_run_with_variants_touches_nothing(self):
        self.add_variants()
        self.git("add", "-A")
        self.git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "variants")
        env = {k: v for k, v in self.env.items() if k != "SESSION_DIR"}
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh")], cwd=self.repo,
                           env=dict(env, DRY_RUN="1", FAKE_CHECK_FAIL="t2-cpu-ngram"), capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("+ (background, nice 19) check + prep of: t1gpu t1cpu t2cpu t2gpu", out)
        self.assertIn("+ bash variants/t1-gpu-stack/182/run_node.sh prep", out)
        self.assertEqual(out.count("DATA_PATH=../.. torchrun --standalone --nproc_per_node=8 train_gpt.py"), 4 + 4 + 1 + 1 + 1)  # + profile t1gpu
        self.assertEqual(out.count("train_gpt.py [CANON_MASK=1]"), 1)
        self.assertIn("    FAIL ram: 64 GiB available", out)          # the real check output: what this host lacks
        st = subprocess.run(["git", "-C", str(self.repo), "status", "--porcelain", "--ignored"], capture_output=True, text=True).stdout
        self.assertEqual(st, "")
        self.assertFalse(list(self.state.glob("*/runs")), "DRY_RUN ran a variant")
        self.assertFalse(list(self.state.glob("*/prepped")), "DRY_RUN ran a prep")
        self.assertFalse(self.marker.exists())



class TestProfileNcclXab(SessionRepo):
    """The profile, nccl and xab phases against the fake run_node.sh (FAKE_SEEDED=1: val/time from the seed plus the
    arm's EXTRA_ENV shifts) and the fake root trainer."""

    def experiment(self, name, body):
        d = self.repo / "182/experiments.d"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.sh").write_text(body)

    def test_xab_runs_every_experiment_paired_resumes_and_summarizes(self):
        self.add_variants("t1gpu")
        self.experiment("moe", 'XAB_NAME=moe\nXAB_TREE=variants/t1-gpu-stack\n'
                               'XAB_ARMS=("base:" "cap:FAKE_DVAL=0.001 FAKE_DVAL_SLOPE=0.0002 FAKE_DT=-0.25")\n'
                               'XAB_SEEDS=4 ; XAB_EST_MIN=20 ; XAB_NOTE="expect -0.25 s; killed if Δval CI > 1 mnat"\n')
        self.experiment("rootx", 'XAB_NAME=rootx\nXAB_TREE=.\nXAB_ARMS=("base:" "ao:FP8_ATTN_O=1 FAKE_ATTNO_DVAL=0.001")\n'
                                 'XAB_SEEDS=2 ; XAB_EST_MIN=15 ; XAB_NOTE="root arm"\n')
        self.experiment("broken", 'XAB_NAME=broken\nXAB_TREE=variants/nope\nXAB_ARMS=("base:" "x:A=1")\n')
        rc, out = self.run_session("xab", "bundle", FAKE_SEEDED="1")
        self.assertEqual(rc, 1, out)                       # the invalid file fails the phase; the other two ran
        self.assertEqual(self.status("xab"), "FAIL rc=1")
        self.assertIn("xab broken: INVALID", out)
        self.assertIn("XAB_TREE=variants/nope: no such directory", (self.sd / "xab/broken.invalid").read_text())
        calls = [c for c in self.calls("t1gpu") if c.startswith("run ")]
        self.assertEqual(len(calls), 8)
        self.assertEqual(sum("EXTRA_ENV=FAKE_DVAL=0.001" in c for c in calls), 4)
        self.assertEqual((self.sd / "xab/moe/status.txt").read_text().strip(), "PASS")
        rep = (self.sd / "xab/moe/report.txt").read_text()
        self.assertIn("Δval  +1.50 ± 0.13 mnat (SE), 95% CI [+1.09, +1.91]", rep)
        self.assertIn("net_s = Δtime + Δval x 164 ms/mnat = -0.004 s", rep)
        # the root experiment went through ab_run.py: same seeds in both arms, the fake trainer's -0.5 s for FP8_ATTN_O
        ends = self.ends("xab/rootx")
        self.assertEqual({(e["arm"], e["seed"]) for e in ends}, {(a, s) for a in ("base", "ao") for s in (1, 2)})
        rrep = (self.sd / "xab/rootx/report.txt").read_text()
        self.assertIn("ao - base (n=2 seed pairs", rrep)
        self.assertIn("Δval  +1.00 ± 0.00 mnat", rrep)
        self.assertIn("Δtime -0.500 ± 0.000 s", rrep)
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("== xab moe: PASS  (expect -0.25 s; killed if Δval CI > 1 mnat)", summary)
        self.assertIn("== xab broken: INVALID experiments.d file", summary)
        self.assertIn("ao - base (n=2 seed pairs", summary)
        # resume: complete experiments are not re-run; XAB_SKIP drops one by name
        rc, out = self.run_session("xab", FAKE_SEEDED="1", XAB_SKIP="broken")
        self.assertIn("xab moe: complete earlier", out)
        self.assertEqual(len([c for c in self.calls("t1gpu") if c.startswith("run ")]), 8)
        (self.repo / "182/experiments.d/broken.sh").unlink()
        rc, out = self.run_session("xab", FAKE_SEEDED="1", XAB_ONLY="rootx")
        self.assertEqual(rc, 0, out)
        self.assertIn("xab moe: not selected", out)
        self.assertEqual(self.status("xab"), "PASS")

    def test_xab_skips_a_variant_whose_check_fails_and_dry_run_plans(self):
        self.add_variants("t1gpu")
        self.experiment("moe", 'XAB_NAME=moe\nXAB_TREE=variants/t1-gpu-stack\nXAB_ARMS=("base:" "cap:MOE_CAP=1")\n'
                               'XAB_SEEDS=2 ; XAB_EST_MIN=17 ; XAB_NOTE="n"\n')
        rc, out = self.run_session("xab", DRY_RUN="1")
        self.assertEqual(rc, 0, out)
        self.assertRegex(out, r"\n  xab +17  \(1 file\(s\) in 182/experiments\.d")
        self.assertIn("EXTRA_ENV=MOE_CAP=1 bash variants/t1-gpu-stack/182/run_node.sh run   (cap)", out)
        self.assertEqual(self.calls("t1gpu"), [])
        rc, out = self.run_session("xab", FAKE_CHECK_FAIL="t1-gpu-stack")
        self.assertEqual(rc, 0, out)
        self.assertTrue((self.sd / "xab/moe/status.txt").read_text().startswith("SKIP check: FAIL ram"))
        self.assertEqual([c for c in self.calls("t1gpu") if c.startswith("run ")], [])

    def test_nccl_skips_without_nvls_and_runs_when_forced(self):
        self.add_variants("t1gpu")
        rc, out = self.run_session("nccl", NCCL_TREES="t1gpu", NCCL_PROBE_TIMEOUT_S="20")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.status("nccl").startswith("SKIP NVLS: unavailable -- the forced-NVLS probe failed"), self.status("nccl"))
        self.assertTrue((self.sd / "nccl/nvls_check.json").exists())
        rc, out = self.run_session("nccl", "bundle", FORCE="1", FORCE_NCCL="1", NCCL_TREES="t1gpu", NCCL_SEEDS="1,2",
                                   NCCL_PROBE_TIMEOUT_S="20", FAKE_SEEDED="1")
        self.assertEqual(rc, 0, out)
        calls = [c for c in self.calls("t1gpu") if c.startswith("run ")]
        self.assertEqual(len(calls), 6)
        self.assertEqual(sum(c.endswith("EXTRA_ENV=NCCL_NVLS_ENABLE=0") for c in calls), 2)
        self.assertEqual(sum("'NCCL_ALGO=allreduce:nvls;reducescatter:nvls;allgather:nvls'" in c for c in calls), 2)
        rep = (self.sd / "nccl/t1gpu/report.txt").read_text()
        self.assertIn("nvls_off - base (n=2 seed pairs", rep)
        self.assertIn("nvls_force - base (n=2 seed pairs", rep)
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("== nccl (NVLink SHARP):", summary)
        self.assertIn("NVLS: unavailable", summary)
        self.assertIn("== nccl t1gpu: tree variants/t1-gpu-stack, baseline base, seeds 1,2", summary)

    def test_profile_fails_loudly_when_the_trainer_bypasses_the_hook_and_skips_unready_trees(self):
        self.add_variants("t1gpu")
        # the fake run_node.sh never calls torchrun: nothing can be profiled, and the phase says so
        rc, out = self.run_session("profile", "bundle", PROFILE_TREES="t1gpu")
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.status("profile"), "FAIL rc=1")
        self.assertIn("did not start through 182/profile/shim/torchrun", (self.sd / "profile/t1gpu/run.txt").read_text())
        self.assertIn("== profile t1gpu: no analysis", (self.sd / "summary.txt").read_text())
        rc, out = self.run_session("profile", FORCE="1", PROFILE_TREES="t1gpu", FAKE_CHECK_FAIL="t1-gpu-stack")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.status("profile").startswith("SKIP t1gpu: check: FAIL ram"), self.status("profile"))
        self.assertTrue(list(self.sd.glob("profile/t1gpu.attempt1/run.txt")))      # the failed attempt is kept

    def test_profile_done_earlier_is_kept_and_summarized(self):
        d = self.sd / "profile/root"
        d.mkdir(parents=True)
        rows = "\n".join(f"{1.0 / (i + 1):8.3f} {10.0:6.1f}% {1.0:6.1f} {100.0:8.1f}  gemm       kernel_{i}" for i in range(25))
        (d / "analysis.txt").write_text("steps profiled: 20 (device 0); GPU window 30.00 ms/step\n\n"
                                        "NCCL: 2.000 ms/step of kernels, 0.500 ms exposed\n\ntop 25 kernels by total time:\n"
                                        f"{'ms/step':>8} header\n{rows}\n\nlargest idle gaps:\n  12.0 us in step 300\n")
        rc, out = self.run_session("profile", "bundle", PROFILE_TREES="root")
        self.assertEqual(rc, 0, out)
        self.assertIn("profile root: done earlier", out)
        summary = (self.sd / "summary.txt").read_text()
        self.assertIn("== profile root (torch.profiler; not a timing run", summary)
        self.assertIn("kernel_11", summary)
        self.assertNotIn("kernel_12", summary)              # the summary keeps the top 12; analysis.txt has all 25
        self.assertIn("12.0 us in step 300", summary)

    def test_skip_phases_and_quick(self):
        env = {k: v for k, v in self.env.items() if k != "SESSION_DIR"}
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh")], cwd=self.repo,
                           env=dict(env, DRY_RUN="1", SKIP_PHASES="nccl xab t1cpu"), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("phases: env tests sanity base t1gpu profile t2cpu t2gpu gate attno hadamard bundle\n", r.stdout)
        self.assertRegex(r.stdout, r"profile: \S+ steps 300:20 -> \S+/profile/root")   # run.sh's DRY_RUN plan (t1gpu absent)
        self.assertRegex(r.stdout, r"\+ cd \S+ && PROFILE_STEPS=300:20 .*torchrun --standalone --nproc_per_node=8 \S+/hook\.py train_gpt\.py")
        r = subprocess.run(["bash", str(self.repo / "182/gpu_session.sh"), "gate"], cwd=self.repo,
                           env=dict(env, DRY_RUN="1", SKIP_PHASES="bogus"), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown phase 'bogus'", r.stderr)


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
                torch_cuda="12.6", triton="3.6.0", driver="575.57.08", python="3.12.3", fa3="ok")))
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

    def test_parse_result_contract(self):
        ok = "noise\nRESULT variant=t1-gpu-stack val_loss=3.27743 train_time_s=30.67 steps=1100 target=3.28 hit=1 log=logs/a.txt\n"
        r = ss.parse_run(ok, 0, 77, "3", "t1-gpu-stack")
        self.assertEqual((r["ok"], r["val_loss"], r["train_time_s"], r["steps"], r["hit"], r["log"], r["wall_s"], r["seed"]),
                         (True, 3.27743, 30.67, 1100, 1, "logs/a.txt", 77, "3"))
        self.assertNotIn("note", r)
        # the LAST RESULT line counts; a failed line carries its reason whole
        r = ss.parse_run(ok + "RESULT variant=t1-gpu-stack status=failed reason=val loss is nan at step 900\n", 1)
        self.assertEqual((r["ok"], r["reason"]), (False, "val loss is nan at step 900"))
        self.assertFalse(ss.parse_run(ok, 3)["ok"])                                  # a result with a non-zero exit is not a result
        self.assertIn("exit code 3", ss.parse_run(ok, 3)["reason"])
        for bad in ("RESULT variant=x val_loss=nan train_time_s=30 steps=1 target=3.28 hit=0 log=l",
                    "RESULT variant=x val_loss=3.27 steps=1 target=3.28 hit=1 log=l",
                    "RESULT variant=x val_loss=3.27 train_time_s=fast steps=1 target=3.28 hit=1 log=l",
                    "step:1100/1100 val_loss:3.2774 train_time:30670ms", ""):
            r = ss.parse_run(bad, 0)
            self.assertFalse(r["ok"], bad)
            self.assertTrue(r["reason"], bad)
        r = ss.parse_run(ok.replace("hit=1", "hit=0"), 0, variant="t2-gpu-port")
        self.assertTrue(r["ok"])
        self.assertIn("expected t2-gpu-port", r["note"])
        self.assertIn("hit=0 disagrees", ss.parse_result(ok.replace("hit=1", "hit=0"))["note"])

    def test_methods_table_is_empty_for_a_session_without_methods_and_honest_about_one_run(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ss.methods_table(d, {"env": "PASS"}), [])
            os.makedirs(os.path.join(d, "t2cpu/runs"))
            Path(d, "t2cpu/runs/01.result").write_text(json.dumps(ss.parse_run(
                "RESULT variant=t2-cpu-ngram val_loss=2.9212 train_time_s=230.4 steps=5800 target=2.92 hit=0 log=l", 0)))
            Path(d, "t2cpu/runs/02.result").write_text('{"ok": tr')            # torn write
            Path(d, "t2cpu/planned").write_text("2\n")
            text = "\n".join(ss.methods_table(d, {"t2cpu": "FAIL rc=1", "t1cpu": "SKIP t1-cpu-ngram check: FAIL ram: 64 GiB"}))
            self.assertRegex(text, r"t2cpu: t2-cpu-ngram +\| 1/2 ok \(ONE run\) +\| 2\.9212 +\| 230\.40 +\| 221 s, 1 run +\| 17\.35 min \(1041 s\), <= 2\.92 +\| 0/1 runs")
            self.assertRegex(text, r"t1cpu: t1-cpu-ngram \(PR #381\) +\| SKIPPED ")
            self.assertRegex(text, r"base: root record #92 ANVIL2 +\| not run ")
            self.assertIn("not measured by us", text)
            self.assertIn("One run is one run", text)
            self.assertIn("t1cpu SKIP t1-cpu-ngram check: FAIL ram: 64 GiB", text)
            lines = "\n".join(ss.variant_lines(d, "t2cpu"))
            self.assertIn("run 2: FAILED (rc ?): unreadable result file", lines)

    def test_attno_net_matches_hand_computation(self):
        dv, dt = [0.4, -0.2, 0.9, 0.1], [-0.30, -0.10, -0.25, -0.15]     # attn_o - fp8: millinats, seconds
        groups = {"fp8": [], "attn_o": []}
        for rep, (v, t) in enumerate(zip(dv, dt)):
            groups["fp8"].append(dict(idx=2 * rep, rep=rep, status="ok", val_loss=3.278, train_time_s=40.0, details={}))
            groups["attn_o"].append(dict(idx=2 * rep + 1, rep=rep, status="ok", val_loss=3.278 + v / 1000, train_time_s=40.0 + t, details={}))
        ab = dict(arms=["fp8", "attn_o"], baseline="fp8", groups=groups, env={"fp8": {}, "attn_o": {"FP8_ATTN_O": "1"}})
        text = "\n".join(ss.attno_lines(ab))
        net = [t + v * 0.164 for v, t in zip(dv, dt)]
        m, sd = stats.mean_sd(net)
        h = 3.182446305284263 * sd / 2
        self.assertIn(f"= {m:+.3f} s, 95% CI [{m - h:+.3f}, {m + h:+.3f}]", text)
        mv, sdv = stats.mean_sd(dv)
        self.assertIn(f"val delta        {mv:+.2f} ± {sdv / 2:.2f} millinats (SE), 95% CI [{mv - 3.182446305284263 * sdv / 2:+.2f}, {mv + 3.182446305284263 * sdv / 2:+.2f}]", text)
        lo, hi = (sum(dt) / 4 + mv * p for p in (0.120, 0.250))
        self.assertIn(f"band: {lo:+.3f} to {hi:+.3f} s", text)
        self.assertIn("the flag pays on this node", text)                  # CI of the net is below 0
        # one usable pair: said so, no CI
        groups["attn_o"] = groups["attn_o"][:1]
        one = "\n".join(ss.attno_lines(ab))
        self.assertIn("n=1 seed pair: ONE pair, no confidence interval, not a result", one)
        groups["attn_o"][0]["status"] = "failed"
        self.assertIn("no complete seed pair", "\n".join(ss.attno_lines(ab)))

if __name__ == "__main__":
    unittest.main()
