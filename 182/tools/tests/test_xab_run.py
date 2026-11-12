"""xab_run.py (the session's xab and nccl phases) and nvls_check.py on CPU: experiments.d parsing and validation, paired
interleaved runs through a fake variants/<name>/182/run_node.sh (tests/fixtures/fake_run_node.sh, FAKE_SEEDED=1:
val/time from the seed plus the arm's EXTRA_ENV shifts), hand-computed paired statistics, resume, dead arms,
timeouts; nvls_check's parsing, verdicts and probe handling against stub nvidia-smi / torchrun."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from _paths import FIXTURES, TOOLS
import nvls_check
import xab_run

XAB = str(Path(TOOLS) / "xab_run.py")
ON = "on:FAKE_DVAL=0.001 FAKE_DVAL_SLOPE=0.0002 FAKE_DT=-0.25"


class XabRepo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        d = self.repo / "variants/t1-gpu-stack/182"
        d.mkdir(parents=True)
        shutil.copy(Path(FIXTURES) / "fake_run_node.sh", d / "run_node.sh")
        (self.repo / "train_gpt.py").write_text("# root trainer\n")
        (self.repo / "182/experiments.d").mkdir(parents=True)
        self.state = self.root / "state"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("FAKE_", "EXTRA_ENV"))}
        self.env.update(FAKE_STATE=str(self.state), FAKE_SEEDED="1")

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, body, name="exp.sh"):
        p = self.repo / "182/experiments.d" / name
        p.write_text(body)
        return xab_run.load_spec(str(p), str(self.repo))

    def xab(self, *args, **env):
        r = subprocess.run([sys.executable, XAB, *args], cwd=self.repo, env=dict(self.env, **env), capture_output=True,
                           text=True, timeout=300)
        return r.returncode, r.stdout + r.stderr

    def calls(self):
        f = self.state / "t1-gpu-stack/calls"
        return [c for c in f.read_text().splitlines() if c.startswith("run ")] if f.exists() else []


class TestSpec(XabRepo):
    def test_valid_spec_and_its_shell_form(self):
        s = self.spec('XAB_NAME=moe-cap\nXAB_TREE=variants/t1-gpu-stack\n'
                      'XAB_ARMS=("base:" "cap:MOE_CAP=1.25 NOTE=\'a b\'")\nXAB_SEEDS=3 ; XAB_EST_MIN=30 ; '
                      'XAB_NOTE="expect -0.2 s; killed if Δval > 1 mnat"\n')
        self.assertTrue(s["ok"], s)
        self.assertEqual((s["name"], s["tree"], s["seeds"], s["est"], s["root"]), ("moe-cap", "variants/t1-gpu-stack", "1,2,3", 30, False))
        self.assertEqual(xab_run.parse_arm(s["arms"][1]), ("cap", {"MOE_CAP": "1.25", "NOTE": "a b"}))
        out = subprocess.run(["bash", "-c", f'eval "$({sys.executable} {XAB} spec "$1" --repo "$2")"; '
                              'echo "$XS_OK|$XS_NAME|$XS_N|${#XS_ARMS[@]}|${XS_ARMS[1]}|$XS_NOTE"', "_",
                              str(self.repo / "182/experiments.d/exp.sh"), str(self.repo)], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "1|moe-cap|3|2|cap:MOE_CAP=1.25 NOTE='a b'|expect -0.2 s; killed if Δval > 1 mnat")

    def test_root_tree_and_defaults(self):
        s = self.spec('XAB_NAME=r\nXAB_TREE=.\nXAB_ARMS=("base:" "x:FOO=1")\n')
        self.assertTrue(s["ok"], s)
        self.assertEqual((s["tree"], s["root"], s["n"], s["est"]), (".", True, 4, 0))

    def test_invalid_specs_say_why(self):
        cases = {
            'XAB_NAME="a b"\nXAB_TREE=.\nXAB_ARMS=("base:" "x:FOO=1")\n': "letters, digits",
            'XAB_NAME=a\nXAB_TREE=variants/nope\nXAB_ARMS=("base:" "x:FOO=1")\n': "no such directory",
            'XAB_NAME=a\nXAB_TREE=/abs\nXAB_ARMS=("base:" "x:FOO=1")\n': "repo-relative",
            'XAB_NAME=a\nXAB_TREE=../x\nXAB_ARMS=("base:" "x:FOO=1")\n': "repo-relative",
            'XAB_NAME=a\nXAB_TREE=.\nXAB_ARMS=("base:")\n': "a baseline and at least one more",
            'XAB_NAME=a\nXAB_TREE=.\nXAB_ARMS=("base:" "base:FOO=1")\n': "duplicate arm names",
            'XAB_NAME=a\nXAB_TREE=.\nXAB_ARMS=("base:" "x:FOO")\n': "is not KEY=VALUE",
            'XAB_NAME=a\nXAB_TREE=.\nXAB_ARMS=("base:" "x-y:FOO=1")\n': "is not NAME:ENV",
            'XAB_NAME=a\nXAB_TREE=.\nXAB_ARMS=("base:" "x:FOO=1")\nXAB_SEEDS=0\n': "1-32",
            'XAB_NAME=a\nXAB_TREE=variants\nXAB_ARMS=("base:" "x:FOO=1")\n': "has no 182/run_node.sh",
            'XAB_NAME=a\nXAB_ARMS=(\n': "could not source",
            '# nothing\n': "XAB_NAME=''",
        }
        for body, why in cases.items():
            s = self.spec(body)
            self.assertFalse(s["ok"], body)
            self.assertIn(why, s["error"], body)
        (self.repo / "train_gpt.py").unlink()
        self.assertIn("no train_gpt.py", self.spec('XAB_NAME=a\nXAB_TREE=.\nXAB_ARMS=("base:" "x:FOO=1")\n')["error"])

    def test_slot_order_rotates_by_seed(self):
        self.assertEqual(xab_run.slot_order(["a", "b", "c"], ["1", "2", "3"]),
                         [("1", "a"), ("1", "b"), ("1", "c"), ("2", "b"), ("2", "c"), ("2", "a"),
                          ("3", "c"), ("3", "a"), ("3", "b")])


class TestRun(XabRepo):
    def test_paired_run_report_numbers_resume(self):
        out = self.root / "x"
        rc, log = self.xab("run", "--out", str(out), "--tree", "variants/t1-gpu-stack", "--name", "demo",
                           "--seeds", "1,2,3,4", "--arm", "base:", "--arm", ON, "--arm", "same:")
        self.assertEqual(rc, 0, log)
        calls = self.calls()
        self.assertEqual(len(calls), 12)
        # interleaved by seed, arm order rotating; the arm's env arrives as EXTRA_ENV, LOG_DIR in the experiment dir
        self.assertEqual([c.split(" SEED=")[1].split()[0] for c in calls], ["1"] * 3 + ["2"] * 3 + ["3"] * 3 + ["4"] * 3)
        self.assertEqual(sum("EXTRA_ENV=FAKE_DVAL=0.001 FAKE_DVAL_SLOPE=0.0002 FAKE_DT=-0.25" in c for c in calls), 4)
        self.assertTrue(calls[3].endswith("FAKE_DT=-0.25"), calls[3])        # seed 2 starts with the second arm
        self.assertIn(f"LOG_DIR={out}/logs", calls[0])
        self.assertIn("complete: 3 arms x 4 seeds", log)
        # paired diffs (mnat) 1.2, 1.4, 1.6, 1.8: mean 1.5, SE 0.2582/2, t(0.975, 3) = 3.1824 -> CI [1.089, 1.911]
        R = xab_run.records(str(out))
        p = xab_run.paired_stats(R, "on")
        self.assertEqual(p["n"], 4)
        self.assertAlmostEqual(p["dval"], 1.5, places=6)
        self.assertAlmostEqual(p["dval_se"], math.sqrt(0.2 / 3) / 2, places=6)
        self.assertAlmostEqual(p["dval_lo"], 1.5 - 3.182446 * math.sqrt(0.2 / 3) / 2, places=4)
        self.assertAlmostEqual(p["dtime"], -0.25, places=6)
        self.assertAlmostEqual(p["net"], -0.25 + 1.5 * 0.164, places=6)
        self.assertAlmostEqual(p["net_band"][0], -0.25 + 1.5 * 0.120, places=6)
        self.assertAlmostEqual(p["net_band"][1], -0.25 + 1.5 * 0.250, places=6)
        self.assertEqual(xab_run.paired_stats(R, "same")["dval"], 0.0)
        rc, rep = self.xab("report", str(out), "--json", str(out / "s.json"))
        self.assertEqual(rc, 0, rep)
        self.assertIn("== demo: tree variants/t1-gpu-stack, baseline base, seeds 1,2,3,4", rep)
        self.assertRegex(rep, r"on +4/4 +3\.27\d+±")
        self.assertIn("Δval  +1.50 ± 0.13 mnat (SE), 95% CI [+1.09, +1.91]  -> Δval CI excludes 0 (the loss moved)", rep)
        self.assertIn("Δtime -0.250 ± 0.000 s (SE)", rep)
        self.assertIn("net_s = Δtime + Δval x 164 ms/mnat = -0.004 s", rep)
        self.assertIn("at 120-250 ms/mnat: -0.070 to +0.125 s  -> net CI includes 0: n=4 pairs do not settle it", rep)
        self.assertIn("same - base (n=4 seed pairs", rep)
        self.assertIn("Δval CI includes 0 (loss unchanged within noise)", rep)
        js = json.loads((out / "s.json").read_text())
        self.assertAlmostEqual(js["arms"]["on"]["dval"], 1.5, places=6)
        # resume: nothing re-runs
        rc, log = self.xab("run", "--out", str(out), "--tree", "variants/t1-gpu-stack", "--seeds", "1,2,3,4",
                           "--arm", "base:", "--arm", ON, "--arm", "same:")
        self.assertEqual((rc, len(self.calls())), (0, 12), log)
        # a different plan in the same directory is refused
        rc, log = self.xab("run", "--out", str(out), "--tree", "variants/t1-gpu-stack", "--seeds", "1,2",
                           "--arm", "base:", "--arm", ON)
        self.assertEqual(rc, 2, log)
        self.assertIn("planned with other arms/seeds/tree", log)

    def test_dead_arm_failed_slot_retry_and_check_codes(self):
        out = self.root / "x"
        args = ["run", "--out", str(out), "--tree", "variants/t1-gpu-stack", "--seeds", "1,2,3", "--arm", "base:FAKE_ARM=base",
                "--arm", "bad:FAKE_ARM=bad", "--arm", "ok:FAKE_ARM=ok FAKE_DT=0.1"]
        rc, log = self.xab(*args, FAKE_FAIL_ARM_ENV="bad")
        self.assertEqual(rc, 5, log)
        self.assertIn("abandoning arm bad: its first 2 run(s) all failed", log)
        self.assertEqual(sum("FAKE_ARM=bad" in c for c in self.calls()), 2)   # the third slot is not attempted
        self.assertEqual(xab_run.check_dir(str(out)), (5, "arm(s) bad failed every run"))
        rc, rep = self.xab("report", str(out))
        self.assertIn("[abandoned: bad]", rep)
        self.assertIn("failed: arm bad seed/rep 1: arm bad broke", rep)
        self.assertIn("ok - base (n=3 seed pairs", rep)
        # a dead arm stays dead on a plain rerun (the session re-plans only on FORCE=1)
        rc, log = self.xab(*args)
        self.assertEqual((rc, sum("FAKE_ARM=bad" in c for c in self.calls())), (5, 2), log)
        # one failed slot (not a dead arm): pending (3), then set aside as .fail1 and retried on the next invocation
        shutil.rmtree(out)
        shutil.rmtree(self.state)
        rc, log = self.xab(*args, FAKE_RUN_FAIL="t1-gpu-stack:1")
        self.assertEqual(rc, 3, log)
        self.assertEqual(xab_run.check_dir(str(out)), (3, "1 slot(s) not done"))
        rc, log = self.xab(*args)
        self.assertEqual(rc, 0, log)
        self.assertTrue((out / "runs/s1_base.fail1.result").exists())
        self.assertEqual(len(self.calls()), 10)

    def test_timeout_kills_the_run_and_counts_as_failed(self):
        out = self.root / "x"
        rc, log = self.xab("run", "--out", str(out), "--tree", "variants/t1-gpu-stack", "--seeds", "1", "--arm", "base:",
                           "--arm", "on:X=1", "--timeout-s", "1", "--kill-grace-s", "1", FAKE_RUN_SLEEP="30")
        self.assertNotEqual(rc, 0)
        self.assertIn("run exceeded 1 s", log)
        r = json.loads((out / "runs/s1_base.result").read_text())
        self.assertEqual((r["ok"], r["rc"]), (False, 124))
        self.assertIn("timed out", r["reason"])

    def test_dry_run_prints_and_writes_nothing(self):
        out = self.root / "x"
        rc, log = self.xab("run", "--out", str(out), "--tree", "variants/t1-gpu-stack", "--seeds", "1,2", "--arm", "base:",
                           "--arm", "nvls:NCCL_NVLS_ENABLE=1 NCCL_ALGO='allreduce:nvls;allgather:nvls'", "--dry-run")
        self.assertEqual(rc, 0, log)
        self.assertIn("+ SEED=2 NPROC=8 LOG_DIR=", log)
        self.assertIn("EXTRA_ENV='NCCL_NVLS_ENABLE=1 '\"'\"'NCCL_ALGO=allreduce:nvls;allgather:nvls'\"'\"''", log)
        self.assertFalse(out.exists())
        self.assertEqual(self.calls(), [])

    def test_quoted_arm_values_reach_the_tree_intact(self):
        out = self.root / "x"
        rc, log = self.xab("run", "--out", str(out), "--tree", "variants/t1-gpu-stack", "--seeds", "1,2", "--arm", "base:",
                           "--arm", "nvls:NCCL_NVLS_ENABLE=1 NCCL_ALGO='allreduce:nvls;allgather:nvls'")
        self.assertEqual(rc, 0, log)
        self.assertIn("EXTRA_ENV=NCCL_NVLS_ENABLE=1 'NCCL_ALGO=allreduce:nvls;allgather:nvls'", self.calls()[1])
        # what a real run_node.sh (node.py parse_extra_env: shlex.split) recovers from it
        import shlex
        extra = self.calls()[1].split("EXTRA_ENV=", 1)[1]
        self.assertEqual(shlex.split(extra), ["NCCL_NVLS_ENABLE=1", "NCCL_ALGO=allreduce:nvls;allgather:nvls"])

    def test_root_tree_is_refused_by_run(self):
        rc, log = self.xab("run", "--out", str(self.root / "x"), "--tree", ".", "--seeds", "1", "--arm", "a:", "--arm", "b:X=1")
        self.assertEqual(rc, 2)
        self.assertIn("goes through ab_run.py", log)


NVSMI_Q = """
GPU 00000000:18:00.0
    Product Name                          : NVIDIA H100 80GB HBM3
    Fabric
        State                             : {state}
        Status                            : {status}
        CliqueId                          : 0
    Processes                             : None
GPU 00000000:2A:00.0
    Fabric
        State                             : Completed
        Status                            : Success
"""


class TestNvlsCheck(unittest.TestCase):
    def test_fabric_state_parsing(self):
        f = nvls_check.fabric_state(NVSMI_Q.format(state="In Progress", status="N/A"))
        self.assertEqual(f, dict(state=["In Progress", "Completed"], status=["N/A", "Success"]))
        self.assertEqual(nvls_check.fabric_state("GPU 0\n    Product Name : X\n"), dict(state=[], status=[]))

    def test_verdicts(self):
        ok_fab = dict(state=["Completed"] * 8, status=["Success"] * 8)
        forced_ok = {"forced": dict(ok=True, nvls_negative=[])}
        v = nvls_check.verdict
        self.assertEqual(v([1] * 8, ok_fab, forced_ok)[0], "available")
        self.assertEqual(v(None, ok_fab, forced_ok)[0], "available")
        self.assertEqual(v([1] * 7 + [0], ok_fab, forced_ok), ("unavailable", "multicast not supported on GPU(s) [7]"))
        self.assertIn("fabric manager not up", v([1] * 8, dict(state=["In Progress"], status=[]), forced_ok)[1])
        self.assertIn("fabric status", v([1] * 8, dict(state=["Completed"], status=["Failure"]), forced_ok)[1])
        self.assertIn("forced-NVLS probe failed", v([1] * 8, ok_fab, {"forced": dict(ok=False, error="rc 1")})[1])
        self.assertIn("NCCL says", v([1] * 8, ok_fab, {"forced": dict(ok=True, nvls_negative=["NVLS disabled"])})[1])
        self.assertEqual(v([1] * 8, ok_fab, {})[0], "unknown")
        self.assertEqual(v(None, dict(state=[], status=[]), {})[0], "unknown")

    def run_main(self, fake_torchrun, smi_q, *args):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d)
            (b / "nvidia-smi").write_text('#!/usr/bin/env bash\nif [[ "$1" == topo ]]; then echo "GPU0 X NV18"; '
                                          f'else cat <<"EOF"\n{smi_q}\nEOF\nfi\n')
            (b / "torchrun").write_text("#!/usr/bin/env bash\n" + fake_torchrun)
            for f in ("nvidia-smi", "torchrun"):
                (b / f).chmod(0o755)
            env = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}")
            r = subprocess.run([sys.executable, str(Path(TOOLS) / "nvls_check.py"), "--nproc", "2", "--timeout-s", "30",
                                "--json", str(b / "o.json"), *args], env=env, capture_output=True, text=True, timeout=120)
            return r.returncode, r.stdout, json.loads((b / "o.json").read_text())

    def test_main_with_probes(self):
        # the stub "torchrun" answers like the probe: NVLS lines + bandwidths; NVLS off -> slower all_reduce
        fake = ('if [[ "${NCCL_NVLS_ENABLE:-2}" == 0 ]]; then us=200; else us=100; echo "NCCL INFO NVLS multicast support is available"; fi\n'
                'echo "PROBE_JSON {\\"all_reduce_64MiB\\": {\\"us\\": $us, \\"busbw_GBps\\": 400.0}}"\n')
        rc, out, js = self.run_main(fake, NVSMI_Q.format(state="Completed", status="Success"))
        self.assertIn("GPU0 X NV18", out)
        self.assertIn("## probe forced: NCCL_NVLS_ENABLE=1 NCCL_ALGO=allreduce:nvls;reducescatter:nvls;allgather:nvls: ok", out)
        self.assertIn("default vs NVLS off, all_reduce_64MiB: -50.0% time", out)
        self.assertEqual(set(js["probes"]), {"default", "nvls_off", "forced"})
        if js["multicast"] is None:           # no libcuda here: the forced probe decides
            self.assertEqual((rc, js["verdict"]), (0, "available"))
            self.assertIn("NVLS: available -- the forced-NVLS probe ran (multicast attribute not read)", out)

    def test_main_forced_probe_failure_and_fabric_down(self):
        fake = ('if [[ -n "${NCCL_ALGO:-}" ]]; then echo "NCCL WARN NVLS not supported on this system"; exit 1; fi\n'
                'echo "PROBE_JSON {}"\n')
        rc, out, js = self.run_main(fake, NVSMI_Q.format(state="Completed", status="Success"))
        self.assertEqual(rc, 1, out)
        self.assertIn("NVLS: unavailable -- the forced-NVLS probe failed: rc 1", out)
        rc, out, js = self.run_main(fake, NVSMI_Q.format(state="In Progress", status="N/A"), "--no-probe")
        self.assertEqual((rc, js["verdict"]), (1, "unavailable"))
        self.assertIn("fabric manager not up", out)
        self.assertEqual(js["probes"], {})

    def test_hung_probe_times_out(self):
        rc, out, js = self.run_main("sleep 60\n", NVSMI_Q.format(state="Completed", status="Success"), "--timeout-s", "1")
        self.assertEqual(rc, 1)
        self.assertIn("timed out after 1 s", js["probes"]["forced"]["error"])


if __name__ == "__main__":
    unittest.main()
