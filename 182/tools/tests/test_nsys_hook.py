"""nsys_step_hook.py against a stub torch (no CUDA): which steps get NVTX ranges, and where the
profiler starts and stops."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from _paths import TOOLS

STUB_TORCH = textwrap.dedent('''
    import types
    EVENTS = []
    class _Cudart:
        def cudaProfilerStart(self): EVENTS.append("start")
        def cudaProfilerStop(self): EVENTS.append("stop")
    _rt = _Cudart()
    cuda = types.SimpleNamespace(
        cudart=lambda: _rt,
        synchronize=lambda: EVENTS.append("sync"),
        nvtx=types.SimpleNamespace(range_push=lambda s: EVENTS.append("push " + s), range_pop=lambda: EVENTS.append("pop")),
    )
''')

TRAINER = textwrap.dedent('''
    import json, sys, torch
    def train_step(manager, graphs, step, flag=False):
        torch.EVENTS.append(f"run {step}")
    def main():
        for s in [0, 1, 2, 7, 8]:          # warmup pass (sampled steps)
            train_step(None, None, s)
        for s in range(12):                # timed loop
            train_step(None, None, step=s, flag=True)
        print("EVENTS=" + json.dumps(torch.EVENTS))
    if __name__ == "__main__":
        raise SystemExit("the guard must not fire under the hook")
''')


class TestHook(unittest.TestCase):
    def run_hook(self, env_extra, trainer=TRAINER):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "stub", "torch"))
            with open(os.path.join(d, "stub", "torch", "__init__.py"), "w") as f:
                f.write(STUB_TORCH)
            with open(os.path.join(d, "train_gpt.py"), "w") as f:
                f.write(trainer)
            env = dict(os.environ, PYTHONPATH=os.path.join(d, "stub"), **env_extra)
            r = subprocess.run([sys.executable, os.path.join(TOOLS, "nsys_step_hook.py"), "train_gpt.py"],
                               cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            line = [l for l in r.stdout.splitlines() if l.startswith("EVENTS=")][-1]
            return json.loads(line[7:]), r.stdout

    def test_timed_loop_window(self):
        ev, _ = self.run_hook({"NSYS_START_STEP": "5", "NSYS_NUM_STEPS": "3"})
        # the warmup pass (phase 0) is untouched even though it also visits steps 7 and 8
        self.assertEqual(ev[:5], ["run 0", "run 1", "run 2", "run 7", "run 8"])
        timed = ev[5:]
        i = timed.index("start")
        self.assertEqual(timed[i - 1], "run 3")          # starts one unannotated step early (step 4)
        self.assertEqual(timed[i + 1], "run 4")
        self.assertEqual([e for e in timed if e.startswith("push")], ["push step 5", "push step 6", "push step 7"])
        j = timed.index("stop")
        self.assertEqual(timed[j - 1], "sync")
        self.assertEqual(timed[j - 2], "run 8")          # step 8 (unannotated) traced, stop at step 9's entry
        self.assertEqual(timed[j + 1], "run 9")
        self.assertEqual(timed.count("start"), 1)
        self.assertEqual(timed.count("stop"), 1)

    def test_phase0_and_unhooked_fallback(self):
        ev, _ = self.run_hook({"NSYS_START_STEP": "1", "NSYS_NUM_STEPS": "1", "NSYS_HOOK_PHASE": "0"})
        self.assertEqual(ev[:6], ["start", "run 0", "push step 1", "run 1", "pop", "run 2"])
        inline = "import json, torch\nfor s in range(3): pass\nprint('EVENTS=' + json.dumps(torch.EVENTS))\n"
        ev, out = self.run_hook({}, trainer=inline)
        self.assertEqual(ev, [])
        self.assertIn("running it unhooked", out)


if __name__ == "__main__":
    unittest.main()
