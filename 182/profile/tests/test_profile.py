"""182/profile on CPU: analyze.py on a synthetic Kineto trace with hand-computed numbers, hook.py's window logic and
an end-to-end hooked run against a stub torch, and the torchrun shim. No GPU, no real torch needed."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROFILE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROFILE))
import analyze  # noqa: E402
import hook  # noqa: E402


def kernel(name, ts, dur, corr, stream=7, cat="kernel"):
    return dict(ph="X", cat=cat, name=name, pid=0, tid=stream, ts=ts, dur=dur,
                args=dict(device=0, stream=stream, correlation=corr))


def launch(name, ts, corr):
    return dict(ph="X", cat="cuda_runtime", name=name, pid=1, tid=1, ts=ts, dur=5, args=dict(correlation=corr))


def synthetic_trace():
    """Two profiled steps on device 0 (times in us).

    step 300 (CPU 0-1000): flash fwd [100,300]; one cudaGraphLaunch whose replay ran nvjet GEMM [300,600] and an
      inductor pointwise kernel [600,700]; an NCCL reduce-scatter on another stream [500,900]. GPU window [100,1100)
      (the next step's first kernel): busy 800, idle 200 (the bubble 900-1100); NCCL 400 of which 200 exposed.
    step 301 (CPU 1000-2000): XXT (Newton-Schulz gram) [1100,1400], an fp8 quantize [1400,1500], a DtoD memcpy
      [1500,1550]. Window [1100,1550]: busy 450, idle 0.
    Means: window 0.725 ms, busy 0.625, idle 0.100, NCCL 0.200, exposed 0.100. A kernel launched before the window
    ([50,70]) and the GPU-side mirror of a step range are ignored.
    """
    ev = [
        dict(ph="X", cat="user_annotation", name="prof_step 300", pid=1, tid=1, ts=0, dur=1000),
        dict(ph="X", cat="user_annotation", name="prof_step 301", pid=1, tid=1, ts=1000, dur=1000),
        dict(ph="X", cat="gpu_user_annotation", name="prof_step 300", pid=0, tid=7, ts=100, dur=900),
        launch("cudaLaunchKernel", -50, 7), kernel("pre_window_kernel", 50, 20, 7),
        launch("cudaLaunchKernel", 10, 1), kernel("void flash::flash_fwd_kernel<Sm90>", 100, 200, 1),
        launch("cudaGraphLaunch", 20, 2), kernel("nvjet_tst_128x256_64x4_1x2_h_bz_coopA_TNT", 300, 300, 2),
        kernel("triton_poi_fused_add_mul_0", 600, 100, 2),
        launch("cudaLaunchKernel", 30, 3), kernel("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL", 500, 400, 3, stream=20),
        launch("cuLaunchKernel", 1010, 4), kernel("XXT_kernel", 1100, 300, 4),
        launch("cuLaunchKernel", 1020, 5), kernel("_quantize_weights_dual_kernel", 1400, 100, 5),
        launch("cudaMemcpyAsync", 1030, 6), kernel("Memcpy DtoD (Device -> Device)", 1500, 50, 6, cat="gpu_memcpy"),
        dict(ph="s", cat="ac2g", name="ac2g", id=1, pid=1, tid=1, ts=10),
    ]
    return dict(traceEvents=ev, deviceProperties=[])


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.s = analyze.summarize(synthetic_trace()["traceEvents"])

    def test_step_windows_busy_idle_nccl(self):
        s = self.s
        self.assertEqual(s["steps_profiled"], 2)
        self.assertAlmostEqual(s["mean_window_ms"], 0.725)
        self.assertAlmostEqual(s["mean_busy_ms"], 0.625)
        self.assertAlmostEqual(s["mean_idle_ms"], 0.100)
        self.assertAlmostEqual(s["nccl_ms"], 0.200)
        self.assertAlmostEqual(s["nccl_exposed_ms"], 0.100)
        self.assertEqual([(x["step"], x["kernels"]) for x in s["steps"]], [(300, 4), (301, 3)])
        self.assertAlmostEqual(s["steps"][0]["window_ms"], 1.0)
        self.assertAlmostEqual(s["steps"][1]["idle_ms"], 0.0)
        self.assertAlmostEqual(s["mean_cpu_ms"], 1.0)

    def test_categories(self):
        c = {x["key"]: x["ms_per_step"] for x in self.s["categories"]}
        want = dict(attn_fwd=0.1, gemm=0.15, other=0.05, nccl=0.2, optim=0.15, fp8_quant=0.05, memcpy=0.025,
                    attn_bwd=0, gemm_moe=0, loss_head=0, rows=0)
        for k, v in want.items():
            self.assertAlmostEqual(c[k], v, msg=k)
        pct = {x["key"]: x["pct"] for x in self.s["categories"]}
        self.assertAlmostEqual(pct["nccl"], 100 * 0.2 / 0.725)

    def test_graph_coverage_top_kernels_and_gaps(self):
        s = self.s
        self.assertEqual((s["graph_launches"], s["graph_kernels"], s["kernels"]), (1, 2, 7))
        self.assertEqual(s["top_kernels"][0]["name"], "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL")
        self.assertEqual([k["cat"] for k in s["top_kernels"][:2]], ["nccl", "gemm"])
        g = s["gaps"][0]
        self.assertEqual((g["us"], g["step"]), (200, 300))
        self.assertTrue(g["after"].startswith("ncclDevKernel"), g)
        self.assertEqual(g["before"], "XXT_kernel")
        self.assertEqual(s["warnings"], [])

    def test_missing_graph_kernels_are_flagged(self):
        ev = [e for e in synthetic_trace()["traceEvents"] if (e.get("args") or {}).get("correlation") != 2
              or e.get("cat") != "kernel"]
        s = analyze.summarize(ev)
        self.assertEqual((s["graph_launches"], s["graph_kernels"]), (1, 0))
        self.assertTrue(any("CUPTI did not record the graph replays" in w for w in s["warnings"]), s["warnings"])

    def test_known_kernel_names(self):
        cats = analyze.compile_categories()
        names = {
            "void flash::FlashAttnBwdSm90<...>": "attn_bwd", "_qk_norm_rope_forward_kernel": "attn_fwd",
            "ncclDevKernel_AllGather_RING_LL": "nccl", "grouped_mm_kernel": "gemm_moe", "_experts_forward": "gemm_moe",
            "sm90_xmma_gemm_e4m3e4m3_bf16_tn": "gemm", "linear_relu_square_kernel": "gemm", "ba_plus_cAA_kernel": "optim",
            "_fused_adam_kernel": "optim", "_hadamard_quant_dual_kernel": "fp8_quant", "_gather_quant_dual_kernel": "fp8_quant",
            "cplm_ce_fwd_bwd": "loss_head", "_ngram_embed_fwd_kernel": "rows", "_scatter_add_rows_kernel": "rows",
            "triton_red_fused_sum_3": "other",
        }
        for n, k in names.items():
            self.assertEqual(analyze.categorize(n, "kernel", cats), k, n)

    def test_cli_on_a_profile_dir_with_rank_summaries(self):
        with tempfile.TemporaryDirectory() as d:
            with gzip.open(os.path.join(d, "trace_rank0.json.gz"), "wt") as f:
                json.dump(synthetic_trace(), f)
            for r, w in ((0, 0.725), (1, 0.9)):
                json.dump(dict(self.s, mean_window_ms=w, meta=dict(rank=r, status="ok", window_ms=w * 20, count=20)),
                          open(os.path.join(d, f"rank{r}_summary.json"), "w"))
            out = subprocess.run([sys.executable, str(PROFILE / "analyze.py"), d, "--json", os.path.join(d, "a.json")],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("GPU window 0.72 ms/step", out.stdout)
            self.assertRegex(out.stdout, r"NCCL communication +0\.200 +27\.6%")
            self.assertIn("NCCL: 0.200 ms/step of kernels, 0.100 ms exposed", out.stdout)
            self.assertIn("50% overlapped", out.stdout)
            self.assertIn("spread of the GPU window across ranks: 0.72-0.90 ms", out.stdout)
            self.assertEqual(len(json.load(open(os.path.join(d, "a.json")))["ranks"]), 2)
            os.remove(os.path.join(d, "trace_rank0.json.gz"))
            out = subprocess.run([sys.executable, str(PROFILE / "analyze.py"), d], capture_output=True, text=True)
            self.assertEqual(out.returncode, 1)
            self.assertIn("no trace_rank0.json.gz", out.stdout)


class TestWindow(unittest.TestCase):
    def drive(self, calls, start, count):
        w, opened, inside, closed = hook.Window(start, count), [], [], []
        for i, s in enumerate(calls):
            if w.before(s):
                opened.append(i)
            if w.inside(s):
                inside.append(s)
            if w.after(s):
                closed.append(i)
        return opened, inside, closed

    def test_skips_the_warmup_pass(self):
        warm = sorted({0, 1, 2, 3, 5, 6, 40, 41, 300, 301, 302})   # warmup visits the window's steps too
        calls = warm + list(range(400))
        opened, inside, closed = self.drive(calls, 300, 3)
        self.assertEqual(opened, [len(warm) + 300])
        self.assertEqual(inside, [300, 301, 302])
        self.assertEqual(closed, [len(warm) + 302])

    def test_no_warmup_and_too_short_runs(self):
        self.assertEqual(self.drive(list(range(10)), 2, 3)[1], [2, 3, 4])
        self.assertEqual(self.drive(list(range(5)), 7, 3), ([], [], []))
        opened, inside, closed = self.drive(list(range(10)), 8, 5)    # cut short: opened, never closed
        self.assertEqual((opened, inside, closed), ([8], [8, 9], []))

    def test_parse_steps(self):
        self.assertEqual(hook.parse_steps("300:20"), (300, 20))
        for bad in ("300", "a:b", "0:5", "5:0"):
            with self.assertRaises(SystemExit):
                hook.parse_steps(bad)


STUB_TORCH = r'''
import json, os
__version__ = "0.0-stub"
class _Cuda:
    calls = []
    @staticmethod
    def is_available(): return True
    @staticmethod
    def synchronize(): _Cuda.calls.append("sync")
    @staticmethod
    def set_device(d): _Cuda.calls.append(f"set_device {d}")
cuda = _Cuda()
def zeros(*a, **k):
    class T:
        def add_(self, x): return self
    return T()
class _Act:
    CPU, CUDA = "cpu", "cuda"
class _Prof:
    sessions = []
    def __init__(self, activities=None, record_shapes=False, with_stack=False, profile_memory=False):
        self.acts, self.shapes, self.events, self.on = activities, record_shapes, [], False
        _Prof.sessions.append(self)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def start(self): self.on = True; _Prof.log.append("start")
    def stop(self): self.on = False; _Prof.log.append("stop")
    def export_chrome_trace(self, path):
        import gzip
        ev = [e for e in _Prof.marks]
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "wt") as f:
            json.dump({"traceEvents": ev}, f)
_Prof.log, _Prof.marks = [], []
class _RF:
    def __init__(self, name): self.name = name
    def __enter__(self):
        s = int(self.name.split()[1])
        t = 1000 * s
        _Prof.marks.extend([
            {"ph": "X", "cat": "user_annotation", "name": self.name, "ts": t, "dur": 900},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": t + 1, "dur": 1, "args": {"correlation": s}},
            {"ph": "X", "cat": "kernel", "name": "nvjet_gemm", "ts": t + 10, "dur": 500, "args": {"correlation": s, "device": 0}}])
    def __exit__(self, *a): pass
class profiler:
    ProfilerActivity = _Act
    profile = _Prof
    record_function = _RF
'''

FAKE_TRAINER = r'''
import os, sys
def train_step(model, step: int, flag=False):
    with open(os.environ["TRACE_CALLS"], "a") as f:
        f.write(f"{step}\n")
def main():
    for s in [0, 1, 5, 6, 7]:          # warmup visits the window's steps
        train_step(None, s)
    for s in range(12):
        train_step(None, step=s)
    print("trainer done", sys.argv[0], __name__)
if __name__ == "__main__":
    raise SystemExit("the guard must not fire under the hook")
'''


class TestHookEndToEnd(unittest.TestCase):
    def test_hooked_run_writes_trace_summary_and_meta(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "stub/torch").mkdir(parents=True)
            (d / "stub/torch/__init__.py").write_text(STUB_TORCH)
            (d / "train_gpt.py").write_text(FAKE_TRAINER)
            out = d / "prof"
            env = dict(os.environ, PYTHONPATH=str(d / "stub"), PROFILE_STEPS="5:3", PROFILE_DIR=str(out), RANK="0",
                       LOCAL_RANK="3", TRACE_CALLS=str(d / "calls"))
            r = subprocess.run([sys.executable, str(PROFILE / "hook.py"), "train_gpt.py"], cwd=d, env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("window opened at step 5 (3 steps)", r.stdout)
            self.assertIn("window closed after step 7", r.stdout)
            self.assertIn("trainer done train_gpt.py train_gpt_prof", r.stdout)   # argv[0] kept: the run log embeds the trainer
            self.assertEqual((d / "calls").read_text().split(), [str(s) for s in [0, 1, 5, 6, 7] + list(range(12))])
            meta = json.loads((out / "meta_rank0.json").read_text())
            self.assertEqual((meta["status"], meta["cupti_init"], len(meta["step_cpu_ms"])), ("ok", "done", 3))
            summ = json.loads((out / "rank0_summary.json").read_text())
            self.assertEqual([s["step"] for s in summ["steps"]], [5, 6, 7])
            self.assertEqual((summ["graph_launches"], summ["graph_kernels"]), (3, 3))
            self.assertTrue((out / "trace_rank0.json.gz").exists())
            # rank != 0: a summary, no trace left behind
            env.update(RANK="1", PROFILE_CUPTI_INIT="0")
            r = subprocess.run([sys.executable, str(PROFILE / "hook.py"), "train_gpt.py"], cwd=d, env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((out / "rank1_summary.json").exists())
            self.assertEqual(sorted(p.name for p in out.iterdir() if "trace" in p.name), ["trace_rank0.json.gz"])
            self.assertEqual(json.loads((out / "meta_rank1.json").read_text())["cupti_init"], "off")

    def test_window_never_opened_is_said(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "stub/torch").mkdir(parents=True)
            (d / "stub/torch/__init__.py").write_text(STUB_TORCH)
            (d / "train_gpt.py").write_text(FAKE_TRAINER)
            env = dict(os.environ, PYTHONPATH=str(d / "stub"), PROFILE_STEPS="50:3", PROFILE_DIR=str(d / "p"),
                       TRACE_CALLS=str(d / "calls"))
            r = subprocess.run([sys.executable, str(PROFILE / "hook.py"), "train_gpt.py"], cwd=d, env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("window never opened", r.stdout)
            self.assertFalse((d / "p/trace_rank0.json.gz").exists())


class TestShimAndRunScript(unittest.TestCase):
    def test_shim_inserts_the_hook_and_execs_the_real_torchrun(self):
        with tempfile.TemporaryDirectory() as d:
            real = Path(d) / "bin"
            real.mkdir()
            (real / "torchrun").write_text('#!/usr/bin/env bash\necho "REAL $*"\n')
            (real / "torchrun").chmod(0o755)
            env = dict(os.environ, PATH=f"{PROFILE / 'shim'}:{real}:{os.environ['PATH']}")
            r = subprocess.run(["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py", "--x"], env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.stdout.strip(), f"REAL --standalone --nproc_per_node=8 {PROFILE / 'hook.py'} train_gpt.py --x")
            r = subprocess.run(["torchrun", "--standalone"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            env["PATH"] = f"{PROFILE / 'shim'}:/usr/bin:/bin"
            r = subprocess.run([str(PROFILE / "shim/torchrun"), "x.py"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 127, r.stderr)

    def test_run_script_through_a_tree_run_node_and_root(self):
        """run.sh drives a tree's run_node.sh with the shim first on PATH (the fake run_node calls `torchrun`), and the
        root through torchrun + hook.py directly; the real torchrun here is a stub that runs python on its script."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "stub/torch").mkdir(parents=True)
            (d / "stub/torch/__init__.py").write_text(STUB_TORCH)
            b = d / "bin"
            b.mkdir()
            (b / "torchrun").write_text('#!/usr/bin/env bash\nargs=(); for a in "$@"; do [[ $a == --* ]] || args+=("$a"); done\n'
                                        f'exec {sys.executable} "${{args[@]}}"\n')
            (b / "torchrun").chmod(0o755)
            tree = d / "variants/t1"
            (tree / "182").mkdir(parents=True)
            (tree / "train_gpt.py").write_text(FAKE_TRAINER)
            (tree / "182/run_node.sh").write_text('#!/usr/bin/env bash\ncd "$(dirname "$0")/.." && torchrun --standalone '
                                                  '--nproc_per_node=$NPROC train_gpt.py\n'
                                                  'echo "RESULT variant=t1 val_loss=3.2700 train_time_s=31.0 steps=12 target=3.28 hit=1"\n')
            (d / "train_gpt.py").write_text(FAKE_TRAINER)
            env = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", PYTHONPATH=str(d / "stub"), PROFILE_STEPS="5:3",
                       TRACE_CALLS=str(d / "calls"), PY=sys.executable)
            outs = {}
            for t, o in ((tree, d / "out_t1"), (d, d / "out_root")):
                r = subprocess.run(["bash", str(PROFILE / "run.sh"), str(t), str(o)], env=env, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("window opened at step 5", r.stdout)
                self.assertIn("graph coverage: 3 cudaGraphLaunch call(s), 3 of 3 kernels", r.stdout)
                self.assertTrue((o / "analysis.json").exists())
                outs[o.name] = r.stdout
            self.assertIn("profiled run (NOT a timing run): RESULT variant=t1 val_loss=3.2700", outs["out_t1"])
            self.assertTrue((d / "out_t1/rank0_summary.json").exists())
            r = subprocess.run(["bash", str(PROFILE / "run.sh"), str(d), str(d / "dry")], env=dict(env, DRY_RUN="1"),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0)
            self.assertIn("hook.py train_gpt.py", r.stdout)
            self.assertFalse((d / "dry/train.out").exists())

    def test_run_script_keeps_the_shim_first_when_run_node_would_activate_a_venv(self):
        """A variant run_node.sh activates <repo>/.venv-run when no venv is active; run.sh activates it first so the
        shim stays ahead of the venv's torchrun. A run that bypasses the shim is a FAIL, not a silent unprofiled run."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prof = d / "repo/182/profile"
            (prof / "shim").mkdir(parents=True)
            for f in ("run.sh", "hook.py", "analyze.py"):
                (prof / f).write_bytes((PROFILE / f).read_bytes())
            (prof / "shim/torchrun").write_bytes((PROFILE / "shim/torchrun").read_bytes())
            (prof / "shim/torchrun").chmod(0o755)
            venv_bin = d / "repo/.venv-run/bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "activate").write_text(f'export VIRTUAL_ENV="{d}/repo/.venv-run"; export PATH="$VIRTUAL_ENV/bin:$PATH"\n')
            (venv_bin / "torchrun").write_text('#!/usr/bin/env bash\necho "VENV TORCHRUN $*"\n')
            (venv_bin / "torchrun").chmod(0o755)
            tree = d / "repo/variants/t1"
            (tree / "182").mkdir(parents=True)
            (tree / "train_gpt.py").write_text(FAKE_TRAINER)
            # the activation rule of variants/*/182/run_node.sh
            (tree / "182/run_node.sh").write_text(
                '#!/usr/bin/env bash\nACT="$(cd "$(dirname "$0")/../../.." && pwd)/.venv-run/bin/activate"\n'
                '[[ -z "${VIRTUAL_ENV:-}" && -f "$ACT" ]] && source "$ACT"\n'
                'cd "$(dirname "$0")/.." && torchrun --standalone train_gpt.py\necho "RESULT variant=t1 val_loss=3.27"\n')
            env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
            env.update(PY=sys.executable, PROFILE_STEPS="5:3")
            r = subprocess.run(["bash", str(prof / "run.sh"), str(tree), str(d / "o")], env=env, capture_output=True, text=True)
            # the shim ran and exec'd the venv's torchrun (the next on PATH) with hook.py inserted
            self.assertIn(f"VENV TORCHRUN --standalone {prof / 'hook.py'} train_gpt.py", r.stdout)
            self.assertTrue((d / "o/.shim_used").exists())
            # bypass: a run_node.sh that calls a torchrun by absolute path never meets the shim
            (tree / "182/run_node.sh").write_text(f'#!/usr/bin/env bash\n"{venv_bin}/torchrun" train_gpt.py\n')
            r = subprocess.run(["bash", str(prof / "run.sh"), str(tree), str(d / "o2")], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stdout)
            self.assertIn("did not start through 182/profile/shim/torchrun", r.stdout)


if __name__ == "__main__":
    unittest.main()
