"""torch.profiler over a few steady-state steps of a trainer's TIMED loop, without editing the trainer.

    PROFILE_STEPS=300:20 PROFILE_DIR=out torchrun --standalone --nproc_per_node=8 182/profile/hook.py train_gpt.py

Normally launched through 182/profile/run.sh, which puts 182/profile/shim/torchrun on PATH so a tree's own
`182/run_node.sh run` (data path, seed, checks, RESULT line) starts the trainer through this file.

How: like 182/tools/nsys_step_hook.py, the script is executed as a module named "<stem>_prof" (its
`if __name__ == "__main__"` guard does not fire), its step function is wrapped, and its main() is called.
sys.argv[0] stays the script path, so the run log still embeds the trainer's own source. Nothing changes in a
run that does not go through this file: the trainers have no profiling code at all.

Which call is the timed loop: the warmup also calls the step function (with a sorted subset of the step
indices, at most 40 of them), so the window opens at the first call with step == START that is preceded by the
calls 0, 1, ..., START-1 in a row. Only the timed loop makes such a run (for START >= 40 the warmup cannot).

All ranks open and close the window at the same step, with a torch.cuda.synchronize() at both ends (so the
window's wall time is a real GPU time and every rank is perturbed alike). After the trainer's main() returns
(or raises), rank 0 writes the Chrome trace and every rank writes a summary of its own trace (analyze.py's
numbers, so the ranks can be compared). The profiled run is NOT a timing run.

CUDA graphs: both trees replay the step's forward/backward, the optimizer and the fp8 refresh as CUDA graphs.
CUPTI reports the kernels inside a graph replay only if it was initialised before the graphs were captured
(pytorch issue 75504; torch.profiler._utils._init_for_cuda_graphs is the same workaround), and the trainers
capture during warmup, long before the window. So by default (PROFILE_CUPTI_INIT=1) this file starts and stops
an empty CUDA profiler session before the trainer runs, with TEARDOWN_CUPTI=0 and DISABLE_CUPTI_LAZY_REINIT=1
(what torch itself sets when it knows graphs are in use). analyze.py says whether graph kernels made it into
the trace. There is no graphs-off mode: neither trainer has an eager path (the step, optimizer and fp8
graphs are structural), so if graph kernels are missing the fallback is 182/tools/profile_nsys.sh
(nsys --cuda-graph-trace=node), not a different program.

Env:
  PROFILE_STEPS      START:COUNT of the timed loop (default 300:20)
  PROFILE_DIR        output directory (default ./profile_out); created
  PROFILE_FN         step function to wrap (default train_step), PROFILE_STEP_ARG its step argument (default step)
  PROFILE_CUPTI_INIT 1 (default) / 0: the early CUPTI initialisation above
  PROFILE_SHAPES     1 (default) / 0: record_shapes
Outputs in PROFILE_DIR: trace_rank0.json.gz, rank<R>_summary.json (every rank), meta_rank<R>.json.
"""
from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_steps(spec: str) -> tuple[int, int]:
    try:
        a, b = spec.split(":")
        start, count = int(a), int(b)
    except ValueError:
        raise SystemExit(f"[profile] PROFILE_STEPS={spec!r} is not START:COUNT (e.g. 300:20)")
    if start < 1 or count < 1:
        raise SystemExit(f"[profile] PROFILE_STEPS={spec!r}: START and COUNT must be >= 1")
    return start, count


class Window:
    """Decides, call by call, when the profiling window opens and closes (pure logic; no torch)."""

    def __init__(self, start: int, count: int):
        self.start, self.count = start, count
        self.prev = None
        self.run = 0          # == s + 1 iff the calls so far ended with 0, 1, ..., s
        self.state = "wait"   # wait -> on -> done

    def before(self, step: int) -> bool:
        """Called before each step; True when the window opens at this step."""
        if step == 0:
            self.run = 1
        elif self.prev is not None and step == self.prev + 1 and self.run == step:
            self.run = step + 1
        else:
            self.run = 0
        self.prev = step
        if self.state == "wait" and step == self.start and self.run == step + 1:
            self.state = "on"
            return True
        return False

    def inside(self, step: int) -> bool:
        return self.state == "on" and self.start <= step < self.start + self.count

    def after(self, step: int) -> bool:
        """Called after each step; True when the window closes after this step."""
        if self.state == "on" and step >= self.start + self.count - 1:
            self.state = "done"
            return True
        return False


def _load_analyze():
    spec = importlib.util.spec_from_file_location("_profile_analyze", os.path.join(HERE, "analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def init_cupti(torch, local_rank: int) -> str:
    """An empty CUDA profiler session before any graph capture, so CUPTI traces graph-replay kernels later."""
    os.environ.setdefault("TEARDOWN_CUPTI", "0")
    os.environ.setdefault("DISABLE_CUPTI_LAZY_REINIT", "1")
    if not torch.cuda.is_available():
        return "skipped (no CUDA)"
    torch.cuda.set_device(local_rank)  # the trainer sets the same device from LOCAL_RANK
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    return "done"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    start, count = parse_steps(os.environ.get("PROFILE_STEPS", "300:20"))
    out = os.path.abspath(os.environ.get("PROFILE_DIR") or "profile_out")
    fn_name = os.environ.get("PROFILE_FN", "train_step")
    arg_name = os.environ.get("PROFILE_STEP_ARG", "step")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.makedirs(out, exist_ok=True)
    with open(script) as f:
        src = f.read()
    if f"def {fn_name}(" not in src or "def main(" not in src:
        raise SystemExit(f"[profile] {script} has no {fn_name}() or main(): nothing to hook (PROFILE_FN?)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))

    import torch

    meta = dict(script=script, cwd=os.getcwd(), rank=rank, start=start, count=count, torch=torch.__version__,
                cupti_init="off", status="window never opened")
    if os.environ.get("PROFILE_CUPTI_INIT", "1") == "1":
        meta["cupti_init"] = init_cupti(torch, local_rank)
    name = os.path.splitext(os.path.basename(script))[0] + "_prof"
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(script)
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    inner = getattr(mod, fn_name)
    sig = inspect.signature(inner)
    win = Window(start, count)
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    prof = torch.profiler.profile(activities=acts, record_shapes=os.environ.get("PROFILE_SHAPES", "1") == "1",
                                  with_stack=False, profile_memory=False)
    st = dict(t0=None, cpu_ms=[])

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @functools.wraps(inner)
    def wrapped(*args, **kwargs):
        step = sig.bind_partial(*args, **kwargs).arguments.get(arg_name)
        if step is None:
            return inner(*args, **kwargs)
        if win.before(step):
            sync()
            prof.start()
            st["t0"] = time.perf_counter()
            if rank == 0:
                print(f"[profile] window opened at step {step} ({count} steps) -> {out}", flush=True)
        if not win.inside(step):
            return inner(*args, **kwargs)
        t = time.perf_counter()
        try:
            with torch.profiler.record_function(f"prof_step {step}"):
                return inner(*args, **kwargs)
        finally:
            st["cpu_ms"].append(1000 * (time.perf_counter() - t))
            if win.after(step):
                sync()
                meta["window_ms"] = 1000 * (time.perf_counter() - st["t0"])
                prof.stop()
                meta["status"] = "ok"
                if rank == 0:
                    print(f"[profile] window closed after step {step}: {meta['window_ms'] / count:.2f} ms/step "
                          f"(synchronized wall, profiler on)", flush=True)

    setattr(mod, fn_name, wrapped)
    try:
        mod.main()
    finally:
        if win.state == "on":  # the run ended (or crashed) inside the window
            prof.stop()
            meta["status"] = "window cut short"
        meta["step_cpu_ms"] = [round(x, 3) for x in st["cpu_ms"]]
        if win.state != "wait":
            write_outputs(prof, meta, out, rank)
        with open(os.path.join(out, f"meta_rank{rank}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if rank == 0 and meta["status"] != "ok":
            print(f"[profile] {meta['status']} (PROFILE_STEPS={start}:{count}; does the run have that many steps?)",
                  flush=True)


def write_outputs(prof, meta: dict, out: str, rank: int):
    """Rank 0 keeps its trace; every rank summarizes its own (same code as analyze.py) and drops the rest."""
    path = os.path.join(out, "trace_rank0.json.gz" if rank == 0 else f".trace_rank{rank}.tmp.json")
    try:
        prof.export_chrome_trace(path)
        an = _load_analyze()
        summary = an.summarize(an.load_trace(path), steps_hint=meta["count"])
        summary["meta"] = {k: v for k, v in meta.items() if k != "step_cpu_ms"}
        with open(os.path.join(out, f"rank{rank}_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    except Exception as e:  # a failed export must not hide the run's own result
        meta["export_error"] = f"{type(e).__name__}: {e}"[:500]
        print(f"[profile] rank {rank}: trace export/summary failed: {meta['export_error']}", flush=True)
    finally:
        if rank != 0 and os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
