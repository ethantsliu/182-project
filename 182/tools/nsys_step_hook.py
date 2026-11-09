"""Run a trainer with NVTX "step N" ranges around a few steps of its timed loop, and the CUDA profiler
API (cudaProfilerStart/Stop) bracketing them, so `nsys --capture-range=cudaProfilerApi` records only
those steps. The trainer's source is not modified.

    python 182/tools/nsys_step_hook.py train_gpt.py [trainer args]

How: the script is executed as a module named "<stem>_nsys" (so its `if __name__ == "__main__"` guard
does not fire), its step function is wrapped, and its main() is called. sys.argv[0] stays the script
path, so the run log still embeds the trainer's own source.

Env:
  NSYS_HOOK_FN        step function to wrap (default train_step; train_gpt.py's per-step call)
  NSYS_HOOK_STEP_ARG  its argument holding the step index (default step)
  NSYS_HOOK_PHASE     which pass over the steps to profile: the index restarts at the timed loop, after
                      the warmup pass, so 1 = timed loop (default); 0 for a trainer with no warmup pass
  NSYS_START_STEP     first annotated step (default 300)
  NSYS_NUM_STEPS      annotated steps (default 5). Capture spans one unannotated step either side.
  NSYS_HOOK_EMIT_NVTX 1 = also torch.autograd.profiler.emit_nvtx(record_shapes=True) over the captured
                      steps (eager trainers only: GEMM shapes for analyze_nsys.py; CUDA graphs bypass it)

If the script has no such function or no main(), it runs unchanged (as __main__) and the capture has to
be time-windowed instead (profile_nsys.sh --mode window).
"""
import atexit
import functools
import inspect
import os
import runpy
import sys
import types


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    script_dir = os.path.dirname(os.path.abspath(script))
    sys.path.insert(0, script_dir)
    fn_name = os.environ.get("NSYS_HOOK_FN", "train_step")
    arg_name = os.environ.get("NSYS_HOOK_STEP_ARG", "step")
    phase_want = int(os.environ.get("NSYS_HOOK_PHASE", "1"))
    start = int(os.environ.get("NSYS_START_STEP", "300"))
    num = int(os.environ.get("NSYS_NUM_STEPS", "5"))
    emit = os.environ.get("NSYS_HOOK_EMIT_NVTX") == "1"
    rank0 = os.environ.get("RANK", "0") == "0"

    with open(script) as f:
        src = f.read()
    if f"def {fn_name}(" not in src or "def main(" not in src:
        if rank0:
            print(f"[nsys_hook] {script} has no {fn_name}() or main(): running it unhooked "
                  f"(use profile_nsys.sh --mode window)", flush=True)
        runpy.run_path(script, run_name="__main__")
        return

    import torch

    name = os.path.splitext(os.path.basename(script))[0] + "_nsys"
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(script)
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    inner = getattr(mod, fn_name)
    sig = inspect.signature(inner)
    state = dict(last=None, phase=0, on=False, done=False, nvtx_ctx=None)
    cudart = torch.cuda.cudart()

    def stop():
        if state["on"]:
            torch.cuda.synchronize()
            if state["nvtx_ctx"] is not None:
                state["nvtx_ctx"].__exit__(None, None, None)
                state["nvtx_ctx"] = None
            cudart.cudaProfilerStop()
            state["on"], state["done"] = False, True
            if rank0:
                print(f"[nsys_hook] capture stopped", flush=True)

    atexit.register(stop)

    @functools.wraps(inner)
    def wrapped(*args, **kwargs):
        step = sig.bind_partial(*args, **kwargs).arguments.get(arg_name)
        if step is None:
            return inner(*args, **kwargs)
        if state["last"] is not None and step <= state["last"]:
            state["phase"] += 1
        state["last"] = step
        live = state["phase"] == phase_want and not state["done"]
        if live and not state["on"] and step == max(start - 1, 0):
            torch.cuda.synchronize()
            cudart.cudaProfilerStart()
            state["on"] = True
            if emit:
                state["nvtx_ctx"] = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                state["nvtx_ctx"].__enter__()
            if rank0:
                print(f"[nsys_hook] capture started at step {step} (phase {state['phase']})", flush=True)
        if state["on"] and step >= start + num + 1:
            stop()
        if state["on"] and start <= step < start + num:
            torch.cuda.nvtx.range_push(f"step {step}")
            try:
                return inner(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        return inner(*args, **kwargs)

    setattr(mod, fn_name, wrapped)
    mod.main()
    stop()


if __name__ == "__main__":
    main()
