"""Compile the record's MLP / Hadamard Triton kernels for H100 (sm_90a) on a CPU-only machine.

    TRITON_INTERPRET=0 bash 182/tools/triton_cpu/run.sh python 182/tools/triton_cpu/compile_check.py

The real host launchers (hadamard_dual_quant, hadamard_t_quant, quantize_weights_rotated,
quantize_mlp_weights_dual, linear_relu_square) run on CPU tensors; every `kernel[grid](...)` they issue is
turned into a JIT warmup against a fake CUDA driver whose target is sm_90, so the kernels go through the
same frontend, TTGIR/LLVM passes and the bundled ptxas as on the node -- with exactly the descriptors,
dtypes and constexprs the launchers build. Nothing runs. Per variant it reports shared memory per
num_stages (H100 opt-in limit 232448 B; linear_relu_square steps num_stages down from its default on
OutOfResources) and ptxas registers / spills.

Compile-tested only: no numerics (that is the interpreter's job) and no timing.
"""
import os
import re
import subprocess
import sys
import tempfile
import traceback

assert os.environ.get("TRITON_INTERPRET", "0") != "1", "compile_check needs TRITON_INTERPRET=0"

import torch  # noqa: E402
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.runtime import driver  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H100_SMEM = 232448
H100_SMS = 132
TARGET = GPUTarget("cuda", 90, 32)


class _FakeDriver:
    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return TARGET

    def get_active_torch_device(self):
        return torch.device("cpu")


RECORDS = []  # (kernel name, variant key, num_stages, CompiledKernel or exception)
_orig_run = JITFunction.run


def _warmup_run(self, *args, grid, warmup, **kwargs):
    """Compile instead of launching. For linear_relu_square_kernel, compile every num_stages 1..4."""
    stages = [kwargs.get("num_stages", 3)]
    if self.fn.__name__ == "linear_relu_square_kernel":
        stages = [4, 3, 2, 1]
    key = {k: v for k, v in kwargs.items() if k.isupper() and k not in ("NUM_SMS", "BLOCK_SIZE_M", "BLOCK_SIZE_N")}
    for st in stages:
        kw = dict(kwargs, num_stages=st)
        try:
            k = _orig_run(self, *args, grid=grid, warmup=True, **kw)
            RECORDS.append((self.fn.__name__, key, st, k))
        except Exception as e:  # a frontend / compiler error is the finding
            RECORDS.append((self.fn.__name__, key, st, e))
    return None


def ptxas_stats(kernel):
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.ptx")
        with open(src, "w") as f:
            f.write(kernel.asm["ptx"])
        out = subprocess.run([ptxas, "-arch=sm_90a", "-v", src, "-o", os.path.join(d, "k.cubin")],
                             capture_output=True, text=True)
    txt = out.stdout + out.stderr
    regs = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", txt)
    return (int(regs.group(1)) if regs else None, int(spill.group(1)) if spill else None,
            int(spill.group(2)) if spill else None)


def run_launchers():
    """Call every launcher the way training does (shapes: one 768-wide MLP slot, 1024 tokens), for every
    rotation block (HADAMARD_BLOCK 32 / 64 / 128) and every HADAMARD_GEMMS subset the kernels distinguish."""
    import shim
    shim.cpu_fallbacks(H100_SMS)
    from track_1_short.perf.hadamard_ref import signed_hadamard
    from track_1_short.perf.kernels import hadamard as hd
    from track_1_short.perf.kernels import mlp

    E4, E5, BF = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16
    T, D, H = 1024, 768, 3072
    one = torch.ones(1)
    a, b = torch.zeros(T, D, dtype=BF), torch.zeros(H, D, dtype=BF)
    amax = torch.zeros(H100_SMS)
    calls = []
    for hb in (32, 64, 128):
        q = signed_hadamard(hb, 1234)
        for emit_row, rot_row, rot_t, rec, e5 in ((1, 1, 1, 0, 1), (1, 1, 0, 0, 1), (0, 0, 1, 0, 1), (0, 0, 1, 1, 0)):
            calls.append((f"_hadamard_quant_dual_kernel HB={hb}", lambda q=q, hb=hb, f=(emit_row, rot_row, rot_t, rec, e5):
                          hd._hadamard_quant_dual_kernel[(T // 128, D // 128)](
                              a, q, torch.empty(T, D, dtype=E5 if f[4] else E4), torch.empty(D, T, dtype=E5 if f[4] else E4), one, one, one, T, D, D,
                              EMIT_ROW=bool(f[0]), ROT_ROW=bool(f[1]), ROT_T=bool(f[2]), RECORD_T_AMAX=bool(f[3]),
                              E5M2=bool(f[4]), BLOCK=128, HB=hb, num_warps=8)))
        for along_h in (True, False):
            bank = torch.zeros(2, 2, H, D, dtype=BF)[:, 1]
            nt = hd.quantize_weights_rotated_ntiles(H, D)
            out = torch.empty((2, D, H) if along_h else (2, H, D), dtype=E4)
            calls.append((f"_quantize_weights_rotated_kernel HB={hb} along_h={along_h}",
                          lambda q=q, hb=hb, bank=bank, nt=nt, out=out, along_h=along_h:
                          hd._quantize_weights_rotated_kernel[(2, nt)](
                              bank, q, out, torch.ones(2), torch.zeros(2, nt), *bank.stride(), H=H, D=D,
                              num_tiles_d=D // 128, num_tiles=nt, ALONG_H=along_h, BLOCK=128, HB=hb,
                              num_warps=8, num_stages=1)))
    bank = torch.zeros(2, 2, H, D, dtype=BF)[:, 0]
    calls.append(("quantize_mlp_weights_dual", lambda: mlp.quantize_mlp_weights_dual(
        bank, torch.ones(2), torch.zeros(2, mlp.quantize_weights_dual_ntiles(H, D)), torch.empty(2, H, D, dtype=E4),
        torch.empty(2, D, H, dtype=E4), False)))
    fwd = dict(a_f8=torch.zeros(T, D, dtype=E4), b_f8=torch.zeros(H, D, dtype=E4), dequant_scale_ptr=one,
               emit_f8=True, emit_t=True, post_scale=one, post_amax=amax, store_pre=False, store_post_bf=False)
    bwd = dict(aux=torch.zeros(T, H, dtype=E4), a_f8=torch.zeros(T, D, dtype=E5), b_f8=torch.zeros(H, D, dtype=E4),
               dequant_scale_ptr=one, post_scale=one, dpre_scale=one, dpre_amax=amax)
    calls.append(("linear_relu_square bf16 forward", lambda: mlp.linear_relu_square(a, b)))
    calls.append(("linear_relu_square fp8 forward (record)", lambda: mlp.linear_relu_square(a, b, **fwd)))
    calls.append(("linear_relu_square fp8 backward (record)", lambda: mlp.linear_relu_square(a, b, **bwd)))
    for hb in (32, 64, 128):
        q = signed_hadamard(hb, 1234)
        calls.append((f"linear_relu_square fp8 forward ROT_T HB={hb}", lambda q=q: mlp.linear_relu_square(
            a, b, **fwd, hadamard_q=q, rot_t=True, post_t_scale=one, post_t_amax=amax)))
        for rr, rt in ((True, True), (True, False), (False, True)):
            calls.append((f"linear_relu_square fp8 backward ROT_ROW={int(rr)} ROT_T={int(rt)} HB={hb}",
                          lambda q=q, rr=rr, rt=rt: mlp.linear_relu_square(a, b, **bwd, hadamard_q=q, rot_row=rr,
                                                                           rot_t=rt)))
    results = []
    for name, fn in calls:
        n0 = len(RECORDS)
        try:
            fn()
            err = None
        except Exception as e:  # host-side failure (assert, shape) -- also a finding
            err = e
        results.append((name, RECORDS[n0:], err))
    return results


def main():
    driver.set_active(_FakeDriver())
    JITFunction.run = _warmup_run
    results = run_launchers()
    failed = False
    for name, recs, err in results:
        print(f"== {name}")
        if err is not None:
            failed = True
            print(f"   HOST ERROR: {type(err).__name__}: {err}")
        for kname, key, st, k in recs:
            flags = " ".join(f"{a}={int(v) if isinstance(v, bool) else v}" for a, v in key.items())
            if isinstance(k, Exception):
                failed = True
                msg = str(k).strip().splitlines()
                print(f"   {kname} stages={st}: COMPILE ERROR {type(k).__name__}: {msg[-1] if msg else ''}")
                if os.environ.get("COMPILE_CHECK_TRACE"):
                    traceback.print_exception(k)
                continue
            smem = k.metadata.shared
            regs, ss, sl = ptxas_stats(k)
            fits = "fits" if smem <= H100_SMEM else "EXCEEDS 227KB"
            print(f"   {kname} stages={st} {flags}: smem {smem} B ({fits}), {regs} regs, spill st/ld {ss}/{sl} B")
    print("RESULT:", "FAILURES" if failed else "all variants compiled")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
