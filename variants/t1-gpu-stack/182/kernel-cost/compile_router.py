"""Compile MOE_ROUTER_FUSED's two Triton kernels for H100 (sm_90a) on a CPU-only machine; nothing runs.

    TRITON_INTERPRET=0 bash 182/tools/triton_cpu/run.sh python variants/t1-gpu-stack/182/kernel-cost/compile_router.py

(The root `run.sh aot` covers the root MLP kernels only, not these.)

The launchers in perf/kernels/moe_router.py run on CPU tensors; every `kernel[grid](...)` they issue becomes a JIT
warmup against a fake CUDA driver whose target is sm_90 (the mechanism of 182/tools/triton_cpu/compile_check.py), so
each kernel goes through the frontend, TTGIR/LLVM and the wheel's ptxas with the record's constexprs: D = 768, E = 8,
and every per-rank token count of training (16384, 32768, 40960, 49152). Prints, per kernel and token count: shared
memory (H100 limit 232448 B), registers, spills, and whether the token counts share one binary. Exit 1 on any error.
Says nothing about speed or numerics (bench_router.py and test_moe_router.py).
"""
import os
import re
import subprocess
import sys
import tempfile

assert os.environ.get("TRITON_INTERPRET", "0") != "1", "compile_router needs TRITON_INTERPRET=0"

import torch  # noqa: E402
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.runtime import driver  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

H100_SMEM = 232448
TARGET = GPUTarget("cuda", 90, 32)
D, E = 768, 8
TOKENS = (16384, 32768, 40960, 49152)


class _FakeDriver:
    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return TARGET

    def get_active_torch_device(self):
        return torch.device("cpu")


RECORDS = []
_orig_run = JITFunction.run


def _warmup_run(self, *args, grid, warmup, **kwargs):
    try:
        RECORDS.append((self.fn.__name__, _orig_run(self, *args, grid=grid, warmup=True, **kwargs)))
    except Exception as e:  # a frontend / compiler error is the finding
        RECORDS.append((self.fn.__name__, e))
    return None


def _ptxas(kernel):
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.ptx")
        with open(src, "w") as f:
            f.write(kernel.asm["ptx"])
        out = subprocess.run([ptxas, "-arch=sm_90a", "-v", src, "-o", os.path.join(d, "k.cubin")],
                             capture_output=True, text=True)
    txt = out.stdout + out.stderr
    if out.returncode:
        raise RuntimeError(f"ptxas failed: {txt[-400:]}")
    regs = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", txt)
    return int(regs.group(1)) if regs else -1, int(spill.group(1)) if spill else -1, int(spill.group(2)) if spill else -1


def main():
    driver.set_active(_FakeDriver())
    JITFunction.run = _warmup_run
    from track_1_short.perf.kernels.moe_mlp import padded_rows
    from track_1_short.perf.kernels.moe_router import _dx_combine_drouter_impl, _router_topk_impl
    for T in TOKENS:
        bf = torch.bfloat16
        rows, w = torch.empty(T, D, dtype=bf), torch.empty(E, D, dtype=bf)
        _router_topk_impl(rows, w)
        m = padded_rows(T, E)
        _dx_combine_drouter_impl(torch.empty(m, D, dtype=bf), torch.empty(T, dtype=torch.int32),
                                 torch.empty(T, E, dtype=bf), w, rows)
    for rpp in (128, 256):      # bench_router.py's alternatives, at the stage-2 token count
        T = TOKENS[-1]
        bf = torch.bfloat16
        _dx_combine_drouter_impl(torch.empty(padded_rows(T, E), D, dtype=bf), torch.empty(T, dtype=torch.int32),
                                 torch.empty(T, E, dtype=bf), torch.empty(E, D, dtype=bf), torch.empty(T, D, dtype=bf),
                                 rows_per_prog=rpp)
    failed = False
    hashes = {}
    for i, (name, k) in enumerate(RECORDS):
        T = TOKENS[i // 2] if i < 2 * len(TOKENS) else f"{TOKENS[-1]} rows_per_prog={(128, 256)[i - 2 * len(TOKENS)]}"
        if isinstance(k, Exception):
            failed = True
            print(f"FAIL {name} T={T}: {type(k).__name__}: {str(k)[:300]}")
            continue
        regs, st, ld = _ptxas(k)
        smem = k.metadata.shared
        if i < 2 * len(TOKENS):
            hashes.setdefault(name, set()).add(k.hash)
        ok = smem <= H100_SMEM and st == 0 and ld == 0
        failed |= smem > H100_SMEM
        print(f"{'OK  ' if ok else 'WARN'} {name} T={T}: shared={smem} B, regs={regs}, spill stores/loads={st}/{ld}, "
              f"warps={k.metadata.num_warps}, stages={k.metadata.num_stages}")
    for name, hs in hashes.items():
        print(f"{name}: {len(hs)} binary(ies) over T={TOKENS}")
    print("RESULT:", "FAILURES" if failed else "all router kernels compiled for sm_90a")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
