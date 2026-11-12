"""Compile this tree's GPU kernels for H100 (sm_90a) on a machine without a GPU.

    bash 182/compile_check.sh          # runs this file in the triton-cpu:3.6.0 image (torch 2.10.0, triton 3.6.0)

What it compiles, with the arguments the trainer uses:
  * the fused cross-entropy CUDA kernel, exactly as triton_kernels.py hands it to torch.cuda._compile_kernel
    (NVRTC 12.8, the version torch 2.10.0+cu128 ships, -lineinfo --use_fast_math, sm_90), then ptxas -arch=sm_90a;
  * its prefix-token variant (PREFIX_WEIGHT > 0), the same way;
  * the Triton relu^2 MLP kernel FUSED_MLP=1 uses, forward and backward, at GPT-2 Medium's shapes
    (K=1024, N=4096, every per-rank token count of the batch schedule and the validation batch), through
    Triton's own frontend and the ptxas in its wheel, against a fake sm_90 driver. Shared memory is checked
    against the H100's 232448 B opt-in limit.
Nothing is executed: this proves the sources compile and assemble, not that they compute the right thing
(182/test_cpu.py checks the arithmetic with a Python transliteration) or how fast they are.
"""
import ctypes
import glob
import os
import re
import subprocess
import sys
import tempfile
import types

import torch
import torch.cuda._utils as cuda_utils
import torch.utils.cpp_extension as cpp_extension
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
from triton.runtime.jit import JITFunction

HERE = os.path.dirname(os.path.abspath(__file__))
H100_SMEM = 232448
H100_SMS = 132
PTXAS = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
FAILED = []


def find_nvidia_wheel_dirs():
    """(dir of libnvrtc.so.12, dir holding include/cuda_bf16.h) from the nvidia-*-cu12 pip wheels."""
    lib = inc = None
    for sp in sys.path:
        for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib", "libnvrtc.so.12")):
            lib = p
        for p in glob.glob(os.path.join(sp, "nvidia", "*", "include", "cuda_bf16.h")):
            inc = os.path.dirname(os.path.dirname(p))
    return lib, inc


def ptxas(ptx: bytes, what: str):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.ptx")
        with open(src, "wb") as f:
            f.write(ptx.rstrip(b"\0"))
        out = subprocess.run([PTXAS, "-arch=sm_90a", "-v", src, "-o", os.path.join(d, "k.cubin")],
                             capture_output=True, text=True)
    txt = out.stdout + out.stderr
    if out.returncode:
        FAILED.append(what)
        print(f"FAIL {what}: ptxas rc {out.returncode}: {txt[-800:]}")
        return
    regs = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    print(f"PASS {what}: NVRTC + ptxas sm_90a ok ({len(ptx)} B PTX, {regs.group(1) if regs else '?'} registers, "
          f"{spill.group(1) if spill else '?'} B spill stores)")


class _FakeKernel:
    def set_shared_memory_config(self, n):
        assert n <= H100_SMEM, f"{n} B of shared memory exceeds the H100 limit {H100_SMEM}"

    def __call__(self, *a, **k):
        raise RuntimeError("compile check only: kernels are not launched")


def fake_compile_kernel(source, name, compute_capability=None, cuda_include_dirs=None, nvcc_options=None):
    """torch.cuda._compile_kernel without the module load: the same _nvrtc_compile call, then ptxas."""
    try:
        ptx, mangled = cuda_utils._nvrtc_compile(source, name, compute_capability, cuda_include_dirs, nvcc_options)
    except Exception as e:
        FAILED.append(name)
        print(f"FAIL {name}: NVRTC: {str(e)[:3000]}")
        return _FakeKernel()
    assert mangled == name, (mangled, name)  # extern "C"
    ptxas(ptx, f"CUDA kernel {name}")
    return _FakeKernel()


class _FakeDriver:
    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return GPUTarget("cuda", 90, 32)

    def get_active_torch_device(self):
        return torch.device("cpu")


_orig_run = JITFunction.run
_compiled = []


def _warmup_run(self, *args, grid, warmup, **kwargs):
    _compiled.append(_orig_run(self, *args, grid=grid, warmup=True, **kwargs))
    return None


def main():
    nvrtc, cuda_home = find_nvidia_wheel_dirs()
    if not nvrtc or not cuda_home:
        print("FAIL setup: needs the pip wheels nvidia-cuda-nvrtc-cu12==12.8.93 and nvidia-cuda-runtime-cu12==12.8.90 "
              "(182/compile_check.sh installs them in the container)")
        return 2
    if not torch.cuda.is_available():
        # a CPU torch build has no CUDA version / CUDA_HOME; give _nvrtc_compile what a cu128 build has
        ctypes.CDLL(nvrtc.replace("libnvrtc.so.12", "libnvrtc-builtins.so.12.8"), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(nvrtc)
        cuda_utils._get_gpu_rtc_library = lambda: lib
        torch.version.cuda = "12.8"
        cpp_extension.CUDA_HOME = cuda_home
        torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(multi_processor_count=H100_SMS)
    torch.cuda._compile_kernel = fake_compile_kernel
    print(f"torch {torch.__version__}, triton {triton.__version__}, NVRTC from {nvrtc}")

    sys.path.insert(0, os.path.dirname(HERE))
    import triton_kernels as tk  # compiles the base CE kernel at import, as on the node
    tk.prefix_ce_kernel()

    # Triton MLP kernel at Medium's shapes
    driver.set_active(_FakeDriver())
    JITFunction.run = _warmup_run
    D, H, world = 1024, 4096, 8
    train_tokens = sorted({bs // world for bs in (131072, 262144, 393216, 524288)})  # train_bs_schedule
    val_tokens = 4 * 64 * 1024 * 8 // world
    bf = torch.bfloat16
    w = torch.empty(H, D, dtype=bf)
    for T in train_tokens + [val_tokens]:
        for fwd in (True, False):
            if not fwd and T == val_tokens:
                continue  # validation has no backward
            what = f"Triton linear_relu_square {'forward' if fwd else 'backward'} T={T} K={D} N={H}"
            n0 = len(_compiled)
            try:
                if fwd:
                    tk.linear_relu_square(torch.empty(T, D, dtype=bf), w)
                else:
                    tk.linear_relu_square(torch.empty(T, D, dtype=bf), w, aux=torch.empty(T, H, dtype=bf))
                (k,) = _compiled[n0:]
                smem = k.metadata.shared
                assert smem <= H100_SMEM, f"{smem} B shared memory > H100 limit {H100_SMEM}"
                print(f"PASS {what}: compiled for sm_90, {smem} B shared memory (limit {H100_SMEM})")
            except Exception as e:
                FAILED.append(what)
                print(f"FAIL {what}: {type(e).__name__}: {str(e)[:1500]}")
    print("RESULT:", f"{len(FAILED)} FAILED" if FAILED else "all kernels compiled")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
