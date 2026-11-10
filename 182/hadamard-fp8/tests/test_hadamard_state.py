"""The HADAMARD_FP8 fp8 MLP state cases (hadamard_state_cases.py), run in a subprocess.

Why a subprocess: the cases need HADAMARD_FP8=1 (read when track_1_short.config is imported) and the Triton
interpreter (chosen when each kernel is decorated), and the rest of a pytest session needs whatever the
caller set. Setting either in os.environ from a test module leaks into every module collected after it:
on an H100 with TRITON_INTERPRET unset, the interpreter-only modules would then run against compiled
kernels and fail. So this module never touches os.environ; the child gets TRITON_INTERPRET=1,
HADAMARD_FP8=1, CUDA hidden (the cases are CPU-only, on the interpreter, with gpt.py's CUDA-only modules
stubbed) and the interpreter numerics shim (182/tools/triton_cpu). HADAMARD_GEMMS / HADAMARD_BLOCK pass
through. The CUDA side of the same state (inductor + CUDA graphs) is test_gpu_state.py.

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/hadamard-fp8/tests/test_hadamard_state.py -q
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip("triton", reason="the HADAMARD_FP8 state cases run the Triton interpreter")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHIM = ROOT / "182" / "tools" / "triton_cpu"
MIN_PASSED = 14  # every case runs in the child; fewer passes means something skipped silently


def test_hadamard_state_cases():
    env = dict(os.environ, TRITON_INTERPRET="1", HADAMARD_FP8="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(SHIM)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("TRITON_CPU_NO_SHIM", None)
    p = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "hadamard_state_cases.py"), "-q", "-rs",
                        "-p", "no:cacheprovider"], env=env, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    out = p.stdout + p.stderr
    summary = (out.strip().splitlines() or ["(no output)"])[-1]
    print(f"hadamard_state_cases: {summary}")
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
    assert p.returncode == 0 and passed >= MIN_PASSED, f"rc={p.returncode}, {passed} passed (want >= {MIN_PASSED})\n{out[-6000:]}"
