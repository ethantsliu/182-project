"""The FP8_ATTN_O GPT state cases (attn_o_state_cases.py), each configuration in its own subprocess.

Why subprocesses: the cases need FP8_ATTN_O / FP8_ATTN_O_SCALES / HADAMARD_SAT_STATS at config import and the Triton
interpreter at kernel decoration; a test module must not change os.environ for the rest of the session (see
182/hadamard-fp8/tests/test_hadamard_state.py). The child gets TRITON_INTERPRET=1, CUDA hidden and the interpreter
numerics shim (182/tools/triton_cpu).

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/fp8-attn-o/tests/test_attn_o_state.py -q
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip("triton", reason="the FP8_ATTN_O state cases run the Triton interpreter")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHIM = ROOT / "182" / "tools" / "triton_cpu"
CASES = {  # case -> (env, minimum passes: every case runs; fewer means something skipped silently)
    "off": ({}, 4),
    "static": ({"FP8_ATTN_O": "1"}, 8),
    "static_sat": ({"FP8_ATTN_O": "1", "HADAMARD_SAT_STATS": "1"}, 8),
    "delayed": ({"FP8_ATTN_O": "1", "FP8_ATTN_O_SCALES": "delayed"}, 8),
}


@pytest.mark.parametrize("case", list(CASES))
def test_attn_o_state_cases(case):
    extra, min_passed = CASES[case]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FP8_", "HADAMARD"))}
    env.update(extra, FP8_ATTN_O_CASE=case, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(SHIM)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    env.pop("TRITON_CPU_NO_SHIM", None)
    p = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "attn_o_state_cases.py"), "-q", "-rs",
                        "-p", "no:cacheprovider"], env=env, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    out = p.stdout + p.stderr
    summary = (out.strip().splitlines() or ["(no output)"])[-1]
    print(f"attn_o_state_cases[{case}]: {summary}")
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
    assert p.returncode == 0 and passed >= min_passed, f"rc={p.returncode}, {passed} passed (want >= {min_passed})\n{out[-6000:]}"
