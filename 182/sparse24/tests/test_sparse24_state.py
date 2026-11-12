"""The SPARSE24_MLP GPT state cases (sparse24_state_cases.py), each configuration in its own subprocess.

Why subprocesses: the cases need SPARSE24_* at config import and the Triton interpreter at kernel decoration; a
test module must not change os.environ for the rest of the session (as 182/fp8-attn-o/tests/test_attn_o_state.py).

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/sparse24/tests/test_sparse24_state.py -q
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip("triton", reason="the SPARSE24 state cases run the Triton interpreter")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHIM = ROOT / "182" / "tools" / "triton_cpu"
DENSE = {"SPARSE24_BACKEND": "dense"}
CASES = {  # case -> (env, minimum passes: fewer means something skipped silently)
    "off": (DENSE, 6),
    "greedy": ({**DENSE, "SPARSE24_MLP": "1"}, 10),
    "optimal": ({**DENSE, "SPARSE24_MLP": "1", "SPARSE24_MASK": "optimal"}, 10),
    "tail13": ({**DENSE, "SPARSE24_MLP": "1", "SPARSE24_DENSE_FROM_WS": "13"}, 10),
}


@pytest.mark.parametrize("case", list(CASES))
def test_sparse24_state_cases(case):
    extra, min_passed = CASES[case]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FP8_", "HADAMARD", "SPARSE24", "MONARCH"))}
    env.update(extra, SPARSE24_CASE=case, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(SHIM)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    env.pop("TRITON_CPU_NO_SHIM", None)
    p = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "sparse24_state_cases.py"), "-q", "-rs",
                        "-p", "no:cacheprovider"], env=env, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    out = p.stdout + p.stderr
    summary = (out.strip().splitlines() or ["(no output)"])[-1]
    print(f"sparse24_state_cases[{case}]: {summary}")
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
    assert p.returncode == 0 and passed >= min_passed, f"rc={p.returncode}, {passed} passed (want >= {min_passed})\n{out[-6000:]}"
