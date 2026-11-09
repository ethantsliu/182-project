"""GPU test (skipped without CUDA): the HADAMARD_FP8 amax buffers under torch.compile(fullgraph) + CUDA graphs.

    HADAMARD_FP8=1 python -m pytest 182/hadamard-fp8/tests/test_gpu_state.py -q -s

The check is gpu_sanity.check_compiled_amax (also part (ii) of gpu_sanity.py). It runs GPT's real
_quantize_mlp_input and the fp8 MLP inside a compiled module, captures forward and backward in a CUDA graph,
and replays it. Only the target slot's x_t / post / post_t / dpre amax may become > 0, each must match its
torch reference, and a replay of the captured refresh must move only that slot's x_t and post_t scales.
"""
import os
import pathlib
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
if os.environ.get("HADAMARD_FP8") != "1":
    pytest.skip("run with HADAMARD_FP8=1 (an import-time flag)", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gpu_sanity  # noqa: E402


def test_compiled_cuda_graph_amax_and_refresh():
    res = gpu_sanity.check_compiled_amax()
    for name, (ok, detail) in res.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    bad = {n: d for n, (ok, d) in res.items() if not ok}
    assert not bad, bad
