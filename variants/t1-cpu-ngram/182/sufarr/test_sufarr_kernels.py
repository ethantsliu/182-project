"""The GPU scan's own kernels (ExactCounts in track_1_short/exact_counts.py: _ins_kernel, _scan_kernel, _find_kernel)
against the exact reference (sufarr_ref.py) that test_sufarr_counts.py checks the suffix arrays against: the same
cases, one ExactCounts per rank over its share of the same shard files, the ranks' counts summed. With both tests
passing, EXACT_BACKEND=sa and the scan give equal counts on these cases (no 64-bit hash collision occurs here).

Needs real Triton under TRITON_INTERPRET=1 (skipped otherwise). From the repo root:

    bash 182/tools/triton_cpu/run.sh python -m pytest variants/t1-cpu-ngram/182/sufarr/test_sufarr_kernels.py -q"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # variants/t1-cpu-ngram
sys.path.insert(0, HERE)
from sufarr_ref import cases, reference, segment_starts, write_shard  # noqa: E402

try:
    import triton  # noqa: F401
    import triton.language  # noqa: F401
    REAL_TRITON = hasattr(triton, "__version__")
except ImportError:
    REAL_TRITON = False
pytestmark = pytest.mark.skipif(not REAL_TRITON or os.environ.get("TRITON_INTERPRET") != "1",
                                reason="needs real Triton under TRITON_INTERPRET=1 (182/tools/triton_cpu/run.sh)")
HB_TEST = 17  # a small query table (the run's 2^27 buckets are 4 GiB); the cases insert at most ~60k grams
CASES = cases()


class _NoCuda:
    def __init__(self, *a, **k): pass
    def synchronize(self): pass
    def record(self, *a): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_scan_kernels_match_reference(case, tmp_path, monkeypatch):
    for name, value in (("synchronize", lambda *a, **k: None), ("set_device", lambda *a, **k: None), ("Stream", _NoCuda),
                        ("Event", _NoCuda), ("stream", lambda s: _NoCuda())):
        monkeypatch.setattr(torch.cuda, name, value)
    monkeypatch.syspath_prepend(ROOT)
    from track_1_short import exact_counts as ec_mod
    monkeypatch.setattr(ec_mod, "HB", HB_TEST)
    name, world, files, tok, tgt, q_chunk = case
    paths = []
    for r, share in enumerate(files):  # sorted()[r::world] gives share r's files back in order
        for k, toks in enumerate(share):
            paths.append(os.path.join(str(tmp_path), f"fineweb_train_{k * world + r:06d}.bin"))
            write_shard(paths[-1], toks)
    total = torch.zeros(len(tok), 14, dtype=torch.int32)
    for r in range(world):
        ec = ec_mod.ExactCounts(paths, 0, r, world, "cpu", torch.as_tensor(q_chunk), pinned=False, n_val=0)
        ec.load_corpus()
        ec.loaded.wait()
        assert ec.load_error is None, ec.load_error
        ec.q_tok.copy_(torch.as_tensor(tok))
        ec.q_tgt.copy_(torch.as_tensor(tgt))
        ec.launch()
        total += ec.out
    start = segment_starts(tok, q_chunk)
    want = reference([np.concatenate(share) for share in files], tok, tgt, start)
    got = total.numpy()
    bad = np.flatnonzero((got != want).any(1))
    assert not len(bad), f"{name}: {len(bad)} of {len(tok)} rows differ; first {bad[0]}: got {got[bad[0]]}, want {want[bad[0]]}"
