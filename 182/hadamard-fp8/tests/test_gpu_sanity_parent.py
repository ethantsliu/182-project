"""gpu_sanity.py's parent-side logic on CPU (no torch, no triton): the exit decision and the Triton kernel
introspection against stand-ins shaped like triton 3.6's JITFunction / CompiledKernel."""
import os
import sys
from collections import namedtuple
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sanity  # noqa: E402

MD = namedtuple("KernelMetadata", "name num_stages num_warps shared target")


class FakeCompiled:
    def __init__(self, regs, spills, constants, fail=False):
        self.metadata = MD("k", 3, 8, 200_000, None)
        self.src = SimpleNamespace(fn=SimpleNamespace(arg_names=["x", "FWD", "HAD", "BLOCK"]), constants=constants)
        self._regs, self._spills, self._fail = regs, spills, fail

    def _init_handles(self):
        if self._fail:
            raise RuntimeError("out of resources: shared memory")
        self.n_regs, self.n_spills = self._regs, self._spills


def fake_jit(name, compiled, dev=0):
    fn = SimpleNamespace(__name__=name)
    fn.device_caches = {dev: ({i: c for i, c in enumerate(compiled)}, {}, None, None, None)}
    return fn


def test_kernel_resources_triton36_layout():
    k = fake_jit("linear_relu_square_kernel", [FakeCompiled(255, 120, {(1,): True, (2,): True, (3,): 128}),
                                               FakeCompiled(168, 0, {(1,): False, (2,): False})])
    rows = gpu_sanity.kernel_resources([k], dev=0)
    assert rows[0] == ["linear_relu_square_kernel", "FWD,HAD", 3, 8, 200_000, 255, 120]
    assert rows[1][1] == "" and rows[1][5:] == [168, 0]


def test_kernel_resources_never_raises():
    autotuned = SimpleNamespace(__name__="wrapped", fn=fake_jit("inner", [FakeCompiled(64, 0, {})]))
    rows = gpu_sanity.kernel_resources([
        autotuned,                                               # @autotune-style wrapper: unwrapped
        fake_jit("other_device", [FakeCompiled(1, 0, {})], dev=3),  # nothing compiled on this device
        SimpleNamespace(__name__="no_caches"),                   # some other triton
        fake_jit("too_big", [FakeCompiled(1, 0, {}, fail=True)]),
        fake_jit("weird_src", [SimpleNamespace(_init_handles=lambda: None, metadata=MD("k", 2, 4, 1, None),
                                               n_regs=32, n_spills=0, src="ptx text")]),
    ], dev=0)
    assert rows[0][5:] == [64, 0]
    assert rows[1][0] == "no_caches" and rows[1][1].startswith("unavailable: AttributeError")
    assert rows[2][0] == "too_big" and rows[2][1].startswith("unavailable: RuntimeError")
    assert rows[3] == ["weird_src", "", 2, 4, 1, 32, 0]
    assert len(rows) == 4


def test_failures_counts_crashed_workers_not_only_checks():
    ok_on = {"amax_check": {"a": [True, ""]}, "bench": {"T": {}}}
    assert gpu_sanity.failures({"bench": {"T": {}}}, ok_on) == []
    assert gpu_sanity.failures({}, {"amax_check": {"a": [False, "x"]}}) == ["amax check FAIL: a"]
    # a worker that died (before this fix the exit code was 0: only amax checks counted)
    bad = gpu_sanity.failures({"bench": {}}, {"error": "worker HADAMARD_FP8=1 exited -11 without a result"})
    assert bad == ["worker on: worker HADAMARD_FP8=1 exited -11 without a result"]
    assert gpu_sanity.failures({"bench": {"error": "OutOfMemoryError: x"}}, ok_on) == ["bench off: OutOfMemoryError: x"]
    assert gpu_sanity.failures({}, {"bench": {}}) == ["the HADAMARD_FP8=1 worker returned no amax check"]
    assert gpu_sanity.failures({}, {"bench": {}}, skip_compile=True) == []
