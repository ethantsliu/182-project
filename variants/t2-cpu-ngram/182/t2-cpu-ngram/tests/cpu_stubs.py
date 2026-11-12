"""CPU stand-ins so the model and schedules can be built (not run) on a machine without CUDA or Triton.

triton, the FA3 hub kernel, the NVRTC CE kernel and the exact_match Rust extension are replaced by inert
objects; every kernel stays unlaunched. Enough for GPT(...) on the meta device, the schedules and the
sampled-softmax switch table. Import this before anything from track_1_short.
"""
import hashlib
import importlib.abc
import importlib.machinery
import sys
import tempfile
import types
from pathlib import Path


class Anything:
    """Any attribute, call or subscript; called with one function (a decorator), returns it unchanged."""
    def __init__(self, name="stub"):
        self._name = name

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k and not isinstance(a[0], Anything):
            return a[0]
        return Anything(self._name + "()")

    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return Anything(f"{self._name}.{n}")

    def __getitem__(self, k):
        return Anything(self._name + "[]")

    def __iter__(self):  # empty: with __getitem__ alone, iterating (torch's has_triton() does) never ends
        return iter(())

    def __mro_entries__(self, bases):
        return (object,)

    def __or__(self, o):
        return self
    __ror__ = __or__


class _StubModule(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return Anything(f"{self.__name__}.{n}")


class _TritonFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == "triton":
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, m):
        if m.__name__ == "triton":
            m.__version__ = "0.0-stub"
            m.cdiv = lambda a, b: -(-a // b)
            m.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()


def _install_fa3(expected_sha: str):
    """model/attention.py loads FA3 from the hub and checks its binary's sha256: hand it a dummy binary and
    accept that one file's digest only."""
    fa3_dir = Path(tempfile.mkdtemp(prefix="fa3stub"))
    (fa3_dir / "__init__.py").write_text("")
    marker = b"CPU-STUB-FA3"
    (fa3_dir / "_flash_attn3_cuda_stub.so").write_bytes(marker)

    class _FA3:
        __file__ = str(fa3_dir / "__init__.py")
        flash_attn_interface = Anything("flash_attn_interface")

    kernels = types.ModuleType("kernels")
    kernels.get_kernel = lambda *a, **k: _FA3()
    sys.modules["kernels"] = kernels
    real = hashlib.sha256

    class _Sha:
        def __init__(self, data=b""):
            self._data, self._h = data, real(data)

        def hexdigest(self):
            return expected_sha if self._data == marker else self._h.hexdigest()

        def __getattr__(self, n):
            return getattr(self._h, n)
    hashlib.sha256 = _Sha


def install_all(variant_root: Path):
    """Stubs for triton, exact_match, FA3 and NVRTC; puts variant_root on sys.path."""
    if getattr(install_all, "done", False):
        return
    install_all.done = True
    sys.meta_path.insert(0, _TritonFinder())
    em = types.ModuleType("exact_match")
    em.CELLS = 4 * 7 * 6  # exact_match/src/lib.rs: (LEVELS + 2) * (COUNTS + 1) * (PURITIES + 1)
    em.SlotIndex = em.StreamIndex = Anything("exact_match.index")
    sys.modules["exact_match"] = em
    src = (variant_root / "track_1_short" / "model" / "attention.py").read_text()
    sha = src.split('FA3_BINARY_SHA256 = "', 1)[1].split('"', 1)[0]
    _install_fa3(sha)
    import torch.cuda
    torch.cuda._compile_kernel = lambda *a, **k: Anything("nvrtc_kernel")
    sys.path.insert(0, str(variant_root))
