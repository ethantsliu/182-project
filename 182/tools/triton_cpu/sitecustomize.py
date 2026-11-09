# Loaded by every python process that has 182/tools/triton_cpu on PYTHONPATH (run.sh sets it): applies the
# interpreter numerics shim and the CPU launch fallbacks. TRITON_CPU_NO_SHIM=1 runs the raw interpreter.
import os

if os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("TRITON_CPU_NO_SHIM") != "1":
    import shim

    shim.install()
    shim.cpu_fallbacks()
