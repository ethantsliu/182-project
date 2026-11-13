"""The run log: every record's log file starts with the full source code and the environment."""
import atexit
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton

PACKAGE_DIR = Path(__file__).resolve().parent
LOG_BUFFER_BYTES = 1 << 20


def read_source(entry_script: str) -> str:
    """Concatenate the entry script, every module of this package and the exact_match extension's Rust
    source, for the run log.

    Called first thing at startup so the log holds the code as it was at launch.
    """
    root = Path(entry_script).resolve().parent
    files = [Path(entry_script)] + sorted(PACKAGE_DIR.rglob("*.py")) + sorted((root / "exact_match" / "src").glob("*.rs"))
    chunks = []
    for path in files:
        text = path.read_text()
        if chunks:
            chunks.append(f"\n\n{'-'*40}\n# {path.resolve().relative_to(root)}\n{'-'*40}\n\n")
        chunks.append(text)
    return "".join(chunks)


def start_run_log(master_process: bool, run_id: str):
    """Create TRAIN_LOG_DIR/<run_id>.txt (default logs/) on rank 0 and return (print0, flush).

    print0(s, console=False) always appends to the log file; with console=True it also prints to
    stdout. The file is one block-buffered handle rather than an open() + close() per line (record #360):
    call flush() where the clock is stopped; it is also flushed at exit. A forked child (the canonical
    mask build) leaves with os._exit, which never flushes the inherited buffer, so no line is written twice.
    """
    logfile = None
    if master_process:
        log_dir = os.environ.get("TRAIN_LOG_DIR") or "logs"
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{run_id}.txt")
        try:
            logfile = open(path, "a", buffering=LOG_BUFFER_BYTES)
        except OSError as exc:
            raise OSError(f"cannot write trainer log {path}: {exc}; set TRAIN_LOG_DIR to a writable directory") from exc
        print(path, flush=True)
        print(f"TRAIN_LOG_PATH={os.path.abspath(path)}", flush=True)
        atexit.register(logfile.close)  # close() flushes

    def print0(s, console=False):
        if master_process:
            if console:
                print(s)
            print(s, file=logfile)

    def flush():
        if logfile is not None:
            logfile.flush()

    return print0, flush


def log_environment(print0, code: str) -> None:
    print0(code)
    print0("=" * 100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0(f"Running Triton version {triton.__version__}")
    print0(attention_kernel_line(), console=True)
    print0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    print0("=" * 100)


def attention_kernel_line() -> str:
    """One line naming the attention kernel the model calls (model/attention.py's flash_attn_interface, whatever
    loaded it) and whether it is the pinned FA3 binary, so a run on another loader or kernel is visible in its log
    (182/node.py puts pinned= into the RESULT line)."""
    import hashlib
    import importlib.metadata
    import pathlib
    try:
        kernels = importlib.metadata.version("kernels")
    except importlib.metadata.PackageNotFoundError:
        kernels = "none"
    try:
        from track_1_short.model import attention
        fai = attention.flash_attn_interface
        where = pathlib.Path(getattr(fai, "__file__", "") or ".").parent
        binaries = sorted(where.glob("_flash_attn3_cuda_*.so"))
        digest = "none"
        if binaries:
            b = str(binaries[0])
            digest = attention.FA3_DIGESTS.get(b) or hashlib.sha256(pathlib.Path(b).read_bytes()).hexdigest()
        pinned = int(digest == attention.FA3_BINARY_SHA256)
        return (f"attention kernel: pinned={pinned} module={getattr(fai, '__name__', type(fai).__name__)} "
                f"binary={binaries[0].name if binaries else 'none'} sha256={digest} repo={attention.FA3_REPO}@{attention.FA3_REVISION[:12]} "
                f"kernels={kernels}")
    except Exception as e:  # a report: it must not cost the run
        return f"attention kernel: pinned=0 unknown ({type(e).__name__}: {e}) kernels={kernels}"
