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
    """Concatenate the entry script (and train_gpt.py, if the entry script is a wrapper), every module of this package and the exact_match extension's Rust
    source, for the run log.

    Called first thing at startup so the log holds the code as it was at launch.
    """
    root = Path(entry_script).resolve().parent
    # A wrapper entry point (train_gpt_medium_cpu.py) runs train_gpt.py's main: log both.
    outline = root / "train_gpt.py"
    extra = [outline] if outline.exists() and outline != Path(entry_script).resolve() else []
    files = [Path(entry_script)] + extra + sorted(PACKAGE_DIR.rglob("*.py")) + sorted((root / "exact_match" / "src").glob("*.rs"))
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
    print0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    print0("=" * 100)
