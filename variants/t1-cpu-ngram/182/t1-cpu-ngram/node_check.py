#!/usr/bin/env python3
"""Does this node meet the exact-match (CPU n-gram) retrieval's host requirements? Standard library only.

    python3 182/t1-cpu-ngram/node_check.py [--data DATA_PATH] [--gpus 8] [--shared-arena]

Prints the host (CPUs, cores, NUMA, RAM, /dev/shm, THP, AVX-512), the retrieval's estimated host memory for the
shards found (track_1_short/host.py: the model the trainer itself checks before the clock), and one line per check:
FAIL (the run cannot work as configured), WARN (it runs, but not as the records were measured), PASS. Exit status 1
on any FAIL.

The reference: the #380/#381 records ran on 2x Xeon Platinum 8462Y+ (128 vCPU: 16 per GPU, so 32 build threads and
LARGE_HOST off), 2 TB RAM, 8x H100, data in /dev/shm. The large-host mode (>= 24 vCPUs per GPU: 64 build threads, the
corpus load after the 6-token index) and the ~700 GiB / 64-thread targets are the requirements this check names
"recommended"; the hard floor is what the run itself needs.
"""
import argparse
import glob
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from track_1_short import host as H  # noqa: E402

RECOMMENDED_RAM_GIB = 700
RECOMMENDED_THREADS = 64
LARGE_PER_GPU, RECORD_PER_GPU = 24, 16
TRAIN_SHARDS = 103


def gpu_count():
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
            return sum(1 for l in out.splitlines() if l.startswith("GPU "))
        except (OSError, subprocess.SubprocessError):
            pass
    return len(glob.glob("/proc/driver/nvidia/gpus/*"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATA_PATH", "."), help="DATA_PATH (holds data/fineweb10B)")
    ap.add_argument("--gpus", type=int, default=None, help="GPUs to plan for (default: detected; 8 for the plan if none)")
    ap.add_argument("--exclude", type=int, default=2, help="leading shards the loader trains on (the run computes it; 2 at ~700 steps)")
    ap.add_argument("--shared-arena", action="store_true", default=os.environ.get("SLOT_SHARED_ARENA") == "1",
                    help="plan for SLOT_SHARED_ARENA=1")
    a = ap.parse_args()

    detected = gpu_count()
    gpus = a.gpus or detected or 8  # the memory plan needs a count; no detected GPU is still a FAIL below
    h = H.detect(gpus=gpus)
    print(H.describe(h))
    per_gpu = h.allowed // gpus
    threads = int(os.environ.get("VAL_BUILD_THREADS", 0)) or (64 if per_gpu >= 24 else 32 if per_gpu >= 16 else 16)

    files = sorted(glob.glob(os.path.join(a.data, "data/fineweb10B/fineweb_train_*.bin")))
    if files:
        tokens = [H.shard_tokens(f) for f in files[a.exclude:]]
    else:  # no data here: plan for the standard 103 shards of 100M tokens
        tokens = [100_000_000] * (TRAIN_SHARDS - a.exclude)
    need = H.plan(sum(tokens), sum(tokens[:20]), threads, gpus, shared=a.shared_arena)
    print(f"plan: {gpus} GPUs, {threads} build threads, 6-token index over {len(tokens)} shards ({sum(tokens) / 1e9:.2f}B tokens)"
          f"{', shared arena' if a.shared_arena else ''}: arenas {need.arenas / H.GiB:.0f} GiB + tables {need.tables / H.GiB:.0f} GiB "
          f"(/dev/shm) + ranks {need.ranks / H.GiB:.0f} GiB = {need.total / H.GiB:.0f} GiB")

    checks = []  # (level, message)
    def add(ok, fail_level, msg):
        checks.append(("PASS" if ok else fail_level, msg))

    add(platform.system() == "Linux", "FAIL", f"Linux (the index uses Linux mmap/madvise/affinity): {platform.system()}")
    add(platform.machine() in ("x86_64", "AMD64"), "WARN", f"x86-64 (the tuned paths; elsewhere scalar fallbacks): {platform.machine()}")
    add(h.avx512, "WARN", "AVX-512 F/BW/DQ/VL (else AVX2/scalar hashing and plain stores: a slower build)")
    kv = tuple(int(x) for x in (platform.release().split("-")[0].split(".") + ["0", "0"])[:2] if x.isdigit())
    add(platform.system() != "Linux" or kv >= (5, 14), "WARN", f"kernel >= 5.14 (MADV_POPULATE_READ/WRITE; older: page faults on the clock): {platform.release()}")
    add("[never]" not in h.thp, "WARN", f"transparent huge pages enabled or madvise: {h.thp}")
    add(detected > 0, "FAIL", f"GPUs detected (nvidia-smi / /proc/driver/nvidia): {detected}"
        + ("" if detected else f" (the plan above assumes {gpus})"))
    if detected or a.gpus:
        add(gpus == 8, "WARN", f"8 GPUs (the schedule is tuned for 8): {gpus}")
    add(h.allowed >= RECOMMENDED_THREADS, "FAIL", f">= {RECOMMENDED_THREADS} CPUs allowed: {h.allowed}")
    add(per_gpu >= RECORD_PER_GPU, "FAIL", f">= {RECORD_PER_GPU} vCPUs per GPU (the records' box; fewer: 16 build threads, not measured): {per_gpu}")
    add(per_gpu >= LARGE_PER_GPU, "WARN", f">= {LARGE_PER_GPU} vCPUs per GPU (LARGE_HOST: 64 build threads; the records ran with 16 per GPU): {per_gpu}")
    add(h.numa_nodes >= 1, "PASS", f"NUMA nodes: {h.numa_nodes} (each rank pins to the cores local to its GPU)")
    add(h.shm_total > 0 and need.tables <= h.shm_free, "FAIL",
        f"/dev/shm free >= the index tables: {h.shm_free / H.GiB:.0f} GiB free, {need.tables / H.GiB:.0f} GiB needed")
    add(need.total <= h.mem_available, "FAIL", f"RAM available >= the plan: {h.mem_available / H.GiB:.0f} GiB available, {need.total / H.GiB:.0f} GiB needed")
    add(h.mem_total >= RECOMMENDED_RAM_GIB * H.GiB, "WARN", f"RAM >= {RECOMMENDED_RAM_GIB} GiB (headroom; the records' box had 2 TB): {h.mem_total / H.GiB:.0f} GiB")
    stale = glob.glob("/dev/shm/exact-match-*")
    add(not stale, "WARN", f"no stale index tables in /dev/shm (a killed run leaves them): {len(stale)}")
    add(len(files) == TRAIN_SHARDS, "FAIL", f"{TRAIN_SHARDS} training shards under {a.data}/data/fineweb10B (the index asserts it): {len(files)}")
    add(not files or os.path.realpath(files[0]).startswith("/dev/shm"), "WARN", "data in /dev/shm (as the records: warm reads)")
    add(importlib.util.find_spec("exact_match") is not None or shutil.which("cargo") is not None, "FAIL",
        "the exact_match extension is installed, or cargo can build it (pip install ./exact_match)")

    for level, msg in checks:
        print(f"{level:4}  {msg}")
    failed = sum(level == "FAIL" for level, _ in checks)
    print(f"{'NOT READY' if failed else 'READY'}: {failed} FAIL, {sum(l == 'WARN' for l, _ in checks)} WARN")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
