#!/usr/bin/env python3
"""Is NVLink SHARP (NVLS) usable on this node, and what does it do to NCCL bandwidth? For gpu_session.sh's nccl phase.

    python 182/tools/nvls_check.py [--nproc 8] [--json OUT] [--no-probe] [--timeout-s 300]

1. `nvidia-smi topo -m` (printed as is) and the Fabric lines of `nvidia-smi -q` (NVLS on HGX H100 needs the
   NVSwitches and a running fabric manager: State Completed, Status Success).
2. CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED for every GPU through libcuda (ctypes; the attribute id is read from
   $CUDA_HOME/include/cuda.h when present, else 132). NVLS is built on multicast objects.
3. Unless --no-probe: three short torchrun runs of this file (--probe) with NCCL_DEBUG=INFO, each timing bf16
   all_reduce / reduce_scatter / all_gather (64 MiB, and a 4 MiB all_reduce) and printing bus bandwidth:
     default   NCCL as the trainers get it (NCCL_NVLS_ENABLE unset = 2: NVLS used where NCCL supports it)
     nvls_off  NCCL_NVLS_ENABLE=0
     forced    NCCL_NVLS_ENABLE=1 NCCL_ALGO="allreduce:nvls;reducescatter:nvls;allgather:nvls" (2.24+ syntax)
   The NCCL log lines that mention NVLS are kept.
Verdict (last line, and "verdict" in the JSON): `NVLS: available` when every GPU reports multicast and the forced
probe ran; `NVLS: unavailable -- <why>` when multicast is off on any GPU, the fabric manager is not up, or the
forced probe failed; `NVLS: unknown -- <why>` otherwise (e.g. --no-probe and no libcuda). Exit 0 / 1 / 3.

The NCCL here is torch 2.10.0+cu128's: nvidia-nccl-cu12==2.27.5 (its PyPI requirements). In 2.27 and 2.28 a
failed NVLS buffer allocation falls back silently and can hang if it failed on only some ranks (NCCL's docs).
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

DEFAULT_ATTR = 132  # CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED in cuda.h (CUDA 12.x)
FORCED_ALGO = "allreduce:nvls;reducescatter:nvls;allgather:nvls"
CONFIGS = [("default", {}), ("nvls_off", {"NCCL_NVLS_ENABLE": "0"}),
           ("forced", {"NCCL_NVLS_ENABLE": "1", "NCCL_ALGO": FORCED_ALGO})]
NVLS_NO_RE = re.compile(r"NVLS.*(not (available|supported)|disabl|fail)|multicast.*not", re.I)


def sh(*cmd, timeout=60) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{cmd[0]}: {e}"


def multicast_attr_id() -> int:
    for base in filter(None, [os.environ.get("CUDA_HOME"), "/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*"))):
        try:
            m = re.search(r"CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED\s*=\s*(\d+)", open(os.path.join(base, "include/cuda.h")).read())
        except OSError:
            continue
        if m:
            return int(m.group(1))
    return DEFAULT_ATTR


def multicast_support() -> tuple[list[int] | None, str]:
    """[0/1 per GPU] or None, and a note."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return None, f"libcuda.so.1 not loadable ({e})"
    attr = multicast_attr_id()
    if cu.cuInit(0) != 0:
        return None, "cuInit failed"
    n = ctypes.c_int()
    if cu.cuDeviceGetCount(ctypes.byref(n)) != 0:
        return None, "cuDeviceGetCount failed"
    out = []
    for i in range(n.value):
        dev, v = ctypes.c_int(), ctypes.c_int()
        if cu.cuDeviceGet(ctypes.byref(dev), i) != 0 or cu.cuDeviceGetAttribute(ctypes.byref(v), attr, dev) != 0:
            return None, f"cuDeviceGetAttribute({attr}) failed on GPU {i}"
        out.append(int(v.value))
    return out, f"attribute id {attr}"


def fabric_state(q: str) -> dict:
    """The Fabric sections of `nvidia-smi -q` (one per GPU): {'state': [...], 'status': [...]}."""
    state, status, ind = [], [], None
    for ln in q.splitlines():
        lead = len(ln) - len(ln.lstrip())
        if ln.strip() == "Fabric":
            ind = lead
            continue
        if ind is not None:
            if not ln.strip() or lead <= ind:
                ind = None
                continue
            k, _, v = ln.partition(":")
            if k.strip() == "State":
                state.append(v.strip())
            elif k.strip() == "Status":
                status.append(v.strip())
    return dict(state=state, status=status)


def verdict(mc: list[int] | None, fab: dict, probes: dict) -> tuple[str, str]:
    if mc is not None and not all(mc):
        return "unavailable", f"multicast not supported on GPU(s) {[i for i, v in enumerate(mc) if not v]}"
    if fab["state"] and any(s.lower() not in ("completed", "n/a") for s in fab["state"]):
        return "unavailable", f"fabric state {sorted(set(fab['state']))} (fabric manager not up?)"
    if fab["status"] and any(s.lower() not in ("success", "n/a") for s in fab["status"]):
        return "unavailable", f"fabric status {sorted(set(fab['status']))}"
    f = probes.get("forced")
    if f is not None:
        if not f.get("ok"):
            return "unavailable", f"the forced-NVLS probe failed: {f.get('error', '?')}"
        if f.get("nvls_negative"):
            return "unavailable", f"NCCL says: {f['nvls_negative'][0][:200]}"
        if mc is None:
            return "available", "the forced-NVLS probe ran (multicast attribute not read)"
        return "available", "multicast on every GPU and the forced-NVLS probe ran"
    if mc is not None:
        return "unknown", "multicast on every GPU, but no NCCL probe ran (--no-probe)"
    return "unknown", "no libcuda and no probe"


def run_probe(nproc: int, extra: dict, timeout_s: float, torchrun: str) -> dict:
    env = dict(os.environ, NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NVLS,TUNING", **extra)
    cmd = [torchrun, "--standalone", f"--nproc_per_node={nproc}", os.path.abspath(__file__), "--probe"]
    try:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    except OSError as e:
        return dict(ok=False, error=f"cannot run {torchrun}: {e}")
    try:
        out, _ = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return dict(ok=False, error=f"timed out after {timeout_s:.0f} s (an NVLS allocation hang?)", log_tail=out[-2000:])
    m = re.findall(r"^PROBE_JSON (.*)$", out, re.M)
    nv = sorted({re.sub(r"^\S+:\d+:\d+ \[\d+\] ", "", ln).strip() for ln in out.splitlines() if "NVLS" in ln})
    res = dict(ok=p.returncode == 0 and bool(m), rc=p.returncode, nvls_lines=nv[:40],
               nvls_negative=[ln for ln in nv if NVLS_NO_RE.search(ln)][:10])
    if m:
        res["bw"] = json.loads(m[-1])
    if not res["ok"]:
        tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
        res["error"] = f"rc {p.returncode}: " + " | ".join(tail)[:600]
    return res


def probe():  # under torchrun
    import torch
    import torch.distributed as dist
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    ws, rank = dist.get_world_size(), dist.get_rank()
    res = {}
    for op, mib in (("all_reduce", 64), ("reduce_scatter", 64), ("all_gather", 64), ("all_reduce", 4)):
        n = mib * 2**20 // 2
        x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
        y = torch.empty(n // ws, dtype=torch.bfloat16, device="cuda")
        f = {"all_reduce": lambda: dist.all_reduce(x),
             "reduce_scatter": lambda: dist.reduce_scatter_tensor(y, x),
             "all_gather": lambda: dist.all_gather_into_tensor(x, y)}[op]
        for _ in range(5):
            f()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters
        factor = 2 * (ws - 1) / ws if op == "all_reduce" else (ws - 1) / ws
        res[f"{op}_{mib}MiB"] = dict(us=round(t * 1e6, 1), busbw_GBps=round(factor * n * 2 / t / 1e9, 1))
    if rank == 0:
        print("PROBE_JSON " + json.dumps(res), flush=True)
    dist.destroy_process_group()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--json")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=300)
    ap.add_argument("--torchrun", default="torchrun")
    a = ap.parse_args(argv)
    if a.probe:
        probe()
        return 0
    rc, topo = sh("nvidia-smi", "topo", "-m")
    print("## nvidia-smi topo -m")
    print(topo.rstrip() if rc == 0 else f"(not available: {topo.strip()[:200]})")
    rc, q = sh("nvidia-smi", "-q")
    fab = fabric_state(q) if rc == 0 else dict(state=[], status=[])
    print(f"## fabric (nvidia-smi -q): state {sorted(set(fab['state'])) or '-'} status {sorted(set(fab['status'])) or '-'}")
    mc, note = multicast_support()
    print(f"## multicast supported per GPU: {mc if mc is not None else '?'} ({note})")
    probes = {}
    if not a.no_probe:
        for name, extra in CONFIGS:
            r = run_probe(a.nproc, extra, a.timeout_s, a.torchrun)
            probes[name] = r
            env_s = " ".join(f"{k}={v}" for k, v in extra.items()) or "(NCCL defaults)"
            print(f"## probe {name}: {env_s}: {'ok' if r['ok'] else 'FAILED ' + r.get('error', '')}")
            for k, v in (r.get("bw") or {}).items():
                print(f"   {k:<22} {v['us']:>9.1f} us  busbw {v['busbw_GBps']:>7.1f} GB/s")
            for ln in r.get("nvls_lines", [])[:8]:
                print(f"   nccl: {ln[:200]}")
        d, o = probes.get("default", {}).get("bw") or {}, probes.get("nvls_off", {}).get("bw") or {}
        for k in d:
            if k in o and o[k]["us"]:
                print(f"   default vs NVLS off, {k}: {100 * (d[k]['us'] / o[k]['us'] - 1):+.1f}% time")
    v, why = verdict(mc, fab, probes)
    out = dict(verdict=v, why=why, multicast=mc, multicast_note=note, fabric=fab, probes=probes, topo=topo)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    print(f"NVLS: {v} -- {why}")
    return {"available": 0, "unavailable": 1}.get(v, 3)


if __name__ == "__main__":
    sys.exit(main())
