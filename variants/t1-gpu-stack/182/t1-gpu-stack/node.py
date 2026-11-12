"""The node contract for variants/t1-gpu-stack, behind 182/run_node.sh (which cds here and picks the python):

    182/run_node.sh check   CPU-only preflight: PASS|WARN|FAIL <what>: <detail and the fix>, exit 1 on any FAIL
    182/run_node.sh prep    off the clock: data shards, the pinned FA3 binary, the tokenizer file (idempotent)
    182/run_node.sh run     one timed run of #397 (our flags off unless exported); last stdout line is RESULT ...
    182/run_node.sh ab      #397 against our flags, one flag per arm, interleaved by seed

Environment:
  check  SHARDS (9), ALLOW_STACK=1 (a torch / triton version mismatch is a WARN, not a FAIL)
  prep   SHARDS (9), PREP_TIMEOUT_S (3600, per step), SKIP_DATA=1, SKIP_FA3=1, DRY_RUN=1
  run    NPROC (8; the trainer refuses anything else), SEED (-> TRAIN_SEED; unset = random init, as #397),
         STEPS (-> NUM_SCHEDULED_ITERATIONS, default 981), EXTRA_ENV ("K=V K=V"), exported HOST_PREP / DOC_COPY /
         LAYER_ROPE / COPY_MIX / COPY_MIX_FIT / OPTIM_* / MOE_ROUTER_FUSED / HOST_ASYNC / HOST_ASYNC_DEPTH / EXACT_LEVERS,
         LOG_DIR (logs/node), RUN_TIMEOUT_S (1800), DRY_RUN=1
  ab     AB_RUNS (4), AB_ARMS ("base hostprep copymix"), AB_STEPS ("981"; several counts = equal-loss comparison
         through sweep_steps.py), AB_SEED_BASE (1), LOG_DIR, RUN_TIMEOUT_S, DRY_RUN=1. Resumable: finished runs are kept.

Standard library only: `check` has to say what is missing on a machine that has nothing.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlib  # noqa: E402
from runlib import REPO_DIR, TARGET_LOSS, VARIANT_DIR, VARIANT_NAME, WORLD_SIZE  # noqa: E402

TORCH_WANT = "2.10.0+cu128"
TRITON_WANT = "3.6.0"
KERNELS_WANT = "0.16.1"
SHARDS_DEFAULT = 9          # the repo README's `cached_fineweb10B.py 9`; the 1053-step run reads fewer
SHARD_HEADER_BYTES = 1024   # 256 int32: magic, version, token count
SHARD_MAGIC = 20240520
SETUP_CMD = "bash 182/tools/node_setup.sh --mode venv && source .venv-run/bin/activate   (from the repo root)"

# One flag per arm. `exact` arms leave the model and the training arithmetic alone, so their loss is #397's
# and one step count is enough; the others move the loss per step and are judged at equal loss.
ARMS = {
    "base":         dict(env={}, exact=True, what="#397, no flag"),
    "hostprep":     dict(env={"HOST_PREP": "all"}, exact=True, what="#387 host batch plan + candidate gather"),
    "hostasync":    dict(env={"HOST_PREP": "all", "HOST_ASYNC": "1"}, exact=True,
                         what="hostprep + the loader's host batches on a producer thread (182/hostcpu)"),
    "copymix":      dict(env={"COPY_MIX": "1"}, exact=True, what="#376/#382 eval-time copy mixture, fit on consumed tokens"),
    "copymixshard": dict(env={"COPY_MIX": "1", "COPY_MIX_FIT": "shard"}, exact=True, what="#376's fit source (unconsumed shard head)"),
    "lropeattn":    dict(env={"LAYER_ROPE": "attn"}, exact=False, what="#385 gains on the attention sites only (weight folds)"),
    "lrope":        dict(env={"LAYER_ROPE": "1"}, exact=False, what="#385 gains on all four sites (MLP ones as activation scales)"),
    "doccopy":      dict(env={"DOC_COPY": "1"}, exact=False, what="#378 document-local copy feature"),
    # r6-optim (182/optim-cost/README.md): the ANVIL cascade's cost
    "optprof":      dict(env={"OPTIM_PROFILE": "1"}, exact=True, what="CUDA-event timing of the optimizer tail (diagnostic)"),
    "optmoefused":  dict(env={"OPTIM_MOEFUSED": "1"}, exact=True, what="expert bank on the fused ANVIL tail (time only)"),
    "optmaps3":     dict(env={"OPTIM_MAPS": "3"}, exact=False, what="three-map ANVIL cascade (one round less per bank)"),
    "opts4maps3":   dict(env={"OPTIM_MAPS": "3", "OPTIM_NORM": "s4"}, exact=False, what="three maps after a Schatten-4 scale"),
    "opts4":        dict(env={"OPTIM_NORM": "s4"}, exact=False, what="Schatten-4 cascade scale, four maps (loss only)"),
    # r6-kernel (182/kernel-cost/README.md)
    "router":       dict(env={"MOE_ROUTER_FUSED": "1"}, exact=True, what="fused MoE router kernels (182/kernel-cost; same math, not bitwise)"),
}
# What a millinat of val loss costs in train time on this stack, to price a loss change at a single step count:
# #393 cut 999 -> 981 scheduled steps for about +3.0 millinats and -0.75 s. A sweep (AB_STEPS) measures it instead.
MS_PER_MILLINAT = 250.0


# ----------------------------------------------------------------------------------------------- check
class Report:
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def line(self, level: str, what: str, detail: str):
        if level == "FAIL":
            self.fails += 1
        elif level == "WARN":
            self.warns += 1
        print(f"{level} {what}: {detail}", flush=True)

    def ok(self, what, detail):
        self.line("PASS", what, detail)

    def warn(self, what, detail):
        self.line("WARN", what, detail)

    def fail(self, what, detail):
        self.line("FAIL", what, detail)


STACK_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0]}
def ver(name, mod=None):
    try:
        m = __import__(mod or name)
        out[name] = getattr(m, "__version__", "?")
    except Exception as e:
        out[name] = None
        out[name + "_error"] = f"{type(e).__name__}: {str(e)[:160]}"
for name in ("torch", "triton", "numpy", "tiktoken", "huggingface_hub"):
    ver(name)
try:
    from importlib.metadata import version
    out["kernels"] = version("kernels")
except Exception as e:
    out["kernels"] = None
if out.get("torch"):
    import torch
    out["cuda_built"] = torch.version.cuda
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["gpu_count"] = torch.cuda.device_count()
    except Exception as e:
        out["cuda_available"], out["gpu_count"] = False, 0
if out.get("triton"):
    try:
        from triton.experimental import gluon  # the dense and MoE G3 kernels; the run asserts they are active
        out["gluon"] = True
    except Exception as e:
        out["gluon"] = False
        out["gluon_error"] = f"{type(e).__name__}: {str(e)[:160]}"
print("PROBE" + json.dumps(out))
"""


def probe_stack(timeout_s: float = 45) -> dict:
    try:
        p = subprocess.run([sys.executable, "-c", STACK_PROBE], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"probe_error": f"importing torch / triton took more than {timeout_s:.0f} s"}
    for line in p.stdout.splitlines():
        if line.startswith("PROBE"):
            return json.loads(line[5:])
    return {"probe_error": (p.stderr.strip().splitlines() or ["no output"])[-1][:200]}


def shard_problem(path: str) -> str | None:
    """None if the .bin is whole: right magic, and as many bytes as its header's token count says."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            header = f.read(12)
    except OSError as e:
        return str(e)
    if len(header) < 12:
        return f"{size} bytes, no header"
    magic, version, tokens = struct.unpack("<3i", header)
    if magic != SHARD_MAGIC or version != 1:
        return f"bad header (magic {magic}, version {version})"
    want = SHARD_HEADER_BYTES + 2 * tokens
    return None if size == want else f"{size} bytes, header says {want} (partial download?)"


def data_status(shards: int) -> tuple[list[str], list[str], str]:
    """(train shards that are whole, problems, data dir) for shards 1..N and the validation shard."""
    data_dir = os.path.normpath(os.path.join(VARIANT_DIR, runlib.data_path(), "data", "fineweb10B"))
    good, problems = [], []
    wanted = [f"fineweb_train_{i:06d}.bin" for i in range(1, shards + 1)]
    for name in ["fineweb_val_000000.bin"] + wanted:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            problems.append(f"{name} missing")
            continue
        bad = shard_problem(path)
        if bad:
            problems.append(f"{name}: {bad}")
        elif name in wanted:
            good.append(path)
    return good, problems, data_dir


def _run_text(cmd: list[str], timeout_s: float = 15) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def find_cudart13() -> str | None:
    out = _run_text(["ldconfig", "-p"]) or ""
    if "libcudart.so.13" in out:
        return "ldconfig"
    dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        dirs += [os.path.join(cuda_home, "targets/x86_64-linux/lib"), os.path.join(cuda_home, "lib64")]
    dirs += glob.glob("/usr/local/cuda-13*/targets/x86_64-linux/lib") + ["/usr/local/cuda/lib64"]
    dirs += glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/cu13/lib"))
    for d in dirs:
        if os.path.exists(os.path.join(d, "libcudart.so.13")):
            return d
    return None


def fa3_pin() -> dict:
    """The FA3 build the trainer loads, read from its source so this never drifts from the code."""
    import re
    text = open(os.path.join(VARIANT_DIR, "track_1_short", "model", "attention.py")).read()
    return {k: re.search(rf'^FA3_{k} = "(.*)"', text, re.M).group(1) for k in ("REPO", "REVISION", "BINARY_SHA256")}


def hf_hub_dir() -> str:
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.environ.get("HF_HUB_CACHE") or os.path.join(home, "hub")


def fa3_cached(pin: dict) -> bool:
    snap = os.path.join(hf_hub_dir(), "kernels--" + pin["REPO"].replace("/", "--"), "snapshots", pin["REVISION"])
    return bool(glob.glob(os.path.join(snap, "**", "_flash_attn3_cuda_*.so"), recursive=True))


def mem_total_gb() -> float | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    out = _run_text(["sysctl", "-n", "hw.memsize"])
    return int(out) / 2 ** 30 if out and out.strip().isdigit() else None


def flag_env_problem(env: dict, clear: tuple[str, ...] = ()) -> str | None:
    """Parse our flags exactly as the trainer will (folds.py reads only os.environ), in a child process: the
    exported ones (minus `clear`) plus `env`."""
    code = ("import sys; sys.path.insert(0, %r); from track_1_short import folds; "
            "from track_1_short.optim import cascade_flags; print(folds.describe())" % VARIANT_DIR)
    child = {k: v for k, v in os.environ.items() if k not in clear}
    child.update(env)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=child, timeout=30)
    if p.returncode == 0:
        return None
    return (p.stderr.strip().splitlines() or ["folds.py failed"])[-1][:200]


def check() -> int:
    r = Report()
    allow = os.environ.get("ALLOW_STACK") == "1"
    stack_bad = r.warn if allow else r.fail
    shards = int(os.environ.get("SHARDS", SHARDS_DEFAULT))

    # ---- python stack
    info = probe_stack()
    if "probe_error" in info:
        r.fail("python stack", f"{info['probe_error']}; fix: {SETUP_CMD}")
        info = {}
    py = info.get("python", sys.version.split()[0])
    (r.ok if sys.version_info >= (3, 10) else r.fail)("python", f"{py} at {sys.executable} (needs >= 3.10; #397 ran 3.12)")
    torch_v = info.get("torch")
    if not torch_v:
        r.fail("torch", f"not importable ({info.get('torch_error', 'no probe')}); fix: {SETUP_CMD}")
    elif torch_v != TORCH_WANT:
        stack_bad("torch", f"{torch_v}, this tree needs {TORCH_WANT} (the pinned FA3 binary is built for its ABI); fix: "
                           f"pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128  (ALLOW_STACK=1 to try anyway)")
    else:
        r.ok("torch", torch_v)
    triton_v = info.get("triton")
    if not triton_v:
        r.fail("triton", f"not importable ({info.get('triton_error', 'no probe')}); it ships with the cu128 torch wheel: {SETUP_CMD}")
    elif triton_v != TRITON_WANT:
        stack_bad("triton", f"{triton_v}, #397 ran {TRITON_WANT} (torch {TORCH_WANT}'s pin); fix: pip install triton=={TRITON_WANT}")
    else:
        r.ok("triton", triton_v)
    if triton_v:
        if info.get("gluon"):
            r.ok("triton gluon", "triton.experimental.gluon imports (the dense and MoE G3 kernels)")
        else:
            r.fail("triton gluon", f"{info.get('gluon_error')}: the run asserts the Gluon G3 kernels are active; "
                                   f"fix: pip install triton=={TRITON_WANT}")
    kernels_v = info.get("kernels")
    if not kernels_v:
        r.fail("kernels", f"not installed (the hub loader of the patched FA3); fix: pip install kernels=={KERNELS_WANT} "
                          f"huggingface-hub==1.29.0")
    elif kernels_v != KERNELS_WANT:
        r.warn("kernels", f"{kernels_v}, requirements.txt pins {KERNELS_WANT}")
    else:
        r.ok("kernels", kernels_v)
    missing = [m for m in ("numpy", "tiktoken", "huggingface_hub") if not info.get(m)]
    if missing:
        r.fail("python packages", f"missing {', '.join(missing)}; fix: pip install -r requirements.txt  (in {VARIANT_DIR})")
    else:
        r.ok("python packages", "numpy, tiktoken, huggingface_hub import")
    if shutil.which("torchrun"):
        r.ok("torchrun", shutil.which("torchrun"))
    else:
        r.fail("torchrun", f"not on PATH; activate the venv that has torch: source {REPO_DIR}/.venv-run/bin/activate")

    # ---- GPUs
    smi = _run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"])
    if smi is None:
        r.fail("gpus", "nvidia-smi not found or failing; this trainer needs exactly 8 CUDA GPUs (8xH100)")
    else:
        rows = [[c.strip() for c in line.split(",")] for line in smi.strip().splitlines() if line.strip()]
        names = sorted({row[0] for row in rows})
        if len(rows) != WORLD_SIZE:
            r.fail("gpus", f"{len(rows)} visible ({', '.join(names)}); train_gpt.py asserts world_size == 8. "
                           f"Check CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
        elif not all("H100" in n for n in names):
            r.warn("gpus", f"8 x {', '.join(names)}: not H100, times are not comparable with the records")
        else:
            r.ok("gpus", f"8 x {names[0]}")
        if rows:
            driver = rows[0][1]
            major = int(driver.split(".")[0]) if driver.split(".")[0].isdigit() else 0
            (r.ok if major >= 580 else r.warn)("driver", f"{driver} (the ANVIL2 record README measured a ~35% slowdown below 580)")
            busy = [f"gpu{i} {row[2]} MiB" for i, row in enumerate(rows) if row[2].isdigit() and int(row[2]) > 2000]
            small = [row[3] for row in rows if row[3].isdigit() and int(row[3]) < 60000]
            if busy:
                r.warn("gpu memory", f"in use before the run: {', '.join(busy)} (#397 reserves ~52.6 GiB per GPU); "
                                     f"see `nvidia-smi`, kill stale ranks")
            elif small:
                r.fail("gpu memory", f"{small[0]} MiB per GPU; #397 peaks at 34.3 GiB allocated / 52.6 GiB reserved")
            else:
                r.ok("gpu memory", f"{rows[0][3]} MiB per GPU, idle")
    if torch_v and smi is not None and not info.get("cuda_available"):
        r.fail("torch cuda", f"torch.cuda.is_available() is False (torch built for CUDA {info.get('cuda_built')}); "
                             f"install the cu128 wheel: pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128")

    # ---- CUDA 13 runtime and headers (FA3 dlopens libcudart.so.13; the CE kernels compile through nvrtc)
    if sys.platform.startswith("linux"):
        where = find_cudart13()
        if where:
            r.ok("libcudart.so.13", f"found ({where})")
        else:
            r.fail("libcudart.so.13", "not found (the patched FA3 links it); fix: sudo apt-get install -y cuda-cudart-13-1 "
                                      "cuda-nvrtc-dev-13-1 cuda-cudart-dev-13-1 and put /usr/local/cuda-13.1/targets/x86_64-linux/lib "
                                      f"on LD_LIBRARY_PATH, or: {SETUP_CMD}")
    else:
        r.fail("libcudart.so.13", f"{sys.platform} is not Linux: this trainer runs on a Linux 8xH100 node only")
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    header = os.path.join(cuda_home, "include", "cuda_bf16.h")
    if os.path.exists(header):
        r.ok("cuda headers", header)
    else:
        r.fail("cuda headers", f"no {header}: the cross-entropy kernels compile through nvrtc at import; fix: "
                               f"export CUDA_HOME=<a CUDA toolkit with include/cuda_bf16.h> (e.g. /usr/local/cuda-13.1)")

    # ---- FA3 binary and data
    pin = fa3_pin()
    if fa3_cached(pin):
        r.ok("fa3", f"{pin['REPO']}@{pin['REVISION'][:8]} is in the HF cache")
    else:
        r.warn("fa3", f"{pin['REPO']}@{pin['REVISION'][:8]} is not in {hf_hub_dir()}: the first run downloads it "
                      f"(needs network); fix: 182/run_node.sh prep")
    good, problems, data_dir = data_status(shards)
    if problems:
        shown = "; ".join(problems[:3]) + (f"; and {len(problems) - 3} more" if len(problems) > 3 else "")
        r.fail("data", f"{len(good)}/{shards} train shards whole in {data_dir}: {shown}; fix: 182/run_node.sh prep")
    else:
        r.ok("data", f"{len(good)} train shards + the validation shard in {data_dir}")

    # ---- host
    cpus = os.cpu_count() or 0
    (r.ok if cpus >= 4 * WORLD_SIZE else r.warn)(
        "vcpus", f"{cpus} ({cpus / WORLD_SIZE:.1f} per GPU; each rank runs a prep thread and a loader beside its main thread, "
                 f"want >= 4 per GPU)")
    ram = mem_total_gb()
    if ram is None:
        r.warn("ram", "could not read the host memory size")
    else:
        (r.ok if ram >= 96 else r.warn)("ram", f"{ram:.0f} GiB (8 ranks hold a 200 MB shard or two each plus pinned staging; "
                                               f"want >= 96 GiB)")
    if os.path.isdir("/dev/shm"):
        shm = shutil.disk_usage("/dev/shm").free / 2 ** 30
        (r.ok if shm >= 1 else r.warn)("/dev/shm", f"{shm:.1f} GiB free (NCCL's shared-memory transport; in docker pass --ipc=host "
                                                   f"or --shm-size=16g)")
    else:
        r.warn("/dev/shm", "absent")
    for label, path in (("disk (tree)", VARIANT_DIR), ("disk (compile caches)", os.path.expanduser("~")), ("disk (/tmp)", "/tmp")):
        free = shutil.disk_usage(path).free / 2 ** 30
        (r.ok if free >= 10 else r.warn)(label, f"{free:.0f} GiB free at {path} (logs, Inductor / Triton caches; want >= 10)")
    r.ok("extensions", "none to build: every kernel is JIT (Triton, nvrtc) or the prebuilt FA3; no rust / cargo")

    # ---- our flags, as exported now
    exported = {k: os.environ[k] for k in runlib.FLAG_KEYS + runlib.BASE_KEYS if k in os.environ}
    bad = flag_env_problem({})
    if bad:
        r.fail("flags", f"{exported}: {bad}")
    else:
        r.ok("flags", f"{exported or 'none exported'} (run = #397 unless a flag is exported)")

    print(f"check: {r.fails} FAIL, {r.warns} WARN" + ("" if r.fails else " -- ready: 182/run_node.sh run"), flush=True)
    return 1 if r.fails else 0


# ------------------------------------------------------------------------------------------------ prep
FA3_PREFETCH = r"""
import hashlib, os, pathlib, shutil, sys
from huggingface_hub import snapshot_download
repo, rev, want = sys.argv[1:4]
home = os.environ.get("HF_HOME") or str(pathlib.Path.home() / ".cache/huggingface")
hub = pathlib.Path(os.environ.get("HF_HUB_CACHE") or pathlib.Path(home) / "hub")
path = snapshot_download(repo, revision=rev)
# kernels >= 0.16 resolves through its own cache namespace: seed it from the model download.
src, dst = hub / ("models--" + repo.replace("/", "--")), hub / ("kernels--" + repo.replace("/", "--"))
if src.exists() and not dst.exists():
    shutil.copytree(src, dst, symlinks=True)
bins = list(pathlib.Path(path).rglob("_flash_attn3_cuda_*.so"))
variant = [b for b in bins if "torch-stable-abi29-cu128-x86_64-linux" in str(b)] or bins
if not variant:
    sys.exit("no _flash_attn3_cuda_*.so in the snapshot")
got = hashlib.sha256(variant[0].read_bytes()).hexdigest()
if got != want:
    sys.exit(f"FA3 sha256 {got} != pinned {want}")
print("FA3", variant[0])
"""


def _step(what: str, cmd: list[str], timeout_s: float, dry: bool, cwd: str = REPO_DIR) -> bool:
    print(f"prep {what}: {' '.join(shlex.quote(c) if len(c) < 200 else '<inline python>' for c in cmd)}", flush=True)
    if dry:
        return True
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 15)
            print(f"FAIL prep {what}: timeout after {timeout_s:.0f} s (PREP_TIMEOUT_S); re-run prep, it resumes", flush=True)
            return False
    except OSError as e:
        print(f"FAIL prep {what}: {e}", flush=True)
        return False
    if code != 0:
        print(f"FAIL prep {what}: exit code {code}", flush=True)
    return code == 0


def prep() -> int:
    dry = os.environ.get("DRY_RUN") == "1"
    timeout_s = float(os.environ.get("PREP_TIMEOUT_S", 3600))
    shards = int(os.environ.get("SHARDS", SHARDS_DEFAULT))
    ok = True
    if os.environ.get("SKIP_DATA") == "1":
        print("prep data: skipped (SKIP_DATA=1)")
    else:
        good, problems, data_dir = data_status(shards)
        broken = [p for p in problems if "missing" not in p]
        if broken:
            # The downloader skips files that exist, so a damaged one has to go first. Not deleted here.
            print(f"FAIL prep data: {'; '.join(broken)} in {data_dir}: delete the file(s) and re-run prep", flush=True)
            ok = False
        elif not problems:
            print(f"prep data: {len(good)} train shards + validation already in {data_dir}")
        elif os.path.normpath(os.path.join(REPO_DIR, "data", "fineweb10B")) != data_dir:
            print(f"FAIL prep data: DATA_PATH points at {data_dir}, which this script does not fill ({'; '.join(problems[:2])}); "
                  f"unset DATA_PATH to use the repo root's data/", flush=True)
            ok = False
        else:
            ok &= _step("data", [sys.executable, os.path.join("data", "cached_fineweb10B.py"), str(shards)], timeout_s, dry)
            if ok and not dry:
                _, problems, _ = data_status(shards)
                if problems:
                    print(f"FAIL prep data: after the download: {'; '.join(problems[:3])}", flush=True)
                    ok = False
    if os.environ.get("SKIP_FA3") == "1":
        print("prep fa3: skipped (SKIP_FA3=1)")
    else:
        pin = fa3_pin()
        if fa3_cached(pin) and not dry:
            print(f"prep fa3: {pin['REPO']}@{pin['REVISION'][:8]} already in the HF cache")
        else:
            ok &= _step("fa3", [sys.executable, "-c", FA3_PREFETCH, pin["REPO"], pin["REVISION"], pin["BINARY_SHA256"]],
                        timeout_s, dry)
    # The trainer builds its prefix table and canonical mask from the GPT-2 tokenizer, which tiktoken downloads once.
    ok &= _step("tokenizer", [sys.executable, "-c", "import tiktoken; tiktoken.get_encoding('gpt2'); print('gpt2 encoding cached')"],
                min(timeout_s, 600), dry)
    print("prep: nothing to build (no extension). The first run compiles for ~7 minutes before its clock starts; "
          "later runs reuse the Inductor / Triton caches.")
    print("prep: done" if ok else "prep: FAILED (see the FAIL lines above)", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------- run
def parse_extra_env(raw: str) -> dict:
    env = {}
    for item in shlex.split(raw or ""):
        key, sep, value = item.partition("=")
        if not sep or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"EXTRA_ENV item {item!r} is not KEY=VALUE")
        env[key] = value
    return env


def run_env() -> dict:
    """What `run` sets on top of the caller's environment (exported flags pass through untouched)."""
    env = {"DATA_PATH": runlib.data_path()}
    if os.environ.get("SEED", "") != "":
        env["TRAIN_SEED"] = str(int(os.environ["SEED"]))
    if os.environ.get("STEPS", "") != "":
        env["NUM_SCHEDULED_ITERATIONS"] = str(int(os.environ["STEPS"]))
    env.update(parse_extra_env(os.environ.get("EXTRA_ENV", "")))
    return env


def shown_env(env_set: dict, clear: tuple[str, ...] = ()) -> dict:
    """env_set plus the trainer knobs already exported, for the printed command (so it reproduces the run)."""
    shown = {k: os.environ[k] for k in runlib.FLAG_KEYS + runlib.BASE_KEYS if k in os.environ and k not in clear}
    shown.update(env_set)
    return shown


def result_failed(reason: str) -> int:
    print(f"RESULT variant={VARIANT_NAME} status=failed reason={' '.join(reason.split())}", flush=True)
    return 1


def preflight_problem(nproc: int, env_set: dict, clear: tuple[str, ...] = ()) -> str | None:
    """The cheap reasons a launch cannot work, so `run` says so in a second instead of after a traceback."""
    if nproc != WORLD_SIZE:
        return f"NPROC={nproc}: this trainer runs on exactly 8 GPUs (track_1_short/distributed.py)"
    if not shutil.which("torchrun"):
        return "torchrun not on PATH (activate the venv: source .venv-run/bin/activate at the repo root; see run_node.sh check)"
    good, problems, data_dir = data_status(2)  # the loader opens the first shard and preloads the second
    if problems:
        return f"data: {problems[0]} in {data_dir} (run_node.sh prep)"
    bad = flag_env_problem({k: v for k, v in env_set.items() if k in runlib.FLAG_KEYS}, clear)
    if bad:
        return f"flags: {bad}"
    return None


def run() -> int:
    try:
        nproc = int(os.environ.get("NPROC", WORLD_SIZE))
        env_set = run_env()
        timeout_s = float(os.environ.get("RUN_TIMEOUT_S", 1800))
    except ValueError as e:
        return result_failed(f"bad environment: {e}")
    if os.environ.get("DRY_RUN") == "1":
        print(runlib.format_cmd(shown_env(env_set), runlib.torchrun_cmd(nproc)))
        return 0
    problem = preflight_problem(nproc, env_set)
    if problem:
        return result_failed(problem)
    log_dir = os.path.abspath(os.environ.get("LOG_DIR") or os.path.join(VARIANT_DIR, "logs", "node"))
    seed = env_set.get("TRAIN_SEED", os.environ.get("TRAIN_SEED", "rand"))
    stem = f"{VARIANT_NAME}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_seed{seed}"
    console = os.path.join(log_dir, stem + ".console.txt")
    print(f"run: {runlib.format_cmd(shown_env(env_set), runlib.torchrun_cmd(nproc))}", flush=True)
    print(f"run: console -> {console}", flush=True)
    code, note = runlib.launch(env_set, console, nproc=nproc, timeout_s=timeout_s, echo=True)
    return report_run(console, code, note, log_dir, stem)


def report_run(console: str, code: int | None, note: str, log_dir: str, stem: str) -> int:
    text = open(console, errors="replace").read()
    result = runlib.parse_text(text)
    log = console
    trainer_log = runlib.trainer_log_path(text)
    if trainer_log:  # the record-format log (source, environment, every line): keep it next to the console
        log = os.path.join(log_dir, stem + ".txt")
        shutil.copyfile(trainer_log, log)
    if result is None or code != 0:
        return result_failed(f"{note or runlib.failure_reason(text, code)} log={log}")
    if result.copy:
        c = result.copy
        print(f"INFO copy mixture: val_loss_no_copy={c['val_loss_no_copy']:.6f} copy_gain={c['copy_gain']:.6f} "
              f"copy_fit_ms={c['copy_fit_ms']} fit={c['fit']}")
    print(f"INFO scheduled={result.scheduled} flags={ {k: os.environ[k] for k in runlib.FLAG_KEYS if k in os.environ} or 'none'}")
    print(f"RESULT variant={VARIANT_NAME} val_loss={result.val_loss:.6f} train_time_s={result.train_time_s:.3f} "
          f"steps={result.steps} target={TARGET_LOSS:.2f} hit={int(result.hit)} log={log}", flush=True)
    return 0


# -------------------------------------------------------------------------------------------------- ab
def ab_plan(arms: list[str], steps: list[int], runs: int, seed_base: int, out_dir: str) -> list[dict]:
    """Every run of the A/B in launch order: rounds (one seed each) x step counts x arms. Within a seed every arm
    runs back to back, and the arm order rotates by one each round so no arm always goes first on a warm node."""
    plan = []
    for k in range(runs):
        order = arms[k % len(arms):] + arms[:k % len(arms)]
        for count in steps:
            for arm in order:
                plan.append(dict(arm=arm, steps=count, seed=seed_base + k, round=k,
                                 path=os.path.join(out_dir, arm, f"{arm}_s{count}_r{k}.txt")))
    return plan


def arm_env(arm: str, steps: int, seed: int) -> dict:
    return {"DATA_PATH": runlib.data_path(), "TRAIN_SEED": str(seed), "NUM_SCHEDULED_ITERATIONS": str(steps),
            **ARMS[arm]["env"]}


def paired_stats(deltas: list[float]) -> tuple[float, float, float]:
    """(mean, standard error, t) of paired differences; se and t are nan below two pairs."""
    mean = statistics.mean(deltas)
    if len(deltas) < 2:
        return mean, float("nan"), float("nan")
    se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    return mean, se, (mean / se if se > 0 else float("inf") * (1 if mean > 0 else -1))


# Two-sided 95% t quantiles by degrees of freedom (1..10), then the normal value.
T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228)


def t95(df: int) -> float:
    return T95[df - 1] if 1 <= df <= len(T95) else 1.96


def ab_summary(plan: list[dict], results: dict, arms: list[str], steps: list[int]) -> list[str]:
    """The table and one verdict line per arm. `results` maps a plan path to a runlib.Run (finished runs only)."""
    out = []
    by = {}
    for p in plan:
        if p["path"] in results:
            by.setdefault((p["arm"], p["steps"]), {})[p["seed"]] = results[p["path"]]
    out.append(f"{'arm':<13} {'steps':>5} {'n':>2} {'time s':>8} {'sd':>6} {'val loss':>9} {'sd':>8}")
    for arm in arms:
        for count in steps:
            runs = list(by.get((arm, count), {}).values())
            if not runs:
                out.append(f"{arm:<13} {count:>5}  0  (no finished run)")
                continue
            times, losses = [x.train_time_s for x in runs], [x.val_loss for x in runs]
            sd = (lambda v: statistics.stdev(v) if len(v) > 1 else float("nan"))
            out.append(f"{arm:<13} {count:>5} {len(runs):>2} {statistics.mean(times):>8.3f} {sd(times):>6.3f} "
                       f"{statistics.mean(losses):>9.5f} {sd(losses):>8.5f}")
    ref = max(steps)  # the default count: where #397's 30.67 s was measured
    base = by.get(("base", ref), {})
    if base:
        times = [x.train_time_s for x in base.values()]
        losses = [x.val_loss for x in base.values()]
        out.append(f"VERDICT base: #397 on this node {statistics.mean(times):.3f} s, mean val {statistics.mean(losses):.5f} "
                   f"at {ref} steps, n={len(times)} (author's node: 30.675 s, 3.27743, n=12; the root record here: 40.93 s, n=1)")
    for arm in arms:
        if arm == "base":
            continue
        mine = by.get((arm, ref), {})
        seeds = sorted(set(base) & set(mine))
        if not seeds:
            out.append(f"VERDICT {arm}: no seed finished in both arms at {ref} steps")
            continue
        dt, dt_se, dt_t = paired_stats([mine[s].train_time_s - base[s].train_time_s for s in seeds])
        dl, dl_se, _ = paired_stats([mine[s].val_loss - base[s].val_loss for s in seeds])
        half = t95(len(seeds) - 1) * dt_se if len(seeds) > 1 else float("nan")
        head = (f"VERDICT {arm}: n={len(seeds)} pairs at {ref} steps: time {dt:+.3f} s (95% +-{half:.3f}, t={dt_t:.1f}), "
                f"val loss {dl * 1000:+.2f} mnat (se {dl_se * 1000:.2f})")
        copies = [mine[s].copy for s in seeds if mine[s].copy]
        if copies:
            gain = statistics.mean(c["copy_gain"] for c in copies) * 1000
            fit_ms = statistics.mean(c["copy_fit_ms"] for c in copies)
            worth = gain * MS_PER_MILLINAT
            keep = "KEEP (retune steps down by the gain)" if worth > fit_ms and gain > 0 else "DROP"
            out.append(f"{head}; copy_gain {gain:.2f} mnat (same forward, no seed noise) is worth ~{worth:.0f} ms of steps at "
                       f"{MS_PER_MILLINAT:.0f} ms/mnat against a {fit_ms:.0f} ms fit on the clock -> {keep}")
        elif ARMS[arm]["exact"]:
            if len(seeds) < 2:
                keep = "one pair only: no interval, run more"
            elif dt + half < 0:
                keep = "KEEP (the whole interval is below 0)"
            elif dt - half > 0:
                keep = "DROP (slower)"
            else:
                keep = "NO DIFFERENCE resolved at this n"
            out.append(f"{head}; exact lever, judged on time alone -> {keep}")
        else:
            net = dt * 1000 + dl * 1000 * MS_PER_MILLINAT
            out.append(f"{head}; changes the model: net {net:+.0f} ms at {MS_PER_MILLINAT:.0f} ms/mnat (rough). "
                       + ("See the equal-loss table below." if len(steps) > 1 else
                          f"For the real answer sweep it: AB_ARMS=\"base {arm}\" AB_STEPS=\"951 966 981\" 182/run_node.sh ab"))
    return out


def ab() -> int:
    try:
        arms = (os.environ.get("AB_ARMS") or "base hostprep copymix").split()
        steps = [int(s) for s in (os.environ.get("AB_STEPS") or "981").split()]
        runs = int(os.environ.get("AB_RUNS", 4))
        seed_base = int(os.environ.get("AB_SEED_BASE", 1))
        nproc = int(os.environ.get("NPROC", WORLD_SIZE))
        timeout_s = float(os.environ.get("RUN_TIMEOUT_S", 1800))
    except ValueError as e:
        print(f"FAIL ab: bad environment: {e}")
        return 2
    unknown = [a for a in arms if a not in ARMS]
    if unknown or not arms or runs < 1 or not steps:
        print(f"FAIL ab: unknown arm(s) {unknown}; known: {', '.join(ARMS)}")
        return 2
    if "base" not in arms:
        arms = ["base"] + arms  # every verdict is against #397 on this node, in this session
    out_dir = os.path.join(os.path.abspath(os.environ.get("LOG_DIR") or os.path.join(VARIANT_DIR, "logs", "node")), "ab")
    plan = ab_plan(arms, steps, runs, seed_base, out_dir)
    clear = runlib.FLAG_KEYS + runlib.BASE_KEYS  # an arm is its own flags and nothing the shell happened to export
    dry = os.environ.get("DRY_RUN") == "1"
    print(f"ab: arms {arms} x steps {steps} x {runs} seeds ({seed_base}..{seed_base + runs - 1}) = {len(plan)} runs -> {out_dir}")
    for arm in arms:
        print(f"ab:   {arm:<13} {ARMS[arm]['what']}")
    if not dry:
        problem = preflight_problem(nproc, {}, clear)
        if problem:
            print(f"FAIL ab: {problem}")
            return 1
    results = {}
    failures = {}
    for i, p in enumerate(plan):
        env_set = arm_env(p["arm"], p["steps"], p["seed"])
        tag = f"[{i + 1}/{len(plan)}] {p['arm']} s{p['steps']} seed {p['seed']}"
        if dry:
            print(f"{tag}: {runlib.format_cmd(shown_env(env_set, clear), runlib.torchrun_cmd(nproc))}")
            continue
        done = runlib.parse_log(p["path"])
        if done is not None:
            print(f"{tag}: have {p['path']}")
            results[p["path"]] = done
            continue
        if failures.get(p["arm"], 0) >= 2:
            print(f"{tag}: skipped, the arm failed twice")
            continue
        print(f"{tag}: running -> {p['path']}", flush=True)
        code, note = runlib.launch(env_set, p["path"], nproc=nproc, timeout_s=timeout_s, clear=clear)
        done = runlib.parse_log(p["path"]) if code == 0 else None
        if done is None:
            text = open(p["path"], errors="replace").read()
            failures[p["arm"]] = failures.get(p["arm"], 0) + 1
            print(f"{tag}: FAILED: {note or runlib.failure_reason(text, code)}", flush=True)
            os.replace(p["path"], p["path"] + ".failed")  # a re-run retries it
            if p["arm"] == "base" and failures["base"] >= 2:
                print("FAIL ab: #397 itself failed twice on this node; nothing to compare against")
                return 1
            if code == 130:
                return 130
            continue
        results[p["path"]] = done
        print(f"{tag}: val_loss {done.val_loss:.6f} train_time {done.train_time_s:.3f} s", flush=True)
    if dry:
        return 0
    print()
    for line in ab_summary(plan, results, arms, steps):
        print(line)
    if len(steps) > 1:
        import sweep_steps
        print()
        sweep_steps.compare([os.path.join(out_dir, arm) for arm in arms], sweep_steps.TARGET_MEAN)
    return 0 if results else 1


def main(argv: list[str]) -> int:
    modes = {"check": check, "prep": prep, "run": run, "ab": ab}
    if len(argv) != 1 or argv[0] not in modes:
        print("usage: 182/run_node.sh check|prep|run|ab   (environment knobs: see the top of 182/t1-gpu-stack/node.py)")
        return 2
    return modes[argv[0]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
