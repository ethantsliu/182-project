#!/usr/bin/env bash
# Set up a fresh 8xH100 Linux box for the speedrun and write a node report. Run from the repo root.
#
#   bash 182/tools/node_setup.sh [--mode venv|docker|report] [--shards 9] [--venv .venv-run]
#                                [--image modded-nanogpt] [--skip-data] [--skip-fa3] [--skip-check]
#
#   report  only the node report (GPUs, driver, CPU cores, RAM, NUMA, disk, CUDA runtimes) -> node_report.txt
#   venv    (default) report + python venv with requirements.txt and torch==2.10 (cu128 index), CUDA 13
#           runtime check, pinned patched-FA3 prefetch + sha256 check, data download, GPU/NCCL sanity check
#   docker  report + `docker build` of the repo Dockerfile (CUDA 13.1 base, torch 2.10 cu128), then the
#           FA3 prefetch, data and sanity check inside the container, with the HF cache mounted from the host
#
# The stack is the one records/track_1_short/2026-08-30_ANVIL2/README.md certifies: torch 2.10.0+cu128
# (NOT a nightly), kernels==0.16.1, huggingface-hub==1.29.0, driver >= 580, libcudart.so.13 present,
# CUDA_HOME with include/cuda_bf16.h (the CE kernel compiles through nvrtc), >= 10 GB free disk.
# Exits nonzero on a hard failure; prints WARN for the soft ones (they are in the report too).
set -uo pipefail

MODE=venv; SHARDS=9; VENV=.venv-run; IMAGE=modded-nanogpt; SKIP_DATA=0; SKIP_FA3=0; SKIP_CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --shards) SHARDS="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --skip-data) SKIP_DATA=1; shift;;
    --skip-fa3) SKIP_FA3=1; shift;;
    --skip-check) SKIP_CHECK=1; shift;;
    -h|--help) sed -n '2,17p' "$0"; exit 0;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[[ "$MODE" =~ ^(venv|docker|report)$ ]] || { echo "invalid mode: $MODE" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO" || exit 1
REPORT="$REPO/node_report.txt"
FAILS=0
fail() { echo "FAIL: $*" | tee -a "$REPORT"; FAILS=$((FAILS + 1)); }
warn() { echo "WARN: $*" | tee -a "$REPORT"; }
sec() { printf '\n## %s\n' "$*" >> "$REPORT"; }
run() { { echo "\$ $*"; "$@" 2>&1; } >> "$REPORT" || true; }

# The pinned FA3 build, read from the trainer so this never drifts from the code.
ATTN="$REPO/track_1_short/model/attention.py"
FA3_REPO=$(sed -n 's/^FA3_REPO = "\(.*\)"/\1/p' "$ATTN" 2>/dev/null)
FA3_REV=$(sed -n 's/^FA3_REVISION = "\(.*\)"/\1/p' "$ATTN" 2>/dev/null)
FA3_SHA=$(sed -n 's/^FA3_BINARY_SHA256 = "\(.*\)"/\1/p' "$ATTN" 2>/dev/null)
FA3_REPO=${FA3_REPO:-devenpzak/flash-attn3-12864}
FA3_REV=${FA3_REV:-64c1e6d1f2780e7931839f41426ddcdb564a7cb9}

# ------------------------------------------------------------------------------------ node report
node_report() {
  : > "$REPORT"
  echo "# node report $(hostname) $(date -Is)" >> "$REPORT"
  echo "repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null))" >> "$REPORT"
  sec "OS"; run uname -a; [[ -f /etc/os-release ]] && run grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
  sec "GPUs"
  if command -v nvidia-smi >/dev/null; then
    run nvidia-smi -L
    run nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,clocks.max.mem,power.limit,persistence_mode,ecc.mode.current,temperature.gpu --format=csv
    run nvidia-smi topo -m
    run nvidia-smi nvlink -s -i 0
    local drv ngpu
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    ngpu=$(nvidia-smi -L | grep -c GPU)
    echo "driver $drv, $ngpu GPU(s), CUDA (driver API) $(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')" >> "$REPORT"
    [[ "${drv%%.*}" -ge 580 ]] 2>/dev/null || warn "driver $drv < 580: the ANVIL2 README measured a silent ~35% uniform slowdown below 580"
    [[ "$ngpu" -eq 8 ]] || warn "$ngpu GPUs (records are 8xH100)"
    nvidia-smi -L | grep -q "H100" || warn "no H100 in nvidia-smi -L"
    nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | grep -qv Enabled && warn "persistence mode off on some GPU (sudo nvidia-smi -pm 1)"
  else
    fail "nvidia-smi not found"
  fi
  sec "CPU"
  run lscpu
  echo "nproc $(nproc)" >> "$REPORT"
  sec "NUMA"
  if command -v numactl >/dev/null; then run numactl -H; else run sh -c "lscpu | grep -i numa"; fi
  sec "Memory"; run free -g; run sh -c "grep -E 'HugePages_Total|Hugepagesize|MemTotal' /proc/meminfo"
  sec "Disk"; run df -h "$REPO" /tmp "${HF_HOME:-$HOME/.cache/huggingface}"
  local free_gb
  free_gb=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1048576)}')
  [[ "$free_gb" -ge 10 ]] || warn "only ${free_gb} GB free under $REPO (need >= 10 for data + Inductor/Triton caches)"
  sec "CUDA runtimes"
  run sh -c "ldconfig -p | grep -E 'libcudart.so|libnccl.so' || true"
  run sh -c "ls -d /usr/local/cuda* 2>/dev/null || true"
  echo "CUDA_HOME=${CUDA_HOME:-<unset>}" >> "$REPORT"
  sec "Software"
  run sh -c "python3 --version; command -v docker && docker --version; command -v nsys && nsys --version; true"
  echo "node report: $REPORT"
}

find_cudart13() {
  ldconfig -p 2>/dev/null | grep 'libcudart.so.13' >/dev/null && return 0
  local d libs="${LD_LIBRARY_PATH:-}"
  for d in /usr/local/cuda-13*/targets/x86_64-linux/lib /usr/local/cuda/lib64 ${libs//:/ }; do
    [[ -e "$d/libcudart.so.13" ]] && { echo "$d"; return 0; }
  done
  return 1
}

check_cuda13() {
  sec "CUDA 13 runtime (the patched FA3 links libcudart.so.13)"
  local d
  if d=$(find_cudart13); then
    echo "libcudart.so.13 found ${d:+in $d}" >> "$REPORT"
    if [[ -n "$d" && ":${LD_LIBRARY_PATH:-}:" != *":$d:"* ]]; then
      warn "libcudart.so.13 is in $d, which is not on LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$d:\$LD_LIBRARY_PATH"
    fi
  else
    fail "no libcudart.so.13. Install: wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update && sudo apt-get install -y cuda-cudart-13-1 cuda-nvrtc-dev-13-1 cuda-cudart-dev-13-1; then put /usr/local/cuda-13.1/targets/x86_64-linux/lib on LD_LIBRARY_PATH (on a 'Signed-By conflict', delete the duplicate .list under /etc/apt/sources.list.d/). Or use --mode docker."
  fi
  local ch="${CUDA_HOME:-/usr/local/cuda}"
  if [[ -f "$ch/include/cuda_bf16.h" ]]; then
    echo "CUDA headers: $ch/include/cuda_bf16.h" >> "$REPORT"
  else
    fail "no $ch/include/cuda_bf16.h: set CUDA_HOME to a CUDA toolkit with headers (the CE kernel compiles via nvrtc at import)"
  fi
}

# ------------------------------------------------------------------------------- python payloads
# These run with whichever python is the run environment (venv or container).
fa3_prefetch_py() {
  cat <<EOF
import hashlib, os, pathlib, shutil, sys
from huggingface_hub import snapshot_download
hub = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface")) / "hub"
path = snapshot_download("$FA3_REPO", revision="$FA3_REV")
print("FA3 snapshot", path)
# kernels>=0.16 resolves through the kernel-repo namespace; seed it from the model download (README)
src, dst = hub / "models--${FA3_REPO/\//--}", hub / "kernels--${FA3_REPO/\//--}"
if src.exists() and not dst.exists():
    shutil.copytree(src, dst, copy_function=os.link if hasattr(os, "link") else shutil.copy2, symlinks=True)
    print("seeded", dst)
want = "$FA3_SHA"
bins = list(pathlib.Path(path).rglob("_flash_attn3_cuda_*.so"))
variant = [b for b in bins if "torch-stable-abi29-cu128-x86_64-linux" in str(b)] or bins
if not variant:
    sys.exit("FAIL: no _flash_attn3_cuda_*.so in the snapshot")
got = hashlib.sha256(variant[0].read_bytes()).hexdigest()
print("FA3 binary", variant[0], got)
if want and got != want:
    sys.exit(f"FAIL: FA3 sha256 {got} != pinned {want}")
EOF
}

stack_check_py() {
  cat <<'EOF'
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
ok = True
if torch.__version__ != "2.10.0+cu128":
    print("FAIL: torch must be 2.10.0+cu128 (a nightly has produced NaNs; see the ANVIL2 README)"); ok = False
try:
    import kernels, huggingface_hub
    print("kernels", kernels.__version__, "huggingface_hub", huggingface_hub.__version__)
except Exception as e:
    print("FAIL: kernels/huggingface_hub import:", e); ok = False
try:
    from kernels import get_kernel
    import pathlib, hashlib
    sys.path.insert(0, ".")
    src = open("track_1_short/model/attention.py").read()
    ns = {}
    for line in src.splitlines():
        if line.startswith(("FA3_REPO", "FA3_REVISION", "FA3_BINARY_SHA256")):
            exec(line, ns)
    fa3 = get_kernel(ns["FA3_REPO"], revision=ns["FA3_REVISION"], trust_remote_code=True)
    b = next(pathlib.Path(fa3.__file__).parent.glob("_flash_attn3_cuda_*.so"))
    d = hashlib.sha256(b.read_bytes()).hexdigest()
    print("FA3 loads:", b, "sha256 ok" if d == ns["FA3_BINARY_SHA256"] else f"FAIL sha256 {d}")
    ok &= d == ns["FA3_BINARY_SHA256"]
except Exception as e:
    print("FAIL: FA3 get_kernel:", repr(e)[:400], "(try `hf auth login`, or HF_HUB_OFFLINE=1 after the prefetch)"); ok = False
sys.exit(0 if ok else 1)
EOF
}

data_download() {
  [[ "$SKIP_DATA" == 1 ]] && return 0
  local n
  # Shard filenames are fixed by cached_fineweb10B.py, without whitespace or newlines.
  # shellcheck disable=SC2012
  n=$(ls data/fineweb10B/fineweb_train_*.bin 2>/dev/null | wc -l)
  if [[ "$n" -ge "$SHARDS" && -f data/fineweb10B/fineweb_val_000000.bin ]]; then
    echo "data: $n train shards + val already present"
  else
    "$@" data/cached_fineweb10B.py "$SHARDS" || fail "data download"
  fi
  # Human-readable listing only, not filename parsing.
  # shellcheck disable=SC2012
  { echo; echo "## Data"; ls -la data/fineweb10B 2>&1 | head -20; } >> "$REPORT"
}

# ----------------------------------------------------------------------------------------- modes
node_report
[[ "$MODE" == "report" ]] && { cat "$REPORT"; exit 0; }

if [[ "$MODE" == "venv" ]]; then
  check_cuda13
  PY=python3.12; command -v "$PY" >/dev/null || PY=python3
  if [[ ! -x "$VENV/bin/python" ]]; then
    if command -v uv >/dev/null; then uv venv --seed -p 3.12 "$VENV" || "$PY" -m venv "$VENV"; else "$PY" -m venv "$VENV"; fi
  fi
  VPY="$VENV/bin/python"
  "$VPY" -m pip install -q --upgrade pip || fail "pip bootstrap"
  "$VPY" -m pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 || fail "pip install -r requirements.txt"
  "$VPY" -m pip install -q torch==2.10 --index-url https://download.pytorch.org/whl/cu128 || fail "pip install torch==2.10 cu128"
  [[ "$SKIP_FA3" == 1 ]] || { fa3_prefetch_py | "$VPY" - || fail "FA3 prefetch"; }
  sec "Stack check (offline against the seeded cache unless HF_HUB_OFFLINE is set)"
  stack_check_py | HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-$((1 - SKIP_FA3))}" "$VPY" - 2>&1 | tee -a "$REPORT"; [[ ${PIPESTATUS[1]} -eq 0 ]] || fail "stack check"
  data_download "$VPY"
  if [[ "$SKIP_CHECK" == 0 ]]; then
    sec "GPU / NCCL sanity"
    "$VENV/bin/torchrun" --standalone --nproc_per_node="$(nvidia-smi -L | grep -c GPU)" "$HERE/node_check.py" 2>&1 | tee -a "$REPORT"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || fail "GPU / NCCL sanity"
  fi
  echo; echo "activate with: source $VENV/bin/activate   (torchrun must be this venv's: $VENV/bin/torchrun)"
elif [[ "$MODE" == "docker" ]]; then
  command -v docker >/dev/null || { fail "docker not installed"; exit 1; }
  DOCKER=(docker); docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
  "${DOCKER[@]}" build -t "$IMAGE" . || { fail "docker build"; exit 1; }
  HFC="${HF_HOME:-$HOME/.cache/huggingface}"; mkdir -p "$HFC"
  DOPTS=(--rm --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864
         -v "$REPO":/modded-nanogpt -v "$HFC":/root/.cache/huggingface -w /modded-nanogpt)
  DRUN=("${DOCKER[@]}" run "${DOPTS[@]}" "$IMAGE")            # a command follows
  DPIPE=("${DOCKER[@]}" run -i "${DOPTS[@]}" "$IMAGE" python -)  # a script on stdin
  [[ "$SKIP_FA3" == 1 ]] || { fa3_prefetch_py | "${DPIPE[@]}" || fail "FA3 prefetch (docker)"; }
  sec "Stack check (docker, offline against the seeded cache)"
  stack_check_py | "${DOCKER[@]}" run -i -e HF_HUB_OFFLINE=1 "${DOPTS[@]}" "$IMAGE" python - 2>&1 | tee -a "$REPORT"
  [[ ${PIPESTATUS[1]} -eq 0 ]] || fail "stack check (docker)"
  data_download "${DRUN[@]}" python
  if [[ "$SKIP_CHECK" == 0 ]]; then
    sec "GPU / NCCL sanity (docker)"
    "${DRUN[@]}" torchrun --standalone --nproc_per_node=8 182/tools/node_check.py 2>&1 | tee -a "$REPORT"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || fail "GPU / NCCL sanity (docker)"
  fi
  echo; echo "run with: ${DRUN[*]} sh run.sh   (add -e HF_HUB_OFFLINE=1 if get_kernel returns 401)"
fi

echo; echo "node report: $REPORT ($FAILS failure(s))"
exit $(( FAILS > 0 ))
