#!/usr/bin/env bash
# One GPU session on the 8xH100 box -> one file to send back. Run from anywhere inside the repo.
#
#   bash 182/gpu_session.sh              # = all: env tests sanity gate hadamard t2 bundle
#   QUICK=1 bash 182/gpu_session.sh      # = env sanity gate bundle: the decision gate only (n=4 still)
#   bash 182/gpu_session.sh sanity gate  # just those phases (in the canonical order)
#   DRY_RUN=1 bash 182/gpu_session.sh    # print every GPU/pip/download command; writes only to a temp dir
#                                        # (with SESSION_DIR: previews a resume of that session from a copy)
#
# Resumable: re-run the same command after a crash or ctrl-C. A phase whose status is PASS is skipped
# (FORCE=1 redoes it); the A/B phases resume inside ab_run.py at the first unfinished run. Every phase is
# logged to 182/results/session-<sha>/ (gitignored); `all` always ends with the bundle, even after a failed
# phase. An A/B phase stops after STOP_AFTER_FAILURES consecutive failed runs, or once one arm's first
# ARM_FAILURES runs all failed (slots stay pending; re-run resumes them, probing that arm once more); a hadamard
# variant that fails that way is abandoned and the other variants continue. One that finished unusable from
# scattered failures (gate < 2 seed pairs) is FAIL, and the next re-run moves it to <phase>.attemptN and
# re-plans it; one where an arm failed EVERY run is not re-planned (it would burn the same runs again): fix the
# cause, then FORCE=1 (FORCE=1 always re-plans). Every GPU, test and download command runs under a wall-clock
# limit that kills its whole process group. After ctrl-C, `bash 182/gpu_session.sh bundle` packs what exists.
# Run it inside tmux/screen. The script never commits or pushes.
#
# Phases (est. node-minutes printed before each; GPU-minutes = 8x):
#   env       git sha/status, node report (182/tools/node_setup.sh --mode report), driver >= 580 check,
#             CPUs, RAM, /dev/shm; node_check.py (GEMM/NCCL); 9 data shards if missing. The stack check FAILs
#             env (and the session skips to the bundle) unless torch 2.10.0+cu128 and triton 3.6.0 import,
#             torchrun belongs to $PY's environment, and the pinned FA3 kernel loads with its sha256
#             (ALLOW_STACK=1 downgrades that to a warning). Writes env/stack.json and env/pip_freeze.txt.
#             When t2 is selected and the host can run it, starts the 103-shard download + exact_match build
#             in the background (CPU/network only, overlaps the GPU phases).
#   tests     pytest 182/hadamard-fp8/tests, then again with HADAMARD_FP8=1
#   sanity    182/hadamard-fp8/gpu_sanity.py (skipped if absent)
#   gate      step 1 of 182/hadamard-fp8/README.md's workflow (the attribution/dump/offline steps are not run
#             here): record vs FP8_OFF=1, GATE_SEEDS paired, interleaved (ab_run.py); prints the paired gap
#             val(fp8) - val(bf16) mean +- SE and its 95% CI vs 0.5 mnat -> STOP / GAP / INCONCLUSIVE, but only
#             once every gate slot has finished (INCOMPLETE otherwise: no decision)
#   hadamard  only if the gate phase PASSed with decision GAP/INCONCLUSIVE and mean gap >= HADAMARD_MIN_GAP_MNAT
#             (2.0: a loose prescreen for the default variants, which need ~3-4 mnat to pay at 0.164 s/mnat), and
#             the HADAMARD_FP8 tests / sanity did not fail (FORCE_HADAMARD=1 bypasses all of that):
#             record vs HADAMARD_FP8=1 per HADAMARD_VARIANTS (GEMMS@BLOCK), HADAMARD_SEEDS
#   t2        variants/t2-cpu-ngram at its default 5800 steps, once (skipped with the reason if the host
#             lacks ~390 GiB RAM, 144 GiB free /dev/shm, 24 vCPU/GPU, or the 103 shards)
#   bundle    summary.txt + tar.gz of every log -> 182/results/session-<date>-<sha>.tar.gz (prints the path)
#
# Knobs (env): PY (python), NPROC (8), CACHE (warm|cold; warm = one compile cache per arm), GATE_SEEDS
# (1,2,3,4), HADAMARD_VARIANTS ("all@128 dw1,dw2@32"), HADAMARD_SEEDS (1,2), HADAMARD_MIN_GAP_MNAT (2.0),
# FORCE_HADAMARD, TIMEOUT_S (1500 per A/B run), STOP_AFTER_FAILURES (3), ARM_FAILURES (2), T2_TIMEOUT_S (5400),
# TESTS_TIMEOUT_S (1800 per pytest), SANITY_TIMEOUT_S (1200), DATA_TIMEOUT_S (3600, the 9-shard download),
# T2_DATA_TIMEOUT_S (14400, the 103-shard download), T2_PREFETCH_WAIT_S (3600), COLD_RUN_MIN (9) / WARM_RUN_MIN (4)
# (the estimates), T2_MIN_RAM_GIB (390), T2_MIN_SHM_GIB (144), T2_MIN_CPU_PER_GPU (24), QUICK, RUN_CMD (ab_run
# --cmd template), SANITY_CMD, SKIP_DATA, ALLOW_STACK, T2_FORCE (ignore host checks), T2_INSTALL_RUST (auto: only
# inside a container), SESSION_DIR, FORCE, DRY_RUN, SESSION_DOCKER=<image> (re-run this script inside that image,
# e.g. modded-nanogpt from node_setup.sh --mode docker; SESSION_DOCKER_GPUS: --gpus value, default all, none = omit).
# In docker mode every knob above that YOU set is forwarded into the container, except PY (the image's python is
# used) and SESSION_DIR (mapped explicitly); defaults are computed inside the container. HF_HUB_OFFLINE and
# HF_TOKEN are forwarded too.
set -euo pipefail

PHASES=(env tests sanity gate hadamard t2 bundle)
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$REPO"

# The knobs the user set, captured before any default is applied (docker mode forwards exactly these).
FORWARD_VARS=(QUICK NPROC CACHE GATE_SEEDS HADAMARD_VARIANTS HADAMARD_SEEDS HADAMARD_MIN_GAP_MNAT FORCE_HADAMARD TIMEOUT_S
  STOP_AFTER_FAILURES ARM_FAILURES T2_TIMEOUT_S TESTS_TIMEOUT_S SANITY_TIMEOUT_S DATA_TIMEOUT_S T2_DATA_TIMEOUT_S
  T2_PREFETCH_WAIT_S COLD_RUN_MIN WARM_RUN_MIN T2_MIN_RAM_GIB T2_MIN_SHM_GIB T2_MIN_CPU_PER_GPU RUN_CMD SANITY_CMD
  SKIP_DATA ALLOW_STACK T2_FORCE T2_INSTALL_RUST FORCE DRY_RUN HF_HUB_OFFLINE HF_TOKEN)
USER_ENV=()
for v in "${FORWARD_VARS[@]}"; do [[ -n "${!v+x}" ]] && USER_ENV+=(-e "$v=${!v}"); done
SANITY_DEFAULT=0; [[ -z "${SANITY_CMD+x}" ]] && SANITY_DEFAULT=1

DRY_RUN=${DRY_RUN:-0}
PY=${PY:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
NPROC=${NPROC:-8}
CACHE=${CACHE:-warm}
GATE_SEEDS=${GATE_SEEDS:-1,2,3,4}
HADAMARD_VARIANTS=${HADAMARD_VARIANTS:-"all@128 dw1,dw2@32"}
HADAMARD_SEEDS=${HADAMARD_SEEDS:-1,2}
TIMEOUT_S=${TIMEOUT_S:-1500}      # a cold run is ~9 min; a hung NCCL run should not cost 40
STOP_AFTER_FAILURES=${STOP_AFTER_FAILURES:-3}
ARM_FAILURES=${ARM_FAILURES:-2}   # one arm crashing every run (the arms interleave, so the consecutive count misses it)
T2_TIMEOUT_S=${T2_TIMEOUT_S:-5400}; TESTS_TIMEOUT_S=${TESTS_TIMEOUT_S:-1800}; SANITY_TIMEOUT_S=${SANITY_TIMEOUT_S:-1200}
DATA_TIMEOUT_S=${DATA_TIMEOUT_S:-3600}; T2_DATA_TIMEOUT_S=${T2_DATA_TIMEOUT_S:-14400}
RUN_CMD=${RUN_CMD:-"torchrun --standalone --nproc_per_node={nproc} {script}"}
SANITY_CMD=${SANITY_CMD:-"$PY 182/hadamard-fp8/gpu_sanity.py"}
COLD_RUN_MIN=${COLD_RUN_MIN:-9}   # one Track 1 run with a cold compile cache (~7 min compile + train + val)
WARM_RUN_MIN=${WARM_RUN_MIN:-4}
T2_MIN_RAM_GIB=${T2_MIN_RAM_GIB:-390}; T2_MIN_SHM_GIB=${T2_MIN_SHM_GIB:-144}; T2_MIN_CPU_PER_GPU=${T2_MIN_CPU_PER_GPU:-24}
T2_SHARDS=103; T2_DIR=variants/t2-cpu-ngram
export GIT_OPTIONAL_LOCKS=0       # `git status` must not rewrite the index (DRY_RUN writes nothing in the repo)
[[ "$DRY_RUN" == 1 ]] && export PYTHONDONTWRITEBYTECODE=1
SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || echo nogit)

if [[ -z "${SESSION_DIR:-}" ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then SESSION_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu_session_dry.XXXXXX")/session-$SHA"
  else SESSION_DIR="$REPO/182/results/session-$SHA"; fi
fi
[[ "$SESSION_DIR" == /* ]] || SESSION_DIR="$REPO/$SESSION_DIR"
SD="$SESSION_DIR"
export SESSION_DIR DRY_RUN PY

log() { printf '%s\n' "$*"; }
q() { printf '%q ' "$@"; }
# x CMD...: run it, or print it under DRY_RUN (for GPU work, installs, downloads)
x() { if [[ "$DRY_RUN" == 1 ]]; then log "+ $(q "$@")"; else "$@"; fi; }

# tmo SECS CMD...: CMD in its own process group under a wall-clock limit. On expiry the whole group gets
# SIGTERM, then SIGKILL 60 s later (exit 124), so a TERM-ignoring grandchild (a torchrun rank) cannot survive;
# a ctrl-C/TERM/HUP of the session is forwarded to the group the same way, and leftovers of a finished CMD are
# killed. Python, not coreutils timeout: that has no group kill with --foreground and is absent on some hosts.
TMO_PY='
import os, signal, subprocess, sys, time
secs, cmd = float(sys.argv[1]), sys.argv[2:]
try:
    p = subprocess.Popen(cmd, start_new_session=True)
except OSError as e:
    print(f"tmo: cannot run {cmd[0]}: {e}", file=sys.stderr)
    sys.exit(127)
got = []
def on_sig(sig, frame):
    got.append(sig)
for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s, on_sig)
def group(sig):
    try:
        os.killpg(p.pid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False
def wait_group(t):
    end = time.time() + t
    while time.time() < end:
        p.poll()
        if not group(0):
            return True
        time.sleep(0.2)
    return False
def stop(first):
    group(first)
    if not wait_group(60 if first == signal.SIGTERM else 30):
        if first != signal.SIGTERM:
            group(signal.SIGTERM)
            if wait_group(30):
                return
        group(signal.SIGKILL)
        wait_group(10)
deadline, sent = time.time() + secs, 0
while True:
    try:
        rc = p.wait(timeout=0.5)
        break
    except subprocess.TimeoutExpired:
        pass
    if len(got) > sent:
        sent = len(got)
        group(got[-1])
        stop(got[-1])
        p.poll()
        rc = p.returncode
        break
    if time.time() >= deadline:
        print(f"tmo: {cmd[0]} exceeded {secs:.0f} s: stopping its process group", file=sys.stderr, flush=True)
        stop(signal.SIGTERM)
        sys.exit(124)
group(signal.SIGKILL)  # leftovers of a finished command (it ran in its own group)
if got:  # die of the signal, so a waiting bash sees an interrupt, not a normal exit
    signal.signal(got[0], signal.SIG_DFL)
    os.kill(os.getpid(), got[0])
sys.exit(rc if rc is not None and rc >= 0 else 128 - (rc or -1))
'
tmo() { local s=$1; shift; "$PY" -c "$TMO_PY" "$s" "$@"; }
# xlog FILE CMD...: same, with output tee'd to FILE; returns CMD's status
xlog() {
  local f=$1; shift
  if [[ "$DRY_RUN" == 1 ]]; then log "+ $(q "$@")> $f"; return 0; fi
  "$@" 2>&1 | tee "$f"
}
skip() { echo "SKIP $*" > "$SD/status/$PHASE"; log "SKIP $PHASE: $*"; }
status_of() { [[ -f "$SD/status/$1" ]] && head -n1 "$SD/status/$1" || echo none; }
gpu_count() { if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L 2>/dev/null | grep -c GPU || true; else echo 0; fi; }

# ------------------------------------------------------------------------------------------- estimates
ab_estimate() {  # arms n -> node-minutes
  local arms=$1 n=$2
  if [[ "$CACHE" == cold ]]; then echo $((arms * n * COLD_RUN_MIN)); else echo $((arms * COLD_RUN_MIN + arms * (n - 1) * WARM_RUN_MIN)); fi
}
# shellcheck disable=SC2086  # split on commas on purpose
count_csv() { local IFS=,; set -- $1; echo $#; }
n_variants() { local v n=0; for v in $HADAMARD_VARIANTS; do n=$((n + 1)); done; echo "$n"; }
estimate() {
  case "$1" in
    env) echo 2;;
    tests) echo 8;;
    sanity) echo 3;;
    gate) ab_estimate 2 "$(count_csv "$GATE_SEEDS")";;
    hadamard) echo $(( $(ab_estimate $(( $(n_variants) + 1 )) "$(count_csv "$HADAMARD_SEEDS")") - COLD_RUN_MIN + WARM_RUN_MIN ));;  # baseline reuses the gate's cache
    t2) echo 20;;  # ~7 min compile/capture + ~4 min train (t2 README) + the table build
    *) echo 0;;
  esac
}

# ------------------------------------------------------------------------------------------- phase: env
ensure_data() {  # ensure_data N: val shard + >= N train shards at the repo root
  [[ "${SKIP_DATA:-0}" == 1 ]] && return 0
  local n
  n=$(find data/fineweb10B -maxdepth 1 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ' || true)
  if [[ "$n" -ge "$1" && -f data/fineweb10B/fineweb_val_000000.bin ]]; then log "data: $n train shards + val present"; return 0; fi
  x tmo "$DATA_TIMEOUT_S" "$PY" data/cached_fineweb10B.py "$1" || {
    log "FAIL: the $1-shard download did not finish (rc $?; limit DATA_TIMEOUT_S=${DATA_TIMEOUT_S}s). Check the network /"
    log "      Hugging Face access, re-run to resume it (finished shards are kept), or raise DATA_TIMEOUT_S."; return 1; }
}

t2_host_problems() {  # prints one line per reason the host cannot run t2; nothing if it can
  local avail_gib shm_gib cpus ngpu lim cur
  if [[ -r /proc/meminfo ]]; then
    avail_gib=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo || echo 0)
    lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
    if [[ "$lim" =~ ^[0-9]+$ ]]; then
      cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
      (( (lim - cur) / 1073741824 < avail_gib )) && avail_gib=$(( (lim - cur) / 1073741824 ))
    fi
  else
    avail_gib=0
  fi
  shm_gib=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}' || true); shm_gib=${shm_gib:-0}
  cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1); command -v nproc >/dev/null && cpus=$(nproc)
  ngpu=$(gpu_count); [[ "$ngpu" -ge 1 ]] || ngpu=$NPROC
  [[ "$avail_gib" -ge "$T2_MIN_RAM_GIB" ]] || echo "RAM available ${avail_gib} GiB < ${T2_MIN_RAM_GIB} GiB (retrieval tables + arenas + ranks)"
  [[ "$shm_gib" -ge "$T2_MIN_SHM_GIB" ]] || echo "/dev/shm free ${shm_gib} GiB < ${T2_MIN_SHM_GIB} GiB (remount larger: sudo mount -o remount,size=200G /dev/shm; docker: --ipc=host)"
  (( cpus / ngpu >= T2_MIN_CPU_PER_GPU )) || echo "$cpus CPUs for $ngpu GPUs < ${T2_MIN_CPU_PER_GPU}/GPU (LARGE_HOST)"
  local have need free_gib
  have=$(find data/fineweb10B -maxdepth 1 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ' || true)
  need=$(( (T2_SHARDS - ${have:-0}) * 19 / 100 + 5 ))   # 0.19 GiB per shard + slack
  free_gib=$(df -Pk data 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}' || true)
  [[ "${have:-0}" -ge "$T2_SHARDS" || "${free_gib:-0}" -ge "$need" ]] || echo "disk: ${free_gib:-?} GiB free under data/, the missing shards need ~$need GiB"
  return 0
}

t2_prefetch() {  # data + exact_match build, run in the background from env (or inline from t2)
  set -euo pipefail
  tmo "$T2_DATA_TIMEOUT_S" "$PY" data/cached_fineweb10B.py "$T2_SHARDS" || {
    echo "FAIL: the $T2_SHARDS-shard download did not finish (limit T2_DATA_TIMEOUT_S=${T2_DATA_TIMEOUT_S}s); re-run to resume it"; return 4; }
  if ! (cd "$T2_DIR" && "$PY" -c "import exact_match" 2>/dev/null); then
    if ! command -v cargo >/dev/null 2>&1 && [[ -x "$HOME/.cargo/bin/cargo" ]]; then export PATH="$HOME/.cargo/bin:$PATH"; fi
    if ! command -v cargo >/dev/null 2>&1; then
      local inst=${T2_INSTALL_RUST:-auto}
      [[ "$inst" == auto ]] && { [[ -f /.dockerenv ]] && inst=1 || inst=0; }
      if [[ "$inst" == 1 ]]; then
        curl -sSf --max-time 600 https://sh.rustup.rs | sh -s -- -y --profile minimal
        export PATH="$HOME/.cargo/bin:$PATH"
      else
        echo "no cargo: exact_match (Rust) cannot build. Install rustup (https://rustup.rs) or set T2_INSTALL_RUST=1"; return 3
      fi
    fi
    tmo 1800 "$PY" -m pip install -q "maturin>=1.9,<2"
    tmo 3600 "$PY" -m pip install -q "./$T2_DIR/exact_match"
  fi
  echo "t2 prefetch done"
}

# The stack the record needs, checked with the interpreter and torchrun the session will actually use.
# Writes env/stack.json (read by summarize_session.py); exits 1 when something is wrong.
STACK_PY='
import hashlib, json, os, pathlib, platform, re, shutil, subprocess, sys
want = {"torch": "2.10.0+cu128", "triton": "3.6.0"}
out, P = {"python": platform.python_version(), "executable": sys.executable, "prefix": sys.prefix}, []
def sh(*cmd, strip=True):
    try:
        o = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        return o.strip() if strip else o
    except Exception:
        return ""
out["git_sha"] = sh("git", "rev-parse", "HEAD") or None
out["dirty_files"] = [ln[3:] for ln in sh("git", "status", "--porcelain", "--untracked-files=no", strip=False).splitlines() if ln.strip()]
out["git_dirty"] = bool(out["dirty_files"]) if out["git_sha"] else None
out["driver"] = (sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines() or [None])[0]
try:
    import torch
    out.update(torch=torch.__version__, torch_cuda=torch.version.cuda, devices=torch.cuda.device_count(), torch_file=torch.__file__)
except Exception as e:
    out["torch"] = None
    P.append(f"torch does not import with {sys.executable}: {e!r}"[:300])
try:
    import triton
    out["triton"] = triton.__version__
except Exception as e:
    out["triton"] = None
    P.append(f"triton does not import with {sys.executable}: {e!r}"[:300])
for k, v in want.items():
    if out.get(k) and out[k] != v:
        P.append(f"{k} {out[k]} != {v} (the record needs exactly this; node_setup.sh --mode venv installs it)")
tr = shutil.which("torchrun")
out["torchrun"] = tr
if not tr:
    P.append("torchrun is not on PATH (activate the venv: source .venv-run/bin/activate)")
else:
    lines = open(tr, "rb").read(4096).decode(errors="replace").splitlines() or [""]
    interp = lines[0][2:].split() if lines[0].startswith("#!") else []
    if interp and os.path.basename(interp[0]) == "env":
        interp = [a for a in interp[1:] if not a.startswith("-")]
    if interp and os.path.basename(interp[0]) in ("sh", "bash") and len(lines) > 1:  # pip long-path launcher
        m = re.search(r"exec\x27? +\"?([^\" ]+)", lines[1])
        interp = [m.group(1)] if m else interp
    exe = (shutil.which(interp[0]) or interp[0]) if interp else None
    if not exe or "python" not in os.path.basename(exe):
        P.append(f"torchrun at {tr} is not a python console script ({lines[0][:80]!r}): cannot tell whose torch it runs")
    else:
        pre = sh(exe, "-c", "import sys; print(sys.prefix)")
        out["torchrun_prefix"] = pre
        pre_s = pre or "?"
        if os.path.realpath(pre or "/nonexistent") != os.path.realpath(sys.prefix):
            P.append(f"torchrun at {tr} runs {exe} (prefix {pre_s}), not the session interpreter {sys.executable} "
                     f"(prefix {sys.prefix}): the ranks would import a different torch. Activate the venv, or set PY")
attn = pathlib.Path("track_1_short/model/attention.py")
if attn.is_file():
    ns = {}
    for line in attn.read_text().splitlines():
        if line.startswith(("FA3_REPO", "FA3_REVISION", "FA3_BINARY_SHA256")):
            exec(line, ns)
    try:
        from kernels import get_kernel
        fa3 = get_kernel(ns["FA3_REPO"], revision=ns["FA3_REVISION"], trust_remote_code=True)
        b = next(pathlib.Path(fa3.__file__).parent.glob("_flash_attn3_cuda_*.so"))
        d = hashlib.sha256(b.read_bytes()).hexdigest()
        out["fa3"] = "ok" if d == ns.get("FA3_BINARY_SHA256") else f"sha256 {d[:16]} != pinned"
        if out["fa3"] != "ok":
            pinned = ns.get("FA3_BINARY_SHA256")
            P.append(f"FA3 binary {b} sha256 {d} != pinned {pinned}")
    except Exception as e:
        out["fa3"] = "FAILED"
        P.append(f"FA3 does not load the way the trainer loads it (track_1_short/model/attention.py): {e!r}"[:400]
                 + " -- libcudart.so.13 on LD_LIBRARY_PATH? `hf auth login`, or HF_HUB_OFFLINE=1 after the node_setup.sh prefetch")
else:
    out["fa3"] = "not checked (no track_1_short/model/attention.py)"
out["problems"] = P
out["allow_stack"] = os.environ.get("ALLOW_STACK") == "1"
json.dump(out, open(sys.argv[1], "w"), indent=1)
g = out.get
print("python", g("python"), f"({sys.executable})")
print("torch", g("torch"), "cuda", g("torch_cuda"), "devices", g("devices"))
print("triton", g("triton"))
print("torchrun", tr)
print("FA3", g("fa3"))
for p in P:
    print("STACK PROBLEM: " + p)
sys.exit(1 if P else 0)
'

phase_env() {
  local d="$SD/env" drv stack_rc=0
  mkdir -p "$d"
  {
    echo "date $(date -Is 2>/dev/null || date)  host $(hostname)"
    echo "git $(git rev-parse HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
    git log -1 --format='%h %s' 2>/dev/null || true
    echo "modified tracked files (git status --porcelain --untracked-files=no):"; git status --porcelain --untracked-files=no 2>/dev/null || true
    echo "untracked files (not used by the A/B arms, which run the committed HEAD):"; git status --porcelain 2>/dev/null | grep '^??' || true
    echo; echo "## nvidia-smi"; nvidia-smi 2>&1 || echo "nvidia-smi not found"
    echo; echo "## python stack"
    "$PY" -c "$STACK_PY" "$d/stack.json" 2>&1 || echo "stack check rc=$?"
    echo; echo "## CPU / RAM / shm"
    echo "CPUs $(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
    free -g 2>/dev/null || true
    df -h /dev/shm 2>/dev/null || true
    echo; echo "## t2 host check"; t2_host_problems | sed 's/^/t2 blocker: /'
  } 2>&1 | tee "$d/env.txt"
  "$PY" -m pip freeze > "$d/pip_freeze.txt" 2>/dev/null || "$PY" -c 'import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()): print(d.metadata["Name"] + "==" + d.version)' > "$d/pip_freeze.txt" 2>&1 || true
  if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    log "WARN: modified tracked files. The gate/hadamard arms run the COMMITTED HEAD $SHA (ab_run pins refs), not the working tree."
  fi
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)
  if [[ -n "$drv" && "${drv%%.*}" -lt 580 ]]; then
    log "WARN: driver $drv < 580: the ANVIL2 README measured a uniform ~35% slowdown below 580 (val loss unaffected)" | tee -a "$d/env.txt"
  fi
  # The node report from the existing tool; keep any report node_setup.sh wrote earlier (it has the stack check).
  [[ -f node_report.txt ]] && cp node_report.txt "$d/node_report.setup.txt"
  if [[ "$DRY_RUN" == 1 ]]; then
    log "+ bash 182/tools/node_setup.sh --mode report   (writes node_report.txt; not run under DRY_RUN)"
  else
    bash 182/tools/node_setup.sh --mode report >/dev/null 2>&1 || true
    if [[ -f node_report.txt ]]; then
      mv node_report.txt "$d/node_report.txt"
      [[ -f "$d/node_report.setup.txt" ]] && cp "$d/node_report.setup.txt" node_report.txt
    fi
  fi
  xlog "$d/node_check.txt" tmo 300 torchrun --standalone --nproc_per_node="$NPROC" 182/tools/node_check.py || log "WARN: node_check.py failed (see $d/node_check.txt)"
  ensure_data 9 || return 1
  if [[ " ${SELECTED:-} " == *" t2 "* && -z "$(t2_host_problems)" && ! -f "$SD/t2/prefetch.pid" ]]; then
    mkdir -p "$SD/t2"
    if [[ "$DRY_RUN" == 1 ]]; then log "+ (background) t2 prefetch: $PY data/cached_fineweb10B.py $T2_SHARDS; pip install ./$T2_DIR/exact_match"
    else
      local lowprio=(nice -n 19); command -v ionice >/dev/null 2>&1 && lowprio+=(ionice -c2 -n7)
      # low priority: it overlaps the timed A/B runs (their data loaders and the hadamard step times)
      nohup "${lowprio[@]}" bash -c "bash $(q "$SELF") --t2-prefetch; echo \$? > $(q "$SD/t2/prefetch.rc")" > "$SD/t2/prefetch.txt" 2>&1 &
      echo $! > "$SD/t2/prefetch.pid"
      log "t2: 103-shard download + exact_match build started in the background (pid $!, log $SD/t2/prefetch.txt)"
    fi
  fi
  [[ -n "$drv" || "$DRY_RUN" == 1 ]] || { log "FAIL: no GPU driver visible"; return 1; }
  if grep -q '^STACK PROBLEM: ' "$d/env.txt"; then
    if [[ "$DRY_RUN" == 1 ]]; then log "(dry run) env would FAIL on the stack check above (ALLOW_STACK=1 overrides)"
    elif [[ "${ALLOW_STACK:-0}" == 1 ]]; then log "WARN: ALLOW_STACK=1: running on a stack the record was not certified on (see STACK PROBLEM above)"
    else
      log "FAIL: the stack is not the record's (STACK PROBLEM lines above; $d/stack.json). The GPU phases are skipped:"
      log "      results on another torch/triton/FA3 would not say anything about the record. Fix it (bash 182/tools/node_setup.sh"
      log "      --mode venv && source .venv-run/bin/activate), then re-run; or ALLOW_STACK=1 to run anyway."
      return 1
    fi
  fi
}

# ----------------------------------------------------------------------------------------- phase: tests
phase_tests() {
  local d="$SD/tests" rc1=0 rc2=0
  mkdir -p "$d"
  "$PY" -c "import pytest" 2>/dev/null || x tmo 600 "$PY" -m pip install -q pytest
  # -rfEs: name every failed/errored test and every skip reason (the summary shows them)
  xlog "$d/pytest.txt" tmo "$TESTS_TIMEOUT_S" "$PY" -m pytest 182/hadamard-fp8/tests -q -rfEs -p no:cacheprovider || rc1=$?
  echo "$rc1" > "$d/rc_plain"
  xlog "$d/pytest_hadamard.txt" tmo "$TESTS_TIMEOUT_S" env HADAMARD_FP8=1 "$PY" -m pytest 182/hadamard-fp8/tests -q -rfEs -s -p no:cacheprovider || rc2=$?
  echo "$rc2" > "$d/rc_hadamard"
  log "tests: plain rc=$rc1, HADAMARD_FP8=1 rc=$rc2"
  local f
  for f in pytest.txt pytest_hadamard.txt; do
    [[ -f "$d/$f" ]] && grep -E '^(FAILED|ERROR) ' "$d/$f" | sed "s|^|  $f: |" | head -n 12 || true
  done
  [[ "$rc1" == 0 && "$rc2" == 0 ]]
}

# ---------------------------------------------------------------------------------------- phase: sanity
phase_sanity() {
  local d="$SD/sanity"
  mkdir -p "$d"
  if [[ "$SANITY_DEFAULT" == 1 && ! -f 182/hadamard-fp8/gpu_sanity.py ]]; then
    skip "182/hadamard-fp8/gpu_sanity.py is not in this checkout (not merged yet)"; return 0
  fi
  local cmd=$SANITY_CMD
  [[ "$SANITY_DEFAULT" == 1 ]] && cmd+=" --json $d/sanity.json"
  # shellcheck disable=SC2086  # a command line
  xlog "$d/sanity.txt" tmo "$SANITY_TIMEOUT_S" $cmd
}

# ----------------------------------------------------------------------------------------- A/B helpers
ab_check() {  # ab_check gate|ab DIR: 0 complete+usable, 3 slots pending, 4 unusable (re-plan), 5 an arm failed every run, 2 no plan
  if [[ "$1" == gate ]]; then "$PY" 182/tools/summarize_session.py --check "$2" --check-gate
  else "$PY" 182/tools/summarize_session.py --check "$2"; fi
}
ab_plan_run() {  # ab_plan_run gate|ab DIR [plan args...]: plan once, run/resume, report; returns ab_check's code
  local kind=$1 out=$2 rc=0 k=1 why dead=stop; shift 2
  [[ "$kind" == ab ]] && dead=abandon   # a broken hadamard variant must not stop the other variants
  if [[ -f "$out/plan.json" ]]; then
    why=$(ab_check "$kind" "$out") || rc=$?
    if [[ "${FORCE:-0}" == 1 || "$rc" == 4 ]]; then
      # ab_run never re-runs a finished slot, so a dead attempt (e.g. scattered crashes) is set aside, kept
      # in the bundle, and the phase is planned afresh.
      while [[ -e "$out.attempt$k" ]]; do k=$((k + 1)); done
      log "re-planning: previous attempt moved to $out.attempt$k ($([[ "$rc" == 4 ]] && echo "unusable: $why" || echo FORCE=1))"
      x mv "$out" "$out.attempt$k"
      [[ "$DRY_RUN" == 1 ]] && { log "(dry run) the re-plan would start from an empty $(basename "$out")"; return 0; }
    elif [[ "$rc" == 5 ]]; then
      log "NOT re-planning $(basename "$out"): $why."
      log "  A re-plan would burn the same runs again. Fix the cause, then: FORCE=1 bash 182/gpu_session.sh $(basename "$out")"
    else
      log "plan exists: $out/plan.json (resuming: $why)"
    fi
  fi
  if [[ ! -f "$out/plan.json" ]]; then
    "$PY" 182/tools/ab_run.py plan --out "$out" --repo "$REPO" --work-root "$SD/work" --cache "$CACHE" \
      --order blocked --order-seed 182 --nproc "$NPROC" --timeout-s "$TIMEOUT_S" --cmd "$RUN_CMD" "$@" || return 1
  fi
  x "$PY" 182/tools/ab_run.py run "$out" --no-report --stop-after-failures "$STOP_AFTER_FAILURES" \
    --arm-failures "$ARM_FAILURES" --dead-arm "$dead" || return 1
  [[ "$DRY_RUN" == 1 ]] && return 0
  "$PY" 182/tools/ab_run.py report "$out" --quiet || true
  rc=0; why=$(ab_check "$kind" "$out") || rc=$?
  [[ "$rc" == 0 ]] || log "A/B $(basename "$out") not usable: $why"
  return "$rc"
}

# ------------------------------------------------------------------------------------------ phase: gate
phase_gate() {
  local g="$SD/gate"
  mkdir -p "$SD"
  ensure_data 9 || return 1
  local rc=0
  ab_plan_run gate "$g" --arm fp8=ref:HEAD --arm bf16=ref:HEAD --env bf16:FP8_OFF=1 --baseline fp8 \
    --n "$(count_csv "$GATE_SEEDS")" --seeds "$GATE_SEEDS" || rc=$?
  [[ -f "$g/plan.json" ]] || { [[ "$DRY_RUN" == 1 ]] && return 0; return 1; }
  # decision.env says GATE_RUN_HADAMARD=0 unless every slot finished (summarize_session: INCOMPLETE otherwise)
  "$PY" 182/tools/summarize_session.py --gate "$g" --write-decision "$g/decision.env" | tee "$g/decision.txt"
  [[ "$DRY_RUN" == 1 || "$rc" == 0 ]] && return 0
  case "$rc" in
    3) log "FAIL: the gate stopped with slots pending (causes above): no decision. Fix the cause and re-run the same command to resume.";;
    4) log "FAIL: the gate finished without 2 usable seed pairs: no decision. The next re-run re-plans it.";;
    5) log "FAIL: one gate arm failed every run (causes above): no decision. Fix it, then FORCE=1 bash 182/gpu_session.sh gate";;
    *) log "FAIL: the gate did not run (rc $rc)";;
  esac
  return 1
}

# -------------------------------------------------------------------------------------- phase: hadamard
phase_hadamard() {
  local h="$SD/hadamard" why="" v gemms block name args=() gst
  if [[ "${FORCE_HADAMARD:-0}" != 1 ]]; then
    gst=$(status_of gate)
    if [[ ! -f "$SD/gate/decision.env" || "$gst" != PASS* ]]; then
      why="the gate phase did not complete (status: $gst): there is no decision to act on (finish the gate, or FORCE_HADAMARD=1)"
    else
      # shellcheck disable=SC1091
      source "$SD/gate/decision.env"
      if [[ "${GATE_RUN_HADAMARD:-0}" != 1 || "${GATE_COMPLETE:-0}" != 1 ]]; then
        case "${GATE_DECISION:-?}" in
          NODATA|INCOMPLETE) why="gate decision ${GATE_DECISION}: no usable gate (fewer than 2 complete seed pairs), nothing to act on (FORCE_HADAMARD=1 overrides)";;
          STOP) why="gate decision STOP: 95% CI upper ${GATE_HI_MNAT:-?} mnat < 0.5, the fp8 recipe loses nothing worth recovering (FORCE_HADAMARD=1 overrides)";;
          *) why="gate decision ${GATE_DECISION:-?}: mean gap ${GATE_MEAN_MNAT:-?} mnat < HADAMARD_MIN_GAP_MNAT (${HADAMARD_MIN_GAP_MNAT:-2.0}), a loose prescreen for the default HADAMARD_VARIANTS (~3-4 mnat to pay at 0.164 s/mnat). Cheaper variants are not excluded: HADAMARD_VARIANTS=dpost@32 FORCE_HADAMARD=1 times one (~0.5 mnat to pay)";;
        esac
      fi
    fi
    [[ -z "$why" && "$(cat "$SD/tests/rc_hadamard" 2>/dev/null || echo 0)" != 0 ]] && why="the HADAMARD_FP8=1 tests failed ($SD/tests/pytest_hadamard.txt)"
    [[ -z "$why" && "$(status_of sanity)" == FAIL* ]] && why="gpu_sanity.py failed ($SD/sanity/sanity.txt)"
    if [[ -n "$why" ]]; then
      if [[ "$DRY_RUN" == 1 ]]; then log "(dry run) would skip: $why; showing the commands anyway"; else skip "$why"; return 0; fi
    fi
  fi
  local spec='^(all|(dw1|dw2|dx|dpost)(,(dw1|dw2|dx|dpost))*)@(32|64|128)$'
  for v in $HADAMARD_VARIANTS; do  # a bad spec would crash every run at import: refuse before any GPU time
    [[ "$v" =~ $spec ]] || { log "FAIL: HADAMARD_VARIANTS entry '$v' is not GEMMS@BLOCK (GEMMS: all or a comma list of dw1,dw2,dx,dpost; BLOCK: 32, 64 or 128)"; return 1; }
  done
  if grep -rqs HADAMARD_GEMMS track_1_short && grep -rqs HADAMARD_BLOCK track_1_short; then
    for v in $HADAMARD_VARIANTS; do
      gemms=${v%@*}; block=${v#*@}
      name="h_$(printf '%s_b%s' "$gemms" "$block" | tr -c 'A-Za-z0-9_\n' '_')"
      args+=(--arm "$name=ref:HEAD" --env "$name:HADAMARD_FP8=1" --env "$name:HADAMARD_GEMMS=$gemms" --env "$name:HADAMARD_BLOCK=$block")
    done
  else
    log "WARN: this checkout has no HADAMARD_GEMMS/HADAMARD_BLOCK knobs: one arm, HADAMARD_FP8=1 (block 128, all four GEMMs)"
    args+=(--arm "h_default=ref:HEAD" --env "h_default:HADAMARD_FP8=1")
  fi
  local rc=0
  ab_plan_run ab "$h" --arm fp8=ref:HEAD "${args[@]}" --baseline fp8 \
    --n "$(count_csv "$HADAMARD_SEEDS")" --seeds "$HADAMARD_SEEDS" || rc=$?
  [[ "$DRY_RUN" == 1 || ! -f "$h/plan.json" ]] || "$PY" 182/tools/summarize_session.py --hadamard "$SD" || true
  return "$rc"
}

# ---------------------------------------------------------------------------------------------- phase: t2
phase_t2() {
  local d="$SD/t2" problems pid rc n rel
  problems=$(t2_host_problems)
  if [[ -n "$problems" ]]; then
    mkdir -p "$d"; printf '%s\n' "$problems" > "$d/preflight.txt"
    if [[ "${T2_FORCE:-0}" != 1 ]]; then skip "host cannot run t2: $(printf '%s' "$problems" | paste -sd ';' -) (T2_FORCE=1 overrides)"; return 0; fi
    log "WARN: T2_FORCE=1, running despite: $problems"
  fi
  mkdir -p "$d"
  if [[ -f "$d/prefetch.pid" ]]; then
    pid=$(cat "$d/prefetch.pid")
    local waited=0
    # prefetch.rc is written when the background job ends (an unreaped zombie still answers kill -0)
    while [[ ! -f "$d/prefetch.rc" ]] && kill -0 "$pid" 2>/dev/null; do
      if (( waited >= ${T2_PREFETCH_WAIT_S:-3600} )); then
        log "FAIL: the background prefetch (pid $pid) is still running after ${waited}s: $(tail -n2 "$d/prefetch.txt")"; return 1
      fi
      (( waited % 300 == 0 )) && log "t2: waiting for the background prefetch (pid $pid; $d/prefetch.txt)"
      sleep 30; waited=$((waited + 30))
    done
    rm -f "$d/prefetch.pid"
  fi
  if [[ -f "$d/prefetch.rc" ]]; then  # the background prefetch finished
    rc=$(cat "$d/prefetch.rc"); rm -f "$d/prefetch.rc"
    if [[ "$rc" != 0 ]]; then log "FAIL: t2 prefetch exited $rc: $(tail -n3 "$d/prefetch.txt")"; return 1; fi
  elif [[ "$DRY_RUN" == 1 ]]; then
    log "+ t2 prefetch: $PY data/cached_fineweb10B.py $T2_SHARDS; pip install ./$T2_DIR/exact_match"
  else
    t2_prefetch 2>&1 | tee "$d/prefetch.txt"
  fi
  if [[ "$DRY_RUN" != 1 ]]; then
    n=$(find data/fineweb10B -maxdepth 1 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ' || true)
    [[ "$n" -eq "$T2_SHARDS" ]] || { log "FAIL: $n train shards, t2 needs all $T2_SHARDS (the exact-count corpus depends on it)"; return 1; }
  fi
  (cd "$T2_DIR" && x "$PY" -m track_1_short.memory_budget) > "$d/memory_budget.txt" 2>&1 || true
  rc=0
  (cd "$T2_DIR" && DATA_PATH=../.. xlog "$d/stdout.txt" tmo "$T2_TIMEOUT_S" torchrun --standalone --nproc_per_node="$NPROC" train_gpt_medium_cpu.py) || rc=$?
  if [[ "$DRY_RUN" != 1 ]]; then
    rel=$(grep -m1 -E '^logs/[0-9A-Za-z_.-]+\.txt$' "$d/stdout.txt" || true)
    [[ -n "$rel" && -f "$T2_DIR/$rel" ]] && cp "$T2_DIR/$rel" "$d/"
    grep -E '^(endpoint_result|step:[0-9]+/[0-9]+ val_loss)' "$d/stdout.txt" | tail -n2 || true
    if [[ "$rc" == 0 ]] && ! grep -qE '^(endpoint_result|step:([0-9]+)/\2 val_loss)' "$d/stdout.txt"; then
      log "FAIL: torchrun exited 0 but printed no final result (no endpoint_result / final val_loss line): $(tail -n2 "$d/stdout.txt")"
      rc=1
    fi
  fi
  return "$rc"
}

# ------------------------------------------------------------------------------------------ phase: bundle
phase_bundle() {
  local date base out
  date=$(sed -n 's/^session_date //p' "$SD/meta.txt" 2>/dev/null | head -n1); date=${date:-$(date +%F)}
  base=$(basename "$SD")
  out="$(dirname "$SD")/session-$date-$SHA.tar.gz"
  "$PY" 182/tools/summarize_session.py "$SD" > "$SD/summary.txt" || log "WARN: summarize_session.py failed"
  cat "$SD/summary.txt"
  COPYFILE_DISABLE=1 tar -czf "$out.tmp" -C "$(dirname "$SD")" --exclude="$base/work" "$base"
  mv "$out.tmp" "$out"
  log ""
  log "SEND THIS FILE BACK: $out ($(du -h "$out" | cut -f1))"
  echo "$out" > "$SD/bundle_path"
}

# ---------------------------------------------------------------------------------------------- driver
run_phase() {
  local p=$1 rc=0 est
  if [[ "${FORCE:-0}" != 1 && "$(status_of "$p")" == PASS* && "$p" != bundle ]]; then
    log "== $p: already PASS ($SD/status/$p); FORCE=1 redoes it"; return 0
  fi
  est=$(estimate "$p")
  log ""; log "== $p: est. $est node-min (~$((est * 8)) GPU-min) -- $(date '+%F %T')"
  rm -f "$SD/status/$p"
  SELECTED="${SELECTED_PHASES[*]}" PHASE="$p" bash "$SELF" --run-phase "$p" || rc=$?
  if [[ "$rc" != 0 ]]; then echo "FAIL rc=$rc" > "$SD/status/$p"
  elif [[ ! -f "$SD/status/$p" ]]; then echo "PASS" > "$SD/status/$p"; fi
  log "== $p: $(status_of "$p")"
}

case "${1:-}" in
  --run-phase) PHASE=$2; "phase_$2"; exit $?;;
  --t2-prefetch) t2_prefetch; exit $?;;
  -h|--help) sed -n '2,56p' "$SELF"; exit 0;;
esac

WANT=("$@")
if [[ ${#WANT[@]} -eq 0 ]]; then
  if [[ "${QUICK:-0}" == 1 ]]; then WANT=(env sanity gate bundle); else WANT=(all); fi
fi
SELECTED_PHASES=()
for p in "${PHASES[@]}"; do
  for w in "${WANT[@]}"; do
    if [[ "$w" == all || "$w" == "$p" ]]; then SELECTED_PHASES+=("$p"); break; fi
  done
done
for w in "${WANT[@]}"; do
  [[ "$w" == all || " ${PHASES[*]} " == *" $w "* ]] || { echo "unknown phase '$w' (phases: ${PHASES[*]} | all)" >&2; exit 2; }
done

if [[ -n "${SESSION_DOCKER:-}" && ! -f /.dockerenv ]]; then
  DOCKER=(docker); docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
  HFC="${HF_HOME:-$HOME/.cache/huggingface}"
  NAME="gpu_session_${SHA}_$$"
  GPUS=(); [[ "${SESSION_DOCKER_GPUS:-all}" != none ]] && GPUS=(--gpus "${SESSION_DOCKER_GPUS:-all}")
  # -t: ctrl-C in the terminal reaches the container's foreground processes directly; --init (tini, killing the
  # whole process group): a signal proxied by the docker client reaches the running phase, not just a PID-1 bash
  # that would defer its trap until the ~40 min child exits.
  TTY=(); [[ -t 0 && -t 1 ]] && TTY=(-it)
  CMD=("${DOCKER[@]}" run --rm --init --name "$NAME" "${TTY[@]}" "${GPUS[@]}" --ipc=host --network=host
       --ulimit memlock=-1 --ulimit stack=67108864
       -v "$REPO:$REPO" -v "$HFC:/root/.cache/huggingface" -w "$REPO" -e SESSION_DIR="$SD" -e TINI_KILL_PROCESS_GROUP=1
       -e SESSION_HOST_UID="$(id -u)" -e SESSION_HOST_GID="$(id -g)"
       -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' ${USER_ENV[@]+"${USER_ENV[@]}"}
       "$SESSION_DOCKER" bash 182/gpu_session.sh "${WANT[@]}")
  if [[ "$DRY_RUN" == 1 ]]; then log "+ $(q "${CMD[@]}")"; exit 0; fi
  mkdir -p "$(dirname "$SD")" "$HFC"
  # The client runs in the background so this shell can act on TERM/HUP (tmux pane closed, kill) at once:
  # `docker stop` the container rather than leave it holding all 8 GPUs.
  stop_container() {
    trap '' INT TERM HUP
    log ""; log "stopping container $NAME (docker stop; the phase's processes get SIGTERM, then SIGKILL after 60 s)"
    "${DOCKER[@]}" stop -t 60 "$NAME" >/dev/null 2>&1 || true
    wait "$DPID" 2>/dev/null || true
    log "interrupted. Re-run the same command to resume, or pack what exists: SESSION_DOCKER=$SESSION_DOCKER bash 182/gpu_session.sh bundle"
    exit 130
  }
  trap stop_container TERM HUP
  [[ ${#TTY[@]} -eq 0 ]] && trap stop_container INT   # with -t the terminal sends ctrl-C into the container itself
  "${CMD[@]}" <&0 &
  DPID=$!
  rc=0; wait "$DPID" || rc=$?
  exit "$rc"
fi

# DRY_RUN never writes outside a temp dir: with an existing SESSION_DIR it previews a resume from a copy.
if [[ "$DRY_RUN" == 1 && "$SD" != "${TMPDIR:-/tmp}"/gpu_session_dry.*/* ]]; then
  REAL_SD=$SD
  SD="$(mktemp -d "${TMPDIR:-/tmp}/gpu_session_dry.XXXXXX")/$(basename "$REAL_SD")"
  if [[ -d "$REAL_SD" ]]; then
    mkdir -p "$SD"
    tar -cf - -C "$REAL_SD" --exclude=./work . | tar -xf - -C "$SD"
    echo "DRY_RUN: previewing a resume of $REAL_SD from a copy in $SD (the real session is not touched)"
  fi
  SESSION_DIR=$SD; export SESSION_DIR
fi

mkdir -p "$SD/status"
if [[ -f /.dockerenv && -n "${SESSION_HOST_UID:-}" && "${SESSION_HOST_UID}" != 0 && "$DRY_RUN" != 1 ]]; then
  # docker runs as root: hand what this session created (results, worktrees, data, HF cache, __pycache__) back to the host user
  chown_back() { find "$REPO" "$SD" /root/.cache/huggingface -xdev -uid 0 -exec chown -h "$SESSION_HOST_UID:${SESSION_HOST_GID:-$SESSION_HOST_UID}" {} + 2>/dev/null || true; }
  trap chown_back EXIT
fi
# The log tee ignores ctrl-C/TERM/HUP (set from python, since bash 5.3 resets a subshell's ignored INT before
# exec), so the interrupt message below still reaches the terminal and session.log.
exec > >(exec "$PY" -c 'import os, signal, sys
for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(s, signal.SIG_IGN)
os.execvp("tee", ["tee", "-a", sys.argv[1]])' "$SD/session.log") 2>&1
if [[ ! -f "$SD/meta.txt" ]]; then
  { echo "session_date $(date +%F)"; echo "sha $(git rev-parse HEAD 2>/dev/null || echo nogit)"; echo "host $(hostname)"; } > "$SD/meta.txt"
fi
echo "invocation $(date '+%F %T') phases=${SELECTED_PHASES[*]}$([[ "$DRY_RUN" == 1 ]] && echo " DRY_RUN=1")" >> "$SD/meta.txt"

total=0; for p in "${SELECTED_PHASES[@]}"; do total=$((total + $(estimate "$p"))); done
log "GPU session $SHA -> $SD"
log "phases: ${SELECTED_PHASES[*]}   upper-bound estimate $total node-min (~$((total * 8)) GPU-min; hadamard/t2 may skip)"
[[ "$DRY_RUN" == 1 ]] && log "DRY_RUN=1: GPU/pip/download commands are printed, not run; files go to $SD"

CUR_PHASE=""
on_interrupt() {
  [[ -n "$CUR_PHASE" ]] && echo "FAIL interrupted" > "$SD/status/$CUR_PHASE"
  log ""; log "interrupted during ${CUR_PHASE:-startup}. Re-run the same command to resume, or pack what exists:"
  log "  bash 182/gpu_session.sh bundle"
  exit 130
}
trap on_interrupt INT TERM HUP
FAILED=()
ENV_FAILED=0
for p in "${SELECTED_PHASES[@]}"; do
  CUR_PHASE=$p
  if [[ "$ENV_FAILED" == 1 && "$p" != bundle ]]; then
    echo "SKIP env failed (stack/driver; see env above): nothing GPU-side ran" > "$SD/status/$p"
    log ""; log "== $p: SKIP (env failed: fix the stack and re-run, or ALLOW_STACK=1)"; continue
  fi
  run_phase "$p"
  [[ "$(status_of "$p")" == FAIL* ]] && FAILED+=("$p")
  [[ "$p" == env && "$(status_of env)" == FAIL* ]] && ENV_FAILED=1
done
log ""
for p in "${SELECTED_PHASES[@]}"; do log "  $p: $(status_of "$p")"; done
[[ -f "$SD/bundle_path" && " ${SELECTED_PHASES[*]} " == *" bundle "* ]] && log "SEND THIS FILE BACK: $(cat "$SD/bundle_path")"
[[ ${#FAILED[@]} -eq 0 ]] || { log "failed phases: ${FAILED[*]}"; exit 1; }
