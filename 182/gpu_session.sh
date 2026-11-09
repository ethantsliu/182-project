#!/usr/bin/env bash
# One GPU session on the 8xH100 box -> one file to send back. Run from anywhere inside the repo.
#
#   bash 182/gpu_session.sh              # = all: env tests sanity gate hadamard t2 bundle
#   QUICK=1 bash 182/gpu_session.sh      # = env sanity gate bundle: the decision gate only (n=4 still)
#   bash 182/gpu_session.sh sanity gate  # just those phases (in the canonical order)
#   DRY_RUN=1 bash 182/gpu_session.sh    # print every GPU/pip/download command; writes only to a temp dir
#
# Resumable: re-run the same command after a crash or ctrl-C. A phase whose status is PASS is skipped
# (FORCE=1 redoes it); the A/B phases resume inside ab_run.py at the first unfinished run. Every phase is
# logged to 182/results/session-<sha>/ (gitignored); `all` always ends with the bundle, even after a failed
# phase. An A/B phase stops after STOP_AFTER_FAILURES consecutive failed runs (slots stay pending; re-run
# resumes them); one that finished unusable (gate < 2 seed pairs, or an arm with no ok run) is FAIL, and the
# next re-run moves it to <phase>.attemptN and re-plans it (FORCE=1 always re-plans). Every GPU command has
# a timeout. After ctrl-C, `bash 182/gpu_session.sh bundle` packs what exists. Run it inside tmux/screen.
# The script never commits or pushes.
#
# Phases (est. node-minutes printed before each; GPU-minutes = 8x):
#   env       git sha/status, node report (182/tools/node_setup.sh --mode report), driver >= 580 check,
#             torch/triton, CPUs, RAM, /dev/shm; node_check.py (GEMM/NCCL); 9 data shards if missing.
#             When t2 is selected and the host can run it, starts the 103-shard download + exact_match build
#             in the background (CPU/network only, overlaps the GPU phases).
#   tests     pytest 182/hadamard-fp8/tests, then again with HADAMARD_FP8=1
#   sanity    182/hadamard-fp8/gpu_sanity.py (skipped if absent)
#   gate      README decision gate: record vs FP8_OFF=1, GATE_SEEDS paired, interleaved (ab_run.py);
#             prints the paired gap mean +- SE, its 95% CI vs 0.5 mnat, and the decision
#   hadamard  only if the gate is GAP/INCONCLUSIVE with mean gap >= HADAMARD_MIN_GAP_MNAT (2.0) (or FORCE_HADAMARD=1), and the HADAMARD_FP8 tests /
#             sanity did not fail: record vs HADAMARD_FP8=1 per HADAMARD_VARIANTS (GEMMS@BLOCK), HADAMARD_SEEDS
#   t2        variants/t2-cpu-ngram at its default 5800 steps, once (skipped with the reason if the host
#             lacks ~390 GiB RAM, 144 GiB free /dev/shm, 24 vCPU/GPU, or the 103 shards)
#   bundle    summary.txt + tar.gz of every log -> 182/results/session-<date>-<sha>.tar.gz (prints the path)
#
# Knobs (env): PY (python), NPROC (8), CACHE (warm|cold; warm = one compile cache per arm), GATE_SEEDS
# (1,2,3,4), HADAMARD_VARIANTS ("all@128 dw1,dw2@32"), HADAMARD_SEEDS (1,2), FORCE_HADAMARD, TIMEOUT_S
# (1500 per A/B run), STOP_AFTER_FAILURES (3), T2_TIMEOUT_S (5400), TESTS_TIMEOUT_S (1800 per pytest),
# SANITY_TIMEOUT_S (1200), T2_PREFETCH_WAIT_S (3600), QUICK, RUN_CMD (ab_run --cmd template), SANITY_CMD, SKIP_DATA, T2_FORCE (ignore host checks),
# T2_INSTALL_RUST (auto: only inside a container), SESSION_DIR, SESSION_DOCKER=<image> (re-run this script
# inside that image, e.g. modded-nanogpt from node_setup.sh --mode docker), FORCE, DRY_RUN.
set -euo pipefail

PHASES=(env tests sanity gate hadamard t2 bundle)
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$REPO"

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
T2_TIMEOUT_S=${T2_TIMEOUT_S:-5400}; TESTS_TIMEOUT_S=${TESTS_TIMEOUT_S:-1800}; SANITY_TIMEOUT_S=${SANITY_TIMEOUT_S:-1200}
RUN_CMD=${RUN_CMD:-"torchrun --standalone --nproc_per_node={nproc} {script}"}
SANITY_CMD=${SANITY_CMD:-"$PY 182/hadamard-fp8/gpu_sanity.py"}
COLD_RUN_MIN=${COLD_RUN_MIN:-9}   # one Track 1 run with a cold compile cache (~7 min compile + train + val)
WARM_RUN_MIN=${WARM_RUN_MIN:-4}
T2_MIN_RAM_GIB=${T2_MIN_RAM_GIB:-390}; T2_MIN_SHM_GIB=${T2_MIN_SHM_GIB:-144}; T2_MIN_CPU_PER_GPU=${T2_MIN_CPU_PER_GPU:-24}
T2_SHARDS=103; T2_DIR=variants/t2-cpu-ngram
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
# tmo SECS CMD...: CMD under a wall-clock limit (exit 124 on expiry). --foreground keeps CMD in our process
# group, so ctrl-C still reaches torchrun; torchrun forwards the SIGTERM to its ranks.
tmo() {
  local s=$1; shift
  if timeout --foreground 5 true >/dev/null 2>&1; then timeout --foreground -k 60 "$s" "$@"
  else log "WARN: no coreutils timeout: running without the ${s}s limit"; "$@"; fi
}
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
  x "$PY" data/cached_fineweb10B.py "$1"
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
  "$PY" data/cached_fineweb10B.py "$T2_SHARDS"
  if ! (cd "$T2_DIR" && "$PY" -c "import exact_match" 2>/dev/null); then
    if ! command -v cargo >/dev/null 2>&1 && [[ -x "$HOME/.cargo/bin/cargo" ]]; then export PATH="$HOME/.cargo/bin:$PATH"; fi
    if ! command -v cargo >/dev/null 2>&1; then
      local inst=${T2_INSTALL_RUST:-auto}
      [[ "$inst" == auto ]] && { [[ -f /.dockerenv ]] && inst=1 || inst=0; }
      if [[ "$inst" == 1 ]]; then
        curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        export PATH="$HOME/.cargo/bin:$PATH"
      else
        echo "no cargo: exact_match (Rust) cannot build. Install rustup (https://rustup.rs) or set T2_INSTALL_RUST=1"; return 3
      fi
    fi
    "$PY" -m pip install -q "maturin>=1.9,<2"
    "$PY" -m pip install -q "./$T2_DIR/exact_match"
  fi
  echo "t2 prefetch done"
}

phase_env() {
  local d="$SD/env" drv
  mkdir -p "$d"
  {
    echo "date $(date -Is 2>/dev/null || date)  host $(hostname)"
    echo "git $(git rev-parse HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
    git log -1 --format='%h %s' 2>/dev/null || true
    echo "git status --short:"; git status --short 2>/dev/null || true
    echo; echo "## nvidia-smi"; nvidia-smi 2>&1 || echo "nvidia-smi not found"
    echo; echo "## python stack"
    "$PY" - <<'EOF' 2>&1 || true
import platform
print("python", platform.python_version())
try:
    import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
except Exception as e: print("torch: import failed:", e)
try:
    import triton; print("triton", triton.__version__)
except Exception as e: print("triton: import failed:", e)
EOF
    echo; echo "## CPU / RAM / shm"
    echo "CPUs $(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
    free -g 2>/dev/null || true
    df -h /dev/shm 2>/dev/null || true
    echo; echo "## t2 host check"; t2_host_problems | sed 's/^/t2 blocker: /'
  } 2>&1 | tee "$d/env.txt"
  if [[ -n "$(git status --short 2>/dev/null)" ]]; then
    log "WARN: uncommitted changes. The gate/hadamard arms run the COMMITTED HEAD $SHA (ab_run pins refs), not the working tree."
  fi
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)
  if [[ -n "$drv" && "${drv%%.*}" -lt 580 ]]; then
    log "WARN: driver $drv < 580: the ANVIL2 README measured a uniform ~35% slowdown below 580 (val loss unaffected)" | tee -a "$d/env.txt"
  fi
  # The node report from the existing tool; keep any report node_setup.sh wrote earlier (it has the stack check).
  [[ -f node_report.txt ]] && cp node_report.txt "$d/node_report.setup.txt"
  bash 182/tools/node_setup.sh --mode report >/dev/null 2>&1 || true
  if [[ -f node_report.txt ]]; then
    mv node_report.txt "$d/node_report.txt"
    [[ -f "$d/node_report.setup.txt" ]] && cp "$d/node_report.setup.txt" node_report.txt
  fi
  xlog "$d/node_check.txt" tmo 300 torchrun --standalone --nproc_per_node="$NPROC" 182/tools/node_check.py || log "WARN: node_check.py failed (see $d/node_check.txt)"
  ensure_data 9
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
}

# ----------------------------------------------------------------------------------------- phase: tests
phase_tests() {
  local d="$SD/tests" rc1=0 rc2=0
  mkdir -p "$d"
  "$PY" -c "import pytest" 2>/dev/null || x "$PY" -m pip install -q pytest
  xlog "$d/pytest.txt" tmo "$TESTS_TIMEOUT_S" "$PY" -m pytest 182/hadamard-fp8/tests -q -rs || rc1=$?
  echo "$rc1" > "$d/rc_plain"
  HADAMARD_FP8=1 xlog "$d/pytest_hadamard.txt" tmo "$TESTS_TIMEOUT_S" "$PY" -m pytest 182/hadamard-fp8/tests -q -rs -s || rc2=$?
  echo "$rc2" > "$d/rc_hadamard"
  log "tests: plain rc=$rc1, HADAMARD_FP8=1 rc=$rc2"
  [[ "$rc1" == 0 && "$rc2" == 0 ]]
}

# ---------------------------------------------------------------------------------------- phase: sanity
phase_sanity() {
  local d="$SD/sanity"
  mkdir -p "$d"
  if [[ ! -f 182/hadamard-fp8/gpu_sanity.py && "$SANITY_CMD" == "$PY 182/hadamard-fp8/gpu_sanity.py" ]]; then
    skip "182/hadamard-fp8/gpu_sanity.py is not in this checkout (not merged yet)"; return 0
  fi
  local cmd=$SANITY_CMD
  [[ "$cmd" == "$PY 182/hadamard-fp8/gpu_sanity.py" ]] && cmd+=" --json $d/sanity.json"
  # shellcheck disable=SC2086  # a command line
  xlog "$d/sanity.txt" tmo "$SANITY_TIMEOUT_S" $cmd
}

# ----------------------------------------------------------------------------------------- A/B helpers
ab_check() {  # ab_check gate|ab DIR: 0 complete+usable, 3 slots pending, 4 complete but unusable, 2 no plan
  if [[ "$1" == gate ]]; then "$PY" 182/tools/summarize_session.py --check "$2" --check-gate
  else "$PY" 182/tools/summarize_session.py --check "$2"; fi
}
ab_plan_run() {  # ab_plan_run gate|ab DIR [plan args...]: plan once, run/resume, report; returns ab_check's code
  local kind=$1 out=$2 rc=0 k=1 why; shift 2
  if [[ -f "$out/plan.json" ]]; then
    why=$(ab_check "$kind" "$out") || rc=$?
    if [[ "${FORCE:-0}" == 1 || "$rc" == 4 ]]; then
      # ab_run never re-runs a finished slot, so a dead attempt (e.g. every run crashed) is set aside, kept
      # in the bundle, and the phase is planned afresh.
      while [[ -e "$out.attempt$k" ]]; do k=$((k + 1)); done
      log "re-planning: previous attempt moved to $out.attempt$k ($([[ "$rc" == 4 ]] && echo "unusable: $why" || echo FORCE=1))"
      mv "$out" "$out.attempt$k"
    else
      log "plan exists: $out/plan.json (resuming: $why)"
    fi
  fi
  if [[ ! -f "$out/plan.json" ]]; then
    "$PY" 182/tools/ab_run.py plan --out "$out" --repo "$REPO" --work-root "$SD/work" --cache "$CACHE" \
      --order blocked --order-seed 182 --nproc "$NPROC" --timeout-s "$TIMEOUT_S" --cmd "$RUN_CMD" "$@" || return 1
  fi
  x "$PY" 182/tools/ab_run.py run "$out" --no-report --stop-after-failures "$STOP_AFTER_FAILURES" || return 1
  [[ "$DRY_RUN" == 1 ]] && return 0
  "$PY" 182/tools/ab_run.py report "$out" --quiet || true
  rc=0; why=$(ab_check "$kind" "$out") || rc=$?
  [[ "$rc" == 0 ]] || log "A/B $(basename "$out") not usable yet: $why"
  return "$rc"
}

# ------------------------------------------------------------------------------------------ phase: gate
phase_gate() {
  local g="$SD/gate"
  mkdir -p "$SD"
  ensure_data 9
  local rc=0
  ab_plan_run gate "$g" --arm fp8=ref:HEAD --arm bf16=ref:HEAD --env bf16:FP8_OFF=1 --baseline fp8 \
    --n "$(count_csv "$GATE_SEEDS")" --seeds "$GATE_SEEDS" || rc=$?
  [[ -f "$g/plan.json" ]] || return 1
  "$PY" 182/tools/summarize_session.py --gate "$g" --write-decision "$g/decision.env" | tee "$g/decision.txt"
  [[ "$DRY_RUN" == 1 || "$rc" == 0 ]] || { log "FAIL: the gate has no decision yet (re-run to resume / re-plan)"; return 1; }
}

# -------------------------------------------------------------------------------------- phase: hadamard
phase_hadamard() {
  local h="$SD/hadamard" why="" v gemms block name args=()
  if [[ "${FORCE_HADAMARD:-0}" != 1 ]]; then
    if [[ ! -f "$SD/gate/decision.env" ]]; then
      why="the gate has not produced a decision (run the gate phase, or FORCE_HADAMARD=1)"
    else
      # shellcheck disable=SC1091
      source "$SD/gate/decision.env"
      [[ "${GATE_RUN_HADAMARD:-0}" == 1 ]] || why="gate decision ${GATE_DECISION:-?}: mean gap ${GATE_MEAN_MNAT:-?} mnat (CI upper ${GATE_HI_MNAT:-?}) is below what any Hadamard variant needs to pay for itself (HADAMARD_MIN_GAP_MNAT, default 2.0; FORCE_HADAMARD=1 overrides)"
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
  [[ "$DRY_RUN" == 1 || ! -f "$h/plan.json" ]] || "$PY" - "$h" <<'EOF'
import sys; sys.path.insert(0, "182/tools")
import summarize_session as s
ab = s.load_ab(sys.argv[1]); print("\n".join(s.ab_table("hadamard", ab)))
EOF
  return "$rc"
}

# ---------------------------------------------------------------------------------------------- phase: t2
phase_t2() {
  local d="$SD/t2" problems pid rc n rel
  mkdir -p "$d"
  problems=$(t2_host_problems)
  if [[ -n "$problems" ]]; then
    printf '%s\n' "$problems" > "$d/preflight.txt"
    if [[ "${T2_FORCE:-0}" != 1 ]]; then skip "host cannot run t2: $(printf '%s' "$problems" | paste -sd ';' -) (T2_FORCE=1 overrides)"; return 0; fi
    log "WARN: T2_FORCE=1, running despite: $problems"
  fi
  if [[ -f "$d/prefetch.pid" ]]; then
    pid=$(cat "$d/prefetch.pid")
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
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
  -h|--help) sed -n '2,40p' "$SELF"; exit 0;;
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
  PASS_ENV=()
  for v in QUICK HADAMARD_MIN_GAP_MNAT STOP_AFTER_FAILURES T2_TIMEOUT_S TESTS_TIMEOUT_S SANITY_TIMEOUT_S T2_PREFETCH_WAIT_S NPROC CACHE GATE_SEEDS HADAMARD_VARIANTS HADAMARD_SEEDS FORCE_HADAMARD TIMEOUT_S RUN_CMD SANITY_CMD SKIP_DATA \
           T2_FORCE T2_INSTALL_RUST FORCE DRY_RUN HF_HUB_OFFLINE COLD_RUN_MIN WARM_RUN_MIN; do
    [[ -n "${!v:-}" ]] && PASS_ENV+=(-e "$v=${!v}")
  done
  CMD=("${DOCKER[@]}" run --rm --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864
       -v "$REPO:$REPO" -v "$HFC:/root/.cache/huggingface" -w "$REPO" -e SESSION_DIR="$SD"
       -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' "${PASS_ENV[@]}"
       "$SESSION_DOCKER" bash 182/gpu_session.sh "${WANT[@]}")
  if [[ "$DRY_RUN" == 1 ]]; then log "+ $(q "${CMD[@]}")"; exit 0; fi
  mkdir -p "$(dirname "$SD")"
  exec "${CMD[@]}"
fi

mkdir -p "$SD/status"
# tee ignores ctrl-C, so the interrupt message below still reaches the terminal and session.log
exec > >(trap '' INT TERM HUP; exec tee -a "$SD/session.log") 2>&1
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
for p in "${SELECTED_PHASES[@]}"; do
  CUR_PHASE=$p
  run_phase "$p"
  [[ "$(status_of "$p")" == FAIL* ]] && FAILED+=("$p")
done
log ""
for p in "${SELECTED_PHASES[@]}"; do log "  $p: $(status_of "$p")"; done
[[ -f "$SD/bundle_path" && " ${SELECTED_PHASES[*]} " == *" bundle "* ]] && log "SEND THIS FILE BACK: $(cat "$SD/bundle_path")"
[[ ${#FAILED[@]} -eq 0 ]] || { log "failed phases: ${FAILED[*]}"; exit 1; }
