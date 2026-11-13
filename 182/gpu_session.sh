#!/usr/bin/env bash
# One GPU session on the 8xH100 box: every method in this repo, most promising first -> one file to send back.
# Run from anywhere inside the repo, inside tmux/screen. The script never commits or pushes.
#
#   bash 182/gpu_session.sh              # = all: env tests sanity base t1gpu profile nccl xab t1cpu t2cpu t2gpu gate
#                                        #   attno hadamard bundle
#   QUICK=1 bash 182/gpu_session.sh      # = env sanity base t1gpu profile bundle (2 runs each): something real in < 45 min
#   bash 182/gpu_session.sh t2cpu        # just those phases (always in the canonical order); `t2` = t2cpu
#   DRY_RUN=1 bash 182/gpu_session.sh    # print the plan, the estimates and every GPU/pip/download command; writes
#                                        # only to a temp dir (with SESSION_DIR: previews a resume from a copy)
#
# Phases (node-minute estimates are printed up front and before each phase; GPU-minutes = 8x):
#   env       git sha/status, node report (182/tools/node_setup.sh --mode report), driver >= 580 check, CPUs, RAM,
#             /dev/shm; node_check.py (GEMM/NCCL); 9 data shards if missing. The ROOT stack check FAILs env unless
#             torch 2.10.0+cu128 and triton 3.6.0 import, torchrun belongs to $PY's environment, and the pinned FA3
#             kernel loads with its sha256 (ALLOW_STACK=1 downgrades that to a warning). A failed env skips the
#             root phases (tests sanity base gate attno hadamard); the variant phases still run: each variant
#             checks its own stack. Writes env/stack.json and env/pip_freeze.txt.
#   tests     pytest 182/hadamard-fp8/tests, then again with HADAMARD_FP8=1
#   sanity    182/hadamard-fp8/gpu_sanity.py (skipped if absent)
#   base      the root record (Track 1 #92, ANVIL2, official 39.9 s) as it is, BASE_RUNS times, unseeded: this
#             node's own baseline and its spread, with the logs
#   t1gpu     variants/t1-gpu-stack   (Track 1, GPU only; reported 30.67 s, n=12)          T1GPU_RUNS=4
#   t1cpu     variants/t1-cpu-ngram   (Track 1, CPU retrieval; reported 9.653 s, n=16)     T1CPU_RUNS=4
#   t2cpu     variants/t2-cpu-ngram   (Track 2, CPU retrieval; reported 221 s, one run)    T2CPU_RUNS=1
#   t2gpu     variants/t2-gpu-port    (Track 2, GPU only; reported 14.34 min)              T2GPU_RUNS=1
#             Each variant phase drives variants/<name>/182/run_node.sh: `check` (a FAIL line -> the phase is
#             SKIPPED with the reasons, not a session failure; FORCE_CHECK=1 preps and runs anyway), `prep` (data, builds;
#             off the clock), then `run` N times, reading the final `RESULT ...` line of each run. A FAIL line
#             that names `prep` is treated as fixable by prep; any other FAIL line blocks prep too. The preps of
#             all selected variants run one after another in ONE low-priority background job from the start of
#             the session (PREP_BACKGROUND=0: inline instead); a variant phase waits for that whole job, so only
#             root phases overlap it (noted in the summary). Two consecutive failed runs stop a variant phase.
#   profile   where the step time goes (182/profile/README.md): one run per PROFILE_TREES entry (t1gpu root) with
#             torch.profiler over PROFILE_STEPS (300:20) of the timed loop, on all ranks, through 182/profile/run.sh
#             (no trainer edits: a torchrun shim starts the trainer through 182/profile/hook.py). Writes
#             profile/<tree>/analysis.txt (per-category ms/step, NCCL exposed vs overlapped, idle gaps, top 25
#             kernels, per-rank table) and trace_rank0.json.gz. Not timing runs. The root tree is skipped if env
#             failed; a variant tree whose `check` FAILs (after prep, if the FAIL names it) is skipped.
#   nccl      NVLink SHARP A/B: 182/tools/nvls_check.py first (nvidia-smi topo -m, fabric state, per-GPU multicast
#             support, three NCCL bandwidth probes); NVLS unavailable -> SKIP (FORCE_NCCL=1 runs anyway). Then per
#             NCCL_TREES entry (t1gpu root) a paired, interleaved A/B over NCCL_SEEDS of NCCL_ARMS ('|'-separated
#             NAME:K=V..., default base | nvls_off: NCCL_NVLS_ENABLE=0 | nvls_force: NCCL_NVLS_ENABLE=1 +
#             NCCL_ALGO per collective): Δval mnat and Δtime with 95% CIs, net seconds. NCCL 2.27.5 (torch 2.10
#             cu128) defaults NCCL_NVLS_ENABLE to 2 = NVLS on where supported, so `base` already has it.
#   xab       every 182/experiments.d/*.sh (format: 182/experiments.d/README.md): a paired A/B of its XAB_ARMS over
#             seeds 1..XAB_SEEDS on XAB_TREE, interleaved by seed: a variant tree through its run_node.sh run with
#             EXTRA_ENV (182/tools/xab_run.py), the root (".") through ab_run.py. Δval (mnat, 95% CI), Δtime, net_s
#             per arm. XAB_ONLY / XAB_SKIP (comma lists of XAB_NAMEs) pick experiments; an invalid file is reported
#             and the others still run. Resumable per experiment.
#   gate      record vs FP8_OFF=1, GATE_SEEDS paired, interleaved (ab_run.py): the paired gap val(fp8) - val(bf16)
#             mean +- SE and its 95% CI vs 0.5 mnat -> STOP / GAP / INCONCLUSIVE, only once every slot finished
#   attno     record vs FP8_ATTN_O=1, ATTNO_SEEDS paired, interleaved: val delta in millinats with its 95% CI,
#             time delta, and net seconds at 164 ms per millinat (120-250 band). Runs whatever the gate says;
#             skipped with the reason while track_1_short/config.py at HEAD does not know FP8_ATTN_O.
#   hadamard  LAST (one measurement on this node had it +3.17 s slower; it needs a 3-4 mnat fp8 gap to pay): only
#             if the gate PASSed with decision GAP/INCONCLUSIVE and mean gap >= HADAMARD_MIN_GAP_MNAT (2.0), and
#             the HADAMARD_FP8 tests / sanity did not fail (FORCE_HADAMARD=1 bypasses all of that): record vs
#             HADAMARD_FP8=1 per HADAMARD_VARIANTS (GEMMS@BLOCK), HADAMARD_SEEDS
#   bundle    summary.txt + tar.gz of every log -> 182/results/session-<date>-<sha>.tar.gz (prints the path)
#
# Resumable: re-run the same command after a crash or ctrl-C. A phase whose status is PASS is skipped (FORCE=1
# redoes it, keeping the old attempt); the A/B phases (base gate attno hadamard) resume inside ab_run.py at the
# first unfinished run, a variant phase at its first run without a result (failed runs are retried). Every phase
# is logged to 182/results/session-<sha>/ (gitignored). A failed or skipped phase never stops the phases after it,
# and `all` always ends with the bundle. An A/B phase stops after STOP_AFTER_FAILURES consecutive failed runs, or
# once one arm's first ARM_FAILURES runs all failed (slots stay pending; re-run resumes them); a hadamard variant
# that fails that way is abandoned and the others continue. An A/B that finished unusable from scattered failures
# is FAIL and the next re-run moves it to <phase>.attemptN and re-plans it; one where an arm failed EVERY run is
# not re-planned (fix the cause, then FORCE=1). Every GPU, test and download command runs under a wall-clock limit
# that kills its whole process group. After ctrl-C, `bash 182/gpu_session.sh bundle` packs what exists.
#
# Knobs (env): PY (python), NPROC (8), CACHE (warm|cold; warm = one compile cache per arm), BASE_RUNS (2),
# T1GPU_RUNS / T1CPU_RUNS (4; 2 under QUICK), T2CPU_RUNS / T2GPU_RUNS (1), <PHASE>_SEEDS (comma list passed as
# SEED to run i; default unseeded; t2-gpu-port ignores SEED), <PHASE>_RUN_ENV (extra K=V for that variant's run;
# default T2GPU_RUN_ENV=CANON_MASK=1, e.g. T1CPU_RUN_ENV=CONFIG=380 in its own SESSION_DIR), <PHASE>_TIMEOUT_S (per variant run: 1500 Track 1, 5400 Track 2),
# VARIANT_STOP_AFTER (2), CHECK_TIMEOUT_S (300), PREP_TIMEOUT_S (14400 per variant), PREP_WAIT_S (7200),
# PREP_BACKGROUND (1), FORCE_CHECK, GATE_SEEDS (1,2,3,4), ATTNO_SEEDS (1,2,3,4), HADAMARD_VARIANTS ("all@128
# dw1,dw2@32"), HADAMARD_SEEDS (1,2), HADAMARD_MIN_GAP_MNAT (2.0), FORCE_HADAMARD, TIMEOUT_S (1500 per A/B run),
# STOP_AFTER_FAILURES (3), ARM_FAILURES (2), TESTS_TIMEOUT_S (1800 per pytest), SANITY_TIMEOUT_S (1200),
# DATA_TIMEOUT_S (3600, the 9-shard download), COLD_RUN_MIN (9) / WARM_RUN_MIN (4) / T2CPU_RUN_MIN (20) /
# T2GPU_RUN_MIN (25) (the estimates), QUICK, RUN_CMD (ab_run --cmd template), SANITY_CMD, SKIP_DATA, ALLOW_STACK,
# SESSION_DIR, FORCE, DRY_RUN, SESSION_DOCKER=<image> (re-run this script inside that image, e.g. modded-nanogpt
# from node_setup.sh --mode docker; SESSION_DOCKER_GPUS: --gpus value, default all, none = omit). In docker mode
# every knob above that YOU set is forwarded into the container, except PY (the image's python is used) and
# SESSION_DIR (mapped explicitly); defaults are computed inside the container. HF_HUB_OFFLINE and HF_TOKEN are
# forwarded too, and INSTALL_RUST (lets a CPU-retrieval prep install rustup). The image must carry what a variant needs (its `check` says what is missing).
# profile/nccl/xab knobs: PROFILE_TREES ("t1gpu root"; a variant phase name, root, or a repo-relative dir), PROFILE_STEPS
# (300:20), PROFILE_RUN_MIN (5, the estimate per tree), PROFILE_TIMEOUT_S (1800), PROFILE_BUNDLE_MB (150: a bigger
# trace_rank0.json.gz stays out of the bundle), NCCL_TREES ("t1gpu root"), NCCL_ARMS,
# NCCL_SEEDS (1,2,3,4), NCCL_PROBE_TIMEOUT_S (300 per probe), FORCE_NCCL, XAB_ONLY, XAB_SKIP, XAB_TIMEOUT_S (1500 per
# run), SKIP_PHASES (phase names dropped from the selection, e.g. "nccl xab").
set -euo pipefail

PHASES=(env tests sanity base t1gpu profile nccl xab t1cpu t2cpu t2gpu gate attno hadamard bundle)
ROOT_PHASES=" tests sanity base gate attno hadamard "   # need the record's stack (env checks it)
VARIANT_PHASES=" t1gpu t1cpu t2cpu t2gpu "              # check their own stack (run_node.sh check)
QUICK_PHASES=(env sanity base t1gpu profile bundle)     # profile is cheap (one run per tree); nccl and xab are not
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$REPO"

# The knobs the user set, captured before any default is applied (docker mode forwards exactly these).
FORWARD_VARS=(QUICK NPROC CACHE BASE_RUNS T1GPU_RUNS T1CPU_RUNS T2CPU_RUNS T2GPU_RUNS T1GPU_SEEDS T1CPU_SEEDS T2CPU_SEEDS
  T2GPU_SEEDS T1GPU_RUN_ENV T1CPU_RUN_ENV T2CPU_RUN_ENV T2GPU_RUN_ENV T1GPU_TIMEOUT_S T1CPU_TIMEOUT_S T2CPU_TIMEOUT_S T2GPU_TIMEOUT_S VARIANT_STOP_AFTER CHECK_TIMEOUT_S
  PREP_TIMEOUT_S PREP_WAIT_S PREP_BACKGROUND FORCE_CHECK GATE_SEEDS ATTNO_SEEDS HADAMARD_VARIANTS HADAMARD_SEEDS
  HADAMARD_MIN_GAP_MNAT FORCE_HADAMARD TIMEOUT_S STOP_AFTER_FAILURES ARM_FAILURES TESTS_TIMEOUT_S SANITY_TIMEOUT_S
  DATA_TIMEOUT_S COLD_RUN_MIN WARM_RUN_MIN T2CPU_RUN_MIN T2GPU_RUN_MIN RUN_CMD SANITY_CMD SKIP_DATA ALLOW_STACK FORCE
  DRY_RUN HF_HUB_OFFLINE HF_TOKEN INSTALL_RUST PROFILE_TREES PROFILE_STEPS PROFILE_RUN_MIN PROFILE_TIMEOUT_S PROFILE_BUNDLE_MB
  NCCL_TREES NCCL_ARMS NCCL_SEEDS NCCL_PROBE_TIMEOUT_S FORCE_NCCL XAB_ONLY XAB_SKIP XAB_TIMEOUT_S SKIP_PHASES)
USER_ENV=()
for v in "${FORWARD_VARS[@]}"; do [[ -n "${!v+x}" ]] && USER_ENV+=(-e "$v=${!v}"); done
SANITY_DEFAULT=0; [[ -z "${SANITY_CMD+x}" ]] && SANITY_DEFAULT=1

DRY_RUN=${DRY_RUN:-0}
PY=${PY:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
NPROC=${NPROC:-8}
CACHE=${CACHE:-warm}
GATE_SEEDS=${GATE_SEEDS:-1,2,3,4}
ATTNO_SEEDS=${ATTNO_SEEDS:-1,2,3,4}
BASE_RUNS=${BASE_RUNS:-2}
# QUICK halves the Track 1 variant runs (2 is the least that shows a spread) unless the caller set them.
if [[ "${QUICK:-0}" == 1 ]]; then T1GPU_RUNS=${T1GPU_RUNS:-2}; T1CPU_RUNS=${T1CPU_RUNS:-2}; fi
T1GPU_RUNS=${T1GPU_RUNS:-4}; T1CPU_RUNS=${T1CPU_RUNS:-4}; T2CPU_RUNS=${T2CPU_RUNS:-1}; T2GPU_RUNS=${T2GPU_RUNS:-1}
T1GPU_TIMEOUT_S=${T1GPU_TIMEOUT_S:-1500}; T1CPU_TIMEOUT_S=${T1CPU_TIMEOUT_S:-1500}
T2CPU_TIMEOUT_S=${T2CPU_TIMEOUT_S:-5400}; T2GPU_TIMEOUT_S=${T2GPU_TIMEOUT_S:-5400}
VARIANT_STOP_AFTER=${VARIANT_STOP_AFTER:-2}
# Extra environment for a variant's `run` (space-separated K=V). t2gpu: its README's recommended first run is the
# canonical mask (a CANON line above RESULT); T2GPU_RUN_ENV= (empty) runs plain PR #261.
T2GPU_RUN_ENV=${T2GPU_RUN_ENV-CANON_MASK=1}
CHECK_TIMEOUT_S=${CHECK_TIMEOUT_S:-300}; PREP_TIMEOUT_S=${PREP_TIMEOUT_S:-14400}; PREP_WAIT_S=${PREP_WAIT_S:-7200}
PREP_BACKGROUND=${PREP_BACKGROUND:-1}
HADAMARD_VARIANTS=${HADAMARD_VARIANTS:-"all@128 dw1,dw2@32"}
HADAMARD_SEEDS=${HADAMARD_SEEDS:-1,2}
TIMEOUT_S=${TIMEOUT_S:-1500}      # a cold run is ~9 min; a hung NCCL run should not cost 40
STOP_AFTER_FAILURES=${STOP_AFTER_FAILURES:-3}
ARM_FAILURES=${ARM_FAILURES:-2}   # one arm crashing every run (the arms interleave, so the consecutive count misses it)
TESTS_TIMEOUT_S=${TESTS_TIMEOUT_S:-1800}; SANITY_TIMEOUT_S=${SANITY_TIMEOUT_S:-1200}
DATA_TIMEOUT_S=${DATA_TIMEOUT_S:-3600}
RUN_CMD=${RUN_CMD:-"torchrun --standalone --nproc_per_node={nproc} {script}"}
SANITY_CMD=${SANITY_CMD:-"$PY 182/hadamard-fp8/gpu_sanity.py"}
COLD_RUN_MIN=${COLD_RUN_MIN:-9}   # one Track 1 run with a cold compile cache (~7 min compile + train + val)
WARM_RUN_MIN=${WARM_RUN_MIN:-4}
# Guesses until a run exists: t2cpu ~7 min compile/capture + ~4 min train + the table build (its README);
# t2gpu 14.34 min reported + compile + val.
T2CPU_RUN_MIN=${T2CPU_RUN_MIN:-20}; T2GPU_RUN_MIN=${T2GPU_RUN_MIN:-25}
PROFILE_TREES=${PROFILE_TREES-t1gpu root}; PROFILE_STEPS=${PROFILE_STEPS:-300:20}
PROFILE_RUN_MIN=${PROFILE_RUN_MIN:-5}; PROFILE_TIMEOUT_S=${PROFILE_TIMEOUT_S:-1800}; PROFILE_BUNDLE_MB=${PROFILE_BUNDLE_MB:-150}
NCCL_TREES=${NCCL_TREES-t1gpu root}; NCCL_SEEDS=${NCCL_SEEDS:-1,2,3,4}; NCCL_PROBE_TIMEOUT_S=${NCCL_PROBE_TIMEOUT_S:-300}
# base: NCCL as it is (2.27.5 defaults NCCL_NVLS_ENABLE=2: NVLS wherever supported). nvls_off: what NVLS buys now.
# nvls_force: NVLS forced for the three collectives that support it (NCCL >= 2.24 per-collective syntax).
NCCL_ARMS=${NCCL_ARMS:-"base:|nvls_off:NCCL_NVLS_ENABLE=0|nvls_force:NCCL_NVLS_ENABLE=1 NCCL_ALGO='allreduce:nvls;reducescatter:nvls;allgather:nvls'"}
XAB_TIMEOUT_S=${XAB_TIMEOUT_S:-1500}
export GIT_OPTIONAL_LOCKS=0      # `git status` must not rewrite the index (DRY_RUN writes nothing in the repo)
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

# -------------------------------------------------------------------------------------------- variants
variant_dir() {
  case "$1" in
    t1gpu) echo variants/t1-gpu-stack;; t1cpu) echo variants/t1-cpu-ngram;;
    t2cpu) echo variants/t2-cpu-ngram;; t2gpu) echo variants/t2-gpu-port;;
    *) return 1;;
  esac
}
is_variant() { [[ "$VARIANT_PHASES" == *" $1 "* ]]; }
knob() {  # knob PHASE SUFFIX [DEFAULT]: the value of e.g. T1GPU_RUNS
  local v; v="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')_$2"; printf '%s' "${!v:-${3:-}}"
}
chain_alive() {  # the background prep job of this session is still working
  [[ -f "$SD/prep_chain.pid" && ! -f "$SD/prep_chain.rc" ]] && kill -0 "$(cat "$SD/prep_chain.pid")" 2>/dev/null
}

# ------------------------------------------------------------------------------------------- estimates
ab_estimate() {  # arms n -> node-minutes
  local arms=$1 n=$2
  if [[ "$CACHE" == cold ]]; then echo $((arms * n * COLD_RUN_MIN)); else echo $((arms * COLD_RUN_MIN + arms * (n - 1) * WARM_RUN_MIN)); fi
}
# shellcheck disable=SC2086  # split on commas on purpose
count_csv() { local IFS=,; set -- $1; echo $#; }
n_variants() { local v n=0; for v in $HADAMARD_VARIANTS; do n=$((n + 1)); done; echo "$n"; }
estimate() {  # node-minutes; the A/B phases after `base` reuse its warm compile cache for the fp8 arm
  local n
  case "$1" in
    env) echo 2;;
    tests) echo 8;;
    sanity) echo 3;;
    base) ab_estimate 1 "$BASE_RUNS";;
    t1gpu|t1cpu) n=$(knob "$1" RUNS); if (( n < 1 )); then echo 0; else echo $((COLD_RUN_MIN + (n - 1) * WARM_RUN_MIN)); fi;;  # compile once, then the on-disk cache
    t2cpu) echo $(( $(knob t2cpu RUNS) * T2CPU_RUN_MIN ));;
    t2gpu) echo $(( $(knob t2gpu RUNS) * T2GPU_RUN_MIN ));;
    gate) ab_estimate 2 "$(count_csv "$GATE_SEEDS")";;
    attno) ab_estimate 2 "$(count_csv "$ATTNO_SEEDS")";;
    hadamard) echo $(( $(ab_estimate $(( $(n_variants) + 1 )) "$(count_csv "$HADAMARD_SEEDS")") - COLD_RUN_MIN + WARM_RUN_MIN ));;  # baseline reuses the fp8 cache
    profile) n=0; for t in $PROFILE_TREES; do n=$((n + 1)); done; echo $((n * PROFILE_RUN_MIN));;
    # warm runs: the variant's on-disk cache from t1gpu, the root arms share base's cache (NCCL env compiles nothing)
    nccl) n=0; for t in $NCCL_TREES; do n=$((n + 1)); done; echo $((3 + n * $(nccl_arm_count) * $(count_csv "$NCCL_SEEDS") * WARM_RUN_MIN));;
    xab) xab_estimate;;
    *) echo 0;;
  esac
}
# shellcheck disable=SC2086  # split on '|' on purpose
nccl_arm_count() { local IFS='|'; set -- $NCCL_ARMS; echo $#; }
xab_selected() {  # xab_selected NAME: XAB_ONLY / XAB_SKIP allow it
  [[ -z "${XAB_ONLY:-}" || ",$XAB_ONLY," == *",$1,"* ]] && [[ ",${XAB_SKIP:-}," != *",$1,"* ]]
}
xab_files() { local f; for f in "$REPO"/182/experiments.d/*.sh; do [[ -f "$f" ]] && printf '%s\n' "$f"; done; return 0; }
xab_estimate() {  # sum of XAB_EST_MIN over the valid, selected experiments
  local f t=0 e
  while IFS= read -r f; do
    e=$( XS_OK=0; eval "$("$PY" "$REPO/182/tools/xab_run.py" spec "$f" --repo "$REPO" 2>/dev/null)"
         if [[ "$XS_OK" == 1 ]] && xab_selected "$XS_NAME"; then echo "$XS_EST"; else echo 0; fi )
    t=$((t + ${e:-0}))
  done < <(xab_files)
  echo "$t"
}
tree_of() {  # tree_of NAME: a variant phase name, root, or a repo-relative directory -> the directory
  case "$1" in root|.) echo .;; *) if is_variant "$1"; then variant_dir "$1"; else echo "$1"; fi;; esac
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
  } 2>&1 | tee "$d/env.txt"
  "$PY" -m pip freeze > "$d/pip_freeze.txt" 2>/dev/null || "$PY" -c 'import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()): print(d.metadata["Name"] + "==" + d.version)' > "$d/pip_freeze.txt" 2>&1 || true
  if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    log "WARN: modified tracked files. The base/gate/attno/hadamard arms run the COMMITTED HEAD $SHA (ab_run pins refs), not the working tree."
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
  [[ -n "$drv" || "$DRY_RUN" == 1 ]] || { log "FAIL: no GPU driver visible"; return 1; }
  if grep -q '^STACK PROBLEM: ' "$d/env.txt"; then
    if [[ "$DRY_RUN" == 1 ]]; then log "(dry run) env would FAIL on the stack check above (ALLOW_STACK=1 overrides)"
    elif [[ "${ALLOW_STACK:-0}" == 1 ]]; then log "WARN: ALLOW_STACK=1: running on a stack the record was not certified on (see STACK PROBLEM above)"
    else
      log "FAIL: the stack is not the record's (STACK PROBLEM lines above; $d/stack.json). The root phases are skipped:"
      log "      results on another torch/triton/FA3 would not say anything about the record. Fix it (bash 182/tools/node_setup.sh"
      log "      --mode venv && source .venv-run/bin/activate), then re-run; or ALLOW_STACK=1 to run anyway."
      log "      The variant phases still run: each variant checks its own stack."
      return 1
    fi
  fi
}

# ----------------------------------------------------------------------------------------- phase: tests
phase_tests() {
  local d="$SD/tests" rc1=0 rc2=0
  mkdir -p "$d"
  if ! "$PY" -c "import pytest" 2>/dev/null && ! x tmo 600 "$PY" -m pip install -q pytest; then
    log "FAIL: pytest is not importable with $PY and \`$PY -m pip install pytest\` failed (wrong interpreter? activate the venv)" \
      | tee "$d/pytest.txt" "$d/pytest_hadamard.txt"
    echo 1 > "$d/rc_plain"; echo 1 > "$d/rc_hadamard"; return 1
  fi
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

# ------------------------------------------------------------------------------------ phases: variants
# variant_prep PHASE: `check`, then `prep` unless a FAIL line that prep cannot fix (one not naming `prep`) says
# the host cannot run this variant anyway. Writes <phase>/check_pre.txt, prep.txt and prep.rc (0 | blocked | rc).
variant_prep() {
  local p=$1 d="$SD/$1" s rc=0
  s="$(variant_dir "$p")/182/run_node.sh"
  mkdir -p "$d"
  [[ -f "$s" ]] || return 0
  tmo "$CHECK_TIMEOUT_S" bash "$s" check > "$d/check_pre.txt" 2>&1 || true
  if [[ "${FORCE_CHECK:-0}" != 1 ]] && grep -E '^FAIL ' "$d/check_pre.txt" | grep -qv 'prep'; then echo blocked > "$d/prep.rc"; return 0; fi
  tmo "$PREP_TIMEOUT_S" bash "$s" prep > "$d/prep.txt" 2>&1 || rc=$?
  echo "$rc" > "$d/prep.rc"
}

prep_chain() {  # the background job: one variant after another, so two downloads never write the same shards
  local p
  for p in "$@"; do variant_prep "$p" || true; done
  echo 0 > "$SD/prep_chain.rc"
}

variant_wait_prep() {  # wait for the WHOLE background job: a download or a cargo build must not overlap a timed variant run
  local p=$1 d="$SD/$1" waited=0
  while chain_alive; do
    if (( waited >= PREP_WAIT_S )); then
      if [[ -f "$d/prep.rc" ]]; then
        log "WARN: the background prep job is still running after ${waited}s; $p's own prep finished, so it runs now, overlapping that job"
        echo "$p started while the background prep job was still running (gave up waiting after ${waited}s)" >> "$SD/prep_overlap.txt"
        return 0
      fi
      log "FAIL: $p's prep has not finished after ${waited}s (PREP_WAIT_S); $d/prep.txt: $(tail -n2 "$d/prep.txt" 2>/dev/null || true)"
      return 1
    fi
    (( waited % 300 == 0 )) && log "$p: waiting for the background prep job (pid $(cat "$SD/prep_chain.pid"); logs $SD/<phase>/prep.txt)"
    sleep "${PREP_POLL_S:-15}"; waited=$((waited + ${PREP_POLL_S:-15}))
  done
  if [[ ! -f "$d/prep.rc" ]]; then
    log "$p: check + prep (inline; log $d/prep.txt)"
    variant_prep "$p"
  fi
}

phase_variant() {
  local p=$PHASE d="$SD/$PHASE" vdir s name n i tag rc crc=0 reasons prc seeds seed k fails=0 consec=0 t0 wall logp renv
  vdir=$(variant_dir "$p"); s="$vdir/182/run_node.sh"; name=$(basename "$vdir")
  n=$(knob "$p" RUNS); seeds=$(knob "$p" SEEDS); renv=$(knob "$p" RUN_ENV)
  if [[ ! -f "$s" ]]; then skip "$s is not in this checkout (this variant has no node script yet)"; return 0; fi
  mkdir -p "$d"
  if [[ "$DRY_RUN" == 1 ]]; then
    log "+ bash $s check   (CPU only; run for real, so the dry run shows what this host lacks)"
    tmo "$CHECK_TIMEOUT_S" bash "$s" check 2>&1 | sed 's/^/    /' || log "    (check exited non-zero: on a real run a FAIL line here skips $p)"
    log "+ bash $s prep    (data, builds; off the clock)"
    for ((i = 1; i <= n; i++)); do
      log "+ NPROC=$NPROC LOG_DIR=$d/logs${renv:+ $renv} bash $s run   (run $i/$n)"
      # shellcheck disable=SC2086  # renv is a list of K=V words
      env NPROC="$NPROC" LOG_DIR="$d/logs" $renv DRY_RUN=1 bash "$s" run 2>&1 | sed 's/^/    /' || true
    done
    return 0
  fi
  if [[ "${FORCE:-0}" == 1 && -d "$d/runs" ]]; then
    k=1; while [[ -e "$d/runs.attempt$k" ]]; do k=$((k + 1)); done
    log "FORCE=1: previous runs moved to $d/runs.attempt$k"; mv "$d/runs" "$d/runs.attempt$k"
  fi
  mkdir -p "$d/runs" "$d/logs"
  echo "$n" > "$d/planned"
  variant_wait_prep "$p" || return 1
  prc=$(cat "$d/prep.rc" 2>/dev/null || echo 0)
  if [[ "$prc" != 0 && "$prc" != blocked ]]; then
    log "FAIL: \`bash $s prep\` exited $prc (124 = PREP_TIMEOUT_S=${PREP_TIMEOUT_S}s); re-run to resume it. Tail of $d/prep.txt:"
    tail -n5 "$d/prep.txt" 2>/dev/null | sed 's/^/    /' || true
    return 1
  fi
  # the authoritative check, after prep
  tmo "$CHECK_TIMEOUT_S" bash "$s" check > "$d/check.txt" 2>&1 || crc=$?
  cat "$d/check.txt"
  if [[ "$crc" != 0 ]]; then
    reasons=$(grep -E '^FAIL ' "$d/check.txt" | paste -sd ';' - || true)
    if [[ -z "$reasons" ]]; then
      log "FAIL: \`bash $s check\` exited $crc without a FAIL line (a broken check, not a host verdict); see $d/check.txt"
      return 1
    fi
    if [[ "${FORCE_CHECK:-0}" != 1 ]]; then skip "$name check: $reasons (FORCE_CHECK=1 runs anyway)"; return 0; fi
    log "WARN: FORCE_CHECK=1, running $name despite: $reasons"
  fi
  for ((i = 1; i <= n; i++)); do
    tag=$(printf '%02d' "$i")
    if [[ -f "$d/runs/$tag.result" ]]; then
      if "$PY" 182/tools/summarize_session.py --result-ok "$d/runs/$tag.result"; then log "run $i/$n: already done"; consec=0; continue; fi
      k=1; while [[ -e "$d/runs/$tag.fail$k.result" ]]; do k=$((k + 1)); done   # keep the failed attempt, retry the run
      mv "$d/runs/$tag.result" "$d/runs/$tag.fail$k.result"
      [[ -f "$d/runs/$tag.out" ]] && mv "$d/runs/$tag.out" "$d/runs/$tag.fail$k.out"
    fi
    seed=""; [[ -n "$seeds" ]] && seed=$(printf '%s' "$seeds" | cut -d, -f"$i")
    log "-- $p run $i/$n${seed:+ (SEED=$seed)}${renv:+ ($renv)} -- $(date '+%T')"
    chain_alive && echo "$p run $i ran while the background prep job was still running" >> "$SD/prep_overlap.txt"
    rc=0; t0=$(date +%s)
    # shellcheck disable=SC2086  # renv is a list of K=V words
    env NPROC="$NPROC" LOG_DIR="$d/logs" ${seed:+SEED="$seed"} $renv "$PY" -c "$TMO_PY" "$(knob "$p" TIMEOUT_S)" bash "$s" run 2>&1 | tee "$d/runs/$tag.out" || rc=$?
    wall=$(( $(date +%s) - t0 ))
    if "$PY" 182/tools/summarize_session.py --parse-run "$d/runs/$tag.out" --rc "$rc" --wall-s "$wall" --seed "$seed" --run-env "$renv" --variant-name "$name" > "$d/runs/$tag.result"; then
      consec=0
      logp=$("$PY" 182/tools/summarize_session.py --result-field log "$d/runs/$tag.result" || true)
      [[ -n "$logp" && "$logp" != /* ]] && logp="$vdir/$logp"
      if [[ -n "$logp" && -f "$logp" && "$(cd "$(dirname "$logp")" && pwd -P)" != "$(cd "$d/logs" && pwd -P)" ]]; then cp "$logp" "$d/logs/"; fi
    else
      fails=$((fails + 1)); consec=$((consec + 1))
      log "run $i/$n FAILED (rc $rc): $("$PY" 182/tools/summarize_session.py --result-field reason "$d/runs/$tag.result" || true)"
      if (( consec >= VARIANT_STOP_AFTER && i < n )); then
        log "stopping $p: $consec consecutive failed runs; $((n - i)) run(s) not attempted (fix the cause, then re-run the same command)"
        fails=$((fails + n - i)); break
      fi
    fi
  done
  "$PY" 182/tools/summarize_session.py --variant "$SD" "$p" || true
  [[ "$fails" == 0 ]]
}

# ------------------------------------------------------------------------------------------ phase: base
phase_base() {
  local b="$SD/base" rc=0
  mkdir -p "$SD"
  ensure_data 9 || return 1
  # One arm, named fp8 like the baseline arm of gate/attno/hadamard: they reuse its worktree and warm compile cache.
  ab_plan_run ab "$b" --arm fp8=ref:HEAD --baseline fp8 --n "$BASE_RUNS" --seeds none || rc=$?
  [[ "$DRY_RUN" == 1 || ! -f "$b/plan.json" ]] || "$PY" 182/tools/summarize_session.py --base "$SD" || true
  return "$rc"
}

# ----------------------------------------------------------------------------------------- phase: attno
phase_attno() {
  local a="$SD/attno" rc=0
  # The arms run the committed HEAD, so that is where the flag must exist. A grep, not an import: no torch needed.
  if ! git grep -q FP8_ATTN_O HEAD -- track_1_short/config.py 2>/dev/null; then
    skip "track_1_short/config.py at HEAD does not read FP8_ATTN_O (the flag has not been merged into this checkout)"; return 0
  fi
  mkdir -p "$SD"
  ensure_data 9 || return 1
  ab_plan_run ab "$a" --arm fp8=ref:HEAD --arm attn_o=ref:HEAD --env attn_o:FP8_ATTN_O=1 --baseline fp8 \
    --n "$(count_csv "$ATTNO_SEEDS")" --seeds "$ATTNO_SEEDS" || rc=$?
  [[ "$DRY_RUN" == 1 || ! -f "$a/plan.json" ]] || "$PY" 182/tools/summarize_session.py --attno "$SD" || true
  return "$rc"
}

# ------------------------------------------------------------------- helpers: variant readiness, one paired A/B
# variant_ready TREE LOGDIR: 0 when TREE's run_node.sh `check` passes (running `prep` once if every FAIL names it);
# otherwise 1 with READY_WHY. Waits for the session's background prep job first (two preps must not write the same
# shards). FORCE_CHECK=1 ignores a failed check, as in the variant phases.
variant_ready() {
  local tree=$1 ld=$2 s="$1/182/run_node.sh" waited=0 rc
  READY_WHY=""
  mkdir -p "$ld"
  [[ -f "$s" ]] || { READY_WHY="$s is not in this checkout"; return 1; }
  if [[ "$DRY_RUN" == 1 ]]; then log "+ bash $s check   (then prep, if a FAIL line names prep)"; return 0; fi
  while chain_alive && (( waited < PREP_WAIT_S )); do sleep "${PREP_POLL_S:-15}"; waited=$((waited + ${PREP_POLL_S:-15})); done
  tmo "$CHECK_TIMEOUT_S" bash "$s" check > "$ld/check.txt" 2>&1 && return 0
  if [[ "${FORCE_CHECK:-0}" == 1 ]]; then log "WARN: FORCE_CHECK=1: running $tree despite its check ($ld/check.txt)"; return 0; fi
  if ! grep -q '^FAIL ' "$ld/check.txt"; then READY_WHY="\`bash $s check\` failed without a FAIL line ($ld/check.txt)"; return 1; fi
  if ! grep -E '^FAIL ' "$ld/check.txt" | grep -qv prep; then
    log "$tree: its check names prep; running it (log $ld/prep.txt)"
    rc=0; tmo "$PREP_TIMEOUT_S" bash "$s" prep > "$ld/prep.txt" 2>&1 || rc=$?
    [[ "$rc" == 0 ]] || { READY_WHY="\`bash $s prep\` exited $rc ($ld/prep.txt)"; return 1; }
    tmo "$CHECK_TIMEOUT_S" bash "$s" check > "$ld/check.txt" 2>&1 && return 0
  fi
  READY_WHY="check: $(grep -E '^FAIL ' "$ld/check.txt" | paste -sd ';' - || true)"
  return 1
}

# xab_tree LABEL TREE OUT SEEDS ARM...: one paired, interleaved A/B (ARM = NAME:K=V K=V, the first the baseline).
# A variant tree runs through 182/tools/xab_run.py (its run_node.sh run, the arm's env in EXTRA_ENV); the root (".")
# through ab_run.py (ab_plan_run; extra plan args in XAB_AB_EXTRA). Re-plans after FORCE=1 or an unusable finish
# (the old attempt is kept as OUT.attemptN). Writes OUT/report.txt and OUT/summary.json; returns xab_run.py check's
# code (0 complete and usable, 3 pending, 4 unusable, 5 an arm failed every run).
xab_tree() {
  local label=$1 tree=$2 out=$3 seeds=$4 rc=0 a nm kv base="" args=() k why; shift 4
  if [[ "$tree" == . ]]; then
    for a in "$@"; do
      nm=${a%%:*}; [[ -z "$base" ]] && base=$nm
      args+=(--arm "$nm=ref:HEAD")
      while IFS= read -r kv; do [[ -n "$kv" ]] && args+=(--env "$nm:$kv"); done \
        < <("$PY" -c 'import shlex, sys; print("\n".join(shlex.split(sys.argv[1])))' "${a#*:}")
    done
    ab_plan_run ab "$out" "${args[@]}" --baseline "$base" --n "$(count_csv "$seeds")" --seeds "$seeds" \
      ${XAB_AB_EXTRA[@]+"${XAB_AB_EXTRA[@]}"} || rc=$?
  else
    if [[ -f "$out/plan.json" ]]; then
      rc=0; why=$("$PY" 182/tools/xab_run.py check "$out") || rc=$?
      if [[ "${FORCE:-0}" == 1 || "$rc" == 4 ]]; then
        k=1; while [[ -e "$out.attempt$k" ]]; do k=$((k + 1)); done
        log "re-planning: previous attempt moved to $out.attempt$k ($([[ "$rc" == 4 ]] && echo "unusable: $why" || echo FORCE=1))"
        x mv "$out" "$out.attempt$k"
      elif [[ "$rc" == 5 ]]; then
        log "NOT re-running $(basename "$out"): $why. Fix the cause, then FORCE=1."
      else
        log "plan exists: $out/plan.json (resuming: $why)"
      fi
    fi
    args=()
    for a in "$@"; do args+=(--arm "$a"); done
    rc=0
    if [[ "$DRY_RUN" == 1 ]]; then
      "$PY" 182/tools/xab_run.py run --out "$out" --tree "$tree" --name "$label" --seeds "$seeds" --nproc "$NPROC" "${args[@]}" --dry-run
      return 0
    fi
    "$PY" 182/tools/xab_run.py run --out "$out" --tree "$tree" --name "$label" --seeds "$seeds" --nproc "$NPROC" \
      --timeout-s "$XAB_TIMEOUT_S" --stop-after-failures "$STOP_AFTER_FAILURES" --arm-failures "$ARM_FAILURES" "${args[@]}" || rc=$?
  fi
  [[ "$DRY_RUN" == 1 ]] && return 0
  [[ -f "$out/plan.json" ]] || return 1
  "$PY" 182/tools/xab_run.py report "$out" --label "$label" --json "$out/summary.json" > "$out/report.txt" 2>&1 || true
  cat "$out/report.txt"
  rc=0; why=$("$PY" 182/tools/xab_run.py check "$out") || rc=$?
  [[ "$rc" == 0 ]] || log "A/B $label not usable yet: $why"
  return "$rc"
}

root_blocked() {  # the root tree needs env's stack check (a dry run never blocks)
  [[ "$DRY_RUN" != 1 && "$(status_of env)" == FAIL* && "${ALLOW_STACK:-0}" != 1 ]]
}

# ----------------------------------------------------------------------------------------- phase: profile
phase_profile() {
  local d="$SD/profile" t tree out rc fails=0 done_=0 skipped=() cache k
  mkdir -p "$d"
  [[ -n "${PROFILE_TREES// }" ]] || { skip "PROFILE_TREES is empty"; return 0; }
  for t in $PROFILE_TREES; do
    tree=$(tree_of "$t"); out="$d/$t"
    if [[ "${FORCE:-0}" != 1 && -f "$out/.profile_complete" && -s "$out/analysis.txt" && -s "$out/analysis.json" ]]; then
      log "profile $t: done earlier ($out/analysis.txt)"; done_=$((done_ + 1)); continue
    fi
    if [[ "${FORCE:-0}" == 1 && -d "$out" && "$DRY_RUN" != 1 ]]; then
      k=1; while [[ -e "$out.attempt$k" ]]; do k=$((k + 1)); done; mv "$out" "$out.attempt$k"
    fi
    mkdir -p "$out"
    cache=()
    if [[ "$tree" == . ]]; then
      if root_blocked; then skipped+=("$t: env failed (root stack)"); echo "SKIP env failed (root stack)" > "$out/skip.txt"; continue; fi
      ensure_data 9 || { fails=$((fails + 1)); continue; }
      # the base phase's warm compile cache, when there is one (it only saves compile time)
      [[ -d "$SD/work/cache_warm/fp8" ]] && cache=("TORCHINDUCTOR_CACHE_DIR=$SD/work/cache_warm/fp8/inductor" "TRITON_CACHE_DIR=$SD/work/cache_warm/fp8/triton")
    elif ! variant_ready "$tree" "$out"; then
      skipped+=("$t: $READY_WHY"); echo "SKIP $READY_WHY" > "$out/skip.txt"; log "profile $t: SKIP ($READY_WHY)"; continue
    fi
    log "-- profile $t ($tree), steps $PROFILE_STEPS -- $(date '+%T')"
    if [[ "$DRY_RUN" == 1 ]]; then
      env NPROC="$NPROC" PY="$PY" PROFILE_STEPS="$PROFILE_STEPS" DRY_RUN=1 bash 182/profile/run.sh "$tree" "$out" || true
      continue
    fi
    rc=0
    env ${cache[@]+"${cache[@]}"} NPROC="$NPROC" PY="$PY" PROFILE_STEPS="$PROFILE_STEPS" \
      "$PY" -c "$TMO_PY" "$PROFILE_TIMEOUT_S" bash 182/profile/run.sh "$tree" "$out" > "$out/run.txt" 2>&1 || rc=$?
    tail -n 60 "$out/run.txt"
    if [[ "$rc" == 0 ]]; then done_=$((done_ + 1)); else fails=$((fails + 1)); log "profile $t FAILED (rc $rc; $out/run.txt)"; fi
  done
  [[ ${#skipped[@]} -gt 0 ]] && printf '%s\n' "${skipped[@]}" > "$d/skipped.txt"
  if (( fails == 0 && done_ == 0 )) && [[ ${#skipped[@]} -gt 0 && "$DRY_RUN" != 1 ]]; then skip "$(IFS=';'; echo "${skipped[*]}")"; return 0; fi
  (( fails == 0 ))
}

# -------------------------------------------------------------------------------------------- phase: nccl
phase_nccl() {
  local d="$SD/nccl" t tree v=0 fails=0 ran=0 skipped=() arms=()
  mkdir -p "$d"
  IFS='|' read -ra arms <<< "$NCCL_ARMS"
  if [[ "$DRY_RUN" == 1 ]]; then
    log "+ $PY 182/tools/nvls_check.py --nproc $NPROC --timeout-s $NCCL_PROBE_TIMEOUT_S --json $d/nvls_check.json   (NVLS unavailable -> SKIP)"
  else
    tmo $((3 * NCCL_PROBE_TIMEOUT_S + 300)) "$PY" 182/tools/nvls_check.py --nproc "$NPROC" --timeout-s "$NCCL_PROBE_TIMEOUT_S" \
      --json "$d/nvls_check.json" > "$d/nvls_check.txt" 2>&1 || v=$?
    cat "$d/nvls_check.txt"
    if [[ "$v" == 1 && "${FORCE_NCCL:-0}" != 1 ]]; then
      skip "$(grep '^NVLS: ' "$d/nvls_check.txt" | tail -n1 || echo 'NVLS unavailable') (the arms would all run the same algorithms; FORCE_NCCL=1 runs anyway)"; return 0
    fi
    [[ "$v" == 0 ]] || log "WARN: NVLS availability not established (nvls_check rc $v); running the A/B, whose nvls_force arm will fail fast if NVLS cannot be used"
  fi
  for t in $NCCL_TREES; do
    tree=$(tree_of "$t")
    XAB_AB_EXTRA=()
    if [[ "$tree" == . ]]; then
      if root_blocked; then skipped+=("$t: env failed (root stack)"); continue; fi
      ensure_data 9 || { fails=$((fails + 1)); continue; }
      # NCCL variables change no compiled code: every arm reuses one warm cache (base's, when it ran)
      XAB_AB_EXTRA=(--cache inherit --common-env "TORCHINDUCTOR_CACHE_DIR=$SD/work/cache_warm/fp8/inductor"
                    --common-env "TRITON_CACHE_DIR=$SD/work/cache_warm/fp8/triton")
    elif ! variant_ready "$tree" "$d/$t"; then
      skipped+=("$t: $READY_WHY"); log "nccl $t: SKIP ($READY_WHY)"; continue
    fi
    log "-- nccl $t ($tree): ${#arms[@]} arms x $(count_csv "$NCCL_SEEDS") seeds -- $(date '+%T')"
    ran=$((ran + 1))
    xab_tree "nccl_$t" "$tree" "$d/$t" "$NCCL_SEEDS" "${arms[@]}" || fails=$((fails + 1))
  done
  [[ ${#skipped[@]} -gt 0 ]] && printf '%s\n' "${skipped[@]}" > "$d/skipped.txt"
  if (( ran == 0 && fails == 0 )) && [[ "$DRY_RUN" != 1 ]]; then skip "no tree could run: $(IFS=';'; echo "${skipped[*]:-NCCL_TREES is empty}")"; return 0; fi
  (( fails == 0 ))
}

# --------------------------------------------------------------------------------------------- phase: xab
phase_xab() {
  local d="$SD/xab" f fails=0 out seen=" " base rc files=()
  mkdir -p "$d"
  while IFS= read -r f; do files+=("$f"); done < <(xab_files)
  [[ ${#files[@]} -gt 0 ]] || { skip "no 182/experiments.d/*.sh in this checkout"; return 0; }
  for f in "${files[@]}"; do
    base=$(basename "$f" .sh)
    XS_OK=0; XS_ERR="xab_run.py spec crashed on it"; XS_NAME=""   # nothing left over from the previous file
    eval "$("$PY" 182/tools/xab_run.py spec "$f" --repo "$REPO")" || true
    if [[ "$XS_OK" != 1 ]]; then
      log "xab $base: INVALID ($f): $XS_ERR"; echo "$f: $XS_ERR" > "$d/$base.invalid"; fails=$((fails + 1)); continue
    fi
    if [[ "$seen" == *" $XS_NAME "* ]]; then
      log "xab $base: INVALID: XAB_NAME=$XS_NAME is used by another file"; echo "$f: duplicate XAB_NAME $XS_NAME" > "$d/$base.invalid"
      fails=$((fails + 1)); continue
    fi
    [[ -f "$d/$base.invalid" ]] && mv "$d/$base.invalid" "$d/$base.invalid.old"
    seen+="$XS_NAME "
    out="$d/$XS_NAME"
    if ! xab_selected "$XS_NAME"; then log "xab $XS_NAME: not selected (XAB_ONLY='${XAB_ONLY:-}' XAB_SKIP='${XAB_SKIP:-}')"; continue; fi
    mkdir -p "$out"
    printf '%s\n' "file $f" "tree $XS_TREE" "seeds $XS_SEEDS" "est_min $XS_EST" "note $XS_NOTE" "${XS_ARMS[@]/#/arm }" > "$out/spec.txt"
    log ""; log "-- xab $XS_NAME: $XS_TREE, ${#XS_ARMS[@]} arms x $XS_N seeds, est. $XS_EST node-min -- $(date '+%T')"
    log "   $XS_NOTE"
    if [[ "${FORCE:-0}" != 1 && -f "$out/plan.json" ]] && "$PY" 182/tools/xab_run.py check "$out" >/dev/null 2>&1; then
      log "xab $XS_NAME: complete earlier"; cat "$out/report.txt" 2>/dev/null || true; echo PASS > "$out/status.txt"; continue
    fi
    XAB_AB_EXTRA=()
    if [[ "$XS_ROOT" == 1 ]]; then
      if root_blocked; then echo "SKIP env failed (root stack)" > "$out/status.txt"; log "xab $XS_NAME: SKIP (env failed: root stack)"; continue; fi
      ensure_data 9 || { fails=$((fails + 1)); echo "FAIL data" > "$out/status.txt"; continue; }
    elif ! variant_ready "$XS_TREE" "$out"; then
      echo "SKIP $READY_WHY" > "$out/status.txt"; log "xab $XS_NAME: SKIP ($READY_WHY)"; continue
    fi
    rc=0; xab_tree "$XS_NAME" "$XS_TREE" "$out" "$XS_SEEDS" "${XS_ARMS[@]}" || rc=$?
    if [[ "$rc" == 0 ]]; then echo PASS > "$out/status.txt"; else echo "FAIL rc=$rc" > "$out/status.txt"; fails=$((fails + 1)); fi
  done
  (( fails == 0 ))
}

# ------------------------------------------------------------------------------------------ phase: bundle
phase_bundle() {
  local date base out
  date=$(sed -n 's/^session_date //p' "$SD/meta.txt" 2>/dev/null | head -n1); date=${date:-$(date +%F)}
  base=$(basename "$SD")
  out="$(dirname "$SD")/session-$date-$SHA.tar.gz"
  "$PY" 182/tools/summarize_session.py "$SD" > "$SD/summary.txt" || log "WARN: summarize_session.py failed"
  cat "$SD/summary.txt"
  # profile traces stay on the node when they are big (analysis.txt/json and the rank summaries are bundled)
  local big=() f
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    big+=("--exclude=$base/${f#"$SD"/}"); log "not bundled (> ${PROFILE_BUNDLE_MB} MB, still in the session dir): $f"
  done < <(find "$SD/profile" -name 'trace_rank*.json.gz' -size +"${PROFILE_BUNDLE_MB}"M 2>/dev/null)
  COPYFILE_DISABLE=1 tar -czf "$out.tmp" -C "$(dirname "$SD")" --exclude="$base/work" ${big[@]+"${big[@]}"} "$base"
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
  PHASE="$p" bash "$SELF" --run-phase "$p" || rc=$?
  if [[ "$rc" != 0 ]]; then echo "FAIL rc=$rc" > "$SD/status/$p"
  elif [[ ! -f "$SD/status/$p" ]]; then echo "PASS" > "$SD/status/$p"; fi
  log "== $p: $(status_of "$p")"
}

case "${1:-}" in
  --run-phase) PHASE=$2; if is_variant "$2"; then phase_variant; else "phase_$2"; fi; exit $?;;
  --prep-chain) shift; prep_chain "$@"; exit $?;;
  -h|--help) sed -n '2,/^set -euo pipefail/p' "$SELF" | sed '$d'; exit 0;;
esac

WANT=()
for w in "$@"; do [[ "$w" == t2 ]] && w=t2cpu; WANT+=("$w"); done   # t2: the old name of t2cpu
if [[ ${#WANT[@]} -eq 0 ]]; then
  # QUICK: the two results that need nothing exotic and matter most. `base` is this node's own reference (the one
  # earlier measurement left no log), t1gpu is the fastest GPU-only tree (reported 30.67 s vs the 39.9 s record)
  # and runs on the record's kind of host; the CPU-retrieval trees need a large-RAM host and long preps, and the
  # fp8 A/Bs need 4 seed pairs to say anything. sanity (3 min) is the first time the fp8 kernels meet real H100s.
  if [[ "${QUICK:-0}" == 1 ]]; then WANT=("${QUICK_PHASES[@]}"); else WANT=(all); fi
fi
SELECTED_PHASES=()
for p in "${PHASES[@]}"; do
  for w in "${WANT[@]}"; do
    if [[ "$w" == all || "$w" == "$p" ]]; then SELECTED_PHASES+=("$p"); break; fi
  done
done
for w in "${WANT[@]}" ${SKIP_PHASES:-}; do
  [[ "$w" == all || " ${PHASES[*]} " == *" $w "* ]] || { echo "unknown phase '$w' (phases: ${PHASES[*]} | all)" >&2; exit 2; }
done
if [[ -n "${SKIP_PHASES:-}" ]]; then   # dropped by name, e.g. SKIP_PHASES="nccl xab"
  KEEP=(); for p in "${SELECTED_PHASES[@]}"; do [[ " $SKIP_PHASES " == *" $p "* ]] || KEEP+=("$p"); done
  SELECTED_PHASES=(${KEEP[@]+"${KEEP[@]}"})
fi

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
log "phases: ${SELECTED_PHASES[*]}"
log "estimates (node-min on 8 GPUs; guesses until a run exists, variant preps not included):"
for p in "${SELECTED_PHASES[@]}"; do
  note=""
  if [[ "${FORCE:-0}" != 1 && "$p" != bundle && "$(status_of "$p")" == PASS* ]]; then note="  (already PASS: skipped)"
  elif is_variant "$p"; then note="  ($(knob "$p" RUNS) run(s) of $(variant_dir "$p"); skipped if its check FAILs)"
  elif [[ "$p" == profile ]]; then note="  (torch.profiler, steps $PROFILE_STEPS of: ${PROFILE_TREES:-nothing}; not timing runs)"
  elif [[ "$p" == nccl ]]; then note="  ($(nccl_arm_count) arms x $(count_csv "$NCCL_SEEDS") seeds on: ${NCCL_TREES:-nothing}; skipped if NVLS is unavailable)"
  elif [[ "$p" == xab ]]; then note="  ($(xab_files | wc -l | tr -d ' ') file(s) in 182/experiments.d; XAB_ONLY/XAB_SKIP select)"
  elif [[ "$p" == hadamard ]]; then note="  (only if the gate finds a gap)"
  elif [[ "$p" == attno ]] && ! git grep -q FP8_ATTN_O HEAD -- track_1_short/config.py 2>/dev/null; then note="  (skipped: FP8_ATTN_O is not in this checkout)"; fi
  log "$(printf '  %-9s %4s%s' "$p" "$(estimate "$p")" "$note")"
done
log "  total     $(printf '%4s' "$total") node-min upper bound (~$((total * 8)) GPU-min)"
[[ "$DRY_RUN" == 1 ]] && log "DRY_RUN=1: GPU/pip/download commands are printed, not run; files go to $SD"

# Variant preps (downloads, builds: CPU/network only) start now, in one low-priority background job in its own
# process group, and overlap env/tests/sanity/base; every variant phase waits for the whole job before it times
# anything.
PREP_LIST=()
for p in "${SELECTED_PHASES[@]}"; do
  is_variant "$p" || continue
  [[ "${FORCE:-0}" != 1 && "$(status_of "$p")" == PASS* ]] && continue
  [[ -f "$(variant_dir "$p")/182/run_node.sh" ]] || continue
  PREP_LIST+=("$p")
done
if [[ ${#PREP_LIST[@]} -gt 0 ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    [[ "$PREP_BACKGROUND" == 1 ]] && log "+ (background, nice 19) check + prep of: ${PREP_LIST[*]}"
  elif chain_alive; then
    log "prep: the background job of an earlier invocation is still running (pid $(cat "$SD/prep_chain.pid")); not starting another"
  else
    rm -f "$SD/prep_chain.pid" "$SD/prep_chain.rc"
    for p in "${PREP_LIST[@]}"; do rm -f "$SD/$p/prep.rc"; done   # prep is idempotent: every invocation re-runs it
    if [[ "$PREP_BACKGROUND" == 1 ]]; then
      LOWPRIO=(nice -n 19); command -v ionice >/dev/null 2>&1 && LOWPRIO+=(ionice -c2 -n7)
      "$PY" -c 'import os, sys
try: os.setsid()
except OSError: pass
os.execvp(sys.argv[1], sys.argv[1:])' "${LOWPRIO[@]}" \
        bash "$SELF" --prep-chain "${PREP_LIST[@]}" > "$SD/prep_chain.txt" 2>&1 < /dev/null &
      echo $! > "$SD/prep_chain.pid"
      log "prep: check + prep of ${PREP_LIST[*]} started in the background (pid $!; logs $SD/<phase>/prep.txt)"
    fi
  fi
fi

CUR_PHASE=""
on_interrupt() {
  [[ -n "$CUR_PHASE" ]] && echo "FAIL interrupted" > "$SD/status/$CUR_PHASE"
  if chain_alive; then kill -TERM -- "-$(cat "$SD/prep_chain.pid")" 2>/dev/null || true; rm -f "$SD/prep_chain.pid"; fi   # preps resume on re-run
  log ""; log "interrupted during ${CUR_PHASE:-startup}. Re-run the same command to resume, or pack what exists:"
  log "  bash 182/gpu_session.sh bundle"
  exit 130
}
trap on_interrupt INT TERM HUP
FAILED=()
ENV_FAILED=0
for p in "${SELECTED_PHASES[@]}"; do
  CUR_PHASE=$p
  if [[ "$ENV_FAILED" == 1 && "$ROOT_PHASES" == *" $p "* ]]; then
    echo "SKIP env failed (root stack/driver/data; see env above): no root phase ran" > "$SD/status/$p"
    log ""; log "== $p: SKIP (env failed: fix the stack and re-run, or ALLOW_STACK=1)"; continue
  fi
  if [[ "$p" != bundle && "$p" != env ]] && ! is_variant "$p" && chain_alive; then
    echo "$p started while the background prep job (downloads/builds, nice 19) was still running" >> "$SD/prep_overlap.txt"
  fi
  run_phase "$p"
  [[ "$(status_of "$p")" == FAIL* ]] && FAILED+=("$p")
  [[ "$p" == env && "$(status_of env)" == FAIL* ]] && ENV_FAILED=1
done
log ""
for p in "${SELECTED_PHASES[@]}"; do log "  $p: $(status_of "$p")"; done
chain_alive && log "note: the background prep job (pid $(cat "$SD/prep_chain.pid")) is still running; it stops by itself"
[[ -f "$SD/bundle_path" && " ${SELECTED_PHASES[*]} " == *" bundle "* ]] && log "SEND THIS FILE BACK: $(cat "$SD/bundle_path")"
[[ ${#FAILED[@]} -eq 0 ]] || { log "failed phases: ${FAILED[*]}"; exit 1; }
