#!/usr/bin/env bash
# Capture an Nsight Systems trace of a few steps of the timed training loop on rank 0, export it to
# sqlite and analyze it (analyze_nsys.py). Run from the repo (or worktree) root on the GPU box.
#
#   bash 182/tools/profile_nsys.sh [-o DIR] [--start 300] [--steps 5] [--nproc 8] [--script train_gpt.py]
#                                  [--mode hook|window] [--delay S --duration S] [--rank R]
#                                  [--analyze-args "..."] [-- trainer args]
#
# Modes:
#   hook   (default) all ranks run the trainer through nsys_step_hook.py, which brackets steps
#          [start-1, start+steps] with cudaProfilerStart/Stop and wraps [start, start+steps) in NVTX
#          "step N" ranges; nsys on rank R only uses --capture-range=cudaProfilerApi. Works for any
#          trainer with a step function and main() (train_gpt.py: train_step). Every rank goes through
#          the hook so the two capture-boundary synchronizes happen on all ranks alike.
#   window plain nsys --delay/--duration on rank R (e.g. train_gpt_medium.py, whose loop is inline);
#          steps then come from analyze_nsys.py --step-marker REGEX (pass it in --analyze-args).
#
# The profiled run is not a timing run (nsys + node-level graph tracing slow rank 0 and with it every
# collective); use ab_run.py for times. Needs nsys >= 2023.4 on PATH (for --cuda-graph-trace=node).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- per-rank wrapper (re-entered by torchrun --no-python) ---------------------------------------
if [[ "${_NSYS_WRAP:-}" == "1" ]]; then
  launch=(python)
  [[ "$_NSYS_MODE" == "hook" ]] && launch=(python "$HERE/nsys_step_hook.py")
  if [[ "${RANK:-0}" == "$_NSYS_RANK" ]]; then
    common=(nsys profile -o "$_NSYS_OUT/rank${RANK}" -f true --trace=cuda,nvtx --cuda-graph-trace=node
            --sample=none --cpuctxsw=none --cuda-memory-usage=false --stats=false)
    if [[ "$_NSYS_MODE" == "hook" ]]; then
      exec "${common[@]}" --capture-range=cudaProfilerApi --capture-range-end=stop "${launch[@]}" "$@"
    else
      exec "${common[@]}" --delay="$_NSYS_DELAY" --duration="$_NSYS_DURATION" "${launch[@]}" "$@"
    fi
  fi
  exec "${launch[@]}" "$@"
fi

# ---- driver -------------------------------------------------------------------------------------
OUT="nsys/$(date +%Y%m%d_%H%M%S)"; START=300; STEPS=5; NPROC=8; SCRIPT=train_gpt.py; MODE=hook
DELAY=600; DURATION=5; PROF_RANK=0; ANALYZE_ARGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out) OUT="$2"; shift 2;;
    --start) START="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --nproc) NPROC="$2"; shift 2;;
    --script) SCRIPT="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --delay) DELAY="$2"; shift 2;;
    --duration) DURATION="$2"; shift 2;;
    --rank) PROF_RANK="$2"; shift 2;;
    --analyze-args) ANALYZE_ARGS="$2"; shift 2;;
    --) shift; break;;
    -h|--help) sed -n '2,22p' "$0"; exit 0;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
command -v nsys >/dev/null || { echo "nsys not on PATH (install Nsight Systems, or use the CUDA toolkit's)" >&2; exit 1; }
command -v torchrun >/dev/null || { echo "torchrun not on PATH (activate the run environment)" >&2; exit 1; }
nsys --version
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
echo "profiling $SCRIPT on rank $PROF_RANK, mode=$MODE, steps [$START, $((START + STEPS))) -> $OUT"

export _NSYS_WRAP=1 _NSYS_MODE="$MODE" _NSYS_RANK="$PROF_RANK" _NSYS_OUT="$OUT" _NSYS_DELAY="$DELAY" _NSYS_DURATION="$DURATION"
export NSYS_START_STEP="$START" NSYS_NUM_STEPS="$STEPS"
torchrun --standalone --nproc_per_node="$NPROC" --no-python bash "$HERE/profile_nsys.sh" "$SCRIPT" "$@" 2>&1 | tee "$OUT/train.out"
unset _NSYS_WRAP

REP="$OUT/rank${PROF_RANK}.nsys-rep"
[[ -f "$REP" ]] || { echo "no report at $REP (did the capture range start? see $OUT/train.out)" >&2; exit 1; }
nsys export --type sqlite --force-overwrite true -o "$OUT/rank${PROF_RANK}.sqlite" "$REP"
# shellcheck disable=SC2086
python "$HERE/analyze_nsys.py" "$OUT/rank${PROF_RANK}.sqlite" --json "$OUT/analysis.json" $ANALYZE_ARGS | tee "$OUT/analysis.md"
echo "trace: $REP  sqlite: $OUT/rank${PROF_RANK}.sqlite  analysis: $OUT/analysis.md"
