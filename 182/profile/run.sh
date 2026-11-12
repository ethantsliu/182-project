#!/usr/bin/env bash
# One profiled training run of a tree (torch.profiler over PROFILE_STEPS of the timed loop), then analyze.py.
#
#   bash 182/profile/run.sh variants/t1-gpu-stack OUT_DIR    # through the tree's own 182/run_node.sh run
#   bash 182/profile/run.sh . OUT_DIR                        # the root record: torchrun ... hook.py train_gpt.py
#
# Env: PROFILE_STEPS (300:20), NPROC (8), PY (python, for analyze.py), PROFILE_CUPTI_INIT / PROFILE_SHAPES (hook.py),
# DRY_RUN=1 (print the commands; a variant's run_node.sh also prints its own). Outputs in OUT_DIR: train.out (the
# console), trace_rank0.json.gz, rank<R>_summary.json, meta_rank<R>.json, analysis.txt, analysis.json, logs/.
# Exit 0 when a trace was analyzed (the run's own RESULT is reported but a profiled run is not a timing run).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -eq 2 ]] || { sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 2; }
TREE=$1
OUT=$2
PY=${PY:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
NPROC=${NPROC:-8}
export PROFILE_STEPS=${PROFILE_STEPS:-300:20}
[[ -d "$TREE" ]] || { echo "FAIL profile: no tree $TREE"; exit 2; }
TREE="$(cd "$TREE" && pwd)"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export PROFILE_DIR="$OUT"
rm -f "$OUT/.shim_used"
if [[ -f "$TREE/182/run_node.sh" ]]; then
  # A variant's run_node.sh activates <repo>/.venv-run when no venv is active, which would put the venv's torchrun
  # ahead of the shim. Activate it here first (same rule: NODE_VENV, RUN_NODE_NO_VENV), so run_node.sh sees a
  # venv, leaves PATH alone, and the shim stays first.
  ACT="$(cd "$HERE/../.." && pwd)/${NODE_VENV:-.venv-run}/bin/activate"
  if [[ -z "${VIRTUAL_ENV:-}" && -z "${RUN_NODE_NO_VENV:-}" && -f "$ACT" && "${DRY_RUN:-0}" != 1 ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$ACT"
    set -u
  fi
  CMD=(env "PATH=$HERE/shim:$PATH" "NPROC=$NPROC" "LOG_DIR=$OUT/logs" bash "$TREE/182/run_node.sh" run)
else
  [[ -f "$TREE/train_gpt.py" ]] || { echo "FAIL profile: $TREE has neither 182/run_node.sh nor train_gpt.py"; exit 2; }
  CMD=(torchrun --standalone "--nproc_per_node=$NPROC" "$HERE/hook.py" train_gpt.py)
fi
echo "profile: $TREE steps $PROFILE_STEPS -> $OUT"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "+ cd $TREE && PROFILE_STEPS=$PROFILE_STEPS PROFILE_DIR=$OUT ${CMD[*]}"
  [[ -f "$TREE/182/run_node.sh" ]] && (cd "$TREE" && "${CMD[@]}" 2>&1 | sed 's/^/    /')
  echo "+ $PY $HERE/analyze.py $OUT --json $OUT/analysis.json"
  exit 0
fi
(cd "$TREE" && "${CMD[@]}") 2>&1 | tee "$OUT/train.out"
rc=${PIPESTATUS[0]}
echo "profile: training run exited $rc"
grep -E '^RESULT ' "$OUT/train.out" | tail -n1 | sed 's/^/profiled run (NOT a timing run): /' || true
if [[ -f "$TREE/182/run_node.sh" && ! -f "$OUT/.shim_used" ]]; then
  echo "FAIL profile: the trainer did not start through 182/profile/shim/torchrun (another torchrun came first on PATH,"
  echo "  or run_node.sh failed before launching it); nothing was profiled. See $OUT/train.out"
  exit 1
fi
if [[ ! -f "$OUT/trace_rank0.json.gz" ]]; then
  echo "FAIL profile: no trace_rank0.json.gz in $OUT (window opened? see the [profile] lines in $OUT/train.out)"
  "$PY" "$HERE/analyze.py" "$OUT" || true
  exit 1
fi
"$PY" "$HERE/analyze.py" "$OUT" --json "$OUT/analysis.json" > "$OUT/analysis.txt" 2>&1 || { cat "$OUT/analysis.txt"; echo "FAIL profile: analyze.py failed"; exit 1; }
cat "$OUT/analysis.txt"
exit 0
