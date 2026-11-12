#!/usr/bin/env bash
# Run after the pinned nanoGPT session. Resumes successful jobs; retains failures.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
PY=${PY:-python}
FNO_OUT=${FNO_OUT:-$REPO/182/results/fno-followup-$(date -u +%Y%m%d)-$(git rev-parse --short=12 HEAD)}
export FNO_OUT PYTHONUNBUFFERED=1
STAGES=${FNO_STAGES:-"controls-darcy ns-boundary refresh-darcy refresh-ns"}
if [[ ${DRY_RUN:-0} == 1 ]]; then
  PLAN=$(mktemp /tmp/fno-plan.XXXXXX.json)
  "$PY" "$HERE/fno_followup.py" plan --out "$PLAN"
  cat "$PLAN"
  printf '\nSelected stages: %s\nResults: %s\nNo GPU work was launched.\n' "$STAGES" "$FNO_OUT"
  exit 0
fi
mkdir -p "$FNO_OUT"
exec 9>"$FNO_OUT/queue.lock"
flock -n 9 || { echo "FNO queue already running in $FNO_OUT" >&2; exit 1; }
exec > >(tee -a "$FNO_OUT/queue.log") 2>&1
printf 'FNO start %s\n' "$(date -u +%FT%TZ)"
"$PY" -c 'import torch, triton; print(torch.__version__, triton.__version__); assert torch.__version__ == "2.10.0+cu128"; assert triton.__version__ == "3.6.0"'
git rev-parse HEAD > "$FNO_OUT/git-sha.txt"
git diff HEAD -- 182/fft-optim > "$FNO_OUT/source-diff.patch"
"$PY" -m pip freeze > "$FNO_OUT/pip-freeze.txt"
nvidia-smi > "$FNO_OUT/nvidia-smi.txt"
"$PY" "$HERE/fno_followup.py" plan --out "$FNO_OUT/plan.json"
step() {
  local name=$1 start rc=0
  shift
  start=$(date -u +%FT%TZ)
  "$@" > >(tee -a "$FNO_OUT/$name.log") 2>&1 || rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$start" "$(date -u +%FT%TZ)" "$rc" >> "$FNO_OUT/status.tsv"
  return "$rc"
}
step prepare "$PY" "$HERE/fno_followup.py" prepare --out "$FNO_OUT"
failed=0
for stage in $STAGES; do
  step "$stage" "$PY" "$HERE/fno_followup.py" stage --stage "$stage" --out "$FNO_OUT" || failed=1
done
"$PY" "$HERE/fno_followup.py" report --out "$FNO_OUT"
tar -czf "$FNO_OUT.tar.gz" -C "$(dirname "$FNO_OUT")" "$(basename "$FNO_OUT")"
printf 'FNO finished %s; failed=%s; bundle=%s.tar.gz\n' "$(date -u +%FT%TZ)" "$failed" "$FNO_OUT"
exit "$failed"
