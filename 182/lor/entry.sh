#!/usr/bin/env bash
# Runs INSIDE a Lor container (nvidia/cuda 13.1.1 devel, root). Sets the node up, runs the given command from the
# repo root in the pinned venv, then prints the results as a base64 tarball between RESULTS markers (a job's disk
# is gone when it exits, and there is no file copy, so the log is the only way out).
#   bash 182/lor/entry.sh '<shell command>'      # results the command wants kept go under $OUT (/work/out)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
export OUT=/work/out; mkdir -p "$OUT"
t0=$(date +%s)
bash 182/tools/node_setup.sh --mode venv > "$OUT/setup.log" 2>&1; echo "LOR setup rc=$? secs=$(( $(date +%s) - t0 ))"
tail -5 "$OUT/setup.log"
# shellcheck disable=SC1091
source .venv-run/bin/activate
git log --oneline -1 | tee "$OUT/sha"
bash -c "$1"; rc=$?
echo "LOR command rc=$rc secs=$(( $(date +%s) - t0 ))"
cp node_report.txt "$OUT/" 2>/dev/null
echo "LOR-RESULTS-BEGIN"; tar -C /work -czf - --exclude='*.pt' --exclude='*.json.gz' out | base64 -w 76; echo "LOR-RESULTS-END"
exit $rc
