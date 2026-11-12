#!/usr/bin/env bash
# The node contract of variants/t1-cpu-ngram. Run from anywhere:
#
#   bash variants/t1-cpu-ngram/182/run_node.sh prep     # off the clock: build the extension, fetch all 103 shards, then check
#   bash variants/t1-cpu-ngram/182/run_node.sh check    # CPU-only preflight: PASS|WARN|FAIL lines, exit 1 on any FAIL
#   bash variants/t1-cpu-ngram/182/run_node.sh run      # one timed run; last stdout line: RESULT variant=t1-cpu-ngram ...
#   bash variants/t1-cpu-ngram/182/run_node.sh ab       # AB_RUNS (4) paired runs, base vs AB_B (default TABLE_VOTE=1)
#
# Knobs (CONFIG, NPROC, SEED, EXTRA_ENV, DATA_PATH, LOG_DIR, DRY_RUN, FORCE, ...): see the top of 182/node.py.
# PYTHON picks the interpreter (default: python3 on PATH, i.e. the activated environment's).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.." || { echo "RESULT variant=t1-cpu-ngram status=failed reason=cannot cd to the variant directory"; exit 1; }
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "FAIL python: $PY not found; fix: activate the run environment or set PYTHON"
  case "${1:-}" in run|ab) echo "RESULT variant=t1-cpu-ngram status=failed reason=$PY not found";; esac
  exit 1
fi
exec "$PY" "$HERE/node.py" "$@"
