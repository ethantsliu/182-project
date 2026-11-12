#!/usr/bin/env bash
# The node entry point of variants/t2-cpu-ngram (Track 2, GPT-2 Medium, val loss <= 2.92, on PR #380's CPU retrieval).
#
#   bash variants/t2-cpu-ngram/182/run_node.sh check   # CPU-only preflight: PASS|WARN|FAIL lines, exit 1 on a FAIL
#   bash variants/t2-cpu-ngram/182/run_node.sh prep    # 103 shards, the Rust extension, the FA3 kernel (off the clock)
#   bash variants/t2-cpu-ngram/182/run_node.sh run     # one timed run; the last line is RESULT variant=t2-cpu-ngram ...
#   bash variants/t2-cpu-ngram/182/run_node.sh ab      # the default against a shorter run, seed-paired (AB_RUNS, AB_ENV)
#
# Run it from anywhere. Python: $PY, else the active virtualenv, else the repository's .venv-run (182/tools/node_setup.sh
# makes it), else python3. The work and the list of environment variables are in 182/t2-cpu-ngram/node.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$VARIANT/../.." && pwd)"
mode="${1:-}"
case "$mode" in
  check|prep|run|ab) ;;
  *) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2;;
esac
fail() {  # the contract's failure lines, for the modes that have one
  echo "FAIL python: $1"
  if [[ "$mode" == run ]]; then echo "RESULT variant=t2-cpu-ngram status=failed reason=$1"; fi
  exit 1
}
if [[ -z "${PY:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then PY="$VIRTUAL_ENV/bin/python"
  elif [[ -x "$ROOT/.venv-run/bin/python" ]]; then PY="$ROOT/.venv-run/bin/python"
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python >/dev/null 2>&1; then PY=python
  else fail "no python found (set PY=/path/to/python, or run bash 182/tools/node_setup.sh from the repository root)"
  fi
fi
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null \
  || fail "$PY is missing or older than Python 3.10 (set PY=/path/to/python; the stack was run on 3.12)"
cd "$VARIANT"
exec "$PY" 182/t2-cpu-ngram/node.py "$@"
