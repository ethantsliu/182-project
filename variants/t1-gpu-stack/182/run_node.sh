#!/usr/bin/env bash
# The node contract for variants/t1-gpu-stack (upstream PR #397 + our flags, all off by default).
#
#   182/run_node.sh check   CPU-only preflight (< 60 s): PASS|WARN|FAIL lines, exit 1 on any FAIL
#   182/run_node.sh prep    data shards, the pinned FA3 binary, the tokenizer file (idempotent, PREP_TIMEOUT_S)
#   182/run_node.sh run     one timed run of #397; the last stdout line is `RESULT variant=t1-gpu-stack ...`
#   182/run_node.sh ab      #397 vs our flags, one per arm, interleaved by seed (AB_RUNS, AB_ARMS, AB_STEPS)
#
# Run it from anywhere. It cds to the variant directory, activates the repo root's .venv-run if no venv is active
# (NODE_VENV names another one, RUN_NODE_NO_VENV=1 skips this, PYTHON picks the interpreter), and hands over to
# 182/t1-gpu-stack/node.py, whose docstring lists every environment knob.
set -uo pipefail

VARIANT=t1-gpu-stack
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT_DIR="$(cd "$HERE/.." && pwd)"
REPO_DIR="$(cd "$VARIANT_DIR/../.." && pwd)"
MODE="${1:-}"

die() {  # a failure before python: in the shape the mode promises
  if [[ "$MODE" == run ]]; then
    echo "RESULT variant=$VARIANT status=failed reason=$*"
  else
    echo "FAIL $*"
  fi
  exit 1
}

case "$MODE" in
  check|prep|run|ab) ;;
  *) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2;;
esac
[[ $# -eq 1 ]] || { echo "usage: 182/run_node.sh check|prep|run|ab (knobs are environment variables)"; exit 2; }
cd "$VARIANT_DIR" || die "cannot cd to $VARIANT_DIR"

ACTIVATE="$REPO_DIR/${NODE_VENV:-.venv-run}/bin/activate"
if [[ -z "${VIRTUAL_ENV:-}" && -z "${RUN_NODE_NO_VENV:-}" && -f "$ACTIVATE" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ACTIVATE"   # also puts libcudart.so.13 on LD_LIBRARY_PATH when 182/tools/node_setup.sh found it off-path
  set -u
fi

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for candidate in python python3; do
    command -v "$candidate" >/dev/null 2>&1 && { PY="$candidate"; break; }
  done
fi
[[ -n "$PY" ]] && command -v "$PY" >/dev/null 2>&1 \
  || die "python: no interpreter found (PYTHON=${PYTHON:-<unset>}); fix: bash 182/tools/node_setup.sh --mode venv (repo root)"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null \
  || die "python: $("$PY" --version 2>&1) at $(command -v "$PY") is older than 3.10; fix: bash 182/tools/node_setup.sh --mode venv (repo root)"

exec "$PY" "$HERE/t1-gpu-stack/node.py" "$MODE"
