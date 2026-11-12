#!/usr/bin/env bash
# Node entry point for variants/t2-gpu-port (GPT-2 Medium, PR #261 + flags). Run from anywhere:
#
#   variants/t2-gpu-port/182/run_node.sh check   # CPU-only preflight: PASS|WARN|FAIL lines, exit 1 on any FAIL
#   variants/t2-gpu-port/182/run_node.sh prep    # data (25 shards), FA3 wheel, tokenizer cache, mask check
#   variants/t2-gpu-port/182/run_node.sh run     # one timed run, PR #261's config; last line: RESULT ...
#   variants/t2-gpu-port/182/run_node.sh ab      # interleaved A/B, default base vs FUSED_MLP=1 (see README)
#
# Flags are plain environment variables (CANON_MASK=1 run_node.sh run) or EXTRA_ENV="K=V K=V".
# Other knobs: NPROC, DATA_PATH, LOG_DIR, DRY_RUN=1, RUN_TIMEOUT, SKIP_CHECK=1, PREP_TIMEOUT, PREP_SHARDS,
# AB_RUNS, AB_A, AB_B, PYTHON. The logic is in node.py (standard library only).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/.."
py="${PYTHON:-}"
if [ -z "$py" ]; then
  for c in python python3; do
    if command -v "$c" >/dev/null 2>&1; then py="$c"; break; fi
  done
fi
if [ -z "$py" ]; then
  echo "FAIL python: no python or python3 on PATH; activate the repo's venv or set PYTHON=<path>"
  [ "${1:-}" = run ] && echo "RESULT variant=t2-gpu-port status=failed reason=no python on PATH"
  exit 1
fi
export PYTHON="$(command -v "$py")"
exec "$PYTHON" "$here/node.py" "$@"
