#!/usr/bin/env bash
# Run 182/val_ce_memory.py on PR #261's torch (2.10.0) without a GPU: docker, the triton-cpu:3.6.0 image (built
# by the repo root's `bash 182/tools/triton_cpu/run.sh true`) and network access for apt's g++, which
# Inductor's CPU backend compiles with (not kept).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant="$(cd "$here/.." && pwd)"
image="${TRITON_CPU_IMAGE:-triton-cpu:3.6.0}"
command -v docker >/dev/null || { echo "FAIL docker not found" >&2; exit 2; }
docker image inspect "$image" >/dev/null 2>&1 || {
  echo "FAIL image $image missing: from the repo root run  bash 182/tools/triton_cpu/run.sh true" >&2; exit 2; }
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$variant":/variant:ro -w /variant "$image" bash -c '
  (apt-get update -qq && apt-get install -y -qq g++) >/dev/null 2>&1 || { echo "FAIL apt-get install g++" >&2; exit 2; }
  python 182/val_ce_memory.py'
