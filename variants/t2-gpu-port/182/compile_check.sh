#!/usr/bin/env bash
# Compile-check this tree's GPU kernels for H100 without a GPU (see compile_check.py). Needs docker, the
# triton-cpu:3.6.0 image (built by the repo root's `bash 182/tools/triton_cpu/run.sh true`) and network
# access for two pip wheels (NVRTC 12.8 and the CUDA headers, ~100 MB, not kept).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant="$(cd "$here/.." && pwd)"
image="${TRITON_CPU_IMAGE:-triton-cpu:3.6.0}"
command -v docker >/dev/null || { echo "FAIL docker not found" >&2; exit 2; }
docker image inspect "$image" >/dev/null 2>&1 || {
  echo "FAIL image $image missing: from the repo root run  bash 182/tools/triton_cpu/run.sh true" >&2; exit 2; }
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -e TRITON_INTERPRET=0 -v "$variant":/variant:ro -w /variant "$image" bash -c '
  pip install -q --root-user-action=ignore --disable-pip-version-check \
      nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90 >&2 &&
  python 182/compile_check.py'
