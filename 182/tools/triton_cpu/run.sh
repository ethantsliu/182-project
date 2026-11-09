#!/usr/bin/env bash
# Run a command in the Triton-interpreter CPU container with the repo mounted at /repo:
#   bash 182/tools/triton_cpu/run.sh python -m pytest 182/tools/triton_cpu 182/hadamard-fp8/tests -q
# Builds the image on first use. Env passed through: TRITON_CPU_NO_SHIM, TRITON_CPU_NUM_SMS, HADAMARD_FP8,
# HADAMARD_SEED. TRITON_CPU_IMAGE overrides the image tag.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
image="${TRITON_CPU_IMAGE:-triton-cpu:3.6.0}"
[ $# -gt 0 ] || { echo "usage: $0 <cmd> [args...]" >&2; exit 2; }
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build -t "$image" "$here" >&2
fi
envs=(-e TRITON_INTERPRET=1 -e PYTHONPATH=/repo/182/tools/triton_cpu -e PYTHONDONTWRITEBYTECODE=1
      -e "PYTEST_ADDOPTS=-p no:cacheprovider ${PYTEST_ADDOPTS:-}")
for v in TRITON_CPU_NO_SHIM TRITON_CPU_NUM_SMS HADAMARD_FP8 HADAMARD_SEED; do
  [ -n "${!v:-}" ] && envs+=(-e "$v=${!v}")
done
tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
exec docker run --rm "${tty[@]}" "${envs[@]}" -v "$repo":/repo -w /repo "$image" "$@"
