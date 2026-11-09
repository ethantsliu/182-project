#!/usr/bin/env bash
# Run a command in the Triton-interpreter CPU container with the repo mounted at /repo:
#   bash 182/tools/triton_cpu/run.sh python -m pytest 182/tools/triton_cpu 182/hadamard-fp8/tests -q
#   TRITON_INTERPRET=0 bash 182/tools/triton_cpu/run.sh python 182/tools/triton_cpu/compile_check.py  # sm_90a compile
# Builds the image on first use. Env passed through: TRITON_INTERPRET (default 1), TRITON_CPU_NO_SHIM,
# TRITON_CPU_NUM_SMS, HADAMARD_FP8, HADAMARD_SEED, HADAMARD_BLOCK, HADAMARD_GEMMS. TRITON_CPU_VERSION=3.5.1 selects the triton 3.5.1 / torch 2.9.1
# image (default 3.6.0 / 2.10.0, the record's pin); TRITON_CPU_IMAGE overrides the tag.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
ver="${TRITON_CPU_VERSION:-3.6.0}"
case "$ver" in
  3.6.0) torch_ver=2.10.0 ;;
  3.5.1) torch_ver=2.9.1 ;;
  *) echo "TRITON_CPU_VERSION=$ver: add its torch pair to run.sh" >&2; exit 2 ;;
esac
image="${TRITON_CPU_IMAGE:-triton-cpu:$ver}"
[ $# -gt 0 ] || { echo "usage: $0 <cmd> [args...]" >&2; exit 2; }
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --build-arg TRITON="$ver" --build-arg TORCH="$torch_ver" -t "$image" "$here" >&2
fi
envs=(-e "TRITON_INTERPRET=${TRITON_INTERPRET:-1}" -e PYTHONPATH=/repo/182/tools/triton_cpu -e PYTHONDONTWRITEBYTECODE=1
      -e "PYTEST_ADDOPTS=-p no:cacheprovider ${PYTEST_ADDOPTS:-}")
for v in TRITON_CPU_NO_SHIM TRITON_CPU_NUM_SMS HADAMARD_FP8 HADAMARD_SEED HADAMARD_BLOCK HADAMARD_GEMMS; do
  [ -n "${!v:-}" ] && envs+=(-e "$v=${!v}")
done
tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
exec docker run --rm "${tty[@]}" "${envs[@]}" -v "$repo":/repo -w /repo "$image" "$@"
