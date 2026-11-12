#!/usr/bin/env bash
# Run a command in the Triton-interpreter CPU container with the repo mounted at /repo:
#   bash 182/tools/triton_cpu/run.sh python -m pytest 182/tools/triton_cpu 182/hadamard-fp8/tests -q
#   bash 182/tools/triton_cpu/run.sh aot [--base <sha>]   # sm_90a compile check (compile_check.py), no GPU
# Builds the image on first use. Env passed through: TRITON_INTERPRET (default 1), TRITON_CPU_NO_SHIM,
# TRITON_CPU_NUM_SMS, HADAMARD_FP8, HADAMARD_SEED, HADAMARD_BLOCK, HADAMARD_GEMMS, FP8_ATTN_O_BASE_ATTENTION (a path
# under /repo: 182/fp8-attn-o/tests/test_interp_attn_o.py). TRITON_CPU_VERSION=3.5.1 selects the triton 3.5.1 / torch 2.9.1
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
[ $# -gt 0 ] || { echo "usage: $0 <cmd> [args...] | aot [--base <sha>]" >&2; exit 2; }
mounts=(-v "$repo":/repo)
if [ "$1" = aot ]; then  # real compilation for sm_90a instead of the interpreter
  shift
  export TRITON_INTERPRET=0
  cmd=(python 182/tools/triton_cpu/compile_check.py)
  if [ "${1:-}" = --base ]; then
    [ -n "${2:-}" ] || { echo "aot --base needs a commit" >&2; exit 2; }
    base_dir="$(mktemp -d)"
    trap 'rm -rf "$base_dir"' EXIT
    git -C "$repo" archive "$2" track_1_short | tar -x -C "$base_dir"
    mounts+=(-v "$base_dir":/base:ro)
    cmd+=(--base /base)
    shift 2
  fi
  set -- "${cmd[@]}" "$@"
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --build-arg TRITON="$ver" --build-arg TORCH="$torch_ver" -t "$image" "$here" >&2
fi
envs=(-e "TRITON_INTERPRET=${TRITON_INTERPRET:-1}" -e PYTHONPATH=/repo/182/tools/triton_cpu -e PYTHONDONTWRITEBYTECODE=1
      -e "PYTEST_ADDOPTS=-p no:cacheprovider ${PYTEST_ADDOPTS:-}")
for v in TRITON_CPU_NO_SHIM TRITON_CPU_NUM_SMS HADAMARD_FP8 HADAMARD_SEED HADAMARD_BLOCK HADAMARD_GEMMS FP8_ATTN_O_BASE_ATTENTION; do
  [ -n "${!v:-}" ] && envs+=(-e "$v=${!v}")
done
tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
docker run --rm "${tty[@]}" "${envs[@]}" "${mounts[@]}" -w /repo "$image" "$@"
