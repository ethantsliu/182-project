#!/usr/bin/env bash
# Rebuild this tree's exact_match extension with EXACT_BACKEND=sa's suffix arrays (cargo feature sa-openmp: libsais
# with OpenMP, needs libgomp; falls back to feature sa, a single-threaded suffix sort), into the active python, and
# stamp it as prep does so that `run_node.sh check` still says "built from this source". Off the clock. The default
# trainer path does not change with this build (the module only gains the SuffixCounts class). Undo: delete the
# module's SOURCE_SHA256 and run `run_node.sh prep` (it reinstalls the default build).
#   bash variants/t1-cpu-ngram/182/sufarr/build.sh          # PYTHON=... picks the interpreter
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VDIR="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3}"
[ -d "$HOME/.cargo/bin" ] && export PATH="$HOME/.cargo/bin:$PATH"
need="1.85"  # libsais-sys 0.2.0 is edition 2024
have="$(rustc --version 2>/dev/null | sed -E 's/^rustc ([0-9]+\.[0-9]+).*/\1/')"
if [ -z "$have" ] || [ "$(printf '%s\n%s\n' "$need" "$have" | sort -V | head -1)" != "$need" ]; then
  echo "FAIL rustc ${have:-missing}: the sa feature needs >= $need (rustup update stable, or INSTALL_RUST=1 run_node.sh prep)" >&2
  exit 1
fi
cd "$VDIR"
for feat in sa-openmp sa; do
  echo "building exact_match with --features $feat"
  if MATURIN_PEP517_ARGS="--features $feat" "$PY" -m pip install --force-reinstall --no-deps ./exact_match; then
    built=$feat
    break
  fi
done
[ -n "${built:-}" ] || { echo "FAIL: the extension did not build with the sa feature" >&2; exit 1; }
module="$(cd / && "$PY" -c 'import exact_match, os; assert hasattr(exact_match, "SuffixCounts"); print(os.path.dirname(os.path.abspath(exact_match.__file__)))')"
"$PY" -c "import sys; sys.path.insert(0, '$VDIR/182'); import node; print(node.source_hash())" > "$module/SOURCE_SHA256"
echo "OK exact_match with --features $built in $module (stamped $(cut -c1-12 "$module/SOURCE_SHA256"))"
