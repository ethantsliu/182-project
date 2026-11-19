#!/usr/bin/env bash
# Iteration on the fastest tree: variants/t1-cpu-ngram, base vs AB_B (default TABLE_VOTE=1), AB_RUNS interleaved pairs.
set -uo pipefail
: "${OUT:=/work/out}"; mkdir -p "$OUT"
t() { echo "LOR $(date -u +%T) $*"; }
V=variants/t1-cpu-ngram
t "prep start"; INSTALL_RUST=1 bash $V/182/run_node.sh prep > "$OUT/prep.log" 2>&1; t "prep rc=$?"; grep -E "^FAIL|^WARN" "$OUT/prep.log" | tail -8
export HF_HUB_OFFLINE=1   # the kernels hub API answers 401 without a login; the cache is seeded
t "ab start"; AB_RUNS=${AB_RUNS:-3} LOG_DIR="$OUT/ab" bash $V/182/run_node.sh ab > "$OUT/ab.log" 2>&1; t "ab rc=$?"
grep -E "RESULT|^AB|delta|mean|pair" "$OUT/ab.log" | cut -c1-400 | tail -40
t "all done"
