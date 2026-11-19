#!/usr/bin/env bash
# Confirm fewer scheduled steps on variants/t1-cpu-ngram: one warm-up run, then AB_RUNS interleaved pairs of the
# default (675) against NUM_SCHEDULED_ITERATIONS=$N (default 668). Run time drifts down over a job's first runs, so
# only the interleaved pairs count for time.
set -uo pipefail
: "${OUT:=/work/out}"; mkdir -p "$OUT"
t() { echo "LOR $(date -u +%T) $*"; }
V=variants/t1-cpu-ngram
t "prep start"; INSTALL_RUST=1 bash $V/182/run_node.sh prep > "$OUT/prep.log" 2>&1; t "prep rc=$?"
export HF_HUB_OFFLINE=1
r=$(SEED=20 LOG_DIR="$OUT/warm" bash $V/182/run_node.sh run 2>&1 | grep '^RESULT' | tail -1); t "WARMUP ${r:0:120}"
t "ab start"; AB_RUNS=${AB_RUNS:-4} AB_SEED0=${AB_SEED0:-3} AB_B="NUM_SCHEDULED_ITERATIONS=${N:-668}" LOG_DIR="$OUT/ab" bash $V/182/run_node.sh ab > "$OUT/ab.log" 2>&1; t "ab rc=$?"
grep -E "^AB" "$OUT/ab.log" | sed -E 's/ target=.*//' | cut -c1-300
t "all done"
