#!/usr/bin/env bash
# Fewer scheduled steps on variants/t1-cpu-ngram (upstream 675): how many can go before the 3.28 target is at risk?
# STEPS="668 662" SEEDS="0 1 2"; seeds match the base arm of jobs/ngram_ab.sh (AB_SEED0=0), so losses pair by seed.
set -uo pipefail
: "${OUT:=/work/out}"; mkdir -p "$OUT"
t() { echo "LOR $(date -u +%T) $*"; }
V=variants/t1-cpu-ngram
t "prep start"; INSTALL_RUST=1 bash $V/182/run_node.sh prep > "$OUT/prep.log" 2>&1; t "prep rc=$?"
export HF_HUB_OFFLINE=1
one() { tag=$1; seed=$2; extra=$3
  r=$(SEED=$seed EXTRA_ENV="$extra" LOG_DIR="$OUT/runs" bash $V/182/run_node.sh run 2>&1 | tee -a "$OUT/runs.log" | grep '^RESULT' | tail -1)
  t "STEPRUN tag=$tag seed=$seed ${r:0:140}"; }
one base-cold ${COLD_SEED:-3} "NUM_SCHEDULED_ITERATIONS=675"          # the first run on a fresh node is the slow one; also one more base sample
for n in ${STEPS:-668 662}; do for s in ${SEEDS:-0 1 2}; do one "sched$n" "$s" "NUM_SCHEDULED_ITERATIONS=$n"; done; done
t "all done"
