#!/usr/bin/env bash
# (1) More interleaved pairs of 675 vs 668 scheduled steps on variants/t1-cpu-ngram (fresh seeds, after a warm-up
# run), so the time difference has enough pairs. (2) One Track 2 run of variants/t2-cpu-ngram through the session.
set -uo pipefail
: "${OUT:=/work/out}"; mkdir -p "$OUT"
t() { echo "LOR $(date -u +%T) $*"; }
V=variants/t1-cpu-ngram
t "t1 prep start"; INSTALL_RUST=1 bash $V/182/run_node.sh prep > "$OUT/prep-t1.log" 2>&1; t "t1 prep rc=$?"
( t "t2 prep start (background)"; INSTALL_RUST=1 nice -n 19 bash variants/t2-cpu-ngram/182/run_node.sh prep > "$OUT/prep-t2.log" 2>&1; echo $? > "$OUT/prep-t2.rc" ) &
T2PREP=$!
export HF_HUB_OFFLINE=1
r=$(SEED=21 LOG_DIR="$OUT/warm" bash $V/182/run_node.sh run 2>&1 | grep '^RESULT' | tail -1); t "WARMUP ${r:0:120}"
# The Track 2 prep (downloads, a build) must not overlap timed runs.
wait $T2PREP; t "t2 prep rc=$(cat "$OUT/prep-t2.rc" 2>/dev/null)"; grep -E "^FAIL" "$OUT/prep-t2.log" | tail -5
t "ab start"; AB_RUNS=${AB_RUNS:-4} AB_SEED0=${AB_SEED0:-7} AB_A="NUM_SCHEDULED_ITERATIONS=675" AB_B="NUM_SCHEDULED_ITERATIONS=${N:-668}" LOG_DIR="$OUT/ab" bash $V/182/run_node.sh ab > "$OUT/ab.log" 2>&1; t "ab rc=$?"
grep -E "^AB(_RUN| n=| A )" "$OUT/ab.log" | sed -E 's/ target=.*//' | cut -c1-300
t "track 2 start"
SESSION_DIR=$OUT/session T2CPU_RUNS=1 bash 182/gpu_session.sh env t2cpu bundle > "$OUT/session.log" 2>&1; t "session rc=$?"
grep -E "RESULT variant|== .*: (PASS|FAIL|SKIP)|OOM|out of memory" "$OUT/session.log" | cut -c1-400 | tail -15
t "all done"
