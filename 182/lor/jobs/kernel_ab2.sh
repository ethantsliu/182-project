#!/usr/bin/env bash
# One batch job on the root record: two loss-curve runs (base, gram222s; VAL_CURVE), then timing runs of base,
# gram222s and gram33s (the Polar Express appendix's restart-every-3) with the arm order rotated per seed.
# Run by 182/lor/entry.sh from the repo root inside the venv. Time-aware: a run is skipped if it would pass DEADLINE_S.
set -uo pipefail
: "${OUT:=/work/out}"; mkdir -p "$OUT/root"
T0=$(date +%s); DEADLINE_S=${DEADLINE_S:-2280}; RUN_S=${RUN_S:-270}
t() { echo "LOR $(date -u +%T) $*"; }
export HF_HUB_OFFLINE=1   # the kernels hub API answers 401 without a login; node_setup.sh seeded the cache
run() { n=$1; seed=$2; shift 2
  el=$(( $(date +%s) - T0 ))
  if (( el + RUN_S > DEADLINE_S )); then t "skip $n (deadline: ${el} s elapsed)"; return; fi
  t "$n start"
  env TRAIN_LOG_DIR="$OUT/root/$n" RUN_METRICS=1 TRAIN_SEED="$seed" "$@" torchrun --standalone --nproc_per_node=8 train_gpt.py > "$OUT/root-$n.log" 2>&1
  t "$n rc=$? $(grep -oE 'train_time_s=[0-9.]+ val_loss=[0-9.]+ final=1' "$OUT/root-$n.log" | tail -1) $(grep -oE 'METRIC clock_split .*' "$OUT/root-$n.log" | cut -c14-90)"; }
C=${CURVE_EVERY:-150}
run curve-base-s5 5 VAL_CURVE=$C                              # also the node's warm-up run
run curve-gram222s-s5 5 VAL_CURVE=$C ANVIL_KERNEL=gram222s
run gram222s-s6 6 ANVIL_KERNEL=gram222s
run base-s6 6
run gram33s-s6 6 ANVIL_KERNEL=gram33s
run base-s7 7
run gram222s-s7 7 ANVIL_KERNEL=gram222s
run gram33s-s7 7 ANVIL_KERNEL=gram33s
t "all done"
