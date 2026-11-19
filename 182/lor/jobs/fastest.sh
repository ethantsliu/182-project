#!/usr/bin/env bash
# One batch job: the two fast Track 1 trees (t1-gpu-stack, t1-cpu-ngram), a root profile, then the last root seeds of
# the ANVIL kernel A/B. Run by 182/lor/entry.sh from the repo root inside the venv. Keeps going past a failed part.
set -uo pipefail
: "${OUT:=/work/out}"; mkdir -p "$OUT"
t() { echo "LOR $(date -u +%T) $*"; }
export INSTALL_RUST=1
t "prep start (online: data shards, rust build)"
( bash variants/t1-gpu-stack/182/run_node.sh prep; echo "t1gpu prep rc=$?"; bash variants/t1-cpu-ngram/182/run_node.sh prep; echo "t1cpu prep rc=$?" ) > "$OUT/prep.log" 2>&1
t "prep done"; grep -E "prep rc=|FAIL|WARN" "$OUT/prep.log" | tail -20
# The kernels hub API answers 401 without a login; the FA3 build is already in the cache node_setup.sh seeded.
export HF_HUB_OFFLINE=1
export SESSION_DIR=$OUT/session T1GPU_RUNS=${T1GPU_RUNS:-2} T1CPU_RUNS=${T1CPU_RUNS:-3} PROFILE_TREES=${PROFILE_TREES:-root}
t "session start"
bash 182/gpu_session.sh env sanity t1gpu t1cpu profile bundle > "$OUT/session.log" 2>&1; t "session rc=$?"
grep -E "^RESULT|RESULT variant|== .*: (PASS|FAIL|SKIP)|STACK PROBLEM" "$OUT/session.log" | cut -c1-400 | tail -40
find "$SESSION_DIR" -name '*.txt' -path '*profile*' | head -3 | while read -r f; do echo "LOR profile $f"; head -60 "$f"; done
run() { n=$1; seed=$2; shift 2; t "$n start"
  env TRAIN_LOG_DIR="$OUT/root/$n" RUN_METRICS=1 TRAIN_SEED="$seed" "$@" torchrun --standalone --nproc_per_node=8 train_gpt.py > "$OUT/root-$n.log" 2>&1
  t "$n rc=$? $(grep -oE 'train_time_s=[0-9.]+ val_loss=[0-9.]+ final=1' "$OUT/root-$n.log" | tail -1)"; }
mkdir -p "$OUT/root"
run gram222s-s4 4 ANVIL_KERNEL=gram222s
run skipmatrix-s4 4 ANVIL_ABLATE=skip_matrix
t "all done"
