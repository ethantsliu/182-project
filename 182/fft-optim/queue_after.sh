#!/usr/bin/env bash
# Runs after queue.sh finishes (editing queue.sh while bash executes it is unsafe). Retries the three nanoGPT
# variant runs that failed in queue.sh, each for a diagnosed cause:
#  * t1-/t2-cpu-ngram: Triton "Loop-carried variable m_i fp32 -> fp64" in _copy_fwd_kernel. On torch 2.11, Inductor
#    passes the Python-float `scale` of the traced launch as fp64. Fixed in track_1_short/cplm_copy.py with
#    tl.cast(scale, tl.float32); compiled == eager checked on GPU.
#  * t2-gpu-port: OOM (>72 GiB) in the compile warmup on torch 2.11 (#261 peaked at 57 GiB on 2.10-dev). Retried with
#    selective activation recompute, ACT_MEM_BUDGET=0.6, then 0.4 (new env knob in train_gpt_medium.py, default off).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
Q=$HERE/runs/queue
while ps -eo args | grep -q "^bash $HERE/queue.sh"; do sleep 60; done
source <(sed -n '/^export LD_LIBRARY_PATH/p; /^drun()/,/^}/p' "$HERE/queue.sh")
REPO=$HOME/182-project

record() {  # record <name> <start> <rc> <log>
  printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$(date -u +%FT%TZ)" "$3" >> "$Q/status.tsv"
  { echo; echo "## retry: $1 (exit $3)"; echo '```'
    grep -E "^\[final\]|endpoint_result|^step:[0-9]+/[0-9]+ val_loss|peak memory|OutOfMemory|Error" "$4" | tail -6; echo '```'; } >> "$HERE/runs/SUMMARY.md"
}

t0=$(date -u +%FT%TZ)
drun nanogpt-t1-cpu "$REPO/variants/t1-cpu-ngram" -- \
  'python3 182/t1-cpu-ngram/node_check.py --data ../.. || exit 3; rm -f /dev/shm/exact-match-*; trap "rm -f /dev/shm/exact-match-*" EXIT;
   TRAIN_SEED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py' > "$Q/t1-cpu-ngram-retry.log" 2>&1
record t1-cpu-ngram-retry "$t0" $? "$Q/t1-cpu-ngram-retry.log"

t0=$(date -u +%FT%TZ)
drun nanogpt-t2-cpu "$REPO/variants/t2-cpu-ngram" PYTHONPATH=/mnt/work/dev_patel/pydeps/t2-exact-match -- \
  'python -c "import exact_match; print(exact_match.__file__)"; rm -f /dev/shm/exact-match-*; trap "rm -f /dev/shm/exact-match-*" EXIT;
   torchrun --standalone --nproc_per_node=8 train_gpt_medium_cpu.py' > "$Q/t2-cpu-ngram-retry.log" 2>&1
record t2-cpu-ngram-retry "$t0" $? "$Q/t2-cpu-ngram-retry.log"

printf "t2-gpu-port\t(first attempt)\t\tOOM in compile warmup; ablate.sh's exit 0 masked it\n" >> "$Q/status.tsv"
for b in 0.6 0.4; do
  t0=$(date -u +%FT%TZ)
  log=$Q/t2-gpu-port-amb$b.log
  drun nanogpt-t2-gpu "$REPO/variants/t2-gpu-port" ACT_MEM_BUDGET=$b -- \
    '182/t2-gpu-port/ablate.sh base 1; f=$(ls -t runs/base/*.out | head -1); echo "OUT=$f"; grep -E "^step:[0-9]+/[0-9]+ val_loss|peak memory" "$f" | tail -3;
     grep -q OutOfMemoryError "$f" && exit 2; grep -qE "^step:([0-9]+)/\1 val_loss" "$f" || exit 3' > "$log" 2>&1
  rc=$?
  record "t2-gpu-port ACT_MEM_BUDGET=$b" "$t0" $rc "$log"
  [ $rc -eq 0 ] && break
done
echo "AFTER-QUEUE DONE"
