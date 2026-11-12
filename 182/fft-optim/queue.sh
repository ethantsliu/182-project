#!/usr/bin/env bash
# Sequential queue on this 8xH100 node: the three broken nanoGPT variant runs (fixed), then the FFT-optimizer
# experiments in the plan's order 2 -> 1 -> 3 -> 4. Each step logs to runs/queue/<step>.log; one failing step
# does not stop the queue. FFT jobs cache their results, so re-running this script resumes.
#   tmux attach -t fftq       # watch;   cat runs/queue/status.tsv   # progress
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$HOME/182-project
Q=$HERE/runs/queue
mkdir -p "$Q"
STATUS=$Q/status.tsv
[ -f "$STATUS" ] || printf "step\tstart\tend\texit\n" > "$STATUS"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$HOME/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$HOME/venv/lib/python3.12/site-packages/torch/lib
PY=$HOME/venv/bin/python

step() {  # step <name> <command...>
  local name=$1; shift
  if grep -qP "^$name\t.*\t0$" "$STATUS"; then echo "[$name] already done, skipping"; return; fi
  local t0; t0=$(date -u +%FT%TZ)
  echo "===== [$name] start $t0"
  "$@" > "$Q/$name.log" 2>&1
  local rc=$?
  printf "%s\t%s\t%s\t%s\n" "$name" "$t0" "$(date -u +%FT%TZ)" "$rc" >> "$STATUS"
  echo "===== [$name] exit $rc  ($(date -u +%T)); tail:"
  tail -5 "$Q/$name.log"
}

# Same container setup as the t1-gpu-stack run that worked, plus --ipc=host (host /dev/shm: the n-gram
# tables need ~144 GiB; docker's default is 64 MB) and the venv on PATH (ablate.sh calls bare torchrun).
drun() {  # drun <name> <workdir> <extra -e args...> -- <bash command>
  local name=$1 wd=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done; shift
  docker rm -f "$name" >/dev/null 2>&1
  timeout 3h docker run --rm --name "$name" --gpus all --ipc=host \
    -v /home/dev_patel:/home/dev_patel -v /mnt/work:/mnt/work \
    -v /usr/local/cuda:/usr/local/cuda:ro -v /usr/local/cuda-13.0:/usr/local/cuda-13.0:ro \
    -w "$wd" -e HOME=/home/dev_patel -e USER=dev_patel -e LOGNAME=dev_patel -e CUDA_HOME=/usr/local/cuda \
    -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH" -e HF_HUB_OFFLINE=1 -e DATA_PATH=../.. \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PATH=/home/dev_patel/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    "${envs[@]}" --entrypoint bash nanogpt-run:gcc -lc "$*"
  local rc=$?
  docker rm -f "$name" >/dev/null 2>&1  # if timeout killed the client, stop the container too
  return $rc
}

# ---- fixed nanoGPT variant runs ----------------------------------------------------------------------
# t2-gpu-port: Track 2 trainer is train_gpt_medium.py (the earlier attempt ran the stale Track-1 train_gpt.py,
# 1480 steps, and OOMed). ablate.sh base 1 = PR #261 defaults; README expects val ~2.917 in ~860 s.
step t2-gpu-port drun nanogpt-t2-gpu "$REPO/variants/t2-gpu-port" -- \
  'python 182/t2-gpu-port/test_cpu.py | tail -3; 182/t2-gpu-port/ablate.sh base 1; f=$(ls -t runs/base/*.out | head -1); echo "OUT=$f"; tail -15 "$f"'

# t1-cpu-ngram: README leg A (= #381 behaviour). exact_match (t1 build) is already in ~/venv.
step t1-cpu-ngram drun nanogpt-t1-cpu "$REPO/variants/t1-cpu-ngram" -- \
  'python -c "import exact_match; print(exact_match.__file__)"; python3 182/t1-cpu-ngram/node_check.py --data ../.. || exit 3;
   rm -f /dev/shm/exact-match-*; trap "rm -f /dev/shm/exact-match-*" EXIT;
   TRAIN_SEED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py'

# t2-cpu-ngram: its own exact_match build (different Rust source), isolated on PYTHONPATH.
step t2-cpu-ngram drun nanogpt-t2-cpu "$REPO/variants/t2-cpu-ngram" PYTHONPATH=/mnt/work/dev_patel/pydeps/t2-exact-match -- \
  'python -c "import exact_match; print(exact_match.__file__)"; python -m track_1_short.memory_budget | tail -5;
   rm -f /dev/shm/exact-match-*; trap "rm -f /dev/shm/exact-match-*" EXIT;
   torchrun --standalone --nproc_per_node=8 train_gpt_medium_cpu.py'

# ---- FFT-optimizer experiments (host venv; 8 GPUs as a job pool) ----------------------------------------
cd "$HERE"
step prep "$PY" prep_data.py cifar pde
step exp2 "$PY" exp2_longconv.py all
step exp1 "$PY" exp1_conv.py all
step exp3 "$PY" exp3_fno.py all
step exp4 "$PY" exp4_dct.py main

# ---- one summary ----------------------------------------------------------------------------------------
{
  echo "# Queue results ($(date -u +%F))"; echo
  echo '## nanoGPT variant runs'; echo; echo '```'
  for s in t2-gpu-port t1-cpu-ngram t2-cpu-ngram; do
    echo "--- $s"; grep -E "^\[final\]|endpoint_result|val_loss:[0-9.]+ train_time|OutOfMemory|Error|FAIL" "$Q/$s.log" | tail -4
  done
  echo '```'; echo
  for f in runs/exp2_kill/summary.md runs/exp2_main/summary.md runs/exp1_kill/summary.md runs/exp1_main/summary.md \
           runs/exp3_kill/summary.md runs/exp3_main/summary.md runs/exp4/summary.md; do
    [ -f "$f" ] && { cat "$f"; echo; }
  done
  echo '## Queue status'; echo '```'; cat "$STATUS"; echo '```'
} > runs/SUMMARY.md
echo "QUEUE DONE -> $HERE/runs/SUMMARY.md"
