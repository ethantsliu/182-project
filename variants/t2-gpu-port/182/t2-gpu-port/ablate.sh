#!/usr/bin/env bash
# A/B runner for the Track 2 GPU port. Run from the repo root on an 8xH100 node with the
# fineweb10B shards in data/ (python data/cached_fineweb10B.py 25; medium trains on ~1.9B tokens).
#
#   182/t2-gpu-port/ablate.sh <arm> <n_runs>      e.g.  182/t2-gpu-port/ablate.sh base 3
#   python 182/t2-gpu-port/summarize.py runs/      # mean/std/p per arm
#
# Each arm is a set of env overrides (defaults == PR #261). Logs land in runs/<arm>/<i>.out; the
# trainer's own log path (logs/<uuid>.txt) is printed on the first line of each .out file.
set -euo pipefail
arm=${1:?arm name}; n=${2:-1}

case "$arm" in
  base)      envs="" ;;
  cdt)       envs="COORD_DESCENT_TUNING=1" ;;
  canon)     envs="CANON_MASK=1 VAL_CE_CHUNKS=8" ;;
  prefix)    envs="PREFIX_WEIGHT=0.25" ;;
  prefix_s)  envs="PREFIX_WEIGHT=0.25 PREFIX_HOLD_FRAC=0.10 PREFIX_END_FRAC=0.25" ;;  # medium's MTP ends at 1/6
  xsa)       envs="XSA=1" ;;
  xsa_nog)   envs="XSA=1 ATTN_GATE=0" ;;
  stack)     envs="COORD_DESCENT_TUNING=1 CANON_MASK=1 VAL_CE_CHUNKS=8 PREFIX_WEIGHT=0.25 XSA=1" ;;
  # step retune of the stack: pass NUM_SCHEDULED_ITERATIONS / NUM_EXTENSION_ITERATIONS in the environment
  stack_steps) envs="COORD_DESCENT_TUNING=1 CANON_MASK=1 VAL_CE_CHUNKS=8 PREFIX_WEIGHT=0.25 XSA=1 NUM_SCHEDULED_ITERATIONS=${NUM_SCHEDULED_ITERATIONS:-4065}" ;;
  *) echo "unknown arm $arm"; exit 1 ;;
esac

mkdir -p "runs/$arm"
for i in $(seq 1 "$n"); do
  out="runs/$arm/$(date +%Y%m%d-%H%M%S)-$i.out"
  echo "[$arm $i/$n] $envs -> $out"
  # VAL_LOSS_EVERY=0 skips the intermediate validations (they are off the clock but cost ~wall time);
  # override with VAL_LOSS_EVERY=250 for loss curves.
  env VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" $envs torchrun --standalone --nproc_per_node=8 train_gpt_medium.py > "$out" 2>&1
  grep -E "^step:[0-9]+/[0-9]+ val_loss" "$out" | tail -1
done
