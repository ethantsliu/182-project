#!/usr/bin/env bash
# CPU stand-in for variants/<name>/182/run_node.sh (the node contract: check | prep | run), for test_gpu_session.py.
# FAKE_* variables hold comma lists of variant names (or name:N for "the N-th `run` invocation", name:all);
# state (call log, run counter, the "prepped" marker) lives in $FAKE_STATE/<name>/.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name=$(basename "$here")
cd "$here" || exit 9
st="${FAKE_STATE:?}/$name"
mkdir -p "$st"
has() { [[ ",${1:-}," == *",$2,"* ]]; }
case "${1:-}" in
  check)
    echo "check" >> "$st/calls"
    echo "PASS python: 3.12"
    has "${FAKE_CHECK_WARN:-}" "$name" && echo "WARN vcpu: 12 per GPU, 24 recommended"
    if has "${FAKE_CHECK_CRASH:-}" "$name"; then echo "run_node.sh: line 3: syntax error" >&2; exit 2; fi
    if has "${FAKE_CHECK_FAIL:-}" "$name"; then echo "FAIL ram: 64 GiB available, needs 390 GiB (use a larger host)"; exit 1; fi
    if has "${FAKE_NEEDS_PREP:-}" "$name" && [[ ! -f "$st/prepped" ]]; then
      echo "FAIL data: 0 of 9 shards (fix: bash variants/$name/182/run_node.sh prep)"; exit 1
    fi
    echo "PASS data: shards present";;
  prep)
    echo "prep" >> "$st/calls"
    sleep "${FAKE_PREP_SLEEP:-0}"
    if has "${FAKE_PREP_FAIL:-}" "$name"; then echo "download failed: 401"; exit 5; fi
    touch "$st/prepped"; echo "prep done";;
  run)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      echo "+ cd $here && DATA_PATH=../.. torchrun --standalone --nproc_per_node=${NPROC:-8} train_gpt.py${CANON_MASK:+ [CANON_MASK=$CANON_MASK]}"; exit 0
    fi
    n=$(( $(cat "$st/runs" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$st/runs"
    echo "run NPROC=${NPROC:-} SEED=${SEED:-} LOG_DIR=${LOG_DIR:-}${CANON_MASK:+ CANON_MASK=$CANON_MASK}${CONFIG:+ CONFIG=$CONFIG}" >> "$st/calls"
    if has "${FAKE_RUN_FAIL:-}" "$name:$n" || has "${FAKE_RUN_FAIL:-}" "$name:all"; then
      echo "Traceback (most recent call last):"
      echo "RESULT variant=$name status=failed reason=CUDA out of memory on rank 3"; exit 1
    fi
    if has "${FAKE_RUN_NORESULT:-}" "$name"; then echo "started, then nothing"; exit 0; fi
    sleep "${FAKE_RUN_SLEEP:-0}"
    mkdir -p "${LOG_DIR:?}"; log="$LOG_DIR/run$n.txt"; echo "fake log $n" > "$log"
    case "$name" in
      t2-*) echo "RESULT variant=$name val_loss=2.9171 train_time_s=860.5 steps=5800 target=2.92 hit=1 log=$log";;
      *)    echo "RESULT variant=$name val_loss=3.27${n}0 train_time_s=30.${n}0 steps=1100 target=3.28 hit=$(( n < 8 ? 1 : 0 )) log=$log";;
    esac;;
  *) echo "usage: run_node.sh check|prep|run" >&2; exit 2;;
esac
