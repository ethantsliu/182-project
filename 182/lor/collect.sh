#!/usr/bin/env bash
# Follow a submitted job to its end, save its log, unpack the results tarball. 182/lor/collect.sh ~/.lor-182/jobs/<dir>
set -uo pipefail
D=$1; LOR=${LOR:-$HOME/.lor/bin/lor}; JOB=$(cat "$D/job_id"); cd "$HOME"
state() { "$LOR" compute get --job-id "$JOB" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("job", d)["state"].rsplit("_",1)[-1])' 2>/dev/null; }
while :; do s=$(state); case "$s" in RUNNING|SUCCEEDED|FAILED|CANCELLED) break;; esac; sleep 20; done
"$LOR" compute logs --job-id "$JOB" --follow > "$D/log.txt" 2>&1 || true
while :; do s=$(state); case "$s" in SUCCEEDED|FAILED|CANCELLED) break;; esac; sleep 10; done
"$LOR" compute logs --job-id "$JOB" 2>&1 | sed -E $'s/^\x1b\\[[0-9;]*m\\([^)]*\\)\x1b\\[0m //' | tr -d '\r' > "$D/log.txt" || true   # drop the "(job, pid=N) " prefix
sed -n '/^LOR-RESULTS-BEGIN/,/^LOR-RESULTS-END/p' "$D/log.txt" | sed '1d;$d' | tr -d '\r' | base64 -d 2>/dev/null | tar -C "$D" -xzf - 2>/dev/null && echo "results: $D/out"
"$LOR" compute get --job-id "$JOB" > "$D/final.json" 2>/dev/null
echo "job $JOB ended: $s"; grep -E "^LOR |RESULT|val_loss" "$D/log.txt" | tail -20
