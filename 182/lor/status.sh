#!/usr/bin/env bash
# What is holding (and billing) GPUs right now, and what has been spent. `182/lor/status.sh stop` also stops all of it.
set -uo pipefail
LOR=${LOR:-$HOME/.lor/bin/lor}; cd "$HOME"
"$LOR" compute list 2>/dev/null > /tmp/lor_jobs.$$; "$LOR" compute instance list 2>/dev/null > /tmp/lor_inst.$$
python3 - /tmp/lor_jobs.$$ /tmp/lor_inst.$$ "${1:-}" "$LOR" <<'PY'
import json, subprocess, sys, uuid
jobs = json.load(open(sys.argv[1])).get("jobs", []); inst = json.load(open(sys.argv[2])).get("gpu_instances", [])
done = ("SUCCEEDED", "FAILED", "CANCELLED"); live = 0
for kind, rows, key, flag in (("job", jobs, "job_id", "--job-id"), ("instance", inst, "gpu_instance_id", "--gpu-instance-id")):
    for r in rows:
        st = r["state"].rsplit("_", 1)[-1]
        if st in done: continue
        live += 1
        print(f"LIVE {kind} {r[key]} {r.get('name','')} {st} spend=${r.get('spend',{}).get('amount','?')}")
        if sys.argv[3] == "stop":
            cmd = [sys.argv[4], "compute"] + (["instance"] if kind == "instance" else []) + ["stop", "--request-key", str(uuid.uuid4()), flag, r[key]]
            print("  stop:", subprocess.run(cmd, capture_output=True, text=True).returncode)
print("nothing is holding GPUs" if not live else f"{live} run(s) still live")
PY
rm -f /tmp/lor_jobs.$$ /tmp/lor_inst.$$
"$LOR" compute usage --start-time "$(date -u -v-30d +%Y-%m-%dT00:00:00Z 2>/dev/null || date -u -d '30 days ago' +%Y-%m-%dT00:00:00Z)" --end-time "$(date -u -v+1d +%Y-%m-%dT00:00:00Z 2>/dev/null || date -u -d tomorrow +%Y-%m-%dT00:00:00Z)" --group-by day 2>/dev/null \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("org_total",{}); print("spent, last 30 days: $%s (%s GPU-hours)" % (t.get("amount","?"), t.get("gpu_hours","?")))'
