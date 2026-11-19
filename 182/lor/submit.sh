#!/usr/bin/env bash
# Claim an 8xH100 Lor node, run one command, release the node. From a laptop with `lor login` done.
#   182/lor/submit.sh NAME EST_MIN MAX_MIN '<command run from the repo root on the node>'
# The job clones REF (default: the pushed HEAD sha) with the read-only deploy key saved as the Lor secret
# `gh-182-deploy`, runs 182/lor/entry.sh, and exits; Lor bills the node only while the job holds it. MAX_MIN is a
# hard stop. Logs and the results tarball land in ~/.lor-182/jobs/NAME-<time>/.  TIER=preemptible ($16/h) by default.
set -euo pipefail
NAME=$1; EST=$2; MAX=$3; CMD=$4
LOR=${LOR:-$HOME/.lor/bin/lor}; TIER=${TIER:-preemptible}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF=${REF:-$(git -C "$REPO" rev-parse HEAD)}
git -C "$REPO" fetch -q origin && git -C "$REPO" merge-base --is-ancestor "$REF" origin/main || { echo "REF $REF is not pushed to origin/main" >&2; exit 2; }
D=$HOME/.lor-182/jobs/$NAME-$(date +%m%d-%H%M%S); mkdir -p "$D"
IMG=${IMG:-docker.io/nvidia/cuda@sha256:ef8a68702fdc93f3acb12919cde4468b9df0aeea214c1c97bf57598dad558967}
BOOT='set -uo pipefail; export DEBIAN_FRONTEND=noninteractive
sed -i "s|http://archive.ubuntu.com|https://archive.ubuntu.com|; s|http://security.ubuntu.com|https://security.ubuntu.com|" /etc/apt/sources.list.d/*.sources
apt-get -o Acquire::https::Timeout=20 update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates openssh-client python3.12 python3.12-venv python3.12-dev python3-pip build-essential pciutils numactl >/dev/null
mkdir -p ~/.ssh /work && cp /secrets/deploy_key ~/.ssh/id_182 && chmod 600 ~/.ssh/id_182 && ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
export GIT_SSH_COMMAND="ssh -i ~/.ssh/id_182 -o IdentitiesOnly=yes"
cd /work && git clone -q git@github.com:ethantsliu/182-project.git && cd 182-project && git checkout -q "$LOR_REF" && exec bash 182/lor/entry.sh "$LOR_CMD"'
python3 - "$D" "$IMG" "$BOOT" "$REF" "$CMD" <<'PY'
import json, sys
d, img, boot, ref, cmd = sys.argv[1:]
json.dump({"imageDigest": img, "command": ["bash", "-c", boot],
           "env": [{"name": "LOR_REF", "value": ref}, {"name": "LOR_CMD", "value": cmd}],
           "secretRefs": [{"secret": "gh-182-deploy", "path": "/secrets/deploy_key"}]}, open(d + "/spec.json", "w"))
json.dump({"workers": 1, "accelerator": {"kind": "COMPUTE_ACCELERATOR_KIND_GPU", "model": "H100", "countPerWorker": 8}},
          open(d + "/resources.json", "w"))
open(d + "/pod.yaml", "w").write('metadata:\n  labels:\n    lor.compute/internet: "true"\nspec:\n  volumes:\n    - name: dshm\n      emptyDir:\n        medium: Memory\n  containers:\n    - name: ray-node\n      volumeMounts:\n        - name: dshm\n          mountPath: /dev/shm\n')
PY
cd "$HOME"
"$LOR" compute create --request-key "$(uuidgen)" --name "$NAME" --run-description "${DESC:-nanogpt speedrun experiment}" --tier "$TIER" \
  --estimated-runtime-seconds $((EST * 60)) --max-runtime-seconds $((MAX * 60)) ${CLUSTER:+--cluster "$CLUSTER"} \
  --resources "@$D/resources.json" --spec "@$D/spec.json" --pod-config "$D/pod.yaml" > "$D/create.json" 2> "$D/create.err"
JOB=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job"]["job_id"])' "$D/create.json"); echo "$JOB" > "$D/job_id"
echo "job $JOB  dir $D  (stop: $LOR compute stop --request-key \$(uuidgen) --job-id $JOB)"
[[ "${NOWAIT:-0}" == 1 ]] && exit 0
exec bash "$REPO/182/lor/collect.sh" "$D"
