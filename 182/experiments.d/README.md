# 182/experiments.d: A/B experiments for the session's `xab` phase

Each `*.sh` file here declares one paired A/B experiment. `bash 182/gpu_session.sh xab` (part of the full session,
after `base` and `t1gpu`) runs every file it finds, so adding an experiment to the node session means adding one
file here. Nothing else needs to change.

## Format

The file is bash and is sourced. It sets exactly these variables:

```bash
XAB_NAME=<name>            # short id: letters, digits, _ and -; it names the results directory
XAB_TREE=<repo-relative dir whose 182/run_node.sh or root trainer runs it>   # "." for the repo root
XAB_ARMS=("base:" "<arm>:<ENV=VAL ENV2=VAL>" ...)   # name:env pairs; first is the baseline
XAB_SEEDS=4 ; XAB_EST_MIN=<node-minutes estimate> ; XAB_NOTE="<one line: expected effect, what confirms/kills it>"
```

Arm names use letters, digits and `_`. The env part is shell words (`K=V`, and values may be quoted). Example:

```bash
XAB_NAME=moe-cap
XAB_TREE=variants/t1-gpu-stack
XAB_ARMS=("base:" "cap125:MOE_CAP=1.25")
XAB_SEEDS=4 ; XAB_EST_MIN=36 ; XAB_NOTE="expect -0.1 to -0.3 s at equal loss; killed if the Δval CI lies above +1 mnat"
```

`python 182/tools/xab_run.py spec 182/experiments.d/<file>.sh` validates a file on any machine and prints what the
session will read (`XS_OK=1`, or `XS_OK=0` and `XS_ERR=...`).

## What the session does with it

- A variant tree (`XAB_TREE` has `182/run_node.sh`) goes through `182/tools/xab_run.py`. Before any run, the tree's
  `check` must pass. If every FAIL line names `prep`, prep runs first. Otherwise the experiment is skipped, with
  the reason. The runs are then interleaved by seed: for seed 1..`XAB_SEEDS` every arm runs once, and the arm order
  rotates by one each seed. Each run is `SEED=<s> NPROC=8 LOG_DIR=... EXTRA_ENV="<the arm's env>" bash
  <tree>/182/run_node.sh run`. Its result is the last `RESULT` line.
- The root tree (`.`) goes through `182/tools/ab_run.py`. Every arm is the committed HEAD with its env, with the
  same seed list and blocked interleaving that the session's other root A/Bs use.
- Report (`182/results/<session>/xab/<name>/report.txt`, also printed in `summary.txt`): for each arm, the number of
  runs ok out of planned, val_loss and train_time mean ± sd. Then, for each arm against the baseline, paired by seed:
  Δval in millinats ± SE with its 95% t-CI, Δtime ± SE with its CI, and net seconds = Δtime + Δval x 164 ms/mnat,
  also given at both ends of the 120-250 ms/mnat band. A + sign means the arm is worse or slower.
- It is resumable. A finished run is never repeated, and a failed run is set aside and retried on the next invocation.
  An arm whose first two runs both failed is abandoned. `FORCE=1` re-plans an experiment from scratch and keeps the
  old directory as `<name>.attemptN`.
- An invalid file is reported (`xab/<file>.invalid`) and fails the phase, but the other files still run.
  `XAB_ONLY=a,b` / `XAB_SKIP=c` select experiments by `XAB_NAME`. `QUICK=1` and `SKIP_PHASES=xab` skip the phase.
  `XAB_TIMEOUT_S` (1500) caps each run.

## Writing one

- Every arm's variables must be flags that are off by default in that tree. The baseline arm should normally be
  `"base:"`, the tree exactly as it is.
- `XAB_EST_MIN` is the node-minutes estimate shown in the session plan. A rough rule: (number of arms x seeds) x
  the minutes per warm run (about 4 for the Track 1 trees), plus one cold compile per arm whose flag changes
  compiled code.
- In `XAB_NOTE`, state the expected Δtime and what would kill the idea. The run-to-run spread on this node has not
  been measured yet. The ANVIL2 record logs show a val_loss sd of about 1 mnat between runs. With 4 pairs, the
  95% t-CI half-width is 3.18 x sd(diff) / 2, so expect several tenths of a millinat at best. An effect smaller
  than that needs more seeds (`XAB_SEEDS`, up to 32).
- Each worker adds only its own file. This README is the only file here that the session tooling owns.

Nothing here has run on a GPU.
