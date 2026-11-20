# 182/lor: rent an 8xH100 node per experiment

One iteration = one batch job on Lor (Mirendil). The job claims a node, sets it up, runs one command, prints its
results into the log and exits; the node is released (and billing stops) when the command exits.

```bash
182/lor/submit.sh NAME EST_MIN MAX_MIN '<command, run from the repo root on the node>'   # claim, run, collect, release
182/lor/status.sh          # what is holding GPUs right now, and the spend so far
182/lor/status.sh stop     # stop everything that is still live
```

- Needs once: `~/.lor/bin/lor login --server https://lor.mirendil.dev`, and the Lor secret `gh-182-deploy` (a
  read-only deploy key for this repo; the job clones the pushed commit, so push before submitting).
- `MAX_MIN` is a hard stop. Cost is $16/h for the node on the preemptible tier (`TIER=nonpreemptible`: $32/h).
- Results: `~/.lor-182/jobs/NAME-<time>/log.txt` and `out/` (files under 1.5 MB that the command left in `$OUT`).
- A fresh node costs about 12 min of waiting (not billed), then about 3 min of setup and about 10 min of cold
  compile on the first run of a tree; warm runs are about 4 min. Put several runs in one job.
- `jobs/fastest.sh` is an example job. Trainers must run with `HF_HUB_OFFLINE=1` after setup (the kernels hub API
  answers 401 without a login; `node_setup.sh` seeds the cache).
- `variants/t1-cpu-ngram` and `variants/t2-cpu-ngram` each install their own build of the `exact_match` extension
  into the same venv. Whichever `prep` ran last wins, and the other tree's check then FAILs ("an older build"). Run
  one tree's prep and its runs before the other tree's prep (`jobs/ngram_final.sh` got this wrong on 2026-11-20:
  its Track 1 pairs did not run).
