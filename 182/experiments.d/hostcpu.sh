# r6-hostcpu: the training loader's host work off the main thread on t1-gpu-stack (variants/t1-gpu-stack/182/hostcpu/README.md).
# Sourced by the session worker's generic xab phase. All three arms are exact (same batches, same model): time only.
XAB_NAME=hostcpu
XAB_TREE=variants/t1-gpu-stack
XAB_ARMS=("base:" "hostprep:HOST_PREP=all" "hostasync:HOST_PREP=all HOST_ASYNC=1")
XAB_SEEDS=4 ; XAB_EST_MIN=65 ; XAB_NOTE="expect ~0 s for both (-0.02..+0.01 s): the graph-replayed host runs ahead of the ~29.4 ms GPU step and the loader is ~0.6-0.9 ms of it; val must match base within seed noise; confirms: paired dt 95% CI below 0 (host was on the critical path after all); kills: CI covering 0 (expected: keep off) or above 0 (GIL contention), any start-up assert, 'HOST_ASYNC=1 depth=4' missing from the [folds] line"
