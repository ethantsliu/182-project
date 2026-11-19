# ANVIL_KERNEL / ANVIL_ABLATE on the root record (182/optim-kernel/README.md). Sourced by the session worker's generic
# xab phase. Run the one-minute microbenchmark first (python 182/optim-kernel/bench.py) and put its best kernel in the
# gram222 arm if another one wins. gram222: the same update rule, mlp_bank's cascade evaluated in Gram space (time
# only; loss equal to base within seed noise). skipmatrix: MEASUREMENT ONLY, no whitening cascade at all: read its
# dtime (the latency the cascade's matrix work exposes, the ceiling for any kernel), never its val or net_s.
XAB_NAME=anvilkernel
XAB_TREE="."
XAB_ARMS=("base:" "gram222:ANVIL_KERNEL=gram222" "skipmatrix:ANVIL_ABLATE=skip_matrix")
XAB_SEEDS=4 ; XAB_EST_MIN=56
XAB_NOTE="gram222: expect -0.02..-0.2 s, val equal to base; confirmed if the paired dtime CI is below 0 with dval inside seed noise, killed by a capture/self-check assert, a val shift beyond noise, or dtime >= 0. skipmatrix is not a training run: only its dtime counts (expect -0.2..-0.9 s if the bank updates are exposed, ~0 if they hide under comms, which would cap every optimizer kernel at ~0)"
