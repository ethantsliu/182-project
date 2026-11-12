# r6-kernel: MOE_ROUTER_FUSED on t1-gpu-stack (variants/t1-gpu-stack/182/kernel-cost/README.md). Sourced by the session
# worker's generic xab phase. Run the 2-minute microbenchmark first (bench_router.py); skip this A/B if it projects < 20 ms.
XAB_NAME=kernel
XAB_TREE=variants/t1-gpu-stack
XAB_ARMS=("base:" "router:MOE_ROUTER_FUSED=1")
XAB_SEEDS=4 ; XAB_EST_MIN=45 ; XAB_NOTE="fused MoE router fwd/bwd kernels, same math; expect -0.25..+0.03 s (centre -0.1 s), loss unchanged; 4 seeds cannot resolve 0.1 s, so bench_router.py's projection decides; this A/B confirms no start-up/capture failure, '[moe] MOE_ROUTER_FUSED=1' in the log and no loss shift; kills: bench projecting < 20 ms, a paired dt CI above 0, or a loss shift beyond seed noise"
