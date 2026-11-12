# r6-optim: the ANVIL cascade's step cost on t1-gpu-stack (variants/t1-gpu-stack/182/optim-cost/README.md).
# Both non-base arms are exact (no model change): optprof prints per-bank [optim-profile] wait/update times (diagnostic,
# read its log lines, not its wall); optmoefused moves the expert bank onto the fused ANVIL tail (time only).
XAB_NAME=optim
XAB_TREE=variants/t1-gpu-stack
XAB_ARMS=("base:" "optprof:OPTIM_PROFILE=1" "optmoefused:OPTIM_MOEFUSED=1")
XAB_SEEDS=4 ; XAB_EST_MIN=60 ; XAB_NOTE="optmoefused: expect -0.03..-0.08 s, val equal to base; confirmed if the paired 95% CI is below 0, killed by a startup assert or val drift. optprof: moe_bank/mlp_bank u= vs w= decides whether OPTIM_MAPS=3 is worth a sweep"
