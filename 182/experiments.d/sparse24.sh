# SPARSE24_MLP (182/sparse24/README.md): 2:4 transposable-mask MLP, bf16 cuSPARSELt, vs the record's fp8 dense MLP.
# Sourced by the session worker's generic xab phase. Arms: the record, the bf16 dense control (separates the
# numeric format from sparsity), the sparse arm (sparse throughout, the 2:4 model is the one evaluated).
XAB_NAME=sparse24
XAB_TREE="."
XAB_ARMS=("base:" "bf16mlp:FP8_OFF=mlp" "sparse:SPARSE24_MLP=1")
XAB_SEEDS=4 ; XAB_EST_MIN=63
XAB_NOTE="cost model: sparse arm +1.6..+11.9 s (centre +6) and +10..+40 mnat, i.e. cannot pay; run 182/sparse24/gpu_bench.py first (a negative projected delta_s would falsify the model); confirm = sparse net_s<0 beyond CI, kill = time>=base or val(sparse)-val(bf16mlp) > 10 mnat"
