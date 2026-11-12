"""Static per-op cost model of one t1-gpu-stack (#397) training step on one H100 SXM rank, from the code's shapes.

    python variants/t1-gpu-stack/182/kernel-cost/cost_model.py                 # stage-2 step table, calibration, run totals
    python variants/t1-gpu-stack/182/kernel-cost/cost_model.py --regime taper --top 25
    python variants/t1-gpu-stack/182/kernel-cost/cost_model.py --router-fused  # with MOE_ROUTER_FUSED's two kernels
    python variants/t1-gpu-stack/182/kernel-cost/cost_model.py --csv ops.csv   # every op of the chosen regime

Stdlib only (it reads track_1_short/config.py for the stage table, which imports no torch).

What it is: every GEMM, attention call, MoE dispatch/combine, quantize / elementwise pass, lookup, optimizer piece and
collective of one step, with FLOPs and HBM bytes taken from the shapes in model/gpt.py, model/moe.py, model/attention.py
and perf/kernels/*.py, and a roofline time on H100 SXM (989 TF/s bf16 dense, 1979 TF/s fp8, 3.35 TB/s HBM) scaled by an
achieved fraction per op class (EFF below) plus a per-kernel launch gap. It is a ranking device and a baseline for the
node profile (182/tools/profile_nsys.sh + analyze_nsys.py), not a prediction to trust per op: the EFF numbers and the
elementwise pass counts are assumptions, stated next to each, and the calibration block shows how far the sum is from
the measured step times of a real #397 run.

Per rank: the batch is split over 8 GPUs, so a stage of batch_size B runs T = B / 8 tokens per rank (49152 at batch 24).
"""
import argparse
import functools
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from track_1_short.config import (  # noqa: E402
    BLOCK_SIZE, SCHEDULE_GROWTH_STAGE, SCHEDULE_GROWTH_STEPS, TRAINING_STAGES, VIRTUAL_SEQ_CAP,
)

# ---- hardware and achieved-fraction assumptions ------------------------------------------------------------------
PEAK_BF16, PEAK_FP8, HBM = 989e12, 1979e12, 3.35e12
NVLINK_BUSBW = 350e9          # achieved NCCL bus bandwidth per GPU for 100+ MB collectives (assumed; node_check measures it)
LAUNCH_S = 2.0e-6             # per-kernel gap under CUDA-graph replay (assumed)
COLLECTIVE_EXPOSED = 0.30     # fraction of collective time not hidden behind compute (rsearly overlaps most; assumed)
# Achieved fraction of the relevant peak, per op class. Compute classes are capped by flops, memory classes by bytes.
EFF = {
    "gemm_fp8_gluon": 0.62,   # warp-specialized Gluon G3 (dense MLP up / dpre, MoE up): record #92 reports ~1.2-1.3x Triton
    "gemm_fp8": 0.55,         # Triton persistent / cuBLAS _scaled_mm fp8 GEMMs at K or N = 768
    "gemm_fp8_grouped": 0.45, # grouped Triton fp8 GEMMs (MoE down, dW over experts): 8 groups of ~6k rows
    "gemm_bf16": 0.65,        # cuBLAS bf16 at T x 768 x 768-ish
    "fa3": 0.38,              # FA3 varlen, sliding window, head dims 64/128 (masked-tile waste included), fwd+bwd
    "triton_flash": 0.22,     # CPLM pointer: Triton flash loop with a per-tile token-equality mask
    "polar": 0.40,            # ANVIL symmetric batched GEMMs on 768-wide matrices
    "mem_tuned": 0.80,        # hand-written memory-bound kernels (Triton/CUDA), coalesced
    "mem_inductor": 0.70,     # inductor pointwise / row-reduction kernels over [T, 768]
    "transpose_fp8": 0.60,    # tiled 1-byte transposes
    "atomic": 0.30,           # scatter-add with fp16/bf16 atomics
    "skinny_cublas": 0.40,    # cuBLAS bf16 with N = 8 (router) or M = 8, K = T (drouter): THE uncertain number here
    "nccl": 1.0,              # handled through NVLINK_BUSBW and COLLECTIVE_EXPOSED
}

# ---- model shapes (model/gpt.py, model/moe.py, perf/kernels/cplm.py) ------------------------------------------------
DIM, HEADS, HEAD_DIM, NARROW = 768, 6, 128, 64
VOCAB = 50304
MLP_H, MOE_H, MOE_E = 2816, 1024, 8
MOE_LAYERS = (1, 3, 5, 6, 8)
DENSE_MLPS = 5                              # layers 0, 2, 4, 10 + layer 8's parallel MLP
ATTN = {  # layer: (qk_dim, v_dim, long window, paired)
    0: (64, 128, False, True), 2: (64, 128, False, True), 5: (64, 128, False, True),
    1: (64, 64, False, False), 8: (64, 64, False, False),
    3: (128, 128, True, False), 10: (128, 128, True, False),
}
XSA_LAYERS = (1, 3)
COPY_DIM, COPY_BAND = 128, 2048
VE_PLANES = 4
# Parameter counts (for the optimizer, refresh and collective lines)
MLP_BANK = 16 * MLP_H * DIM                 # 8 slots x (c_fc, c_proj); 5 used
MOE_BANK = len(MOE_LAYERS) * 2 * MOE_E * MOE_H * DIM
QK_BANK, VO_BANK = 48 * 2 * HEAD_DIM * DIM, 16 * DIM * DIM
LM_HEAD = DIM * VOCAB
# Sampled-softmax candidate counts per stage index (sampled_softmax.CANDIDATES_BY_STAGE; the test checks they agree).
CANDIDATES_BY_STAGE = {0: (10240,), 1: (10240,), 2: (14336, 14336, 24576)}
SAMPLED_HEAD_GROUP, SAMPLED_HEAD_MIN_ROWS = 4, 32768
# Mean document length in tokens (assumed: FineWeb, GPT-2 tokens), for the attention / pointer key counts. The truth is
# a heavy-tailed distribution; the profile's FA3 time is what pins this.
MEAN_DOC_TOKENS = 800

# Measured step times of #397 (records/track_1_short/2026-10-31_ExactSpeedups/this_pr/3012f81e-...txt, seed 3, 30.660 s:
# differences of the cumulative train_time over 25-step windows inside each regime).
MEASURED_MS = {"stage0": 15.45, "stage1": 26.30, "stage2_p14336": 38.60, "stage2_p24576": 40.50, "taper": 40.30,
               "extension": 19.20}


@dataclass
class Op:
    name: str
    group: str
    cls: str
    flops: float = 0.0
    bytes: float = 0.0
    kernels: int = 1
    impl: str = ""            # who wrote it: gluon / triton / cuda / cublas / fa3 / inductor / nccl
    tuned: bool = True        # already a hand-written / vendor-tuned kernel
    note: str = ""

    def seconds(self) -> float:
        if self.cls == "nccl":
            return self.bytes * 7 / 8 / NVLINK_BUSBW * COLLECTIVE_EXPOSED
        e = EFF[self.cls]
        if self.cls.startswith(("gemm", "fa3", "triton_flash", "polar")):
            peak = PEAK_FP8 if "fp8" in self.cls else PEAK_BF16
            t = max(self.flops / (peak * e), self.bytes / (HBM * EFF["mem_tuned"]))
        else:
            t = self.bytes / (HBM * e)
        return t + self.kernels * LAUNCH_S

    def bound(self) -> str:
        if self.cls == "nccl":
            return "comm"
        if self.cls.startswith(("gemm", "fa3", "triton_flash", "polar")):
            peak = PEAK_FP8 if "fp8" in self.cls else PEAK_BF16
            tc, tm = self.flops / (peak * EFF[self.cls]), self.bytes / (HBM * EFF["mem_tuned"])
            return "compute" if tc >= tm else "memory"
        return "memory"


# ---- the schedule -------------------------------------------------------------------------------------------------
@dataclass
class Regime:
    name: str
    tokens: int        # per rank
    vocab: int         # loss width: P on sampled steps, VOCAB otherwise
    ws_short: int      # tokens
    ws_long: int
    seq_cap: int       # documents are cut to this; attention segments to min(this, VIRTUAL_SEQ_CAP)
    sampled: bool


def stage_ends(scheduled=981, extension=20):
    """TrainingSchedule's stage boundaries (schedule.py), recomputed from config.py."""
    ends, acc = [0], 0.0
    for s in TRAINING_STAGES[:-1]:
        acc += s.duration
        ends.append(round(acc * scheduled))
    total = scheduled + SCHEDULE_GROWTH_STEPS + extension
    ends.append(total)
    ends[SCHEDULE_GROWTH_STAGE + 1:-1] = [e + SCHEDULE_GROWTH_STEPS for e in ends[SCHEDULE_GROWTH_STAGE + 1:-1]]
    return list(zip(ends[:-1], ends[1:]))


def regime_at(step: int, bounds=None) -> Regime:
    bounds = bounds or stage_ends()
    idx = next(i for i, (a, b) in enumerate(bounds) if step < b) if step < bounds[-1][1] else len(bounds) - 1
    a, b = bounds[idx]
    st = TRAINING_STAGES[idx]
    counts = CANDIDATES_BY_STAGE.get(idx)
    P = counts[(step - a) * len(counts) // (b - a)] if counts else 0
    names = {0: "stage0", 1: "stage1", 3: "taper", 4: "extension"}
    name = names.get(idx, f"stage2_p{P}")
    return Regime(name, st.batch_size // 8, P or VOCAB, st.window_sizes[0] * BLOCK_SIZE, st.window_sizes[1] * BLOCK_SIZE,
                  st.train_max_seq_len, bool(P))


def regimes():
    """{name: (Regime, number of steps)} over the whole run (981 + 52 + 20 = 1053 steps)."""
    bounds = stage_ends()
    out = {}
    for s in range(bounds[-1][1]):
        r = regime_at(s, bounds)
        out.setdefault(r.name, [r, 0])[1] += 1
    return {k: tuple(v) for k, v in out.items()}


@functools.lru_cache(maxsize=None)
def mean_keys(window: int, seq_cap: int, paired: bool = False, mean_doc: int = MEAN_DOC_TOKENS) -> float:
    """Token-weighted mean number of keys a query attends to: min(position-in-segment + 1, window), documents
    exponential with mean `mean_doc`, cut to seq_cap, attention segments cut to VIRTUAL_SEQ_CAP. A paired layer runs the
    interleaved doubled stream: position p becomes 2p (+1) and the window counts doubled positions."""
    seg = min(seq_cap, VIRTUAL_SEQ_CAP)
    num = den = 0.0
    step = 8
    for L in range(step, seq_cap + 1, step):
        w = math.exp(-L / mean_doc) * step / mean_doc
        if L == seq_cap:
            w = math.exp(-(L - step) / mean_doc)        # the tail mass: documents cut to seq_cap
        full, rem = divmod(L, seg)
        for n, cnt in ((seg, full), (rem, 1)):
            if not n or not cnt:
                continue
            pos = range(n)
            if paired:
                k = sum(min(2 * p + 1.5, window) for p in pos)
            else:
                k = sum(min(p + 1, window) for p in pos)
            num += w * cnt * k
            den += w * cnt * n
    return num / den


# ---- the ops of one step ------------------------------------------------------------------------------------------
def step_ops(r: Regime, router_fused: bool = False) -> list[Op]:
    T, V, D = r.tokens, r.vocab, DIM
    act = T * D * 2                                    # one bf16 [T, 768] activation
    ops: list[Op] = []
    add = ops.append

    # LM head + CPLM CE (perf/kernels/cross_entropy.py, cplm.py, sampled_*): fp8 GEMMs, e4m3 logits, one CE kernel.
    add(Op("lm_head logits GEMM (fp8)", "lm_head", "gemm_fp8", 2 * T * D * V, T * D + D * V + T * V, 1, "cublas/triton"))
    add(Op("CE+CPLM mixture kernel (fwd+bwd, one pass)", "lm_head", "mem_tuned", 0, 2 * T * V, 1, "cuda nvrtc"))
    add(Op("lm_head dX GEMM (fp8)", "lm_head", "gemm_fp8", 2 * T * V * D, T * V + D * V + act, 1, "cublas"))
    if r.sampled and T >= SAMPLED_HEAD_MIN_ROWS:
        rows = T / SAMPLED_HEAD_GROUP
        add(Op("lm_head dW GEMM (fp8, sampled 1/4 rows)", "lm_head", "gemm_fp8", 2 * rows * V * D, rows * V + rows * D,
               2, "cublas+triton"))
        add(Op("sampled dW exact parts + densify", "lm_head", "mem_tuned", 0, 2 * D * VOCAB * 2 + T * 8 * D * 2, 3,
               "triton"))
    else:
        add(Op("lm_head dW GEMM (fp8)", "lm_head", "gemm_fp8", 2 * T * V * D, T * V + T * D + D * V * 2, 1, "cublas"))
        add(Op("CE grad transpose for dW (fp8)", "lm_head", "transpose_fp8", 0, 2 * T * V, 1, "triton"))
        if r.sampled:
            add(Op("sampled densify of dW", "lm_head", "mem_tuned", 0, 2 * D * VOCAB * 2, 2, "triton"))
    if r.sampled:
        add(Op("candidate slab gather (fp8 lm_head rows)", "lm_head", "mem_tuned", 0, 4 * V * D, 2, "triton"))

    # CPLM pointer (cplm.py): q/k projection bf16, flash-style pointer fwd (1 dot) + bwd (dq, dk: 2 dots each).
    keys = min(mean_keys(2 * COPY_BAND, r.seq_cap), 1.5 * COPY_BAND)
    add(Op("CPLM q/k projection (bf16, fwd+bwd)", "cplm", "gemm_bf16", 3 * 2 * T * D * 2 * COPY_DIM, 3 * act, 4,
           "cublas+inductor"))
    add(Op("CPLM pointer fwd+bwd (Triton flash)", "cplm", "triton_flash", 10 * T * keys * COPY_DIM, 8 * T * COPY_DIM * 4,
           3, "triton"))

    # Dense MLPs (mlp.py + mlp_g3.py): 2 forward + 4 backward fp8 GEMMs each; the input / gradient fp8 transposes.
    f = 2 * T * D * MLP_H
    for k in range(DENSE_MLPS):
        tag = f"dense MLP x{DENSE_MLPS}"
        add(Op(f"{tag}: up fwd relu^2 epilogue (G3)", "mlp", "gemm_fp8_gluon", f, T * D + 2 * T * MLP_H, 1, "gluon"))
        add(Op(f"{tag}: down fwd (_scaled_mm)", "mlp", "gemm_fp8", f, T * MLP_H + act, 1, "cublas"))
        add(Op(f"{tag}: dpre bwd (G3)", "mlp", "gemm_fp8_gluon", f, T * D + 3 * T * MLP_H, 1, "gluon"))
        add(Op(f"{tag}: dW1 + dW2 + dx (fp8)", "mlp", "gemm_fp8", 3 * f, 3 * T * MLP_H + 2 * T * D + act, 3, "cublas"))
        add(Op(f"{tag}: x / g fp8 transposes (glue_fp8_t)", "mlp", "transpose_fp8", 0, 4 * T * D, 2, "triton"))
        add(Op(f"{tag}: g quantize (row half, inductor)", "mlp", "mem_inductor", 0, act + T * D, 1, "inductor", True))

    # MoE (moe.py, moe_mlp.py, moe_g3.py): router, dispatch, gathers, grouped fp8 GEMMs, combine.
    M = T + MOE_E * 128
    fe = 2 * M * D * MOE_H
    for layer in MOE_LAYERS:
        tag = f"MoE x{len(MOE_LAYERS)}"
        if router_fused:
            add(Op(f"{tag}: router GEMM+softmax+top1 (fused)", "moe", "mem_tuned", 2 * T * D * MOE_E, act, 1,
                   "triton (MOE_ROUTER_FUSED)", False))
        else:
            add(Op(f"{tag}: router GEMM N=8 (cuBLAS bf16)", "moe", "skinny_cublas", 2 * T * D * MOE_E, act, 1, "cublas",
                   False, "skinny shape: the fusion target"))
            add(Op(f"{tag}: router softmax/max (inductor)", "moe", "mem_inductor", 0, T * MOE_E * 14, 2, "inductor",
                   False))
        add(Op(f"{tag}: dispatch (count + place)", "moe", "mem_tuned", 0, T * 16 + M * 4, 2, "triton"))
        add(Op(f"{tag}: gather x -> e4m3 row + transposed", "moe", "mem_tuned", 0, act + 2 * M * D, 1, "triton"))
        add(Op(f"{tag}: up fwd (G3 grouped)", "moe", "gemm_fp8_gluon", fe, M * D + 2 * M * MOE_H, 1, "gluon"))
        add(Op(f"{tag}: down fwd (grouped)", "moe", "gemm_fp8_grouped", fe, M * MOE_H + M * D * 2, 1, "triton"))
        add(Op(f"{tag}: combine gather*gate (inductor)", "moe", "mem_inductor", 0, 2 * act, 1, "inductor"))
        add(Op(f"{tag}: gather g (e5m2 dual + gate grad)", "moe", "mem_tuned", 0, act + M * D * 2 + 2 * M * D, 1,
               "triton"))
        add(Op(f"{tag}: dW2 + dW1 (grouped over K)", "moe", "gemm_fp8_grouped", 2 * fe, 2 * M * MOE_H + 2 * M * D, 2,
               "triton"))
        add(Op(f"{tag}: dpre bwd (G3 grouped)", "moe", "gemm_fp8_gluon", fe, M * D + 3 * M * MOE_H, 1, "gluon"))
        add(Op(f"{tag}: dx grouped", "moe", "gemm_fp8_grouped", fe, M * MOE_H + M * D * 2, 1, "triton"))
        if router_fused:
            add(Op(f"{tag}: dlogits (inductor)", "moe", "mem_inductor", 0, T * MOE_E * 4 * 6, 3, "inductor", False))
            add(Op(f"{tag}: dx combine + drouter (fused)", "moe", "mem_tuned", 2 * T * D * MOE_E, 3 * act, 2,
                   "triton (MOE_ROUTER_FUSED)", False))
        else:
            add(Op(f"{tag}: dlogits (inductor)", "moe", "mem_inductor", 0, T * MOE_E * 4 * 8, 4, "inductor", False))
            add(Op(f"{tag}: dx combine (Triton)", "moe", "mem_tuned", 0, 2 * act, 1, "triton"))
            add(Op(f"{tag}: drouter GEMM M=8 K=T (cuBLAS bf16)", "moe", "skinny_cublas", 2 * T * D * MOE_E, act, 1,
                   "cublas", False, "skinny shape: the fusion target"))

    # Attention (attention.py, qkv_rope.py, fp8_attention_quant.py): packed fp8 QKV, norm/rope, FA3, XSA, bf16 O.
    for layer, (qk, v, long_w, paired) in ATTN.items():
        tag = f"attn x{len(ATTN)}"
        n_qkv = HEADS * (2 * qk + v)
        add(Op(f"{tag}: QKV GEMM fwd (fp8)", "attn", "gemm_fp8", 2 * T * D * n_qkv, T * D + T * n_qkv * 2, 1, "triton"))
        add(Op(f"{tag}: QKV dX + dW (fp8)", "attn", "gemm_fp8", 2 * 2 * T * D * n_qkv, 2 * T * n_qkv + act, 2, "cublas"))
        add(Op(f"{tag}: qk-norm/rope fwd + bwd (+ e4m3 dual)", "attn", "mem_tuned", 0,
               4 * T * HEADS * 2 * qk * 2 + 3 * T * n_qkv, 2, "triton"))
        w = r.ws_long if long_w else r.ws_short
        k = mean_keys(w, r.seq_cap, paired)
        fa = 2 * T * HEADS * k * (qk + v)
        add(Op(f"{tag}: FA3 fwd + bwd", "attn", "fa3", 3.5 * fa, 2 * T * n_qkv * 2 + 4 * T * HEADS * v * 2, 2, "fa3"))
        add(Op(f"{tag}: O proj fwd + dX + dW (bf16)", "attn", "gemm_bf16", 3 * 2 * T * HEADS * v * D,
               3 * (act + T * HEADS * v * 2), 3, "cublas"))
        add(Op(f"{tag}: x quant (row fused, transpose glue)", "attn", "transpose_fp8", 0, 2 * T * D, 1, "triton"))
        if layer in XSA_LAYERS:
            add(Op(f"XSA x{len(XSA_LAYERS)} (inductor fwd+bwd)", "attn", "mem_inductor", 0, 8 * T * HEADS * v * 2, 3,
                   "inductor"))

    # Residual stream and the MUDD mixes (inductor, compiled forward/backward): counted in [T, 768] bf16 passes.
    # Assumed pass counts (the least grounded lines here): attention sublayer 6 fwd / 12 bwd, MLP sublayer 5 / 10,
    # attention-free layers 3 / 6, the last layer's MUDD 8 / 16, the post-loop MUDD mix of 10 sources 12 / 24,
    # embedding + smear + x0 norm + MUDD gates 6 / 10.
    passes = len(ATTN) * 18 + (DENSE_MLPS - 1 + len(MOE_LAYERS)) * 15 + 4 * 9 + 24 + 36 + 16
    add(Op(f"residual/norm/MUDD elementwise ({passes} passes)", "elementwise", "mem_inductor", 0, passes * act,
           passes // 2, "inductor", False, "assumed pass counts; inductor already fuses these"))

    # Lookups and their sparse backward (ngram_embed.py, value_embed.py, row_scatter.py, embed).
    add(Op("embed gather + dense embed grad", "lookup", "mem_tuned", 0, 2 * act + 2 * VOCAB * D * 2, 3, "aten"))
    add(Op("n-gram embedding fwd (2 rows/token)", "lookup", "mem_tuned", 0, 2 * act + act, 1, "triton"))
    add(Op("n-gram sink grad + prescatter (atomics)", "lookup", "atomic", 0, 2 * act + 2 * act, 2, "triton"))
    add(Op("value-embed gathers (4 planes)", "lookup", "mem_tuned", 0, VE_PLANES * 2 * act, 1, "aten"))
    add(Op("value-embed grad scatter (fp16 atomics)", "lookup", "atomic", 0, VE_PLANES * 2 * act, VE_PLANES, "triton"))

    # Optimizer (optim/anvil.py, polar_express.py, replicated Adam): this rank's shard; owned by r6-optim.
    def polar(m, n, count):     # four-map cascade, symmetric Gram / polynomial at about half cost
        n, m = min(m, n), max(m, n)
        return count * 4 * (2 * m * n * n + n ** 3 + 2 * m * n * n * 0.5)
    pf = polar(MLP_H, DIM, 2) + polar(MOE_H, DIM, 10) + polar(2 * HEAD_DIM, DIM, 6) + polar(DIM, DIM, 2)
    add(Op("ANVIL cascade (this rank's matrices)", "optim", "polar", pf, (MLP_BANK + MOE_BANK + QK_BANK + VO_BANK) / 8 * 24,
           60, "triton"))
    add(Op("replicated Adam (lm_head, embed, scalars; every other step)", "optim", "mem_tuned", 0, 0.5 * 2 * LM_HEAD * 16,
           2, "triton"))
    add(Op("fp8 weight refresh (MLP, attn, lm_head, MoE owner + col transpose)", "optim", "mem_tuned", 0,
           (10 * MLP_H * DIM + QK_BANK + VO_BANK + LM_HEAD) * 4 + MOE_BANK / 8 * 4 + MOE_BANK * 2, 8, "triton"))

    # Collectives (exposed part only).
    rs = (MLP_BANK + MOE_BANK + QK_BANK + VO_BANK) * 2
    ag = (MLP_BANK + QK_BANK + VO_BANK) * 2 + MOE_BANK
    add(Op("bank reduce-scatter + all-gather (exposed)", "comm", "nccl", 0, rs + ag, 6, "nccl"))
    add(Op("lm_head/embed all-reduce, every other step (exposed)", "comm", "nccl", 0, 0.5 * 2 * 2 * LM_HEAD * 2, 2,
           "nccl"))
    add(Op("n-gram rows exchange (exposed)", "comm", "nccl", 0, 4 * T * D * 2, 4, "nccl"))
    return ops


# ---- reporting -----------------------------------------------------------------------------------------------------
def aggregate(ops: list[Op]) -> list[tuple[str, str, str, bool, float, float, float, int]]:
    """Merge the per-layer copies of an op (same name): (name, group, impl, tuned, GFLOP, MB, seconds, count)."""
    acc = {}
    for o in ops:
        a = acc.setdefault(o.name, [o.group, o.impl, o.tuned, 0.0, 0.0, 0.0, 0, o.bound()])
        a[3] += o.flops / 1e9
        a[4] += o.bytes / 1e6
        a[5] += o.seconds()
        a[6] += 1
    return sorted(((n, *v) for n, v in acc.items()), key=lambda t: -t[6])


def step_seconds(r: Regime, router_fused=False) -> float:
    return sum(o.seconds() for o in step_ops(r, router_fused))


def run_totals(router_fused=False):
    """{op name: seconds over the whole run}, and the run total."""
    tot = {}
    for r, n in regimes().values():
        for row in aggregate(step_ops(r, router_fused)):
            tot[row[0]] = tot.get(row[0], 0.0) + row[6] * n
    return tot, sum(tot.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regime", default="stage2_p14336", help="one of: " + ", ".join(regimes()))
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--router-fused", action="store_true", help="model MOE_ROUTER_FUSED's two kernels")
    ap.add_argument("--csv", help="write the chosen regime's merged ops here")
    a = ap.parse_args(argv)
    regs = regimes()
    r, nsteps = regs[a.regime]
    rows = aggregate(step_ops(r, a.router_fused))
    total = sum(x[6] for x in rows)
    print(f"# t1-gpu-stack step, regime {r.name}: {r.tokens} tokens/rank, loss width {r.vocab}, windows "
          f"{r.ws_short}/{r.ws_long}, docs <= {r.seq_cap}, {nsteps} steps of the run; router_fused={a.router_fused}")
    print(f"{'#':>3} {'op':<62} {'impl':<16} {'tuned':<5} {'GFLOP':>9} {'MB':>9} {'bound':<7} {'ms':>7} {'%':>5}")
    for i, (name, group, impl, tuned, gf, mb, s, cnt, bound) in enumerate(rows[:a.top], 1):
        print(f"{i:>3} {name:<62} {impl[:16]:<16} {'yes' if tuned else 'NO':<5} {gf:>9.1f} {mb:>9.1f} {bound:<7} "
              f"{s * 1e3:>7.3f} {100 * s / total:>5.1f}")
    print(f"    {'sum of modelled ops':<62} {'':<16} {'':<5} {'':>9} {'':>9} {'':<7} {total * 1e3:>7.3f}")
    groups = {}
    for row in rows:
        groups[row[1]] = groups.get(row[1], 0.0) + row[6]
    print("\n# by group (ms): " + ", ".join(f"{g} {s * 1e3:.2f}" for g, s in sorted(groups.items(), key=lambda t: -t[1])))

    print("\n# calibration against #397 seed 3 (measured ms/step from its log; the gap is unmodelled time: host/graph")
    print("# gaps between replays, the clock-side syncs, and every assumption above being off)")
    print(f"{'regime':<16} {'tokens':>7} {'width':>6} {'steps':>5} {'model ms':>9} {'measured':>9} {'model/meas':>10}")
    for name, (rg, n) in regs.items():
        m = step_seconds(rg, a.router_fused) * 1e3
        meas = MEASURED_MS.get(name)
        print(f"{name:<16} {rg.tokens:>7} {rg.vocab:>6} {n:>5} {m:>9.2f} {meas if meas else float('nan'):>9.2f} "
              f"{(m / meas) if meas else float('nan'):>10.2f}")

    tot, run_s = run_totals(a.router_fused)
    print(f"\n# whole run ({sum(n for _, n in regs.values())} steps): modelled {run_s:.2f} s of GPU time "
          f"(#397 measured 30.66 s wall, train_time)")
    print(f"{'op':<62} {'s/run':>7} {'tuned':<5}")
    tuned = {row[0]: row[3] for rg, _ in regs.values() for row in aggregate(step_ops(rg, a.router_fused))}
    for name, s in sorted(tot.items(), key=lambda t: -t[1])[:a.top]:
        print(f"{name:<62} {s:>7.3f} {'yes' if tuned[name] else 'NO':<5}")

    print("\n# not-yet-tuned ops, per run (the candidates; memory/launch-bound ones are what a fused kernel can cut)")
    for name, s in sorted(((n, s) for n, s in tot.items() if not tuned[n]), key=lambda t: -t[1]):
        print(f"  {name:<60} {s:>7.3f} s")
    if not a.router_fused:
        _, fused_s = run_totals(True)
        print(f"\n# MOE_ROUTER_FUSED modelled delta: {1e3 * (fused_s - run_s):+.0f} ms per run "
              f"(EFF skinny_cublas = {EFF['skinny_cublas']}; at 0.8 the delta is "
              f"{1e3 * _router_delta_at(0.8):+.0f} ms, at 0.25 it is {1e3 * _router_delta_at(0.25):+.0f} ms)")
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("op,group,impl,tuned,gflop,mb,ms,count,bound\n")
            for name, group, impl, tuned_, gf, mb, s, cnt, bound in rows:
                f.write(f"\"{name}\",{group},{impl},{int(tuned_)},{gf:.3f},{mb:.3f},{s * 1e3:.4f},{cnt},{bound}\n")
    return 0


def _router_delta_at(skinny_eff: float) -> float:
    old = EFF["skinny_cublas"]
    EFF["skinny_cublas"] = skinny_eff
    try:
        return run_totals(True)[1] - run_totals(False)[1]
    finally:
        EFF["skinny_cublas"] = old


if __name__ == "__main__":
    sys.exit(main())
