"""What the ANVIL cascade costs per step on one rank of t1-gpu-stack, from the code (no GPU).

    python 182/optim-cost/cost_model.py

Per bank: the shard each rank updates, the cascade's GEMM work (one Gram, one polynomial, one update GEMM per map), the
CTAs each kernel launches against the H100's 132 SMs, the elementwise bytes of the rest of the update (rails, equalizer,
cautious decay), and a time range under stated throughput assumptions. Then the calibration against the one measured
number there is: #393's six -> four maps, -0.19 s over the run at equal val (paired A/B on its node).

Shapes are this tree's (track_1_short/model/gpt.py, model/moe.py, optim/anvil.py; test_optim_cost.py checks them
against the source). The tile configs are anvil_opt._GRAM_CFG for the fused banks and polar_express's for the stock
moe_bank path.
"""
import math
from dataclasses import dataclass

WORLD = 8
SMS = 132
MAPS = 4
STEPS = 1053                   # 981 + 52 + 20
STEP_MS = 30.957 / STEPS * 1e3  # node: 30.957 s for 1053 steps -> 29.4 ms per step (training time only)
HBM_TBPS = 3.35 * 0.8          # H100 SXM HBM3, ~80% achievable on streaming kernels


@dataclass
class Bank:
    name: str
    matrices: int      # global matrices in the bank (reshape[0])
    rows: int
    cols: int
    live: int          # matrices that ever move (the rest are padding / frozen)
    fused: bool        # anvil_opt fused tail (True) or the inductor anvil_cascade (moe_bank)
    gram_tile: tuple   # (BM, BN) of the Gram kernel
    poly_tile: tuple   # (BM, BN) of b*A + c*A@A
    update: str        # how the a*X + X@B / B@X GEMM runs

    @property
    def chunk(self):
        return self.matrices // WORLD

    @property
    def tall(self):
        return self.rows > self.cols

    @property
    def n(self):       # Gram edge
        return min(self.rows, self.cols)

    @property
    def m(self):       # long edge
        return max(self.rows, self.cols)


BANKS = [
    # 7 attention slots x 6 head pairs = 42 groups of [Q; K] heads (2 x 128 rows), padded to 48
    Bank("qk_bank", 48, 256, 768, 42, True, (64, 64), (64, 64), "cuBLAS baddbmm (B @ X)"),
    # V and O of 7 slots = 14, padded to 16; square, so the wide path (X X^T, left products)
    Bank("vo_bank", 16, 768, 768, 14, True, (64, 128), (64, 128), "cuBLAS baddbmm (B @ X)"),
    # 8 slots x (c_fc, c_proj) of [2816, 768]; slots 5-7 are frozen padding (ranks 5, 6, 7 hold only padding)
    Bank("mlp_bank", 16, 2816, 768, 10, True, (64, 128), (64, 128), "gemm_xb_add (persistent 128x128, fused +aX)"),
    # 5 MoE layers x (c_fc, c_proj) x 8 experts of [1024, 768]; stock path: polar_express tiles 128x128
    Bank("moe_bank", 80, 1024, 768, 80, False, (128, 128), (128, 128), "cuBLAS baddbmm (X @ B)"),
]


def live_tiles(n: int, bm: int, bn: int) -> int:
    """Blocks a symmetric kernel computes (LOWER_UPPER=1: skips blocks entirely above the diagonal, m + bm <= n_idx)."""
    return sum(1 for i in range(math.ceil(n / bm)) for j in range(math.ceil(n / bn)) if not (i * bm + bm <= j * bn))


def bank_cost(b: Bank, maps: int = MAPS) -> dict:
    n, m, c = b.n, b.m, b.chunk
    gram_dense = 2 * m * n * n          # X^T X or X X^T, full
    poly_dense = 2 * n * n * n          # A @ A
    upd = 2 * m * n * n                 # X @ B or B @ X
    tiles_g = live_tiles(n, *b.gram_tile)
    tiles_p = live_tiles(n, *b.poly_tile)
    frac_g = tiles_g / (math.ceil(n / b.gram_tile[0]) * math.ceil(n / b.gram_tile[1]))
    frac_p = tiles_p / (math.ceil(n / b.poly_tile[0]) * math.ceil(n / b.poly_tile[1]))
    per_round_dense = c * (gram_dense + poly_dense + upd)
    per_round_eff = c * (frac_g * gram_dense + frac_p * poly_dense + upd)
    elems = c * b.rows * b.cols
    # Rails + blend + cast (fused: grad 2 B + velocity 2x4 B read and written + x 2 B); normalization (X read/written,
    # 4 B); equalizer + cautious decay (v, p 2 B, mantissa 2 B rw, slow rail 4 B, update ~6 B); the stock moe path
    # also re-reads the rails for the blend and writes a scratch copy (anvil_opt's docstring: "three inductor kernels and
    # a scratch copy of the fp32 rails").
    bytes_pre = elems * (2 + 16 + 2) if b.fused else elems * (2 + 16 + 2 + 16 + 8)
    bytes_rest = elems * (4 + 2 * 2 + 2 * 2 + 4 + 6)
    return dict(
        bank=b.name, chunk=c, shape=f"{b.rows}x{b.cols}", path="tall X^T X" if b.tall else "wide X X^T",
        gflop_round_dense=per_round_dense / 1e9, gflop_round_eff=per_round_eff / 1e9,
        gflop_cascade_eff=maps * per_round_eff / 1e9,
        ctas_gram=c * tiles_g, ctas_poly=c * tiles_p,
        ctas_update=c * math.ceil(m / 128) * math.ceil(n / 128),
        kernels=2 + 3 * maps + 6,  # pre, gram0 + norm, per map (gram, poly, update[, add]), equalizer + decay ~6
        mb_elementwise=(bytes_pre + bytes_rest) / 1e6,
        mb_pre=bytes_pre / 1e6,
        live_frac=b.live / b.matrices,
    )


def report(tflops=(250.0, 450.0)):
    rows = [bank_cost(b) for b in BANKS]
    lo_t, hi_t = tflops
    print(f"Per rank per step (world {WORLD}, {MAPS} maps; step = {STEP_MS:.1f} ms over {STEPS} steps)\n")
    print(f"{'bank':<9} {'chunk':>5} {'shape':>9} {'path':<11} {'GF/round':>9} {'GF/round':>9} {'GF casc':>8} "
          f"{'CTAs':>15} {'MB elem':>8}")
    print(f"{'':<9} {'':>5} {'':>9} {'':<11} {'dense':>9} {'executed':>9} {'executed':>8} {'gram/poly/upd':>15} {'':>8}")
    for r in rows:
        print(f"{r['bank']:<9} {r['chunk']:>5} {r['shape']:>9} {r['path']:<11} {r['gflop_round_dense']:9.2f} "
              f"{r['gflop_round_eff']:9.2f} {r['gflop_cascade_eff']:8.1f} "
              f"{r['ctas_gram']:>4}/{r['ctas_poly']:>4}/{r['ctas_update']:>4} {r['mb_elementwise']:8.0f}")
    gf = sum(r["gflop_cascade_eff"] for r in rows)
    mb = sum(r["mb_elementwise"] for r in rows)
    kern = sum(r["kernels"] for r in rows)
    t_gemm = (gf / hi_t, gf / lo_t)                     # ms (GF / TF/s = ms)
    t_mem = mb / 1e3 / HBM_TBPS                         # GB / (TB/s) = ms
    t_launch = (kern * 2e-3, kern * 4e-3)               # 2-4 us of in-graph launch + ramp per dependent kernel
    total = (t_gemm[0] + t_mem + t_launch[0], t_gemm[1] + t_mem + t_launch[1])
    print(f"\ncascade GEMMs: {gf:.0f} GFLOP executed -> {t_gemm[0]:.2f}-{t_gemm[1]:.2f} ms at {hi_t:.0f}-{lo_t:.0f} TFLOP/s "
          f"(bf16 dense peak ~990; these batched 256-2816 x 768 GEMMs reach a fraction)")
    print(f"elementwise:   {mb:.0f} MB -> {t_mem:.2f} ms at {HBM_TBPS:.2f} TB/s")
    print(f"dependency chain: ~{kern} kernels -> {t_launch[0]:.2f}-{t_launch[1]:.2f} ms of per-kernel latency floor")
    print(f"optimizer bank updates, GPU time: {total[0]:.2f}-{total[1]:.2f} ms per step = "
          f"{100 * total[0] / STEP_MS:.1f}-{100 * total[1] / STEP_MS:.1f}% of {STEP_MS:.1f} ms")
    per_map_measured = 0.19 / 2 / STEPS * 1e3
    print(f"\ncalibration: #393 six -> four maps = -0.19 s over the run -> {per_map_measured * 1e3:.0f} us of step time per map "
          f"round (all banks); the model's per-round GEMM time is {1e3 * t_gemm[0] / MAPS:.0f}-{1e3 * t_gemm[1] / MAPS:.0f} us, so "
          f"the measured cost implies the rounds are nearly fully exposed (or run at ~"
          f"{gf / MAPS / per_map_measured:.0f} TFLOP/s fully exposed).")
    print(f"-> the four-map cascade itself: ~{MAPS * per_map_measured:.2f} ms per step exposed "
          f"(~{MAPS * per_map_measured * STEPS / 1e3:.2f} s per run, {100 * MAPS * per_map_measured / STEP_MS:.1f}% of the step)")
    moe = next(r for r in rows if r["bank"] == "moe_bank")
    fused_pre = moe["mb_pre"] * (2 + 16 + 2) / (2 + 16 + 2 + 16 + 8)
    print(f"\nOPTIM_MOEFUSED: moe_bank pre-pass {moe['mb_pre']:.0f} MB -> ~{fused_pre:.0f} MB, "
          f"{(moe['mb_pre'] - fused_pre) / 1e3 / HBM_TBPS * 1e3:.0f} us per step "
          f"({(moe['mb_pre'] - fused_pre) / 1e3 / HBM_TBPS * STEPS / 1e3:.2f} s per run) plus 2-3 fewer kernels")
    print(f"OPTIM_MAPS=3:   one round less: ~{per_map_measured * 1e3:.0f} us per step measured-scale "
          f"(~{per_map_measured * STEPS / 1e3:.2f} s per run); break-even {per_map_measured * STEPS / 250:.2f} millinats at "
          f"250 ms/mnat, {per_map_measured * STEPS / 164:.2f} at 164")
    under = [r for r in rows if r["ctas_gram"] < SMS]
    print("Gram kernels under one wave (< 132 CTAs): " + ", ".join(f"{r['bank']} ({r['ctas_gram']})" for r in under))
    return rows


if __name__ == "__main__":
    report()
