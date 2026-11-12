"""Run configuration: hyperparameters and the training-stage table.

The run's own environment variables (DATA_PATH, NUM_SCHEDULED_ITERATIONS, TRAIN_SEED, and the FP8
experiment knobs FP8_OFF, HADAMARD_FP8, HADAMARD_SEED, HADAMARD_BLOCK, HADAMARD_GEMMS, FP8_DUMP_DIR /
_STEPS / _ROWS / _SLOTS, FP8_ATTN_O / FP8_ATTN_O_SCALES at the bottom) are read here. The others: the
experiment knob HADAMARD_SAT_STATS (train_gpt.py; read here too, to decide whether FP8_ATTN_O records amaxes)
and MONARCH_MLP / _BLOCKS / _EXPAND / _LR_MUL / _DOWN_LR_MUL (model/monarch.py); torchrun's
RANK / LOCAL_RANK / WORLD_SIZE (distributed.py; train_gpt.py also reads LOCAL_RANK), PYTORCH_ALLOC_CONF
(set by train_gpt.py), and CUDA_HOME (perf/kernels/cross_entropy.py, for the CE kernel's CUDA headers).

Provenance: the five-stage table (batch taper, batch-8 extension, seq 3072 under a 2560 attention
cap, 52 growth steps) is record #360's.
"""
import os
import uuid
from dataclasses import dataclass

@dataclass(slots=True)
class Hyperparameters:
    # data
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin") # input .bin to train on
    val_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin") # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    val_batch_size: int = 4 * 64 * 1024 * 8
    # schedule: the base step count of the main stages (SCHEDULE_GROWTH_STEPS are added on top), then
    # the extension stage. The override is for step-count sweeps (record #360's KX_STEPS).
    num_scheduled_iterations: int = int(os.environ.get("NUM_SCHEDULED_ITERATIONS", "1122"))
    num_extension_iterations: int = 20
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    # Every how many steps to evaluate val loss; 0 = only at the end (record #360: each intermediate
    # validation is on the clock -- the val reads and n-gram pulls, the value_embeds gather, the drained
    # CUDA-graph run-ahead and the cycle re-land after it cost ~0.2-0.5 s over a run at 250).
    val_loss_every: int = 0
    save_checkpoint: bool = False
    run_evals: bool = False  # run additional evaluations after training is completed
    # reproducibility: unset means a random init; any integer (0 included) seeds it
    train_seed: int | None = int(os.environ["TRAIN_SEED"]) if "TRAIN_SEED" in os.environ else None

@dataclass(slots=True)
class TrainingStage:
    lr_mul: float
    batch_size: int
    window_sizes: tuple[int, int]  # (short, long) in block units
    mtp_weights_start: list[float]
    mtp_weights_end: list[float]
    train_max_seq_len: int  # documents are cut to this many tokens
    # Prefix-token-prediction loss weight, stepped: the stage's steps split into equal parts, one weight each.
    prefix_weights: tuple[float, ...]
    duration: float | None = None  # fraction of num_scheduled_iterations; None for the extension stage


# Stage lengths as fractions of num_scheduled_iterations (normalized to sum to 1). The last entry is
# the batch-24 stage plus the batch-20 taper, which takes TAPER_DURATION of it.
STAGE_DURATIONS = [0.2854, 0.3209, 0.3931]
STAGE_DURATIONS = [d / sum(STAGE_DURATIONS) for d in STAGE_DURATIONS]
TAPER_DURATION = 0.06
# Record #360 grew its run by 52 steps without re-deriving the fractions: the extra steps all go to
# the batch-24 stage (index 2), which keeps every later boundary's Adam-step (odd/even) parity.
SCHEDULE_GROWTH_STEPS = 52
SCHEDULE_GROWTH_STAGE = 2
# Attention window sizes are counted in blocks of this many tokens (record #360).
BLOCK_SIZE = 128
# Documents may run to 3072 tokens in the last stages, but attention segments are cut at this many
# tokens (the loader adds segment boundaries; the token stream is untouched). A whole number of blocks.
VIRTUAL_SEQ_CAP = 2560
TAPER_BATCH_UNITS = 20
# The lr decays linearly to its floor over the last LR_COOLDOWN_FRAC of the main stages (record #360).
LR_COOLDOWN_FRAC = 0.80
# embed unties from lm_head at the start of this stage, the extension stage (record #360).
SPLIT_EMBED_STAGE = 4
# The final validation extends the long attention window to this many blocks (record #360).
WS_POST_YARN_EXT = 20

# window_sizes are in units of BLOCK_SIZE tokens.
TRAINING_STAGES = [
    TrainingStage(duration=STAGE_DURATIONS[0], train_max_seq_len=896, batch_size=8 * 2048 * 8, window_sizes=(1, 3), lr_mul=1.0,
                  mtp_weights_start=[1.0, 0.5, 0.25], mtp_weights_end=[1.0, 0.5, 0.0], prefix_weights=(0.25,)),
    TrainingStage(duration=STAGE_DURATIONS[1], train_max_seq_len=2048, batch_size=16 * 2048 * 8, window_sizes=(3, 7), lr_mul=1.52,  # (16/8)**0.6
                  mtp_weights_start=[1.0, 0.5], mtp_weights_end=[1.0, 0.0], prefix_weights=(0.20, 0.15, 0.10, 0.05)),
    TrainingStage(duration=STAGE_DURATIONS[2] - TAPER_DURATION, train_max_seq_len=3072, batch_size=24 * 2048 * 8, window_sizes=(5, 11), lr_mul=1.73,  # (24/8)**0.5
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
    # terminal batch taper: 24 -> 20, lr_mul scaled by sqrt(20/24)
    TrainingStage(duration=TAPER_DURATION, train_max_seq_len=3072, batch_size=TAPER_BATCH_UNITS * 2048 * 8, window_sizes=(5, 11),
                  lr_mul=1.73 * (TAPER_BATCH_UNITS / 24) ** 0.5,
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
    # extension stage at batch 8: wall-matched, ~18 steps at batch 8 cost what 7 do at batch 24
    TrainingStage(train_max_seq_len=3072, batch_size=8 * 2048 * 8, window_sizes=(6, 13), lr_mul=1.0,  # lr_mul is not used (lr sits at the floor)
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
]
assert VIRTUAL_SEQ_CAP % BLOCK_SIZE == 0
assert any(s.train_max_seq_len > VIRTUAL_SEQ_CAP for s in TRAINING_STAGES), "VIRTUAL_SEQ_CAP would be inert"


# -----------------------------------------------------------------------------
# FP8 experiment knobs (182/hadamard-fp8). Every default is the record's behaviour; they are module
# constants, so torch.compile sees them as trace-time constants and the default graphs are unchanged.

def _fp8_off_parts(value: str) -> frozenset[str]:
    """FP8_OFF: "" / "0" = record (all fp8); "1" / "all" = every training GEMM in bf16; or a comma list
    of "mlp", "qkv", "lm" to turn off only those paths (for attributing a bf16-vs-fp8 gap)."""
    value = value.strip().lower()
    if value in ("", "0"):
        return frozenset()
    parts = frozenset({"mlp", "qkv", "lm"} if value in ("1", "all") else (p.strip() for p in value.split(",") if p.strip()))
    assert parts <= {"mlp", "qkv", "lm"}, f"FP8_OFF: unknown parts {sorted(parts - {'mlp', 'qkv', 'lm'})}"
    return parts

FP8_OFF = _fp8_off_parts(os.environ.get("FP8_OFF", ""))
# HADAMARD_FP8=1: block-Hadamard-rotated fp8 operands for the four MLP gradient GEMMs (dW1, dW2, dx and
# the dpre GEMM), rotated along each GEMM's contraction dim (perf/kernels/hadamard.py).
HADAMARD_FP8 = os.environ.get("HADAMARD_FP8", "0") == "1"
HADAMARD_SEED = int(os.environ.get("HADAMARD_SEED", "1234"))
assert not (HADAMARD_FP8 and "mlp" in FP8_OFF), "HADAMARD_FP8 rotates the fp8 MLP; FP8_OFF=mlp removes it"
# HADAMARD_BLOCK=32|64|128: the rotation block (Q's size). The kernels keep 128-wide tiles and rotate each
# one block-diagonally, so the epilogue dots cost 2 * HADAMARD_BLOCK FLOP per element.
HADAMARD_BLOCK = int(os.environ.get("HADAMARD_BLOCK", "128"))
assert HADAMARD_BLOCK in (32, 64, 128), f"HADAMARD_BLOCK={HADAMARD_BLOCK}: use 32, 64 or 128"

def _hadamard_gemms(value: str) -> frozenset[str]:
    """HADAMARD_GEMMS: comma subset of dw1, dw2, dx, dpost (or "all", the default). A GEMM not listed gets no
    rotation on either operand. That is not always the record's exact operand path: with dpost but not dw2,
    dW2's g_t is quantized from the stored bf16 g (one pass with the rotated g_row) where the record rounds
    the producer's fp32; and dx / dW1 share the delayed dpre scale, whose amax covers whichever dpre layout
    is rotated, so rotating one can move the other's scale."""
    names = {"dw1", "dw2", "dx", "dpost"}
    value = value.strip().lower()
    parts = frozenset(names if value in ("", "all") else (p.strip() for p in value.split(",") if p.strip()))
    assert parts and parts <= names, f"HADAMARD_GEMMS: unknown or empty {sorted(parts - names) or value!r}"
    return parts

# Validated even when HADAMARD_FP8 is off, like HADAMARD_BLOCK: a typo must not pass silently in either arm.
_HADAMARD_GEMMS_REQUESTED = _hadamard_gemms(os.environ.get("HADAMARD_GEMMS", "all"))
HADAMARD_GEMMS = _HADAMARD_GEMMS_REQUESTED if HADAMARD_FP8 else frozenset()
HADAMARD_DW1, HADAMARD_DW2, HADAMARD_DX, HADAMARD_DPOST = (g in HADAMARD_GEMMS for g in ("dw1", "dw2", "dx", "dpost"))
# FP8_DUMP_DIR: save sample MLP / QKV operands (x, dy, W) at FP8_DUMP_STEPS from rank 0, for the offline
# quantization-error study (182/hadamard-fp8/fp8_quant_error.py). Off by default; the clock is paused
# around each dump, but a dump run is an analysis run, not a timing run (perf/fp8_dump.py).
FP8_DUMP_DIR = os.environ.get("FP8_DUMP_DIR", "")
FP8_DUMP_STEPS = tuple(int(s) for s in os.environ.get("FP8_DUMP_STEPS", "100,500,1000").split(",") if s.strip())
FP8_DUMP_ROWS = int(os.environ.get("FP8_DUMP_ROWS", "8192"))         # leading tokens kept per tensor
FP8_DUMP_SLOTS = tuple(int(s) for s in os.environ.get("FP8_DUMP_SLOTS", "0,5,10").split(",") if s.strip())
# FP8_ATTN_O=1 (182/fp8-attn-o): the attention output projection (y @ (o_scale * o_w).T) runs in fp8 in
# training, forward, dx and dW, with o_scale folded into the GEMM descale (perf/kernels/attn_o.py). Validation
# stays bf16. FP8_ATTN_O_SCALES: "static" (default) = the power-of-two scales derived in perf/attn_o_scales.py;
# "delayed" = y and g from the previous step's amax, o_w from its current amax (the arm for when the static
# derivation is wrong).
FP8_ATTN_O = os.environ.get("FP8_ATTN_O", "0") == "1"
FP8_ATTN_O_SCALES = os.environ.get("FP8_ATTN_O_SCALES", "static").strip().lower()
assert os.environ.get("FP8_ATTN_O", "0") in ("0", "1"), f"FP8_ATTN_O={os.environ['FP8_ATTN_O']!r}: use 0 or 1"
assert FP8_ATTN_O_SCALES in ("static", "delayed"), f"FP8_ATTN_O_SCALES={FP8_ATTN_O_SCALES!r}: use static or delayed"
# FP8_OFF with every part is the bf16 upper bound of 182/hadamard-fp8; adding an fp8 GEMM to it is a contradiction.
# A partial FP8_OFF (attribution arms) composes with FP8_ATTN_O; so do HADAMARD_FP8 (MLP only) and MONARCH_MLP.
assert not (FP8_ATTN_O and FP8_OFF == {"mlp", "qkv", "lm"}), "FP8_ATTN_O adds an fp8 GEMM; FP8_OFF=all is the bf16 upper bound"
# The amax of y, g and o_w is recorded (one atomic max per tile in the quantize kernels; for y/g that costs the
# inductor-fused row quantize on the gated/XSA layers) only when a scale needs it: the delayed arm, or the
# analysis-only HADAMARD_SAT_STATS=1, whose "fp8 sat" lines then get o_y / o_g / o_w rows.
FP8_ATTN_O_AMAX = FP8_ATTN_O and (FP8_ATTN_O_SCALES == "delayed" or os.environ.get("HADAMARD_SAT_STATS", "0") == "1")
