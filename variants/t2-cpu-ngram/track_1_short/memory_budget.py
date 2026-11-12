"""Back-of-envelope host-RAM and per-GPU memory budget, printed at startup (and runnable offline).

Why: Track 2 doubles model_dim, and the hashed n-gram table's width is model_dim (ngram_table.NGRAM_DIM),
so its per-GPU shard goes from ~10 GiB (d=512) to ~20 GiB (d=1024). ldmberman's d=1024 run peaked at
61242 MiB allocated / 70578 MiB reserved of an H100's 80 GiB with the validation batch halved.

The large terms are counted from shapes (model/gpt.py, ngram_table.py, value_embed_pull.py,
exact_counts.py). The rest -- activations, the private pools of the 12 captured CUDA step graphs,
logits, workspaces, the retrieval buffers -- is one constant, OTHER_BYTES, set so the Track 2 run's
logged peak is matched. Checked out of sample on the Track 1 log: predicted 47.6 GiB vs 47.5 GiB
logged (48643 MiB), i.e. that remainder hardly moves between d=512/11 layers and d=1024/16 layers. Both
calibration runs are assumed to have had 103 train shards on disk (the exact-count corpus is every
train shard the loader cannot reach). Treat it as a budget, not a profile: the trainer also prints
the measured allocation after setup and the measured peak at the end.

Offline:  python -m track_1_short.memory_budget --model-dim 1024 --layers 16 --mlp-hidden 4096 --val-batch 1048576 --steps 5800
(imports only this file; no torch, CUDA or extension needed).
"""
import argparse
import os

GiB = 1 << 30
VOCAB = 50304               # 50257 padded to a multiple of 128 (model/gpt.py)
NGRAM_ROWS = 84_602_880     # ngram_table.NGRAM_VOCAB_SIZE (train_gpt.py asserts they agree)
NGRAM_ROW_AUX_BYTES = 12    # per row: exp_avg_sq fp32, last_event int32, entry_of_row int32
MAX_CYCLE_STEPS = 4         # ngram_table.MAX_CYCLE_STEPS
NUM_VALUE_EMBEDS = 4        # len(model/gpt.py VALUE_EMBED_LAYERS)
NO_ATTN_LAYERS = 4          # len(model/gpt.py NO_ATTN_LAYERS)
H100_BYTES = 79.6 * GiB     # usable on an 80 GB H100 (nvidia-smi: 81559 MiB)
EXACT_TAB_BYTES = 4 * ((1 << 27) + (1 << 12)) * 8 + (1 << 16) * 8 + (1 << 21) * 8  # exact_counts.ExactCounts tab + filt
SHARD_TOKENS = 100_000_000  # tokens per FineWeb train shard (train_gpt.py's `exclude` rounds by it)
TRAIN_SHARDS = 103          # fineweb10B train shards, when all are downloaded

# Logged peaks (torch.cuda.max_memory_allocated, rank 0), 8xH100, max train batch 131072 tokens:
CALIBRATION = [
    # Track 1, records/track_1_short/2026-10-03_ExactCountChain (PR #380): d=512, 11 layers, mlp 1792, val batch 2097152
    dict(model_dim=512, layers=11, mlp_hidden=1792, val_batch=4 * 64 * 1024 * 8, total_steps=725, peak_mib=48643),
    # Track 2, ldmberman's adaptation (PR #380 thread, 2026-10-04): d=1024, 16 layers, mlp 4096, val batch 1048576
    dict(model_dim=1024, layers=16, mlp_hidden=4096, val_batch=2 * 64 * 1024 * 8, total_steps=5800, peak_mib=61242),
]


def _pad(n, m):
    return -(-n // m) * m


def dense_matrix_params(model_dim, layers, mlp_hidden, world_size=8):
    """Params of lm_head, embed, the attention banks and the MLP bank (model/gpt.py shapes)."""
    heads = model_dim // 128
    attn_layers = layers - NO_ATTN_LAYERS
    qk = _pad(attn_layers * heads, world_size) * 256 * model_dim
    vo = _pad(2 * attn_layers, world_size) * model_dim * model_dim
    mlp = (((layers + 1 + 3) // 4) * 4) * 2 * mlp_hidden * model_dim
    return 2 * VOCAB * model_dim + qk + vo + mlp


def device_static(model_dim, layers, mlp_hidden, val_batch, world_size=8, train_batch=131072):
    """Per-GPU bytes of the large allocations whose size follows from shapes."""
    t = train_batch // world_size
    tv = val_batch // world_size
    d = model_dim
    ve = NUM_VALUE_EMBEDS * VOCAB * d
    dense = dense_matrix_params(d, layers, mlp_hidden, world_size)
    return {
        # the hashed bigram/trigram table: this rank's 1/world_size of the rows, bf16, plus per-row Adam state
        "ngram table shard": NGRAM_ROWS // world_size * (2 * d + NGRAM_ROW_AUX_BYTES),
        # per-cycle grads (fp16, two rows per token per step) and the model's row cache (bf16, sized for validation)
        "ngram cycle grads + row cache": 2 * MAX_CYCLE_STEPS * t * d * 2 + 2 * tv * d * 2,
        # bf16 replica + fp32 grad accumulator + this rank's fp32 grad shard and two Adam moments
        "value_embeds": ve * 2 + ve * 4 + 3 * ve * 4 // world_size,
        # bf16 weights + bf16 grads replicated; fp32 optimizer state (two moments/velocity) on this rank's shard
        "dense matrices": dense * (2 + 2) + dense * 8 // world_size,
        # canonical mask (V x V/8 uint8), lm_head fp8 row/col copies, fit/val hidden-state buffers
        "fixed buffers": VOCAB * VOCAB // 8 + 2 * VOCAB * d + (65536 + 262144) * d * 2,
    }


def excluded_shards(total_steps, world_size=8, train_batch=131072):
    """train_gpt.py's `exclude`: shards the loader may train on, which the corpus indexes leave out."""
    return -(-int(1.5 * total_steps * (train_batch + world_size)) // SHARD_TOKENS)


def exact_count_bytes(total_steps, n_shards=TRAIN_SHARDS, world_size=8, train_batch=131072):
    """ExactCounts on one GPU: its 1/world_size of the corpus shards (int16 tokens) plus the hash table."""
    corpus_shards = max(0, n_shards - excluded_shards(total_steps, world_size, train_batch))
    return -(-corpus_shards // world_size) * SHARD_TOKENS * 2 + EXACT_TAB_BYTES


def _other_bytes():
    c = CALIBRATION[-1]
    static = device_static(c["model_dim"], c["layers"], c["mlp_hidden"], c["val_batch"])
    return c["peak_mib"] * (1 << 20) - sum(static.values()) - exact_count_bytes(c["total_steps"])


def estimate(model_dim, layers, mlp_hidden, val_batch, total_steps, n_shards=TRAIN_SHARDS, world_size=8, train_batch=131072):
    parts = device_static(model_dim, layers, mlp_hidden, val_batch, world_size, train_batch)
    parts["exact counts (corpus share + table)"] = exact_count_bytes(total_steps, n_shards, world_size, train_batch)
    parts["other (activations, graph pools; calibrated)"] = _other_bytes()
    return parts


def host_estimate(model_dim, layers, mlp_hidden, val_batch, world_size=8, local_world=8, train_batch=131072):
    """Host bytes per node that depend on the config. The Rust exact-match indexes, the exact-count corpus
    reads and the n-gram row prep scale with the token streams, not model_dim, and are not counted."""
    ve = NUM_VALUE_EMBEDS * VOCAB * model_dim
    dense = dense_matrix_params(model_dim, layers, mlp_hidden, world_size)
    return {
        # GPT() is built on the host in fp32 (value_embeds in bf16) before .cuda(), on every local rank at once
        "model build, transient (all local ranks)": local_world * (dense * 4 + ve * 2),
        # pinned row-prefetch upload rings (value_embeds rows) and validation token slot, per rank
        "pinned staging (all local ranks)": local_world * (ve // model_dim * 4 * 2 + val_batch * 2),
        # the canonical-mask build: one forked child on the master rank, a V x V bool matrix
        "canonical mask build (one child)": VOCAB * VOCAB,
    }


def mem_total_available():
    """(MemTotal, MemAvailable) in bytes from /proc/meminfo; (None, None) off Linux."""
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        return tuple(int(info[k].split()[0]) * 1024 for k in ("MemTotal", "MemAvailable"))
    except (OSError, KeyError, ValueError):
        return None, None


def report(model_dim, layers, mlp_hidden, val_batch, total_steps, n_shards=TRAIN_SHARDS, world_size=8, local_world=8,
           gpu_bytes=None):
    gpu = estimate(model_dim, layers, mlp_hidden, val_batch, total_steps, n_shards, world_size)
    host = host_estimate(model_dim, layers, mlp_hidden, val_batch, world_size, local_world)
    total_gpu = sum(gpu.values())
    cap = gpu_bytes or H100_BYTES
    lines = [f"memory budget (estimate) d={model_dim} layers={layers} mlp_hidden={mlp_hidden} val_batch={val_batch} "
             f"steps={total_steps} train_shards={n_shards} (excluded {excluded_shards(total_steps, world_size)}):"]
    lines += [f"  gpu  {k:<46s} {v / GiB:7.2f} GiB" for k, v in gpu.items()]
    lines.append(f"  gpu  {'peak allocated, estimated':<46s} {total_gpu / GiB:7.2f} GiB of {cap / GiB:.1f} GiB"
                 f"{'  WARNING: over 90% of the GPU' if total_gpu > 0.9 * cap else ''}")
    if model_dim > 512 and val_batch > CALIBRATION[-1]["val_batch"]:
        lines.append("  gpu  WARNING: the validation forward is in the calibrated constant only; ldmberman halved "
                     "VAL_BATCH_SIZE to 1048576 to fit d=1024, so a larger one likely runs out of memory")
    lines += [f"  host {k:<46s} {v / GiB:7.2f} GiB" for k, v in host.items()]
    total, avail = mem_total_available()
    if total is not None:
        lines.append(f"  host MemTotal {total / GiB:.1f} GiB, MemAvailable {avail / GiB:.1f} GiB")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dim", type=int, default=int(os.environ.get("MODEL_DIM", "1024")))
    ap.add_argument("--layers", type=int, default=int(os.environ.get("MODEL_LAYERS", "16")))
    ap.add_argument("--mlp-hidden", type=int, default=int(os.environ.get("MLP_HIDDEN_DIM", "4096")))
    ap.add_argument("--val-batch", type=int, default=int(os.environ.get("VAL_BATCH_SIZE", str(2 * 64 * 1024 * 8))))
    ap.add_argument("--steps", type=int, default=int(os.environ.get("NUM_SCHEDULED_ITERATIONS", "5790")) + 10,
                    help="total steps (scheduled + 10 extension)")
    ap.add_argument("--train-shards", type=int, default=TRAIN_SHARDS)
    a = ap.parse_args()
    print(report(a.model_dim, a.layers, a.mlp_hidden, a.val_batch, a.steps, a.train_shards))
