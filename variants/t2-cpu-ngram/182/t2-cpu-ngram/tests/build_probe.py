"""Build the model (meta device) and the schedule under the current environment; print one JSON line.

Run in a subprocess per configuration, since track_1_short.config reads its knobs at import.
"""
import json
import sys
from pathlib import Path

VARIANT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_stubs  # noqa: E402

cpu_stubs.install_all(VARIANT)

import torch  # noqa: E402

from track_1_short import config  # noqa: E402
from track_1_short.model import gpt  # noqa: E402
from track_1_short.ngram_table import NGRAM_DIM  # noqa: E402
from track_1_short.sampled_softmax import candidate_count_at  # noqa: E402
from track_1_short.schedule import LR_FLOOR, TrainingSchedule  # noqa: E402

args = config.Hyperparameters()
with torch.device("meta"):
    model = gpt.GPT(vocab_size=50257, num_layers=config.MODEL_LAYERS, num_heads=config.MODEL_DIM // 128, head_dim=128,
                    model_dim=config.MODEL_DIM, max_seq_len=131072, ngram_dim=NGRAM_DIM, world_size=8,
                    device=torch.device("meta"))
schedule = TrainingSchedule(config.TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations,
                            device=torch.device("cpu"), cooldown_frac=config.LR_COOLDOWN_FRAC,
                            split_embed_stage=config.SPLIT_EMBED_STAGE, ws_post_yarn_ext=config.WS_POST_YARN_EXT)
counts = [candidate_count_at(schedule, s) for s in range(schedule.total_steps + 1)]
switches = [[0, counts[0]]] + [[s, counts[s]] for s in range(1, len(counts)) if counts[s] != counts[s - 1]]
print(json.dumps(dict(
    parameters=sum(p.numel() for p in model.parameters()),
    attn_bank_order=list(gpt.ATTN_BANK_ORDER), long_window=list(gpt.LONG_WINDOW_LAYERS),
    value_embed_layers=list(gpt.VALUE_EMBED_LAYERS), num_mlp_slots=gpt.NUM_MLP_SLOTS,
    parallel_mlp_slot=gpt.PARALLEL_MLP_SLOT, frozen=sorted(model.mlp_bank.frozen_matrices),
    mlp_bank=list(model.mlp_bank.shape), ngram_dim=NGRAM_DIM, val_batch=args.val_batch_size,
    boundaries=[list(b) for b in schedule.boundaries], switches=switches,
    lr_end=round(float(schedule.get_lr(schedule.total_steps)), 6), lr_floor=LR_FLOOR,
    cooldown=config.LR_COOLDOWN_FRAC)))
