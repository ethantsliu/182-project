"""The final validation's memory: what train_gpt.py drops before the forward, and the forward's out-of-memory retry.

Both run after the clock stops. The t2-cpu-ngram run on a node (torch 2.11, not the pinned 2.10) trained all 5800
steps and then ran out of CUDA memory in this forward (73 GiB in use, 5.7 GiB of it in CUDA-graph pools). Neither
changes what is measured: the release drops only buffers the validation never reads, and the retry recomputes the
same batch with smaller loss slabs, which are row-local, so the same tokens give the same mean up to float rounding
(182/t2-cpu-ngram/tests/test_final_val.py: eager bit for bit; CPU inductor within 4e-6 per token, 2e-8 on the mean).
"""
import gc
import os

import torch

from track_1_short import memory_budget
from track_1_short.model import gpt as gpt_module

# Off the clock, before the final validation forward: drop the buffers only training and the chain's fit read
# (the exact-count corpus and table, the fit steps' hidden states, the step graphs and their pool, the gradients).
# 0 keeps them, as before. The validation reads none of them, so the val loss is the same either way.
FINAL_VAL_RELEASE = os.environ.get("FINAL_VAL_RELEASE", "1") != "0"
# The final validation's out-of-memory fallback halves the loss slab (model/gpt.py EVAL_CE_SLAB_ROWS) down to this.
EVAL_CE_MIN_SLAB_ROWS = 2048
GiB = memory_budget.GiB


def gpu_memory_line(what: str, who: str = "rank 0") -> str:
    return (f"{what} (measured, {who}): allocated {torch.cuda.memory_allocated() / GiB:.2f} GiB "
            f"reserved {torch.cuda.memory_reserved() / GiB:.2f} GiB")


def release_for_final_validation(exact, step_graphs, ngram_table, model, before: int) -> str:
    """Drop what training and the chain's fit needed and the validation does not, with the device synchronized
    and the clock stopped. The caller has already dropped its own references (the fit steps' tensors); `before`
    is torch.cuda.memory_allocated() from before it did."""
    exact_bytes = exact.release()
    graphs = step_graphs.release()
    ngram_table.release_training_state()
    model.zero_grad(set_to_none=True)  # p.grad points into the step graphs' pool (or the Adam accumulators)
    gc.unfreeze()  # the step-graph objects predate the gc.freeze() before the clock
    gc.collect()  # the timed loop runs with gc disabled
    torch.cuda.empty_cache()  # returns the freed blocks, and the private pool once nothing uses it
    return (f"final validation memory release: {(before - torch.cuda.memory_allocated()) / GiB:.2f} GiB freed "
            f"(exact counts {exact_bytes / GiB:.2f} GiB, {graphs} step graphs and their pool, gradients, fit buffers); "
            + gpu_memory_line("now"))


def eval_forward(model, *args, **kwargs):
    """The validation forward. On a CUDA out-of-memory error it retries the same batch with half the loss
    slab: the slab loop is row-local, so the batch's per-token losses, and their mean, are the same up to float
    rounding (182/t2-cpu-ngram/tests/test_final_val.py). Each new slab size recompiles the eval graph (off the clock). No collective runs
    inside the forward, so a rank that retries does not stall the others."""
    while True:
        try:
            return model(*args, **kwargs)
        except torch.OutOfMemoryError:
            rows = gpt_module.EVAL_CE_SLAB_ROWS
            if rows <= EVAL_CE_MIN_SLAB_ROWS:
                raise
        # outside the except block, so the failed attempt's frames (and their tensors) are gone
        gpt_module.EVAL_CE_SLAB_ROWS = rows // 2
        gc.collect()
        torch.cuda.empty_cache()
        rank = os.environ.get("RANK", "0")
        print(f"final validation: CUDA out of memory with {rows}-row loss slabs on rank {rank}; retrying the batch with "
              f"{rows // 2}-row slabs (the eval graph recompiles); " + gpu_memory_line("now", f"rank {rank}"), flush=True)
