"""Open PRs folded onto #397, each behind its own environment variable. Every default is OFF, so an unset
environment runs #397's exact code path; the variant's README.md lists the A/B runs.

  HOST_PREP    #387 (nikhil-lalgudi): exact host-side batch preparation, a comma list or "all" (= batchplan,cands;
               ngramhash must be named, since it adds a torch.compile call site that Track 1's compile rule may not allow):
                 batchplan  Shard.next_batch plans every rank's documents from one cumulative-length array and a
                            searchsorted per rank instead of a per-document Python walk (data.py); same starts/ends
                 cands      the sampled-softmax candidate build gathers with integer indexing instead of np.take
                            into the scratch views (sampled_softmax.py); same arrays
                 ngramhash  the host n-gram row hashes as one functional expression compiled for the CPU
                            (ngram_table.ngram_row_ids_fused), primed for every batch geometry before the clock
               (#387's FA3 batch-axis change is #396's fa3copy lever, already in EXACT_LEVERS.)
  DOC_COPY     #378 (josusanmartin): the document-local copy feature. Each position gets the next token of the longest earlier
               exact match of its (normalized) context in the same document as an extra model input, embedded
               and scaled per match bucket (doc_copy.py, model/gpt.py). Changes the model: tune the step count.
  LAYER_ROPE   #385 (shikhar-srivastava): depth-conditioned rotary gains on the four branch sites (attention / MLP input and residual).
               The attention gains are folded into the QKV and O weights as in #385; the MLP gains (dense AND the
               MoE layers, which #385's base did not have) are applied to the normed MLP input and the MLP output
               instead of being folded into the fp8 bank caches (model/gpt.py). Changes the model.
               LAYER_ROPE=attn keeps the attention sites only: those are weight folds with no per-token work, while the
               MLP sites cost two elementwise passes (and their backward) over every MLP layer's activations each step.
  COPY_MIX     #376 (cyrusghane) / #382 (krishaab): the document-local copy mixture at the final validation,
                 P'(x) = (1 - lam_b) P(x) + lam_b [x == c]
               with c the token after the longest earlier exact match in the same document and b its
               (length, distance) bucket; the 31 weights are fitted ON THE CLOCK by maximum likelihood on training
               tokens (copy_mixture.py). Eval-only: training is untouched.
  HOST_ASYNC   ours (round 6): the training loader's host work (document plan, token gather, cu_seqlens, n-gram row
               hashes) runs on a background producer thread up to HOST_ASYNC_DEPTH batches ahead; the main thread only
               copies each finished batch into its pinned slot and launches the H2D, as before. Same batches, same
               order (data.AsyncBatchLoader; 182/hostcpu/README.md). Best with HOST_PREP=all: the per-document walk
               is pure Python and holds the GIL.
  COPY_MIX_FIT where the fit's training tokens come from: "consumed" (default, #382: the last training batches the
               run already trained on, concatenated into one validation-shaped batch) or "shard" (#376: the head
               of the last train shard, which the run never reaches, read by the unmodified validation loader).
"""
import os

HOST_PREP_KNOWN = ("batchplan", "cands", "ngramhash")
# What "all" means: the plain numpy rewrites only. ngramhash (a CPU torch.compile) is opt-in by name.
HOST_PREP_ALL = ("batchplan", "cands")


def _flag(name: str) -> bool:
    value = os.environ.get(name, "0").strip().lower()
    assert value in ("0", "1", "", "off", "on", "false", "true"), f"{name}={value!r}: use 0 or 1"
    return value in ("1", "on", "true")


def _subset(name: str, known: tuple[str, ...], everything: tuple[str, ...], raw: str | None = None) -> frozenset[str]:
    raw = (os.environ.get(name, "none") if raw is None else raw).replace(" ", "").lower()
    if raw in ("", "none", "off", "0"):
        return frozenset()
    chosen = frozenset(x for x in raw.split(",") if x)
    if "all" in chosen:
        chosen = frozenset(everything) | (chosen - {"all"})
    assert chosen <= set(known), f"unknown {name} {sorted(chosen - set(known))} (known: {known})"
    return chosen


HOST_PREP = _subset("HOST_PREP", HOST_PREP_KNOWN, HOST_PREP_ALL)
HOST_BATCH_PLAN = "batchplan" in HOST_PREP
HOST_CANDIDATE_GATHER = "cands" in HOST_PREP
HOST_NGRAM_HASH = "ngramhash" in HOST_PREP

def _layer_rope(raw: str | None = None) -> str:
    """"" (off), "all" (1: the four sites) or "attn" (the two attention sites only)."""
    value = (os.environ.get("LAYER_ROPE", "0") if raw is None else raw).strip().lower()
    assert value in ("0", "1", "", "off", "on", "false", "true", "all", "attn"), f"LAYER_ROPE={value!r}: use 0, 1 or attn"
    return {"1": "all", "on": "all", "true": "all", "all": "all", "attn": "attn"}.get(value, "")


HOST_ASYNC = _flag("HOST_ASYNC")
HOST_ASYNC_DEPTH = int(os.environ.get("HOST_ASYNC_DEPTH", "4"))
assert HOST_ASYNC_DEPTH >= 1, f"HOST_ASYNC_DEPTH={HOST_ASYNC_DEPTH}: at least 1"
DOC_COPY = _flag("DOC_COPY")
LAYER_ROPE_SITES = _layer_rope()
LAYER_ROPE = bool(LAYER_ROPE_SITES)            # the parameters, the attention folds, the gates' lr schedule
LAYER_ROPE_MLP = LAYER_ROPE_SITES == "all"     # the activation scales around every MLP / MoE layer
COPY_MIX = _flag("COPY_MIX")
COPY_MIX_FIT = os.environ.get("COPY_MIX_FIT", "consumed").strip().lower()
assert COPY_MIX_FIT in ("consumed", "shard"), f"COPY_MIX_FIT={COPY_MIX_FIT!r}: consumed or shard"


def describe() -> str:
    """One line for the run log."""
    return (f"HOST_PREP={','.join(sorted(HOST_PREP)) or 'none'} DOC_COPY={int(DOC_COPY)} "
            f"LAYER_ROPE={ {'': '0', 'all': '1', 'attn': 'attn'}[LAYER_ROPE_SITES]} COPY_MIX={int(COPY_MIX)}"
            + (f" COPY_MIX_FIT={COPY_MIX_FIT}" if COPY_MIX else "")
            + (f" HOST_ASYNC=1 depth={HOST_ASYNC_DEPTH}" if HOST_ASYNC else ""))
