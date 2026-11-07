"""Document-local copy mixture at the final validation (COPY_MIX=1; PR #376 by @cghane, fit source of PR #382 by
@krishaab).

    P'(x) = (1 - lam_b) * P(x) + lam_b * [x == c]

P is the model's validation distribution (here CPLM's mixture under the canonical mask). c is the token that followed
the most recent earlier occurrence, in the same document, of the longest suffix of the context (lengths COPY_SPLITS)
that occurs earlier at all; b buckets that match by length and by distance (COPY_DISTS). c and b depend on the inputs
up to the current position only and P' sums to one, so P' is still a causal probability model. Everything runs
eagerly on the per-token loss the eval forward already returns: the model and its compiled graphs are untouched.

The 31 weights lam_b are fitted parameters, so fitting them is training and runs on the clock: maximum likelihood on
training tokens, with eval-mode forwards only (no backward), after the final weights are in place. Where the tokens
come from (COPY_MIX_FIT, folds.py):
  consumed  the last training tokens the run consumed (#382): the tail of the training stream, concatenated into one
            validation-shaped batch per rank. Nothing the run did not already read.
  shard     the head of the last train shard (#376), which a run never reaches, read as the validation loader reads.
Either way the fit batch has the validation batch's shape, so it runs the validation's compiled eval graph, which
precompile_eval() compiles in the warmup (the final validation would otherwise compile it, off the clock).

What CPLM changes: CPLM (perf/kernels/cplm.py) already mixes in a learned pointer over the previous 8192 tokens of the
document at validation, which covers most of the copy rule's long-distance matches (#376 found 79% of its gain more
than 2560 tokens back, beyond every attention window but inside CPLM's band). The fit gives a bucket the model already
copies well lam = 0 exactly, so the mixture can only help on the fit tokens; how much survives on top of CPLM is
what the A/B (182/t1-gpu-stack/README.md) measures.
"""
from collections import deque

import torch
import torch.distributed as dist
from torch import Tensor

from track_1_short.data import BOS_ID, eval_batch_from_tokens, read_shard_head

# Match lengths tried, each assembled from two shorter ones: (length, left part, right part).
COPY_SPLITS = ((1, 0, 0), (2, 1, 1), (3, 2, 1), (4, 2, 2), (6, 4, 2), (8, 4, 4), (12, 8, 4), (16, 8, 8),
               (24, 16, 8), (32, 16, 16))
COPY_DISTS = (768, 2560)  # distance split at #376's final short and long attention windows, in tokens
COPY_NUM_BUCKETS = 1 + len(COPY_SPLITS) * (len(COPY_DISTS) + 1)  # 31; bucket 0: no match


def _rank_and_prev(key: Tensor):
    """Dense rank of every key, and the latest earlier position holding the same key (-1 if none)."""
    sorted_key, order = torch.sort(key, stable=True)  # stable: equal keys stay in position order
    first = torch.ones_like(sorted_key, dtype=torch.bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    rank, prev = torch.empty_like(order), torch.empty_like(order)
    rank[order] = torch.cumsum(first, 0) - 1
    prev[order] = torch.where(first, -1, order.roll(1))
    return rank, prev


@torch.no_grad()
def copy_candidates(inputs: Tensor) -> tuple[Tensor, Tensor]:
    """(candidate, bucket) per position, from inputs[:t+1] alone; targets are never read.

    Exact, no hashing: the rank of an L-gram is the rank of the pair (rank of its left part, rank of its right
    part), so equal ranks mean equal token strings. The document index is folded into the length-1 rank, which keeps
    every longer match inside one document; a chunk that opens mid-document is its own document."""
    assert inputs.ndim == 1
    N = inputs.numel()
    x = inputs.long()
    ar = torch.arange(N, device=x.device)
    is_bos = x == BOS_ID
    pos_in_doc = ar - torch.cummax(ar * is_bos, 0).values
    ranks = {}
    src = torch.full_like(ar, -1)
    len_idx = torch.zeros_like(ar)
    for k, (L, left, right) in enumerate(COPY_SPLITS):
        if L == 1:
            key = torch.cumsum(is_bos, 0) * 65536 + x
        else:
            key = ranks[left].roll(right) * N + ranks[right]
            key = torch.where(pos_in_doc >= L - 1, key, -1 - ar)  # an L-gram cut by the document start matches nothing
        ranks[L], prev = _rank_and_prev(key)
        found = prev >= 0  # nested: a match at L implies one at every shorter length
        src = torch.where(found, prev, src)
        len_idx = torch.where(found, k, len_idx)
    matched = src >= 0
    cand = x[(src + 1).clamp(min=0)]  # src + 1 <= t: the copied token is itself part of the context
    dist_idx = torch.bucketize(ar - src, torch.tensor(COPY_DISTS, device=x.device))
    bucket = torch.where(matched, 1 + len_idx * (len(COPY_DISTS) + 1) + dist_idx, 0)
    return cand, bucket


def copy_mix_loss(loss: Tensor, hit: Tensor, lam: Tensor) -> Tensor:
    """-log P'(target) from the model's -log P(target) and whether the target is the copied token; lam per token.
    Where lam is 0 this returns `loss` bit for bit."""
    lam = lam.to(loss.dtype)
    keep = torch.log1p(-lam)
    both = -torch.logaddexp(keep - loss, torch.log(lam))
    mixed = torch.where(hit & (lam > 0), both, loss - keep)
    return torch.where(lam == 0, loss, mixed)


@torch.no_grad()
def copy_fit(loss: Tensor, hit: Tensor, bucket: Tensor, iters: int = 40, lam_max: float = 0.98,
             prior_miss: float = 5.0) -> Tensor:
    """Maximum-likelihood mixture weight per bucket, fp64. The log-likelihood is concave in lam, with derivative
    sum_hits (1-p)/(p + lam (1-p)) - n_miss/(1-lam), so bisect on its sign; a bucket the copy rule cannot help keeps
    lam at exactly 0. prior_miss phantom misses per bucket pull thinly populated buckets toward 0 (the model's own
    loss). Sums are all-reduced, so every rank ends with identical weights."""
    p = torch.exp(-loss.double())
    n_miss = torch.bincount(bucket[~hit], minlength=COPY_NUM_BUCKETS).double()
    if dist.is_initialized():
        dist.all_reduce(n_miss)
    n_miss += prior_miss
    hit_bucket, hit_p = bucket[hit], p[hit]
    lo, hi = torch.zeros_like(n_miss), torch.full_like(n_miss, lam_max)
    for _ in range(iters):
        mid = (lo + hi) / 2
        grad = torch.zeros_like(n_miss).index_add_(0, hit_bucket, (1 - hit_p) / (hit_p + mid[hit_bucket] * (1 - hit_p)))
        if dist.is_initialized():
            dist.all_reduce(grad)
        up = grad > n_miss / (1 - mid)
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    lo[0] = 0.0  # no match, nothing to copy
    return lo.float()


class ConsumedTokens:
    """The tail of this rank's training stream (COPY_MIX_FIT=consumed): the host inputs and targets of the last
    batches train_step consumed, enough for one validation-shaped batch. Host references only: a Batch's
    inputs_cpu / targets_cpu are fresh per batch, never a reused pinned slot."""
    def __init__(self, num_tokens: int):
        self.num_tokens = num_tokens
        self.parts: deque[tuple[Tensor, Tensor]] = deque()
        self.held = 0

    def add(self, batch) -> None:
        inputs = torch.from_numpy(batch.inputs_cpu)
        self.parts.append((inputs, batch.targets_cpu))
        self.held += inputs.numel()
        while self.held - self.parts[0][0].numel() >= self.num_tokens:
            self.held -= self.parts.popleft()[0].numel()

    def take(self) -> tuple[Tensor, Tensor]:
        """The last num_tokens (inputs, targets). Within a training batch targets[i] == inputs[i + 1]; each batch
        opens on a BOS, so the concatenation keeps every (input, target) pair of the stream."""
        assert self.held >= self.num_tokens, f"only {self.held} consumed tokens for a {self.num_tokens}-token fit"
        inputs = torch.cat([p[0] for p in self.parts])[-self.num_tokens:]
        targets = torch.cat([p[1] for p in self.parts])[-self.num_tokens:]
        self.parts.clear()
        return inputs, targets


def shard_head_tokens(train_files: list[str], num_tokens_local: int, rank: int, world_size: int):
    """COPY_MIX_FIT=shard: this rank's slice of the head of the last train shard, cut as the unaligned validation
    loader cuts its first batch."""
    tokens = read_shard_head(sorted(train_files)[-1], world_size * num_tokens_local + 1)
    buf = tokens[rank * num_tokens_local: (rank + 1) * num_tokens_local + 1]
    return buf[:-1], buf[1:]


@torch.no_grad()
def fit_copy_mixture(model, inputs: Tensor, targets: Tensor, staging, ngram_table, forward_args) -> Tensor:
    """On the clock, at the final validation's state (final weights, windows, canonical mask, complete tables): one
    validation-shaped eval forward over the fit tokens, its copy features, and the weights. Returns lam [31] fp32,
    identical on every rank. Leaves the n-gram cache holding the fit batch's rows (the validation lands its own)."""
    batch = eval_batch_from_tokens(inputs, targets, staging)
    pull, = ngram_table.eval_pulls([batch.ngram_ids])
    ngram_table.land(pull)
    was_training = model.training
    model.eval()
    loss = model(batch.inputs, batch.targets, batch.cum_seqlens, ngram_table.slots(pull, batch.ngram_ids), forward_args)
    model.train(was_training)
    cand, bucket = copy_candidates(batch.inputs)
    return copy_fit(loss.float(), cand == batch.targets, bucket)


@torch.no_grad()
def precompile_eval(model, staging, num_tokens_local: int, forward_args) -> None:
    """Warmup only: one eval forward on synthetic validation-shaped tokens, so the graph the on-clock fit (and the
    final validation) runs is compiled before the clock. Reads the n-gram cache's first rows (any rows: only
    shapes matter) and changes no state the timed run reads, except the CPLM diagnostics, which every validation
    zeroes first."""
    t = torch.arange(num_tokens_local, dtype=torch.int64)
    inputs = (t * 7919) % 50000
    inputs[::1024] = BOS_ID  # validation-like document count
    targets = inputs.roll(-1)
    batch = eval_batch_from_tokens(inputs.to(torch.int32), targets, staging)
    slots = torch.zeros_like(batch.ngram_ids)
    was_training = model.training
    model.eval()
    model(batch.inputs, batch.targets, batch.cum_seqlens, slots, forward_args)
    model.train(was_training)
    torch.cuda.synchronize()
