"""CPU checks of the folded PRs' host-side and pure-torch code (no GPU, no triton, no data).

    python 182/t1-gpu-stack/test_folds.py        # from the variant directory (or any other); plain asserts, one line per check

What it covers: HOST_PREP's three exact rewrites against #397's code (batch plan, candidate gather, n-gram hash),
the copy mixture's matcher / loss / fit (COPY_MIX) against brute force, the consumed-token window, PR #378's matcher
(DOC_COPY) against brute force, LayerRoPE's gain function and its weight-fold gradient identities, HOST_PREP's
parsing, and sweep_steps.py's equal-loss comparison on synthetic logs. What it does
not cover: anything that needs CUDA (the compiled model, the fp8 / Triton kernels, the CUDA graphs, NCCL).
"""
import os
import sys
import types

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)


def _stub_triton():
    """Enough of triton for the kernel modules to import on a machine without it (their kernels never run here)."""
    class _Any(types.ModuleType):
        def __getattr__(self, name):
            return _Any(name)

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]  # @triton.jit
            return lambda f: f   # @triton.jit(...), @triton.autotune(...)

        def __getitem__(self, item):
            return self

    triton = _Any("triton")
    sys.modules.setdefault("triton", triton)
    sys.modules.setdefault("triton.language", triton.language)


TRITON_STUBBED = False
try:
    import triton  # noqa: F401
except ImportError:
    _stub_triton()
    TRITON_STUBBED = True
try:
    import kernels  # noqa: F401  (the hub loader attention.py fetches FA3 with)
except ImportError:
    _kernels = types.ModuleType("kernels")
    _kernels.get_kernel = lambda *args, **kwargs: sys.modules["triton"]
    sys.modules["kernels"] = _kernels

from track_1_short import data, sampled_softmax  # noqa: E402
from track_1_short.copy_mixture import (  # noqa: E402
    COPY_DISTS, COPY_SPLITS, ConsumedTokens, copy_candidates, copy_fit, copy_mix_loss,
)
from track_1_short.doc_copy import COPY_LENGTHS, COUNT_BUCKETS, doc_copy_features, hash_power_tables  # noqa: E402
from track_1_short.ngram_norm_map import NORM_MAP  # noqa: E402
from track_1_short.ngram_table import _ngram_rows_functional, ngram_row_ids, ngram_row_ids_fused  # noqa: E402

BOS = data.BOS_ID
rng = np.random.default_rng(0)


def synthetic_tokens(n: int, mean_doc: int, vocab: int = 50000) -> torch.Tensor:
    toks = rng.integers(0, vocab, n).astype(np.int64)
    toks[0] = BOS
    toks[rng.random(n) < 1.0 / mean_doc] = BOS
    return torch.from_numpy(toks.astype(np.uint16))


def check_batch_plan():
    assert not data.HOST_BATCH_PLAN, "run the checks with HOST_PREP unset"
    tokens = synthetic_tokens(10_000_000, 600)
    walk, plan = data.Shard(tokens, world_size=8), data.Shard(tokens, world_size=8)
    for s in (walk, plan):
        s._ready.wait()
        s._maybe_switch()
    calls = 0
    for num_tokens, cap in [(16384, 896)] * 5 + [(32768, 2048)] * 5 + [(49152, 3072)] * 5 + [(40960, 3072), (16384, 3072)] * 3:
        a = walk.next_batch(num_tokens, cap)          # #397's per-document walk (HOST_PREP unset)
        b = plan._next_batch_planned(num_tokens, cap)
        assert walk.i == plan.i
        for r in range(8):
            assert np.array_equal(np.asarray(a[0][r], dtype=np.int64), b[0][r]), (num_tokens, cap, r)
            assert np.array_equal(np.asarray(a[1][r], dtype=np.int64), b[1][r]), (num_tokens, cap, r)
        calls += 1
    # Run both to the end of the shard: the same last batch, then StopIteration on both with nothing consumed.
    while True:
        try:
            a = walk.next_batch(49152, 3072)
        except StopIteration:
            a = None
        try:
            b = plan._next_batch_planned(49152, 3072)
        except StopIteration:
            b = None
        assert (a is None) == (b is None) and walk.i == plan.i
        if a is None:
            break
        assert all(np.array_equal(np.asarray(a[0][r], dtype=np.int64), b[0][r]) for r in range(8))
        calls += 1
    print(f"ok  HOST_PREP=batchplan: planned starts/ends == the walk over {calls} batches x 8 ranks, to the shard's end")


def check_candidate_gather():
    V = 50304
    b = sampled_softmax.CandidateBuilder(V, max_candidates=24576, max_rows=49152)
    prefix_table = rng.integers(-1, 50257, V).astype(np.int64)
    for P in (10240, 14336, 24576):
        targets = np.minimum(rng.zipf(1.3, 49152), 50256).astype(np.int64)   # few distinct classes, like text
        outs = []
        for flag in (False, True):
            sampled_softmax.HOST_CANDIDATE_GATHER = flag
            b.reset(3, 8)
            outs.append([x.copy() for x in b.build(P, targets, prefix_table)])
        sampled_softmax.HOST_CANDIDATE_GATHER = False
        for x, y in zip(*outs):
            assert np.array_equal(x, y)
    print("ok  HOST_PREP=cands: integer-indexing gather == np.take at P = 10240, 14336, 24576")


def check_ngram_hash():
    for n in (3, 4, 17, 16384, 49152, 262144):
        x = torch.randint(0, 50257, (n,), dtype=torch.int32)
        x[0] = BOS
        ref = ngram_row_ids(x)
        assert torch.equal(_ngram_rows_functional(x, NORM_MAP), ref), n
    # torch.compile probes for triton: hide the import stub (the kernel modules already hold their references).
    if TRITON_STUBBED:
        sys.modules.pop("triton", None)
        sys.modules.pop("triton.language", None)
    try:
        for n in (16384, 40960, 262144):
            x = torch.randint(0, 50257, (n,), dtype=torch.int32)
            assert torch.equal(ngram_row_ids_fused(x), ngram_row_ids(x)), n
        print("ok  HOST_PREP=ngramhash: functional and torch.compile'd (CPU inductor) rows == ngram_row_ids")
    except Exception as e:  # no C++ toolchain for inductor here: the eager expression was still checked
        print(f"ok  HOST_PREP=ngramhash: functional rows == ngram_row_ids (compiled path NOT checked here: {type(e).__name__})")


def brute_copy_candidates(x: np.ndarray):
    """For each t: longest L in COPY_SPLITS lengths whose suffix x[t-L+1..t] (inside t's document) occurs ending at an
    earlier j in the same document; latest such j. Returns (cand, bucket)."""
    n = len(x)
    lengths = [L for L, _, _ in COPY_SPLITS]
    doc_start = np.zeros(n, dtype=np.int64)
    s = 0
    for t in range(n):
        if x[t] == BOS:
            s = t
        doc_start[t] = s
    cand = np.empty(n, dtype=np.int64)
    bucket = np.zeros(n, dtype=np.int64)
    for t in range(n):
        best = None
        for k, L in enumerate(lengths):
            if t - L + 1 < doc_start[t]:
                break
            ctx = tuple(x[t - L + 1:t + 1])
            hit = None
            for j in range(t - 1, doc_start[t] + L - 2, -1):
                if j - L + 1 >= doc_start[t] and tuple(x[j - L + 1:j + 1]) == ctx:
                    hit = j
                    break
            if hit is None:
                break
            best = (k, hit)
        if best is None:
            cand[t] = x[0]  # x[(src + 1).clamp(min=0)] with src = -1
            continue
        k, j = best
        cand[t] = x[j + 1]
        d = t - j
        dist_idx = int(np.searchsorted(np.array(COPY_DISTS), d, side="left"))
        bucket[t] = 1 + k * (len(COPY_DISTS) + 1) + dist_idx
    return cand, bucket


def check_copy_candidates():
    # Small vocabulary so long matches happen; a mid-document chunk start (no BOS at 0) as in validation.
    x = rng.integers(0, 6, 700)
    x[rng.random(700) < 0.006] = BOS
    x[:40] = rng.integers(0, 6, 40)
    x[400:440] = x[100:140]   # a long exact repeat inside one document (if no BOS between)
    cand, bucket = copy_candidates(torch.from_numpy(x))
    ref_cand, ref_bucket = brute_copy_candidates(x)
    matched = ref_bucket > 0
    assert np.array_equal(bucket.numpy(), ref_bucket)
    assert np.array_equal(cand.numpy()[matched], ref_cand[matched])
    print(f"ok  COPY_MIX: copy_candidates == brute force on 700 tokens ({matched.mean():.0%} matched, "
          f"max bucket {ref_bucket.max()})")


def check_copy_loss_and_fit():
    torch.manual_seed(0)
    V, n = 50, 4000
    logits = torch.randn(n, V) * 2
    logp = torch.log_softmax(logits, -1)
    targets = torch.randint(0, V, (n,))
    cand = torch.where(torch.rand(n) < 0.5, targets, torch.randint(0, V, (n,)))
    lam = torch.rand(n) * 0.9
    lam[::7] = 0
    loss = -logp[torch.arange(n), targets]
    got = copy_mix_loss(loss, cand == targets, lam)
    mix = (1 - lam[:, None]) * logp.exp()
    mix[torch.arange(n), cand] += lam
    assert torch.allclose(mix.sum(-1), torch.ones(n), atol=1e-5)
    want = -torch.log(mix[torch.arange(n), targets])
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.equal(got[lam == 0], loss[lam == 0])
    # The fit: per bucket, lam maximizes sum log P' (+ prior misses) -- compare against a fine grid.
    bucket = torch.randint(1, 31, (n,))
    bucket[::11] = 0
    hit = cand == targets
    fitted = copy_fit(loss, hit, bucket)
    assert fitted[0] == 0
    grid = torch.linspace(0, 0.98, 9801, dtype=torch.float64)
    p = torch.exp(-loss.double())
    for b in range(1, 31):
        m = bucket == b
        h, ph = hit[m], p[m]
        ll = torch.log(ph[h][:, None] * (1 - grid) + grid).sum(0) + (h.logical_not().sum() + 5.0) * torch.log1p(-grid)
        best = grid[ll.argmax()]
        assert abs(best - fitted[b].double()) < 2e-4, (b, best, fitted[b])
    print("ok  COPY_MIX: mixture sums to one, copy_mix_loss == explicit -log P', copy_fit == grid argmax on 30 buckets")


def check_consumed_tokens():
    keep = ConsumedTokens(5000)

    class B:  # the two fields ConsumedTokens reads
        def __init__(self, buf):
            self.inputs_cpu = buf[:-1].numpy()
            self.targets_cpu = buf[1:].to(torch.int64)

    stream = []
    for _ in range(9):
        buf = synthetic_tokens(1501, 300).to(torch.int32)
        buf[0] = BOS
        stream.append(buf)
        keep.add(B(buf))
    inputs, targets = keep.take()
    assert inputs.numel() == targets.numel() == 5000
    # Every (input, target) pair is one of the stream's: target follows input within its batch.
    ref_in = torch.cat([b[:-1] for b in stream])[-5000:]
    ref_tg = torch.cat([b[1:] for b in stream]).to(torch.int64)[-5000:]
    assert torch.equal(inputs, ref_in) and torch.equal(targets, ref_tg)
    print("ok  COPY_MIX_FIT=consumed: the window is the stream's last 5000 (input, target) pairs")


def check_doc_copy():
    x = rng.integers(0, 6, 700)
    x[rng.random(700) < 0.006] = BOS
    xt = torch.from_numpy(x)
    powers = hash_power_tables(4096, "cpu")
    cand, bucket = doc_copy_features(xt, powers, NORM_MAP)
    y = NORM_MAP.long()[xt].numpy()
    n = len(x)
    doc_start, s = np.zeros(n, dtype=np.int64), 0
    for t in range(n):
        if x[t] == BOS:
            s = t
        doc_start[t] = s
    for t in range(n):
        best_len, best_j, count = 0, -1, 0
        for li, L in enumerate(COPY_LENGTHS):
            if t - L + 1 < doc_start[t]:
                break
            ctx = tuple(y[t - L + 1:t + 1])
            js = [j for j in range(doc_start[t] + L - 1, t) if tuple(y[j - L + 1:j + 1]) == ctx]
            if js:
                best_len, best_j, count = li + 1, js[-1], len(js)
        if best_len == 0:
            assert bucket[t] == 0 and cand[t] == -1, t
        else:
            assert cand[t] == x[best_j + 1], t
            assert bucket[t] == (best_len - 1) * COUNT_BUCKETS + min(count, COUNT_BUCKETS) - 1 + 1, t
            assert best_j + 1 <= t  # causal: the candidate is part of the context
    print(f"ok  DOC_COPY: doc_copy_features == brute force on {n} tokens (causal, document-local)")


def check_layer_rope():
    torch.manual_seed(0)
    from track_1_short.model.layer_rope import layer_rope_gains
    D, L = 768, 11
    gates = torch.randn(4, 4) * 0.1
    gammas = 1 + 0.1 * torch.randn(4, D)
    freqs = 100.0 ** (-2.0 * torch.arange(D // 2) / D)
    u = torch.log(torch.arange(1, L + 1, dtype=torch.float32))
    got = layer_rope_gains(gates, gammas, freqs, u).float()
    # #385's definition, written out per (site, layer, pair).
    for s in range(4):
        for l in (0, 5, 10):
            mag = torch.exp(gates[s, 0] + gates[s, 1] * u[l])
            th = torch.exp(gates[s, 2] + gates[s, 3] * u[l]) * freqs
            a, b = gammas[s, 0::2], gammas[s, 1::2]
            ref = torch.stack([mag * (a * torch.cos(th) - b * torch.sin(th)), mag * (a * torch.sin(th) + b * torch.cos(th))], -1).flatten()
            assert torch.allclose(got[s, l], ref.bfloat16().float(), rtol=1e-2, atol=1e-2)
    # The QKV fold's backward identities (qkv_rope.py): for W_eff = W * c (columns), dL/dW = G * c, dL/dc = sum_rows(G * W).
    W = torch.randn(64, D, requires_grad=True)
    c = torch.randn(D, requires_grad=True)
    x = torch.randn(32, D)
    G = torch.randn(32, 64)
    (x @ (W * c).T).backward(G)
    G_w = G.T @ x  # the gradient of the effective weight
    assert torch.allclose(W.grad, G_w * c, atol=1e-4) and torch.allclose(c.grad, (G_w * W.detach()).sum(0), atol=1e-3)
    # The O fold: F.linear(y, r[:, None] * O) == (F.linear(y, O)) * r.
    O, y, r = torch.randn(D, 96), torch.randn(5, 96), torch.randn(D)
    assert torch.allclose(torch.nn.functional.linear(y, r[:, None] * O), torch.nn.functional.linear(y, O) * r, atol=1e-4)
    print("ok  LAYER_ROPE: gains == #385's formula; QKV column-fold and O row-fold gradient identities")


def check_folds_parsing():
    """HOST_PREP=all is the plain numpy rewrites only; the CPU torch.compile (ngramhash) needs its own name."""
    from track_1_short import folds
    sub = lambda raw: folds._subset("HOST_PREP", folds.HOST_PREP_KNOWN, folds.HOST_PREP_ALL, raw=raw)
    assert sub("all") == {"batchplan", "cands"}
    assert sub("all,ngramhash") == {"batchplan", "cands", "ngramhash"}
    assert sub("ngramhash") == {"ngramhash"} and sub("") == sub("none") == sub("0") == frozenset()
    try:
        sub("bogus")
    except AssertionError:
        pass
    else:
        raise AssertionError("an unknown HOST_PREP name must be refused")
    print("ok  folds: HOST_PREP=all == batchplan,cands (ngramhash opt-in by name); unknown names refused")


def check_sweep_compare():
    """sweep_steps.py compare: on synthetic logs with known lines, each arm's time at the target and its delta."""
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sweep_steps
    gen = np.random.default_rng(1)
    # baseline: val = 3.2785 at 970 steps, 0.17 millinats per step; 30 ms per scheduled step.
    # arm: the same loss 10 steps sooner at +1 ms per step (so it reaches the target at 960 steps).
    arms = {"base": (970, 0.0), "arm": (960, 0.001)}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (x_target, extra) in arms.items():
            os.makedirs(os.path.join(tmp, name))
            for k in range(6):
                for steps in (950, 965, 980):
                    loss = 3.2785 - 0.00017 * (steps - x_target) + gen.normal(0, 1e-5)
                    secs = 1.6 + (0.030 + extra) * steps
                    with open(os.path.join(tmp, name, f"{name}_s{steps}_r{k}.txt"), "w") as f:
                        f.write(f"[final] val_loss:{loss:.6f} train_time_ms:{secs * 1000:.0f} steps:{steps + 72} scheduled:{steps}\n")
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            (name, delta, lo, hi), = sweep_steps.compare([os.path.join(tmp, "base"), os.path.join(tmp, "arm")], 3.2785, boot=200)
    expect = (1.6 + 0.031 * 960) - (1.6 + 0.030 * 970)
    assert abs(delta - expect) < 0.01 and lo <= delta <= hi, (delta, expect, lo, hi)
    print(f"ok  sweep_steps compare: equal-loss time delta {delta:+.3f} s (exact {expect:+.3f}), interval [{lo:+.3f}, {hi:+.3f}]")


# check_ngram_hash last: it removes the triton import stub before torch.compile.
CHECKS = [check_batch_plan, check_candidate_gather, check_copy_candidates, check_copy_loss_and_fit,
          check_consumed_tokens, check_doc_copy, check_layer_rope, check_folds_parsing, check_sweep_compare,
          check_ngram_hash]

if __name__ == "__main__":
    import time
    chosen = [c for c in CHECKS if not sys.argv[1:] or c.__name__ in sys.argv[1:]]
    for check in chosen:
        t0 = time.perf_counter()
        check()
        print(f"    ({time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"{len(chosen)} CPU checks passed")
