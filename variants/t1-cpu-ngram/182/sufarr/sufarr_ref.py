"""The exact reference both backends are tested against, and the test cases (pure numpy, no extension, no triton).

reference() is ExactCounts._count (track_1_short/exact_counts.py) with exact gram keys in place of its 64-bit hashes:
the query table holds what _ins_kernel inserts (each query's grams that end with its target, inside its segment),
the scan of each share adds an occurrence of a gram of length 2..9 only while all its shorter suffixes are in the
table (`alive = found`), and _find_kernel reads a context's count and its context + target's count from the table
(0 where the gram is not in it). Summed over the shares, as the trainer's all-reduce does.

cases() gives (shares, files per share, queries) the way the trainer makes them: a validation stream cut into
chunks and fit segments, segment starts at the last BOS (exact_counts.segment_starts), on random BOS-led corpora and
on a real sample of fineweb10B (testdata/fineweb_sample.npz: 6 x 12,000 tokens from 3 training shards, 6,000
validation tokens)."""
import os

import numpy as np

MAXO, COLS, BOS = 8, 14, 50256
HERE = os.path.dirname(os.path.abspath(__file__))


def segment_starts(tok, chunk):  # exact_counts.segment_starts: the latest of the chunk start and the last BOS at or before
    idx = np.arange(len(tok))
    return np.maximum.accumulate(np.maximum(np.where(tok == BOS, idx, 0), chunk)).astype(np.int64)


def reference(shares, tok, tgt, start):
    n = len(tok)
    tok, tgt, start = (np.asarray(a).tolist() for a in (tok, tgt, start))
    table = {}
    for j in range(n):
        for m in range(1, MAXO + 1):
            if j - (m - 1) < start[j]:
                break
            table[tuple(tok[j - m + 1:j + 1]) + (tgt[j],)] = 0
    for share in shares:
        s = np.asarray(share).tolist()
        for p in range(len(s)):
            for L in range(2, MAXO + 2):
                if p + 1 < L:
                    break
                g = tuple(s[p - L + 1:p + 1])
                if g not in table:
                    break
                table[g] += 1
    out = np.zeros((n, COLS), np.int32)
    for i in range(n):
        for m in range(2, MAXO + 1):
            if i - (m - 1) < start[i]:
                break
            ctx = tuple(tok[i - m + 1:i + 1])
            out[i, m - 2] = table.get(ctx, 0)
            out[i, MAXO + m - 3] = table.get(ctx + (tgt[i],), 0)
    return out


def _corpus(rng, n, vocab):  # BOS-led documents over a small vocabulary with repeated phrases, so long grams recur
    phrases = [rng.integers(0, vocab, rng.integers(2, 12)) for _ in range(20)]
    out, size = [], 0
    while size < n:
        r = rng.integers(10)
        piece = np.array([BOS]) if r == 0 else phrases[rng.integers(20)] if r <= 4 else rng.integers(0, vocab, 1)
        out.append(piece)
        size += len(piece)
    return np.concatenate(out)[:n].astype(np.uint16)


def queries(rng, val, fit_src, world, chunk, F, broken=False):
    """The trainer's queries: the validation stream (target = next token) in chunks, then world * F fit positions in
    segments of F // 4 (a fit step's), drawn from fit_src; `broken`: some targets are not the next token."""
    V = len(val) - 1
    i = np.arange(V + world * F)
    S = F // 4
    q_chunk = np.where(i < V, i // chunk * chunk, V + (i - V) // S * S)
    at = rng.integers(0, len(fit_src) - world * F - 1)
    fit = fit_src[at:at + world * F + 1].astype(np.int32)
    tok = np.concatenate([val[:V].astype(np.int32), fit[:-1]])
    tgt = np.concatenate([val[1:].astype(np.int32), fit[1:]])
    if broken:
        k = rng.integers(0, len(tgt), len(tgt) // 7)
        tgt[k] = rng.integers(0, int(tok.max()) + 2, len(k))
    return tok, tgt, q_chunk.astype(np.int64)


def cases():
    """(name, world, files [world][files of the share], tok, tgt, q_chunk); a share is its files concatenated. Every
    case has at least 4096 queries (ExactCounts' constructor compiles its kernels on 4096 of them)."""
    rng = np.random.default_rng(2026)
    out = []
    for vocab, broken in ((6, False), (40, True), (50257, False)):
        files = [[_corpus(rng, int(rng.integers(1500, 4000)), vocab) for _ in range(3)] for _ in range(4)]
        allc = np.concatenate([f for share in files for f in share])
        tok, tgt, q_chunk = queries(rng, _corpus(rng, 4097, vocab), allc, 4, 64, 160, broken)
        out.append((f"random-v{vocab}{'-broken' if broken else ''}", 4, files, tok, tgt, q_chunk))
    d = np.load(os.path.join(HERE, "testdata", "fineweb_sample.npz"))
    train, val = d["train"], d["val"]
    for world in (2, 3):  # files[rank::world], as ExactCounts splits the sorted shard list
        files = [[train[f] for f in range(r, len(train), world)] for r in range(world)]
        tok, tgt, q_chunk = queries(rng, val, np.concatenate(list(train)), world, 1024, 400)
        out.append((f"fineweb-w{world}", world, files, tok, tgt, q_chunk))
    return out


def write_shard(path, tokens):  # a shard file as the trainer reads it: the 1024-byte header, then uint16 tokens
    header = np.zeros(256, np.int32)
    header[:3] = 20240520, 1, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, np.uint16).tobytes())
