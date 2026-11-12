"""Count the validation targets the canonical mask would drop (CPU only, ~20 s, no GPU).

    DATA_PATH=../.. python 182/t2-gpu-port/check_canon_val.py

Every (input, target) pair the final validation scores is a pair of adjacent tokens in the first
`val_tokens` (+1) tokens of the val shards, so counting masked adjacent pairs there is an exact upper bound
on the masked targets. It must print 0: a masked target gets logit -60 and costs ~80 nats instead of a few.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from medium_extras import build_canonical_mask  # noqa: E402

VOCAB = 50304
VAL_TOKENS = 10485760  # train_gpt_medium.py Hyperparameters.val_tokens


def load_shard(path: str) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, f"{path}: not a fineweb .bin shard"
    tokens = np.fromfile(path, dtype=np.uint16, offset=256 * 4)
    assert len(tokens) == header[2], f"{path}: token count does not match header"
    return tokens


def count_masked_pairs(packed: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    prev, cur = tokens[:-1].astype(np.int64), tokens[1:].astype(np.int64)
    bits = (packed[prev, cur // 8] >> (cur % 8)) & 1
    return np.flatnonzero(bits)


def main(argv):
    files = argv[1:] or sorted(glob.glob(os.path.join(os.environ.get("DATA_PATH", "."), "data/fineweb10B/fineweb_val_*.bin")))
    assert files, "no val shards found: set DATA_PATH (../.. from this variant) or pass the .bin paths"
    tokens = np.concatenate([load_shard(f) for f in files])[:VAL_TOKENS + 1]
    assert len(tokens) == VAL_TOKENS + 1, f"only {len(tokens)} val tokens"
    hits = count_masked_pairs(build_canonical_mask(VOCAB), tokens)
    print(f"{len(tokens) - 1} val pairs, {len(hits)} masked")
    for i in hits[:10]:
        print(f"  position {i}: ({tokens[i]}, {tokens[i + 1]})")
    return 1 if len(hits) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
