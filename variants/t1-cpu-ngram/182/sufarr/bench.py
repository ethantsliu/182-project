"""EXACT_BACKEND=sa's memory and time on real shards, outside the trainer: each rank's share suffix sorted and the
run's queries counted, all ranks at once (one process each, SA_THREADS threads each) as in a run, and with --check-gpu
the GPU scan (ExactCounts) on the same share and queries, compared count for count.

    python 182/sufarr/bench.py --data <dir holding data/fineweb10B> [--world 8] [--threads 16] [--check-gpu]

The queries are the run's: the first 10,485,760 validation tokens in 262,144-token chunks, then world x 212,992 fit
positions in 16,384-token segments (13 fit steps of the 685-step schedule), read from the shards the loader trains on
(the first `--exclude`, which the counted corpus leaves out). One line per rank, then
`RESULT sufarr build_s=<max> query_s=<max> gib=<sum> ...`; --check-gpu adds `mismatch=<rows>` per rank (0 expected,
up to the scan's 64-bit hash collisions). --ranks 0 runs one rank alone (no contention: the floor)."""
import argparse
import glob
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COLS = 14


def queries(args, train):
    """(tok, tgt, q_chunk, start) of the run's layout; start as exact_counts.segment_starts makes it."""
    val_file = sorted(glob.glob(os.path.join(args.data, "data/fineweb10B/fineweb_val_*.bin")))[0]
    val = np.fromfile(val_file, np.uint16, args.val + 1, offset=1024).astype(np.int32)
    F = args.world * args.fit
    fit = np.concatenate([np.fromfile(f, np.uint16, offset=1024) for f in train[:args.exclude]])[:F + 1].astype(np.int32)
    tok, tgt = np.concatenate([val[:-1], fit[:-1]]), np.concatenate([val[1:], fit[1:]])
    i = np.arange(len(tok))
    q_chunk = np.where(i < args.val, i // args.chunk * args.chunk, args.val + (i - args.val) // args.fit_seg * args.fit_seg)
    start = np.maximum.accumulate(np.maximum(np.where(tok == 50256, i, 0), q_chunk)).astype(np.int64)
    return tok, tgt, q_chunk.astype(np.int64), start


def rank_main(rank, args, train, q, out_dir):
    import exact_match
    files = train[args.exclude:][rank::args.world]
    tok, tgt, q_chunk, start = q
    t = time.perf_counter()
    sc = exact_match.SuffixCounts(files, args.threads)
    build = time.perf_counter() - t
    out = np.zeros((len(tok), COLS), np.int32)
    t = time.perf_counter()
    sc.counts_into(tok, tgt, start, out)
    query = time.perf_counter() - t
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 2**30
    line = (f"rank {rank}: {sc.tokens} tokens ({len(files)} shards), build {build:.2f} s ({sc.tokens / build / 1e6:.1f} M tokens/s), "
            f"{len(tok)} queries in {query:.2f} s ({len(tok) / query / 1e6:.2f} M/s on {args.threads} threads), "
            f"index {sc.memory / 2**30:.2f} GiB, peak RSS {rss:.2f} GiB")
    mismatch = -1
    if args.check_gpu:
        import torch
        sys.path.insert(0, ROOT)
        from track_1_short import exact_counts
        dev = torch.device("cuda", rank)
        torch.cuda.set_device(dev)
        ec = exact_counts.ExactCounts(train, args.exclude, rank, args.world, dev, torch.as_tensor(q_chunk, device=dev))
        ec.load_corpus()
        ec.q_tok.copy_(torch.as_tensor(tok))
        ec.q_tgt.copy_(torch.as_tensor(tgt))
        torch.cuda.synchronize(dev)
        t = time.perf_counter()
        ec.launch()
        torch.cuda.synchronize(dev)
        gpu_s = time.perf_counter() - t
        g = ec.out.cpu().numpy()
        mismatch = int((g != out).any(1).sum())
        line += f"; GPU scan {gpu_s * 1e3:.0f} ms (after its load), mismatch={mismatch} rows"
        if mismatch:
            k = np.flatnonzero((g != out).any(1))[0]
            line += f" (first {k}: gpu {g[k].tolist()} sa {out[k].tolist()})"
    print(line, flush=True)
    np.save(os.path.join(out_dir, f"r{rank}.npy"), np.array([build, query, sc.memory, rss, mismatch]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=ROOT, help="the directory holding data/fineweb10B")
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--ranks", default="all", help="all, or a comma list (e.g. 0: one share alone)")
    p.add_argument("--threads", type=int, default=0, help="per rank (default: CPUs / world)")
    p.add_argument("--exclude", type=int, default=2, help="leading shards the loader trains on (the trainer's: 2)")
    p.add_argument("--val", type=int, default=10485760)
    p.add_argument("--chunk", type=int, default=262144)
    p.add_argument("--fit", type=int, default=212992, help="fit positions per rank")
    p.add_argument("--fit-seg", type=int, default=16384)
    p.add_argument("--check-gpu", action="store_true")
    args = p.parse_args()
    args.threads = args.threads or max(1, (os.cpu_count() or 1) // args.world)
    train = sorted(glob.glob(os.path.join(args.data, "data/fineweb10B/fineweb_train_*.bin")))
    assert len(train) > args.exclude + args.world, f"{len(train)} training shards under {args.data}/data/fineweb10B"
    ranks = range(args.world) if args.ranks == "all" else [int(r) for r in args.ranks.split(",")]
    q = queries(args, train)
    import tempfile
    import multiprocessing as mp
    with tempfile.TemporaryDirectory() as out_dir:
        ctx = mp.get_context("fork")
        t = time.perf_counter()
        procs = [ctx.Process(target=rank_main, args=(r, args, train, q, out_dir)) for r in ranks]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        wall = time.perf_counter() - t
        if any(pr.exitcode for pr in procs):
            sys.exit(f"FAIL: a rank exited with {[pr.exitcode for pr in procs]}")
        res = np.array([np.load(os.path.join(out_dir, f"r{r}.npy")) for r in ranks])
    print(f"RESULT sufarr ranks={len(ranks)} threads={args.threads} build_s={res[:, 0].max():.2f} query_s={res[:, 1].max():.2f} "
          f"index_gib={res[:, 2].sum() / 2**30:.1f} rss_gib={res[:, 3].sum():.1f} wall_s={wall:.1f}"
          + (f" mismatch={int(res[:, 4].sum())}" if args.check_gpu else ""), flush=True)


if __name__ == "__main__":
    main()
