"""Per-step host work of t1-gpu-stack at the record's shapes, timed on this CPU (no GPU needed).

    python 182/hostcpu/bench_host.py                 # synthetic FineWeb-like tokens (~690-token documents)
    python 182/hostcpu/bench_host.py --shard data/fineweb10B/fineweb_train_000001.bin   # real tokens
    python 182/hostcpu/bench_host.py --quick         # fewer repeats

One torch thread, as under torchrun (it sets OMP_NUM_THREADS=1). For every host op a training step runs, it prints
the median time per call at each stage's geometry (tokens per rank, sequence cap, sampled-softmax P), which thread
runs it today, and the whole run's total over the 1053-step schedule. Then the GIL test: how long a main-thread
loop of small dispatch-like calls takes per iteration while a producer thread builds batches nonstop (walk vs plan)
and paced at one batch per GPU step (HOST_ASYNC=1's steady state).

What it times:
  main thread, every step (what HOST_ASYNC=1 moves to the producer):
    plan       Shard.next_batch: every rank's starts / ends (walk = #397, plan = HOST_PREP=batchplan)
    gather     the token gather (torch.cat of the documents), targets shift, cum_lengths, attention segments
    prep       _prepare_batch: the cu_seqlens table, dtype casts, ngram_row_ids (the host n-gram row hash)
  main thread, every step (stays there):
    pincopy    the four host -> pinned-slot copies of PinnedBatchStaging.upload (here into plain buffers)
  main thread, every event step (2-4 steps):
    request    RowOwners.request's host part for both tables (searchsorted, diff, tolist)
  prep threads today (only their join is on the main thread):
    cands      CandidateBuilder.build (sampled-softmax steps only)
    wants      cycle_wants: sorted_unique_rows of a cycle's n-gram ids + value_embed_rows
Not timed: Python dispatch of the graph replays and the optimizer (needs CUDA), the H2D launches, gloo / NCCL.
"""
import argparse
import os
import statistics
import sys
import tempfile
import threading
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_host_async as harness  # noqa: E402  (stubs triton / kernels, imports the variant)

from track_1_short import data  # noqa: E402
from track_1_short.config import TRAINING_STAGES  # noqa: E402
from track_1_short.ngram_table import MAX_CYCLE_STEPS, NGRAM_VOCAB_SIZE, is_update_step, ngram_row_ids, sorted_unique_rows  # noqa: E402
from track_1_short.sampled_softmax import CandidateBuilder, candidate_count_at  # noqa: E402
from track_1_short.schedule import TrainingSchedule  # noqa: E402

WORLD = 8
VOCAB = 50304
# model/gpt.py VALUE_EMBED_LAYERS = (1, 2, 8, 10): four planes (gpt.py loads FA3 at import, so not imported here).
VALUE_EMBED_PLANES = 4
GPU_STEP_S = 30.957 / 1053  # the node's #397 run: 30.957 s over 1053 steps


def value_embed_rows(token_arrays, vocab_size: int, num_planes: int) -> np.ndarray:
    """perf/value_embed_pull.py's value_embed_rows, copied: importing that module pulls in optim/ and every kernel."""
    seen = np.zeros(vocab_size, dtype=bool)
    for tokens in token_arrays:
        seen[tokens] = True
    distinct = np.flatnonzero(seen).astype(np.int32)
    return np.concatenate([distinct + np.int32(p * vocab_size) for p in range(num_planes)])


def median_ms(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times)


def synthetic_tokens(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    # Document lengths: lognormal with a ~690-token mean (the #387 sample: 6M tokens, 8.7k documents) and a long tail.
    lengths = np.maximum(2, rng.lognormal(mean=5.9, sigma=1.0, size=n // 300).astype(np.int64))
    ends = np.cumsum(lengths)
    ends = ends[ends < n]
    toks = rng.zipf(1.2, n).clip(0, 50255).astype(np.uint16)  # word-like frequencies, no stray BOS
    toks[0] = data.BOS_ID
    toks[ends] = data.BOS_ID
    return torch.from_numpy(toks)


def read_tokens(path: str, n: int) -> torch.Tensor:
    return data.read_shard_head(path, n)


def stage_table(sched: TrainingSchedule):
    """(stage index, steps, tokens per rank, cap, P of the stage's first step) for each stage."""
    rows = []
    for i, (start, end) in enumerate(sched.boundaries):
        stage = TRAINING_STAGES[i]
        ps = sorted({candidate_count_at(sched, s) for s in range(start, end)})
        rows.append((i, end - start, stage.batch_size // WORLD, stage.train_max_seq_len, ps))
    return rows


def time_loader_ops(tokens: torch.Tensor, local: int, cap: int, repeats: int, plan: bool) -> dict:
    """Median ms of plan / gather / prep for one geometry, and of the whole host_batch_generator step."""
    data.HOST_BATCH_PLAN = plan
    shard = data.Shard(tokens, WORLD)
    shard._ready.wait()
    shard._maybe_switch()
    out = {}
    # plan: fresh positions each call (walks forward through the shard; wraps by resetting the cursor)
    def plan_call():
        try:
            return shard.next_batch(local, cap)
        except StopIteration:
            shard.i = 0
            return shard.next_batch(local, cap)
    out["plan"] = median_ms(plan_call, repeats)
    starts, ends = plan_call()
    s, e = torch.tensor(starts[3]), torch.tensor(ends[3])

    def gather_call():
        start_idxs, end_idxs = s, e.clone()
        buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
        _inputs, _targets = buf[:-1], buf[1:]
        end_idxs[-1] -= 1
        cum = (end_idxs - start_idxs).cumsum(0)
        if cap > data.VIRTUAL_SEQ_CAP:
            cum = data.split_attention_segments(cum, data.VIRTUAL_SEQ_CAP)
        return _inputs, _targets, cum
    out["gather"] = median_ms(gather_call, repeats)
    _inputs, _targets, cum = gather_call()
    rows = data.cu_seqlens_rows(local)
    out["prep"] = median_ms(lambda: data._prepare_batch(_inputs, _targets, cum, local, rows), repeats)
    out["hash"] = median_ms(lambda: ngram_row_ids(_inputs.to(torch.int32)), repeats)
    host = data._prepare_batch(_inputs, _targets, cum, local, rows)
    bufs = [torch.empty(t.numel(), dtype=t.dtype) for t in host]

    def pincopy():
        for t, b in zip(host, bufs):
            b[:t.numel()].copy_(t)
    out["pincopy"] = median_ms(pincopy, repeats)
    data.HOST_BATCH_PLAN = False
    return out


def time_prep_threads(tokens: torch.Tensor, local: int, cap: int, P: int, repeats: int) -> dict:
    """cands at P (0 = full softmax: not built) and a 4-step cycle's wants, at one geometry."""
    rng = np.random.default_rng(1)
    starts = rng.integers(0, tokens.numel() - local - 1, MAX_CYCLE_STEPS)
    batches = [tokens[s:s + local + 1].to(torch.int64) for s in starts]
    out = {}
    if P:
        builder = CandidateBuilder(VOCAB, max_candidates=P, max_rows=local)
        builder.reset(3, WORLD)
        prefix = rng.integers(-1, 50257, VOCAB).astype(np.int64)
        targets = batches[0][1:].numpy()
        out["cands"] = median_ms(lambda: builder.build(P, targets, prefix), repeats)
    ids = [ngram_row_ids(b[:-1].to(torch.int32)).numpy() for b in batches]
    inputs = [b[:-1].to(torch.int32).numpy() for b in batches]
    planes = VALUE_EMBED_PLANES
    out["wants"] = median_ms(lambda: (sorted_unique_rows(ids), value_embed_rows(inputs, VOCAB, planes)), max(3, repeats // 4))
    want = sorted_unique_rows(ids)
    bounds = np.arange(WORLD + 1, dtype=np.int64) * (NGRAM_VOCAB_SIZE // WORLD)

    def request():
        cuts = np.searchsorted(want, bounds)
        counts = np.diff(cuts)
        counts[3] = 0
        torch.from_numpy(counts.astype(np.int64))
        return counts.tolist(), int(cuts[3]), int(cuts[4])
    out["request"] = 2 * median_ms(request, repeats)  # both tables
    return out


def gil_test(tokens: torch.Tensor, local: int, cap: int, seconds: float) -> dict:
    """Main-thread loop latency (a small dispatch-like body) while a producer builds batches nonstop."""
    small = torch.zeros(16)

    def body():  # ~ the Python + dispatcher work of launching a few ops
        for _ in range(20):
            small.add_(1.0)
            _ = [k for k in range(8)]

    def measure() -> list[float]:
        lat = []
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            body()
            lat.append(1e6 * (time.perf_counter() - t0))
        return lat

    out = {"idle": measure()}
    # nonstop: the worst case; "paced": one batch per GPU step (what HOST_ASYNC=1 does in steady state)
    for mode, plan, pace in (("plan", True, 0.0), ("walk", False, 0.0), ("paced", True, GPU_STEP_S)):
        data.HOST_BATCH_PLAN = plan
        shard = data.Shard(tokens, WORLD)
        shard._ready.wait()
        shard._maybe_switch()
        stop = threading.Event()
        built = [0]

        def produce():
            while not stop.is_set():
                began = time.perf_counter()
                try:
                    starts, ends = shard.next_batch(local, cap)
                except StopIteration:
                    shard.i = 0
                    continue
                s, e = torch.tensor(starts[0]), torch.tensor(ends[0])
                buf = torch.cat([tokens[i:j] for i, j in zip(s, e)])
                e[-1] -= 1
                cum = (e - s).cumsum(0)
                if cap > data.VIRTUAL_SEQ_CAP:
                    cum = data.split_attention_segments(cum, data.VIRTUAL_SEQ_CAP)
                data._prepare_batch(buf[:-1], buf[1:], cum, local, data.cu_seqlens_rows(local))
                built[0] += 1
                if pace:
                    stop.wait(max(0.0, pace - (time.perf_counter() - began)))
        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        out[mode] = measure()
        stop.set()
        thread.join()
        out[mode + "_rate"] = built[0] / seconds
    data.HOST_BATCH_PLAN = False
    return out


def pct(values, q):
    return float(np.percentile(values, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", help="a FineWeb .bin shard (default: synthetic tokens)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=981, help="NUM_SCHEDULED_ITERATIONS")
    args = ap.parse_args()
    torch.set_num_threads(1)
    repeats = 15 if args.quick else 60
    n = 30_000_000
    tokens = read_tokens(args.shard, n) if args.shard else synthetic_tokens(n)
    harness.fake_dist(0)
    sched = TrainingSchedule(TRAINING_STAGES, args.steps, 20, torch.device("cpu"), cooldown_frac=0.8,
                             split_embed_stage=4, ws_post_yarn_ext=20)
    print(f"tokens: {'synthetic' if not args.shard else args.shard}, {n / 1e6:.0f}M; torch {torch.__version__}, "
          f"numpy {np.__version__}, 1 torch thread; schedule {sched.total_steps} steps")
    totals = {}
    print(f"{'stage':>5} {'steps':>5} {'tok/rank':>8} {'cap':>5} {'P':>11} | main thread ms/step: "
          f"{'walk':>6} {'plan':>6} {'gather':>6} {'prep':>6} (hash) {'pincopy':>7} | event: {'request':>7} | "
          f"prep thr: {'cands':>6} {'wants':>6}")
    for i, steps, local, cap, ps in stage_table(sched):
        walk = time_loader_ops(tokens, local, cap, repeats, plan=False)
        plan = time_loader_ops(tokens, local, cap, repeats, plan=True)
        prep = time_prep_threads(tokens, local, cap, max(ps), repeats)
        start, end = sched.boundaries[i]
        events = sum(is_update_step(s) for s in range(start, end))
        sampled = sum(candidate_count_at(sched, s) > 0 for s in range(start, end))
        print(f"{i:>5} {steps:>5} {local:>8} {cap:>5} {','.join(map(str, ps)) or 'full':>11} | "
              f"{'':>20} {walk['plan']:6.3f} {plan['plan']:6.3f} {plan['gather']:6.3f} {plan['prep']:6.3f} "
              f"({plan['hash']:.3f}) {plan['pincopy']:7.3f} | {'':>7}{prep['request']:7.3f} | "
              f"{'':>10}{prep.get('cands', 0):6.2f} {prep['wants']:6.2f}")
        for key, value in (("walk", walk["plan"]), ("plan", plan["plan"]), ("gather", plan["gather"]),
                           ("prep", plan["prep"]), ("pincopy", plan["pincopy"])):
            totals[key] = totals.get(key, 0.0) + steps * value
        totals["request"] = totals.get("request", 0.0) + events * prep["request"]
        totals["cands"] = totals.get("cands", 0.0) + sampled * prep.get("cands", 0.0)
        totals["wants"] = totals.get("wants", 0.0) + events * prep["wants"]
    moved_sync = totals["walk"] + totals["gather"] + totals["prep"]
    moved_plan = totals["plan"] + totals["gather"] + totals["prep"]
    print("\nwhole run (ms, this CPU):")
    print(f"  loader host build on the main thread, #397 (walk):         {moved_sync:8.1f}  "
          f"(walk {totals['walk']:.1f}, gather {totals['gather']:.1f}, prep {totals['prep']:.1f})")
    print(f"  loader host build on the main thread, HOST_PREP=all (plan): {moved_plan:8.1f}  (plan {totals['plan']:.1f})")
    print(f"  -> HOST_ASYNC=1 moves that to the producer; stays on the main thread: pincopy {totals['pincopy']:.1f}, "
          f"request {totals['request']:.1f}")
    print(f"  already off the main thread (prep threads): cands {totals['cands']:.1f}, wants {totals['wants']:.1f}")
    print(f"  per step on average: build {moved_sync / sched.total_steps:.3f} (walk) / "
          f"{moved_plan / sched.total_steps:.3f} (plan) ms; GPU step ~29.4 ms (30.957 s / 1053 steps on the node)")

    scan = median_ms(lambda: (tokens == data.BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).numpy(), 5)
    print(f"  Shard._scan (the full BOS index; batches use the 6M-token partial index until it lands): "
          f"{scan:.1f} ms on {n / 1e6:.0f}M tokens -> ~{scan * 100e6 / n:.0f} ms for a 100M-token shard")

    seconds = 0.6 if args.quick else 2.0
    stage = TRAINING_STAGES[2]
    gil = gil_test(tokens, stage.batch_size // WORLD, stage.train_max_seq_len, seconds)
    print(f"\nGIL: main-thread loop iteration (us) while a producer builds stage-2 batches ({seconds:.1f} s each)")
    for mode in ("idle", "plan", "walk", "paced"):
        lat = gil[mode]
        rate = f"  producer {gil[mode + '_rate']:.0f} batches/s" if mode != "idle" else ""
        print(f"  {mode:>5}: mean {statistics.fmean(lat):6.1f}  p50 {pct(lat, 50):7.1f}  p99 {pct(lat, 99):7.1f}  p99.9 {pct(lat, 99.9):7.1f}  "
              f"max {max(lat):8.1f}  n={len(lat)}{rate}")


if __name__ == "__main__":
    main()
