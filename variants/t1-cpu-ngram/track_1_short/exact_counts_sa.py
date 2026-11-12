"""EXACT_BACKEND=sa (default gpu): ExactCounts' order-2..8 counts from a suffix array of each rank's corpus share on
the host (exact_match.SuffixCounts, the crate's `sa` feature, src/sa.rs) instead of the GPU scan of exact_counts.py.

Same constructor, same `out` [queries, 14] on the GPU, same events and the same share (the shard files after
`exclude`, rank::world, concatenated in order), so the counts summed over the ranks are those of the GPU scan (equal
up to the scan's 64-bit hash collisions: 182/sufarr/ tests both against one exact reference). The share's suffix
array is built on a side thread once load_corpus() is called (the trainer calls it on the clock, when the 6-token
index is built), and launch() answers every query on the host, then copies the counts to the GPU.

Not supported: SCAN_HINTS (the 381 config's default). Its samples come from the GPU scan's first occurrences of
each marked context, which this backend does not take, so the constructor raises unless SCAN_HINTS=0 (CONFIG=380).
SA_THREADS (default: the host's CPUs / world) sets each rank's build and query threads; the suffix sort itself is
single-threaded unless the extension was built with the `sa-openmp` feature (182/sufarr/build.sh)."""
import os
import threading
import time

import numpy as np
import torch

EXACT_BACKEND = os.environ.get("EXACT_BACKEND", "gpu")
assert EXACT_BACKEND in ("gpu", "sa"), f"EXACT_BACKEND={EXACT_BACKEND}: gpu (the scan, default) or sa (suffix arrays)"
SA_THREADS = int(os.environ.get("SA_THREADS", "0"))
COLS = 14  # 2 * exact_counts.MAX_ORDER - 2: N for context lengths 2..8, then C


class SuffixExactCounts:
    def __init__(self, train_files, exclude, rank, world, device, q_chunk, pinned=False, n_val=0):
        if n_val:
            raise ValueError("EXACT_BACKEND=sa takes no SCAN_HINTS samples: run it with SCAN_HINTS=0 (CONFIG=380)")
        import exact_match
        if not hasattr(exact_match, "SuffixCounts"):
            raise ImportError("EXACT_BACKEND=sa: this exact_match was built without the `sa` feature; "
                              "build it with variants/t1-cpu-ngram/182/sufarr/build.sh")
        self._cls = exact_match.SuffixCounts
        self.device, self.files, self.rank = device, sorted(train_files)[exclude:][rank::world], rank
        self.n = sum((os.path.getsize(f) - 1024) // 2 for f in self.files)
        self.threads = SA_THREADS or max(1, (os.cpu_count() or world) // world)
        self.q_chunk, (self.q_tok, self.q_tgt) = q_chunk, torch.zeros(2, len(q_chunk), dtype=torch.int32, device=device)
        self.out, self.loaded = torch.zeros(len(q_chunk), COLS, dtype=torch.int32, device=device), threading.Event()
        self.queries = threading.Event()
        self.host_out = torch.zeros(len(q_chunk), COLS, dtype=torch.int32, pin_memory=pinned and torch.cuda.is_available())
        self.nv, self.cap, self.index, self.load_error = 0, 0, None, None
        self.build_s = self.query_s = None

    def load_corpus(self):  # a thread: this rank's share read and suffix sorted on the host (the GIL released)
        def run():
            try:  # a failed build still sets `loaded`, and launch() raises its error rather than waiting forever
                t = time.perf_counter()
                self.index = self._cls(self.files, self.threads)
                self.build_s = time.perf_counter() - t
                if self.index.tokens != self.n:
                    raise EOFError(f"the suffix array holds {self.index.tokens} of the {self.n} corpus tokens planned")
                if self.rank == 0:
                    print(f"exact counts (sa): {self.n} tokens suffix sorted in {self.build_s:.2f} s on {self.threads} "
                          f"threads, {self.index.memory / 2**30:.2f} GiB", flush=True)
            except BaseException as e:
                self.load_error = e
                raise
            finally:
                self.loaded.set()
        threading.Thread(target=run, daemon=True).start()

    @torch.no_grad()
    def launch(self):  # after training, the queries in q_tok / q_tgt: this rank's counts into self.out
        from track_1_short.exact_counts import segment_starts
        start = segment_starts(self.q_tok, self.q_chunk).to(torch.int64).cpu().numpy()
        tok, tgt = (np.ascontiguousarray(x.cpu().numpy()) for x in (self.q_tok, self.q_tgt))
        self.loaded.wait()
        if self.load_error is not None:
            raise RuntimeError("the exact-count suffix array build failed") from self.load_error
        t = time.perf_counter()
        self.index.counts_into(tok, tgt, start, self.host_out.numpy())
        self.query_s = time.perf_counter() - t
        if self.rank == 0:
            print(f"exact counts (sa): {len(tok)} queries counted in {self.query_s:.2f} s", flush=True)
        self.out.copy_(self.host_out, non_blocking=self.host_out.is_pinned())

    def launch_marked(self, marks):  # SCAN_HINTS_SKIP only, which needs SCAN_HINTS: never reached
        raise RuntimeError("EXACT_BACKEND=sa does not support SCAN_HINTS")
