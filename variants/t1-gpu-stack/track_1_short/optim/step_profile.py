"""OPTIM_PROFILE=1: where the optimizer tail's time goes on the compute stream, measured inside the real run.

AnvilAndAdam.step marks three CUDA events per label it updates, on the current (compute) stream:
  <label>.issue  just before the reduce-scatter future's wait (a stream wait: the host does not block)
  <label>.ready  just after it: the event completes when the stream resumes, i.e. when the reduce-scatter has landed
  <label>.done   after the update (for an ANVIL bank: its captured graph's replay)
and one pair around Phase 2-3 (tail.begin / tail.end). Per label:
  wait   = ready - issue: the compute stream idles on the collective (comms are the critical path there)
  update = done - ready:  the update's own GPU time (the cascade for the banks)
If a bank's `wait` is ~0 and its `update` large, shortening its cascade shortens the step; if `wait` dominates, the
reduce-scatter chain is the bound and a cheaper cascade buys nothing at that point (overlap would).

Events are read back without blocking: each step's record is resolved once its last event has completed (query()),
so the host keeps running ahead of the GPU. Nothing is recorded while a CUDA graph capture is in progress. Every
REPORT_EVERY steps each rank prints one line with the means over the last window and over the whole run:

  [optim-profile] rank=0 steps=1000 window=100 tail=1.234ms | qk_bank w=0.010 u=0.080 | ... (ms)

Cost when on: three event records per label and step (~40 host-side records per step) and a query() each; when off the
object is never created and step() only tests `prof is not None`.
"""
from collections import deque

import torch

REPORT_EVERY = 100


class _StepRecord:
    __slots__ = ("events", "order")

    def __init__(self):
        self.events: dict[str, torch.cuda.Event] = {}
        self.order: list[str] = []

    def mark(self, name: str):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[name] = event
        self.order.append(name)

    def end(self):
        self.mark("tail.end")


class StepProfile:
    def __init__(self, rank: int):
        self.rank = rank
        self.pending: deque[_StepRecord] = deque()
        self.steps = 0
        self.total: dict[str, float] = {}
        self.window: dict[str, float] = {}
        self.window_steps = 0

    def begin(self) -> _StepRecord | None:
        """At the start of Phase 2: a new record, or None while a capture is in progress (marks would be captured)."""
        if torch.cuda.is_current_stream_capturing():
            return None
        self._drain()
        record = _StepRecord()
        record.mark("tail.begin")
        self.pending.append(record)
        return record

    def _drain(self):
        """Accumulate every finished record, oldest first; stop at the first one the GPU has not reached."""
        while self.pending:
            head = self.pending[0]
            end = head.events.get("tail.end")
            if end is None:  # a step that never reached end() (an exception mid-step): drop it
                self.pending.popleft()
                continue
            if not end.query():
                return
            self._accumulate(self.pending.popleft())

    def _accumulate(self, record: _StepRecord):
        ev = record.events
        times = {"tail": ev["tail.begin"].elapsed_time(ev["tail.end"])}
        for name in record.order:
            if name.endswith(".issue"):
                label = name[:-len(".issue")]
                if f"{label}.ready" in ev and f"{label}.done" in ev:
                    times[f"{label}.w"] = ev[name].elapsed_time(ev[f"{label}.ready"])
                    times[f"{label}.u"] = ev[f"{label}.ready"].elapsed_time(ev[f"{label}.done"])
        for key, value in times.items():
            self.total[key] = self.total.get(key, 0.0) + value
            self.window[key] = self.window.get(key, 0.0) + value
        self.steps += 1
        self.window_steps += 1
        if self.steps % REPORT_EVERY == 0:
            print(self.format(), flush=True)
            self.window, self.window_steps = {}, 0

    def format(self) -> str:
        """The report line (window means, then run means in brackets)."""
        def part(key):
            w = self.window.get(key, 0.0) / max(self.window_steps, 1)
            t = self.total.get(key, 0.0) / max(self.steps, 1)
            return f"{w:.3f}[{t:.3f}]"
        labels = [k[:-2] for k in self.total if k.endswith(".w")]
        body = " | ".join(f"{label} w={part(label + '.w')} u={part(label + '.u')}" for label in labels)
        return (f"[optim-profile] rank={self.rank} steps={self.steps} window={self.window_steps} "
                f"tail={part('tail')}ms | {body} (ms; window mean[run mean])")
