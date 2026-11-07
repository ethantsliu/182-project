"""FP8_DUMP_DIR: save sample fp8-GEMM operands from a few training steps, for the offline study in
182/hadamard-fp8/fp8_quant_error.py. Off by default.

What is saved, per dumped step and per selected MLP slot / attention layer (first FP8_DUMP_ROWS tokens,
bf16, on CPU), as <dir>/step<NNNNNN>.pt:
  mlp<slot>:  x [T, 768] (normed MLP input), W1 [H, 768], W2 [H, 768], g [T, 768] (the MLP output's
              gradient), p_fold (post-lambda folded into W2 by the fp8 path, or None), plus the fp8 scale
              state the GEMMs used (x / g static scales, the delayed post / dpre scales).
  qkv<layer>: x [T, 768] (normed attention input), W [qkv_dim, 768] (sa_lambda * [Wq; Wk; Wv]),
              g [T, qkv_dim] (the projection output's bf16 gradient, after the norm / rotary backward).
post and dpre are not saved: the script rebuilds them from (x, W1, W2, g) exactly as the kernels do.

How (train_gpt.py, `maybe_dump`): the step graphs are CUDA-graph replays, so a dump is one extra EAGER
forward + backward of the uncompiled model on the step's batch, rank 0 only, with no n-gram sink and no
value_embeds gradient buffer (so no gradient lands anywhere), and every fp8 amax buffer snapshotted and
restored around it (the kernels atomic-max into them). The training step itself is untouched. The
clock is stopped for the dump; a dump run is an analysis run and its time is not a result.

The hook points in the autograd Functions check FP8_DUMP.active(), which is False under torch.compile
(`torch.compiler.is_compiling()` short-circuits before the Python flag is read, so no guard is added
and the compiled graphs are unchanged).
"""
import os

import torch


class Fp8Dump:
    def __init__(self):
        self._on = False
        self.tag = None      # the layer currently running its forward (set by GPT.forward, eager only)
        self.rows = 0
        self.records: dict[str, dict] = {}

    def active(self) -> bool:
        return not torch.compiler.is_compiling() and self._on

    def begin(self, rows: int):
        self._on, self.rows, self.records, self.tag = True, rows, {}, None

    def set_tag(self, tag: str | None):
        if self.active():
            self.tag = tag

    def _rows(self, t):
        """An activation [T, *]: its first `rows` tokens, on the CPU."""
        return t.detach()[:self.rows].to("cpu", copy=True)

    def save_mlp(self, tag, x, W1, W2, g, p_fold, **scales):
        if tag is None:
            return
        x2 = x.reshape(-1, x.shape[-1])
        self.records[tag] = dict(kind="mlp", x=self._rows(x2), g=self._rows(g.reshape(-1, g.shape[-1])),
                                 W1=W1.detach().to("cpu", copy=True), W2=W2.detach().to("cpu", copy=True),
                                 p_fold=None if p_fold is None else float(p_fold),
                                 **{k: float(v) for k, v in scales.items()})

    def save_qkv(self, tag, x, W, g):
        if tag is None:
            return
        self.records[tag] = dict(kind="qkv", x=self._rows(x.reshape(-1, x.shape[-1])),
                                 W=W.detach().to("cpu", copy=True), g=self._rows(g.reshape(-1, g.shape[-1])))

    def end(self, path: str, extra: dict):
        self._on, self.tag = False, None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(dict(records=self.records, **extra), path)
        n = len(self.records)
        self.records = {}
        return n


FP8_DUMP = Fp8Dump()


def run_fp8_dump(model, batch, ngram_slots, forward_args, path: str, rows: int, extra: dict) -> int:
    """One eager forward + backward of the UNCOMPILED model on `batch` with FP8_DUMP collecting; returns
    the number of records saved to `path`. No gradient lands anywhere (no n-gram sink, no value_embeds
    buffer, torch.autograd.grad results dropped) and the fp8 amax buffers the kernels atomic-max into are
    restored afterwards, so the training run's state is unchanged. `forward_args` should carry no sampled
    loss: the dump uses the full softmax even on sampled-softmax steps (its operands differ slightly
    from that step's training operands; the quantization statistics do not depend on it)."""
    amax_buffers = {n: b for n, b in model.named_buffers() if "amax" in n}
    saved = {n: b.detach().clone() for n, b in amax_buffers.items()}
    params = [p for p in model.parameters() if p.requires_grad]
    FP8_DUMP.begin(rows)
    try:
        loss = model(batch.inputs, batch.targets, batch.cum_seqlens, ngram_slots, forward_args).sum()
        torch.autograd.grad(loss, params, allow_unused=True)
        del loss
    finally:
        with torch.no_grad():
            for n, b in amax_buffers.items():
                b.copy_(saved[n])
    return FP8_DUMP.end(path, extra)
