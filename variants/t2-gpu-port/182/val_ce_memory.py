"""What Inductor keeps alive for the validation cross-entropy, on CPU, at a small size.

    python 182/val_ce_memory.py          # needs a working C++ compiler for Inductor's CPU backend
    bash 182/val_ce_memory.sh            # the same in docker, triton-cpu:3.6.0 (torch 2.10.0, PR #261's pin)

Compiles the eval-mode loss of train_gpt_medium.py at [rows, vocab] = [256, 1024] three ways and walks the
generated wrapper code: every buffer it allocates and frees, in order. Prints the peak bytes of live
intermediate buffers per element of one chunk's [rows / n, vocab] logit block.

  inline unmasked   PR #261's loop (VAL_CE_CHUNKS row chunks, softcap, cross_entropy)
  inline masked     the same loop with the canonical mask gathered inside it (this tree before the fix)
  op masked         the masked loss through medium_extras.canon_val_ce (an opaque custom op, eager inside)

At the real size a block of 262144 / n rows x 50304 columns has 13.2e9 / n elements, so 6 B per element
at n = 4 is 18.4 GiB, and 48 B per element at n = 8 is 73.7 GiB. Inductor's CPU and CUDA backends share
the scheduler pass that decides these fusions (buffers read by several nodes make those nodes fusion
candidates), but this is the CPU backend's plan, not a measurement of the GPU one.
"""
import ast
import os
import re
import sys

import torch
import torch.nn.functional as F
from torch import Tensor

ROWS, VOCAB, DIM = 256, 1024, 64
SIZES = dict(bfloat16=2, float32=4, uint8=1, bool=1, int64=8, int32=4)
HERE = os.path.dirname(os.path.abspath(__file__))


def canon_op():
    """medium_extras.canon_val_ce, registered once per process. medium_extras itself loads the tiktoken
    vocabulary at import, so this execs only the op's section when the op is not registered yet."""
    if hasattr(torch.ops.t2gpu, "canon_val_ce"):
        return torch.ops.t2gpu.canon_val_ce
    with open(os.path.join(HERE, "..", "medium_extras.py")) as f:
        tree = ast.parse(f.read())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("canon_val_ce", "_")]
    g = dict(torch=torch, F=F, Tensor=Tensor)
    exec(compile(ast.Module(nodes, []), "medium_extras.py[canon_val_ce]", "exec"), g)
    return torch.ops.t2gpu.canon_val_ce


def loss_fn(n: int, how: str):
    op = canon_op() if how == "op masked" else None

    def f(x, w, tgt, inp, mask):
        if op is not None:
            return op(x, w, tgt, inp, mask, n)
        xr, tc, ic = x.chunk(n), tgt.chunk(n), inp.chunk(n)
        loss = 0
        for i in range(n):
            logits = (xr[i] @ w).float()
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)
            if how == "inline masked":
                shifts = torch.arange(8, dtype=torch.uint8, device=logits.device)
                dropped = (mask[ic[i], :, None] >> shifts & 1).view(logits.shape).bool()
                logits = logits.masked_fill(dropped, -60.0)
            loss += F.cross_entropy(logits.view(-1, logits.size(-1)), tc[i], reduction="mean") / n
        return loss
    return f


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, DIM, generator=g).bfloat16()
    w = torch.randn(DIM, VOCAB, generator=g).bfloat16()
    tgt = torch.randint(0, VOCAB, (ROWS,), generator=g)
    inp = torch.randint(0, VOCAB, (ROWS,), generator=g).int()
    bits = torch.rand(VOCAB, VOCAB, generator=g) < 0.3
    bits[inp.long(), tgt] = False  # a sound mask: no real (prev, target) pair is dropped
    mask = torch.from_numpy(__import__("numpy").packbits(bits.numpy(), axis=1, bitorder="little"))
    return x, w, tgt, inp, mask


def peak_live_bytes(code: str) -> tuple[int, int]:
    """(peak bytes of live intermediate buffers, number of [rows / n, vocab]-or-larger buffers) in the
    wrapper's call(): allocations add, `del` lines free."""
    call = code[code.index("def call("):]
    live, cur, peak, big = {}, 0, 0, 0
    for line in call.splitlines():
        m = re.search(r"(buf\d+) = empty_strided_\w+\(\(([^)]*)\), \(([^)]*)\), torch\.(\w+)\)", line)
        if m:
            numel = 1
            for s in m.group(2).split(","):
                if s.strip():
                    numel *= int(s)
            live[m.group(1)] = numel * SIZES[m.group(4)]
            cur += live[m.group(1)]
            peak = max(peak, cur)
            big += numel >= VOCAB * 16
        m = re.match(r"\s*del (.*)$", line)
        if m:
            for name in m.group(1).split(","):
                cur -= live.pop(name.strip(), 0)
    return peak, big


def measure(cases=((4, "inline unmasked"), (8, "inline unmasked"), (8, "inline masked"), (8, "op masked"))):
    """{(n, how): (bytes per element of one chunk's block, number of big buffers, loss)}."""
    from torch._inductor.utils import run_and_get_code
    args = inputs()
    out = {}
    for n, how in cases:
        torch._dynamo.reset()
        with torch.no_grad():
            loss, codes = run_and_get_code(torch.compile(loss_fn(n, how), dynamic=False, fullgraph=True), *args)
        peak, big = peak_live_bytes(codes[0])
        out[(n, how)] = (peak / (ROWS // n * VOCAB), big, float(loss))
    return out


if __name__ == "__main__":
    print(f"torch {torch.__version__}; [rows, vocab] = [{ROWS}, {VOCAB}]")
    res = measure()
    for (n, how), (per_el, big, loss) in res.items():
        print(f"VAL_CE_CHUNKS={n} {how:16s} peak live intermediates {per_el:6.2f} B per element of one "
              f"[rows/{n}, vocab] block, {big} block-sized buffers, loss {loss:.6f}")
    ref = res[(8, "inline masked")][2]
    ok = abs(res[(8, "op masked")][2] - ref) < 1e-4 and res[(8, "op masked")][1] == 0
    print("op masked: same loss as inline masked, no block-sized buffer in the compiled graph" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
