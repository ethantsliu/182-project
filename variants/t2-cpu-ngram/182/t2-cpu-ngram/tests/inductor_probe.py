"""The validation slab loop (model/gpt.py cplm_eval_slab_losses) through CPU inductor at two slab sizes.
Prints one JSON line: for half a slab, the largest per-token and the mean difference from one slab.

A subprocess of test_final_val.py: model/gpt.py is imported under the CPU stubs, then the triton stub is removed
before torch._dynamo is imported, so inductor takes its C++ path (it type-checks triton objects, which the stub
cannot satisfy).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cpu_stubs  # noqa: E402

cpu_stubs.install_all(HERE.parents[2])
import torch  # noqa: E402
from track_1_short.model import gpt  # noqa: E402

sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, cpu_stubs._TritonFinder)]
for name in [m for m in sys.modules if m == "triton" or m.startswith("triton.")]:
    del sys.modules[name]
import torch.utils._triton  # noqa: E402

for fn in vars(torch.utils._triton).values():  # has_triton_package() etc. may have cached the stub's answer
    if hasattr(fn, "cache_clear"):
        fn.cache_clear()
import torch._dynamo.utils  # noqa: E402

VOCAB, T, D, IDS = 50304, 12, 8, 16  # the vocabulary must hold the copy column (50257); the rest is minimal
g = torch.Generator().manual_seed(0)
weight = torch.randn(D, VOCAB, generator=g).bfloat16() * 0.2
xf = torch.randn(T, D, generator=g).bfloat16()
inputs = torch.randint(0, IDS, (T,), generator=g)
targets = torch.randint(0, 50257, (T,), generator=g)
canon = torch.randint(0, 256, (IDS, VOCAB // 8), generator=g, dtype=torch.uint8)
p_copy, a_sink, gate = torch.rand(T, generator=g), torch.rand(T, generator=g), torch.full((1,), 0.7)


def lm_head(x):  # model/layers.py CastedLinearT's validation forward
    return x @ weight.type_as(x)


# Defined in model/gpt.py's globals, so it reads the bare EVAL_CE_SLAB_ROWS exactly as GPT._cplm_loss does.
exec("def _probe_loss(lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate):\n"
     "    return cplm_eval_slab_losses(lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate, EVAL_CE_SLAB_ROWS)\n",
     vars(gpt))


def loss(xf, inputs, targets):
    return gpt._probe_loss(lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate)


compiled = torch.compile(loss, dynamic=False, fullgraph=True)
out = {}
with torch.no_grad():
    for rows in (T, T // 2):  # one slab, then half of it (the retry's step)
        gpt.EVAL_CE_SLAB_ROWS = rows
        out[rows] = compiled(xf, inputs, targets)
    eager = loss(xf, inputs, targets)
a = out[T]
diffs = {r: float((v - a).abs().max()) for r, v in out.items() if r != T}
means = {r: abs(float(v.double().mean() - a.double().mean())) for r, v in out.items() if r != T}
print(json.dumps(dict(max_abs=diffs, mean_abs=means, vs_eager=float((a - eager).abs().max()),
                      graphs=torch._dynamo.utils.counters["stats"]["unique_graphs"])))
