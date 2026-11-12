"""CPU checks for what variants/t2-gpu-port adds to PR #261 (no GPU needed, ~20 s):

    python -m pytest 182 -q            # from variants/t2-gpu-port; or: python 182/test_cpu.py

1. Canonical mask soundness: no adjacent token pair of real GPT-2-encoded text is masked.
2. Prefix table: the per-rank buckets max-reduce to the single-bucket (full) table, and spot checks.
3. Kernels file: everything up to the prefix section is PR #261's triton_kernels.py byte for byte, and the
   prefix CE kernel is that kernel plus six substitutions whose anchors each occur exactly once.
4. Prefix CE kernel logic: a Python transliteration of its loss and two-pass gradient write (dense pass,
   then sparse target-column overwrite incl. the prefix thread) matches autograd.
5. Autograd accepts the surplus None gradients the prefix Function's backward returns.
6. The prefix-weight schedule reproduces track 1's 0.25 -> 0.20/0.15/0.10/0.05 -> 0 shape.
7. Model smoke test: the GPT and TrainingManager sections of train_gpt_medium.py, exec'd on CPU at a
   tiny width with flash-attention and the fused CE replaced by plain-torch references, run forward +
   backward under each flag combination; every parameter gets a gradient, every optimizer label is
   covered, the canonical mask can only lower the eval loss, and XSA at alpha=0 is an exact no-op.
   With every flag off the model calls the base CE with #261's 8 arguments and has #261's parameters
   and buffers. FUSED_MLP runs triton_kernels.FusedLinearReLUSquareFunction's real autograd code with a
   torch stand-in for the Triton kernel and matches the inline MLP's loss and gradients up to bf16 rounding.

The kernels themselves are compiled (not run) for H100 by 182/compile_check.sh.
"""
import ast
import hashlib
import glob
import os
import sys
import time

import numpy as np
import tiktoken
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from medium_extras import build_canonical_mask, build_prefix_table_bucket, canon_val_ce  # noqa: E402

VOCAB = 50304
enc = tiktoken.get_encoding("gpt2")


def corpus_tokens() -> np.ndarray:
    texts = []
    for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True))[:200]:
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    for path in sorted(glob.glob(os.path.join(ROOT, "*.py"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    texts.append("It's a test: we'll see 1,234.5 cafés, naïve “quotes”, 東京, emoji 🙂🙂 and   spaces.\n\n")
    toks = []
    for t in texts:
        toks.extend(enc.encode_ordinary(t))
        toks.append(enc.eot_token)  # documents are BOS-separated in the val stream too
    return np.array(toks, dtype=np.int64)


def test_canonical_mask():
    t = time.perf_counter()
    packed = build_canonical_mask(VOCAB)
    print(f"  canonical mask built in {time.perf_counter() - t:.1f}s, shape {packed.shape}")
    assert packed.shape == (VOCAB, VOCAB // 8) and packed.dtype == np.uint8
    toks = corpus_tokens()
    prev, cur = toks[:-1], toks[1:]
    bits = (packed[prev, cur // 8] >> (cur % 8)) & 1
    assert bits.sum() == 0, f"{int(bits.sum())} canonical pairs masked, e.g. {list(zip(prev[bits == 1][:5], cur[bits == 1][:5]))}"
    # the torch unpack used in train_gpt_medium.py's eval path agrees with numpy's bit order
    rows = torch.from_numpy(packed[prev[:64]])
    shifts = torch.arange(8, dtype=torch.uint8)
    dropped = (rows[:, :, None] >> shifts & 1).view(64, VOCAB).bool()
    ref = np.unpackbits(packed[prev[:64]], axis=1, bitorder="little").astype(bool)
    assert torch.equal(dropped, torch.from_numpy(ref))
    # BOS/EOT is never masked, as target or as context
    assert np.unpackbits(packed[:, enc.eot_token // 8], bitorder="little").reshape(VOCAB, 8)[:, enc.eot_token % 8].sum() == 0
    assert packed[enc.eot_token].sum() == 0
    # check_canon_val.py (run on the box against the real val shard): reads a .bin shard and finds masked pairs
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_canon_val import count_masked_pairs, load_shard
    p_bad, c_bad = map(int, np.argwhere(np.unpackbits(packed[:2000], axis=1, bitorder="little"))[0])
    for extra, want in [([], 0), ([p_bad, c_bad], 1)]:
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            body = np.concatenate([toks, extra]).astype(np.uint16)
            header = np.zeros(256, dtype=np.int32)
            header[:3] = 20240520, 1, len(body)
            f.write(header.tobytes() + body.tobytes())
            f.flush()
            hits = count_masked_pairs(packed, load_shard(f.name))
        assert len(hits) == want and (not want or hits[0] == len(toks)), (want, hits)
    frac = np.unpackbits(packed, axis=1, bitorder="little").mean()
    print(f"  {len(toks)} corpus tokens, 0 canonical pairs masked; {100 * frac:.2f}% of all pairs masked")


def test_prefix_table():
    full = build_prefix_table_bucket(VOCAB, bucket=0, num_buckets=1)
    merged = torch.stack([build_prefix_table_bucket(VOCAB, bucket=r, num_buckets=8) for r in range(8)]).amax(0)
    assert torch.equal(full, merged)
    ranks = enc._mergeable_ranks
    tok = {v: k for k, v in ranks.items()}
    for word in [" hello", " international", "ing", " the", "abc"]:
        for tid in enc.encode_ordinary(word):
            b, p = tok[tid], int(full[tid])
            if p >= 0:
                assert b.startswith(tok[p]) and len(tok[p]) < len(b)
                # longest: no longer proper prefix is a token
                assert not any(b[:n] in ranks for n in range(len(tok[p]) + 1, len(b)))
            else:
                assert not any(b[:n] in ranks for n in range(1, len(b)))
    assert int(full[enc.eot_token]) == -1 and (full[VOCAB - 40:] == -1).all()
    print(f"  prefix table ok, {(full >= 0).float().mean() * 100:.1f}% of tokens have a prefix target")


BASE_KERNELS_SHA256 = "5ad45f3466502cd9d3d60f42a09f7277eb5983d5aa1c13e72aad85f9ed07f857"  # upstream PR #261 (cdddeda8) triton_kernels.py
BASE_KERNELS_BYTES = 34510


def _kernel_sources():
    """(base CE kernel source, prefix CE kernel source), built from triton_kernels.py without importing it
    (the import compiles a CUDA kernel)."""
    with open(os.path.join(ROOT, "triton_kernels.py")) as f:
        tree = ast.parse(f.read())
    ns = {}
    for node in tree.body:
        name = getattr(getattr(node, "targets", [None])[0], "id", None) or getattr(node, "name", "")
        if name.startswith(("CE_KERNEL_", "_CE_PREFIX_EDITS", "_ce_prefix_source", "CE_PREFIX_KERNEL_SOURCE")):
            exec(compile(ast.Module([node], []), "triton_kernels.py", "exec"), ns)
    return ns["CE_KERNEL_SOURCE"], ns["CE_PREFIX_KERNEL_SOURCE"]


def test_kernels_file_is_base_plus_prefix_section():
    with open(os.path.join(ROOT, "triton_kernels.py"), "rb") as f:
        raw = f.read()
    assert hashlib.sha256(raw[:BASE_KERNELS_BYTES]).hexdigest() == BASE_KERNELS_SHA256, \
        "the first part of triton_kernels.py is no longer PR #261's file"
    assert b"prefix" not in raw[:BASE_KERNELS_BYTES].lower()
    base, pfx = _kernel_sources()  # _ce_prefix_source asserts every anchor occurs exactly once
    for needle in ("const int64_t* __restrict__ prefix_targets,", "float prefix_weight = prefix_weight_ptr[0];",
                   "total_loss += prefix_weight * (lse - z_p);", "S_w += prefix_weight;", "bool is_pfx",
                   "term2 += prefix_weight;", "void ce_fwd_bwd_prefix_kernel("):
        assert pfx.count(needle) == 1 and needle not in base, needle
    assert pfx.count("{") == pfx.count("}") and "ce_fwd_bwd_kernel" not in pfx
    print("  triton_kernels.py = PR #261's file + the prefix section; all six kernel edits applied once")


def kernel_reference(logits, targets, mtp_w, prefix_targets, prefix_w, A=23.0, B=5.0, C=7.5):
    """Transliteration of ce_fwd_bwd_prefix_kernel (fp64, no fp8/bf16 rounding): per-row loss and dloss/dlogit."""
    n_rows, V = logits.shape
    n_predict = len(mtp_w)
    losses = torch.zeros(n_rows, dtype=torch.float64)
    grad = torch.zeros_like(logits)
    for r in range(n_rows):
        s = torch.sigmoid(logits[r] / C + B / C)
        z = A * s
        lse = torch.logsumexp(z, 0)
        total = 0.0
        for k in range(n_predict):
            if r + k < n_rows:
                total += mtp_w[k] * (lse - z[targets[r + k]])
        ptgt = int(prefix_targets[r])
        if not 0 <= ptgt < V:
            ptgt = -1  # the kernel treats an out-of-range prefix target as "none"
        if ptgt >= 0:
            total += prefix_w * (lse - z[ptgt])
        losses[r] = total
        S_w = sum(mtp_w) + (prefix_w if ptgt >= 0 else 0.0)  # kernel sums all mtp weights, even past the end
        p = torch.exp(z - lse)
        dz = S_w * p  # dense pass: term2 = 0
        grad[r] = (A / C) * dz * s * (1 - s)
        threads = [k for k in range(n_predict) if r + k < n_rows]
        is_pfx = ptgt >= 0 and prefix_w > 0
        cols = [int(targets[r + k]) for k in threads] + ([ptgt] if is_pfx else [])
        for col in cols:  # sparse pass: overwrite each target column
            term2 = sum(mtp_w[k] for k in range(min(3, n_predict)) if r + k < n_rows and int(targets[r + k]) == col)
            if ptgt >= 0 and ptgt == col:
                term2 += prefix_w
            grad[r, col] = (A / C) * (S_w * p[col] - term2) * s[col] * (1 - s[col])
    return losses, grad


def test_ce_kernel_logic():
    torch.manual_seed(0)
    n_rows, V = 12, 37
    for mtp_w, prefix_w in [([1.0], 0.0), ([1.0, 0.5, 0.25], 0.25), ([1.0, 0.5], 0.1), ([1.0], 0.2)]:
        logits = (5 * torch.randn(n_rows, V, dtype=torch.float64)).requires_grad_()
        targets = torch.randint(0, V, (n_rows,))
        targets[3] = targets[4]  # duplicate MTP targets
        prefix_targets = torch.randint(-1, V, (n_rows,))
        prefix_targets[5] = targets[5]  # prefix target equal to the next-token target
        prefix_targets[6] = targets[7]  # prefix target equal to an MTP target
        prefix_targets[7] = -1
        prefix_targets[8] = V + 3  # out of range: must behave exactly like -1
        losses, grad = kernel_reference(logits.detach(), targets, mtp_w, prefix_targets, prefix_w)
        # ground truth: sum_k w_k CE(z_r, t_{r+k}) + prefix_w CE(z_r, ptgt_r), via autograd
        z = 23.0 * torch.sigmoid((logits + 5.0) / 7.5)
        logp = torch.log_softmax(z, -1)
        loss = torch.zeros(n_rows, dtype=torch.float64)
        for r in range(n_rows):
            for k, w in enumerate(mtp_w):
                if r + k < n_rows:
                    loss[r] = loss[r] - w * logp[r, targets[r + k]]
            if 0 <= prefix_targets[r] < V:
                loss[r] = loss[r] - prefix_w * logp[r, prefix_targets[r]]
        loss.sum().backward()
        assert torch.allclose(losses, loss.detach(), atol=1e-9)
        # rows near the end differ by design: the kernel's S_w counts MTP weights past the batch end
        # (pre-existing upstream behaviour, unchanged here), so compare only rows with all MTP targets
        full = n_rows - (len(mtp_w) - 1)
        assert torch.allclose(grad[:full], logits.grad[:full], atol=1e-9), (mtp_w, prefix_w)
    print("  CE kernel loss + two-pass gradient logic matches autograd (incl. duplicate/prefix-collision columns)")


def test_surplus_none_grads():
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, a=1.0, prefix=None):
            return x * w * a

        @staticmethod
        def backward(ctx, g):
            return g, None, None, None, None, None  # more gradients than the 2 inputs passed

    x = torch.ones(3, requires_grad=True)
    F.apply(x, 2.0).sum().backward()
    assert torch.equal(x.grad, torch.ones(3))
    print("  autograd accepts surplus None gradients (the prefix Function returns 13 for 10 inputs)")


def test_prefix_schedule():
    # Mirrors TrainingManager._build_prefix_schedule with track 1's fractions.
    W, HOLD, END, N = 0.25, 0.2854, 0.6063, 4125
    def w(s):
        x = s / N
        if x < HOLD:
            return W
        if x < END:
            return W * (4 - min(3, int(4 * (x - HOLD) / (END - HOLD)))) / 5
        return 0.0
    vals = [round(w(s), 6) for s in range(N + 26)]
    assert sorted(set(vals), reverse=True) == [0.25, 0.2, 0.15, 0.1, 0.05, 0.0], set(vals)
    assert vals[0] == 0.25 and vals[int(0.5 * N)] in (0.1, 0.05) and vals[-1] == 0.0
    print("  prefix schedule steps 0.25 -> 0.20/0.15/0.10/0.05 -> 0")


class _FlashRef:
    """Plain-torch flash_attn_varlen_func: causal, per-document (cu_seqlens), sliding window (left)."""
    @staticmethod
    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, softmax_scale, window_size):
        T = q.size(0)
        doc = torch.searchsorted(cu_seqlens_q.unique(), torch.arange(T), right=True)
        i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
        allowed = (doc[:, None] == doc[None, :]) & (j <= i) & (i - j <= window_size[0])
        att = torch.einsum("thd,shd->hts", q.float(), k.float()) * softmax_scale
        att = att.masked_fill(~allowed, float("-inf")).softmax(-1)
        return torch.einsum("hts,shd->thd", att, v.float()).type_as(q)


def _ce_ref(x, targets, mtp_weights, W, A, B, C, prefix_targets=None, prefix_weight=None):
    logp = torch.log_softmax(A * torch.sigmoid((x.float() @ W.float() + B) / C), -1)
    n = targets.size(0)
    loss = torch.zeros(n)
    for k, w in enumerate(mtp_weights.tolist()):
        loss[: n - k] = loss[: n - k] - w * logp[: n - k].gather(1, targets[k:, None])[:, 0]
    if prefix_targets is not None:
        pfx = logp.gather(1, prefix_targets.clamp_min(0)[:, None])[:, 0]
        loss = loss - prefix_weight * torch.where(prefix_targets >= 0, pfx, torch.zeros_like(pfx))
    return loss


class _FusedCERef:
    """triton_kernels.FusedSoftcappedCrossEntropy.apply as #261 calls it: exactly 8 positional arguments."""
    calls = 0

    @staticmethod
    def apply(x, targets, mtp_weights, W, x_s, w_s, grad_s, grad_scale):
        _FusedCERef.calls += 1
        return _ce_ref(x, targets, mtp_weights, W, 23.0, 5.0, 7.5)


class _FusedCEPrefixRef:
    """triton_kernels.FusedSoftcappedCrossEntropyPrefix.apply: the base arguments, then the prefix pair."""
    calls = 0

    @staticmethod
    def apply(x, targets, mtp_weights, W, x_s, w_s, grad_s, grad_scale, prefix_targets, prefix_weight, A=23.0, B=5.0, C=7.5):
        _FusedCEPrefixRef.calls += 1
        return _ce_ref(x, targets, mtp_weights, W, A, B, C, prefix_targets, prefix_weight)


def _fused_mlp_class():
    """The real FusedLinearReLUSquareFunction from triton_kernels.py (its source, exec'd), with the Triton
    linear_relu_square kernel replaced by a torch stand-in that also checks the kernel's input contract."""
    with open(os.path.join(ROOT, "triton_kernels.py")) as f:
        src = f.read()
    cls_src = src[src.index("class FusedLinearReLUSquareFunction"):src.index("# Tiled transpose copy kernel")]
    cls_src = cls_src[:cls_src.rindex("# ---")]

    def linear_relu_square(a, b, aux=None):
        # kernel contract: 2D, contiguous bf16 operands (TMA descriptors, tl.bfloat16 epilogue), b is [N, K]
        assert a.ndim == b.ndim == 2 and a.is_contiguous() and b.is_contiguous(), (a.shape, b.shape)
        assert a.dtype == b.dtype == torch.bfloat16 and a.shape[1] == b.shape[1], (a.dtype, b.dtype, a.shape, b.shape)
        c = (a.float() @ b.float().T).bfloat16()  # fp32 accumulate, bf16 out
        if aux is None:
            return c, torch.relu(c).square()  # pre, post = relu(pre)^2
        return 2 * c * torch.where(aux > 0, aux, 0)  # dpre = 2 * (g @ W2.T) * relu(pre)

    g = dict(torch=torch, linear_relu_square=linear_relu_square)
    exec(compile(cls_src, "triton_kernels.py[FusedLinearReLUSquareFunction]", "exec"), g)
    return g["FusedLinearReLUSquareFunction"]


def _load_trainer(flags: dict):
    """Exec the model + training-manager sections of train_gpt_medium.py with CPU stand-ins."""
    import dataclasses
    import math
    import types
    import torch.nn as nn
    import torch.nn.functional as F
    with open(os.path.join(ROOT, "train_gpt_medium.py")) as f:
        src = f.read()
    model_src = src[src.index("def norm(x: Tensor):"):src.index("# Distributed data loader")]
    tm_src = src[src.index("# Training Management"):src.index("# int main")]
    args = types.SimpleNamespace(
        block_size=8, train_max_seq_len=64, val_batch_size=64, bigram_vocab_size=64 * 5,
        num_scheduled_iterations=40, num_extension_iterations=2, num_iterations=42, cooldown_frac=0.7,
        split_embed_frac=2 / 3 / 4, train_bs_schedule=(64,) * 12, train_bs_extension=64,
        ws_schedule=(3, 7, 11, 13, 15, 17, 19, 21, 23, 23, 23, 23), ws_final=23, ws_validate_post_yarn_ext=27)

    class _Opt:  # stands in for NorMuonAndAdam: checks the label bookkeeping the real one asserts
        def __init__(self, named_params, param_table, scatter_order, work_order, adam_defaults, normuon_defaults):
            labels = [p.label for _, p in named_params]
            assert len(labels) == len(set(labels)) and all(l in param_table for l in labels), labels
            assert set(scatter_order) == set(labels) == set(work_order), (sorted(labels), sorted(work_order))
            self.param_cfgs = {}
        def reset(self): pass

    import gc
    g = dict(torch=torch, nn=nn, F=F, Tensor=torch.Tensor, math=math, os=os, np=np, gc=gc, dataclass=dataclasses.dataclass,
             dist=types.SimpleNamespace(get_world_size=lambda: 1), args=args, device=torch.device("cpu"),
             grad_scale=1.0, grad_accum_steps=1, world_size=1, rank=0,
             get_kernel=lambda name: types.SimpleNamespace(flash_attn_interface=_FlashRef),
             FusedSoftcappedCrossEntropy=_FusedCERef, FusedSoftcappedCrossEntropyPrefix=_FusedCEPrefixRef, NorMuonAndAdam=_Opt, _sparse_comms_active=lambda: False,
             CANON_MASK=0, VAL_CE_CHUNKS=4, PREFIX_WEIGHT=0.0, PREFIX_HOLD_FRAC=0.2854, PREFIX_END_FRAC=0.6063,
             XSA=0, XSA_LAYERS="all", ATTN_GATE=1, MUON_LR=0.015, ADAM_LR=0.004,
             FUSED_MLP=0, FusedLinearReLUSquareFunction=_fused_mlp_class(), canon_val_ce=canon_val_ce)
    g.update(flags)
    exec(compile(model_src + tm_src, "train_gpt_medium.py[model+tm]", "exec"), g)
    return g


def _smoke(flags: dict, canon_fn=None, post_init=None, seed=0):
    """Tiny-width GPT from the trainer source: one train fwd+bwd, one eval (and one masked eval)."""
    g = _load_trainer(flags)
    torch.manual_seed(seed)
    V = 1000
    model = g["GPT"](vocab_size=V, num_layers=16, num_heads=8, head_dim=16, model_dim=128, max_seq_len=64)
    with torch.no_grad():  # non-trivial weights so XSA / masking actually move the output
        for name, p in model.named_parameters():
            if p.ndim >= 2 and p.abs().sum() == 0 and name != "xsa_alphas":
                p.normal_(0, 0.02)
        model.attn_bank[:, 3 * 128:, :].normal_(0, 0.02)  # O projections (zero init upstream)
        model.mlp_bank[:, 1].normal_(0, 0.02)  # MLP c_proj (zero init upstream)
        if post_init is not None:
            post_init(model)
        # the bf16 casts train_gpt_medium.py's main applies before compiling
        for m in model.modules():
            if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
                m.weight.data = m.weight.data.bfloat16()
        for name in ("attn_gate_bank", "ve_gate_bank", "skip_gate_bank", "attn_bank", "mlp_bank", "dense_w1", "dense_w2", "dense_bs"):
            if hasattr(model, name):
                getattr(model, name).data = getattr(model, name).data.bfloat16()
    tm = g["TrainingManager"](model)
    tm.advance_schedule(1)
    T = 64
    gen = torch.Generator().manual_seed(seed)
    inp = torch.randint(0, V, (T,), dtype=torch.int32, generator=gen)
    tgt = torch.cat([inp[1:].long(), torch.randint(0, V, (1,), generator=gen)])
    seqlens = torch.tensor([0, 30, T, T], dtype=torch.int32)
    bigram = torch.randint(0, 64 * 5, (T,), dtype=torch.int32, generator=gen)
    if flags.get("PREFIX_WEIGHT", 0) > 0:
        model.prefix_table.copy_(torch.arange(model.vocab_size) // 2 - 1)  # some table of in-vocab ids / -1
    model.train()
    loss = model(inp, tgt, seqlens, bigram, tm.get_forward_args())
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"params without grad: {no_grad}"
    model.eval()
    with torch.no_grad():
        val = float(model(inp, tgt, seqlens, bigram, tm.get_forward_args()))
        val_masked = None
        if canon_fn is not None:
            model.canon_mask.copy_(canon_fn(model.vocab_size, inp, tgt))
            val_masked = float(model(inp, tgt, seqlens, bigram, tm.get_forward_args()))
    _smoke.last = dict(g=g, tm=tm, batch=(inp, tgt, seqlens, bigram))
    return float(loss.detach()), val, val_masked, model


def _random_sound_mask(V, inp, tgt):
    """Random 30% mask with every (prev, target) pair of the batch cleared, bit-packed like the real one."""
    bits = torch.rand(V, V, generator=torch.Generator().manual_seed(1)) < 0.3
    bits[inp.long(), tgt] = False
    return torch.from_numpy(np.packbits(bits.numpy(), axis=1, bitorder="little"))


def test_model_smoke():
    params = lambda m: {n for n, _ in m.named_parameters()}
    _FusedCERef.calls = _FusedCEPrefixRef.calls = 0
    base = _smoke({})
    # flags off: #261's 8-argument base CE, never the prefix variant, and none of this tree's state
    assert _FusedCERef.calls == 1 and _FusedCEPrefixRef.calls == 0
    extra = {"xsa_alphas", "prefix_table", "canon_mask"} & (params(base[3]) | {n for n, _ in base[3].named_buffers()})
    assert not extra and "attn_gate_bank" in params(base[3]), extra
    # XSA at its zero init is an exact identity: train and eval losses match the baseline bit-for-bit
    xsa = _smoke({"XSA": 1})
    assert "xsa_alphas" in params(xsa[3])
    assert xsa[0] == base[0] and xsa[1] == base[1], (xsa[:2], base[:2])
    # ... and a nonzero alpha on the selected layers changes the output
    def set_alpha(m):
        m.xsa_alphas.fill_(0.5)
    xsa_on = _smoke({"XSA": 1, "XSA_LAYERS": "3,15"}, post_init=set_alpha)
    assert xsa_on[0] != base[0]
    nogate = _smoke({"ATTN_GATE": 0})
    assert "attn_gate_bank" not in params(nogate[3]) and nogate[0] != base[0]
    # prefix weight adds a positive term to the training loss; eval is unchanged
    pfx = _smoke({"PREFIX_WEIGHT": 0.25})
    assert pfx[0] > base[0] and pfx[1] == base[1] and _FusedCEPrefixRef.calls == 1
    # canonical mask: all-zero == no-op; a sound mask can only lower the eval loss; chunking is exact
    zero = lambda V, i, t: torch.zeros(V, V // 8, dtype=torch.uint8)
    cz = _smoke({"CANON_MASK": 1}, canon_fn=zero)
    assert cz[1] == base[1] and cz[2] == base[1]
    cm = _smoke({"CANON_MASK": 1, "VAL_CE_CHUNKS": 8}, canon_fn=_random_sound_mask)
    assert abs(cm[1] - base[1]) < 1e-5 and cm[2] < cm[1], (cm[1], cm[2], base[1])
    # fused relu^2 MLP: same math as the inline path up to bf16 rounding, in loss and in every gradient
    fm = _smoke({"FUSED_MLP": 1})
    assert abs(fm[0] - base[0]) < 2e-3 * abs(base[0]) and abs(fm[1] - base[1]) < 2e-3 * abs(base[1]), (fm[:2], base[:2])
    for (n, p), (_, q) in zip(base[3].named_parameters(), fm[3].named_parameters()):
        cos = torch.nn.functional.cosine_similarity(p.grad.float().flatten(), q.grad.float().flatten(), dim=0)
        assert p.grad.norm() == 0 or cos > 0.995, (n, float(cos))
    # everything at once
    _smoke({"XSA": 1, "ATTN_GATE": 0, "PREFIX_WEIGHT": 0.25, "CANON_MASK": 1, "FUSED_MLP": 1}, canon_fn=_random_sound_mask)
    print(f"  fwd/bwd OK for base, XSA, ATTN_GATE=0, PREFIX_WEIGHT, CANON_MASK, FUSED_MLP and all combined; "
          f"masked eval {cm[2]:.4f} < unmasked {cm[1]:.4f}")


def _eval_loss(model, tm, batch):
    with torch.no_grad():
        return model(*batch, tm.get_forward_args())


def test_val_chunks_give_the_same_mean():
    """VAL_CE_CHUNKS only sizes the logit block: the eval loss of one tiny model is the full-batch mean at
    1, 2, 4, 8, 16 and 32 chunks, unmasked and masked; the masked op equals the inline form it replaced."""
    _, base_val, _, model = _smoke({"CANON_MASK": 1}, canon_fn=_random_sound_mask)
    g, tm, batch = _smoke.last["g"], _smoke.last["tm"], _smoke.last["batch"]
    inp, tgt = batch[0], batch[1]
    masked = {}
    for n in (1, 2, 4, 8, 16, 32):
        g["VAL_CE_CHUNKS"] = n
        masked[n] = float(_eval_loss(model, tm, batch))
    model.canon_mask.zero_()
    unmasked = {}
    for n in (1, 2, 4, 8, 16, 32):
        g["VAL_CE_CHUNKS"] = n
        unmasked[n] = float(_eval_loss(model, tm, batch))
    g["CANON_MASK"] = 0  # PR #261's inline path, same weights, no mask buffer read
    inline = {}
    for n in (1, 4, 8):
        g["VAL_CE_CHUNKS"] = n
        inline[n] = float(_eval_loss(model, tm, batch))
    for d in (masked, unmasked, inline):
        assert max(d.values()) - min(d.values()) < 2e-6, d
    assert abs(unmasked[1] - inline[1]) < 2e-6 and masked[1] < unmasked[1], (masked, unmasked, inline)

    # the op against the inline masked loss this tree computed before (same chunks, same formula)
    gen = torch.Generator().manual_seed(3)
    V, rows, d = 1024, 256, 32
    x = torch.randn(rows, d, generator=gen).bfloat16()
    w = torch.randn(d, V, generator=gen).bfloat16()
    t = torch.randint(0, V, (rows,), generator=gen)
    i = torch.randint(0, V, (rows,), generator=gen).int()
    mask = _random_sound_mask(V, i, t)
    def inline_masked(n):
        loss = 0
        for xr, tc, ic in zip(x.chunk(n), t.chunk(n), i.chunk(n)):
            logits = 23 * torch.sigmoid(((xr @ w).float() + 5) / 7.5)
            shifts = torch.arange(8, dtype=torch.uint8)
            dropped = (mask[ic, :, None] >> shifts & 1).view(logits.shape).bool()
            loss += torch.nn.functional.cross_entropy(logits.masked_fill(dropped, -60.0), tc, reduction="mean") / n
        return float(loss)
    full = inline_masked(1)
    for n in (1, 8, 64):
        assert abs(float(canon_val_ce(x, w, t, i, mask, n)) - inline_masked(n)) < 1e-5
        assert abs(float(canon_val_ce(x, w, t, i, mask, n)) - full) < 1e-5
    print(f"  eval loss equal at 1..32 chunks: masked {masked[8]:.6f}, unmasked {unmasked[8]:.6f}")


def test_run_validation_retries_out_of_memory():
    """run_validation repeats a pass that ran out of memory with twice the chunks and returns the same mean;
    other errors, and an out-of-memory error at the chunk limit, still propagate."""
    import pytest
    model = _smoke({"CANON_MASK": 1}, canon_fn=_random_sound_mask)[3]
    g, tm, batch = _smoke.last["g"], _smoke.last["tm"], _smoke.last["batch"]
    rows = batch[0].numel()
    attempts = []

    def flaky(*a):
        attempts.append(g["VAL_CE_CHUNKS"])
        if g["VAL_CE_CHUNKS"] < 16:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 6.14 GiB")
        return model(*a)
    make_loader = lambda: iter([batch + (None,)] * 3)
    g["VAL_CE_CHUNKS"] = 4
    model.eval()
    got = g["run_validation"](flaky, make_loader, 3, tm.get_forward_args, rows, "final_val")
    assert attempts == [4, 8, 16, 16, 16] and g["VAL_CE_CHUNKS"] == 16, attempts
    g["VAL_CE_CHUNKS"] = 4
    want = float(_eval_loss(model, tm, batch))
    assert abs(float(got) - want) < 2e-6, (float(got), want)
    assert g["PHASE"]["phase"] == "final_val" and g["PHASE"]["tokens"] == rows

    def broken(*a):
        raise RuntimeError("NCCL error")
    with pytest.raises(RuntimeError, match="NCCL"):
        g["run_validation"](broken, make_loader, 3, tm.get_forward_args, rows, "val")
    g["VAL_CE_CHUNKS"] = 32  # one retry at 64 chunks (the limit), then the error propagates
    def always_oom(*a):
        raise RuntimeError("CUDA error: out of memory")  # wrapped / non-typed form is recognized too
    with pytest.raises(RuntimeError, match="out of memory"):
        g["run_validation"](always_oom, make_loader, 3, tm.get_forward_args, rows, "val")
    assert g["VAL_CE_CHUNKS"] == 64
    assert g["is_oom"](RuntimeError("compile failed")) is False
    wrapped = RuntimeError("backend failed")
    wrapped.__cause__ = torch.OutOfMemoryError("CUDA out of memory")
    assert g["is_oom"](wrapped)
    lines = g["oom_lines"]()
    assert lines[0].startswith("OOM rank=0 phase=val step=None tokens=64 max_seq_len=64 VAL_CE_CHUNKS=64"), lines[0]


def test_dynamo_recompiles_when_val_ce_chunks_changes():
    """The OOM retry relies on torch.compile guarding the VAL_CE_CHUNKS global the eval forward reads."""
    import torch._dynamo
    _, _, _, model = _smoke({"CANON_MASK": 1}, canon_fn=_random_sound_mask)
    g, tm, batch = _smoke.last["g"], _smoke.last["tm"], _smoke.last["batch"]

    class _FlashTraceable(_FlashRef):  # the same attention without unique(), so the forward is one graph as on GPU
        @staticmethod
        def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, softmax_scale, window_size):
            T = q.size(0)
            doc = torch.searchsorted(cu_seqlens_q, torch.arange(T), right=True)  # repeated ends sit past T
            i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
            allowed = (doc[:, None] == doc[None, :]) & (j <= i) & (i - j <= window_size[0])
            att = torch.einsum("thd,shd->hts", q.float(), k.float()) * softmax_scale
            att = att.masked_fill(~allowed, float("-inf")).softmax(-1)
            return torch.einsum("hts,shd->thd", att, v.float()).type_as(q)
    g["flash_attn_interface"] = _FlashTraceable
    g["VAL_CE_CHUNKS"] = 4
    eager = float(_eval_loss(model, tm, batch))
    compiles = []
    def backend(gm, example_inputs):
        compiles.append(gm)
        return gm.forward
    torch._dynamo.reset()
    cmodel = torch.compile(model, backend=backend, dynamic=False, fullgraph=True)  # as train_gpt_medium.py compiles it
    g["VAL_CE_CHUNKS"] = 4
    a = float(_eval_loss(cmodel, tm, batch))
    first = len(compiles)
    float(_eval_loss(cmodel, tm, batch))
    assert len(compiles) == first, "a repeat call recompiled"
    g["VAL_CE_CHUNKS"] = 8
    b = float(_eval_loss(cmodel, tm, batch))
    assert len(compiles) > first, "changing VAL_CE_CHUNKS did not recompile"
    assert any("canon_val_ce" in str(gm.graph) and ", 8)" in str(gm.graph) for gm in compiles[first:]), \
        [str(gm.graph)[-300:] for gm in compiles[first:]]
    assert abs(a - b) < 2e-6 and abs(a - eager) < 2e-6
    torch._dynamo.reset()


def test_inductor_keeps_masked_chunks_apart():
    """182/val_ce_memory.py: the inline masked loss holds every chunk's logit block at once, the op holds none
    in the compiled graph. Opt-in (T2GPU_INDUCTOR_PROBE=1: it compiles with Inductor's C++ backend, minutes of
    CPU); bash 182/val_ce_memory.sh runs the same measurement in docker."""
    import pytest
    if os.environ.get("T2GPU_INDUCTOR_PROBE") != "1":
        pytest.skip("opt-in: T2GPU_INDUCTOR_PROBE=1 (or bash 182/val_ce_memory.sh)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import val_ce_memory
    try:
        res = val_ce_memory.measure()
    except Exception as e:  # no working C++ compiler for Inductor here
        pytest.skip(f"Inductor cannot compile on this machine ({type(e).__name__}); use bash 182/val_ce_memory.sh")
    assert res[(8, "inline masked")][0] > 4 * res[(8, "inline unmasked")][0], res
    assert res[(8, "op masked")][1] == 0 and abs(res[(8, "op masked")][2] - res[(8, "inline masked")][2]) < 1e-4, res


if __name__ == "__main__":
    for t in [test_kernels_file_is_base_plus_prefix_section, test_surplus_none_grads, test_prefix_schedule, test_ce_kernel_logic, test_model_smoke,
              test_val_chunks_give_the_same_mean, test_run_validation_retries_out_of_memory, test_dynamo_recompiles_when_val_ce_chunks_changes,
              test_prefix_table, test_canonical_mask]:
        print(t.__name__)
        t()
    print("all CPU checks passed")
