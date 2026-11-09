"""Probes of the Triton interpreter's numerics against torch: the ops our kernels rely on.

    bash 182/tools/triton_cpu/run.sh python -m pytest 182/tools/triton_cpu -q
    TRITON_CPU_NO_SHIM=1 bash 182/tools/triton_cpu/run.sh python -m pytest 182/tools/triton_cpu -q   # raw interpreter

Torch is the reference for every cast (round-to-nearest-even). GPU fp32 -> fp8 casts are `satfinite`;
torch's are not, so the reference saturates the out-of-range inputs itself.
"""
import os

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("interpreter probes: run under TRITON_INTERPRET=1 (182/tools/triton_cpu/run.sh)", allow_module_level=True)

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32, E4: tl.float8e4nv,
      E5: tl.float8e5}
FMAX = {E4: 448.0, E5: 57344.0}


@triton.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, ACC: tl.constexpr,
                TRANS_B: tl.constexpr):
    om, on, ok = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
    if TRANS_B:  # b stored [N, K], used as b.T (the mlp kernel's `tl.dot(a, b.T, acc)`)
        b = tl.load(b_ptr + on[:, None] * K + ok[None, :]).T
    else:
        b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
    if ACC:
        acc = tl.load(c_ptr + om[:, None] * N + on[None, :])
        acc = tl.dot(a, b, acc)
    else:
        acc = tl.dot(a, b)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


@pytest.mark.parametrize("da,db", [(torch.bfloat16, torch.bfloat16), (E4, E4), (E5, E4), (E5, E5),
                                   (torch.float16, torch.float16)])
@pytest.mark.parametrize("shape", [(16, 16, 16), (128, 128, 128), (64, 32, 128)])
@pytest.mark.parametrize("acc", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
def test_dot(da, db, shape, acc, trans_b):
    M, N, K = shape
    g = torch.Generator().manual_seed(M + N + K)
    a = (4 * torch.randn(M, K, generator=g)).to(da)
    b = (4 * torch.randn(N, K, generator=g)).to(db)
    c0 = torch.randn(M, N, generator=g)
    c = c0.clone() if acc else torch.empty(M, N)
    _dot_kernel[(1,)](a, b if trans_b else b.T.contiguous(), c, M, N, K, acc, trans_b)
    want = a.double() @ b.double().T + (c0.double() if acc else 0)
    err = ((c.double() - want).abs().max() / want.abs().max()).item()
    assert err < 1e-6, err


@triton.jit
def _cast_kernel(x_ptr, y_ptr, n, DST: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(y_ptr + o, tl.load(x_ptr + o, mask=m).to(DST), mask=m)


def _cast(x, dst):
    y = torch.empty(x.shape, dtype=dst)
    _cast_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), TL[dst], 1024)
    return y


def _edge_values(fmt):
    """Every representable value of fmt, the midpoints between neighbours (ties), points just off the
    midpoints, the subnormal range, and the clamp edges."""
    allv = torch.arange(256, dtype=torch.uint8).view(fmt).float()
    allv = allv[torch.isfinite(allv)].unique()
    pos = allv[allv >= 0]
    mids = (pos[1:] + pos[:-1]) / 2
    off = torch.cat((mids * (1 + 2 ** -20), mids * (1 - 2 ** -20)))
    lim = FMAX[fmt]
    edge = torch.tensor([lim, lim * 1.01, lim * 1.07, lim * 1.2, 2 * lim, 1e30, 0.0, -0.0, 2.0 ** -30, 2.0 ** -17])
    v = torch.cat((pos, mids, off, edge))
    return torch.cat((v, -v))


def _ref_cast(x, fmt):
    """GPU fp32 -> fp8 (`cvt.rn.satfinite`): round to nearest even, out-of-range saturates to +-max."""
    lim = FMAX[fmt]
    return torch.where(x.abs() > lim, x.sign() * lim, x).to(fmt)


@pytest.mark.parametrize("fmt", [E4, E5])
def test_fp32_to_fp8_cast_exhaustive_edges(fmt):
    x = _edge_values(fmt)
    got, want = _cast(x, fmt).view(torch.uint8), _ref_cast(x, fmt).view(torch.uint8)
    bad = (got != want).nonzero().flatten()
    assert bad.numel() == 0, [(x[i].item(), got[i].item(), want[i].item()) for i in bad[:8]]


@pytest.mark.parametrize("fmt", [E4, E5])
def test_fp32_to_fp8_cast_random(fmt):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1 << 16, generator=g) * torch.exp2(torch.randint(-14, 9, (1 << 16,), generator=g).float())
    x = x.clamp(-FMAX[fmt], FMAX[fmt])  # in-range, as the kernels guarantee with an explicit clamp
    assert torch.equal(_cast(x, fmt).view(torch.uint8), x.to(fmt).view(torch.uint8))


@pytest.mark.parametrize("fmt", [E4, E5])
def test_fp8_to_fp32_cast(fmt):
    bits = torch.arange(256, dtype=torch.uint8)
    bits = bits[torch.isfinite(bits.view(fmt).float())]
    assert torch.equal(_cast(bits.view(fmt), torch.float32), bits.view(fmt).float())


def test_fp32_to_bf16_cast_rounds_to_nearest_even():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1 << 16, generator=g)
    bf = torch.randn(4096, generator=g).bfloat16().float()
    step = torch.finfo(torch.bfloat16).eps * bf.abs()
    x = torch.cat((x, bf + step / 2, bf - step / 4, bf + step * 0.75))  # exact ties and off-tie points
    assert torch.equal(_cast(x, torch.bfloat16).view(torch.int16), x.bfloat16().view(torch.int16))
    assert torch.equal(_cast(x.bfloat16(), torch.float32), x.bfloat16().float())


@triton.jit
def _bf16_math_kernel(x_ptr, y_ptr, relu_ptr, sq_ptr, mix_ptr, fma_ptr, sel_ptr, cmp_ptr, max_ptr, n,
                      BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(x_ptr + o, mask=m)  # bf16
    y = tl.load(y_ptr + o, mask=m)  # bf16
    r = tl.maximum(x, 0)             # bf16 op with an int scalar (the mlp forward's relu)
    tl.store(relu_ptr + o, r, mask=m)
    tl.store(sq_ptr + o, r * r, mask=m)                       # bf16 * bf16 -> bf16 (rounded)
    tl.store(mix_ptr + o, (x + y) - x * y * 3, mask=m)  # all bf16 (tl.minimum/maximum promote to fp32 in 3.6)
    tl.store(fma_ptr + o, tl.fma(x, y, x), mask=m)
    tl.store(sel_ptr + o, tl.where(x > y, x, y + 1), mask=m)
    tl.store(cmp_ptr + o, (x >= y).to(tl.int32), mask=m)
    tl.store(max_ptr, tl.max(tl.where(m, x, -1e30), axis=0))  # bf16 reduction (promoted to fp32)


def test_bf16_elementwise():
    n, B = 1000, 1024
    g = torch.Generator().manual_seed(2)
    x, y = torch.randn(n, generator=g).bfloat16(), torch.randn(n, generator=g).bfloat16()
    out = {k: torch.empty(n, dtype=torch.bfloat16) for k in ("relu", "sq", "mix", "fma", "sel")}
    cmp, mx = torch.empty(n, dtype=torch.int32), torch.empty(1)
    _bf16_math_kernel[(1,)](x, y, out["relu"], out["sq"], out["mix"], out["fma"], out["sel"], cmp, mx, n, B)
    r = torch.relu(x)
    want = dict(relu=r, sq=r * r, mix=(x + y) - x * y * 3, sel=torch.where(x > y, x, y + 1))
    for k, w in want.items():
        assert torch.equal(out[k].view(torch.int16), w.view(torch.int16)), k
    # fma: one rounding on the GPU (fp32 math, then bf16); torch's bf16 ops round twice, so compare loosely
    assert torch.allclose(out["fma"].float(), (x.float() * y.float() + x.float()).bfloat16().float(), rtol=1e-2, atol=1e-2)
    assert torch.equal(cmp, (x >= y).int()) and mx.item() == x.float().max().item()


@triton.jit
def _int_to_bf16_kernel(i_ptr, b_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(b_ptr + o, tl.load(i_ptr + o).to(tl.bfloat16))


def test_int_to_bf16_cast():
    i = torch.tensor([0, 1, -1, 3, 255, 257, 1 << 20, -(1 << 24) - 1], dtype=torch.int32)
    b = torch.empty(8, dtype=torch.bfloat16)
    _int_to_bf16_kernel[(1,)](i, b, 8)
    assert torch.equal(b.view(torch.int16), i.bfloat16().view(torch.int16))


@triton.jit
def _atomic_max_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    v = tl.load(x_ptr + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK))
    tl.atomic_max(out_ptr, tl.max(tl.abs(v), axis=0))
    tl.atomic_max(out_ptr + 1 + tl.program_id(0) % 2, tl.max(v, axis=0))


def test_atomic_max_float():
    x = torch.randn(64 * 128)
    x[777] = 9.5
    out = torch.tensor([0.0, -1e30, -1e30])
    _atomic_max_kernel[(64,)](x, out, 128)
    xs = x.view(64, 128)
    assert out[0].item() == x.abs().max().item()
    assert out[1].item() == xs[0::2].max().item() and out[2].item() == xs[1::2].max().item()


@triton.jit
def _trans_split_kernel(x_ptr, t_ptr, s0_ptr, s1_ptr, M: tl.constexpr, N: tl.constexpr):
    om, on = tl.arange(0, M), tl.arange(0, N)
    x = tl.load(x_ptr + om[:, None] * N + on[None, :])
    tl.store(t_ptr + on[:, None] * M + om[None, :], tl.trans(x))
    # the mlp kernel's accumulator split into its two column halves
    acc = tl.permute(tl.reshape(x, (M, 2, N // 2)), (0, 2, 1))
    a0, a1 = tl.split(acc)
    oh = tl.arange(0, N // 2)
    tl.store(s0_ptr + om[:, None] * (N // 2) + oh[None, :], a0)
    tl.store(s1_ptr + om[:, None] * (N // 2) + oh[None, :], a1)


def test_trans_reshape_permute_split():
    M, N = 32, 64
    x = torch.randn(M, N)
    t, s0, s1 = torch.empty(N, M), torch.empty(M, N // 2), torch.empty(M, N // 2)
    _trans_split_kernel[(1,)](x, t, s0, s1, M, N)
    assert torch.equal(t, x.T) and torch.equal(s0, x[:, : N // 2]) and torch.equal(s1, x[:, N // 2:])


@triton.jit
def _persistent_kernel(out_ptr, num_tiles, NUM_PROGS: tl.constexpr, BLOCK: tl.constexpr):
    for tile in tl.range(tl.program_id(0), num_tiles, NUM_PROGS, flatten=True):
        for k in range(3):  # an inner loop, as the GEMM mainloop under the flattened tile loop
            o = tile * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + o, tl.load(out_ptr + o) + tile + 1.0)


def test_range_flatten_persistent_loop():
    num_tiles, block = 11, 16
    out = torch.zeros(num_tiles * block)
    _persistent_kernel[(4,)](out, num_tiles, 4, block)
    want = 3 * (torch.arange(num_tiles).float() + 1).repeat_interleave(block)
    assert torch.equal(out, want)


@triton.jit
def _desc_kernel(a_desc, o_desc, t_desc, BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0) * BM, tl.program_id(1) * BN
    a = a_desc.load([pm, pn])
    o_desc.store([pm, pn], (a.to(tl.float32) * 2).to(tl.float8e4nv))
    t_desc.store([pn, pm], tl.trans(a))


def test_tensor_descriptor_load_store():
    from triton.tools.tensor_descriptor import TensorDescriptor
    M, N, BM, BN = 64, 96, 32, 32
    a = torch.randn(M, N).bfloat16()
    o, t = torch.empty(M, N, dtype=E4), torch.empty(N, M, dtype=torch.bfloat16)
    _desc_kernel[(M // BM, N // BN)](TensorDescriptor.from_tensor(a, [BM, BN]), TensorDescriptor.from_tensor(o, [BM, BN]),
                                     TensorDescriptor.from_tensor(t, [BN, BM]), BM, BN)
    assert torch.equal(o.view(torch.uint8), (a.float() * 2).to(E4).view(torch.uint8))
    assert torch.equal(t, a.T)
