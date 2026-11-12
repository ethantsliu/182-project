"""Compile-only probe for the butterfly question (182/fp8-attn-o/README.md): a quantize kernel that rotates a
128 x 128 bf16 tile along its last dim, compiled for H100 (sm_90a) without a GPU, as
  plain       no rotation
  dense128/32 the block dot HADAMARD_FP8 uses (perf/kernels/hadamard.py rotate_last)
  fwht128/32  an in-register fast Walsh-Hadamard transform: log2(b) butterfly stages
  h16+3bfly   H_128 = H_8 (x) H_16: one K=16 dot, then 3 butterfly stages
Prints shared memory, registers and static SASS counts (every kernel is straight-line, so the counts are what
each thread executes per tile). It measures no time: nothing here runs.

    TRITON_INTERPRET=0 bash 182/tools/triton_cpu/run.sh python 182/tools/triton_cpu/butterfly_probe.py
"""
import os, re, subprocess, sys, tempfile
import torch, triton, triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.runtime import driver

class _Fake:
    def get_current_device(self): return 0
    def get_current_stream(self, device=None): return 0
    def get_current_target(self): return GPUTarget("cuda", 90, 32)
    def get_active_torch_device(self): return torch.device("cpu")
driver.set_active(_Fake())

@triton.jit
def _bfly(x, R: tl.constexpr, C: tl.constexpr, S: tl.constexpr):
    # one stage pairing elements S apart along the last dim
    y = tl.reshape(x, (R, C // (2 * S), 2, S))
    y = tl.permute(y, (0, 1, 3, 2))
    a, b = tl.split(y)
    y = tl.join(a + b, a - b)
    y = tl.permute(y, (0, 1, 3, 2))
    return tl.reshape(y, (R, C))

@triton.jit
def probe(src_ptr, q_ptr, out_ptr, scale_ptr, N, MODE: tl.constexpr, HB: tl.constexpr, BLOCK: tl.constexpr):
    offs_m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tile = tl.load(src_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :])
    if MODE == 0:      # plain
        y = tile.to(tl.float32)
    elif MODE == 1:    # dense block dot, HB wide
        offs = tl.arange(0, HB)
        q = tl.load(q_ptr + offs[:, None] * HB + offs[None, :])
        if HB == BLOCK:
            y = tl.dot(tile, q)
        else:
            y = tl.reshape(tl.dot(tl.reshape(tile, (BLOCK * BLOCK // HB, HB)), q), (BLOCK, BLOCK))
        y = y * 0.0883883476
    elif MODE == 2:    # full FWHT over HB (7 stages at 128), fp32 registers
        y = tile.to(tl.float32)
        if HB > 1:
            y = _bfly(y, BLOCK, BLOCK, 1)
        if HB > 2:
            y = _bfly(y, BLOCK, BLOCK, 2)
        if HB > 4:
            y = _bfly(y, BLOCK, BLOCK, 4)
        if HB > 8:
            y = _bfly(y, BLOCK, BLOCK, 8)
        if HB > 16:
            y = _bfly(y, BLOCK, BLOCK, 16)
        if HB > 32:
            y = _bfly(y, BLOCK, BLOCK, 32)
        if HB > 64:
            y = _bfly(y, BLOCK, BLOCK, 64)
        y = y * 0.0883883476
    else:              # H16 dot then 3 butterfly stages (16, 32, 64 apart): H128 = H8 (x) H16
        offs = tl.arange(0, 16)
        q = tl.load(q_ptr + offs[:, None] * 16 + offs[None, :])
        y = tl.reshape(tl.dot(tl.reshape(tile, (BLOCK * BLOCK // 16, 16)), q), (BLOCK, BLOCK))
        y = _bfly(y, BLOCK, BLOCK, 16)
        y = _bfly(y, BLOCK, BLOCK, 32)
        y = _bfly(y, BLOCK, BLOCK, 64)
        y = y * 0.0883883476
    inv = 1.0 / tl.load(scale_ptr)
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :],
             tl.maximum(tl.minimum(y * inv, 57344.0), -57344.0).to(tl.float8e5))

def tool(n): return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", n)

src = torch.empty(256, 256, dtype=torch.bfloat16); out = torch.empty(256, 256, dtype=torch.float8_e5m2)
q = torch.empty(128, 128, dtype=torch.bfloat16); sc = torch.ones(1)
for name, mode, hb in (("plain", 0, 128), ("dense128", 1, 128), ("dense32", 1, 32), ("fwht128", 2, 128), ("fwht32", 2, 32), ("h16+3bfly", 3, 128)):
    try:
        k = probe.warmup(src, q, out, sc, 256, MODE=mode, HB=hb, BLOCK=128, grid=(2, 2), num_warps=8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.ptx"); open(p, "w").write(k.asm["ptx"])
            o = subprocess.run([tool("ptxas"), "-arch=sm_90a", "-v", p, "-o", os.path.join(d, "k.cubin")], capture_output=True, text=True)
            txt = o.stdout + o.stderr
            regs = re.search(r"Used (\d+) registers", txt); spill = re.search(r"(\d+) bytes spill stores", txt)
            s = subprocess.run([tool("cuobjdump"), "-sass", os.path.join(d, "k.cubin")], capture_output=True, text=True).stdout
        insts = [l for l in s.splitlines() if ";" in l and re.search(r"/\*[0-9a-f]{4}\*/", l)]
        cnt = lambda pat: sum(bool(re.search(pat, l)) for l in insts)
        print(f"{name:10s} ok smem={k.metadata.shared} B regs={regs.group(1) if regs else '?'} spill={spill.group(1) if spill else '?'} "
              f"sass_insts={len(insts)} smem_ld={cnt(r'LDS')} smem_st={cnt(r'STS')} fadd={cnt(r'FADD')} hmma={cnt(r'HGMMA|HMMA')} bar={cnt(r'BAR')}")
    except Exception as e:
        print(f"{name:10s} FAILED {type(e).__name__}: {str(e).strip().splitlines()[-1][:300] if str(e).strip() else ''}")
