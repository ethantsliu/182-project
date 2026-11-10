"""Compile the record's MLP / Hadamard Triton kernels for H100 (sm_90a) on a CPU-only machine.

    bash 182/tools/triton_cpu/run.sh aot                 # every variant: errors, smem per stage, regs, spills
    bash 182/tools/triton_cpu/run.sh aot --base <sha>    # + record-path SASS / PTX identical to <sha>'s?

What it covers. The host launchers run on CPU tensors, and every `kernel[grid](...)` they issue is turned
into a JIT warmup against a fake CUDA driver whose target is sm_90, so the kernel goes through the frontend,
the TTGIR/LLVM passes and the wheel's ptxas with the descriptors, dtypes, constexprs and integer
specializations (==1, %16) that launcher builds. Nothing runs; no tensor is read. Called this way:
  record     quantize_mlp_weights_dual, quantize_dual_layout_fused (fp8 transpose), linear_relu_square
             (bf16 forward, fp8 forward, fp8 backward): through the launchers, with gpt.py's arguments
  rotated    linear_relu_square with the hadamard_q / rot_row / rot_t kwargs FusedLinearReLUSquareFunction
             passes, mlp.quantize_grad (dpost / dw2 / both), quantize_weights_rotated (W1 col, W2 row):
             through the launchers, with HADAMARD_BLOCK / HADAMARD_GEMMS patched per variant;
             hadamard_t_quant (x_t) via its Python body (its registered CPU kernel is a torch reference)
Geometry: the record's D = 768, H = MLP_HIDDEN_DIM read from the tree's model/gpt.py (2816), L = 12 bank
slots, and every per-rank token count the training stages reach (batch_size / 8: 16384, 32768, 40960,
49152), plus the validation batch (262144) for the bf16 forward. Per kernel it prints whether those token
counts compile to one binary (they share a specialization) or several.
Per kernel and num_stages: shared memory (H100 opt-in limit 232448 B), ptxas registers / spills. For
linear_relu_square every num_stages 4..1 is compiled, and the stage the launcher would end up with is
printed: its default (4 for the bf16 forward, else 3), stepped down while smem exceeds the limit, as its
OutOfResources retry does on the GPU.

Not covered: the torch.compile / inductor path (it launches the same kernels with the same arguments, but
the inductor wrapper is not exercised), the _scaled_mm GEMMs (cuBLAS), any kernel outside the MLP, and
numerics (that is the interpreter's job) or timing.

With --base, the record (flags-off) variants are compiled from both trees through the launcher API they
share and compared kernel by kernel: SASS (cuobjdump -sass, addresses stripped) and PTX (line info,
comments and the .file table stripped). Identical SASS means the change cannot alter the record's kernels.
cuobjdump failing, or a disassembly with no instructions, is an error, never "identical".
Adapted from rv-kern's aot_sm90.py / sass_cmp.py.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import traceback

assert os.environ.get("TRITON_INTERPRET", "0") != "1", "compile_check needs TRITON_INTERPRET=0"

import torch  # noqa: E402
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.runtime import driver  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
                 help="tree whose track_1_short is compiled")
_ap.add_argument("--record-only", action="store_true",
                 help="only the flags-off variants, through the API upstream has (works on a pristine upstream tree)")
_ap.add_argument("--sass-dir", help="write normalized SASS and PTX of the record variants here")
_ap.add_argument("--base", help="a second tree (git archive of the base commit): compare record SASS / PTX")
ARGS = _ap.parse_args()
sys.path.insert(0, ARGS.root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H100_SMEM = 232448
H100_SMS = 132
TARGET = GPUTarget("cuda", 90, 32)
WORLD = 8
D = 768          # model_dim (train_gpt.py)
SLOTS = 12       # NUM_MLP_SLOTS (model/gpt.py)
SEED = 1234      # HADAMARD_SEED default


def record_geometry():
    """(H, training token counts per rank, validation token count per rank) of the tree being compiled."""
    with open(os.path.join(ARGS.root, "track_1_short", "model", "gpt.py")) as f:
        m = re.search(r"^MLP_HIDDEN_DIM\s*=\s*(\d+)", f.read(), re.M)
    assert m, "MLP_HIDDEN_DIM not found in model/gpt.py"
    from track_1_short.config import TRAINING_STAGES, Hyperparameters
    tokens = sorted({s.batch_size // WORLD for s in TRAINING_STAGES})
    return int(m.group(1)), tokens, Hyperparameters().val_batch_size // WORLD


class _FakeDriver:
    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return TARGET

    def get_active_torch_device(self):
        return torch.device("cpu")


RECORDS = []  # (kernel name, variant flags, num_stages, launcher default num_stages, CompiledKernel or exception)
_orig_run = JITFunction.run


def _warmup_run(self, *args, grid, warmup, **kwargs):
    """Compile instead of launching. For linear_relu_square_kernel, compile every num_stages 4..1."""
    default = kwargs.get("num_stages", 3)
    stages = [default]
    if self.fn.__name__ == "linear_relu_square_kernel":
        stages = [4, 3, 2, 1]
    key = {k: v for k, v in kwargs.items() if k.isupper() and k not in ("NUM_SMS", "BLOCK_SIZE_M", "BLOCK_SIZE_N")}
    for st in stages:
        kw = dict(kwargs, num_stages=st)
        try:
            k = _orig_run(self, *args, grid=grid, warmup=True, **kw)
            RECORDS.append((self.fn.__name__, key, st, default, k))
        except Exception as e:  # a frontend / compiler error is the finding
            RECORDS.append((self.fn.__name__, key, st, default, e))
    return None


def _tool(name):
    return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", name)


_PTXAS = {}


def ptxas_stats(kernel):
    if kernel.hash in _PTXAS:
        return _PTXAS[kernel.hash]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.ptx")
        with open(src, "w") as f:
            f.write(kernel.asm["ptx"])
        out = subprocess.run([_tool("ptxas"), "-arch=sm_90a", "-v", src, "-o", os.path.join(d, "k.cubin")],
                             capture_output=True, text=True)
    txt = out.stdout + out.stderr
    if out.returncode:
        raise RuntimeError(f"ptxas failed (rc {out.returncode}): {txt[-500:]}")
    regs = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", txt)
    res = (int(regs.group(1)) if regs else None, int(spill.group(1)) if spill else None,
           int(spill.group(2)) if spill else None)
    _PTXAS[kernel.hash] = res
    return res


def sass_text(kernel):
    """Normalized SASS (one instruction per line, addresses and encodings stripped). Raises on a cuobjdump
    failure or a disassembly without instructions: an empty string must never compare equal."""
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(kernel.asm["cubin"])
        f.flush()
        out = subprocess.run([_tool("cuobjdump"), "-sass", f.name], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"cuobjdump -sass failed (rc {out.returncode}): {out.stderr.strip()[-500:]}")
    lines = [re.sub(r"/\*[0-9a-f]{4,}\*/", "", ln).strip() for ln in out.stdout.splitlines()]
    insts = [ln for ln in lines if re.match(r"^[A-Z@{]", ln) and ";" in ln]
    if not insts:
        raise RuntimeError(f"cuobjdump -sass printed no instructions (stderr: {out.stderr.strip()[-300:]!r})")
    return "\n".join(insts)


def ptx_text(kernel):
    """PTX without what only records where the source lives: .loc / .file directives, comments and the
    DWARF sections (line numbers move whenever the file is edited, even far from the kernel)."""
    ptx = re.sub(r"\.section\s+\.debug_\w+\s*\{.*?\n\s*\}", "", kernel.asm["ptx"], flags=re.S)
    keep = []
    for ln in ptx.splitlines():
        s = ln.split("//", 1)[0].rstrip()
        if not s.strip() or re.match(r"\s*\.(loc|file)\b", s):
            continue
        keep.append(s)
    if not any(re.match(r"\s*\.(visible\s+)?\.entry\b", ln) for ln in keep):
        raise RuntimeError("PTX has no .entry")
    return "\n".join(keep)


RECORD_PREFIX = "record: "


def _scalars(slot):
    """The per-slot scale / amax views gpt.py passes (0-d, (1,) and [slot:slot+1] views of [SLOTS] buffers):
    slot 0 is 16-byte aligned, slot 5 is not, and Triton specializes pointer arguments on that."""
    scal, amax = torch.ones(SLOTS), torch.zeros(SLOTS, 1)
    return dict(dequant=scal[slot:slot + 1], scale=scal[slot], amax=amax[slot])


SCALAR_SLOTS = (0, 5)


def record_calls(H, tokens, val_tokens):
    """(name, geometry or None, fn) for the flags-off launches, through the API upstream has as well."""
    from track_1_short.perf.kernels import mlp

    E4, E5, BF = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16
    E = torch.empty
    b = E(H, D, dtype=BF)
    bank = E(SLOTS, 2, H, D, dtype=BF)[:, 0]   # gpt.py: self.mlp_bank[:, 0] / [:, 1]
    calls = [(RECORD_PREFIX + "quantize_mlp_weights_dual", None, lambda: mlp.quantize_mlp_weights_dual(
        bank, torch.ones(SLOTS), torch.zeros(SLOTS, mlp.quantize_weights_dual_ntiles(H, D)), E(SLOTS, H, D, dtype=E4),
        E(SLOTS, D, H, dtype=E4), False))]
    for T in tokens + [val_tokens]:
        calls.append((RECORD_PREFIX + "linear_relu_square bf16 forward", f"T={T}",
                      lambda T=T: mlp.linear_relu_square(E(T, D, dtype=BF), b)))
    static = torch.ones(())  # _mlp_x_scale / _mlp_grad_scale: buffers of their own
    for T in tokens:
        a = E(T, D, dtype=BF)
        calls += [
            (RECORD_PREFIX + "quantize_dual_layout_fused e4m3 (x)", f"T={T}",
             lambda a=a: mlp.quantize_dual_layout_fused(a, static, torch.float8_e4m3fn)),
            (RECORD_PREFIX + "quantize_dual_layout_fused e5m2 (g)", f"T={T}",
             lambda a=a: mlp.quantize_dual_layout_fused(a, static, torch.float8_e5m2))]
        for slot in SCALAR_SLOTS:
            s = _scalars(slot)
            fwd = dict(a_f8=E(T, D, dtype=E4), b_f8=E(H, D, dtype=E4), dequant_scale_ptr=s["dequant"], emit_f8=True,
                       emit_t=True, post_scale=s["scale"], post_amax=s["amax"], store_pre=False, store_post_bf=False)
            bwd = dict(aux=E(T, H, dtype=E4), a_f8=E(T, D, dtype=E5), b_f8=E(H, D, dtype=E4), dequant_scale_ptr=s["scale"],
                       post_scale=s["scale"], dpre_scale=s["scale"], dpre_amax=s["amax"])
            geo = f"T={T} slot={slot}"
            calls += [
                (RECORD_PREFIX + "linear_relu_square fp8 forward", geo, lambda a=a, kw=fwd: mlp.linear_relu_square(a, b, **kw)),
                (RECORD_PREFIX + "linear_relu_square fp8 backward", geo, lambda a=a, kw=bwd: mlp.linear_relu_square(a, b, **kw)),
            ]
    return calls


def hadamard_calls(H, tokens):
    """(name, geometry or None, fn) for every HADAMARD_BLOCK x the rotation subsets HADAMARD_GEMMS reaches."""
    from track_1_short.perf.hadamard_ref import signed_hadamard
    from track_1_short.perf.kernels import hadamard as hd
    from track_1_short.perf.kernels import mlp

    E4, E5, BF = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16
    E = torch.empty
    b = E(H, D, dtype=BF)
    bank = E(SLOTS, 2, H, D, dtype=BF)
    nt = hd.quantize_weights_rotated_ntiles(H, D)
    static = torch.ones(())

    def with_block(hb, fn, **gemms):
        """Run fn with HADAMARD_BLOCK = hb in both modules (and HADAMARD_DPOST / _DW2 for quantize_grad)."""
        saved = {(m, "HADAMARD_BLOCK"): m.HADAMARD_BLOCK for m in (hd, mlp)}
        saved.update({(mlp, k): getattr(mlp, k) for k in gemms})
        try:
            hd.HADAMARD_BLOCK = mlp.HADAMARD_BLOCK = hb
            for k, v in gemms.items():
                setattr(mlp, k, v)
            fn()
        finally:
            for (m, k), v in saved.items():
                setattr(m, k, v)

    t_quant = getattr(hd.hadamard_t_quant, "_init_fn", None)
    assert t_quant is not None, "torch custom_op without _init_fn: call the x_t kernel another way"
    calls = []
    for hb in (32, 64, 128):
        q = signed_hadamard(hb, SEED)
        for along_h, what in ((True, "W1 col (dx)"), (False, "W2 row (dpost)")):
            w = bank[:, 0] if along_h else bank[:, 1]
            out = E((SLOTS, D, H) if along_h else (SLOTS, H, D), dtype=E4)
            calls.append((f"quantize_weights_rotated {what} HB={hb}", None,
                          lambda hb=hb, q=q, w=w, out=out, along_h=along_h: with_block(hb, lambda: hd.quantize_weights_rotated(
                              w, q, torch.ones(SLOTS), torch.zeros(SLOTS, nt), out=out, along_h=along_h,
                              update_scales=False))))
        for T in tokens:
            a = E(T, D, dtype=BF)
            for dpost, dw2 in ((True, True), (True, False), (False, True)):
                calls.append((f"quantize_grad dpost={int(dpost)} dw2={int(dw2)} HB={hb}", f"T={T}",
                              lambda hb=hb, q=q, a=a, g=(dpost, dw2): with_block(
                                  hb, lambda: mlp.quantize_grad(a, q, static), HADAMARD_DPOST=g[0], HADAMARD_DW2=g[1])))
            for slot in SCALAR_SLOTS:
                s = _scalars(slot)
                geo = f"T={T} slot={slot}"
                calls.append((f"hadamard_t_quant x_t (dw1) HB={hb}", geo,
                              lambda hb=hb, q=q, a=a, sc=s["scale"]: with_block(hb, lambda: t_quant(a, q, sc))))
                fwd = dict(a_f8=E(T, D, dtype=E4), b_f8=E(H, D, dtype=E4), dequant_scale_ptr=s["dequant"], emit_f8=True,
                           emit_t=True, post_scale=s["scale"], post_amax=s["amax"], store_pre=False, store_post_bf=False)
                bwd = dict(aux=E(T, H, dtype=E4), a_f8=E(T, D, dtype=E5), b_f8=E(H, D, dtype=E4),
                           dequant_scale_ptr=s["scale"], post_scale=s["scale"], dpre_scale=s["scale"], dpre_amax=s["amax"])
                calls.append((f"linear_relu_square fp8 forward ROT_T (dw2) HB={hb}", geo,
                              lambda q=q, a=a, kw=fwd, s=s: mlp.linear_relu_square(
                                  a, b, **kw, hadamard_q=q, rot_t=True, post_t_scale=s["scale"], post_t_amax=s["amax"])))
                for rr, rt in ((True, True), (True, False), (False, True)):
                    calls.append((f"linear_relu_square fp8 backward ROT_ROW={int(rr)} (dx) ROT_T={int(rt)} (dw1) HB={hb}",
                                  geo, lambda q=q, a=a, kw=bwd, rr=rr, rt=rt: mlp.linear_relu_square(
                                      a, b, **kw, hadamard_q=q, rot_row=rr, rot_t=rt)))
    return calls


def run_launchers():
    """{call name: [(geometry, records, host error)]}, in call order."""
    import shim
    shim.cpu_fallbacks(H100_SMS)
    H, tokens, val_tokens = record_geometry()
    print(f"geometry: D={D} H={H} L={SLOTS} tokens/rank {tokens} (+ validation {val_tokens} for the bf16 forward)")
    calls = record_calls(H, tokens, val_tokens)
    if not ARGS.record_only:  # the Hadamard modules exist only in this tree, not in upstream
        calls += hadamard_calls(H, tokens)
    results = {}
    for name, geo, fn in calls:
        n0 = len(RECORDS)
        try:
            fn()
            err = None
        except Exception as e:  # host-side failure (assert, shape) -- also a finding
            err = e
        results.setdefault(name, []).append((geo, RECORDS[n0:], err))
    return results


def _fmt_geoms(gs, all_gs):
    gs = [g for g in gs if g is not None]
    if not gs:
        return ""
    return f"all {len(gs)} geometries" if len(all_gs) > 1 and len(gs) == len(all_gs) else "; ".join(gs)


def report(results):
    failed = False
    for name, runs in results.items():
        all_gs = [g for g, _, _ in runs if g is not None]
        print(f"== {name}")
        for geo, _, err in runs:
            if err is not None:
                failed = True
                print(f"   HOST ERROR{' ' + geo if geo else ''}: {type(err).__name__}: {err}")
        # (kernel, flags, stages) -> {binary hash or error text: [CompiledKernel or exception, [geometries]]}
        groups = {}
        for geo, recs, _ in runs:
            for kname, key, st, default, k in recs:
                flags = " ".join(f"{a}={int(v) if isinstance(v, bool) else v}" for a, v in key.items())
                g = groups.setdefault((kname, flags, st, default), {})
                tag = k.hash if not isinstance(k, Exception) else f"ERR {type(k).__name__}: {(str(k).strip().splitlines() or [''])[-1]}"
                g.setdefault(tag, [k, []])[1].append(geo)
        picks = {}
        for (kname, flags, st, default), binaries in groups.items():
            multi = len(binaries) > 1
            for tag, (k, gs) in binaries.items():
                where = _fmt_geoms(gs, all_gs)
                where = (f" [{where}]" if where else "") + (" (geometry changes the binary)" if multi else "")
                if isinstance(k, Exception):
                    failed = True
                    print(f"   {kname} stages={st}{where}: COMPILE {tag}")
                    if os.environ.get("COMPILE_CHECK_TRACE"):
                        traceback.print_exception(k)
                    continue
                smem = k.metadata.shared
                try:
                    regs, ss, sl = ptxas_stats(k)
                except Exception as e:
                    failed = True
                    print(f"   {kname} stages={st}{where}: PTXAS ERROR {e}")
                    continue
                fits = smem <= H100_SMEM
                print(f"   {kname} stages={st} {flags}{where}: smem {smem} B ({'fits' if fits else 'EXCEEDS 227KB'}), "
                      f"{regs} regs, spill st/ld {ss}/{sl} B")
                if kname == "linear_relu_square_kernel":
                    fit = picks.setdefault((kname, flags, default), {})
                    fit[st] = fit.get(st, True) and fits
        for (kname, flags, default), fit in picks.items():
            ok = [st for st in range(default, 0, -1) if fit.get(st)]
            if ok:
                print(f"   -> the launcher runs {kname} at stages={ok[0]} (default {default})")
            else:
                failed = True
                print(f"   -> {kname}: no num_stages <= {default} fits: the launcher raises OutOfResources")
    return failed


def write_record_asm(results, out_dir):
    """One .sass and one .ptx per (record call, kernel, num_stages, geometry)."""
    os.makedirs(out_dir, exist_ok=True)
    failed = False
    for name, runs in results.items():
        if not name.startswith(RECORD_PREFIX):
            continue
        for geo, recs, _ in runs:
            for kname, key, st, default, k in recs:
                if isinstance(k, Exception):
                    continue
                base = re.sub(r"[^A-Za-z0-9]+", "_", f"{name[len(RECORD_PREFIX):]} {kname} s{st} {geo or ''}").strip("_")
                for ext, fn in ((".sass", sass_text), (".ptx", ptx_text)):
                    try:
                        txt = fn(k)
                    except Exception as e:
                        failed = True
                        print(f"   {base}{ext}: {e}")
                        continue
                    with open(os.path.join(out_dir, base + ext), "w") as f:
                        f.write(txt)
    return failed


def main():
    driver.set_active(_FakeDriver())
    JITFunction.run = _warmup_run
    results = run_launchers()
    failed = report(results)
    if ARGS.sass_dir:
        failed |= write_record_asm(results, ARGS.sass_dir)
    print("RESULT:", "FAILURES" if failed else "all variants compiled")
    if ARGS.base:
        failed |= compare_base()
    return 1 if failed else 0


def compare_base():
    """Record-variant SASS and PTX of this tree vs ARGS.base (each compiled in a subprocess: the trees share
    module names). Any file missing on one side, unreadable, or empty is a difference."""
    d = tempfile.mkdtemp()
    new, old = os.path.join(d, "new"), os.path.join(d, "base")
    me = os.path.abspath(__file__)
    for root, out in ((ARGS.root, new), (ARGS.base, old)):
        r = subprocess.run([sys.executable, me, "--root", root, "--record-only", "--sass-dir", out],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"SASS compare: compiling {root} --record-only failed (rc {r.returncode})\n"
                  f"{r.stdout[-3000:]}{r.stderr[-3000:]}")
            return True
    names = sorted(set(os.listdir(new)) | set(os.listdir(old)))
    if not names:
        print("SASS compare: neither tree produced any record kernel")
        return True
    diff = False
    print("== record-path SASS / PTX vs base")
    counts = {}
    for n in names:
        texts = []
        for side in (new, old):
            p = os.path.join(side, n)
            texts.append(open(p).read() if os.path.exists(p) else None)
        a, b = texts
        same = a is not None and b is not None and a != "" and a == b
        diff |= not same
        ext = n.rsplit(".", 1)[1]
        counts.setdefault(ext, [0, 0])[0 if same else 1] += 1
        unit = "instructions" if ext == "sass" else "lines"
        print(f"   {n}: {'identical' if same else 'DIFFERS'} ({'missing' if a is None else a.count(chr(10)) + 1} vs "
              f"{'missing' if b is None else b.count(chr(10)) + 1} {unit})")
        if not same and a and b and ext == "ptx":
            pa = re.search(r"\.entry\s+\w+\s*\((.*?)\)", a, re.S)
            pb = re.search(r"\.entry\s+\w+\s*\((.*?)\)", b, re.S)
            if pa and pb:
                print(f"      .entry params: {pa.group(1).count('.param')} vs {pb.group(1).count('.param')}")
    print("SASS/PTX: " + ", ".join(f"{e}: {s} identical, {x} differ" for e, (s, x) in sorted(counts.items())))
    print("SASS:", "record path DIFFERS from base" if diff else "record path identical to base")
    return diff


if __name__ == "__main__":
    sys.exit(main())
