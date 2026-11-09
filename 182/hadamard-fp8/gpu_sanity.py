"""HADAMARD_FP8 GPU self-diagnostics on one GPU, under 3 minutes:

    python 182/hadamard-fp8/gpu_sanity.py                       # all three parts, one compact table
    HADAMARD_GEMMS=dw1,dx HADAMARD_BLOCK=64 python 182/hadamard-fp8/gpu_sanity.py   # knobs pass through
    python 182/hadamard-fp8/gpu_sanity.py --iters 5 --tokens 16384,49152 --skip-compile

Runs one worker subprocess with HADAMARD_FP8=0 and one with HADAMARD_FP8=1. The flags are read at import
time, so each setting needs its own process. Each worker reports:
  (i)   prime_stage_cache's num_stages for every linear_relu_square variant, and n_regs / n_spills / shared
        memory of every compiled variant of the MLP and Hadamard Triton kernels
  (ii)  HADAMARD_FP8=1 only: the compiled amax check, also run by tests/test_gpu_state.py. GPT's real fp8 MLP
        state methods run inside torch.compile(fullgraph=True) and a captured CUDA graph, as in training.
        Only the target slot's x_t amax (and, when mlp.py implements the post_t contract, its post / post_t /
        dpre amaxes) may become > 0, each must match a torch reference, and a replay of the captured fp8
        refresh must move that slot's x_t and post_t scales and nothing else
  (iii) per-kernel device time of one MLP call (forward + backward at slot 0, with real dims) at each token
        count, from torch.profiler, plus the per-step weight refresh, and the projected seconds over the
        1194-step schedule (STAGES below, 11 MLP calls per step)
The parent prints one table of on - off deltas. Nothing here is a training run. Every number is a
microbenchmark of one process on one GPU (eager kernels, warm L2), so treat the projection as a rough guide.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (per-rank tokens, steps) of the record schedule's stages: 1194 steps.
STAGES = ((16384, 320), (32768, 361), (49152, 426), (40960, 67), (16384, 20))
MLP_CALLS_PER_STEP = 11
PASS_ENV = ("HADAMARD_GEMMS", "HADAMARD_BLOCK", "HADAMARD_SEED")
MARK = "GPU_SANITY_JSON "


# -----------------------------------------------------------------------------------------------------
# A model-free harness around GPT's real fp8 MLP state methods (shared with the CPU and GPU tests).

def make_mini_state(h, d, device, slot=0, full=True):
    """An nn.Module that borrows GPT's fp8 MLP state methods (buffers, refresh, input quantize, Function
    args, rearm, saturation snapshot) unchanged, with a [NUM_MLP_SLOTS, 2, h, d] bank. forward(x, r) runs
    slot `slot`'s input quantize and, with `full`, its fp8 MLP, and returns a scalar loss."""
    import torch
    from torch import nn

    from track_1_short.model import gpt
    from track_1_short.perf.kernels.mlp import FusedLinearReLUSquareFunction

    class MiniMLPState(nn.Module):
        init_mlp_fp8 = gpt.GPT.init_mlp_fp8
        # The refresh body, under no_grad as GPT.quantize_mlp_fp8 (its only caller) runs it.
        _quantize_mlp_weights_and_scales = torch.no_grad()(gpt.GPT._quantize_mlp_weights_and_scales)
        _quantize_mlp_input = gpt.GPT._quantize_mlp_input
        _mlp_fp8_args = gpt.GPT._mlp_fp8_args
        rearm_fp8_bootstrap = gpt.GPT.rearm_fp8_bootstrap
        fp8_sat_snapshot = gpt.GPT.fp8_sat_snapshot
        _refresh_delayed_scale = staticmethod(gpt.GPT._refresh_delayed_scale)

        def __init__(self):
            super().__init__()
            self.mlp_hdim, self.device, self.slot, self.full = h, device, slot, full
            g = torch.Generator().manual_seed(1)
            self.mlp_bank = nn.Parameter((0.03 * torch.randn(gpt.NUM_MLP_SLOTS, 2, h, d, generator=g)).bfloat16().to(device))
            self.post_lambdas = nn.Parameter(torch.ones(gpt.NUM_LAYERS, 2, device=device))
            self.init_mlp_fp8(d)

        def forward(self, x, r):
            s = self.slot
            x_f8 = self._quantize_mlp_input(x.detach(), s)
            if not self.full:
                return x_f8[1].float().mean() + x_f8[0].float().mean()
            args = self._mlp_fp8_args(s, self._mlp_up_f8[s], x_f8, None, xt_slot=s)
            out = FusedLinearReLUSquareFunction.apply(x, self.mlp_bank[s, 0], self.mlp_bank[s, 1], *args)
            return (out.float() * r).sum()

    return MiniMLPState()


def mlp_takes_post_t():
    """Whether perf/kernels/mlp.py implements the post_t contract of model/gpt.py (FP8_XT_HEADROOM comment)."""
    import inspect

    from track_1_short.perf.kernels.mlp import FusedLinearReLUSquareFunction
    params = inspect.signature(FusedLinearReLUSquareFunction.forward).parameters
    return "post_t_scale" in params and "post_t_amax" in params


def rms_rows(t):
    return (t / t.float().pow(2).mean(1, keepdim=True).sqrt()).bfloat16()


# -----------------------------------------------------------------------------------------------------
# (ii) the compiled amax check

def check_compiled_amax(T=4096, slot=5):
    """Returns {check: [ok, detail]}. HADAMARD_FP8=1, CUDA. Mirrors training: torch.compile(fullgraph,
    dynamic=False) of a module whose forward calls GPT's _quantize_mlp_input (+ the fp8 MLP), captured
    with its backward in one CUDA graph; the delayed-scale refresh captured in another (as
    perf/cuda_graphs/fp8_refresh_graphs.py does)."""
    from track_1_short.model import gpt

    assert gpt.HADAMARD_FP8, "run with HADAMARD_FP8=1"
    res = {}
    full = mlp_takes_post_t()
    res["mlp.py implements the post_t contract"] = [full, "" if full else
                                                    "x_t checked alone; post / post_t / dpre checks skipped"]
    real_check = gpt.check_mlp_takes_post_t
    if not full:
        gpt.check_mlp_takes_post_t = lambda: None  # let the harness build the buffers; restored below
    try:
        return _check_compiled_amax(res, full, T, slot)
    finally:
        gpt.check_mlp_takes_post_t = real_check


def _check_compiled_amax(res, full, T, slot):
    import torch

    from track_1_short.model import gpt
    from track_1_short.perf.hadamard_ref import rotate
    from track_1_short.perf.kernels.mlp import prime_stage_cache

    dev = torch.device("cuda")
    prime_stage_cache()
    torch.manual_seed(0)
    h, d = gpt.MLP_HIDDEN_DIM, 768
    m = make_mini_state(h, d, dev, slot=slot, full=full)
    for _ in range(gpt.FP8_EXACT_SCALE_CALLS):  # the eager exact-scale bootstrap
        m._quantize_mlp_weights_and_scales()
    x = rms_rows(torch.randn(T, d, device=dev)).requires_grad_()
    r = 1e-2 * torch.randn(T, d, device=dev)
    params = [x, m.mlp_bank] if full else []
    cm = torch.compile(m, fullgraph=True, dynamic=False)

    def step():
        loss = cm(x, r)
        return torch.autograd.grad(loss, params) if full else loss

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
            m._quantize_mlp_weights_and_scales()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    g_step, g_refresh = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_step, pool=pool, stream=side):
        step()
    with torch.cuda.graph(g_refresh, stream=side):
        m._quantize_mlp_weights_and_scales()
    torch.cuda.synchronize()

    # A known state: activation amaxes clear, delayed activation scales at their init values.
    act = ["_mlp_xt_amax", "_mlp_post_amax", "_mlp_dpre_amax", "_mlp_post_t_amax"]
    with torch.no_grad():
        for n in act:
            getattr(m, n).zero_()
        m._mlp_xt_scales.fill_(gpt.FP8_MLP_X_SCALE)
        m._mlp_post_t_scales.fill_(1.0)
        m._mlp_post_scales.fill_(1.0)
        m._mlp_dpre_scales.fill_(gpt.FP8_GRAD_SCALE)
        x.copy_(rms_rows(torch.randn(T, d, device=dev)))
    g_step.replay()
    torch.cuda.synchronize()

    def only_slot(name):
        a = getattr(m, name)[:, 0]
        others = torch.cat((a[:slot], a[slot + 1:]))
        ok = a[slot].item() > 0 and bool((others == 0).all())
        return ok, f"slot {slot}: {a[slot].item():.4g}, max of other slots: {others.abs().max().item():.3g}"

    def close(name, got, want, rtol):
        rel = abs(got - want) / max(abs(want), 1e-30)
        res[name] = [rel <= rtol, f"got {got:.5g}, want {want:.5g} (rel err {rel:.2e}, tolerance {rtol})"]

    q = m._hadamard_q
    xf = x.detach().float()
    res["x_t amax only in the target slot"] = list(only_slot("_mlp_xt_amax"))
    close("x_t amax = max|rotate(x, tokens)|", m._mlp_xt_amax[slot, 0].item(), rotate(xf, 0, q.float()).abs().max().item(), 1e-2)
    if full:
        for n in ("_mlp_post_amax", "_mlp_post_t_amax", "_mlp_dpre_amax"):
            res[f"{n[5:]} only in the target slot"] = list(only_slot(n))
        post = torch.relu(xf @ m.mlp_bank[slot, 0].float().T).square()
        post_ref, post_t_ref = post.abs().max().item(), rotate(post, 0, q.float()).abs().max().item()
        # The fp8 forward GEMM rounds pre by a few %, so post (a square) by up to ~2x that.
        close("post amax = max|post| (unrotated: record forward numerics)", m._mlp_post_amax[slot, 0].item(), post_ref, 0.15)
        close("post_t amax = max|rotate(post, tokens)|", m._mlp_post_t_amax[slot, 0].item(), post_t_ref, 0.15)
        res["post vs post_t references differ enough to tell a shared amax apart"] = [
            True, f"post_t/post reference ratio {post_t_ref / post_ref:.2f} (a check only discriminates if this is far from 1)"]

    before = {n: getattr(m, n).clone() for n in ("_mlp_xt_scales", "_mlp_post_t_scales", "_mlp_xt_amax", "_mlp_post_t_amax")}
    g_refresh.replay()
    torch.cuda.synchronize()
    e4 = 448.0
    for scale_n, amax_n, headroom, init in (("_mlp_xt_scales", "_mlp_xt_amax", gpt.FP8_XT_HEADROOM, gpt.FP8_MLP_X_SCALE),
                                            ("_mlp_post_t_scales", "_mlp_post_t_amax", gpt.FP8_POST_HEADROOM, 1.0)):
        if amax_n == "_mlp_post_t_amax" and not full:
            continue
        s_new = getattr(m, scale_n)
        want = max(before[amax_n][slot, 0].item() * headroom / e4, 1e-6)
        others = torch.cat((s_new[:slot], s_new[slot + 1:]))
        ok = (abs(s_new[slot].item() - want) <= 1e-5 * want and s_new[slot].item() != init
              and bool((others == init).all()) and bool((getattr(m, amax_n) == 0).all()))
        res[f"captured refresh moves {scale_n[5:]}[{slot}] only"] = [
            ok, f"{init:.4g} -> {s_new[slot].item():.5g} (want {want:.5g}); other slots at init: {bool((others == init).all())}"]
    return res


# -----------------------------------------------------------------------------------------------------
# (i) stage cache and kernel resources

def describe_stage_key(key):
    try:
        fwd, fp8, emit_f8, emit_t, _, _, had = key[:7]
        return f"{'fwd' if fwd else 'bwd'}-{'fp8' if fp8 else 'bf16'}{'-emit' if emit_f8 or emit_t else ''}{'-HAD' if had else ''}"
    except (TypeError, ValueError):
        return str(key)


def kernel_resources(kernels):
    """[name, variant, stages, warps, shared, regs, spills] per compiled specialization on the device."""
    import torch
    rows = []
    dev = torch.cuda.current_device()
    for kfn in kernels:
        caches = getattr(kfn, "device_caches", {})
        if dev not in caches:
            continue
        for ck in caches[dev][0].values():
            try:
                ck._init_handles()
            except Exception as e:  # noqa: BLE001 - report, don't die
                rows.append([kfn.__name__, f"init failed: {e}", None, None, None, None, None]); continue
            variant = ""
            try:
                names = ck.src.fn.arg_names
                flags = {names[k[0]]: v for k, v in ck.src.constants.items() if isinstance(v, bool) and v}
                variant = ",".join(sorted(n for n in flags if n.isupper()))
            except Exception:  # noqa: BLE001
                pass
            md = ck.metadata
            rows.append([kfn.__name__, variant, md.num_stages, md.num_warps, md.shared, ck.n_regs, ck.n_spills])
    return rows


# -----------------------------------------------------------------------------------------------------
# (iii) microbench

def short_kernel(name):
    n = name.lower()
    for key, short in (("linear_relu_square", "linear_relu_square"), ("hadamard_quant_dual", "hadamard_quant"),
                       ("quantize_weights_rotated", "w_quant_rotated"), ("quantize_weights_dual", "w_quant_dual"),
                       ("transpose", "fp8_transpose"), ("gemm", "scaled_mm"), ("cutlass", "scaled_mm"),
                       ("xmma", "scaled_mm"), ("memset", "memset"), ("memcpy", "memcpy")):
        if key in n:
            return short
    return "other_pointwise"


def _profile(fn, iters):
    """{short kernel name: ms per iteration} of everything fn launches, and the CUDA-event ms per iteration."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
    per = {}
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None)
        if t is None:
            t = getattr(e, "cuda_time_total", 0)
        if t and t > 0:
            k = short_kernel(e.key)
            per[k] = per.get(k, 0.0) + t / 1000.0 / iters
    return per, start.elapsed_time(end) / iters


def bench(tokens, iters):
    import torch

    from track_1_short.model import gpt
    from track_1_short.perf.kernels.mlp import FusedLinearReLUSquareFunction

    dev = torch.device("cuda")
    if gpt.HADAMARD_FP8 and not mlp_takes_post_t():
        return {"error": "mlp.py lacks the post_t contract (model/gpt.py check_mlp_takes_post_t); no HADAMARD bench"}
    h, d = gpt.MLP_HIDDEN_DIM, 768
    m = make_mini_state(h, d, dev, slot=0)
    for _ in range(gpt.FP8_EXACT_SCALE_CALLS):
        m._quantize_mlp_weights_and_scales()
    out = {"refresh": None, "T": {}}
    per, ms = _profile(lambda: [m._quantize_mlp_weights_and_scales() for _ in range(iters)], iters)
    out["refresh"] = {"kernels": per, "total": ms}
    W1, W2 = (m.mlp_bank[0, i].detach().requires_grad_() for i in (0, 1))
    for T in tokens:
        torch.manual_seed(T)
        x = rms_rows(torch.randn(T, d, device=dev))
        g = (1e-2 * torch.randn(T, d, device=dev)).bfloat16()

        def fwd_one():
            xin = x.detach().requires_grad_()
            args = m._mlp_fp8_args(0, m._mlp_up_f8[0], m._quantize_mlp_input(x, 0), None, xt_slot=0)
            return xin, FusedLinearReLUSquareFunction.apply(xin, W1, W2, *args)

        for _ in range(2):  # warm up, then set the delayed activation scales from real amaxes
            xin, y = fwd_one()
            torch.autograd.grad(y, (xin, W1, W2), g)
            m._quantize_mlp_weights_and_scales()
        saved = []
        per_f, ms_f = _profile(lambda: saved.extend(fwd_one() for _ in range(iters)), iters)
        per_b, ms_b = _profile(lambda: [torch.autograd.grad(y, (xin, W1, W2), g) for xin, y in saved], iters)
        del saved
        out["T"][str(T)] = {"fwd": per_f, "bwd": per_b, "fwd_total": ms_f, "bwd_total": ms_b}
        torch.cuda.empty_cache()
    return out


def worker(args):
    import torch

    from track_1_short.model import gpt
    from track_1_short.perf.kernels import hadamard as hk
    from track_1_short.perf.kernels import mlp as mk

    res = {"hadamard": gpt.HADAMARD_FP8, "device": torch.cuda.get_device_name(),
           "env": {k: os.environ[k] for k in PASS_ENV if k in os.environ}}
    mk.prime_stage_cache()
    res["stages"] = {describe_stage_key(k): v for k, v in mk._lrs_stage_cache.items()}
    if gpt.HADAMARD_FP8 and not args.skip_compile:
        try:
            res["amax_check"] = check_compiled_amax()
        except Exception as e:  # noqa: BLE001 - one failing part must not hide the others
            res["amax_check"] = {"raised": [False, f"{type(e).__name__}: {e}"]}
    try:
        res["bench"] = bench([int(t) for t in args.tokens.split(",")], args.iters)
    except Exception as e:  # noqa: BLE001
        res["bench"] = {"error": f"{type(e).__name__}: {e}"}
    kernels = [mk.linear_relu_square_kernel, mk._quantize_weights_dual_kernel]
    kernels += [getattr(hk, n) for n in dir(hk) if n.startswith("_") and n.endswith("_kernel")]
    res["kernels"] = kernel_resources(kernels)
    print(MARK + json.dumps(res), flush=True)


# -----------------------------------------------------------------------------------------------------
# The parent: two workers, one table.

def run_worker(hadamard, args):
    env = dict(os.environ, HADAMARD_FP8=str(hadamard))
    cmd = [sys.executable, __file__, "--worker", "--iters", str(args.iters), "--tokens", args.tokens]
    if args.skip_compile:
        cmd.append("--skip-compile")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    for line in p.stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
    return {"error": f"worker HADAMARD_FP8={hadamard} exited {p.returncode} without a result (log above)"}


def projected_seconds(delta_ms_by_T, refresh_delta_ms):
    total = 0.0
    for T, steps in STAGES:
        if str(T) not in delta_ms_by_T:
            return None
        total += steps * MLP_CALLS_PER_STEP * delta_ms_by_T[str(T)] / 1000.0
    return total + sum(s for _, s in STAGES) * refresh_delta_ms / 1000.0


def report(off, on):
    print(f"HADAMARD_FP8 gpu_sanity on {on.get('device') or off.get('device')}; env passed through: "
          f"{on.get('env') or {}}")
    for r in (off, on):
        if "error" in r:
            print("ERROR", r["error"])
    print("\n(i) linear_relu_square num_stages after prime_stage_cache")
    for name in sorted(set(off.get("stages", {})) | set(on.get("stages", {}))):
        print(f"  {name:<22} off={off.get('stages', {}).get(name, '-')!s:<3} on={on.get('stages', {}).get(name, '-')}")
    print("    kernel resources (setting kernel [variant flags]: stages warps shared regs spills)")
    for tag, r in (("off", off), ("on", on)):
        for k in r.get("kernels", []):
            spill = "  <-- SPILLS" if k[6] else ""
            print(f"  {tag:<3} {k[0]:<34} [{k[1]}]: {k[2]} {k[3]} {k[4]} {k[5]} {k[6]}{spill}")
    print("\n(ii) compiled x_t / post_t amax check (HADAMARD_FP8=1)")
    checks = on.get("amax_check")
    if not checks:
        print("  skipped")
    else:
        for name, (ok, detail) in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print("\n(iii) device ms per MLP call, on - off  (columns: off / on / delta per token count)")
    bo, bn = off.get("bench", {}), on.get("bench", {})
    for r in (bo, bn):
        if "error" in r:
            print("  bench error:", r["error"])
    if "T" not in bo or "T" not in bn:
        return
    Ts = [t for t in bo["T"] if t in bn["T"]]
    print(f"  {'':<26}" + "".join(f" | {'T=' + t:^23}" for t in Ts))
    print(f"  {'phase':<5} {'kernel':<20}" + "".join(f" | {'off':>7} {'on':>7} {'delta':>7}" for _ in Ts))
    deltas = {t: 0.0 for t in Ts}
    for phase in ("fwd", "bwd"):
        names = sorted(set().union(*(set(bo["T"][t][phase]) | set(bn["T"][t][phase]) for t in Ts)))
        for k in names:
            cells = ""
            for t in Ts:
                a, b = bo["T"][t][phase].get(k, 0.0), bn["T"][t][phase].get(k, 0.0)
                cells += f" | {a:7.3f} {b:7.3f} {b - a:+7.3f}"
            print(f"  {phase:<5} {k:<20}{cells}")
    cells = ""
    for t in Ts:
        a = bo["T"][t]["fwd_total"] + bo["T"][t]["bwd_total"]
        b = bn["T"][t]["fwd_total"] + bn["T"][t]["bwd_total"]
        deltas[t] = b - a
        cells += f" | {a:7.3f} {b:7.3f} {b - a:+7.3f}"
    print(f"  {'all':<5} {'events fwd+bwd':<20}{cells}")
    ro, rn = bo["refresh"]["total"], bn["refresh"]["total"]
    print(f"  weight refresh per step (12 slots): off {ro:.3f} on {rn:.3f} delta {rn - ro:+.3f} ms")
    proj = projected_seconds(deltas, rn - ro)
    steps = sum(s for _, s in STAGES)
    if proj is None:
        print(f"  projection needs every stage's T {sorted({t for t, _ in STAGES})}; pass them with --tokens")
    else:
        print(f"  projected over {steps} steps x {MLP_CALLS_PER_STEP} MLP calls (+ refresh/step): {proj:+.2f} s "
              f"(friend measured +3.17 s on 8xH100 for the round-1 kernels)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tokens", default="16384,32768,40960,49152")
    ap.add_argument("--skip-compile", action="store_true", help="skip part (ii)")
    ap.add_argument("--json", help="also write both workers' raw results here")
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    off, on = run_worker(0, args), run_worker(1, args)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"off": off, "on": on}, indent=1))
    report(off, on)
    checks = on.get("amax_check") or {}
    sys.exit(1 if any(not ok for ok, _ in checks.values()) else 0)


if __name__ == "__main__":
    main()
