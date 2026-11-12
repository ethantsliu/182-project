"""FP8_ATTN_O on one H100 in ~2 min: does the fp8 attention output projection compile, replay in a CUDA graph and
agree with eager, and what does it cost per call?  python 182/fp8-attn-o/gpu_sanity.py [--json out.json]

  (i)  compiled + CUDA graph, FP8_ATTN_O off and on: a stand-in for the projection's surroundings (y = y_pre * gate,
       the elementwise producer the gated layers have; then the output projection; then a loss), torch.compile'd
       (inductor, fullgraph) and its forward + backward captured in one CUDA graph, then replayed. Off: the
       record's F.linear(y, o_scale * o_w). On: AttnOFP8Function -- static (row quantize fused into the gate
       multiply, the gated / XSA layers), static_plain (the Triton dual quantize the plain layers run) and
       delayed (the Triton dual quantize with amaxes folded into a buffer). Each replay is
       compared with eager on out / dy / do_w / do_scale (and, on, the amax slots).
  (ii) per-call ms at T = 16k/32k/40k/48k for H*v = 768 and 384: the record's bf16 path (o_scale * o_w product,
       F.linear and its two backward GEMMs) vs the fp8 path (y / g dual-layout quantize, three _scaled_mm), both
       eager and per kernel (torch.profiler), plus the o_w refresh (quantize_attn_o_weights + the half-V copies).
       Projected over the 1194-step schedule at 7 calls per step: 5 at 768 (layers 0, 2, 5 plain; 3, 10 gated)
       and 2 at 384 (layer 8 plain, layer 1 XSA). The gated / XSA layers' y quantize is timed standalone here
       (aten row + fp8 transpose); in the trainer inductor fuses the row half into the gate / XSA kernel, so
       their number is an upper bound.
Exit status 1 if a check fails or a part raises. Nothing here measures loss: that is the attno A/B phase of
182/gpu_session.sh.
"""
import argparse
import json
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (per-rank tokens, steps) of the record schedule's stages: 1194 steps (as 182/hadamard-fp8/gpu_sanity.py).
STAGES = ((16384, 320), (32768, 361), (49152, 426), (40960, 67), (16384, 20))
# (H*v, y quantize path) of the 7 attention layers: plain layers use the Triton dual quantize, gated / XSA
# layers the producer-fused aten path (static arm).
LAYERS = ((768, "triton"), (768, "triton"), (768, "triton"), (384, "triton"),   # 0, 2, 5, 8
          (768, "fused"), (768, "fused"), (384, "fused"))                       # 3, 10, 1
SEC_PER_MNAT = 0.164
MODES = ("bf16", "static", "static_plain", "delayed")  # part (i)


def _operands(T, K, dev, seed=0):
    import torch
    g = torch.Generator(device=dev).manual_seed(seed)
    y_pre = (0.2 * torch.randn(T, K, device=dev, generator=g)).bfloat16()
    gate = (0.25 + 0.05 * torch.randn(T, 1, device=dev, generator=g)).bfloat16()
    bound = 2.0 ** -5
    o_w = ((torch.rand(768, K, device=dev, generator=g) * 2 - 1) * bound).bfloat16()
    grad = (1e-3 * torch.randn(T, 768, device=dev, generator=g)).bfloat16()
    return y_pre, gate, o_w, grad


class OutProj:
    """y = y_pre * gate, then the output projection, record (bf16) or fp8; .run(y_pre) returns out."""
    def __init__(self, o_w, mode, dev):
        import torch

        from track_1_short.perf import attn_o_scales as S
        self.mode = mode
        self.o_w = o_w.detach().clone().requires_grad_()
        self.lam = torch.tensor(0.8, device=dev, requires_grad=True)
        self.gain = torch.tensor(1.1, device=dev, dtype=torch.bfloat16)
        if mode != "bf16":
            wq = S.quant(self.o_w.detach(), S.ATTN_O_W_SCALE, S.E4M3)
            t = lambda v: torch.tensor(float(v), device=dev)
            self.w = (wq, wq.T.contiguous(), t(S.ATTN_O_W_SCALE), t(S.ATTN_O_Y_SCALE), t(S.ATTN_O_G_SCALE))
            self.amax = (torch.zeros(1, device=dev), torch.zeros(1, device=dev)) if mode == "delayed" else (None, None)

    def run(self, y_pre, gate):
        import torch.nn.functional as F

        from track_1_short.perf.kernels.attn_o import AttnOFP8
        y = y_pre * gate
        o_scale = self.lam * self.gain
        if self.mode == "bf16":
            return F.linear(y, o_scale * self.o_w.type_as(y))
        # static: row quantize fused into y_pre * gate (gated / XSA layers); static_plain: the Triton dual quantize
        # (plain layers, whose y comes straight from FA3); delayed: the Triton quantize returning amaxes
        return AttnOFP8(y, self.o_w, o_scale, *self.w, *self.amax, self.mode != "static_plain", True)


def check_compiled_graph(mode, T=8192, K=768):
    """{check: [ok, detail]} for one mode: eager vs torch.compile(fullgraph) + one CUDA graph of forward+backward."""
    import torch
    dev = torch.device("cuda")
    y_pre0, gate, o_w, grad = _operands(T, K, dev)
    res = {}

    def eager():
        m = OutProj(o_w, mode, dev)
        y_pre = y_pre0.detach().clone().requires_grad_()
        out = m.run(y_pre, gate)
        out.backward(grad)
        return m, dict(out=out.detach(), dy=y_pre.grad, dw=m.o_w.grad, dlam=m.lam.grad)

    me, want = eager()
    m = OutProj(o_w, mode, dev)
    fn = torch.compile(m.run, fullgraph=True, dynamic=False)
    y_pre = y_pre0.detach().clone().requires_grad_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # compile and warm up off the capture stream
            fn(y_pre, gate).backward(grad)
    torch.cuda.current_stream().wait_stream(s)
    for p in (y_pre, m.o_w, m.lam):
        p.grad = None
    if mode == "delayed":
        for a in m.amax:
            a.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(y_pre, gate)
        out.backward(grad)
    # capture records but does not run: zero what the replay accumulates into, then replay once
    for p in (y_pre, m.o_w, m.lam):
        p.grad.zero_()
    if mode == "delayed":
        for a in m.amax:
            a.zero_()
    graph.replay()
    torch.cuda.synchronize()
    got = dict(out=out.detach(), dy=y_pre.grad, dw=m.o_w.grad, dlam=m.lam.grad)
    # static: inductor fuses the row quantize into y_pre * gate and rounds its fp32 product, eager rounds the
    # stored bf16 y: at most one bf16 rounding apart before the cast, so not bitwise
    tol = 2e-2 if mode == "static" else 0.0
    for k in want:
        a, b = got[k].float(), want[k].float()
        err = ((a - b).norm() / b.norm().clamp(min=1e-30)).item()
        res[k] = [err <= tol + 1e-6 or (tol == 0.0 and torch.equal(a, b)), f"rel {err:.2e} (tol {tol})"]
    if mode == "delayed":
        for name, a, ref in (("y_amax", m.amax[0], (y_pre0 * gate).float().abs().max()), ("g_amax", m.amax[1], grad.float().abs().max())):
            res[name] = [abs(a.item() - ref.item()) <= 1e-3 * ref.item(), f"{a.item():.4g} vs {ref.item():.4g}"]
    return res


def _profile(fn, iters):
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
        t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        if t and t > 0:
            per[e.key[:60]] = per.get(e.key[:60], 0.0) + t / 1000.0 / iters
    return per, start.elapsed_time(end) / iters


def bench(tokens, iters):
    """ms per call (fwd + bwd, CUDA events) of bf16 vs fp8 per (T, K, quantize path), and the o_w refresh."""
    import torch

    from track_1_short.perf.kernels.attn_o import quantize_attn_o_weights
    from track_1_short.perf.kernels.fp8_attention_quant import AMAX_PARTS
    dev = torch.device("cuda")
    out = {"T": {}}
    vo = torch.randn(16, 768, 768, device=dev).bfloat16()
    L = 7
    scales, pamax = torch.full((L,), 2.0 ** -6, device=dev), torch.zeros(L, AMAX_PARTS, device=dev)
    row, col = torch.empty(L, 768, 768, device=dev, dtype=torch.float8_e4m3fn), torch.empty(L, 768, 768, device=dev, dtype=torch.float8_e4m3fn)
    half_r, half_c = torch.empty(2, 768, 384, device=dev, dtype=torch.float8_e4m3fn), torch.empty(2, 384, 768, device=dev, dtype=torch.float8_e4m3fn)

    def refresh():
        quantize_attn_o_weights(vo[1:2 * L:2], scales, pamax, row, col, exact=False, record_amax=False)
        half_r.copy_(row[3:5].view(2, 768, 6, 128)[..., :64].reshape(2, 768, 384))
        half_c.copy_(col[3:5].view(2, 6, 128, 768)[:, :, :64].reshape(2, 384, 768))
    refresh()
    per, ms = _profile(lambda: [refresh() for _ in range(iters)], iters)
    out["refresh"] = {"kernels": per, "total": ms}
    for T in tokens:
        out["T"][str(T)] = {}
        for K in (768, 384):
            y_pre, gate, o_w, grad = _operands(T, K, dev)
            row_ms = {}
            for mode, fused in (("bf16", None), ("fp8", False), ("fp8", True)):
                m = OutProj(o_w, "bf16" if mode == "bf16" else "static", dev)
                if mode == "fp8" and not fused:  # the Triton path: what the plain layers run
                    from track_1_short.perf.kernels.attn_o import AttnOFP8
                    m.run = lambda yp, gt, m=m: AttnOFP8(yp * gt, m.o_w, m.lam * m.gain, *m.w, None, None, False, True)

                def step(m=m):
                    yp = y_pre.detach().requires_grad_()
                    m.run(yp, gate).backward(grad)
                for _ in range(3):
                    step()
                per, ms = _profile(lambda: [step() for _ in range(iters)], iters)
                key = "bf16" if mode == "bf16" else ("fp8_fused" if fused else "fp8_triton")
                row_ms[key] = {"total": ms, "kernels": per}
            out["T"][str(T)][str(K)] = row_ms
            torch.cuda.empty_cache()
    return out


def projected(b):
    """Seconds over the schedule: (fp8 - bf16) per step summed over LAYERS, plus one o_w refresh per step."""
    total = 0.0
    for T, steps in STAGES:
        d = b["T"].get(str(T))
        if d is None:
            return None
        per_step = sum(d[str(K)]["fp8_" + path]["total"] - d[str(K)]["bf16"]["total"] for K, path in LAYERS)
        total += steps * per_step / 1000.0
    return total + sum(s for _, s in STAGES) * b["refresh"]["total"] / 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tokens", default="16384,32768,40960,49152")
    ap.add_argument("--skip-compile", action="store_true", help="skip part (i)")
    ap.add_argument("--json", help="also write the raw results here")
    args = ap.parse_args()
    import torch
    assert torch.cuda.is_available(), "gpu_sanity.py needs a GPU"
    res, failed = {"device": torch.cuda.get_device_name()}, []
    if not args.skip_compile:
        res["compiled"] = {}
        for mode in MODES:
            try:
                res["compiled"][mode] = check_compiled_graph(mode)
            except Exception as e:  # noqa: BLE001 - one failing part must not hide the others
                res["compiled"][mode] = {"raised": [False, f"{type(e).__name__}: {e}"]}
                traceback.print_exc()
            for k, (ok, detail) in res["compiled"][mode].items():
                print(f"(i) compiled+graph FP8_ATTN_O={'off' if mode == 'bf16' else 'on (' + mode + ')'} {k}: "
                      f"{'ok' if ok else 'FAIL'} {detail}")
                if not ok:
                    failed.append(f"{mode} {k}")
    try:
        res["bench"] = b = bench([int(t) for t in args.tokens.split(",")], args.iters)
        print(f"(ii) o_w refresh (7 slots + half-V copies): {b['refresh']['total']:.4f} ms/step")
        print(f"(ii) {'T':>6} {'H*v':>4} {'bf16 ms':>9} {'fp8 triton':>11} {'fp8 fused*':>11}   (fwd+bwd per call)")
        for T, d in b["T"].items():
            for K, r in d.items():
                print(f"     {T:>6} {K:>4} {r['bf16']['total']:9.4f} {r['fp8_triton']['total']:11.4f} {r['fp8_fused']['total']:11.4f}")
        p = projected(b)
        if p is not None:
            res["projected_s"] = p
            print(f"(ii) projected Δtime over the 1194-step schedule: {p:+.3f} s (fused* timed standalone: an upper "
                  f"bound for 3 of the 7 layers); break-even Δval {-p / SEC_PER_MNAT:+.2f} mnat at {SEC_PER_MNAT} s/mnat")
    except Exception as e:  # noqa: BLE001
        res["bench"] = {"error": f"{type(e).__name__}: {e}"}
        failed.append("bench raised")
        traceback.print_exc()
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1))
    print("RESULT:", "FAIL " + "; ".join(failed) if failed else "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
