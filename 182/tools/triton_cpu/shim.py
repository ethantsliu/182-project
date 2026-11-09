"""Numerics fixes for the Triton interpreter (TRITON_INTERPRET=1), loaded only by this harness.

The interpreter keeps bf16 as raw uint16 bits and fp8 as raw uint8 bits in numpy arrays, and in triton
3.5.1 / 3.6.0 / 3.9.0 (all identical here) several ops treat those bits as numbers:
  - tl.dot: np.matmul on the uint16 bits of bf16 operands -> garbage (fp8 is upcast and fine)
  - bf16 elementwise ops (add, mul, max, min, compare, where, fma, sqrt, ...) on the bits: tl.maximum(x, 0)
    compares as unsigned, so negative values win; x * x is integer multiplication
  - float constants in bf16 expressions: InterpreterBuilder has no get_bf16
  - int -> bf16 and fp64 -> bf16 casts: astype of the value into the uint16 bit field
  - fp32 -> bf16 cast truncates (GPU: round to nearest even)
  - fp32 -> fp8 cast: rounds ties away from zero, loses the mantissa carry into the exponent (e.g. a value
    just under 2^-6 lands on 2^-7 in e4m3), rounds subnormals before shifting, and is not satfinite
  - numpy >= 2.3 rejects int() on the 1-element arrays the interpreter uses as scalars (tl.range bounds)
`install()` replaces those paths: casts go through torch (round to nearest even, fp8 saturating to +-max
like the GPU's `cvt.rn.satfinite`), dots accumulate in fp64 and round once to the accumulator dtype, and
bf16 elementwise ops compute in fp32 and round back to bf16. fp8 elementwise arithmetic raises instead of
returning garbage (no kernel here does it). 182/tools/triton_cpu/test_interp_probes.py checks every path.

What the shim does NOT model: fp32 x fp32 tl.dot is exact here (GPU default input_precision is tf32); the
GPU's fp32 accumulation order inside a dot; fp8 tensor-core accumulator truncation (`use_fast_accum`);
races between programs (the interpreter runs programs one after another).
"""
import os
import types

import numpy as np
import torch

TESTED_TRITON = ("3.5.1", "3.6.0", "3.9.0")
_installed = False


def install():
    global _installed
    if _installed:
        return
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir as _ir
    from triton.runtime import interpreter as I

    if triton.__version__ not in TESTED_TRITON:
        print(f"[triton_cpu shim] triton {triton.__version__} untested (tested: {TESTED_TRITON}); "
              f"run the probes: python -m pytest 182/tools/triton_cpu", flush=True)

    T = {tl.float64: torch.float64, tl.float32: torch.float32, tl.float16: torch.float16,
         tl.bfloat16: torch.bfloat16, tl.float8e4nv: torch.float8_e4m3fn, tl.float8e5: torch.float8_e5m2}
    FP8_MAX = {tl.float8e4nv: 448.0, tl.float8e5: 57344.0}
    SINT = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
    UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
    TINT = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    BITS_TYPES = (tl.bfloat16, tl.float8e4nv, tl.float8e5, tl.float8e5b16, tl.float8e4b8, tl.float8e4b15)

    def scalar(dt):
        return dt.scalar if hasattr(dt, "scalar") else dt

    def np_to_torch(data, dt):
        """numpy storage of a triton float dtype -> torch tensor of that dtype (same values)."""
        a = np.ascontiguousarray(data)
        width = T[dt].itemsize
        return torch.from_numpy(a.view(SINT[width]).copy()).view(T[dt])

    def torch_to_np(t, dt):
        """torch tensor of triton float dtype dt -> the interpreter's numpy storage for dt."""
        t = t.contiguous()
        width = t.element_size()
        raw = t.view(TINT[width]).numpy()
        return raw.view(I._get_np_dtype(dt))

    def to_torch_dtype(t, dst):
        """Value conversion to triton dtype dst with GPU semantics: RNE, fp8 satfinite."""
        if dst in FP8_MAX:
            lim = FP8_MAX[dst]
            t = t.to(torch.float32) if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else t
            t = torch.where(t.abs() > lim, torch.sign(t) * lim, t)  # NaN stays NaN
        return t.to(T[dst])

    def is_bits(h):
        return scalar(h.dtype) in BITS_TYPES and h.data.dtype in (np.uint8, np.uint16)

    def as_f32(h):
        dt = scalar(h.dtype)
        if dt == tl.bfloat16 and h.data.dtype == np.uint16:
            return (h.data.astype(np.uint32) << 16).view(np.float32)
        if is_bits(h):
            raise NotImplementedError(f"[triton_cpu shim] elementwise arithmetic on {dt} is not supported "
                                      f"by the interpreter; cast to tl.float32 first")
        return h.data

    def from_f32(data, dt):
        if scalar(dt) == tl.bfloat16 and data.dtype.kind == "f":
            return torch_to_np(torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).to(torch.bfloat16),
                               tl.bfloat16)
        return data

    # ---- casts -----------------------------------------------------------------------------------
    orig_convert = I._convert_float

    def _convert_float(input, input_dtype, output_dtype, rounding_mode):
        ok = input_dtype in T and output_dtype in T
        downcast = output_dtype.primitive_bitwidth < input_dtype.primitive_bitwidth
        if not ok or (downcast and rounding_mode not in (None, _ir.ROUNDING_MODE.RTNE)):
            return orig_convert(input, input_dtype, output_dtype, rounding_mode)
        out = to_torch_dtype(np_to_torch(input, input_dtype), output_dtype)
        width = out.element_size()
        return out.contiguous().view(TINT[width]).numpy().view(UINT[width]).reshape(np.shape(input))

    I._convert_float = _convert_float

    orig_cast_impl = I.InterpreterBuilder.cast_impl

    def cast_impl(self, src, dst_type):
        s, d = scalar(src.dtype), scalar(dst_type)
        if s == tl.bfloat16 or d == tl.bfloat16:
            if s == tl.bfloat16:
                val = torch.from_numpy(as_f32(src).copy())
            else:
                val = torch.from_numpy(np.ascontiguousarray(src.data).astype(np.float64))
            if d in T:
                out = torch_to_np(to_torch_dtype(val, d), d)
            else:  # bf16 -> int: truncation toward zero, as fptosi
                out = val.numpy().astype(I._get_np_dtype(d))
            return I.TensorHandle(out.reshape(np.shape(src.data)), d)
        return orig_cast_impl(self, src, dst_type)

    I.InterpreterBuilder.cast_impl = cast_impl
    for name in ("create_si_to_fp", "create_ui_to_fp", "create_fp_to_si", "create_fp_to_ui", "create_fp_ext",
                 "create_fp_trunc"):
        setattr(I.InterpreterBuilder, name, lambda self, src, dst_type: self.cast_impl(src, dst_type))

    # ---- dot -------------------------------------------------------------------------------------
    orig_dot = I.InterpreterBuilder.create_dot

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        da, db, dd = scalar(a.dtype), scalar(b.dtype), scalar(d.dtype)
        if not (da in T and db in T and dd in T):
            return orig_dot(self, a, b, d, input_precision, max_num_imprecise_acc)
        acc = np_to_torch(a.data, da).double() @ np_to_torch(b.data, db).double() + np_to_torch(d.data, dd).double()
        return I.TensorHandle(torch_to_np(acc.to(T[dd]), dd), dd)

    I.InterpreterBuilder.create_dot = create_dot

    # ---- bf16 elementwise ------------------------------------------------------------------------
    orig_binary, orig_unary, orig_ternary = (I.InterpreterBuilder.binary_op, I.InterpreterBuilder.unary_op,
                                             I.InterpreterBuilder.ternary_op)

    def binary_op(self, lhs, rhs, op):
        if not (is_bits(lhs) or is_bits(rhs)) or op in (np.bitwise_and, np.bitwise_or, np.bitwise_xor):
            return orig_binary(self, lhs, rhs, op)
        return I.TensorHandle(from_f32(op(as_f32(lhs), as_f32(rhs)), lhs.dtype), scalar(lhs.dtype))

    def unary_op(self, arg, op):
        if not is_bits(arg):
            return orig_unary(self, arg, op)
        return I.TensorHandle(from_f32(op(as_f32(arg)), arg.dtype), scalar(arg.dtype))

    def ternary_op(self, lhs, rhs, other, op):
        if not (is_bits(lhs) or is_bits(rhs) or is_bits(other)):
            return orig_ternary(self, lhs, rhs, other, op)
        conv = lambda h: as_f32(h) if is_bits(h) else h.data
        return I.TensorHandle(from_f32(op(conv(lhs), conv(rhs), conv(other)), other.dtype), scalar(other.dtype))

    def create_fma(self, x, y, z):
        if not (is_bits(x) or is_bits(y) or is_bits(z)):
            return I.TensorHandle(x.data * y.data + z.data, scalar(z.dtype))
        return I.TensorHandle(from_f32(as_f32(x) * as_f32(y) + as_f32(z), z.dtype), scalar(z.dtype))

    def get_bf16(self, value):  # a float constant in a bf16 expression; missing from the interpreter
        return I.TensorHandle(torch_to_np(torch.tensor([value], dtype=torch.float64).bfloat16(), tl.bfloat16),
                              tl.bfloat16)

    I.InterpreterBuilder.get_bf16 = get_bf16
    I.InterpreterBuilder.binary_op = binary_op
    I.InterpreterBuilder.unary_op = unary_op
    I.InterpreterBuilder.ternary_op = ternary_op
    I.InterpreterBuilder.create_fma = create_fma

    # ---- reductions: the frontend promotes bf16 to fp32 for min/max/sum; refuse anything that is not ----
    for cls_name, meths in (("ReduceOps", ("min_max", "sum")), ("ScanOps", ("cumsum", "cumprod"))):
        cls = getattr(I, cls_name, None)
        for m in meths if cls is not None else ():
            orig = getattr(cls, m)

            def guarded(self, input, *args, _orig=orig, **kw):
                if hasattr(input, "handle") and is_bits(input.handle):
                    raise NotImplementedError(f"[triton_cpu shim] reduction over raw {input.dtype} bits")
                return _orig(self, input, *args, **kw)

            setattr(cls, m, guarded)

    # ---- numpy >= 2.3: int() of a 1-element array ------------------------------------------------
    orig_patch_tensor = I._patch_lang_tensor

    def _patch_lang_tensor(tensor, *scope):  # 3.6+ passes a restore scope
        orig_patch_tensor(tensor, *scope)
        index = lambda self: int(np.asarray(self.handle.data).reshape(-1)[0])
        if scope:
            scope[0].set_attr(tensor, "__index__", index)
        else:
            tensor.__index__ = index

    I._patch_lang_tensor = _patch_lang_tensor
    # Markers: other test files carry their own partial patches (hd-speed's test_interp_kernels.py checks
    # `_hd_patched`); with this shim loaded they skip them, so there is one set of interpreter fixes.
    I._triton_cpu_shim = True
    I._hd_patched = True
    _installed = True


def cpu_fallbacks(num_sms=None):
    """Let host-side launch code written for CUDA run on CPU tensors: torch.cuda.get_device_properties
    (multi_processor_count sizes persistent grids; default 3, so every persistent loop takes several
    tiles per program) and torch.cuda.synchronize. No-op when CUDA is available."""
    if torch.cuda.is_available():
        return
    n = int(num_sms or os.environ.get("TRITON_CPU_NUM_SMS", "3"))
    props = types.SimpleNamespace(multi_processor_count=n, name="triton-interpreter", total_memory=1 << 34,
                                  major=9, minor=0)
    torch.cuda.get_device_properties = lambda *a, **k: props
    torch.cuda.synchronize = lambda *a, **k: None
