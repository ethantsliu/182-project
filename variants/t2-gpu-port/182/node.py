"""Node driver for variants/t2-gpu-port (called by 182/run_node.sh; standard library only).

    node.py check   CPU-only preflight; prints PASS|WARN|FAIL lines, exit 1 on any FAIL
    node.py prep    off-the-clock setup: data shards, FA3 wheel, tokenizer cache, canonical-mask val check
    node.py run     one timed run; last stdout line is "RESULT variant=t2-gpu-port ..."
    node.py ab      interleaved A/B of two flag sets (default: base vs FUSED_MLP=1, both with CANON_MASK=1)
    node.py parse <log>   print the RESULT line for an existing trainer log / captured stdout

Environment (all optional): NPROC (8), DATA_PATH (the repo root), LOG_DIR (<variant>/runs), EXTRA_ENV
("K=V K=V", added to the trainer's environment), DRY_RUN=1, RUN_TIMEOUT (seconds), SKIP_CHECK=1,
PREP_TIMEOUT (3600), PREP_SHARDS (25), AB_RUNS (2 pairs), AB_A / AB_B (flag sets), PYTHON, ALLOW_STACK=1 (run on a
torch / triton other than the pinned 2.10.0+cu128 / 3.6.0: check WARNs instead of FAILing).
"""
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import tempfile
import time

VARIANT = "t2-gpu-port"
TARGET = 2.92
HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(VARIANT_DIR))
TRAINER = "train_gpt_medium.py"
FA3_REPO = "varunneal/flash-attention-3"
# torch (major.minor) x CUDA builds FA3_REPO publishes for x86_64 linux (its build/ folder, 2026-11-11)
FA3_TORCH = ("2.8", "2.9", "2.10", "2.11", "2.12")
FA3_CUDA = ("12.6", "12.8", "13.0")
# The stack PR #261 was measured on and the repo root pins. Anything else FAILs check unless ALLOW_STACK=1:
# Inductor's fusion and memory plan, the FA3 wheel and the timing all depend on it.
WANT_TORCH, WANT_TRITON = "2.10.0+cu128", "3.6.0"
STACK_FIX = ("fix: from the repo root, bash 182/tools/node_setup.sh --mode venv (builds .venv-run with torch 2.10.0+cu128 "
             "and triton 3.6.0), then rerun with PYTHON=<repo>/.venv-run/bin/python; or ALLOW_STACK=1 to run on this stack anyway")
# This tree's knobs (train_gpt_medium.py). Anything else in EXTRA_ENV is passed through but flagged.
FLAGS = ("COORD_DESCENT_TUNING", "CANON_MASK", "VAL_CE_CHUNKS", "PREFIX_WEIGHT", "PREFIX_HOLD_FRAC", "PREFIX_END_FRAC",
         "XSA", "XSA_LAYERS", "ATTN_GATE", "FUSED_MLP", "NUM_SCHEDULED_ITERATIONS", "NUM_EXTENSION_ITERATIONS",
         "COOLDOWN_FRAC", "MUON_LR", "ADAM_LR", "VAL_LOSS_EVERY")
# Tokens one run trains on: 4125 scheduled steps over the 12-stage batch schedule plus 25 extension steps.
TRAIN_TOKENS = 4125 * (131072 + 262144 + 393216 + 9 * 524288) // 12 + 25 * 524288  # 1.905e9
SHARD_TOKENS = 100_000_000
MIN_SHARDS = math.ceil(TRAIN_TOKENS / SHARD_TOKENS)  # 20: below this the loader runs out of files
WANT_SHARDS = 25  # margin for the tokens BOS alignment skips

VAL_RE = re.compile(r"^step:(\d+)/(\d+) val_loss:(\S+) train_time:(\d+)ms", re.M)
TRAIN_RE = re.compile(r"^step:(\d+)/(\d+) train_time:(\d+)ms", re.M)
CANON_RE = re.compile(r"^canon_mask: val_loss_masked:(\S+) val_loss_unmasked:(\S+) delta:(\S+) canon_ms:(\S+)", re.M)
TRAIN_LOG_PATH_RE = re.compile(r"^TRAIN_LOG_PATH=(.+)$", re.M)
LOGPATH_RE = re.compile(r"^(logs/[0-9a-f-]{36}\.txt)\s*$", re.M)
PEAK_RE = re.compile(r"^peak memory allocated: (\d+) MiB", re.M)
TORCH_RE = re.compile(r"^Running PyTorch (\S+)", re.M)
# train_gpt_medium.py's out-of-memory report (first line per rank) and its validation retries
OOM_RE = re.compile(r"^(OOM rank=\d+ phase=.*)$", re.M)
OOM_RECOVERED_RE = re.compile(r"^(OOM rank=\d+ recovered: .*)$", re.M)
MEM_RE = re.compile(r"^(mem rank=\d+ at=\S+ .*)$", re.M)
ERROR_RE = re.compile(r"^(\w*(?:Error|Exception|Interrupt)\b.*|.*CUDA out of memory.*|.*Killed.*)$", re.M)


# ----------------------------------------------------------------------------- log parsing

def parse_log(text: str) -> dict:
    """What a trainer log (or its captured stdout) says. `completed` needs the final validation line
    (step == total steps) with a finite loss."""
    vals = [(int(s), int(t), v, int(ms)) for s, t, v, ms in VAL_RE.findall(text)]
    trains = TRAIN_RE.findall(text)
    out = dict(completed=False, val_loss=None, train_time_s=None, steps=None, total_steps=None,
               last_train_step=int(trains[-1][0]) if trains else None, canon=None, peak_mem_mib=None, error=None,
               torch=None, oom=None, oom_recovered=OOM_RECOVERED_RE.findall(text), last_mem=None)
    if vals:
        step, total, raw, ms = vals[-1]
        try:
            loss = float(raw)
        except ValueError:
            loss = float("nan")
        out.update(val_loss=loss, train_time_s=ms / 1000, steps=step, total_steps=total,
                   completed=step == total and math.isfinite(loss))
    elif trains:
        out["total_steps"] = int(trains[-1][1])
    m = CANON_RE.search(text)
    if m:
        try:
            masked, unmasked, delta, canon_ms = map(float, m.groups())
            out["canon"] = dict(masked=masked, unmasked=unmasked, delta=delta, canon_ms=canon_ms)
        except ValueError:
            pass
    m = PEAK_RE.search(text)
    if m:
        out["peak_mem_mib"] = int(m.group(1))
    m = TORCH_RE.search(text)
    if m:
        out["torch"] = m.group(1)
    ooms = OOM_RE.findall(text)
    if ooms:
        out["oom"] = ooms[0]
    mems = MEM_RE.findall(text)
    if mems:
        out["last_mem"] = mems[-1]
    errs = ERROR_RE.findall(text[-20000:])
    if errs:
        out["error"] = errs[-1].strip()[:200]
    return out


def failure_reason(parsed: dict, rc: int | None = 0, timed_out: bool = False) -> str | None:
    """None when the run produced a usable result, else one line saying why not."""
    if timed_out:
        return f"timed out (last training step {parsed['last_train_step']} of {parsed['total_steps']})"
    if parsed["completed"] and not rc:
        return None
    if parsed["val_loss"] is not None and not math.isfinite(parsed["val_loss"]):
        return f"val_loss is {parsed['val_loss']} at step {parsed['steps']}"
    where = (f"stopped at training step {parsed['last_train_step']} of {parsed['total_steps']}"
             if parsed["last_train_step"] is not None else "no training step was logged (startup or compile failure)")
    if parsed["completed"]:
        where = "final validation was logged"
    bits = [where] + ([f"exit code {rc}"] if rc else []) + ([parsed["error"]] if parsed["error"] else [])
    if parsed.get("oom"):
        # where the trainer was and what the allocator held (memory_summary follows it in the log)
        bits.append(" ".join(parsed["oom"].split())[:300])
    elif parsed.get("last_mem") and parsed["error"] and "out of memory" in parsed["error"].lower():
        bits.append("last memory line: " + parsed["last_mem"][:200])
    return "; ".join(bits)


def result_line(parsed: dict, log: str, reason: str | None) -> str:
    """torch= is the version the trainer logged, else the one the preflight saw, else unknown. It comes before
    reason=, which runs to the end of the line."""
    torch_ver = parsed.get("torch") or (LAST_PROBE.get("torch") or {}).get("version") or "unknown"
    if reason is not None:
        return f"RESULT variant={VARIANT} status=failed torch={torch_ver} reason={' '.join(reason.split())}"
    return (f"RESULT variant={VARIANT} val_loss={parsed['val_loss']:.4f} train_time_s={parsed['train_time_s']:.3f} "
            f"steps={parsed['steps']} target={TARGET:.2f} hit={int(parsed['val_loss'] <= TARGET)} torch={torch_ver} log={log}")


def canon_line(parsed: dict) -> str | None:
    c = parsed["canon"]
    if not c:
        return None
    return (f"CANON variant={VARIANT} val_loss_masked={c['masked']:.6f} val_loss_unmasked={c['unmasked']:.6f} "
            f"delta={c['delta']:.6f} canon_ms={c['canon_ms']:.0f}")


# ----------------------------------------------------------------------------- environment

def python_exe() -> str:
    return os.environ.get("PYTHON") or sys.executable


def data_path() -> str:
    return os.path.abspath(os.environ.get("DATA_PATH") or REPO_ROOT)


def parse_extra_env(spec: str) -> dict:
    out = {}
    for tok in shlex.split(spec or ""):
        if "=" not in tok or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok.split("=", 1)[0]):
            raise ValueError(f"EXTRA_ENV / flag set entry {tok!r} is not NAME=value")
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def cuda_headers() -> tuple[str | None, str]:
    """(CUDA home whose include/ has cuda_bf16.h, how it was found). torch compiles the CE kernel with
    NVRTC at import and adds <CUDA home>/include, where CUDA home is $CUDA_HOME, else nvcc's prefix,
    else /usr/local/cuda. The pip wheel nvidia-cuda-runtime-cu12 (a torch dependency) has the headers too."""
    cands = []
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            cands.append((os.environ[var], f"${var}"))
            break
    else:
        nvcc = shutil.which("nvcc")
        cands.append((os.path.dirname(os.path.dirname(nvcc)), "nvcc on PATH") if nvcc else ("/usr/local/cuda", "/usr/local/cuda"))
    home, how = cands[0]
    if os.path.isfile(os.path.join(home, "include", "cuda_bf16.h")):
        return home, how
    try:
        out = subprocess.run([python_exe(), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    wheel = os.path.join(out, "nvidia", "cuda_runtime") if out else ""
    if wheel and os.path.isfile(os.path.join(wheel, "include", "cuda_bf16.h")):
        return wheel, f"pip wheel (no cuda_bf16.h under {home}/include from {how})"
    return None, f"no cuda_bf16.h under {home}/include ({how}) or in the nvidia-cuda-runtime pip wheel"


def run_env(extra: dict) -> tuple[dict, dict]:
    """(full environment for the trainer, the part of it this script sets or that selects flags)."""
    shown = {"DATA_PATH": data_path()}
    home, how = cuda_headers()
    if home and how.startswith("pip wheel"):
        shown["CUDA_HOME"] = home  # torch would otherwise fail with "CUDA_HOME environment variable is not set"
    for k in FLAGS:
        if k in os.environ:
            shown[k] = os.environ[k]
    shown.update(parse_extra_env(os.environ.get("EXTRA_ENV", "")))
    shown.update(extra)
    env = dict(os.environ)
    env.update(shown)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env, shown


def trainer_cmd() -> list[str]:
    nproc = os.environ.get("NPROC", "8")
    return [python_exe(), "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", TRAINER]


def shell_form(cmd: list[str], shown: dict) -> str:
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in shown.items())
    return f"cd {shlex.quote(VARIANT_DIR)} && env {envs} {' '.join(shlex.quote(c) for c in cmd)}"


# ----------------------------------------------------------------------------- check

PROBE = r'''
import json, os, sys, ast
out = {"python": sys.version.split()[0]}
def attempt(name, fn):
    try:
        out[name] = fn()
    except BaseException as e:
        out[name + "_error"] = f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"
def torch_info():
    import torch
    return dict(version=torch.__version__, cuda=torch.version.cuda, gpus=torch.cuda.device_count() if torch.version.cuda else 0)
attempt("torch", torch_info)
def triton_info():
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor
    return triton.__version__
attempt("triton", triton_info)
def kernels_info():
    import kernels, huggingface_hub
    from huggingface_hub.constants import HF_HUB_CACHE
    from importlib.metadata import version
    cached = os.path.isdir(os.path.join(HF_HUB_CACHE, "models--" + sys.argv[2].replace("/", "--")))
    return dict(version=version("kernels"), hub=huggingface_hub.__version__, fa3_cached=cached)
attempt("kernels", kernels_info)
attempt("numpy", lambda: __import__("numpy").__version__)
def tok():
    import tiktoken
    return len(tiktoken.get_encoding("gpt2")._mergeable_ranks)
attempt("tiktoken", tok)
def nvrtc():
    import torch, torch.cuda._utils as u
    ns = {}
    for node in ast.parse(open(os.path.join(sys.argv[1], "triton_kernels.py")).read()).body:
        name = getattr(getattr(node, "targets", [None])[0], "id", None) or getattr(node, "name", "")
        if name.startswith(("CE_KERNEL_", "_CE_PREFIX_EDITS", "_ce_prefix_source", "CE_PREFIX_KERNEL_SOURCE")):
            exec(compile(ast.Module([node], []), "triton_kernels.py", "exec"), ns)
    done = []
    for src, name in ((ns["CE_KERNEL_SOURCE"], "ce_fwd_bwd_kernel"), (ns["CE_PREFIX_KERNEL_SOURCE"], "ce_fwd_bwd_prefix_kernel")):
        ptx, _ = u._nvrtc_compile(ns["CE_KERNEL_DECLS"] + src, name, "90", ["/usr/local/cuda/include/"], ["-lineinfo", "--use_fast_math"])
        done.append(f"{name} ({len(ptx)} B PTX)")
    return ", ".join(done)
if out.get("torch", {}).get("cuda"):
    attempt("nvrtc", nvrtc)
print("PROBE " + json.dumps(out))
'''


LAST_PROBE: dict = {}  # the preflight's probe result, for the RESULT line's torch= when the trainer logged none


def probe(env: dict) -> dict:
    try:
        r = subprocess.run([python_exe(), "-c", PROBE, VARIANT_DIR, FA3_REPO], capture_output=True, text=True, timeout=45, env=env)
    except subprocess.TimeoutExpired:
        return {"probe_error": "the import probe did not finish in 45 s"}
    except OSError as e:
        return {"probe_error": f"cannot run {python_exe()}: {e}"}
    for line in r.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    return {"probe_error": f"python probe failed (rc {r.returncode}): {' '.join((r.stderr or r.stdout).split())[-200:]}"}


def gpu_list() -> tuple[list[tuple[str, int]] | None, str]:
    if not shutil.which("nvidia-smi"):
        return None, "nvidia-smi not found"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"nvidia-smi failed: {e}"
    if r.returncode:
        return None, f"nvidia-smi exit {r.returncode}: {r.stderr.strip()[:200]}"
    gpus = []
    for line in r.stdout.strip().splitlines():
        name, _, mem = line.rpartition(",")
        gpus.append((name.strip(), int(float(mem)) if mem.strip().replace(".", "").isdigit() else 0))
    return gpus, ""


def mem_total_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return None


def collect_checks() -> list[tuple[str, str, str]]:
    """[(PASS|WARN|FAIL, what, detail)]. Never raises for a missing tool: that is a FAIL line."""
    res = []
    add = lambda level, what, detail: res.append((level, what, detail))
    try:
        nproc = int(os.environ.get("NPROC", "8"))
        assert nproc > 0 and 8 % nproc == 0
        add("PASS", "NPROC", f"{nproc} (the trainer needs a divisor of 8 and accumulates gradients over 8/NPROC batches)")
    except (ValueError, AssertionError):
        nproc = 8
        add("FAIL", "NPROC", f"NPROC={os.environ.get('NPROC')!r} must be 1, 2, 4 or 8")

    home, how = cuda_headers()
    env, _ = run_env({})
    p = probe(env)
    LAST_PROBE.clear()
    LAST_PROBE.update(p)
    if "probe_error" in p:
        add("FAIL", "python", p["probe_error"] + f"; set PYTHON=<the env's python> (now {python_exe()})")
        p = {}
    else:
        ok = tuple(int(x) for x in p["python"].split(".")[:2]) >= (3, 10)
        add("PASS" if ok else "FAIL", "python", f"{p['python']} at {python_exe()}" + ("" if ok else "; needs >= 3.10 (base PR ran 3.12.3)"))

    t = p.get("torch")
    torch_cuda = None
    if t is None:
        if p:
            add("FAIL", "torch", f"{p.get('torch_error')}; fix: pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128")
    else:
        mm = ".".join(t["version"].split(".")[:2])
        torch_cuda = t["cuda"]
        if not t["cuda"]:
            add("FAIL", "torch", f"{t['version']} is a CPU build; fix: pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128")
        elif mm not in FA3_TORCH or t["cuda"] not in FA3_CUDA:
            add("FAIL", "torch", f"{t['version']} (CUDA {t['cuda']}): {FA3_REPO} has no build for it (torch {'/'.join(FA3_TORCH)}, "
                                 f"CUDA {'/'.join(FA3_CUDA)}); fix: pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128")
        elif t["version"] != WANT_TORCH:
            allow = os.environ.get("ALLOW_STACK") == "1"
            add("WARN" if allow else "FAIL", "torch",
                f"{t['version']} (CUDA {t['cuda']}); PR #261 was measured on {WANT_TORCH}, the repo root's pin"
                + ("; running anyway (ALLOW_STACK=1)" if allow else f"; {STACK_FIX}"))
        else:
            add("PASS", "torch", f"{t['version']} (CUDA {t['cuda']}), as PR #261's logs")

    if p:
        if "triton" in p:
            allow = os.environ.get("ALLOW_STACK") == "1"
            lvl = "PASS" if p["triton"] == WANT_TRITON else "WARN" if allow else "FAIL"
            add(lvl, "triton", p["triton"] + ("" if lvl == "PASS" else f"; PR #261 ran {WANT_TRITON} (the version torch 2.10 requires)"
                                              + ("; running anyway (ALLOW_STACK=1)" if allow else f"; {STACK_FIX}")))
        else:
            add("FAIL", "triton", f"{p.get('triton_error')}; fix: pip install triton==3.6.0 (comes with the torch cu128 wheel)")
        if "kernels" in p:
            k = p["kernels"]
            add("PASS", "kernels", f"kernels {k['version']}, huggingface-hub {k['hub']}")
            add("PASS" if k["fa3_cached"] else "WARN", "flash-attention wheel",
                f"{FA3_REPO} " + ("is in the Hugging Face cache" if k["fa3_cached"] else
                                  "not in the Hugging Face cache; every rank downloads it at start. fix: run_node.sh prep"))
        else:
            add("FAIL", "kernels", f"{p.get('kernels_error')}; fix: pip install -r {os.path.join(REPO_ROOT, 'requirements.txt')}")
        if "numpy" not in p:
            add("FAIL", "numpy", f"{p.get('numpy_error')}; fix: pip install numpy")
        if "tiktoken" in p:
            add("PASS", "tokenizer", "tiktoken gpt2 encoding loads (medium_extras.py needs it at import)")
        else:
            add("FAIL", "tokenizer", f"{p.get('tiktoken_error')}; fix: pip install tiktoken, then run_node.sh prep (caches the gpt2 vocabulary, needs network once)")

    # CUDA headers + the real NVRTC compile of both CE kernels (CPU only, about 2 s)
    if home is None:
        add("FAIL", "CUDA headers", how + "; the CE kernel includes <cuda_bf16.h>. fix: export CUDA_HOME=<CUDA toolkit root> "
                                          "or pip install nvidia-cuda-runtime-cu12==12.8.90")
    else:
        add("PASS", "CUDA headers", f"{home}/include ({how})" + ("; run exports CUDA_HOME to it" if how.startswith("pip wheel") else ""))
    if "nvrtc" in p:
        add("PASS", "CE kernel compile", f"NVRTC sm_90: {p['nvrtc']}")
    elif "nvrtc_error" in p:
        add("FAIL", "CE kernel compile", f"NVRTC: {p['nvrtc_error']}; the trainer compiles this at import. Check CUDA_HOME and that libnvrtc.so.12 loads")
    elif p and torch_cuda is None and t is not None:
        add("WARN", "CE kernel compile", "skipped (CPU torch build); bash 182/compile_check.sh does it in docker")

    gpus, err = gpu_list()
    if gpus is None:
        add("FAIL", "GPUs", f"{err}; this tree needs {nproc} H100-class GPUs")
    elif len(gpus) < nproc:
        add("FAIL", "GPUs", f"{len(gpus)} visible, NPROC={nproc}; fix: NPROC={max(n for n in (1, 2, 4, 8) if n <= max(len(gpus), 1))} or expose more GPUs")
    else:
        names = sorted({g[0] for g in gpus})
        small = [g for g in gpus if g[1] and g[1] < 79000]
        not_h100 = [n for n in names if "H100" not in n]
        if small:
            add("FAIL", "GPUs", f"{len(gpus)} x {', '.join(names)}; {len(small)} have < 80 GB and PR #261 peaks at 59977 MiB allocated per GPU")
        elif not_h100:
            add("WARN", "GPUs", f"{len(gpus)} x {', '.join(names)}: not H100, so times do not compare with the 860.5 s record "
                                "(the CE kernel is compiled for sm_90)")
        else:
            add("PASS", "GPUs", f"{len(gpus)} x {', '.join(names)}, {min(g[1] for g in gpus)} MiB each")
        if t and t.get("cuda") and t["gpus"] < nproc:
            add("FAIL", "GPUs (torch)", f"torch sees {t['gpus']} CUDA devices, NPROC={nproc}; check CUDA_VISIBLE_DEVICES and the driver")

    # data
    d = os.path.join(data_path(), "data", "fineweb10B")
    import glob
    train = sorted(glob.glob(os.path.join(d, "fineweb_train_*.bin")))
    val = sorted(glob.glob(os.path.join(d, "fineweb_val_*.bin")))
    short = [f for f in train + val if os.path.getsize(f) < 1024 + 2 * 90_000_000]
    fix = "fix: run_node.sh prep" + ("" if data_path() == REPO_ROOT else f" (downloads to {REPO_ROOT}/data; DATA_PATH points elsewhere)")
    if not val:
        add("FAIL", "val shard", f"no fineweb_val_*.bin in {d}; {fix}")
    else:
        add("PASS", "val shard", f"{len(val)} in {d}")
    if len(train) < MIN_SHARDS:
        add("FAIL", "train shards", f"{len(train)} in {d}; one run reads {TRAIN_TOKENS / 1e9:.3f}B tokens = at least {MIN_SHARDS} shards "
                                    f"(the root record needs only 9), {WANT_SHARDS} recommended; {fix}")
    elif len(train) < WANT_SHARDS:
        add("WARN", "train shards", f"{len(train)} in {d}; enough for {TRAIN_TOKENS / 1e9:.3f}B tokens only if BOS alignment skips "
                                    f"< {100 * (len(train) * SHARD_TOKENS / TRAIN_TOKENS - 1):.0f}%; {WANT_SHARDS} recommended; {fix}")
    else:
        add("PASS", "train shards", f"{len(train)} in {d} (needs {MIN_SHARDS}, {WANT_SHARDS} recommended)")
    if short:
        add("FAIL", "shard sizes", f"{len(short)} shard(s) look truncated, e.g. {short[0]} ({os.path.getsize(short[0])} B); delete them and rerun prep")

    # host
    gb = mem_total_gb()
    if gb is None:
        add("WARN", "RAM", "could not read the host memory size")
    else:
        add("PASS" if gb >= 64 else "WARN", "RAM", f"{gb:.0f} GB" + ("" if gb >= 64 else "; 8 ranks each hold pinned data buffers, 64 GB or more recommended"))
    if os.path.isdir("/dev/shm"):
        free = shutil.disk_usage("/dev/shm").free / 2 ** 30
        add("PASS" if free >= 1 else "WARN", "/dev/shm", f"{free:.1f} GB free" + ("" if free >= 1 else "; CANON_MASK=1 puts a 316 MB shared buffer there"))
    else:
        add("WARN", "/dev/shm", "missing; CANON_MASK=1 needs shared memory for its 316 MB mask buffer")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    per = cpus / nproc
    add("PASS" if per >= 4 else "WARN", "vCPUs", f"{cpus} usable, {per:.1f} per rank" + ("" if per >= 4 else "; each rank runs a data-loader thread, "
        "under 4 per rank can slow steps"))
    free = shutil.disk_usage(VARIANT_DIR).free / 2 ** 30
    add("PASS" if free >= 2 else "WARN", "disk", f"{free:.0f} GB free at {VARIANT_DIR} (logs and compile caches)")
    return res


def cmd_check() -> int:
    res = collect_checks()
    for level, what, detail in res:
        print(f"{level} {what}: {detail}")
    fails = sum(1 for r in res if r[0] == "FAIL")
    warns = sum(1 for r in res if r[0] == "WARN")
    print(f"check: {len(res) - fails - warns} PASS, {warns} WARN, {fails} FAIL")
    return 1 if fails else 0


# ----------------------------------------------------------------------------- prep

def cmd_prep() -> int:
    timeout = float(os.environ.get("PREP_TIMEOUT", "3600"))
    shards = int(os.environ.get("PREP_SHARDS", str(WANT_SHARDS)))
    dry = os.environ.get("DRY_RUN") == "1"
    py = python_exe()
    steps = []
    if data_path() == REPO_ROOT:
        steps.append((f"fineweb10B: val shard + {shards} train shards into {REPO_ROOT}/data/fineweb10B (skips files already there)",
                      [py, os.path.join(REPO_ROOT, "data", "cached_fineweb10B.py"), str(shards)], REPO_ROOT, True))
    else:
        print(f"prep: DATA_PATH={data_path()} is not the repo root, so no download; put the shards in {data_path()}/data/fineweb10B")
    steps += [
        ("GPT-2 vocabulary into the tiktoken cache", [py, "-c", "import tiktoken; tiktoken.get_encoding('gpt2')"], VARIANT_DIR, True),
        (f"flash-attention wheel {FA3_REPO} into the Hugging Face cache (needs the CUDA torch build)",
         [py, "-c", f"from kernels import get_kernel; get_kernel({FA3_REPO!r}); print('ok')"], VARIANT_DIR, False),
        ("canonical-mask check on the real validation tokens (must count 0 masked pairs; about 20 s of CPU)",
         [py, os.path.join(HERE, "check_canon_val.py")], VARIANT_DIR, False),
    ]
    env = dict(os.environ, DATA_PATH=data_path())
    failed = 0
    for what, cmd, cwd, required in steps:
        print(f"prep: {what}\n  $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        if dry:
            continue
        rc, timed_out = run_group(cmd, cwd, env, timeout, None)
        if rc or timed_out:
            why = f"timed out after {timeout:.0f} s (PREP_TIMEOUT); rerun to resume" if timed_out else f"exit code {rc}"
            print(f"{'FAIL' if required else 'WARN'} prep step: {what}: {why}", flush=True)
            failed += required
    print("prep: " + ("FAILED, see above" if failed else "dry run, nothing done" if dry else "done; now run_node.sh check"))
    return 1 if failed else 0


# ----------------------------------------------------------------------------- run

def run_group(cmd, cwd, env, timeout, out_path):
    """Run cmd in its own process group, echo its output (and tee it to out_path), kill the whole group on
    timeout or on SIGINT/SIGTERM. Returns (exit code, timed_out)."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", start_new_session=True)
    state = {"timed_out": False}

    def kill_group(sig=signal.SIGTERM):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def on_timeout():
        state["timed_out"] = True
        kill_group()
        time.sleep(10)
        kill_group(signal.SIGKILL)

    def on_signal(signum, frame):
        kill_group()
        raise KeyboardInterrupt(f"signal {signum}")

    timer = threading.Timer(timeout, on_timeout) if timeout and timeout > 0 else None
    old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    out = open(out_path, "w") if out_path else None
    try:
        if timer:
            timer.daemon = True
            timer.start()
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if out:
                out.write(line)
                out.flush()
        proc.wait()
    finally:
        if timer:
            timer.cancel()
        if proc.poll() is None:
            kill_group()
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                kill_group(signal.SIGKILL)
                proc.wait()
        if out:
            out.close()
        for s, h in old.items():
            signal.signal(s, h)
    return proc.returncode, state["timed_out"]


def default_timeout(shown: dict) -> float:
    if os.environ.get("RUN_TIMEOUT"):
        return float(os.environ["RUN_TIMEOUT"])
    # ~14.4 min of training + ~7 min of compile and warmup + off-clock validations; coordinate-descent
    # tuning adds 30+ min of compile
    return 10800.0 if shown.get("COORD_DESCENT_TUNING", "0") not in ("0", "") else 5400.0


def run_once(extra: dict, tag: str) -> tuple[dict, str | None, str]:
    """One trainer run. Returns (parsed log, failure reason or None, log path)."""
    env, shown = run_env(extra)
    cmd = trainer_cmd()
    log_dir = os.path.abspath(os.environ.get("LOG_DIR") or os.path.join(VARIANT_DIR, "runs"))
    out_path = os.path.join(log_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}.out")
    print(f"run: {shell_form(cmd, shown)}\nrun: stdout -> {out_path}", flush=True)
    trainer_dir = env.get("TRAIN_LOG_DIR") or os.path.join(log_dir, "trainer")
    env["TRAIN_LOG_DIR"] = os.path.abspath(os.path.join(VARIANT_DIR, trainer_dir))
    try:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(env["TRAIN_LOG_DIR"], exist_ok=True)
        with tempfile.TemporaryFile(dir=env["TRAIN_LOG_DIR"]):
            pass
        with open(out_path, "w"):
            pass
    except OSError as exc:
        return parse_log(""), f"cannot write run logs: {exc}; set LOG_DIR and TRAIN_LOG_DIR to writable directories", out_path
    interrupted = None
    try:
        rc, timed_out = run_group(cmd, VARIANT_DIR, env, default_timeout(shown), out_path)
    except KeyboardInterrupt as e:
        rc, timed_out, interrupted = 130, False, str(e) or "interrupted"
    with open(out_path, errors="replace") as f:
        text = f.read()
    parsed = parse_log(text)
    reason = f"interrupted ({interrupted})" if interrupted else failure_reason(parsed, rc, timed_out)
    m = TRAIN_LOG_PATH_RE.search(text) or LOGPATH_RE.search(text)
    log = os.path.join(VARIANT_DIR, m.group(1)) if m and os.path.isfile(os.path.join(VARIANT_DIR, m.group(1))) else out_path
    return parsed, reason, log


def preflight() -> str | None:
    if os.environ.get("SKIP_CHECK") == "1":
        return None
    res = collect_checks()
    for level, what, detail in res:
        if level != "PASS":
            print(f"{level} {what}: {detail}")
    fails = [r for r in res if r[0] == "FAIL"]
    return f"preflight: {len(fails)} FAIL, first: {fails[0][1]}: {fails[0][2]}" if fails else None


def notes(shown: dict):
    if os.environ.get("SEED"):
        print("NOTE SEED is ignored: train_gpt_medium.py (PR #261) does not seed its RNGs, every run is a fresh draw")
    unknown = [k for k in parse_extra_env(os.environ.get("EXTRA_ENV", "")) if k not in FLAGS]
    if unknown:
        print(f"NOTE EXTRA_ENV sets {', '.join(unknown)}, which are not flags of this tree (passed through anyway)")


def cmd_run() -> int:
    try:
        env, shown = run_env({})
    except ValueError as e:
        print(result_line({}, "", str(e)))
        return 2
    notes(shown)
    if os.environ.get("DRY_RUN") == "1":
        print(shell_form(trainer_cmd(), shown))
        return 0
    bad = preflight()
    if bad:
        print(result_line({}, "", bad))
        return 1
    parsed, reason, log = run_once({}, "run")
    c = canon_line(parsed)
    if c and reason is None:
        print(c)
    for line in parsed["oom_recovered"][:1]:
        print(f"NOTE a validation ran out of memory and was repeated with more row chunks: {line}")
    print(result_line(parsed, log, reason), flush=True)
    return 0 if reason is None else 1


# ----------------------------------------------------------------------------- ab

def summarize_ab(rows: list[dict], spec: dict) -> list[str]:
    """Lines for the A/B summary. rows: dicts with arm, pair, val_loss, train_time_s, unmasked, canon_ms."""
    lines = []
    arms = {}
    for arm in ("A", "B"):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        arms[arm] = rs
        un = [r["unmasked"] for r in rs if r["unmasked"] is not None]
        ms = lambda xs: f"{statistics.fmean(xs):.4f}" + (f" (sd {statistics.stdev(xs):.4f})" if len(xs) > 1 else "")
        ts = [r["train_time_s"] for r in rs]
        lines.append(f"AB arm={arm} flags=[{spec[arm]}] n={len(rs)} val_loss={ms([r['val_loss'] for r in rs])}"
                     + (f" val_loss_unmasked={ms(un)}" if un else "")
                     + f" train_time_s={statistics.fmean(ts):.1f}" + (f" (sd {statistics.stdev(ts):.1f})" if len(ts) > 1 else ""))
    pairs = sorted({r["pair"] for r in rows})
    dt, dl = [], []
    for p in pairs:
        a = [r for r in rows if r["pair"] == p and r["arm"] == "A"]
        b = [r for r in rows if r["pair"] == p and r["arm"] == "B"]
        if a and b:
            dt.append(b[0]["train_time_s"] - a[0]["train_time_s"])
            key = "unmasked" if a[0]["unmasked"] is not None and b[0]["unmasked"] is not None else "val_loss"
            dl.append(b[0][key] - a[0][key])
    if dt:
        base_t = statistics.fmean(r["train_time_s"] for r in arms["A"])
        mean_dt = statistics.fmean(dt)
        lines.append(f"AB paired B-A n={len(dt)} train_time_s={mean_dt:+.2f} ({100 * mean_dt / base_t:+.2f}%)"
                     + (f" sd {statistics.stdev(dt):.2f}" if len(dt) > 1 else "")
                     + f" pairs_faster={sum(d < 0 for d in dt)}/{len(dt)} val_loss={statistics.fmean(dl):+.4f}"
                     + (f" sd {statistics.stdev(dl):.4f}" if len(dl) > 1 else ""))
    deltas = [r["unmasked"] - r["val_loss"] for r in rows if r["unmasked"] is not None]
    if deltas:
        cms = [r["canon_ms"] for r in rows if r["canon_ms"] is not None]
        lines.append(f"AB canon_mask n={len(deltas)} unmasked-masked={statistics.fmean(deltas):.6f} "
                     f"(min {min(deltas):.6f}, max {max(deltas):.6f}) canon_ms={statistics.fmean(cms):.0f}")
    return lines


def cmd_ab() -> int:
    n = int(os.environ.get("AB_RUNS", "2"))
    spec = {"A": os.environ.get("AB_A", "CANON_MASK=1"), "B": os.environ.get("AB_B", "CANON_MASK=1 FUSED_MLP=1")}
    try:
        arms = {k: parse_extra_env(v) for k, v in spec.items()}
        run_env({})
    except ValueError as e:
        print(f"AB status=failed reason={e}")
        return 2
    for a in arms.values():
        a.setdefault("VAL_LOSS_EVERY", os.environ.get("VAL_LOSS_EVERY", "0"))  # final validation only: saves wall time
    order = [(p, arm) for p in range(n) for arm in (("A", "B") if p % 2 == 0 else ("B", "A"))]
    print(f"ab: {n} pair(s), order {' '.join(a for _, a in order)}; A=[{spec['A']}] B=[{spec['B']}]")
    if os.environ.get("DRY_RUN") == "1":
        for p, arm in order:
            print(f"pair {p + 1} arm {arm}: {shell_form(trainer_cmd(), run_env(arms[arm])[1])}")
        return 0
    bad = preflight()
    if bad:
        print(f"AB status=failed reason={bad}")
        return 1
    rows = []
    for p, arm in order:
        parsed, reason, log = run_once(arms[arm], f"ab-p{p + 1}-{arm}")
        print(result_line(parsed, log, reason), flush=True)
        if reason is not None:
            for line in summarize_ab(rows, spec):
                print(line)
            print(f"AB status=failed reason=pair {p + 1} arm {arm}: {reason}")
            return 1
        c = parsed["canon"] or {}
        rows.append(dict(arm=arm, pair=p, val_loss=parsed["val_loss"], train_time_s=parsed["train_time_s"],
                         unmasked=c.get("unmasked"), canon_ms=c.get("canon_ms"), log=log))
        if c:
            rows[-1]["val_loss"] = c["masked"]  # 6 decimals instead of the step line's 4
    for line in summarize_ab(rows, spec):
        print(line)
    print(f"AB status=ok runs={len(rows)}")
    return 0


def cmd_parse(path: str) -> int:
    with open(path, errors="replace") as f:
        parsed = parse_log(f.read())
    reason = failure_reason(parsed)
    c = canon_line(parsed)
    if c and reason is None:
        print(c)
    print(result_line(parsed, os.path.abspath(path), reason))
    return 0 if reason is None else 1


def main(argv) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode == "check":
        return cmd_check()
    if mode == "prep":
        return cmd_prep()
    if mode == "run":
        return cmd_run()
    if mode == "ab":
        return cmd_ab()
    if mode == "parse" and len(argv) == 3:
        return cmd_parse(argv[2])
    print("usage: run_node.sh check|prep|run|ab   (or: node.py parse <log>)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
