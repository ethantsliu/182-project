#!/usr/bin/env python3
"""The node contract of variants/t1-cpu-ngram: check | prep | run | ab. Standard library only (it must run on a box
without torch); called through 182/run_node.sh, which sets the working directory to the variant.

    check   CPU-only preflight: PASS|WARN|FAIL <what>: <detail and the fix>; exit 1 on any FAIL
    prep    off the clock: build the exact_match extension, download the 103 training shards and the validation shard,
            prefetch the attention kernel and the tokenizer, warm the page cache; then runs check
    run     one timed run; the last stdout line is the RESULT line
    ab      AB_RUNS paired runs of AB_A (default: the run's config) against AB_B (default TABLE_VOTE=1), interleaved

Environment (all optional):
    CONFIG=381|380   381 (default): this tree's defaults, PR #381 (685 steps). 380: PR #380's configuration on this
                     code (725 steps, SCAN_HINTS=0, VAL_TOPK=2), the one with a published 8xH100 time (9.653 s, n=16).
    NPROC=8  SEED=<int> (TRAIN_SEED; unset: a random init)  EXTRA_ENV="K=V K2=V2" (or export the knobs)
    DATA_PATH=<dir holding data/fineweb10B> (default: the repo root)  LOG_DIR (default: logs/node under the variant)
    DRY_RUN=1 (print the command, change nothing)  FORCE=1 (run although check fails)  ALLOW_STACK=1 (another torch)
    ALLOW_DIRTY=1 (run although the trainer's source differs from the commit: RESULT then says tree=<commit>-dirty)
    RUN_TIMEOUT_S=2400  PREP_TIMEOUT_S=7200  INSTALL_RUST=1 (prep may install rustup)  WARM_DATA=0 (skip the read pass)
    AB_RUNS=4  AB_A=""  AB_B="TABLE_VOTE=1"  AB_SEED0=0
"""
import glob
import hashlib
import importlib.metadata
import math
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import threading
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.dirname(HERE)                      # variants/t1-cpu-ngram
REPO = os.path.dirname(os.path.dirname(VDIR))     # the repo root (where the data lives by default)
sys.path.insert(0, VDIR)
from track_1_short import host as H  # noqa: E402  (standard library only)

VARIANT = "t1-cpu-ngram"
TARGET = 3.28
TRAIN_SHARDS = 103
SHARD_MAGIC = 20240520
DATASET = "kjj0/fineweb10B-gpt2"
# PR #380's configuration expressed in this tree's knobs (the code is #381's: the validation lookups also use the
# build cores, a timing-only difference)
CONFIGS = {"381": {}, "380": {"NUM_SCHEDULED_ITERATIONS": "715", "SCAN_HINTS": "0", "VAL_TOPK": "2"}}
# the stack #380's logs ran on (and the root record's): records/track_1_short/2026-10-03_ExactCountChain upstream
STACK = {"torch": "2.10.0+cu128", "triton": "3.6.0", "kernels": "0.16.1", "huggingface-hub": "1.29.0"}
MIN_RUSTC = (1, 80)   # std::sync::LazyLock (src/sys.rs), rayon 1.12
FA3_REPO = "devenpzak/flash-attn3-12864"
KNOBS = ("SLOT_SHARED_ARENA", "TABLE_VOTE", "VOTE_K", "STREAM_THREADS", "VAL_BUILD_THREADS", "HOST_CHECK", "HOST_RANK_GIB",
         "NUM_SCHEDULED_ITERATIONS", "SCAN_HINTS", "SCAN_HINTS_SKIP", "VAL_TOPK")
_NUM = r"(?:[0-9]+(?:\.[0-9]*)?|nan|inf)"
VAL_RE = re.compile(rf"^step:(\d+)/(\d+) val_loss:({_NUM}) train_time:([0-9.]+)ms", re.M)
TRAIN_RE = re.compile(r"^step:(\d+)/(\d+) train_time:([0-9.]+)ms", re.M)
VOTE_RE = re.compile(rf"^table vote: val_loss:({_NUM}) with the table's links, ({_NUM}) without them, ({_NUM}) with PR #399's", re.M)
HOSTMEM_RE = re.compile(r"^host memory: .*$", re.M)
TRAIN_LOG_PATH_RE = re.compile(r"^TRAIN_LOG_PATH=(.+)$", re.M)
LOGPATH_RE = re.compile(r"^(logs/[0-9a-f\-]{36}\.txt)\s*$", re.M)
STAMP_RE = re.compile(r"^\[(\d+\.\d+)\] (.*)$")
BUILT_RE = re.compile(r"validation index built in ([0-9.]+) s(?: \((\d)-token key\))?")
STACK_RE = re.compile(r"^Running PyTorch (\S+) compiled for CUDA (\S+)$", re.M)
TRITON_RE = re.compile(r"^Running Triton version (\S+)$", re.M)
ATTN_RE = re.compile(r"^attention kernel: pinned=([01])\b.*?(?:kernels=(\S+))?$", re.M)
DIAG_RE = re.compile(rf"^val diagnostics: network val_loss:({_NUM}) .*?hint rows: 6-token ({_NUM}), 3/4-token ({_NUM}), "
                     rf"2-token ({_NUM}), none ({_NUM})", re.M)
THREADS_RE = re.compile(r"^retrieval host plan: (\d+) build threads", re.M)
# The trainer's own source: a change here changes what is trained or how (the attention loader, above all)
TREE_PATHS = ("train_gpt.py", "track_1_short", "exact_match/src", "exact_match/Cargo.toml", "requirements.txt")
# What a run's loss is judged against (val loss; each config's own published numbers). #381 never ran on 8 GPUs:
# its 8xH100 expectation is #380's 8xH100 mean plus the -0.0009 #381 measured against #380 on 4 GPUs (an estimate).
REFERENCE = {"381": dict(mean=3.2750, std=0.0016, what="estimate: PR #380's 8xH100 3.2759 (n=16) + PR #381's -0.0009 on "
                         "4 GPUs (#381 itself: 3.2725, n=3, 4x RTX PRO 6000 only)"),
             "380": dict(mean=3.2759, std=0.0016, what="PR #380, 8xH100, n=16, range 3.2738-3.2796")}
ERROR_RE = re.compile(r"^(?:\w+\.)*\w*(?:Error|Exception|Interrupt)\b.*$|^HOST CHECK: .*$|.*CUDA out of memory.*|.*Killed.*", re.M)


# ---------------------------------------------------------------------------------------------- logs

def parse_log(text):
    """A trainer log (or the console capture) -> dict: completed, val_loss, train_time_s, steps, total_steps,
    last_train_step, vote (with, without, fixed) or None, host_memory. A log starts with the trainer's source, whose
    f-strings never match: every pattern needs literal numbers."""
    vals = [(int(s), int(n), float(v), float(t) / 1000) for s, n, v, t in VAL_RE.findall(text)]
    trains = [(int(s), int(n)) for s, n, _ in TRAIN_RE.findall(text)]
    out = dict(completed=False, val_loss=None, train_time_s=None, steps=None, total_steps=trains[-1][1] if trains else None,
               last_train_step=trains[-1][0] if trains else None, vote=None, host_memory=None, torch=None, cuda=None,
               triton=None, fa3=None, kernels=None, diag=None, build_threads=None)
    m = STACK_RE.search(text)
    if m:
        out.update(torch=m.group(1), cuda=m.group(2))
    m = TRITON_RE.search(text)
    if m:
        out["triton"] = m.group(1)
    m = ATTN_RE.search(text)
    if m:
        out.update(fa3="pinned" if m.group(1) == "1" else "other", kernels=m.group(2))
    m = DIAG_RE.search(text)
    if m:
        out["diag"] = dict(zip(("network", "six", "three", "two", "none"), map(float, m.groups())))
    m = THREADS_RE.search(text)
    if m:
        out["build_threads"] = int(m.group(1))
    if vals:
        step, total, loss, t = vals[-1]
        out.update(steps=step, total_steps=total, val_loss=loss, train_time_s=t, completed=step == total and loss == loss)
    m = VOTE_RE.search(text)
    if m:
        out["vote"] = tuple(float(x) for x in m.groups())
    m = HOSTMEM_RE.search(text)
    if m:
        out["host_memory"] = m.group(0)
    return out


def failure_reason(text, parsed):
    """One line saying why a run has no final validation: the last error-looking line, else where it stopped."""
    errors = [m.group(0).strip() for m in ERROR_RE.finditer(text)]
    if errors:
        return errors[-1][:300]
    if parsed["val_loss"] is not None and parsed["val_loss"] != parsed["val_loss"]:
        return "val_loss is nan"
    if parsed["last_train_step"] is not None:
        return f"no final validation line (stopped after step {parsed['last_train_step']}/{parsed['total_steps']})"
    return "no step line (died before the timed run: see the console log)"


def timeline(console_text):
    """From a console capture whose lines carry `[wall seconds] ` stamps: when, on the run's clock, each index build
    finished and when the last training step did. None if the stamps or the lines are missing. The clock's start is
    estimated from the first step line (its wall stamp less its train_time), so the figures are good to a few ms."""
    t0, builds, last_train, stopped = None, [], None, None
    for line in console_text.splitlines():
        m = STAMP_RE.match(line)
        if not m:
            continue
        wall, body = float(m.group(1)), m.group(2)
        s = TRAIN_RE.match(body)
        if s and t0 is None:
            t0 = wall - float(s.group(3)) / 1000
        if s and int(s.group(1)) == int(s.group(2)):
            last_train = float(s.group(3)) / 1000
        b = BUILT_RE.search(body)
        if b:
            builds.append((wall, float(b.group(1)), b.group(2) or "?"))
        v = VAL_RE.match(body)
        if v:
            stopped = float(v.group(4)) / 1000
    if t0 is None or not builds:
        return None
    return dict(builds=[(key, wall - t0, took) for wall, took, key in builds], last_train_s=last_train, clock_s=stopped)


def describe_timeline(tl):
    parts = ", ".join(f"{key}-token at {at:.2f} s (took {took:.2f} s)" for key, at, took in tl["builds"])
    line = f"timeline: index builds finished {parts} on the clock"
    if tl["last_train_s"] is not None:
        last = max(at for _, at, _ in tl["builds"])
        slack = tl["last_train_s"] - last
        line += (f"; last training step at {tl['last_train_s']:.2f} s: the builds "
                 + (f"finished {slack:.2f} s before it (not on the critical path)" if slack > 0 else
                    f"finished {-slack:.2f} s AFTER it (the CPU build is on the critical path)"))
    if tl["clock_s"] is not None:
        line += f"; clock stopped at {tl['clock_s']:.3f} s"
    return line


def result_line(parsed=None, log="", reason=None, tags=None):
    """The RESULT line. tags (config, torch, triton, fa3, tree, ...) go before log= / reason=, which stay last (a
    reason runs to the end of the line). Spaces in a tag's value become underscores."""
    extra = "".join(f" {k}={'_'.join(str(v).split()) or 'unknown'}" for k, v in (tags or {}).items())
    if reason is not None:
        return f"RESULT variant={VARIANT} status=failed{extra} reason={' '.join(str(reason).split())}"
    return (f"RESULT variant={VARIANT} val_loss={parsed['val_loss']:.4f} train_time_s={parsed['train_time_s']:.3f} "
            f"steps={parsed['steps']} target={TARGET:.2f} hit={int(parsed['val_loss'] <= TARGET)}{extra} log={log}")


def run_tags(s, parsed=None, tree=None):
    """What a result is attributed to: the config, the stack the trainer reported in its log (else the one installed
    where node.py runs, marked ?), whether the attention kernel was the pinned FA3 binary, the source commit."""
    parsed = parsed or {}
    installed = lambda pkg: (pkg_version(pkg) or "none") + "?"
    commit, dirty = tree if tree is not None else tree_state()
    return dict(config=s.config, torch=parsed.get("torch") or installed("torch"), triton=parsed.get("triton") or installed("triton"),
                fa3=parsed.get("fa3") or "unknown", tree=(commit or "unknown") + ("-dirty" if dirty else ""))


def reference_line(config, parsed):
    """The run's loss against what its config should give, in units of PR #380's 16-run std."""
    ref = REFERENCE[config]
    d = parsed["val_loss"] - ref["mean"]
    line = (f"reference: val_loss {parsed['val_loss']:.4f} vs {ref['mean']:.4f} ({ref['what']}): {d:+.4f}, "
            f"{d / ref['std']:+.1f} std")
    if parsed.get("diag"):
        g = parsed["diag"]
        line += (f"; network alone {g['network']:.4f}; hint rows 6-token {g['six']:.3f}, 3/4-token {g['three']:.3f}, "
                 f"2-token {g['two']:.3f}, none {g['none']:.3f}")
    return line


# ------------------------------------------------------------------------------------------- settings

def parse_kv(text):
    """"K=V K2=V2" -> dict (shell quoting allowed)."""
    out = {}
    for item in shlex.split(text or ""):
        k, eq, v = item.partition("=")
        if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            raise ValueError(f"not KEY=VALUE: {item!r}")
        out[k] = v
    return out


class Settings:
    def __init__(self, env=None):
        env = dict(os.environ if env is None else env)
        self.env = env
        self.config = env.get("CONFIG", "381")
        if self.config not in CONFIGS:
            raise ValueError(f"CONFIG={self.config}: 381 (this tree's default, PR #381) or 380 (PR #380's configuration)")
        self.nproc = int(env.get("NPROC", "8"))
        self.seed = env.get("SEED", "")
        if self.seed and not re.fullmatch(r"-?\d+", self.seed):
            raise ValueError(f"SEED={self.seed}: an integer")
        self.extra = parse_kv(env.get("EXTRA_ENV", ""))
        self.data = os.path.abspath(os.path.join(VDIR, env["DATA_PATH"])) if env.get("DATA_PATH") else REPO
        self.log_dir = os.path.abspath(env.get("LOG_DIR") or os.path.join(VDIR, "logs", "node"))
        self.dry = env.get("DRY_RUN", "0") == "1"
        self.force = env.get("FORCE", "0") == "1"
        self.allow_stack = env.get("ALLOW_STACK", "0") == "1"
        self.allow_dirty = env.get("ALLOW_DIRTY", "0") == "1"
        self.run_timeout = float(env.get("RUN_TIMEOUT_S", "2400"))
        self.prep_timeout = float(env.get("PREP_TIMEOUT_S", "7200"))

    def knobs(self, more=None):
        """The trainer's variables this run sets: the config's, then exported knobs, EXTRA_ENV, `more`, the seed."""
        k = dict(CONFIGS[self.config])
        k.update({n: self.env[n] for n in KNOBS if n in self.env})
        k.update(self.extra)
        k.update(more or {})
        if self.seed:
            k["TRAIN_SEED"] = self.seed
        k["DATA_PATH"] = self.data
        return k

    def command(self):
        return ["torchrun", "--standalone", f"--nproc_per_node={self.nproc}", "train_gpt.py"]

    def shell_line(self, more=None):
        return (f"cd {shlex.quote(VDIR)} && " + " ".join(f"{k}={shlex.quote(v)}" for k, v in self.knobs(more).items())
                + " PYTHONUNBUFFERED=1 " + " ".join(self.command()))

    def data_dir(self):
        return os.path.join(self.data, "data", "fineweb10B")


# ---------------------------------------------------------------------------------------------- check

def pkg_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def tree_state(vdir=None):
    """(short commit or None, the trainer source files that differ from it or are untracked). None, [] outside a git
    checkout or without git."""
    vdir = vdir or VDIR
    try:
        head = subprocess.run(["git", "-C", vdir, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=30)
        if head.returncode:
            return None, []
        st = subprocess.run(["git", "-C", vdir, "status", "--porcelain", "--", *TREE_PATHS], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None, []
    return head.stdout.strip(), [line[3:] for line in st.stdout.splitlines() if line.strip()]


def shard_ok(path):
    """A shard file is whole: the header's magic, version and token count agree with its size (as exact_match checks)."""
    try:
        with open(path, "rb") as f:
            magic, version, n = struct.unpack("<3i", f.read(12))
        return magic == SHARD_MAGIC and version == 1 and n >= 0 and os.path.getsize(path) == 1024 + 2 * n
    except (OSError, struct.error):
        return False


def shard_names():
    return ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, TRAIN_SHARDS + 1)]


def missing_shards(data_dir):
    return [n for n in shard_names() if not shard_ok(os.path.join(data_dir, n))]


def source_hash():
    """sha256 of the extension's source: prep stamps the installed module with it, check compares."""
    h = hashlib.sha256()
    root = os.path.join(VDIR, "exact_match")
    for f in [os.path.join(root, n) for n in ("Cargo.toml", "Cargo.lock", "pyproject.toml")] + sorted(glob.glob(os.path.join(root, "src", "*.rs"))):
        h.update(os.path.basename(f).encode() + b"\0" + open(f, "rb").read())
    return h.hexdigest()


def extension_state():
    """(module directory or None, problem or None, stamp or None) of the exact_match installed for this python."""
    code = ("import exact_match, os; s = exact_match.SlotIndex; "
            "print(os.path.dirname(os.path.abspath(exact_match.__file__))); "
            "print(','.join(a for a in ('query_counts', 'memory', 'create', 'attach') if not hasattr(s, a)))")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd="/")
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e), None
    if r.returncode:
        return None, (r.stderr.strip().splitlines() or ["import failed"])[-1], None
    lines = r.stdout.splitlines() + ["", ""]
    stamp = None
    try:
        stamp = open(os.path.join(lines[0], "SOURCE_SHA256")).read().strip()
    except OSError:
        pass
    return lines[0], (f"an older build (no {lines[1]})" if lines[1] else None), stamp


def tool_version(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", out)
    return tuple(int(x) for x in m.groups(default="0")) if m else None


def cargo_path():
    """Put rustup's default bin directory on PATH (a non-login shell does not have it)."""
    d = os.path.expanduser("~/.cargo/bin")
    if os.path.isdir(d) and d not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def gpus():
    """(count, names, driver) from nvidia-smi; (0, [], None) without it."""
    if not shutil.which("nvidia-smi"):
        return len(glob.glob("/proc/driver/nvidia/gpus/*")), [], None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 0, [], None
    rows = [[c.strip() for c in l.split(",")] for l in out.splitlines() if "," in l]
    return len(rows), [r[0] for r in rows], rows[0][1] if rows else None


def find_cudart13():
    try:
        if "libcudart.so.13" in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=20).stdout:
            return "ldconfig"
    except (OSError, subprocess.SubprocessError):
        pass
    home = os.environ.get("CUDA_HOME")
    dirs = ([os.path.join(home, "targets/x86_64-linux/lib"), os.path.join(home, "lib64")] if home else []) \
        + glob.glob("/usr/local/cuda-13*/targets/x86_64-linux/lib") + ["/usr/local/cuda/lib64"] \
        + [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]
    return next((d for d in dirs if os.path.exists(os.path.join(d, "libcudart.so.13"))), None)


def is_trainer(argv):
    """A process's argv is a trainer's: python or torchrun running a train_gpt*.py script (not a shell or an editor
    whose command line merely mentions one)."""
    return bool(argv) and re.match(r"(python|torchrun)", os.path.basename(argv[0])) is not None \
        and any(re.fullmatch(r"train_gpt\w*\.py", os.path.basename(a)) for a in argv[1:])


def trainers_running(proc="/proc"):
    pids = []
    for d in glob.glob(os.path.join(proc, "[0-9]*")):
        try:
            argv = open(os.path.join(d, "cmdline"), "rb").read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if is_trainer(argv) and int(os.path.basename(d)) != os.getpid():
            pids.append(int(os.path.basename(d)))
    return sorted(pids)


def build_threads(s, per_gpu):
    return int(s.knobs().get("VAL_BUILD_THREADS", 0) or 0) or (64 if per_gpu >= 24 else 32 if per_gpu >= 16 else 16)


def check(s, out=print):
    """Print the host, the plan and one line per check; return the number of FAILs."""
    me = "bash " + os.path.join(os.path.relpath(HERE, REPO), "run_node.sh")
    rows = []

    def add(ok, level, what, detail):  # a passing row drops its "; fix: ..." tail
        rows.append(("PASS", what, detail.split("; fix:")[0]) if ok else (level, what, detail))

    knobs = s.knobs()
    shared = knobs.get("SLOT_SHARED_ARENA") == "1"
    n_gpus, names, driver = gpus()
    h = H.detect(gpus=s.nproc)
    per_gpu = h.allowed // max(1, s.nproc)
    threads = build_threads(s, per_gpu)
    out(H.describe(h))

    # the software stack
    linux = platform.system() == "Linux"
    add(linux, "FAIL", "linux", f"{platform.system()} {platform.machine()} (the index uses Linux mmap/madvise/affinity, the trainer /proc and /sys)")
    add(sys.version_info >= (3, 10), "FAIL", "python", f"{platform.python_version()} at {sys.executable} (>= 3.10; the records ran 3.12)")
    for pkg, want in STACK.items():
        try:
            got = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            got = None
        fix = ("pip install -r requirements.txt && pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128 (or, from the "
               "repo root: bash 182/tools/node_setup.sh --mode venv && source .venv-run/bin/activate)" if pkg == "torch" else f"pip install {pkg}=={want}")
        level = "WARN" if s.allow_stack and got else "FAIL"
        add(got == want, level if pkg in ("torch", "triton") or not got else "WARN", pkg,
            f"{got or 'not installed'} (the records' stack: {want}; ALLOW_STACK=1 runs with another version); fix: {fix}")
    for pkg in ("numpy", "tiktoken"):
        try:
            add(True, "FAIL", pkg, importlib.metadata.version(pkg))
        except importlib.metadata.PackageNotFoundError:
            add(False, "FAIL", pkg, "not installed; fix: pip install -r requirements.txt")
    commit, dirty = tree_state()
    if commit is None:
        add(False, "WARN", "trainer source", "not a git checkout: a locally changed trainer (e.g. the attention loader) cannot be detected")
    else:
        add(not dirty, "WARN" if s.allow_dirty else "FAIL", "trainer source",
            (f"as committed ({commit})" if not dirty else
             f"{len(dirty)} file(s) differ from commit {commit}: {', '.join(dirty[:6])}{' ...' if len(dirty) > 6 else ''} "
             f"(a changed trainer is not the run this tree describes; ALLOW_DIRTY=1 runs it, and RESULT says tree={commit}-dirty); "
             f"fix: git -C {VDIR} diff -- {' '.join(TREE_PATHS)} to see the change, git stash or git checkout to undo it"))
    tr = shutil.which("torchrun")
    add(tr is not None, "FAIL", "torchrun", tr or "not on PATH; fix: activate the environment that has torch")
    if tr and os.path.dirname(os.path.realpath(tr)) != os.path.dirname(os.path.realpath(sys.executable)) \
            and os.path.dirname(tr) != os.path.dirname(sys.executable):
        add(False, "WARN", "torchrun env", f"torchrun is {tr} but this check ran under {sys.executable}: the versions above may not be the run's (set PYTHON)")

    # GPUs and CUDA
    add(n_gpus >= s.nproc, "FAIL", "gpus", f"{n_gpus} visible, NPROC={s.nproc}" + (f" ({names[0]})" if names else "")
        + ("" if n_gpus >= s.nproc else "; fix: run on the GPU node, or set NPROC"))
    if n_gpus:
        add(s.nproc == 8, "WARN", "8 ranks", f"NPROC={s.nproc} (the schedule and every published time are for 8)")
        add(all("H100" in n for n in names), "WARN", "H100", f"{sorted(set(names))} (times are comparable only on 8xH100)")
        if driver:
            add(int(driver.split(".")[0]) >= 580, "WARN", "driver", f"{driver} (the records ran 580; ANVIL2's README measured a slowdown below it)")
    lib = find_cudart13()
    add(lib is not None, "FAIL" if linux else "WARN", "libcudart.so.13",
        (f"found ({lib})" if lib else "not found: the pinned attention kernel links it; fix: install cuda-cudart-13-1 and put its lib "
         "directory on LD_LIBRARY_PATH (the repo root's 182/tools/node_setup.sh does both)"))
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    add(os.path.isfile(os.path.join(cuda_home, "include", "cuda_bf16.h")), "FAIL" if linux else "WARN", "CUDA headers",
        f"{cuda_home}/include/cuda_bf16.h (the loss kernel compiles through nvrtc); fix: export CUDA_HOME=<a CUDA toolkit with headers>")
    hub = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    fa3 = [d for d in (os.path.join(hub, f"{kind}--{FA3_REPO.replace('/', '--')}") for kind in ("kernels", "models")) if os.path.isdir(d)]
    add(bool(fa3), "WARN", "attention kernel", (f"cached ({fa3[0]})" if fa3 else f"{FA3_REPO} is not in the Hugging Face cache: the first run "
                                                f"downloads it (needs network); fix: {me} prep"))

    # the extension
    cargo_path()
    module, problem, stamp = extension_state()
    want = source_hash()
    if module and not problem and stamp == want:
        add(True, "FAIL", "exact_match", f"built from this source ({module})")
    elif module and not problem and stamp is None:
        add(False, "WARN", "exact_match", f"installed ({module}) but not by prep: cannot tell whether it matches this source; fix: {me} prep")
    else:
        why = problem if (problem and module) else "built from other source" if module else f"not importable ({problem})"
        add(False, "FAIL", "exact_match", f"{why}; fix: {me} prep")
        rustc, cargo = tool_version(["rustc", "--version"]), tool_version(["cargo", "--version"])
        ok = rustc is not None and cargo is not None and rustc[:2] >= MIN_RUSTC
        add(ok, "FAIL", "rust toolchain", f"rustc {'.'.join(map(str, rustc)) if rustc else 'missing'}, cargo {'.'.join(map(str, cargo)) if cargo else 'missing'}"
            f" (need >= {MIN_RUSTC[0]}.{MIN_RUSTC[1]}; Ubuntu 24.04's apt rustc is 1.75)"
            + ("" if ok else f"; fix: INSTALL_RUST=1 {me} prep (installs rustup into ~/.cargo), or curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal"))
        add(shutil.which("cc") is not None or shutil.which("gcc") is not None, "FAIL", "C linker", "cc (rustc links with it); fix: sudo apt-get install -y build-essential")

    # data
    ddir = s.data_dir()
    missing = missing_shards(ddir)
    have = len(shard_names()) - len(missing)
    add(not missing, "FAIL", "data", f"{have}/{len(shard_names())} shards whole under {ddir} (all {TRAIN_SHARDS} training shards and the validation shard)"
        + ("" if not missing else f"; first missing: {missing[0]}; fix: {me} prep (about 20 GB)"))
    in_shm = os.path.realpath(ddir).startswith("/dev/shm")
    add(in_shm, "WARN", "data in /dev/shm", "as the records (reads never touch a disk)" if in_shm else
        "the data is on a disk: run reads it once before the clock to warm the page cache; the records kept it in /dev/shm "
        f"(fix: DATA_PATH=/dev/shm {me} prep, +20 GiB of RAM)")

    # the host against the plan
    files = sorted(glob.glob(os.path.join(ddir, "fineweb_train_*.bin")))
    tokens = [H.shard_tokens(f) for f in files][2:] if len(files) == TRAIN_SHARDS else [100_000_000] * (TRAIN_SHARDS - 2)
    need = H.plan(sum(tokens), sum(tokens[:20]), threads, s.nproc, shared=shared)
    alt = H.plan(sum(tokens), sum(tokens[:20]), threads, s.nproc, shared=True)
    data_shm = 0 if not in_shm else sum(os.path.getsize(f) for f in files)
    out(f"plan: config {s.config}, {s.nproc} ranks, {threads} build threads{', shared arena' if shared else ''}: arenas {need.arenas / H.GiB:.0f} GiB + "
        f"tables {need.tables / H.GiB:.0f} GiB (/dev/shm) + ranks {need.ranks / H.GiB:.0f} GiB = {need.total / H.GiB:.0f} GiB of host RAM"
        f" ({alt.total / H.GiB:.0f} GiB with SLOT_SHARED_ARENA=1)")
    add(need.total <= h.mem_available, "FAIL", "RAM", f"{h.mem_available / H.GiB:.0f} GiB available of {h.mem_total / H.GiB:.0f} GiB"
        f"{f' (cgroup limit {h.cgroup_limit / H.GiB:.0f} GiB)' if h.cgroup_limit else ''}, the plan needs {need.total / H.GiB:.0f} GiB"
        + ("" if need.total <= h.mem_available else
           (f"; fix: SLOT_SHARED_ARENA=1 (needs {alt.total / H.GiB:.0f} GiB)" if not shared and alt.total <= h.mem_available else
            "; fix: a node with more RAM (the estimate allows 12 GiB per rank unmeasured: HOST_CHECK=0 FORCE=1 tries anyway)")))
    add(h.shm_total > 0 and need.tables <= h.shm_free, "FAIL", "/dev/shm", f"{h.shm_free / H.GiB:.0f} GiB free of {h.shm_total / H.GiB:.0f} GiB, "
        f"the index tables need {need.tables / H.GiB:.0f} GiB" + (f" (the data there holds {data_shm / H.GiB:.0f} GiB)" if data_shm else "")
        + ("" if h.shm_total > 0 and need.tables <= h.shm_free else
           "; fix: sudo mount -o remount,size=200G /dev/shm (docker: --ipc=host or --shm-size=200g), and remove stale /dev/shm/exact-match-*"))
    add(h.mem_total >= 700 * H.GiB, "WARN", "RAM headroom", f"{h.mem_total / H.GiB:.0f} GiB in total (the records' boxes had 1.5-2 TB; under 700 GiB is untested)")
    add(per_gpu >= 16, "WARN", "vCPUs per GPU", f"{per_gpu} ({h.allowed} allowed, {h.cores} cores): {threads} build threads"
        + (" as the records' box (16 per GPU, 32 threads)" if 16 <= per_gpu < 24 else
           " (LARGE_HOST mode, 64 threads: never measured in the published logs)" if per_gpu >= 24 else
           " (fewer than the records' 16 per GPU: the index build may finish after training, on the clock)"))
    add(h.cores // max(1, s.nproc) >= 4, "FAIL", "cores per GPU", f"{h.cores // max(1, s.nproc)} physical cores per rank (a rank needs a main core, "
        "worker cores and build cores: at least 4)")
    add(h.avx512, "WARN", "AVX-512", "F/BW/DQ/VL" if h.avx512 else "missing: the build falls back to AVX2/scalar hashing and plain stores (slower; never timed)")
    add(platform.machine() in ("x86_64", "AMD64"), "WARN", "x86-64", platform.machine())
    kv = tuple(int(x) for x in re.findall(r"\d+", platform.release())[:2])
    add(not linux or kv >= (5, 14), "WARN", "kernel", f"{platform.release()} (>= 5.14 for MADV_POPULATE_READ/WRITE; older: page faults on the clock)")
    add("[never]" not in h.thp and h.thp != "n/a", "WARN", "transparent huge pages", f"{h.thp} (the arenas ask for huge pages; "
        "fix: echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled)")
    stale, running = glob.glob("/dev/shm/exact-match-*"), trainers_running()
    add(not running, "WARN", "no trainer running", f"{len(running)} train_gpt process(es): {running[:8]}" if running else "none")
    add(not stale, "WARN", "stale index tables", f"{len(stale)} /dev/shm/exact-match-* file(s)" + (" (a killed run's; run removes them when no trainer is running)" if stale else ""))
    try:
        free = shutil.disk_usage(ddir if os.path.isdir(ddir) else s.data).free
        add(not missing or free >= 22 * H.GiB, "FAIL", "disk for data", f"{free / H.GiB:.0f} GiB free under {s.data} (the download needs about 21 GiB)")
    except OSError:
        pass

    for level, what, detail in rows:
        out(f"{level} {what}: {detail}")
    fails, warns = sum(r[0] == "FAIL" for r in rows), sum(r[0] == "WARN" for r in rows)
    out(f"{'NOT READY' if fails else 'READY'}: {fails} FAIL, {warns} WARN ({VARIANT}, config {s.config})")
    return fails


# ----------------------------------------------------------------------------------------------- prep

def warm(data_dir, out=print):
    """Read every shard once, so the run's reads come from the page cache (the 6-token build reads 20 GB on the clock)."""
    t, n = time.time(), 0
    for name in shard_names():
        try:
            with open(os.path.join(data_dir, name), "rb", buffering=0) as f:
                while True:
                    chunk = f.read(1 << 24)
                    if not chunk:
                        break
                    n += len(chunk)
        except OSError:
            pass
    out(f"warmed the page cache: {n / H.GiB:.1f} GiB in {time.time() - t:.0f} s")


DOWNLOAD = """
import os, sys
from huggingface_hub import hf_hub_download
dest, names = sys.argv[1], sys.argv[2:]
for i, name in enumerate(names):
    hf_hub_download(repo_id=%r, filename=name, repo_type="dataset", local_dir=dest)
    print(f"downloaded {name} ({i + 1}/{len(names)})", flush=True)
""" % DATASET

PREFETCH = """
import os, pathlib, shutil
from huggingface_hub import snapshot_download
import tiktoken
tiktoken.get_encoding("gpt2")
print("tokenizer cached")
src = open(os.path.join("track_1_short", "model", "attention.py")).read()
pin = lambda name: src.split(name + ' = "')[1].split('"')[0]
repo, rev = pin("FA3_REPO"), pin("FA3_REVISION")
path = snapshot_download(repo, revision=rev)
hub = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface")) / "hub"
a, b = hub / ("models--" + repo.replace("/", "--")), hub / ("kernels--" + repo.replace("/", "--"))
if a.exists() and not b.exists():  # kernels >= 0.16 resolves through the kernel-repo namespace
    shutil.copytree(a, b, symlinks=True)
print("attention kernel cached", path)
"""


def prep(s, out=print):
    deadline = time.time() + s.prep_timeout
    left = lambda: max(1.0, deadline - time.time())
    me = "bash " + os.path.join(os.path.relpath(HERE, REPO), "run_node.sh")
    cargo_path()
    module, problem, stamp = extension_state()
    want = source_hash()
    ddir = s.data_dir()
    missing = missing_shards(ddir)
    if s.dry:
        out(f"DRY_RUN prep: extension {'up to date' if module and not problem and stamp == want else 'would be built: ' + sys.executable + ' -m pip install ./exact_match'}; "
            f"{len(missing)} shard(s) to fetch into {ddir}; then the attention kernel and tokenizer prefetch, a page-cache read and check")
        return 0

    # 1. the extension
    if module and not problem and stamp == want:
        out(f"exact_match: up to date ({module})")
    else:
        rustc = tool_version(["rustc", "--version"])
        if (rustc is None or rustc[:2] < MIN_RUSTC or not shutil.which("cargo")) and os.environ.get("INSTALL_RUST") == "1":
            out("installing rustup (stable, minimal profile) into ~/.cargo")
            subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable",
                           shell=True, timeout=left(), check=False)
            cargo_path()
            rustc = tool_version(["rustc", "--version"])
        if rustc is None or rustc[:2] < MIN_RUSTC:
            out(f"FAIL rust toolchain: rustc {'.'.join(map(str, rustc)) if rustc else 'missing'} (need >= {MIN_RUSTC[0]}.{MIN_RUSTC[1]}); "
                f"fix: INSTALL_RUST=1 {me} prep")
            return 1
        out(f"building exact_match with rustc {'.'.join(map(str, rustc))} (a few minutes)")
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "./exact_match"], cwd=VDIR, timeout=left())
        except subprocess.TimeoutExpired:
            out("FAIL exact_match: the build timed out (PREP_TIMEOUT_S); re-run prep")
            return 1
        module, problem, _ = extension_state()
        if r.returncode or not module or problem:
            out(f"FAIL exact_match: pip install ./exact_match failed ({problem or 'see above'})")
            return 1
        with open(os.path.join(module, "SOURCE_SHA256"), "w") as f:
            f.write(want + "\n")
        out(f"exact_match: built and installed ({module})")

    # 2. the data: copy what the repo root already has, download the rest (resumable: whole files are skipped)
    os.makedirs(ddir, exist_ok=True)
    root_dir = os.path.join(REPO, "data", "fineweb10B")
    if os.path.realpath(root_dir) != os.path.realpath(ddir):
        for name in list(missing):
            src = os.path.join(root_dir, name)
            if shard_ok(src):
                shutil.copyfile(src, os.path.join(ddir, name) + ".part")
                os.replace(os.path.join(ddir, name) + ".part", os.path.join(ddir, name))
                missing.remove(name)
                out(f"copied {name} from {root_dir}")
    for name in missing:  # a partial file from an interrupted download fails the header check: start it again
        if os.path.exists(os.path.join(ddir, name)):
            os.remove(os.path.join(ddir, name))
    if missing:
        out(f"downloading {len(missing)} shard(s) of {DATASET} into {ddir}")
        try:
            subprocess.run([sys.executable, "-c", DOWNLOAD, ddir, *missing], timeout=left(), check=False)
        except subprocess.TimeoutExpired:
            out("the download timed out (PREP_TIMEOUT_S); re-run prep to resume")
        missing = missing_shards(ddir)
        if missing:
            out(f"FAIL data: {len(missing)} shard(s) still missing (first: {missing[0]}); re-run prep to resume")
            return 1
    out(f"data: {len(shard_names())} shards whole under {ddir}")

    # 3. what the first run would otherwise download; not fatal (the run downloads it itself, given network)
    try:
        r = subprocess.run([sys.executable, "-c", PREFETCH], cwd=VDIR, timeout=min(left(), 900))
        if r.returncode:
            out("WARN prefetch: the attention kernel / tokenizer prefetch failed (see above); the first run will try again")
    except subprocess.TimeoutExpired:
        out("WARN prefetch: timed out; the first run will try again")
    if os.environ.get("WARM_DATA", "1") == "1" and not os.path.realpath(ddir).startswith("/dev/shm"):
        warm(ddir, out)
    return 1 if check(s, out) else 0


# ------------------------------------------------------------------------------------------------ run

def kill_group(proc):
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            pass


def clear_tables(out=print):
    """Remove a killed run's index tables (a finished run unlinks its own), unless a trainer is running."""
    stale = glob.glob("/dev/shm/exact-match-*")
    if stale and not trainers_running():
        for f in stale:
            try:
                os.remove(f)
            except OSError:
                pass
        out(f"removed {len(stale)} stale /dev/shm/exact-match-* table(s)")


def one_run(s, more=None, tag="run", out=print):
    """Launch one timed run. Returns (parsed or None, log path, failure reason or None)."""
    running = trainers_running()
    if running:
        return None, "", f"another trainer is running (pids {running[:8]}): wait for it or kill it"
    clear_tables(out)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}-c{s.config}" + (f"-s{(more or {}).get('TRAIN_SEED', s.seed)}" if (more or {}).get("TRAIN_SEED", s.seed) else "")
    console = os.path.join(s.log_dir, name + ".console.txt")
    env = dict(os.environ, **s.knobs(more), PYTHONUNBUFFERED="1")
    log_dir = env.get("TRAIN_LOG_DIR") or os.path.join(s.log_dir, "trainer")
    env["TRAIN_LOG_DIR"] = os.path.abspath(os.path.join(VDIR, log_dir))
    try:
        os.makedirs(s.log_dir, exist_ok=True)
        os.makedirs(env["TRAIN_LOG_DIR"], exist_ok=True)
        with tempfile.TemporaryFile(dir=env["TRAIN_LOG_DIR"]):
            pass
        with open(console, "w"):
            pass
    except OSError as exc:
        return None, console, f"cannot write run logs: {exc}; set LOG_DIR and TRAIN_LOG_DIR to writable directories"
    out(f"launch: {s.shell_line(more)}")
    out(f"console: {console}")
    lines = []
    try:
        proc = subprocess.Popen(s.command(), cwd=VDIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1, start_new_session=True)
    except OSError as e:
        return None, "", f"cannot launch torchrun: {e}"

    def pump():
        with open(console, "w") as f:
            for line in proc.stdout:
                stamped = f"[{time.time():.3f}] {line.rstrip()}"
                lines.append(stamped)
                f.write(stamped + "\n")
                f.flush()
                out(line.rstrip())

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    reason = None
    try:
        proc.wait(s.run_timeout)
    except subprocess.TimeoutExpired:
        reason = f"timed out after {s.run_timeout:.0f} s (RUN_TIMEOUT_S)"
        kill_group(proc)
    except KeyboardInterrupt:
        kill_group(proc)
        reader.join(5)
        clear_tables(out)
        raise
    reader.join(30)
    kill_group(proc)  # a rank that outlived torchrun
    clear_tables(out)
    text = "\n".join(lines)
    plain = "\n".join(STAMP_RE.sub(r"\2", l) for l in lines)
    log, parsed = console, parse_log(plain)
    m = TRAIN_LOG_PATH_RE.search(plain) or LOGPATH_RE.search(plain)
    if m and os.path.isfile(os.path.join(VDIR, m.group(1))):
        log = os.path.join(s.log_dir, name + ".log.txt")
        shutil.copyfile(os.path.join(VDIR, m.group(1)), log)
        from_log = parse_log(open(log, errors="replace").read())
        if from_log["val_loss"] is not None:
            parsed = dict(parsed, **{k: v for k, v in from_log.items() if v is not None})
    if reason is None and proc.returncode:
        reason = f"exit {proc.returncode}: {failure_reason(plain, parsed)}"
    if reason is None and not parsed["completed"]:
        reason = failure_reason(plain, parsed)
    tl = timeline(text)
    if tl:
        out(describe_timeline(tl))
    if parsed["host_memory"]:
        out(parsed["host_memory"])
    if parsed["vote"]:
        w, wo, fx = parsed["vote"]
        out(f"table vote on this run's weights: val_loss {w:.5f} with the links, {wo:.5f} without ({w - wo:+.5f}), {fx:.5f} with PR #399's fixed vote")
    return (None if reason else parsed), log, reason


def run(s, out=print):
    if s.dry:
        out(f"DRY_RUN config={s.config} nproc={s.nproc} seed={s.seed or 'random'} log_dir={s.log_dir}")
        out(s.shell_line())
        return 0
    tree = tree_state()
    if check(s, out) and not s.force:
        out(result_line(reason="preflight failed: see the FAIL lines above (FORCE=1 runs anyway)", tags=run_tags(s, tree=tree)))
        return 1
    if os.environ.get("WARM_DATA", "1") == "1" and not os.path.realpath(s.data_dir()).startswith("/dev/shm"):
        warm(s.data_dir(), out)
    parsed, log, reason = one_run(s, out=out)
    if reason:
        out(result_line(reason=reason + (f" (log: {log})" if log else ""), tags=run_tags(s, tree=tree)))
        return 1
    tags = run_tags(s, parsed, tree)
    for line in attribution_lines(s, parsed, tags):
        out(line)
    out(result_line(parsed, log, tags=tags))
    return 0


def attribution_lines(s, parsed, tags):
    """WARN lines for what makes a result not this tree's run on the records' stack, then the reference line."""
    lines = []
    if tags["torch"] != STACK["torch"] or tags["triton"] != STACK["triton"]:
        lines.append(f"WARN this run's stack: torch {tags['torch']} / triton {tags['triton']}, not the records' {STACK['torch']} / "
                     f"{STACK['triton']}: its loss and time are not comparable to theirs")
    if tags["fa3"] != "pinned":
        lines.append(f"WARN this run's attention kernel: {tags['fa3']} (the log's 'attention kernel:' line): not the pinned FA3 binary, "
                     "or a log without the line")
    if tags["tree"].endswith("-dirty") or tags["tree"] == "unknown":
        lines.append(f"WARN this run's trainer source: tree={tags['tree']}: not, or not known to be, the committed trainer")
    if parsed.get("val_loss") is not None and parsed["val_loss"] == parsed["val_loss"]:
        lines.append(reference_line(s.config, parsed))
    return lines


# ------------------------------------------------------------------------------------------------- ab

T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131]  # two-sided, df 1..15


def paired(diffs):
    """(mean, half-width of the 95% CI or None) of paired differences."""
    n = len(diffs)
    if n < 2:
        return (diffs[0] if diffs else float("nan")), None
    t = T95[n - 2] if n - 2 < len(T95) else 1.96
    return statistics.mean(diffs), t * statistics.stdev(diffs) / math.sqrt(n)


def ab_summary(pairs, a_flags, b_flags):
    """Lines summarising completed (a, b) parsed pairs: B - A on train_time and val_loss, and B's own vote report."""
    lines = []
    if not pairs:
        return ["AB no completed pair"]
    dt, dl = paired([b["train_time_s"] - a["train_time_s"] for a, b in pairs]), paired([b["val_loss"] - a["val_loss"] for a, b in pairs])
    ci = lambda x, unit, fmt: f"{x[0]:{fmt}} {unit}" + (f" (95% CI +-{x[1]:{fmt[1:]}})" if x[1] is not None else " (n=1: no interval)")
    lines.append(f"AB n={len(pairs)} pairs, B - A: train_time {ci(dt, 's', '+.3f')}, val_loss {ci(dl, '', '+.4f')}")
    lines.append(f"AB A [{a_flags or 'base'}]: mean train_time {statistics.mean(a['train_time_s'] for a, _ in pairs):.3f} s, "
                 f"mean val_loss {statistics.mean(a['val_loss'] for a, _ in pairs):.4f}; B [{b_flags}]: "
                 f"{statistics.mean(b['train_time_s'] for _, b in pairs):.3f} s, {statistics.mean(b['val_loss'] for _, b in pairs):.4f}")
    votes = [b["vote"] for _, b in pairs if b.get("vote")]
    if votes:
        d = paired([w - wo for w, wo, _ in votes])
        f = paired([fx - wo for _, wo, fx in votes])
        lines.append(f"AB table vote within B's runs (same weights, n={len(votes)}): links - none {ci(d, '', '+.5f')}; "
                     f"PR #399's fixed vote - none {ci(f, '', '+.5f')}")
    return lines


def ab(s, out=print):
    n = int(s.env.get("AB_RUNS", "4"))
    a_flags, b_flags = s.env.get("AB_A", ""), s.env.get("AB_B", "TABLE_VOTE=1")
    arms = {"A": parse_kv(a_flags), "B": parse_kv(b_flags)}
    seed0 = int(s.env.get("AB_SEED0", "0"))
    plan = [(i, arm) for i in range(n) for arm in (("A", "B") if i % 2 == 0 else ("B", "A"))]  # interleaved: AB BA AB BA
    if s.dry:
        out(f"DRY_RUN ab: {n} pairs, A=[{a_flags or 'base'}] B=[{b_flags}], seeds {seed0}..{seed0 + n - 1}, order {' '.join(a for _, a in plan)}")
        for i, arm in plan:
            out(s.shell_line(dict(arms[arm], TRAIN_SEED=str(seed0 + i))))
        return 0
    tree = tree_state()
    if check(s, out) and not s.force:
        out(result_line(reason="preflight failed: see the FAIL lines above (FORCE=1 runs anyway)", tags=run_tags(s, tree=tree)))
        return 1
    done = {}
    for i, arm in plan:
        parsed, log, reason = one_run(s, dict(arms[arm], TRAIN_SEED=str(seed0 + i)), tag=f"ab{arm}", out=out)
        tags = run_tags(s, parsed, tree)
        out(f"AB_RUN arm={arm} seed={seed0 + i} " + (result_line(reason=reason, tags=tags) if reason else result_line(parsed, log, tags=tags)))
        if parsed:
            done[(i, arm)] = parsed
    pairs = [(done[(i, "A")], done[(i, "B")]) for i in range(n) if (i, "A") in done and (i, "B") in done]
    for line in ab_summary(pairs, a_flags, b_flags):
        out(line)
    return 0 if len(pairs) == n else 1


def main(argv):
    mode = argv[1] if len(argv) > 1 else ""
    if mode not in ("check", "prep", "run", "ab"):
        print(__doc__)
        return 2
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        s = Settings()
    except ValueError as e:
        print(f"FAIL settings: {e}")
        if mode in ("run", "ab"):
            print(result_line(reason=str(e)))
        return 2
    try:
        if mode == "check":
            return 1 if check(s) else 0
        return {"prep": prep, "run": run, "ab": ab}[mode](s)
    except KeyboardInterrupt:
        if mode in ("run", "ab"):
            print(result_line(reason="interrupted"))
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv))
