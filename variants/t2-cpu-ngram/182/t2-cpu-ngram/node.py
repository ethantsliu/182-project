"""The node runner behind 182/run_node.sh: check | prep | run | ab for the Track 2 CPU-retrieval run.

  check  CPU-only preflight (no GPU work, no network): one `PASS|WARN|FAIL <what>: <detail and fix>` line per
         item, exit 1 on any FAIL.
  prep   everything slow and off the clock: the 103 train shards, the exact_match Rust extension, the FA3
         kernel and the GPT-2 tokenizer file. Idempotent; a second call does only what is still missing.
  run    one timed run of train_gpt_medium_cpu.py. The last stdout line is
           RESULT variant=t2-cpu-ngram val_loss=<f> train_time_s=<f> steps=<n> target=2.92 hit=<0|1> torch=<v> log=<path>
         or, with a nonzero exit, RESULT variant=t2-cpu-ngram status=failed torch=<v> reason=<one line>. torch= is the
         version the trainer printed (its `stack:` line), else the one installed for this Python.
  ab     the default run against one changed setting (AB_ENV, default a shorter run), interleaved and seed-paired.

Environment (all optional): NPROC (8), SEED (unset: random init), STEPS (NUM_SCHEDULED_ITERATIONS; the default is
5790 scheduled + 10 extension = 5800 steps), EXTRA_ENV ("K=V K=V", or export the variables), LOG_DIR,
DRY_RUN=1 (print the command, run nothing), RUN_TIMEOUT_S (5400), PREP_TIMEOUT_S (14400, the download),
BUILD_TIMEOUT_S (3600, the Rust build), SKIP_CHECK=1 (run without the preflight; the torch/triton check still runs),
ALLOW_STACK=1 (another torch or triton version is a WARN, and the trainer starts anyway), HOST_CHECK=0 (a host too small for the estimate is a WARN; the trainer reads it too),
INSTALL_RUST=1 (prep may install rustup), AB_RUNS (4), AB_ENV ("NUM_SCHEDULED_ITERATIONS=5200"), AB_SEEDS.

Standard library only, so that check works before anything is installed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANT = HERE.parents[1]
ROOT = VARIANT.parents[1]  # the repository root: data/ lives there
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(VARIANT))

import runlib  # noqa: E402
from track_1_short import host, memory_budget  # noqa: E402  (both standard library only)

NAME = "t2-cpu-ngram"
ENTRY = "train_gpt_medium_cpu.py"
TARGET = runlib.TARGET
SHARDS = runlib.FULL_TRAIN_SHARDS
DEFAULT_STEPS = 5790  # train_gpt_medium_cpu.py's NUM_SCHEDULED_ITERATIONS; + 10 extension steps = 5800
EXTENSION_STEPS = 10
TORCH, TRITON = "2.10.0+cu128", "3.6.0"
PINS = {"kernels": "0.16.1", "huggingface-hub": "1.29.0"}  # requirements.txt
RECORD_S = 17.35 * 60  # the official Track 2 record
RESULTS = HERE / "results"
STAMP = RESULTS / "exact_match.sha256"
GiB = 1 << 30
SHARD_GIB = 0.19


# ------------------------------------------------------------------------------------------------- helpers

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        raise SystemExit(f"{name}={os.environ[name]!r} is not an integer")


def parse_extra_env(text: str) -> dict:
    """EXTRA_ENV / AB_ENV: shell-style words, each K=V."""
    out = {}
    for word in shlex.split(text or ""):
        key, eq, value = word.partition("=")
        if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(f"expected K=V words, got {word!r}")
        out[key] = value
    return out


def fa3_pin() -> dict:
    src = (VARIANT / "track_1_short" / "model" / "attention.py").read_text()
    return {k: re.search(rf'^{k} = "([^"]+)"', src, re.M).group(1) for k in ("FA3_REPO", "FA3_REVISION", "FA3_BINARY_SHA256")}


def hf_hub_dir(env=None) -> Path:
    env = os.environ if env is None else env
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    return Path(env.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"


def fa3_cached(hub: Path | None = None, pin: dict | None = None) -> tuple[bool, str]:
    """(ok, detail): the pinned FA3 binary is in the Hugging Face cache, under the namespace kernels 0.16 reads."""
    pin, hub = pin or fa3_pin(), hub or hf_hub_dir()
    folder = pin["FA3_REPO"].replace("/", "--")
    snap = hub / f"kernels--{folder}" / "snapshots" / pin["FA3_REVISION"]
    if not snap.is_dir():
        where = "only under models-- (prep seeds the kernels-- copy)" if (hub / f"models--{folder}").exists() else "not downloaded"
        return False, f"{pin['FA3_REPO']}@{pin['FA3_REVISION'][:8]} {where} in {hub}"
    bins = sorted(snap.rglob("_flash_attn3_cuda_*.so"))
    bins = [b for b in bins if "cu128-x86_64-linux" in str(b)] or bins
    if not bins:
        return False, f"no _flash_attn3_cuda_*.so under {snap}"
    digest = hashlib.sha256(bins[0].read_bytes()).hexdigest()
    if digest != pin["FA3_BINARY_SHA256"]:
        return False, f"{bins[0]} has sha256 {digest[:12]}..., the code pins {pin['FA3_BINARY_SHA256'][:12]}..."
    return True, str(bins[0])


def find_cudart13(env=None) -> tuple[bool, str | None]:
    """(found, directory to add to LD_LIBRARY_PATH or None when the loader already finds it)."""
    env = os.environ if env is None else env
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=10).stdout
        if "libcudart.so.13" in out:
            return True, None
    except (OSError, subprocess.SubprocessError):
        pass
    on_path = [d for d in env.get("LD_LIBRARY_PATH", "").split(":") if d]
    cuda_home = env.get("CUDA_HOME") or "/usr/local/cuda"
    candidates = on_path + [f"{cuda_home}/targets/x86_64-linux/lib", f"{cuda_home}/lib64"]
    candidates += sorted(str(p) for p in Path("/usr/local").glob("cuda-13*/targets/x86_64-linux/lib"))
    for d in candidates:
        if os.path.exists(os.path.join(d, "libcudart.so.13")):
            return True, None if d in on_path else d
    return False, None


def source_digest() -> str:
    h = hashlib.sha256()
    for f in [VARIANT / "exact_match" / n for n in ("Cargo.toml", "Cargo.lock", "pyproject.toml")] + sorted((VARIANT / "exact_match" / "src").glob("*.rs")):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def extension_state(py: str = sys.executable) -> tuple[str, str]:
    """("ok" | "stale" | "missing", detail). Imported in a child, from the variant folder, as the trainer does."""
    code = "import exact_match as m; from exact_match import CELLS, SlotIndex, StreamIndex; print(m.__file__)"
    try:
        r = subprocess.run([py, "-c", code], cwd=VARIANT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return "missing", str(e)
    if r.returncode != 0:
        return "missing", (r.stderr.strip().splitlines() or ["import failed"])[-1]
    where = r.stdout.strip()
    if STAMP.exists() and STAMP.read_text().strip() != source_digest():
        return "stale", f"{where} was built from other exact_match/ sources"
    return "ok", where


def cargo_path() -> str | None:
    return shutil.which("cargo") or next((str(p) for p in [Path.home() / ".cargo" / "bin" / "cargo"] if p.exists()), None)


def data_dir(data_path: str) -> Path:
    return (VARIANT / data_path / "data" / "fineweb10B").resolve()


def shard_problems(base: Path) -> list[str]:
    """Train/val shards whose size is not what their header says (an interrupted download)."""
    bad = []
    for f in sorted(base.glob("fineweb_train_*.bin")) + sorted(base.glob("fineweb_val_*.bin")):
        try:
            with open(f, "rb") as fh:
                head = fh.read(12)
            tokens = int.from_bytes(head[8:12], "little")
            if len(head) < 12 or int.from_bytes(head[:4], "little") != 20240520 or f.stat().st_size != 1024 + 2 * tokens:
                bad.append(f.name)
        except OSError:
            bad.append(f.name)
    return bad


def gpu_query() -> list[dict] | None:
    """One dict per GPU from nvidia-smi, None when it cannot be run."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    gpus = []
    for line in r.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4 and p[1].isdigit():
            gpus.append(dict(name=p[0], total_mib=int(p[1]), used_mib=int(p[2]) if p[2].isdigit() else 0, driver=p[3]))
    return gpus


def scheduled_steps(env=None) -> int:
    env = os.environ if env is None else env
    return int(env.get("NUM_SCHEDULED_ITERATIONS") or DEFAULT_STEPS)


def retrieval_plan(base: Path, total_steps: int, threads: int, ranks: int):
    """host.plan for this run, as retrieval.host_plan computes it at startup: from the shards on disk when all
    are there, else from 100M tokens per shard."""
    exclude = memory_budget.excluded_shards(total_steps, ranks)
    files = sorted(base.glob("fineweb_train_*.bin"))
    tokens = [host.shard_tokens(f) for f in files[exclude:]] if len(files) == SHARDS else [memory_budget.SHARD_TOKENS] * (SHARDS - exclude)
    return host.plan(sum(tokens), sum(tokens[:20]), threads, ranks)


# --------------------------------------------------------------------------------------------------- check

STACK_FIX = (f"fix: bash 182/tools/node_setup.sh --mode venv from the repository root (makes .venv-run with the pinned stack), "
             f"or pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 (it brings triton {TRITON})")


def installed_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def stack_problems(relax: bool, version=None) -> list[tuple[str, str, str]]:
    """(level, what, detail) for torch and triton in this Python. Another version is a FAIL: the tree was only
    written against torch 2.10.0+cu128 / triton 3.6.0, and a node on torch 2.11 ran out of memory after training.
    ALLOW_STACK=1 makes it a WARN."""
    version = version or installed_version
    out = []
    for dist, want in (("torch", TORCH), ("triton", TRITON)):
        got = version(dist)
        if got == want:
            out.append(("PASS", dist, got))
        elif got is None:
            out.append(("FAIL", dist, f"not installed in {sys.executable} ({STACK_FIX})"))
        else:
            out.append(("WARN" if relax else "FAIL", dist,
                        f"{got} in {sys.executable}, the tree needs {want} ({STACK_FIX}; ALLOW_STACK=1 runs anyway, unsupported)"))
    return out


def run_check(env=None, out=print) -> int:
    """Print the preflight lines; returns the number of FAILs."""
    env = os.environ if env is None else env
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def say(level, what, detail):
        counts[level] += 1
        out(f"{level} {what}: {detail}")

    me = f"variants/{NAME}/182/run_node.sh"
    nproc = int(env.get("NPROC") or 8)
    relax = env.get("ALLOW_STACK") == "1"

    def check_platform():
        ok = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")
        say("PASS" if ok else "FAIL", "platform", f"{platform.system()} {platform.machine()}"
            + ("" if ok else " (needs Linux x86_64: CUDA, /dev/shm tables, CPU pinning; run this on the GPU node)"))
        v = sys.version_info
        say("PASS" if v[:2] == (3, 12) else "WARN" if v >= (3, 10) else "FAIL", "python",
            f"{platform.python_version()} at {sys.executable}" + ("" if v[:2] == (3, 12) else " (the stack was run on 3.12; set PY=/path/to/python to choose another)"))

    def check_packages():
        def version(dist):
            try:
                return importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                return None
        for level, what, detail in stack_problems(relax):
            say(level, what, detail)
        for dist, want in PINS.items():
            got = version(dist)
            if got is None:
                say("FAIL", dist, f"not installed (pip install -r variants/{NAME}/requirements.txt)")
            else:
                say("PASS" if got == want else "WARN", dist, got + ("" if got == want else f", requirements.txt pins {want}"))
        missing = [d for d in ("numpy", "tiktoken") if version(d) is None]
        say("FAIL" if missing else "PASS", "numpy, tiktoken", f"not installed: {', '.join(missing)} (pip install -r variants/{NAME}/requirements.txt)" if missing else "installed")

    def check_gpus():
        gpus = gpu_query()
        if not gpus:
            say("FAIL", "gpus", "nvidia-smi found no GPU (run this on the GPU node; in docker add --gpus all)")
            return
        names = sorted({g["name"] for g in gpus})
        detail = f"{len(gpus)} x {', '.join(names)}, {gpus[0]['total_mib']} MiB, driver {gpus[0]['driver']}"
        if len(gpus) < nproc:
            say("FAIL", "gpus", f"{detail}: NPROC={nproc} needs {nproc} (set NPROC={len(gpus)}; anything but 8 is untested)")
        elif nproc != 8 or not all("H100" in n for n in names):
            say("WARN", "gpus", f"{detail}; NPROC={nproc} (the reported time is for 8xH100; other shapes are untested)")
        else:
            say("PASS", "gpus", detail)
        major = gpus[0]["driver"].split(".")[0]
        if major.isdigit() and int(major) < 580:
            say("WARN", "driver", f"{gpus[0]['driver']} < 580 (the ANVIL2 record README measured a uniform ~35% slowdown below 580)")
        busy = [i for i, g in enumerate(gpus) if g["used_mib"] > 2048]
        if busy:
            say("WARN", "gpus in use", f"GPUs {busy} already hold > 2 GiB (another run? this one needs ~60 of 80 GiB per GPU)")
        need = sum(memory_budget.estimate(1024, 16, 4096, 2 * 64 * 1024 * 8, scheduled_steps(env) + EXTENSION_STEPS, world_size=nproc).values())
        have = min(g["total_mib"] for g in gpus) * (1 << 20)
        say("PASS" if need <= 0.9 * have else "FAIL" if need > have else "WARN", "gpu memory",
            f"estimated peak {need / GiB:.1f} GiB of {have / GiB:.1f} GiB per GPU (python -m track_1_short.memory_budget)")

    def check_cuda():
        found, add = find_cudart13(env)
        if not found:
            say("FAIL", "libcudart.so.13", "not found (the patched FA3 kernel links it). sudo apt-get install -y cuda-cudart-13-1 cuda-nvrtc-dev-13-1 "
                "cuda-cudart-dev-13-1 after adding NVIDIA's cuda-keyring, or use the repository Dockerfile (182/tools/node_setup.sh --mode docker)")
        else:
            say("PASS", "libcudart.so.13", "on the loader path" if add is None else f"in {add} (run adds it to LD_LIBRARY_PATH)")
        home = env.get("CUDA_HOME") or "/usr/local/cuda"
        ok = os.path.isfile(os.path.join(home, "include", "cuda_bf16.h"))
        say("PASS" if ok else "FAIL", "CUDA headers", f"{home}/include/cuda_bf16.h" + ("" if ok else
            " is missing: set CUDA_HOME to a CUDA toolkit with headers (the cross-entropy kernel compiles through NVRTC at startup)"))

    def check_caches():
        ok, detail = fa3_cached(hf_hub_dir(env))
        say("PASS" if ok else "FAIL", "FA3 kernel", detail + ("" if ok else f" (bash {me} prep)"))
        cache = Path(env.get("TIKTOKEN_CACHE_DIR") or env.get("DATA_GYM_CACHE_DIR") or Path(env.get("TMPDIR") or "/tmp") / "data-gym-cache")
        have = cache.is_dir() and any(cache.iterdir())
        say("PASS" if have else "WARN", "GPT-2 tokenizer file", f"cached in {cache}" if have else
            f"nothing in {cache}: every rank would download it at startup (bash {me} prep fetches it once)")

    def check_extension():
        state, detail = extension_state()
        if state == "ok":
            say("PASS", "exact_match extension", detail)
        elif state == "stale":
            say("FAIL", "exact_match extension", f"{detail} (bash {me} prep rebuilds it)")
        else:
            cargo = cargo_path()
            say("FAIL", "exact_match extension", f"not importable: {detail}. " + (f"cargo is at {cargo}: bash {me} prep builds it" if cargo else
                f"No cargo either: INSTALL_RUST=1 bash {me} prep installs rustup and builds it (or: curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal)"))

    def check_data():
        data_path = runlib.resolve_data_path(env, VARIANT)
        base = data_dir(data_path)
        n = len(list(base.glob("fineweb_train_*.bin")))
        val = bool(list(base.glob("fineweb_val_*.bin")))
        bad = shard_problems(base)
        if n != SHARDS or not val:
            say("FAIL", "data", f"{n} of {SHARDS} train shards{'' if val else ' and no validation shard'} under {base} "
                f"(bash {me} prep downloads the rest; the retrieval asserts all {SHARDS})")
        elif bad:
            say("FAIL", "data", f"{len(bad)} shard(s) with a wrong size, e.g. {bad[0]} under {base}: delete them, then bash {me} prep")
        else:
            say("PASS", "data", f"{n} train shards + validation under {base} (DATA_PATH={data_path})")
        if n < SHARDS:
            try:
                free = shutil.disk_usage(base if base.exists() else ROOT).free / GiB
                need = (SHARDS - n) * SHARD_GIB + 5
                say("PASS" if free >= need else "FAIL", "disk for data", f"{free:.0f} GiB free, the missing shards need ~{need:.0f} GiB")
            except OSError as e:
                say("WARN", "disk for data", str(e))

    def check_host():
        h = host.detect(gpus=nproc)
        threads = host.build_threads(h.per_gpu, env)
        base = data_dir(runlib.resolve_data_path(env, VARIANT))
        total = scheduled_steps(env) + EXTENSION_STEPS
        need = retrieval_plan(base, total, threads, nproc)
        soft = env.get("HOST_CHECK", "1") != "1"
        problems = host.check(h, need)
        if h.shm_total == 0:
            problems.insert(0, "no /dev/shm (the index tables are files there)")
        hint = ""
        if problems and not env.get("VAL_BUILD_THREADS") and threads > 16 and not host.check(h, retrieval_plan(base, total, 16, nproc)):
            hint = "; VAL_BUILD_THREADS=16 fits (36 GiB less arena at 64 threads, a slower index build; README: Smaller hosts)"
        summary = (f"needs ~{need.total / GiB:.0f} GiB RAM incl. {need.tables / GiB:.0f} GiB in /dev/shm ({threads} build threads); "
                   f"has {h.mem_available / GiB:.0f} GiB available, /dev/shm {h.shm_free / GiB:.0f} GiB free")
        if problems:
            fix = "sudo mount -o remount,size=200G /dev/shm (docker: --ipc=host or --shm-size=200g)" if any("/dev/shm" in p for p in problems) else "free memory or use a larger host"
            say("WARN" if soft else "FAIL", "host memory", f"{summary}. {'; '.join(problems)}. Fix: {fix}{hint}"
                + ("" if soft else "; HOST_CHECK=0 reports only (a too-small host then dies mid-build)"))
        else:
            say("PASS", "host memory", summary)
        if h.stale_tables:
            say("WARN", "stale tables", f"{h.stale_tables / GiB:.0f} GiB of /dev/shm/exact-match-* left by a killed run: rm /dev/shm/exact-match-*")
        cpu = f"{h.allowed} CPUs allowed, {h.per_gpu} per GPU over {nproc} ({h.cores} cores)"
        if h.per_gpu >= 24:
            say("PASS", "cpus", cpu)
        elif h.per_gpu >= 4:
            say("WARN", "cpus", f"{cpu}: under 24 per GPU the run takes PR #380's small-host path ({threads} build threads, unpinned "
                "corpus reads); it should finish but slower, and nobody has reported a Track 2 time on it")
        else:
            say("FAIL", "cpus", f"{cpu}: each rank needs its own cores for the loader, the online index and the index build (>= 4 per GPU)")
        if platform.system() == "Linux":
            if not h.avx512:
                say("WARN", "avx-512", "not available: the index build uses its slower AVX2 path")
            if "[never]" in h.thp:
                say("WARN", "transparent huge pages", "disabled (never): the index arenas ask for them; echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled")
            kernel = re.match(r"(\d+)\.(\d+)", h.kernel)
            if kernel and (int(kernel[1]), int(kernel[2])) < (5, 14):
                say("WARN", "kernel", f"{h.kernel} < 5.14: MADV_POPULATE_WRITE is ignored, so the tables fault in on the clock")

    def check_writable():
        for d in (VARIANT / "logs", RESULTS):
            try:
                d.mkdir(parents=True, exist_ok=True)
                probe = d / f".write-test-{os.getpid()}"
                probe.write_text("")
                probe.unlink()
            except OSError as e:
                say("FAIL", "writable", f"{d}: {e}")
                return
        say("PASS", "writable", f"{VARIANT / 'logs'} and {RESULTS}")

    for fn in (check_platform, check_packages, check_gpus, check_cuda, check_caches, check_extension, check_data, check_host, check_writable):
        try:
            fn()
        except Exception as e:  # a check that cannot run is a failed check, not a crash
            say("FAIL", fn.__name__.removeprefix("check_"), f"the check itself failed: {type(e).__name__}: {e}")
    out(f"check: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail"
        + (f" -> fix the FAIL lines (most are fixed by: bash {me} prep)" if counts["FAIL"] else f" -> ready: bash {me} run"))
    return counts["FAIL"]


# ---------------------------------------------------------------------------------------------------- prep

FA3_PREFETCH = r"""
import hashlib, os, pathlib, shutil, sys
from huggingface_hub import snapshot_download
repo, rev, want, hub = sys.argv[1:5]
hub = pathlib.Path(hub)
path = snapshot_download(repo, revision=rev)
# kernels >= 0.16 resolves through the kernels-- namespace; seed it from the model download
src, dst = hub / ("models--" + repo.replace("/", "--")), hub / ("kernels--" + repo.replace("/", "--"))
if src.exists() and not dst.exists():
    shutil.copytree(src, dst, copy_function=os.link, symlinks=True)
bins = list(pathlib.Path(path).rglob("_flash_attn3_cuda_*.so"))
bins = [b for b in bins if "cu128-x86_64-linux" in str(b)] or bins
if not bins:
    sys.exit("no _flash_attn3_cuda_*.so in the snapshot")
got = hashlib.sha256(bins[0].read_bytes()).hexdigest()
if got != want:
    sys.exit(f"FA3 sha256 {got} != pinned {want}")
print("FA3", bins[0])
"""


def step(cmd, timeout, dry, cwd=None, env=None) -> int:
    print("+ " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if dry:
        return 0
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"  timed out after {timeout}s", flush=True)
        return 124
    except OSError as e:
        print(f"  could not start: {e}", flush=True)
        return 127


def run_prep(dry: bool) -> int:
    py, fails = sys.executable, []
    env = dict(os.environ)
    # 1) the data, into the repository root's data/ (finished shards are kept, so a re-run resumes)
    base = ROOT / "data" / "fineweb10B"
    n = len(list(base.glob("fineweb_train_*.bin")))
    bad = shard_problems(base)
    for name in bad:
        print(f"removing {base / name}: its size does not match its header (an interrupted download)")
        if not dry:
            (base / name).unlink()
    if n == SHARDS and not bad and (base / "fineweb_val_000000.bin").exists():
        print(f"data: {SHARDS} train shards + validation present in {base}")
    else:
        rc = step([py, str(ROOT / "data" / "cached_fineweb10B.py"), str(SHARDS)], env_int("PREP_TIMEOUT_S", 14400), dry, cwd=ROOT)
        if rc:
            fails.append(f"data download exited {rc} (re-run to resume; PREP_TIMEOUT_S raises the limit; check Hugging Face access)")
    # 2) the Rust extension
    state, detail = ("missing", "dry run") if dry and not STAMP.exists() else extension_state(py)
    if state == "ok" and STAMP.exists():
        print(f"exact_match: built ({detail})")
    else:
        cargo = cargo_path()
        if cargo is None and os.environ.get("INSTALL_RUST") == "1":
            rc = step(["sh", "-c", "curl -sSf --max-time 600 https://sh.rustup.rs | sh -s -- -y --profile minimal"], 1200, dry)
            cargo = cargo_path() or (str(Path.home() / ".cargo" / "bin" / "cargo") if dry else None)
        if cargo is None:
            fails.append("no cargo for the exact_match build: INSTALL_RUST=1 lets prep install rustup, or install it from https://rustup.rs")
        else:
            env["PATH"] = f"{Path(cargo).parent}{os.pathsep}{env.get('PATH', '')}"
            rc = step([py, "-m", "pip", "install", "-q", "maturin>=1.9,<2"], 1800, dry, env=env)
            rc = rc or step([py, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", str(VARIANT / "exact_match")],
                            env_int("BUILD_TIMEOUT_S", 3600), dry, env=env)
            if rc:
                fails.append(f"exact_match build exited {rc}")
            elif not dry:
                state, detail = extension_state(py)
                if state == "missing":
                    fails.append(f"exact_match built but does not import: {detail}")
                else:
                    RESULTS.mkdir(parents=True, exist_ok=True)
                    STAMP.write_text(source_digest() + "\n")
    # 3) the FA3 kernel and the tokenizer file, so that no rank needs the network at startup
    pin = fa3_pin()
    if fa3_cached()[0]:
        print("FA3 kernel: cached")
    else:
        rc = step([py, "-c", FA3_PREFETCH, pin["FA3_REPO"], pin["FA3_REVISION"], pin["FA3_BINARY_SHA256"], str(hf_hub_dir())], 1800, dry)
        if rc:
            fails.append(f"FA3 prefetch exited {rc} (needs huggingface-hub from requirements.txt and network; `hf auth login` on a 401)")
    rc = step([py, "-c", "import tiktoken; tiktoken.get_encoding('gpt2'); print('GPT-2 tokenizer file cached')"], 600, dry)
    if rc:
        fails.append(f"tokenizer fetch exited {rc} (needs tiktoken from requirements.txt and network)")
    for f in fails:
        print(f"FAIL prep: {f}")
    print(f"prep: {'dry run' if dry else 'done' if not fails else f'{len(fails)} step(s) failed'}; next: bash variants/{NAME}/182/run_node.sh check")
    return 1 if fails else 0


# ----------------------------------------------------------------------------------------------------- run

def build_run(overrides: dict | None = None, environ=None) -> tuple[list, dict, dict]:
    """(command, full environment, the variables this runner set) for one run. `overrides` are an arm's settings;
    SEED, STEPS and EXTRA_ENV come from the environment."""
    environ = os.environ if environ is None else environ
    setv = {}
    if environ.get("STEPS"):
        setv["NUM_SCHEDULED_ITERATIONS"] = str(int(environ["STEPS"]))
    if environ.get("SEED"):
        setv["TRAIN_SEED"] = str(int(environ["SEED"]))
    setv.update(parse_extra_env(environ.get("EXTRA_ENV", "")))
    setv.update(overrides or {})
    env = {**environ, **setv}
    env["DATA_PATH"] = setv["DATA_PATH"] = runlib.resolve_data_path(env, VARIANT) if "DATA_PATH" not in env else env["DATA_PATH"]
    if "HF_HUB_OFFLINE" not in env and fa3_cached(hf_hub_dir(env))[0]:
        env["HF_HUB_OFFLINE"] = setv["HF_HUB_OFFLINE"] = "1"  # eight ranks read the cache instead of the hub
    found, add = find_cudart13(env)
    if found and add:
        env["LD_LIBRARY_PATH"] = setv["LD_LIBRARY_PATH"] = add + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    nproc = int(environ.get("NPROC") or 8)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", ENTRY]
    return cmd, env, setv


def shown(cmd, setv) -> str:
    return f"cd {shlex.quote(str(VARIANT))} && " + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(setv.items())) + " " + " ".join(shlex.quote(c) for c in cmd)


def execute(cmd, env, stdout_path: Path, timeout: int) -> tuple[int, str | None]:
    """Run in its own process group, stdout+stderr to the console and stdout_path. (exit code, "timeout" |
    "interrupted" | None). The whole group is killed on a timeout or a signal, so no rank outlives the run."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    why = []
    with open(stdout_path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=VARIANT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", start_new_session=True)

        def kill(reason):
            why.append(reason)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    return
                try:
                    proc.wait(20)
                    return
                except subprocess.TimeoutExpired:
                    pass

        timer = threading.Timer(timeout, kill, ("timeout",))
        timer.daemon = True
        timer.start()
        old = {s: signal.signal(s, lambda *_: threading.Thread(target=kill, args=("interrupted",), daemon=True).start())
               for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
            rc = proc.wait()
        finally:
            timer.cancel()
            for s, h in old.items():
                signal.signal(s, h)
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # stragglers of a crashed launcher
            except (ProcessLookupError, PermissionError):
                pass
    return rc, (why[0] if why else None)


STACK_LINE = re.compile(r"^stack: torch=(\S+)", re.M)


def torch_version(text: str = "") -> str:
    """The torch version of a run: the trainer's `stack:` line when the output has one, else what this Python has."""
    m = STACK_LINE.search(text)
    return m.group(1) if m else (installed_version("torch") or "none")


def result_line(result: dict | None, log: Path | str | None, reason: str | None = None, torch: str | None = None) -> str:
    torch = torch or torch_version()
    if result is None or reason:
        return f"RESULT variant={NAME} status=failed torch={torch} reason={' '.join((reason or 'no final validation line').split())[:300]}"
    return (f"RESULT variant={NAME} val_loss={result['val_loss']:.6f} train_time_s={result['train_time_ms'] / 1000:.3f} "
            f"steps={result['steps']} target={TARGET} hit={int(result['val_loss'] <= TARGET)} torch={torch} log={log}")


def summarize(result: dict, details: dict) -> list[str]:
    """What one run says, for the console and summary.txt (every number is this run's, n = 1)."""
    loss, secs = result["val_loss"], result["train_time_ms"] / 1000
    lines = [f"run summary ({NAME}, n=1, seed {result.get('seed')}):",
             f"  val loss {loss:.4f} at step {result['steps']}: {abs(TARGET - loss) * 1000:.1f} millinats {'under' if loss <= TARGET else 'OVER'} the {TARGET} target",
             f"  train time {secs:.1f} s ({result['train_time_ms'] / result['steps']:.2f} ms/step); the official Track 2 record is {RECORD_S:.0f} s"]
    if "model_only" in details:
        lines.append(f"  model alone {details['model_only']:.4f}, the exact-count chain takes off {details['model_only'] - loss:.4f}")
    if "peak_alloc_mib" in details:
        total = details.get("gpu_total_mib")
        lines.append(f"  GPU peak {details['peak_alloc_mib'] / 1024:.1f} GiB allocated, {details['peak_reserved_mib'] / 1024:.1f} GiB reserved"
                     + (f" of {total / 1024:.1f} GiB ({(total - details['peak_reserved_mib']) / 1024:.1f} GiB headroom)" if total else ""))
    if "host_used_gib" in details:
        lines.append(f"  host: rank 0 peak RSS {details['rank0_rss_gib']:.1f} GiB, node in use at the end {details['host_used_gib']:.0f} GiB")
    if details.get("index_build_s"):
        lines.append("  validation index builds (2-, 3-, 6-token): " + ", ".join(f"{t:.1f} s" for t in details["index_build_s"]))
    if loss <= TARGET:
        lines.append(f"  next: a shorter run gives the slope of loss against steps (bash variants/{NAME}/182/run_node.sh ab)")
    return lines


def run_once(overrides: dict | None, log_dir: Path, dry: bool = False) -> tuple[dict | None, str]:
    """One run. Returns (parsed result or None, the RESULT line); the run's files are in log_dir."""
    cmd, env, setv = build_run(overrides)
    print(f"+ {shown(cmd, setv)}  > {log_dir / 'stdout.txt'}", flush=True)
    if dry:
        return None, ""
    timeout = env_int("RUN_TIMEOUT_S", 5400)
    t0 = time.time()
    rc, why = execute(cmd, env, log_dir / "stdout.txt", timeout)
    text = (log_dir / "stdout.txt").read_text(errors="replace")
    result = runlib.parse_result(text)
    log = log_dir / "stdout.txt"
    if (m := runlib.LOG_PATH.search(text)) and (VARIANT / m.group(1)).exists():
        log = log_dir / Path(m.group(1)).name  # the run log: the source, the environment and every step line
        shutil.copyfile(VARIANT / m.group(1), log)
    if rc != 0 or result is None or why:
        reason = {"timeout": f"timeout after {timeout}s (RUN_TIMEOUT_S)", "interrupted": "interrupted"}.get(why) or runlib.failure_reason(text, rc)
        return None, result_line(None, log, f"{reason}; see {log_dir / 'stdout.txt'}", torch=torch_version(text))
    details = runlib.parse_details(text)
    gpus = gpu_query()
    if gpus:
        details["gpu_total_mib"] = min(g["total_mib"] for g in gpus)
    summary = summarize(result, details)
    (log_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)
    knobs = {k: v for k, v in setv.items() if k not in ("DATA_PATH", "HF_HUB_OFFLINE", "LD_LIBRARY_PATH", "TRAIN_SEED", "NUM_SCHEDULED_ITERATIONS")}
    row = dict(result, scheduled_iterations=result["steps"] - EXTENSION_STEPS, out=str(log), wall_s=round(time.time() - t0), **details)
    if knobs:
        row["knobs"] = knobs  # sweep_steps.py fits only runs without other settings changed
    runlib.append_jsonl(RESULTS / "sweep.jsonl", row)
    return row, result_line(result, log, torch=torch_version(text))


def default_log_dir(tag: str) -> Path:
    return Path(os.environ["LOG_DIR"]).resolve() if os.environ.get("LOG_DIR") else RESULTS / "node" / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"


def preflight(dry: bool) -> str | None:
    """The check before a run; the reason not to start, or None."""
    if dry:
        return None
    if os.environ.get("SKIP_CHECK") == "1":
        # Still the stack: a wrong torch only shows up after a 20-minute run (torch 2.11 ran out of memory after
        # training). ALLOW_STACK=1 skips this too.
        bad = [p for p in stack_problems(os.environ.get("ALLOW_STACK") == "1") if p[0] != "PASS"]
        for level, what, detail in bad:
            print(f"{level} {what}: {detail}")
        fails = sum(level == "FAIL" for level, _, _ in bad)
        return f"preflight (SKIP_CHECK=1, stack only): {fails} FAIL line(s) above" if fails else None
    fails = run_check()
    return f"preflight: {fails} FAIL line(s) above (SKIP_CHECK=1 runs anyway)" if fails else None


def mode_run(dry: bool) -> int:
    reason = preflight(dry)
    if reason:
        print(result_line(None, None, reason))
        return 1
    result, line = run_once(None, default_log_dir("run"), dry)
    if dry:
        return 0
    print(line, flush=True)
    return 0 if result else 1


# ------------------------------------------------------------------------------------------------------ ab

def ab_report(pairs: list[tuple[dict, dict]], arm: dict) -> list[str]:
    """The paired comparison of finished (base, alt) runs."""
    label = " ".join(f"{k}={v}" for k, v in arm.items())
    lines = [f"A/B: default vs {label}, {len(pairs)} seed-paired run(s)", f"{'seed':>6} {'base loss':>10} {'alt loss':>10} {'d loss':>9} {'base s':>8} {'alt s':>8} {'d s':>8}"]
    dl, dt = [], []
    for a, b in pairs:
        dl.append(b["val_loss"] - a["val_loss"])
        dt.append((b["train_time_ms"] - a["train_time_ms"]) / 1000)
        lines.append(f"{str(a.get('seed')):>6} {a['val_loss']:>10.4f} {b['val_loss']:>10.4f} {dl[-1]:>+9.4f} {a['train_time_ms'] / 1000:>8.1f} {b['train_time_ms'] / 1000:>8.1f} {dt[-1]:>+8.1f}")
    n = len(pairs)
    mean = lambda xs: sum(xs) / len(xs)
    alt_losses = [b["val_loss"] for _, b in pairs]
    lines.append(f"mean d loss {mean(dl):+.4f}, mean d time {mean(dt):+.1f} s; the alt arm is at or under {TARGET} in {sum(x <= TARGET for x in alt_losses)}/{n} runs")
    if n >= 2:
        _, sd = runlib.mean_std(dl)
        lines.append(f"paired std of d loss {sd:.4f} (n={n}); p(alt mean < {TARGET}) = {runlib.one_sided_p(alt_losses)[1]:.4f} "
                     f"(a record needs p < 0.01: significance.py)" if len(set(alt_losses)) > 1 else f"paired std of d loss {sd:.4f} (n={n})")
    sa, sb = pairs[0][0]["steps"], pairs[0][1]["steps"]
    if sa != sb and mean(dl) > 0:
        import math
        slope = mean(dl) / math.log(sb / sa)  # val_loss = a + b ln(steps), through the two arms' means
        base = mean([a["val_loss"] for a, _ in pairs])
        fit = sa * math.exp((TARGET - 0.002 - base) / slope)
        lines.append(f"two-point fit: {slope:+.4f} loss per ln(steps) -> mean {TARGET - 0.002:.3f} near {fit:.0f} total steps "
                     f"(NUM_SCHEDULED_ITERATIONS={math.ceil(fit) - EXTENSION_STEPS}); refine with sweep_steps.py, which pools these runs")
    lines.append(f"AB variant={NAME} arm={label.replace(' ', ',')} n={n} base_val_loss={mean([a['val_loss'] for a, _ in pairs]):.6f} "
                 f"alt_val_loss={mean(alt_losses):.6f} d_val_loss={mean(dl):+.6f} base_train_time_s={mean([a['train_time_ms'] for a, _ in pairs]) / 1000:.3f} "
                 f"alt_train_time_s={mean([b['train_time_ms'] for _, b in pairs]) / 1000:.3f} alt_hit={sum(x <= TARGET for x in alt_losses)}/{n}")
    return lines


def mode_ab(dry: bool) -> int:
    arm = parse_extra_env(os.environ.get("AB_ENV", "NUM_SCHEDULED_ITERATIONS=5200"))
    if not arm:
        raise SystemExit("AB_ENV is empty: nothing to compare")
    n = env_int("AB_RUNS", 4)
    seeds = [int(s) for s in os.environ.get("AB_SEEDS", "").replace(",", " ").split()] or list(range(n))
    if len(seeds) < n:
        raise SystemExit(f"AB_SEEDS has {len(seeds)} seeds for AB_RUNS={n}")
    reason = preflight(dry)
    if reason:
        print(result_line(None, None, reason))
        return 1
    root, pairs, failures = default_log_dir("ab"), [], 0
    for i, seed in enumerate(seeds[:n]):
        order = [("base", {}), ("alt", arm)] if i % 2 == 0 else [("alt", arm), ("base", {})]  # interleaved: drift hits both arms
        got = {}
        for name, overrides in order:
            row, line = run_once({**overrides, "TRAIN_SEED": str(seed)}, root / f"pair{i}_seed{seed}_{name}", dry)
            if not dry:
                print(line, flush=True)
                if row is None:
                    failures += 1
                    break
                got[name] = row
        if len(got) == 2:
            pairs.append((got["base"], got["alt"]))
        if failures >= 2:
            print("A/B stopped: 2 failed runs")
            break
    if dry:
        return 0
    if not pairs:
        print(f"AB variant={NAME} status=failed reason=no complete pair; see {root}")
        return 1
    report = ab_report(pairs, arm)
    (root / "ab.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)
    return 0


def main(argv) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    dry = os.environ.get("DRY_RUN") == "1"
    if mode == "check":
        return 1 if run_check() else 0
    if mode == "prep":
        return run_prep(dry)
    if mode == "run":
        return mode_run(dry)
    if mode == "ab":
        return mode_ab(dry)
    print(__doc__.split("\n\nEnvironment")[0], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
