"""The host the exact-match retrieval runs on: what it has (CPUs, cores, NUMA nodes, RAM, /dev/shm), what the
retrieval's host memory needs on it, and a report of both for the run log, so a submission can disclose the CPU and
RAM it used. Standard library only: 182/t1-cpu-ngram/node_check.py imports it on a box without torch.

The memory model mirrors exact_match/src (slot.rs: SLOTS, PBITS, nbits, BLOCK; stream.rs: the online index); the
build reports its real arena and table sizes after it allocates them (SlotIndex.memory), which the report prints.
"""
import glob
import os
import platform
import resource
from dataclasses import dataclass

GiB = 1 << 30
# exact_match/src/slot.rs
SLOTS, PBITS, BLOCK, MIN = 48, 14, 2048, 6
NBITS = {6: 14, 3: 12, 2: 12}
# A rank's host memory besides the retrieval (Python, torch, CUDA context, pinned staging, ExactCounts' pinned
# corpus buffers, the online index): a generous allowance, not measured (HOST_RANK_GIB overrides it).
RANK_GIB = float(os.environ.get("HOST_RANK_GIB", "12"))
# HOST_CHECK (default 1): fail before the clock when the estimate does not fit; 0: report only
HOST_CHECK = os.environ.get("HOST_CHECK", "1") == "1"


def _read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _cpulist(text):
    return {c for r in text.split(",") if r for a, _, b in [r.partition("-")] for c in range(int(a), int(b or a) + 1)}


@dataclass
class Host:
    cpu_model: str
    allowed: int          # CPUs this process may run on
    cores: int            # physical cores among them
    numa_nodes: int
    mem_total: int        # bytes
    mem_available: int
    shm_total: int        # /dev/shm (where the index tables live)
    shm_free: int
    thp: str              # transparent huge pages mode
    avx512: bool
    kernel: str
    gpus: int
    cgroup_limit: int = 0  # this process's memory cgroup limit (0: none); mem_available is clamped to its headroom

    @property
    def per_gpu(self):
        return self.allowed // max(1, self.gpus)


def cgroup_memory(root="/sys/fs/cgroup", self_cgroup="/proc/self/cgroup"):
    """(limit, headroom) bytes of this process's memory cgroup (v2, else v1), or None if it has no limit. Headroom is
    the limit less the cgroup's usage without its reclaimable page cache (/dev/shm pages, shmem, count as used: a
    container's tmpfs tables are charged to it). /proc/meminfo shows the host's memory, not a container's limit."""
    lines = _read(self_cgroup).splitlines()
    v2 = [l.split(":", 2)[2] for l in lines if l.startswith("0::")]
    v1 = [l.split(":", 2)[2] for l in lines if len(l.split(":", 2)) == 3 and "memory" in l.split(":", 2)[1].split(",")]
    candidates = [(os.path.join(root, p.lstrip("/")), "memory.max", "memory.current", "file", "shmem") for p in v2 + ["/"]]
    candidates += [(os.path.join(root, "memory", p.lstrip("/")), "memory.limit_in_bytes", "memory.usage_in_bytes", "total_cache", "total_shmem")
                   for p in v1 + ["/"]]
    best = None
    for d, lim, cur, cache, shmem in candidates:
        text = _read(os.path.join(d, lim))
        if not text.isdigit() or int(text) >= 1 << 60:  # absent, "max", or v1's "unlimited" (a page-rounded 2^63)
            continue
        stat = dict((k, int(v)) for k, _, v in (l.partition(" ") for l in _read(os.path.join(d, "memory.stat")).splitlines()) if v.isdigit())
        used = int(_read(os.path.join(d, cur), "0") or 0) - max(0, stat.get(cache, stat.get(cache.replace("total_", ""), 0))
                                                              - stat.get(shmem, stat.get(shmem.replace("total_", ""), 0)))
        cg = (int(text), max(0, int(text) - max(0, used)))
        best = cg if best is None or cg[1] < best[1] else best
    return best


def detect(gpus=None) -> Host:
    allowed = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    siblings = {frozenset(_cpulist(_read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list", str(c)))) & allowed
                for c in allowed}
    meminfo = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        meminfo[k] = int(v.split()[0]) * 1024 if v.split() else 0
    if not meminfo:  # not Linux: total memory only
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            total = 0
        meminfo = {"MemTotal": total, "MemAvailable": total}
    try:
        st = os.statvfs("/dev/shm")
        shm_total, shm_free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
    except OSError:
        shm_total = shm_free = 0
    model = next((l.split(":", 1)[1].strip() for l in _read("/proc/cpuinfo").splitlines() if l.startswith("model name")), platform.processor())
    flags = next((l for l in _read("/proc/cpuinfo").splitlines() if l.startswith("flags")), "")
    thp = _read("/sys/kernel/mm/transparent_hugepage/enabled", "n/a")
    cg = cgroup_memory()
    if cg is not None:
        meminfo["MemAvailable"] = min(meminfo.get("MemAvailable", cg[1]), cg[1])
    if gpus is None:
        gpus = int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or len(glob.glob("/proc/driver/nvidia/gpus/*")) or 1
    return Host(cpu_model=model, allowed=len(allowed), cores=len(siblings), numa_nodes=len(glob.glob("/sys/devices/system/node/node[0-9]*")) or 1,
                mem_total=meminfo.get("MemTotal", 0), mem_available=meminfo.get("MemAvailable", 0), shm_total=shm_total, shm_free=shm_free,
                thp=thp, avx512=all(f" {f}" in flags for f in ("avx512f", "avx512bw", "avx512dq", "avx512vl")), kernel=platform.release(), gpus=gpus,
                cgroup_limit=cg[0] if cg is not None else 0)


def shard_tokens(path):
    return max(0, (os.path.getsize(path) - 1024) // 2)


def slot_bytes(tokens, threads, key, shared=False):
    """(scan arena, table, per-thread buffers) bytes of one SlotIndex over `tokens` training tokens (slot.rs create)."""
    full = -(-max(0, tokens) // BLOCK)
    arena = 0 if shared else (full * 51 // 50 + threads * (1 << PBITS) + 64) * BLOCK * 8
    table = (SLOTS << (PBITS + NBITS[key])) * 8
    buffers = threads * ((1 << PBITS) * 9 + ((SLOTS << NBITS[key]) + 8) * 8 + (1 << NBITS[key]))
    return arena, table, buffers


@dataclass
class Plan:
    """The retrieval's host memory on this node (bytes)."""
    arenas: int       # rank 0's scan arenas (private, faulted in before the clock)
    tables: int       # the index tables (files in /dev/shm)
    buffers: int
    ranks: int        # every rank's other host memory (RANK_GIB each)

    @property
    def total(self):
        return self.arenas + self.tables + self.buffers + self.ranks


def plan(index_tokens, lower_tokens, threads, ranks, shared=False) -> Plan:
    """index_tokens: the 6-token index's training tokens; lower_tokens: each of the 3- and 2-token indexes'."""
    parts = [slot_bytes(index_tokens, threads, 6)] + [slot_bytes(lower_tokens, threads, k, shared) for k in (3, 2)]
    return Plan(arenas=sum(p[0] for p in parts), tables=sum(p[1] for p in parts), buffers=sum(p[2] for p in parts),
                ranks=int(ranks * RANK_GIB * GiB))


def check(host: Host, need: Plan):
    """The problems (empty: none) of running `need` on `host`: the tables must fit /dev/shm's free space, and
    everything (tables included: /dev/shm is RAM) the memory available."""
    problems = []
    if host.shm_total and need.tables > host.shm_free:
        problems.append(f"/dev/shm has {host.shm_free / GiB:.0f} GiB free, the index tables need {need.tables / GiB:.0f} GiB "
                        f"(remount it larger, or remove stale /dev/shm/exact-match-* tables)")
    if need.total > host.mem_available:
        problems.append(f"{host.mem_available / GiB:.0f} GiB of RAM available, the retrieval needs about {need.total / GiB:.0f} GiB "
                        f"(arenas {need.arenas / GiB:.0f} + tables {need.tables / GiB:.0f} + ranks {need.ranks / GiB:.0f}); "
                        f"SLOT_SHARED_ARENA=1 saves the 3-/2-token indexes' arenas")
    return problems


def describe(host: Host):
    return (f"host: {host.cpu_model} | {host.allowed} CPUs allowed ({host.cores} cores, {host.numa_nodes} NUMA nodes, "
            f"{host.per_gpu} per GPU over {host.gpus}) | AVX-512 {'yes' if host.avx512 else 'no'} | RAM {host.mem_total / GiB:.0f} GiB "
            f"({host.mem_available / GiB:.0f} available{f', cgroup limit {host.cgroup_limit / GiB:.0f} GiB' if host.cgroup_limit else ''}) | /dev/shm {host.shm_total / GiB:.0f} GiB ({host.shm_free / GiB:.0f} free) | "
            f"THP {host.thp} | kernel {host.kernel}")


def usage():
    """This process's peak RSS and the node's memory in use now (incl. /dev/shm), GiB, for the end-of-run report."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    now = detect(gpus=1)
    total = now.cgroup_limit or now.mem_total  # in a memory cgroup: its usage (mem_available is its headroom there)
    return peak / GiB, (total - now.mem_available) / GiB, (now.shm_total - now.shm_free) / GiB
