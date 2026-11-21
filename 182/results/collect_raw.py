#!/usr/bin/env python3
"""Copy the small text logs from the laptop's ~/.lor-182 into 182/results/raw/, redacted.

    python3 182/results/collect_raw.py [SRC]        # SRC defaults to ~/.lor-182

Only the machine that submitted the Lor jobs has SRC. Everyone else uses raw/ as committed.
What is copied, what is skipped and why is written to raw/MANIFEST.txt.

Rules
- Text only. A file over 200 kB is skipped: those are the trainers' own logs, which start with a ~13,000-line
  dump of the trainer source; the same result lines are in the small console/stdout file next to each one.
- `trainer/` folders are byte-for-byte duplicates of the `*.log.txt` files, `out/session.log` of
  `out/session/session.log`: skipped.
- A job's log.txt keeps everything except the base64 tarball between LOR-RESULTS-BEGIN/END (that tarball is
  the `out/` folder, already unpacked).
- A job's final.json is reduced to a whitelist of fields (job.json). The submit-side spec files hold account
  ids and are not copied. Key files are never read.
- Every copied file is passed through the redaction patterns below.
"""
import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.lor-182")
DST = os.path.join(HERE, "raw")
MAX_BYTES = 200_000

REDACT = [
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("api-key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[bap]-[A-Za-z0-9-]{10,})")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
JOB_FIELDS = ["name", "state", "release_state", "created_at", "observed_at", "max_runtime_seconds",
              "preemption_count", "resources", "spend"]
SKIP_JOB_TOP = {"create.err", "create.json", "spec.json", "pod.yaml", "resources.json", "job_id", "log_now.txt",
                "final.json"}

copied, skipped, hits = [], [], {}


def redact(text, rel):
    for name, pat in REDACT:
        text, n = pat.subn(f"<redacted-{name}>", text)
        if n:
            hits[(rel, name)] = hits.get((rel, name), 0) + n
    return text


def put(rel, text):
    text = redact(text, rel)
    path = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    copied.append((rel, len(text.encode())))


def read_text(path):
    with open(path, "rb") as f:
        data = f.read()
    if b"\0" in data:
        return None
    return data.decode("utf-8", "replace")


def copy_file(src, rel):
    size = os.path.getsize(src)
    if size > MAX_BYTES:
        skipped.append((rel, size, "over 200 kB (trainer log with a source dump; see the console file beside it)"))
        return
    text = read_text(src)
    if text is None:
        skipped.append((rel, size, "binary"))
        return
    put(rel, text)


def job_log(src, rel):
    out, inside, dropped = [], False, 0
    with open(src, errors="replace") as f:
        for line in f:
            line = ANSI.sub("", line)
            if "LOR-RESULTS-BEGIN" in line:
                inside = True
                out.append(line)
                continue
            if "LOR-RESULTS-END" in line:
                inside = False
                out.append(f"[collect_raw.py: {dropped} base64 lines removed here; they are the out/ folder]\n")
            if inside:
                dropped += 1
            else:
                out.append(line)
    put(rel, "".join(out))


def main():
    if not os.path.isdir(SRC):
        sys.exit(f"{SRC} not found: only the submitting laptop has it. raw/ in the repo is the archive.")
    b = os.path.join(SRC, "runs", "bringup-1119")
    for name in sorted(os.listdir(b)):
        if name.endswith(".tgz"):
            skipped.append((f"bringup-1119/{name}", os.path.getsize(os.path.join(b, name)), "archive of the files beside it"))
            continue
        copy_file(os.path.join(b, name), f"bringup-1119/{name}")
    jobs = os.path.join(SRC, "jobs")
    for job in sorted(os.listdir(jobs)):
        jd = os.path.join(jobs, job)
        if not os.path.isdir(jd):
            continue
        for name in sorted(os.listdir(jd)):
            p = os.path.join(jd, name)
            if name == "log.txt":
                job_log(p, f"jobs/{job}/log.txt")
            elif name == "final.json":
                d = json.load(open(p))
                put(f"jobs/{job}/job.json", json.dumps({k: d[k] for k in JOB_FIELDS if k in d}, indent=1) + "\n")
            elif name in SKIP_JOB_TOP:
                skipped.append((f"jobs/{job}/{name}", os.path.getsize(p), "submit-side file (account ids) or a partial copy of log.txt"))
        for root, dirs, files in os.walk(os.path.join(jd, "out")):
            dirs.sort()
            for name in sorted(files):
                p = os.path.join(root, name)
                rel = f"jobs/{job}/" + os.path.relpath(p, jd)
                if os.path.basename(root) == "trainer":
                    skipped.append((rel, os.path.getsize(p), "duplicate of the *.log.txt / *.txt one folder up"))
                elif rel == f"jobs/{job}/out/session.log":
                    skipped.append((rel, os.path.getsize(p), "duplicate of out/session/session.log"))
                else:
                    copy_file(p, rel)
    total = sum(n for _, n in copied)
    with open(os.path.join(DST, "MANIFEST.txt"), "w") as f:
        f.write(f"raw/ built by 182/results/collect_raw.py: {len(copied)} files, {total} bytes\n\n")
        f.write("REDACTIONS (file, pattern, count)\n")
        for (rel, name), n in sorted(hits.items()):
            f.write(f"  {rel}  {name}  {n}\n")
        if not hits:
            f.write("  none\n")
        f.write("\nCOPIED (bytes, sha256[:12], path)\n")
        for rel, n in copied:
            h = hashlib.sha256(open(os.path.join(DST, rel), "rb").read()).hexdigest()[:12]
            f.write(f"  {n:8d}  {h}  {rel}\n")
        f.write("\nSKIPPED (bytes, path, reason)\n")
        for rel, n, why in skipped:
            f.write(f"  {n:8d}  {rel}  ({why})\n")
    print(f"copied {len(copied)} files, {total / 1e6:.2f} MB; skipped {len(skipped)}; "
          f"redactions: {sum(hits.values())} in {len({r for r, _ in hits})} file(s) (values not shown; see raw/MANIFEST.txt)")


if __name__ == "__main__":
    main()
