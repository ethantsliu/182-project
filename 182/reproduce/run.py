"""CPU-only replay of committed analysis; outputs stay under reproduce/output.

Run from any working directory: python /path/to/182-project/182/reproduce/run.py
No model training, downloads, GPU, shell utilities, or LaTeX are required.
"""
from pathlib import Path
import contextlib
import csv
import hashlib
import html
import importlib.util
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "output"
FIGURES = {
    "fig_maps.png": "Six scalar maps: algebraic illustration, not an experiment.",
    "fig_bench.png": "One-H100 microbenchmark: median; whiskers are 10th–90th percentiles over 300 repetitions, not confidence intervals.",
    "fig_steploop.png": "First-session unpaired training clock decomposition, seeds 1–3 per arm: mean ± sample SD; dots are runs. Timing excludes compilation and warm-up. Supplementary asset not included in current main.tex.",
    "fig_profile.png": "Early rank-0 profile from archived analysis.txt; overlapping streams are not additive.",
    "fig_paired.png": "Ten same-seed, same-session comparisons. Positive means faster. Bars show mean and sample SD, not SE or a confidence interval. Skip-matrix is a timing-only ablation, not a usable optimizer.",
    "fig_curve.png": "Three seeds (5–7), mean ± sample SD. Training objective is rank-0 loss divided by tokens, includes changing auxiliary terms, and is not validation cross-entropy. Training y-axis is log; validation/differences are linear.",
    "fig_sweep.png": "Shape sweep: measured speedup over the record. Left x-axis is log base 2; horizontal line is parity.",
    "fig_profile_late.png": "Late rank-0 profile from archived analysis.txt; kernel categories can overlap across streams.",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rebuild():
    import numpy
    import matplotlib
    matplotlib.use("Agg")
    results = ROOT / "182/results"
    paper = ROOT / "182/paper"
    if not (results / "raw").is_dir():
        raise FileNotFoundError("Clone the complete repository: 182/results/raw is required.")
    for directory in (OUTPUT, OUTPUT / "paper", OUTPUT / "results/figures"):
        directory.mkdir(parents=True, exist_ok=True)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        tables = load("make_tables", results / "make_tables.py")
        # Inputs are absolute and remain unchanged; redirect only output locations.
        tables.HERE = str(OUTPUT / "results")
        tables.FIG = str(OUTPUT / "results/figures")
        rows = tables.collect_runs()
        with (OUTPUT / "results/runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tables.COLS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in tables.COLS})
        table_text, pairs = tables.build_tables(rows)
        (OUTPUT / "results/tables.md").write_text(table_text, encoding="utf-8")
        tables.figures(rows, pairs)
        figures = load("project_paper_figures", paper / "make_figs.py")
        figures.HERE = OUTPUT / "paper"
        # Matplotlib bundles DejaVu Sans; avoid machine-dependent missing-font warnings.
        figures.plt.rcParams["font.family"] = ["DejaVu Sans"]
        for name in FIGURES:
            getattr(figures, Path(name).stem)()
        figures.losses()
    (OUTPUT / "analysis.log").write_text(capture.getvalue(), encoding="utf-8")
    expected = [OUTPUT / "paper" / name for name in FIGURES]
    expected += [OUTPUT / "results/runs.csv", OUTPUT / "results/tables.md"]
    for path in expected:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected output missing: {path}")
    source_paths = sorted(p for p in (results / "raw").rglob("*") if p.is_file())
    source_paths += [results / "make_tables.py", results / "README.md", paper / "make_figs.py", HERE / "run.py"]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable (downloaded source archive)"
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Analysis rerun from archived logs; no new training or benchmark measurements.",
        "git_head": commit,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "figure_font": "DejaVu Sans (portable bundled font)",
        "inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in source_paths},
        "outputs_sha256": {str(p.relative_to(OUTPUT)).replace("\\", "/"): digest(p) for p in expected},
        "limitations": ["Paper LaTeX tables are manually populated, not rewritten here.", "Restart-interval appendix table requires its separate CPU emulation; not rebuilt by default.", "Checkpoint availability and full GPU training are not verified by this replay."],
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    gallery = "".join(f'<figure><img src="paper/{name}" alt="{html.escape(name)}"><figcaption>{html.escape(caption)}</figcaption></figure>' for name, caption in FIGURES.items())
    page = '<!doctype html><meta charset="utf-8"><title>EECS 182: archived experiment replay</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#16283b}img{max-width:100%}figure{margin:35px 0}figcaption{line-height:1.5}pre{overflow:auto;background:#f1f4f7;padding:16px;font-size:13px}a{color:#1453a0}</style>'
    page += '<h1>EECS 182: archived experiment replay</h1><p>CPU analysis of committed results; no new training runs. Read the captions for the meaning of error bars and ablations.</p><p><a href="results/tables.md">Generated tables</a> · <a href="results/runs.csv">Run-level CSV</a> · <a href="provenance.json">Input hashes and environment</a> · <a href="analysis.log">Analysis log</a></p>'
    page += '<p><strong>Coverage limits:</strong> paper tables still need comparison against these generated values; the restart-interval emulation table is not rebuilt. Checkpoints/full GPU training are outside this replay.</p>'
    page += gallery + '<h2>Generated table source</h2><pre>' + html.escape((OUTPUT / "results/tables.md").read_text(encoding="utf-8")) + '</pre>'
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    with (OUTPUT / "results/runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    print(f"Rebuilt {len(FIGURES)} paper figures and analysis tables from {len(rows)} archived run records.")
    print(f"Open {OUTPUT / 'index.html'}")
    return OUTPUT


if __name__ == "__main__":
    rebuild()
