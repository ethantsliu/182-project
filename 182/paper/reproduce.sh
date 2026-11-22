#!/usr/bin/env bash
# Regenerate every figure and table of the report from the archived training results, then build the PDF.
# No GPU is needed: the inputs are the logs in 182/results/raw/.
#   bash 182/paper/reproduce.sh
# Needs: python3 with matplotlib and numpy (set PY to use another interpreter, e.g. PY="arch -arm64 python3");
# latexmk with pdflatex for the last step (skipped if missing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python3}
( cd "$HERE/../results" && $PY make_tables.py )     # runs.csv, tables.md, results/figures/*.png
( cd "$HERE" && $PY make_figs.py )                   # the report's fig_*.png
if command -v latexmk >/dev/null; then ( cd "$HERE" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null && echo "built $HERE/main.pdf" ); else echo "latexmk not found: figures and tables regenerated, PDF not rebuilt"; fi
