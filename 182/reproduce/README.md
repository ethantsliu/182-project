# Start here: reproduce the evidence on a laptop

Open **[review.ipynb](review.ipynb)** to read the saved, genuinely executed results and all eight paper figures. GitHub/Jupyter can display these without a GPU. The notebook calls the same parsers and plotting functions as the report, using the committed raw logs.

## Run it

From the repository root, using Python 3.10 or newer:

```sh
python -m pip install -r 182/reproduce/requirements.txt
python -X utf8 182/reproduce/run.py
```

Open `182/reproduce/output/index.html` in your browser. It includes all eight figures with captions, the generated table source, and links to the run CSV and provenance manifest. This command works on Windows, macOS and Linux; neither Bash nor LaTeX is required. It does not install the GPU training dependencies.

For the notebook version:

```sh
python -m notebook 182/reproduce/review.ipynb
```

Choose **Kernel → Restart Kernel and Run All Cells**. Run All rebuilds the results; it does not train a model. The notebook locates the repo from either its own folder or the repository root. No hardcoded personal paths, cloud credentials, or dataset downloads are needed. Setup is a one-time install; the rerun is one command/Run All.

All generated files go under ignored `output/`; original archived logs, saved figures, and report text are left alone. The saved notebook outputs are intentionally checked in as the readable preview. Re-running it updates those outputs only if you save the notebook.

## What is reproduced

- All eight `182/paper/fig_*.png` figure assets (seven included in current `main.tex`; `fig_steploop` is supplementary), including training/validation curves, paired step-time savings, microbenchmarks, shape sweep and two profiles.
- `runs.csv`, `tables.md`, and the existing results-analysis figures, from the actual archived logs.
- Input SHA-256 hashes, current checkout commit, Python/package versions, output hashes, and analysis stdout (`output/provenance.json`, `output/analysis.log`). The commit alone does not describe uncommitted edits; the input hashes cover the scripts actually used.

The portable replay uses Matplotlib's bundled DejaVu Sans font, so pixels can differ from figures rendered with Helvetica/Arial. Numerical reproduction matters; bit-identical rendering across font/library versions is not promised. Python summation/rounding can also change the last displayed digit of a summary mean; the 89 parsed run records matched the archived CSV in the local check. Tested here with Python 3.14.0, NumPy 2.3.5, Matplotlib 3.10.7, nbformat 5.10.4 and nbclient 0.10.2; the requirements specify compatible ranges rather than a GPU environment lock.

## What still needs a separate step

1. **Restart-interval appendix table:** run `python -X utf8 182/optim-kernel/restart_interval.py` with compatible CPU PyTorch and Python 3.10-3.13 installed. The archived output is [restart_interval.md](../optim-kernel/restart_interval.md). A fresh attempt on this host was blocked by the PyTorch `torch.compile` Python 3.14 limitation; no fresh emulation result is claimed. This is a real numerical emulation on synthetic test matrices (not new training evidence), and is excluded from default Run All to keep setup small. It prints per-seed values that must be averaged/formatted for the report table. CPU PyTorch is not a default notebook dependency.
2. **Handwritten LaTeX tables/hyperparameters:** compare `output/results/tables.md` against `182/paper/main.tex`; this runner does not replace manually typed paper values or verify their rationale.
3. **Checkpoints and new training:** obtain the real files/access links from the teammate who ran the jobs. See `182/paper/README.md` for the 8×H100 rerun commands. This notebook proves archived analysis can run; it does not prove a fresh GPU training reproduction or checkpoint availability.
4. **Paper PDF:** the existing `182/paper/reproduce.sh` optionally compiles it with LaTeX. The notebook does not compile the paper.

Do not treat skip-matrix as a proposed trainable optimizer: it deliberately removes essential work to estimate recoverable runtime. Timing runs and instrumented learning-curve runs are kept separate by the existing parser. Error bars differ across figures; each caption explains their meaning.
