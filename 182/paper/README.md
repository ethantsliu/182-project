# Report

`main.tex` is the report and `main.pdf` the built copy. The report uses the NeurIPS 2015 style (`nips15submit_e.sty`, from https://neurips.cc/Conferences/2015/PaperInformation/StyleFiles). `overleaf.zip` holds `main.tex`, the style file and the figures: in Overleaf choose New Project, Upload Project, and select the zip.

## Reproducing the figures and tables

```bash
bash 182/paper/reproduce.sh
```

This needs no GPU. It reads the archived training results in `182/results/raw/` (the trainer's logs for every run) and rewrites:

| Output | Made by | From |
|---|---|---|
| `fig_maps.png` | `make_figs.py` | the six map coefficients (no data) |
| `fig_bench.png`, Table 1 | `make_figs.py`; `182/results/tables.md` section C | `raw/bringup-1119/bench.log` |
| `fig_profile.png` | `make_figs.py` | the profile table in `182/results/README.md` section 3.6 (source: `raw/jobs/fastest-1-*/out/session/profile/root/analysis.txt`) |
| `fig_steploop.png`, Table 2 | `make_figs.py`; `tables.md` section B | `raw/bringup-1119/*.log`, `raw/jobs/fastest-1-*/out/root-*.log` |
| `fig_paired.png`, stage table | `make_figs.py` (pairs from `make_tables.py`); `tables.md` sections B2b, B2c and B4 | the logs above, `raw/jobs/kernel-ab2-*/out/root-*.log` and `raw/jobs/qlabs-13354/root-time-*.log` |
| `fig_curve.png` | `make_figs.py`; `tables.md` section B2c | `raw/jobs/qlabs-13354/root-curve-{base,gram222s}-s{5,6,7}.log` |
| `fig_sweep.png` | `make_figs.py`; `tables.md` section C2 | `raw/jobs/qlabs-13354/sweep.log` |
| `fig_profile_late.png` | `make_figs.py`; `tables.md` section D2 | `raw/jobs/qlabs-13354/profile-late/analysis.txt` |
| `182/results/runs.csv`, `tables.md` | `182/results/make_tables.py` | all of `raw/` |

The numbers in the tables of `main.tex` are typed from `tables.md` and the `METRIC` lines of the logs; `182/results/README.md` says where each one comes from.

## Rerunning the experiments

These need one 8xH100 node. From the repository root, after `bash 182/tools/node_setup.sh`:

```bash
python 182/optim-kernel/bench.py --json kbench.json                       # kernel microbenchmark (one GPU)
python -m pytest 182/optim-kernel/test_optim_kernel.py -q                 # kernel tests
RUN_METRICS=1 TRAIN_SEED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py                          # record
RUN_METRICS=1 TRAIN_SEED=1 ANVIL_KERNEL=gram222s torchrun --standalone --nproc_per_node=8 train_gpt.py    # Gram-space kernel
RUN_METRICS=1 TRAIN_SEED=1 ANVIL_ABLATE=skip_matrix torchrun --standalone --nproc_per_node=8 train_gpt.py # ablation
```

Add `VAL_CURVE=150` for validation curves, `TRAIN_LOSS_EVERY=25` for training loss and `SAVE_CHECKPOINT=1` to save the final model. With no flag set the trainer is the record.

## Not in the repository yet

Training-loss curves and model checkpoints. The runs that produce them were submitted and had not finished when this was written.
