"""How many maps can be chained on the Gram matrix before a restart? CPU only, no GPU needed.

    python 182/optim-kernel/restart_interval.py            # the record's mlp_bank shape (1 x 2816 x 768), two seeds

Each grouping composes that many consecutive maps on the Gram side before re-reading X (anvil_kernel.gram_cascade),
with every product accumulated in fp32 and stored in bf16 (RefOps, the emulation the tests use). The number printed
is the relative Frobenius error against the exact cascade (bench.exact_cascade), for three synthetic spectra
(bench.synthetic): p05 (sigma_i ~ i^-0.5), p1 (sigma_i ~ 1/i, heavy tail) and gauss (flat)."""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402
from track_1_short.perf import anvil_kernel as AK  # noqa: E402

GROUPINGS = {"record (restart every map)": None, "2+2+2": (2, 2, 2), "3+3": (3, 3), "4+2": (4, 2), "2+4": (2, 4),
             "5+1": (5, 1), "6": (6,)}


def rel(a, b):
    return (torch.linalg.matrix_norm(a.double() - b.double()) / torch.linalg.matrix_norm(b.double())).max().item()


def main():
    torch.set_num_threads(2)
    shape = (1, 2816, 768)
    print("| grouping | " + " | ".join(f"{sp} seed {sd}" for sp in ("p05", "p1", "gauss") for sd in (0, 1)) + " |")
    print("|---|" + "---|" * 6)
    cols = {}
    for sp in ("p05", "p1", "gauss"):
        for sd in (0, 1):
            X = bench.synthetic(shape, sp, torch.device("cpu"), sd)
            exact = bench.exact_cascade(X)
            for name, g in GROUPINGS.items():
                ops = AK.RefOps(g is not None)  # the symmetric small product for the Gram-space groupings
                out = (AK.reference_cascade(X, ops, bench.MAPS, bench.MARGIN, bench.EPS) if g is None
                       else AK.gram_cascade(X, ops, bench.MAPS, g, bench.MARGIN, bench.EPS))
                cols.setdefault(name, []).append(rel(out.float(), exact))
    for name, vals in cols.items():
        print(f"| {name} | " + " | ".join(f"{v:.2%}" if v < 10 else f"{v:.1e}" for v in vals) + " |")


if __name__ == "__main__":
    main()
