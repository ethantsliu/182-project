"""CPU bf16 simulation of the ANVIL cascade (optim/anvil.py anvil_cascade, from the bf16 cast on), for the OPTIM_MAPS /
OPTIM_NORM flags: how close each schedule gets to the polar factor U V^T on matrices of this trainer's bank shapes.

    python 182/optim-cost/cascade_sim.py            # the table in this directory's README
    python 182/optim-cost/cascade_sim.py --quick    # two shapes, fewer spectra (what the test runs)

The arithmetic follows anvil_cascade: X in bf16; the Gram, the polynomial b*A + c*A@A and the a*X + X@B step each a bf16
matmul (fp32 accumulate, bf16 out); the normalization in fp32 from the Gram of the unnormalized X. It does not reproduce
the GPU kernels' bytes (tile order, fma contraction), only their precision class. The spectra are synthetic (no trained
velocity is available off the node): flat, power laws, a Gaussian matrix, and a low-rank signal over a noise floor.
"""
import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the variant root, for track_1_short

from track_1_short.optim.cascade_flags import ANVIL_MAPS3  # noqa: E402


def shipped_maps() -> list[tuple[float, float, float]]:
    """ANVIL_MAPS, read from optim/anvil.py's source (importing it needs triton)."""
    import ast
    src = open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "track_1_short", "optim", "anvil.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "ANVIL_MAPS":
            return [tuple(t) for t in ast.literal_eval(node.value)]
    raise LookupError("ANVIL_MAPS not found in optim/anvil.py")


ANVIL_MAPS = shipped_maps()

FROBENIUS_EPS = 1e-6


def bmm16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 matmul with fp32 accumulation, bf16 result (cuBLAS / Triton tl.dot class)."""
    return (a.float() @ b.float()).bfloat16()


def cascade(X: torch.Tensor, maps, margin: float = 1.05, s4: bool = False) -> torch.Tensor:
    """X [..., m, n] -> the cascade's bf16 output. Tall matrices use X^T X and right products, wide/square X X^T and left."""
    X = X.bfloat16()
    tall = X.size(-2) > X.size(-1)
    gram = (lambda Y: bmm16(Y.mT, Y)) if tall else (lambda Y: bmm16(Y, Y.mT))
    A = gram(X)
    if s4:
        nrm = A.float().square().sum(dim=(-2, -1))[..., None, None].sqrt().sqrt()
    else:
        nrm = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None].sqrt()
    d = nrm * margin + FROBENIUS_EPS
    X = (X.float() / d).bfloat16()
    A = (A.float() / d.square()).bfloat16()
    for k, (a, b, c) in enumerate(maps):
        if k > 0:
            A = gram(X)
        B = (b * A.float() + c * bmm16(A, A).float()).bfloat16()
        XB = bmm16(X, B) if tall else bmm16(B, X)
        X = (XB.float() + a * X.float()).bfloat16()
    return X


def make_matrix(m: int, n: int, spectrum: str, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(X, U V^T) in float64 with X = U diag(s) V^T for a named spectrum; r = min(m, n) singular values."""
    r = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, r, generator=gen, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, r, generator=gen, dtype=torch.float64))
    i = torch.arange(1, r + 1, dtype=torch.float64)
    if spectrum == "gaussian":
        X = torch.randn(m, n, generator=gen, dtype=torch.float64)
        Uf, _, Vh = torch.linalg.svd(X, full_matrices=False)
        return X, Uf @ Vh
    if spectrum == "flat":
        s = torch.ones(r, dtype=torch.float64)
    elif spectrum.startswith("pow"):
        s = i ** (-float(spectrum[3:]))
    elif spectrum == "lowrank":  # 16 strong directions, 30x over a flat noise floor
        s = torch.where(i <= 16, 30.0 * (17 - i) / 16, torch.ones_like(i))
    else:
        raise ValueError(spectrum)
    return (U * s) @ V.T, U @ V.T


def score(out: torch.Tensor, polar: torch.Tensor) -> dict:
    """Distance to the polar factor (relative Frobenius) and the output's singular-value envelope."""
    o = out.double()
    sv = torch.linalg.svdvals(o)
    rel = (torch.linalg.norm(o - polar) / torch.linalg.norm(polar)).item()
    return dict(rel=rel, smin=sv.min().item(), p10=sv.quantile(0.10).item(), med=sv.median().item(), smax=sv.max().item(),
                below=(sv < 0.5).double().mean().item())


VARIANTS = {  # name -> (maps, s4)
    "F4 (shipped)": (ANVIL_MAPS, False),
    "F3": (ANVIL_MAPS3, False),
    "S4+4": (ANVIL_MAPS, True),
    "S4+3": (ANVIL_MAPS3, True),
}
SHAPES = {"qk_bank": (256, 768), "vo_bank": (768, 768), "moe_bank": (1024, 768), "mlp_bank": (2816, 768)}
SPECTRA = ("flat", "gaussian", "pow0.5", "pow1", "lowrank")


def table(shapes, spectra, seed: int = 0):
    rows = []
    for bank, (m, n) in shapes.items():
        margin = 1.1 if bank == "moe_bank" else 1.05
        for spec in spectra:
            gen = torch.Generator().manual_seed(seed)
            X, P = make_matrix(m, n, spec, gen)
            fro = torch.linalg.norm(X).item()
            s4n = torch.linalg.svdvals(X).pow(4).sum().pow(0.25).item()
            for name, (maps, s4) in VARIANTS.items():
                rows.append(dict(bank=bank, spectrum=spec, variant=name, fro_over_s4=fro / s4n,
                                 **score(cascade(X.float(), maps, margin, s4), P)))
    return rows


def orth_vs_iterations(m=1024, n=768, spectrum="pow0.5", seed=0):
    """Distance to U V^T after each map, for the shipped maps and MAPS3 under both normalizations."""
    gen = torch.Generator().manual_seed(seed)
    X, P = make_matrix(m, n, spectrum, gen)
    out = {}
    for name, (maps, s4) in VARIANTS.items():
        out[name] = [score(cascade(X.float(), maps[:k], 1.05, s4), P)["rel"] for k in range(1, len(maps) + 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    shapes = {k: SHAPES[k] for k in ("qk_bank", "moe_bank")} if args.quick else SHAPES
    spectra = ("gaussian", "pow1") if args.quick else SPECTRA
    print("orthogonality error ||out - UV^T||_F / ||UV^T||_F after k maps (1024x768, sigma_i ~ i^-0.5):")
    for name, errs in orth_vs_iterations().items():
        print(f"  {name:<13} " + "  ".join(f"k={k + 1}: {e:.4f}" for k, e in enumerate(errs)))
    print()
    print(f"{'bank':<9} {'spectrum':<9} {'F/S4':>5}  {'variant':<13} {'rel err':>8} {'smin':>6} {'p10':>6} {'median':>6} "
          f"{'smax':>6} {'<0.5':>6}")
    for r in table(shapes, spectra):
        print(f"{r['bank']:<9} {r['spectrum']:<9} {r['fro_over_s4']:5.2f}  {r['variant']:<13} {r['rel']:8.4f} "
              f"{r['smin']:6.3f} {r['p10']:6.3f} {r['med']:6.3f} {r['smax']:6.3f} {r['below']:6.1%}")


if __name__ == "__main__":
    main()
