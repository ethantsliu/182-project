"""Shared pieces: inverse roots, eigenbases, complex polar factors, and a batched matrix optimizer.

`MatrixOpt` runs Shampoo / shared-factor Shampoo / SOAP / Muon on a parameter after a linear "view"
maps it to a batch of matrices (B, m, n), real or complex. Each experiment supplies its own views:
  exp1  conv kernel -> per-frequency rfft2 matrices, per-offset matrices, or a flattened (Cout, Cin*k*k)
  exp3  FNO spectral weight (K, Cin, Cout) -> per-mode matrices or the flattened (Cin, K*Cout)
Momentum (and the grafting Adam) live in parameter space; the views are linear, so the view of the
EMA is the EMA of the views. Factors are EMAs in the view; inverse roots / eigenbases are computed in
float64 eigh with damping eps = 1e-6 * tr/dim, refreshed every `precond_every` steps.
"""
import math

import torch


def _f64(A):
    return A.to(torch.complex128 if A.is_complex() else torch.float64)


def _damped(A, rel_eps):
    A = _f64(A)
    n = A.shape[-1]
    tr = torch.diagonal(A, dim1=-2, dim2=-1).real.sum(-1)
    eps = (rel_eps * tr / n).clamp_min(1e-30)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    A = 0.5 * (A + A.mH)
    return A + eps[..., None, None] * eye, eps


def inv_root(A, p, rel_eps=1e-6, out_dtype=None):
    """(A + eps I)^(-1/p) for a batch of Hermitian PSD matrices, via float64 eigh."""
    Ad, eps = _damped(A, rel_eps)
    w, V = torch.linalg.eigh(Ad)
    w = torch.maximum(w, eps[..., None])
    R = (V * w.pow(-1.0 / p).to(V.dtype)[..., None, :]) @ V.mH
    return R.to(out_dtype or A.dtype)


def eigvecs(A, rel_eps=1e-6, out_dtype=None):
    Ad, _ = _damped(A, rel_eps)
    _, V = torch.linalg.eigh(Ad)
    return V.to(out_dtype or A.dtype)


def eig_condition(A):
    """Undamped condition numbers lambda_max / lambda_min of Hermitian PSD matrices."""
    w = torch.linalg.eigvalsh(0.5 * (_f64(A) + _f64(A).mH))
    return (w[..., -1] / w[..., 0].clamp_min(1e-300)).float()


def polar(G, steps=5):
    """Approximate polar factor U V^H (Muon's quintic Newton-Schulz); works for complex matrices."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.complex64 if G.is_complex() else torch.float32)
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.mH
    X = X / (torch.linalg.matrix_norm(X, keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mH
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.mH
    return X


def nbytes_state(opt):
    tot = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
            elif isinstance(v, dict):
                tot += sum(t.numel() * t.element_size() for t in v.values() if torch.is_tensor(t))
    return tot


class View:
    """Linear map param <-> batch of matrices. fwd(x) -> (B, m, n); bwd(Y) -> param-shaped real/complex."""

    def __init__(self, fwd, bwd):
        self.fwd, self.bwd = fwd, bwd


def blocked_flat_view(shape, max_dim=1152, flatten=None, unflatten=None):
    """(m, n) matrix (after optional flatten) split along columns into equal blocks of <= max_dim."""
    flatten = flatten or (lambda x: x)
    unflatten = unflatten or (lambda x: x)

    def nblocks(n):
        nb = math.ceil(n / max_dim)
        while n % nb:
            nb += 1
        return nb

    def fwd(x):
        X = flatten(x)
        m, n = X.shape
        nb = nblocks(n)
        return X.reshape(m, nb, n // nb).transpose(0, 1)

    def bwd(Y):
        nb, m, nn = Y.shape
        return unflatten(Y.transpose(0, 1).reshape(m, nb * nn))

    return View(fwd, bwd)


class MatrixOpt(torch.optim.Optimizer):
    """mode in {shampoo, shampoo_shared, soap, muon}. Each param group carries `view`.

    shampoo / shampoo_shared: U = L^-1/4 M R^-1/4 per matrix (shared: one L, R summed over the batch),
      grafted to the norm of the Adam update (per parameter).
    soap: Adam in the eigenbasis of each matrix's (L, R); no grafting.
    muon: polar factor of the Nesterov momentum per matrix, times sqrt(max(1, m/n)); no grafting.
    Decoupled weight decay: p *= 1 - lr_mult * decay, with decay given absolutely (not times lr).
    """

    def __init__(self, params, lr, mode, betas=(0.9, 0.99), shampoo_beta=None, eps=1e-8,
                 precond_every=20, decay=0.0, rel_eps=1e-6, nesterov=True):
        assert mode in ("shampoo", "shampoo_shared", "soap", "muon")
        if mode == "muon":
            betas = (0.95, betas[1])
        if mode == "soap" and shampoo_beta is None:
            betas, shampoo_beta = (0.95, 0.95), 0.95
        defaults = dict(lr=lr, mode=mode, betas=betas, shampoo_beta=shampoo_beta or betas[1], eps=eps,
                        precond_every=precond_every, decay=decay, rel_eps=rel_eps, nesterov=nesterov,
                        lr_mult=1.0, view=None)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            mode, (b1, b2), view = g["mode"], g["betas"], g["view"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                G = p.grad.float() if not p.grad.is_complex() else p.grad
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(G)
                    if mode in ("shampoo", "shampoo_shared"):
                        st["v"] = torch.zeros_like(G, dtype=G.real.dtype)
                st["t"] += 1
                t = st["t"]
                st["m"].lerp_(G, 1 - b1)
                if mode == "muon":
                    M = G.lerp(st["m"], b1) if g["nesterov"] else st["m"]
                    X = view.fwd(M)
                    U = polar(X) * math.sqrt(max(1.0, X.shape[-2] / X.shape[-1]))
                    upd = view.bwd(U)
                else:
                    Gv = view.fwd(G)
                    self._update_factors(st, Gv, g, t)
                    if mode == "soap":
                        upd = self._soap(st, Gv, g, t, view)
                    else:
                        st["v"].mul_(b2).add_((G.abs() ** 2), alpha=1 - b2)
                        mh = st["m"] / (1 - b1 ** t)
                        adam = mh / ((st["v"] / (1 - b2 ** t)).sqrt() + g["eps"])
                        Mv = view.fwd(mh)
                        U = st["PL"] @ Mv @ st["PR"]
                        upd = view.bwd(U)
                        upd = upd * (adam.norm() / (upd.norm() + 1e-30))
                if g["decay"]:
                    p.mul_(1 - g["lr_mult"] * g["decay"])
                p.add_(upd.to(p.dtype), alpha=-g["lr"] * g["lr_mult"])

    def _update_factors(self, st, Gv, g, t):
        sb = g["shampoo_beta"]
        L = Gv @ Gv.mH
        R = Gv.mH @ Gv
        if g["mode"] == "shampoo_shared":
            L, R = L.sum(0, keepdim=True), R.sum(0, keepdim=True)
        if "L" not in st:
            st["L"], st["R"] = torch.zeros_like(L), torch.zeros_like(R)
        st["L"].mul_(sb).add_(L, alpha=1 - sb)
        st["R"].mul_(sb).add_(R, alpha=1 - sb)
        if t == 1 or t % g["precond_every"] == 0:
            if g["mode"] == "soap":
                st["QL"] = eigvecs(st["L"], g["rel_eps"])
                st["QR"] = eigvecs(st["R"], g["rel_eps"])
            else:
                st["PL"] = inv_root(st["L"], 4, g["rel_eps"])
                st["PR"] = inv_root(st["R"], 4, g["rel_eps"])

    def _soap(self, st, Gv, g, t, view):
        b1, b2 = g["betas"]
        QL, QR = st["QL"], st["QR"]
        Gr = QL.mH @ Gv @ QR
        if "vr" not in st:
            st["vr"] = torch.zeros_like(Gr, dtype=Gr.real.dtype)
        st["vr"].mul_(b2).add_(Gr.abs() ** 2, alpha=1 - b2)
        mh = st["m"] / (1 - b1 ** t)
        Mr = QL.mH @ view.fwd(mh) @ QR
        Ur = Mr / ((st["vr"] / (1 - b2 ** t)).sqrt() + g["eps"])
        return view.bwd(QL @ Ur @ QR.mH)


def set_lr_mult(opts, mult):
    for o in opts:
        for g in o.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            if "lr_mult" in g:
                g["lr_mult"] = mult
            else:
                g["lr"] = g["base_lr"] * mult


def warmup_linear(step, total, warm_frac=0.05, floor=0.0):
    w = max(1, int(warm_frac * total))
    if step < w:
        return (step + 1) / w
    return floor + (1 - floor) * max(0.0, (total - step) / max(1, total - w))
