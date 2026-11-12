"""Data for the FFT-optimizer experiments. Writes to $FFT_DATA (default /mnt/work/dev_patel/data).

cifar   CIFAR-10 as uint8 tensors (HF parquet mirror).
pde     FNO benchmarks generated here on the GPU (same families as Li et al. 2021, smaller):
  burgers  u_t + (u^2/2)_x = nu u_xx on the unit torus, nu=0.01, u0 ~ GRF(alpha=2.5, tau=7), map u0 -> u(1);
           IF-RK4 pseudo-spectral at 1024 points, dt=1e-4, 2/3 dealiasing; stored at 512 points.
  darcy    -div(a grad u) = 1 on the unit square, u=0 on the boundary, a = 12 if psi>=0 else 3,
           psi ~ GRF(alpha=2, tau=3) cosine series; 5-point FD (harmonic face means) on 64x64 nodes, batched CG.
  ns       2D vorticity, nu=1e-3, f = 0.1(sin 2pi(x+y) + cos 2pi(x+y)), w0 ~ GRF(alpha=2.5, tau=7),
           Crank-Nicolson / explicit advection pseudo-spectral at 64x64, dt=2e-3, map w0 -> w(T=2).
  1000 train + 200 test samples each.
"""
import math
import os
import sys

import torch

D = os.environ.get("FFT_DATA", "/mnt/work/dev_patel/data")
NTR, NTE = 1000, 200


def cifar():
    """CIFAR-10 from the Hugging Face parquet mirror (uoft-cs/cifar10); toronto.edu was ~100 KB/s here."""
    import io
    import subprocess
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    out = os.path.join(D, "cifar10.pt")
    if os.path.exists(out):
        return
    raw = os.path.join(D, "cifar_raw")
    os.makedirs(raw, exist_ok=True)
    arrs = {}
    for split in ("train", "test"):
        f = os.path.join(raw, f"cifar10_{split}.parquet")
        if not os.path.exists(f):
            url = f"https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
            subprocess.run(["curl", "-sSfL", "-o", f + ".part", url], check=True)
            os.rename(f + ".part", f)
        t = pq.read_table(f).to_pydict()
        x = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
        arrs[split] = (torch.from_numpy(x), torch.tensor(t["label"]))
    assert arrs["train"][0].shape == (50000, 32, 32, 3) and arrs["test"][0].shape == (10000, 32, 32, 3)
    torch.save(dict(train_x=arrs["train"][0], train_y=arrs["train"][1], test_x=arrs["test"][0], test_y=arrs["test"][1]), out)
    print("wrote", out)


def burgers(n, N=1024, nu=0.01, T=1.0, dt=1e-4, dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    k = torch.fft.rfftfreq(N, d=1.0 / N, device=dev).double()  # integer wavenumbers
    alpha, tau = 2.5, 7.0
    sigma = tau ** (0.5 * (2 * alpha - 1))
    sq = sigma * ((2 * math.pi * k) ** 2 + tau ** 2) ** (-alpha / 2)
    sq[0] = 0
    c = (torch.randn(n, k.numel(), generator=g, device=dev, dtype=torch.float64)
         + 1j * torch.randn(n, k.numel(), generator=g, device=dev, dtype=torch.float64)) * sq / math.sqrt(2)
    uh = c * N
    u0 = torch.fft.irfft(uh, n=N)
    ik = 2j * math.pi * k
    lin = -nu * (2 * math.pi * k) ** 2
    dealias = (k < N / 3).double()

    def nl(vh):
        u = torch.fft.irfft(vh, n=N)
        return -0.5 * ik * torch.fft.rfft(u * u) * dealias

    E, E2 = torch.exp(lin * dt), torch.exp(lin * dt / 2)
    vh = uh.clone()
    for _ in range(int(round(T / dt))):  # integrating-factor RK4, one step at a time (no overflow)
        k1 = nl(vh)
        k2 = nl(E2 * (vh + 0.5 * dt * k1))
        k3 = nl(E2 * vh + 0.5 * dt * k2)
        k4 = nl(E * vh + dt * E2 * k3)
        vh = E * vh + dt / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
    uT = torch.fft.irfft(vh, n=N)
    s = N // 512
    return u0[:, ::s].float().cpu(), uT[:, ::s].float().cpu()


def darcy(n, s=64, dev="cuda", seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    alpha, tau = 2.0, 3.0
    K = 32
    kk = torch.arange(K, device=dev).double()
    lam = (math.pi ** 2 * (kk[:, None] ** 2 + kk[None, :] ** 2) + tau ** 2) ** (-alpha / 2)
    lam[0, 0] = 0
    x = torch.linspace(0, 1, s, device=dev).double()
    cosx = torch.cos(math.pi * kk[:, None] * x[None, :])  # (K, s)
    xi = torch.randn(n, K, K, generator=g, device=dev, dtype=torch.float64) * lam
    psi = torch.einsum("nab,ai,bj->nij", xi, cosx, cosx)
    a = torch.where(psi >= 0, 12.0, 3.0).double()
    h = 1.0 / (s - 1)
    hm = lambda p, q: 2 * p * q / (p + q)
    ax = hm(a[:, 1:, :], a[:, :-1, :])  # faces between i and i+1 (n, s-1, s)
    ay = hm(a[:, :, 1:], a[:, :, :-1])
    ax_, ay_ = ax[:, :, 1:-1], ay[:, 1:-1, :]  # restricted to interior rows/cols

    def A(u):  # u (n, s-2, s-2) interior; zero Dirichlet
        U = torch.nn.functional.pad(u, (1, 1, 1, 1))
        fx = ax_ * (U[:, 1:, 1:-1] - U[:, :-1, 1:-1])  # (n, s-1, s-2)
        fy = ay_ * (U[:, 1:-1, 1:] - U[:, 1:-1, :-1])  # (n, s-2, s-1)
        return -((fx[:, 1:] - fx[:, :-1]) + (fy[:, :, 1:] - fy[:, :, :-1])) / h ** 2

    b = torch.ones(n, s - 2, s - 2, device=dev, dtype=torch.float64)
    u = torch.zeros_like(b)
    r = b - A(u)
    p = r.clone()
    rs = (r * r).sum((1, 2))
    b2 = (b * b).sum((1, 2))
    for it in range(5000):
        Ap = A(p)
        al = rs / (p * Ap).sum((1, 2))
        u = u + al[:, None, None] * p
        r = r - al[:, None, None] * Ap
        rs_new = (r * r).sum((1, 2))
        if (rs_new / b2).max().sqrt() < 1e-9:
            break
        p = r + (rs_new / rs)[:, None, None] * p
        rs = rs_new
    print(f"darcy CG: {it + 1} iterations, max rel residual {(rs_new / b2).max().sqrt().item():.2e}")
    uf = torch.nn.functional.pad(u, (1, 1, 1, 1))
    return a.float().cpu(), uf.float().cpu()


def ns(n, s=64, nu=1e-3, T=2.0, dt=2e-3, dev="cuda", seed=2):
    g = torch.Generator(device=dev).manual_seed(seed)
    alpha, tau = 2.5, 7.0
    k1 = torch.fft.fftfreq(s, d=1.0 / s, device=dev).double()
    k2 = torch.fft.rfftfreq(s, d=1.0 / s, device=dev).double()
    KX, KY = torch.meshgrid(k1, k2, indexing="ij")
    sigma = tau ** (0.5 * (2 * alpha - 2))
    sq = math.sqrt(2) * sigma * (4 * math.pi ** 2 * (KX ** 2 + KY ** 2) + tau ** 2) ** (-alpha / 2)
    sq[0, 0] = 0
    wh = (torch.randn(n, s, s // 2 + 1, generator=g, device=dev, dtype=torch.float64)
          + 1j * torch.randn(n, s, s // 2 + 1, generator=g, device=dev, dtype=torch.float64)) * sq * s * s / math.sqrt(2)
    w0 = torch.fft.irfft2(wh, s=(s, s))
    wh = torch.fft.rfft2(w0)
    x = torch.arange(s, device=dev).double() / s
    X, Y = torch.meshgrid(x, x, indexing="ij")
    fh = torch.fft.rfft2(0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))))
    lap = 4 * math.pi ** 2 * (KX ** 2 + KY ** 2)
    lap_inv = torch.where(lap > 0, 1.0 / lap, torch.zeros_like(lap))
    dealias = ((KX.abs() <= s / 3) & (KY.abs() <= s / 3)).double()
    for _ in range(int(round(T / dt))):
        psih = wh * lap_inv
        u = torch.fft.irfft2(2j * math.pi * KY * psih, s=(s, s))
        v = torch.fft.irfft2(-2j * math.pi * KX * psih, s=(s, s))
        wx = torch.fft.irfft2(2j * math.pi * KX * wh, s=(s, s))
        wy = torch.fft.irfft2(2j * math.pi * KY * wh, s=(s, s))
        F = torch.fft.rfft2(u * wx + v * wy) * dealias
        wh = (-dt * F + dt * fh + (1 - 0.5 * dt * nu * lap) * wh) / (1 + 0.5 * dt * nu * lap)
    wT = torch.fft.irfft2(wh, s=(s, s))
    return w0.float().cpu(), wT.float().cpu()


def pde():
    os.makedirs(os.path.join(D, "fno"), exist_ok=True)
    for name, fn in (("burgers", burgers), ("darcy", darcy), ("ns", ns)):
        out = os.path.join(D, "fno", f"{name}.pt")
        if os.path.exists(out):
            continue
        x, y = fn(NTR + NTE)
        assert torch.isfinite(x).all() and torch.isfinite(y).all(), name
        torch.save(dict(x_train=x[:NTR], y_train=y[:NTR], x_test=x[NTR:], y_test=y[NTR:]), out)
        print(f"wrote {out}: x {tuple(x.shape)} rms {x.pow(2).mean().sqrt():.3g}, y rms {y.pow(2).mean().sqrt():.3g}, "
              f"rel change {((y - x).norm() / x.norm()).item():.3g}")


if __name__ == "__main__":
    for c in sys.argv[1:]:
        {"cifar": cifar, "pde": pde}[c]()
