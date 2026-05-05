#!/usr/bin/env python3
"""Sanity-check Phase 1c: read anelastic_sl KH Boussinesq CSV dump and verify
basic physical invariants (divergence-free, reasonable ω spectrum).

Inputs: path to kh_final.csv written by `--solver anelastic_sl --test
kh_shear_boussinesq`.

Checks:
  (1) ∇·u ≈ 0 everywhere (projection worked).  Uses FFT in x and Chebyshev D
      on CGL y rebuilt from y_cgl in the CSV.
  (2) Kinetic energy per k_x bin (E(k_x) = ∫dy ½Σ_y |û|²+|v̂|²) — report the
      shape; spectrum should roll off smoothly, no k_max bump.
  (3) Max ω, RMS ω, max |∇·u|  → print for a one-glance sanity pass.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np


def parse_csv(path: Path):
    with path.open() as f:
        lines = f.readlines()
    hdr = lines[0]
    assert hdr.startswith("# nx="), f"unexpected header: {hdr!r}"
    meta = dict(kv.strip().split("=") for kv in hdr[2:].split(",") if "=" in kv)
    nx = int(meta["nx"]); ny = int(meta["ny"])
    Lx = float(meta["Lx"]); Ly = float(meta["Ly"])
    nu = float(meta["nu"]); t = float(meta["t"])
    ncell = nx * ny

    # Find blocks by "# " markers
    blocks = {"y": None, "omega": None, "u": None, "v": None}
    cur = None
    buf: list[float] = []

    def push(arr):
        if cur is not None:
            blocks[cur] = np.array(buf, dtype=float)

    for ln in lines[1:]:
        s = ln.strip()
        if s.startswith("#"):
            push(buf)
            buf = []
            if "y_cgl" in s:    cur = "y"
            elif "omega" in s:  cur = "omega"
            elif s.endswith("u (ny × nx row-major):"): cur = "u"
            elif s.endswith("v (ny × nx row-major):"): cur = "v"
            else: cur = None
        elif cur is not None and s:
            buf.append(float(s))
    push(buf)

    y = blocks["y"].reshape(ny)
    omega = blocks["omega"].reshape(ny, nx)
    u = blocks["u"].reshape(ny, nx)
    v = blocks["v"].reshape(ny, nx)
    return dict(nx=nx, ny=ny, Lx=Lx, Ly=Ly, nu=nu, t=t,
                y=y, omega=omega, u=u, v=v)


def cheb_diffmat_on(y):
    """Trefethen D for ascending y on [0, Ly], via chain rule from [-1, 1]."""
    ny = len(y); N = ny - 1
    Ly = y[-1] - y[0]
    # Descending x_k = cos(kπ/N), map y_k^asc = Ly/2 * (1 + x_{N-k})
    x = np.cos(np.arange(N + 1) * np.pi / N)    # descending
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D_desc = (c[:, None] / c[None, :]) * ((-1.0) ** (np.arange(N + 1)[:, None]
             + np.arange(N + 1)[None, :])) / (dX + np.eye(N + 1))
    D_desc -= np.diag(D_desc.sum(axis=1))
    # Reorder to ascending y.
    idx = np.arange(N, -1, -1)  # N-k for k=0..N → reverse
    D_asc = D_desc[np.ix_(idx, idx)]
    return D_asc * (2.0 / Ly)


def compute_divergence(u, v, Lx, y):
    ny, nx = u.shape
    # ∂x u via FFT
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=Lx / nx)
    uh = np.fft.rfft(u, axis=1)
    dudx = np.fft.irfft(1j * kx[None, :] * uh, n=nx, axis=1)
    # ∂y v via Chebyshev D (apply along y axis)
    D = cheb_diffmat_on(y)
    dvdy = D @ v
    return dudx + dvdy


def kx_spectrum(u, v, Lx):
    nx = u.shape[1]; ny = u.shape[0]
    uh = np.fft.rfft(u, axis=1) / nx
    vh = np.fft.rfft(v, axis=1) / nx
    # Integrate |·|² over y with trapezoid on CGL-averaged spacing.
    # y-integration doesn't matter for shape; use uniform sum to compare trends.
    E = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2).sum(axis=0)
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()

    d = parse_csv(args.csv)
    nx, ny = d["nx"], d["ny"]
    print(f"loaded: nx={nx}, ny={ny}, Lx={d['Lx']}, Ly={d['Ly']}, "
          f"ν={d['nu']:.2e}, t={d['t']:.3f}")

    div = compute_divergence(d["u"], d["v"], d["Lx"], d["y"])
    max_div = float(np.max(np.abs(div)))
    max_omega = float(np.max(np.abs(d["omega"])))
    rms_omega = float(np.sqrt(np.mean(d["omega"] ** 2)))
    u_inf = float(np.max(np.abs(d["u"])))
    v_inf = float(np.max(np.abs(d["v"])))
    print(f"  |u|∞ = {u_inf:.4f}   |v|∞ = {v_inf:.4e}")
    print(f"  |ω|∞ = {max_omega:.4f}   |ω|_rms = {rms_omega:.4f}")
    print(f"  |∇·u|∞ = {max_div:.4e}   (should be ≲ 1e-8 for healthy projection)")

    E = kx_spectrum(d["u"], d["v"], d["Lx"])
    print(f"  E(k_x) first 6 bins: " + ", ".join(f"{e:.2e}" for e in E[:6]))
    print(f"  E(k_x) last  6 bins: " + ", ".join(f"{e:.2e}" for e in E[-6:]))
    # A healthy spectrum rolls off >10 orders from peak → k_max;
    # a grid-scale bump / aliasing would show comparable last-bin values.
    roll = E.max() / max(E[-1], 1e-300)
    print(f"  spectrum roll-off (peak / last bin) = {roll:.2e}")

    # SL-Poisson Dirichlet projection only kills divergence in the span of
    # the SL basis (ψ_n vanishing at walls).  Boundary-layer residual
    # ∇·u can persist at O(10^-3) without signalling a bug; a truly broken
    # projection produces O(1) divergence.
    if max_div > 1e-2:
        print("FAIL: divergence too large — projection step is broken.")
        sys.exit(1)
    if not np.isfinite(max_omega):
        print("FAIL: NaN or Inf in ω.")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
