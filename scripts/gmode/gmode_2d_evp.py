#!/usr/bin/env python3
"""2D anelastic g-mode EVP on (Fourier x, Chebyshev y).

Per k_x, solve the scalar eigenproblem for v(y):

    A v = ω²  B v,
        A = diag(k_x² N²(y) ρ₀(y))
        B = -∂_y (ρ₀(y) ∂_y) + k_x² diag(ρ₀(y))

with Dirichlet v(0) = v(L_y) = 0.

This derives from the linearised anelastic system

    ∂_t u  = -(1/ρ₀) ∂_x p
    ∂_t v  = -(1/ρ₀) ∂_y p + b
    ∂_t b  = -N²(y) v
    ∂_x(ρ₀ u) + ∂_y(ρ₀ v) = 0

by eliminating u (continuity), b (buoyancy), and p (momentum/Poisson).

Boussinesq (ρ₀=1) reduces to  (k_x² − ∂_yy) v ω² = k_x² N² v, matching the
simple Schwarzschild oscillator ω² = N² k_x² / (k_x² + k_y²).

This script is the **reference** for the CUDA 2D EVP solver: machine
precision, all k_x batched, supports arbitrary ρ₀(y), N²(y).
"""
from __future__ import annotations
import argparse
import numpy as np
import scipy.linalg


def trefethen_D(N: int):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = (np.outer(c, 1.0 / c)) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def cgl_asc(ny: int, Ly: float):
    N = ny - 1
    D, x = trefethen_D(N)
    idx = np.arange(N, -1, -1)
    y = (1.0 + x[idx]) * Ly / 2.0
    D_asc = (2.0 / Ly) * D[np.ix_(idx, idx)]
    return y, D_asc


def solve_gmode_2d_scalar(ny: int, Ly: float, kx_phys: float,
                          rho_fn, N2_fn, n_modes: int = 10):
    """Solve  A v = ω² B v  on interior CGL nodes (Dirichlet BCs).

    Returns omega_sq (ascending ω²), v_modes on interior nodes (shape
    (ny-2, n_modes)).  For Boussinesq this matches (n π / Ly)² eigenvalues.

    rho_fn, N2_fn: callables y → array, evaluated on CGL grid.
    """
    y, D = cgl_asc(ny, Ly)
    rho = rho_fn(y)
    N2  = N2_fn(y)

    # B = -D diag(ρ) D + k_x² diag(ρ)
    B = -D @ (np.diag(rho) @ D) + kx_phys ** 2 * np.diag(rho)
    # A = k_x² diag(N² ρ)
    A = kx_phys ** 2 * np.diag(N2 * rho)

    interior = slice(1, ny - 1)
    A_int = A[interior, interior]
    B_int = B[interior, interior]

    # Generalised eigenproblem  A v = ω² B v  → scipy.linalg.eig(A, B)
    omega_sq, v_modes = scipy.linalg.eig(A_int, B_int)
    omega_sq = np.real(omega_sq)
    v_modes = np.real(v_modes)
    # Drop infinite / negative modes
    good = np.isfinite(omega_sq) & (omega_sq > 0)
    omega_sq = omega_sq[good]
    v_modes = v_modes[:, good]
    order = np.argsort(omega_sq)          # ascending: smallest first (high-n g-mode)
    # For Boussinesq, n=1 is *largest* ω² (low-k_y).  Users often want that first,
    # so return descending.
    order = order[::-1]
    omega_sq = omega_sq[order]
    v_modes = v_modes[:, order]
    return y, omega_sq[:n_modes], v_modes[:, :n_modes]


def boussinesq_analytic(kx_phys: float, ky_int: int, Ly: float, N2: float):
    ky = ky_int * np.pi / Ly
    return N2 * kx_phys ** 2 / (kx_phys ** 2 + ky ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--kx_int", type=int, default=1)
    ap.add_argument("--N2", type=float, default=1.0)
    ap.add_argument("--n_show", type=int, default=5)
    ap.add_argument("--profile", choices=["boussinesq", "gaussian_N2",
                                          "lane_emden_1_5"],
                    default="boussinesq")
    args = ap.parse_args()

    kx_phys = args.kx_int * 2 * np.pi / args.Lx

    if args.profile == "boussinesq":
        rho_fn = lambda y: np.ones_like(y)
        N2_fn  = lambda y: args.N2 * np.ones_like(y)
    elif args.profile == "gaussian_N2":
        # N²(y) Gaussian bump in the middle, ρ uniform
        rho_fn = lambda y: np.ones_like(y)
        yc = args.Ly / 2; sigma = 0.2 * args.Ly
        N2_fn = lambda y: args.N2 * np.exp(-((y - yc) / sigma) ** 2)
    elif args.profile == "lane_emden_1_5":
        # crude: rho = (1 - (y/Ly - 0.5)²)^1.5 + 0.01,  N² ∝ -ρ'/ρ (stable above)
        def rho_fn(y):
            q = 1.0 - ((y / args.Ly - 0.5) * 2) ** 2
            return np.clip(q, 0.0, None) ** 1.5 + 0.01
        def N2_fn(y):
            eps = 1e-3
            rho = rho_fn(y)
            drho = np.gradient(rho, y, edge_order=2)
            return np.maximum(-drho / (rho + eps), 0.0)   # stable zones only

    y, om2, v_modes = solve_gmode_2d_scalar(args.ny, args.Ly, kx_phys,
                                            rho_fn, N2_fn, args.n_show)

    print(f"{'2D anelastic g-mode EVP (scalar form)':^72}")
    print("=" * 72)
    print(f"  profile={args.profile}  ny={args.ny}  Ly={args.Ly}  "
          f"Lx={args.Lx}  k_x_int={args.kx_int}")
    print(f"  k_x_phys = {kx_phys:.6f}")
    print()
    print(f"{'n':>3}  {'ω²':>16}  {'ω':>10}  {'ω²_analytic':>14}  {'rel err':>12}")
    print("-" * 72)
    for n, w2 in enumerate(om2, 1):
        if args.profile == "boussinesq":
            w2_exact = boussinesq_analytic(kx_phys, n, args.Ly, args.N2)
            rel = abs(w2 - w2_exact) / w2_exact
            print(f"{n:>3}  {w2:16.10e}  {np.sqrt(w2):10.6f}  "
                  f"{w2_exact:14.10e}  {rel:12.3e}")
        else:
            print(f"{n:>3}  {w2:16.10e}  {np.sqrt(w2):10.6f}")


if __name__ == "__main__":
    main()
