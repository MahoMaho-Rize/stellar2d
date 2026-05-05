#!/usr/bin/env python3
"""Phase 1d EVP verification: compute g-mode ω² from the linearised
Boussinesq-stratified operator using **the same Chebyshev discretisation
as the CUDA solver**.  Decouples spatial error from time-integration error.

Derivation:
  Linearised about (ρ₀ = 1, constant N², u = v = b = 0):
    ∂t u = -∂x π
    ∂t v = -∂y π + b
    ∂t b = -N² v
    ∂x u + ∂y v = 0

  Fourier in x: ansatz ∝ e^{i k_x x}.  Eliminate u via continuity:
    u = -∂y v / (i k_x)
  Apply div to momentum to get π equation:  ∇² π = ∂_t(∂_x u + ∂_y v) + ∂_y b = ∂_y b
  in the linear limit (continuity makes the first term zero).

  Eliminate π from the v-equation by taking ∂t of continuity
  + substituting π:  (k_x² - ∂_{yy}) ∂_t v = i k_x² · 0 + k_x² b - ...
  The compact form is

      (k_x² − ∂_{yy}) ∂_{tt} v = −N² k_x² v

  ω² eigenproblem with Dirichlet v(0) = v(Ly) = 0:

      (k_x² − ∂_{yy}) v · ω² = N² k_x² v

  Analytically solvable: v_n = sin(n π y / Ly) gives
      ω²_n = N² k_x² / (k_x² + (n π / Ly)²).

  This script solves the EVP on the exact CGL grid the CUDA solver uses
  (Trefethen D), so the output ω²_n matches the **spatial** accuracy of
  the solver, independent of RK3 / projection splitting.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import scipy.linalg


def trefethen_D(N):
    """Chebyshev-Gauss-Lobatto D matrix on [-1, 1], descending nodes.
    Copied from Trefethen SMMML to match the CUDA implementation bit-exactly.
    """
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = (np.outer(c, 1.0 / c)) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def cgl_grid_ascending(ny, Ly):
    """Same ascending map as AnelasticSLSolver::set_background."""
    N = ny - 1
    D, x = trefethen_D(N)
    # y_k^asc = (1 + x_{N-k}) * Ly / 2  and D_asc[i,j] = (2/Ly) D[N-i, N-j]
    idx = np.arange(N, -1, -1)
    y = (1.0 + x[idx]) * Ly / 2.0
    D_asc = (2.0 / Ly) * D[np.ix_(idx, idx)]
    return y, D_asc


def eigensolve_omega_sq(ny, Ly, N2, kx_phys):
    y, D = cgl_grid_ascending(ny, Ly)
    D2 = D @ D
    # Interior-node problem (Dirichlet on v)
    interior = slice(1, ny - 1)
    M = kx_phys ** 2 * np.eye(ny - 2) - D2[interior, interior]
    rhs = N2 * kx_phys ** 2 * np.eye(ny - 2)
    # Generalized EVP: M v · ω² = rhs · v   ⇒   M⁻¹ rhs · v = ω² · v
    # scipy.linalg.eig(A, B) solves A x = λ B x, i.e. rhs x = λ M x → λ = ω²
    eigvals, eigvecs = scipy.linalg.eig(rhs, M)
    eigvals = np.real(eigvals)
    eigvals = eigvals[np.isfinite(eigvals)]
    eigvals = eigvals[eigvals > 0]
    eigvals.sort()                 # ascending ω²
    eigvals_desc = eigvals[::-1]   # descending (largest first; g-mode n=1 is largest for this form)
    return y, eigvals_desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny",   type=int, default=128)
    ap.add_argument("--Ly",   type=float, default=1.0)
    ap.add_argument("--Lx",   type=float, default=1.0)
    ap.add_argument("--kx",   type=int, default=1)
    ap.add_argument("--N2",   type=float, default=1.0)
    ap.add_argument("--n_show", type=int, default=5)
    args = ap.parse_args()

    kx_phys = args.kx * 2 * np.pi / args.Lx
    y, omega_sq_sorted = eigensolve_omega_sq(args.ny, args.Ly, args.N2, kx_phys)

    print(f"{'Chebyshev EVP on CGL grid (Trefethen D)':^72}")
    print("=" * 72)
    print(f"  ny={args.ny}  Ly={args.Ly}  kx_phys={kx_phys:.4f}  N²={args.N2}")
    print()
    print(f"{'n':>3}  {'ω²_EVP':>14}  {'ω²_analytic':>14}  {'rel err':>12}")
    print("-" * 54)
    for n in range(1, args.n_show + 1):
        if n - 1 >= len(omega_sq_sorted):
            break
        ky = n * np.pi / args.Ly
        omega_sq_exact = args.N2 * kx_phys ** 2 / (kx_phys ** 2 + ky ** 2)
        omega_sq_num = omega_sq_sorted[n - 1]
        rel = abs(omega_sq_num - omega_sq_exact) / omega_sq_exact
        print(f"{n:>3}  {omega_sq_num:14.8e}  {omega_sq_exact:14.8e}  {rel:12.3e}")


if __name__ == "__main__":
    main()
