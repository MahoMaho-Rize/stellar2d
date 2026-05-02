#!/usr/bin/env python3
"""
E7b: analytic-solution spectral ceiling test.

The Exp K benchmark against GYRE's 999-point poly3.txt is limited by
GYRE's internal ~1e-9 precision, not by our Chebyshev discretisation.
To see the TRUE spectral ceiling, we need problems with analytically-known
solutions.

Two tests:

TEST A  Poisson manufactured solution with analytic σ=3 density.
  ρ(r) = (1-r)^3,  π_exact(r) = sin(2π r),
  f(r) = [ρ π_exact']' - k² ρ π_exact  (SymPy, machine-precision).
  Solve with Chebyshev.  Expected: err drops to ~1e-13 around N=40-60,
  then climbs slowly due to roundoff.

TEST B  Analytic Sturm-Liouville EVP.
  Quantum harmonic oscillator on [-L, L]:
     -ψ''(x) + x² ψ = λ ψ
  Eigenvalues: λ_n = 2n+1 (for n=0,1,2,...) in the L→∞ limit.
  For finite L, eigenvalues approach these exponentially.
  Expected: Chebyshev reaches machine precision on the low-n eigenvalues
  for L ~ 10 and N ~ 50.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmode_infra as gi


def cheb_matrices(N, a, b):
    """CGL grid + D1, D2 on [a, b], ascending."""
    D_raw, x_raw = gi.cheb(N)
    scale = 2.0 / (b - a)
    D1 = D_raw * scale
    idx = np.argsort(x_raw)
    x = a + (x_raw[idx] + 1.0) * (b - a) / 2.0
    P = np.zeros_like(D1)
    P[np.arange(N + 1), idx] = 1.0
    D1 = P @ D1 @ P.T
    D2 = D1 @ D1
    return x, D1, D2


def test_A_poisson_manufactured(N_list=(16, 24, 32, 48, 64, 96, 128, 192, 256)):
    """Poisson with ρ=(1-r)^3, π_exact = sin(2πr), SymPy-exact f."""
    print("=" * 72)
    print("TEST A  Poisson manufactured, σ=3 analytic, SymPy-exact forcing")
    print("=" * 72)
    r_sym = sp.Symbol('r', real=True, positive=True)
    rho_sym = (1 - r_sym) ** 3
    pi_sym = sp.sin(2 * sp.pi * r_sym)
    k_x = 2.0
    lhs_sym = sp.diff(rho_sym * sp.diff(pi_sym, r_sym), r_sym) - k_x**2 * rho_sym * pi_sym
    rho_fn = sp.lambdify(r_sym, rho_sym, "numpy")
    pi_fn = sp.lambdify(r_sym, pi_sym, "numpy")
    f_fn = sp.lambdify(r_sym, sp.simplify(lhs_sym), "numpy")

    errs = []
    for N in N_list:
        x, D1, D2 = cheb_matrices(N, 0.0, 1.0)
        rho = rho_fn(x)
        f = f_fn(x)
        pi_exact = pi_fn(x)

        A = D1 @ np.diag(rho) @ D1 - k_x ** 2 * np.diag(rho)
        b = f.copy()
        A[0, :] = 0.0;  A[0, 0] = 1.0;   b[0] = 0.0
        A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0

        pi_num = np.linalg.solve(A, b)
        err = np.max(np.abs(pi_num - pi_exact))
        errs.append(err)
        print(f"  N = {N:4d}  DOF = {N+1:4d}  err = {err:.3e}")
    return np.array(N_list), np.array(errs)


def test_B_harmonic_oscillator(N_list=(16, 24, 32, 48, 64, 96, 128),
                                 L=10.0, n_compare=5):
    """Quantum harmonic oscillator eigenvalue problem on [-L, L].

    -ψ''(x) + x² ψ = λ ψ,  ψ(±L) = 0 (approximate ground state decays).
    Eigenvalues: λ_n = 2n+1 in the L→∞ limit.
    """
    print()
    print("=" * 72)
    print(f"TEST B  Quantum harmonic oscillator on [-{L}, {L}], first {n_compare} eigenvalues")
    print("=" * 72)
    print(f"  Exact (L→∞):  λ_n = 2n+1  for n = 0, 1, 2, ...")

    errs_table = []
    for N in N_list:
        x, D1, D2 = cheb_matrices(N, -L, L)
        # Build operator -D² + x²  on interior only (Dirichlet at ±L)
        M_full = -D2 + np.diag(x ** 2)
        M = M_full[1:-1, 1:-1]   # Strip boundary rows/cols (Dirichlet)
        # Chebyshev D2 is NOT symmetric; must use general eig solver
        eigvals_raw = np.linalg.eigvals(M)
        # Keep finite, positive, real-part eigenvalues
        mask = np.isfinite(eigvals_raw.real) & (eigvals_raw.real > 0) \
             & (np.abs(eigvals_raw.imag) < 1e-8 * (np.abs(eigvals_raw.real) + 1e-30))
        eigvals = np.sort(eigvals_raw[mask].real)
        # Compare first n_compare to 2n+1
        exact = np.array([2 * n + 1.0 for n in range(n_compare)])
        computed = eigvals[:n_compare]
        rel_err = np.abs(computed - exact) / np.abs(exact)
        errs_table.append((N, computed, rel_err))

        print(f"  N = {N:4d}  "
              + "  ".join(f"λ{n}-ref={rel_err[n]:.2e}" for n in range(n_compare)))
    return errs_table


def test_C_analytic_singular_sl(N_list=(16, 24, 32, 48, 64, 96, 128, 192)):
    """Analytic SL eigenvalue test with a weight similar to our problem.

       -d/dx [(1-x²)^(1/2) u'] = λ (1-x²)^(1/2) u,  on (-1, 1).

    This is the Chebyshev-T eigenvalue problem, with eigenvalues
    λ_n = n² and eigenfunctions T_n(x).  A collocation Chebyshev
    solver should converge to machine precision for the first few
    eigenvalues at very low N.

    Actually simpler: the scaled Laplace eigenvalue problem on (-1, 1)
    with Dirichlet,  -u''(x) = λ u(x),  has λ_n = (nπ/2)² for n=1,2,...
    (for half-interval mapping etc.).  Use (0, 1) with Dirichlet for
    cleaner result:
       -u''(x) = λ u(x),  u(0)=u(1)=0  →  λ_n = (nπ)²
    """
    print()
    print("=" * 72)
    print(f"TEST C  Laplacian Dirichlet eigenproblem on [0, 1]")
    print("=" * 72)
    print(f"  Exact:  λ_n = (n π)²  for n = 1, 2, 3, ...")
    n_compare = 5

    errs_table = []
    for N in N_list:
        x, D1, D2 = cheb_matrices(N, 0.0, 1.0)
        M_full = -D2
        M = M_full[1:-1, 1:-1]
        # Chebyshev D2 is NOT symmetric; must use general eig solver
        eigvals_raw = np.linalg.eigvals(M)
        # Keep finite, positive, real-part eigenvalues
        mask = np.isfinite(eigvals_raw.real) & (eigvals_raw.real > 0) \
             & (np.abs(eigvals_raw.imag) < 1e-8 * (np.abs(eigvals_raw.real) + 1e-30))
        eigvals = np.sort(eigvals_raw[mask].real)
        exact = np.array([(n * np.pi) ** 2 for n in range(1, n_compare + 1)])
        computed = eigvals[:n_compare]
        rel_err = np.abs(computed - exact) / np.abs(exact)
        errs_table.append((N, computed, rel_err))
        print(f"  N = {N:4d}  "
              + "  ".join(f"λ{n+1}={rel_err[n]:.2e}" for n in range(n_compare)))
    return errs_table


def main():
    print()
    print(" " * 14 + "SPECTRAL-METHOD ANALYTICAL CEILING TESTS")
    print(" " * 10 + "=" * 50)
    print()

    # TEST A
    N_A, err_A = test_A_poisson_manufactured()

    # TEST B
    errs_B = test_B_harmonic_oscillator()

    # TEST C
    errs_C = test_C_analytic_singular_sl()

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=140)

    # A
    err_A_safe = np.maximum(err_A, 1e-16)
    axes[0].semilogy(N_A, err_A_safe, "C2o-", lw=1.5, ms=7)
    axes[0].set_xlabel("N (Chebyshev order)")
    axes[0].set_ylabel("max |π_num - π_exact|")
    axes[0].set_title("TEST A: Poisson with ρ=(1-r)³, analytic forcing")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].axhline(1e-13, ls="--", color="gray", lw=0.7, label="~ round-off")
    axes[0].legend()
    mask = (N_A >= 16) & (err_A > 0) & (err_A < 1e-2)
    if mask.sum() >= 2:
        slope = np.polyfit(N_A[mask], np.log(err_A[mask]), 1)[0]
        axes[0].set_title(axes[0].get_title() + f"\nsemilog slope = {slope:+.3f}")

    # B
    Ns_B = np.array([r[0] for r in errs_B])
    for n_level in range(5):
        re = np.array([r[2][n_level] for r in errs_B])
        re_safe = np.maximum(re, 1e-16)
        axes[1].semilogy(Ns_B, re_safe, "o-", lw=1.3, ms=6, label=f"λ_{n_level} (= {2*n_level+1})")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("rel err vs 2n+1")
    axes[1].set_title("TEST B: Quantum harmonic oscillator on [-10, 10]")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].axhline(1e-13, ls="--", color="gray", lw=0.7)
    axes[1].legend(fontsize=8)

    # C
    Ns_C = np.array([r[0] for r in errs_C])
    for n_level in range(5):
        re = np.array([r[2][n_level] for r in errs_C])
        re_safe = np.maximum(re, 1e-16)
        axes[2].semilogy(Ns_C, re_safe, "o-", lw=1.3, ms=6, label=f"λ_{n_level+1} = ({n_level+1}π)²")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("rel err vs (nπ)²")
    axes[2].set_title("TEST C: Laplacian Dirichlet eigenproblem on [0, 1]")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].axhline(1e-13, ls="--", color="gray", lw=0.7)
    axes[2].legend(fontsize=8)

    fig.suptitle("Chebyshev spectral method — analytical-solution ceiling tests",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = gi.VID / "spectral_analytical_ceiling.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\n  => {out}")


if __name__ == "__main__":
    main()
