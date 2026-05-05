#!/usr/bin/env python3
"""
E6 v2: HONEST convergence test for Chebyshev spectral Poisson with
singular ρ at the surface.

Previous attempt used f = D1 @ (rho * D1 @ pi_exact) - k² · rho · pi_exact,
which cancels the discretisation error by construction (A π_exact = f
to machine precision).  This gives a trivial "err = machine eps" result
that is not a convergence test.

Correct approach: use an ANALYTIC ρ with surface ρ → (R-r)^σ behaviour,
pick analytic pi_exact, compute f ANALYTICALLY (by hand or SymPy), and
measure err = max |pi_num - pi_exact| on the grid.
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


def cheb_matrices(N, r_lo, r_hi):
    r, D2, w = gi.cheb_on_interval(N, r_lo, r_hi)
    D_raw, x_raw = gi.cheb(N)
    scale = 2.0 / (r_hi - r_lo)
    D1_raw = D_raw * scale
    idx = np.argsort(x_raw)
    P = np.zeros_like(D1_raw)
    P[np.arange(N + 1), idx] = 1.0
    D1 = P @ D1_raw @ P.T
    return r, D1, D2, w


def generate_analytic_forcing(sigma, k_x, k_r_factor=2, R=1.0):
    """Use SymPy to produce callable ρ(r), π(r), f(r) for convergence test.

    ρ(r) = (R - r)^σ · (1 + small smooth stuff) — we use ρ(r) = (1-r)^σ · (1+r)^? ...
    Actually for clean analytic test, pick
        ρ(r) = (1 - r²)^σ_eff       (zero at r = 1 with exponent σ_eff · 2... hmm)
    To match Lane-Emden n=3 surface ρ ~ (R-r)^3, pick σ_eff so that
    (1-r²)^σ_eff ~ 2^σ_eff · (1-r)^σ_eff near r=1.  So σ_eff = σ.
    But at r=0 we have (1-0)^σ = 1, matching Lane-Emden central value.
    """
    r_sym = sp.Symbol('r', real=True, positive=True)
    rho_sym = (R - r_sym) ** sigma           # simple: (1-r)^σ

    pi_sym = sp.sin(k_r_factor * sp.pi * r_sym / R)   # zero at r=0 and r=R

    # f = [ρ π']' - k² ρ π
    lhs_sym = sp.diff(rho_sym * sp.diff(pi_sym, r_sym), r_sym) - k_x**2 * rho_sym * pi_sym
    f_sym = sp.simplify(lhs_sym)

    rho_fn = sp.lambdify(r_sym, rho_sym, "numpy")
    pi_fn = sp.lambdify(r_sym, pi_sym, "numpy")
    f_fn = sp.lambdify(r_sym, f_sym, "numpy")
    return rho_fn, pi_fn, f_fn


def poisson_solve_raw(r, rho, k_x, f):
    """Solve  D1 ρ D1 π - k²ρ π = f  with Dirichlet π(0) = π(R) = 0."""
    N1 = len(r)
    _r, D1, D2, w = cheb_matrices(N1 - 1, r[0], r[-1])
    A = D1 @ np.diag(rho) @ D1 - k_x ** 2 * np.diag(rho)
    b = f.copy()
    A[0, :] = 0.0;  A[0, 0] = 1.0;   b[0] = 0.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0
    return np.linalg.solve(A, b)


def poisson_solve_with_prefactor(r, rho, k_x, f, R, alpha_star):
    """Solve via substitution π = t^α · u, t = R - r.

    Operator on u:  (ρ g u'' + [ρ g' + (ρ g)'] u' + [(ρ g')' - k² ρ g]) u = f
    with g = t^α.
    """
    N1 = len(r)
    _r, D1, D2, w = cheb_matrices(N1 - 1, r[0], r[-1])

    t = R - r
    t_safe = np.where(t > 1e-14, t, 1e-14)
    g = t_safe ** alpha_star
    g_p_analytic = -alpha_star * t_safe ** (alpha_star - 1.0)   # exact derivative

    rho_g = rho * g
    rho_gp = rho * g_p_analytic
    rho_gp_p = D1 @ rho_gp
    rho_g_p  = D1 @ rho_g

    a_uu = rho_g
    a_u  = rho_gp + rho_g_p
    a_0  = rho_gp_p - k_x ** 2 * rho_g

    A = np.diag(a_uu) @ D2 + np.diag(a_u) @ D1 + np.diag(a_0)
    b = f.copy()

    # Scale rows for conditioning
    rs = np.max(np.abs(A), axis=1)
    rs = np.where(rs > 0, rs, 1.0)
    A = A / rs[:, None]
    b = b / rs

    A[0, :] = 0.0;  A[0, 0] = 1.0;   b[0] = 0.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0

    u = np.linalg.solve(A, b)
    return g * u


def run_convergence(sigma, k_x=2.0, k_r_factor=2,
                    N_list=(16, 24, 32, 48, 64, 96, 128),
                    alpha_cases=None, R=1.0):
    rho_fn, pi_fn, f_fn = generate_analytic_forcing(sigma, k_x, k_r_factor, R)

    results = {}
    for alpha_key, label in alpha_cases:
        errs = []
        for N in N_list:
            r, D1, D2, w = cheb_matrices(N, 0.0, R)
            rho = rho_fn(r)
            f = f_fn(r)
            pi_exact = pi_fn(r)
            if alpha_key is None:
                pi_num = poisson_solve_raw(r, rho, k_x, f)
            else:
                pi_num = poisson_solve_with_prefactor(r, rho, k_x, f, R, alpha_key)
            err = np.max(np.abs(pi_num - pi_exact))
            errs.append(err)
        results[label] = (np.array(N_list), np.array(errs))
    return results


def main():
    print("=" * 72)
    print("E6 v2: HONEST spectral Poisson convergence test (analytic forcing)")
    print("=" * 72)
    print()
    print("  ρ(r) = (1-r)^σ   (simplest analytic surface ρ→0 model)")
    print("  π_exact(r) = sin(k_r π r)   (zero at r=0, r=R=1)")
    print("  f(r) computed ANALYTICALLY by SymPy.")
    print()

    for sigma in [3, sp.Rational(3, 2)]:
        print(f"\n  σ = {sigma}  ({'Lane-Emden n=' + str(sigma)})")
        alpha_cases = [
            (None,                    "raw (no prefactor)"),
            (float(1 - sigma / 2),   f"α = 1 - σ/2 = {float(1-sigma/2):+.3f}"),
            (float(1 - sigma),       f"α = 1 - σ   = {float(1-sigma):+.3f}"),
            (float(-sigma / 2),      f"α = -σ/2    = {float(-sigma/2):+.3f}"),
        ]
        results = run_convergence(float(sigma), alpha_cases=alpha_cases,
                                   N_list=(16, 32, 48, 64, 96, 128, 192, 256))
        for lbl, (N_arr, err_arr) in results.items():
            print(f"    {lbl:<30}  N=16: {err_arr[0]:.3e}   N=64: {err_arr[3]:.3e}   N=256: {err_arr[-1]:.3e}")
            mask = N_arr >= 32
            if mask.sum() >= 2 and np.all(err_arr[mask] > 0):
                slope = np.polyfit(N_arr[mask], np.log(err_arr[mask]), 1)[0]
                # Also fit algebraic: log(err) = a + p*log(N)
                if np.all(err_arr[mask] > 0):
                    plog = np.polyfit(np.log(N_arr[mask]), np.log(err_arr[mask]), 1)[0]
                    print(f"      → semilog slope log(err)/N = {slope:+.4f}   "
                          f"algebraic slope d log(err)/d log(N) = {plog:+.3f}")

    # Plot case σ=3 (n=3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    for ax, sigma in zip(axes, [3, 1.5]):
        alpha_cases = [
            (None,                    "raw (no prefactor)"),
            (float(1 - sigma / 2),   f"α = 1 - σ/2"),
            (float(1 - sigma),       f"α = 1 - σ"),
            (float(-sigma / 2),      f"α = -σ/2"),
        ]
        results = run_convergence(float(sigma), alpha_cases=alpha_cases,
                                   N_list=(16, 32, 48, 64, 96, 128, 192, 256))
        for i, (lbl, (N_arr, err_arr)) in enumerate(results.items()):
            err_plot = np.maximum(err_arr, 1e-16)
            ax.semilogy(N_arr, err_plot, "o-", lw=1.3, ms=6, label=lbl)
        ax.set_xlabel("N")
        ax.set_ylabel(r"max $|\pi_{\rm num} - \pi_{\rm exact}|$")
        ax.set_title(f"σ = {sigma}  (ρ = (1-r)^σ)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle("Chebyshev spectral Poisson convergence: raw vs prefactor,\n"
                 "analytic forcing on [0, 1] without cutoff")
    fig.tight_layout()
    out = gi.VID / "spectral_liouville_convergence_v2.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\n  => {out}")


if __name__ == "__main__":
    main()
