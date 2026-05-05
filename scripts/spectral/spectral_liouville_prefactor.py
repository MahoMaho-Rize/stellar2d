#!/usr/bin/env python3
"""
E6: Chebyshev + α★ prefactor spectral solver for the reduced-pressure
    Liouville Poisson problem on a Lane-Emden polytrope.

Problem (from reduced_pressure_liouville.md §3):

    d/dr [ρ₀(r) dπ̂/dr] - k_x² ρ₀(r) π̂ = f̂(r),    r ∈ [0, R]

where ρ₀(r) is the Lane-Emden polytropic density profile with surface
behaviour ρ₀ ~ c·(R - r)^σ at the surface r = R.  σ = 3 for n=3,
σ = 3/2 for n=3/2.

E5 showed that the substitution chain

    π̂(r) = t^{α★} · ρ₀(r)^{-1/2} · u(r),    α★ = 1 - σ/2,  t = R - r

yields  ρ₀ · t^{2α★} = c · t²,  and transforms the equation to the
**constant-coefficient** form

    u''(r) + W_eff(r) u - k_x² u = g(r)

with W_eff ≡ 0 in the leading power (i.e. the t⁻² singularity at r = R
is fully cancelled).  Subleading corrections from the finite-r
Lane-Emden θ(ξ) deviation from pure-power are smooth and smaller by O(t).

Test strategy (E6a):
  1. Manufactured solution.  Pick u_exact(r) that is smooth throughout
     [0, R] and has no endpoint singularities (e.g. u_exact(r) = sin(k_r r)
     on [0, R] with Dirichlet at both ends — sine vanishes at r=0, set
     R so sin(k_r R) = 0 as well, i.e. k_r R = n π).
  2. Compute f = π̂-side forcing that produces this u via the transform.
  3. Discretise u'' - k_x² u = g on CGL grid with Dirichlet at r=0 and r=R.
  4. Measure error ||u_num - u_exact||_∞ vs N.  Expect exponential decay
     (err ~ exp(-c N)) since the transformed equation is smooth and there
     is no cutoff.

Comparison target:
  Phase 0 ext E3 showed algebraic N^{-2.4} without prefactor (cutoff ρ > 0.01).
  With α★ = 1 - σ/2 we expect exponential convergence at full domain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.integrate
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmode_infra as gi

REF_DOC = "docs/spectral_liouville_plan_2026-05-03.md"
SCRIPT_REL = "scripts/spectral_liouville_prefactor.py"


def lane_emden_rho_profile(n_poly, R_target=None, n_pts=4000):
    """Lane-Emden polytrope of index n_poly.  Returns (r, rho) on [0, R]
    where R = xi_1/xi_1 = 1 (unit radius).  rho = theta^n_poly.
    """
    xi, theta, xi_1 = gi.solve_lane_emden(n=n_poly, n_pts=n_pts)
    r = xi / xi_1                                    # r ∈ [0, 1]
    rho = np.where(theta > 0, theta ** n_poly, 0.0)
    return r, rho, xi_1


def cheb_matrices(N, r_lo, r_hi):
    """Chebyshev-Gauss-Lobatto grid on [r_lo, r_hi] with 2nd-derivative matrix.

    Returns (r, D1, D2, w) — r ascending, D1/D2 acting on the FULL grid
    (endpoints included), w Clenshaw-Curtis weights.
    """
    r, D2, w = gi.cheb_on_interval(N, r_lo, r_hi)
    # D1 by applying D twice idea; easier: get raw cheb D from gi.cheb
    D_raw, x_raw = gi.cheb(N)
    scale = 2.0 / (r_hi - r_lo)
    D1_raw = D_raw * scale
    # Reorder to ascending to match r ordering (gi.cheb_on_interval uses argsort)
    idx = np.argsort(x_raw)
    P = np.zeros_like(D1_raw)
    P[np.arange(N + 1), idx] = 1.0
    D1 = P @ D1_raw @ P.T
    return r, D1, D2, w


def poisson_solve_spectral_transformed(r, rho, k_x, f, R, sigma, alpha_star):
    """Solve the Poisson problem

        [rho(r) · pi'(r)]' - k_x² · rho(r) · pi(r) = f(r),   pi(0) = pi(R) = 0

    by discretising DIRECTLY in pi on the Chebyshev grid.

    This is the "raw discretisation" — no α★ prefactor applied, serving
    as the baseline.  The prefactor version is `poisson_solve_with_prefactor`.
    """
    r = np.asarray(r, dtype=float)
    N1 = len(r)
    _r, D1, D2, w = cheb_matrices(N1 - 1, r[0], r[-1])
    assert np.allclose(_r, r, atol=1e-12)

    # Self-adjoint form: A pi = [rho · D1] @ D1 @ pi + ... but it's easier
    # to just assemble A = D1 @ diag(rho) @ D1 - k_x² · diag(rho)
    A = D1 @ np.diag(rho) @ D1 - k_x ** 2 * np.diag(rho)

    b = f.copy()

    # Dirichlet BCs: overwrite first and last rows
    A[0, :] = 0.0;  A[0, 0] = 1.0;  b[0] = 0.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0

    pi_hat = np.linalg.solve(A, b)
    return pi_hat, pi_hat


def poisson_solve_with_prefactor(r, rho, k_x, f, R, sigma, alpha_star):
    """Solve the same Poisson problem via the E5 prefactor transformation

        pi(r) = t^{alpha_star} · u(r),    t = R - r

    (note: we do NOT divide by sqrt(rho) in this implementation; the E5
    analysis showed that the KEY substitution is π = t^α · (something),
    and the Liouville sqrt(rho) can be applied separately or together.
    Here we stick with pure t^α for clarity and numerical stability.
    The resulting equation for u:

        [rho · (t^α u)']' - k² rho · t^α u = f
     ⇒  [rho · (α t^{α-1} u + t^α u')]' - k² rho t^α u = f
     ⇒  ... messy.

    Simpler: substitute pi = t^α · u.  Since rho ~ c·t^σ near surface,
    the coefficient rho·t^α ~ c·t^{σ+α} ~ c·t^2 (for α = 1-σ/2, but we
    want σ+2α=2 which means α = 1-σ/2.  Wait actually σ+α ≠ σ+2α!).
    Let me recompute: the operator is (rho pi')', so the leading
    singular factor rho ~ t^σ multiplies pi' first.  To make the
    operator non-degenerate we want rho · t^α · [...] to behave like
    a constant at t=0.  Taking pi = t^α · u gives pi' = α t^{α-1} u +
    t^α u', so rho·pi' = c t^σ (α t^{α-1} u + t^α u') = α c t^{σ+α-1} u
    + c t^{σ+α} u'.  For this to have leading non-singular
    integrable behaviour at t=0, pick σ+α such that the LEADING term is
    finite and differentiable.  The standard choice for "regular
    Frobenius" is the larger root, which for our operator structure is
    α = 1 - σ (the OTHER Frobenius branch, not 1 - σ/2 !).

    Hmm — my E5 derivation was for the Liouville-substituted Schrödinger
    form, not the raw operator.  Let me just do this numerically:
    try α = 1 - σ  vs  α = 1 - σ/2  vs α = -σ/2 and see which gives
    exponential convergence.

    **For now (E6 first pass), we implement α = 1 - σ (the "simple Frobenius
    regular branch"):**  pi = t^{1-σ} · u.  For n=3 (σ=3), α = -2.
    For n=1.5 (σ=3/2), α = -1/2.

    We'll test multiple α values in the convergence sweep.
    """
    r = np.asarray(r, dtype=float)
    N1 = len(r)
    _r, D1, D2, w = cheb_matrices(N1 - 1, r[0], r[-1])
    assert np.allclose(_r, r, atol=1e-12)

    t = R - r
    t_safe = np.where(t > 1e-14, t, 1e-14)

    # Prefactor g(r) = t^alpha_star.  (alpha_star passed in; the outer code
    # tries different values.)
    g = t_safe ** alpha_star
    g_p = D1 @ g

    # pi = g · u, so pi' = g' u + g u'.
    # rho pi' = rho g' u + rho g u'.
    # (rho pi')' = (rho g')' u + rho g' u' + (rho g)' u' + rho g u''
    #            = (rho g')' u + [rho g' + (rho g)'] u' + rho g u''
    rho_g = rho * g
    rho_gp = rho * g_p
    rho_gp_p = D1 @ rho_gp      # (rho g')'
    rho_g_p  = D1 @ rho_g       # (rho g)'

    a_uu = rho_g
    a_u  = rho_gp + rho_g_p
    a_0  = rho_gp_p - k_x ** 2 * rho_g

    A = np.diag(a_uu) @ D2 + np.diag(a_u) @ D1 + np.diag(a_0)

    # Normalise rows to unit scale to improve conditioning (each row / max|row|).
    # Note: must scale RHS same way.
    b = f.copy()
    row_scales = np.max(np.abs(A), axis=1)
    row_scales = np.where(row_scales > 0, row_scales, 1.0)
    A = A / row_scales[:, None]
    b = b / row_scales

    # BCs: u(r=0) and u(r=R) controlled via Dirichlet on pi.
    # pi(0) = g(0) · u(0), and g(0) = R^alpha > 0, so u(0) = 0.
    # pi(R) = g(R) · u(R) = 0 · u(R) (for alpha_star > 0) or ∞ · u(R) (alpha<0).
    # For alpha_star < 0 (n=3 case) we impose u(R) = 0 directly so that the
    # product g·u at R is resolved as limit 0 (regular solution).
    A[0, :] = 0.0;  A[0, 0] = 1.0;   b[0] = 0.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0

    u = np.linalg.solve(A, b)
    # Reconstruct pi = g · u
    pi_hat = g * u
    # Endpoints: pi(0) = g(0)·0 = 0; at r=R, g is huge but u=0 so product is 0.
    pi_hat[0] = 0.0
    pi_hat[-1] = 0.0
    return u, pi_hat


def manufactured_solution_test(n_poly=3, N_list=(16, 24, 32, 48, 64, 96, 128), k_x=2.0,
                                k_r_factor=2, alpha_choice=None, label="no-prefactor"):
    """Manufactured-solution convergence test.

    Pick pi_exact(r) = sin(k_r r) — smooth, zero at r=0 and r=R=1 when
    k_r = k_r_factor · π.  Compute f = [ρ π_exact']' - k_x² ρ π_exact
    on the CGL grid, then solve the discretised system and compare.

    alpha_choice:  None  → raw discretisation (no prefactor)
                   float → apply pi = t^alpha_choice · u transform
    """
    sigma = n_poly
    R = 1.0
    r_le, rho_le, _ = lane_emden_rho_profile(n_poly)

    errors = []
    for N in N_list:
        r, D1, D2, w = cheb_matrices(N, 0.0, R)
        rho = np.interp(r, r_le, rho_le)
        k_r = k_r_factor * np.pi / R
        pi_exact = np.sin(k_r * r)

        # Compute f = [rho pi_exact']' - k_x² rho pi_exact on the SAME grid
        pi_p_exact = D1 @ pi_exact
        rho_pi_p = rho * pi_p_exact
        f = D1 @ rho_pi_p - k_x ** 2 * rho * pi_exact

        if alpha_choice is None:
            pi_num, _ = poisson_solve_spectral_transformed(r, rho, k_x, f, R, sigma, 0.0)
        else:
            _, pi_num = poisson_solve_with_prefactor(r, rho, k_x, f, R, sigma, alpha_choice)

        err = np.max(np.abs(pi_num - pi_exact))
        errors.append(err)
        print(f"  [{label:<25}] N = {N:4d}:  max |err| = {err:.3e}")
    return np.array(N_list), np.array(errors)


def main():
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" E6: Chebyshev + α★ prefactor — manufactured solution convergence")
    print("=" * 72)
    print()

    n_poly = 3
    sigma = n_poly
    N_list_base = (16, 24, 32, 48, 64, 96, 128)
    k_x = 2.0
    k_r_factor = 2

    # Compare multiple α choices against the raw no-prefactor baseline.
    # α = 0          : no prefactor (baseline, expect algebraic)
    # α = 1 - σ/2    : E5 derivation (regularises Liouville-form Schrödinger potential)
    # α = 1 - σ      : Frobenius "regular branch" of the raw operator
    # α = -σ/2       : the OTHER Liouville root
    print(f"  Lane-Emden n={n_poly} (σ={sigma}).  Domain [0, R=1], k_x={k_x}.")
    print(f"  Manufactured pi_exact = sin({k_r_factor}π r).")
    print()

    alpha_cases = [
        (None,                   "raw no-prefactor"),
        (0.0,                    "α = 0 (trivial prefactor)"),
        (1.0 - sigma / 2.0,      f"α = 1 - σ/2 = {1.0 - sigma/2.0}"),
        (1.0 - sigma,            f"α = 1 - σ   = {1.0 - sigma}"),
        (-sigma / 2.0,           f"α = -σ/2    = {-sigma/2.0}"),
        (2.0,                    "α = +2 (surface-quiet)"),
    ]

    results = {}
    for alpha, label in alpha_cases:
        print(f"\n  ── {label} ──")
        N_arr, err_arr = manufactured_solution_test(
            n_poly=n_poly, N_list=N_list_base, k_x=k_x, k_r_factor=k_r_factor,
            alpha_choice=alpha, label=label[:24]
        )
        results[label] = (N_arr, err_arr)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5), dpi=140)
    colors = plt.cm.tab10.colors
    for i, (label, (N_arr, err_arr)) in enumerate(results.items()):
        # Clamp near-zero floor
        err_plot = np.maximum(err_arr, 1e-16)
        ax.semilogy(N_arr, err_plot, "o-", lw=1.3, ms=6, color=colors[i % 10],
                    label=label)
    ax.axhline(1e-10, ls="--", color="gray", lw=0.7)
    ax.axhline(1e-2,  ls=":", color="red", lw=0.7, label="1% floor")
    ax.set_xlabel("N (Chebyshev order)")
    ax.set_ylabel(r"max $|\pi_\mathrm{num} - \pi_\mathrm{exact}|$")
    ax.set_title(
        fr"Poisson spectral convergence, Lane-Emden $n={n_poly}$ ($\sigma={sigma}$), "
        r"prefactor sweep"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")

    out = gi.VID / "spectral_liouville_poisson_convergence.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"\n  => {out}")

    # Print slopes
    print("\n  Convergence slope summary (log(err)/N, fit on N ≥ 32):")
    print("  " + "-" * 60)
    for label, (N_arr, err_arr) in results.items():
        mask = N_arr >= 32
        if mask.sum() >= 2 and np.all(err_arr[mask] > 0):
            slope = np.polyfit(N_arr[mask], np.log(err_arr[mask]), 1)[0]
            print(f"    {label:<32}  slope = {slope:+.4f}")
        else:
            print(f"    {label:<32}  (err too small or too few points)")


if __name__ == "__main__":
    main()
