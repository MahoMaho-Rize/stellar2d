#!/usr/bin/env python3
"""Run Exp K Python solver on the SAME CGL grid and profile that CUDA uses.

Usage:
    ./build/stellar2d --solver anelastic_sl --test gmode_exp_k --nr N ...
       → writes /tmp/ansl_expk/.../gmode_exp_k.csv

Then run this script:
    python scripts/verify_exp_k_cuda.py <N>

It rebuilds the exact CGL grid + polytrope profile CUDA used (via Python
equivalents of build_polytrope_profile_at), runs solve_gmode_full_chebyshev,
and prints the raw ω² list before any classifier filtering.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_poly3_python import build_profile as build_poly3_profile
from gmode_exp_k_chebyshev_full import (
    solve_gmode_full_chebyshev,
    cheb_D1_on_interval,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",  type=int, default=48)
    ap.add_argument("--ell", type=int, default=1)
    ap.add_argument("--inner_cut", type=float, default=1e-4)
    ap.add_argument("--outer_cut", type=float, default=0.9999)
    ap.add_argument("--n_modes", type=int, default=15)
    args = ap.parse_args()

    # CGL grid matching CUDA main.cpp
    x_cheb, _ = cheb_D1_on_interval(args.N, args.inner_cut, args.outer_cut)
    # Sample Python polytrope profile at those nodes (same as CUDA does)
    # Need a "build_profile_at(x_query)" equivalent.  Reuse build_profile to
    # get dense sol, then eval at x_cheb.
    import scipy.integrate

    def rhs(xi, y):
        theta, dtheta = y
        if xi < 1e-12:
            return [dtheta, -theta**3.0 / 3.0]
        tn = np.sign(theta) * np.abs(theta)**3.0 if theta >= 0 else 0.0
        return [dtheta, -2.0/xi * dtheta - tn]
    def ev(xi, y): return y[0]
    ev.terminal = True; ev.direction = -1
    sol = scipy.integrate.solve_ivp(
        rhs, [1e-6, 12.0], [1.0 - 1e-14, 0.0],
        events=ev, max_step=0.005, rtol=1e-12, atol=1e-14,
        dense_output=True,
    )
    xi_1 = sol.t_events[0][0]
    theta_surf, eta_surf = sol.sol(xi_1)
    mtot_fac = xi_1**2 * (-eta_surf)

    xi = x_cheb * xi_1
    theta, eta = sol.sol(xi)
    theta = np.clip(theta, 0.0, None)
    gamma_1 = 5.0/3.0
    V = -(3.0 + 1.0) * xi * eta / np.maximum(theta, 1e-300)
    V_2 = V / (x_cheb ** 2)
    U   = -xi * theta**3.0 / eta
    M_r = xi**2 * (-eta) / mtot_fac
    c_1 = x_cheb**3 / M_r
    A_star = V * (1.0/gamma_1 - 3.0/4.0)
    Gamma_1 = np.full_like(x_cheb, gamma_1)

    print(f"N = {args.N}, DOF = {4*(args.N+1)}")
    print(f"ξ_1 = {xi_1:.10f}")
    print(f"A_star range: [{A_star.min():.3e}, {A_star.max():.3e}]")

    omsq, _ = solve_gmode_full_chebyshev(
        x_cheb, V_2, U, A_star, c_1, Gamma_1,
        args.ell, args.n_modes, p_frac_cutoff=0.05)
    print(f"n_found after classifier = {len(omsq)}")
    print(f"{'n':>3}  {'ω²_python':>20}")
    for i, w in enumerate(omsq):
        print(f"{i+1:>3}  {w:20.12e}")

    # Also try without classifier
    import gmode_exp_k_chebyshev_full as ekc
    # Monkey-patch p_frac_cutoff to 1.0 → keep all
    omsq_all, _ = solve_gmode_full_chebyshev(
        x_cheb, V_2, U, A_star, c_1, Gamma_1,
        args.ell, args.n_modes * 3, p_frac_cutoff=1.01)
    print()
    print("=== classifier disabled ===")
    print(f"n_found = {len(omsq_all)}")
    for i, w in enumerate(omsq_all[:args.n_modes * 2]):
        print(f"{i+1:>3}  {w:20.12e}")


if __name__ == "__main__":
    main()
