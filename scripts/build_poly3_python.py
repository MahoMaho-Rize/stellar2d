#!/usr/bin/env python3
"""Build the reference n=3 polytropic profile with all 5 GYRE structure
coefficients, using Python.  Output: /tmp/poly3_ref.csv

Columns: x  rho  P  M_r  V_2  A_star  U  c_1  Gamma_1
Purpose: C++ build_polytrope_profile() comparison target.

For a polytrope with index n, gamma_1 = 5/3 (distinct from polytrope
index):
   ρ = ρ_c θ^n,           P / P_c = θ^{n+1}
   V(ξ) = -(n+1) ξ (dθ/dξ) / θ             (polytrope identity)
   V_2  = V / x²    with x = ξ/ξ_1
   U    = -ξ θ^n / η                        (standard)
   c_1  = x³ / (M_r / M_tot)
   A*   = V · ( 1/Γ_1 − n/(n+1) )

No external dependencies beyond numpy, scipy.integrate.solve_ivp.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import scipy.integrate


def lane_emden(n_poly: float, xi_max: float = 12.0, rtol: float = 1e-12):
    def rhs(xi, y):
        theta, dtheta = y
        if xi < 1e-12:
            return [dtheta, -theta**n_poly / 3.0]
        tn = np.sign(theta) * np.abs(theta)**n_poly if theta >= 0 else 0.0
        return [dtheta, -2.0/xi * dtheta - tn]

    def event_zero(xi, y): return y[0]
    event_zero.terminal = True; event_zero.direction = -1

    sol = scipy.integrate.solve_ivp(
        rhs, [1e-6, xi_max], [1.0 - 1e-14, 0.0],
        events=event_zero, max_step=0.005, rtol=rtol, atol=1e-14,
        dense_output=True,
    )
    xi_1 = sol.t_events[0][0] if sol.t_events[0].size else sol.t[-1]
    # surface derivative
    theta_surf, eta_surf = sol.sol(xi_1)
    return sol, xi_1, eta_surf


def build_profile(n_poly=3.0, n_pts=5000, gamma_1=5.0/3.0,
                  inner_cut=1e-4, outer_cut=0.9999):
    sol, xi_1, eta_surf = lane_emden(n_poly)
    mtot_fac = xi_1**2 * (-eta_surf)

    x = np.linspace(inner_cut, outer_cut, n_pts)
    xi = x * xi_1
    theta, eta = sol.sol(xi)
    theta = np.clip(theta, 0.0, None)

    rho  = theta**n_poly
    Pres = theta**(n_poly + 1)
    M_r  = (xi**2 * (-eta)) / mtot_fac

    V   = -(n_poly + 1) * xi * eta / np.maximum(theta, 1e-300)
    V_2 = V / (x**2)
    U   = np.where(np.abs(eta) > 1e-30,
                   -xi * rho / eta,       # = -ξ θ^n / η
                   0.0)
    c_1 = x**3 / np.maximum(M_r, 1e-300)
    A_star = V * (1.0/gamma_1 - n_poly/(n_poly + 1.0))
    Gamma_1 = np.full_like(x, gamma_1)

    return dict(x=x, rho=rho, P=Pres, M_r=M_r,
                V_2=V_2, A_star=A_star, U=U, c_1=c_1, Gamma_1=Gamma_1,
                xi_1=xi_1, eta_surf=eta_surf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_poly", type=float, default=3.0)
    ap.add_argument("--n_pts",  type=int, default=5000)
    ap.add_argument("--out",    type=Path, default=Path("/tmp/poly3_ref.csv"))
    args = ap.parse_args()

    p = build_profile(args.n_poly, args.n_pts)
    print(f"Lane-Emden n={args.n_poly}: ξ_1 = {p['xi_1']:.10f}, "
          f"η_surf = {p['eta_surf']:.10f}")

    hdr = "# x rho P M_r V_2 A_star U c_1 Gamma_1"
    arr = np.column_stack([p["x"], p["rho"], p["P"], p["M_r"],
                           p["V_2"], p["A_star"], p["U"],
                           p["c_1"], p["Gamma_1"]])
    np.savetxt(args.out, arr, header=hdr, fmt="%.15e")
    print(f"wrote {args.out} ({len(p['x'])} rows)")

    # Sanity check: print a few key values
    for i in [0, args.n_pts//4, args.n_pts//2, args.n_pts-1]:
        print(f"  x={p['x'][i]:.4f}  V_2={p['V_2'][i]:.4e}  "
              f"U={p['U'][i]:.4e}  c_1={p['c_1'][i]:.4e}  "
              f"A*={p['A_star'][i]:.4e}")


if __name__ == "__main__":
    main()
