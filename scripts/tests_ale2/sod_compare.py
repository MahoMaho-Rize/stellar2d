#!/usr/bin/env python3
"""Sod shock tube compare: y-averaged profile from cart_ale2 vs exact solution.

Test case: cfg.test_case = "sod"
Expected: IC ρL=1, pL=1 | ρR=0.125, pR=0.1 at x=0.5, γ=1.4, v=0.
Measure at t_end (default 0.2): L1 error on ρ(x) between simulation (x-slice
taken as y-average of the 2D field) and exact 1D Riemann solution.

Usage:
    python3 scripts/tests_ale2/sod_compare.py --run-dir runs/sod_256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import parse_vtk, latest_vtk, cell_centers, pass_fail


def sod_exact(x: np.ndarray, t: float, gamma: float = 1.4,
              x0: float = 0.5,
              rhoL: float = 1.0, pL: float = 1.0,
              rhoR: float = 0.125, pR: float = 0.1) -> np.ndarray:
    """Exact Sod rho(x, t) — closed-form Toro/Sod (1978) construction."""
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    # Pressure in star region: solve (p*-pR) f(p*) with shock+rarefaction forms.
    def f_L(p):
        if p > pL:  # shock
            A = 2.0 / ((gamma + 1.0) * rhoL)
            B = (gamma - 1.0) / (gamma + 1.0) * pL
            return (p - pL) * np.sqrt(A / (p + B))
        else:       # rarefaction
            return 2.0 * cL / (gamma - 1.0) * ((p / pL) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    def f_R(p):
        if p > pR:
            A = 2.0 / ((gamma + 1.0) * rhoR)
            B = (gamma - 1.0) / (gamma + 1.0) * pR
            return (p - pR) * np.sqrt(A / (p + B))
        else:
            return 2.0 * cR / (gamma - 1.0) * ((p / pR) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    def F(p):
        return f_L(p) + f_R(p) + 0.0   # Δu = 0 at IC

    # Bracket and bisect on p*.
    plo, phi = 1e-6, 10.0 * max(pL, pR)
    for _ in range(200):
        pm = 0.5 * (plo + phi)
        if F(pm) > 0:
            phi = pm
        else:
            plo = pm
    p_star = 0.5 * (plo + phi)
    u_star = 0.5 * (f_R(p_star) - f_L(p_star))

    rho = np.empty_like(x)
    for i, xi in enumerate(x):
        s = (xi - x0) / t if t > 0 else 0.0
        if s < u_star:   # left of contact
            if p_star > pL:
                # left shock
                S_L = cL * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / pL
                                  + (gamma - 1.0) / (2.0 * gamma))
                if s < -S_L:  # ambient pre-shock... but we started with u=0
                    rho[i] = rhoL
                else:
                    rho[i] = rhoL * ((p_star / pL + (gamma - 1.0) / (gamma + 1.0))
                                    / ((gamma - 1.0) / (gamma + 1.0) * p_star / pL + 1.0))
            else:
                # left rarefaction fan
                aL_star = cL - (gamma - 1.0) / 2.0 * (-u_star)
                c_fan_head = -cL
                c_fan_tail = u_star - aL_star
                if s < c_fan_head:
                    rho[i] = rhoL
                elif s > c_fan_tail:
                    rho[i] = rhoL * (p_star / pL) ** (1.0 / gamma)
                else:
                    # inside fan
                    u_fan = 2.0 / (gamma + 1.0) * (cL + (gamma - 1.0) / 2.0 * 0.0 + s)
                    c_fan = 2.0 / (gamma + 1.0) * (cL + (gamma - 1.0) / 2.0 * (0.0 - s))
                    rho[i] = rhoL * (c_fan / cL) ** (2.0 / (gamma - 1.0))
        else:            # right of contact
            if p_star > pR:
                # right shock
                S_R = cR * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / pR
                                  + (gamma - 1.0) / (2.0 * gamma))
                if s > S_R:
                    rho[i] = rhoR
                else:
                    rho[i] = rhoR * ((p_star / pR + (gamma - 1.0) / (gamma + 1.0))
                                    / ((gamma - 1.0) / (gamma + 1.0) * p_star / pR + 1.0))
            else:
                # right rarefaction
                aR_star = cR - (gamma - 1.0) / 2.0 * u_star
                c_fan_head = cR
                c_fan_tail = u_star + aR_star
                if s > c_fan_head:
                    rho[i] = rhoR
                elif s < c_fan_tail:
                    rho[i] = rhoR * (p_star / pR) ** (1.0 / gamma)
                else:
                    c_fan = 2.0 / (gamma + 1.0) * (cR - (gamma - 1.0) / 2.0 * (0.0 - s))
                    rho[i] = rhoR * (c_fan / cR) ** (2.0 / (gamma - 1.0))
    return rho


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--t", type=float, default=0.2, help="time of the analyzed frame")
    ap.add_argument("--tol-L1", type=float, default=0.08,
                    help="pass tolerance on rho L1 (default 8% of |Δρ|). ALE"
                         " swept-remap: ~7% at 128², ~5% at 256², ~4% at 512²"
                         " on this Sod IC.")
    args = ap.parse_args()

    vtk = latest_vtk(args.run_dir)
    data = parse_vtk(vtk)
    xc, yc = cell_centers(data)
    rho = data["rho"]                      # (ny, nx)
    rho_1d = rho.mean(axis=0)              # y-average → (nx,)
    rho_exact = sod_exact(xc, args.t)

    L1 = float(np.mean(np.abs(rho_1d - rho_exact)))
    Linf = float(np.max(np.abs(rho_1d - rho_exact)))
    dRho = 1.0 - 0.125
    print(f"Sod @ t={args.t}  grid nx={len(xc)}")
    print(f"  L1(ρ)   = {L1:.4e}   ({100*L1/dRho:.2f}% of |Δρ|)")
    print(f"  Linf(ρ) = {Linf:.4e}  ({100*Linf/dRho:.2f}% of |Δρ|)")
    ok = pass_fail("L1(ρ)/|Δρ|", L1 / dRho, args.tol_L1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
