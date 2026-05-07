#!/usr/bin/env python3
"""Yee-Vinokur-Djomehri isentropic vortex round-trip compare.

Domain [0,10]² with periodic BC (centred on (5,5) by init_yee_vortex).
Carrier flow (u,v) = (1,1), so after t=10 the vortex has traveled exactly
once around in each direction and should return to the initial state.
We compare ρ_sim(t=10) against the IC ρ profile — L1/L∞ of the difference
measures numerical diffusion through the remap.

Usage:
    python3 scripts/tests_ale2/yee_compare.py --run-dir runs/yee_256 --t 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import parse_vtk, latest_vtk, cell_centers, pass_fail


def rho_exact(x: np.ndarray, y: np.ndarray, gamma: float = 1.4,
              beta: float = 5.0, xc0: float = 5.0, yc0: float = 5.0) -> np.ndarray:
    dx = x - xc0
    dy = y - yc0
    r2 = dx * dx + dy * dy
    T  = 1.0 - (gamma - 1.0) * beta * beta \
        / (8.0 * gamma * np.pi * np.pi) * np.exp(1.0 - r2)
    return np.power(T, 1.0 / (gamma - 1.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--t", type=float, required=True,
                    help="time of analyzed frame (expect integer domain periods)")
    ap.add_argument("--tol-L1", type=float, default=0.04,
                    help="pass tolerance on L1(ρ − ρ_IC). Measured 2.7-3.0% on"
                         " 128²/256² with 1st-order rebuild (default stable config);"
                         " ~1% would require higher-order rebuild which needs further"
                         " work to stabilise on strong vortex cores (see TODO).")
    args = ap.parse_args()

    vtk = latest_vtk(args.run_dir)
    data = parse_vtk(vtk)
    xc, yc = cell_centers(data)
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")

    rho_sim = data["rho"]
    rho_ic  = rho_exact(Xc, Yc)

    err = np.abs(rho_sim - rho_ic)
    L1 = float(err.mean())
    Linf = float(err.max())
    print(f"Yee vortex @ t={args.t}  grid nx={len(xc)}×ny={len(yc)}")
    print(f"  L1(ρ−ρ_IC)   = {L1:.4e}")
    print(f"  Linf(ρ−ρ_IC) = {Linf:.4e}")
    print(f"  ρ_min = {float(rho_sim.min()):.3f}   ρ_max = {float(rho_sim.max()):.3f}")
    ok = pass_fail("L1 diffusion", L1, args.tol_L1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
