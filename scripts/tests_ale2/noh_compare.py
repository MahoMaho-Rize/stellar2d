#!/usr/bin/env python3
"""Noh 2D implosion compare.

Exact post-shock state (γ=5/3, 2D cylindrical):
    ρ_post = ρ0 · ((γ+1)/(γ-1))² = 16 ρ0
    p_post = ρ_post/3
    shock radius r_sh(t) = t/3      (speed 1/3 from Noh 1987)

We measure ρ inside r_sh/2 (well behind the shock) and compare to 16.

Usage:
    python3 scripts/tests_ale2/noh_compare.py --run-dir runs/noh_256 --t 2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import parse_vtk, latest_vtk, cell_centers, pass_fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--t",   type=float, required=True)
    ap.add_argument("--rho0", type=float, default=1.0)
    ap.add_argument("--tol-rel", type=float, default=0.15,
                    help="pass tolerance on |ρ_avg - 16|/16 inside probe disk")
    args = ap.parse_args()

    vtk = latest_vtk(args.run_dir)
    data = parse_vtk(vtk)
    xc, yc = cell_centers(data)
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")

    Lx = float(data["x_nodes"][-1] - data["x_nodes"][0])
    Ly = float(data["y_nodes"][-1] - data["y_nodes"][0])
    xc0, yc0 = 0.5 * Lx, 0.5 * Ly
    R = np.sqrt((Xc - xc0) ** 2 + (Yc - yc0) ** 2)

    r_sh = args.t / 3.0      # Noh shock speed = 1/3
    probe = R < 0.5 * r_sh
    if probe.sum() < 4:
        print(f"  FAIL  probe disk empty (r_sh={r_sh:.3f} too small for this grid)")
        return 1

    rho_post_sim  = float(data["rho"][probe].mean())
    rho_post_th   = 16.0 * args.rho0
    rel_err = abs(rho_post_sim - rho_post_th) / rho_post_th

    print(f"Noh @ t={args.t}  grid nx={len(xc)}×ny={len(yc)}")
    print(f"  ρ_post sim  = {rho_post_sim:.3f}   (probe r < r_sh/2 = {0.5*r_sh:.3f})")
    print(f"  ρ_post theor = {rho_post_th:.3f}")
    print(f"  rel err     = {100*rel_err:.2f}%")
    ok = pass_fail("|Δρ_post|/ρ_post", rel_err, args.tol_rel)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
