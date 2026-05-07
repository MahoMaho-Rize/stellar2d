#!/usr/bin/env python3
"""Sedov 2D cylindrical blast compare.

Tests the self-similar shock-radius scaling:
    r_sh(t) = ξ₀ (E₀/ρ₀)^{1/4} · t^{1/2}        (2D cylindrical, γ=1.4)

with ξ₀ ≈ 1.0 (Kamm 2007, LA-UR-07-2849 Table 6 gives ξ₀ = 1.0 for γ=1.4,
ν=2). We locate the shock as the outermost radius where ρ(r) exceeds
(1+α)·ρ₀ with α=0.2 — this ignores the floor pedestal and picks the shock
jump, which is ρ_post/ρ₀ = (γ+1)/(γ-1) = 6 for γ=1.4.

Usage:
    python3 scripts/tests_ale2/sedov_compare.py \\
        --run-dir runs/sedov_256 --t 0.05 --E 1 --rho 1
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
    ap.add_argument("--t",  type=float, required=True, help="simulation time of the analyzed frame")
    ap.add_argument("--E",  type=float, default=1.0, help="total deposited energy E0")
    ap.add_argument("--rho", type=float, default=1.0, help="ambient density ρ0")
    ap.add_argument("--xi0", type=float, default=1.0, help="Kamm ξ0 (γ=1.4, ν=2)")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="shock detector: outermost r where ρ ≥ (1+α)·ρ0. Absolute"
                         " threshold; sensitive to resolution but widely used in"
                         " ALE Sedov regression. Default 20% excess.")
    ap.add_argument("--tol-rel", type=float, default=0.10,
                    help="pass tolerance on |r_sh_sim - r_sh_th|/r_sh_th."
                         " ALE/Caramana typically gets 3-10% here. Default 10%.")
    args = ap.parse_args()

    vtk = latest_vtk(args.run_dir)
    data = parse_vtk(vtk)
    xc, yc = cell_centers(data)
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")   # rho is (ny, nx)

    Lx = float(data["x_nodes"][-1] - data["x_nodes"][0])
    Ly = float(data["y_nodes"][-1] - data["y_nodes"][0])
    xc0, yc0 = 0.5 * Lx, 0.5 * Ly
    R = np.sqrt((Xc - xc0) ** 2 + (Yc - yc0) ** 2)
    rho = data["rho"]

    # Outermost radius where ρ crosses a fixed excess threshold.
    mask = rho >= (1.0 + args.alpha) * args.rho
    if not mask.any():
        print("  FAIL  no cells above detector threshold — shock not found")
        return 1
    r_sh_sim = float(R[mask].max())
    rho_peak = float(rho.max())

    r_sh_th = args.xi0 * (args.E / args.rho) ** 0.25 * args.t ** 0.5
    rel_err = abs(r_sh_sim - r_sh_th) / r_sh_th

    print(f"Sedov @ t={args.t}  grid nx={len(xc)}×ny={len(yc)}")
    print(f"  r_sh sim   = {r_sh_sim:.4f}")
    print(f"  r_sh theor = {r_sh_th:.4f}  (ξ₀={args.xi0}, 2D cyl)")
    print(f"  rel err    = {100*rel_err:.2f}%")
    print(f"  ρ_peak sim = {rho_peak:.2f}   (theor 6.0 for γ=1.4)")
    ok = pass_fail("|Δr_sh|/r_sh", rel_err, args.tol_rel)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
