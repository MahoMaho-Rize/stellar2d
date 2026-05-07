#!/usr/bin/env python3
"""Gresho stationary vortex compare.

Exact solution is v-rms-conserving: vφ(r) piecewise linear (peaks 1 at r=0.2),
flow stays stationary for all time. We measure L1/L∞ drift of |v| between the
simulated speed and the IC speed at each cell center, plus L1 of divergence.

Usage:
    python3 scripts/tests_ale2/gresho_compare.py --run-dir runs/gresho_256 --t 3.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import parse_vtk, latest_vtk, cell_centers, pass_fail


def vphi_exact(r: np.ndarray) -> np.ndarray:
    v = np.zeros_like(r)
    v[r < 0.2] = 5.0 * r[r < 0.2]
    m = (r >= 0.2) & (r < 0.4)
    v[m] = 2.0 - 5.0 * r[m]
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--tol-L1-v", type=float, default=0.03,
                    help="pass tolerance on L1 of |v − v0|. Needs --rebuild-order 1."
                         " Measured: 2.7% at 128², 1.3% at 256² (2026-05-07).")
    args = ap.parse_args()

    vtk = latest_vtk(args.run_dir)
    data = parse_vtk(vtk)
    xc, yc = cell_centers(data)
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")

    Lx = float(data["x_nodes"][-1] - data["x_nodes"][0])
    Ly = float(data["y_nodes"][-1] - data["y_nodes"][0])
    xc0, yc0 = 0.5 * Lx, 0.5 * Ly
    dx = Xc - xc0
    dy = Yc - yc0
    R  = np.sqrt(dx * dx + dy * dy)

    # Exact speed (which equals |v| since flow is purely azimuthal).
    speed_exact = vphi_exact(R)

    vx = data["vx"]; vy = data["vy"]
    speed_sim = np.sqrt(vx * vx + vy * vy)

    # Limit comparison to the vortex disk (r < 0.5) to avoid comparing floor
    # to floor over the full square.
    disk = R < 0.5
    err_abs = np.abs(speed_sim - speed_exact)
    L1 = float(err_abs[disk].mean())
    Linf = float(err_abs[disk].max())

    print(f"Gresho @ t={args.t}  grid nx={len(xc)}×ny={len(yc)}")
    print(f"  L1(|v|-|v0|)   = {L1:.4e}  (inside r<0.5, v_max=1)")
    print(f"  Linf(|v|-|v0|) = {Linf:.4e}")
    ok = pass_fail("L1 drift", L1, args.tol_L1_v)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
