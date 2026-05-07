#!/usr/bin/env python3
"""Analyze Athena++ Andrassy runs: merge meshblocks, compute v_rms / y_ub / Ṁ_e.

Reads every set of block{0..N-1}.out1.NNNNN.vtk files in --run-dir, merges
into one global field, applies the SAME diagnostic definitions as our
cart_ale2 scripts (Andrassy Eq. 15 mass-weighted v_rms, ε=1e-3 y_ub
threshold, buffer=0.1 M_e integration).

Usage:
    python3 scripts/andrassy2022/athena_analyze.py \\
        --run-dir runs/athena_andrassy_128_smoke
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "athena/vis/python"))
from athena_read import vtk as athena_vtk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose import find_y_ub, Y_SHIFT, GAMMA


def merge_blocks(files):
    """Merge a list of Athena block VTKs into a single global grid.

    Returns dict: t, rho, P, vx, vy, X1, y_nodes, x_nodes
    """
    blocks = []
    for f in files:
        xf, yf, zf, data = athena_vtk(f)
        # Extract time
        m = re.search(r"time=([\d.eE+-]+)", open(f, "rb").read(200).decode(errors="ignore"))
        t = float(m.group(1)) if m else float('nan')
        vel = np.asarray(data["vel"])   # shape (nz, ny, nx, 3)
        blocks.append({
            "x": xf, "y": yf, "t": t,
            "rho": np.asarray(data["rho"]).squeeze(),
            "press": np.asarray(data["press"]).squeeze(),
            "vel1": vel.squeeze()[..., 0],   # x-component
            "vel2": vel.squeeze()[..., 1],   # y-component
            "r0":   np.asarray(data["r0"]).squeeze() if "r0" in data else None,
        })

    # Determine global extent & block positions
    x_mins = sorted(set(b["x"][0] for b in blocks))
    y_mins = sorted(set(b["y"][0] for b in blocks))
    nbx, nby = len(x_mins), len(y_mins)

    b_nx = blocks[0]["rho"].shape[-1]
    b_ny = blocks[0]["rho"].shape[-2]
    Nx = nbx * b_nx
    Ny = nby * b_ny

    rho = np.zeros((Ny, Nx))
    P   = np.zeros((Ny, Nx))
    vx  = np.zeros((Ny, Nx))
    vy  = np.zeros((Ny, Nx))
    X1  = np.zeros((Ny, Nx))

    for b in blocks:
        bix = x_mins.index(b["x"][0])
        biy = y_mins.index(b["y"][0])
        xslice = slice(bix * b_nx, (bix + 1) * b_nx)
        yslice = slice(biy * b_ny, (biy + 1) * b_ny)
        rho[yslice, xslice] = b["rho"]
        P  [yslice, xslice] = b["press"]
        vx [yslice, xslice] = b["vel1"]
        vy [yslice, xslice] = b["vel2"]
        if b["r0"] is not None:
            X1[yslice, xslice] = b["r0"]

    # Global node coordinates
    x_nodes = np.unique(np.concatenate([b["x"] for b in blocks]))
    y_nodes = np.unique(np.concatenate([b["y"] for b in blocks]))
    return dict(t=blocks[0]["t"], rho=rho, P=P, vx=vx, vy=vy, X1=X1,
                x_nodes=x_nodes, y_nodes=y_nodes,
                nx=Nx, ny=Ny)


def horizontal_profiles(data):
    rho = data["rho"]; P = data["P"]; vx = data["vx"]; vy = data["vy"]
    y_nodes = data["y_nodes"]
    y_c = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    ny, nx = rho.shape

    rho_bar = rho.mean(axis=1)
    P_bar   = P.mean(axis=1)
    M_row = rho.sum(axis=1)
    vx_tilde = (rho * vx).sum(axis=1) / np.maximum(M_row, 1e-30)
    vy_tilde = (rho * vy).sum(axis=1) / np.maximum(M_row, 1e-30)
    sigma2_vx = (rho * (vx - vx_tilde[:, None])**2).sum(axis=1) / np.maximum(M_row, 1e-30)
    sigma2_vy = (rho * (vy - vy_tilde[:, None])**2).sum(axis=1) / np.maximum(M_row, 1e-30)
    v_rms = np.sqrt(sigma2_vx + sigma2_vy)
    A_bar = P_bar / rho_bar**GAMMA
    return dict(y=y_c, rho=rho_bar, P=P_bar, A=A_bar,
                sigma_vx=np.sqrt(sigma2_vx), sigma_vy=np.sqrt(sigma2_vy),
                v_rms=v_rms)


def bulk_stats(prof, y_ub_local, rho_bar, buffer=0.1):
    y = prof["y"]
    mask_conv = y < (y_ub_local - buffer)
    mask_stab = y > (y_ub_local + buffer)
    def _avg(mask):
        if mask.sum() == 0: return float('nan')
        w = rho_bar[mask]
        s2 = prof["sigma_vx"][mask]**2 + prof["sigma_vy"][mask]**2
        return float(np.sqrt((w * s2).sum() / w.sum()))
    return _avg(mask_conv), _avg(mask_stab)


def compute_Me(data, y_ub_local, buffer=0.1):
    x_nodes = data["x_nodes"]
    y_nodes = data["y_nodes"]
    nx = data["nx"]; ny = data["ny"]
    dx = (x_nodes[-1] - x_nodes[0]) / nx
    dy = (y_nodes[-1] - y_nodes[0]) / ny
    dV = dx * dy
    y_c = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    mask_conv = y_c < (y_ub_local - buffer)
    M_e = 0.0
    for jy in range(ny):
        if not mask_conv[jy]: continue
        M_e += data["rho"][jy, :].dot(data["X1"][jy, :]) * dV
    return M_e


# The Athena coord system: y axis is x2, so local-slab y = y_paper.
# But find_y_ub expects slab coords y_slab ∈ [0, 2].  Shift = -Y_SHIFT = -1.
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.run_dir

    block_files = sorted(glob.glob(str(args.run_dir / "*.block*.out1.*.vtk")))
    # Group by timestep index
    by_step = {}
    for f in block_files:
        m = re.search(r"out1\.(\d+)\.vtk", f)
        if m:
            by_step.setdefault(m.group(1), []).append(f)
    steps = sorted(by_step.keys())
    print(f"Found {len(steps)} time steps × {len(by_step[steps[0]])} blocks")

    rows = []
    for i, s in enumerate(steps):
        data = merge_blocks(by_step[s])
        prof = horizontal_profiles(data)
        # y_ub: Athena grid is paper coords y∈[1,3]; diagnose expects slab [0,2]
        # so shift y: make a copy of prof with y -= 1
        prof_slab = dict(prof); prof_slab["y"] = prof["y"] - 1.0
        y_ub_local = find_y_ub(prof_slab)  # local slab coord
        y_ub_paper = y_ub_local + Y_SHIFT   # paper coord
        # Bulk stats
        v_conv, v_stab = bulk_stats(prof, y_ub_paper, prof["rho"])
        # M_e: integrate in paper coords (y > 1)
        M_e = compute_Me(data, y_ub_paper)
        rows.append((data["t"], y_ub_paper, v_conv, v_stab, M_e))
        if i % 5 == 0 or i == len(steps) - 1:
            print(f"  [{i+1}/{len(steps)}] t={data['t']:.2f}  y_ub={y_ub_paper:.3f}  "
                  f"v_rms_conv={v_conv:.4e}  M_e={M_e:.4e}")

    csv = out_dir / "athena_diagnostics.csv"
    with csv.open("w") as f:
        f.write("t,y_ub,v_rms_conv,v_rms_stable,M_e\n")
        for r in rows:
            f.write(f"{r[0]:.6e},{r[1]:.6e},{r[2]:.6e},{r[3]:.6e},{r[4]:.6e}\n")
    print(f"Wrote {csv}")


if __name__ == "__main__":
    raise SystemExit(main())
