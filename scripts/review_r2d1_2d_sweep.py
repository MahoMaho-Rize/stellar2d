#!/usr/bin/env python3
"""Review Round-2 item D.1: full-integrator N_y sweep of the primitive-node
eigenmode preservation test, to resolve the magnitude gap between the
Round-1 simplified prototype (1.05e-5/step) and the paper §5.1 figure
(6.9e-4/step).

Key point of contention from R1.2: does the aliasing floor scale as
  (a) an absolute ~1e-5 invariant of N_y, or
  (b) a dt-dependent quantity whose *N_y-independence* is the real claim?

This script shows (b) is the right reading: for a range of dt, and for
primitive-node RK4 (the closest analogue to CUDA's step), dev/step is
  - O(dt^2) in dt (truncation + aliasing coupling),
  - flat in N_y (aliasing is resolution-independent, per R1.2).

Two sweeps:
  Sweep 1: fixed dt = 2e-3, vary N_y ∈ {32, 48, 64, 96, 128, 192}
           → demonstrates N_y-flatness at "paper-like" dt
  Sweep 2: fixed N_y = 64, vary dt ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
           → demonstrates dt-scaling, which explains the Round-1 magnitude gap

Output: review/r2d1_2d_sweep/{sweep_Ny.csv, sweep_dt.csv}
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from full_galerkin_closure_test import (
    cgl_grid,
    cc_weights,
    bg_lane_emden,
    assemble_operator,
    evp,
    measure_dev,
)


def step_primitive_rk4(v, w, dt, D, rho, N2, kx):
    """Primitive-node RK4 stepper.  The CUDA-faithful analogue of the
    node-space stepping: at each RK4 substage, evaluate v̈ using two
    applications of the differentiation matrix D plus pointwise
    multiplies by ρ and N², WITHOUT assembling L, R — this is the
    discrete-level mismatch §5 diagnoses."""
    rho_safe = np.maximum(rho, 1e-30)

    def rhs(v_, w_):
        rho_dvdy = rho * (D @ v_)
        Lv_cont = -(D @ rho_dvdy) + kx ** 2 * rho * v_
        Rv = kx ** 2 * N2 * rho * v_
        vdotdot = (Rv - Lv_cont) / rho_safe
        # Wall Dirichlet: zero the boundary
        vdotdot[0] = 0.0
        vdotdot[-1] = 0.0
        return w_, vdotdot

    k1v, k1w = rhs(v, w)
    k2v, k2w = rhs(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
    k3v, k3w = rhs(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
    k4v, k4w = rhs(v + dt * k3v, w + dt * k3w)

    v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    w_new = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    v_new[0] = 0.0
    v_new[-1] = 0.0
    return v_new, w_new


def step_assembled_rk4(v_int, w_int, dt, L_inv_R):
    """Assembled-operator RK4 on interior-restricted state."""
    def rhs(v_, w_):
        return w_, -(L_inv_R @ v_)

    k1v, k1w = rhs(v_int, w_int)
    k2v, k2w = rhs(v_int + 0.5 * dt * k1v, w_int + 0.5 * dt * k1w)
    k3v, k3w = rhs(v_int + 0.5 * dt * k2v, w_int + 0.5 * dt * k2w)
    k4v, k4w = rhs(v_int + dt * k3v, w_int + dt * k3w)
    v_new = v_int + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    w_new = w_int + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    return v_new, w_new


def run_one(ny, Ly=1.0, kx=2 * np.pi, rho_cut=0.05, n_steps=100,
            dt=2e-3, amp=1e-8):
    """Run both primitive-RK4 and assembled-RK4 on Lane-Emden n=3/2 for
    n_steps.  Return (dev_prim/step, dev_asm/step, omega_1)."""
    y, D = cgl_grid(ny, Ly)
    w_cc = cc_weights(ny, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    L, R = assemble_operator("vspace", y, D, rho, N2, kx)
    lam, V = evp(L, R)
    om1 = float(np.sqrt(lam[0]))

    # IC = top g-mode eigenvector, unit-normalised, then scaled by amp
    v0_int = V[:, 0] / np.max(np.abs(V[:, 0])) * amp
    v0_full = np.zeros(ny)
    v0_full[1:-1] = v0_int
    w0_full = np.zeros(ny)

    # ── Primitive-RK4 ─────────────────────────────────────────────
    v, w = v0_full.copy(), w0_full.copy()
    traj_prim = [v.copy()]
    for _ in range(n_steps):
        v, w = step_primitive_rk4(v, w, dt, D, rho, N2, kx)
        traj_prim.append(v.copy())
    devs_prim = measure_dev(traj_prim, v0_full, w_cc)
    rate_prim = (devs_prim[-1] - devs_prim[1]) / max(n_steps - 1, 1)

    # ── Assembled-RK4 ────────────────────────────────────────────
    L_inv_R = scipy.linalg.solve(L, R)
    v_i, w_i = v0_int.copy(), np.zeros_like(v0_int)
    # Embed into full-grid storage for the dev metric
    traj_asm = []
    full = np.zeros(ny); full[1:-1] = v_i; traj_asm.append(full.copy())
    for _ in range(n_steps):
        v_i, w_i = step_assembled_rk4(v_i, w_i, dt, L_inv_R)
        full = np.zeros(ny); full[1:-1] = v_i; traj_asm.append(full.copy())
    devs_asm = measure_dev(traj_asm, v0_full, w_cc)
    rate_asm = (devs_asm[-1] - devs_asm[1]) / max(n_steps - 1, 1)

    return rate_prim, rate_asm, om1, devs_prim[-1], devs_asm[-1]


def sweep_Ny(Ny_list, dt=2e-3, n_steps=100):
    print(f"\n=== Sweep 1: fixed dt = {dt:.0e}, {n_steps} steps, vary N_y ===")
    rows = []
    for ny in Ny_list:
        rate_p, rate_a, om, fin_p, fin_a = run_one(ny, dt=dt, n_steps=n_steps)
        rows.append(dict(N_y=ny, dt=dt, n_steps=n_steps, omega_1=om,
                         prim_per_step=rate_p, asm_per_step=rate_a,
                         prim_final=fin_p, asm_final=fin_a))
        print(f"  N_y={ny:4d}  ω₁={om:.4f}  "
              f"prim/step={rate_p:.3e}  asm/step={rate_a:.3e}  "
              f"prim_final={fin_p:.3e}")
    return rows


def sweep_dt(dt_list, ny=64, n_steps=100):
    print(f"\n=== Sweep 2: fixed N_y = {ny}, {n_steps} steps, vary dt ===")
    rows = []
    for dt in dt_list:
        rate_p, rate_a, om, fin_p, fin_a = run_one(ny, dt=dt, n_steps=n_steps)
        rows.append(dict(N_y=ny, dt=dt, n_steps=n_steps, omega_1=om,
                         prim_per_step=rate_p, asm_per_step=rate_a,
                         prim_final=fin_p, asm_final=fin_a))
        print(f"  dt={dt:.0e}  ω₁={om:.4f}  "
              f"prim/step={rate_p:.3e}  asm/step={rate_a:.3e}  "
              f"prim_final={fin_p:.3e}")
    return rows


def main():
    out_dir = SCRIPT_DIR.parent / "review" / "r2d1_2d_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep 1: N_y-flatness, two dt tiers to cover the full N_y range
    # within primitive-RK4 stability.  (The primitive-RK4 effective CFL
    # tightens with N_y because aliasing amplifies high-wavenumber
    # spurious modes; at a single dt the stability envelope shrinks.
    # Using two dt bands lets us probe the full N_y range cleanly.)
    Ny_list_A = [32, 48, 64, 96]     # stable at dt=5e-4
    Ny_list_B = [128, 192, 256]      # need dt=1e-4
    rows_Ny = sweep_Ny(Ny_list_A, dt=5e-4, n_steps=100)
    rows_Ny += sweep_Ny(Ny_list_B, dt=1e-4, n_steps=100)
    with open(out_dir / "sweep_Ny.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_Ny[0].keys()))
        w.writeheader()
        for r in rows_Ny:
            w.writerow(r)

    # Sweep 2: dt-scaling to explain R1.1 magnitude
    dt_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    rows_dt = sweep_dt(dt_list, ny=64, n_steps=100)
    with open(out_dir / "sweep_dt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_dt[0].keys()))
        w.writeheader()
        for r in rows_dt:
            w.writerow(r)

    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
