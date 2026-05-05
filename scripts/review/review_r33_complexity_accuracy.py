#!/usr/bin/env python3
"""Review Round-3 item P3: complexity-accuracy data collection for
three-method comparison.

For each N_y in {32, 48, 64, 96, 128, 192, 256} and each method
(primitive, τ, assembled), record:
  - per-step eigenmode deviation   (accuracy)
  - setup time                     (one-time cost)
  - per-step time                  (recurring cost)
  - peak memory (working matrices) (cost)

Output: review/r33_complexity/runtime_scan.csv

Intended figure (drawn later): loglog(N_y, per-step-dev) overlaid with
  (N_y, setup_time), (N_y, step_time) on twin y-axis, for each method
  colour-coded.  Aim is ONE figure that gives reviewer the full
  complexity/accuracy picture.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy.linalg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from full_galerkin_closure_test import (
    cgl_grid, cc_weights, bg_lane_emden,
    assemble_operator, evp, measure_dev,
)
from review_r2d1_2d_sweep import step_primitive_rk4, step_assembled_rk4
from review_r32_three_method import (
    assemble_tau_LR, step_tau_rk4, cheb_to_node_mat,
)

OUT_DIR = SCRIPT_DIR.parent / "review" / "r33_complexity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def time_method_primitive(N_y, Ly, kx, rho, N2, v0_full, dt, n_steps):
    y, D = cgl_grid(N_y, Ly)
    # No real "setup" for primitive (D is already computed in cgl_grid,
    # amortised in our convention).  Record cgl_grid + bg_lane_emden time.
    t0 = time.perf_counter()
    # No per-run setup — primitive uses D directly
    setup_t = time.perf_counter() - t0

    v, w = v0_full.copy(), np.zeros_like(v0_full)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        v, w = step_primitive_rk4(v, w, dt, D, rho, N2, kx)
    step_t = (time.perf_counter() - t0) / n_steps

    # Memory: D matrix + misc working arrays ≈ 8 · N_y² bytes
    mem_bytes = 8 * N_y * N_y + 8 * 4 * N_y
    return dict(setup_time=setup_t, step_time=step_t, memory_bytes=mem_bytes,
                v_final=v)


def time_method_tau(N_y, Ly, kx, rho, N2, v0_full, dt, n_steps):
    y, D = cgl_grid(N_y, Ly)
    t0 = time.perf_counter()
    L_tau, R_tau, T, Tinv = assemble_tau_LR(N_y, Ly, rho, N2, kx)
    M_tau = np.linalg.solve(L_tau, R_tau)
    setup_t = time.perf_counter() - t0

    c_v = Tinv @ v0_full
    c_v[-2] = 0.0; c_v[-1] = 0.0
    c_w = np.zeros_like(c_v)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        c_v, c_w = step_tau_rk4(c_v, c_w, dt, M_tau)
    step_t = (time.perf_counter() - t0) / n_steps

    # Memory: M_tau (N_y × N_y) + T + Tinv ≈ 3 · 8 · N_y² bytes
    mem_bytes = 3 * 8 * N_y * N_y + 8 * 4 * N_y
    v_final = T @ c_v
    return dict(setup_time=setup_t, step_time=step_t, memory_bytes=mem_bytes,
                v_final=v_final)


def time_method_assembled(N_y, Ly, kx, rho, N2, v0_full, dt, n_steps):
    y, D = cgl_grid(N_y, Ly)
    t0 = time.perf_counter()
    L, R = assemble_operator("vspace", y, D, rho, N2, kx)
    L_inv_R = scipy.linalg.solve(L, R)
    setup_t = time.perf_counter() - t0

    v_int = v0_full[1:-1].copy()
    w_int = np.zeros_like(v_int)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        v_int, w_int = step_assembled_rk4(v_int, w_int, dt, L_inv_R)
    step_t = (time.perf_counter() - t0) / n_steps

    # Memory: L_inv_R (N_int × N_int) ≈ 8 · (N_y-2)² bytes
    n_int = N_y - 2
    mem_bytes = 8 * n_int * n_int + 8 * 4 * N_y
    v_final = np.zeros(N_y); v_final[1:-1] = v_int
    return dict(setup_time=setup_t, step_time=step_t, memory_bytes=mem_bytes,
                v_final=v_final)


def run_one_Ny(N_y, Ly=1.0, kx=2 * np.pi, rho_cut=0.05, dt=5e-4,
               n_steps=100, amp=1e-8):
    y, D = cgl_grid(N_y, Ly)
    w_cc = cc_weights(N_y, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)

    L, R = assemble_operator("vspace", y, D, rho, N2, kx)
    lam, V = evp(L, R)
    v0_int = V[:, 0] / np.max(np.abs(V[:, 0])) * amp
    v0_full = np.zeros(N_y); v0_full[1:-1] = v0_int

    # Run each method twice: 1st run for timing, 2nd run (longer) for
    # per-step dev measurement.
    results = {}
    for name, fn in [("primitive", time_method_primitive),
                     ("tau",       time_method_tau),
                     ("assembled", time_method_assembled)]:
        # Warmup + timing (short)
        r = fn(N_y, Ly, kx, rho, N2, v0_full, dt, n_steps)
        # Accuracy from a fresh run (same length; we already captured
        # final state, so reconstruct dev as ‖v_final − v0 cos(ωT)‖)
        # For per-step we need the full trajectory — easier: rerun and
        # collect trajectory.
        traj = [v0_full.copy()]
        if name == "primitive":
            y, D = cgl_grid(N_y, Ly)
            v, w = v0_full.copy(), np.zeros_like(v0_full)
            for _ in range(n_steps):
                v, w = step_primitive_rk4(v, w, dt, D, rho, N2, kx)
                traj.append(v.copy())
        elif name == "tau":
            L_tau, R_tau, T, Tinv = assemble_tau_LR(N_y, Ly, rho, N2, kx)
            M_tau = np.linalg.solve(L_tau, R_tau)
            c_v = Tinv @ v0_full; c_v[-2] = 0.0; c_v[-1] = 0.0
            c_w = np.zeros_like(c_v)
            for _ in range(n_steps):
                c_v, c_w = step_tau_rk4(c_v, c_w, dt, M_tau)
                traj.append(T @ c_v)
        else:  # assembled
            L_inv_R = scipy.linalg.solve(L, R)
            v_int = v0_int.copy(); w_int = np.zeros_like(v_int)
            for _ in range(n_steps):
                v_int, w_int = step_assembled_rk4(v_int, w_int, dt, L_inv_R)
                full = np.zeros(N_y); full[1:-1] = v_int
                traj.append(full.copy())
        devs = measure_dev(traj, v0_full, w_cc)
        per_step = (devs[-1] - devs[1]) / max(n_steps - 1, 1)

        results[name] = dict(
            method=name, N_y=N_y,
            setup_time=r["setup_time"],
            step_time=r["step_time"],
            memory_bytes=r["memory_bytes"],
            per_step_dev=float(per_step),
            final_dev=float(devs[-1]),
        )
    return results


def main():
    N_list = [32, 48, 64, 96, 128, 192, 256]
    dt = 5e-4
    n_steps = 100

    all_rows = []
    for N_y in N_list:
        print(f"\n=== N_y = {N_y} ===")
        # Primitive-RK4 can be unstable at higher N_y; adjust dt
        dt_here = dt * (64.0 / N_y) ** 0.5
        res = run_one_Ny(N_y, dt=dt_here, n_steps=n_steps)
        for name, r in res.items():
            r["dt"] = dt_here
            print(f"  {name:10s}: setup={r['setup_time']*1e3:.2f}ms  "
                  f"step={r['step_time']*1e6:.2f}μs  "
                  f"mem={r['memory_bytes']/1024:.1f}kB  "
                  f"dev/step={r['per_step_dev']:.2e}")
            all_rows.append(r)

    with open(OUT_DIR / "runtime_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {OUT_DIR / 'runtime_scan.csv'}")


if __name__ == "__main__":
    main()
