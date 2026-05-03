#!/usr/bin/env python3
"""Review Round-3 item P1: Proposition 1 — the *time-stepping* defect
between primitive-node RK4 and assembled RK4, measured as the per-step
leakage of the top g-mode eigenvector, does NOT vanish as N_y → ∞ on
half-integer polytropic backgrounds; it DOES vanish algebraically on
integer polytropic backgrounds.

The defect being measured is NOT the operator-residual
  ‖(M_prim − ω²) v_n‖  (misleading, since CUDA uses a pressure-projection
  pipeline that approximates L⁻¹ indirectly, not diag(1/ρ))
but rather the per-step *time evolution* leakage
  δ_N := ‖RK4[step_prim](v_n, 0) − v_n cos(ω_n Δt)‖ / ‖v_n‖
         − ‖RK4[step_asm](v_n, 0) − v_n cos(ω_n Δt)‖ / ‖v_n‖
which is the observable from paper §5.1.

This script does three scans:
  (A) Lane-Emden n = 3/2 (half-integer surface σ=3/2)
      Sweep N_y ∈ {32..256}.  Fixed Δt.
  (B) Lane-Emden n = 1    (integer σ=1)
  (C) Lane-Emden n = 3    (integer σ=3)

Proposition 1 predicts:
  (A): δ_N flat in N_y  (structural lower bound c > 0)
  (B): δ_N → 0 algebraically or exponentially
  (C): δ_N → 0 algebraically or exponentially

Output: review/r31_prop1/{timestep_defect_scan.csv, defect_operator_scan.csv}
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from full_galerkin_closure_test import cgl_grid, cc_weights, assemble_operator, evp, measure_dev

OUT_DIR = SCRIPT_DIR.parent / "review" / "r31_prop1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bg_lane_emden_n(y, Ly, n, rho_cut=0.05):
    def rhs(xi, st):
        theta, dtheta = st
        theta_safe = max(theta, 0.0)
        return [dtheta, -(theta_safe ** n) - 2 * dtheta / max(xi, 1e-8)]

    sol = solve_ivp(rhs, [1e-6, 10.0], [1.0, 0.0], dense_output=True,
                    events=lambda t, st: st[0], max_step=1e-3)
    xi_s = sol.t_events[0][0]
    xi = np.linspace(1e-6, xi_s, 4000)
    theta = np.clip(sol.sol(xi)[0], 0, None)
    rho_full = theta ** n
    mask = rho_full > rho_cut
    xi_lo, xi_hi = xi[mask].min(), xi[mask].max()
    xi_q = xi_lo + (y / Ly) * (xi_hi - xi_lo)
    rho = np.clip(np.interp(xi_q, xi, rho_full), rho_cut, None)
    drho = np.gradient(rho, y, edge_order=2)
    N2 = np.maximum(-drho / rho, 0.0)
    return rho, N2


def step_primitive_rk4(v, w, dt, D, rho, N2, kx):
    """Primitive-node RK4: at each substage evaluate v̈ as (R v − L_fact v)/ρ
    without assembling L⁻¹R.  Same as review_r2d1_2d_sweep.py but inlined here."""
    rho_safe = np.maximum(rho, 1e-30)

    def rhs(v_, w_):
        rho_dvdy = rho * (D @ v_)
        Lv_cont = -(D @ rho_dvdy) + kx ** 2 * rho * v_
        Rv = kx ** 2 * N2 * rho * v_
        vdotdot = (Rv - Lv_cont) / rho_safe
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
    def rhs(v_, w_):
        return w_, -(L_inv_R @ v_)
    k1v, k1w = rhs(v_int, w_int)
    k2v, k2w = rhs(v_int + 0.5 * dt * k1v, w_int + 0.5 * dt * k1w)
    k3v, k3w = rhs(v_int + 0.5 * dt * k2v, w_int + 0.5 * dt * k2w)
    k4v, k4w = rhs(v_int + dt * k3v, w_int + dt * k3w)
    v_new = v_int + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    w_new = w_int + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    return v_new, w_new


def per_step_defect(N_y, poly_n, n_steps, dt, Ly=1.0, kx=2 * np.pi,
                    rho_cut=0.05, amp=1e-8):
    """Run both schemes n_steps and return per-step eigenmode deviation."""
    y, D = cgl_grid(N_y, Ly)
    w_cc = cc_weights(N_y, Ly)
    rho, N2 = bg_lane_emden_n(y, Ly, poly_n, rho_cut=rho_cut)
    L, R = assemble_operator("vspace", y, D, rho, N2, kx)
    lam, V = evp(L, R)
    if len(lam) == 0:
        return None
    om2 = float(lam[0])
    om1 = float(np.sqrt(om2))

    v0_int = V[:, 0] / np.max(np.abs(V[:, 0])) * amp
    v0_full = np.zeros(N_y); v0_full[1:-1] = v0_int
    w0_full = np.zeros(N_y)

    # Primitive-RK4
    v, w = v0_full.copy(), w0_full.copy()
    traj_p = [v.copy()]
    for _ in range(n_steps):
        v, w = step_primitive_rk4(v, w, dt, D, rho, N2, kx)
        traj_p.append(v.copy())
    devs_p = measure_dev(traj_p, v0_full, w_cc)
    rate_p = (devs_p[-1] - devs_p[1]) / max(n_steps - 1, 1)

    # Assembled-RK4
    L_inv_R = scipy.linalg.solve(L, R)
    v_i, w_i = v0_int.copy(), np.zeros_like(v0_int)
    traj_a = []
    full = np.zeros(N_y); full[1:-1] = v_i
    traj_a.append(full.copy())
    for _ in range(n_steps):
        v_i, w_i = step_assembled_rk4(v_i, w_i, dt, L_inv_R)
        full = np.zeros(N_y); full[1:-1] = v_i
        traj_a.append(full.copy())
    devs_a = measure_dev(traj_a, v0_full, w_cc)
    rate_a = (devs_a[-1] - devs_a[1]) / max(n_steps - 1, 1)

    return dict(
        N_y=N_y,
        poly_n=poly_n,
        omega1=om1,
        dt=dt,
        n_steps=n_steps,
        primitive_per_step=rate_p,
        assembled_per_step=rate_a,
        primitive_final=float(devs_p[-1]),
        assembled_final=float(devs_a[-1]),
    )


def sweep_over_n(poly_n, N_list, dt_choice, label):
    rows = []
    print(f"\n=== {label}: Lane-Emden n = {poly_n}, dt = {dt_choice:.0e} ===")
    print("  N_y   ω₁      prim_per_step   asm_per_step   prim_final")
    print("  " + "-" * 60)
    for N in N_list:
        # Re-scale dt with N to keep primitive-RK4 within its N-dependent
        # CFL envelope.  The aliasing amplification tightens the effective
        # CFL roughly as N^{-1/2}, so dt ← dt_base · (N_ref/N)^{1/2}.
        N_ref = 64
        dt = dt_choice * (N_ref / N) ** 0.5
        r = per_step_defect(N, poly_n, n_steps=100, dt=dt)
        if r is None:
            print(f"  N={N}: no g-mode")
            continue
        # Normalise per-step by dt² (the paper-§5.1 scaling) to remove
        # the dt-variation across N.  That way any remaining N-dependence
        # is structural, not CFL-padded.
        r["prim_per_step_dt2"] = r["primitive_per_step"] / dt ** 2
        r["asm_per_step_dt2"] = r["assembled_per_step"] / dt ** 2
        r["dt_effective"] = dt
        print(f"  {r['N_y']:4d}  {r['omega1']:.4f}  "
              f"{r['primitive_per_step']:.3e}   "
              f"{r['assembled_per_step']:.3e}   "
              f"{r['primitive_final']:.3e}")
        rows.append(r)
    return rows


def main():
    N_list = [32, 48, 64, 96, 128, 192, 256]
    all_rows = []
    all_rows.extend(sweep_over_n(1.5, N_list, 5e-4, "Half-integer σ=3/2"))
    all_rows.extend(sweep_over_n(1.0, N_list, 5e-4, "Integer σ=1"))
    all_rows.extend(sweep_over_n(3.0, N_list, 5e-4, "Integer σ=3"))

    with open(OUT_DIR / "timestep_defect_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\nWrote {OUT_DIR / 'timestep_defect_scan.csv'}")

    # Summary: compute the "floor ratio" per poly_n
    print("\nSummary — per_step/dt² (scaled) vs N_y:")
    for poly in (1.5, 1.0, 3.0):
        subset = [r for r in all_rows if r["poly_n"] == poly]
        if not subset:
            continue
        print(f"  n = {poly}:")
        for r in subset:
            print(f"    N={r['N_y']:4d}  prim/dt²={r['prim_per_step_dt2']:.3e}")


if __name__ == "__main__":
    main()
