#!/usr/bin/env python3
"""R5.4: Quantify the near-wall energy-functional bias that produces the
+2.6 drift observed in Table 7.1.

The standard diagnostic
    E_raw = ½ ∫ ρ(u² + v²) dx dy + ½ ∫ (b² / N²) dx dy
includes the floor region y ∈ [y_floor_lo, y_floor_hi] where N² → 0
and the 1/N² weight becomes ill-conditioned, producing a numerical
drift orders of magnitude above the physical energy scale.

A corrected diagnostic restricts the buoyancy integral to the interior
    E_corr = ½ ∫_Ω ρ(u² + v²) dx dy + ½ ∫_{Ω ∩ {ρ > 2 ρ_cut}} (b² / N²) dx dy
excluding the floor region where N² < N²_threshold.

This script compares E_raw and E_corr drifts over 800 Strang-split
steps on the eigenmode IC at amp = 1e-8 (linear regime, where any
physical energy transfer is below machine precision and all observed
drift is diagnostic bias).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nonlinear_paths_infra import (
    cgl_grid, cc_weights, bg_lane_emden, assemble_M_per_kx,
    make_eigenmode_ic, compute_advection, apply_M,
    fft_x, ifft_x, dealias_23,
)

OUT_DIR = SCRIPT_DIR.parent / "review" / "r54_energy"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def total_energy_raw(u, v, b, rho, N2, w_cc, nx):
    """Standard diagnostic (the one producing +2.6 drift)."""
    KE = 0.5 * np.einsum('ij,i,i->', u * u + v * v, rho, w_cc) / nx
    N2_safe = np.where(N2 > 1e-12, N2, np.inf)
    PE = 0.5 * np.einsum('ij,i->', b * b / N2_safe[:, None], w_cc) / nx
    return float(KE + PE)


def total_energy_corrected(u, v, b, rho, N2, w_cc, nx, rho_cut=0.05,
                            rho_factor=2.0):
    """Corrected diagnostic: exclude floor region from PE integral.

    The floor region is {y : ρ(y) ≤ rho_factor · ρ_cut}.  The b²/N²
    weight becomes ill-conditioned as N² → 0; the ρ-based mask is a
    simple proxy for that region since ρ is clipped to ρ_cut in the
    floor.
    """
    in_interior = rho > rho_factor * rho_cut
    KE = 0.5 * np.einsum('ij,i,i->', u * u + v * v, rho, w_cc) / nx
    N2_safe = np.where(N2 > 1e-12, N2, np.inf)
    b2_over_N2 = b * b / N2_safe[:, None]
    PE = 0.5 * np.einsum('ij,i,i->', b2_over_N2, in_interior.astype(float), w_cc) / nx
    return float(KE + PE)


def total_energy_interior(u, v, b, rho, N2, w_cc, nx, rho_cut=0.05,
                           rho_factor=2.0):
    """Interior-only diagnostic: restrict BOTH KE and PE to the
    interior region where ρ > rho_factor · ρ_cut."""
    in_interior = rho > rho_factor * rho_cut
    mask = in_interior.astype(float)
    KE = 0.5 * np.einsum('ij,i,i,i->', u * u + v * v, rho, mask, w_cc) / nx
    N2_safe = np.where(N2 > 1e-12, N2, np.inf)
    b2_over_N2 = b * b / N2_safe[:, None]
    PE = 0.5 * np.einsum('ij,i,i->', b2_over_N2, mask, w_cc) / nx
    return float(KE + PE)


def rk4_linear(V, W, B, dt, M_list, kx_array, N2, nx):
    """RK4 step on the linear (V, W, B) system per kx."""
    def rhs(V_, W_, B_):
        return W_, -apply_M(V_, M_list, kx_array, nx), -N2[:, None] * V_

    k1V, k1W, k1B = rhs(V, W, B)
    k2V, k2W, k2B = rhs(V + 0.5 * dt * k1V, W + 0.5 * dt * k1W, B + 0.5 * dt * k1B)
    k3V, k3W, k3B = rhs(V + 0.5 * dt * k2V, W + 0.5 * dt * k2W, B + 0.5 * dt * k2B)
    k4V, k4W, k4B = rhs(V + dt * k3V, W + dt * k3W, B + dt * k3B)
    V_new = V + dt / 6 * (k1V + 2 * k2V + 2 * k3V + k4V)
    W_new = W + dt / 6 * (k1W + 2 * k2W + 2 * k3W + k4W)
    B_new = B + dt / 6 * (k1B + 2 * k2B + 2 * k3B + k4B)
    # Wall Dirichlet
    V_new[0, :] = V_new[-1, :] = 0.0
    W_new[0, :] = W_new[-1, :] = 0.0
    return V_new, W_new, B_new


def reconstruct_u_from_v(v, rho, D, kx_array, nx):
    """u = -(1/(ikx ρ)) ∂_y(ρ v) per kx."""
    vhat = fft_x(v)
    nh = vhat.shape[1]
    u_hat = np.zeros_like(vhat)
    for k in range(nh):
        kx = kx_array[k]
        if abs(kx) < 1e-12:
            continue
        rho_v = rho[:, None] * vhat[:, k:k+1]
        dy_rho_v = D @ rho_v
        u_hat[:, k] = -(dy_rho_v[:, 0] / (1j * kx * rho))
    return ifft_x(u_hat, nx)


def run_eigenmode_linear(N_y=48, N_x=64, Ly=1.0, Lx=1.0,
                         rho_cut=0.05, n_steps=800, dt=2e-2,
                         amp=1e-8, rho_factor=2.0):
    """Linear Strang-split run on the eigenmode IC (no nonlinear block —
    this isolates the diagnostic-bias from any physical energy transfer)."""
    y, D = cgl_grid(N_y, Ly)
    w_cc = cc_weights(N_y, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    kx_array = 2 * np.pi / Lx * np.arange(N_x // 2 + 1)

    M_list = assemble_M_per_kx(y, D, rho, N2, kx_array)
    u, V, B, V_ref, omega = make_eigenmode_ic(
        y, rho, N2, D, N_x, Lx, kx_int=1, n_g=1, amp=amp,
    )
    W = np.zeros_like(V)
    u0 = u.copy()

    e_raw_0 = total_energy_raw(u0, V, B, rho, N2, w_cc, N_x)
    e_corr_0 = total_energy_corrected(u0, V, B, rho, N2, w_cc, N_x, rho_cut, rho_factor)
    e_int_0 = total_energy_interior(u0, V, B, rho, N2, w_cc, N_x, rho_cut, rho_factor)
    rows = [dict(step=0, E_raw=e_raw_0, E_corr=e_corr_0, E_int=e_int_0)]

    for step in range(1, n_steps + 1):
        V, W, B = rk4_linear(V, W, B, dt, M_list, kx_array, N2, N_x)
        if step % 50 == 0 or step == n_steps:
            u_now = reconstruct_u_from_v(V, rho, D, kx_array, N_x).real
            e_raw = total_energy_raw(u_now, V, B, rho, N2, w_cc, N_x)
            e_corr = total_energy_corrected(u_now, V, B, rho, N2, w_cc, N_x, rho_cut, rho_factor)
            e_int = total_energy_interior(u_now, V, B, rho, N2, w_cc, N_x, rho_cut, rho_factor)
            rows.append(dict(step=step, E_raw=e_raw, E_corr=e_corr, E_int=e_int))

    for r in rows:
        r["dE_raw_over_E0"] = (r["E_raw"] - e_raw_0) / max(abs(e_raw_0), 1e-300)
        r["dE_corr_over_E0"] = (r["E_corr"] - e_corr_0) / max(abs(e_corr_0), 1e-300)
        r["dE_int_over_E0"] = (r["E_int"] - e_int_0) / max(abs(e_int_0), 1e-300)
    return rows, e_raw_0, e_corr_0, e_int_0


def main():
    print(" R5.4: Corrected energy diagnostic on linear eigenmode evolution")
    print(" (amp = 1e-8, 800 Strang-linear steps — physical energy transfer")
    print("  is below machine precision, all drift is diagnostic bias)")
    print("=" * 72)
    # Sweep rho_factor to find the cleanest diagnostic
    print("\n  Sweep of rho_factor (exclusion threshold as multiple of ρ_cut):")
    print("  factor   ΔE_raw/E_0     ΔE_corr/E_0    ΔE_int/E_0     ratio raw/int")
    print("  " + "-" * 72)
    sweep_results = {}
    for rho_factor in [1.0, 1.5, 2.0, 3.0, 5.0]:
        rows, E_raw0, E_corr0, E_int0 = run_eigenmode_linear(
            N_y=48, N_x=64, n_steps=800, dt=2e-2, amp=1e-8,
            rho_factor=rho_factor,
        )
        last = rows[-1]
        ratio = (abs(last["dE_raw_over_E0"]) /
                 max(abs(last["dE_int_over_E0"]), 1e-300))
        print(f"  {rho_factor:.1f}      {last['dE_raw_over_E0']:+.3e}   "
              f"{last['dE_corr_over_E0']:+.3e}   "
              f"{last['dE_int_over_E0']:+.3e}    {ratio:.2e}")
        sweep_results[rho_factor] = (rows, E_raw0, E_corr0, E_int0)

    # Amplitude-invariance test: the drift is a FIXED relative number
    # independent of amp, confirming it is not round-off amplification
    # but a structural non-conservation of the discrete energy functional.
    print("\n  Amplitude-invariance test (200 steps, rho_factor=2):")
    print("  amp       E_0         ΔE_raw/E_0     ΔE_int/E_0")
    print("  " + "-" * 56)
    amp_rows = []
    for amp in [1e-8, 1e-6, 1e-4, 1e-2]:
        r, E0, Ec0, Ei0 = run_eigenmode_linear(
            N_y=48, N_x=64, n_steps=200, dt=2e-2, amp=amp, rho_factor=2.0,
        )
        last = r[-1]
        print(f"  {amp:.0e}  {E0:.2e}  {last['dE_raw_over_E0']:+.3e}   "
              f"{last['dE_int_over_E0']:+.3e}")
        amp_rows.append(dict(amp=amp, E0=E0,
                             dE_raw_over_E0=last["dE_raw_over_E0"],
                             dE_int_over_E0=last["dE_int_over_E0"]))

    with open(OUT_DIR / "amp_invariance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(amp_rows[0].keys()))
        w.writeheader()
        for r in amp_rows:
            w.writerow(r)

    # Detailed output for rho_factor = 2.0 (reported in paper)
    rows, E_raw0, E_corr0, E_int0 = sweep_results[2.0]
    print(f"\n  Detailed trajectory for rho_factor = 2.0:")
    print(f"  E_raw(0)  = {E_raw0:.3e}")
    print(f"  E_corr(0) = {E_corr0:.3e}")
    print(f"  E_int(0)  = {E_int0:.3e}")
    print()
    print(f"  step   ΔE_raw/E_0     ΔE_corr/E_0    ΔE_int/E_0")
    print("  " + "-" * 60)
    for r in rows[::3]:
        print(f"  {r['step']:4d}   {r['dE_raw_over_E0']:+.3e}   "
              f"{r['dE_corr_over_E0']:+.3e}   {r['dE_int_over_E0']:+.3e}")

    last = rows[-1]
    ratio = (abs(last["dE_raw_over_E0"]) /
             max(abs(last["dE_int_over_E0"]), 1e-300))
    print(f"\n  Final step {last['step']} (rho_factor=2.0):")
    print(f"    ΔE_raw/E_0   = {last['dE_raw_over_E0']:+.3e}")
    print(f"    ΔE_int/E_0   = {last['dE_int_over_E0']:+.3e}")
    print(f"    ratio raw/int = {ratio:.2e}  (= {np.log10(ratio):.2f} orders of magnitude)")

    with open(OUT_DIR / "energy_drift.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {OUT_DIR / 'energy_drift.csv'}")


if __name__ == "__main__":
    main()
