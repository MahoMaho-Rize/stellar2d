#!/usr/bin/env python3
"""R6.1: Test whether the +2.6 energy drift of Section 7.3 is a mismatch
between the *raw* quadrature functional and the natural inner product
induced by the assembled operator.

Three energy definitions measured on the identical linear (V, W, B) evolution:
    E_raw  = ½ ∫ ρ(u²+v²) + ½ ∫ b²/N²            (current §7.3 definition)
    E_int  = same but restricted to ρ > 2 ρ_cut  (floor-excluded)
    E_asm  = ½ ⟨W, W⟩_{CC,x} + ½ ⟨V, M V⟩_{CC,x}  (assembled-operator induced)

E_asm is by construction the conserved quantity of the second-order
oscillator ̈V = -M V that Theorem 6.1 closes.  RK4 on this oscillator
preserves E_asm to O((ωΔt)⁸) per step — ~10⁻⁹ over 800 steps at our
parameters — whereas the raw functional picks up structural bias from
its ρ and 1/N² weights being inconsistent with the discrete algebra of M.

Hypothesis: E_asm drift is 5-10 orders of magnitude smaller than E_raw
across all amplitudes and all Ny — converting the "open-problem structural
non-conservation" of §7.3 into a "chose the wrong quadrature" story.
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
    make_eigenmode_ic, apply_M,
    fft_x, ifft_x,
)

OUT_DIR = SCRIPT_DIR.parent / "review" / "r61_inner_product"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def reconstruct_u_from_v(v, rho, D, kx_array, nx):
    """u = -(1/(i kx ρ)) ∂_y(ρ v) per kx (for raw-energy diagnostic only)."""
    vhat = fft_x(v)
    nh = vhat.shape[1]
    u_hat = np.zeros_like(vhat)
    for k in range(nh):
        kx = kx_array[k]
        if abs(kx) < 1e-12:
            continue
        rho_v = rho[:, None] * vhat[:, k:k + 1]
        dy_rho_v = D @ rho_v
        u_hat[:, k] = -(dy_rho_v[:, 0] / (1j * kx * rho))
    return ifft_x(u_hat, nx)


def E_raw(u, v, b, rho, N2, w_cc, nx):
    KE = 0.5 * np.einsum('ij,i,i->', u * u + v * v, rho, w_cc) / nx
    N2_safe = np.where(N2 > 1e-12, N2, np.inf)
    PE = 0.5 * np.einsum('ij,i->', b * b / N2_safe[:, None], w_cc) / nx
    return float(KE + PE)


def E_int(u, v, b, rho, N2, w_cc, nx, rho_cut=0.05, factor=2.0):
    mask = (rho > factor * rho_cut).astype(float)
    KE = 0.5 * np.einsum('ij,i,i,i->', u * u + v * v, rho, mask, w_cc) / nx
    N2_safe = np.where(N2 > 1e-12, N2, np.inf)
    b2N2 = b * b / N2_safe[:, None]
    PE = 0.5 * np.einsum('ij,i,i->', b2N2, mask, w_cc) / nx
    return float(KE + PE)


def E_asm(V, W, M_list, kx_array, w_cc, nx):
    """E_asm = ½ ⟨W, W⟩_{CC,x} + ½ ⟨V, M V⟩_{CC,x}.

    Both terms assembled in physical space on the CGL grid.  The MV term
    uses the per-kx assembled M (interior DOFs only; walls are zero
    for Dirichlet-V, so contributing only 0 · 0)."""
    KE_term = 0.5 * np.einsum('ij,ij,i->', W, W, w_cc) / nx
    MV = apply_M(V, M_list, kx_array, nx)
    PE_term = 0.5 * np.einsum('ij,ij,i->', V, MV, w_cc) / nx
    return float(KE_term + PE_term)


def rk4_linear(V, W, B, dt, M_list, kx_array, N2, nx):
    """RK4 step of the oscillator ̈V = -MV with slaved B (dB/dt = -N² V)."""
    def rhs(V_, W_, B_):
        return W_, -apply_M(V_, M_list, kx_array, nx), -N2[:, None] * V_

    k1V, k1W, k1B = rhs(V, W, B)
    k2V, k2W, k2B = rhs(V + 0.5 * dt * k1V, W + 0.5 * dt * k1W, B + 0.5 * dt * k1B)
    k3V, k3W, k3B = rhs(V + 0.5 * dt * k2V, W + 0.5 * dt * k2W, B + 0.5 * dt * k2B)
    k4V, k4W, k4B = rhs(V + dt * k3V, W + dt * k3W, B + dt * k3B)
    V_new = V + dt / 6 * (k1V + 2 * k2V + 2 * k3V + k4V)
    W_new = W + dt / 6 * (k1W + 2 * k2W + 2 * k3W + k4W)
    B_new = B + dt / 6 * (k1B + 2 * k2B + 2 * k3B + k4B)
    V_new[0, :] = V_new[-1, :] = 0.0
    W_new[0, :] = W_new[-1, :] = 0.0
    return V_new, W_new, B_new


def exp_propagator_diagonalise(M_list, kx_array):
    """Pre-diagonalise M per kx into eigenpairs (Λ, Q, Q⁻¹).
    Returns list of dicts per kx: {'Q', 'Qi', 'omega'} or None for kx=0."""
    diag = []
    for k, kx in enumerate(kx_array):
        if abs(kx) < 1e-14 or M_list[k].shape[0] == 0:
            diag.append(None)
            continue
        M = M_list[k]
        lam, Q = scipy.linalg.eig(M)
        lam = np.real(lam)
        Q = np.real(Q)
        # clip tiny-negative eigenvalues (numerical noise) to zero
        lam = np.maximum(lam, 0.0)
        omega = np.sqrt(lam)
        Qi = scipy.linalg.inv(Q)
        diag.append({'Q': Q, 'Qi': Qi, 'omega': omega})
    return diag


def exp_linear_step(V, W, dt, diag, kx_array, nx):
    """Exact oscillator step: V_n+1 = cos(Ω dt) V_n + Ω⁻¹ sin(Ω dt) W_n
                               W_n+1 = -Ω sin(Ω dt) V_n + cos(Ω dt) W_n
    applied in the eigenbasis of M per kx."""
    Vhat = fft_x(V)
    What = fft_x(W)
    nh = Vhat.shape[1]
    V_new_hat = np.zeros_like(Vhat)
    W_new_hat = np.zeros_like(What)
    for k in range(nh):
        if diag[k] is None:
            V_new_hat[:, k] = Vhat[:, k]
            W_new_hat[:, k] = What[:, k]
            continue
        Q, Qi, om = diag[k]['Q'], diag[k]['Qi'], diag[k]['omega']
        V_int = Vhat[1:-1, k]
        W_int = What[1:-1, k]
        V_eig = Qi @ V_int
        W_eig = Qi @ W_int
        c = np.cos(om * dt)
        s = np.sin(om * dt)
        # sinΩΔt / Ω, with fallback to Δt when Ω→0
        s_over_om = np.where(om > 1e-14, s / np.where(om > 1e-14, om, 1.0), dt)
        V_new_eig = c * V_eig + s_over_om * W_eig
        W_new_eig = -om * s * V_eig + c * W_eig
        V_new_hat[1:-1, k] = Q @ V_new_eig
        W_new_hat[1:-1, k] = Q @ W_new_eig
    V_out = ifft_x(V_new_hat, nx)
    W_out = ifft_x(W_new_hat, nx)
    V_out[0, :] = V_out[-1, :] = 0.0
    W_out[0, :] = W_out[-1, :] = 0.0
    return V_out, W_out


def run(N_y, N_x, Lx, Ly, rho_cut, n_steps, dt, amp, integrator):
    y, D = cgl_grid(N_y, Ly)
    w_cc = cc_weights(N_y, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    kx_array = 2 * np.pi / Lx * np.arange(N_x // 2 + 1)

    M_list = assemble_M_per_kx(y, D, rho, N2, kx_array)
    diag = exp_propagator_diagonalise(M_list, kx_array) if integrator == 'exp' else None

    u0, V, B, V_ref, omega = make_eigenmode_ic(
        y, rho, N2, D, N_x, Lx, kx_int=1, n_g=1, amp=amp,
    )
    W = np.zeros_like(V)

    def u_from_V(V_now):
        return reconstruct_u_from_v(V_now, rho, D, kx_array, N_x).real

    E0_raw = E_raw(u0, V, B, rho, N2, w_cc, N_x)
    E0_int = E_int(u0, V, B, rho, N2, w_cc, N_x, rho_cut=rho_cut)
    E0_asm = E_asm(V, W, M_list, kx_array, w_cc, N_x)

    for step in range(n_steps):
        if integrator == 'rk4':
            V, W, B = rk4_linear(V, W, B, dt, M_list, kx_array, N2, N_x)
        elif integrator == 'exp':
            V, W = exp_linear_step(V, W, dt, diag, kx_array, N_x)
            B = B + dt * (-N2[:, None] * V)  # trapezoid-free slaved update
        else:
            raise ValueError(integrator)

    u_end = u_from_V(V)
    E1_raw = E_raw(u_end, V, B, rho, N2, w_cc, N_x)
    E1_int = E_int(u_end, V, B, rho, N2, w_cc, N_x, rho_cut=rho_cut)
    E1_asm = E_asm(V, W, M_list, kx_array, w_cc, N_x)

    def rel(a0, a1):
        return (a1 - a0) / max(abs(a0), 1e-300)

    return dict(
        E0_raw=E0_raw, E0_int=E0_int, E0_asm=E0_asm,
        dE_raw=rel(E0_raw, E1_raw),
        dE_int=rel(E0_int, E1_int),
        dE_asm=rel(E0_asm, E1_asm),
        omega=omega,
    )


def main():
    print(" R6.1: assembled-inner-product energy diagnostic")
    print("  E_asm(V,W) = ½⟨W,W⟩_w + ½⟨V, M V⟩_w")
    print("  tests whether the §7.3 +2.6 drift is a quadrature-functional mismatch")
    print("=" * 74)

    cases = [
        # Amplitude sweep at fixed Ny, Strang+RK4
        ('amp-sweep-RK4', dict(N_y=48, N_x=64, n_steps=800, dt=2e-2, integrator='rk4'), [1e-8, 1e-4, 1e-1]),
        # Amplitude sweep at fixed Ny, exp-propagator
        ('amp-sweep-exp', dict(N_y=48, N_x=64, n_steps=800, dt=2e-2, integrator='exp'), [1e-8, 1e-4, 1e-1]),
        # Ny refinement at amp=1e-8, Strang+RK4 (to test grid-invariance of prefactor)
        ('Ny-sweep-RK4', dict(N_x=64, n_steps=800, dt=2e-2, amp=1e-8, integrator='rk4'), [48, 96, 192]),
        # Ny refinement at amp=1e-8, exp-propagator
        ('Ny-sweep-exp', dict(N_x=64, n_steps=800, dt=2e-2, amp=1e-8, integrator='exp'), [48, 96, 192]),
        # dt refinement (to verify O(Δt²) for RK4, machine-precision for exp)
        ('dt-sweep-RK4', dict(N_y=48, N_x=64, amp=1e-8, integrator='rk4'), [(2e-2, 800), (1e-2, 1600), (5e-3, 3200)]),
        ('dt-sweep-exp', dict(N_y=48, N_x=64, amp=1e-8, integrator='exp'), [(2e-2, 800), (1e-2, 1600), (5e-3, 3200)]),
    ]

    all_rows = []
    for label, fixed, sweep in cases:
        print(f"\n  [{label}]  fixed: {fixed}")
        if 'amp' in label:
            print(f"  {'amp':>8s}  {'dE_raw/E0':>14s}  {'dE_int/E0':>14s}  {'dE_asm/E0':>14s}  {'ratio raw/asm':>12s}")
            print("  " + "-" * 72)
            for amp in sweep:
                r = run(amp=amp, Lx=1.0, Ly=1.0, rho_cut=0.05, **fixed)
                ratio = abs(r['dE_raw']) / max(abs(r['dE_asm']), 1e-300)
                print(f"  {amp:8.0e}  {r['dE_raw']:+14.3e}  {r['dE_int']:+14.3e}  {r['dE_asm']:+14.3e}  {ratio:12.2e}")
                all_rows.append(dict(label=label, sweep_var=amp, **r, ratio_raw_asm=ratio))
        elif 'Ny' in label:
            print(f"  {'Ny':>6s}  {'dE_raw/E0':>14s}  {'dE_int/E0':>14s}  {'dE_asm/E0':>14s}  {'ratio raw/asm':>12s}")
            print("  " + "-" * 70)
            for Ny in sweep:
                r = run(N_y=Ny, Lx=1.0, Ly=1.0, rho_cut=0.05, **fixed)
                ratio = abs(r['dE_raw']) / max(abs(r['dE_asm']), 1e-300)
                print(f"  {Ny:6d}  {r['dE_raw']:+14.3e}  {r['dE_int']:+14.3e}  {r['dE_asm']:+14.3e}  {ratio:12.2e}")
                all_rows.append(dict(label=label, sweep_var=Ny, **r, ratio_raw_asm=ratio))
        elif 'dt' in label:
            print(f"  {'dt':>8s}  {'n_steps':>8s}  {'dE_raw/E0':>14s}  {'dE_int/E0':>14s}  {'dE_asm/E0':>14s}")
            print("  " + "-" * 72)
            for dt, n_steps in sweep:
                r = run(dt=dt, n_steps=n_steps, Lx=1.0, Ly=1.0, rho_cut=0.05, **fixed)
                print(f"  {dt:8.0e}  {n_steps:8d}  {r['dE_raw']:+14.3e}  {r['dE_int']:+14.3e}  {r['dE_asm']:+14.3e}")
                all_rows.append(dict(label=label, sweep_var=dt, n_steps=n_steps, **r))

    # Write results
    csv_path = OUT_DIR / "energy_three_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        # Take union of keys
        keys = []
        for r in all_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    print("\n" + "=" * 74)
    print("  Decision rule (§7.3 rewrite vs keep-as-is):")
    print("  (A) If E_asm drift ≲ 1e-10 for RK4 AND machine-precision for exp:")
    print("      §7.3 story changes from 'structural non-conservation, open problem'")
    print("      to 'raw functional uses wrong inner product; assembled functional")
    print("      is conserved by construction'. Positive upgrade.")
    print("  (B) If E_asm drift is similar to E_raw: keep §7.3 as-is, add negative")
    print("      result to appendix.")


if __name__ == "__main__":
    main()
