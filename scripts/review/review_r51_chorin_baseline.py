#!/usr/bin/env python3
"""R5.1: Python reproducer of the §5.1 primitive-node baseline numbers
(4.5e-5 Boussinesq, 6.9e-4 Lane-Emden).

Background.  These numbers originated in the CUDA stellar2d binary
running the "anelastic_sl" path with a RK3 + Chorin projection time
loop, documented in docs/variable_density_lane_emden_td_2026-05-03.md.
The measurement is a one-step leakage rate of an EVP eigenvector
through the coupled (momentum RHS, buoyancy RHS, Chorin projection)
linear operator, with IC-projection skipped, filter off, viscosity off.

The key mechanism (§5.3 of that doc):
  - EVP assembles B = -D·diag(ρ)·D + k²·diag(ρ).
  - The SL-Poisson projection uses T = -D² + W-potential basis (via
    Liouville substitution π = ρ^{-1/2} q).
  - For ρ = 1, B and T coincide exactly (up to k² shift), so V_EVP is
    also a T-eigenvector and the projection does not scatter it.
    Residual dev/step is the RK3 truncation floor ≈ 4.5e-5 at dt=1e-4.
  - For Lane-Emden, ρ'≠0 causes B ≠ T at the discrete level by
    O(‖ρ'‖), so V_EVP is NOT a T-eigenvector and the projection
    scatters it at a rate set by the B–T mismatch.  Residual
    dev/step jumps to 6.9e-4 (~15× the Boussinesq floor).

This script reproduces the Python equivalent.  We build:
  - V_EVP from the assembled EVP B V = ω² A V.
  - A Python Chorin-projection SL-Poisson solver using T.
  - An RK3 loop of (V, B, U) with IC set to V_EVP and no IC projection.

Measure dev/step = ‖V^1 - (coeff)·V_EVP‖ / ‖V_EVP‖ after one step.
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

from nonlinear_paths_infra import cgl_grid, cc_weights, bg_lane_emden

OUT_DIR = SCRIPT_DIR.parent / "review" / "r51_chorin_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_sl_basis(D, rho, intr_slice):
    """Build the SL-Poisson basis for the Chorin projection.

    The projection solves ∇·(ρ∇π) = div, which after Liouville substitution
    π = ρ^{-1/2} q becomes a Schrödinger-type eigenproblem
        (-D² - W(y)) ψ_n = μ_n ψ_n,    W = (1/2)ρ''/ρ - (1/4)(ρ'/ρ)²

    Return the eigenvectors Ψ (interior-restricted) and eigenvalues {μ_n}.
    """
    # Liouville potential
    drho = D @ rho
    ddrho = D @ drho
    W = 0.5 * ddrho / rho - 0.25 * (drho / rho)**2

    # SL operator: T = -D² - diag(W)
    D2 = D @ D
    T = -D2[intr_slice, intr_slice] - np.diag(W[intr_slice])
    mu, Psi = scipy.linalg.eigh(T)
    return mu, Psi, W


def sl_poisson_solve(div_rhs, rho, D, w_cc, mu, Psi, kx2, intr_slice):
    """Solve ∇·(ρ∇π) = div_rhs for π with π(walls)=0 via SL expansion.

    In Liouville variables: (-D² - W - k²) q = sqrt(ρ) · div_rhs / (??) ...
    Simplified implementation: directly apply T+k² inverse on interior.
    """
    # Full-node SL: (T + k²I) π = -div / ρ  on interior
    rhs_int = -div_rhs[intr_slice] / rho[intr_slice]
    # Expand in Ψ basis
    w_int = w_cc[intr_slice]
    coeffs = (Psi.T @ (w_int * rhs_int)) / (mu + kx2)
    pi_int = Psi @ coeffs
    pi = np.zeros_like(div_rhs)
    pi[intr_slice] = pi_int
    return pi


def rk3_step_primitive(V, B, U, dt, D, rho, N2, kx, w_cc, mu, Psi, intr_slice):
    """RK3 + Chorin-projection step, mirroring CUDA's anelastic_sl path.

    RHS(V, B, U):
      dV/dt = B - dπ/dy     (vertical momentum with buoyancy and pressure)
      dB/dt = -N² V
      dU/dt = -i k_x π      (horizontal momentum with pressure)
    Continuity:  i k_x ρ U + d(ρV)/dy = 0  enforced via Chorin projection.

    This is a simplified *real* scalar version for the first Fourier
    mode k_x; we track only the real parts using sin/cos decomposition
    implicit in the CUDA code.  The key structural feature is that the
    pressure Π = π solves  ∇·(ρ∇π) = ∇·(ρ RHS*) via SL expansion —
    that's where the B/T operator mismatch leaks into dev/step.
    """
    def rhs_novp(V_, B_, U_):
        """Un-projected RHS before Chorin."""
        dVdt = B_.copy()
        dBdt = -N2 * V_
        dUdt = np.zeros_like(U_)
        return dVdt, dBdt, dUdt

    def chorin_project(V_, U_):
        """Subtract gradient of Π such that continuity ∇·(ρu)=0 holds."""
        # div(ρ·u) on interior, where u = (U, V)
        # In 2D Fourier: div = i k_x ρ U + d(ρ V)/dy
        # We want to solve ∇·(ρ∇π) = div(ρ u) and subtract ∇π from u.
        rho_V = rho * V_
        dy_rhoV = D @ rho_V
        # For a single-mode analysis, express div as a grid-scalar
        div = (1j * kx) * (rho * U_) + dy_rhoV  # complex RHS for full kx mode
        # Solve SL-Poisson for π (complex)
        div_re = div.real
        div_im = div.imag
        pi_re = sl_poisson_solve(div_re, rho, D, w_cc, mu, Psi, kx**2, intr_slice)
        pi_im = sl_poisson_solve(div_im, rho, D, w_cc, mu, Psi, kx**2, intr_slice)
        pi = pi_re + 1j * pi_im
        # Subtract gradient
        dpi_dy = D @ pi
        V_new = V_ - dpi_dy.real  # V is real; use real part
        U_new = U_ - (1j * kx * pi).real
        V_new[0] = 0.0
        V_new[-1] = 0.0
        return V_new, U_new

    # Shu-Osher RK3 (TVD)
    # Stage 1
    dV1, dB1, dU1 = rhs_novp(V, B, U)
    V1 = V + dt * dV1
    B1 = B + dt * dB1
    U1 = U + dt * dU1
    V1, U1 = chorin_project(V1, U1)

    # Stage 2
    dV2, dB2, dU2 = rhs_novp(V1, B1, U1)
    V2 = 0.75 * V + 0.25 * (V1 + dt * dV2)
    B2 = 0.75 * B + 0.25 * (B1 + dt * dB2)
    U2 = 0.75 * U + 0.25 * (U1 + dt * dU2)
    V2, U2 = chorin_project(V2, U2)

    # Stage 3
    dV3, dB3, dU3 = rhs_novp(V2, B2, U2)
    V_new = (1.0 / 3.0) * V + (2.0 / 3.0) * (V2 + dt * dV3)
    B_new = (1.0 / 3.0) * B + (2.0 / 3.0) * (B2 + dt * dB3)
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * dU3)
    V_new, U_new = chorin_project(V_new, U_new)

    return V_new, B_new, U_new


def measure_leakage(bg_name, Ny=64, Ly=1.0, kx=2.0 * np.pi, n_steps=100,
                    dt=1e-4, amp=1e-8, rho_cut=0.05):
    y, D = cgl_grid(Ny, Ly)
    w_cc = cc_weights(Ny, Ly)

    if bg_name == "boussinesq":
        rho = np.ones(Ny)
        N2 = np.ones(Ny)  # constant N² = 1 to match the anelastic_sl test
    elif bg_name == "lane_emden":
        rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    else:
        raise ValueError(bg_name)

    intr = slice(1, Ny - 1)

    # Build the two discrete operators:
    #   B_evp = -D·diag(ρ)·D + k²·diag(ρ)   (assembled EVP operator)
    #   A_evp = k²·diag(N²·ρ)
    B_evp_full = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
    A_evp_full = kx**2 * np.diag(N2 * rho)
    B_int = B_evp_full[intr, intr]
    A_int = A_evp_full[intr, intr]

    # EVP: A V = ω² B V
    lam, V_eig = scipy.linalg.eig(A_int, B_int)
    lam = np.real(lam); V_eig = np.real(V_eig)
    mask = np.isfinite(lam) & (lam > 0)
    lam, V_eig = lam[mask], V_eig[:, mask]
    order = np.argsort(lam)[::-1]
    lam, V_eig = lam[order], V_eig[:, order]
    omega2 = float(lam[0])
    omega = np.sqrt(omega2)

    # Top eigenvector V_EVP on full grid, scaled by amp
    V_EVP = np.zeros(Ny)
    V_EVP[intr] = V_eig[:, 0]
    V_EVP = V_EVP / np.max(np.abs(V_EVP)) * amp

    # Build SL-Poisson basis for Chorin projection
    mu, Psi, W = build_sl_basis(D, rho, intr)

    # IC: (V, B, U) at the eigenmode.  Use the linearised eigenmode:
    #   V = V_EVP,  B = 0,  U = (ρV_EVP)' / (k_x ρ) · (i)  from continuity.
    # For sin/cos mode with V ~ cos(kx·x) V_EVP(y), the continuity gives
    # U ~ sin(kx·x) · dy(ρ V)/(k_x ρ).  We skip IC projection per the
    # docs's ANSL_SKIP_IC_PROJECT=1 setting.
    V = V_EVP.copy()
    B = np.zeros(Ny)
    rho_V = rho * V
    dy_rhoV = D @ rho_V
    U = np.zeros(Ny, dtype=complex)
    U_interior = dy_rhoV[intr] / (1j * kx * rho[intr])
    U[intr] = U_interior

    # Reference norm (CC-weighted) for dev metric
    V_IC = V.copy()
    IC_norm = np.sqrt(np.sum(w_cc * V_IC**2))

    devs = [0.0]
    for step in range(n_steps):
        V, B, U = rk3_step_primitive(V, B, U, dt, D, rho, N2, kx, w_cc,
                                       mu, Psi, intr)
        # Project-out V_IC component
        coeff = np.sum(w_cc * V * V_IC) / np.sum(w_cc * V_IC**2)
        r = V - coeff * V_IC
        dev = np.sqrt(np.sum(w_cc * r**2)) / IC_norm
        devs.append(dev)

    dev_per_step = (devs[-1] - devs[1]) / max(n_steps - 1, 1) if n_steps > 1 else devs[-1]
    return dict(bg=bg_name, omega=omega, n_steps=n_steps, dt=dt,
                dev_step1=devs[1], dev_final=devs[-1],
                dev_per_step=dev_per_step)


def main():
    print("R5.1: Python reproducer of §5.1 primitive-RK3 + Chorin baseline")
    print("    Configuration: ANSL_SKIP_IC_PROJECT=1, ANSL_FILTER_ALPHA=0,")
    print("                   ANSL_DT_MAX=1e-4, amp=1e-8, 100 steps, N_y=64")
    print("=" * 74)

    rows = []
    for bg in ("boussinesq", "lane_emden"):
        print(f"\n  [{bg}]")
        r = measure_leakage(bg, Ny=64, n_steps=100, dt=1e-4, amp=1e-8)
        print(f"    omega        = {r['omega']:.4f}")
        print(f"    dev @ step 1 = {r['dev_step1']:.3e}")
        print(f"    dev @ step 100 = {r['dev_final']:.3e}")
        print(f"    dev / step   = {r['dev_per_step']:.3e}")
        rows.append(r)

    with open(OUT_DIR / "chorin_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {OUT_DIR / 'chorin_baseline.csv'}")

    # Ratio check
    ratio = rows[1]["dev_per_step"] / rows[0]["dev_per_step"]
    print(f"\n  Lane-Emden / Boussinesq ratio = {ratio:.2f}")
    print("  Expected (CUDA stellar2d, doc §4.2): ~15×")
    print("  Expected absolute values: Boussinesq 4.5e-5,  Lane-Emden 6.9e-4")


if __name__ == "__main__":
    main()
