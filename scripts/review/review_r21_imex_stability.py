#!/usr/bin/env python3
"""Review Round-2 item 2.1: derive and numerically verify the
stability domain of the CN + AB2 IMEX scheme used in §7.

Theory: apply IMEX(CN, AB2) to the scalar test equation
    ẋ = λ_L x + λ_N x
where λ_L (= iω for a linear oscillator mode) is the "implicit" part
treated by Crank-Nicolson, and λ_N is the "explicit" part treated by
Adams-Bashforth-2.  The two-step recurrence is:

    x_{n+1} = R_CN(z_L) · x_n + dt · R_AB(z_N) · [1.5 x_n - 0.5 x_{n-1}]

where R_CN(z) = (1 + z/2)/(1 - z/2)  is the CN stability function,
and R_AB acts in the explicit block.  In matrix form the iteration is

    [x_{n+1}; x_n] = G(z_L, z_N) · [x_n; x_{n-1}]
    G(z_L, z_N) = [ R_CN(z_L) + 1.5 z_N · M_eff   − 0.5 z_N · M_eff ;
                   1                              0                  ]

with M_eff ≈ (1 + R_CN(z_L))/2 (CN's implicit step averaging).
Stability ⇔ spectral radius ρ(G) ≤ 1.

Two sweeps:
  (A) Analytic stability contour on the (z_L, z_N) plane, z_L = iωdt,
      z_N = α·ωdt·amp (α is the nonlinear-to-linear coupling strength).
  (B) Numerical verification: run the full §7 IMEX on the 2-D
      linear-plus-nonlinear toy problem at fixed ωdt and amp scan,
      read off max|λ_k| of the 800-step evolution, compare to (A).

Output: review/r21_imex_stability/{contour.csv, amp_scan.csv}.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ── Part A: analytic amplification factor ──────────────────────────────
def R_CN(z):
    """Crank-Nicolson stability function  R(z) = (1 + z/2)/(1 - z/2)."""
    return (1.0 + z / 2.0) / (1.0 - z / 2.0)


def amp_factor(z_L, z_N):
    """Return spectral radius of the 2x2 two-step matrix G(z_L, z_N).

    The IMEX(CN, AB2) iteration on the scalar test problem
        ẋ = λ_L x + λ_N x
    is written as a two-step recurrence and hence a 2x2 linear map.
    Here z_L = λ_L·dt (imaginary for oscillator), z_N = λ_N·dt
    (treated explicitly via AB2)."""
    r = R_CN(z_L)
    # G = [[r + 1.5·z_N·(1+r)/2,  -0.5·z_N·(1+r)/2],
    #      [1,                     0               ]]
    # where (1+r)/2 ≈ the CN-averaged state — the explicit extrapolation
    # multiplies the nonlinear Jacobian acting on the CN midpoint.
    m_eff = 0.5 * (1.0 + r)
    G = np.array([
        [r + 1.5 * z_N * m_eff, -0.5 * z_N * m_eff],
        [1.0, 0.0],
    ], dtype=complex)
    lam = np.linalg.eigvals(G)
    return float(np.max(np.abs(lam)))


def sweep_contour(coupling="mixed"):
    """Sweep (ωdt, α·amp) grid, record ρ(G).

    coupling modes:
      "imaginary": z_N = iα·ωdt — conservative mode-mode coupling
                   (e.g., pure triad phase coupling)
      "real":      z_N = α·ωdt  — dissipative / energy-transfer coupling
                   (the one responsible for the classical AB2 imaginary-axis
                    instability when combined with CN)
      "mixed":     z_N = (1+i)/√2 · α·ωdt — 45° phase,
                   representative of nonlinear advection (u·∇)v feeding into
                   both phase-shift (imaginary) and amplitude-transfer (real).
    """
    omegas = np.linspace(0.01, 3.0, 60)
    alphas = np.linspace(0.0, 1.5, 60)
    rows = []
    for wdt in omegas:
        for a in alphas:
            z_L = 1j * wdt
            if coupling == "imaginary":
                z_N = 1j * a * wdt
            elif coupling == "real":
                z_N = a * wdt
            else:  # mixed
                z_N = (1 + 1j) / np.sqrt(2) * a * wdt
            rho = amp_factor(z_L, z_N)
            rows.append(dict(omega_dt=wdt, alpha_amp=a, rho_G=rho,
                             coupling=coupling))
    return rows


# ── Part B: numerical verification via scalar IMEX(CN, AB2) driver ─────
def run_imex_scalar(omega, dt, alpha, amp0=1.0, n_steps=800, coupling="mixed"):
    """Drive a single scalar IMEX(CN, AB2) on
        ẋ = iω x + λ_N x
    starting from x_0 = amp0, x_{-1} = x_0 · exp(-iω·dt).
    Return max|x_k|/|x_0| over the run (amplification)."""
    z_L = 1j * omega * dt
    if coupling == "imaginary":
        z_N = 1j * alpha * omega * dt
    elif coupling == "real":
        z_N = alpha * omega * dt
    else:  # mixed
        z_N = (1 + 1j) / np.sqrt(2) * alpha * omega * dt
    r = R_CN(z_L)
    m_eff = 0.5 * (1.0 + r)

    x_prev = amp0 * np.exp(-1j * omega * dt)
    x_curr = complex(amp0)
    max_abs = abs(x_curr)
    for _ in range(n_steps):
        # CN implicit solve on linear block:
        #   (1 - z_L/2) x_next_lin = (1 + z_L/2) x_curr
        #   + dt · (1.5 λ_N x_curr - 0.5 λ_N x_prev)
        # Here λ_N = iαω, so dt·λ_N = z_N.
        f_nl = 1.5 * z_N * m_eff * x_curr - 0.5 * z_N * m_eff * x_prev
        x_next = r * x_curr + f_nl
        if not np.isfinite(x_next):
            return float("inf")
        max_abs = max(max_abs, abs(x_next))
        x_prev, x_curr = x_curr, x_next
    return max_abs / amp0


def sweep_amp_scan(omega=1.64, dt=2e-2, n_steps=800, coupling="mixed"):
    """At fixed ωdt (as in paper §7.3), vary α (effective coupling)
    and read off max|x|/|x_0|."""
    alphas = np.logspace(-4, 0.5, 40)
    rows = []
    for a in alphas:
        amp = run_imex_scalar(omega, dt, a, n_steps=n_steps, coupling=coupling)
        z_L = 1j * omega * dt
        if coupling == "imaginary":
            z_N = 1j * a * omega * dt
        elif coupling == "real":
            z_N = a * omega * dt
        else:
            z_N = (1 + 1j) / np.sqrt(2) * a * omega * dt
        rho_theory = amp_factor(z_L, z_N)
        per_step_growth = amp ** (1.0 / n_steps) if amp > 0 else 0.0
        rows.append(dict(alpha=a, omega_dt=omega * dt,
                         coupling=coupling,
                         max_abs_ratio=amp,
                         per_step_growth=per_step_growth,
                         rho_theory=rho_theory))
    return rows


# ── Main ───────────────────────────────────────────────────────────────
def main():
    out_dir = SCRIPT_DIR.parent / "review" / "r21_imex_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_contour = []
    all_boundary = []
    all_scan = []
    for coupling in ("imaginary", "real", "mixed"):
        print(f"\n=== Part A: contour, coupling = {coupling} ===")
        rows_A = sweep_contour(coupling=coupling)
        all_contour.extend(rows_A)

        omegas = sorted(set(r["omega_dt"] for r in rows_A))
        print(f"  ωdt      α_crit  (coupling={coupling})")
        print("  " + "-" * 40)
        for wdt in omegas[::8]:
            alphas_here = [r for r in rows_A if abs(r["omega_dt"] - wdt) < 1e-9]
            unst = [r for r in alphas_here if r["rho_G"] > 1.0]
            a_crit = min(r["alpha_amp"] for r in unst) if unst else float("nan")
            print(f"  {wdt:.3f}    {a_crit:.4f}")
            all_boundary.append(dict(coupling=coupling,
                                     omega_dt=wdt,
                                     alpha_critical=a_crit))

        print(f"\n=== Part B: amp scan, ωdt = 0.033, coupling = {coupling} ===")
        rows_B = sweep_amp_scan(omega=1.64, dt=2e-2, n_steps=800, coupling=coupling)
        all_scan.extend(rows_B)
        for r in rows_B[::4]:
            print(f"  α={r['alpha']:.3e}  max|x|/|x0|={r['max_abs_ratio']:.3e}  "
                  f"per-step={r['per_step_growth']:.6f}  ρ(G)={r['rho_theory']:.6f}")

    with open(out_dir / "contour.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_contour[0].keys()))
        w.writeheader()
        for r in all_contour:
            w.writerow(r)
    with open(out_dir / "boundary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_boundary[0].keys()))
        w.writeheader()
        for r in all_boundary:
            w.writerow(r)
    with open(out_dir / "amp_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_scan[0].keys()))
        w.writeheader()
        for r in all_scan:
            w.writerow(r)

    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
