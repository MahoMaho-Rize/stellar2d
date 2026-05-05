#!/usr/bin/env python3
"""Review Round-3 addendum (user-proposed reframing): verify that the
true origin of the primitive-node per-step defect is the mass-inverse
inconsistency
    L⁻¹ · R   (correct)
    diag(1/ρ) · R_applied  (what the primitive scheme evaluates)
NOT the discrete Leibniz defect (which is truncation-level, ~1e-6).

For the top g-mode eigenvector v_n:
  L⁻¹ R v_n = ω²_n · v_n         (exact, by EVP)
  diag(1/ρ) R v_n = k² N²(y) · v_n   (pointwise; NOT an eigenvector action)

The ratio
    ε_N := ‖(L⁻¹ - diag(1/ρ)) R v_n‖ / ‖R v_n‖
measures this mismatch.  If the user's framing is correct, ε_N should
be O(1) and N-independent — the smoking gun.

Comparison:
  (a) Full operator residual:  ‖M_prim v_n − M_asm v_n‖
      where M_prim = diag(1/ρ)(R_applied − L_applied)
            M_asm  = L⁻¹ R
  (b) Pure mass-inverse term:  ‖(L⁻¹ − diag(1/ρ)) R v_n‖
  (c) Leibniz defect:           ‖Δ_L v_n‖  (control from P1 data)

All three quantities reported for N_y ∈ {32..256}, Lane-Emden n=3/2.
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

from full_galerkin_closure_test import (
    cgl_grid, cc_weights, bg_lane_emden,
    assemble_operator, evp,
)

OUT_DIR = SCRIPT_DIR.parent / "review" / "r34_mass_inverse"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def probe_one(N_y, Ly=1.0, kx=2 * np.pi, rho_cut=0.05):
    y, D = cgl_grid(N_y, Ly)
    w_cc = cc_weights(N_y, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    drho = D @ rho

    # Assembled operators (interior-restricted)
    L_int, R_int = assemble_operator("vspace", y, D, rho, N2, kx)
    lam, V = evp(L_int, R_int)
    if len(lam) == 0:
        return None
    om2 = float(lam[0])
    v_int = V[:, 0] / np.linalg.norm(V[:, 0])

    rho_i = rho[1:-1]
    N2_i = N2[1:-1]

    # ── (a) Full operator residual (what the time-stepper 'sees') ─────
    # Assembled M_asm · v = L⁻¹ R v  (interior)
    M_asm = np.linalg.solve(L_int, R_int)
    M_asm_v = M_asm @ v_int
    # Primitive M_prim · v := diag(1/ρ) · (R − L)_applied · v
    # R_applied v = k² N² ρ · v    (pointwise)
    # L_applied v = -D diag(ρ) D v + k² ρ v   (full-grid apply, then restrict)
    v_full = np.zeros(N_y); v_full[1:-1] = v_int
    L_app_full = -(D @ (rho * (D @ v_full))) + kx ** 2 * rho * v_full
    R_app_full = kx ** 2 * N2 * rho * v_full
    M_prim_v = ((R_app_full - L_app_full) / np.maximum(rho, 1e-30))[1:-1]
    # But wait — the CUDA primitive scheme computes v̈ = (R v − L v)/ρ
    # whose eigenvalue should be +ω².  The discrete mismatch is:
    resid_full = M_prim_v - M_asm_v
    full_resid_norm = np.linalg.norm(resid_full) / np.linalg.norm(M_asm_v)

    # ── (b) Pure mass-inverse inconsistency ───────────────────────────
    # Compare L⁻¹ R v vs diag(1/ρ) R v
    L_inv_Rv = M_asm_v  # = ω² v_int (by EVP, up to numerical)
    diag_inv_rho_R_v = (R_app_full / np.maximum(rho, 1e-30))[1:-1]
    mass_inv_residual = L_inv_Rv - diag_inv_rho_R_v
    mass_inv_norm = np.linalg.norm(mass_inv_residual) / np.linalg.norm(L_inv_Rv)

    # ── (c) Leibniz defect on same v (for control) ────────────────────
    A1 = D @ np.diag(rho) @ D
    A2 = np.diag(rho) @ (D @ D)
    A3 = np.diag(drho) @ D
    Delta_L = A1 - A2 - A3
    Delta_L_v = (Delta_L @ v_full)[1:-1]
    leibniz_norm = np.linalg.norm(Delta_L_v) / np.linalg.norm(v_int)

    # ── (d) What does diag(1/ρ) R v look like spectrally? ─────────────
    # If pointwise, it's k² N²(y) · v — which is NOT a scalar multiple
    # of v (unless N² is constant).  Measure the "variance" of
    # k² N²(y) across interior:
    k2N2_values = kx ** 2 * N2_i
    k2N2_mean = np.mean(k2N2_values)
    k2N2_std = np.std(k2N2_values)
    # Compare with ω²:
    k2N2_vs_om2 = k2N2_mean / om2

    return dict(
        N_y=N_y,
        omega2=om2,
        full_primitive_residual=float(full_resid_norm),
        mass_inverse_inconsistency=float(mass_inv_norm),
        leibniz_defect=float(leibniz_norm),
        k2N2_mean=float(k2N2_mean),
        k2N2_std=float(k2N2_std),
        k2N2_over_omega2_mean=float(k2N2_vs_om2),
    )


def main():
    N_list = [32, 48, 64, 96, 128, 192, 256, 384]
    rows = []
    print("  Mass-inverse inconsistency probe (Lane-Emden n=3/2)")
    print("  " + "-" * 94)
    print("  N_y   ω²       full-prim    mass-inv    Leibniz    k²N²/ω² mean   k²N² std")
    print("  " + "-" * 94)
    for N in N_list:
        r = probe_one(N)
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['N_y']:4d}  {r['omega2']:.4f}  "
              f"{r['full_primitive_residual']:.3e}  "
              f"{r['mass_inverse_inconsistency']:.3e}  "
              f"{r['leibniz_defect']:.3e}  "
              f"{r['k2N2_over_omega2_mean']:.4f}          "
              f"{r['k2N2_std']:.3f}")

    with open(OUT_DIR / "probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print(f"Wrote {OUT_DIR / 'probe.csv'}")
    print()
    print("  EXPECTED signature of mass-inverse-inconsistency framing:")
    print("    full-prim  ≈  mass-inv       (two are the dominant effect)")
    print("    full-prim  >>  Leibniz       (Leibniz is subleading)")
    print("    mass-inv   N_y-independent   (structural, not truncation)")


if __name__ == "__main__":
    main()
