#!/usr/bin/env python3
"""E2: c(rho_0, n_g, k_x) scan for Proposition 1.

The paper currently reports c(rho_0) ≈ 57 for the Lane-Emden n=3/2
background at (n_g=1, k_x = 2π/L_y).  Reviewers will ask whether this is
a mode-specific coincidence or a general property.  This script answers
that by sweeping the relative Proposition-1 gap (shape metric) over

    n_g in {1, 2, 3, 5, 7, 10}
    k_x / (2π/L_y) in {1, 2, 4, 8}

on Lane-Emden n = 3/2 at N_y = 96 (well-converged, beyond the N_y
floor documented in §5.2).

Outputs:
  - Table of c(n_g, k_x) measured vs predicted (variance formula 5.1b).
  - Scaling check: c should scale roughly linearly in k_x^2 (since
    var_mu(k_x^2 N^2) = k_x^4 var_mu(N^2), and lambda_n ~ k_x^2 N_eff^2,
    giving c ~ sqrt(var_mu(N^2)) / N_eff^2 * k_x^0, i.e. weakly k_x-dependent
    with a weight-dependent correction).
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

OUT_DIR = SCRIPT_DIR.parent / "review" / "e2_c_scan"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_c_at(rho, N2, y, w, D, kx_phys, Ny, n_g):
    """Return c_measured (shape gap) and c_predicted (variance formula)
    for the n_g-th g-mode at wavenumber kx_phys on the given background."""
    intr = slice(1, Ny - 1)
    L = -D @ (np.diag(rho) @ D) + kx_phys**2 * np.diag(rho)
    R = kx_phys**2 * np.diag(N2 * rho)
    L_int = L[intr, intr]
    R_int = R[intr, intr]
    w_int = w[intr]
    rho_int = rho[intr]
    N2_int = N2[intr]

    # assembled generalised EVP: R V = omega^2 L V
    lam, V = scipy.linalg.eig(R_int, L_int)
    lam = np.real(lam); V = np.real(V)
    mask = np.isfinite(lam) & (lam > 0)
    lam, V = lam[mask], V[:, mask]
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]

    if n_g - 1 >= len(lam):
        return dict(n_g=n_g, available=False)

    omega2 = lam[n_g - 1]
    Vn = V[:, n_g - 1]
    Vn = Vn / np.sqrt(np.sum(Vn**2 * w_int))

    # M_asm V_n = omega^2 V_n
    # M_prim V_n = diag(k^2 N^2) V_n  (primitive-node surrogate)
    # Shape gap: project out V_n component
    MpVn = kx_phys**2 * N2_int * Vn
    inner = np.sum(MpVn * Vn * w_int)
    Vn_norm2 = np.sum(Vn * Vn * w_int)
    scalar = inner / Vn_norm2
    orth = MpVn - scalar * Vn
    shape_gap_abs = np.sqrt(np.sum(orth**2 * w_int))
    c_measured = shape_gap_abs / omega2 / np.sqrt(Vn_norm2)

    # Predicted from variance formula
    mu = (Vn**2) * w_int
    mu = mu / np.sum(mu)
    k2N2 = kx_phys**2 * N2_int
    mean_k2N2 = np.sum(k2N2 * mu)
    var_k2N2 = np.sum((k2N2 - mean_k2N2)**2 * mu)
    c_variance = np.sqrt(var_k2N2) / omega2

    return dict(n_g=n_g, available=True, omega2=float(omega2),
                c_measured=float(c_measured),
                c_variance_predicted=float(c_variance),
                k2N2_mean=float(mean_k2N2),
                k2N2_var=float(var_k2N2))


def main():
    print("E2: c(rho_0, n_g, k_x) scan on Lane-Emden n = 3/2")
    print("=" * 78)

    Ly = 1.0
    Ny = 96
    rho_cut = 0.05
    y, D = cgl_grid(Ny, Ly)
    w = cc_weights(Ny, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)

    kx_units = 2.0 * np.pi / Ly
    kx_multipliers = [1, 2, 4, 8]
    n_g_list = [1, 2, 3, 5, 7, 10]

    all_rows = []
    for km in kx_multipliers:
        kx_phys = km * kx_units
        print(f"\n  k_x = {km} * 2π/L_y  (= {kx_phys:.3f})")
        print(f"  {'n_g':>3s}  {'omega^2':>10s}  "
              f"{'c measured':>12s}  {'c variance':>12s}  "
              f"{'<k^2 N^2>':>10s}  {'Var(k^2 N^2)':>12s}")
        print("  " + "-" * 72)
        for n_g in n_g_list:
            r = compute_c_at(rho, N2, y, w, D, kx_phys, Ny, n_g)
            r["kx_multiplier"] = km
            r["kx_phys"] = float(kx_phys)
            if not r.get("available", True):
                print(f"  {n_g:3d}  (mode not present at this k_x)")
                continue
            all_rows.append(r)
            print(f"  {n_g:3d}  {r['omega2']:10.3e}  "
                  f"{r['c_measured']:12.4e}  {r['c_variance_predicted']:12.4e}  "
                  f"{r['k2N2_mean']:10.3e}  {r['k2N2_var']:12.4e}")

    csv_path = OUT_DIR / "c_scan.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["kx_multiplier", "kx_phys", "n_g", "omega2",
                  "c_measured", "c_variance_predicted",
                  "k2N2_mean", "k2N2_var"]
        w_csv = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w_csv.writeheader()
        for r in all_rows:
            w_csv.writerow(r)
    print(f"\nWrote {csv_path}")

    # Scaling summary: c vs k_x at fixed n_g=1
    print("\n" + "=" * 78)
    print("  Scaling summary: c(n_g=1, k_x) as a function of k_x")
    print(f"  {'k_x/(2π/L)':>10s}  {'c measured':>12s}  {'c variance':>12s}  "
          f"{'c / k_x^2':>10s}")
    print("  " + "-" * 54)
    for km in kx_multipliers:
        for r in all_rows:
            if r.get("kx_multiplier") == km and r.get("n_g") == 1:
                print(f"  {km:10d}  {r['c_measured']:12.4e}  "
                      f"{r['c_variance_predicted']:12.4e}  "
                      f"{r['c_measured'] / km**2:10.4e}")

    print("\n  Scaling summary: c(k_x=2π/L, n_g) as a function of n_g")
    print(f"  {'n_g':>3s}  {'c measured':>12s}  {'c variance':>12s}")
    print("  " + "-" * 34)
    for r in all_rows:
        if r.get("kx_multiplier") == 1:
            print(f"  {r['n_g']:3d}  {r['c_measured']:12.4e}  "
                  f"{r['c_variance_predicted']:12.4e}")


if __name__ == "__main__":
    main()
