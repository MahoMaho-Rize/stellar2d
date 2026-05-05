#!/usr/bin/env python3
"""Review round-1, item R1.3: decompose the GYRE benchmark error
budget into three independent sources:

  (i)  radial spectral resolution   N_r in {48, 64, 96, 128, 192}
  (ii) profile interpolation order  linear (np.interp) vs cubic spline
  (iii) GYRE source profile density 1000 → 400 → 200 → 100 rows

This lets us answer the reviewer's question: the 3.6e-5 residual is
NOT due to physics-model simplification (we use the full 4-variable
Dziembowski system with Φ', identical to GYRE's adiabatic non-rotating
case, see §4.2), only to discrete approximation — which of the three
discretisation axes contributes most?

Output:
  review/r13_error_budget/sweep_resolution.csv   (i)
  review/r13_error_budget/sweep_interpolation.csv (ii)
  review/r13_error_budget/sweep_gyre_density.csv  (iii)
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gmode_infra as gi
from gmode_exp_k_chebyshev_full import (
    cheb_D1_on_interval,
    solve_gmode_full_chebyshev,
    EXPECTED_OMSQ_GYRE,
)

POLY3_TXT = Path("/tmp/gyre_run/poly3.txt")
OUT_DIR = SCRIPT_DIR.parent / "review" / "r13_error_budget"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_poly3(path=POLY3_TXT):
    data = np.loadtxt(path, skiprows=1)
    return dict(x=data[:, 0], V_2=data[:, 1], A_star=data[:, 2],
                U=data[:, 3], c_1=data[:, 4], Gamma_1=data[:, 5])


def subsample(prof, n_keep):
    """Keep every k-th row to downsample GYRE profile to ~n_keep points
    (used for axis-iii sweep)."""
    n = len(prof["x"])
    idx = np.linspace(0, n - 1, n_keep).astype(int)
    return {k: v[idx] for k, v in prof.items()}


def interp_to_cgl(prof, N, method="cubic", inner_cut=1e-4, outer_cut=0.9999):
    """Interpolate a stellar profile onto CGL nodes of order N over
    [inner_cut, outer_cut]."""
    mask = (prof["x"] > inner_cut) & (prof["x"] < outer_cut)
    x_src = prof["x"][mask]
    x_cgl, _ = cheb_D1_on_interval(N, x_src[0], x_src[-1])

    def do(v):
        if method == "linear":
            return np.interp(x_cgl, x_src, v)
        return CubicSpline(x_src, v)(x_cgl)

    return (x_cgl,
            do(prof["V_2"][mask]),
            do(prof["A_star"][mask]),
            do(prof["U"][mask]),
            do(prof["c_1"][mask]),
            do(prof["Gamma_1"][mask]))


def run_benchmark(prof, N_r, method="cubic", ell=1, n_compare=10):
    """Full pipeline: interpolate → 4-var EVP → rel_err vs frozen GYRE."""
    x, V_2, A_star, U, c_1, G1 = interp_to_cgl(prof, N_r, method=method)
    omsq, _ = solve_gmode_full_chebyshev(x, V_2, U, A_star, c_1, G1, ell, n_compare + 5)
    omsq = omsq[:n_compare]
    if len(omsq) < n_compare:
        omsq = np.concatenate([omsq, np.full(n_compare - len(omsq), np.nan)])
    rel = np.abs(omsq - np.array(EXPECTED_OMSQ_GYRE)) / np.array(EXPECTED_OMSQ_GYRE)
    return omsq, rel


def sweep_resolution(prof, Nr_list):
    """Axis (i): vary N_r, cubic spline profile, full 1000-row GYRE."""
    rows = []
    for N in Nr_list:
        try:
            omsq, rel = run_benchmark(prof, N, method="cubic")
            rd_ng1 = rel[0]
            rd_max = np.nanmax(rel)
            print(f"  (i)  N_r={N:4d}  cubic/1000pts   rd(n_g=1)={rd_ng1:.3e}   max_rd={rd_max:.3e}")
            rows.append(dict(N_r=N, method="cubic", gyre_rows=1000,
                             rd_ng1=rd_ng1, rd_max=rd_max))
        except Exception as e:
            print(f"  (i)  N_r={N:4d}  FAILED: {e}")
            rows.append(dict(N_r=N, method="cubic", gyre_rows=1000,
                             rd_ng1=float("nan"), rd_max=float("nan")))
    return rows


def sweep_interpolation(prof, N_r=96):
    """Axis (ii): fixed N_r = 96, linear vs cubic."""
    rows = []
    for method in ("linear", "cubic"):
        omsq, rel = run_benchmark(prof, N_r, method=method)
        rd_ng1 = rel[0]; rd_max = np.nanmax(rel)
        print(f"  (ii) N_r={N_r}   {method:6s}/1000pts   rd(n_g=1)={rd_ng1:.3e}   max_rd={rd_max:.3e}")
        rows.append(dict(N_r=N_r, method=method, gyre_rows=1000,
                         rd_ng1=rd_ng1, rd_max=rd_max))
    return rows


def sweep_gyre_density(prof_full, N_r=96, densities=(1000, 400, 200, 100)):
    """Axis (iii): fixed N_r = 96, cubic, vary GYRE source rows."""
    rows = []
    for n_rows in densities:
        prof = prof_full if n_rows == 1000 else subsample(prof_full, n_rows)
        omsq, rel = run_benchmark(prof, N_r, method="cubic")
        rd_ng1 = rel[0]; rd_max = np.nanmax(rel)
        print(f"  (iii) N_r={N_r}  cubic/{n_rows:4d}pts   rd(n_g=1)={rd_ng1:.3e}   max_rd={rd_max:.3e}")
        rows.append(dict(N_r=N_r, method="cubic", gyre_rows=n_rows,
                         rd_ng1=rd_ng1, rd_max=rd_max))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print("=" * 72)
    print(" R1.3: GYRE error budget — three-axis sweep")
    print("=" * 72)
    prof = load_poly3()
    print(f" Loaded {len(prof['x'])} rows from {POLY3_TXT}")
    print()

    # Remove surface rows with V_2 or A_star = Inf (GYRE's poly3.txt quirk).
    mask = np.all([np.isfinite(prof[k]) for k in
                   ("V_2", "A_star", "U", "c_1", "Gamma_1")], axis=0)
    prof = {k: v[mask] for k, v in prof.items()}
    print(f" After Inf-row filter: {len(prof['x'])} rows")
    print()

    print(" --- Axis (i): N_r resolution scan ---")
    rows_i = sweep_resolution(prof, [48, 64, 96, 128, 192])
    write_csv(rows_i, OUT_DIR / "sweep_resolution.csv")

    print()
    print(" --- Axis (ii): interpolation order (N_r=96) ---")
    rows_ii = sweep_interpolation(prof, N_r=96)
    write_csv(rows_ii, OUT_DIR / "sweep_interpolation.csv")

    print()
    print(" --- Axis (iii): GYRE source density (N_r=96, cubic) ---")
    rows_iii = sweep_gyre_density(prof, N_r=96)
    write_csv(rows_iii, OUT_DIR / "sweep_gyre_density.csv")

    print()
    print(f" Wrote CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
