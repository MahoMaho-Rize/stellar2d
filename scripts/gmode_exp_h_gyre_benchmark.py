#!/usr/bin/env python3
"""
Experiment H: external benchmark against GYRE.

PURPOSE
  First truly-independent numerical validation.  GYRE (Townsend 2013,
  https://gyre.readthedocs.io) is the standard Fortran stellar-oscillation
  solver, widely used and independently implemented.  Exps B-G all
  share our Python codebase; Exp H is the external cross-check.

PROCEDURE
  1. Take GYRE's shipped Lane-Emden n=3 polytrope structure (dimensionless
     x, V_2, A*, U, c_1, Gamma_1 on 1001 points).  GYRE computes all of
     these internally from the Emden theta solution.
  2. GYRE solves the adiabatic Cowling equations on this model and
     produces omega_GYRE^2 for each ell=1 g-mode.  We use the shipped
     `build_poly` output and the `gyre` binary built earlier.
  3. Convert the dimensionless structure to the (r, rho, N^2) triple our
     Python solvers consume:
          r = x        (dimensionless radius, R = 1)
          N^2 = A* / c_1    (dimensionless Brunt^2, GM/R^3 = 1 unit)
          rho = a function of x derived from the polytropic EOS
                (the 2-var solver uses rho0 in continuity only)
  4. Run `solve_anelastic_2var` and `solve_gmode_cowling_spherical` on
     the same dimensionless structure.  The eigenvalues we obtain are
     directly omega^2 in GYRE's (GM/R^3) unit system.
  5. Compare mode-by-mode.

  If the agreement at each n_g is good (say, within 1% for n_g <= 10),
  this certifies that our 2-var assembly matches an independent, widely-
  used external code -- not just our own scalar reduction.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §12

REPRO
  # Prerequisites (one-time):
  #   1. MESA SDK installed at $HOME/mesasdk-*
  #   2. GYRE cloned at $HOME/gyre with submodules, built (make in
  #      same shell that sourced mesasdk_init.sh)
  # Then the standard reference run is carried out once by
  #   bash scripts/gmode_exp_h_run_gyre.sh
  # which writes /tmp/gyre_run/summary.h5 (the GYRE output).
  #
  # Once that file exists:
  python scripts/gmode_exp_h_gyre_benchmark.py
  python scripts/gmode_exp_h_gyre_benchmark.py --verify
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmode_infra as gi
from gmode_exp_e_anelastic_linop import solve_anelastic_2var

REF_DOC = "docs/gmode_experiments_2026-05-02.md"
SCRIPT_REL = "scripts/gmode_exp_h_gyre_benchmark.py"

GYRE_RUN_DIR = Path("/tmp/gyre_run")
GYRE_STRUCTURE_TXT = GYRE_RUN_DIR / "poly3.txt"
GYRE_SUMMARY_H5 = GYRE_RUN_DIR / "summary.h5"

# Reference values filled after first clean run.
EXPECTED_OMSQ_GYRE = [0.0] * 10
EXPECTED_OMSQ_OURS_2VAR = [0.0] * 10
EXPECTED_MAX_REL_DIFF = 0.0
REL_TOL = 0.02


def load_gyre_structure(path):
    """Load dimensionless polytrope structure produced by `poly_to_txt`:
    columns x V_2 A* U c_1 Gamma_1 on the first 1001 rows (poly n=3).

    Derive the triple (x, rho, N2_dimless) that our Python solvers consume.
    In GYRE dimensionless units: GM/R^3 = 1 (frequency scale),
    R = 1 (length scale), so:
      - x = r / R = r (dimensionless radius)
      - N^2 / (GM/R^3) = A*/c_1    (derivation in Exp H docstring)
    The density normalisation does not enter the Cowling scalar equation
    beyond the rho_0 r^2 weight, but our 2-variable operator DOES need
    a density profile.  Since the Cowling equation is scale-invariant in
    rho_0 (it only uses rho_0 through 1/rho_0 * d(rho_0 r^2 xi_r)/dr),
    we supply the unnormalised polytropic profile rho(x) = theta(xi)^n
    with xi = x * xi_1, theta = 1 at centre.
    """
    data = np.loadtxt(path, skiprows=1)
    x = data[:, 0]
    V_2 = data[:, 1]
    A_star = data[:, 2]
    U = data[:, 3]
    c_1 = data[:, 4]
    Gamma_1 = data[:, 5]

    # N^2 in GM/R^3 units, derived in the Exp H docstring / §12:
    #   rN^2/g = A*,  g = GM_r/r^2,  M_r/M = x^3/c_1
    #   => N^2 = A* * g / r = A* * GM / (c_1 R^3)
    #   => N^2 / (GM/R^3) = A*/c_1
    with np.errstate(divide="ignore", invalid="ignore"):
        N2 = np.where(c_1 > 0, A_star / c_1, 0.0)
    # At x=0, A*=0 and c_1 is finite; N^2(0) = 0.

    # Density profile: rho/rho_c = theta^n.  We reconstruct theta from the
    # Lane-Emden solution consistent with GYRE's shipped structure by
    # integrating theta'' + (2/xi)theta' + theta^n = 0 to the same xi_1.
    # But all we need is a MONOTONE decreasing rho; the Cowling eigenvalue
    # has been shown to depend on N^2 and the spatial weight structure, not
    # on the absolute normalisation.  We recompute rho from the Lane-Emden
    # solution to keep things clean.
    from gmode_infra import solve_lane_emden
    xi_arr, theta_arr, xi_1 = solve_lane_emden(n=3.0)
    r_over_R = xi_arr / xi_1
    rho_ref = np.where(theta_arr > 0, theta_arr ** 3, 0.0)
    # Interpolate onto the GYRE x grid
    rho = np.interp(x, r_over_R, rho_ref)

    return x, rho, N2


def load_gyre_modes(path):
    """Return (n_g, omega_sq) sorted by ascending n_g (n_g = -n_pg)."""
    import h5py
    with h5py.File(path, "r") as f:
        n_pg = f["n_pg"][()]
        omega = f["omega"][()]["re"]
    mask = n_pg < 0          # g-modes have negative n_pg
    n_g = -n_pg[mask]
    om_sq = omega[mask] ** 2
    order = np.argsort(n_g)
    return n_g[order], om_sq[order]


def main(verify=False):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment H: external benchmark against GYRE")
    print("=" * 72)

    if not GYRE_SUMMARY_H5.exists() or not GYRE_STRUCTURE_TXT.exists():
        print(f"\n  Missing GYRE artefacts:")
        print(f"    {GYRE_SUMMARY_H5}  (exists: {GYRE_SUMMARY_H5.exists()})")
        print(f"    {GYRE_STRUCTURE_TXT} (exists: {GYRE_STRUCTURE_TXT.exists()})")
        print(f"\n  Run first:  bash scripts/gmode_exp_h_run_gyre.sh")
        sys.exit(2)

    # Load GYRE structure + eigenvalues.  Also load the full structure
    # coefficients for the gyre_compat operator.
    x, rho, N2 = load_gyre_structure(GYRE_STRUCTURE_TXT)
    # Full structure (x, V_2, A_star, U, c_1, Gamma_1) on GYRE's native grid.
    _gyre_data = np.loadtxt(GYRE_STRUCTURE_TXT, skiprows=1)
    x_full     = _gyre_data[:, 0]
    V_2_full   = _gyre_data[:, 1]
    A_star_full= _gyre_data[:, 2]
    U_full     = _gyre_data[:, 3]
    c_1_full   = _gyre_data[:, 4]
    Gamma_1_full = _gyre_data[:, 5]
    ng_gyre, omsq_gyre = load_gyre_modes(GYRE_SUMMARY_H5)
    n_compare = 10
    # GYRE may miss modes near freq_max; take the first n_compare n_g values.
    ng_gyre = ng_gyre[:n_compare]
    omsq_gyre = omsq_gyre[:n_compare]
    print(f"  GYRE structure: {len(x)} points,  x in [{x[0]:.4f}, {x[-1]:.4f}]")
    print(f"  GYRE g-modes: first 10 n_g = {ng_gyre.tolist()}")

    # GYRE uses r=0 regular + vacuum-at-surface; our FD solvers use
    # Dirichlet psi=0 at both endpoints.  Artificially cutting the domain
    # changes the spectrum.  To minimise cutoff artefact we push the
    # domain as close to [0, 1] as numerically safe (avoiding the
    # centrifugal 1/r^2 at r=0 and the N^2 pole at r=1).
    inner_cut, outer_cut = 0.01, 0.99
    mask = (x > inner_cut) & (x < outer_cut) & np.isfinite(N2)
    r = x[mask]
    rho_cut = rho[mask]
    N2_cut = N2[mask]
    # Resample onto a uniform grid to match our FD solvers
    Nr = 1024
    r_u = np.linspace(r[0], r[-1], Nr)
    rho_u = np.interp(r_u, r, rho_cut)
    N2_u = np.interp(r_u, r, N2_cut)

    print(f"  resampled: Nr={Nr}, r in [{r_u[0]:.4f}, {r_u[-1]:.4f}]")
    print(f"  rho range: [{rho_u.min():.3e}, {rho_u.max():.3e}]")
    print(f"  N^2 range (dimless, GM/R^3=1): [{N2_u.min():.3e}, {N2_u.max():.3e}]")
    print(f"  N^2 at inner cutoff ({r_u[0]:.3f}): {N2_u[0]:.3e}")

    ell = 1
    # (1) Our scalar spherical reduction (Boussinesq-like, no V/U/Gamma_1)
    omsq_sph, _ = gi.solve_gmode_cowling_spherical(r_u, N2_u, ell, n_compare + 5)
    omsq_sph = np.sort(omsq_sph)[::-1][:n_compare]

    # (2) Our 2-variable incompressible-buoyancy operator (no V/U/Gamma_1)
    omsq_2var = solve_anelastic_2var(r_u, rho_u, N2_u, ell, n_compare + 5)
    omsq_2var = np.sort(omsq_2var)[::-1][:n_compare]

    # (3) Our GYRE-compatible 2-variable Cowling operator (uses V_2, U, A*,
    # c_1, Gamma_1 — the full set of structure coefficients).  This should
    # agree with GYRE's own Cowling (alpha_grv=0) to ~1e-4, and with the
    # full-gravity GYRE run (the target here) to ~13% (the Cowling
    # approximation error at low n_g).
    V_2_cut   = V_2_full[(x_full > inner_cut) & (x_full < outer_cut)]
    A_cut     = A_star_full[(x_full > inner_cut) & (x_full < outer_cut)]
    U_cut     = U_full[(x_full > inner_cut) & (x_full < outer_cut)]
    c_1_cut   = c_1_full[(x_full > inner_cut) & (x_full < outer_cut)]
    G1_cut    = Gamma_1_full[(x_full > inner_cut) & (x_full < outer_cut)]
    x_src     = x_full[(x_full > inner_cut) & (x_full < outer_cut)]
    V_2_u  = np.interp(r_u, x_src, V_2_cut)
    A_u    = np.interp(r_u, x_src, A_cut)
    U_u    = np.interp(r_u, x_src, U_cut)
    c_1_u  = np.interp(r_u, x_src, c_1_cut)
    G1_u   = np.interp(r_u, x_src, G1_cut)
    omsq_compat, _ = gi.solve_gmode_cowling_gyre_compat(
        r_u, V_2_u, U_u, A_u, c_1_u, G1_u, ell, n_compare + 5)
    omsq_compat = omsq_compat[:n_compare]

    print()
    print(f"  {'n_g':>4}  {'ω²_GYRE':>12}  {'ω²_sph':>12}  {'ω²_2var':>12}  "
          f"{'ω²_compat':>12}  {'rd_sph':>9}  {'rd_2var':>9}  {'rd_compat':>9}")
    print("  " + "-" * 92)

    rel_diffs_2var = []
    rel_diffs_sph = []
    rel_diffs_compat = []
    for i in range(n_compare):
        om_g = omsq_gyre[i]
        om_v = omsq_2var[i] if i < len(omsq_2var) else float("nan")
        om_s = omsq_sph[i]  if i < len(omsq_sph)  else float("nan")
        om_c = omsq_compat[i] if i < len(omsq_compat) else float("nan")
        rd_v = abs(om_v - om_g) / abs(om_g)
        rd_s = abs(om_s - om_g) / abs(om_g)
        rd_c = abs(om_c - om_g) / abs(om_g)
        rel_diffs_2var.append(rd_v); rel_diffs_sph.append(rd_s); rel_diffs_compat.append(rd_c)
        print(f"  {ng_gyre[i]:>4}  {om_g:12.4e}  {om_s:12.4e}  {om_v:12.4e}  "
              f"{om_c:12.4e}  {rd_s:9.2e}  {rd_v:9.2e}  {rd_c:9.2e}")

    max_rd_2var = max(rel_diffs_2var)
    max_rd_sph = max(rel_diffs_sph)
    max_rd_compat = max(rel_diffs_compat)
    print()
    print(f"  max rel_diff (sph vs GYRE)       = {max_rd_sph:.3e}")
    print(f"  max rel_diff (2var vs GYRE)      = {max_rd_2var:.3e}")
    print(f"  max rel_diff (gyre_compat vs GYRE) = {max_rd_compat:.3e}")
    print()
    print(f"  NOTE: GYRE here uses FULL gravity (alpha_grv=1).  Our gyre_compat")
    print(f"        is Cowling (alpha_grv=0).  The gyre_compat-vs-GYRE discrepancy")
    print(f"        at low n_g is the known Cowling-approximation error (~13% for")
    print(f"        n_g=1 on a Lane-Emden n=3 polytrope) -- it is NOT a bug.")
    print(f"        The Boussinesq-like operators (sph, 2var) drop V/U/Gamma_1")
    print(f"        coupling and are off by a larger factor (~2.2x) at low n_g.")
    ok_compat = max_rd_compat < 0.15   # <= 15% = Cowling approximation error

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    axes[0].semilogy(ng_gyre, omsq_gyre,  "ko-",  lw=1.5, ms=8, label="GYRE (full)")
    axes[0].semilogy(ng_gyre, omsq_compat,"C2D--",lw=1, ms=7, label="ours (gyre_compat, Cowling)")
    axes[0].semilogy(ng_gyre, omsq_2var,  "C3s--",lw=1, ms=6, label="ours (2-var Boussinesq)")
    axes[0].semilogy(ng_gyre, omsq_sph,   "C0^--",lw=1, ms=6, label="ours (spherical scalar)")
    axes[0].set_xlabel(r"$n_g$")
    axes[0].set_ylabel(r"$\omega^2$ (GM/R³ units)")
    axes[0].set_title(f"l={ell} g-mode spectrum: ours vs GYRE (full gravity)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogy(ng_gyre, rel_diffs_compat, "C2D-", lw=1.5, ms=7,
                     label="|gyre_compat - GYRE|/GYRE")
    axes[1].semilogy(ng_gyre, rel_diffs_2var, "C3s-", lw=1.5, ms=7,
                     label="|2var - GYRE|/GYRE")
    axes[1].semilogy(ng_gyre, rel_diffs_sph, "C0^-", lw=1.5, ms=7,
                     label="|spherical - GYRE|/GYRE")
    axes[1].axhline(0.15, ls="--", color="gray", lw=1, label="Cowling err (~15%)")
    axes[1].set_xlabel(r"$n_g$")
    axes[1].set_ylabel("relative difference")
    axes[1].set_title("Agreement with GYRE (full gravity)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = gi.VID / "gmode_exp_h_gyre_benchmark.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    if verify:
        print("\n--- VERIFY against EXPECTED ---")
        if EXPECTED_MAX_REL_DIFF == 0.0:
            print("  SKIP: EXPECTED unset (first run)")
            return
        n_fail = 0
        for n in range(n_compare):
            for label, arr, ref in [
                ("ω²_GYRE  ", omsq_gyre, EXPECTED_OMSQ_GYRE),
                ("ω²_ours  ", omsq_2var, EXPECTED_OMSQ_OURS_2VAR),
            ]:
                v, ref_v = arr[n], ref[n]
                d = abs(v - ref_v) / max(abs(ref_v), 1e-300)
                if d > REL_TOL:
                    print(f"  [DRIFT] n_g={ng_gyre[n]} {label} {v:.4e} vs {ref_v:.4e} ({d*100:.2f}%)")
                    n_fail += 1
        d = abs(max_rd_2var - EXPECTED_MAX_REL_DIFF) / max(abs(EXPECTED_MAX_REL_DIFF), 1e-300)
        mark = "OK" if d < REL_TOL else "DRIFT"
        print(f"  [{mark:<5}] max_rel_diff_2var {max_rd_2var:.4e} vs {EXPECTED_MAX_REL_DIFF:.4e}  ({d*100:.2f}%)")
        if d >= REL_TOL:
            n_fail += 1
        if n_fail:
            sys.exit(1)
        print("\n  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    main(verify=args.verify)
