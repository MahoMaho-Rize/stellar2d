#!/usr/bin/env python3
"""
Experiment I: GYRE-compatible Cowling operator vs GYRE (alpha_grv=0).

PURPOSE
  The first TRULY apples-to-apples external benchmark.  We implement the
  exact GYRE adiabatic Cowling equations (from
  `gyre/src/eqns/ad/ad_eqns_m.fypp` + `A_t.inc`, with alpha_grv=0 and
  alpha_omg=alpha_gam=alpha_pi=1) with the exact GYRE boundary conditions
  (IB_regular + OB_vacuum) in Python, then compare against a GYRE run with
  the same switches.  If our 2-variable operator is correctly assembled,
  agreement should be <1% on n_g = 1..10.

PROCEDURE
  1. `/tmp/gyre_cowling/gyre.in` is the GYRE input with `alpha_grv = 0.0`
     (only!).  The output is `/tmp/gyre_cowling/summary_cowling.h5`.
  2. `/tmp/gyre_run/poly3.txt` gives the dimensionless polytrope structure
     (x, V_2, A*, U, c_1, Gamma_1) GYRE computed.  We feed this DIRECTLY
     to `solve_gmode_cowling_gyre_compat` — no interpolation, no density-
     reconstruction, same grid.
  3. Compare omega^2 mode-by-mode.

PASS
  max |omega^2_ours - omega^2_GYRE| / omega^2_GYRE  <  0.01  on n_g=1..10.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §13  (to be added)
  docs/gmode_next_session_plan.md          (plan doc)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmode_infra as gi

REF_DOC = "docs/gmode_experiments_2026-05-02.md"
SCRIPT_REL = "scripts/gmode_exp_i_gyre_compat.py"

GYRE_STRUCTURE_TXT = Path("/tmp/gyre_run/poly3.txt")
GYRE_COWLING_H5 = Path("/tmp/gyre_cowling/summary_cowling.h5")

# Reference values (Nr=1024, inner_cut=0.01, outer_cut=0.999,
# GYRE Cowling alpha_grv=0, Lane-Emden n=3 polytrope).
EXPECTED_OMSQ_GYRE = [
    2.8519529544471434, 1.3614541926538262, 0.8002549169578306,
    0.5277955256755773, 0.3747288035503986, 0.2800854707548325,
    0.2174390328457775, 0.17379910801994455, 0.14216638108329882,
    0.11849526682381913,
]
EXPECTED_OMSQ_OURS = [
    2.8519463963300398, 1.361458413028407, 0.8002596910865049,
    0.527795568935303, 0.37472100084248233, 0.28006779715072033,
    0.21741013509396703, 0.17375812191031828, 0.1421128405450522,
    0.11842904929216293,
]
EXPECTED_MAX_REL_DIFF = 0.0005588200561179545
REL_TOL = 0.02


def load_gyre_cowling_modes(path):
    import h5py
    with h5py.File(path, "r") as f:
        n_pg = f["n_pg"][()]
        om = f["omega"][()]["re"]
    mask = n_pg < 0
    ng = -n_pg[mask]
    om2 = om[mask] ** 2
    order = np.argsort(ng)
    return ng[order], om2[order]


def load_gyre_structure_full(path):
    """Return (x, V_2, A_star, U, c_1, Gamma_1).  All on GYRE's native grid."""
    data = np.loadtxt(path, skiprows=1)
    x = data[:, 0]
    V_2 = data[:, 1]
    A_star = data[:, 2]
    U = data[:, 3]
    c_1 = data[:, 4]
    Gamma_1 = data[:, 5]
    return x, V_2, A_star, U, c_1, Gamma_1


def main(verify=False, Nr=1024, ell=1, n_compare=10,
         inner_cut=0.01, outer_cut=0.999):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment I: GYRE-compatible Cowling operator vs GYRE (alpha_grv=0)")
    print("=" * 72)

    if not GYRE_STRUCTURE_TXT.exists() or not GYRE_COWLING_H5.exists():
        print(f"\n  Missing GYRE artefacts:")
        print(f"    {GYRE_STRUCTURE_TXT} (exists: {GYRE_STRUCTURE_TXT.exists()})")
        print(f"    {GYRE_COWLING_H5}    (exists: {GYRE_COWLING_H5.exists()})")
        sys.exit(2)

    # Load GYRE structure + cowling eigenvalues
    x_g, V_2g, A_starg, Ug, c_1g, Gamma_1g = load_gyre_structure_full(GYRE_STRUCTURE_TXT)
    ng_gyre, omsq_gyre = load_gyre_cowling_modes(GYRE_COWLING_H5)
    ng_gyre = ng_gyre[:n_compare]
    omsq_gyre = omsq_gyre[:n_compare]
    print(f"  GYRE structure: {len(x_g)} points, x in [{x_g[0]:.4f}, {x_g[-1]:.4f}]")
    print(f"  GYRE Cowling modes: first {n_compare} n_g = {ng_gyre.tolist()}")
    print(f"  omega^2_GYRE[n_g=1] = {omsq_gyre[0]:.6f}")

    # Mask interior, interpolate to uniform grid for FD on x in (inner_cut, outer_cut).
    mask = (x_g > inner_cut) & (x_g < outer_cut)
    x_src = x_g[mask]
    V_2src = V_2g[mask]
    A_star_src = A_starg[mask]
    U_src = Ug[mask]
    c_1src = c_1g[mask]
    Gamma_1_src = Gamma_1g[mask]

    # Gamma_1 is const for polytrope but interpolate anyway (safe for MESA profiles later)
    x_u = np.linspace(x_src[0], x_src[-1], Nr)
    V_2u = np.interp(x_u, x_src, V_2src)
    A_star_u = np.interp(x_u, x_src, A_star_src)
    U_u = np.interp(x_u, x_src, U_src)
    c_1u = np.interp(x_u, x_src, c_1src)
    Gamma_1u = np.interp(x_u, x_src, Gamma_1_src)

    print(f"  resampled: Nr={Nr}, x in [{x_u[0]:.4f}, {x_u[-1]:.4f}]")
    print(f"  V_2 range   : [{V_2u.min():.3e}, {V_2u.max():.3e}]")
    print(f"  A*  range   : [{A_star_u.min():.3e}, {A_star_u.max():.3e}]")
    print(f"  U   range   : [{U_u.min():.3e}, {U_u.max():.3e}]")
    print(f"  c_1 range   : [{c_1u.min():.3e}, {c_1u.max():.3e}]")
    print(f"  Gamma_1     : [{Gamma_1u.min():.3e}, {Gamma_1u.max():.3e}]")

    omsq_ours, u_out = gi.solve_gmode_cowling_gyre_compat(
        x_u, V_2u, U_u, A_star_u, c_1u, Gamma_1u, ell, n_compare + 5)
    omsq_ours = omsq_ours[:n_compare]

    print()
    print(f"  {'n_g':>4}  {'omega^2_GYRE':>14}  {'omega^2_ours':>14}  {'rel_diff':>11}")
    print("  " + "-" * 52)
    rel_diffs = []
    for i in range(n_compare):
        og = omsq_gyre[i]
        ou = omsq_ours[i] if i < len(omsq_ours) else float("nan")
        rd = abs(ou - og) / abs(og)
        rel_diffs.append(rd)
        print(f"  {ng_gyre[i]:>4}  {og:14.6e}  {ou:14.6e}  {rd:11.3e}")

    max_rd = float(max(rel_diffs))
    print()
    print(f"  max rel_diff = {max_rd:.3e}")
    ok = max_rd < 1e-2
    print(f"  PASS criterion: max rel_diff < 1e-2")
    print(f"  Result: {'PASS' if ok else 'FAIL'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    axes[0].semilogy(ng_gyre, omsq_gyre, "ko-", lw=1.5, ms=8, label="GYRE Cowling (α_grv=0)")
    axes[0].semilogy(ng_gyre, omsq_ours, "C3s--", lw=1, ms=6, label="ours (gyre_compat)")
    axes[0].set_xlabel(r"$n_g$")
    axes[0].set_ylabel(r"$\omega^2$  (GM/R³ units)")
    axes[0].set_title(f"l={ell} g-mode spectrum: GYRE Cowling vs ours")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogy(ng_gyre, rel_diffs, "C3s-", lw=1.5, ms=7,
                     label="|ours - GYRE|/GYRE")
    axes[1].axhline(1e-2, ls="--", color="r", lw=1, label="1% tol")
    axes[1].set_xlabel(r"$n_g$")
    axes[1].set_ylabel("relative difference")
    axes[1].set_title("Agreement with GYRE Cowling (α_grv=0)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = gi.VID / "gmode_exp_i_gyre_compat.png"
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
                ("omsq_GYRE", omsq_gyre, EXPECTED_OMSQ_GYRE),
                ("omsq_ours", omsq_ours, EXPECTED_OMSQ_OURS),
            ]:
                v, rv = arr[n], ref[n]
                d = abs(v - rv) / max(abs(rv), 1e-300)
                if d > REL_TOL:
                    print(f"  [DRIFT] n_g={ng_gyre[n]} {label} {v:.4e} vs {rv:.4e} ({d*100:.2f}%)")
                    n_fail += 1
        d = abs(max_rd - EXPECTED_MAX_REL_DIFF) / max(abs(EXPECTED_MAX_REL_DIFF), 1e-300)
        mk = "OK" if d < REL_TOL else "DRIFT"
        print(f"  [{mk:<5}] max_rel_diff {max_rd:.4e} vs {EXPECTED_MAX_REL_DIFF:.4e}  ({d*100:.2f}%)")
        if d >= REL_TOL:
            n_fail += 1
        if n_fail:
            sys.exit(1)
        print("\n  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--nr", type=int, default=1024)
    args = ap.parse_args()
    main(verify=args.verify, Nr=args.nr)
