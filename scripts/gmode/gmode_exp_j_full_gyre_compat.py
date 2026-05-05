#!/usr/bin/env python3
"""
Experiment J: full-gravity 4-variable GYRE-compatible operator vs GYRE.

PURPOSE
  Exp I verified the 2-variable Cowling operator matches GYRE with
  alpha_grv=0 to <1%.  This experiment closes the remaining 13%
  Cowling-approximation gap by comparing our full 4-variable operator
  (alpha_grv=1) against the standard GYRE run at /tmp/gyre_run/summary.h5.

PROCEDURE
  1. Load the same dimensionless polytrope structure (x, V_2, A*, U, c_1,
     Gamma_1) from /tmp/gyre_run/poly3.txt.
  2. Run solve_gmode_full_gyre_compat on the uniform resample.
  3. Compare omega^2 mode-by-mode against GYRE's full-gravity l=1
     g-mode spectrum at /tmp/gyre_run/summary.h5.

PASS
  max |ω²_ours - ω²_GYRE| / ω²_GYRE < 0.01 on n_g=1..10.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md  (§14 to be added)
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
SCRIPT_REL = "scripts/gmode_exp_j_full_gyre_compat.py"

GYRE_STRUCTURE_TXT = Path("/tmp/gyre_run/poly3.txt")
GYRE_FULL_H5 = Path("/tmp/gyre_run/summary.h5")

# Nr=1024, inner_cut=0.01, outer_cut=0.999, GYRE full gravity (alpha_grv=1),
# Lane-Emden n=3 polytrope.  Max rel_diff < 6e-4 (PASS < 1%).
EXPECTED_OMSQ_GYRE = [
    2.5159279360877496, 1.2857077544856306, 0.7757327764772477,
    0.5177759762324133, 0.36992549567563754, 0.2775028154601723,
    0.21592664733814718, 0.1728536032702941, 0.1415440904624304,
    0.11806842352742726,
]
EXPECTED_OMSQ_OURS = [
    2.515926918086264, 1.2857135035640723, 0.7757411482101223,
    0.5177795560861602, 0.369921731845274, 0.2774891906305836,
    0.21590187364264868, 0.17281669693789967, 0.1414946019591157,
    0.1180061944217357,
]
EXPECTED_MAX_REL_DIFF = 0.0005270596814321637
REL_TOL = 0.02


def load_gyre_modes(path):
    import h5py
    with h5py.File(path, "r") as f:
        n_pg = f["n_pg"][()]
        om = f["omega"][()]["re"]
    mask = n_pg < 0
    ng = -n_pg[mask]
    om2 = om[mask] ** 2
    order = np.argsort(ng)
    return ng[order], om2[order]


def main(verify=False, Nr=1024, ell=1, n_compare=10,
         inner_cut=0.01, outer_cut=0.999):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment J: full-gravity 4-variable GYRE-compatible operator")
    print("=" * 72)

    if not GYRE_STRUCTURE_TXT.exists() or not GYRE_FULL_H5.exists():
        print(f"\n  Missing GYRE artefacts. Run scripts/gmode_exp_h_run_gyre.sh first.")
        sys.exit(2)

    data = np.loadtxt(GYRE_STRUCTURE_TXT, skiprows=1)
    x_full = data[:, 0]; V_2 = data[:, 1]; A_star = data[:, 2]
    U = data[:, 3]; c_1 = data[:, 4]; Gamma_1 = data[:, 5]
    ng_gyre, omsq_gyre = load_gyre_modes(GYRE_FULL_H5)
    ng_gyre = ng_gyre[:n_compare]; omsq_gyre = omsq_gyre[:n_compare]

    print(f"  GYRE structure: {len(x_full)} points,  x in [{x_full[0]:.4f}, {x_full[-1]:.4f}]")
    print(f"  GYRE full-gravity modes: first 10 n_g = {ng_gyre.tolist()}")
    print(f"  ω²_GYRE[n_g=1] = {omsq_gyre[0]:.6f}")

    mask = (x_full > inner_cut) & (x_full < outer_cut)
    x_src = x_full[mask]
    x_u = np.linspace(x_src[0], x_src[-1], Nr)
    V_2u = np.interp(x_u, x_src, V_2[mask])
    A_u  = np.interp(x_u, x_src, A_star[mask])
    U_u  = np.interp(x_u, x_src, U[mask])
    c_1u = np.interp(x_u, x_src, c_1[mask])
    G1_u = np.interp(x_u, x_src, Gamma_1[mask])

    print(f"  resampled: Nr={Nr}, x in [{x_u[0]:.4f}, {x_u[-1]:.4f}]")

    omsq_ours, _ = gi.solve_gmode_full_gyre_compat(
        x_u, V_2u, U_u, A_u, c_1u, G1_u, ell, n_compare + 5)
    omsq_ours = omsq_ours[:n_compare]

    print()
    print(f"  {'n_g':>4}  {'ω²_GYRE_full':>14}  {'ω²_ours_full':>14}  {'rel_diff':>11}")
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    axes[0].semilogy(ng_gyre, omsq_gyre, "ko-", lw=1.5, ms=8, label="GYRE (full α_grv=1)")
    axes[0].semilogy(ng_gyre, omsq_ours, "C2D--", lw=1, ms=7, label="ours (4-var full)")
    axes[0].set_xlabel(r"$n_g$")
    axes[0].set_ylabel(r"$\omega^2$ (GM/R³ units)")
    axes[0].set_title(f"l={ell} g-mode spectrum: 4-var full vs GYRE")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogy(ng_gyre, rel_diffs, "C2D-", lw=1.5, ms=7,
                     label="|ours - GYRE|/GYRE")
    axes[1].axhline(1e-2, ls="--", color="r", lw=1, label="1% tol")
    axes[1].set_xlabel(r"$n_g$")
    axes[1].set_ylabel("relative difference")
    axes[1].set_title("Agreement with GYRE (full gravity)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = gi.VID / "gmode_exp_j_full_gyre_compat.png"
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
                ("ω²_ours  ", omsq_ours, EXPECTED_OMSQ_OURS),
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
