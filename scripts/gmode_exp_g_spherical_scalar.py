#!/usr/bin/env python3
"""
Experiment G: spherical scalar reduction vs 2-variable operator.

PURPOSE
  Rigorous cross-check: the 2-variable system (xi_r, p') and the fully
  spherical scalar reduction psi = rho0 r^2 xi_r are ALGEBRAICALLY
  equivalent (see docs §10 for the sympy-verified derivation).  Their
  omega^2 spectra must therefore agree to within discretisation error
  at every n, not just in a high-n limit.

  This experiment replaces the earlier incorrect claim (Exp E §7.3) that
  the slab-Cowling solver is the "Boussinesq limit" of the 2-variable
  operator.  That claim conflated two distinct issues:
    * the slab-vs-spherical geometry (the actual source of the ratio != 1)
    * any Boussinesq / compressibility truncation (neither solver makes
      such an approximation in the g-mode-only form)

  Passing Exp G requires the two spectra to agree at every n, with the
  error driven by the shared 2nd-order FD discretisation of the 1/r^2
  operator (expected O(Nr^{-2})).  If they do not agree at LOW n too,
  the 2-variable matrix has a bug.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §11

REPRO
  python scripts/gmode_exp_g_spherical_scalar.py
  python scripts/gmode_exp_g_spherical_scalar.py --verify
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
from gmode_exp_e_anelastic_linop import solve_anelastic_2var

REF_DOC = "docs/gmode_experiments_2026-05-02.md"
SCRIPT_REL = "scripts/gmode_exp_g_spherical_scalar.py"


def n2_profile(r, r_lo=0.2, r_hi=1.0):
    """Same Gaussian bump as Exps B/C/D/E/F."""
    rc = 0.5 * (r_lo + r_hi)
    sigma = 0.25 * (r_hi - r_lo)
    bump = np.exp(-((r - rc) / sigma) ** 2)
    taper = np.sin(np.pi * (r - r_lo) / (r_hi - r_lo)) ** 2
    return bump * taper


# Reference values.  Exp G is a two-pipeline algebraic-equivalence check,
# so we pin:
#   (a) each spectrum entry at Nr=512,
#   (b) the FD-truncation constant Nr^2 * max_rd, which should plateau.
EXPECTED_OMSQ_SCALAR_SPH_NR512 = [
    1.65060466e-01, 2.96676620e-02, 1.19388795e-02, 6.40358201e-03,
    3.98577333e-03, 2.71788157e-03, 1.97132971e-03, 1.49501083e-03,
    1.17264144e-03, 9.44371492e-04,
]
EXPECTED_OMSQ_2VAR_NR512 = [
    1.65062302e-01, 2.96696075e-02, 1.19409612e-02, 6.40571891e-03,
    3.98793766e-03, 2.72006177e-03, 1.97352021e-03, 1.49720870e-03,
    1.17484500e-03, 9.46579756e-04,
]
EXPECTED_SCALED = {  # Nr -> Nr^2 * max_rel_diff
    128:  6.522e+02,
    256:  6.213e+02,
    512:  6.130e+02,
    1024: 6.103e+02,
}
REL_TOL = 0.02


def convergence_sweep(r_lo, r_hi, ell, n_compare, Nr_list):
    """Compute max relative difference between spherical scalar and 2-var
    spectra on a sequence of uniform grids; the difference should decay
    roughly as Nr^{-2} for 2nd-order FD."""
    results = {}
    for Nr in Nr_list:
        r = np.linspace(r_lo, r_hi, Nr)
        N2 = n2_profile(r, r_lo, r_hi)
        rho0 = np.ones_like(r)
        omsq_sph, _ = gi.solve_gmode_cowling_spherical(r, N2, ell, n_compare)
        omsq_2var = solve_anelastic_2var(r, rho0, N2, ell, n_compare)
        omsq_sph = np.sort(omsq_sph)[::-1]
        omsq_2var = np.sort(omsq_2var)[::-1]
        rel = np.abs(omsq_2var - omsq_sph) / np.abs(omsq_sph)
        results[Nr] = {
            "omsq_sph": omsq_sph,
            "omsq_2var": omsq_2var,
            "rel": rel,
            "max_rel": float(rel.max()),
        }
    return results


def main(verify=False):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment G: spherical scalar vs 2-variable operator")
    print("=" * 72)

    r_lo, r_hi = 0.2, 1.0
    ell = 1
    n_compare = 10

    # (1) Main comparison at a representative resolution
    Nr = 512
    r = np.linspace(r_lo, r_hi, Nr)
    N2 = n2_profile(r, r_lo, r_hi)
    rho0 = np.ones_like(r)

    omsq_sph, _ = gi.solve_gmode_cowling_spherical(r, N2, ell, n_compare)
    omsq_2var = solve_anelastic_2var(r, rho0, N2, ell, n_compare)
    # Both represent g-modes; align by descending order (n=1 highest omega^2).
    omsq_sph = np.sort(omsq_sph)[::-1]
    omsq_2var = np.sort(omsq_2var)[::-1]

    print(f"  Nr = {Nr},  ell = {ell},  cavity [{r_lo}, {r_hi}],  rho0 = 1")
    print(f"  N² max = {N2.max():.4f}")
    print()
    print(f"  {'n':>3}  {'ω²_scalar_sph':>17}  {'ω²_2var':>15}  "
          f"{'rel_diff':>12}")
    print("  " + "-" * 58)
    rel_diffs = []
    for n in range(n_compare):
        om_s = omsq_sph[n]
        om_v = omsq_2var[n]
        rd = abs(om_v - om_s) / abs(om_s)
        rel_diffs.append(rd)
        print(f"  {n+1:>3}  {om_s:17.8e}  {om_v:15.8e}  {rd:12.3e}")

    max_rd = max(rel_diffs)
    print()
    print(f"  max rel_diff at Nr={Nr} = {max_rd:.3e}")

    # (2) Convergence sweep: the key evidence is O(Nr^{-2}) decay of max rel diff,
    # which certifies algebraic equivalence + 2nd-order FD.
    print()
    print("  --- Nr convergence sweep ---")
    sweep = convergence_sweep(r_lo, r_hi, ell, n_compare,
                               Nr_list=[128, 256, 512, 1024])
    print(f"  {'Nr':>6}  {'max rel_diff':>16}  {'(Nr^2 × max_rd)':>16}")
    scaled_vals = []
    for Nr_s, res in sweep.items():
        scaled = res["max_rel"] * Nr_s**2
        scaled_vals.append(scaled)
        print(f"  {Nr_s:>6}  {res['max_rel']:16.3e}  {scaled:16.3e}")
    # The scaled column should plateau to a constant (the FD operator
    # truncation coefficient).  If the 2-var operator had a bug that did
    # not vanish in the continuum limit, this column would grow or
    # oscillate.
    scaled_arr = np.array(scaled_vals)
    scaled_variation = scaled_arr.max() / scaled_arr.min() - 1.0
    print(f"  scaled-column variation = {scaled_variation*100:.1f}%")
    print(f"  PASS criterion: Nr^2 * max_rd varies by < 10%  (O(Nr^{{-2}}) FD)")
    ok = scaled_variation < 0.10
    print(f"  Result: {'PASS' if ok else 'FAIL'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    n_arr = np.arange(1, n_compare + 1)
    axes[0].semilogy(n_arr, omsq_sph, "o-", color="C0", lw=1.5, ms=7,
                     label="scalar (spherical)")
    axes[0].semilogy(n_arr, omsq_2var, "s--", color="C3", lw=1.2, ms=6,
                     label="2-variable")
    axes[0].set_xlabel("radial order n")
    axes[0].set_ylabel(r"$\omega^2$")
    axes[0].set_title(f"Spectrum agreement (ell={ell})")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    for Nr_s, res in sweep.items():
        axes[1].semilogy(np.arange(1, n_compare + 1), res["rel"],
                         "o-", lw=1.2, ms=4, label=f"Nr={Nr_s}")
    axes[1].set_xlabel("radial order n")
    axes[1].set_ylabel("rel diff")
    axes[1].set_title(f"FD convergence: O(Nr⁻²) decay")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = gi.VID / "gmode_exp_g_spherical_scalar.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    if verify:
        print("\n--- VERIFY against EXPECTED ---")
        n_fail = 0
        # Per-mode spectra (at Nr=512)
        for n in range(n_compare):
            for label, arr, ref in [
                ("scalar_sph", omsq_sph, EXPECTED_OMSQ_SCALAR_SPH_NR512),
                ("2var      ", omsq_2var, EXPECTED_OMSQ_2VAR_NR512),
            ]:
                v = arr[n]; ref_v = ref[n]
                d = abs(v - ref_v) / max(abs(ref_v), 1e-300)
                if d > REL_TOL:
                    print(f"  [DRIFT] n={n+1} {label} {v:.4e} vs {ref_v:.4e} ({d*100:.2f}%)")
                    n_fail += 1
        # FD scaled constant
        for Nr_s, expected_scaled in EXPECTED_SCALED.items():
            res = sweep[Nr_s]
            actual = res["max_rel"] * Nr_s**2
            d = abs(actual - expected_scaled) / max(abs(expected_scaled), 1e-300)
            mark = "OK" if d < REL_TOL else "DRIFT"
            print(f"  [{mark:<5}] Nr={Nr_s:<5} Nr²·max_rd {actual:.4e} vs {expected_scaled:.4e}  ({d*100:.2f}%)")
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
