#!/usr/bin/env python3
"""
Experiment F: anelastic 2-variable operator on variable-rho₀ (polytrope).

PURPOSE
  Final operator-level validation before the C++/CUDA assembly.  Experiment
  E verified the 2-variable anelastic operator against the Boussinesq
  scalar Cowling reduction in the simplifying limit rho_0 = const.  This
  experiment relaxes that constraint: feed the Lane-Emden n=3 polytrope
  density rho_0(r) from the MESA-style fixture built in Exp D, with the
  same Gaussian-bump N²(r), and verify:

    (1) The continuity term (1/r²) ∂_r(rho_0 r² xi_r) — which contains the
        rho_0' coefficient that was identically zero in Exp E — is
        correctly discretised.
    (2) The anelastic 2-var spectrum still approaches the scalar Cowling
        spectrum as n grows, with the same monotone-from-below pattern
        observed in Exp E (ratio < 1, ratio → 1 as n → infinity).

  If either check fails, the likely culprit is a sign error or index
  slip in the rho_0' bookkeeping — the single most error-prone piece of
  the operator-assembly code that must later be translated to C++.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §9

REPRO
  python scripts/gmode_exp_f_variable_rho.py
  python scripts/gmode_exp_f_variable_rho.py --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmode_infra as gi
from mesa_profile import build_polytrope_fixture, read_profile
from gmode_exp_e_anelastic_linop import solve_anelastic_2var

REF_DOC = "docs/gmode_experiments_2026-05-02.md"
SCRIPT_REL = "scripts/gmode_exp_f_variable_rho.py"
FIXTURE = gi.VID / "polytrope_fixture.dat"

# Reference values bound to docs §9.  Populated after first clean run.
EXPECTED_OMSQ_SCALAR = [
    1.73123357e-02, 3.63901665e-03, 1.53554060e-03, 8.42644748e-04,
    5.31588350e-04, 3.65715085e-04, 2.66931625e-04, 2.03385425e-04,
    1.60109160e-04, 1.29315109e-04,
]
EXPECTED_OMSQ_2VAR = [
    1.54588381e-02, 3.49660373e-03, 1.50270298e-03, 8.31107382e-04,
    5.26550957e-04, 3.63227416e-04, 2.65619491e-04, 2.02680218e-04,
    1.59744750e-04, 1.29154898e-04,
]
EXPECTED_RATIOS = {
    "ratio_lo": 0.9441,
    "ratio_hi": 0.9977,
}
REL_TOL = 0.02


def main(verify=False):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment F: anelastic operator on variable-rho₀ (Lane-Emden n=3)")
    print("=" * 72)

    # Build / reload the polytrope fixture so the run is fully self-contained.
    build_polytrope_fixture(FIXTURE,
                             cavity_r_lo=0.2, cavity_r_hi=1.0, N2_amp=1.0)
    prof = read_profile(FIXTURE)
    print(f"  fixture: {FIXTURE.name},  {len(prof['r'])} rows")

    # Restrict to the cavity (positive N²) AND impose a density floor so the
    # operator does not see rho_0 -> 0 near the surface (that region is
    # outside the g-mode cavity anyway, but the 1/rho_0 factors in the
    # momentum equation would otherwise blow up at grid edges).
    rho_cut = 0.02
    mask = (prof["N2"] > 1e-10) & (prof["rho"] > rho_cut)
    r_raw = prof["r"][mask]
    rho_raw = prof["rho"][mask]
    N2_raw = prof["N2"][mask]
    if len(r_raw) < 100:
        print(f"  ERROR: cavity has only {len(r_raw)} points after rho_cut")
        sys.exit(2)

    # Resample on uniform grid for both solvers.
    Nr = 512
    ell = 1
    n_compare = 10

    r = np.linspace(r_raw[0], r_raw[-1], Nr)
    rho0 = np.interp(r, r_raw, rho_raw)
    N2 = np.interp(r, r_raw, N2_raw)

    print(f"  grid: Nr={Nr}, r in [{r[0]:.4f}, {r[-1]:.4f}]")
    print(f"  rho₀ range in cavity: [{rho0.min():.3e}, {rho0.max():.3e}]  "
          f"(variation: {rho0.max() / rho0.min():.1f}x)")
    print(f"  N² range: [{N2.min():.3e}, {N2.max():.3e}]")

    # (1) Scalar Cowling (unchanged from Exp D)
    omsq_scalar, _ = gi.solve_gmode_cowling(r, N2, ell=ell, n_modes=n_compare + 5)
    omsq_scalar = omsq_scalar[:n_compare]

    # (2) 2-variable anelastic WITH variable rho_0
    omsq_2var = solve_anelastic_2var(r, rho0, N2, ell, n_compare)

    print()
    print(f"  {'n':>3}  {'ω²_Bouss (scalar)':>18}  {'ω²_anelastic':>15}  "
          f"{'ratio':>10}  {'rel_diff':>10}")
    print("  " + "-" * 65)

    rel_diffs = []
    ratios = []
    for n in range(n_compare):
        om_s = omsq_scalar[n]
        om_v = omsq_2var[n]
        ratio = om_v / om_s
        rd = abs(om_v - om_s) / abs(om_s)
        rel_diffs.append(rd)
        ratios.append(ratio)
        print(f"  {n+1:>3}  {om_s:18.8e}  {om_v:15.8e}  "
              f"{ratio:10.4f}  {rd:10.3e}")

    ratio_lo = float(np.mean(ratios[:3]))
    ratio_hi = float(np.mean(ratios[-3:]))
    hn_conv = abs(ratio_hi - 1.0)
    print()
    print(f"  low-n (n=1..3) avg ratio  = {ratio_lo:.4f}  "
          f"(anelastic < Boussinesq expected from Exp E)")
    print(f"  high-n (n=8..10) avg ratio = {ratio_hi:.4f}  "
          f"(should tend to 1 even with variable ρ₀)")
    print(f"  |ratio_hi - 1|             = {hn_conv:.4f}")
    print()
    print(f"  PASS criterion: |ratio_hi - 1| < 0.05 (tighter than Exp E's 0.2)")
    print(f"  Result: {'PASS' if hn_conv < 0.05 else 'FAIL'}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=140)
    axes[0].semilogy(r, rho0, color="C3", lw=1.5, label=r"$\rho_0(r)$")
    ax0b = axes[0].twinx()
    ax0b.plot(r, N2, color="C2", lw=1.5, label=r"$N^2(r)$")
    ax0b.set_ylabel(r"$N^2(r)$", color="C2")
    axes[0].set_xlabel("r")
    axes[0].set_ylabel(r"$\rho_0(r)$", color="C3")
    axes[0].set_title(f"Lane-Emden n=3 + Gaussian cavity")
    axes[0].grid(alpha=0.3, which="both")

    n_arr = np.arange(1, n_compare + 1)
    axes[1].semilogy(n_arr, omsq_scalar, "o-", color="C0", lw=1.5, ms=7,
                      label="scalar (Boussinesq)")
    axes[1].semilogy(n_arr, omsq_2var, "s--", color="C3", lw=1.2, ms=6,
                      label="2-var (full anelastic)")
    axes[1].set_xlabel("radial order n")
    axes[1].set_ylabel(r"$\omega^2$")
    axes[1].set_title(f"g-mode spectrum, variable $\\rho_0$ (ell={ell})")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    axes[2].plot(n_arr, ratios, "o-", color="C2", lw=1.5, ms=7)
    axes[2].axhline(1.0, ls="--", color="k", lw=1, label="Boussinesq limit")
    axes[2].axhline(1 - 0.05, ls=":", color="r", lw=1, label="5% tol")
    axes[2].set_xlabel("radial order n")
    axes[2].set_ylabel(r"$\omega^2_\mathrm{anel} / \omega^2_\mathrm{Bouss}$")
    axes[2].set_title(f"ratio → 1 at high n  (hi avg = {ratio_hi:.4f})")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = gi.VID / "gmode_exp_f_variable_rho.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    if verify:
        print("\n--- VERIFY against EXPECTED ---")
        n_fail = 0
        runtime = {"ratio_lo": ratio_lo, "ratio_hi": ratio_hi}
        for key, ref in EXPECTED_RATIOS.items():
            val = runtime[key]
            d = abs(val - ref) / max(abs(ref), 1e-300)
            ok = d < REL_TOL
            mark = "OK" if ok else "DRIFT"
            print(f"  [{mark:<5}] {key:<12} {val:.6f} vs {ref:.6f}  ({d*100:.3f}%)")
            if not ok:
                n_fail += 1
        for n in range(n_compare):
            om_v = omsq_2var[n]
            om_s = omsq_scalar[n]
            ref_v = EXPECTED_OMSQ_2VAR[n]
            ref_s = EXPECTED_OMSQ_SCALAR[n]
            dv = abs(om_v - ref_v) / max(abs(ref_v), 1e-300)
            ds = abs(om_s - ref_s) / max(abs(ref_s), 1e-300)
            if dv > REL_TOL or ds > REL_TOL:
                print(f"  [DRIFT] n={n+1}: 2var {om_v:.4e} vs {ref_v:.4e} "
                      f"({dv*100:.2f}%)  scalar {om_s:.4e} vs {ref_s:.4e} ({ds*100:.2f}%)")
                n_fail += 1
        if n_fail:
            sys.exit(1)
        print("\n  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    main(verify=args.verify)
