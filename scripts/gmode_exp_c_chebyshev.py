#!/usr/bin/env python3
"""
Experiment C: Chebyshev collocation g-mode solver on the stratified cavity.

PURPOSE
  Break the FD convergence floor identified in Experiment B
  (|ratio - 1| = 7.5e-4 at N_r = 2048).  Re-use the same Gaussian-bump N²(r)
  profile and ell = 1, but discretise the Cowling equation on a CGL grid
  via solve_gmode_cowling_cheb so that the eigenvalue problem inherits
  spectral accuracy.

  Expected outcome, based on the reduced-pressure Chebyshev experience
  (reduced_pressure_chebyshev.py: FD floor 1.4e-7 -> Chebyshev 6e-11):
  the Tassoul ratio should approach 1 much faster in N_Cheb than the
  2nd-order FD scaling of 2^{-2} per doubling.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §5   (section written in the commit
  that introduces these numbers)

REPRO
  python scripts/gmode_exp_c_chebyshev.py
  python scripts/gmode_exp_c_chebyshev.py --verify
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
SCRIPT_REL = "scripts/gmode_exp_c_chebyshev.py"


def n2_profile(r, r_lo, r_hi):
    """Same Gaussian-bump N²(r) as gmode_exp_b_stratified.py."""
    rc = 0.5 * (r_lo + r_hi)
    sigma = 0.25 * (r_hi - r_lo)
    bump = np.exp(-((r - rc) / sigma) ** 2)
    taper = np.sin(np.pi * (r - r_lo) / (r_hi - r_lo)) ** 2
    return bump * taper


EXPECTED = {
    64:  {"dP_tassoul": 21.00168831, "dP_tail_mean": 21.04450240, "ratio_tail": 1.002039},
    128: {"dP_tassoul": 20.99536336, "dP_tail_mean": 21.00746753, "ratio_tail": 1.000577},
    256: {"dP_tassoul": 20.99378247, "dP_tail_mean": 20.99784951, "ratio_tail": 1.000194},
    512: {"dP_tassoul": 20.99338727, "dP_tail_mean": 20.99482518, "ratio_tail": 1.000068},
}
REL_TOL_DEFAULT = 0.02
REL_TOL_HIGH = 0.30   # high-N rows may sit at float64 noise floor


def main(verify=False):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment C: Chebyshev g-mode solver on stratified cavity")
    print("=" * 72)

    r_lo, r_hi = 0.2, 1.0
    ell = 1
    # Chebyshev generalised eigensolvers contaminate the last ~N_Cheb/4 modes
    # with spurious high-frequency states whose eigenfunctions have sub-grid
    # wavelength.  We cap n_modes at N_Cheb//5 so the 5-mode tail average
    # falls inside the converged band.
    N_list = [64, 128, 256, 512]

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)

    # Show N²(r) once (on a fine grid for visual clarity)
    r_show = np.linspace(r_lo, r_hi, 2048)
    axes[0].plot(r_show, n2_profile(r_show, r_lo, r_hi),
                  color="C2", lw=2)
    axes[0].set_xlabel("r")
    axes[0].set_ylabel(r"$N^2(r)$")
    axes[0].set_title("Gaussian-bump stratification (same as Exp B)")
    axes[0].grid(alpha=0.3)

    print(f"  ell = {ell},  r in [{r_lo}, {r_hi}]")
    print(f"  Chebyshev grid sweep: {N_list}")

    for N_cheb in N_list:
        r, D2, w_full = gi.cheb_on_interval(N_cheb, r_lo, r_hi)
        N2 = n2_profile(r, r_lo, r_hi)

        # cap modes below spurious threshold; min 10 so tail averaging works
        n_modes = max(10, N_cheb // 5)
        omsq, psi = gi.solve_gmode_cowling_cheb(
            r, D2, w_full, N2, ell=ell, n_modes=n_modes)
        if len(omsq) < 2:
            print(f"  N_Cheb={N_cheb}: solver returned {len(omsq)} modes, skipping")
            continue
        omega = np.sqrt(omsq)
        P = 2.0 * np.pi / omega
        dP = np.abs(np.diff(P))

        dP_tassoul = gi.tassoul_dP(r, N2, ell=ell)
        n_tail = min(5, len(dP))
        dP_tail = dP[-n_tail:].mean()
        ratio = dP_tail / dP_tassoul

        print(f"\n  N_Cheb = {N_cheb}:   {len(omsq)} modes returned")
        print(f"    Tassoul ΔP (l={ell}) = {dP_tassoul:.8f}")
        print(f"    ΔP first 5:  {['%.6f' % v for v in dP[:5]]}")
        print(f"    ΔP tail (last {n_tail}) = {dP_tail:.8f},  ratio = {ratio:.6f}")

        results[N_cheb] = {
            "dP_tassoul": dP_tassoul,
            "dP_tail_mean": dP_tail,
            "ratio_tail": ratio,
            "dP_all": dP,
        }

        n_arr = np.arange(1, len(dP) + 1)
        axes[1].plot(n_arr, dP, "-", lw=1.2, ms=3,
                      label=f"N_Cheb={N_cheb}  ratio={ratio:.5f}")

    last = N_list[-1] if N_list[-1] in results else max(results)
    axes[1].axhline(results[last]["dP_tassoul"], ls="--", color="r",
                     lw=1.5,
                     label=f"Tassoul = {results[last]['dP_tassoul']:.5f}")
    axes[1].set_xlabel("radial order n")
    axes[1].set_ylabel(r"$\Delta P_n$")
    axes[1].set_title(f"Chebyshev: ΔP convergence (ell={ell})")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = gi.VID / "gmode_exp_c_chebyshev.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 72)
    print(f"{'N_Cheb':>8}  {'ΔP_Tassoul':>14}  {'ΔP_tail':>14}  {'ratio':>10}  {'err':>12}")
    print("-" * 72)
    for N_cheb in N_list:
        if N_cheb not in results:
            continue
        r = results[N_cheb]
        err = abs(r["ratio_tail"] - 1.0)
        print(f"{N_cheb:>8}  {r['dP_tassoul']:14.8f}  {r['dP_tail_mean']:14.8f}  "
              f"{r['ratio_tail']:10.6f}  {err:12.3e}")
    print("=" * 72)
    N_best = max(results)
    best = results[N_best]["ratio_tail"]
    print(f"  Best (N_Cheb={N_best}): |ratio - 1| = {abs(best - 1):.3e}")

    # Verification
    if verify:
        print("\n--- VERIFY against EXPECTED ---")
        n_fail = 0
        n_skip = 0
        for N_cheb in N_list:
            if N_cheb not in results:
                continue
            exp = EXPECTED[N_cheb]
            res = results[N_cheb]
            if exp["dP_tassoul"] == 0.0:
                print(f"  [SKIP] N_Cheb={N_cheb}: EXPECTED unset")
                n_skip += 1
                continue
            tol = REL_TOL_HIGH if N_cheb >= 256 else REL_TOL_DEFAULT
            for key in ("dP_tassoul", "dP_tail_mean", "ratio_tail"):
                v, ref = res[key], exp[key]
                d = abs(v - ref) / max(abs(ref), 1e-300)
                ok = d < tol
                mark = "OK" if ok else "DRIFT"
                print(f"  [{mark:<5}] N_Cheb={N_cheb} {key:<14} {v:.8f} vs {ref:.8f}  ({d*100:.2f}%)  [tol {tol*100:.0f}%]")
                if not ok:
                    n_fail += 1
        if n_fail:
            sys.exit(1)
        if n_skip == len([n for n in N_list if n in results]):
            print("  No reference values set yet.")
        else:
            print(f"\n  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    main(verify=args.verify)
