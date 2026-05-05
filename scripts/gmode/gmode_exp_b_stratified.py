#!/usr/bin/env python3
"""
Experiment B: real g-modes on an artificial stratified cavity.

PURPOSE
  Validate the full g-mode pipeline on a physically sensible (positive
  Brunt-Väisälä frequency) radial profile so that the Tassoul (1980)
  asymptotic period spacing is the genuine limit:

      lim_{n→∞} ΔP_n  =  2π² / ( sqrt(ell*(ell+1)) · ∫ N/r dr )

  We use a smooth Gaussian-bump N²(r) on r ∈ [r_lo, r_hi].  The Cowling
  generalised eigenproblem

      -ψ''(r) = ω^{-2} · ell(ell+1) · N²(r)/r² · ψ(r)

  is solved at 4 grid resolutions, the last-5-modes-mean ΔP is compared to
  the Tassoul integral, and the ratio should approach 1.0 as N_r grows.

  This is the first clean demonstration in this repo that the g-mode
  infrastructure (scripts/gmode_infra.py) reproduces a textbook result.
  Lane-Emden (Experiment A) is isentropic and cannot host real g-modes;
  the stratified layer here can.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §3

REPRO
  python scripts/gmode_exp_b_stratified.py
  python scripts/gmode_exp_b_stratified.py --verify
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
SCRIPT_REL = "scripts/gmode_exp_b_stratified.py"


def n2_profile(r, r_lo, r_hi):
    """Smooth Gaussian-bump N²(r) peaked at (r_lo + r_hi)/2.

    Vanishes at r_lo and r_hi so the g-mode cavity is well-defined.
    Normalised so max(N²) = 1 -> ΔP_tassoul ~ 2π² / (log ratio) is O(1).
    """
    rc = 0.5 * (r_lo + r_hi)
    sigma = 0.25 * (r_hi - r_lo)
    bump = np.exp(-((r - rc) / sigma) ** 2)
    # Smoothly drop to zero at the edges (taper so the Dirichlet BC is consistent).
    taper = np.sin(np.pi * (r - r_lo) / (r_hi - r_lo)) ** 2
    return bump * taper


EXPECTED = {
    256:  {"dP_tassoul": 20.993338, "dP_tail_mean": 19.523643, "ratio_tail": 0.9300},
    512:  {"dP_tassoul": 20.993276, "dP_tail_mean": 20.638170, "ratio_tail": 0.9831},
    1024: {"dP_tassoul": 20.993261, "dP_tail_mean": 20.910023, "ratio_tail": 0.9960},
    2048: {"dP_tassoul": 20.993257, "dP_tail_mean": 20.977525, "ratio_tail": 0.9993},
}
REL_TOL = 0.02


def main(verify=False):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment B: stratified layer g-modes, Tassoul convergence")
    print("=" * 72)

    r_lo, r_hi = 0.2, 1.0
    ell = 1
    n_modes = 40
    N_list = [256, 512, 1024, 2048]

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)

    # Plot N²(r) once (identical across resolutions up to grid)
    r_show = np.linspace(r_lo, r_hi, 2048)
    axes[0].plot(r_show, n2_profile(r_show, r_lo, r_hi),
                  color="C2", lw=2)
    axes[0].set_xlabel("r")
    axes[0].set_ylabel(r"$N^2(r)$")
    axes[0].set_title("Gaussian-bump stratification (artificial)")
    axes[0].grid(alpha=0.3)

    print(f"  ell = {ell},  r in [{r_lo}, {r_hi}]")
    print(f"  Resolution sweep: {N_list}")

    for Nr in N_list:
        r = np.linspace(r_lo, r_hi, Nr)
        N2 = n2_profile(r, r_lo, r_hi)

        omsq, psi = gi.solve_gmode_cowling(r, N2, ell=ell, n_modes=n_modes)
        omega = np.sqrt(omsq)
        P = 2.0 * np.pi / omega
        dP = np.abs(np.diff(P))

        dP_tassoul = gi.tassoul_dP(r, N2, ell=ell)
        dP_tail = dP[-5:].mean()
        ratio = dP_tail / dP_tassoul

        print(f"\n  Nr = {Nr}:")
        print(f"    Tassoul ΔP (l={ell}) = {dP_tassoul:.6f}")
        print(f"    ΔP first 5:  {['%.4f' % v for v in dP[:5]]}")
        print(f"    ΔP tail (last 5) = {dP_tail:.6f},  ratio = {ratio:.4f}")

        results[Nr] = {
            "dP_tassoul": dP_tassoul,
            "dP_tail_mean": dP_tail,
            "ratio_tail": ratio,
            "dP_all": dP,
        }

        n_arr = np.arange(1, len(dP) + 1)
        axes[1].plot(n_arr, dP, "-", lw=1.2, ms=3,
                      label=f"Nr={Nr}  ratio={ratio:.3f}")

    axes[1].axhline(results[N_list[-1]]["dP_tassoul"], ls="--", color="r",
                     lw=1.5, label=f"Tassoul = {results[N_list[-1]]['dP_tassoul']:.3f}")
    axes[1].set_xlabel("radial order n")
    axes[1].set_ylabel(r"$\Delta P_n$")
    axes[1].set_title(f"ΔP convergence to Tassoul (ell={ell})")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = gi.VID / "gmode_exp_b_stratified.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    # Summary convergence table
    print("\n" + "=" * 72)
    print(f"{'Nr':>6}  {'ΔP_Tassoul':>12}  {'ΔP_tail':>12}  {'ratio':>8}  {'err':>10}")
    print("-" * 72)
    for Nr in N_list:
        r = results[Nr]
        err = abs(r["ratio_tail"] - 1.0)
        print(f"{Nr:>6}  {r['dP_tassoul']:12.6f}  {r['dP_tail_mean']:12.6f}  "
              f"{r['ratio_tail']:8.4f}  {err:10.2e}")
    print("=" * 72)
    final_ratio = results[N_list[-1]]["ratio_tail"]
    print(f"  PASS threshold: |ratio - 1| < 0.05 at Nr = {N_list[-1]}")
    print(f"  Result: |{final_ratio:.4f} - 1| = {abs(final_ratio - 1):.4f}  "
          f"{'PASS' if abs(final_ratio - 1) < 0.05 else 'FAIL'}")

    # Verification
    if verify:
        print("\n--- VERIFY against EXPECTED ---")
        n_fail = 0
        n_skip = 0
        for Nr in N_list:
            exp = EXPECTED[Nr]
            res = results[Nr]
            if exp["dP_tassoul"] == 0.0:
                print(f"  [SKIP] Nr={Nr}: EXPECTED unset")
                n_skip += 1
                continue
            for key in ("dP_tassoul", "dP_tail_mean", "ratio_tail"):
                v, ref = res[key], exp[key]
                d = abs(v - ref) / max(abs(ref), 1e-300)
                ok = d < REL_TOL
                mark = "OK" if ok else "DRIFT"
                print(f"  [{mark:<5}] Nr={Nr}  {key:<14} {v:.6f} vs {ref:.6f}  ({d*100:.2f}%)")
                if not ok:
                    n_fail += 1
        if n_fail:
            sys.exit(1)
        if n_skip == len(N_list):
            print("  No reference values set yet.")
        else:
            print(f"\n  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    main(verify=args.verify)
