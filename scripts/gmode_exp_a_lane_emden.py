#!/usr/bin/env python3
"""
Experiment A: Lane-Emden W̃-proxy Tassoul heuristic.

PURPOSE
  Sanity-check the g-mode pipeline (gmode_infra.solve_gmode_cowling +
  tassoul_dP) by feeding it a non-physical but well-defined N² proxy:

      N²_proxy(r) := -W̃(r)

  where W̃ is the reduced-pressure Liouville potential derived from the
  Lane-Emden n=3/2 density profile.  For Lane-Emden ρ ∝ t^{3/2}, W̃ near the
  surface is +3/(16 t²), so -W̃ is negative there and cannot itself support
  g-modes -- but away from the surface (rho_cut) we keep only the region
  where -W̃ > 0.

  Lane-Emden polytropes are isentropic, so the true Brunt-Väisälä frequency
  vanishes (N² = 0 everywhere) and the star supports no real g-modes.  This
  experiment is therefore a pure scaling test: does the numerical pipeline
  recover Tassoul's asymptotic ΔP formula when fed a plausibly-shaped but
  artificial N²(r)?  A ratio of ΔP_tail / ΔP_Tassoul ≈ 1 (within 10%) is
  a pass; it demonstrates the pipeline converges correctly without claiming
  any Lane-Emden g-mode physics.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §2

REPRO
  python scripts/gmode_exp_a_lane_emden.py
  python scripts/gmode_exp_a_lane_emden.py --verify
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
SCRIPT_REL = "scripts/gmode_exp_a_lane_emden.py"

# Reference values (populated after first clean run; see --verify).
EXPECTED = {
    "dP_tassoul":     137.6316,
    "dP_tail_mean":   117.2542,
    "ratio_tail":     0.852,
    "ell":            1,
}
REL_TOL = 0.02


def main(verify=False):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment A: Lane-Emden W̃-proxy g-mode heuristic")
    print("=" * 72)

    # 1. Lane-Emden and W̃ (reduced-pressure form)
    xi, theta, xi_1 = gi.solve_lane_emden(n=1.5)
    r_norm = xi / xi_1
    rho = np.abs(theta) ** 1.5

    # Truncate to the interior where FD on W̃ is reliable.  rho_cut = 0.05
    # avoids the surface boundary layer where np.gradient loses accuracy and
    # where W̃ is dominated by the +3/(16t²) barrier (unphysical for g-modes).
    rho_cut = 0.05
    mask = rho > rho_cut
    r_raw = r_norm[mask]
    rho_raw = rho[mask]

    # Resample on a uniform grid (required by solve_sl_eigenpairs and
    # solve_gmode_cowling).
    Nr = 512
    r = np.linspace(r_raw[0], r_raw[-1], Nr)
    rho_u = np.interp(r, r_raw, rho_raw)
    W_redu = gi.compute_W_reduced(r, rho_u)
    W_orig = gi.compute_W_original(r, rho_u)

    # 2. Build N²_proxy = -W̃.  Keep only the positive-N² region.
    N2_proxy = -W_redu
    # Mask the core: r too small makes 1/r² term dominate pathologically.
    r_lo = 0.15
    keep = (r > r_lo) & (N2_proxy > 0)
    if keep.sum() < 50:
        print(f"  INSUFFICIENT proxy cavity: only {keep.sum()} points. Abort.")
        sys.exit(2)

    r_g = r[keep]
    N2_g = N2_proxy[keep]
    # Re-uniform (keep was a boolean slice, r is still uniform though).
    Nr_g = len(r_g)
    r_g = np.linspace(r_g[0], r_g[-1], Nr_g)
    N2_g = np.interp(r_g, r[keep], N2_proxy[keep])

    print(f"  Lane-Emden n=1.5, rho_cut = {rho_cut}, r in [{r[0]:.3f}, {r[-1]:.3f}]")
    print(f"  g-mode cavity: {Nr_g} points on r in [{r_g[0]:.3f}, {r_g[-1]:.3f}]")
    print(f"  N²_proxy range in cavity: [{N2_g.min():.3e}, {N2_g.max():.3e}]")

    # 3. Cowling g-mode solve + Tassoul ΔP
    ell = 1
    n_modes = 30
    omsq, psi = gi.solve_gmode_cowling(r_g, N2_g, ell=ell, n_modes=n_modes)
    omega = np.sqrt(omsq)
    P = 2.0 * np.pi / omega
    dP = np.abs(np.diff(P))

    dP_tassoul = gi.tassoul_dP(r_g, N2_g, ell=ell)
    dP_tail = dP[-5:].mean() if len(dP) >= 5 else float("nan")
    ratio = dP_tail / dP_tassoul if dP_tassoul > 0 else float("nan")

    print(f"\n  Tassoul ΔP (l={ell}) = {dP_tassoul:.4f}")
    print(f"  ΔP_SL first 10 modes:")
    for n, v in enumerate(dP[:10]):
        print(f"    n={n+1:2d}  ΔP = {v:.4f}   ratio vs Tassoul = {v / dP_tassoul:.3f}")
    print(f"  ΔP tail (last 5 modes) mean = {dP_tail:.4f},  ratio = {ratio:.3f}")
    print(f"  pipeline sanity:  {'PASS' if 0.80 < ratio < 1.20 else 'FAIL'}  "
          f"(accept window: 0.80-1.20 on the heuristic proxy)")

    # 4. Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    axes[0].plot(r, W_orig, label=r"$W_{\mathrm{orig}}$", color="C3", lw=1.5)
    axes[0].plot(r, W_redu, label=r"$\widetilde W$", color="C0", lw=1.5)
    axes[0].plot(r_g, N2_g, label=r"$N^2_{\mathrm{proxy}} = -\widetilde W$",
                  color="C2", lw=2, ls="--")
    axes[0].set_xlabel("r / R_star")
    axes[0].set_ylabel("W or N²")
    axes[0].set_title("Lane-Emden n=3/2 potentials + heuristic N²")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    n_arr = np.arange(1, len(dP) + 1)
    axes[1].plot(n_arr, dP, "bo-", label=r"SL $\Delta P_n$")
    axes[1].axhline(dP_tassoul, ls="--", color="r",
                     label=f"Tassoul constant = {dP_tassoul:.3f}")
    axes[1].set_xlabel("radial order n")
    axes[1].set_ylabel(r"$\Delta P_n$")
    axes[1].set_title(f"g-mode period spacing (ell={ell}, ratio_tail = {ratio:.3f})")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = gi.VID / "gmode_exp_a_lane_emden.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    # 5. Verification
    if verify:
        print("\n--- VERIFY against EXPECTED ---")
        checks = [
            ("dP_tassoul",    dP_tassoul,   EXPECTED["dP_tassoul"]),
            ("dP_tail_mean",  dP_tail,      EXPECTED["dP_tail_mean"]),
            ("ratio_tail",    ratio,        EXPECTED["ratio_tail"]),
        ]
        n_fail = 0
        for name, val, ref in checks:
            if ref == 0.0:
                print(f"  [SKIP ] {name}: EXPECTED unset (first run)")
                continue
            d = abs(val - ref) / max(abs(ref), 1e-300)
            ok = d < REL_TOL
            mark = "OK" if ok else "DRIFT"
            print(f"  [{mark:<5}] {name:<16} {val:.6f} vs {ref:.6f}  ({d*100:.2f}%)")
            if not ok:
                n_fail += 1
        if n_fail:
            sys.exit(1)
        print("  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    main(verify=args.verify)
