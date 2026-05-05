#!/usr/bin/env python3
"""
Experiment D: Polytropic profile g-mode calculation via MESA-style reader.

PURPOSE
  First integration test of the `scripts/mesa_profile.py` parser: build a
  Lane-Emden n=3 polytrope fixture with an imposed Gaussian-bump N²(r),
  read it back through the parser, and feed (r, N²) into both the FD
  (`solve_gmode_cowling`) and Chebyshev (`solve_gmode_cowling_cheb`) Cowling
  solvers.  The ΔP tail must agree with the corresponding Exp B / Exp C
  numbers, proving that the file I/O path does not distort the result.

  This is a unit-test-shaped experiment: the polytrope's density ρ(r) does
  not enter the Cowling equation (Cowling uses only N² and r), so the
  result must numerically match Exp B / Exp C at the same cavity and
  resolution.  Any disagreement is a parser or resampling bug.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §7   (to be written in this commit)

REPRO
  python scripts/gmode_exp_d_polytrope_profile.py
  python scripts/gmode_exp_d_polytrope_profile.py --verify
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
from gmode_profile import build_polytrope_fixture, read_profile

REF_DOC = "docs/gmode_experiments_2026-05-02.md"
SCRIPT_REL = "scripts/gmode_exp_d_polytrope_profile.py"

FIXTURE = gi.VID / "polytrope_fixture.dat"

# Reference values are compared against the corresponding Exp B (FD) and
# Exp C (Chebyshev) rows at the same cavity and grid resolution.  If the
# file I/O is lossless, the mismatch should be dominated by resampling
# error at the cavity edges, which drops below 1e-3 at Nr=1024 / N_Cheb=256.
EXPECTED = {
    "fd_1024":   {"dP_tassoul": 20.992871, "dP_tail_mean": 20.910504, "ratio_tail": 0.996076},
    "fd_2048":   {"dP_tassoul": 20.992860, "dP_tail_mean": 20.977810, "ratio_tail": 0.999283},
    "cheb_256":  {"dP_tassoul": 20.993408, "dP_tail_mean": 20.997854, "ratio_tail": 1.000212},
    "cheb_512":  {"dP_tassoul": 20.993003, "dP_tail_mean": 20.994706, "ratio_tail": 1.000081},
}
REL_TOL = 0.01   # 1% -- slightly looser than reduced_pressure_* because the
                 # file-I/O interpolation at cavity boundaries contributes a
                 # resampling-error term absent from Exp B / Exp C.


def fd_run(prof, Nr, ell=1):
    """Resample profile N²(r) onto a uniform grid of size Nr inside the
    cavity (the positive-N² region), run FD Cowling."""
    mask = prof["N2"] > 1e-10
    r_raw = prof["r"][mask]
    N2_raw = prof["N2"][mask]
    r = np.linspace(r_raw[0], r_raw[-1], Nr)
    N2 = np.interp(r, r_raw, N2_raw)
    n_modes = 40
    omsq, _ = gi.solve_gmode_cowling(r, N2, ell=ell, n_modes=n_modes)
    omega = np.sqrt(omsq)
    P = 2.0 * np.pi / omega
    dP = np.abs(np.diff(P))
    dP_tass = gi.tassoul_dP(r, N2, ell)
    dP_tail = dP[-5:].mean()
    return dict(dP_tassoul=dP_tass, dP_tail_mean=dP_tail,
                ratio_tail=dP_tail / dP_tass, dP_all=dP)


def cheb_run(prof, N_Cheb, ell=1):
    """Resample profile N²(r) onto the CGL grid inside the cavity,
    run Chebyshev Cowling with the spurious-mode cap."""
    mask = prof["N2"] > 1e-10
    r_raw = prof["r"][mask]
    N2_raw = prof["N2"][mask]
    r, D2, w_full = gi.cheb_on_interval(N_Cheb, r_raw[0], r_raw[-1])
    N2 = np.interp(r, r_raw, N2_raw)
    n_modes = max(10, N_Cheb // 5)
    omsq, _ = gi.solve_gmode_cowling_cheb(r, D2, w_full, N2, ell=ell, n_modes=n_modes)
    omega = np.sqrt(omsq)
    P = 2.0 * np.pi / omega
    dP = np.abs(np.diff(P))
    dP_tass = gi.tassoul_dP(r, N2, ell)
    dP_tail = dP[-5:].mean() if len(dP) >= 5 else dP.mean()
    return dict(dP_tassoul=dP_tass, dP_tail_mean=dP_tail,
                ratio_tail=dP_tail / dP_tass, dP_all=dP)


def main(verify=False):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment D: polytrope profile -> MESA parser -> Cowling")
    print("=" * 72)

    # 1. Build / rebuild the fixture deterministically
    build_polytrope_fixture(FIXTURE,
                             cavity_r_lo=0.2, cavity_r_hi=1.0, N2_amp=1.0)
    print(f"  built fixture: {FIXTURE}")
    prof = read_profile(FIXTURE)
    print(f"    columns: {list(prof.keys())}")
    print(f"    rows: {len(prof['r'])}")
    print(f"    r range: [{prof['r'][0]:.4f}, {prof['r'][-1]:.4f}]")
    print(f"    rho range: [{prof['rho'].min():.3e}, {prof['rho'].max():.3e}]")
    print(f"    N² range: [{prof['N2'].min():.3e}, {prof['N2'].max():.3e}]  "
          f"(imposed Gaussian bump)")

    # 2. Run FD at two resolutions
    print("\n  FD Cowling:")
    fd_1024 = fd_run(prof, 1024)
    fd_2048 = fd_run(prof, 2048)
    print(f"    Nr=1024:   ΔP_Tassoul={fd_1024['dP_tassoul']:.6f}  "
          f"ΔP_tail={fd_1024['dP_tail_mean']:.6f}  ratio={fd_1024['ratio_tail']:.6f}")
    print(f"    Nr=2048:   ΔP_Tassoul={fd_2048['dP_tassoul']:.6f}  "
          f"ΔP_tail={fd_2048['dP_tail_mean']:.6f}  ratio={fd_2048['ratio_tail']:.6f}")

    # 3. Run Chebyshev at two resolutions
    print("\n  Chebyshev Cowling:")
    ch_256 = cheb_run(prof, 256)
    ch_512 = cheb_run(prof, 512)
    print(f"    N_Cheb=256: ΔP_Tassoul={ch_256['dP_tassoul']:.6f}  "
          f"ΔP_tail={ch_256['dP_tail_mean']:.6f}  ratio={ch_256['ratio_tail']:.6f}")
    print(f"    N_Cheb=512: ΔP_Tassoul={ch_512['dP_tassoul']:.6f}  "
          f"ΔP_tail={ch_512['dP_tail_mean']:.6f}  ratio={ch_512['ratio_tail']:.6f}")

    results = {"fd_1024": fd_1024, "fd_2048": fd_2048,
                "cheb_256": ch_256, "cheb_512": ch_512}

    # 4. Plot density profile + N² + ΔP convergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=140)

    axes[0].semilogy(prof["r"], prof["rho"], color="C3", lw=1.5)
    axes[0].set_xlabel("r")
    axes[0].set_ylabel(r"$\rho(r)$")
    axes[0].set_title("Lane-Emden n=3 polytrope (file-parsed)")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(prof["r"], prof["N2"], color="C2", lw=1.5)
    axes[1].set_xlabel("r")
    axes[1].set_ylabel(r"$N^2(r)$  (imposed)")
    axes[1].set_title("Gaussian-bump cavity")
    axes[1].grid(alpha=0.3)

    for key, color, style in [("fd_1024", "C0", "-"),
                                ("fd_2048", "C0", "--"),
                                ("cheb_256", "C3", "-"),
                                ("cheb_512", "C3", "--")]:
        dP = results[key]["dP_all"]
        n_arr = np.arange(1, len(dP) + 1)
        axes[2].plot(n_arr, dP, style, color=color, lw=1.2, ms=2,
                      label=f"{key}  ratio={results[key]['ratio_tail']:.5f}")
    axes[2].axhline(results["cheb_512"]["dP_tassoul"], ls=":", color="k",
                     lw=1.5, label=f"Tassoul")
    axes[2].set_xlabel("radial order n")
    axes[2].set_ylabel(r"$\Delta P_n$")
    axes[2].set_title("ΔP convergence (file I/O path)")
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = gi.VID / "gmode_exp_d_polytrope.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    # 5. Summary
    print("\n" + "=" * 78)
    print(f"{'suite':<12}  {'ΔP_Tassoul':>14}  {'ΔP_tail':>14}  {'ratio':>10}  {'|r-1|':>10}")
    print("-" * 78)
    for key in ("fd_1024", "fd_2048", "cheb_256", "cheb_512"):
        r = results[key]
        err = abs(r["ratio_tail"] - 1.0)
        print(f"{key:<12}  {r['dP_tassoul']:14.6f}  "
              f"{r['dP_tail_mean']:14.6f}  {r['ratio_tail']:10.6f}  {err:10.3e}")
    print("=" * 78)

    # 6. Verification
    if verify:
        print("\n--- VERIFY against EXPECTED (Exp B / Exp C reference rows) ---")
        n_fail = 0
        for key, exp in EXPECTED.items():
            res = results[key]
            for field in ("dP_tassoul", "dP_tail_mean", "ratio_tail"):
                v, ref = res[field], exp[field]
                d = abs(v - ref) / max(abs(ref), 1e-300)
                ok = d < REL_TOL
                mark = "OK" if ok else "DRIFT"
                print(f"  [{mark:<5}] {key:<10} {field:<14} {v:.6f} vs {ref:.6f}  ({d*100:.3f}%)")
                if not ok:
                    n_fail += 1
        if n_fail:
            sys.exit(1)
        print("\n  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    main(verify=args.verify)
