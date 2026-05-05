#!/usr/bin/env python3
"""Review Round-2 item 2.2: ρ_cut sensitivity sweep.

The surface-truncation parameter ρ_cut (§2.3) replaces the vanishing
Lane-Emden tail by a flat floor, making the SL eigenproblem well-posed
on the interior.  Reviewer asks: how sensitive are the physical
observables (g-mode frequencies, eigenvector shapes) to the choice?

Sweep: ρ_cut ∈ {0.01, 0.02, 0.05, 0.10} on Lane-Emden n=3/2,
       record ω² for the first 5 g-modes, ℓ = 1, N_y = 96.
       Also report ΔV/V_ref (L² deviation of eigenvector vs ρ_cut=0.05
       reference) for each (mode, ρ_cut) combination.

Output: review/r22_rhocut/{sweep.csv, eigenvector_deviation.csv}
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from full_galerkin_closure_test import (
    cgl_grid, cc_weights, bg_lane_emden, assemble_operator, evp,
)


def bg_lane_emden_fixed_domain(y, Ly, rho_cut=0.01):
    """Alternative Lane-Emden background in which the *physical domain*
    [ξ_lo, ξ_hi] is held fixed (= [0, ξ_1] = full polytrope interior),
    and the cutoff acts as a floor: ρ(y) = max(Lane-Emden(y), ρ_cut).
    This isolates the *regularisation* effect of ρ_cut from the
    implicit domain-rescaling that happens in `bg_lane_emden`.
    """
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda xi, st: [st[1],
                      -max(st[0], 0.0) ** 1.5 - 2 * st[1] / max(xi, 1e-8)],
                    [1e-6, 4.0], [1.0, 0.0], dense_output=True,
                    events=lambda t, st: st[0], max_step=1e-3)
    xi_s = sol.t_events[0][0]
    xi_lo, xi_hi = 1e-3, xi_s * 0.999
    xi = np.linspace(xi_lo, xi_hi, 4000)
    rho_full = np.clip(sol.sol(xi)[0], 0, None) ** 1.5
    xi_q = xi_lo + (y / Ly) * (xi_hi - xi_lo)
    rho_true = np.interp(xi_q, xi, rho_full)
    rho = np.maximum(rho_true, rho_cut)
    drho = np.gradient(rho, y, edge_order=2)
    N2 = np.maximum(-drho / rho, 0.0)
    return rho, N2

OUT_DIR = SCRIPT_DIR.parent / "review" / "r22_rhocut"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_rhocut(rho_cut, N_y=96, Ly=1.0, kx=2 * np.pi, n_modes=5,
               mode="rescaled_domain"):
    y, D = cgl_grid(N_y, Ly)
    w_cc = cc_weights(N_y, Ly)
    if mode == "rescaled_domain":
        rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    else:  # fixed_domain
        rho, N2 = bg_lane_emden_fixed_domain(y, Ly, rho_cut=rho_cut)
    L, R = assemble_operator("vspace", y, D, rho, N2, kx)
    lam, V = evp(L, R)
    n_got = min(n_modes, len(lam))
    omega2 = lam[:n_got]
    omega = np.sqrt(omega2)
    return dict(
        y=y, rho=rho, N2=N2, w_cc=w_cc,
        omega2=omega2, omega=omega, V=V[:, :n_got],
    )


def interp_eigenvector(V_src, y_src, y_tgt):
    """Interpolate V (interior-only) onto a common grid y_tgt for
    cross-ρ_cut comparison.  Zero-pad at walls."""
    N_src = len(y_src)
    V_full_src = np.zeros((N_src, V_src.shape[1]))
    V_full_src[1:-1, :] = V_src
    V_tgt = np.zeros((len(y_tgt), V_src.shape[1]))
    for k in range(V_src.shape[1]):
        V_tgt[:, k] = np.interp(y_tgt, y_src, V_full_src[:, k])
    return V_tgt


def main_mode(mode, rho_cut_list, N_y):
    """Run the sweep for one of the two domain conventions."""
    print(f"\n{'='*72}")
    print(f"  Mode: {mode}")
    print(f"{'='*72}")
    results = {}
    for rc in rho_cut_list:
        r = run_rhocut(rc, N_y=N_y, mode=mode)
        # Normalise each eigenvector to unit CC-weighted L²
        V_full = np.zeros((N_y, r["V"].shape[1]))
        V_full[1:-1, :] = r["V"]
        for k in range(V_full.shape[1]):
            norm2 = np.sum(r["w_cc"] * V_full[:, k] ** 2)
            V_full[:, k] /= np.sqrt(max(norm2, 1e-300))
        r["V_full_norm"] = V_full
        results[rc] = r
        print(f"  ρ_cut={rc:.2f}  ω²[1..5] = {r['omega2']}")

    # Eigenvector deviations (reference = ρ_cut = 0.05)
    ref = results[0.05]
    y_ref = ref["y"]
    w_cc_ref = ref["w_cc"]

    print()
    print(f"  Eigenvector deviation vs ρ_cut=0.05 reference")
    print(f"  ρ_cut   mode      ΔV L²     Δω/ω_ref")
    sweep_rows = []
    dev_rows = []
    for rc in rho_cut_list:
        r = results[rc]
        V_on_ref = interp_eigenvector(r["V"], r["y"], y_ref)
        # Re-normalise after interpolation
        for k in range(V_on_ref.shape[1]):
            n2 = np.sum(w_cc_ref * V_on_ref[:, k] ** 2)
            V_on_ref[:, k] /= np.sqrt(max(n2, 1e-300))

        for k in range(5):
            # Sign-correction: match sign via inner product with ref
            s = np.sign(np.sum(w_cc_ref * V_on_ref[:, k] * ref["V_full_norm"][:, k]))
            diff = V_on_ref[:, k] * s - ref["V_full_norm"][:, k]
            dev = np.sqrt(np.sum(w_cc_ref * diff ** 2))
            omega_shift = (r["omega"][k] - ref["omega"][k]) / ref["omega"][k]
            print(f"  {rc:.2f}    n_g={k+1}    {dev:.3e}    {omega_shift:+.3e}")
            dev_rows.append(dict(
                rho_cut=rc, mode=k + 1,
                omega2=float(r["omega2"][k]),
                omega=float(r["omega"][k]),
                dev_L2=float(dev),
                omega_shift=float(omega_shift),
            ))

        for k in range(5):
            sweep_rows.append(dict(
                rho_cut=rc, mode=k + 1,
                omega2=float(r["omega2"][k]),
                omega=float(r["omega"][k]),
            ))

    return results, sweep_rows, dev_rows


def main():
    rho_cut_list = [0.01, 0.02, 0.05, 0.10]
    N_y = 96

    all_sweep = []
    all_dev = []
    all_prof = []
    for mode in ("rescaled_domain", "fixed_domain"):
        results, sweep_rows, dev_rows = main_mode(mode, rho_cut_list, N_y)
        for r in sweep_rows:
            r["mode"] = mode
        for r in dev_rows:
            r["mode"] = mode
        all_sweep.extend(sweep_rows)
        all_dev.extend(dev_rows)
        for rc in rho_cut_list:
            r = results[rc]
            for j in range(len(r["y"])):
                all_prof.append(dict(
                    mode=mode, rho_cut=rc, y=float(r["y"][j]),
                    rho=float(r["rho"][j]), N2=float(r["N2"][j]),
                ))

    with open(OUT_DIR / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_sweep[0].keys()))
        w.writeheader()
        for row in all_sweep:
            w.writerow(row)

    with open(OUT_DIR / "eigenvector_deviation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_dev[0].keys()))
        w.writeheader()
        for row in all_dev:
            w.writerow(row)

    with open(OUT_DIR / "profiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_prof[0].keys()))
        w.writeheader()
        for row in all_prof:
            w.writerow(row)

    print()
    print(f"Wrote CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
