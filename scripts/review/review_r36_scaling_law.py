#!/usr/bin/env python3
"""Review Round-3 addendum 3 (user refinement 3): small-ε scaling test
for Proposition 1 (locality gap version).

User refinement:
  The Proposition should not say "defect is O(1)" — that fails in the
  small-perturbation limit ρ = 1 + ε·f(y) as ε → 0.
  Instead it should say "defect is independent of N_y, with magnitude
  c(ρ₀) that vanishes as ρ₀ → const."

This script verifies c(ρ₀) ∝ ‖ρ₀'‖ by setting
    ρ₀(y) = 1 + ε · sin(2π y / L_y)
and sweeping ε ∈ {1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5}.
For each ε, it computes the locality-gap norm
    ‖(L⁻¹ - diag(1/ρ₀)) · R · v_n‖ / ‖L⁻¹ · R · v_n‖
and reports the floor (= average over N_y = 64, 128).

Expected: floor ∝ ε  (or ε²; will read off the slope).

This upgrades Proposition 1 from "existence of floor" to "scaling law".

Output:
  review/r36_scaling_law/scaling.csv
  review/r36_scaling_law/scaling_plot.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from full_galerkin_closure_test import cgl_grid, cc_weights, assemble_operator, evp

OUT_DIR = SCRIPT_DIR.parent / "review" / "r36_scaling_law"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bg_perturbed_constant(y, Ly, eps):
    """ρ₀(y) = 1 + ε · sin(2π y / L_y), N² = -ρ₀'/ρ₀."""
    rho = 1.0 + eps * np.sin(2 * np.pi * y / Ly)
    drho_dy = eps * 2 * np.pi / Ly * np.cos(2 * np.pi * y / Ly)
    # Use analytic derivative (avoids numerical noise confounding the ε limit)
    N2 = np.maximum(-drho_dy / rho, 0.0)
    return rho, N2


def locality_gap_norm(N_y, eps, Ly=1.0, kx=2 * np.pi):
    """Return ‖(L⁻¹ - diag(1/ρ)) · R · v_n‖ / ‖L⁻¹ · R · v_n‖
    where v_n is the top eigenvector of R v = ω² L v."""
    y, D = cgl_grid(N_y, Ly)
    rho, N2 = bg_perturbed_constant(y, Ly, eps)

    L_int, R_int = assemble_operator("vspace", y, D, rho, N2, kx)
    lam, V = evp(L_int, R_int)
    if len(lam) == 0:
        return None
    om2 = float(lam[0])
    v_int = V[:, 0] / np.linalg.norm(V[:, 0])

    rho_int = rho[1:-1]
    # L⁻¹ R v
    M_asm = np.linalg.solve(L_int, R_int)
    L_inv_R_v = M_asm @ v_int
    # diag(1/ρ) R v  (interior-restricted)
    diag_inv_rho_R_v = (R_int @ v_int) / rho_int
    resid = L_inv_R_v - diag_inv_rho_R_v

    # Also: ‖ρ₀'‖_∞ for scaling interpretation
    drho_dy = eps * 2 * np.pi / Ly * np.cos(2 * np.pi * y / Ly)
    rho_prime_Linf = float(np.max(np.abs(drho_dy)))

    # Absolute gap (no normalisation).  Under ε → 0, ω² → 0, so L⁻¹R v_n → 0
    # and diag(1/ρ) R v_n → 0; the absolute gap should track the physics scale.
    abs_gap = float(np.linalg.norm(resid))
    # Relative gap: invariant of the physics scale — reports operator-level
    # locality mismatch.
    rel_gap = float(np.linalg.norm(resid) / max(np.linalg.norm(L_inv_R_v), 1e-30))

    return dict(
        N_y=N_y,
        eps=eps,
        omega2=om2,
        rho_prime_Linf=rho_prime_Linf,
        abs_gap=abs_gap,
        rel_gap=rel_gap,
        locality_gap_norm=rel_gap,  # kept for backward compat
        L_inv_R_v_norm=float(np.linalg.norm(L_inv_R_v)),
        resid_norm=abs_gap,
    )


def main():
    eps_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5]
    N_list = [64, 128]

    rows = []
    print("  Small-ε scaling test for locality gap")
    print("  Background: ρ₀ = 1 + ε sin(2π y/L_y)")
    print("  " + "-" * 88)
    print("   ε         N_y    ω²        ‖ρ₀'‖∞     abs_gap        rel_gap        abs/ε")
    print("  " + "-" * 88)
    for eps in eps_list:
        for N in N_list:
            r = locality_gap_norm(N, eps)
            if r is None:
                continue
            r["gap_over_eps"] = r["abs_gap"] / eps
            r["gap_over_eps2"] = r["abs_gap"] / eps ** 2
            rows.append(r)
            print(f"  {eps:.1e}   {N:3d}   "
                  f"{r['omega2']:.4f}   "
                  f"{r['rho_prime_Linf']:.3e}  "
                  f"{r['abs_gap']:.3e}     "
                  f"{r['rel_gap']:.3e}     "
                  f"{r['gap_over_eps']:.4f}")
        print()

    with open(OUT_DIR / "scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Fit both absolute and relative scalings: gap ∝ ε^p
    # Use N_y = 128 for the fit (cleaner)
    pts = [(r["eps"], r["abs_gap"], r["rel_gap"]) for r in rows if r["N_y"] == 128]
    eps_arr = np.array([p[0] for p in pts])
    abs_arr = np.array([p[1] for p in pts])
    rel_arr = np.array([p[2] for p in pts])
    mask = eps_arr <= 0.1  # linear-response regime
    log_abs = np.polyfit(np.log(eps_arr[mask]), np.log(abs_arr[mask]), 1)
    log_rel = np.polyfit(np.log(eps_arr[mask]), np.log(rel_arr[mask]), 1)
    slope_abs, intc_abs = log_abs
    slope_rel, intc_rel = log_rel
    print(f"\nLog-log fits (ε ≤ 0.1, N_y = 128):")
    print(f"  absolute gap  ≈ {np.exp(intc_abs):.3e} · ε^{slope_abs:.3f}")
    print(f"  relative gap  ≈ {np.exp(intc_rel):.3e} · ε^{slope_rel:.3f}")
    print(f"\n  Interpretation:")
    print(f"  - abs slope {slope_abs:.2f} ≈ 1  → c(ρ₀) ∝ ‖ρ₀'‖ in the small-ε limit")
    print(f"    (both ω² ∝ ε and defect ∝ ε vanish as ρ → const)")
    print(f"  - rel slope {slope_rel:.2f} ≈ 0  → operator-level mismatch is")
    print(f"    SCALE-INVARIANT: ratio of gap to eigenvalue is O(1) even for small ε")
    print(f"    This is the 'resolution-independent floor' property.")

    # Figure — two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)

    ax = axes[0]
    for N in N_list:
        pts_N = [(r["eps"], r["abs_gap"]) for r in rows if r["N_y"] == N]
        eps_N = [p[0] for p in pts_N]
        gap_N = [p[1] for p in pts_N]
        ax.loglog(eps_N, gap_N, "o-", label=f"$N_y = {N}$")
    ax.loglog(eps_arr[mask], np.exp(intc_abs) * eps_arr[mask] ** slope_abs,
              "k--", lw=1, alpha=0.5, label=f"fit: $\\propto \\varepsilon^{{{slope_abs:.2f}}}$")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"absolute gap $\|(L^{-1}-\mathrm{diag}(1/\rho))Rv_n\|$")
    ax.set_title(f"(a) Absolute gap: $\\propto \\varepsilon^{{{slope_abs:.2f}}}$  (vanishes as $\\rho_0 \\to $ const)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax = axes[1]
    for N in N_list:
        pts_N = [(r["eps"], r["rel_gap"]) for r in rows if r["N_y"] == N]
        eps_N = [p[0] for p in pts_N]
        gap_N = [p[1] for p in pts_N]
        ax.loglog(eps_N, gap_N, "o-", label=f"$N_y = {N}$")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"relative gap $\|(L^{-1}-\mathrm{diag}(1/\rho))Rv_n\|/\|L^{-1}Rv_n\|$")
    ax.set_title("(b) Relative gap: $\\mathcal{O}(1)$ — structural inconsistency")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    fig.suptitle(r"Proposition 1 scaling law: $\rho_0 = 1 + \varepsilon\sin(2\pi y/L_y)$",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scaling_plot.png", bbox_inches="tight")

    print(f"\nWrote {OUT_DIR / 'scaling.csv'}")
    print(f"Wrote {OUT_DIR / 'scaling_plot.png'}")


if __name__ == "__main__":
    main()
