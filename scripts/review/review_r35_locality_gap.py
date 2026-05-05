#!/usr/bin/env python3
"""Review Round-3 addendum 2 (user refinement 2): quantify the
locality gap between L⁻¹ (global, nonlocal elliptic inverse) and
diag(1/ρ) (pointwise local scaling).

User's refined framing:
  "It is not about singularity — it's about locality: a global inverse
   operator is being replaced by a pointwise scaling."

This script measures:
  (a) L⁻¹ matrix structure:
      - off-diagonal energy ratio
      - effective bandwidth (where |L⁻¹_ij| > 0.01 · max)
      - row decay rate (exponential / algebraic)
  (b) diag(1/ρ) matrix structure (trivially diagonal)
  (c) The two matrices' spectral properties (eigenvalue distributions)

Key figure of merit:
    nonlocality(L⁻¹) / locality(diag(1/ρ)) → quantified as
    off_diag_frac := Σ_{i≠j} |L⁻¹_ij|² / Σ_ij |L⁻¹_ij|²

For diag(1/ρ), off_diag_frac = 0 exactly.
For L⁻¹, we expect off_diag_frac ~ 0.5 or higher — confirming L⁻¹ is
genuinely global.

Output:
  review/r35_locality/locality_gap.csv
  review/r35_locality/matrix_heatmap.png (for visualisation)
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

from full_galerkin_closure_test import (
    cgl_grid, bg_lane_emden, assemble_operator,
)

OUT_DIR = SCRIPT_DIR.parent / "review" / "r35_locality"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def off_diag_fraction(M):
    diag = np.diag(np.diag(M))
    off = M - diag
    off_sq = np.sum(off ** 2)
    total_sq = np.sum(M ** 2)
    return off_sq / max(total_sq, 1e-300)


def effective_bandwidth(M, threshold_rel=0.01):
    """Count, for each row, how far off-centre the entries extend above
    threshold · row_max.  Return average effective half-bandwidth."""
    n = M.shape[0]
    bws = []
    for i in range(n):
        row = np.abs(M[i, :])
        mx = np.max(row)
        if mx <= 0:
            continue
        nonzero = np.where(row > threshold_rel * mx)[0]
        # distance from diagonal
        dist = np.abs(nonzero - i)
        bws.append(dist.max() if len(dist) > 0 else 0)
    return float(np.mean(bws)) if bws else 0.0


def row_decay_rate(M, row_idx=None):
    """Fit |M[i, j]| vs |i-j| for j≠i; return (algebraic α, residual)
    where row ~ |i-j|^{-α}."""
    n = M.shape[0]
    if row_idx is None:
        row_idx = n // 2  # middle row for diagnostic
    row = np.abs(M[row_idx, :])
    row[row_idx] = 0.0  # exclude diagonal
    d = np.abs(np.arange(n) - row_idx)
    mask = (d > 0) & (row > 1e-16)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    x = np.log(d[mask])
    y = np.log(row[mask])
    # linear fit
    p = np.polyfit(x, y, 1)
    alpha = -p[0]  # exponent in row ~ d^{-α}
    y_fit = np.polyval(p, x)
    resid = float(np.sqrt(np.mean((y - y_fit) ** 2)))
    return float(alpha), resid


def measure_one(N_y, Ly=1.0, kx=2 * np.pi, rho_cut=0.05):
    y, D = cgl_grid(N_y, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    L_int, R_int = assemble_operator("vspace", y, D, rho, N2, kx)

    # The "inverse operator" L⁻¹
    L_inv = np.linalg.inv(L_int)
    # The "surrogate" diag(1/ρ)_int
    rho_int = rho[1:-1]
    inv_rho = np.diag(1.0 / rho_int)

    # Off-diagonal fraction (main locality metric)
    off_L_inv = off_diag_fraction(L_inv)
    off_inv_rho = off_diag_fraction(inv_rho)

    # Effective bandwidth
    bw_L_inv = effective_bandwidth(L_inv, threshold_rel=0.01)
    bw_inv_rho = 0.0  # diagonal exactly

    # Row decay rate (middle row)
    alpha_L_inv, resid = row_decay_rate(L_inv)

    # Frobenius norms
    fro_L_inv = float(np.linalg.norm(L_inv, ord="fro"))
    fro_inv_rho = float(np.linalg.norm(inv_rho, ord="fro"))

    # Spectral radius / min-max singular values (conditioning)
    svd_L_inv = np.linalg.svd(L_inv, compute_uv=False)
    cond_L_inv = float(svd_L_inv[0] / svd_L_inv[-1])

    return dict(
        N_y=N_y,
        off_diag_frac_L_inv=float(off_L_inv),
        off_diag_frac_diag_inv_rho=float(off_inv_rho),
        effective_bw_L_inv=bw_L_inv,
        effective_bw_diag_inv_rho=bw_inv_rho,
        row_decay_alpha=alpha_L_inv,
        row_decay_fit_resid=resid,
        frobenius_L_inv=fro_L_inv,
        frobenius_diag_inv_rho=fro_inv_rho,
        cond_L_inv=cond_L_inv,
    )


def visualise_matrices(N_y=64, Ly=1.0, kx=2 * np.pi, rho_cut=0.05):
    """Render side-by-side log-magnitude heatmaps of L⁻¹ vs diag(1/ρ)."""
    y, D = cgl_grid(N_y, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    L_int, R_int = assemble_operator("vspace", y, D, rho, N2, kx)
    L_inv = np.linalg.inv(L_int)
    rho_int = rho[1:-1]
    inv_rho = np.diag(1.0 / rho_int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=130)
    # Normalise both to same colour scale
    vmax = max(np.max(np.abs(L_inv)), np.max(np.abs(inv_rho)))
    extent = [0, N_y - 2, N_y - 2, 0]

    im0 = axes[0].imshow(np.log10(np.abs(L_inv) + 1e-20),
                         cmap="viridis", aspect="equal",
                         vmin=np.log10(1e-6 * vmax),
                         vmax=np.log10(vmax), extent=extent)
    axes[0].set_title(r"$L^{-1}$  (global, nonlocal)")
    axes[0].set_xlabel("column $j$")
    axes[0].set_ylabel("row $i$")
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label=r"$\log_{10}|M_{ij}|$")

    im1 = axes[1].imshow(np.log10(np.abs(inv_rho) + 1e-20),
                         cmap="viridis", aspect="equal",
                         vmin=np.log10(1e-6 * vmax),
                         vmax=np.log10(vmax), extent=extent)
    axes[1].set_title(r"$\mathrm{diag}(1/\rho_0)$  (local, pointwise)")
    axes[1].set_xlabel("column $j$")
    axes[1].set_ylabel("row $i$")
    plt.colorbar(im1, ax=axes[1], shrink=0.85, label=r"$\log_{10}|M_{ij}|$")

    fig.suptitle(f"The locality gap: $L^{{-1}}$ versus $\\mathrm{{diag}}(1/\\rho_0)$,  $N_y = {N_y}$",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = OUT_DIR / "matrix_heatmap.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Wrote {out_png}")


def main():
    N_list = [32, 48, 64, 96, 128, 192, 256]
    rows = []
    print("  Locality-gap probe: L⁻¹ vs diag(1/ρ)")
    print("  " + "-" * 80)
    print("  N_y   off(L⁻¹)   off(diag)  bw(L⁻¹)   decay α   ‖L⁻¹‖_F    ‖diag‖_F")
    print("  " + "-" * 80)
    for N in N_list:
        r = measure_one(N)
        rows.append(r)
        print(f"  {r['N_y']:4d}  "
              f"{r['off_diag_frac_L_inv']:.4f}     "
              f"{r['off_diag_frac_diag_inv_rho']:.4f}     "
              f"{r['effective_bw_L_inv']:5.1f}     "
              f"{r['row_decay_alpha']:.3f}      "
              f"{r['frobenius_L_inv']:.3e}   "
              f"{r['frobenius_diag_inv_rho']:.3e}")
        rows.append(r)
    # Drop duplicates that slipped in
    seen = set(); unique = []
    for r in rows:
        k = r["N_y"]
        if k not in seen:
            seen.add(k); unique.append(r)

    with open(OUT_DIR / "locality_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(unique[0].keys()))
        w.writeheader()
        for r in unique:
            w.writerow(r)
    print(f"\nWrote {OUT_DIR / 'locality_gap.csv'}")

    # Visualisation
    print("\nGenerating matrix heatmap figure (N_y = 64)...")
    visualise_matrices(N_y=64)


if __name__ == "__main__":
    main()
