#!/usr/bin/env python3
"""
E7 / Exp K: Chebyshev spectral 4-variable GYRE-full operator.

Uses raw Chebyshev-Gauss-Lobatto collocation (no prefactor, no cutoff —
relies on E6 v2 observation that Lane-Emden n=3 with σ=3 integer surface
exponent has spectral convergence on Chebyshev polynomials).

Equations from `gyre/src/eqns/ad/ad_eqns_m.fypp` + `A_t.inc` with
alpha_grv = alpha_omg = alpha_gam = alpha_pi = 1:

    x dy_1/dx = (V_g - ell - 1) y_1 + (λ/(c_1 ω²) - V_g) y_2 + λ/(c_1 ω²) y_3
    x dy_2/dx = (c_1 ω² - A*_iso) y_1 + (A* - U + 3 - ell) y_2 - y_4
    x dy_3/dx = (3 - U - ell) y_3 + y_4
    x dy_4/dx = U A* y_1 + U V_g y_2 + λ y_3 + (2 - U - ell) y_4

BCs (IB_regular + OB_vacuum):
    IB1: c_1(0) ω² y_1(0) - ell y_2(0) - ell y_3(0) = 0
    IB2: ell y_3(0) - y_4(0) = 0
    OB1: y_1(R) - y_2(R) = 0
    OB2: U(R) y_1(R) + (ell+1) y_3(R) + y_4(R) = 0

Multiply eq1 by ω² to linearise (ω² appears as c_1 ω² AND λ/(c_1 ω²)):

    ω² [x dy_1/dx - (V_g-ell-1)y_1 + V_g y_2] = λ/c_1 (y_2 + y_3)
    ω² [c_1 y_1]  = x dy_2/dx + A*_iso y_1 - (A*-U+3-ell) y_2 + y_4
    0             = x dy_3/dx - (3-U-ell) y_3 - y_4
    0             = x dy_4/dx - U A* y_1 - U V_g y_2 - λ y_3 - (2-U-ell) y_4

⇒ ω² P u = Q u,  u = [y_1; y_2; y_3; y_4] stacked.

PASS criterion: max_rel < 1e-3 vs frozen EXPECTED from Exp J at N <= 128.
Stretch goal: N = 64 reaches 5e-4 (= Exp J's Nr=1024 accuracy).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmode_infra as gi
from gmode_exp_j_full_gyre_compat import load_gyre_modes

REF_DOC = "docs/spectral_liouville_plan_2026-05-03.md"
SCRIPT_REL = "scripts/gmode_exp_k_chebyshev_full.py"

GYRE_STRUCTURE_TXT = Path("/tmp/gyre_run/poly3.txt")
GYRE_FULL_H5 = Path("/tmp/gyre_run/summary.h5")

# Frozen reference from Exp J Nr=1024 (the FD staggered solver matched
# GYRE to 5.3e-4 max_rel, n_g=1 to 6 digits).  We re-target the SAME
# GYRE EXPECTED here — the goal is "does Chebyshev reach similar
# accuracy with far fewer DOF".
EXPECTED_OMSQ_GYRE = [
    2.5159279360877496, 1.2857077544856306, 0.7757327764772477,
    0.5177759762324133, 0.36992549567563754, 0.2775028154601723,
    0.21592664733814718, 0.1728536032702941, 0.1415440904624304,
    0.11806842352742726,
]


def cheb_D1_on_interval(N, a, b):
    """Chebyshev-Gauss-Lobatto grid + 1st-derivative matrix on [a, b],
    reordered ascending."""
    D_raw, x_raw = gi.cheb(N)
    scale = 2.0 / (b - a)
    D1 = D_raw * scale
    idx = np.argsort(x_raw)
    x = a + (x_raw[idx] + 1.0) * (b - a) / 2.0
    P = np.zeros_like(D1)
    P[np.arange(N + 1), idx] = 1.0
    D1 = P @ D1 @ P.T
    return x, D1


def solve_gmode_full_chebyshev(x_nodes, V_2, U, A_star, c_1, Gamma_1, ell, n_modes,
                               alpha_gam=1.0, p_frac_cutoff=0.05):
    """Chebyshev spectral version of solve_gmode_full_gyre_compat.

    x_nodes     : CGL nodes on [x_lo, x_hi] (ascending), length Nr+1
    V_2, U, ... : structure coefficients evaluated at x_nodes
    ell, n_modes: as before

    Returns (omega_sq, u_out).  omega_sq descending (n_g=1 first).
    """
    x = np.asarray(x_nodes, dtype=float)
    Nr = len(x)
    _, D1 = cheb_D1_on_interval(Nr - 1, x[0], x[-1])

    V_2 = np.asarray(V_2, dtype=float)
    U = np.asarray(U, dtype=float)
    A_star = np.asarray(A_star, dtype=float)
    c_1 = np.asarray(c_1, dtype=float)
    Gamma_1 = np.asarray(Gamma_1, dtype=float)
    lam = ell * (ell + 1.0)
    V_g = V_2 * x ** 2 / Gamma_1
    A_iso = A_star * np.where(A_star > 0, alpha_gam, 1.0)

    # Unknown layout: u = [y_1; y_2; y_3; y_4], each Nr long, total 4Nr.
    def iy1(n): return n
    def iy2(n): return Nr + n
    def iy3(n): return 2 * Nr + n
    def iy4(n): return 3 * Nr + n

    size = 4 * Nr
    P = np.zeros((size, size))
    Q = np.zeros((size, size))

    # Convenience: x·D1 matrix (N_node × N_node)
    xD1 = np.diag(x) @ D1

    # ---- eq1 (y_1 equation, row index 0..Nr-1) ----
    # P row: x dy_1/dx - (V_g - ell - 1) y_1 + V_g y_2
    # Q row: (λ/c_1) y_2 + (λ/c_1) y_3
    # → P[r, :Nr]   = x·D1 - diag(V_g-ell-1)
    #   P[r, Nr:2Nr]= diag(V_g)
    #   Q[r, Nr:2Nr]= diag(λ/c_1)
    #   Q[r, 2Nr:3Nr] = diag(λ/c_1)
    for i in range(Nr):
        P[i, :Nr]       = xD1[i, :]
        P[i, iy1(i)]   -= (V_g[i] - ell - 1.0)
        P[i, iy2(i)]   += V_g[i]
        Q[i, iy2(i)]   += lam / c_1[i]
        Q[i, iy3(i)]   += lam / c_1[i]

    # ---- eq2 (y_2 equation, row index Nr..2Nr-1) ----
    # P row: c_1 y_1
    # Q row: x dy_2/dx + A_iso y_1 - (A* - U + 3 - ell) y_2 + y_4
    # → P[r, :Nr]    diag(c_1)
    #   Q[r, :Nr]    diag(A_iso)
    #   Q[r, Nr:2Nr] x·D1 - diag(A*-U+3-ell)
    #   Q[r, 3Nr:]   diag(1)
    for i in range(Nr):
        r = Nr + i
        P[r, iy1(i)]   = c_1[i]
        Q[r, iy1(i)]   = A_iso[i]
        Q[r, Nr:2 * Nr] = xD1[i, :]
        Q[r, iy2(i)]  -= (A_star[i] - U[i] + 3.0 - ell)
        Q[r, iy4(i)]  += 1.0

    # ---- eq3 (y_3 equation, row index 2Nr..3Nr-1) ----
    # No ω². P row = 0; put whole eq on Q.
    # Equation: x dy_3/dx - (3 - U - ell) y_3 - y_4 = 0
    for i in range(Nr):
        r = 2 * Nr + i
        Q[r, 2 * Nr:3 * Nr] = xD1[i, :]
        Q[r, iy3(i)]       -= (3.0 - U[i] - ell)
        Q[r, iy4(i)]       -= 1.0

    # ---- eq4 (y_4 equation, row index 3Nr..4Nr-1) ----
    # No ω². Equation: x dy_4/dx - U A* y_1 - U V_g y_2 - λ y_3 - (2-U-ell) y_4 = 0
    for i in range(Nr):
        r = 3 * Nr + i
        Q[r, iy1(i)]       -= U[i] * A_star[i]
        Q[r, iy2(i)]       -= U[i] * V_g[i]
        Q[r, iy3(i)]       -= lam
        Q[r, 3 * Nr:4 * Nr] = xD1[i, :]
        Q[r, iy4(i)]       -= (2.0 - U[i] - ell)

    # ---- BCs ----
    # Overwrite 4 rows at the boundary nodes.  Chebyshev endpoints are
    # x[0] (inner) and x[-1] (outer).
    # IB1 (has ω²):  c_1(0) ω² y_1(0) - ell y_2(0) - ell y_3(0) = 0
    #   equiv:  ω² c_1(0) y_1(0)  =  ell y_2(0) + ell y_3(0)
    # Overwrite eq1 row at i=0 (index 0)
    r = 0
    P[r, :] = 0.0; Q[r, :] = 0.0
    P[r, iy1(0)] = c_1[0]
    Q[r, iy2(0)] = ell
    Q[r, iy3(0)] = ell

    # IB2 (no ω²): ell y_3(0) - y_4(0) = 0
    # Overwrite eq3 row at i=0 (index 2Nr)
    r = 2 * Nr
    P[r, :] = 0.0; Q[r, :] = 0.0
    Q[r, iy3(0)] = ell
    Q[r, iy4(0)] = -1.0

    # OB1 (no ω²): y_1(R) - y_2(R) = 0
    # Overwrite eq1 row at i=Nr-1 (index Nr-1)
    r = Nr - 1
    P[r, :] = 0.0; Q[r, :] = 0.0
    Q[r, iy1(Nr - 1)] =  1.0
    Q[r, iy2(Nr - 1)] = -1.0

    # OB2 (no ω²): U(R) y_1(R) + (ell+1) y_3(R) + y_4(R) = 0
    # Overwrite eq4 row at i=Nr-1 (index 4Nr-1)
    r = 4 * Nr - 1
    P[r, :] = 0.0; Q[r, :] = 0.0
    Q[r, iy1(Nr - 1)] = U[-1]
    Q[r, iy3(Nr - 1)] = (ell + 1.0)
    Q[r, iy4(Nr - 1)] = 1.0

    # Solve ω² P u = Q u
    mu, vecs = scipy.linalg.eig(Q, P)
    mu_r = mu.real
    good = (np.isfinite(mu_r)
            & (np.abs(mu.imag) < 1e-6 * (np.abs(mu_r) + 1e-30))
            & (mu_r > 0))
    mu_g = mu_r[good]
    vec_g = vecs[:, good].real

    # Propagation-cavity g-mode classification (same as infra)
    N2_profile = A_star / np.maximum(c_1, 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        L2_profile = np.where(V_2 > 0, lam * Gamma_1 / (V_2 * x ** 4), np.inf)

    def classify(y1, mu_val):
        w = x ** 3 * y1 ** 2
        total = w.sum()
        if total <= 0:
            return 0.0, 0.0
        in_g = (N2_profile > mu_val) & (L2_profile > mu_val)
        in_p = (N2_profile < mu_val) & (L2_profile < mu_val)
        return (w * in_g).sum() / total, (w * in_p).sum() / total

    kept = []
    for k in range(vec_g.shape[1]):
        y1 = vec_g[:Nr, k]
        gf, pf = classify(y1, mu_g[k])
        if pf < p_frac_cutoff:
            kept.append(k)
    if not kept:
        return np.empty(0), np.empty((size, 0))
    mu_ok = mu_g[kept]; vec_ok = vec_g[:, kept]

    order = np.argsort(-mu_ok)
    mu_sel = mu_ok[order]; vec_sel = vec_ok[:, order]

    # Dedup
    keep = []
    for i in range(len(mu_sel)):
        if not keep:
            keep.append(i); continue
        if abs(mu_sel[i] - mu_sel[keep[-1]]) / abs(mu_sel[keep[-1]] + 1e-30) > 1e-4:
            keep.append(i)
        if len(keep) >= n_modes:
            break
    mu_sel = mu_sel[keep]; vec_sel = vec_sel[:, keep]

    n_out = min(n_modes, len(mu_sel))
    return mu_sel[:n_out], vec_sel[:, :n_out]


def load_gyre_structure_interp_cheb(path, N, inner_cut=0.0001, outer_cut=0.9999):
    """Load GYRE structure, interpolate onto CGL grid with CubicSpline.

    Linear interpolation (np.interp) limits precision to O(h²) in input
    resolution, which dominates the spectral error floor.  CubicSpline gives
    O(h⁴) and lets the Chebyshev spectral accuracy show through.
    """
    data = np.loadtxt(path, skiprows=1)
    x_full = data[:, 0]; V_2 = data[:, 1]; A_star = data[:, 2]
    U = data[:, 3]; c_1 = data[:, 4]; Gamma_1 = data[:, 5]

    mask = (x_full > inner_cut) & (x_full < outer_cut)
    x_src = x_full[mask]
    x_cheb, _ = cheb_D1_on_interval(N, x_src[0], x_src[-1])
    V_2u = CubicSpline(x_src, V_2[mask])(x_cheb)
    A_u  = CubicSpline(x_src, A_star[mask])(x_cheb)
    U_u  = CubicSpline(x_src, U[mask])(x_cheb)
    c_1u = CubicSpline(x_src, c_1[mask])(x_cheb)
    G1_u = CubicSpline(x_src, Gamma_1[mask])(x_cheb)
    return x_cheb, V_2u, A_u, U_u, c_1u, G1_u


def main():
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Exp K: Chebyshev spectral 4-var GYRE-full operator (Lane-Emden n=3)")
    print("=" * 72)
    print()

    ng_gyre, omsq_gyre_h5 = load_gyre_modes(GYRE_FULL_H5)
    n_compare = 10
    ng_gyre = ng_gyre[:n_compare]
    omsq_gyre_h5 = omsq_gyre_h5[:n_compare]

    # Sanity: frozen EXPECTED should match h5
    assert np.allclose(omsq_gyre_h5, EXPECTED_OMSQ_GYRE, rtol=1e-10), \
        f"GYRE h5 drifted: {omsq_gyre_h5[:3]} vs {EXPECTED_OMSQ_GYRE[:3]}"
    print(f"  GYRE full-gravity EXPECTED n_g=1..10 loaded (frozen).")
    print(f"  ω²_GYRE[n_g=1] = {omsq_gyre_h5[0]:.6f}")
    print()

    N_list = [16, 24, 32, 48, 64, 96, 128]
    results = []
    for N in N_list:
        x, V_2, A_star, U, c_1, G1 = load_gyre_structure_interp_cheb(GYRE_STRUCTURE_TXT, N)
        omsq, _ = solve_gmode_full_chebyshev(x, V_2, U, A_star, c_1, G1, 1, n_compare + 5)
        omsq = omsq[:n_compare]
        n_found = len(omsq)
        # Pad with NaN if fewer than n_compare
        if n_found < n_compare:
            omsq_padded = np.concatenate([omsq, np.full(n_compare - n_found, np.nan)])
        else:
            omsq_padded = omsq[:n_compare]

        rd = np.abs(omsq_padded - np.array(EXPECTED_OMSQ_GYRE)) / np.array(EXPECTED_OMSQ_GYRE)
        rd_max = np.nanmax(rd)
        rd_ng1 = rd[0] if np.isfinite(rd[0]) else np.nan
        results.append((N, n_found, rd_ng1, rd_max, omsq_padded))
        print(f"  N = {N:4d}  (DOF = {4*N:5d})   n_found={n_found:2d}   "
              f"rd(n_g=1)={rd_ng1:.3e}   max_rd={rd_max:.3e}")

    # Compare with Exp J frozen reference
    EXP_J_MAX_REL = 5.3e-4
    EXP_J_NR = 1024
    print()
    print(f"  Exp J reference: Nr={EXP_J_NR} (DOF = {4*EXP_J_NR}), "
          f"max_rd={EXP_J_MAX_REL:.2e}")
    print()

    # Find smallest N that beats Exp J max_rel
    beats_idx = None
    for i, (N, nf, rd1, rdm, _) in enumerate(results):
        if np.isfinite(rdm) and rdm <= EXP_J_MAX_REL:
            beats_idx = i
            break
    if beats_idx is not None:
        N_win = results[beats_idx][0]
        speedup = EXP_J_NR / N_win
        print(f"  Chebyshev matches Exp J accuracy at N = {N_win} "
              f"(DOF = {4*N_win}), {speedup:.1f}× fewer DOF.")
    else:
        print(f"  Chebyshev did NOT beat Exp J in tested N range.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    N_arr = np.array([r[0] for r in results])
    rd1_arr = np.array([r[2] for r in results])
    rdm_arr = np.array([r[3] for r in results])

    axes[0].loglog(N_arr, rd1_arr, "C0o-", lw=1.5, ms=7, label="rel_diff(n_g=1)")
    axes[0].loglog(N_arr, rdm_arr, "C3s-", lw=1.5, ms=7, label="max rel_diff (n_g=1..10)")
    axes[0].axhline(EXP_J_MAX_REL, ls="--", color="gray", lw=1,
                    label=f"Exp J Nr=1024 ({EXP_J_MAX_REL:.1e})")
    axes[0].axhline(1e-3, ls=":", color="red", lw=1, label="Gate: 1e-3")
    axes[0].set_xlabel("N (Chebyshev order)")
    axes[0].set_ylabel("relative difference vs GYRE EXPECTED")
    axes[0].set_title("Chebyshev spectral convergence — Lane-Emden n=3 g-modes")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    # Semilog to detect exponential
    axes[1].semilogy(N_arr, rdm_arr, "C3s-", lw=1.5, ms=7, label="max rel_diff")
    axes[1].axhline(EXP_J_MAX_REL, ls="--", color="gray", lw=1)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("max rel_diff")
    axes[1].set_title("Semilog — exponential convergence check")
    axes[1].grid(alpha=0.3, which="both")

    # Slope fits
    mask = np.isfinite(rdm_arr) & (rdm_arr > 0) & (N_arr >= 24)
    if mask.sum() >= 2:
        slope_semilog = np.polyfit(N_arr[mask], np.log(rdm_arr[mask]), 1)[0]
        slope_loglog  = np.polyfit(np.log(N_arr[mask]), np.log(rdm_arr[mask]), 1)[0]
        print(f"  semilog slope log(max_rd)/N     = {slope_semilog:+.4f}  "
              f"(exponential if < -0.05)")
        print(f"  loglog  slope d log(max_rd)/d log(N) = {slope_loglog:+.3f}  "
              f"(algebraic order)")
        axes[1].set_title(f"Semilog slope = {slope_semilog:+.3f}")

    fig.tight_layout()
    out = gi.VID / "gmode_exp_k_chebyshev_full.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\n  => {out}")


if __name__ == "__main__":
    main()
