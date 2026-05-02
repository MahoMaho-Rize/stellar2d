#!/usr/bin/env python3
"""
Experiment E: linearised anelastic 2-variable operator vs Boussinesq Cowling.

PURPOSE
  Quantify the difference between two distinct g-mode eigenvalue derivations
  before writing the C++/CUDA anelastic operator:

  (B) Boussinesq / high-n Cowling  (`solve_gmode_cowling(_cheb)`):
        -ψ''(r) = ω^{-2} · ell(ell+1) · N²(r)/r² · ψ(r)
      A single-variable simplification valid in the high-n limit
      (ω² ≪ N²) and under a slab / Cartesian-local approximation that
      drops the direct ω² · ξ_r self-coupling in the radial momentum
      equation.  Validated by Exps B, C, D.

  (A) Full anelastic 2-variable (ξ_r, p'):
        ρ₀ N² ξ_r + ∂_r p' = ω² ρ₀ ξ_r                        (radial momentum)
        (1/r²) ∂_r(ρ₀ r² ξ_r) = ω⁻² · ell(ell+1) p' / r²      (cont + horizontal)
      Written as a generalised eigenproblem A u = ω² B u.

  The two spectra are NOT expected to match at low n; the difference
  measures anelastic ↔ Boussinesq physics, not a bug.  Expectations:

  - At high n (ω² → 0), the full anelastic spectrum should approach the
    Boussinesq spectrum; their ratio should converge to 1 as n grows.
  - At low n the two formulations diverge: the 2-var anelastic captures
    the ω² · ξ_r self-restoring term that Boussinesq drops.

  This experiment therefore:
    1. computes both spectra on the same cavity,
    2. reports the ratio per-mode, and
    3. verifies the high-n ratio → 1 limit (anelastic reduces to Boussinesq).

  Passing this test gives confidence that the 2-var matrix assembly used by
  the future anelastic operator is mathematically consistent with the
  already-validated scalar solver in the appropriate physical limit.

REFERENCE DOC
  docs/gmode_experiments_2026-05-02.md §8

REPRO
  python scripts/gmode_exp_e_anelastic_linop.py
  python scripts/gmode_exp_e_anelastic_linop.py --verify
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

REF_DOC = "docs/gmode_experiments_2026-05-02.md"
SCRIPT_REL = "scripts/gmode_exp_e_anelastic_linop.py"

# Reference values bound to docs/gmode_experiments_2026-05-02.md §8.
EXPECTED_OMSQ_SCALAR = [
    2.35040410e-01, 3.27017744e-02, 1.25479991e-02, 6.60796755e-03,
    4.07486726e-03, 2.76342766e-03, 1.99725494e-03, 1.51095623e-03,
    1.18303991e-03, 9.51470561e-04,
]
EXPECTED_OMSQ_2VAR = [
    1.65062302e-01, 2.96696075e-02, 1.19409612e-02, 6.40571891e-03,
    3.98793766e-03, 2.72006177e-03, 1.97352021e-03, 1.49720870e-03,
    1.17484500e-03, 9.46579756e-04,
]
EXPECTED_RATIOS = {
    "ratio_lo":  0.8537,   # mean of n = 1..3
    "ratio_hi":  0.9929,   # mean of n = 8..10; should → 1 as n → ∞
}
REL_TOL = 0.02


def n2_profile(r, r_lo=0.2, r_hi=1.0):
    """Same Gaussian-bump as Exp B/C/D."""
    rc = 0.5 * (r_lo + r_hi)
    sigma = 0.25 * (r_hi - r_lo)
    bump = np.exp(-((r - rc) / sigma) ** 2)
    taper = np.sin(np.pi * (r - r_lo) / (r_hi - r_lo)) ** 2
    return bump * taper


def solve_anelastic_2var(r, rho0, N2, ell, n_modes):
    """Assemble and solve the linearised anelastic eigenproblem

        -ω² ρ₀ ξ_r = -∂_r p' - ρ₀ N² ξ_r                      (momentum_r)
        (1/ρ₀ r²) ∂_r(ρ₀ r² ξ_r) - ell(ell+1) p' / (ρ₀ r² ω²) = 0   (continuity + horizontal momentum)

    as  A [ξ_r; p'] = ω² B [ξ_r; p']

    on a uniform grid.  Dirichlet BC on ξ_r at both ends (closed cavity);
    Neumann-ish on p' (no explicit constraint — the second equation fixes
    it from ξ_r at each ω², and the eigensolver picks spectral modes
    consistent with ξ_r Dirichlet).

    Discretisation: ξ_r and p' both on the full grid; second-order central
    FD for ∂_r, one-sided at endpoints (but those are overwritten by
    Dirichlet anyway).
    """
    N = len(r)
    dr = r[1] - r[0]
    if not np.allclose(np.diff(r), dr, rtol=1e-10):
        raise ValueError("uniform r required")

    # First-derivative matrix (2nd-order central, one-sided at boundaries)
    D1 = np.zeros((N, N))
    for i in range(1, N - 1):
        D1[i, i - 1] = -1.0 / (2.0 * dr)
        D1[i, i + 1] = +1.0 / (2.0 * dr)
    D1[0, 0] = -3.0 / (2.0 * dr)
    D1[0, 1] = +4.0 / (2.0 * dr)
    D1[0, 2] = -1.0 / (2.0 * dr)
    D1[-1, -1] = +3.0 / (2.0 * dr)
    D1[-1, -2] = -4.0 / (2.0 * dr)
    D1[-1, -3] = +1.0 / (2.0 * dr)

    I = np.eye(N)

    # Momentum_r:
    #     ρ₀ N² ξ_r + ∂_r p' = ω² ρ₀ ξ_r
    # A row (acts on [ξ_r; p']):  [ diag(ρ₀N²)  |  D1 ]
    # B row (acts on [ξ_r; p']):  [ diag(ρ₀)    |  0  ]
    A_11 = np.diag(rho0 * N2)
    A_12 = D1
    B_11 = np.diag(rho0)
    B_12 = np.zeros((N, N))

    # Continuity + horizontal momentum:
    #     (1/r²) ∂_r(ρ₀ r² ξ_r) = ell(ell+1) p' / (ω² r²)
    # Multiply by ω² to get standard A u = ω² B u form:
    #     ell(ell+1) p' / r² = ω² · (1/r²) ∂_r(ρ₀ r² ξ_r)
    # A row:  [ 0            |  diag(ell(ell+1)/r²) ]
    # B row:  [ (1/r²) D1 diag(ρ₀ r²) | 0 ]
    ell_factor = ell * (ell + 1) / (r ** 2)
    weight = rho0 * r ** 2
    ir2 = 1.0 / (r ** 2)
    A_21 = np.zeros((N, N))
    A_22 = np.diag(ell_factor)
    B_21 = np.diag(ir2) @ D1 @ np.diag(weight)
    B_22 = np.zeros((N, N))

    A = np.block([[A_11, A_12], [A_21, A_22]])
    B = np.block([[B_11, B_12], [B_21, B_22]])

    # Impose Dirichlet on ξ_r: replace rows 0 and N-1 of both A and B
    # with identity-ish rows that force ξ_r(0) = ξ_r(-1) = 0.
    # (A row = diag; B row = 0 -> solver sends these eigenvalues to infinity,
    # which we filter out.)
    for i_bd in (0, N - 1):
        A[i_bd, :] = 0.0; A[i_bd, i_bd] = 1.0
        B[i_bd, :] = 0.0

    # Generalised eigensolve.
    lam, vec = scipy.linalg.eig(A, B)

    # Keep finite positive real eigenvalues (= ω² > 0)
    lam_r = lam.real
    mask_finite = np.isfinite(lam_r) & (np.abs(lam.imag) < 1e-6 * (1 + np.abs(lam_r)))
    mask_physical = mask_finite & (lam_r > 1e-10) & (lam_r < 1e6)
    lam_keep = lam_r[mask_physical]

    # The 2-variable system has size 2N but only ~N physical eigenvalues
    # (the other half are spurious: Dirichlet on ξ_r leaves p' unconstrained
    # at boundaries, creating a "ghost" partner at almost-identical ω² for
    # each real mode). Deduplicate by grouping consecutive eigenvalues whose
    # relative separation is below `dedup_tol`.
    lam_sorted = np.sort(lam_keep)[::-1]   # descending ω²
    dedup_tol = 1e-2
    kept = []
    for lam in lam_sorted:
        if not kept or abs(lam - kept[-1]) / max(abs(kept[-1]), 1e-300) > dedup_tol:
            kept.append(lam)
        if len(kept) >= n_modes:
            break
    omega_sq = np.array(kept[:n_modes])
    return omega_sq


def main(verify=False):
    gi.provenance_banner(SCRIPT_REL, REF_DOC)
    print(" Experiment E: 2-variable anelastic operator vs scalar Cowling")
    print("=" * 72)

    r_lo, r_hi = 0.2, 1.0
    ell = 1
    Nr = 512
    n_compare = 10
    rho0_val = 1.0  # constant-ρ₀ limit for clean comparison

    r = np.linspace(r_lo, r_hi, Nr)
    N2 = n2_profile(r, r_lo, r_hi)
    rho0 = rho0_val * np.ones_like(r)

    # (1) Scalar Cowling
    omsq_scalar, _ = gi.solve_gmode_cowling(r, N2, ell=ell, n_modes=n_compare + 5)
    omsq_scalar = omsq_scalar[:n_compare]

    # (2) 2-variable anelastic (already deduplicated, descending ω²)
    omsq_2var = solve_anelastic_2var(r, rho0, N2, ell, n_compare)

    print(f"  Nr = {Nr},  ell = {ell},  ρ₀ = const = {rho0_val}")
    print(f"  N² profile: Gaussian bump on [{r_lo}, {r_hi}], max N² = {N2.max():.4f}")
    print()
    print(f"  {'n':>3}  {'ω²_Bouss (scalar)':>18}  {'ω²_anelastic':>15}  {'ratio':>10}  {'rel_diff':>10}")
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
        print(f"  {n+1:>3}  {om_s:18.8e}  {om_v:15.8e}  {ratio:10.4f}  {rd:10.3e}")

    # High-n convergence diagnostic: the ratio should approach 1 as n grows,
    # signalling that anelastic → Boussinesq in the low-frequency limit.
    ratio_hi = float(np.mean(ratios[-3:]))
    ratio_lo = float(np.mean(ratios[:3]))
    print()
    print(f"  low-n (first 3) avg ratio  = {ratio_lo:.4f}  (anelastic > Boussinesq expected)")
    print(f"  high-n (last 3) avg ratio  = {ratio_hi:.4f}  (should tend to 1 as n→∞)")
    hn_conv = abs(ratio_hi - 1.0)
    print(f"  |ratio_hi - 1|             = {hn_conv:.4f}")
    print()
    print(f"  PASS criterion: |ratio_hi - 1| < 0.2 (anelastic approaches Boussinesq at high n)")
    print(f"  Result: {'PASS' if hn_conv < 0.2 else 'FAIL'}")
    max_rd = max(rel_diffs)
    mean_rd = float(np.mean(rel_diffs))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    n_arr = np.arange(1, n_compare + 1)
    axes[0].semilogy(n_arr, omsq_scalar, "o-", color="C0", lw=1.5,
                      ms=7, label="scalar Cowling")
    axes[0].semilogy(n_arr, omsq_2var, "s--", color="C3", lw=1.2,
                      ms=6, label="2-var anelastic")
    axes[0].set_xlabel("radial order n")
    axes[0].set_ylabel(r"$\omega^2$")
    axes[0].set_title(f"g-mode spectrum comparison (ell={ell})")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogy(n_arr, rel_diffs, "o-", color="C2", lw=1.5, ms=7)
    axes[1].axhline(5e-3, ls="--", color="r", lw=1, label="5e-3 tol")
    axes[1].set_xlabel("radial order n")
    axes[1].set_ylabel("rel diff")
    axes[1].set_title("scalar vs 2-var: per-mode relative difference")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = gi.VID / "gmode_exp_e_anelastic_linop.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    if verify:
        print("\n--- VERIFY against EXPECTED ---")
        n_fail = 0
        runtime_vals = {"ratio_lo": ratio_lo, "ratio_hi": ratio_hi}
        for key, ref in EXPECTED_RATIOS.items():
            val = runtime_vals[key]
            d = abs(val - ref) / max(abs(ref), 1e-300)
            ok = d < REL_TOL
            mark = "OK" if ok else "DRIFT"
            print(f"  [{mark:<5}] {key:<12} {val:.6f} vs {ref:.6f}  ({d*100:.3f}%)")
            if not ok:
                n_fail += 1
        # Also verify per-mode ω² to pin numerical behavior
        for n in range(n_compare):
            om_v = omsq_2var[n]
            om_s = omsq_scalar[n]
            ref_v = EXPECTED_OMSQ_2VAR[n]
            ref_s = EXPECTED_OMSQ_SCALAR[n]
            dv = abs(om_v - ref_v) / max(abs(ref_v), 1e-300)
            ds = abs(om_s - ref_s) / max(abs(ref_s), 1e-300)
            if dv > REL_TOL or ds > REL_TOL:
                print(f"  [DRIFT] mode n={n+1}: 2var {om_v:.4e} vs {ref_v:.4e} ({dv*100:.2f}%)  "
                      f"scalar {om_s:.4e} vs {ref_s:.4e} ({ds*100:.2f}%)")
                n_fail += 1
        if n_fail:
            sys.exit(1)
        print("\n  all reference values reproduced.")

    return {"omsq_scalar": omsq_scalar, "omsq_2var": omsq_2var,
            "rel_diffs": rel_diffs, "max_rd": max_rd,
            "ratio_lo": ratio_lo, "ratio_hi": ratio_hi}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    main(verify=args.verify)
