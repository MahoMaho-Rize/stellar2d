#!/usr/bin/env python3
"""E1: Variable-coefficient generalised EVP test for Proposition 1.

The purpose is to show that Proposition 1 is a structural property of
primitive-node pseudo-spectral discretisations of variable-coefficient
elliptic generalised eigenproblems, not a phenomenon specific to
stellar anelastic g-modes.

Problem.  Consider the generalised EVP
    L u = lambda R u,    u(0) = u(L_y) = 0
with
    L = -d/dy( a(y) du/dy ) + k_x^2 a(y) u   (elliptic with coeff a)
    R = k_x^2 b(y) u                          (reaction with coeff b)

The stellar anelastic problem is the case a = rho_0, b = N^2 rho_0
(so b/a = N^2), but the construction is generic: any two positive
smooth functions a, b produce a well-defined EVP, and the primitive-
node time-stepping scheme substitutes

    L^{-1}  ->  diag(1/a)

giving the operator M_prim = diag(b/a) instead of M_asm = L^{-1} R.

Proposition 1 predicts the relative gap on the top eigenvector V_n:

    || (M_prim - M_asm) V_n ||_w / || M_asm V_n ||_w
      = sqrt( Var_mu( k_x^2 b/a ) ) / | lambda_n |

with dmu = V_n(y)^2 w(y) dy.  The gap is nonzero whenever b/a is
non-constant on the support of V_n, independent of N_y, and zero in
the Boussinesq limit b/a = const.

We test four (a, b) pairs chosen to have nothing to do with stellar
structure:

  (A)  a = 1 + 0.5 sin(pi y / L_y),    b = 2 + cos(2 pi y / L_y)
  (B)  a = exp(-2 y / L_y),             b = exp(-y / L_y)
  (C)  a = 1 + 0.8 y / L_y,             b = (1 + y / L_y)^2
  (D)  a = 1 + 0.5 sin(pi y / L_y),    b = 1 + 0.5 sin(pi y / L_y)
       ==>  b/a = 1 identically; Proposition 1 predicts c = 0.

For each pair, sweep N_y in {32, 48, 64, 96, 128, 192, 256} at
k_x = 2 pi / L_y.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nonlinear_paths_infra import cgl_grid, cc_weights

OUT_DIR = SCRIPT_DIR.parent / "review" / "e1_poisson"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CASES = {
    "A_trig":
        dict(
            a=lambda y, L: 1.0 + 0.5 * np.sin(np.pi * y / L),
            b=lambda y, L: 2.0 + np.cos(2.0 * np.pi * y / L),
            desc="a = 1 + 0.5 sin(pi y/L),  b = 2 + cos(2 pi y/L)"),
    "B_exp":
        dict(
            a=lambda y, L: np.exp(-2.0 * y / L),
            b=lambda y, L: np.exp(-1.0 * y / L),
            desc="a = exp(-2 y/L),  b = exp(-y/L)"),
    "C_poly":
        dict(
            a=lambda y, L: 1.0 + 0.8 * y / L,
            b=lambda y, L: (1.0 + y / L) ** 2,
            desc="a = 1 + 0.8 y/L,  b = (1 + y/L)^2"),
    "D_ratio1":
        dict(
            a=lambda y, L: 1.0 + 0.5 * np.sin(np.pi * y / L),
            b=lambda y, L: 1.0 + 0.5 * np.sin(np.pi * y / L),
            desc="a = b  ==> b/a = 1, Proposition 1 predicts c = 0"),
}


def run_one(name, case, Ny, Ly=1.0, kx=2.0 * np.pi):
    y, D = cgl_grid(Ny, Ly)
    w = cc_weights(Ny, Ly)
    a = case["a"](y, Ly)
    b = case["b"](y, Ly)

    # Assembled L and R, interior-restricted (Dirichlet walls).
    L_full = -D @ (np.diag(a) @ D) + kx**2 * np.diag(a)
    R_full = kx**2 * np.diag(b)
    intr = slice(1, Ny - 1)
    L_int = L_full[intr, intr]
    R_int = R_full[intr, intr]
    w_int = w[intr]
    a_int = a[intr]
    b_int = b[intr]

    # M_asm = L^{-1} R  ( correct operator )
    M_asm = scipy.linalg.solve(L_int, R_int)
    # M_prim = diag(1/a) R = diag(k^2 b/a)  ( primitive-node surrogate )
    M_prim = np.diag(kx**2 * b_int / a_int)

    # Top eigenvector of M_asm (largest |eigenvalue|).
    lam, V = scipy.linalg.eig(M_asm)
    lam = np.real(lam)
    V = np.real(V)
    idx = int(np.argmax(np.abs(lam)))
    lam_n = lam[idx]
    Vn = V[:, idx]
    Vn = Vn / np.sqrt(np.sum(Vn**2 * w_int))  # normalise w.r.t. CC norm

    # Raw relative gap (Proposition-1 form, eq 5.1):
    #   || (M_prim - M_asm) V_n ||_w  /  || M_asm V_n ||_w
    # which measures both scale mismatch and shape mismatch together.
    residual = M_prim @ Vn - M_asm @ Vn
    num_w = np.sqrt(np.sum(residual**2 * w_int))
    den_w = np.sqrt(np.sum((M_asm @ Vn)**2 * w_int))
    rel_gap = num_w / den_w if den_w > 0 else float('nan')

    # Shape-only mismatch: project out the V_n component of M_prim V_n,
    # measure the remainder. This is zero iff M_prim V_n is a scalar
    # multiple of V_n (i.e., V_n is also an eigenvector of M_prim), which
    # is the "same discrete evolution" criterion.
    Mp_Vn = M_prim @ Vn
    # CC-weighted inner product
    inner = np.sum(Mp_Vn * Vn * w_int)
    Vn_norm2 = np.sum(Vn * Vn * w_int)
    proj_coeff = inner / Vn_norm2
    orth = Mp_Vn - proj_coeff * Vn
    shape_gap_abs = np.sqrt(np.sum(orth**2 * w_int))
    shape_gap_rel = shape_gap_abs / abs(lam_n) / np.sqrt(Vn_norm2) if abs(lam_n) > 0 else float('nan')

    # Proposition-1 prediction:  rel_gap² = E_μ[(k²b/a − λ_n)²] / λ_n²
    # (second-moment form; (5.1b) of the paper is the variance limit when
    # mean(k²b/a) ≈ λ_n, which holds exactly for the stellar generalised
    # EVP weighted inner product but only approximately for CC weights
    # on a generic (a,b) pair).
    mu = (Vn**2) * w_int
    mu = mu / np.sum(mu)
    k2_ba = kx**2 * b_int / a_int
    mean_k2ba = np.sum(k2_ba * mu)
    var_k2ba = np.sum((k2_ba - mean_k2ba)**2 * mu)
    second_moment = np.sum((k2_ba - lam_n)**2 * mu)
    c_variance = np.sqrt(var_k2ba) / abs(lam_n) if abs(lam_n) > 0 else float('nan')
    c_predicted = np.sqrt(second_moment) / abs(lam_n) if abs(lam_n) > 0 else float('nan')

    return dict(
        case=name, Ny=Ny,
        lam_n=lam_n,
        rel_gap_measured=rel_gap,
        shape_gap_rel=shape_gap_rel,
        c_predicted=c_predicted,
        c_variance=c_variance,
        a_prime_max=float(np.max(np.abs(np.gradient(a, y)))),
        ba_variation=float(np.max(b / a) - np.min(b / a)),
    )


def main():
    print("E1: Proposition 1 in a non-stellar variable-coefficient setting")
    print("    Generalised EVP  L u = lambda R u")
    print("    L = -d/dy(a du/dy) + k^2 a u,  R = k^2 b u")
    print("    M_prim = diag(1/a)·R = diag(k^2 b/a)  vs  M_asm = L^{-1} R")
    print("=" * 78)

    all_rows = []
    Ny_list = [32, 48, 64, 96, 128, 192, 256]
    kx = 2.0 * np.pi

    for name, case in CASES.items():
        print(f"\n  [{name}]  {case['desc']}")
        print(f"  {'Ny':>4s}  {'|lam_n|':>10s}  "
              f"{'raw gap':>12s}  {'shape gap':>12s}  "
              f"{'c variance':>12s}  {'b/a spread':>10s}")
        print("  " + "-" * 76)
        for Ny in Ny_list:
            r = run_one(name, case, Ny, kx=kx)
            all_rows.append(r)
            print(f"  {Ny:4d}  {abs(r['lam_n']):10.3e}  "
                  f"{r['rel_gap_measured']:12.4e}  "
                  f"{r['shape_gap_rel']:12.4e}  "
                  f"{r['c_variance']:12.4e}  "
                  f"{r['ba_variation']:10.3e}")

    csv_path = OUT_DIR / "poisson_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    print("\n" + "=" * 78)
    print("  Expected (Proposition 1 generalised):")
    print("    - A/B/C (b/a varying): rel_gap and c_predicted agree, both N_y-independent")
    print("    - D (b/a = 1):          rel_gap ≈ 0, c_predicted ≈ 0 (up to round-off)")


if __name__ == "__main__":
    main()
