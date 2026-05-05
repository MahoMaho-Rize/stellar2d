#!/usr/bin/env python3
"""
Experiment A: Chebyshev collocation to break the FD eigenvalue floor.

The FD-based SL eigensolver used in Phase 0 and in
`reduced_pressure_sl_convergence.py` has 2nd-order accuracy and caps the
achievable L2 error at ~1e-7 regardless of how many SL modes are kept.
That floor masks the true convergence behaviour of the two Liouville
potentials (original vs reduced-pressure) beyond ~N=80 modes.

This script rebuilds the SL eigenvalue problem on Chebyshev-Gauss-Lobatto
(CGL) nodes using the Trefethen (SMMMLAB) differentiation matrices. Spectral
accuracy lets us see whether the reduced-pressure form eventually separates
from the original at high N, or whether both asymptote to machine precision.

Notes:
  - The SL problem T psi = -mu psi is discretised as
        (D2 + diag(W)) psi = -mu psi  on INTERIOR CGL points (Dirichlet).
  - Eigenfunctions are L2-normalised using the Clenshaw-Curtis quadrature
    weights to preserve Parseval.
  - The q-space manufactured test is reused as-is from
    reduced_pressure_sl_convergence.py so results are directly comparable.

Usage:
  python scripts/reduced_pressure_chebyshev.py
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.integrate
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
VID = REPO / "videos"
VID.mkdir(exist_ok=True)

# ── Reference values bound to docs/reduced_pressure_experiments_2026-05-02.md ──
# Keys: N_Cheb -> (err_orig @ max N_modes, err_redu @ max N_modes).
# Experiment A reports q-norm errors at rho_cut=0.01, N_modes=320 (except N=64
# which caps at N_modes=40 for stability).
EXPECTED = {
    64:  (5.098e-6,  5.383e-7),
    128: (2.648e-7,  2.945e-8),
    256: (1.240e-8,  1.401e-9),
    512: (5.584e-10, 6.335e-11),
}
# Chebyshev converges much faster than FD; small drift floors are OK, but
# N_Cheb=512 at ~6e-11 is near the float64 limit, so we relax tol for that row.
REL_TOL_DEFAULT = 0.05
REL_TOL_HIGH = 0.25   # for N_Cheb=512 row (double-precision noise floor)
REF_DOC = "docs/reduced_pressure_experiments_2026-05-02.md"


def git_head():
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def print_provenance(script_name):
    print("#" * 72)
    print(f"#  {script_name}")
    print(f"#  git HEAD: {git_head()}    date: {datetime.date.today().isoformat()}")
    print(f"#  reference doc: {REF_DOC}")
    print("#" * 72)


# Shared infrastructure (Lane-Emden + Chebyshev machinery) lives in
# gmode_infra.py to maintain a single source of truth across the
# reduced-pressure and g-mode experiment suites.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmode_infra import (
    solve_lane_emden,
    cheb,
    clenshaw_curtis_weights,
    cheb_on_interval,
)


# ── W potentials on a generic (possibly non-uniform) grid ───────────────
def W_from_rho(y_query, rho_fn, drho_fn, d2rho_fn, coeff_d2rho, coeff_drho2):
    """W = coeff_d2rho * rho''/(2 rho) + coeff_drho2 * (rho')^2 / rho^2.

    Original form (pi_hat = sqrt(rho) * q on op div(1/rho grad p)):
        W_orig = + rho''/(2 rho) - 3 (rho')^2 / (4 rho^2)
        coeff_d2rho = +1,  coeff_drho2 = -3/4
    Reduced-pressure form (pi = rho^{-1/2} q on op div(rho grad pi)):
        W_redu = - rho''/(2 rho) +   (rho')^2 / (4 rho^2)
        coeff_d2rho = -1,  coeff_drho2 = +1/4
    The two terms carry INDEPENDENT signs; conflating them with a single
    coefficient was a bug in earlier versions of this function (reported 2026-05-02).
    """
    rho = rho_fn(y_query)
    drho = drho_fn(y_query)
    d2rho = d2rho_fn(y_query)
    return coeff_d2rho * d2rho / (2.0 * rho) + coeff_drho2 * drho ** 2 / (rho ** 2)


# ── SL eigenvalue solve (Chebyshev) ─────────────────────────────────────
def sl_eig_chebyshev(y, D2, w_full, W_vals, n_modes):
    """
    Solve (-D2 - diag(W)) psi = mu psi with homogeneous Dirichlet BCs.

    interior-only formulation:
        A = -D2_interior - diag(W_interior);  eig(A)
    eigenvectors extended by zero at the endpoints, then normalised in the
    Clenshaw-Curtis (weighted) L2 inner product so that sum(w * psi_i * psi_j)
    = delta_ij. Returns mu (ascending), psi (N+1, n_modes), w_full.
    """
    N = len(y) - 1
    inner = slice(1, N)
    A = -D2[inner, inner] - np.diag(W_vals[inner])
    mu, V = np.linalg.eig(A)
    order = np.argsort(mu.real)
    mu = mu.real[order]
    V = V.real[:, order]
    psi = np.zeros((N + 1, mu.size))
    psi[inner, :] = V
    # L2-normalise with CC weights
    for i in range(psi.shape[1]):
        norm = np.sqrt(np.sum(w_full * psi[:, i] ** 2))
        if norm > 0:
            psi[:, i] /= norm
    return mu[:n_modes], psi[:, :n_modes]


# ── Manufactured q-space test (analogue of the sin-based test) ──────────
def q_space_convergence_cheb(y, D2, w_full, W_vals, n_modes_list, label):
    """Matches the test in reduced_pressure_sl_convergence.py but with
    Chebyshev collocation: q_exact = sin(pi (y-y0)/L), compute g analytically,
    solve via SL expansion in the Chebyshev basis.
    """
    y0, L = y[0], y[-1] - y[0]
    kx = 2.0 * np.pi * 2  # same as the FD test
    q_exact = np.sin(np.pi * (y - y0) / L)
    d2q_exact = -(np.pi / L) ** 2 * q_exact
    g = d2q_exact + W_vals * q_exact - kx ** 2 * q_exact

    n_max = max(n_modes_list)
    # eigensolve needs n_max <= (N-1) interior points
    N_interior = len(y) - 2
    n_cap = min(n_max, N_interior - 1)
    mu, psi = sl_eig_chebyshev(y, D2, w_full, W_vals, n_cap)

    errs = []
    for Nm in n_modes_list:
        if Nm > psi.shape[1]:
            continue
        # forward SL transform using Clenshaw-Curtis inner product
        G = (psi[:, :Nm] * w_full[:, None]).T @ g
        Q = -G / (mu[:Nm] + kx ** 2)
        q_rec = psi[:, :Nm] @ Q
        # L2 error with CC quadrature
        err = np.sqrt(np.sum(w_full * (q_rec - q_exact) ** 2) / np.sum(w_full))
        errs.append((Nm, err))
        print(f"  [{label}]  N={Nm:3d}  err={err:.3e}")
    return errs


def fit_slope(errs):
    if len(errs) < 2:
        return 0.0
    x = np.log10([e[0] for e in errs])
    y = np.log10([max(e[1], 1e-300) for e in errs])
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    return float(np.polyfit(x[mask], y[mask], 1)[0])


# ── Main ────────────────────────────────────────────────────────────────
def main(verify=False):
    print_provenance("scripts/reduced_pressure_chebyshev.py")
    print(" Experiment A: Chebyshev collocation SL solver")
    print("=" * 72)

    # Lane-Emden rho(r) on a fine ODE grid, then wrap as interpolators
    xi, theta, xi_1 = solve_lane_emden(n=1.5)
    r_ode = xi / xi_1
    rho_ode = np.abs(theta) ** 1.5

    # Spline derivatives analytically via gradient on the fine ODE grid
    drho_ode = np.gradient(rho_ode, r_ode, edge_order=2)
    d2rho_ode = np.gradient(drho_ode, r_ode, edge_order=2)

    def make_interp(r, f):
        # Linear interpolation is enough — CGL grid sits inside [r[0], r[-1]].
        return lambda y: np.interp(y, r, f)

    rho_fn = make_interp(r_ode, rho_ode)
    drho_fn = make_interp(r_ode, drho_ode)
    d2rho_fn = make_interp(r_ode, d2rho_ode)

    # Match Phase 0: use rho_cut = 0.01 for primary comparison
    rho_cut = 0.01
    mask = rho_ode > rho_cut
    a, b = float(r_ode[mask][0]), float(r_ode[mask][-1])
    print(f"  Lane-Emden n=3/2,  rho_cut = {rho_cut},  domain [{a:.4f}, {b:.4f}]")

    # Chebyshev resolutions: push N up to 320 so we go past N_modes = 256
    N_cheb_list = [64, 128, 256, 512]
    n_modes_list = [5, 10, 20, 40, 80, 160, 320]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    all_results = {}

    for N_cheb in N_cheb_list:
        y, D2, w_full = cheb_on_interval(N_cheb, a, b)
        W_orig_vals = W_from_rho(y, rho_fn, drho_fn, d2rho_fn,
                                  coeff_d2rho=+1.0, coeff_drho2=-3.0 / 4.0)
        W_redu_vals = W_from_rho(y, rho_fn, drho_fn, d2rho_fn,
                                  coeff_d2rho=-1.0, coeff_drho2=+1.0 / 4.0)

        print(f"\n--- N_Cheb = {N_cheb} (interior = {N_cheb - 1}) ---")
        print(f"  |W_orig|_inf = {np.max(np.abs(W_orig_vals)):.3e}")
        print(f"  |W_redu|_inf = {np.max(np.abs(W_redu_vals)):.3e}")

        errs_o = q_space_convergence_cheb(y, D2, w_full, W_orig_vals,
                                           n_modes_list, f"orig  N={N_cheb}")
        errs_r = q_space_convergence_cheb(y, D2, w_full, W_redu_vals,
                                           n_modes_list, f"redu  N={N_cheb}")
        so = fit_slope(errs_o)
        sr = fit_slope(errs_r)
        print(f"  slope orig = {so:.2f}, slope redu = {sr:.2f}")
        all_results[N_cheb] = (errs_o, errs_r, so, sr)

        # Plot: original (left), reduced-p (right)
        if errs_o:
            xs, ys = zip(*errs_o)
            axes[0].loglog(xs, ys, "o-", lw=1.3, ms=5,
                            label=f"$N_{{Cheb}}$={N_cheb} (slope {so:.2f})")
        if errs_r:
            xs, ys = zip(*errs_r)
            axes[1].loglog(xs, ys, "s-", lw=1.3, ms=5,
                            label=f"$N_{{Cheb}}$={N_cheb} (slope {sr:.2f})")

    for ax, title in zip(axes,
                          ["Original  $W = \\rho''/2\\rho - 3\\rho'^2/4\\rho^2$",
                           "Reduced-p  $\\widetilde W = \\rho'^2/4\\rho^2 - \\rho''/2\\rho$"]):
        ax.set_xlabel("N modes")
        ax.set_ylabel(r"$\|q_{\mathrm{rec}} - q_{\mathrm{exact}}\|_{L^2}$  (CC norm)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Experiment A: Chebyshev collocation, Lane-Emden rho_cut = {rho_cut}",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = VID / "reduced_pressure_chebyshev.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)

    # ── Summary: did the floor drop, and did the two forms separate? ─────
    print("\n" + "=" * 75)
    print(" SUMMARY  (Chebyshev collocation)")
    print("=" * 75)
    print(f"{'N_Cheb':>8}  {'err_orig(max Nm)':>18}  {'err_redu(max Nm)':>18}  {'ratio':>8}")
    print("-" * 75)
    for N_cheb in N_cheb_list:
        errs_o, errs_r, _, _ = all_results[N_cheb]
        eo = errs_o[-1][1] if errs_o else float("nan")
        er = errs_r[-1][1] if errs_r else float("nan")
        ratio = eo / er if er > 0 else float("nan")
        print(f"{N_cheb:>8}  {eo:18.3e}  {er:18.3e}  {ratio:8.2f}x")
    print("=" * 75)

    # ── Verify against EXPECTED ──────────────────────────────────────────
    if verify:
        print("\n--- VERIFY against EXPECTED (see docs/reduced_pressure_experiments_2026-05-02.md) ---")
        n_ok = 0
        n_fail = 0
        for N_cheb, (eo_ref, er_ref) in EXPECTED.items():
            if N_cheb not in all_results:
                continue
            errs_o, errs_r, _, _ = all_results[N_cheb]
            if not errs_o or not errs_r:
                continue
            eo = errs_o[-1][1]
            er = errs_r[-1][1]
            tol = REL_TOL_HIGH if N_cheb >= 512 else REL_TOL_DEFAULT
            do = abs(eo - eo_ref) / max(abs(eo_ref), 1e-300)
            dr = abs(er - er_ref) / max(abs(er_ref), 1e-300)
            ok = do < tol and dr < tol
            mark = "OK" if ok else "DRIFT"
            print(f"  [{mark:<5}] N_Cheb={N_cheb:>4}  "
                  f"orig {eo:.3e} vs {eo_ref:.3e} ({do*100:.2f}%)  "
                  f"redu {er:.3e} vs {er_ref:.3e} ({dr*100:.2f}%)  [tol {tol*100:.0f}%]")
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        print(f"\n  {n_ok} passed, {n_fail} drifted")
        if n_fail:
            sys.exit(1)
        print("  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="compare results against EXPECTED and exit nonzero on drift")
    args = ap.parse_args()
    main(verify=args.verify)
