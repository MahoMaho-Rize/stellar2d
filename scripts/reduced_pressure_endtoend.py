#!/usr/bin/env python3
"""
Experiment B: End-to-end Poisson convergence for the reduced-pressure Liouville
formulation vs the original (1/rho grad p) formulation.

Scope difference vs `reduced_pressure_sl_convergence.py`:
  - That script tests ONLY the q-space SL expansion step (bypasses weighting,
    RHS construction, and variable recovery) in isolation.
  - This script runs the FULL pipeline end-to-end:
        manufactured physical state  ->  analytic RHS
            ->  weight by rho^{+-1/2}  ->  SL expansion (Nm modes)
                ->  solve in spectral space  ->  unweight
                    ->  recover (pi or p)  ->  compare to exact

Manufactured state (single k_x; x-dependence factored out, since the full 2D
FFT step is linear & exact):
    pi_exact(x, y)  =  cos(k_x * x) * sin(pi * (y - y0) / L)
    p_exact(x, y)   =  rho0(y) * pi_exact(x, y)
Both vanish at y = y0, y0 + L (Dirichlet), periodic in x.

For a single horizontal mode k_x the y-reduction gives:
    reduced-p:  d/dy[rho * dpi/dy] - k_x^2 * rho * pi = f_tilde
    original:   d/dy[(1/rho) * dp/dy] - k_x^2 * (1/rho) * p = f_orig

RHS computed analytically from pi_exact / p_exact + rho, then passed through
each solver to measure the final L2 error on the recovered field.

Usage:
  python scripts/reduced_pressure_endtoend.py
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
VID = REPO / "videos"
VID.mkdir(exist_ok=True)

# ── Reference values bound to docs/reduced_pressure_experiments_2026-05-02.md ──
# Keys: (rho_cut, N_modes) -> (err_orig, err_redu).  Any drift >REL_TOL in
# --verify mode exits nonzero and tells you which row changed.
EXPECTED = {
    (0.1,   20): (1.742e-4, 1.746e-4),
    (0.01,  20): (5.673e-4, 5.891e-4),
    (0.001, 20): (1.114e-3, 1.567e-3),
    (0.1,   80): (6.245e-6, 6.247e-6),
    (0.01,  80): (2.372e-5, 2.380e-5),
    (0.001, 80): (7.854e-5, 8.265e-5),
}
REL_TOL = 0.02   # 2% relative tolerance

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


# ── Shared infrastructure (matched to reduced_pressure_sl_convergence.py) ──
def solve_lane_emden(n=1.5, xi_max=10.0, n_pts=5000):
    def rhs(xi, y):
        theta, dtheta = y
        if xi < 1e-10:
            return [dtheta, -theta ** n / 3.0]
        theta_pow = np.sign(theta) * np.abs(theta) ** n if theta >= 0 else 0.0
        return [dtheta, -2.0 / xi * dtheta - theta_pow]

    def event_zero(xi, y):
        return y[0]
    event_zero.terminal = True
    event_zero.direction = -1

    sol = scipy.integrate.solve_ivp(
        rhs, [1e-6, xi_max], [1.0 - 1e-12, 0.0],
        events=event_zero, max_step=0.01, rtol=1e-10, atol=1e-12,
        dense_output=True,
    )
    xi_1 = sol.t_events[0][0] if sol.t_events[0].size else sol.t[-1]
    xi = np.linspace(1e-5, xi_1 * 0.999, n_pts)
    theta = sol.sol(xi)[0]
    return xi, theta, xi_1


def compute_W_original(y, rho):
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    return d2rho / (2.0 * rho) - 3.0 * drho ** 2 / (4.0 * rho ** 2)


def compute_W_reduced(y, rho):
    # Correct formula (derived by direct substitution pi = rho^{-1/2} q):
    #   d/dy[rho * dpi/dy] = sqrt(rho) * [q'' + W_tilde * q]
    # where W_tilde = (rho')^2/(4 rho^2) - rho''/(2 rho).
    # Signs flipped relative to the original draft of docs/reduced_pressure_liouville.md
    # eq (12); sanity check for Lane-Emden n=3/2 (rho = c * t^{3/2}):
    #   rho''/(2 rho) = 3/(8 t^2),  (rho')^2/(4 rho^2) = 9/(16 t^2)
    #   W_tilde = 9/(16 t^2) - 6/(16 t^2) = +3/(16 t^2)   (repulsive, consistent with
    #   the symbolic-verified result in eq (13)).
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    return drho ** 2 / (4.0 * rho ** 2) - d2rho / (2.0 * rho)


def solve_sl_eigenpairs(y_uniform, W_uniform, n_modes):
    """Finite-difference SL eigenvalue solver (Dirichlet BC).

    Tu = -u'' - W u  acts on interior nodes; eigpairs (mu_n, psi_n) with
    sum(psi_n^2) * dy = 1 (L2-orthonormal in the FD quadrature).
    """
    N = len(y_uniform)
    dy = y_uniform[1] - y_uniform[0]
    M = N - 2
    W_int = W_uniform[1:-1]
    main = 2.0 / dy ** 2 - W_int
    off = -np.ones(M - 1) / dy ** 2
    A = scipy.sparse.diags([off, main, off], [-1, 0, 1], format="csr")
    mu, psi_int = scipy.sparse.linalg.eigsh(A, k=n_modes, which="SA")
    psi = np.zeros((N, n_modes))
    psi[1:-1, :] = psi_int
    for i in range(psi.shape[1]):
        norm = np.sqrt(np.sum(psi[:, i] ** 2) * dy)
        psi[:, i] /= norm
    order = np.argsort(mu)
    return mu[order], psi[:, order]


# ── Manufactured solution + analytic RHS ─────────────────────────────────
def manufactured_pi(y, y0, L):
    """pi_exact(y) for a single k_x Fourier mode (x-factor applied later).

    Smooth, Dirichlet at y = y0 and y = y0 + L.
    """
    return np.sin(np.pi * (y - y0) / L)


def d2_manufactured_pi(y, y0, L):
    return -(np.pi / L) ** 2 * np.sin(np.pi * (y - y0) / L)


def rhs_reduced(y, rho, drho, pi_ex, dpi_ex, d2pi_ex, k_x):
    """
    f_tilde = d/dy[rho dpi/dy] - k_x^2 * rho * pi
            = drho * dpi + rho * d2pi - k_x^2 * rho * pi
    """
    return drho * dpi_ex + rho * d2pi_ex - k_x ** 2 * rho * pi_ex


def rhs_original(y, rho, drho, p_ex, dp_ex, d2p_ex, k_x):
    """
    f_orig = d/dy[(1/rho) dp/dy] - k_x^2 (1/rho) p
           = -(drho/rho^2) * dp + (1/rho) * d2p - k_x^2 * (1/rho) * p
    """
    return (-drho / rho ** 2) * dp_ex + (1.0 / rho) * d2p_ex - k_x ** 2 * (1.0 / rho) * p_ex


# ── End-to-end solver for each formulation ───────────────────────────────
def solve_reduced_pressure(y, rho, f_tilde, k_x, mu, psi):
    """Full pipeline for reduced-pressure form.

    pi_hat satisfies  d/dy[rho * dpi_hat/dy] - k_x^2 * rho * pi_hat = f_tilde
    Substitution:       pi_hat = rho^{-1/2} * q
    Yields:              q'' + W_tilde*q - k_x^2*q = g,  g = rho^{-1/2} * f_tilde

    Return pi_rec on the grid y.
    """
    dy = y[1] - y[0]
    g = f_tilde / np.sqrt(rho)
    Nm = psi.shape[1]
    G = psi.T @ g * dy                   # forward SL transform
    Q = -G / (mu + k_x ** 2)             # diagonal solve in spectral space
    q_rec = psi @ Q
    return q_rec / np.sqrt(rho)          # unweight -> pi


def solve_original(y, rho, f_orig, k_x, mu, psi):
    """Full pipeline for original form.

    p_hat satisfies   d/dy[(1/rho) * dp_hat/dy] - k_x^2 * (1/rho) * p_hat = f_orig
    Substitution:     p_hat = rho^{+1/2} * q
    Yields:           q'' + W_orig*q - k_x^2*q = g,  g = rho^{+1/2} * f_orig
    Return p_rec.
    """
    dy = y[1] - y[0]
    g = np.sqrt(rho) * f_orig
    Nm = psi.shape[1]
    G = psi.T @ g * dy
    Q = -G / (mu + k_x ** 2)
    q_rec = psi @ Q
    return np.sqrt(rho) * q_rec


def fit_slope(errs):
    if len(errs) < 2:
        return 0.0
    x = np.log10([e[0] for e in errs])
    y = np.log10([e[1] for e in errs])
    mask = np.isfinite(y) & (np.array([e[1] for e in errs]) > 0)
    if mask.sum() < 2:
        return 0.0
    coeffs = np.polyfit(x[mask], y[mask], 1)
    return coeffs[0]


# ── Main ─────────────────────────────────────────────────────────────────
def main(verify=False):
    print_provenance("scripts/reduced_pressure_endtoend.py")
    print(" Experiment B: end-to-end Poisson convergence")
    print("   pi_exact = sin(pi (y-y0)/L),  p_exact = rho(y) * pi_exact")
    print("=" * 72)

    xi, theta, xi_1 = solve_lane_emden(n=1.5)
    r_norm = xi / xi_1
    rho = np.abs(theta) ** 1.5

    Ny_fd = 512
    n_modes_list = [5, 10, 20, 40, 80, 160, 256]
    rho_cutoffs = [0.1, 0.01, 0.001]
    k_x = 2.0 * np.pi * 2            # single horizontal mode

    results = {}

    for rho_cut in rho_cutoffs:
        mask = rho > rho_cut
        r_in = r_norm[mask]
        rho_in = rho[mask]

        y = np.linspace(r_in[0], r_in[-1], Ny_fd)
        rho_y = np.interp(y, r_in, rho_in)
        drho_y = np.gradient(rho_y, y, edge_order=2)
        y0, L = y[0], y[-1] - y[0]

        W_orig = compute_W_original(y, rho_y)
        W_redu = compute_W_reduced(y, rho_y)

        pi_ex = manufactured_pi(y, y0, L)
        dpi_ex = (np.pi / L) * np.cos(np.pi * (y - y0) / L)
        d2pi_ex = d2_manufactured_pi(y, y0, L)

        # original form: p_exact = rho * pi_exact
        p_ex = rho_y * pi_ex
        dp_ex = drho_y * pi_ex + rho_y * dpi_ex
        # d2(rho*pi) = d2rho*pi + 2*drho*dpi + rho*d2pi
        d2rho_y = np.gradient(drho_y, y, edge_order=2)
        d2p_ex = d2rho_y * pi_ex + 2.0 * drho_y * dpi_ex + rho_y * d2pi_ex

        f_tilde = rhs_reduced(y, rho_y, drho_y, pi_ex, dpi_ex, d2pi_ex, k_x)
        f_orig = rhs_original(y, rho_y, drho_y, p_ex, dp_ex, d2p_ex, k_x)

        print(f"\n--- rho_cut = {rho_cut} | y in [{y0:.3f}, {y[-1]:.3f}] ---")
        print(f"  rho range: [{rho_y.min():.3e}, {rho_y.max():.3e}]")
        print(f"  |f_tilde|_inf = {np.max(np.abs(f_tilde)):.3e}")
        print(f"  |f_orig|_inf  = {np.max(np.abs(f_orig)):.3e}")

        # Eigenpairs for both potentials (build once, reuse for all N_m)
        n_max = max(n_modes_list)
        mu_o, psi_o = solve_sl_eigenpairs(y, W_orig, min(n_max, Ny_fd - 3))
        mu_r, psi_r = solve_sl_eigenpairs(y, W_redu, min(n_max, Ny_fd - 3))

        errs_orig = []
        errs_redu = []
        for Nm in n_modes_list:
            if Nm > psi_o.shape[1]:
                continue

            # Original form: recover p -> divide by rho to get pi_recovered
            p_rec = solve_original(y, rho_y, f_orig, k_x,
                                    mu_o[:Nm], psi_o[:, :Nm])
            pi_from_p = p_rec / rho_y
            err_o = np.sqrt(np.mean((pi_from_p - pi_ex) ** 2))
            errs_orig.append((Nm, err_o))

            # Reduced-p form: recover pi directly
            pi_rec = solve_reduced_pressure(y, rho_y, f_tilde, k_x,
                                              mu_r[:Nm], psi_r[:, :Nm])
            err_r = np.sqrt(np.mean((pi_rec - pi_ex) ** 2))
            errs_redu.append((Nm, err_r))

            ratio = err_o / err_r if err_r > 0 else float('inf')
            print(f"  N_m={Nm:3d}  err_orig(pi)={err_o:.3e}  err_redu(pi)={err_r:.3e}  ratio={ratio:6.1f}x")

        results[rho_cut] = {
            "orig": errs_orig, "redu": errs_redu,
            "slope_orig": fit_slope(errs_orig),
            "slope_redu": fit_slope(errs_redu),
        }
        print(f"  slope_orig = {results[rho_cut]['slope_orig']:.2f}  "
              f"slope_redu = {results[rho_cut]['slope_redu']:.2f}")

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    for ax, cut in zip(axes, rho_cutoffs):
        res = results[cut]
        if res["orig"]:
            xs, ys = zip(*res["orig"])
            ax.loglog(xs, ys, "o-", color="C3", lw=1.8, ms=6,
                      label=f"original (slope {res['slope_orig']:.2f})")
        if res["redu"]:
            xs, ys = zip(*res["redu"])
            ax.loglog(xs, ys, "s-", color="C0", lw=1.8, ms=6,
                      label=f"reduced-p (slope {res['slope_redu']:.2f})")
        ax.set_xlabel("N modes")
        ax.set_ylabel(r"$\|\pi_{\mathrm{rec}} - \pi_{\mathrm{exact}}\|_{L^2}$")
        ax.set_title(fr"$\rho_{{\mathrm{{cut}}}}$ = {cut}")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Experiment B: end-to-end Poisson — original vs reduced-pressure",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path = VID / "reduced_pressure_endtoend.png"
    fig.savefig(out_path)
    print(f"\n  => {out_path}")
    plt.close(fig)

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print(" SUMMARY (end-to-end Poisson, err measured on pi)")
    print("=" * 85)
    print(f"{'rho_cut':<10}  {'err_orig(20)':>14}  {'err_redu(20)':>14}  {'ratio@20':>10}  {'slope_o':>8}  {'slope_r':>8}")
    print("-" * 85)
    for cut in rho_cutoffs:
        res = results[cut]
        e_o_20 = [e for n, e in res["orig"] if n == 20]
        e_r_20 = [e for n, e in res["redu"] if n == 20]
        eo = e_o_20[0] if e_o_20 else float("nan")
        er = e_r_20[0] if e_r_20 else float("nan")
        ratio = eo / er if er > 0 else float("nan")
        print(f"{cut:<10.3g}  {eo:14.3e}  {er:14.3e}  {ratio:10.1f}x  {res['slope_orig']:8.2f}  {res['slope_redu']:8.2f}")
    print("=" * 85)

    # ── Verify against EXPECTED ──────────────────────────────────────────
    if verify:
        print("\n--- VERIFY against EXPECTED (see docs/reduced_pressure_experiments_2026-05-02.md) ---")
        n_ok = 0
        n_fail = 0
        for (cut, Nm), (eo_ref, er_ref) in EXPECTED.items():
            res = results.get(cut)
            if not res:
                continue
            eo_hit = [e for n, e in res["orig"] if n == Nm]
            er_hit = [e for n, e in res["redu"] if n == Nm]
            if not eo_hit or not er_hit:
                continue
            eo, er = eo_hit[0], er_hit[0]
            do = abs(eo - eo_ref) / max(abs(eo_ref), 1e-300)
            dr = abs(er - er_ref) / max(abs(er_ref), 1e-300)
            ok = do < REL_TOL and dr < REL_TOL
            mark = "OK" if ok else "DRIFT"
            print(f"  [{mark:<5}] rho_cut={cut:<6} N={Nm:3d}  "
                  f"orig {eo:.3e} vs {eo_ref:.3e} ({do*100:.2f}%)  "
                  f"redu {er:.3e} vs {er_ref:.3e} ({dr*100:.2f}%)")
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        print(f"\n  {n_ok} passed, {n_fail} drifted (tol={REL_TOL*100:.0f}%)")
        if n_fail:
            sys.exit(1)
        print("  all reference values reproduced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="compare results against EXPECTED and exit nonzero on drift")
    args = ap.parse_args()
    main(verify=args.verify)
