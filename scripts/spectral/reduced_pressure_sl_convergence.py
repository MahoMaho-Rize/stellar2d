#!/usr/bin/env python3
"""
Convergence comparison: original vs reduced-pressure Liouville formulation.

This script performs the same Phase 0 manufactured-solution experiment for
BOTH formulations and plots the convergence order side by side.

  Original:         div(rho^{-1} grad p) = f,   sub p = sqrt(rho) * q
  Reduced pressure:  div(rho * grad pi)  = f~,   sub pi = (1/sqrt(rho)) * q

For Lane-Emden n=3/2, the Liouville potentials are:
  W_orig  = rho''/(2 rho) - 3(rho')^2 / (4 rho^2)  ~  -21/16 / t^2
  W_reduc = rho''/(2 rho) -   (rho')^2 / (4 rho^2)  ~  +3/16  / t^2

Usage:
  python scripts/reduced_pressure_sl_convergence.py
"""
import os
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


# ── Lane-Emden solver (reused from phase0) ──────────────────────────────
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
    """W = rho''/(2 rho) - 3(rho')^2/(4 rho^2)"""
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    return d2rho / (2.0 * rho) - 3.0 * drho ** 2 / (4.0 * rho ** 2)


def compute_W_reduced(y, rho):
    """W_tilde = rho''/(2 rho) - (rho')^2/(4 rho^2)"""
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    return d2rho / (2.0 * rho) - drho ** 2 / (4.0 * rho ** 2)


# ── SL eigensolver ───────────────────────────────────────────────────────
def solve_sl_eigenpairs(y_uniform, W_uniform, n_modes):
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


# ── Manufactured solution + SL Poisson solve ─────────────────────────────
def q_space_convergence(y, W, n_modes_list, label):
    """Direct q-space convergence test.

    Bypasses the manufactured-solution / FFT pipeline entirely.
    Constructs q_exact = sin(pi*(y-y0)/L) (Dirichlet), computes
    g = q'' + W*q - kx^2*q analytically, solves via SL expansion,
    and measures the reconstruction error.
    """
    Ny = len(y)
    dy = y[1] - y[0]
    y0, L = y[0], y[-1] - y[0]
    kx = 2.0 * np.pi * 2  # single horizontal mode

    q_exact = np.sin(np.pi * (y - y0) / L)
    d2q_exact = -(np.pi / L) ** 2 * q_exact
    g = d2q_exact + W * q_exact - kx ** 2 * q_exact

    n_max = max(n_modes_list)
    mu, psi = solve_sl_eigenpairs(y, W, min(n_max, Ny - 3))

    errs = []
    for Nm in n_modes_list:
        if Nm > psi.shape[1]:
            continue
        G = psi[:, :Nm].T @ g * dy
        Q = -G / (mu[:Nm] + kx ** 2)
        q_rec = psi[:, :Nm] @ Q
        err = np.sqrt(np.mean((q_rec - q_exact) ** 2))
        errs.append((Nm, err))
        print(f"  [{label}] N_modes = {Nm:3d}  err_L2 = {err:.3e}")

    return errs


def fit_slope(errs):
    if len(errs) < 2:
        return 0.0
    x = np.log10([e[0] for e in errs])
    y = np.log10([e[1] for e in errs])
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    coeffs = np.polyfit(x[mask], y[mask], 1)
    return coeffs[0]


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" Convergence comparison: original vs reduced-pressure Liouville")
    print("=" * 70)

    xi, theta, xi_1 = solve_lane_emden(n=1.5)
    r_norm = xi / xi_1
    rho = np.abs(theta) ** 1.5

    rho_cutoffs = [0.1, 0.01, 0.001]
    n_modes_list = [5, 10, 20, 40, 80, 160, 256]
    Ny_fd = 512

    results = {}

    for rho_cut in rho_cutoffs:
        mask = rho > rho_cut
        r_in = r_norm[mask]
        rho_in = rho[mask]

        W_orig_raw = compute_W_original(r_in, rho_in)
        W_redu_raw = compute_W_reduced(r_in, rho_in)

        print(f"\n--- rho_cut = {rho_cut} | domain r in [{r_in[0]:.3f}, {r_in[-1]:.3f}] ---")
        print(f"  W_orig range: [{W_orig_raw.min():.3e}, {W_orig_raw.max():.3e}]")
        print(f"  W_redu range: [{W_redu_raw.min():.3e}, {W_redu_raw.max():.3e}]")

        y_u = np.linspace(r_in[0], r_in[-1], Ny_fd)
        W_orig_u = np.interp(y_u, r_in, W_orig_raw)
        W_redu_u = np.interp(y_u, r_in, W_redu_raw)

        errs_orig = q_space_convergence(
            y_u, W_orig_u, n_modes_list, "original")

        errs_redu = q_space_convergence(
            y_u, W_redu_u, n_modes_list, "reduced-p")

        slope_orig = fit_slope(errs_orig)
        slope_redu = fit_slope(errs_redu)
        print(f"  Slope original:  {slope_orig:.2f}")
        print(f"  Slope reduced-p: {slope_redu:.2f}")

        results[rho_cut] = {
            "orig": errs_orig, "redu": errs_redu,
            "slope_orig": slope_orig, "slope_redu": slope_redu,
        }

    # --- Smooth Gaussian rho control ---
    print(f"\n--- Smooth Gaussian rho (control, no singularity) ---")
    y_smooth = np.linspace(0, 1, Ny_fd)
    rho_smooth = 1.0 + 0.5 * np.exp(-((y_smooth - 0.5) / 0.15) ** 2)

    W_orig_sm = compute_W_original(y_smooth, rho_smooth)
    W_redu_sm = compute_W_reduced(y_smooth, rho_smooth)

    errs_orig_sm = q_space_convergence(
        y_smooth, W_orig_sm, n_modes_list, "orig-smooth")

    errs_redu_sm = q_space_convergence(
        y_smooth, W_redu_sm, n_modes_list, "redu-smooth")

    slope_orig_sm = fit_slope(errs_orig_sm)
    slope_redu_sm = fit_slope(errs_redu_sm)
    print(f"  Slope original (smooth):  {slope_orig_sm:.2f}")
    print(f"  Slope reduced-p (smooth): {slope_redu_sm:.2f}")

    results["smooth"] = {
        "orig": errs_orig_sm, "redu": errs_redu_sm,
        "slope_orig": slope_orig_sm, "slope_redu": slope_redu_sm,
    }

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=140)

    plot_keys = list(rho_cutoffs) + ["smooth"]
    titles = [rf"$\rho_{{cut}}$ = {c}" for c in rho_cutoffs] + ["Smooth Gaussian (control)"]

    for ax, key, title in zip(axes.flat, plot_keys, titles):
        res = results[key]
        e_o = res["orig"]
        e_r = res["redu"]

        if e_o:
            xs, ys = zip(*e_o)
            ax.loglog(xs, ys, "o-", color="C3", lw=1.8, ms=5,
                      label=f"original (slope {res['slope_orig']:.2f})")
        if e_r:
            xs, ys = zip(*e_r)
            ax.loglog(xs, ys, "s-", color="C0", lw=1.8, ms=5,
                      label=f"reduced-p (slope {res['slope_redu']:.2f})")

        ax.set_xlabel("N modes")
        ax.set_ylabel(r"$\mathrm{err}_{L^2}$")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("SL-Poisson convergence: original vs reduced-pressure Liouville",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = VID / "reduced_pressure_convergence.png"
    fig.savefig(out_path)
    print(f"\n  => {out_path}")
    plt.close(fig)

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25s}  {'err_orig(256)':>14s}  {'err_redu(256)':>14s}  {'slope_orig':>11s}  {'slope_redu':>11s}  {'improvement':>12s}")
    print("-" * 95)
    for key in plot_keys:
        res = results[key]
        e_o_256 = [e for n, e in res["orig"] if n == 256]
        e_r_256 = [e for n, e in res["redu"] if n == 256]
        eo = e_o_256[0] if e_o_256 else float("nan")
        er = e_r_256[0] if e_r_256 else float("nan")
        imp = eo / er if er > 0 else float("nan")
        label = f"rho_cut={key}" if isinstance(key, float) else key
        print(f"{label:<25s}  {eo:14.3e}  {er:14.3e}  {res['slope_orig']:11.2f}  {res['slope_redu']:11.2f}  {imp:12.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
