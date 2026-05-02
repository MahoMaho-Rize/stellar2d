#!/usr/bin/env python3
"""
Experiment 2: Chebyshev collocation replaces FD for the SL eigenvalue problem.

The Phase 0 experiments showed both formulations converge to an FD floor
at ~1.4e-7 (512 grid points, 2nd-order stencil).  This floor masks the
true difference between original and reduced-pressure potentials.

This script solves the same SL eigenvalue problem using Chebyshev collocation,
which has spectral (exponential) convergence for smooth potentials and
much lower floor than FD.

Usage:
  python scripts/sl_chebyshev_convergence.py
"""
import os
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


# ── Lane-Emden ───────────────────────────────────────────────────────────
def solve_lane_emden(n=1.5, n_pts=5000):
    def rhs(xi, y):
        theta, dtheta = y
        if xi < 1e-10:
            return [dtheta, -theta ** n / 3.0]
        theta_pow = np.sign(theta) * np.abs(theta) ** n if theta >= 0 else 0.0
        return [dtheta, -2.0 / xi * dtheta - theta_pow]
    def ev(xi, y): return y[0]
    ev.terminal = True; ev.direction = -1
    sol = scipy.integrate.solve_ivp(
        rhs, [1e-6, 10.0], [1.0 - 1e-12, 0.0],
        events=ev, max_step=0.01, rtol=1e-10, atol=1e-12,
        dense_output=True)
    xi_1 = sol.t_events[0][0]
    xi = np.linspace(1e-5, xi_1 * 0.999, n_pts)
    theta = sol.sol(xi)[0]
    return xi / xi_1, np.abs(theta) ** 1.5, sol, xi_1


def lane_emden_rho_at(r_normalized, sol, xi_1):
    """Evaluate rho at arbitrary r/R points using the dense ODE solution."""
    xi = np.clip(r_normalized * xi_1, 1e-5, xi_1 * 0.999)
    theta = sol.sol(xi)[0]
    return np.abs(theta) ** 1.5


# ── W computation ────────────────────────────────────────────────────────
def compute_W(y, rho, form="original"):
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    if form == "original":
        return d2rho / (2.0 * rho) - 3.0 * drho ** 2 / (4.0 * rho ** 2)
    else:
        return d2rho / (2.0 * rho) - drho ** 2 / (4.0 * rho ** 2)


def compute_W_spectral(y, D1, rho, form="original"):
    """Compute W using Chebyshev differentiation matrix for derivatives."""
    drho = D1 @ rho
    d2rho = D1 @ drho
    if form == "original":
        return d2rho / (2.0 * rho) - 3.0 * drho ** 2 / (4.0 * rho ** 2)
    else:
        return d2rho / (2.0 * rho) - drho ** 2 / (4.0 * rho ** 2)


# ── Chebyshev differentiation matrix ────────────────────────────────────
def cheb_diffmat(N):
    """Chebyshev differentiation matrix on N+1 Gauss-Lobatto points in [-1,1]."""
    if N == 0:
        return np.array([[0.0]]), np.array([0.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2.0; c[-1] = 2.0
    c *= (-1.0) ** np.arange(N + 1)
    X = np.outer(x, np.ones(N + 1))
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def solve_sl_chebyshev(y_lo, y_hi, rho_func, N, form="original", n_modes=None):
    """
    Solve -[d^2/dy^2 + W(y)] psi = mu psi on [y_lo, y_hi]
    with Dirichlet BCs, using Chebyshev collocation.

    rho_func: callable rho(y) -> array
    form: "original" or "reduced"
    N: number of Chebyshev intervals (N+1 points total)
    Returns: mu (n_modes,), psi (N+1, n_modes), y, W_vals
    """
    D, xi = cheb_diffmat(N)
    # Map [-1,1] -> [y_lo, y_hi]
    y = y_lo + (y_hi - y_lo) * (1 - xi) / 2  # xi=-1 -> y_hi, xi=1 -> y_lo
    # Chain rule: d/dy = (d/dxi) * (dxi/dy) = D * (-2/(y_hi-y_lo))
    scale = -2.0 / (y_hi - y_lo)
    D1 = scale * D
    D2 = D1 @ D1  # second derivative

    # Compute W using spectral derivatives
    rho_vals = rho_func(y)
    drho = D1 @ rho_vals
    d2rho = D1 @ drho
    if form == "original":
        W_vals = d2rho / (2.0 * rho_vals) - 3.0 * drho ** 2 / (4.0 * rho_vals ** 2)
    else:
        W_vals = d2rho / (2.0 * rho_vals) - drho ** 2 / (4.0 * rho_vals ** 2)

    # Full operator: -(D2 + diag(W)) psi = mu psi
    # Interior points only (exclude first and last = boundary)
    idx = slice(1, -1)
    M_int = N - 1  # number of interior points
    A = -(D2[idx, :][:, idx] + np.diag(W_vals[idx]))

    if n_modes is None or n_modes > M_int:
        n_modes = M_int

    mu_all, psi_int = scipy.linalg.eigh(A)
    mu = mu_all[:n_modes]
    psi = np.zeros((N + 1, n_modes))
    psi[1:-1, :] = psi_int[:, :n_modes]

    # Normalize
    # Trapezoidal weights on Chebyshev grid
    dy_trap = np.abs(np.diff(y))
    for i in range(n_modes):
        w = np.sum(0.5 * (psi[:-1, i] ** 2 + psi[1:, i] ** 2) * dy_trap)
        if w > 1e-30:
            psi[:, i] /= np.sqrt(w)

    return mu, psi, y, W_vals


# ── FD eigensolver (for comparison) ──────────────────────────────────────
def solve_sl_fd(y_uniform, W_uniform, n_modes):
    N = len(y_uniform)
    dy = y_uniform[1] - y_uniform[0]
    M = N - 2
    W_int = W_uniform[1:-1]
    main = 2.0 / dy ** 2 - W_int
    off = -np.ones(M - 1) / dy ** 2
    import scipy.sparse
    A = scipy.sparse.diags([off, main, off], [-1, 0, 1], format="csr")
    mu, psi_int = scipy.sparse.linalg.eigsh(A, k=n_modes, which="SA")
    psi = np.zeros((N, n_modes))
    psi[1:-1, :] = psi_int
    for i in range(n_modes):
        norm = np.sqrt(np.sum(psi[:, i] ** 2) * dy)
        if norm > 1e-30:
            psi[:, i] /= norm
    order = np.argsort(mu)
    return mu[order], psi[:, order]


# ── q-space convergence test ─────────────────────────────────────────────
def q_space_test(y, W_vals, mu, psi, n_modes_list, label):
    """Expand q_exact in SL basis, measure reconstruction error."""
    y0, L = y[0], y[-1] - y[0]
    kx = 2.0 * np.pi * 2

    q_exact = np.sin(np.pi * (y - y0) / L)
    d2q = -(np.pi / L) ** 2 * q_exact
    g = d2q + W_vals * q_exact - kx ** 2 * q_exact

    # Integration weights (trapezoidal on possibly non-uniform grid)
    dy_trap = np.abs(np.diff(y))
    def integrate(f):
        return np.sum(0.5 * (f[:-1] + f[1:]) * dy_trap)

    errs = []
    for Nm in n_modes_list:
        if Nm > psi.shape[1]:
            continue
        G = np.array([integrate(psi[:, n] * g) for n in range(Nm)])
        Q = -G / (mu[:Nm] + kx ** 2)
        q_rec = psi[:, :Nm] @ Q
        err = np.sqrt(integrate((q_rec - q_exact) ** 2) / L)
        errs.append((Nm, err))
        print(f"  [{label}] N={Nm:3d}  err={err:.3e}")
    return errs


def fit_slope(errs):
    if len(errs) < 3:
        return 0.0
    x = np.log10([e[0] for e in errs])
    y = np.log10([e[1] for e in errs])
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    return np.polyfit(x[mask], y[mask], 1)[0]


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" Experiment 2: Chebyshev vs FD for SL eigenvalue problem")
    print("=" * 70)

    r_norm, rho_full, le_sol, xi_1 = solve_lane_emden()
    n_modes_list = [5, 10, 20, 40, 80, 160, 256]

    rho_cut = 0.01
    mask = rho_full > rho_cut
    r_in = r_norm[mask]
    rho_in = rho_full[mask]
    y_lo, y_hi = r_in[0], r_in[-1]

    results = {}

    for form_name, form_key in [("original", "original"), ("reduced-p", "reduced")]:
        print(f"\n{'='*70}")
        print(f" Form: {form_name} (rho_cut={rho_cut})")
        print(f"{'='*70}")

        # --- FD (N=512) ---
        Ny_fd = 512
        y_fd = np.linspace(y_lo, y_hi, Ny_fd)
        rho_fd = np.interp(y_fd, r_in, rho_in)
        W_fd = compute_W(y_fd, rho_fd, form_key)

        print(f"\n--- FD (N={Ny_fd}) ---")
        mu_fd, psi_fd = solve_sl_fd(y_fd, W_fd, max(n_modes_list))
        e_fd = q_space_test(y_fd, W_fd, mu_fd, psi_fd, n_modes_list,
                            f"{form_name}-FD")

        # --- Chebyshev (N=512) ---
        N_cheb = 512
        def rho_exact(y_arr):
            return lane_emden_rho_at(y_arr, le_sol, xi_1)

        print(f"\n--- Chebyshev (N={N_cheb}) ---")
        mu_ch, psi_ch, y_ch, W_ch = solve_sl_chebyshev(
            y_lo, y_hi, rho_exact, N_cheb, form=form_key,
            n_modes=min(max(n_modes_list), N_cheb - 1))
        e_ch = q_space_test(y_ch, W_ch, mu_ch, psi_ch, n_modes_list,
                            f"{form_name}-Cheb")

        # --- Chebyshev (N=256) ---
        N_cheb2 = 256
        print(f"\n--- Chebyshev (N={N_cheb2}) ---")
        mu_ch2, psi_ch2, y_ch2, W_ch2 = solve_sl_chebyshev(
            y_lo, y_hi, rho_exact, N_cheb2, form=form_key,
            n_modes=min(max(n_modes_list), N_cheb2 - 1))
        e_ch2 = q_space_test(y_ch2, W_ch2, mu_ch2, psi_ch2, n_modes_list,
                             f"{form_name}-Cheb256")

        results[form_name] = {"fd": e_fd, "cheb512": e_ch, "cheb256": e_ch2}

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)

    for ax, form_name in zip(axes, ["original", "reduced-p"]):
        res = results[form_name]
        for key, color, marker, ls in [
            ("fd", "C3", "o", "-"),
            ("cheb256", "C0", "s", "--"),
            ("cheb512", "C2", "D", "-"),
        ]:
            errs = res[key]
            if not errs:
                continue
            xs, ys = zip(*errs)
            slope = fit_slope(errs)
            label_map = {"fd": "FD-512", "cheb256": "Cheb-256", "cheb512": "Cheb-512"}
            ax.loglog(xs, ys, f"{marker}{ls}", color=color, lw=1.5, ms=5,
                      label=f"{label_map[key]} (slope {slope:.2f})")

        ax.set_xlabel("N modes")
        ax.set_ylabel(r"$\mathrm{err}_{L^2}$")
        ax.set_title(f"{form_name} ($\\rho_{{cut}}={rho_cut}$)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Chebyshev vs FD discretisation for SL eigenproblem",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = VID / "sl_chebyshev_vs_fd.png"
    fig.savefig(out)
    print(f"\n=> {out}")
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print("=" * 70)
    print(f"{'Config':<30s}  {'err(5)':>10s}  {'err(40)':>10s}  {'err(256)':>10s}  {'slope':>7s}")
    print("-" * 75)
    for form_name in ["original", "reduced-p"]:
        for key in ["fd", "cheb256", "cheb512"]:
            errs = results[form_name][key]
            def get_err(n):
                e = [e for nn, e in errs if nn == n]
                return e[0] if e else float("nan")
            s = fit_slope(errs)
            label_map = {"fd": "FD-512", "cheb256": "Cheb-256", "cheb512": "Cheb-512"}
            print(f"  {form_name:<12s} {label_map[key]:<15s}  {get_err(5):10.3e}  {get_err(40):10.3e}  {get_err(256):10.3e}  {s:7.2f}")
        print()


if __name__ == "__main__":
    main()
