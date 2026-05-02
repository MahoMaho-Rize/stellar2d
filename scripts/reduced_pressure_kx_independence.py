#!/usr/bin/env python3
"""
Experiment C: numerical verification of k_x-independence claim in §5 of
docs/reduced_pressure_liouville.md.

Claim: the SL operator T = d^2/dy^2 + W_tilde(y) depends only on rho_0(y);
       a single eigenpair set {mu_n, psi_n} simultaneously diagonalises the
       full operator (T - k_x^2) for ALL horizontal wavenumbers k_x.

Test: build ONE set of eigenpairs for the chosen formulation, then run the
full end-to-end Poisson solve for many k_x values. If the claim holds:
  (a) the relative L2 error of pi_rec vs pi_exact is bounded independently
      of k_x (modulo the spectral truncation floor);
  (b) the basis does not need to be rebuilt when k_x changes.

Both formulations are tested and compared on the same k_x sweep.

Usage:
  python scripts/reduced_pressure_kx_independence.py
"""
import os
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


# ── Shared infrastructure ───────────────────────────────────────────────
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
    # Sign-corrected reduced-pressure potential (see reduced_pressure_endtoend.py
    # for the derivation note).
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    return drho ** 2 / (4.0 * rho ** 2) - d2rho / (2.0 * rho)


def solve_sl_eigenpairs(y, W, n_modes):
    N = len(y)
    dy = y[1] - y[0]
    M = N - 2
    W_int = W[1:-1]
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


def solve_pipeline(y, rho, W, pi_ex, kind, k_x, mu, psi):
    """End-to-end Poisson solve for a given k_x using a pre-computed
    (mu, psi) eigenpair set.

    kind = 'reduced' or 'original'. Returns recovered pi on grid y.
    """
    dy = y[1] - y[0]
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)

    y0, L = y[0], y[-1] - y[0]
    dpi_ex = (np.pi / L) * np.cos(np.pi * (y - y0) / L)
    d2pi_ex = -(np.pi / L) ** 2 * pi_ex

    if kind == "reduced":
        # f_tilde = drho * dpi + rho * d2pi - k_x^2 * rho * pi
        f = drho * dpi_ex + rho * d2pi_ex - k_x ** 2 * rho * pi_ex
        g = f / np.sqrt(rho)
        G = psi.T @ g * dy
        Q = -G / (mu + k_x ** 2)
        q_rec = psi @ Q
        return q_rec / np.sqrt(rho)
    else:  # original
        p_ex = rho * pi_ex
        dp_ex = drho * pi_ex + rho * dpi_ex
        d2p_ex = d2rho * pi_ex + 2.0 * drho * dpi_ex + rho * d2pi_ex
        f = (-drho / rho ** 2) * dp_ex + (1.0 / rho) * d2p_ex \
            - k_x ** 2 * (1.0 / rho) * p_ex
        g = np.sqrt(rho) * f
        G = psi.T @ g * dy
        Q = -G / (mu + k_x ** 2)
        q_rec = psi @ Q
        return (np.sqrt(rho) * q_rec) / rho   # p_rec / rho -> pi


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print(" Experiment C: k_x-independence of the SL eigenbasis")
    print("=" * 72)

    xi, theta, xi_1 = solve_lane_emden(n=1.5)
    r_norm = xi / xi_1
    rho = np.abs(theta) ** 1.5
    rho_cut = 0.01
    mask = rho > rho_cut
    r_in, rho_in = r_norm[mask], rho[mask]

    Ny = 512
    y = np.linspace(r_in[0], r_in[-1], Ny)
    rho_y = np.interp(y, r_in, rho_in)
    y0, L = y[0], y[-1] - y[0]
    pi_ex = np.sin(np.pi * (y - y0) / L)

    W_orig = compute_W_original(y, rho_y)
    W_redu = compute_W_reduced(y, rho_y)

    Nm = 80   # fixed basis size; past the low-N advantage regime but before FD floor
    print(f"  Ny = {Ny},  N_modes = {Nm},  rho_cut = {rho_cut}")
    print(f"  Domain y in [{y0:.3f}, {y[-1]:.3f}],  rho range "
          f"[{rho_y.min():.2e}, {rho_y.max():.2e}]")

    # Build eigenbases ONCE; never rebuild as k_x varies.
    print("\n  Building eigenpair sets (ONCE)...")
    mu_o, psi_o = solve_sl_eigenpairs(y, W_orig, Nm)
    mu_r, psi_r = solve_sl_eigenpairs(y, W_redu, Nm)
    print(f"    original:    mu[0]={mu_o[0]:.4e}, mu[-1]={mu_o[-1]:.4e}")
    print(f"    reduced-p:   mu[0]={mu_r[0]:.4e}, mu[-1]={mu_r[-1]:.4e}")

    # k_x sweep. 2pi multiples so k_x^2 spans several decades.
    k_x_list = np.array([1, 2, 4, 8, 16, 32, 64, 128]) * 2.0 * np.pi
    print(f"\n  k_x sweep: {len(k_x_list)} values from {k_x_list[0]:.3f} to {k_x_list[-1]:.3f}")

    errs_o = []
    errs_r = []
    header = f"  {'k_x':>10}  {'k_x^2':>11}  {'err_orig':>12}  {'err_redu':>12}"
    print("\n" + header)
    print("-" * len(header))
    for k_x in k_x_list:
        pi_o = solve_pipeline(y, rho_y, W_orig, pi_ex, "original", k_x, mu_o, psi_o)
        pi_r = solve_pipeline(y, rho_y, W_redu, pi_ex, "reduced", k_x, mu_r, psi_r)
        err_o = np.sqrt(np.mean((pi_o - pi_ex) ** 2))
        err_r = np.sqrt(np.mean((pi_r - pi_ex) ** 2))
        errs_o.append(err_o)
        errs_r.append(err_r)
        print(f"  {k_x:10.3f}  {k_x**2:11.3e}  {err_o:12.4e}  {err_r:12.4e}")

    errs_o = np.array(errs_o)
    errs_r = np.array(errs_r)

    # Relative spread across k_x
    rel_o = errs_o.max() / errs_o.min()
    rel_r = errs_r.max() / errs_r.min()
    print(f"\n  err_orig spread (max/min) = {rel_o:.2f}x")
    print(f"  err_redu spread (max/min) = {rel_r:.2f}x")
    print("  k_x-independence holds if spread is O(1); divergence at high k_x")
    print("  would indicate spurious k_x dependence or basis insufficiency.")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.loglog(k_x_list, errs_o, "o-", color="C3", lw=1.8, ms=7,
              label=f"original (spread {rel_o:.1f}x)")
    ax.loglog(k_x_list, errs_r, "s-", color="C0", lw=1.8, ms=7,
              label=f"reduced-p (spread {rel_r:.1f}x)")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$\|\pi_{\mathrm{rec}} - \pi_{\mathrm{exact}}\|_{L^2}$")
    ax.set_title(
        f"Experiment C: k_x-independence at fixed basis (Nm={Nm}, rho_cut={rho_cut})")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = VID / "reduced_pressure_kx_independence.png"
    fig.savefig(out)
    print(f"\n  => {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
