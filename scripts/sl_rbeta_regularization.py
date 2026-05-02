#!/usr/bin/env python3
"""
Experiment 1: r^beta regularization of the SL eigenvalue problem.

For both formulations, extract the leading singular behavior from the
eigenfunctions before expanding in the SL basis:

  Original form:   W ~ -21/16 / t^2,  indicial alpha = 3/4
                    => set psi_n(t) = t^{3/4} * phi_n(t), solve for phi_n
  Reduced-pressure: W ~ +3/16 / t^2,  indicial alpha = 1/4
                    => set psi_n(t) = t^{1/4} * phi_n(t), solve for phi_n

If the regularization works, the effective potential W_eff for phi should
be bounded, and the convergence should improve toward exponential.

Usage:
  python scripts/sl_rbeta_regularization.py
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
    return xi / xi_1, np.abs(theta) ** 1.5


# ── W computation ────────────────────────────────────────────────────────
def compute_W(y, rho, form="original"):
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    if form == "original":
        return d2rho / (2.0 * rho) - 3.0 * drho ** 2 / (4.0 * rho ** 2)
    else:
        return d2rho / (2.0 * rho) - drho ** 2 / (4.0 * rho ** 2)


# ── SL eigensolver ───────────────────────────────────────────────────────
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
    for i in range(n_modes):
        norm = np.sqrt(np.sum(psi[:, i] ** 2) * dy)
        if norm > 1e-30:
            psi[:, i] /= norm
    order = np.argsort(mu)
    return mu[order], psi[:, order]


# ── Regularized W: extract t^beta factor ─────────────────────────────────
def compute_W_regularized(y, W_raw, beta, y_surface):
    """
    Given psi ~ t^beta * phi near the surface (t = y_surface - y),
    substitute into the eigenvalue equation to get the effective
    potential W_eff for phi.

    -phi'' - (2*beta/t)*phi' + [beta*(1-beta)/t^2 - W]*phi = mu*phi

    This is NOT in standard SL form (has a phi' term).
    To bring it to SL form, apply a second substitution:
    phi = t^{-beta_2} * chi, choosing beta_2 to kill the first-derivative term.

    For the equation phi'' + (2*beta/t)*phi' + (W - beta*(1-beta)/t^2)*phi = -mu*phi,
    set phi = t^{-beta} * chi (since the phi' coeff is 2*beta/t):
    => chi'' + W_eff * chi = -mu * chi
    where W_eff = W - beta*(1-beta)/t^2 - beta*(beta+1)/t^2 + 2*beta^2/t^2
                = W - beta*(1-beta)/t^2

    Wait, let me redo this properly.
    The original equation is: psi'' + W*psi = -mu*psi
    Substitution psi = t^beta * phi:
      psi' = beta*t^{beta-1}*phi + t^beta*phi'
      psi'' = beta*(beta-1)*t^{beta-2}*phi + 2*beta*t^{beta-1}*phi' + t^beta*phi''
    So: t^beta*[phi'' + 2*beta/t * phi' + (beta*(beta-1)/t^2 + W)*phi] = -mu*t^beta*phi
    => phi'' + (2*beta/t)*phi' + (beta*(beta-1)/t^2 + W)*phi = -mu*phi

    To remove the phi' term, set phi = t^{-beta} * chi:
      phi' = -beta*t^{-beta-1}*chi + t^{-beta}*chi'
      phi'' = beta*(beta+1)*t^{-beta-2}*chi - 2*beta*t^{-beta-1}*chi' + t^{-beta}*chi''
    Substituting:
      t^{-beta}*[chi'' - 2*beta/t*chi' + beta*(beta+1)/t^2*chi]
      + 2*beta/t * t^{-beta}*[-beta/t*chi + chi']
      + (beta*(beta-1)/t^2 + W)*t^{-beta}*chi = -mu*t^{-beta}*chi

    chi'' - 2*beta/t*chi' + beta*(beta+1)/t^2*chi
      - 2*beta^2/t^2*chi + 2*beta/t*chi'
      + beta*(beta-1)/t^2*chi + W*chi = -mu*chi

    The chi' terms cancel! (-2*beta/t + 2*beta/t = 0)
    Remaining:
    chi'' + [beta*(beta+1) - 2*beta^2 + beta*(beta-1)]/t^2 * chi + W*chi = -mu*chi
    = chi'' + [beta^2+beta - 2*beta^2 + beta^2-beta]/t^2 * chi + W*chi
    = chi'' + [0]/t^2 * chi + W*chi
    = chi'' + W*chi = -mu*chi

    So psi = t^beta * t^{-beta} * chi = chi. The double substitution is trivial!
    This means: extracting t^beta and then removing the first-derivative term
    just gives back the original equation. The r^beta trick does NOT change the
    eigenvalue problem; it only changes the REPRESENTATION.
    """
    # The above derivation shows that the r^beta substitution on the
    # EIGENVALUE PROBLEM is circular. The real benefit is in the
    # EXPANSION: instead of expanding q in psi_n, we expand
    # q / t^beta in psi_n, which improves convergence if q ~ t^beta near surface.
    #
    # So the correct experiment is:
    # 1. Solve the SAME SL eigenproblem (no change to W)
    # 2. When expanding a test function q_exact, pre-divide by t^beta:
    #    f(y) = q_exact(y) / t^beta, expand f in psi_n, reconstruct, multiply by t^beta
    # 3. Compare convergence with and without the t^beta extraction.

    return None  # W is unchanged; the benefit is in the expansion step


def q_space_convergence(y, W, n_modes_list, label, beta=0.0, y_surface=None):
    """
    Direct q-space convergence test with optional r^beta extraction.

    If beta > 0: expand q_exact / t^beta in SL basis, reconstruct, multiply back.
    If beta = 0: standard expansion (no extraction).
    """
    Ny = len(y)
    dy = y[1] - y[0]
    y0, L = y[0], y[-1] - y[0]
    kx = 2.0 * np.pi * 2

    n_max = max(n_modes_list)
    mu, psi = solve_sl_eigenpairs(y, W, min(n_max, Ny - 3))

    # q_exact: smooth function that vanishes at boundaries
    q_exact = np.sin(np.pi * (y - y0) / L)
    d2q_exact = -(np.pi / L) ** 2 * q_exact
    g = d2q_exact + W * q_exact - kx ** 2 * q_exact

    if beta > 0 and y_surface is not None:
        t = np.abs(y_surface - y)
        t = np.maximum(t, 1e-14)
        weight = t ** beta
        # Modify: expand g/weight in basis, reconstruct, multiply by weight
        # The SL equation for f = q / t^beta is:
        #   We solve for q via SL, then the question is whether
        #   pre-weighting improves convergence.
        # Actually, the proper approach is:
        #   q = t^beta * f, so if we expand q in psi_n, the coefficients
        #   a_n = <q, psi_n> = <t^beta * f, psi_n>
        #   If psi_n ~ t^alpha near surface and q ~ t^beta, then
        #   a_n ~ integral of t^{alpha+beta} which converges better
        #   when alpha + beta > -1/2.
        #
        # The r^beta extraction means: define f = q / t^beta, find f via SL
        # with a MODIFIED source g_f = g / t^beta.
        # Then q_reconstructed = t^beta * f_reconstructed.
        g_weighted = g / weight
        # Clamp extreme values at boundaries
        g_weighted[0] = 0.0
        g_weighted[-1] = 0.0
    else:
        weight = None
        g_weighted = g

    errs = []
    for Nm in n_modes_list:
        if Nm > psi.shape[1]:
            continue
        G = psi[:, :Nm].T @ g_weighted * dy
        Q = -G / (mu[:Nm] + kx ** 2)
        f_rec = psi[:, :Nm] @ Q

        if weight is not None:
            q_rec = weight * f_rec
        else:
            q_rec = f_rec

        err = np.sqrt(np.mean((q_rec - q_exact) ** 2))
        errs.append((Nm, err))
        print(f"  [{label}] N_modes = {Nm:3d}  err_L2 = {err:.3e}")

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


def check_exponential(errs):
    """Check if convergence is exponential: log(err) vs N should be linear."""
    if len(errs) < 3:
        return 0.0
    ns = np.array([e[0] for e in errs])
    log_errs = np.log10([e[1] for e in errs])
    mask = np.isfinite(log_errs) & (log_errs > -14)
    if mask.sum() < 3:
        return 0.0
    # Fit log(err) = a + b*N (exponential) vs log(err) = a + b*log(N) (algebraic)
    coeffs_exp = np.polyfit(ns[mask], log_errs[mask], 1)
    coeffs_alg = np.polyfit(np.log10(ns[mask]), log_errs[mask], 1)
    # R^2 for both
    pred_exp = np.polyval(coeffs_exp, ns[mask])
    pred_alg = np.polyval(coeffs_alg, np.log10(ns[mask]))
    ss_tot = np.sum((log_errs[mask] - log_errs[mask].mean()) ** 2)
    r2_exp = 1 - np.sum((log_errs[mask] - pred_exp) ** 2) / max(ss_tot, 1e-30)
    r2_alg = 1 - np.sum((log_errs[mask] - pred_alg) ** 2) / max(ss_tot, 1e-30)
    return r2_exp, r2_alg, coeffs_exp[0], coeffs_alg[0]


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" Experiment 1: r^beta regularization of SL convergence")
    print("=" * 70)

    r_norm, rho_full = solve_lane_emden()
    n_modes_list = [5, 10, 20, 40, 80, 160, 256]
    Ny = 512

    rho_cutoffs = [0.01, 0.001]
    results = {}

    for rho_cut in rho_cutoffs:
        mask = rho_full > rho_cut
        r_in = r_norm[mask]
        rho_in = rho_full[mask]
        y_surface = r_in[-1]

        y = np.linspace(r_in[0], r_in[-1], Ny)
        rho = np.interp(y, r_in, rho_in)

        W_orig = compute_W(y, rho, "original")
        W_redu = compute_W(y, rho, "reduced")

        print(f"\n{'='*70}")
        print(f"rho_cut = {rho_cut}, domain [{y[0]:.3f}, {y[-1]:.3f}]")
        print(f"{'='*70}")

        # --- Original form ---
        print(f"\n--- Original W (C = -21/16), no extraction ---")
        e_orig = q_space_convergence(y, W_orig, n_modes_list, "orig")

        print(f"\n--- Original W, beta=3/4 extraction ---")
        e_orig_b = q_space_convergence(y, W_orig, n_modes_list, "orig+b=3/4",
                                        beta=0.75, y_surface=y_surface)

        # --- Reduced pressure form ---
        print(f"\n--- Reduced-p W (C = +3/16), no extraction ---")
        e_redu = q_space_convergence(y, W_redu, n_modes_list, "redu")

        print(f"\n--- Reduced-p W, beta=1/4 extraction ---")
        e_redu_b = q_space_convergence(y, W_redu, n_modes_list, "redu+b=1/4",
                                        beta=0.25, y_surface=y_surface)

        results[rho_cut] = {
            "orig": e_orig, "orig_beta": e_orig_b,
            "redu": e_redu, "redu_beta": e_redu_b,
        }

    # --- Smooth Gaussian control ---
    print(f"\n{'='*70}")
    print(f"Smooth Gaussian rho (control)")
    print(f"{'='*70}")
    y_sm = np.linspace(0, 1, Ny)
    rho_sm = 1.0 + 0.5 * np.exp(-((y_sm - 0.5) / 0.15) ** 2)
    W_orig_sm = compute_W(y_sm, rho_sm, "original")
    W_redu_sm = compute_W(y_sm, rho_sm, "reduced")

    print(f"\n--- Original W, smooth ---")
    e_sm_o = q_space_convergence(y_sm, W_orig_sm, n_modes_list, "orig-sm")
    print(f"\n--- Reduced-p W, smooth ---")
    e_sm_r = q_space_convergence(y_sm, W_redu_sm, n_modes_list, "redu-sm")

    results["smooth"] = {"orig": e_sm_o, "redu": e_sm_r}

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(rho_cutoffs), figsize=(6 * len(rho_cutoffs), 5), dpi=140)
    if len(rho_cutoffs) == 1:
        axes = [axes]

    colors = {"orig": "C3", "orig_beta": "C1", "redu": "C0", "redu_beta": "C2"}
    labels = {
        "orig": r"original ($C=-21/16$)",
        "orig_beta": r"original + $t^{3/4}$ extraction",
        "redu": r"reduced-p ($C=+3/16$)",
        "redu_beta": r"reduced-p + $t^{1/4}$ extraction",
    }
    markers = {"orig": "o", "orig_beta": "^", "redu": "s", "redu_beta": "D"}

    for ax, rho_cut in zip(axes, rho_cutoffs):
        res = results[rho_cut]
        for key in ["orig", "orig_beta", "redu", "redu_beta"]:
            errs = res[key]
            if not errs:
                continue
            xs, ys = zip(*errs)
            slope = fit_slope(errs)
            ax.loglog(xs, ys, f"{markers[key]}-", color=colors[key],
                      lw=1.5, ms=5, label=f"{labels[key]} ({slope:.2f})")

        ax.set_xlabel("N modes")
        ax.set_ylabel(r"$\mathrm{err}_{L^2}$")
        ax.set_title(rf"$\rho_{{\mathrm{{cut}}}} = {rho_cut}$")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(r"SL convergence: $r^\beta$ regularization experiment",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = VID / "sl_rbeta_regularization.png"
    fig.savefig(out)
    print(f"\n=> {out}")
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35s}  {'err(256)':>10s}  {'slope':>7s}")
    print("-" * 60)
    for rho_cut in rho_cutoffs:
        res = results[rho_cut]
        for key in ["orig", "orig_beta", "redu", "redu_beta"]:
            errs = res[key]
            e256 = [e for n, e in errs if n == 256]
            e = e256[0] if e256 else float("nan")
            s = fit_slope(errs)
            print(f"  rho_cut={rho_cut} {key:<20s}  {e:10.3e}  {s:7.2f}")
        print()


if __name__ == "__main__":
    main()
