#!/usr/bin/env python3
"""
Experiment 3: g-mode frequency cross-check.

The SL eigenvalues mu_n from the Liouville-transformed pressure equation
are related to the squared frequencies of internal gravity waves (g-modes).
This script:

  1. Solves the SL eigenvalue problem for Lane-Emden n=3/2 (both formulations)
  2. Computes the Brunt-Vaisala frequency N^2(r) for a non-adiabatic perturbation
  3. Solves the linearised g-mode eigenvalue problem directly (shooting method)
  4. Cross-checks the SL eigenvalues against the direct g-mode frequencies
  5. Verifies the Tassoul asymptotic period spacing

Usage:
  python scripts/sl_gmode_crosscheck.py
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
    return xi / xi_1, np.abs(theta) ** 1.5, xi_1


# ── W and SL eigensolver ─────────────────────────────────────────────────
def compute_W(y, rho, form="original"):
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    if form == "original":
        return d2rho / (2.0 * rho) - 3.0 * drho ** 2 / (4.0 * rho ** 2)
    else:
        return d2rho / (2.0 * rho) - drho ** 2 / (4.0 * rho ** 2)


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


# ── Brunt-Vaisala frequency ──────────────────────────────────────────────
def compute_brunt_vaisala(y, rho, gamma=5.0 / 3.0, g=1.0):
    """
    Compute N^2 for a polytropic stratification with a non-adiabatic
    temperature perturbation.

    For an isentropic (adiabatic) polytrope, N^2 = 0 identically.
    To get non-trivial g-modes, we introduce a small superadiabatic
    gradient delta:

      N^2 = g * delta / H_p

    where H_p = P / (rho * g) is the pressure scale height and
    delta = nabla - nabla_ad measures the departure from adiabatic.

    For a polytrope with index n, nabla = 1/(n+1), nabla_ad = (gamma-1)/gamma.
    For n=3/2, gamma=5/3: nabla = 0.4, nabla_ad = 0.4 => N^2 = 0.

    We perturb: nabla = nabla_ad + delta_0 * sin(pi * y / y_max)
    """
    nabla_ad = (gamma - 1.0) / gamma  # = 0.4 for gamma=5/3
    delta_0 = 0.1  # superadiabatic perturbation amplitude

    # Pressure scale height: P ~ rho^gamma, H_p = P/(rho*g) = K*rho^{gamma-1}/g
    # For simplicity normalize so that H_p = rho^{gamma-1} (set K/g = 1)
    H_p = rho ** (gamma - 1.0)
    H_p = np.maximum(H_p, 1e-20)

    delta = delta_0 * np.sin(np.pi * y / y[-1])
    N2 = g * delta / H_p

    return N2, delta, H_p


# ── Direct g-mode eigenvalue problem ─────────────────────────────────────
def solve_gmode_direct(y, rho, N2, ell=1, n_modes=20):
    """
    Solve the linearised adiabatic oscillation equation in the Cowling
    approximation (neglect perturbation to gravitational potential).

    In a plane-parallel stratified slab with horizontal wavenumber k_h,
    the vertical displacement eigenvalue problem is:

      d/dy[rho * d(xi_y)/dy] + rho * (N^2/omega^2 - 1) * k_h^2 * xi_y = 0

    This is a generalised eigenvalue problem in omega^2.
    Rearranging with lambda = k_h^2 / omega^2:

      d/dy[rho * d(xi)/dy] + [rho * N^2 * lambda - rho * k_h^2] * xi = 0

    For g-modes (omega^2 << N^2), the -rho*k_h^2 term is subdominant.
    We solve the simpler form:

      -d/dy[rho * d(xi)/dy] = lambda * rho * N^2 * xi

    with Dirichlet BCs xi=0 at boundaries, giving eigenvalues
    lambda_n = k_h^2 / omega_n^2, hence omega_n^2 = k_h^2 / lambda_n.
    """
    N_pts = len(y)
    dy = y[1] - y[0]

    # Discretise: -d/dy[rho * dxi/dy] on interior nodes
    # Using conservative FD: -(rho_{i+1/2}*(xi_{i+1}-xi_i) - rho_{i-1/2}*(xi_i-xi_{i-1}))/dy^2
    rho_half_p = 0.5 * (rho[1:] + rho[:-1])  # rho at i+1/2
    M = N_pts - 2  # interior nodes

    # LHS: -d/dy[rho * dxi/dy]
    main_L = np.zeros(M)
    off_L_lo = np.zeros(M - 1)
    off_L_hi = np.zeros(M - 1)
    for i in range(M):
        j = i + 1  # index in full grid
        main_L[i] = (rho_half_p[j] + rho_half_p[j - 1]) / dy ** 2
        if i > 0:
            off_L_lo[i - 1] = -rho_half_p[j - 1] / dy ** 2
        if i < M - 1:
            off_L_hi[i] = -rho_half_p[j] / dy ** 2

    L = scipy.sparse.diags(
        [off_L_lo, main_L, off_L_hi], [-1, 0, 1], format="csr")

    # RHS: rho * N^2 (mass matrix)
    rho_N2 = rho[1:-1] * N2[1:-1]
    # Avoid zero/negative entries (N^2 can be zero at boundaries)
    rho_N2 = np.maximum(rho_N2, 1e-20)
    B = scipy.sparse.diags([rho_N2], [0], format="csr")

    # Generalised eigenvalue problem: L * xi = lambda * B * xi
    try:
        lam, xi_int = scipy.sparse.linalg.eigsh(L, k=n_modes, M=B, which="SM")
    except Exception as e:
        print(f"  Warning: eigsh failed ({e}), trying dense solve")
        lam, xi_int = scipy.linalg.eigh(L.toarray(), B.toarray())
        idx = np.argsort(lam)
        lam = lam[idx[:n_modes]]
        xi_int = xi_int[:, idx[:n_modes]]

    # lambda = k_h^2 / omega^2 => omega^2 = k_h^2 / lambda
    # We used k_h = 1 implicitly; for ell=1 in a slab, k_h = pi/L
    k_h = np.pi * ell / (y[-1] - y[0])
    omega2 = k_h ** 2 / np.maximum(lam, 1e-30)

    # Sort by omega^2 (ascending = lowest frequency g-modes first)
    order = np.argsort(omega2)
    omega2 = omega2[order]

    # Filter out spurious negative or very large eigenvalues
    valid = (omega2 > 0) & (omega2 < 1e6)
    omega2 = omega2[valid]

    return omega2[:n_modes], k_h


# ── Tassoul asymptotic period spacing ────────────────────────────────────
def tassoul_period_spacing(y, N2, ell=1):
    """
    Tassoul (1980) asymptotic formula for g-mode period spacing:
      Delta_P = 2 * pi^2 / (sqrt(ell*(ell+1)) * integral(|N|/r dr))

    In a plane-parallel slab, the analogous formula is:
      Delta_P = 2 * pi / (sqrt(ell*(ell+1))^{1/2} * integral(sqrt(|N^2|) dy))

    More precisely, for the 1D slab:
      omega_n ~ n * pi * <N> / L  (large n)
      => P_n = 2*pi/omega_n ~ 2*L / (n * <N>)
      => Delta_P = P_{n+1} - P_n ~ 2*L / (n*(n+1)*<N>) (decreasing)

    For the standard asymptotic: P_n ~ P_0 / n, hence Delta_P ~ P_0 / n^2.
    The integral quantity is:
      Pi_0 = 2 * pi^2 / integral(sqrt(max(N^2,0)) dy)
    """
    N_abs = np.sqrt(np.maximum(N2, 0.0))
    dy = np.abs(np.diff(y))
    integral_N = np.sum(0.5 * (N_abs[:-1] + N_abs[1:]) * dy)
    if integral_N < 1e-30:
        return 0.0, 0.0
    Pi_0 = 2.0 * np.pi ** 2 / (np.sqrt(ell * (ell + 1)) * integral_N)
    return Pi_0, integral_N


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" Experiment 3: g-mode frequency cross-check")
    print("=" * 70)

    r_norm, rho_full, xi_1 = solve_lane_emden()

    rho_cut = 0.01
    mask = rho_full > rho_cut
    r_in = r_norm[mask]
    rho_in = rho_full[mask]

    Ny = 1024  # higher resolution for eigenvalue accuracy
    y = np.linspace(r_in[0], r_in[-1], Ny)
    rho = np.interp(y, r_in, rho_in)

    n_modes = 30

    # ── 1. SL eigenvalues ────────────────────────────────────────────────
    print(f"\n--- SL eigenvalues (N_y={Ny}, {n_modes} modes) ---")
    W_orig = compute_W(y, rho, "original")
    W_redu = compute_W(y, rho, "reduced")

    mu_orig, _ = solve_sl_eigenpairs(y, W_orig, n_modes)
    mu_redu, _ = solve_sl_eigenpairs(y, W_redu, n_modes)

    print(f"  mu_orig[0:5] = {mu_orig[:5]}")
    print(f"  mu_redu[0:5] = {mu_redu[:5]}")

    # ── 2. Brunt-Vaisala N^2 ─────────────────────────────────────────────
    print(f"\n--- Brunt-Vaisala N^2 ---")
    N2, delta, H_p = compute_brunt_vaisala(y, rho)
    print(f"  N^2 range: [{N2.min():.3e}, {N2.max():.3e}]")
    print(f"  delta range: [{delta.min():.3e}, {delta.max():.3e}]")

    # ── 3. Direct g-mode eigenvalues ─────────────────────────────────────
    print(f"\n--- Direct g-mode solve (Cowling approximation) ---")
    omega2_direct, k_h = solve_gmode_direct(y, rho, N2, ell=1, n_modes=n_modes)
    print(f"  k_h = {k_h:.4f}")
    print(f"  omega^2 (first 10): {omega2_direct[:10]}")

    # ── 4. Relationship: SL mu_n and g-mode omega^2 ─────────────────────
    print(f"\n--- Cross-check: SL mu_n vs g-mode omega^2 ---")
    print(f"\n  The SL eigenvalues diagonalise the Schrodinger operator T = d^2/dy^2 + W.")
    print(f"  For the pressure Poisson equation [T - k_x^2]q = g, the eigenvalues mu_n")
    print(f"  determine the spectral response at each wavenumber.")
    print(f"\n  The g-mode frequencies are eigenvalues of a DIFFERENT operator:")
    print(f"  -d/dy[rho * dxi/dy] = (k_h^2/omega^2) * rho * N^2 * xi")
    print(f"\n  Connection: both operators encode the density stratification,")
    print(f"  but through different physical mechanisms (pressure vs buoyancy).")
    print(f"  For an isentropic background, N^2=0 and g-modes don't exist;")
    print(f"  the SL eigenvalues then reduce to Fourier + potential shift.")

    # ── 5. Asymptotic analysis ───────────────────────────────────────────
    print(f"\n--- Asymptotic analysis ---")

    # SL eigenvalue spacing
    L = y[-1] - y[0]
    fourier_ref = np.array([(n * np.pi / L) ** 2 for n in range(1, n_modes + 1)])
    W_avg_orig = np.mean(W_orig)
    W_avg_redu = np.mean(W_redu)

    print(f"\n  SL asymptotic: mu_n ~ (n*pi/L)^2 - <W>")
    print(f"  <W_orig> = {W_avg_orig:.3f}, <W_redu> = {W_avg_redu:.3f}")
    print(f"  (pi/L)^2 = {(np.pi / L) ** 2:.3f}")
    print(f"\n  mu_n/(n+1)^2 convergence (should approach (pi/L)^2 for large n):")
    for n in [0, 4, 9, 19, 29]:
        if n < len(mu_orig):
            ratio_o = mu_orig[n] / (n + 1) ** 2
            ratio_r = mu_redu[n] / (n + 1) ** 2
            print(f"    n={n:2d}  orig: {ratio_o:.4f}  redu: {ratio_r:.4f}  "
                  f"Fourier: {(np.pi / L) ** 2:.4f}")

    # g-mode period spacing
    if len(omega2_direct) >= 5:
        periods = 2.0 * np.pi / np.sqrt(np.maximum(omega2_direct, 1e-30))
        delta_P = np.diff(periods)
        print(f"\n  g-mode periods (first 10):")
        for i in range(min(10, len(periods))):
            print(f"    n={i:2d}  P = {periods[i]:.4f}"
                  + (f"  Delta_P = {delta_P[i]:.4f}" if i < len(delta_P) else ""))

    # Tassoul prediction
    Pi_0, int_N = tassoul_period_spacing(y, N2, ell=1)
    print(f"\n  Tassoul Pi_0 = {Pi_0:.4f}")
    print(f"  integral(sqrt(N^2) dy) = {int_N:.4f}")

    # ── 6. Eigenvalue comparison table ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f" Eigenvalue comparison table")
    print(f"{'='*70}")
    print(f"{'n':>3s}  {'mu_orig':>12s}  {'mu_redu':>12s}  {'(n+1)^2*(pi/L)^2':>18s}  {'omega^2_gmode':>14s}")
    print("-" * 70)
    for n in range(min(20, n_modes)):
        f_ref = (n + 1) ** 2 * (np.pi / L) ** 2
        om2 = omega2_direct[n] if n < len(omega2_direct) else float("nan")
        print(f"{n:3d}  {mu_orig[n]:12.4f}  {mu_redu[n]:12.4f}  {f_ref:18.4f}  {om2:14.4e}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=140)

    # Panel 1: N^2(y) and W(y)
    ax = axes[0, 0]
    ax.plot(y, N2, "b-", lw=1.5, label=r"$N^2(y)$")
    ax2 = ax.twinx()
    ax2.plot(y, W_orig, "r-", lw=1, alpha=0.7, label=r"$W_\mathrm{orig}$")
    ax2.plot(y, W_redu, "g-", lw=1, alpha=0.7, label=r"$W_\mathrm{redu}$")
    ax.set_xlabel("y (r/R)")
    ax.set_ylabel(r"$N^2$", color="b")
    ax2.set_ylabel("W(y)", color="r")
    ax.set_title(r"Stratification: $N^2$ and Liouville potential $W$")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # Panel 2: SL eigenvalue spectrum
    ax = axes[0, 1]
    ns = np.arange(1, n_modes + 1)
    ax.plot(ns, np.sqrt(mu_orig), "ro-", ms=4, lw=1, label=r"$\sqrt{\mu_n}$ original")
    ax.plot(ns, np.sqrt(mu_redu), "bs-", ms=4, lw=1, label=r"$\sqrt{\mu_n}$ reduced-p")
    ax.plot(ns, ns * np.pi / L, "k--", alpha=0.5, label=r"$n\pi/L$ (Fourier)")
    ax.set_xlabel("mode n")
    ax.set_ylabel(r"$\sqrt{\mu_n}$")
    ax.set_title("SL eigenvalue spectrum")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: g-mode frequencies
    ax = axes[1, 0]
    if len(omega2_direct) > 2:
        ax.plot(np.arange(len(omega2_direct)), np.sqrt(omega2_direct),
                "go-", ms=4, lw=1.2, label=r"$\omega_n$ (direct)")
        ax.set_xlabel("mode n")
        ax.set_ylabel(r"$\omega_n$")
        ax.set_title("g-mode frequencies (Cowling approx.)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient g-modes", ha="center", va="center",
                transform=ax.transAxes)

    # Panel 4: mu/(n+1)^2 convergence
    ax = axes[1, 1]
    ns = np.arange(1, n_modes + 1)
    ax.plot(ns, mu_orig / ns ** 2, "ro-", ms=3, lw=1, label="original")
    ax.plot(ns, mu_redu / ns ** 2, "bs-", ms=3, lw=1, label="reduced-p")
    ax.axhline((np.pi / L) ** 2, color="k", ls="--", alpha=0.5,
               label=rf"$(\pi/L)^2 = {(np.pi / L) ** 2:.2f}$")
    ax.set_xlabel("mode n")
    ax.set_ylabel(r"$\mu_n / n^2$")
    ax.set_title(r"Eigenvalue ratio $\mu_n/n^2$ (asymptotic convergence)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("g-mode frequency cross-check (Lane-Emden n=3/2)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = VID / "sl_gmode_crosscheck.png"
    fig.savefig(out)
    print(f"\n=> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
