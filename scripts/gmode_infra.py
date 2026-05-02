"""
Shared infrastructure for g-mode experiments.

Reused by:
  - scripts/gmode_exp_a_lane_emden.py   (Lane-Emden W̃-proxy Tassoul heuristic)
  - scripts/gmode_exp_b_stratified.py   (artificial N²(r)>0 stratified layer)

Provides:
  * solve_lane_emden          Lane-Emden n=3/2 polytrope via scipy.integrate
  * compute_W_original        W = rho''/(2 rho) - 3 (rho')^2 / (4 rho^2)
  * compute_W_reduced         W̃ = (rho')^2/(4 rho^2) - rho''/(2 rho)
  * solve_sl_eigenpairs       standard SL:  (-d²/dy² - W) psi = mu psi,
                              homogeneous Dirichlet, FD discretisation.
  * solve_gmode_cowling       Cowling generalised eig:
                              -psi'' = omega^(-2) * l(l+1) * N²(r)/r² * psi.
                              returns omega² sorted ascending, psi.
  * tassoul_dP                ΔP = 2π² / [sqrt(l(l+1)) · ∫ N/r dr]
  * provenance_banner         date + git HEAD + reference doc header

Sign conventions mirror those in:
  docs/reduced_pressure_experiments_2026-05-02.md
  scripts/reduced_pressure_*.py
so a future CI run of all three suites sees consistent W definitions.
"""
from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import numpy as np
import scipy.integrate
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

REPO = Path(__file__).resolve().parent.parent
VID = REPO / "videos"
VID.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
def git_head():
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def provenance_banner(script_rel_path, reference_doc):
    print("#" * 72)
    print(f"#  {script_rel_path}")
    print(f"#  git HEAD: {git_head()}    date: {datetime.date.today().isoformat()}")
    print(f"#  reference doc: {reference_doc}")
    print("#" * 72)


# ─────────────────────────────────────────────────────────────────────────
# Lane-Emden n=3/2 (matches reduced_pressure_*.py for comparability)
# ─────────────────────────────────────────────────────────────────────────
def solve_lane_emden(n=1.5, xi_max=10.0, n_pts=5000):
    """Returns (xi, theta, xi_1). theta >= 0 is the Emden function; rho = theta^n."""
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


# ─────────────────────────────────────────────────────────────────────────
# Liouville potentials (signs consistent with reduced_pressure_*.py)
# ─────────────────────────────────────────────────────────────────────────
def compute_W_original(y, rho):
    """W_orig = +rho''/(2 rho) - 3 (rho')² / (4 rho²).

    Lane-Emden n=3/2 near surface: C = -21/16 (attractive well).
    """
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    return d2rho / (2.0 * rho) - 3.0 * drho ** 2 / (4.0 * rho ** 2)


def compute_W_reduced(y, rho):
    """W̃ = (rho')² / (4 rho²) - rho''/(2 rho).

    Verified symbolically (SymPy) for Lane-Emden n=3/2: C = +3/16 (repulsive).
    """
    drho = np.gradient(rho, y, edge_order=2)
    d2rho = np.gradient(drho, y, edge_order=2)
    return drho ** 2 / (4.0 * rho ** 2) - d2rho / (2.0 * rho)


# ─────────────────────────────────────────────────────────────────────────
# Standard SL eigenproblem (Dirichlet, FD)
# ─────────────────────────────────────────────────────────────────────────
def solve_sl_eigenpairs(y, W, n_modes):
    """Solve (-d²/dy² - W) psi = mu psi with homogeneous Dirichlet BCs.

    Returns (mu (ascending), psi (N, n_modes) L2-normalised in the uniform-grid
    FD quadrature: sum(psi_i * psi_j) * dy = delta_{ij}).
    """
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


# ─────────────────────────────────────────────────────────────────────────
# Cowling generalised eigenproblem for g-modes
# ─────────────────────────────────────────────────────────────────────────
def solve_gmode_cowling(r, N2, ell, n_modes):
    """Solve the Cowling-approximation g-mode equation:

        -psi''(r) = omega^(-2) · ell*(ell+1) * N²(r) / r² · psi(r)
                 ≡ lam * ell*(ell+1) * N²(r) / r² · psi(r)

    where lam = 1/omega² is the eigenvalue (so bigger lam <-> smaller omega,
    the typical g-mode limit).  Homogeneous Dirichlet: psi(r_lo)=psi(r_hi)=0.

    This is the standard generalised form of the radial g-mode equation
    under the Cowling approximation (neglect of perturbed gravity) with
    horizontal wavenumber k_h² = ell(ell+1)/r² -- equivalent to the
    "slowness" form found in Unno et al. (1989) §14.

    For positive N², all omega² are positive (stable oscillations) and the
    g-mode frequencies accumulate at omega → 0 (i.e. lam → ∞).

    Parameters
    ----------
    r      : uniform radial grid (len N)
    N2     : Brunt-Väisälä squared, same shape as r; must be positive where
             g-modes live.
    ell    : spherical harmonic degree (>= 1).
    n_modes: number of smallest-omega² modes to return.

    Returns
    -------
    omega_sq : ndarray, shape (n_modes,), ascending omega².  (i.e. descending
               lam).
    psi      : ndarray, shape (len r, n_modes), eigenfunctions, Dirichlet BCs.
    """
    r = np.asarray(r, dtype=float)
    N = len(r)
    dr = r[1] - r[0]
    if not np.allclose(np.diff(r), dr, rtol=1e-10):
        raise ValueError("solve_gmode_cowling currently assumes uniform r grid")
    if np.min(r) <= 0:
        raise ValueError("r must be strictly positive (1/r² weight)")

    # Interior FD 2nd derivative, same convention as solve_sl_eigenpairs.
    # A psi = (-psi'') on interior.
    M = N - 2
    main = 2.0 / dr ** 2 * np.ones(M)
    off = -np.ones(M - 1) / dr ** 2
    A = scipy.sparse.diags([off, main, off], [-1, 0, 1], format="csr").toarray()

    # Weight matrix B = diag(ell*(ell+1) * N²/r²) on interior.
    w = ell * (ell + 1) * N2[1:-1] / (r[1:-1] ** 2)
    # Guard: negative N² (convectively unstable) would give negative eigenvalues,
    # which are p-modes' g-mode analogue (imaginary omega). We keep only positive.
    B = np.diag(w)

    # Generalised eig: A psi = lam * B psi  => lam = 1/omega²
    # Use scipy.linalg.eig (dense, ~N³ but N~few hundred is fine).
    lam, vec = scipy.linalg.eig(A, B)
    # Keep real positive eigenvalues (g-modes).
    lam_r = lam.real
    lam_r[np.abs(lam.imag) > 1e-6 * np.abs(lam_r)] = np.nan
    mask = np.isfinite(lam_r) & (lam_r > 0)
    lam_r = lam_r[mask]
    vec = vec.real[:, mask]
    # g-mode ordering: n=1 has the LARGEST omega² (smallest lam); omega² -> 0
    # as n -> infinity.  To get the lowest-radial-order modes, sort by
    # ASCENDING lam.
    order = np.argsort(lam_r)[:n_modes]
    lam_sel = lam_r[order]
    vec_sel = vec[:, order]
    omega_sq = 1.0 / lam_sel  # already descending (large omega² first)
    # Return in descending omega² order: index 0 = n=1 (highest frequency).

    psi = np.zeros((N, vec_sel.shape[1]))
    psi[1:-1, :] = vec_sel
    for i in range(psi.shape[1]):
        norm = np.sqrt(np.sum(psi[:, i] ** 2) * dr)
        if norm > 0:
            psi[:, i] /= norm

    return omega_sq, psi


# ─────────────────────────────────────────────────────────────────────────
# Tassoul (1980) asymptotic period spacing
# ─────────────────────────────────────────────────────────────────────────
def solve_gmode_cowling_spherical(r, N2, ell, n_modes):
    """Full spherical scalar reduction of the Cowling g-mode system.

    Solves the eigenproblem derived (symbolically verified in
    `docs/gmode_experiments_2026-05-02.md` §10) by eliminating p' from

        rho0 N^2 xi_r + dp'/dr = omega^2 rho0 xi_r
        (1/r^2) d/dr(rho0 r^2 xi_r) = ell(ell+1) p' / (omega^2 r^2)

    via psi = rho0 r^2 xi_r:

        -psi'' + ell(ell+1)/r^2 * psi = omega^{-2} * ell(ell+1) * N^2/r^2 * psi

    This is the COMPLETE spherical reduction.  It differs from
    `solve_gmode_cowling` (which drops the ell(ell+1)/r^2 centrifugal term
    on the LHS, implementing a local-slab approximation) by exactly that
    term.  In the high-n limit the two agree because psi'' dominates
    psi/r^2; at low n the spherical version is correct and the slab version
    is off by O(1/n^2).

    Parameters
    ----------
    r      : uniform positive radial grid, len N.
    N2     : Brunt squared array, same shape as r.
    ell    : spherical harmonic degree (>=1).
    n_modes: number of smallest-omega^2 modes to return (ascending).

    Returns
    -------
    omega_sq : ndarray (n_modes,) ascending omega^2
    psi      : ndarray (N, n_modes) eigenfunctions, Dirichlet, FD-L2 normalised.
    """
    r = np.asarray(r, dtype=float)
    N = len(r)
    dr = r[1] - r[0]
    if not np.allclose(np.diff(r), dr, rtol=1e-10):
        raise ValueError("uniform r required")
    if np.min(r) <= 0:
        raise ValueError("r must be strictly positive")

    M = N - 2
    # A = -d^2/dr^2 + ell(ell+1)/r^2  on interior
    main = 2.0 / dr**2 * np.ones(M) + ell * (ell + 1) / (r[1:-1] ** 2)
    off  = -np.ones(M - 1) / dr**2
    A = scipy.sparse.diags([off, main, off], [-1, 0, 1], format="csr").toarray()

    # B = diag(ell*(ell+1)*N^2/r^2)
    w = ell * (ell + 1) * N2[1:-1] / (r[1:-1] ** 2)
    B = np.diag(w)

    lam, vec = scipy.linalg.eig(A, B)
    lam_r = lam.real
    lam_r[np.abs(lam.imag) > 1e-6 * np.abs(lam_r)] = np.nan
    mask = np.isfinite(lam_r) & (lam_r > 0)
    lam_r = lam_r[mask]
    vec = vec.real[:, mask]

    # g-mode: n=1 highest-omega^2 has smallest lam (= 1/omega^2)
    order = np.argsort(lam_r)[:n_modes]
    lam_sel = lam_r[order]
    vec_sel = vec[:, order]
    omega_sq = 1.0 / lam_sel

    psi = np.zeros((N, vec_sel.shape[1]))
    psi[1:-1, :] = vec_sel
    for i in range(psi.shape[1]):
        norm = np.sqrt(np.sum(psi[:, i] ** 2) * dr)
        if norm > 0:
            psi[:, i] /= norm

    return omega_sq, psi


def cheb(N):
    """Chebyshev-Gauss-Lobatto points on [-1,1] and the differentiation matrix D.

    Trefethen, "Spectral Methods in MATLAB", `cheb.m`.  Returns (D, x) with
    D shape (N+1, N+1) and x descending from +1 to -1 in Trefethen's order.
    """
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2; c[-1] = 2
    c *= (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def clenshaw_curtis_weights(N):
    """Quadrature weights for the CGL grid on [-1,1]; sum = 2.

    Trefethen p. 128.  Correct for N >= 2.
    """
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)
    for k in range(2, N, 2):
        v -= 2.0 * np.cos(k * theta[1:N]) / (k * k - 1)
    if N % 2 == 0:
        v -= np.cos(N * theta[1:N]) / (N * N - 1)
    w[1:N] = 2.0 * v / N
    w[0] = w[-1] = 1.0 / (N * N - 1 + (N % 2))
    return w


def cheb_on_interval(N, a, b):
    """CGL grid + D2 operator mapped to [a, b], reordered ascending.

    Returns (y_full, D2, w_full) with:
      y_full[0] = a, y_full[-1] = b, uniform-in-theta CGL clustering;
      D2 the second-derivative matrix acting on the FULL grid (endpoints
        included — callers that need Dirichlet strip the boundary rows/cols);
      w_full the Clenshaw-Curtis weights on [a, b] (sum = b - a).
    """
    D, x = cheb(N)
    scale = 2.0 / (b - a)
    D_scaled = D * scale
    D2 = D_scaled @ D_scaled
    w_full = clenshaw_curtis_weights(N) * (b - a) / 2.0

    idx = np.argsort(x)
    y_full = a + (x[idx] + 1.0) * (b - a) / 2.0
    P = np.zeros_like(D2)
    P[np.arange(N + 1), idx] = 1.0
    D2 = P @ D2 @ P.T
    w_full = w_full[idx]
    return y_full, D2, w_full


def solve_gmode_cowling_cheb(r, D2, w_full, N2, ell, n_modes):
    """Chebyshev-collocation version of solve_gmode_cowling.

    Same generalised eigenproblem as the FD routine:
        -ψ''(r) = ω^{-2} · ℓ(ℓ+1) · N²(r)/r² · ψ(r),  ψ(r_lo)=ψ(r_hi)=0

    but with D2 built from `cheb_on_interval` so spectral accuracy is
    achievable on smooth N² profiles.  Eigenfunctions are L2-normalised in
    the Clenshaw-Curtis inner product: Σ w * ψ² = 1.
    """
    r = np.asarray(r, dtype=float)
    N = len(r)
    inner = slice(1, N - 1)
    if np.min(r) <= 0:
        raise ValueError("r must be strictly positive (1/r² weight)")

    A = -D2[inner, inner]
    w_weight = ell * (ell + 1) * N2[1:-1] / (r[1:-1] ** 2)
    B = np.diag(w_weight)

    lam, vec = scipy.linalg.eig(A, B)
    lam_r = lam.real
    lam_r[np.abs(lam.imag) > 1e-6 * np.abs(lam_r)] = np.nan
    mask = np.isfinite(lam_r) & (lam_r > 0)
    lam_r = lam_r[mask]
    vec = vec.real[:, mask]

    # ascending lam = descending omega²; n=1 (highest frequency) has smallest lam.
    order = np.argsort(lam_r)[:n_modes]
    lam_sel = lam_r[order]
    vec_sel = vec[:, order]
    omega_sq = 1.0 / lam_sel

    psi = np.zeros((N, vec_sel.shape[1]))
    psi[inner, :] = vec_sel
    for i in range(psi.shape[1]):
        norm = np.sqrt(np.sum(w_full * psi[:, i] ** 2))
        if norm > 0:
            psi[:, i] /= norm

    return omega_sq, psi


def tassoul_dP(r, N2, ell):
    """ΔP = 2π² / [sqrt(ell*(ell+1)) · ∫ N/r dr].

    High-n g-modes of degree ell have approximately constant period spacing
    ΔP as n → ∞ (Tassoul 1980).  The integral is over the g-mode cavity
    (region with positive N²).
    """
    N = np.sqrt(np.maximum(N2, 0.0))
    integrand = np.where(r > 0, N / r, 0.0)
    I = np.trapezoid(integrand, r)
    if I <= 0:
        return float("inf")
    return 2.0 * np.pi * np.pi / np.sqrt(ell * (ell + 1)) / I
