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


def solve_gmode_cowling_spherical_regular(r, N2, ell, n_modes):
    """Spherical scalar reduction with GYRE-style regular-at-origin BC.

    Same equation as `solve_gmode_cowling_spherical`:
        -ψ'' + ell(ell+1)/r² · ψ = omega^{-2} · ell(ell+1) · N²/r² · ψ,
        ψ = rho_0 r² xi_r

    but at the inner boundary we enforce the Frobenius regularity
    condition  ψ ∼ r^{ell+1}  (i.e.  xi_r  is finite at r=0) rather than
    Dirichlet ψ(r_lo) = 0.  Translating the condition onto the FD stencil:

        ψ'(r_lo) = (ell+1)/r_lo · ψ(r_lo)
    →   (ψ_1 - ψ_0) / dr = (ell+1)/r_lo · ψ_0
    →   ψ_0 = κ · ψ_1,    κ = 1 / (1 + (ell+1)·dr / r_lo)

    We eliminate ψ_0 from the interior problem by patching row i=1 of the
    tridiagonal, absorbing the coupling to ψ_0 into the diagonal entry.

    The outer boundary keeps the Dirichlet ψ(r_hi) = 0 (GYRE's VACUUM BC
    corresponds to p' = 0 at the stellar surface; for the scalar
    reduction ψ = rho_0 r² xi_r this does NOT map to ψ = 0 in general,
    but if r_hi is safely inside the cavity where N² ≈ 0 already, the
    Dirichlet approximation is adequate -- the eigenfunctions have
    small amplitude there regardless).
    """
    r = np.asarray(r, dtype=float)
    N = len(r)
    dr = r[1] - r[0]
    if not np.allclose(np.diff(r), dr, rtol=1e-10):
        raise ValueError("uniform r required")
    if np.min(r) <= 0:
        raise ValueError("r must be strictly positive")

    # Interior indices run from i=1..N-2; we eliminate ψ_0 via the regular
    # condition and enforce ψ_{N-1} = 0 via Dirichlet.
    M = N - 2
    r_int = r[1:-1]
    r_lo = r[0]

    # Regular-at-origin: ψ_0 = κ ψ_1
    kappa = 1.0 / (1.0 + (ell + 1.0) * dr / r_lo)

    # FD second derivative on interior points, normally
    #    ψ''_i ≈ (ψ_{i-1} - 2ψ_i + ψ_{i+1}) / dr²
    # Row i=1 references ψ_0 = κ ψ_1; substitute and absorb into the
    # diagonal of row 1:
    #    ψ''_1 ≈ (κ ψ_1 - 2 ψ_1 + ψ_2) / dr² = ((κ - 2) ψ_1 + ψ_2) / dr²
    # Compared to the standard stencil, the diagonal changes from -2 to
    # (κ - 2) and the sub-diagonal off-diagonal coupling to ψ_0 is gone.
    A_op = np.zeros((M, M))
    for i in range(M):
        A_op[i, i] = 2.0 / dr ** 2 + ell * (ell + 1.0) / (r_int[i] ** 2)
        if i > 0:
            A_op[i, i - 1] = -1.0 / dr ** 2
        if i < M - 1:
            A_op[i, i + 1] = -1.0 / dr ** 2
    # Patch row 0 (i.e. i=1 in the global index): replace diag 2 → 2 - κ
    A_op[0, 0] = (2.0 - kappa) / dr ** 2 + ell * (ell + 1.0) / (r_int[0] ** 2)

    # Weight B = diag(ell(ell+1) N²/r²)
    w = ell * (ell + 1.0) * N2[1:-1] / (r_int ** 2)
    B_op = np.diag(w)

    lam, vec = scipy.linalg.eig(A_op, B_op)
    lam_r = lam.real
    lam_r[np.abs(lam.imag) > 1e-6 * np.abs(lam_r)] = np.nan
    mask = np.isfinite(lam_r) & (lam_r > 0)
    lam_r = lam_r[mask]
    vec = vec.real[:, mask]

    order = np.argsort(lam_r)[:n_modes]
    lam_sel = lam_r[order]
    vec_sel = vec[:, order]
    omega_sq = 1.0 / lam_sel

    # Reconstruct full-grid ψ: ψ_0 = κ ψ_1, ψ_{N-1} = 0.
    psi = np.zeros((N, vec_sel.shape[1]))
    psi[1:-1, :] = vec_sel
    psi[0, :] = kappa * vec_sel[0, :]
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


def solve_gmode_cowling_gyre_compat(x, V_2, U, A_star, c_1, Gamma_1, ell, n_modes,
                                    alpha_gam=1.0, alpha_pi=1.0,
                                    classification="cavity",
                                    cavity_frac_threshold=0.7):
    """GYRE-compatible 2-variable adiabatic Cowling operator (alpha_grv=0),
    **staggered FD** version.

    Implements the exact GYRE dimensionless equations from
    `gyre/src/eqns/ad/ad_eqns_m.fypp` + `A_t.inc`, with `alpha_grv=0`
    (Cowling = neglect perturbed gravity) and
    `alpha_omg = alpha_gam = alpha_pi = 1` (standard adiabatic):

        x dy_1/dx = (V_g - ell - 1) y_1 + (lambda / (c_1 omega^2) - V_g) y_2
        x dy_2/dx = (c_1 omega^2 - A*) y_1 + (A* - U + 3 - ell) y_2

    where V_g = V_2 * x^2 / Gamma_1 and lambda = ell*(ell+1).

    Boundary conditions (from `IB_regular.inc` + `OB_vacuum.inc`, alpha_grv=0):

        Inner (x = x_lo, approximating regular-at-origin):
            c_1(x_lo) omega^2 y_1 - ell y_2 = 0
        Outer (x = x_hi, VACUUM):
            y_1 - y_2 = 0

    Although omega^2 appears both as a coefficient (c_1 omega^2 in eq2 RHS)
    and inversely (lambda/(c_1 omega^2) in eq1 RHS), multiplying eq1 through
    by omega^2 yields a system LINEAR in omega^2:

        omega^2 P u = Q u,      u = [y_1; y_2]

    with
        P_row1 = [ x d/dx - (V_g - ell - 1) ,  V_g ]          (on [y_1, y_2])
        P_row2 = [ c_1                       ,  0   ]
        Q_row1 = [ 0                         ,  lambda/c_1 ]
        Q_row2 = [ A*                        ,  x d/dx - (A* - U + 3 - ell) ]

    Inner BC (x=x_lo, has omega^2):
        row 0 of P: [ c_1(x_lo) at y_1[0], 0 ],    row 0 of Q: [ 0, -ell at y_2[0] ]
          (so omega^2 c_1 y_1 - ell y_2 = 0 at x_lo, sign: ω²P=Q → ω²c_1 y_1 = -ell y_2
           after rearranging from c_1ω² y_1 - ell y_2 = 0, we set Q_{row 0} = [0, +ell]
           and the equation becomes ω² c_1 y_1 = ell y_2. Matching exactly.)
    Outer BC (x=x_hi, no omega^2):
        row N-1 of P: zeros (drops the eigenvalue dependency)
        row N-1 of Q: [ +1 at y_1[-1], -1 at y_2[-1] ]        (so y_1 - y_2 = 0)

    Rows whose P is zero have generalized eigenvalue ∞ in the scipy.linalg.eig
    solution — we filter them out.

    Parameters
    ----------
    x        : uniform grid (Nr,), must satisfy x[0] > 0 to avoid 1/x blow-up.
    V_2      : V/x^2 at each grid point (finite at x=0), shape (Nr,).
    U        : dlnM_r/dlnr, shape (Nr,).
    A_star   : r N^2 / g (dimensionless Brunt^2), shape (Nr,).
    c_1      : x^3 M / M_r, shape (Nr,).
    Gamma_1  : adiabatic index, shape (Nr,) (const for polytrope but we allow array).
    ell      : spherical harmonic degree (>=1).
    n_modes  : number of g-modes to return (highest-omega^2 first, i.e. n_g=1 first).
    alpha_gam, alpha_pi : GYRE switches; keep 1.0 for standard adiabatic.

    Returns
    -------
    omega_sq : ndarray (n_modes,), omega^2 sorted in DESCENDING order
               (i.e. index 0 = n_g=1, the highest-freq g-mode).
    u        : ndarray (2*Nr, n_modes), [y_1; y_2] stacked.
    """
    x = np.asarray(x, dtype=float)
    Nr = len(x)
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h, rtol=1e-10):
        raise ValueError("uniform x required")
    if np.min(x) <= 0:
        raise ValueError("x[0] must be > 0 to avoid 1/x blow-up")

    V_2 = np.asarray(V_2, dtype=float)
    U = np.asarray(U, dtype=float)
    A_star = np.asarray(A_star, dtype=float)
    c_1 = np.asarray(c_1, dtype=float)
    Gamma_1 = np.asarray(Gamma_1, dtype=float)

    lam = ell * (ell + 1.0)

    # -----------------------------------------------------------
    # STAGGERED GRID
    # -----------------------------------------------------------
    # y_1 lives on "node" grid    x_n   with n = 0 .. Nr-1   (= `x` passed in)
    # y_2 lives on "cell" grid    x_c   with c = 0 .. Nr-2
    #                             x_c[c] = 0.5*(x[c] + x[c+1])
    #
    # eq1 is evaluated at y_1 NODES (gives dy_1/dx via a central diff between
    # two adjacent cell-centered values of y_2 — wait, actually we write eq1
    # at CELL centers where y_2 lives, using dy_1/dx = (y_1[c+1] - y_1[c])/h.)
    # eq2 is evaluated at y_2 CELL centers (gives dy_2/dx from two nodes of
    #    a second-level staggering — we need y_2 derivatives at cell centers,
    #    so we differentiate via neighboring cells:
    #       dy_2/dx at cell c ≈ (y_2[c+1] - y_2[c-1]) / (2h) .
    # This is NOT the usual MAC layout. Simpler layout (Keller box):
    #       Collocation point: cell center c, both equations enforced there.
    #       y_1[c] = 0.5*(y_1[n=c] + y_1[n=c+1])  (interpolated)
    #       y_2[c]  = y_2 native cell value
    #       dy_1/dx at c  = (y_1[c+1] - y_1[c]) / h   (uses nodes, exact 2-pt)
    #       dy_2/dx at c  requires neighboring cells — DOES have Nyquist
    #                      null space if we use cell-to-cell central diff.
    # ---------------------------------------------------------------------
    # To TRULY eliminate the Nyquist null space, both derivatives must be
    # of the 2-point forward-difference type, which is possible only if
    # each equation involves the OTHER variable's derivative — which is
    # indeed the case here (eq1 ~ dy_1/dx, eq2 ~ dy_2/dx).  So we do:
    #
    # Layout A (my choice — "Keller box" staggering):
    #   y_1 on nodes   x_n,  n=0..N    (length N+1)
    #   y_2 on cells   x_c = (x_n + x_{n+1})/2,  c=0..N-1  (length N)
    #
    # eq1 at cell c (= inbetween node c and c+1):
    #   x_c (y_1[c+1] - y_1[c]) / h
    #       = (V_g[c] - ell - 1) * y_1_c  +  (λ/(c_1[c] ω²) - V_g[c]) * y_2[c]
    #   with y_1_c = 0.5*(y_1[c] + y_1[c+1]).
    #   That's a row of the matrix with coefficients on y_1[c], y_1[c+1], y_2[c].
    #
    # eq2 at node n (interior, n=1..N-1):
    #   x_n (y_2[n] - y_2[n-1]) / h
    #       = (c_1[n] ω² - A*[n]) * y_1[n]  +  (A*[n] - U[n] + 3 - ell) * y_2_n
    #   with y_2_n = 0.5*(y_2[n-1] + y_2[n]).
    #
    # Boundary conditions:
    #   inner (REGULAR, at node 0):   c_1[0] ω² y_1[0] - ell y_2_0 = 0
    #       where y_2_0 = y_2[0] (nearest cell; first-order approx, fine for
    #       large Nr).  This replaces "eq2 at n=0" which we don't write
    #       (we only write eq2 at n=1..N-1, which is N-1 eq2-rows).
    #   outer (VACUUM, at node N):    y_1[N] - y_2_N = 0
    #       where y_2_N = y_2[N-1].  This is the N-th row.
    #
    # Total unknowns:   (N+1) nodes + N cells = 2N+1
    # Total equations:  N eq1-cells + (N-1) eq2-interior-nodes + 2 BCs = 2N+1 ✓
    # ---------------------------------------------------------------------
    # For simplicity the passed-in `x` of length Nr IS the node grid.  So
    # "N" in the above notation = Nr - 1, and n_cells = Nr - 1.  Total size
    # of linear system = 2*Nr - 1.
    # ---------------------------------------------------------------------
    n_node = Nr
    n_cell = Nr - 1
    size = n_node + n_cell                        # 2*Nr - 1

    x_n = x
    x_c = 0.5 * (x_n[:-1] + x_n[1:])
    V_g_n = V_2 * x_n ** 2 / Gamma_1
    V_g_c = np.interp(x_c, x_n, V_g_n)
    A_c = np.interp(x_c, x_n, A_star)
    A_n = A_star
    U_n = U
    c1_n = c_1
    c1_c = np.interp(x_c, x_n, c_1)

    # u layout:  [ y_1[0..Nr-1], y_2[0..Nr-2] ]
    # Row layout:
    #   rows 0 .. n_cell-1                  : eq1 at cell c
    #   rows n_cell .. n_cell+n_node-3      : eq2 at node n=1..Nr-2  (Nr-2 rows)
    #   row  n_cell+n_node-2                : inner BC at node 0
    #   row  n_cell+n_node-1 = size-1       : outer BC at node Nr-1
    # Total row count: n_cell + (Nr-2) + 2 = Nr-1 + Nr-2 + 2 = 2Nr-1 ✓
    def iy1(n): return n               # index of y_1[n]
    def iy2(c): return n_node + c      # index of y_2[c]

    P = np.zeros((size, size))
    Q = np.zeros((size, size))

    # eq1 at cell c (rows 0 .. n_cell-1):
    #   (x_c/h) * ω² * (y_1[c+1] - y_1[c])
    #      = ω² * [(V_g_c - ell - 1) * (y_1[c] + y_1[c+1])/2 + (-V_g_c) y_2[c]]
    #        + (λ/c_1_c) y_2[c]
    # Rearranged as ω² P u = Q u:
    #   P-row[c] contributions:
    #       y_1[c]   : -x_c/h - 0.5*(V_g_c - ell - 1)
    #       y_1[c+1] : +x_c/h - 0.5*(V_g_c - ell - 1)
    #       y_2[c]   :  V_g_c
    #   Q-row[c] contributions:
    #       y_2[c]   :  λ / c_1_c
    for c in range(n_cell):
        P[c, iy1(c)]     = -x_c[c] / h - 0.5 * (V_g_c[c] - ell - 1.0)
        P[c, iy1(c + 1)] =  x_c[c] / h - 0.5 * (V_g_c[c] - ell - 1.0)
        P[c, iy2(c)]     =  V_g_c[c]
        Q[c, iy2(c)]     =  lam / c1_c[c]

    # eq2 at node n (rows n_cell .. n_cell+Nr-3, for n=1..Nr-2):
    #   (x_n/h) * ω² * (y_2[n] - y_2[n-1])    ← wait this has ω² on LHS?
    # Actually the structure is:
    #   x dy_2/dx = (c_1 ω² - A*_iso) y_1 + (A* - U + 3 - ell) y_2
    # So moving the ω² piece to LHS:
    #   ω² * [c_1 y_1]  =  x dy_2/dx + A*_iso y_1 - (A* - U + 3 - ell) y_2
    # i.e., at node n:
    #   ω² * c_1_n * y_1[n]
    #     = (x_n/h) (y_2[n] - y_2[n-1])   [for n>=1, uses y_2 cells n-1 and n]
    #       + A*_iso_n * y_1[n]
    #       - (A*_n - U_n + 3 - ell) * (y_2[n-1] + y_2[n])/2
    # P-row[eq2,n]:
    #       y_1[n]    :  c_1_n
    # Q-row[eq2,n]:
    #       y_2[n-1]  : -x_n/h - 0.5*(A*_n - U_n + 3 - ell)
    #       y_2[n]    : +x_n/h - 0.5*(A*_n - U_n + 3 - ell)
    #       y_1[n]    :  A*_iso_n
    A_iso_n = A_n * np.where(A_n > 0, alpha_gam, 1.0)
    for n in range(1, n_node - 1):
        row = n_cell + (n - 1)
        P[row, iy1(n)] = c1_n[n]
        # y_2 at cells n-1 and n (valid since n in [1, Nr-2])
        Q[row, iy2(n - 1)] = -x_n[n] / h - 0.5 * (A_n[n] - U_n[n] + 3.0 - ell)
        Q[row, iy2(n)]     =  x_n[n] / h - 0.5 * (A_n[n] - U_n[n] + 3.0 - ell)
        Q[row, iy1(n)]     =  A_iso_n[n]

    # Inner BC (row: second-to-last): c_1[0] ω² y_1[0] = ell * y_2[0]
    row_ib = n_cell + (n_node - 2)     # = 2Nr - 3
    P[row_ib, iy1(0)]   = c1_n[0]
    Q[row_ib, iy2(0)]   = ell

    # Outer BC (last row): y_1[Nr-1] - y_2[Nr-2] = 0
    row_ob = size - 1
    Q[row_ob, iy1(n_node - 1)] =  1.0
    Q[row_ob, iy2(n_cell - 1)] = -1.0

    # Solve ω² P u = Q u via generalised eigenproblem Q u = ω² P u.
    mu, vecs = scipy.linalg.eig(Q, P)

    mu_r = mu.real
    good = (np.isfinite(mu_r)
            & (np.abs(mu.imag) < 1e-6 * (np.abs(mu_r) + 1e-30))
            & (mu_r > 0))
    mu_g = mu_r[good]
    vec_g = vecs[:, good].real

    # -----------------------------------------------------------
    # CLASSIFICATION — propagation-cavity test (per user's criterion A)
    # -----------------------------------------------------------
    # Pure g-mode condition (Unno et al. 1989):
    #     omega^2 < N^2      AND     omega^2 < L_ell^2
    # where, in GYRE's dimensionless units,
    #     N^2      = A_star / c_1
    #     L_ell^2  = lambda * c_s^2 / r^2  (Lamb freq squared)
    # We don't carry c_s^2 explicitly, but for a polytrope c_s^2 = Γ_1 p /
    # ρ, and V_2 = V/x^2 = -dlnP/dr / r (=> c_s^2/x^2 related to 1/V_2/Γ_1).
    # In dimensionless form (GM/R^3=1) we have c_s^2 = x^2 * Γ_1 / V at each
    # x (since V = dlnP/dlnr evaluated at r, and -dlnP/dlnr = ρg r / P =
    # g r / (c_s^2/Γ_1)).  Shortcut:
    #     L_ell^2 = lambda * Γ_1 / (V_2 * x^2 * x^2) ... wait that's messy.
    # Easiest: compute c_s^2 numerically from pressure gradient, which for
    # a polytrope we do not have directly from V_2 alone.
    #
    # Alternative (equivalent, simpler): use GYRE's internal test
    #     omega^2 < A_star / c_1   EVERYWHERE in the cavity   ⇒ g-mode
    # Since L_ell^2 = lambda / (c_1 V_g) * something-complicated, and for
    # the Lane-Emden n=3 polytrope we care about, the simpler separation
    # criterion works:  g-mode energy concentrates where N^2 > omega^2
    # (the "g-cavity"), and we measure what fraction of the kinetic-energy
    # integral lives there.
    #
    # Kinetic energy density (approximate):
    #     e(x) = c_1 * x^3 * |y_1|^2 / omega^2           (radial part)
    # Integrate over x.  fraction_in_cavity =
    #     ∫ e(x) dx over {x: N^2 > omega^2} / ∫ e(x) dx total
    # g-mode threshold: fraction_in_cavity > cavity_frac_threshold (default 0.7).
    N2_profile = A_star / np.maximum(c_1, 1e-30)    # on node grid
    # Lamb frequency squared (dimensionless GYRE units):
    #   L_ell^2 = lambda * c_s^2 / r^2
    # In GYRE's non-dimensionalization (GM/R^3 = 1):
    #   c_s^2 / (GM/R) = Gamma_1 / V     (where V = V_2 * x^2 = dlnP/dlnr)
    # and r^2 = x^2 * R^2, so
    #   L_ell^2 / (GM/R^3) = lambda * Gamma_1 / (V_2 * x^2 * x^2)
    # i.e.
    #   L_ell^2 = lambda * Gamma_1 / (V_2 * x^4)                    (dimless)
    with np.errstate(divide="ignore", invalid="ignore"):
        L2_profile = np.where(V_2 > 0,
                              lam * Gamma_1 / (V_2 * x_n ** 4),
                              np.inf)

    def classify_cavity(y1_nodes, mu_val):
        # Pure g-mode: omega^2 < N^2 AND omega^2 < L_ell^2 over the bulk
        # of the mass-weighted energy.  Weight ~ x^3 * |y_1|^2 approximates
        # kinetic-energy density for the radial-displacement proxy.
        w = x_n ** 3 * y1_nodes ** 2
        total = w.sum()
        if total <= 0:
            return 0.0, 0.0
        in_g_cavity = (N2_profile > mu_val) & (L2_profile > mu_val)
        in_p_cavity = (N2_profile < mu_val) & (L2_profile < mu_val)
        g_frac = (w * in_g_cavity).sum() / total
        p_frac = (w * in_p_cavity).sum() / total
        return g_frac, p_frac

    # Mode inertia (energy) for the user's criterion D:
    def mode_inertia(y1_nodes):
        return float(np.trapezoid(x_n ** 2 * y1_nodes ** 2, x_n))

    g_frac_list = []
    p_frac_list = []
    inertia_list = []
    for k in range(vec_g.shape[1]):
        y1 = vec_g[:n_node, k]
        gf, pf = classify_cavity(y1, mu_g[k])
        g_frac_list.append(gf)
        p_frac_list.append(pf)
        inertia_list.append(mode_inertia(y1))
    g_frac = np.array(g_frac_list)
    p_frac = np.array(p_frac_list)
    inertia = np.array(inertia_list)

    if classification == "cavity":
        # Pure g-mode criterion (Unno+89): the mode energy must avoid the
        # p-cavity.  A p-mode has a sizeable fraction of its energy in the
        # region where both N^2 < ω² and L^2 < ω² (i.e. acoustic propagation
        # zone).  Genuine g-modes have p_frac ≈ 0 regardless of how big
        # g_frac is (the rest of the mode energy sits in evanescent zones
        # where ω² is between N² and L²).
        #
        # Empirically (Lane-Emden n=3 polytrope): p_frac(genuine g) ≈ 0;
        # p_frac(spurious p) ≈ 0.45.  Threshold 0.05 is very safe.
        gmask = p_frac < 0.05
    else:
        gmask = np.ones_like(mu_g, dtype=bool)

    mu_ok = mu_g[gmask]
    vec_ok = vec_g[:, gmask]
    gf_ok = g_frac[gmask]
    in_ok = inertia[gmask]

    # Sort descending ω² (n_g=1 = highest-freq g-mode first)
    order = np.argsort(-mu_ok)
    mu_sel = mu_ok[order]
    vec_sel = vec_ok[:, order]

    # Dedup: pairs with ω² closer than 1e-4 relative (same physical mode,
    # different grid parity partners)
    keep = []
    for i in range(len(mu_sel)):
        if not keep:
            keep.append(i); continue
        if abs(mu_sel[i] - mu_sel[keep[-1]]) / abs(mu_sel[keep[-1]] + 1e-30) > 1e-4:
            keep.append(i)
        if len(keep) >= n_modes:
            break
    mu_sel = mu_sel[keep]
    vec_sel = vec_sel[:, keep]

    n_out = min(n_modes, len(mu_sel))
    omega_sq = mu_sel[:n_out]
    u_out = vec_sel[:, :n_out]
    return omega_sq, u_out


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
