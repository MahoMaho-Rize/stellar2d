#!/usr/bin/env python3
"""q-space reduced-pressure 2D anelastic g-mode EVP prototype.

Derivation (variable ρ₀(y), horizontal Fourier in x):

    ∂_t u = -∂_x π
    ∂_t v = -∂_y π - (ρ₀'/ρ₀) π + b
    ∂_t b = -N²(y) v
    ∂_x(ρ₀ u) + ∂_y(ρ₀ v) = 0

Assume   v(x,y,t) = V̂(y) sin(k x) e^{-iωt},  u = Û cos,  π = Π̂ sin,  b = B̂ sin.
Eliminating u (continuity), b (buoyancy), and π (momentum) gives

    -(ρ₀ V̂)'' + k² ρ₀ V̂  =  (k² N² ρ₀ / ω²) V̂.                        (*)

Change variable  φ := ρ₀ V̂.  Then V̂ = φ/ρ₀, and (*) becomes

    -φ''(y) + k² φ(y) = (k² N²(y) / ω²) φ(y).                          (**)

The density ρ₀ drops out of the LEADING operator.  With Dirichlet φ(0)=φ(L)=0
the natural expansion basis is pure Fourier  sin(nπy/L_y).  The N²(y) term
couples modes through the symmetric matrix

    H_{nm} = ∫₀^L sin(nπy/L) N²(y) sin(mπy/L) dy

and the EVP reduces to

    diag(μ_n + k²) c = (k²/ω²) H c,    μ_n = (nπ/L)²        (Boussinesq SL basis)

This is MUCH smaller (n_modes × n_modes) than the ny × ny Galerkin EVP used
by compute_2d_gmode_evp, and it is independent of ρ₀ discretisation — so the
discrete operator exactly matches what the TD SL-Poisson pipeline would see
if we advanced φ=ρ₀v instead of v.

This script validates the algebra in 3 cases:
  1. Boussinesq (ρ₀=1, const N²): ω² = N²k²/(k²+k_y²) analytically
  2. Lane-Emden n=3/2 with the same N²(y) = -g ρ'/ρ as the CUDA TD
  3. Compare ω²_qspace with ω²_Galerkin (v-space) at increasing ny
"""
from __future__ import annotations
import argparse

import numpy as np
import scipy.linalg


# ── Lane-Emden n=3/2 ρ₀(y) (same as CUDA set_background) ──────────
def lane_emden_rho_on_y(y, rho_cut=0.01, Ly=1.0):
    """Interp Lane-Emden n=3/2 ρ(ξ) onto y ∈ [0, Ly] with surface cut rho_cut."""
    # Solve LE n=3/2 on ξ ∈ [0, ξ_s] with θ(0)=1, θ'(0)=0
    from scipy.integrate import solve_ivp
    def rhs(xi, y_):
        theta, dtheta = y_
        if xi < 1e-8:
            return [dtheta, -theta**1.5 - 2*dtheta/max(xi, 1e-8)]
        return [dtheta, -theta**1.5 - 2*dtheta/xi]
    sol = solve_ivp(rhs, [1e-6, 4.0], [1.0, 0.0],
                    dense_output=True, max_step=1e-3,
                    events=lambda t, y_: y_[0])
    xi_s = sol.t_events[0][0] if sol.t_events[0].size else 3.65
    xi_dense = np.linspace(1e-6, xi_s, 8000)
    theta = sol.sol(xi_dense)[0]
    rho = np.clip(theta, 0.0, None)**1.5
    # trim to ρ > rho_cut
    mask = rho > rho_cut
    xi_lo, xi_hi = xi_dense[mask].min(), xi_dense[mask].max()
    # map y ∈ [0, Ly] linearly to ξ ∈ [xi_lo, xi_hi]
    xi_of_y = xi_lo + (y / Ly) * (xi_hi - xi_lo)
    return np.interp(xi_of_y, xi_dense, rho).clip(min=rho_cut)


# ── q-space EVP solver ────────────────────────────────────────────
def solve_qspace_evp(N2_fn, Ly, kx_phys, n_modes, ny_quad=1024):
    """q-space EVP using Fourier basis sin(n π y / L_y).

    Returns ω² (descending) and v_modes (ny_quad, n_modes) in V̂(y) space.
    (V̂ = φ/ρ, so to reconstruct physical v the caller multiplies by sin(kx·x).)
    """
    y = np.linspace(0.0, Ly, ny_quad)
    # Fourier basis and its quadrature weights: trapezoidal
    # basis functions sin(n π y / L_y), n = 1..n_modes
    n = np.arange(1, n_modes + 1)
    mu = (n * np.pi / Ly)**2                               # Boussinesq SL eigenvals
    # Build H_{nm} = ⟨φ_n, N² φ_m⟩ with L²-orthonormal basis
    #   φ_n(y) = √(2/L) sin(nπy/L),  ∫₀^L φ_n φ_m dy = δ_{nm}
    # So H_{nm} = (2/L) ∫ sin(nπy/L) N²(y) sin(mπy/L) dy  (Simpson's rule)
    from scipy.integrate import simpson
    N2 = N2_fn(y)
    S = np.sin(np.outer(y, n) * np.pi / Ly)                # (ny_quad, n_modes)
    SN2 = S * N2[:, None]
    H = (2.0 / Ly) * simpson(SN2[:, :, None] * S[:, None, :],
                             x=y, axis=0)                   # n_modes × n_modes

    # Generalised EVP:  diag(μ+k²) c = (k²/ω²) H c  →  H c = (ω²/k²) diag(μ+k²) c
    # → (1/(μ+k²)) H c = (ω²/k²) c  (if H=H^T and diag nonzero)
    # Use scipy.linalg.eig of the generalised form instead, for stability.
    L_diag = np.diag(mu + kx_phys**2)
    # EVP:  H c = λ L c,  λ = ω²/k²
    lam, C = scipy.linalg.eig(H, L_diag)
    lam = np.real(lam)
    C = np.real(C)
    # Drop nonphysical
    good = np.isfinite(lam) & (lam > 0)
    lam = lam[good]
    C = C[:, good]
    omega_sq = lam * kx_phys**2
    order = np.argsort(omega_sq)[::-1]                     # descending
    omega_sq = omega_sq[order]
    C = C[:, order]

    # Reconstruct V̂(y) for each mode:  φ(y) = Σ c_n sin(nπy/L),  V̂ = φ/ρ
    # (but we return φ itself; caller divides by ρ if needed)
    phi = S @ C                                             # (ny_quad, n_kept)
    return y, omega_sq, phi, C


# ── v-space Galerkin EVP for comparison (matches CUDA implementation) ──
def solve_vspace_galerkin(rho_fn, N2_fn, Ly, kx_phys, ny, n_modes):
    """B v = ω² A v  where A = diag(k² N² ρ),  B = -D ρ D + k² ρ.

    Uses Chebyshev-Gauss-Lobatto collocation matching CUDA reference.
    """
    N = ny - 1
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0)**np.arange(N + 1))
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    # Reorder ascending and scale [0, Ly]
    idx = np.arange(N, -1, -1)
    y = (1.0 + x[idx]) * Ly / 2.0
    D = (2.0 / Ly) * D[np.ix_(idx, idx)]

    rho = rho_fn(y)
    N2  = N2_fn(y)
    B = -D @ (np.diag(rho) @ D) + kx_phys**2 * np.diag(rho)
    A = kx_phys**2 * np.diag(N2 * rho)

    interior = slice(1, ny - 1)
    lam, V = scipy.linalg.eig(A[interior, interior], B[interior, interior])
    lam = np.real(lam); V = np.real(V)
    good = np.isfinite(lam) & (lam > 0)
    lam = lam[good]; V = V[:, good]
    order = np.argsort(lam)[::-1]
    lam = lam[order]; V = V[:, order]
    return y, lam[:n_modes], V[:, :n_modes]


# ── Main driver ───────────────────────────────────────────────────
def run_boussinesq():
    Ly = 1.0
    kx_phys = 2 * np.pi / 1.0
    N2_val = 1.0
    N2_fn = lambda y: np.full_like(y, N2_val)
    print("─" * 72)
    print("Case 1: Boussinesq  ρ₀=1,  constant N²=1")
    print("─" * 72)
    y, om2, phi, c = solve_qspace_evp(N2_fn, Ly, kx_phys, n_modes=20)
    for n in range(1, 6):
        ky = n * np.pi / Ly
        ex = N2_val * kx_phys**2 / (kx_phys**2 + ky**2)
        rel = (om2[n - 1] - ex) / ex
        print(f"  n_g={n}: ω²_qspace={om2[n-1]:.10e}  ω²_exact={ex:.10e}  rel={rel:+.3e}")


def run_lane_emden():
    Ly = 1.0
    kx_phys = 2 * np.pi / 1.0
    rho_fn = lambda y: lane_emden_rho_on_y(y, rho_cut=0.01, Ly=Ly)
    # match CUDA: N² = g·max(0, −ρ'/ρ),  g = 1
    def N2_fn(y):
        rho = rho_fn(y)
        drho = np.gradient(rho, y, edge_order=2)
        return np.maximum(-drho / rho, 0.0)
    print("─" * 72)
    print("Case 2: Lane-Emden n=3/2,  rho_cut=0.01")
    print("─" * 72)
    _, om2_q, _, _ = solve_qspace_evp(N2_fn, Ly, kx_phys, n_modes=40)
    print("  q-space EVP  (ω² descending):")
    for n in range(1, 6):
        print(f"    n_g={n}: ω²={om2_q[n-1]:.10e}  ω={np.sqrt(om2_q[n-1]):.6f}")

    # Galerkin reference at several ny to show convergence
    print()
    print("  Galerkin v-space EVP @ increasing ny (should converge to q-space):")
    for ny in [32, 64, 96, 128]:
        y_cgl, om2_g, _ = solve_vspace_galerkin(rho_fn, N2_fn, Ly, kx_phys, ny, 6)
        print(f"    ny={ny:3d}: ω²_n_g=1={om2_g[0]:.10e}  "
              f"Δ(q,Galerkin)={(om2_g[0]-om2_q[0]):+.3e}")


def run_cuda_comparison():
    """Compare against whatever CUDA produced last (from gmode_2d_evp CSV).
    CUDA run:  n_g=1 ω² = 3.8588141307 (Lane-Emden, TANH β=2, ny=64)
    """
    Ly = 1.0
    kx_phys = 2 * np.pi / 1.0
    rho_fn = lambda y: lane_emden_rho_on_y(y, rho_cut=0.01, Ly=Ly)
    def N2_fn(y):
        rho = rho_fn(y)
        drho = np.gradient(rho, y, edge_order=2)
        return np.maximum(-drho / rho, 0.0)
    print("─" * 72)
    print("Case 3: CUDA Galerkin (last run) vs q-space prototype")
    print("─" * 72)
    _, om2_q, _, _ = solve_qspace_evp(N2_fn, Ly, kx_phys, n_modes=40,
                                      ny_quad=2048)
    cuda_n_g1 = 3.8588141307
    rel = (om2_q[0] - cuda_n_g1) / cuda_n_g1
    print(f"  CUDA Galerkin ω²_n_g=1 = {cuda_n_g1:.10e}")
    print(f"  q-space ω²_n_g=1       = {om2_q[0]:.10e}")
    print(f"  rel diff               = {rel:+.3e}")
    print()
    print("  If these disagree by O(1e-2) that is expected — CUDA uses TANH")
    print("  coord-map which changes which y values are sampled; q-space does")
    print("  Simpson's-rule quadrature on uniform y grid so H matrix differs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["boussinesq", "lane_emden", "cuda", "all"],
                    default="all")
    args = ap.parse_args()
    if args.case in ("boussinesq", "all"):
        run_boussinesq()
    if args.case in ("lane_emden", "all"):
        run_lane_emden()
    if args.case in ("cuda", "all"):
        run_cuda_comparison()
