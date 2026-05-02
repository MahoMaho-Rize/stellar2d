#!/usr/bin/env python3
"""Verify the CUDA time-stepping operator against the analytic Boussinesq-
stratified g-mode dispersion.

We reproduce *exactly* what CUDA does per RK3 Shu-Osher step in the
linearised limit (drop advection, keep only buoyancy + N²·v + projection):

  Stage 1:
     u_star = u
     v_star = v + dt · b
     b_star = b - dt · N² · v
     Poisson: (∂yy - k_x²) π = ∂x u_star + ∂y v_star
     u_new  = u_star - ∂x π
     v_new  = v_star - ∂y π  (Dirichlet at walls)
     b_new  = b_star
  Then 3/4 u_n + 1/4 (u^(1) + dt·R(u^(1))), project  (stage 2)
       1/3 u_n + 2/3 (u^(2) + dt·R(u^(2))), project  (stage 3)

For Fourier in x (one kx at a time), state vector s = (u, v, b) on
interior CGL y-nodes (Dirichlet walls).  The operator M(dt) maps s_n → s_{n+1};
complex eigenvalues λ of M give numerical ω via λ = exp(-i ω_num · dt),
so ω_num = -arg(λ) / dt.  Compare ω² with the analytic formula.
"""
from __future__ import annotations
import argparse
import numpy as np
import scipy.linalg


def trefethen_D(N):
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = (np.outer(c, 1.0 / c)) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def cgl_asc(ny, Ly):
    N = ny - 1
    D, x = trefethen_D(N)
    idx = np.arange(N, -1, -1)
    y = (1.0 + x[idx]) * Ly / 2.0
    D_asc = (2.0 / Ly) * D[np.ix_(idx, idx)]
    return y, D_asc


def build_poisson_inverse(ny, Ly, kx_phys):
    """Return the matrix P_inv such that π = P_inv · RHS, where the Poisson
    problem is (∂yy - k_x²) π = RHS with Dirichlet π at both walls.

    Uses CUDA's SL diagonalisation: eigenmodes ψ_n = sin(n π y / Ly) on
    CGL grid (interior-node eigenproblem of -D²).
    """
    y, D = cgl_asc(ny, Ly)
    D2 = D @ D
    interior = slice(1, ny - 1)
    # Operator -D² on interior (Dirichlet): eigendecomp = SL basis
    A = -D2[interior, interior]
    mu, V = np.linalg.eig(A)
    mu = mu.real
    V = V.real
    order = np.argsort(mu)
    mu = mu[order]
    V = V[:, order]
    # Normalise CC-weighted
    # CC weights on ascending CGL
    N = ny - 1
    w = np.zeros(ny)
    # Trefethen SMMML Ch.12 weights
    # Standard formula; simpler: integrate |ψ|² via trapezoid for our purposes
    # (here we only use ψ·ψᵀ inverse, so normalisation cancels)
    # Just L2-normalise
    for i in range(V.shape[1]):
        n = np.sqrt(np.sum(V[:, i] ** 2))
        if n > 0:
            V[:, i] /= n

    # Extend to full ny with zeros at boundaries
    Psi = np.zeros((ny, len(mu)))
    Psi[interior, :] = V

    # Poisson inverse for (-D² + k_x²):  invoke sin basis with eigenvalues μ_n + k_x²
    denom = mu + kx_phys ** 2
    # P = Ψ diag(-1/denom) Ψᵀ  (matches CUDA sign convention: q = -g/denom)
    P_full = Psi @ np.diag(-1.0 / denom) @ Psi.T
    return y, P_full


def rhs_linear(u, v, b, dt, D, D_poisson_inv, kx_phys, N2):
    """Apply CUDA's linearised RHS: ∂t u = 0, ∂t v = b, ∂t b = -N²·v.
    Project (u, v) to ∇·u = 0 after stepping.
    Returns the stepped (u_new, v_new, b_new).
    """
    # Euler stage: u_star, v_star, b_star
    u_star = u.copy()
    v_star = v + dt * b
    b_star = b - dt * N2 * v

    # Projection: solve (∂yy - k_x²) π = ∂x u_star + ∂y v_star
    # ∂x on (kx Fourier): → i kx · u_star     (linear in complex, but we keep
    # real/imag components via real representation: u = u_R cos + u_I sin,
    # same for v, etc.  Here stick to complex-valued state, real eigenvalues
    # of projection are preserved.)
    divergence = 1j * kx_phys * u_star + D @ v_star
    pi = D_poisson_inv @ divergence

    u_new = u_star - 1j * kx_phys * pi
    v_new = v_star - D @ pi

    # Wall BC: v = 0 at endpoints  (not applied here because we work on
    # interior nodes only — extension code below assumes `u, v, b` are length
    # ny-2 vectors on interior nodes)

    return u_new, v_new, b_star


def shu_osher_step(s, dt, op_args):
    """One RK3 Shu-Osher step using rhs_linear above.  s = (u, v, b)."""
    u, v, b = s
    u0, v0, b0 = u.copy(), v.copy(), b.copy()
    # Stage 1:  y^(1) = y + dt · R(y)
    u1, v1, b1 = rhs_linear(u, v, b, dt, *op_args)
    # Stage 2:  y^(2) = 3/4 y + 1/4 (y^(1) + dt · R(y^(1)))
    u2_rhs, v2_rhs, b2_rhs = rhs_linear(u1, v1, b1, dt, *op_args)
    u2 = 0.75 * u0 + 0.25 * u2_rhs
    v2 = 0.75 * v0 + 0.25 * v2_rhs
    b2 = 0.75 * b0 + 0.25 * b2_rhs
    # Stage 3
    u3_rhs, v3_rhs, b3_rhs = rhs_linear(u2, v2, b2, dt, *op_args)
    un = (1.0 / 3.0) * u0 + (2.0 / 3.0) * u3_rhs
    vn = (1.0 / 3.0) * v0 + (2.0 / 3.0) * v3_rhs
    bn = (1.0 / 3.0) * b0 + (2.0 / 3.0) * b3_rhs
    return un, vn, bn


def build_step_matrix(ny, Ly, kx_phys, N2, dt):
    """Assemble the discrete RK3+projection operator M for a single kx mode.
    State vector: (u, v, b) stacked on interior CGL nodes, complex.
    Returns M of shape (3*(ny-2), 3*(ny-2)).
    """
    y, D = cgl_asc(ny, Ly)
    D_int = D[1:ny-1, 1:ny-1]
    _, P = build_poisson_inverse(ny, Ly, kx_phys)
    P_int = P[1:ny-1, 1:ny-1]
    op_args = (D_int, P_int, kx_phys, N2)

    nint = ny - 2
    dim = 3 * nint
    M = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        s = (e[0:nint], e[nint:2*nint], e[2*nint:3*nint])
        u_new, v_new, b_new = shu_osher_step(s, dt, op_args)
        col = np.concatenate([u_new, v_new, b_new])
        M[:, j] = col
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny",   type=int, default=64)
    ap.add_argument("--Ly",   type=float, default=1.0)
    ap.add_argument("--Lx",   type=float, default=1.0)
    ap.add_argument("--kx",   type=int, default=1)
    ap.add_argument("--N2",   type=float, default=1.0)
    ap.add_argument("--dt",   type=float, default=1e-4)
    ap.add_argument("--n_show", type=int, default=5)
    args = ap.parse_args()

    kx_phys = args.kx * 2 * np.pi / args.Lx

    print(f"Assembling CUDA-equivalent RK3+projection step matrix (complex)...")
    print(f"  ny={args.ny}, Ly={args.Ly}, Lx={args.Lx}, kx_int={args.kx}")
    print(f"  kx_phys={kx_phys:.4f}, N²={args.N2}, dt={args.dt}")
    M = build_step_matrix(args.ny, args.Ly, kx_phys, args.N2, args.dt)

    eigs = scipy.linalg.eigvals(M)
    # For eigenvalue λ = |λ| e^{-iϕ} of a one-step oscillator, ω_num·dt = ϕ.
    # (We use convention  u(t+dt) = λ u(t), so λ ≈ 1 - iω·dt + O(dt²).)
    mags = np.abs(eigs)
    phases = -np.angle(eigs)  # positive ω convention
    omega = phases / args.dt

    # Filter: physical modes have |λ|≈1 (marginally stable RK3 is |λ|<1 for viscous)
    good = (mags > 0.9) & (mags < 1.1) & (omega > 0)
    om = omega[good]
    mg = mags[good]
    # Sort by descending ω² so n=1 first
    order = np.argsort(-om)
    om = om[order]
    mg = mg[order]

    # Dedup close pairs (complex eigenvalues come in conjugates)
    kept_omega = []
    kept_mag = []
    for o, m in zip(om, mg):
        if any(abs(o - k) / max(abs(k), 1e-30) < 1e-4 for k in kept_omega):
            continue
        kept_omega.append(o)
        kept_mag.append(m)

    print()
    print(f"{'n':>3}  {'ω_discrete':>14}  {'ω²':>14}  {'|λ|':>10}  {'ω²_analytic':>14}  {'rel err':>10}")
    print("-" * 84)
    for n, (o, m) in enumerate(zip(kept_omega[:args.n_show], kept_mag[:args.n_show]), 1):
        ky = n * np.pi / args.Ly
        omega_sq_exact = args.N2 * kx_phys ** 2 / (kx_phys ** 2 + ky ** 2)
        rel = abs(o ** 2 - omega_sq_exact) / omega_sq_exact
        print(f"{n:>3}  {o:14.8e}  {o**2:14.8e}  {m:10.6f}  {omega_sq_exact:14.8e}  {rel:10.3e}")


if __name__ == "__main__":
    main()
