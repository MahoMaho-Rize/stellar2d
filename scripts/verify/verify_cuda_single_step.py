#!/usr/bin/env python3
"""Compare CUDA and Python reference for a SINGLE linear RK3+projection step.

Setup:
  * ρ₀ = 1 (Boussinesq-stratified), constant N².
  * IC: u=0, v=0, b = amp · sin(k_x·x) · sin(k_y·π·y/Ly).
  * Same CGL grid, Trefethen D, SL basis ψ_n = (-D²)-interior-eigenvectors.
  * Project IC; run one RK3 step (dt_max); download u, v, b.

Compares CUDA output (from kh_final.csv-like dump) to analytic prediction
for a single timestep.  Expected: after 1 step of Shu-Osher with small dt,
v should evolve as the eigenmode rotation: v(dt) ≈ 0 + dt·ω·B₀·shape + O(dt²).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.linalg


def trefethen_D(N):
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


def sl_basis_cc(ny, Ly):
    """Return SL eigenvalues + CC-normalised ψ_n, on full (ny,) grid."""
    y, D = cgl_asc(ny, Ly)
    D2 = D @ D
    interior = slice(1, ny - 1)
    A = -D2[interior, interior]
    mu, V = np.linalg.eig(A)
    mu = mu.real
    V = V.real
    order = np.argsort(mu)
    mu = mu[order]
    V = V[:, order]
    # Extend to full grid (zeros at boundaries)
    Psi = np.zeros((ny, len(mu)))
    Psi[interior, :] = V
    # CC weights
    N = ny - 1
    xc = np.cos(np.pi * np.arange(N + 1) / N)   # descending
    # Trefethen Ch. 12 weights (clenshaw-curtis)
    w_raw = np.zeros(N + 1)
    K = np.arange(1, N // 2 + 1)
    for k in range(N + 1):
        s = 1.0 - sum(2 * np.cos(2 * kk * k * np.pi / N) / (4 * kk ** 2 - 1) for kk in K)
        if k == 0 or k == N:
            s *= 1.0 / N
        else:
            s *= 2.0 / N
        w_raw[k] = s
    # Map to ascending + scale to [0, Ly]
    idx = np.arange(N, -1, -1)
    w_asc = w_raw[idx] * Ly / 2.0
    # Normalise ψ_n CC-weighted
    for n in range(Psi.shape[1]):
        norm = np.sqrt(np.sum(w_asc * Psi[:, n] ** 2))
        if norm > 0:
            Psi[:, n] /= norm
    return y, D, mu, Psi, w_asc


def poisson_solve_cc(rhs_phys, kx_phys, Psi, mu, w_asc):
    """Solve (∂yy - kx²) π = rhs.  rhs and π are physical values on CGL grid.
    Uses the CC-weighted SL expansion: π = Σ (-RHS_n / (μ_n + kx²)) ψ_n."""
    # Forward: rhs_n = Σ_y w_asc · ψ_n · rhs
    rhs_n = (Psi * w_asc[:, None]).T @ rhs_phys
    pi_n = -rhs_n / (mu + kx_phys ** 2)
    pi_phys = Psi @ pi_n
    return pi_phys


def linear_rhs(u_vec, v_vec, b_vec, kx_phys, D, N2):
    """Evaluate linearised RHS for each field (no advection, no viscosity).

    u, v, b are length-ny arrays (one column from the 2D field at this kx).
    They are the Fourier coefficients at kx.
    Returns ru, rv, rb — all ny-long.
    """
    ru = np.zeros_like(u_vec)
    rv = b_vec.copy()
    rb = -N2 * v_vec
    return ru, rv, rb


def project(u, v, kx_phys, D, Psi, mu, w_asc):
    """Chorin projection.  u, v are physical at one kx."""
    # divergence = ikx u + D v
    div = 1j * kx_phys * u + D @ v
    pi = poisson_solve_cc(div, kx_phys, Psi, mu, w_asc)
    u_new = u - 1j * kx_phys * pi
    v_new = v - D @ pi
    # Dirichlet walls
    v_new[0] = 0.0
    v_new[-1] = 0.0
    u_new[0] = 0.0
    u_new[-1] = 0.0
    return u_new, v_new


def rk3_step(u, v, b, dt, kx_phys, D, Psi, mu, w_asc, N2):
    u0, v0, b0 = u.copy(), v.copy(), b.copy()
    # Stage 1
    ru, rv, rb = linear_rhs(u, v, b, kx_phys, D, N2)
    u1 = u + dt * ru
    v1 = v + dt * rv
    b1 = b + dt * rb
    u1, v1 = project(u1, v1, kx_phys, D, Psi, mu, w_asc)
    # Stage 2
    ru, rv, rb = linear_rhs(u1, v1, b1, kx_phys, D, N2)
    u2 = 0.75 * u0 + 0.25 * (u1 + dt * ru)
    v2 = 0.75 * v0 + 0.25 * (v1 + dt * rv)
    b2 = 0.75 * b0 + 0.25 * (b1 + dt * rb)
    u2, v2 = project(u2, v2, kx_phys, D, Psi, mu, w_asc)
    # Stage 3
    ru, rv, rb = linear_rhs(u2, v2, b2, kx_phys, D, N2)
    un = (1/3) * u0 + (2/3) * (u2 + dt * ru)
    vn = (1/3) * v0 + (2/3) * (v2 + dt * rv)
    bn = (1/3) * b0 + (2/3) * (b2 + dt * rb)
    un, vn = project(un, vn, kx_phys, D, Psi, mu, w_asc)
    return un, vn, bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny",   type=int, default=128)
    ap.add_argument("--Ly",   type=float, default=1.0)
    ap.add_argument("--Lx",   type=float, default=1.0)
    ap.add_argument("--kx",   type=int, default=1)
    ap.add_argument("--ky",   type=int, default=1)
    ap.add_argument("--N2",   type=float, default=1.0)
    ap.add_argument("--amp",  type=float, default=1e-3)
    ap.add_argument("--dt",   type=float, default=1.17e-4)
    ap.add_argument("--nsteps", type=int, default=1)
    args = ap.parse_args()

    kx_phys = args.kx * 2 * np.pi / args.Lx
    ky_phys = args.ky * np.pi / args.Ly
    y, D, mu, Psi, w_asc = sl_basis_cc(args.ny, args.Ly)

    # Python EVP analytic freq
    omega_analytic = np.sqrt(args.N2 * kx_phys**2 / (kx_phys**2 + ky_phys**2))

    # IC at kx: v=u=0,  b = amp · sin(ky π y / Ly).  The sin(kx x) in x
    # becomes "b_hat at kx_int=kx (imag amp only in R2C convention)".
    # But we work per-kx: state arrays are complex ny-long representing
    # Fourier coefficient at kx.
    b_shape = args.amp * np.sin(ky_phys * y)
    u = np.zeros(args.ny, dtype=complex)
    v = np.zeros(args.ny, dtype=complex)
    b = b_shape.astype(complex)   # real; in R2C at kx=1 it'd be a pure imag,
                                   # but absolute amp doesn't change dispersion.

    # Project IC
    u, v = project(u, v, kx_phys, D, Psi, mu, w_asc)

    print(f"{'Python RK3+projection single-mode reference':^72}")
    print("=" * 72)
    print(f"  ny={args.ny}, kx_phys={kx_phys:.4f}, ky_phys={ky_phys:.4f}, N²={args.N2}")
    print(f"  analytic ω = {omega_analytic:.6f},  ω² = {omega_analytic**2:.6f}")
    print(f"  dt = {args.dt:.3e}, nsteps = {args.nsteps}")
    print()

    # Evolve + log probe
    probe = []
    for step in range(args.nsteps):
        u, v, b = rk3_step(u, v, b, args.dt, kx_phys, D, Psi, mu, w_asc, args.N2)
        probe.append((args.dt * (step + 1), np.real(v[args.ny // 2])))
    probe = np.array(probe)
    # FFT to extract ω
    if len(probe) > 100:
        t_arr = probe[:, 0]
        v_arr = probe[:, 1] - probe[:, 1].mean()
        window = np.hanning(len(v_arr))
        V = np.fft.rfft(v_arr * window)
        freqs = np.fft.rfftfreq(len(t_arr), d=args.dt)
        omega_arr = 2 * np.pi * freqs
        power = np.abs(V) ** 2
        mask = omega_arr > 0
        peak = np.argmax(power[mask])
        omega_peak = omega_arr[mask][peak]
        print(f"  Python FFT peak:  ω = {omega_peak:.4f}, ω² = {omega_peak**2:.4f}")
        print(f"                    rel err vs analytic = "
              f"{abs(omega_peak**2 - omega_analytic**2)/omega_analytic**2:.3e}")

    t_final = args.dt * args.nsteps
    # Probe at y=Ly/2
    jy = args.ny // 2
    print(f"  after {args.nsteps} steps (t={t_final:.4e}):")
    print(f"    |u|_max = {np.max(np.abs(u)):.4e}")
    print(f"    |v|_max = {np.max(np.abs(v)):.4e}")
    print(f"    |b|_max = {np.max(np.abs(b)):.4e}")
    print(f"    v[jy={jy}] = {v[jy]}")
    print(f"    b[jy={jy}] = {b[jy]}")

    # Expected from eigenmode analysis: starting from b=B₀·sin(ky y), v=0,
    # the eigenmode rotation gives
    #   v(t) = (B₀/ω) sin(ω t) · sin(ky y)
    #   b(t) = B₀ cos(ω t) · sin(ky y)
    # (complex-mode picture; real sign conventions may flip)
    ky_shape = np.sin(ky_phys * y[jy])
    v_pred = args.amp / omega_analytic * np.sin(omega_analytic * t_final) * ky_shape
    b_pred = args.amp * np.cos(omega_analytic * t_final) * ky_shape
    print()
    print(f"    analytic v[jy] = {v_pred:.6e}   (from eigenmode oscillator)")
    print(f"    analytic b[jy] = {b_pred:.6e}")


if __name__ == "__main__":
    main()
