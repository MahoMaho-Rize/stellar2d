#!/usr/bin/env python3
"""Shared infrastructure for Phase 3 nonlinear TD prototype comparison.

Three Phase-3 nonlinear paths (op-split / IMEX / exponential) all share:
  - CGL grid + Chebyshev D
  - Clenshaw-Curtis weights
  - Lane-Emden n=3/2 background on [0, Ly]
  - 2D anelastic state (V, W=∂_t V, B) on (ny × nx) grid
  - L, R matrices (v-space) per kx
  - Nonlinear advection term f_nl = -(u·∇)v etc. with 2/3 dealias
  - EVP-based IC and 3-wave triad probe

This module only provides building blocks; each path script composes them
into its own time integrator.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp


# ── CGL grid & Chebyshev D  (lifted from full_galerkin_closure_test.py) ─
def cgl_grid(ny, Ly):
    N = ny - 1
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    idx = np.arange(N, -1, -1)
    y = (1.0 + x[idx]) * Ly / 2.0
    D = (2.0 / Ly) * D[np.ix_(idx, idx)]
    return y, D


def cc_weights(ny, Ly):
    N = ny - 1
    w = np.zeros(N + 1)
    for k in range(N + 1):
        s = 0.0
        for j in range(1, N // 2 + 1):
            b = 2.0 if (2 * j != N) else 1.0
            s += b / (4.0 * j * j - 1) * np.cos(2.0 * j * k * np.pi / N)
        w[k] = (1.0 - s) * 2.0 / N
    w[0] /= 2.0; w[-1] /= 2.0
    return w[::-1] * Ly / 2.0


# ── Lane-Emden n=3/2 ρ(y), N²(y) ───────────────────────────────────────
def bg_lane_emden(y, Ly, rho_cut=0.01):
    sol = solve_ivp(
        lambda xi, st: [st[1],
                        -max(st[0], 0.0) ** 1.5 - 2 * st[1] / max(xi, 1e-8)],
        [1e-6, 4.0], [1.0, 0.0], dense_output=True,
        events=lambda t, st: st[0], max_step=1e-3,
    )
    xi_s = sol.t_events[0][0]
    xi = np.linspace(1e-6, xi_s, 4000)
    rho_full = np.clip(sol.sol(xi)[0], 0, None) ** 1.5
    mask = rho_full > rho_cut
    xi_lo, xi_hi = xi[mask].min(), xi[mask].max()
    xi_q = xi_lo + (y / Ly) * (xi_hi - xi_lo)
    rho = np.clip(np.interp(xi_q, xi, rho_full), rho_cut, None)
    drho = np.gradient(rho, y, edge_order=2)
    N2 = np.maximum(-drho / rho, 0.0)
    return rho, N2


# ── Per-kx assembled operator M = L⁻¹ R  (v-space) ────────────────────
def assemble_M_per_kx(y, D, rho, N2, kx_array):
    """Returns list of M (n_int × n_int) matrices, one per kx in kx_array.
    For k=0 (dc), returns zero matrix (no g-mode dynamics)."""
    ny = len(y)
    n_int = ny - 2
    M_list = []
    intr = slice(1, ny - 1)
    for kx in kx_array:
        if abs(kx) < 1e-14:
            M_list.append(np.zeros((n_int, n_int)))
            continue
        L = -D @ (np.diag(rho) @ D) + kx ** 2 * np.diag(rho)
        R = kx ** 2 * np.diag(N2 * rho)
        Li = L[intr, intr]
        Ri = R[intr, intr]
        M_list.append(scipy.linalg.solve(Li, Ri))
    return M_list


def evp_per_kx(y, D, rho, N2, kx_array):
    """Returns list of (omega2, V_modes) per kx (descending omega2)."""
    out = []
    intr = slice(1, len(y) - 1)
    for kx in kx_array:
        if abs(kx) < 1e-14:
            out.append((np.array([]), np.zeros((len(y) - 2, 0))))
            continue
        L = -D @ (np.diag(rho) @ D) + kx ** 2 * np.diag(rho)
        R = kx ** 2 * np.diag(N2 * rho)
        Li = L[intr, intr]; Ri = R[intr, intr]
        lam, V = scipy.linalg.eig(Ri, Li)
        lam = np.real(lam); V = np.real(V)
        mask = np.isfinite(lam) & (lam > 0)
        lam, V = lam[mask], V[:, mask]
        order = np.argsort(lam)[::-1]
        out.append((lam[order], V[:, order]))
    return out


# ── FFT helpers for x direction ──────────────────────────────────────
def fft_x(arr):
    """Real FFT along x (axis=1).  arr shape (ny, nx) → (ny, nh) complex."""
    return np.fft.rfft(arr, axis=1)


def ifft_x(arr_hat, nx):
    """Inverse real FFT: (ny, nh) → (ny, nx) real."""
    return np.fft.irfft(arr_hat, n=nx, axis=1)


def dealias_23(arr_hat, nh):
    """Zero out top 1/3 x-modes (2/3 dealias rule)."""
    kcut = (2 * (nh - 1)) // 3
    out = arr_hat.copy()
    out[:, kcut + 1:] = 0.0
    return out


# ── Nonlinear advection RHS (physical space + 2/3 dealias) ───────────
def compute_advection(u, v, b, D, kx_array, nx):
    """Returns (adv_u, adv_v, adv_b) = -(u·∇)(u, v, b) with 2/3 dealias.

    u, v, b: (ny, nx) row-major physical fields.
    kx_array: (nh,) physical wavenumbers (kx[k] = k·2π/Lx).
    """
    ny = u.shape[0]
    nh = kx_array.shape[0]

    def d_dx(f):
        fhat = fft_x(f)
        fhat = dealias_23(fhat, nh)
        dfhat = 1j * kx_array[None, :] * fhat
        return ifft_x(dfhat, nx)

    def d_dy(f):
        return D @ f  # D is (ny, ny), acts on axis 0

    adv_u = -(u * d_dx(u) + v * d_dy(u))
    adv_v = -(u * d_dx(v) + v * d_dy(v))
    adv_b = -(u * d_dx(b) + v * d_dy(b))
    return adv_u, adv_v, adv_b


def pressure_project(u, v, rho, D, kx_array, nx):
    """Solve ∇·(ρ∇π) = ∇·(ρu) per kx, return (u - ∂x π, v - ∂y π) with
    v(wall) = 0 enforced.  Pure linear-algebra per kx (assembled L_π).
    Used to clean the divergence after advection step."""
    ny = u.shape[0]
    nh = kx_array.shape[0]
    n_int = ny - 2
    intr = slice(1, ny - 1)

    uhat = fft_x(u)
    vhat = fft_x(v)
    # div in spectral space per kx: ik·ρ·û + ∂y(ρ v̂)  (anelastic)
    # rho is (ny,), broadcast
    rho_u_hat = (rho[:, None] * uhat)
    rho_v = (rho[:, None] * vhat)
    d_rho_v_dy = D @ rho_v
    div_hat = 1j * kx_array[None, :] * rho_u_hat + d_rho_v_dy
    # Solve per-kx: -(D(ρD) - k² ρ) π̂ = div → π̂ on interior
    pi_hat = np.zeros_like(uhat)
    for k in range(nh):
        kx = kx_array[k]
        if abs(kx) < 1e-14:
            continue
        L = -D @ (np.diag(rho) @ D) + kx ** 2 * np.diag(rho)
        Li = L[intr, intr]
        rhs = div_hat[intr, k]
        sol = scipy.linalg.solve(Li, rhs)
        pi_hat[intr, k] = sol  # walls = 0

    # ∂x π, ∂y π, subtract
    dxpi_hat = 1j * kx_array[None, :] * pi_hat
    dypi = D @ pi_hat  # still complex, but D is real
    uhat_new = uhat - dxpi_hat
    vhat_new = vhat - dypi
    u_new = ifft_x(uhat_new, nx)
    v_new = ifft_x(vhat_new, nx)
    # Enforce v(wall) = 0 (stress-free + impermeable)
    v_new[0, :] = 0.0; v_new[-1, :] = 0.0
    return u_new, v_new


# ── EVP-based IC on a 2D grid ────────────────────────────────────────
def make_eigenmode_ic(y, rho, N2, D, nx, Lx, kx_int, n_g, amp):
    """Seed V(x, y) = V_EVP(y) · cos(kx·x), B(x, y) = 0, with w = 0.
    u is recovered from continuity (∂x(ρu) + ∂y(ρv) = 0) so the IC is
    already divergence-free.

    Returns (u, v, b, V_EVP_2d, omega).  V_EVP_2d is the clean pure-mode
    reference used to measure deviation later.
    """
    ny = len(y)
    kx_phys = kx_int * 2.0 * np.pi / Lx
    intr = slice(1, ny - 1)
    L = -D @ (np.diag(rho) @ D) + kx_phys ** 2 * np.diag(rho)
    R = kx_phys ** 2 * np.diag(N2 * rho)
    Li = L[intr, intr]; Ri = R[intr, intr]
    lam, V_int = scipy.linalg.eig(Ri, Li)
    lam = np.real(lam); V_int = np.real(V_int)
    mask = np.isfinite(lam) & (lam > 0)
    lam, V_int = lam[mask], V_int[:, mask]
    order = np.argsort(lam)[::-1]
    lam = lam[order]; V_int = V_int[:, order]
    om2 = lam[n_g - 1]
    omega = float(np.sqrt(om2))
    V_y = np.zeros(ny)
    V_y[1:-1] = V_int[:, n_g - 1]
    # normalise
    V_y = V_y / np.max(np.abs(V_y)) * amp
    # 2D: V(x,y) = V_y(y) · cos(kx x)
    x = np.arange(nx) * (Lx / nx)
    cos_kx = np.cos(kx_phys * x)
    V2d = V_y[:, None] * cos_kx[None, :]
    # u from continuity: ∂x(ρu) + ∂y(ρv) = 0
    # ρ u(x,y) = -(1/ikx) ∂y(ρ V̂)  with V̂ at this kx
    rho_V = rho * V_y
    d_rho_V = D @ rho_V
    u_y = np.zeros(ny)
    u_y[1:-1] = -d_rho_V[1:-1] / (rho[1:-1] * kx_phys)
    sin_kx = np.sin(kx_phys * x)
    u2d = u_y[:, None] * sin_kx[None, :]
    # b(x,y) = 0 initially (eigenmode: b = (N²/iω)·v has definite phase; for
    # cosine-based v we choose the initial phase where b=0 and ∂_t v ≠ 0).
    # But using the oscillator with b=0, v=amplitude extreme gives wrong IC
    # for the coupled (v, b) system — better to use exact eigenmode phase.
    # For (V, W=∂_t V, B):  v = V_y cos(ωt), b = -N²/ω · V_y sin(ωt).
    # At t=0: v = V_y, b = 0, w = 0.  But Ẅ = -(M·V) gives ẃ = -MV, which
    # via -N² ρ coupling translates to buoyancy.  The 3-field code is more
    # straightforward — here we seed as (u, v, b=0).
    b2d = np.zeros_like(V2d)
    return u2d, V2d, b2d, V2d.copy(), omega


# ── Measurement ──────────────────────────────────────────────────────
def eigmode_deviation(V_now, V_ref, w_cc, nx):
    """dev = ‖V_now - a V_ref‖_{w_cc,x} / ‖V_ref‖_{w_cc,x}, where a is the
    weighted projection coefficient.  V_now, V_ref shape (ny, nx)."""
    # y-integral via w_cc, x-sum; then ratio is dimensionless.
    num = np.einsum('ij,ij,i->', V_now, V_ref, w_cc) / nx
    den = np.einsum('ij,ij,i->', V_ref, V_ref, w_cc) / nx
    if den <= 0:
        return float('nan')
    a = num / den
    r = V_now - a * V_ref
    num2 = np.einsum('ij,ij,i->', r, r, w_cc) / nx
    return float(np.sqrt(num2 / den))


def total_energy(u, v, b, rho, N2, w_cc, nx):
    """E = ½ ∫ ρ(u²+v²) dx dy + ½ ∫ (b²/N²) dx dy (linearised anelastic).

    Excludes cells where N² ≤ 0 (surface) from the buoyancy term.
    """
    KE = 0.5 * np.einsum('ij,i,i->', u * u + v * v, rho, w_cc) / nx
    N2_safe = np.where(N2 > 1e-12, N2, np.inf)
    PE = 0.5 * np.einsum('ij,i->', b * b / N2_safe[:, None], w_cc) / nx
    return float(KE + PE)


# ── Apply M (assembled, per kx) to v via FFT_x → DGEMM → IFFT_x ───────
def apply_M(v, M_list, kx_array, nx):
    """Returns M·v where v is (ny, nx), M_list is per-kx (n_int, n_int) list."""
    ny = v.shape[0]
    n_int = ny - 2
    vhat = fft_x(v)
    nh = vhat.shape[1]
    out_hat = np.zeros_like(vhat)
    for k in range(nh):
        if M_list[k].shape[0] == 0:
            continue
        # Apply to interior rows of this column
        out_hat[1:-1, k] = M_list[k] @ vhat[1:-1, k]
    # walls zero already
    return ifft_x(out_hat, nx).real
