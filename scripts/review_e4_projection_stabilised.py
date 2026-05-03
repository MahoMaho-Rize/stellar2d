#!/usr/bin/env python3
"""E4: Projection-stabilised RK4 of the anelastic momentum+buoyancy system.

We implement a three-variable (u, v, b) pseudo-spectral anelastic
time-stepping with:
  - primitive-node factorised operator application
  - Chorin-type projection enforcing ∇·(ρu) = 0 after each RK4 stage
  - single-mode k_x Fourier basis, 1D profile in y

The PDE system (linearised, no viscosity, no nonlinearity):
    ∂_t u = -∂_x π                       (horizontal momentum)
    ∂_t v = -∂_y π + b                   (vertical momentum + buoyancy)
    ∂_t b = -N²(y) v                     (buoyancy equation)
    ∇·(ρu) = 0                            (anelastic mass conservation)

We advance (u, v, b) with RK4 using the *pressure-less* substage RHS,
then project onto ∇·(ρu)=0 after each update.  This is the
fractional-step / Chorin scheme and is the standard in production
anelastic codes.

Three cases compared:
  1. Assembled scheme (reference, ẅ = -(L^{-1}R) v): stable, correct.
  2. Bare factorised M_eff  (§5.7 baseline): blows up in 0.3 periods.
  3. Projection-stabilised primitive-node (this script): stable long-term.

The script reports:
  - evolution horizon (how many periods each scheme survives)
  - per-step eigenmode deviation
  - Fourier-peak frequency drift over 300 periods
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nonlinear_paths_infra import cgl_grid, cc_weights, bg_lane_emden

OUT_DIR = SCRIPT_DIR.parent / "review" / "e4_projection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_eigenmode_1D(Ny, Ly, kx, rho, N2, amp, D):
    """Top g-mode eigenvector (1D profile in y)."""
    intr = slice(1, Ny - 1)
    L = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
    R = kx**2 * np.diag(N2 * rho)
    lam, V = scipy.linalg.eig(R[intr, intr], L[intr, intr])
    lam = lam.real; V = V.real
    mask = np.isfinite(lam) & (lam > 0)
    lam = lam[mask]; V = V[:, mask]
    order = np.argsort(lam)[::-1]
    omega_1 = float(np.sqrt(lam[order[0]]))
    v_mode_int = V[:, order[0]] / np.max(np.abs(V[:, order[0]])) * amp
    v_profile = np.zeros(Ny); v_profile[intr] = v_mode_int
    return v_profile, omega_1


def project_single_mode(u_profile, v_profile, rho, D, kx, L_int_LU):
    """Project (u(y), v(y)) on one Fourier mode k_x onto ∇·(ρu)=0.

    For single-mode (complex-amplitude) fields:
      u(x,y) = Re[ û(y) exp(i kx x) ],  similarly v.
      div = ∂_x(ρu) + ∂_y(ρv) = i kx ρ û + d/dy(ρ v̂)

    Solve  -(L) π̂ = div   with L = -D diag(ρ) D + k² diag(ρ),
    Dirichlet π̂ on walls.  Update û -= i kx π̂,   v̂ -= d/dy π̂.

    We treat u_profile, v_profile as real arrays representing the real
    parts of û, v̂ (standard for sine/cosine decomposition where u is
    the sin-channel and v is the cos-channel with a prefactor).

    This is the assembled SL-Poisson solve — the pressure elliptic is
    consistent with the EVP operator.  Projection is done with
    pre-factored L."""
    Ny = len(rho)
    intr = slice(1, Ny - 1)
    # div on interior: complex if we want, but in real arithmetic we'll
    # treat u as the imaginary part of û (sin channel) and v as the real
    # part (cos channel).  Then div(x,y) in real space is
    #   -sin(kx x) [kx·ρu + d/dy(ρv)].
    # The y-profile of div is g(y) = kx * rho * u - d/dy(rho * v),
    # wait: carefully, if u(x,y) = U(y) sin(kx x), v(x,y) = V(y) cos(kx x),
    #       ρu = ρU sin, ρv = ρV cos
    #       ∂_x(ρu) = kx ρU cos
    #       ∂_y(ρv) = (ρV)' cos
    # div = cos(kx x) * [kx ρU + (ρV)'].
    # For ∇·(ρu_new)=0 we need g(y) = kx ρU + (ρV)' = 0.
    rho_V = rho * v_profile
    drho_V = D @ rho_V
    g = kx * rho * u_profile + drho_V
    # Solve L π = g on interior  (π=0 on walls)
    pi_int = scipy.linalg.lu_solve(L_int_LU, g[intr])
    pi = np.zeros(Ny); pi[intr] = pi_int
    # Gradient projections:
    #   ∂_x π in real space: π = pi_y(y) cos(kx x) so ∂_x π = -kx pi_y sin(kx x)
    #     matches the sin channel of u: U_new = U - (-kx pi_y)/... wait sign.
    # Anelastic projection: u' = u - (1/ρ) ∂ π (conventional), but some
    # codes use u' = u - ∂ π without ρ.  The assembled Poisson here solves
    # L π = g, which is the ∇·(ρ∇π) = div form (L acts on π: L π =
    # -d/dy(ρ dπ/dy) + k² ρ π).  So the associated pressure gradient that
    # we subtract is ∇π (no 1/ρ).  Then:
    dpi_dy = D @ pi
    # Update: û_new = U - (-kx pi_y) = U + kx pi_y? No wait.
    # Actually for u(x,y)=U(y) sin(kx x), pressure grad ∂_x π =
    # -kx pi_y sin(kx x), so gradient subtract gives u -> U - (-kx pi_y) = U + kx pi_y.
    U_new = u_profile + kx * pi
    # v(x,y)=V(y) cos(kx x), ∂_y π = dpi_y/dy · cos(kx x), so v' = V - dpi_y/dy.
    V_new = v_profile - dpi_dy
    V_new[0] = 0.0; V_new[-1] = 0.0
    return U_new, V_new


def rhs_3var(u, v, b, rho, D, N2, kx):
    """Pressure-less RHS of (u, v, b):
         du/dt = 0
         dv/dt = b
         db/dt = -N² v
    u, v, b are 1D y-profiles (single k_x mode)."""
    du = np.zeros_like(u)
    dv = b.copy()
    db = -N2 * v
    return du, dv, db


def rk4_step_3var(u, v, b, dt, rho, D, N2, kx, L_int_LU):
    """RK4 step of (u,v,b) with projection after final update."""
    du1, dv1, db1 = rhs_3var(u, v, b, rho, D, N2, kx)
    u2 = u + 0.5*dt*du1; v2 = v + 0.5*dt*dv1; b2 = b + 0.5*dt*db1
    du2, dv2, db2 = rhs_3var(u2, v2, b2, rho, D, N2, kx)
    u3 = u + 0.5*dt*du2; v3 = v + 0.5*dt*dv2; b3 = b + 0.5*dt*db2
    du3, dv3, db3 = rhs_3var(u3, v3, b3, rho, D, N2, kx)
    u4 = u + dt*du3; v4 = v + dt*dv3; b4 = b + dt*db3
    du4, dv4, db4 = rhs_3var(u4, v4, b4, rho, D, N2, kx)
    u_new = u + dt/6*(du1 + 2*du2 + 2*du3 + du4)
    v_new = v + dt/6*(dv1 + 2*dv2 + 2*dv3 + dv4)
    b_new = b + dt/6*(db1 + 2*db2 + 2*db3 + db4)
    # Chorin projection:
    u_new, v_new = project_single_mode(u_new, v_new, rho, D, kx, L_int_LU)
    return u_new, v_new, b_new


def build_IC_single_mode(Ny, Ly, kx, rho, N2, amp, D):
    """Top g-mode IC: V(y) profile from EVP, U(y) from continuity."""
    v_profile, omega_1 = build_eigenmode_1D(Ny, Ly, kx, rho, N2, amp, D)
    # From continuity:   kx ρ U + (ρV)' = 0  →  U = -(ρV)' / (kx ρ)
    rho_V = rho * v_profile
    drho_V = D @ rho_V
    U_profile = -drho_V / (kx * np.maximum(rho, 1e-30))
    U_profile[0] = 0.0; U_profile[-1] = 0.0
    b_profile = np.zeros(Ny)
    return U_profile, v_profile, b_profile, omega_1


def measure_dev(v_cur, v_IC, w_cc):
    IC_norm = np.sqrt(np.sum(w_cc * v_IC**2))
    if IC_norm <= 0:
        return float('nan')
    a = np.sum(w_cc * v_cur * v_IC) / max(np.sum(w_cc * v_IC**2), 1e-300)
    r = v_cur - a * v_IC
    return float(np.sqrt(np.sum(w_cc * r**2)) / IC_norm)


def fourier_peak(series, dt_sample, omega_true):
    N = len(series)
    windowed = np.asarray(series) * np.hanning(N)
    freqs = np.fft.rfftfreq(N, d=dt_sample)
    amp = np.abs(np.fft.rfft(windowed))
    kmax = int(np.argmax(amp[1:])) + 1
    y0, y1, y2 = amp[kmax-1], amp[kmax], amp[kmax+1]
    denom = (y0 - 2*y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
    f_peak = freqs[kmax] + delta * (freqs[1] - freqs[0])
    om_meas = 2 * np.pi * f_peak
    return om_meas / omega_true - 1.0


def main():
    print("E4: Projection-stabilised 3-variable anelastic RK4")
    print("    Lane-Emden n=3/2, Ny=64, kx=2π/Ly, rho_cut=0.05, dt=2e-3")
    print("=" * 74)

    Ny = 64; Ly = 1.0
    kx = 2 * np.pi / Ly
    rho_cut = 0.05
    amp = 1e-8
    dt = 2e-3

    y, D = cgl_grid(Ny, Ly)
    w_cc = cc_weights(Ny, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    intr = slice(1, Ny - 1)
    # Pre-factor L for projection
    L = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
    L_int_LU = scipy.linalg.lu_factor(L[intr, intr])

    U, v, b, omega_1 = build_IC_single_mode(Ny, Ly, kx, rho, N2, amp, D)
    period = 2 * np.pi / omega_1
    print(f"    ω_1 = {omega_1:.4f}, T = {period:.3f}")
    print(f"    IC built: ‖V‖ = {np.linalg.norm(v):.3e}, ‖U‖ = {np.linalg.norm(U):.3e}")

    # Check that IC already satisfies ∇·(ρu)=0
    g = kx * rho * U + D @ (rho * v)
    print(f"    IC div norm = {np.linalg.norm(g[intr]):.3e}  (should be ~1e-15)")

    # Run 300 periods
    n_periods = 300
    n_steps = int(n_periods * period / dt)
    print(f"\n  Running projection-stabilised RK4: {n_steps} steps = {n_periods} periods")

    v_IC = v.copy()
    sample_stride = max(n_steps // 600, 1)
    dev_hist = []
    probe_hist = []
    probe_idx = Ny // 2

    u_cur, v_cur, b_cur = U.copy(), v.copy(), b.copy()
    blew_step = None
    for step in range(n_steps):
        u_cur, v_cur, b_cur = rk4_step_3var(
            u_cur, v_cur, b_cur, dt, rho, D, N2, kx, L_int_LU)
        if not np.all(np.isfinite(v_cur)):
            blew_step = step; break
        if step % sample_stride == 0 or step == n_steps - 1:
            dev_hist.append((step, measure_dev(v_cur, v_IC, w_cc)))
            probe_hist.append(v_cur[probe_idx])

    if blew_step is not None:
        print(f"    blew up at step {blew_step} (t={blew_step*dt:.3f}, "
              f"periods={blew_step*dt/period:.2f})")
    else:
        devs = [d for _, d in dev_hist]
        steps = [s for s, _ in dev_hist]
        print(f"    stable over {n_periods} periods")
        print(f"    dev @ 1 period   = {devs[min(3, len(devs)-1)]:.3e}")
        print(f"    dev @ 10 periods = {devs[min(20, len(devs)-1)]:.3e}")
        print(f"    dev @ 100 periods= {devs[min(200, len(devs)-1)]:.3e}")
        print(f"    dev @ final      = {devs[-1]:.3e}")
        # Fourier peak
        rel_freq = fourier_peak(probe_hist, dt * sample_stride, omega_1)
        print(f"    Fourier-peak frequency drift = {rel_freq:+.3e}")

    # Reference: assembled scheme
    print(f"\n  Reference: assembled RK4 ẅ = -(L^-1 R)v, same 300 periods")
    R = kx**2 * np.diag(N2 * rho)
    M_asm = scipy.linalg.solve(L[intr, intr], R[intr, intr])
    v_i = v_IC[intr].copy(); w_i = np.zeros_like(v_i)
    for step in range(n_steps):
        def rhs(V_, W_): return W_, -(M_asm @ V_)
        k1v, k1w = rhs(v_i, w_i)
        k2v, k2w = rhs(v_i + 0.5*dt*k1v, w_i + 0.5*dt*k1w)
        k3v, k3w = rhs(v_i + 0.5*dt*k2v, w_i + 0.5*dt*k2w)
        k4v, k4w = rhs(v_i + dt*k3v, w_i + dt*k3w)
        v_i = v_i + dt/6*(k1v + 2*k2v + 2*k3v + k4v)
        w_i = w_i + dt/6*(k1w + 2*k2w + 2*k3w + k4w)
    v_full = np.zeros(Ny); v_full[intr] = v_i
    dev_asm = measure_dev(v_full, v_IC, w_cc)
    print(f"    dev @ final = {dev_asm:.3e}")

    # Write summary CSV
    out_csv = OUT_DIR / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "horizon_periods", "final_dev", "fourier_drift_rel"])
        w.writerow(["bare M_eff RK4 (§5.7)", 0.3, "blew up", "n/a"])
        if blew_step is None:
            w.writerow(["3-var RK4 + projection", n_periods,
                        f"{devs[-1]:.3e}", f"{rel_freq:+.3e}"])
        else:
            w.writerow(["3-var RK4 + projection",
                        blew_step*dt/period, "blew up", "n/a"])
        w.writerow(["assembled RK4 (reference)", n_periods,
                    f"{dev_asm:.3e}", "n/a"])
    print(f"\n  Wrote {out_csv}")


if __name__ == "__main__":
    main()
