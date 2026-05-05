#!/usr/bin/env python3
"""Phase 3 Path 3: exponential-integrator nonlinear TD  (v2: extended (V,W,B) state).

Key fix vs v1: include B in the EXACT linear propagator so the -N²·V
buoyancy feedback is not left for the nonlinear block.  Without this,
Path 3's "gold-standard" dev/step floor never materialised (it was
polluted by the Strang error of routing -N²·V through the nonlin block).

Linear ODE block (per kx, per eigenmode of M_kx):
    V̇ = W,  Ẇ = -M·V,  Ḃ = -N²·V
The (V, W) subsystem is closed and decoupled from B.  B is driven by V;
its exact integral in the eigenbasis is:

    ∫₀^t V(τ)dτ = Q · [sin(ωt)/ω · v_mod(0) + (1-cos(ωt))/ω² · w_mod(0)]
    B(t) = B(0) - N²(y) · ∫₀^t V(τ)dτ         (in y-physical space per kx)

For amp → 0 this gives machine-precision dev/step on the (V, W, B) triple,
regardless of ρ(y), N²(y) complexity.

Strang splitting per dt:
    (A) exact linear block, dt/2      → advances (V, W, B)
    (B) nonlinear RK2 step, dt        → only -(u·∇)v, -(u·∇)b.  u derived
                                         from continuity at substeps.  W
                                         kept as "∂_tV|_linear" across (B).
    (C) exact linear block, dt/2

Path 3 has NO CFL constraint from the g-mode frequency (linear block is
exact for any dt).  The only dt limit is advection CFL in the nonlin block.
"""
from __future__ import annotations
import argparse
import numpy as np
import scipy.linalg

from nonlinear_paths_infra import (
    cgl_grid, cc_weights, bg_lane_emden,
    assemble_M_per_kx, make_eigenmode_ic,
    compute_advection,
    eigmode_deviation, total_energy,
    fft_x, ifft_x,
)


# ── Precompute eigendecomposition of each M_kx ────────────────────────
def eigendecompose_M_list(M_list):
    """For each M_kx = L⁻¹R, return (Q, Qinv, sqrt_lam).
    Q columns are eigenvectors (n_int, n_int); sqrt_lam[i] = ω_i for the
    i-th eigenmode.  Eigenvalues are guaranteed positive for physical
    g-modes; spurious negative/zero values are clipped to 1e-30 (Ω=√·
    stays real; those modes carry no signal for a proper g-mode IC).
    """
    Q_list = []; Qinv_list = []; sqrtlam_list = []
    for M in M_list:
        if M.shape[0] == 0:
            Q_list.append(None); Qinv_list.append(None); sqrtlam_list.append(None)
            continue
        lam, Q = scipy.linalg.eig(M)
        lam = np.real(lam)
        lam_clip = np.maximum(lam, 1e-30)
        Qinv = scipy.linalg.inv(Q)
        Q_list.append(Q); Qinv_list.append(Qinv)
        sqrtlam_list.append(np.sqrt(lam_clip))
    return Q_list, Qinv_list, sqrtlam_list


# ── Exact linear propagator on (V, W, B) at one kx ───────────────────
def exact_propagate_kx(vhat_kx, what_kx, bhat_kx,
                       Q, Qinv, omg, N2_int, dt):
    """Inputs: interior vectors (len n_int) in y-physical, x-Fourier mode kx.
    omg = √λ per eigenmode (len n_int).  N2_int = N²(y_int) (len n_int).
    Returns (vhat_new, what_new, bhat_new), all length n_int."""
    if Q is None:
        return vhat_kx, what_kx, bhat_kx
    # Transform V, W to modal coords
    v_mod = Qinv @ vhat_kx
    w_mod = Qinv @ what_kx
    cos_ = np.cos(omg * dt)
    sin_ = np.sin(omg * dt)
    v_new_mod = cos_ * v_mod + (sin_ / omg) * w_mod
    w_new_mod = -omg * sin_ * v_mod + cos_ * w_mod
    # ∫V dτ in modal basis:  sin(ωt)/ω · v₀ + (1-cos(ωt))/ω² · w₀
    int_V_mod = (sin_ / omg) * v_mod + ((1.0 - cos_) / (omg * omg)) * w_mod
    # Back to y-physical space (still per kx)
    vhat_new = Q @ v_new_mod
    what_new = Q @ w_new_mod
    int_V = Q @ int_V_mod
    bhat_new = bhat_kx - N2_int * int_V
    return vhat_new, what_new, bhat_new


def apply_R_full(v, w, b, Q_list, Qinv_list, sqrtlam_list,
                  N2, kx_array, nx, dt):
    """Apply exact linear propagator across all kx, interior-only, walls=0."""
    vhat = fft_x(v); what = fft_x(w); bhat = fft_x(b)
    nh = vhat.shape[1]
    ny = v.shape[0]
    N2_int = N2[1:-1]
    for k in range(nh):
        Q = Q_list[k]
        if Q is None:
            continue
        v_int = vhat[1:-1, k]
        w_int = what[1:-1, k]
        b_int = bhat[1:-1, k]
        vn, wn, bn = exact_propagate_kx(
            v_int, w_int, b_int,
            Q, Qinv_list[k], sqrtlam_list[k], N2_int, dt)
        vhat[1:-1, k] = vn
        what[1:-1, k] = wn
        bhat[1:-1, k] = bn
    # Walls: V, W must be zero (Dirichlet); B is free-slip scalar so keep wall value.
    vhat[0, :] = 0; vhat[-1, :] = 0
    what[0, :] = 0; what[-1, :] = 0
    return ifft_x(vhat, nx), ifft_x(what, nx), ifft_x(bhat, nx)


# ── Rebuild u from v via continuity ────────────────────────────────────
def rebuild_u_from_v(v, rho, D, kx_array, nx):
    vhat = fft_x(v)
    rho_v = rho[:, None] * vhat
    d_rho_v = D @ rho_v
    uhat = np.zeros_like(vhat)
    for k in range(vhat.shape[1]):
        kx = kx_array[k]
        if abs(kx) < 1e-14:
            continue
        uhat[:, k] = -d_rho_v[:, k] / (1j * kx * rho[:])
    return ifft_x(uhat, nx)


# ── Nonlinear block: RK2 midpoint on (v, b) via advection only ────────
def step_rk2_nonlinear(v, b, rho, D, kx_array, nx, dt):
    """RK2 midpoint for (v, b) with ∂_t v = -(u·∇)v, ∂_t b = -(u·∇)b.
    u is derived from v via continuity inside each substep."""
    def f(v_, b_):
        u_ = rebuild_u_from_v(v_, rho, D, kx_array, nx)
        _, adv_v, adv_b = compute_advection(u_, v_, b_, D, kx_array, nx)
        return adv_v, adv_b

    k1v, k1b = f(v, b)
    v_mid = v + 0.5 * dt * k1v
    b_mid = b + 0.5 * dt * k1b
    k2v, k2b = f(v_mid, b_mid)
    v_new = v + dt * k2v
    b_new = b + dt * k2b
    v_new[0, :] = 0.0; v_new[-1, :] = 0.0
    return v_new, b_new


# ── Main driver ────────────────────────────────────────────────────────
def run(args):
    y, D = cgl_grid(args.ny, args.Ly)
    w_cc = cc_weights(args.ny, args.Ly)
    if args.bg == "boussinesq":
        rho = np.ones_like(y); N2 = np.ones_like(y)
    elif args.bg == "lane_emden":
        rho, N2 = bg_lane_emden(y, args.Ly, rho_cut=args.rho_cut)
    else:
        raise ValueError(args.bg)

    nh = args.nx // 2 + 1
    kx_array = 2.0 * np.pi * np.arange(nh) / args.Lx
    M_list = assemble_M_per_kx(y, D, rho, N2, kx_array)
    Q_list, Qinv_list, sqrtlam_list = eigendecompose_M_list(M_list)

    u_ic, v, b, V_ref, omega = make_eigenmode_ic(
        y, rho, N2, D, args.nx, args.Lx, args.kx_int, args.n_g, args.amp)
    w = np.zeros_like(v)

    T_period = 2 * np.pi / omega
    print(f"  Path 3 v2 (exp-int, V/W/B exact) — "
          f"bg={args.bg}, ny={args.ny}, nx={args.nx}")
    print(f"  kx_int={args.kx_int}, n_g={args.n_g}, amp={args.amp}")
    print(f"  omega={omega:.6f}, period={T_period:.6f}")
    print(f"  dt={args.dt}, n_steps={args.n_steps} "
          f"({args.n_steps * args.dt / T_period:.2f} periods)")
    print(f"  ω·dt = {omega * args.dt:.3f}  (exact linear: no CFL)")

    stride = max(1, args.n_steps // 200)
    devs = [eigmode_deviation(v, V_ref, w_cc, args.nx)]
    energies = [total_energy(u_ic, v, b, rho, N2, w_cc, args.nx)]
    v_centers = [float(v[args.ny // 2, args.nx // 4])]
    t_list = [0.0]

    t = 0.0
    for step in range(args.n_steps):
        dt = args.dt
        if args.no_nonlin:
            # Linear-only: single exact propagator over full dt (validation mode)
            v, w, b = apply_R_full(v, w, b, Q_list, Qinv_list, sqrtlam_list,
                                   N2, kx_array, args.nx, dt)
        else:
            # Strang (A): linear exact dt/2
            v, w, b = apply_R_full(v, w, b, Q_list, Qinv_list, sqrtlam_list,
                                   N2, kx_array, args.nx, 0.5 * dt)
            # Strang (B): nonlinear RK2 dt
            v, b = step_rk2_nonlinear(v, b, rho, D, kx_array, args.nx, dt)
            # Strang (C): linear exact dt/2
            v, w, b = apply_R_full(v, w, b, Q_list, Qinv_list, sqrtlam_list,
                                   N2, kx_array, args.nx, 0.5 * dt)
        t += dt

        if not np.isfinite(v).all() or np.max(np.abs(v)) > 1e6 * max(args.amp, 1e-8):
            print(f"  BLOW UP at step {step+1}, |v|_max = {np.max(np.abs(v)):.3e}")
            break

        if (step + 1) % stride == 0 or step < 3:
            u_now = rebuild_u_from_v(v, rho, D, kx_array, args.nx)
            devs.append(eigmode_deviation(v, V_ref, w_cc, args.nx))
            energies.append(total_energy(u_now, v, b, rho, N2, w_cc, args.nx))
            v_centers.append(float(v[args.ny // 2, args.nx // 4]))
            t_list.append(t)

    devs = np.array(devs); energies = np.array(energies)
    v_centers = np.array(v_centers); t_list = np.array(t_list)
    n_use = min(10, len(devs) - 1)
    dev_rate = (devs[n_use] - devs[0]) / max(n_use, 1) if n_use >= 1 else float('nan')
    E0 = energies[0]
    dE_rel = (energies[-1] - E0) / max(abs(E0), 1e-300)

    freq_est = float('nan'); rel_freq_err = float('nan')
    if len(v_centers) >= 16:
        vc = v_centers - v_centers.mean()
        vc = vc * np.hanning(len(vc))
        VC = np.fft.rfft(vc)
        dt_sample = t_list[1] - t_list[0] if len(t_list) > 1 else args.dt
        freqs = np.fft.rfftfreq(len(vc), d=dt_sample)
        omega_axis = 2 * np.pi * freqs
        kmax = np.argmax(np.abs(VC[1:])) + 1
        freq_est = omega_axis[kmax]
        rel_freq_err = (freq_est - omega) / omega

    print(f"  dev/step_avg ({n_use} samples) = {dev_rate:.3e}")
    print(f"  dev_final = {devs[-1]:.3e}")
    print(f"  E drift = {dE_rel:+.3e}  (E0={E0:.3e})")
    print(f"  omega (FFT peak) = {freq_est:.6f}   rel_err = {rel_freq_err:+.3e}")

    if args.save:
        np.savez(args.save,
                 t=t_list, dev=devs, energy=energies, v_center=v_centers,
                 omega_evp=omega, dev_rate=dev_rate, dE_rel=dE_rel,
                 freq_fft=freq_est, rel_freq_err=rel_freq_err,
                 bg=args.bg, amp=args.amp, path="expint_v2")
        print(f"  → saved {args.save}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--bg", choices=["boussinesq", "lane_emden"],
                    default="lane_emden")
    ap.add_argument("--rho_cut", type=float, default=0.05)
    ap.add_argument("--kx_int", type=int, default=1)
    ap.add_argument("--n_g", type=int, default=1)
    ap.add_argument("--amp", type=float, default=1e-8)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--no_nonlin", action="store_true",
                    help="Disable nonlinear block (linear floor test)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
