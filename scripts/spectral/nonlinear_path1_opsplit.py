#!/usr/bin/env python3
"""Phase 3 Path 1: operator-splitting nonlinear TD.

Algorithm (Strang splitting per dt):

  (A) Linear half-step (dt/2):
      RK4 on (v, w, b) with
         v̇ = w
         ẇ = -M·v         (M = L⁻¹R per kx, assembled Path D-style)
         ḃ = -N²·v
      All linear anelastic physics — buoyancy, pressure, continuity —
      is embedded in M.  b is tracked as an independent diagnostic /
      passive scalar for the nonlinear block; its feedback to v is
      already in M.

  (B) Nonlinear full-step (dt):
      u is rebuilt from v per kx via continuity
         ikx·ρ·û + ∂_y(ρ v̂) = 0  →  û = -(1/(ikx ρ)) ∂_y(ρ v̂)
      RK4 on (v, b) with
         v̇ = -(u·∇)v
         ḃ = -(u·∇)b
      2/3 dealias on x.  w is untouched (O(amp²) effect on w is second-order
      Strang error, acceptable for small/moderate amp).

  (C) Linear half-step (dt/2): same as (A).

In the linear limit amp → 0 the nonlinear block becomes a no-op and the
scheme reduces to Path D's dev/step floor (~1e-15 on Lane-Emden).

Measurement at output time:
  - dev/step_avg over first 10 samples (linear-regime drift)
  - total energy drift over the full run
  - FFT peak of v_center trajectory → ω_freq, rel_err vs EVP
  - mode spectrum at end-of-run  (triad-coupling growth)
"""
from __future__ import annotations
import argparse
import numpy as np
import scipy.linalg

from nonlinear_paths_infra import (
    cgl_grid, cc_weights, bg_lane_emden,
    assemble_M_per_kx, make_eigenmode_ic,
    compute_advection,
    eigmode_deviation, total_energy, apply_M,
    fft_x, ifft_x, dealias_23,
)


# ── Rebuild u from v via continuity (per kx) ──────────────────────────
def rebuild_u_from_v(v, rho, D, kx_array, nx):
    """û(kx, y) = -(1/(ikx ρ)) · ∂_y(ρ · v̂(kx, y)).
    Returns u (ny, nx) real."""
    ny, _ = v.shape
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


# ── Linear block: RK4 on (v, w, b) ────────────────────────────────────
def step_rk4_linear(v, w, b, M_list, N2, kx_array, nx, dt):
    def f(v_, w_, b_):
        Mv = apply_M(v_, M_list, kx_array, nx)
        Mv[0, :] = 0.0; Mv[-1, :] = 0.0
        dv = w_
        dw = -Mv
        db = -N2[:, None] * v_
        return dv, dw, db

    k1v, k1w, k1b = f(v, w, b)
    k2v, k2w, k2b = f(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, b + 0.5 * dt * k1b)
    k3v, k3w, k3b = f(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, b + 0.5 * dt * k2b)
    k4v, k4w, k4b = f(v + dt * k3v, w + dt * k3w, b + dt * k3b)
    v_n = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    w_n = w + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
    b_n = b + dt / 6.0 * (k1b + 2 * k2b + 2 * k3b + k4b)
    v_n[0, :] = 0.0; v_n[-1, :] = 0.0
    return v_n, w_n, b_n


# ── Nonlinear block: RK4 on (v, b) with u derived each substep ────────
def step_rk4_nonlinear(v, b, rho, D, kx_array, nx, dt):
    def f(v_, b_):
        u_ = rebuild_u_from_v(v_, rho, D, kx_array, nx)
        _, adv_v, adv_b = compute_advection(u_, v_, b_, D, kx_array, nx)
        return adv_v, adv_b

    k1v, k1b = f(v, b)
    k2v, k2b = f(v + 0.5 * dt * k1v, b + 0.5 * dt * k1b)
    k3v, k3b = f(v + 0.5 * dt * k2v, b + 0.5 * dt * k2b)
    k4v, k4b = f(v + dt * k3v, b + dt * k3b)
    v_n = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    b_n = b + dt / 6.0 * (k1b + 2 * k2b + 2 * k3b + k4b)
    v_n[0, :] = 0.0; v_n[-1, :] = 0.0
    return v_n, b_n


# ── Main run ───────────────────────────────────────────────────────────
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

    # IC: single eigenmode, v = V_EVP(y) cos(kx x), u from continuity, b=0, w=0.
    u_ic, v, b, V_ref, omega = make_eigenmode_ic(
        y, rho, N2, D, args.nx, args.Lx, args.kx_int, args.n_g, args.amp)
    w = np.zeros_like(v)
    u_ic_norm = float(np.max(np.abs(u_ic)))

    T_period = 2 * np.pi / omega
    print(f"  Path 1 (op-split) — bg={args.bg}, ny={args.ny}, nx={args.nx}")
    print(f"  kx_int={args.kx_int}, n_g={args.n_g}, amp={args.amp}")
    print(f"  omega={omega:.6f}, period={T_period:.6f}")
    print(f"  dt={args.dt}, n_steps={args.n_steps} "
          f"({args.n_steps * args.dt / T_period:.2f} periods)")

    # Storage: dense sampling at low n_steps, subsample at high
    stride = max(1, args.n_steps // 200)

    devs = [eigmode_deviation(v, V_ref, w_cc, args.nx)]
    energies = [total_energy(u_ic, v, b, rho, N2, w_cc, args.nx)]
    v_centers = [float(v[args.ny // 2, args.nx // 4])]
    t_list = [0.0]

    t = 0.0
    for step in range(args.n_steps):
        dt = args.dt
        # Strang (A)
        v, w, b = step_rk4_linear(v, w, b, M_list, N2, kx_array, args.nx, 0.5 * dt)
        # Strang (B)
        v, b = step_rk4_nonlinear(v, b, rho, D, kx_array, args.nx, dt)
        # Strang (C)
        v, w, b = step_rk4_linear(v, w, b, M_list, N2, kx_array, args.nx, 0.5 * dt)
        t += dt

        # Check blowup
        if not np.isfinite(v).all() or np.max(np.abs(v)) > 1e6 * args.amp:
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

    # Diagnostics
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
                 bg=args.bg, amp=args.amp, path="opsplit")
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
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
