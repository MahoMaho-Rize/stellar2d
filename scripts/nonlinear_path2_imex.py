#!/usr/bin/env python3
"""Phase 3 Path 2: IMEX nonlinear TD.

Time integrator per dt:
  Implicit Crank-Nicolson for linear block  +  AB2 explicit for nonlinear.

  State: (V, W, B) each (ny, nx).  Reduced coupled ODE (in x-Fourier per kx):

      ∂_t [V; W] = [ 0   I ] · [V; W] + f_nl,   let A_kx := [ 0   I ; -M_kx   0 ]
                   [-M   0 ]

  Crank-Nicolson in the linear 2×2 block:
      (I - dt/2 A_kx) · [V^{n+1}; W^{n+1}] = (I + dt/2 A_kx) · [V^n; W^n]
                                            + dt · f_nl_AB2
  For each kx this is a solve of a 2·n_int × 2·n_int linear system.

  f_nl_AB2 = 1.5 f_nl^n − 0.5 f_nl^{n-1}  (2nd-order extrapolation)
  f_nl only feeds W equation via advection of v; B tracked separately.

Advantages vs Path 1:
  - Linear part A-stable → dt can exceed the g-mode CFL (ω_max·dt < O(1))
  - Machine-precision dev/step in linear limit (Crank-Nicolson preserves
    eigenvectors of M exactly)

Drawback:
  - Requires factorisation of (I - dt/2 A_kx) per kx (one-time cost if dt
    fixed; cheap — 2·n_int matrices of size 2·n_int × 2·n_int).

State layout per kx: stacked column [V̂_int; Ŵ_int] of length 2·n_int.
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


def build_CN_operators(M_list, dt, n_int):
    """For each kx, build LU of (I - dt/2 A)  and  explicit matrix (I + dt/2 A).
    A = [[0, I], [-M, 0]] (2·n_int × 2·n_int).
    Returns (LU_lhs list, rhs_mat list)."""
    out_lu = []; out_rhs = []
    I2 = np.eye(2 * n_int)
    for M in M_list:
        if M.shape[0] == 0:
            out_lu.append(None); out_rhs.append(None); continue
        A = np.zeros((2 * n_int, 2 * n_int))
        A[:n_int, n_int:] = np.eye(n_int)
        A[n_int:, :n_int] = -M
        lhs = I2 - 0.5 * dt * A
        rhs = I2 + 0.5 * dt * A
        lu = scipy.linalg.lu_factor(lhs)
        out_lu.append(lu); out_rhs.append(rhs)
    return out_lu, out_rhs


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


def compute_nl_rhs_vw(v, b, rho, N2, D, kx_array, nx):
    """Returns f_V=0  (V̇ has no nonlinear part),  f_W=adv_v,  f_B=adv_b-N²v.
    Actually in CN-IMEX the -N²v part is LINEAR so goes into A implicitly;
    here b is semi-decoupled (its linear evolution ḃ = -N²v is explicit).
    Keep it simple: treat b linearly explicit, RK4-ish, via adv_b − N²·v.
    """
    u = rebuild_u_from_v(v, rho, D, kx_array, nx)
    _, adv_v, adv_b = compute_advection(u, v, b, D, kx_array, nx)
    return adv_v, adv_b


def cn_linear_step(v, w, LU_list, RHS_list, kx_array, nx, n_int,
                   f_W=None):
    """Apply Crank-Nicolson per kx:  [V^{n+1}; W^{n+1}] =
         LU⁻¹ · ( RHS · [V^n; W^n] + dt·[0; f_W] )
    f_W: (ny, nx) real, extrapolated nonlinear RHS for w equation (optional).
    """
    vhat = fft_x(v); what = fft_x(w)
    nh = vhat.shape[1]
    out_v = np.zeros_like(vhat); out_w = np.zeros_like(what)

    if f_W is not None:
        fhat_W = fft_x(f_W)
    else:
        fhat_W = None

    for k in range(nh):
        lu = LU_list[k]; rhs_mat = RHS_list[k]
        if lu is None:
            continue
        state = np.concatenate([vhat[1:-1, k], what[1:-1, k]])
        rhs = rhs_mat @ state
        if fhat_W is not None:
            rhs[n_int:] += (LU_list[k][0].shape[0] and 1.0) * 0.0  # placeholder
            # dt is embedded in LU/RHS build; f_W contribution added outside:
            pass
        # We pass dt·f_W added by caller into rhs before calling; see caller.
        sol = scipy.linalg.lu_solve(lu, rhs)
        out_v[1:-1, k] = sol[:n_int]
        out_w[1:-1, k] = sol[n_int:]
    v_new = ifft_x(out_v, nx)
    w_new = ifft_x(out_w, nx)
    v_new[0, :] = 0.0; v_new[-1, :] = 0.0
    return v_new, w_new


def cn_linear_step_with_fW(v, w, f_W, LU_list, RHS_list, kx_array, nx, n_int, dt):
    """Crank-Nicolson with AB2 nonlinear input:
      LU · [V^{n+1}; W^{n+1}] = RHS · [V^n; W^n] + dt · [0; f_W_hat]
    f_W is (ny, nx) real;  its FFT_x rows are dispatched per kx.
    """
    vhat = fft_x(v); what = fft_x(w); fwhat = fft_x(f_W)
    nh = vhat.shape[1]
    out_v = np.zeros_like(vhat); out_w = np.zeros_like(what)
    for k in range(nh):
        lu = LU_list[k]; rhs_mat = RHS_list[k]
        if lu is None:
            continue
        state = np.concatenate([vhat[1:-1, k], what[1:-1, k]])
        rhs = rhs_mat @ state
        rhs[n_int:] += dt * fwhat[1:-1, k]
        sol = scipy.linalg.lu_solve(lu, rhs)
        out_v[1:-1, k] = sol[:n_int]
        out_w[1:-1, k] = sol[n_int:]
    v_new = ifft_x(out_v, nx)
    w_new = ifft_x(out_w, nx)
    v_new[0, :] = 0.0; v_new[-1, :] = 0.0
    return v_new, w_new


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
    n_int = args.ny - 2

    LU_list, RHS_list = build_CN_operators(M_list, args.dt, n_int)

    u_ic, v, b, V_ref, omega = make_eigenmode_ic(
        y, rho, N2, D, args.nx, args.Lx, args.kx_int, args.n_g, args.amp)
    w = np.zeros_like(v)

    T_period = 2 * np.pi / omega
    print(f"  Path 2 (CN-IMEX) — bg={args.bg}, ny={args.ny}, nx={args.nx}")
    print(f"  kx_int={args.kx_int}, n_g={args.n_g}, amp={args.amp}")
    print(f"  omega={omega:.6f}, period={T_period:.6f}")
    print(f"  dt={args.dt}, n_steps={args.n_steps} "
          f"({args.n_steps * args.dt / T_period:.2f} periods)")
    print(f"  ω·dt = {omega * args.dt:.3f}  (Path 1 CFL stability ~ 2.8;"
          f" CN has no CFL on linear part)")

    stride = max(1, args.n_steps // 200)
    devs = [eigmode_deviation(v, V_ref, w_cc, args.nx)]
    energies = [total_energy(u_ic, v, b, rho, N2, w_cc, args.nx)]
    v_centers = [float(v[args.ny // 2, args.nx // 4])]
    t_list = [0.0]

    # AB2 needs previous nonlinear RHS — init by one Euler nonlin eval.
    adv_v_prev, adv_b_prev = compute_nl_rhs_vw(v, b, rho, N2, D, kx_array, args.nx)
    t = 0.0
    for step in range(args.n_steps):
        dt = args.dt
        adv_v_now, adv_b_now = compute_nl_rhs_vw(v, b, rho, N2, D, kx_array, args.nx)
        # AB2 extrapolation for nonlinear ẇ contribution
        fW = 1.5 * adv_v_now - 0.5 * adv_v_prev
        # b is linear explicit: ḃ = -N²·v + adv_b  (nonlinear part AB2'd)
        fB_ab2 = 1.5 * adv_b_now - 0.5 * adv_b_prev
        # advance b with Crank-Nicolson in linear part (trapezoid) + AB2 nonlin:
        # b^{n+1} = b^n + dt·[(-N² v^n − N² v^{n+1})/2] + dt·fB_ab2
        # After linear CN step we know v^{n+1}; iterate: first solve v, w CN,
        # then update b analytically.
        v_new, w_new = cn_linear_step_with_fW(
            v, w, fW, LU_list, RHS_list, kx_array, args.nx, n_int, dt)
        b_new = b + 0.5 * dt * (-N2[:, None] * (v + v_new)) + dt * fB_ab2
        # Update
        v, w, b = v_new, w_new, b_new
        adv_v_prev, adv_b_prev = adv_v_now, adv_b_now
        t += dt

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
                 bg=args.bg, amp=args.amp, path="imex")
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
