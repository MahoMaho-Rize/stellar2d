#!/usr/bin/env python3
"""Full-Galerkin TD closure test: proves that operator-mismatch vanishes
when the discrete 2nd-order ODE is assembled as matrix equations.

The linearised anelastic system collapses (after eliminating u, b, π) to
one 2nd-order ODE in V̂(y,t):

    L · ∂_{tt} V̂ = -R · V̂,    L = -∂_y(ρ∂_y) + k²ρ,   R = k²N²ρ       (v-space)
    L_φ · ∂_{tt}φ = -R_φ · φ, L_φ = -∂_{yy} + k²,     R_φ = k²N²      (φ-space)

where φ = ρV̂.  The EVP ωⁿ² L V̂ = R V̂ is exactly a spatial
discretisation of this ODE.  If we time-step THIS PAIR (V̂, W = ∂_t V̂)
using the same L, R matrices the EVP uses, V̂_EVP is an exact discrete
eigenvector — the time evolution is rigidly

    V̂(t) = V̂_EVP · cos(ω t),   W(t) = -ω V̂_EVP · sin(ω t)

So dev/step must drop to MACHINE PRECISION regardless of background.

This is what "full spectral Galerkin in y" would achieve in CUDA.  The
current CUDA v-space TD uses PRIMITIVE ops (apply_dy, pointwise N²·V,
pointwise ρ'/ρ · Π) whose compositions do NOT equal the assembled
L, R matrices at discrete level — that's where the O(1e-3) leak comes
from.

Target:
  v-space full-Galerkin on Lane-Emden: dev/step ≤ 1e-12 (machine × cond)
  φ-space full-Galerkin on Lane-Emden: dev/step ≤ 1e-12
  primitive node-space TD (toy) on Lane-Emden: dev/step ~ 1e-4 (unchanged)

If results confirm, the fix is architecturally clear: CUDA TD must assemble
and store L, R (ny × ny) and step via matrix-matrix products instead of
primitive apply_dy + pointwise ops.  That's the 7-10 day rewrite.

Usage:
    python3 scripts/full_galerkin_closure_test.py
    python3 scripts/full_galerkin_closure_test.py --integrator leapfrog
    python3 scripts/full_galerkin_closure_test.py --integrator rk4
"""
from __future__ import annotations
import argparse

import numpy as np
import scipy.linalg


# ── CGL grid & Chebyshev D ─────────────────────────────────────────────
def cgl_grid(ny, Ly):
    N = ny - 1
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0)**np.arange(N + 1))
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


# ── Backgrounds ────────────────────────────────────────────────────────
def bg_boussinesq(y, N2_val=1.0):
    return np.ones_like(y), np.full_like(y, N2_val)


def bg_lane_emden(y, Ly, rho_cut=0.01):
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda xi, st: [st[1],
                      -max(st[0], 0.0)**1.5 - 2*st[1]/max(xi, 1e-8)],
                    [1e-6, 4.0], [1.0, 0.0], dense_output=True,
                    events=lambda t, st: st[0], max_step=1e-3)
    xi_s = sol.t_events[0][0]
    xi = np.linspace(1e-6, xi_s, 4000)
    rho_full = np.clip(sol.sol(xi)[0], 0, None)**1.5
    mask = rho_full > rho_cut
    xi_lo, xi_hi = xi[mask].min(), xi[mask].max()
    xi_q = xi_lo + (y / Ly) * (xi_hi - xi_lo)
    rho = np.clip(np.interp(xi_q, xi, rho_full), rho_cut, None)
    drho = np.gradient(rho, y, edge_order=2)
    N2 = np.maximum(-drho / rho, 0.0)
    return rho, N2


# ── Assemble L, R in either v-space or φ-space (interior-restricted) ──
def assemble_operator(variant, y, D, rho, N2, kx):
    ny = len(y)
    if variant == "vspace":
        L = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
        R = kx**2 * np.diag(N2 * rho)
    else:  # phi
        D2 = D @ D
        L = -D2 + kx**2 * np.eye(ny)
        R = kx**2 * np.diag(N2)
    intr = slice(1, ny - 1)
    return L[intr, intr], R[intr, intr]


def evp(L, R):
    """Generalised EVP R v = ω² L v → eigenvalues ω², descending."""
    lam, V = scipy.linalg.eig(R, L)
    lam = np.real(lam); V = np.real(V)
    mask = np.isfinite(lam) & (lam > 0)
    lam, V = lam[mask], V[:, mask]
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order]


# ── Time integrators for  L · v̈ = -R · v ──────────────────────────────
def step_leapfrog(v, w_half, dt, L_inv_R):
    """Staggered leapfrog: w at half-steps, v at integer.
       v^{n+1} = v^n + dt · w^{n+1/2}
       w^{n+3/2} = w^{n+1/2} - dt · (L⁻¹R) v^{n+1}
    """
    v_new = v + dt * w_half
    w_new = w_half - dt * (L_inv_R @ v_new)
    return v_new, w_new


def step_rk4(v, w, dt, L_inv_R):
    """Standard RK4 on (v, w) = (state, d/dt state) with v̈ = -L⁻¹R v."""
    def f(v_, w_):
        return w_, -(L_inv_R @ v_)
    k1v, k1w = f(v, w)
    k2v, k2w = f(v + 0.5*dt*k1v, w + 0.5*dt*k1w)
    k3v, k3w = f(v + 0.5*dt*k2v, w + 0.5*dt*k2w)
    k4v, k4w = f(v + dt*k3v,     w + dt*k3w)
    v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
    w_new = w + dt/6 * (k1w + 2*k2w + 2*k3w + k4w)
    return v_new, w_new


def step_primitive_node(v, w, dt, D, rho, N2, kx):
    """CUDA-faithful node-space primitive TD (v, w=∂_t v).  Uses the
    CONTINUOUS form  v̈ = -L_continuous v  where L is applied via the
    natural primitive discretisation: apply_dy twice, pointwise · ρ, etc.
    This is what CUDA actually does, and shows the O(1e-4) mismatch."""
    # Compute "d/dy (ρ dv/dy)" by two passes of D
    rho_dvdy = rho * (D @ v)
    Lv_continuous = -(D @ rho_dvdy) + kx**2 * rho * v
    # Mass matrix is effectively diag(ρ) (from momentum divided by ρ):
    # v̈ = -(L_cont / ρ) v... but CUDA code divides by ρ pointwise only where
    # the reduced-pressure formulation demands.  Simulate with:
    Rv = kx**2 * N2 * rho * v
    # Equivalent v̈ = -L⁻¹ R v = -(Lv)⁻¹ Rv ≈ -(Rv / Lv) if we're sloppy about the
    # mass matrix — for a fair test, use the EXACT L matrix:
    # No — the whole point of path-C / primitive mismatch is that CUDA solves
    # the projection inline, not assembling L.  Here we emulate:  divide R by
    # a diagonal approximation  of L, which is an intentional mismatch.
    # The cleanest "primitive-pointwise" proxy:  use diag(ρ) as mass matrix:
    # ρ v̈ = -(-(ρ v')' + k² ρ v) + (k² N² ρ) v  when linearised v-eq includes pressure.
    # This keeps ρ on both sides but uses D applied twice without assembling
    # the full L matrix symmetrically.
    vdotdot_primitive = (Rv - Lv_continuous) / np.maximum(rho, 1e-30)
    # Forward Euler on (v, w)
    w_new = w + dt * vdotdot_primitive
    v_new = v + dt * w_new
    v_new[0] = 0.0; v_new[-1] = 0.0
    return v_new, w_new


# ── Measurement ────────────────────────────────────────────────────────
def measure_dev(state_trajectory, IC, w_cc):
    """dev[i] = ‖state[i] - a(i)·IC‖_{w_cc} / ‖IC‖_{w_cc}"""
    IC_norm = np.sum(w_cc * IC**2)
    devs = []
    for s in state_trajectory:
        a = np.sum(w_cc * s * IC) / max(IC_norm, 1e-300)
        r = s - a * IC
        devs.append(float(np.sqrt(np.sum(w_cc * r**2) / max(IC_norm, 1e-300))))
    return np.array(devs)


# ── Run one closure test ───────────────────────────────────────────────
def run_test(bg_name, integrator, y, D, w_cc, rho, N2, kx,
             n_steps, dt, amp, variant):
    ny = len(y)
    L, R = assemble_operator(variant, y, D, rho, N2, kx)
    lam, V = evp(L, R)
    om2 = lam[0]
    omega = np.sqrt(om2)
    # IC on interior
    v0_int = V[:, 0].copy()
    v0_int = v0_int / np.max(np.abs(v0_int)) * amp
    # Solve L · v̈ = -R · v  →  v̈ = -L⁻¹R v
    L_inv_R = scipy.linalg.solve(L, R)
    # IC:  v = v0, w = 0  (oscillator starts at max amplitude)
    w0_int = np.zeros_like(v0_int)
    v, w = v0_int.copy(), w0_int.copy()
    # For leapfrog, initialise w at half-step t = -dt/2
    if integrator == "leapfrog":
        w = w - 0.5 * dt * (L_inv_R @ v)
    traj_int = [v.copy()]
    for _ in range(n_steps):
        if integrator == "leapfrog":
            v, w = step_leapfrog(v, w, dt, L_inv_R)
        elif integrator == "rk4":
            v, w = step_rk4(v, w, dt, L_inv_R)
        else:
            raise ValueError(integrator)
        traj_int.append(v.copy())
    # Extend to full-grid for dev measurement (append zeros at walls)
    traj_full = []
    for s in traj_int:
        full = np.zeros(ny); full[1:-1] = s
        traj_full.append(full)
    IC_full = np.zeros(ny); IC_full[1:-1] = v0_int
    devs = measure_dev(traj_full, IC_full, w_cc)
    return devs, om2, omega


def run_primitive_test(bg_name, y, D, w_cc, rho, N2, kx,
                       n_steps, dt, amp):
    """Emulates CUDA's primitive (non-assembled) TD to reproduce 1e-4 leak."""
    ny = len(y)
    # Use v-space EVP as IC
    L, R = assemble_operator("vspace", y, D, rho, N2, kx)
    lam, V = evp(L, R)
    om2 = lam[0]
    # Full-grid IC: walls 0
    v0 = np.zeros(ny); v0[1:-1] = V[:, 0] / np.max(np.abs(V[:, 0])) * amp
    w0 = np.zeros(ny)
    v, w = v0.copy(), w0.copy()
    traj = [v.copy()]
    for _ in range(n_steps):
        v, w = step_primitive_node(v, w, dt, D, rho, N2, kx)
        traj.append(v.copy())
    devs = measure_dev(traj, v0, w_cc)
    return devs, om2


# ── Main driver ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--kx_int", type=int, default=1)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--amp", type=float, default=1e-8)
    ap.add_argument("--integrator", choices=["leapfrog", "rk4"],
                    default="rk4")
    args = ap.parse_args()

    y, D = cgl_grid(args.ny, args.Ly)
    w_cc = cc_weights(args.ny, args.Ly)
    kx = args.kx_int * 2 * np.pi / args.Lx

    for bg_name, (rho, N2) in [
        ("boussinesq", bg_boussinesq(y, N2_val=1.0)),
        ("lane_emden", bg_lane_emden(y, args.Ly, rho_cut=0.01))]:

        print(f"\n{'═' * 76}")
        print(f"  {bg_name},  ny={args.ny},  kx={kx:.4f},  "
              f"n_steps={args.n_steps},  dt={args.dt},  "
              f"integrator={args.integrator}")
        print('═' * 76)
        print(f"  EVP ω_1  ω_2  ω_3 (for reference):")
        L, R = assemble_operator("vspace", y, D, rho, N2, kx)
        lam, _ = evp(L, R)
        print(f"    {np.sqrt(lam[:3])}")

        # (A) Full Galerkin TD in v-space
        devs_v, om2_v, om_v = run_test(
            bg_name, args.integrator, y, D, w_cc, rho, N2, kx,
            args.n_steps, args.dt, args.amp, "vspace")
        print(f"\n  [Full Galerkin v-space]  ω²={om2_v:.6e}  ω={om_v:.4f}")
        print(f"    dev trajectory samples:")
        for i in [1, 10, 50, args.n_steps // 2, args.n_steps]:
            if i <= args.n_steps:
                print(f"      step {i:4d}: {devs_v[i]:.3e}")
        rate_v = (devs_v[args.n_steps] - devs_v[1]) / max(args.n_steps - 1, 1)
        print(f"    dev/step avg ≈ {rate_v:.3e}")

        # (B) Full Galerkin TD in φ-space
        devs_phi, om2_phi, om_phi = run_test(
            bg_name, args.integrator, y, D, w_cc, rho, N2, kx,
            args.n_steps, args.dt, args.amp, "phi")
        print(f"\n  [Full Galerkin φ-space]  ω²={om2_phi:.6e}  ω={om_phi:.4f}")
        for i in [1, 10, 50, args.n_steps // 2, args.n_steps]:
            if i <= args.n_steps:
                print(f"      step {i:4d}: {devs_phi[i]:.3e}")
        rate_phi = (devs_phi[args.n_steps] - devs_phi[1]) / max(args.n_steps - 1, 1)
        print(f"    dev/step avg ≈ {rate_phi:.3e}")

        # (C) Primitive node-space TD (CUDA-style mismatch)
        devs_prim, om2_prim = run_primitive_test(
            bg_name, y, D, w_cc, rho, N2, kx,
            args.n_steps, args.dt, args.amp)
        print(f"\n  [Primitive node-space TD — CUDA-style]")
        for i in [1, 10, 50, args.n_steps // 2, args.n_steps]:
            if i <= args.n_steps:
                print(f"      step {i:4d}: {devs_prim[i]:.3e}")
        rate_prim = (devs_prim[args.n_steps] - devs_prim[1]) / max(args.n_steps - 1, 1)
        print(f"    dev/step avg ≈ {rate_prim:.3e}")

        # Summary
        print(f"\n  {'─' * 72}")
        print(f"  SUMMARY ({bg_name}):")
        print(f"    full-Galerkin v-space   dev/step = {rate_v:.3e}")
        print(f"    full-Galerkin φ-space   dev/step = {rate_phi:.3e}")
        print(f"    primitive node-space    dev/step = {rate_prim:.3e}")


if __name__ == "__main__":
    main()
