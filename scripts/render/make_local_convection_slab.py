#!/usr/bin/env python3
"""Extract a thin plane-parallel slab from a MESA profile for cart_ale2.

Carves a radial strip from a MESA ZAMS profile (e.g. the convective envelope
of a 1 M⊙ star, r/R ∈ [0.85, 0.99]) and flattens it to a 1-D vertical
stratification `(y, ρ, P, T)` plus a mean gravity `g_y = ⟨G·M_enc/r²⟩`.

The slab is resampled onto a uniform-in-y grid (cart_ale2 wants a regular
Cartesian mesh).  cart_ale2 is ideal γ = 5/3; for a 1 M⊙ ZAMS convective
envelope MESA Γ₁ ≈ 1.6655 → γ = 5/3 is a < 0.05 % error.

Output format (`slab.txt`):
    # Ly_cm Lx_cm g_y gamma rho_top P_top T_top mean_molec_wt
    # y[cm] rho[g/cm^3] P[erg/cm^3] T[K]
    0    ρ0  P0  T0
    dy   ρ1  P1  T1
    ...
    Ly   ρN  PN  TN        (top face)
Bottom of slab = y=0 (deeper = higher pressure).  Top = y=Ly.

Usage:
    python3 scripts/make_local_convection_slab.py \
        /tmp/mesa_work_1Msol/LOGS/profile5.data slab.txt \
        --r-lo 0.85 --r-hi 0.99 --ny 128 --lx-over-ly 2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mesa_profile import read_profile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesa_profile", type=Path)
    ap.add_argument("out_file", type=Path)
    ap.add_argument("--r-lo", type=float, default=0.85,
                    help="bottom of slab, units of R_star (default 0.85)")
    ap.add_argument("--r-hi", type=float, default=0.99,
                    help="top of slab, units of R_star (default 0.99)")
    ap.add_argument("--ny", type=int, default=128,
                    help="number of vertical cells (default 128)")
    ap.add_argument("--lx-over-ly", type=float, default=2.0,
                    help="horizontal extent Lx in units of slab thickness (default 2)")
    args = ap.parse_args()

    mp = read_profile(args.mesa_profile)
    mc = mp.to_cgs_profile()
    G = 6.674e-8

    # MESA data is surface→core; reverse to core→surface.
    r_s   = mc["r"][::-1]
    m_s   = mc["m_enc"][::-1]
    rho_s = mc["rho"][::-1]
    T_s   = mc["T"][::-1]
    P_s   = mc["P"][::-1]
    X_s   = mc["X"][::-1]
    Y_s   = mc["Y"][::-1]
    R_star = mc["R_star"]
    M_star = mc["M_star"]

    r_lo_cm = args.r_lo * R_star
    r_hi_cm = args.r_hi * R_star

    # Drop zones with r=0 (MESA synthesises this) to keep interpolation clean.
    mask = r_s > 0.0
    r_s = r_s[mask]; m_s = m_s[mask]; rho_s = rho_s[mask]
    T_s = T_s[mask]; P_s = P_s[mask]; X_s = X_s[mask]; Y_s = Y_s[mask]

    # Uniform y grid with y = r - r_lo (so slab bottom = y=0).
    ny = args.ny
    y_faces = np.linspace(0.0, r_hi_cm - r_lo_cm, ny + 1)
    y_nodes = y_faces  # we'll write one row per face for the slab file
    r_nodes = r_lo_cm + y_nodes

    # Log-interpolate ρ and P onto the uniform y grid.
    def log_interp(r_target, r_src, f_src):
        lo, hi = r_src[0], r_src[-1]
        r_clip = np.clip(r_target, lo, hi)
        return np.exp(np.interp(r_clip, r_src, np.log(f_src)))

    rho_mesa = log_interp(r_nodes, r_s, rho_s)
    P_mesa   = log_interp(r_nodes, r_s, P_s)
    T_mesa   = log_interp(r_nodes, r_s, T_s)
    m_node   = np.interp(r_nodes, r_s, m_s)
    X_node   = np.interp(r_nodes, r_s, X_s)
    Y_node   = np.interp(r_nodes, r_s, Y_s)

    # Mean gravity over the slab: use ⟨G·M_enc/r²⟩.
    g_arr = G * m_node / (r_nodes ** 2)
    g_y = float(np.mean(g_arr))

    # Rebuild HSE under constant g_y, preserving MESA's entropy profile
    # s(y) = P/ρ^γ.  We anchor at the top boundary (MESA's outer ρ, P) and
    # integrate down with dP/dy = -ρg.  At each step, given s(y) and P, the
    # density is ρ = (P/s)^(1/γ).  Trapezoidal: average ρ between two faces.
    gamma = 5.0 / 3.0
    s_mesa = P_mesa / rho_mesa ** gamma

    rho_node = np.empty_like(rho_mesa)
    P_node   = np.empty_like(P_mesa)
    rho_node[-1] = rho_mesa[-1]
    P_node[-1]   = P_mesa[-1]
    for j in range(ny - 1, -1, -1):
        dy = y_nodes[j + 1] - y_nodes[j]
        # Predictor: P_j ≈ P_{j+1} + ρ_{j+1}·g·dy, then iterate with
        # trapezoidal rule until converged.
        P_cur  = P_node[j + 1] + rho_node[j + 1] * g_y * dy
        for _ in range(8):
            rho_cur = (P_cur / s_mesa[j]) ** (1.0 / gamma)
            rho_avg = 0.5 * (rho_cur + rho_node[j + 1])
            P_new = P_node[j + 1] + rho_avg * g_y * dy
            if abs(P_new - P_cur) < 1e-12 * P_cur:
                P_cur = P_new
                break
            P_cur = P_new
        rho_node[j] = (P_cur / s_mesa[j]) ** (1.0 / gamma)
        P_node[j]   = P_cur

    # T from ideal gas with MESA's local mean molecular weight for
    # reference only (cart_ale2 runs pure γ-law and never uses T).
    mu_node = 1.0 / (2.0 * X_node + 0.75 * Y_node + 0.5 * (1.0 - X_node - Y_node))
    k_B = 1.380649e-16
    m_H = 1.6735575e-24
    T_node = P_node * mu_node * m_H / (rho_node * k_B)

    mid = ny // 2
    X_mid = X_node[mid]; Y_mid = Y_node[mid]
    mu = float(mu_node[mid])

    Ly = float(y_faces[-1])
    Lx = args.lx_over_ly * Ly

    # Store the 'top' (which is y=Ly) as the reference — convection people
    # typically quote envelope top conditions.
    rho_top = float(rho_node[-1])
    P_top   = float(P_node[-1])
    T_top   = float(T_node[-1])

    with args.out_file.open("w") as f:
        f.write(
            f"# slab from {args.mesa_profile.name} "
            f"r/R=[{args.r_lo}, {args.r_hi}]  ny={ny}\n"
        )
        f.write(
            f"# Ly_cm Lx_cm g_y gamma rho_top P_top T_top mu\n"
        )
        f.write(
            f"{Ly:.10e} {Lx:.10e} {g_y:.10e} {gamma:.10e} "
            f"{rho_top:.10e} {P_top:.10e} {T_top:.10e} {mu:.10e}\n"
        )
        f.write(f"# y_cm rho P T\n")
        for i in range(ny + 1):
            f.write(
                f"{y_nodes[i]:.10e} {rho_node[i]:.10e} "
                f"{P_node[i]:.10e} {T_node[i]:.10e}\n"
            )

    # Diagnostics: HSE residual vs prescribed constant g_y (what cart_ale2
    # will see). dP/dy + ρ·g_y should be ≈ 0 under the trap scheme.
    dP = np.diff(P_node)
    dy = np.diff(y_nodes)
    rho_mid = 0.5 * (rho_node[:-1] + rho_node[1:])
    hse_lhs = dP / dy
    hse_rhs = -rho_mid * g_y
    hse_rel = np.abs(hse_lhs - hse_rhs) / np.maximum(np.abs(hse_rhs), 1e-30)

    print(f"slab written to {args.out_file}")
    print(f"  geometry: Ly={Ly:.3e} cm, Lx={Lx:.3e} cm, ny={ny}")
    print(f"  gravity:  g_y={g_y:.3e} cm/s^2 (mean over slab)")
    print(f"  top:      ρ={rho_top:.3e}, P={P_top:.3e}, T={T_top:.3e} K")
    print(f"  bot:      ρ={rho_node[0]:.3e}, P={P_node[0]:.3e}, T={T_node[0]:.3e} K")
    print(f"  mu:       {mu:.4f} at mid-slab (X={X_mid:.3f}, Y={Y_mid:.3f})")
    print(f"  HSE resid (rel): median {np.median(hse_rel):.2e}, "
          f"p90 {np.percentile(hse_rel, 90):.2e}, max {np.max(hse_rel):.2e}")
    print(f"  P/ρ range: {P_node[0]/rho_node[0]:.3e} .. {P_node[-1]/rho_node[-1]:.3e}")
    print(f"  c_s bottom: {np.sqrt(gamma*P_node[0]/rho_node[0]):.3e} cm/s")
    print(f"  c_s top:    {np.sqrt(gamma*P_node[-1]/rho_node[-1]):.3e} cm/s")
    tau_dyn = Ly / np.sqrt(gamma * P_node[-1] / rho_node[-1])
    print(f"  τ_dyn (Ly/c_s_top): {tau_dyn:.3e} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
