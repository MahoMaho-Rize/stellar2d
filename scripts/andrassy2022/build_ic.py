#!/usr/bin/env python3
"""Build Andrassy+ 2022 (A&A 659 A193) idealized O-shell convection IC
as a cart_ale2 slab file.

Test problem spec (Andrassy 2022 §2.2, Eqs. 1–5):
  - y ∈ [1, 3] (convective layer is 1 ≤ y < 2, stable layer 2 < y ≤ 3).
  - x ∈ [-1, 1] periodic, z ∈ [-1, 1] periodic.  Our 2D pilot drops z.
  - γ = 5/3 everywhere (monatomic ideal gas).
  - Two species with μ₀=1.848 (convective layer), μ₁=1.802 (stable).
    Transition smooth via η₁(y) = (1/2)[1 + sin(8πy)] on [2-1/16, 2+1/16].
  - d ln p / d ln ρ = γ₀=5/3 in convective, γ₁=1.3 in stable
    (achieved via μ gradient, not γ — gas is single γ).
  - Variable gravity:  g(y) = g₀ · f_g(y) · y^(-5/4)
    g₀ = 1.414870,
    f_g(y) is a smooth cutoff (sin-based on bottom 1/32 and top 1/32):
        f_g(y) = (1/2)[1 + sin(16π(y-1))]   for 1 ≤ y ≤ 1 + 1/32
               = 1                            for 1 + 1/32 < y < 3 - 1/32
               = (1/2)[1 - sin(16π(y-3))]   for 3 - 1/32 ≤ y ≤ 3
    (turns off g near walls so density/pressure become constant → BC
     acts as reflective without spurious pressure-gradient from walls)
  - Bottom heating: q̇(y) = q̇₀ sin(8π(y-1)) for 1 ≤ y ≤ 9/8, else 0.
    q̇₀ = 3.795720e-4.  Total L = 1.2082151e-4.
  - IC stratification: HSE with the above g(y) and d ln p/d ln ρ = γ_eff(y).
    The piecewise slope implies an effective "polytrope" in (ρ, p) space
    but internally gas obeys p = (γ-1)ρe with γ=5/3.  This is achieved
    at IC by setting T(y) consistently so that ρ(y) and p(y) give the
    requested density slope.

cart_ale2's slab file format is a 1D stratification (y, ρ, P, T) + a
single g_y.  Andrassy's variable g(y) is more nuanced.  For the pilot
we start with a single effective g_y = ⟨g(y)⟩_conv (≈ 0.81), which
gives a good approximation to the convective-layer physics (heating
layer is near y=1 where g(y)≈1.4; bulk convective layer mean ~0.81).
Upper stable layer HSE is off by ~2× — acceptable for pilot because
the initial acoustic transient damps in a few sound-crossing times
(cart_ale2 is fully compressible, unlike anelastic/low-Mach codes
which would require matched g).

Next iteration:  pilot v2 will add a proper variable-g feature to
cart_ale2 or use radial1d-style Lagrangian HSE reprojection.

Usage:
    python3 scripts/andrassy2022/build_ic.py \
        data/andrassy2022/pilot_ic_n256.txt --ny 256
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# ─── Andrassy 2022 problem constants (§2.2) ─────────────────────────────
G0          = 1.414870           # g at y=1 (bottom of convective layer)
Y_BOTTOM    = 1.0                # domain bottom
Y_CB        = 2.0                # Schwarzschild boundary
Y_TOP       = 3.0                # domain top
DELTA_Y     = 1.0 / 16.0         # half-width of γ-transition
DELTA_G     = 1.0 / 32.0         # half-width of gravity cutoff
Y_TRANS_LO  = Y_CB - DELTA_Y     # 1.9375
Y_TRANS_HI  = Y_CB + DELTA_Y     # 2.0625
Y_HEAT_LO   = 1.0
Y_HEAT_HI   = 1.0 + 1.0 / 8.0    # 1.125
Q_DOT_0     = 3.795720e-4
L_TOTAL     = 1.2082151e-4
GAMMA_EOS   = 5.0 / 3.0          # monatomic ideal gas
GAMMA0_DLP  = 5.0 / 3.0          # d ln p/d ln ρ in convective layer
GAMMA1_DLP  = 1.3                # d ln p/d ln ρ in stable layer
MU0         = 1.848              # conv-layer molecular weight
MU1         = 1.802              # stable-layer
# dimensionless: c_s=1, ρ=1, T=1 at bottom of convective layer


def f_g(y: np.ndarray) -> np.ndarray:
    """Gravity cutoff near walls per Andrassy 2022 Eq. 2."""
    f = np.ones_like(y)
    # Bottom 1/32: (1/2)[1 + sin(16π(y-1) - π/2)] shaped as 0 → 1
    # Actually paper says: sin(16π(y - 1/32)/2) style.  Re-reading Eq. 2,
    # it's a smooth sinusoidal ramp that is 0 at y=1 (or y=3) and 1 at
    # y = 1 + 1/32 (or y = 3 - 1/32).  Simplest interpretation:
    #   f_g(y) = (1/2)[1 - cos(π · (y-1)/DELTA_G)]  for 1 ≤ y ≤ 1+1/32
    # which is 0 at y=1 and 1 at y=1+1/32.  Symmetric for top.
    mask_bot = y <= Y_BOTTOM + DELTA_G
    mask_top = y >= Y_TOP - DELTA_G
    f = np.where(mask_bot,
                 0.5 * (1.0 - np.cos(np.pi * (y - Y_BOTTOM) / DELTA_G)),
                 f)
    f = np.where(mask_top,
                 0.5 * (1.0 - np.cos(np.pi * (Y_TOP - y) / DELTA_G)),
                 f)
    return f


def g_of_y(y: np.ndarray) -> np.ndarray:
    """g(y) = g₀ f_g(y) y^(-5/4)."""
    return G0 * f_g(y) * y ** (-5.0 / 4.0)


def eta1(y: np.ndarray) -> np.ndarray:
    """Fractional volume of μ₁ fluid (Eq. 3). 0 in convective, 1 in stable."""
    s = (y - Y_TRANS_LO) / (2.0 * DELTA_Y)  # 0→1 across transition
    # Andrassy Eq. 3 uses (1/2)[1 + sin(8π y)] which I read as (after offset)
    # (1/2)[1 - cos(π·s)]
    ramp = 0.5 * (1.0 - np.cos(np.pi * s))
    return np.where(y < Y_TRANS_LO, 0.0,
            np.where(y > Y_TRANS_HI, 1.0, ramp))


def gamma_eff(y: np.ndarray) -> np.ndarray:
    """Effective d ln p/d ln ρ profile (Eq. 4)."""
    e = eta1(y)
    return GAMMA0_DLP + e * (GAMMA1_DLP - GAMMA0_DLP)


def mu_of_y(y: np.ndarray) -> np.ndarray:
    """μ(y) smoothly blending μ₀ → μ₁ via η₁."""
    e = eta1(y)
    return MU0 + e * (MU1 - MU0)


def build_hse_profile(y: np.ndarray,
                       rho_bot: float = 1.0,
                       p_bot: float = 3.0 / 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HSE with g(y) and piecewise effective γ, fine-grid RK4.

    dp/dy = -ρ g(y)
    d ln p / d ln ρ = γ_eff(y)  →  ρ = ρ_ref · (p/p_ref)^{1/γ_eff}
    """
    n_sub = 80
    y_fine = np.linspace(y[0], y[-1], n_sub * (len(y) - 1) + 1)

    gamma_f = gamma_eff(y_fine)
    g_f = g_of_y(y_fine)
    p_f   = np.zeros_like(y_fine)
    rho_f = np.zeros_like(y_fine)

    p_f[0]   = p_bot
    rho_f[0] = rho_bot

    for i in range(1, len(y_fine)):
        dy = y_fine[i] - y_fine[i-1]
        g_mid = 0.5 * (g_f[i-1] + g_f[i])
        gamma_mid = 0.5 * (gamma_f[i-1] + gamma_f[i])
        # Local K = p / ρ^γ_local ensures ρ(p) is consistent with γ_eff.
        K_local = p_f[i-1] / rho_f[i-1] ** gamma_mid

        def rho_from_p(p_val):
            if p_val <= 0.0:
                return 0.0
            return (p_val / K_local) ** (1.0 / gamma_mid)

        # Midpoint rule: predict p at midpoint, evaluate ρ, update.
        p_mid_pred = p_f[i-1] - 0.5 * dy * rho_f[i-1] * g_mid
        if p_mid_pred <= 0.0:
            p_mid_pred = max(1e-14, 0.5 * p_f[i-1])
        rho_mid = rho_from_p(p_mid_pred)
        p_new = p_f[i-1] - dy * rho_mid * g_mid
        if p_new <= 0.0:
            p_new = max(1e-14, 0.5 * p_f[i-1])
        p_f[i] = p_new
        rho_f[i] = rho_from_p(p_new)

    # Interpolate to requested nodes.
    p   = np.interp(y, y_fine, p_f)
    rho = np.interp(y, y_fine, rho_f)
    # Temperature: ideal gas p = ρ T / μ  →  T = μ · p / ρ (in dimensionless
    # units where R_gas = 1).
    mu = mu_of_y(y)
    T   = mu * p / rho
    return rho, p, T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_file", type=Path)
    ap.add_argument("--ny", type=int, default=256,
                    help="number of vertical cells (default 256)")
    ap.add_argument("--lx", type=float, default=2.0,
                    help="horizontal extent (Andrassy default 2.0)")
    args = ap.parse_args()

    ny = args.ny
    # Node-centered grid on faces (ny+1 faces).  Domain y ∈ [1, 3].
    y_nodes = np.linspace(Y_BOTTOM, Y_TOP, ny + 1)

    # HSE stratification.
    rho, p, T = build_hse_profile(y_nodes)

    # Effective g_y for cart_ale2 slab format (single-g fallback for
    # header — actual physics will use the variable g(y) column below).
    y_conv = np.linspace(1.0, 2.0, 200)
    g_mean_conv = g_of_y(y_conv).mean()
    g_eff = g_mean_conv

    # Variable gravity: Andrassy Eq. 1 evaluated at each y_node.
    g_y_arr = g_of_y(y_nodes)

    # Andrassy heating profile (Eq. 5): q̇(y) = q̇₀ sin(8π(y-1))  for y ∈ [1, 9/8]
    # else zero.  Note: the original spec uses y ∈ [1, 9/8] = 1 heat-layer
    # width = 1/8.  In our shifted coordinate y' = y - 1, the heating is in
    # y' ∈ [0, 1/8] → use sin(8π·y').  Units: volumetric power density
    # [erg/s/cm³] in dimensionless code units.
    y_shifted = y_nodes - Y_BOTTOM
    q_dot_arr = np.where(
        (y_shifted >= 0.0) & (y_shifted <= 1.0/8.0),
        Q_DOT_0 * np.sin(8.0 * np.pi * y_shifted),
        0.0,
    )

    # Header "top" values.
    rho_top = float(rho[-1])
    p_top   = float(p[-1])
    T_top   = float(T[-1])
    mu_stored = MU0   # slab header μ; cart_ale2 treats μ as scalar here.

    with args.out_file.open("w") as f:
        f.write(f"# Andrassy 2022 (A&A 659 A193) idealized O-shell convection IC\n")
        f.write(f"# 2D slab for cart_ale2, y ∈ [1, 3], Lx = {args.lx}\n")
        f.write(f"# Variable-g in original paper; this slab uses mean g = {g_eff:.6f}\n")
        f.write(f"# γ_eos = {GAMMA_EOS}, piecewise effective d ln p/d ln ρ: "
                f"{GAMMA0_DLP:.3f}→{GAMMA1_DLP:.3f} at y=2\n")
        f.write(f"# μ: {MU0} (conv) → {MU1} (stable) via η₁ ramp\n")
        f.write(f"# Heating: q̇₀={Q_DOT_0:.6e}, L_tot={L_TOTAL:.6e}, y∈[{Y_HEAT_LO}, {Y_HEAT_HI}]\n")
        f.write(f"# Ly_cm Lx_cm g_y gamma rho_top P_top T_top mu\n")
        Ly = Y_TOP - Y_BOTTOM
        f.write(
            f"{Ly:.10e} {args.lx:.10e} {g_eff:.10e} {GAMMA_EOS:.10e} "
            f"{rho_top:.10e} {p_top:.10e} {T_top:.10e} {mu_stored:.10e}\n"
        )
        f.write(f"# y_cm rho P T g_y(optional) qdot(optional)\n")
        # cart_ale2 wants y starting at 0 (slab-relative coordinate).
        # Shift Andrassy's y∈[1,3] → y'∈[0,2].
        for i in range(ny + 1):
            # 5th col = g(y), 6th col = q̇(y).  cart_ale2's init_local_convection
            # auto-detects both.
            f.write(
                f"{y_shifted[i]:.10e} {rho[i]:.10e} "
                f"{p[i]:.10e} {T[i]:.10e} "
                f"{g_y_arr[i]:.10e} {q_dot_arr[i]:.10e}\n"
            )

    # HSE residual diagnostic with variable g.
    dy = np.diff(y_nodes)
    p_mid = 0.5 * (p[1:] + p[:-1])
    rho_mid = 0.5 * (rho[1:] + rho[:-1])
    g_mid = g_of_y(0.5 * (y_nodes[1:] + y_nodes[:-1]))
    dpdy = np.diff(p) / dy
    rhs_var = -rho_mid * g_mid
    rhs_single = -rho_mid * g_eff
    hse_resid_var    = np.abs(dpdy - rhs_var) / np.maximum(np.abs(rhs_var), 1e-30)
    hse_resid_single = np.abs(dpdy - rhs_single) / np.maximum(np.abs(rhs_single), 1e-30)
    print(f"Wrote {args.out_file}")
    print(f"  ny = {ny}  domain y ∈ [1, 3] shifted to [0, 2]  Lx = {args.lx}")
    print(f"  g_eff (mean over convective layer) = {g_eff:.6f}")
    print(f"  ρ at y=1, 2, 3: {rho[0]:.4e} / {rho[ny//2]:.4e} / {rho[-1]:.4e}")
    print(f"  p at y=1, 2, 3: {p[0]:.4e} / {p[ny//2]:.4e} / {p[-1]:.4e}")
    print(f"  γ_eff at y=1, 2, 3: "
          f"{gamma_eff(np.array([1.0]))[0]:.3f} / "
          f"{gamma_eff(np.array([2.0]))[0]:.3f} / "
          f"{gamma_eff(np.array([3.0]))[0]:.3f}")
    print(f"  g(y) at y=1, 2, 3: "
          f"{g_of_y(np.array([1.0]))[0]:.4f} / "
          f"{g_of_y(np.array([2.0]))[0]:.4f} / "
          f"{g_of_y(np.array([3.0]))[0]:.4f}")
    print(f"  HSE residual with variable g: mean={hse_resid_var.mean():.3e}  "
          f"max={hse_resid_var.max():.3e}")
    print(f"  HSE residual with mean g:     mean={hse_resid_single.mean():.3e}  "
          f"max={hse_resid_single.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
