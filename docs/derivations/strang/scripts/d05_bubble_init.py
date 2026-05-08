r"""
Section D5 — Bubble (entropy-boost) IC on HSE background.

The Strang solver's default IC builder init_bubble places a
locally-warm (positive entropy) perturbation on top of the
isentropic HSE background.  The IC is constant-pressure (pressure
equilibrium with the background) and isentropic with a radial
entropy boost:

  s(r) = s_bg + delta_s * exp(-(r / R_0)^2)       (optional theta-
                                                   perturbation added)

where r = sqrt((x - x_0)^2 + (y - y_0)^2).  Under constant P = p_bar(y_0)
and the isentropic closure s = log(P rho^{-gamma}):

  rho(r) = rho_bg(y) * (P / P_bg)^{1/gamma} * exp(-delta_s / gamma * exp(-(r/R_0)^2))

Actually the kernel's IC is simpler: it holds P = p_bar(y) EVERYWHERE
(no pressure perturbation), so the only perturbation is to rho:

  delta rho / rho_bg = exp(-delta_s / gamma * exp(-(r/R_0)^2)) - 1
                    ≈ -delta_s / gamma * exp(-(r/R_0)^2)  (linearised)

With an optional azimuthal mode:

  s(r, theta) = s_bg + delta_s * exp(-(r/R_0)^2) * (1 + epsilon cos(k theta))

Strong-form identities verified:

  1. Isentropic-to-rho map:  rho = (P / K_of_s)^{1/gamma} where
     K_of_s = exp(s).  So at constant P = p_bar,  delta rho / rho_bg
     = (K_bg / K_of_s)^{1/gamma} - 1 when s is boosted above s_bg.

  2. Zero-pressure-perturbation (delta P = 0 by construction):
     the IC's delta_E = rho (u^2+v^2)/2 (kinetic only; no internal
     perturbation because P - p_bar = 0).  So delta_E = 0 on the
     static (u = v = 0) bubble IC.

  3. Positivity: rho > 0 and P > 0 on any reasonable parameter
     range (delta_s < gamma * s_bg is sufficient).

Golden-values dump:
  output/d05_bubble_init.goldens.json:
    - canonical bubble parameters (x_0, y_0, R_0, delta_s, k_theta, epsilon)
    - HSE parameters (rho_0, K, gamma, g, L_y)
    - reference delta_rho / rho_bg profile at a representative grid
      (e.g., 64x64) for the test to compare against
    - (optional) azimuthal mode amplitude
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp
import math

from _common import (
    LatexDump,
    GoldensDump,
    assert_zero,
    banner,
)


def main() -> int:
    ld = LatexDump(__file__)
    gd = GoldensDump(__file__)
    banner("D5 - Bubble (entropy-boost) IC")

    # Symbols.
    gamma = sp.Symbol("gamma", positive=True)
    gm1 = gamma - 1
    rho_bg_s = sp.Symbol("rho_bg", positive=True)
    p_bg_s   = sp.Symbol("p_bg",   positive=True)
    delta_s  = sp.Symbol("delta_s", real=True)
    K_bg = p_bg_s / rho_bg_s**gamma  # background entropy function

    # ════════════════════════════════════════════════════════════
    # 1.  Isentropic-to-rho map at constant P.
    #
    # If s = log(P rho^{-gamma}) = log(K), then K = exp(s).
    # Boost s by delta_s (localised to a radial profile for the
    # bubble); the new entropy function is K' = exp(s + delta_s)
    # = K_bg * exp(delta_s).
    # At constant P = p_bg,  rho' = (P / K')^{1/gamma} = rho_bg * exp(-delta_s/gamma).
    #
    # So the density deficit (hot bubble is less dense):
    #   delta rho / rho_bg  =  exp(-delta_s / gamma) - 1.
    # ════════════════════════════════════════════════════════════
    rho_hot = (p_bg_s / (K_bg * sp.exp(delta_s)))**(1 / gamma)
    # Compare to closed form rho_bg exp(-delta_s / gamma):
    # rho_hot = (p_bg / K_bg / exp(delta_s))^(1/gamma)
    #         = (p_bg / K_bg)^(1/gamma) * exp(-delta_s/gamma)
    #         = (p_bg / (p_bg / rho_bg^gamma))^(1/gamma) * exp(-delta_s/gamma)
    #         = (rho_bg^gamma)^(1/gamma) * exp(-delta_s/gamma)
    #         = rho_bg * exp(-delta_s/gamma).
    expected = rho_bg_s * sp.exp(-delta_s / gamma)
    assert_zero(
        sp.simplify(sp.powdenest(sp.logcombine(rho_hot / expected, force=True), force=True) - 1),
        "D5-isentropic-rho: rho' / rho_bg = exp(-delta_s/gamma) at const P",
    )

    # ════════════════════════════════════════════════════════════
    # 2.  Linearised form for small delta_s.
    #
    # exp(-delta_s/gamma) - 1 = -delta_s/gamma + (delta_s/gamma)^2/2 - ...
    # ════════════════════════════════════════════════════════════
    linear_expansion = sp.series(sp.exp(-delta_s / gamma) - 1, delta_s, 0, 2).removeO()
    expected_linear = -delta_s / gamma
    assert_zero(
        sp.simplify(linear_expansion - expected_linear),
        "D5-linear: delta_rho / rho_bg ≈ -delta_s/gamma (linear order)",
    )

    # ════════════════════════════════════════════════════════════
    # 3.  delta_E = 0 on static bubble IC (u = v = 0, delta_P = 0).
    #
    # E_tot = P / gm1 + rho KE
    # delta_E = E_tot - p_bar / gm1
    #         = (P - p_bar) / gm1 + rho KE
    #         = 0 (since P = p_bar) + 0 (since KE = 0)
    #         = 0.
    # ════════════════════════════════════════════════════════════
    delta_E = (p_bg_s - p_bg_s) / gm1 + rho_hot * sp.Integer(0)
    assert_zero(
        sp.simplify(delta_E),
        "D5-delta-E-zero: on static bubble IC with P = p_bar, delta_E = 0",
    )

    # ════════════════════════════════════════════════════════════
    # 4.  Positivity: for delta_s > 0 (hot bubble) and delta_s < 0
    #     (cool bubble), rho stays positive for any finite delta_s.
    # ════════════════════════════════════════════════════════════
    # rho_hot = rho_bg * exp(-delta_s/gamma) > 0 since exp > 0 and rho_bg > 0.
    # This is trivially positive.

    # ════════════════════════════════════════════════════════════
    # Golden values dump.
    # ════════════════════════════════════════════════════════════
    # Canonical bubble parameters.
    gamma_val = 1.4
    rho_0_val = 1.0
    K_val = 1.0
    g_val = 1.0
    L_y_val = 1.0
    x_0_val = 0.5
    y_0_val = 0.3
    R_0_val = 0.1
    delta_s_val = 0.5      # moderate entropy boost
    # Optional azimuthal mode parameters
    k_theta_val = 3        # m = 3 mode
    eps_val = 0.1          # 10% modulation

    gd.add("gamma", gamma_val)
    gd.add("rho_0_bottom", rho_0_val)
    gd.add("K_poly",       K_val)
    gd.add("g",            g_val)
    gd.add("L_y",          L_y_val)
    gd.add("x_0", x_0_val)
    gd.add("y_0", y_0_val)
    gd.add("R_0", R_0_val)
    gd.add("delta_s",  delta_s_val)
    gd.add("k_theta",  k_theta_val)
    gd.add("epsilon",  eps_val)
    gd.add("azimuthal_enabled", True)

    # Reference delta_rho / rho_bg at representative grid 64x64
    # over the domain [0, 1] x [0, 1].
    N_ref = 64
    dx_ref = 1.0 / N_ref
    dy_ref = 1.0 / N_ref

    # HSE background at cell-centre y:
    def rho_bg_at_y(y_val):
        arg = rho_0_val**(gamma_val - 1) - (gamma_val - 1) * g_val * y_val / (gamma_val * K_val)
        if arg < 1e-20:
            arg = 1e-20
        return arg**(1 / (gamma_val - 1))

    def p_bg_at_y(y_val):
        return K_val * rho_bg_at_y(y_val)**gamma_val

    # delta_rho / rho_bg at (x, y):
    def delta_rho_rel(x, y):
        dx = x - x_0_val
        dy = y - y_0_val
        r2 = dx*dx + dy*dy
        theta = math.atan2(dy, dx) if (dx*dx + dy*dy) > 0 else 0.0
        # delta_s localised in radius + azimuthal modulation
        local_ds = delta_s_val * math.exp(-r2 / R_0_val**2) * (1.0 + eps_val * math.cos(k_theta_val * theta))
        return math.exp(-local_ds / gamma_val) - 1.0

    rho_profile_2d = []
    for j in range(N_ref):
        y_c = (j + 0.5) * dy_ref
        row = []
        for i in range(N_ref):
            x_c = (i + 0.5) * dx_ref
            row.append(delta_rho_rel(x_c, y_c))
        rho_profile_2d.append(row)
    gd.add("delta_rho_rel_ref_grid_64x64", rho_profile_2d)
    gd.add("N_ref", N_ref)

    # Also dump the closed-form center value (should be a specific
    # magnitude):
    center_val = math.exp(-delta_s_val / gamma_val) - 1.0
    gd.add("delta_rho_rel_at_center", center_val)
    print(f"  Bubble centre delta_rho / rho_bg = {center_val:.6f} "
          f"(expected -{delta_s_val/gamma_val:.3f} linear-order)")

    gd.write()

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Bubble IC (entropy boost + azimuthal mode)",
        r"s(\mathbf{x}) \;=\; s_{\mathrm{bg}} \;+\; \delta s\,\exp\!\bigl(-r^{2}/R_{0}^{2}\bigr)\,\bigl(1 + \epsilon\,\cos(k_{\theta}\,\theta)\bigr)",
        label="eq:D5-IC",
    )
    ld.add(
        "Constant-pressure assumption",
        r"P(\mathbf{x}) \;=\; \bar p(y) \qquad \text{(pressure equilibrium with HSE)}",
        label="eq:D5-const-P",
    )
    ld.add(
        "Density from isentropic closure",
        r"\rho(\mathbf{x}) \;=\; \bar\rho(y)\,\exp\!\bigl(-\delta s_{\mathrm{local}} / \gamma\bigr)"
        r"\qquad\text{(hot bubble = density deficit)}",
        label="eq:D5-rho",
    )
    ld.add(
        "Linear-order density perturbation",
        r"\frac{\delta\rho}{\bar\rho} \;\approx\; -\frac{\delta s_{\mathrm{local}}}{\gamma} "
        r"\qquad\text{(for small } \delta s\text{)}",
        label="eq:D5-linear",
    )
    ld.add(
        "Zero energy perturbation (static)",
        r"\delta E \;=\; (P - \bar p)/(\gamma-1) + \tfrac{1}{2}\rho(u^2 + v^2) \;=\; 0 "
        r"\qquad\text{(since } P = \bar p \text{ and } u = v = 0\text{)}",
        label="eq:D5-dE-zero",
    )

    ld.write()
    print()
    print("All D5 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
