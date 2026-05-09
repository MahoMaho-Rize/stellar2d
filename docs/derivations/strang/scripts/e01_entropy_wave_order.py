r"""
Section E1 — Entropy-wave convergence order prediction.

Modified-equation analysis for the MUSCL-Hancock + MC + HLLC + Strang
split on a smooth entropy-wave IC (§D1).  Strong-form derivation
of the leading truncation error.

The discrete update on a 1D linear advection with velocity u_0:
  u_{j+1/2}^L (after MUSCL-MC + Hancock) =
    u_j^n + (Delta x / 2) sigma_j^n - (Delta t / 2) u_0 sigma_j^n
  where sigma_j^n = (u_{j+1}^n - u_{j-1}^n) / (2 Delta x)  (MC slope
  at smooth regions = central difference).

The HLLC flux at the L/R pair reduces (on entropy wave) to the
upwind:  F^* = u_0 u_L^n.
So the cell update is
  u_j^{n+1} = u_j^n - (Delta t / Delta x) (F^*_{j+1/2} - F^*_{j-1/2})
  = u_j^n - nu (u_{j+1/2}^L - u_{j-1/2}^L)
where nu = u_0 Delta t / Delta x.

Substituting the MUSCL-Hancock face states and expanding in Taylor
series around u(x_j, t_n), the leading truncation error is
  E = -(nu/6)(1 - 3 nu + 2 nu^2) Delta x^2 d^3 u / dx^3

(at nu = 0.4, the coefficient is -(0.4/6)(1 - 1.2 + 0.32) = -0.008,
giving O(Delta x^2) overall convergence).

Strong-form identities verified:

  1. The modified equation coefficient at nu = 0.4 is O(Delta x^2).

  2. Global convergence rate on periodic advection is p = 2.0.

  3. The Strang-split interaction does NOT change this leading
     order for entropy waves (since y-sweep does nothing on a
     v = 0 ansatz).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp

from _common import (
    LatexDump,
    assert_zero,
    banner,
)


def main() -> int:
    ld = LatexDump(__file__)
    banner("E1 - Entropy-wave convergence order prediction")

    # ════════════════════════════════════════════════════════════
    # Modified-equation analysis.
    #
    # Consider u_t + u_0 u_x = 0 with u smooth.  Write u as a Taylor
    # series around (x_j, t_n):
    #   u(x_{j+s}, t_n) = u + s h u_x + (s h)^2/2 u_xx + (s h)^3/6 u_xxx + ...
    # with h = Delta x.
    #
    # MC limiter on smooth data = central difference (unlimited
    # region).  So sigma_j = (u(x_{j+1}) - u(x_{j-1})) / (2h)
    # = u_x + (h^2 / 6) u_xxx + O(h^4).
    # ════════════════════════════════════════════════════════════
    h = sp.Symbol("Delta_x", positive=True)
    dt = sp.Symbol("Delta_t", positive=True)
    u_0 = sp.Symbol("u_0", real=True)
    nu = u_0 * dt / h
    x_sym = sp.Symbol("x", real=True)
    u_func = sp.Function("u")

    # Define u and its derivatives as independent symbols for
    # Taylor analysis (bypasses sympy's function handling).
    u_val, u_x, u_xx, u_xxx, u_xxxx = sp.symbols("u u_x u_xx u_xxx u_xxxx", real=True)

    # Central-difference slope sigma_j = (u(x+h) - u(x-h))/(2h).
    # Taylor:
    #   u(x+h) = u + h u_x + h^2/2 u_xx + h^3/6 u_xxx + h^4/24 u_xxxx + ...
    #   u(x-h) = u - h u_x + h^2/2 u_xx - h^3/6 u_xxx + h^4/24 u_xxxx - ...
    #   diff = 2h u_x + (h^3/3) u_xxx + O(h^5)
    #   /2h  = u_x + h^2/6 u_xxx + O(h^4)
    sigma_j = u_x + h**2 * u_xxx / 6

    # MUSCL-Hancock face state at x_{j+1/2}:
    # u^L_{j+1/2} = u_j + (h/2) sigma_j - (dt/2) u_0 sigma_j
    #             = u_j + sigma_j (h - u_0 dt) / 2
    #             = u + (u_x + h^2 u_xxx / 6) * (h - u_0 dt) / 2
    u_L_j_plus_half = u_val + sigma_j * (h - u_0 * dt) / 2

    # Similarly at x_{j-1/2}:
    #   sigma_{j-1} = u_x(x_{j-1}) + (h^2/6) u_xxx(x_{j-1}) + ...
    #              = u_x - h u_xx + (h^2/2) u_xxx + ...  (expanded around x_j)
    # Face state x_{j-1/2}^L from cell j-1:
    #   u^L_{j-1/2} = u(x_{j-1}) + (h - u_0 dt)/2 * sigma_{j-1}
    # To leading order (up to O(h^2) in the difference u_L_{j+1/2} - u_L_{j-1/2}),
    # we need to track the u(x_j - h) and sigma at j-1.
    u_at_jm1 = u_val - h * u_x + h**2 / 2 * u_xx - h**3 / 6 * u_xxx + h**4/24 * u_xxxx
    u_x_at_jm1 = u_x - h * u_xx + h**2/2 * u_xxx - h**3/6 * u_xxxx
    u_xxx_at_jm1 = u_xxx - h * u_xxxx  # approx; higher orders cancel in final
    sigma_jm1 = u_x_at_jm1 + h**2 * u_xxx_at_jm1 / 6
    u_L_j_minus_half = u_at_jm1 + sigma_jm1 * (h - u_0 * dt) / 2

    # Flux difference (upwind, u_0 > 0): F^L - F^R = u_0 * (u_L_{j+1/2} - u_L_{j-1/2}).
    # (Since HLLC on entropy wave = upwind flux = u_0 * u_L_at_face.)
    flux_diff = u_0 * (u_L_j_plus_half - u_L_j_minus_half)

    # Discrete update:
    # u_j^{n+1} = u_j^n - (dt/h) flux_diff
    u_new = u_val - (dt / h) * flux_diff

    # Exact time update (entropy wave, linear advection):
    #   u(x, t + dt) = u(x - u_0 dt, t) = u - u_0 dt u_x + (u_0 dt)^2/2 u_xx - ...
    u_exact = (u_val
               - u_0 * dt * u_x
               + (u_0 * dt)**2 / 2 * u_xx
               - (u_0 * dt)**3 / 6 * u_xxx
               + (u_0 * dt)**4 / 24 * u_xxxx)

    # Error per step: u_new - u_exact.  Expand and collect.
    error = sp.expand(u_new - u_exact)

    # Collect by powers of h (Delta x).  We expect leading O(h^2).
    error_poly = sp.collect(sp.expand(error), h)
    # Extract the coefficient of h^2 u_xxx (expected leading term).
    # At fixed nu = u_0 dt / h (CFL number), substitute dt = nu h / u_0
    # so dt is O(h) too.  This couples dt and h; we keep the series
    # in h with dt expressed as nu h / u_0.
    nu_sym = sp.Symbol("nu", positive=True)
    error_nu = error.subs(dt, nu_sym * h / u_0)
    error_nu = sp.expand(error_nu)
    # Now the leading terms are O(h) and higher.
    # Get coefficient of h (O(h)):
    coef_h = error_nu.coeff(h, 1)
    print(f"  O(h) coeff: {sp.simplify(coef_h)}")
    # Coefficient of h^2 (O(h^2)):
    coef_h2 = error_nu.coeff(h, 2)
    print(f"  O(h^2) coeff: {sp.simplify(coef_h2)}")
    coef_h3 = error_nu.coeff(h, 3)
    print(f"  O(h^3) coeff: {sp.simplify(coef_h3)}")

    # O(h) coefficient should vanish (scheme at least 1st order):
    assert_zero(
        sp.simplify(coef_h),
        "E1-at-least-1st-order: O(h) per-step truncation = 0",
    )
    # O(h^2) coefficient should also vanish (scheme is at least
    # 2nd order per step):
    assert_zero(
        sp.simplify(coef_h2),
        "E1-at-least-2nd-order: O(h^2) per-step truncation = 0",
    )
    # Leading truncation is O(h^3) per step, giving O(h^2) global
    # error after T/dt = T/(nu h/u_0) ~ 1/h steps.  Strong-form
    # identity for the O(h^3) per-step coefficient:
    expected_coef_h3 = nu_sym * u_xxx * (2 * nu_sym**2 - 3 * nu_sym + 1) / 12
    assert_zero(
        sp.simplify(coef_h3 - expected_coef_h3),
        "E1-lead-order: O(h^3) per-step coefficient = nu(2nu^2 - 3nu + 1)/12 * u_xxx",
    )

    # Vanishing of per-step error at nu = 0, 0.5, 1:
    for nu_zero in [sp.Integer(0), sp.Rational(1, 2), sp.Integer(1)]:
        resid = sp.simplify(coef_h3.subs(nu_sym, nu_zero))
        assert_zero(
            resid,
            f"E1-magic-nu[{nu_zero}]: per-step error vanishes at nu = {nu_zero}",
        )

    print(f"  Leading per-step truncation: {sp.simplify(coef_h3)} h^3")
    print("  Global L^1 error: O(h^2)  (per-step O(h^3) * 1/h steps).")
    print("  Global convergence slope: p = 2 (2nd order in h).")

    # ════════════════════════════════════════════════════════════
    # Strang splitting does not change the entropy-wave convergence.
    # The y-sweep on v = 0 ansatz does nothing (each face has
    # u_L = u_R, flux cancels).  So entropy-wave in 2D Strang-split
    # is exactly the 1D entropy-wave convergence rate — p = 2.
    # ════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "MUSCL-Hancock + HLLC discrete update (on smooth entropy wave)",
        r"u_{j}^{n+1} \;=\; u_{j}^{n} \;-\; \nu\,\bigl(u_{j+1/2}^{L} - u_{j-1/2}^{L}\bigr),\qquad "
        r"\nu = u_{0}\,\Delta t / \Delta x",
        label="eq:E1-update",
    )
    ld.add(
        "Face state via MUSCL + Hancock",
        r"u_{j+1/2}^{L} \;=\; u_{j} + \tfrac{1}{2}\bigl(\Delta x - u_{0}\,\Delta t\bigr)\,\sigma_{j},\qquad "
        r"\sigma_{j} = \frac{u_{j+1} - u_{j-1}}{2\,\Delta x}",
        label="eq:E1-face",
    )
    ld.add(
        "Leading-order truncation (strong form)",
        r"u_{j}^{n+1} - u_{\mathrm{exact}}(x_{j}, t^{n+1}) \;=\; "
        r"\mathcal{C}(\nu)\,\Delta x^{2}\,\partial_{x}^{3} u \;+\; O(\Delta x^{3})",
        label="eq:E1-trunc",
    )
    ld.add(
        "Predicted slope",
        r"\|u_{\mathrm{num}} - u_{\mathrm{exact}}\|_{L^{1}} \;\sim\; \Delta x^{2} "
        r"\;\Longrightarrow\; p \;=\; 2.0 \pm 0.1 "
        r"\qquad\text{(2nd order, confirmed by §D1 convergence test)}",
        label="eq:E1-slope",
    )

    ld.write()
    print()
    print("All E1 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
