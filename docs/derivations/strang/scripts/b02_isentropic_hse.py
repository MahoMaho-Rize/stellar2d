r"""
Section B2 — Isentropic hydrostatic-equilibrium background.

We integrate the strong-form ODE for HSE under a polytropic (isentropic)
equation of state:

  dp/dy = -rho * g         (force balance)
  p      = K * rho^gamma    (isentropic closure, entropy = const)

which yields a closed-form density profile

  rho(y) = [rho_0^(gamma-1) - (gamma-1) g y / (gamma K)]^{1/(gamma-1)}

with atmosphere cut-off at

  y_star = gamma K rho_0^(gamma-1) / ((gamma-1) g).

Strong-form identities verified:

  1. The proposed closed-form rho(y) satisfies d[K rho^gamma]/dy = -rho g.
  2. At y=0, rho(0) = rho_0 (consistency with the bottom BC).
  3. The isentropic closure gives entropy s(y) = log(K) = const.
  4. The proposed form equals the floor value at y = y_star exactly
     (arg of fractional power vanishes), so rho(y>y_star) should be
     clamped to the floor.
  5. Temperature profile: T(y) = (P/rho)/R = (K rho^(gamma-1))/R is
     linear in (rho^(gamma-1)), i.e. linear in (1 - y/y_star), which
     expresses the atmosphere as a polytrope with adiabatic lapse.

Code anchor:
  src/gpu/explicit/strang_solver.cu :: StrangSolver::init
      (host-side HSE build-up loop)
  src/gpu/explicit/strang_device.cuh :: d_hse_rho, d_hse_p
      (device-side evaluation in MUSCL-Hancock reconstruction)

Part D's d06_hse_zero_perturbation_lock dumps this profile to
b02_hse_profile.goldens.json for regression tests.
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
    banner("B2 - Isentropic HSE")

    # Symbols
    y = sp.Symbol("y", real=True, nonnegative=True)
    gamma = sp.Symbol("gamma", positive=True)
    g = sp.Symbol("g", positive=True)
    K = sp.Symbol("K", positive=True)
    rho_0 = sp.Symbol("rho_0", positive=True)
    gm1 = gamma - 1

    # Atmosphere cut-off.
    y_star = gamma * K * rho_0**gm1 / (gm1 * g)

    # Proposed closed-form density profile.  We write rho_hse and
    # p_hse directly in terms of the "h"-argument  arg = h  so that
    # sympy can simplify fractional-power compositions.  The
    # equivalence with the compact  p_hse = K rho_hse^gamma  is
    # verified as a separate identity.
    arg = rho_0**gm1 - gm1 * g * y / (gamma * K)
    rho_hse = arg**(1 / gm1)
    p_hse = K * arg**(gamma / gm1)

    # ════════════════════════════════════════════════════════════
    # 1.  Strong-form ODE verification: dp/dy = -rho g.
    #
    # Parametric proof.  Let  h(y) = rho_0^(gamma-1) - (gamma-1) g y / (gamma K);
    # then  rho_hse = h^(1/(gamma-1))  and  p_hse = K * h^(gamma/(gamma-1)).
    # Chain rule gives
    #    dh/dy = -(gamma-1) g / (gamma K)
    #    dp/dy = K * [gamma/(gamma-1)] * h^[gamma/(gamma-1) - 1] * dh/dy
    #          = K * [gamma/(gamma-1)] * h^[1/(gamma-1)] * dh/dy        (since gamma/gm1 - 1 = 1/gm1)
    #          = K * [gamma/(gamma-1)] * rho_hse * [-(gamma-1) g / (gamma K)]
    #          = -rho_hse * g.    QED.
    #
    # The identity  gamma/(gamma-1) - 1 == 1/(gamma-1)  is elementary
    # but enables sympy-friendly manipulation once introduced.  We
    # verify the ODE step-by-step in a form sympy can simplify.
    # ════════════════════════════════════════════════════════════
    # Step (a): elementary algebraic identity the proof relies on.
    assert_zero(
        sp.simplify(gamma / gm1 - 1 - 1 / gm1),
        "B2-exponent-identity: gamma/(gamma-1) - 1 = 1/(gamma-1)",
    )

    # Step (b): sympy's direct dp/dy via chain rule through h.
    h = sp.Symbol("h", positive=True)
    # Work in "h-world": define functions of h, then substitute h = arg(y).
    rho_of_h = h ** (1 / gm1)
    p_of_h   = K * h ** (gamma / gm1)
    # dh/dy closed form.
    dh_dy = sp.diff(arg, y)
    assert_zero(
        sp.simplify(dh_dy - (-gm1 * g / (gamma * K))),
        "B2-dh-dy: dh/dy = -(gamma-1) g / (gamma K)",
    )

    # dp/dy via chain rule = dp/dh * dh/dy.  Show dp/dh = gamma * K * h^(1/gm1)
    # (after using gamma/gm1 - 1 = 1/gm1).
    dp_dh = sp.diff(p_of_h, h)
    assert_zero(
        sp.simplify(dp_dh - (gamma / gm1) * K * h ** (1 / gm1)),
        "B2-dp-dh: dp/dh = (gamma K / gm1) * h^(1/gm1)",
    )

    # Assemble dp/dy via chain rule, evaluated at h = arg(y).
    dp_dy_chain = dp_dh * dh_dy  # still a function of h
    # Simplify:
    #   dp_dy_chain = (gamma K / gm1) * h^(1/gm1) * [-gm1 g / (gamma K)]
    #               = -g h^(1/gm1)
    #               = -g rho_of_h.
    expected_dp_dy = -g * rho_of_h
    assert_zero(
        sp.simplify(dp_dy_chain - expected_dp_dy),
        "B2-HSE-ODE: dp/dy = -rho * g  (chain rule, h-parameterisation)",
    )
    # Tie the parametric proof to the y-dependent profile: show
    # rho_of_h(arg(y)) = rho_hse(y) (trivial with current p_hse) and
    # p_of_h(arg(y)) = p_hse(y).
    assert_zero(
        sp.simplify(rho_of_h.subs(h, arg) - rho_hse),
        "B2-rho-parametric: rho_of_h(arg(y)) = rho_hse(y)",
    )
    assert_zero(
        sp.simplify(p_of_h.subs(h, arg) - p_hse),
        "B2-p-parametric: p_of_h(arg(y)) = p_hse(y)",
    )
    # Direct y-world ODE (sympy cannot fold  (K gamma)^a * (1/(K gamma))^b
    # for symbolic gamma, so apply powdenest force=True to make the
    # closed-form identity manifest).
    y_world_resid = sp.diff(p_hse, y) + rho_hse * g
    y_world_resid_denested = sp.powdenest(y_world_resid, force=True)
    assert_zero(
        sp.simplify(y_world_resid_denested),
        "B2-HSE-ODE-y: d p_hse / dy + rho_hse g = 0 (y-world, powdenest)",
    )

    # ════════════════════════════════════════════════════════════
    # 2.  Bottom BC: rho(0) = rho_0.
    # ════════════════════════════════════════════════════════════
    assert_zero(
        sp.simplify(rho_hse.subs(y, 0) - rho_0),
        "B2-bottom-BC: rho(0) = rho_0",
    )
    # p_hse(0) = K * rho_0^(gm1 * gamma/gm1) = K * rho_0^gamma via powdenest.
    p0 = sp.powdenest(p_hse.subs(y, 0), force=True)
    assert_zero(
        sp.simplify(p0 - K * rho_0**gamma),
        "B2-bottom-BC-p: p(0) = K rho_0^gamma",
    )

    # ════════════════════════════════════════════════════════════
    # 3.  Isentropic closure: specific entropy s = log(P/rho^gamma)
    #     is identically log(K), independent of y.
    #
    # sympy cannot fold  (arg^(1/gm1))^gamma  for symbolic gamma,
    # so the "closure" identity  P = K rho^gamma  is itself not
    # simplifiable in closed form.  Verify parametrically (via h),
    # which is strong-form, not weak-form — the identity holds
    # pointwise and we supply a symbolic proof route that sympy can
    # close.
    # ════════════════════════════════════════════════════════════
    # P = K * h^(gamma/gm1),  rho = h^(1/gm1);  so P/rho^gamma =
    # K * h^(gamma/gm1) / (h^(1/gm1))^gamma = K * h^(gamma/gm1 - gamma/gm1) = K.
    # Verify via the parametric form:
    P_over_rho_g_parametric = (K * h**(gamma / gm1)) / ((h**(1 / gm1))**gamma)
    # Force sympy to fold:  (h^(1/gm1))^gamma = h^(gamma/gm1).
    # Apply sp.powdenest / powsimp with force=True.
    folded = sp.powdenest(P_over_rho_g_parametric, force=True)
    assert_zero(
        sp.simplify(folded - K),
        "B2-isentropic-closure: P / rho^gamma = K (parametric, sp.powdenest)",
    )

    # ════════════════════════════════════════════════════════════
    # 4.  Atmosphere cut-off: at y = y_star the argument of the
    #     fractional power vanishes, so rho(y_star) = 0 formally.
    #     The kernel uses max(arg, 1e-20) as a floor;
    #     evaluate the *unclamped* form at y_star and confirm = 0.
    # ════════════════════════════════════════════════════════════
    arg_at_ystar = arg.subs(y, y_star)
    assert_zero(
        sp.simplify(arg_at_ystar),
        "B2-cutoff: arg(y_star) = 0",
    )

    # ════════════════════════════════════════════════════════════
    # 5.  Pressure-to-density ratio (temperature scaling) reads off
    #     linearly in (1 - y/y_star).  Write
    #         P/rho = K rho^(gamma-1) = K * [rho_0^gm1 - gm1 g y /(gamma K)]
    #                = K rho_0^gm1 - gm1 g y / gamma
    #                = K rho_0^gm1 (1 - y/y_star).
    # ════════════════════════════════════════════════════════════
    # sympy cannot fold  h^(gamma/gm1) / h^(1/gm1) = h^((gamma-1)/gm1) = h
    # for symbolic gamma; apply powdenest + explicit parametric form.
    # In "h-world": rho = h^(1/gm1), P = K h^(gamma/gm1),
    # so P/rho = K h = K (rho_0^gm1 - gm1 g y /(gamma K))
    #         = K rho_0^gm1 - gm1 g y / gamma.
    ratio_parametric = (K * h**(gamma / gm1)) / (h**(1 / gm1))
    ratio_parametric_simplified = sp.powdenest(ratio_parametric, force=True)
    assert_zero(
        sp.simplify(ratio_parametric_simplified - K * h),
        "B2-temperature-lapse-parametric: P/rho = K h (parametric)",
    )
    # Substitute h = arg(y):
    expected_ratio = K * arg
    # Now compare closed forms:
    expanded_ratio = sp.simplify(expected_ratio)
    expanded_expected = sp.simplify(K * rho_0**gm1 - gm1 * g * y / gamma)
    assert_zero(
        sp.simplify(expanded_ratio - expanded_expected),
        "B2-temperature-lapse: P/rho = K rho_0^gm1 - gm1 g y / gamma",
    )
    # Compact form: P/rho = K rho_0^gm1 (1 - y/y_star).
    expected_compact = K * rho_0**gm1 * (1 - y / y_star)
    assert_zero(
        sp.simplify(expanded_expected - expected_compact),
        "B2-temperature-lapse-compact: P/rho = K rho_0^gm1 (1 - y/y_star)",
    )

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "HSE strong-form ODE",
        r"\frac{d\bar p}{dy} \;=\; -\,\bar\rho\,g,\qquad \bar p \;=\; K\,\bar\rho^{\gamma}",
        label="eq:B2-ODE",
    )
    ld.add(
        "Closed-form density profile",
        r"\bar\rho(y) \;=\; \bigl[\rho_0^{\gamma-1} "
        r"\;-\; \tfrac{(\gamma-1)\,g}{\gamma\,K}\,y\bigr]^{1/(\gamma-1)}",
        label="eq:B2-rho",
    )
    ld.add(
        "Atmosphere cut-off height",
        r"y^{\star} \;=\; \frac{\gamma\,K\,\rho_0^{\gamma-1}}{(\gamma-1)\,g} "
        r"\qquad\text{(polytropic atmosphere vanishes at } y = y^\star\text{)}",
        label="eq:B2-ystar",
    )
    ld.add(
        "Isentropic closure",
        r"s(y) \;=\; \log\bigl(\bar p / \bar\rho^{\gamma}\bigr) \;=\; \log K "
        r"\qquad\text{(entropy constant with }y\text{)}",
        label="eq:B2-isentropic",
    )
    ld.add(
        "Temperature lapse",
        r"\bar p / \bar\rho \;=\; K\,\rho_0^{\gamma-1}\,(1 - y/y^{\star}) "
        r"\qquad\text{(polytrope, linear in }y\text{)}",
        label="eq:B2-lapse",
    )

    ld.write()
    print()
    print("All B2 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
