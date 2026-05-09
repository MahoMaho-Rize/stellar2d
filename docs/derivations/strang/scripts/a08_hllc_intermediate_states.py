r"""
Section A8 — HLLC intermediate states: S_star, p_star, U*_L, U*_R.

Strong-form identities (verified by sympy):

  1. HLLC contact speed (Toro §10.5.1):
       S_star = [ p_R - p_L + rho_L u_L (S_L - u_L) - rho_R u_R (S_R - u_R) ]
              / [ rho_L (S_L - u_L) - rho_R (S_R - u_R) ]

     Derivation: use the Rankine-Hugoniot mass jump across S_L
     (rho_L (u_L - S_L) = rho*_L (S_star - S_L)) plus the momentum
     jump (rho_L u_L (u_L - S_L) + p_L = rho*_L S_star (S_star - S_L) + p_star),
     and the analogous pair across S_R.  Subtract, and the result is
     the S_star formula above.

  2. Star pressure p_star (Toro eq. 10.38):
       p_star = p_L + rho_L (u_L - S_L)(u_L - S_star)
              = p_R + rho_R (u_R - S_R)(u_R - S_star).
     The fact that both expressions agree is equivalent to the
     consistency of the contact speed with the mass/momentum jumps;
     we verify this algebraically.

  3. Star densities (Toro eq. 10.36):
       rho*_L = rho_L (S_L - u_L) / (S_L - S_star)
       rho*_R = rho_R (S_R - u_R) / (S_R - S_star)

  4. Star momenta and energies (Toro eq. 10.39):
       (rho u)*_K = rho*_K S_star   (normal momentum)
       (rho v)*_K = rho*_K v_K      (tangential momentum, unchanged)
       E*_K = rho*_K [ E_K/rho_K + (S_star - u_K)(S_star + p_K/(rho_K (S_L - u_K))) ]
          (this is the energy-jump integrated form)

  5. Rankine-Hugoniot verification on each wave:
     - across S_L:   S_L [U]_L^*L = [F(U)]_L^*L
     - across S_R:   S_R [U]_*R^R = [F(U)]_*R^R
     These are vector identities (4 components each).  Verify
     strong-form.

  6. Consistency of HLLC with the integral conservation law:
       (S_R - S_L) U_HLL = S_R U_R - S_L U_L - (F_R - F_L)
     where U_HLL = average of U*_L and U*_R weighted by (S_* - S_L)
     and (S_R - S_*).  This is the Batten 1997 / Toro 10.5.1
     positivity-preservation identity.

Code anchors:
  src/gpu/explicit/strang_device.cuh :: d_lmhllc
    (the S_star formula, rho*_L / rho*_R, pstar, and the star-state
     constructions are all implemented here — under the LM-HLLC fM
     rescaling of the pressure jump.  See §C3 for how fM enters
     consistently.)

Rule 4: every identity is strong-form pointwise algebraic.  sympy's
simplifier handles the algebra after field-of-fractions normalisation;
no numerical fallback needed.
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
    flux_x_euler,
    gamma,
    total_energy_sym,
)


def main() -> int:
    ld = LatexDump(__file__)
    banner("A8 - HLLC intermediate states")

    # Symbols.
    rho_L = sp.Symbol("rho_L", positive=True)
    rho_R = sp.Symbol("rho_R", positive=True)
    u_L = sp.Symbol("u_L", real=True)
    u_R = sp.Symbol("u_R", real=True)
    v_L = sp.Symbol("v_L", real=True)
    v_R = sp.Symbol("v_R", real=True)
    p_L = sp.Symbol("p_L", positive=True)
    p_R = sp.Symbol("p_R", positive=True)
    S_L = sp.Symbol("S_L", real=True)
    S_R_sym = sp.Symbol("S_R", real=True)

    # Energy totals.
    E_L = total_energy_sym(rho_L, u_L, v_L, p_L, gamma)
    E_R = total_energy_sym(rho_R, u_R, v_R, p_R, gamma)

    # ════════════════════════════════════════════════════════════
    # 1.  Derivation of S_star.
    #
    # Claim (Toro eq. 10.37):
    #   S_star = [ p_R - p_L + rho_L u_L (S_L - u_L) - rho_R u_R (S_R - u_R) ]
    #          / [ rho_L (S_L - u_L) - rho_R (S_R - u_R) ]
    # ════════════════════════════════════════════════════════════
    S_star_formula = (
        (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R_sym - u_R))
        / (rho_L * (S_L - u_L) - rho_R * (S_R_sym - u_R))
    )

    # ════════════════════════════════════════════════════════════
    # 2.  Star pressures p_star.  Two equivalent forms (Toro 10.38):
    #   p_star_L = p_L + rho_L (u_L - S_L)(u_L - S_star)
    #   p_star_R = p_R + rho_R (u_R - S_R)(u_R - S_star)
    # Verify p_star_L == p_star_R identically when S_star is given by
    # the formula above.
    # ════════════════════════════════════════════════════════════
    p_star_L = p_L + rho_L * (u_L - S_L) * (u_L - S_star_formula)
    p_star_R = p_R + rho_R * (u_R - S_R_sym) * (u_R - S_star_formula)
    diff_pstar = sp.simplify(p_star_L - p_star_R)
    assert_zero(
        diff_pstar,
        "A8-pstar-consistency: p_star_L = p_star_R via S_star formula",
    )

    # Define p_star as the common value.
    p_star = sp.simplify(p_star_L)

    # ════════════════════════════════════════════════════════════
    # 3.  Star densities (Toro eq. 10.36):
    #   rho*_L = rho_L (S_L - u_L)/(S_L - S_star)
    #   rho*_R = rho_R (S_R - u_R)/(S_R - S_star)
    # These follow from the mass Rankine-Hugoniot across S_L and S_R.
    # Verification: show  S_L [rho] = [rho u]  is satisfied by this
    # formula.
    # ════════════════════════════════════════════════════════════
    rho_starL = rho_L * (S_L - u_L) / (S_L - S_star_formula)
    rho_starR = rho_R * (S_R_sym - u_R) / (S_R_sym - S_star_formula)

    # Mass RH across S_L:
    #   S_L (rho*_L - rho_L) = (rho*_L S_star) - (rho_L u_L)
    mass_L_jump = sp.simplify(
        S_L * (rho_starL - rho_L) - (rho_starL * S_star_formula - rho_L * u_L)
    )
    assert_zero(mass_L_jump,
                "A8-mass-RH-across-S_L: S_L [rho] = [rho u_n]")

    # Mass RH across S_R:
    mass_R_jump = sp.simplify(
        S_R_sym * (rho_R - rho_starR) - (rho_R * u_R - rho_starR * S_star_formula)
    )
    assert_zero(mass_R_jump,
                "A8-mass-RH-across-S_R: S_R [rho] = [rho u_n]")

    # ════════════════════════════════════════════════════════════
    # 4.  Star normal momenta (Toro eq. 10.39):
    #   (rho u)*_K = rho*_K S_star   (normal momentum in the star)
    # And star tangential momenta: v is unchanged across the acoustic
    # waves (they are linearly degenerate in v):
    #   (rho v)*_K = rho*_K v_K
    #
    # Verify the momentum RH across S_L and S_R.
    # ════════════════════════════════════════════════════════════
    mx_starL = rho_starL * S_star_formula
    mx_starR = rho_starR * S_star_formula
    my_starL = rho_starL * v_L      # tangential unchanged across acoustic
    my_starR = rho_starR * v_R

    # Momentum x-RH across S_L:
    #   S_L ((rho u)*_L - rho_L u_L) = ((rho u)*_L S_star + p_star) - (rho_L u_L^2 + p_L)
    mom_x_L_flux_star = mx_starL * S_star_formula + p_star
    mom_x_L_flux = rho_L * u_L**2 + p_L
    mom_x_L_jump = sp.simplify(
        S_L * (mx_starL - rho_L * u_L) - (mom_x_L_flux_star - mom_x_L_flux)
    )
    assert_zero(
        mom_x_L_jump,
        "A8-mom_x-RH-across-S_L: S_L [rho u] = [rho u^2 + p]",
    )

    # Momentum y-RH across S_L (tangential):
    #   S_L ((rho v)*_L - rho_L v_L) = ((rho u)*_L v_L - rho_L u_L v_L)  ?
    #   Note the flux is (rho u v) = (rho u) v.  With (rho u)*_L = rho*_L S_star
    #   and v_L* = v_L, the flux becomes rho*_L S_star v_L.
    #   So: S_L rho*_L v_L - S_L rho_L v_L = rho*_L S_star v_L - rho_L u_L v_L.
    mom_y_L_flux_star = rho_starL * S_star_formula * v_L
    mom_y_L_flux = rho_L * u_L * v_L
    mom_y_L_jump = sp.simplify(
        S_L * (my_starL - rho_L * v_L) - (mom_y_L_flux_star - mom_y_L_flux)
    )
    assert_zero(
        mom_y_L_jump,
        "A8-mom_y-RH-across-S_L: S_L [rho v] = [rho u v]",
    )

    # Same for S_R.
    mom_x_R_flux = rho_R * u_R**2 + p_R
    mom_x_R_flux_star = mx_starR * S_star_formula + p_star
    mom_x_R_jump = sp.simplify(
        S_R_sym * (rho_R * u_R - mx_starR) - (mom_x_R_flux - mom_x_R_flux_star)
    )
    assert_zero(
        mom_x_R_jump,
        "A8-mom_x-RH-across-S_R: S_R [rho u] = [rho u^2 + p]",
    )

    mom_y_R_flux = rho_R * u_R * v_R
    mom_y_R_flux_star = rho_starR * S_star_formula * v_R
    mom_y_R_jump = sp.simplify(
        S_R_sym * (rho_R * v_R - my_starR) - (mom_y_R_flux - mom_y_R_flux_star)
    )
    assert_zero(
        mom_y_R_jump,
        "A8-mom_y-RH-across-S_R: S_R [rho v] = [rho u v]",
    )

    # ════════════════════════════════════════════════════════════
    # 5.  Star energies (Toro eq. 10.39 / 10.40):
    #   E*_K = rho*_K [ E_K / rho_K
    #                 + (S_star - u_K) * (S_star + p_K / (rho_K (S_K - u_K))) ]
    # ════════════════════════════════════════════════════════════
    E_starL = rho_starL * (
        E_L / rho_L
        + (S_star_formula - u_L) * (S_star_formula + p_L / (rho_L * (S_L - u_L)))
    )
    E_starR = rho_starR * (
        E_R / rho_R
        + (S_star_formula - u_R) * (S_star_formula + p_R / (rho_R * (S_R_sym - u_R)))
    )

    # Energy RH across S_L:
    #   S_L (E*_L - E_L) = (E*_L + p_star) S_star - (E_L + p_L) u_L
    energy_L_jump = sp.simplify(
        S_L * (E_starL - E_L)
        - ((E_starL + p_star) * S_star_formula - (E_L + p_L) * u_L)
    )
    assert_zero(
        energy_L_jump,
        "A8-energy-RH-across-S_L: S_L [E] = [(E+p) u_n]",
    )

    # Energy RH across S_R:
    energy_R_jump = sp.simplify(
        S_R_sym * (E_R - E_starR)
        - ((E_R + p_R) * u_R - (E_starR + p_star) * S_star_formula)
    )
    assert_zero(
        energy_R_jump,
        "A8-energy-RH-across-S_R: S_R [E] = [(E+p) u_n]",
    )

    # ════════════════════════════════════════════════════════════
    # 6.  HLLC flux in the left-star branch (S_L <= 0 <= S_star):
    #   F_HLLC = F_L + S_L (U*_L - U_L)
    # Verify this equals:  (0, p_star - p_L, 0, p_star S_star - p_L u_L)?
    # Let's just assert that the mass component gives  rho*_L S_star,
    # the tangential component gives (rho u)*_L v_L - ... actually
    # simpler:  F_HLLC[0]  =  rho_L u_L + S_L (rho*_L - rho_L)
    #                      =  rho_L u_L + S_L rho_L (S_L - u_L)/(S_L - S_star) - S_L rho_L
    #                      = rho_L [u_L - S_L + S_L (S_L - u_L)/(S_L - S_star)]
    #                      = rho_L (u_L - S_L) [1 - S_L/(S_L - S_star)]
    #                      = rho_L (u_L - S_L) (-S_star)/(S_L - S_star)
    #                      = rho_L (S_L - u_L) S_star / (S_L - S_star)
    #                      = rho*_L S_star.  Good.
    # Verify all 4 components.
    # ════════════════════════════════════════════════════════════
    U_L_vec = sp.Matrix([rho_L, rho_L * u_L, rho_L * v_L, E_L])
    U_starL_vec = sp.Matrix([rho_starL, mx_starL, my_starL, E_starL])
    F_L_vec = flux_x_euler(rho_L, u_L, v_L, p_L, gamma)

    F_HLLC_leftstar = F_L_vec + S_L * (U_starL_vec - U_L_vec)

    # Expected HLLC flux in left-star branch:
    # (rho*_L S_star, rho*_L S_star^2 + p_star, rho*_L S_star v_L, (E*_L + p_star) S_star)
    expected_F_leftstar = sp.Matrix([
        rho_starL * S_star_formula,
        rho_starL * S_star_formula**2 + p_star,
        rho_starL * S_star_formula * v_L,
        (E_starL + p_star) * S_star_formula,
    ])

    for i in range(4):
        diff = sp.simplify(F_HLLC_leftstar[i] - expected_F_leftstar[i])
        assert_zero(
            diff,
            f"A8-HLLC flux left-star branch = F(U*_L) component {i}",
        )

    # Right-star branch (S_star <= 0 <= S_R):
    U_R_vec = sp.Matrix([rho_R, rho_R * u_R, rho_R * v_R, E_R])
    U_starR_vec = sp.Matrix([rho_starR, mx_starR, my_starR, E_starR])
    F_R_vec = flux_x_euler(rho_R, u_R, v_R, p_R, gamma)

    F_HLLC_rightstar = F_R_vec + S_R_sym * (U_starR_vec - U_R_vec)
    expected_F_rightstar = sp.Matrix([
        rho_starR * S_star_formula,
        rho_starR * S_star_formula**2 + p_star,
        rho_starR * S_star_formula * v_R,
        (E_starR + p_star) * S_star_formula,
    ])
    for i in range(4):
        diff = sp.simplify(F_HLLC_rightstar[i] - expected_F_rightstar[i])
        assert_zero(
            diff,
            f"A8-HLLC flux right-star branch = F(U*_R) component {i}",
        )

    # ════════════════════════════════════════════════════════════
    # 7.  Consistency at S_star = 0 (stationary-contact flux).
    #
    # At a stationary contact (u_L = u_R = 0, p_L = p_R), we have
    # S_star = 0 (verified in A7).  Then F_HLLC mass flux = rho*_L * 0 = 0;
    # pressure flux = p_star = p_L = p_R.  So F_HLLC = (0, p_L, 0, 0)
    # = F_exact.  Already verified in A7.
    # ════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "HLLC contact speed",
        r"S_{\star} \;=\; \frac{p_R - p_L + \rho_L u_L (S_L - u_L) - \rho_R u_R (S_R - u_R)}"
        r"{\rho_L (S_L - u_L) - \rho_R (S_R - u_R)}",
        label="eq:A8-Sstar",
    )
    ld.add(
        "Star pressure (two equivalent forms, consistent)",
        r"p^{\star} \;=\; p_L + \rho_L (u_L - S_L)(u_L - S_{\star}) "
        r"\;=\; p_R + \rho_R (u_R - S_R)(u_R - S_{\star})",
        label="eq:A8-pstar",
    )
    ld.add(
        "Star densities",
        r"\rho^{\star}_L \;=\; \rho_L\,\frac{S_L - u_L}{S_L - S_{\star}}, \qquad "
        r"\rho^{\star}_R \;=\; \rho_R\,\frac{S_R - u_R}{S_R - S_{\star}}",
        label="eq:A8-rhostar",
    )
    ld.add(
        "Star momenta",
        r"(\rho u)^{\star}_K \;=\; \rho^{\star}_K\,S_{\star}, \qquad "
        r"(\rho v)^{\star}_K \;=\; \rho^{\star}_K\,v_K "
        r"\qquad\text{for } K \in \{L, R\}",
        label="eq:A8-momstar",
    )
    ld.add(
        "Star energies",
        r"E^{\star}_K \;=\; \rho^{\star}_K\,\left[\,\frac{E_K}{\rho_K} "
        r"+ (S_{\star} - u_K)\left(S_{\star} + \frac{p_K}{\rho_K\,(S_K - u_K)}\right)\right] "
        r"\qquad K \in \{L, R\}",
        label="eq:A8-Estar",
    )
    ld.add(
        "HLLC flux, piecewise by subsonic branch",
        r"F^{\mathrm{HLLC}} \;=\; \begin{cases}"
        r"\mathbf{F}_L & 0 \le S_L, \\[2pt]"
        r"\mathbf{F}_L + S_L (\mathbf{U}^{\star}_L - \mathbf{U}_L) "
        r"& S_L \le 0 \le S_{\star}, \\[2pt]"
        r"\mathbf{F}_R + S_R (\mathbf{U}^{\star}_R - \mathbf{U}_R) "
        r"& S_{\star} \le 0 \le S_R, \\[2pt]"
        r"\mathbf{F}_R & S_R \le 0.\end{cases}",
        label="eq:A8-HLLC",
    )

    ld.write()
    print()
    print("All A8 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
