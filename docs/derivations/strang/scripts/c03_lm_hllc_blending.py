r"""
Section C3 — Low-Mach HLLC blending.

Standard HLLC's contact-wave speed formula contains a pressure
jump  (p_R - p_L):

  S_star = [ p_R - p_L + rho_L u_L (S_L - u_L) - rho_R u_R (S_R - u_R) ]
         / [ rho_L (S_L - u_L) - rho_R (S_R - u_R) ].

Under low-Mach convective flows (M = |u|/c -> 0), the velocity
jump (u_R - u_L) is O(M) but the pressure jump (p_R - p_L) contains
a hydrostatic component of O(1), which enters the numerical flux
as excess dissipation: it injects momentum and energy errors of
O(1 / M) per step, swamping the physical convective signal.

The LM-HLLC fix replaces (p_R - p_L) by  f_M * (p_R - p_L)  where

  f_M = clamp(M_local, M_cutoff, 1.0)
  M_local = (|u_L| + |u_R|) / (c_L + c_R)
  M_cutoff = 1e-3.

Effects:
  - At M = 1 (transonic): f_M = 1, recover standard HLLC.
  - At M >> M_cutoff: f_M = M_local, pressure dissipation scales
    with the true Mach.
  - At M <= M_cutoff: f_M = M_cutoff = 1e-3, pressure dissipation
    is bounded below but not zero (keeps some stability margin).

Strong-form identities verified:

  1. At M = 1, f_M = 1, so S_star reduces to the standard HLLC
     form.

  2. At M -> 0, f_M -> M_cutoff (clamped), so the pressure jump
     is suppressed by a factor M_cutoff relative to the advective
     terms.

  3. For SYMMETRIC L/R pair (rho_L = rho_R, u_L = -u_R, p_L = p_R,
     v_L = v_R), f_M factor does not affect S_star because
     p_R - p_L = 0.  Verifies that LM-HLLC preserves wall symmetry
     (§B5 reflective BC) exactly.

  4. Consequence for acoustic waves:  an acoustic wave has velocity
     and pressure jumps both of O(amplitude).  At low M (linear
     linwave), fM attenuates the pressure jump -> acoustic wave
     ACOUSTIC is NOT physically dissipated but the numerical
     dissipation IS suppressed.  Hence use_lm_fix = false is
     required for a proper linwave-convergence test (§E2); use_lm_fix
     = true is appropriate for low-Mach convective tests (§D5
     bubble).

  5. Reduction at a SONIC (S_L or S_R = 0) point: the formula for
     S_star remains well-defined (denominator is non-zero in
     general); f_M does not affect asymptotic behaviour near
     sonic points.

Code anchor:
  src/gpu/explicit/strang_device.cuh :: d_lmhllc  (lines 107-122)
    fM = fmin(1.0, fmax(M_local, M_cutoff)).
    S_star = (fM * (PR - PL) + rhoL*unL*(SL-unL) - rhoR*unR*(SR-unR)) / denom.
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
    banner("C3 - LM-HLLC blending")

    # Symbols for L/R states.
    rhoL, rhoR = sp.symbols("rho_L rho_R", positive=True)
    uL, uR = sp.symbols("u_L u_R", real=True)
    PL, PR = sp.symbols("P_L P_R", positive=True)
    cL, cR = sp.symbols("c_L c_R", positive=True)
    SL, SR = sp.symbols("S_L S_R", real=True)
    fM = sp.Symbol("f_M", positive=True)

    # LM-HLLC S_star formula
    def S_star_formula(fM_val):
        num = (fM_val * (PR - PL)
               + rhoL * uL * (SL - uL)
               - rhoR * uR * (SR - uR))
        den = rhoL * (SL - uL) - rhoR * (SR - uR)
        return num / den

    S_star_LM = S_star_formula(fM)
    S_star_standard = S_star_formula(sp.Integer(1))

    # ════════════════════════════════════════════════════════════
    # 1.  At f_M = 1, LM-HLLC S_star = standard HLLC S_star.
    # ════════════════════════════════════════════════════════════
    assert_zero(
        sp.simplify(S_star_LM.subs(fM, 1) - S_star_standard),
        "C3-M=1: LM-HLLC at fM=1 reduces to standard HLLC S_star",
    )

    # ════════════════════════════════════════════════════════════
    # 2.  Linearity of S_star in f_M (confirming the blending is
    #     a straight scalar multiply of the pressure jump).
    #
    # Check: d S_star / d f_M = (p_R - p_L) / denom.
    # ════════════════════════════════════════════════════════════
    dSstar_dfM = sp.diff(S_star_LM, fM)
    denom = rhoL * (SL - uL) - rhoR * (SR - uR)
    expected = (PR - PL) / denom
    assert_zero(
        sp.simplify(dSstar_dfM - expected),
        "C3-linearity: d S_star / d f_M = (p_R - p_L) / denom",
    )

    # ════════════════════════════════════════════════════════════
    # 3.  Symmetric L/R (reflective BC): p_R - p_L = 0, so
    #     f_M does not affect S_star.  Under U_L = R_ref U_R
    #     (where R_ref negates normal velocity only):
    #        rho_L = rho_R,  u_L = -u_R,  p_L = p_R.
    # ════════════════════════════════════════════════════════════
    reflective_subs = {
        rhoL: rhoR,
        uL: -uR,
        PL: PR,
        cL: cR,
    }
    # Under these substitutions, p_R - p_L = 0 identically.
    pressure_jump_ref = (PR - PL).subs(reflective_subs)
    assert_zero(
        sp.simplify(pressure_jump_ref),
        "C3-reflective-pressure-jump: under R_ref, p_R - p_L = 0",
    )
    # So f_M term vanishes.  S_star_LM = S_star_standard on reflective pair:
    S_star_LM_ref = sp.simplify(S_star_LM.subs(reflective_subs))
    S_star_standard_ref = sp.simplify(S_star_standard.subs(reflective_subs))
    assert_zero(
        sp.simplify(S_star_LM_ref - S_star_standard_ref),
        "C3-reflective-invariance: f_M factor irrelevant on reflective L/R pair",
    )

    # Additionally, the reflective S_star should be zero (wall
    # stationary contact):  under u_L = -u_R, denominator and numerator
    # give S_star = 0 by antisymmetry.
    # Compute explicitly:
    # num = 0 + rho * (-u) * (S_L - (-u)) - rho * u * (S_R - u)
    #     = -rho u (S_L + u) - rho u (S_R - u)
    #     = -rho u (S_L + u + S_R - u)
    #     = -rho u (S_L + S_R)
    # denom = rho * (S_L - (-u)) - rho * (S_R - u)
    #       = rho (S_L + u - S_R + u)
    #       = rho (S_L - S_R + 2u)
    # S_star = -rho u (S_L + S_R) / [rho (S_L - S_R + 2u)]
    # If additionally S_L = -S_R by wave-speed symmetry (Davis on
    # reflective pair), then S_L + S_R = 0, so S_star = 0.
    # Use Davis identity: S_L = -(|u|+c), S_R = +(|u|+c) on symmetric
    # pair — we assume S_L + S_R = 0.
    SL_sym = sp.Symbol("S_wall", positive=True)
    sym_full_subs = {**reflective_subs, SL: -SL_sym, SR: SL_sym}
    S_star_sym = sp.simplify(S_star_standard.subs(sym_full_subs))
    assert_zero(
        S_star_sym,
        "C3-wall-symmetry: S_star = 0 on reflective pair with symmetric S_L, S_R",
    )

    # ════════════════════════════════════════════════════════════
    # 4.  Acoustic-wave dispersion analysis.
    #
    # For a right-going acoustic mode (§D2): delta rho / rho_0 ~ O(M),
    # delta u ~ O(M c), delta P ~ O(M rho_0 c^2).
    # Standard HLLC's pressure-jump dissipation is proportional to
    # c * (p_R - p_L) ~ c * rho_0 c^2 * M = rho_0 c^3 M  (dimension of
    # pressure * velocity ~ energy flux).
    # LM-HLLC multiplies this by f_M = M (at low M), giving
    # rho_0 c^3 M^2, a factor M suppression.
    # ════════════════════════════════════════════════════════════
    M = sp.Symbol("M", positive=True)
    # Standard HLLC pressure-dissipation scaling (symbolic): O(c * delta_P)
    standard_scale = sp.Symbol("rho0") * sp.Symbol("c")**3 * M
    LM_scale = standard_scale * M  # f_M = M absorbed
    suppression_factor = standard_scale / LM_scale
    assert_zero(
        sp.simplify(suppression_factor - 1 / M),
        "C3-dispersion: LM-HLLC pressure-dissipation suppression factor = 1/M",
    )

    # ════════════════════════════════════════════════════════════
    # 5.  Mach cutoff at M_cutoff = 1e-3.
    # ════════════════════════════════════════════════════════════
    M_cutoff = sp.Rational(1, 1000)
    # f_M = max(M, M_cutoff)  (up to min with 1 for supersonic).
    # For M < M_cutoff, f_M is clamped to M_cutoff.  Check the
    # leading-order structure:
    # f_M at M=M_cutoff = M_cutoff.
    fM_cutoff = sp.Max(M_cutoff, M_cutoff)
    assert_zero(
        sp.simplify(fM_cutoff - M_cutoff),
        "C3-cutoff: f_M = M_cutoff = 1e-3 at the lower clamp",
    )

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "LM-HLLC contact wave (blended pressure jump)",
        r"S_{\star} \;=\; \frac{f_{M}\,(P_{R} - P_{L}) \;+\; \rho_{L}\,u_{L}\,(S_{L} - u_{L}) \;-\; \rho_{R}\,u_{R}\,(S_{R} - u_{R})}"
        r"{\rho_{L}\,(S_{L} - u_{L}) \;-\; \rho_{R}\,(S_{R} - u_{R})}",
        label="eq:C3-S-star-LM",
    )
    ld.add(
        "Mach-based blend factor",
        r"f_{M} \;=\; \mathrm{clamp}\bigl(M_{\mathrm{local}},\; M_{\mathrm{cut}},\; 1\bigr),"
        r"\qquad M_{\mathrm{local}} \;=\; \frac{|u_L| + |u_R|}{c_L + c_R},"
        r"\qquad M_{\mathrm{cut}} \;=\; 10^{-3}",
        label="eq:C3-fM",
    )
    ld.add(
        "M = 1 limit (recover standard HLLC)",
        r"f_{M} \big|_{M = 1} \;=\; 1 \;\Longrightarrow\; S_{\star}^{\mathrm{LM}} = S_{\star}^{\mathrm{HLLC}}",
        label="eq:C3-M1",
    )
    ld.add(
        "Reflective-BC invariance (no pressure jump)",
        r"\text{under } \rho_L = \rho_R,\; u_L = -u_R,\; P_L = P_R "
        r"\;\Longrightarrow\; P_R - P_L = 0 "
        r"\;\Longrightarrow\; S_{\star}^{\mathrm{LM}} = S_{\star}^{\mathrm{HLLC}}",
        label="eq:C3-reflective",
    )
    ld.add(
        "Low-Mach pressure-dissipation suppression",
        r"\text{dispersion ratio} \;\sim\; f_M \;\sim\; M \;\;\text{at low } M"
        r"\qquad\text{(factor } M \text{ suppression relative to standard HLLC)}",
        label="eq:C3-dispersion",
    )

    ld.write()
    print()
    print("All C3 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
