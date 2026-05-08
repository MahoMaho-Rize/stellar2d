r"""
Section E3 — Effective numerical viscosity of LM-HLLC at low Mach.

Builds on §E2's dispersion analysis: quantify the numerical
viscosity coefficient nu_eff under LM-HLLC, compare to standard
HLLC, and estimate the Reynolds number (effective resolution)
the scheme allows at fixed grid resolution and Mach number.

Strong-form identities verified:

  1. nu_eff at fM = M_local: at general Mach,
         nu_eff = M_local * c * Delta x / 2.
     At M = 1, nu_eff = c Dx / 2 (standard).
     At M = 1e-3 (stratified convection ambient), nu_eff =
     5e-4 c Dx.

  2. Effective Reynolds number at fixed resolution N:
         Re_eff = L * u_conv / nu_eff
               = L u_conv / (M * c * Dx / 2)
               = (L / Dx) * (u_conv / (c M / 2))
               = 2 N * (M_conv / M_loc)
     where M_conv = u_conv / c is the convective Mach.
     At M_conv = M_loc: Re_eff = 2 N — depends only on resolution.
     At M_conv < M_loc (subsonic convection): Re_eff < 2 N.

  3. Condition for accurate low-Mach flow:  M_conv ~ M_loc, so
     the LM-HLLC fix reduces nu_eff to the level of the
     physically-relevant convective velocity.  If M_conv << M_loc
     (e.g., M_cut > M_conv in the clamped regime), the
     numerical dissipation dominates the physical transport.

Code anchor:
  src/gpu/explicit/strang_device.cuh :: d_lmhllc fM clamp.
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
    banner("E3 - LM-HLLC effective numerical viscosity")

    c = sp.Symbol("c", positive=True)
    dx = sp.Symbol("Delta_x", positive=True)
    M = sp.Symbol("M", positive=True)
    M_cut = sp.Symbol("M_cut", positive=True)
    L = sp.Symbol("L", positive=True)
    u_conv = sp.Symbol("u_conv", positive=True)
    N_res = sp.Symbol("N", positive=True)

    # ════════════════════════════════════════════════════════════
    # 1.  Effective numerical viscosity.
    #
    # From §E2: nu_eff = fM * c * Dx / 2.
    # With fM = clamp(M_loc, M_cut, 1):
    #   M_loc > 1:       fM = 1
    #   M_cut < M_loc:   fM = M_loc
    #   M_loc <= M_cut:  fM = M_cut (clamped).
    # ════════════════════════════════════════════════════════════
    # Generic form (active regime):
    nu_eff_standard = 1 * c * dx / 2
    nu_eff_LM = M * c * dx / 2   # at active LM regime M > M_cut

    # Ratio:
    ratio_nu = nu_eff_LM / nu_eff_standard
    assert_zero(
        sp.simplify(ratio_nu - M),
        "E3-ratio: nu_eff_LM / nu_eff_standard = M (at M > M_cut)",
    )

    # ════════════════════════════════════════════════════════════
    # 2.  Effective Reynolds number.
    #
    # Re_eff = L u_conv / nu_eff.  For LM-HLLC at fM = M:
    # Re_eff = L u_conv / (M c Dx / 2).
    # Using N = L / Dx and M_conv = u_conv / c:
    # Re_eff = L (M_conv c) / (M c Dx / 2)
    #        = 2 L M_conv / (M Dx)
    #        = 2 N (M_conv / M).
    # ════════════════════════════════════════════════════════════
    Re_LM = L * u_conv / nu_eff_LM
    M_conv = u_conv / c
    expected = 2 * (L / dx) * (M_conv / M)
    assert_zero(
        sp.simplify(Re_LM - expected),
        "E3-Re-eff-LM: Re_eff = 2 N (M_conv / M_loc) under LM-HLLC",
    )

    Re_std = L * u_conv / nu_eff_standard
    expected_std = 2 * (L / dx) * M_conv
    assert_zero(
        sp.simplify(Re_std - expected_std),
        "E3-Re-eff-std: Re_eff = 2 N M_conv under standard HLLC",
    )

    # ════════════════════════════════════════════════════════════
    # 3.  Ratio between LM-HLLC and standard HLLC Re_eff.
    #
    # Re_eff_LM / Re_eff_std = 1/M.
    # At M = 1e-3 (stratified convection), LM-HLLC buys Re * 1000.
    # ════════════════════════════════════════════════════════════
    ratio_Re = Re_LM / Re_std
    assert_zero(
        sp.simplify(ratio_Re - 1/M),
        "E3-Re-ratio: Re_eff_LM / Re_eff_std = 1 / M_local",
    )

    # ════════════════════════════════════════════════════════════
    # 4.  Clamped-regime viscosity: at M <= M_cut, fM = M_cut.
    # ════════════════════════════════════════════════════════════
    nu_eff_LM_clamped = M_cut * c * dx / 2
    Re_LM_clamped = L * u_conv / nu_eff_LM_clamped
    expected_clamped = 2 * (L / dx) * M_conv / M_cut
    assert_zero(
        sp.simplify(Re_LM_clamped - expected_clamped),
        "E3-Re-clamped: Re_eff = 2 N M_conv / M_cut in clamped regime",
    )

    # ════════════════════════════════════════════════════════════
    # 5.  Numerical example: Andrassy convection at M_local = 1e-2,
    # N = 512, L = 1, c = 1.
    # Re_eff_std = 2 * 512 * (1e-2) = 10.24.
    # Re_eff_LM  = 2 * 512 * (1e-2) / (1e-2) = 1024.
    # LM buys ~100x Reynolds.
    # ════════════════════════════════════════════════════════════
    sample_subs = {
        L: 1, dx: sp.Rational(1, 512), u_conv: sp.Rational(1, 100),
        c: 1, M: sp.Rational(1, 100), M_cut: sp.Rational(1, 1000),
    }
    Re_std_val = float(Re_std.subs(sample_subs))
    Re_LM_val = float(Re_LM.subs(sample_subs))
    print(f"  Example (N=512, M_conv = M_loc = 1e-2):")
    print(f"    Re_eff_std = {Re_std_val:.2f}")
    print(f"    Re_eff_LM  = {Re_LM_val:.2f}")
    print(f"    Ratio      = {Re_LM_val/Re_std_val:.1f}x improvement")

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Effective numerical viscosity under LM-HLLC (active regime)",
        r"\nu_{\mathrm{eff}}^{\mathrm{LM}} \;=\; M_{\mathrm{loc}}\,\frac{c\,\Delta x}{2}, "
        r"\qquad\text{(standard HLLC: } \nu_{\mathrm{eff}}^{\mathrm{std}} = c\,\Delta x/2\text{)}",
        label="eq:E3-nu-eff",
    )
    ld.add(
        "Effective Reynolds number (LM-HLLC, active regime)",
        r"\mathrm{Re}_{\mathrm{eff}}^{\mathrm{LM}} \;=\; 2\,N\,\frac{M_{\mathrm{conv}}}{M_{\mathrm{loc}}}, "
        r"\qquad N = L / \Delta x, \;\; M_{\mathrm{conv}} = u_{\mathrm{conv}} / c",
        label="eq:E3-Re-eff",
    )
    ld.add(
        "LM advantage over standard HLLC",
        r"\frac{\mathrm{Re}_{\mathrm{eff}}^{\mathrm{LM}}}{\mathrm{Re}_{\mathrm{eff}}^{\mathrm{std}}} \;=\; \frac{1}{M_{\mathrm{loc}}} "
        r"\qquad\text{(factor of } 10^{3} \text{ at } M = 10^{-3}\text{)}",
        label="eq:E3-advantage",
    )
    ld.add(
        "Clamped regime (M < M_cut)",
        r"\nu_{\mathrm{eff}}^{\mathrm{LM, clamp}} \;=\; M_{\mathrm{cut}}\,\frac{c\,\Delta x}{2}, "
        r"\qquad \mathrm{Re}_{\mathrm{eff}}^{\mathrm{LM, clamp}} \;=\; 2\,N\,\frac{M_{\mathrm{conv}}}{M_{\mathrm{cut}}}",
        label="eq:E3-clamp",
    )

    ld.write()
    print()
    print("All E3 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
