"""
Section B3 — Parker critical point on a super-radial flux tube.
See sections/b3_parker_critical_point.md for context.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("B3 — Parker critical point (super-radial tube, isothermal)")

    r, v, c_s, G, M_star, r_c, dlnf = sp.symbols(
        "r v c_s G M_* r_c dlnf", positive=True, real=True
    )
    v_f = sp.Function("v")(r)
    A_f = sp.Function("A")(r)
    rho_f = sp.Function("rho")(r)

    # Mass cons → dρ/dr
    mass = (sp.diff(rho_f, r) * v_f * A_f + rho_f * sp.diff(v_f, r) * A_f
            + rho_f * v_f * sp.diff(A_f, r))
    drho_dr = sp.solve(mass, sp.diff(rho_f, r))[0]

    # Momentum: ρv dv/dr + c_s² dρ/dr + ρ GM/r² = 0
    mom = (rho_f * v_f * sp.diff(v_f, r) + c_s**2 * sp.diff(rho_f, r)
           + rho_f * G * M_star / r**2)
    mom_sub = sp.simplify(mom.subs(sp.diff(rho_f, r), drho_dr) / rho_f)
    parker_expected = ((v_f - c_s**2 / v_f) * sp.diff(v_f, r)
                       - c_s**2 * sp.diff(A_f, r) / A_f
                       + G * M_star / r**2)
    assert_zero(sp.simplify(mom_sub - parker_expected),
                "Parker wind equation derivation")
    ld.add(
        "Parker wind equation (super-radial, isothermal)",
        r"\left(v - \frac{c_s^2}{v}\right)\frac{dv}{dr} = "
        r"c_s^2\,\frac{d\ln A}{dr} - \frac{GM_*}{r^2}",
        label="eq:B3_Parker",
    )

    # Spherical limit critical radius
    crit = c_s**2 * (2/r_c + dlnf) - G * M_star / r_c**2
    sols = sp.solve(crit.subs(dlnf, 0), r_c)
    expected = G * M_star / (2 * c_s**2)
    matches = any(sp.simplify(s - expected) == 0 for s in sols)
    assert matches, f"Classical Parker radius not recovered: {sols}"
    print(f"  [OK] Spherical Parker radius: r_c = {expected}")
    ld.add(
        "Classical Parker radius (spherical, f=1)",
        r"r_c = \frac{GM_*}{2 c_s^2}",
        label="eq:B3_rc",
    )

    ld.write()
    print()
    print("All B3 identities verified.")


if __name__ == "__main__":
    main()
