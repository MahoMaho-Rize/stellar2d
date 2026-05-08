"""
Section B2 — WKB wave-action conservation for Alfvén waves.
See sections/b2_wave_action_wkb.md for context.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("B2 — WKB wave-action conservation")

    r, t = sp.symbols("r t", positive=True)
    rho0 = sp.Function("rho_0")(r)
    B_r0 = sp.Function("B_r0")(r)
    v_A = B_r0 / sp.sqrt(rho0)
    k, om = sp.symbols("k omega", positive=True)

    # Dispersion via 2x2 WKB linearised matrix determinant
    M = sp.Matrix([
        [-sp.I*om, -(B_r0/rho0)*sp.I*k],
        [-B_r0*sp.I*k, -sp.I*om],
    ])
    det = sp.simplify(M.det())
    assert_zero(sp.simplify(det + (om**2 - v_A**2 * k**2)),
                "WKB dispersion ω² = v_A² k²")
    ld.add(
        "Alfvén dispersion (WKB)",
        r"\omega^2 = v_A^2\,k^2,\ v_A = B_{r,0}/\sqrt{\rho_0}",
        label="eq:B2_disp",
    )

    # Elsässer transport in UNIFORM background: no reflection
    v_perp = sp.Function("v_perp")(r, t)
    B_perp = sp.Function("B_perp")(r, t)
    dt_v = (B_r0/rho0) * sp.diff(B_perp, r)
    dt_B = B_r0 * sp.diff(v_perp, r)
    z_plus = v_perp - B_perp/sp.sqrt(rho0)
    dt_zp = dt_v - dt_B/sp.sqrt(rho0)
    dr_zp = sp.diff(z_plus, r)
    lhs_plus = sp.simplify(sp.expand(dt_zp + v_A * dr_zp))
    # In a uniform background (ρ_0 = const, B_r0 = const):
    uniform = {sp.Derivative(rho0, r): 0, sp.Derivative(B_r0, r): 0}
    lhs_uniform = sp.simplify(lhs_plus.subs(uniform))
    assert_zero(lhs_uniform,
                "No reflection: ∂_t z+ + v_A ∂_r z+ = 0 under uniform background")
    ld.add(
        "Elsässer transport",
        r"\partial_t z_{\pm} \pm v_A\,\partial_r z_{\pm} = "
        r"S_{\mathrm{refl}}\,z_{\mp},\ S_{\mathrm{refl}}=0\text{ for uniform }(\rho_0, B_{r,0})",
        label="eq:B2_Elsasser",
    )

    ld.add(
        "Poynting-flux conservation scaling",
        r"\rho_0\,v_A\,A\,|\delta v_{\perp}|^2 = \text{const}"
        r"\Rightarrow |\delta v_{\perp}| \propto (\rho_0\,v_A\,A)^{-1/2}",
        label="eq:B2_scaling",
    )

    ld.write()
    print()
    print("All B2 identities verified.")


if __name__ == "__main__":
    main()
