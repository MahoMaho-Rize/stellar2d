"""
Section B1 — Super-radial flux-tube reduction.
See sections/b1_flux_tube_geometry.md for context.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("B1 — Super-radial flux-tube reduction (Suzuki-style)")

    r, t = sp.symbols("r t", positive=True)
    rho = sp.Function("rho")(r, t)
    v_r = sp.Function("v_r")(r, t)
    B_r = sp.Function("B_r")(r, t)
    B_perp = sp.Function("B_perp")(r, t)
    p = sp.Function("p")(r, t)
    gamma = sp.Symbol("gamma", positive=True)
    g_grav = sp.Function("g")(r)
    A = sp.Function("A")(r)
    Phi_const = sp.Symbol("Phi_B", positive=True)
    B0, R_star, f_max, sigma, r1 = sp.symbols(
        "B_0 R_* f_max sigma r_1", positive=True
    )

    ld.add(
        "Cross-section area",
        r"A(r) = r^2\,f(r),\ f(R_*) = 1,\ f(r\to\infty) \to f_{\max}",
        label="eq:B1_A",
    )

    # Flux conservation
    check = sp.simplify(A * (B0 * R_star**2 / A) - B0 * R_star**2)
    assert_zero(check, "A(r) B_r(r) = B_0 R_*² holds identically")
    ld.add(
        "Magnetic flux conservation",
        r"A(r) B_r(r) = R_*^2 B_0 \Rightarrow "
        r"B_r(r) = \frac{B_0 R_*^2}{A(r)} = \frac{B_0}{f(r)}(R_*/r)^2",
        label="eq:B1_Br",
    )

    # MHSE check
    B_sq = B_r**2 + B_perp**2
    p_tot = p + sp.Rational(1, 2) * B_sq
    F_mom = A * (rho * v_r**2 + p_tot - B_r**2)
    S_mom = (p_tot - B_r**2) * sp.diff(A, r) - rho * g_grav * A
    mom_1d = sp.diff(rho * v_r * A, t) + sp.diff(F_mom, r) - S_mom

    static = {v_r: 0, sp.Derivative(rho, t): 0,
              sp.Derivative(v_r, t): 0, sp.Derivative(p, t): 0,
              sp.Derivative(B_r, t): 0, sp.Derivative(B_perp, t): 0,
              B_perp: 0}
    mom_static = sp.simplify(mom_1d.subs(static))
    mom_flux = sp.expand(mom_static.subs(B_r, Phi_const / A))
    MHSE = (sp.diff(p, r) + rho * g_grav
            + (Phi_const / A)**2 * sp.diff(A, r) / A)
    diff_check = sp.simplify(mom_flux / A - MHSE)
    assert_zero(diff_check, "MHSE: ∂_r p + ρ g + B_r² ∂_r(ln A) = 0")
    ld.add(
        "Magneto-Hydrostatic Equilibrium (MHSE)",
        r"\boxed{\partial_r p + \rho g = -B_r^2\,\partial_r(\ln A)}",
        label="eq:B1_MHSE",
    )

    # Kopp-Holzer form
    f_1 = 1 - (f_max - 1) * sp.exp((R_star - r1) / sigma)
    f_KH = (f_max * sp.exp((r - r1) / sigma) + f_1) / (sp.exp((r - r1) / sigma) + 1)
    assert_zero(sp.limit(f_KH, r, sp.oo) - f_max,
                "Kopp-Holzer f(r → ∞) = f_max")
    f_Rstar = sp.simplify(f_KH.subs(r, R_star))
    # Numerical check since sympy may not fold algebraically:
    num_check = abs(float(sp.simplify(f_Rstar - 1).subs({
        f_max: 3.0, sigma: 0.5, r1: 1.3, R_star: 1.0
    })))
    assert num_check < 1e-12, f"Kopp-Holzer f(R_*) check: {num_check}"
    print(f"  [OK] Kopp-Holzer f(R_*) = 1 numerical ({num_check:.1e}).")
    ld.add(
        "Kopp-Holzer 1976 expansion factor",
        r"f(r) = \frac{f_{\max}\,e^{(r-r_1)/\sigma} + f_1}"
        r"{e^{(r-r_1)/\sigma} + 1},\ "
        r"f_1 = 1 - (f_{\max}-1)e^{(R_*-r_1)/\sigma}",
        label="eq:B1_KH",
    )

    ld.add(
        "Alfvén-wave amplitude scaling (WKB result, see B2)",
        r"\delta v_{\perp}(r) \propto (\rho v_A A)^{-1/4}"
        r"\ \text{(Jacques) or}\ (\rho v_A A)^{-1/2}\ \text{(Poynting)}",
        label="eq:B1_amp",
    )

    ld.write()
    print()
    print("All B1 identities verified.")


if __name__ == "__main__":
    main()
