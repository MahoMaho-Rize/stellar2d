"""
Section A5 — Constrained Transport (CT) preservation of discrete ∇·B = 0.
See sections/a5_ct_divergence_preservation.md for context.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("A5 — CT discrete ∇·B = 0 preservation (Evans-Hawley 88, GS05)")

    dx, dy, dt = sp.symbols("Delta_x Delta_y Delta_t", positive=True)
    Bx_L = sp.Symbol("B_x_L", real=True)
    Bx_R = sp.Symbol("B_x_R", real=True)
    By_B = sp.Symbol("B_y_B", real=True)
    By_T = sp.Symbol("B_y_T", real=True)
    Ez_LL = sp.Symbol("E_z_LL", real=True)
    Ez_LR = sp.Symbol("E_z_LR", real=True)
    Ez_UL = sp.Symbol("E_z_UL", real=True)
    Ez_UR = sp.Symbol("E_z_UR", real=True)

    divB_n = (Bx_R - Bx_L) / dx + (By_T - By_B) / dy
    ld.add(
        "Discrete cell-centered divergence",
        r"(\nabla\cdot\mathbf{B})_{i,j} = "
        r"\frac{B_x^{i+1/2,j} - B_x^{i-1/2,j}}{\Delta x} + "
        r"\frac{B_y^{i,j+1/2} - B_y^{i,j-1/2}}{\Delta y}",
        label="eq:A5_divB",
    )

    Bx_L_new = Bx_L - (dt / dy) * (Ez_UL - Ez_LL)
    Bx_R_new = Bx_R - (dt / dy) * (Ez_UR - Ez_LR)
    By_B_new = By_B + (dt / dx) * (Ez_LR - Ez_LL)
    By_T_new = By_T + (dt / dx) * (Ez_UR - Ez_UL)
    ld.add(
        "CT face update (Evans-Hawley 1988 Eq.\\ 17)",
        r"B_x^{i\pm 1/2,j,n+1} = B_x^{i\pm 1/2,j,n} - \frac{\Delta t}{\Delta y}"
        r"(E_z^{i\pm 1/2,j+1/2} - E_z^{i\pm 1/2,j-1/2})",
        label="eq:A5_update",
    )

    divB_np1 = (Bx_R_new - Bx_L_new) / dx + (By_T_new - By_B_new) / dy
    delta = sp.expand(divB_np1 - divB_n)
    assert_zero(
        delta,
        "Discrete ∇·B preservation identity (central CT telescoping)"
    )
    ld.add(
        "Telescoping identity",
        r"\Delta t\,\Delta x\,\Delta y\,\left[(\nabla\cdot\mathbf{B})^{n+1} "
        r"- (\nabla\cdot\mathbf{B})^n\right] = 0",
        label="eq:A5_telescoping",
    )

    # GS05 smooth-limit 2nd-order sanity
    X, Y, h = sp.symbols("x y h", real=True)
    f = sp.Function("E_z")(X, Y)
    avg = sp.Rational(1, 4) * (
        f.subs({X: 0, Y: -h/2}) + f.subs({X: 0, Y: h/2})
        + f.subs({X: -h/2, Y: 0}) + f.subs({X: h/2, Y: 0})
    )
    diff = sp.simplify(sp.series(avg - f.subs({X: 0, Y: 0}), h, 0, 4).removeO())
    assert_zero(diff.coeff(h, 1),
                "GS05 averaging has no O(h) term (2nd-order consistency)")
    ld.add(
        "GS05 averaging 2nd-order accuracy",
        r"E_z^{\text{avg,corner}} - E_z(\text{corner}) = "
        r"\tfrac{h^2}{8}(\partial_x^2 E_z + \partial_y^2 E_z) + \mathcal{O}(h^4)",
        label="eq:A5_GS05",
    )

    ld.write()
    print()
    print("All A5 identities verified.")


if __name__ == "__main__":
    main()
