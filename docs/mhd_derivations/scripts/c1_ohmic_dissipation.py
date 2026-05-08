"""
Section C1 — Ohmic dissipation.
See sections/c1_ohmic_dissipation.md for context.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner, curl_cart, grad_cart, cross


def main():
    ld = LatexDump(__file__)
    banner("C1 — Ohmic (resistive) dissipation")

    x, y, z, t = sp.symbols("x y z t", real=True)
    eta_O = sp.Function("eta_O")(x, y, z)
    Bx = sp.Function("B_x")(x, y, z, t)
    By = sp.Function("B_y")(x, y, z, t)
    Bz = sp.Function("B_z")(x, y, z, t)
    B = sp.Matrix([Bx, By, Bz])
    J = curl_cart(B)

    ld.add(
        "Ohm's law with resistivity",
        r"\mathbf{E} = -\mathbf{v}\times\mathbf{B} + \eta_O\,\mathbf{J},"
        r"\ \mathbf{J} = \nabla\times\mathbf{B}",
        label="eq:C1_Ohm",
    )

    # ∇×(∇×B) = ∇(∇·B) - ∇²B identity
    laplacian_B = sp.Matrix([
        sum(sp.diff(B[i], c, 2) for c in (x, y, z)) for i in range(3)
    ])
    from _common import div_cart
    identity = sp.Matrix([
        curl_cart(curl_cart(B))[i]
        - (grad_cart(div_cart(B))[i] - laplacian_B[i])
        for i in range(3)
    ])
    for i in range(3):
        assert_zero(sp.simplify(identity[i]),
                    f"∇×(∇×B) = ∇(∇·B) − ∇²B comp {i}", verbose=False)
    print("  [OK] ∇×(∇×B) = ∇(∇·B) − ∇²B (3 components).")

    # ∇×(η_O J) = η_O ∇×J + (∇η_O)×J
    curl_etaJ = curl_cart([eta_O * J[i] for i in range(3)])
    expected = sp.Matrix([eta_O * curl_cart(J)[i]
                          + cross(grad_cart(eta_O), J)[i] for i in range(3)])
    for i in range(3):
        assert_zero(sp.simplify(curl_etaJ[i] - expected[i]),
                    f"∇×(η_O J) = η_O ∇×J + (∇η_O)×J comp {i}",
                    verbose=False)
    print("  [OK] ∇×(η_O J) variable-η expansion (3 components).")

    ld.add(
        "Induction with constant resistivity",
        r"\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B}) "
        r"+ \eta_O\,\nabla^2\mathbf{B}",
        label="eq:C1_induction_const",
    )
    ld.add(
        "Induction with variable resistivity",
        r"\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B}) "
        r"+ \eta_O\,\nabla^2\mathbf{B} - (\nabla\eta_O)\times\mathbf{J}",
        label="eq:C1_induction_var",
    )
    ld.add(
        "Joule heating",
        r"Q_{\mathrm{Ohm}} = \eta_O\,|\mathbf{J}|^2 \ge 0",
        label="eq:C1_Q",
    )
    ld.add(
        "Explicit CFL",
        r"\Delta t \le (\Delta x)^2/(2\eta_O)\ \text{1D},\ "
        r"(\Delta x)^2/(4\eta_O)\ \text{2D},\ "
        r"(\Delta x)^2/(6\eta_O)\ \text{3D}",
        label="eq:C1_CFL",
    )

    ld.write()
    print()
    print("All C1 identities verified.")


if __name__ == "__main__":
    main()
