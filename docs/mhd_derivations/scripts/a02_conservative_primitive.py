"""
Section A2 — Conservative ↔ primitive variable transformation.

Derivation targets:
  1. Define  U = (ρ, m_x, m_y, m_z, B_x, B_y, B_z, E)
            W = (ρ, v_x, v_y, v_z, B_x, B_y, B_z, p)
     with E = ρe + ½ρ|v|² + ½|B|², e = p/((γ-1)ρ).
  2. Compute the Jacobian dW/dU symbolically.
  3. Compute dU/dW symbolically.
  4. sympy-verify dW/dU · dU/dW = I_8 (as symbolic 8×8 matrices).
  5. Print the pressure extraction from conservatives:
     p = (γ-1)·(E − m²/(2ρ) − |B|²/2)

These are the formulas every MHD kernel uses at the primitive-reconstruction
step before a Riemann solve.  Any bug here propagates to every Godunov
flux, so this identity is worth locking symbolically.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp                                # noqa: E402
from _common import (                             # noqa: E402
    rho, p, gamma, v_x, v_y, v_z, B_x, B_y, B_z, E as E_sym,
    LatexDump, assert_zero, banner,
)

def main():
    ld = LatexDump(__file__)
    banner("A2 — Conservative ↔ primitive variable transformation")

    # ════════════════════════════════════════════════════════════
    # Primitive and conservative 8-vectors
    # ════════════════════════════════════════════════════════════
    W = sp.Matrix([rho, v_x, v_y, v_z, B_x, B_y, B_z, p])

    m_x = rho * v_x
    m_y = rho * v_y
    m_z = rho * v_z
    v_sq = v_x**2 + v_y**2 + v_z**2
    B_sq = B_x**2 + B_y**2 + B_z**2
    E_expr = p / (gamma - 1) + sp.Rational(1, 2) * rho * v_sq + sp.Rational(1, 2) * B_sq

    U = sp.Matrix([rho, m_x, m_y, m_z, B_x, B_y, B_z, E_expr])

    ld.add(
        "Primitive 8-vector",
        r"\mathbf{W} = (\rho,\ v_x,\ v_y,\ v_z,\ B_x,\ B_y,\ B_z,\ p)^{\mathrm{T}}",
        label="eq:W_def",
    )
    ld.add(
        "Conservative 8-vector",
        r"\mathbf{U} = (\rho,\ \rho v_x,\ \rho v_y,\ \rho v_z,\ "
        r"B_x,\ B_y,\ B_z,\ E)^{\mathrm{T}}",
        label="eq:U_def",
    )
    ld.add(
        "Total energy density",
        r"E = \frac{p}{\gamma - 1} + \tfrac{1}{2}\rho|\mathbf{v}|^{2} "
        r"+ \tfrac{1}{2}|\mathbf{B}|^{2}",
        label="eq:E_def",
    )

    # ════════════════════════════════════════════════════════════
    # dU / dW
    # ════════════════════════════════════════════════════════════
    dUdW = U.jacobian(W)
    print("\ndU/dW =")
    sp.pprint(dUdW)
    ld.add_expr("Jacobian dU/dW (8×8)", dUdW, label="eq:dUdW")

    # ════════════════════════════════════════════════════════════
    # Express W as a function of U (reverse map)
    # ════════════════════════════════════════════════════════════
    # Use U1..U8 as fresh conservative symbols; we'll solve for primitives.
    U1, U2, U3, U4, U5, U6, U7, U8 = sp.symbols(
        "U_1 U_2 U_3 U_4 U_5 U_6 U_7 U_8", real=True
    )
    # primitives as explicit functions of (U_i, γ):
    rho_p  = U1
    v_x_p  = U2 / U1
    v_y_p  = U3 / U1
    v_z_p  = U4 / U1
    B_x_p  = U5
    B_y_p  = U6
    B_z_p  = U7
    m_sq   = U2**2 + U3**2 + U4**2
    B_sq_p = U5**2 + U6**2 + U7**2
    p_p = (gamma - 1) * (U8 - m_sq / (2 * U1) - B_sq_p / 2)

    W_of_U = sp.Matrix([rho_p, v_x_p, v_y_p, v_z_p, B_x_p, B_y_p, B_z_p, p_p])
    U_sym = sp.Matrix([U1, U2, U3, U4, U5, U6, U7, U8])

    ld.add(
        "Pressure extraction",
        r"p = (\gamma - 1)\!\left(E - \frac{|\mathbf{m}|^{2}}{2\rho} "
        r"- \tfrac{1}{2}|\mathbf{B}|^{2}\right)",
        label="eq:p_extract",
    )

    # ════════════════════════════════════════════════════════════
    # dW / dU
    # ════════════════════════════════════════════════════════════
    dWdU = W_of_U.jacobian(U_sym)
    print("\ndW/dU =")
    sp.pprint(dWdU)
    ld.add_expr("Jacobian dW/dU (8×8)", dWdU, label="eq:dWdU")

    # ════════════════════════════════════════════════════════════
    # Verify dW/dU · dU/dW  =  I_8  (after substituting U_i with U(W))
    # ════════════════════════════════════════════════════════════
    subs_map = {
        U1: U[0], U2: U[1], U3: U[2], U4: U[3],
        U5: U[4], U6: U[5], U7: U[6], U8: U[7],
    }
    dWdU_at_W = dWdU.subs(subs_map)

    product = dWdU_at_W * dUdW
    product_simp = sp.simplify(product)

    print("\n(dW/dU)·(dU/dW) =")
    sp.pprint(product_simp)

    I8 = sp.eye(8)
    residual = sp.simplify(product_simp - I8)
    for i in range(8):
        for j in range(8):
            assert_zero(
                residual[i, j],
                f"(dW/dU·dU/dW - I)[{i},{j}]",
                verbose=False,
            )
    print("  [OK] dW/dU · dU/dW = I_8 verified symbolically (all 64 entries).")

    ld.add(
        "Inverse relation",
        r"\frac{\partial\mathbf{W}}{\partial\mathbf{U}}\cdot"
        r"\frac{\partial\mathbf{U}}{\partial\mathbf{W}} = \mathsf{I}_{8}",
        label="eq:jacobian_inverse",
    )

    ld.write()
    print()
    print("All A2 identities verified by sympy.")

if __name__ == "__main__":
    main()
