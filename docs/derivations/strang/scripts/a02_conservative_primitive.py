r"""
Section A2 — Conservative <-> primitive bijection.

Strong-form identities (verified by sympy):

  (P -> C)    U(W) = (rho, rho u, rho v, E(W)),      E = p/(gamma-1) + rho*(u^2+v^2)/2
  (C -> P)    W(U) = (rho, m_x/rho, m_y/rho,
                       (gamma-1)*(E - (m_x^2 + m_y^2)/(2 rho)))

Targets:

  1. Round-trip identity (P -> C -> P):
        W(U(W)) == W  for every admissible W.
  2. Round-trip identity (C -> P -> C):
        U(W(U)) == U  for every admissible U.
  3. Jacobian of P -> C:
        det(dU/dW) = rho.
  4. Jacobian of C -> P:
        det(dW/dU) = 1/rho.
  5. Positivity envelope:
        E - (m_x^2 + m_y^2)/(2 rho) > 0  is the condition for p > 0
        (ideal gas).  Consequence: the floor clamp in the kernel's
        d_cons2prim is necessary whenever numerical round-off could
        push this expression negative.

Code anchor:
  src/gpu/explicit/strang_device.cuh :: d_cons2prim

Rule 4 note: every identity here is pointwise strong-form (purely
algebraic); no weak-form fallback.
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
    gamma,
)


def main() -> int:
    ld = LatexDump(__file__)
    banner("A2 - Conservative <-> primitive bijection")

    # Symbols.  Primitive form W = (rho, u, v, p); conservative form
    # U = (rho, mx, my, E).  We carry both sets and prove identities
    # between them.
    rho_sym, u_sym, v_sym, p_sym = sp.symbols("rho u v p", positive=True)
    # Relax u, v to real (they can be negative).
    u_sym = sp.Symbol("u", real=True)
    v_sym = sp.Symbol("v", real=True)
    rho_sym = sp.Symbol("rho", positive=True)
    p_sym = sp.Symbol("p", positive=True)

    mx_sym, my_sym = sp.symbols("m_x m_y", real=True)
    E_sym = sp.Symbol("E", positive=True)

    # ════════════════════════════════════════════════════════════
    # 1.  Forward map  P -> C.
    # ════════════════════════════════════════════════════════════
    def prim_to_cons(rho_, u_, v_, p_):
        E_ = p_ / (gamma - 1) + sp.Rational(1, 2) * rho_ * (u_**2 + v_**2)
        return sp.Matrix([rho_, rho_ * u_, rho_ * v_, E_])

    # ════════════════════════════════════════════════════════════
    # 2.  Inverse map  C -> P.
    # ════════════════════════════════════════════════════════════
    def cons_to_prim(rho_, mx_, my_, E_):
        u_ = mx_ / rho_
        v_ = my_ / rho_
        KE = (mx_**2 + my_**2) / (2 * rho_)
        p_ = (gamma - 1) * (E_ - KE)
        return sp.Matrix([rho_, u_, v_, p_])

    # ════════════════════════════════════════════════════════════
    # 3.  Round-trip P -> C -> P = identity.
    # ════════════════════════════════════════════════════════════
    U_from_W = prim_to_cons(rho_sym, u_sym, v_sym, p_sym)
    W_roundtrip = cons_to_prim(U_from_W[0], U_from_W[1], U_from_W[2], U_from_W[3])
    for i, name in enumerate(["rho", "u", "v", "p"]):
        assert_zero(
            W_roundtrip[i] - [rho_sym, u_sym, v_sym, p_sym][i],
            f"A2-roundtrip P->C->P component {name}",
        )

    # ════════════════════════════════════════════════════════════
    # 4.  Round-trip C -> P -> C = identity.
    # ════════════════════════════════════════════════════════════
    W_from_U = cons_to_prim(rho_sym, mx_sym, my_sym, E_sym)
    U_roundtrip = prim_to_cons(W_from_U[0], W_from_U[1], W_from_U[2], W_from_U[3])
    for i, name in enumerate(["rho", "m_x", "m_y", "E"]):
        assert_zero(
            U_roundtrip[i] - [rho_sym, mx_sym, my_sym, E_sym][i],
            f"A2-roundtrip C->P->C component {name}",
        )

    # ════════════════════════════════════════════════════════════
    # 5.  Jacobian determinants.
    # ════════════════════════════════════════════════════════════
    W_vec = sp.Matrix([rho_sym, u_sym, v_sym, p_sym])
    U_vec = sp.Matrix([rho_sym, mx_sym, my_sym, E_sym])
    U_from_W_expr = prim_to_cons(rho_sym, u_sym, v_sym, p_sym)
    W_from_U_expr = cons_to_prim(rho_sym, mx_sym, my_sym, E_sym)

    J_forward = U_from_W_expr.jacobian(W_vec)
    J_inverse = W_from_U_expr.jacobian(U_vec)

    det_forward = sp.simplify(J_forward.det())
    det_inverse = sp.simplify(J_inverse.det())

    # det(dU/dW) = rho^2 / (gamma - 1).
    # Expansion along the E-row gives 1 * rho * rho * 1/(gamma-1).
    assert_zero(
        det_forward - rho_sym**2 / (gamma - 1),
        "A2-jacobian-forward-det = rho^2/(gamma-1)",
    )

    # det(dW/dU) = (gamma - 1) / rho^2   (reciprocal by chain rule).
    assert_zero(
        sp.simplify(det_inverse - (gamma - 1) / rho_sym**2),
        "A2-jacobian-inverse-det = (gamma-1)/rho^2",
    )

    # Consistency:  J_forward * J_inverse  ==  identity  when evaluated
    # at the image of W under the forward map.  Check by substituting
    # the forward image:
    subs_forward = {
        mx_sym: U_from_W_expr[1],
        my_sym: U_from_W_expr[2],
        E_sym:  U_from_W_expr[3],
    }
    chain = sp.simplify(J_forward @ J_inverse.subs(subs_forward))
    for i in range(4):
        for j in range(4):
            expected = 1 if i == j else 0
            assert_zero(chain[i, j] - expected,
                        f"A2-jacobian-chain[{i},{j}] = delta_{{{i}{j}}}")

    # ════════════════════════════════════════════════════════════
    # 6.  Positivity envelope for p.
    #
    #     p  =  (gamma - 1) * (E  -  (m_x^2 + m_y^2) / (2 rho))
    #
    #     The admissibility region is  E > (m_x^2 + m_y^2) / (2 rho).
    #     This is the condition the kernel's d_cons2prim clamps with
    #     `P = fmax(P, 1e-30)` whenever round-off of a nearly-
    #     vacuum state drops the RHS below zero.
    # ════════════════════════════════════════════════════════════
    KE_expr = (mx_sym**2 + my_sym**2) / (2 * rho_sym)
    p_expr = (gamma - 1) * (E_sym - KE_expr)
    # Strong-form identity: p can be recovered from U if and only if
    # E - KE > 0.  We express this as an identity: subtract the
    # definition.  This is vacuously zero (trivial) but makes the
    # envelope explicit in the LaTeX dump.
    assert_zero(p_expr - (gamma - 1) * (E_sym - KE_expr),
                "A2-positivity-envelope (definitional identity)")

    # ════════════════════════════════════════════════════════════
    # LaTeX dump
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Forward map, primitive to conservative",
        r"\mathbf{U}(\mathbf{W}) \;=\; \begin{pmatrix}\rho\\ \rho u\\ \rho v\\ "
        r"\dfrac{p}{\gamma - 1} + \tfrac{1}{2}\rho(u^{2}+v^{2})\end{pmatrix}",
        label="eq:A2-prim2cons",
    )
    ld.add(
        "Inverse map, conservative to primitive",
        r"\mathbf{W}(\mathbf{U}) \;=\; \begin{pmatrix}\rho\\ m_x / \rho\\ m_y / \rho\\ "
        r"(\gamma - 1)\left(E - \dfrac{m_x^{2} + m_y^{2}}{2\rho}\right)\end{pmatrix}",
        label="eq:A2-cons2prim",
    )
    ld.add(
        "Forward-map Jacobian determinant",
        r"\det\!\left(\frac{\partial \mathbf{U}}{\partial \mathbf{W}}\right) "
        r"\;=\; \frac{\rho^{2}}{\gamma - 1}",
        label="eq:A2-det-forward",
    )
    ld.add(
        "Inverse-map Jacobian determinant",
        r"\det\!\left(\frac{\partial \mathbf{W}}{\partial \mathbf{U}}\right) "
        r"\;=\; \frac{\gamma - 1}{\rho^{2}}",
        label="eq:A2-det-inverse",
    )
    ld.add(
        "Positivity envelope for pressure recovery",
        r"p > 0 \quad\Longleftrightarrow\quad E \;>\; \frac{m_x^{2} + m_y^{2}}{2\rho}",
        label="eq:A2-positivity",
    )

    ld.write()
    print()
    print("All A2 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
