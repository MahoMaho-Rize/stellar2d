r"""
Section A4 — Rotational covariance of the y-direction flux.

The 2D Euler flux is rotationally covariant: the y-flux F_y(U) can be
obtained from the x-flux F_x(U) by a coordinate swap (u <-> v,
m_x <-> m_y).  Formally,

  F_y(U) = R^{-1} F_x(R U)

where R is the component-permutation matrix

      | 1  0  0  0 |
  R = | 0  0  1  0 |       (swap m_x <-> m_y)
      | 0  1  0  0 |
      | 0  0  0  1 |

and R is its own inverse (R^2 = I).

Verifying this strong-form identity means the y-sweep kernel can
reuse the x-sweep Riemann solver verbatim by swapping the
argument order for momentum components, which is exactly what
k_hllc_update_y does:

    d_lmhllc(d_wR[k0*4+0], d_wR[k0*4+2], d_wR[k0*4+1], d_wR[k0*4+3],
             d_wL[kT*4+0], d_wL[kT*4+2], d_wL[kT*4+1], d_wL[kT*4+3],
             ... )

The components 1 and 2 are swapped at both the left and right
states when calling the shared HLLC device function — this is the
kernel realisation of the A4 identity.

Strong-form targets:

  1. R^2 = I (component swap is an involution).
  2. F_y(U) = R F_x(R U), pointwise, 4 components.
  3. Consequence for the Jacobian:  A_y(U) = R A_x(R U) R.
     Proof via chain rule.
  4. Eigenvalue mapping: if A_x has eigenvalues {u-c, u, u, u+c}
     with eigenvectors R_k, then A_y has eigenvalues
     {v-c, v, v, v+c} with eigenvectors R (R_k evaluated at R U).

Code anchors:
  src/gpu/explicit/strang_solver.cu :: k_hllc_update_y (argument
                                                       permutation)
  src/gpu/explicit/strang_device.cuh :: d_euler_flux_y

Rule 4 note: every identity is pointwise strong-form.
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
    flux_y_euler,
    gamma,
)


def main() -> int:
    ld = LatexDump(__file__)
    banner("A4 - Rotational covariance of the y-flux")

    # Primitive symbols.  u is the x-velocity, v is the y-velocity.
    rho = sp.Symbol("rho", positive=True)
    u = sp.Symbol("u", real=True)
    v = sp.Symbol("v", real=True)
    p = sp.Symbol("p", positive=True)

    # ════════════════════════════════════════════════════════════
    # 1.  The component permutation R.
    # ════════════════════════════════════════════════════════════
    R = sp.Matrix([
        [1, 0, 0, 0],
        [0, 0, 1, 0],   # swap m_x <-> m_y
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    # Involution:  R^2 = I.
    R_sq = sp.simplify(R @ R)
    for i in range(4):
        for j in range(4):
            expected = 1 if i == j else 0
            assert_zero(
                R_sq[i, j] - expected,
                f"A4-involution R^2 = I at ({i},{j})",
            )

    # ════════════════════════════════════════════════════════════
    # 2.  Strong-form pointwise identity  F_y(U) = R F_x(R U).
    # ════════════════════════════════════════════════════════════
    # Build U from primitive (rho, u, v, p) in conservative form.
    # Total energy E = p/(gamma-1) + rho (u^2+v^2)/2.
    from _common import cons_from_prim

    U = cons_from_prim(rho, u, v, p, gamma)

    # F_x(U) and F_y(U) directly from primitive form:
    Fx_U = flux_x_euler(rho, u, v, p, gamma)
    Fy_U = flux_y_euler(rho, u, v, p, gamma)

    # R U is the conservative state with m_x and m_y swapped:
    #   (rho, rho v, rho u, E).
    RU = R @ U
    # F_x(R U) is the x-flux evaluated at the permuted state;
    # under (u, v) -> (v, u) swap, this is flux_x_euler(rho, v, u, p, gamma).
    Fx_RU = flux_x_euler(rho, v, u, p, gamma)

    # Verify R F_x(R U) equals F_y(U).
    lhs = sp.simplify(R @ Fx_RU)
    for i in range(4):
        assert_zero(
            sp.simplify(lhs[i] - Fy_U[i]),
            f"A4-covariance F_y = R F_x(R U) at component {i}",
        )

    # ════════════════════════════════════════════════════════════
    # 3.  Jacobian consequence:  A_y(U) = R A_x(R U) R.
    #
    #     Proof (sympy).  By chain rule, partial F_y(U)/partial U
    #     equals  R (partial F_x/ partial (R U)) (partial R U / partial U)
    #           = R A_x(R U) R     (since partial(RU)/partial U = R
    #                               and R is constant).
    # ════════════════════════════════════════════════════════════
    # Build A_x and A_y from the U-component derivatives.
    mx_c, my_c, E_c = sp.symbols("m_xc m_yc E_c", real=True)
    rho_c = sp.Symbol("rho_c", positive=True)
    U_vec_c = sp.Matrix([rho_c, mx_c, my_c, E_c])

    # Primitive from U.
    u_of = mx_c / rho_c
    v_of = my_c / rho_c
    p_of = (gamma - 1) * (E_c - (mx_c**2 + my_c**2) / (2 * rho_c))
    Fx_of = sp.Matrix([
        mx_c,
        mx_c * u_of + p_of,
        mx_c * v_of,
        (E_c + p_of) * u_of,
    ])
    Fy_of = sp.Matrix([
        my_c,
        mx_c * v_of,
        my_c * v_of + p_of,
        (E_c + p_of) * v_of,
    ])

    Ax = Fx_of.jacobian(U_vec_c)
    Ay = Fy_of.jacobian(U_vec_c)

    # Evaluate A_x at the permuted state R U = (rho, my, mx, E).
    perm_subs = {mx_c: my_c, my_c: mx_c}
    # To avoid name collision during substitution, use temporary:
    tmp_m1, tmp_m2 = sp.symbols("tmp_m1 tmp_m2", real=True)
    Ax_at_RU = Ax.subs({mx_c: tmp_m1, my_c: tmp_m2}).subs(
        {tmp_m1: my_c, tmp_m2: mx_c}
    )

    rhs = sp.simplify(R @ Ax_at_RU @ R)
    diff_AyRHS = sp.simplify(Ay - rhs)
    for i in range(4):
        for j in range(4):
            assert_zero(
                diff_AyRHS[i, j],
                f"A4-Jacobian covariance A_y = R A_x(R U) R at ({i},{j})",
            )

    # ════════════════════════════════════════════════════════════
    # 4.  Eigenvalue mapping.  If A_x has eigenvalue lambda(U) with
    #     right eigenvector R_k(U), then A_y has eigenvalue
    #     lambda(R U) with right eigenvector R R_k(R U).
    #
    #     Equivalently, the eigenvalues of A_y(U) are {v-c, v, v, v+c}
    #     with c = sqrt(gamma p / rho), swapping u->v relative to A_x.
    #
    #     Strong-form verification: det(A_y(U) - lam I) factored.
    # ════════════════════════════════════════════════════════════
    lam = sp.Symbol("lam", real=True)
    # Use primitive-form substitution for cleanliness.
    subs_prim = {
        rho_c: rho,
        mx_c: rho * u,
        my_c: rho * v,
        E_c: p / (gamma - 1) + sp.Rational(1, 2) * rho * (u**2 + v**2),
    }
    Ay_prim = sp.simplify(Ay.subs(subs_prim))
    char_y = sp.simplify((Ay_prim - lam * sp.eye(4)).det())
    expected_y = (v - lam)**2 * ((v - lam)**2 - gamma * p / rho)
    assert_zero(
        sp.expand(char_y - expected_y),
        "A4-char-poly-y: det(A_y - lam I) = (v-lam)^2 ((v-lam)^2 - c^2)",
    )

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Component-permutation matrix R (momentum swap)",
        r"R \;=\; \begin{pmatrix}1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1\end{pmatrix}, "
        r"\qquad R^{2} = I",
        label="eq:A4-R-def",
    )
    ld.add(
        "Rotational covariance of the flux",
        r"\mathbf{F}_y(\mathbf{U}) \;=\; R\,\mathbf{F}_x\!\bigl(R\,\mathbf{U}\bigr)",
        label="eq:A4-flux-covariance",
    )
    ld.add(
        "Consequence for the flux Jacobian",
        r"A_y(\mathbf{U}) \;=\; R\,A_x\!\bigl(R\,\mathbf{U}\bigr)\,R",
        label="eq:A4-Jacobian-covariance",
    )
    ld.add(
        "Eigenvalues of A_y",
        r"\{\lambda_k^{(y)}\}_{k=0}^{3} \;=\; \{\,v - c,\ v,\ v,\ v + c\,\}, "
        r"\qquad c = \sqrt{\gamma p / \rho}",
        label="eq:A4-eigvals-y",
    )

    ld.write()
    print()
    print("All A4 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
