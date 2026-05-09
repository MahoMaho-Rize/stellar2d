r"""
Section A3 — x-direction flux Jacobian eigensystem.

Strong-form identities (verified by sympy):

  1. Compute  A_x = partial F_x / partial U  directly.
     4x4 matrix in terms of (rho, u, v, p, gamma).  No shortcut
     via characteristic variables.

  2. Eigenvalues:  {u - c, u, u, u + c},  c = sqrt(gamma p / rho).
     Verified by  det(A_x - lambda I) == 0  at each of the four
     candidate eigenvalues.

  3. Right eigenvectors R_k (k = 0,1,2,3) in primitive-projected form,
     then mapped back to conservative-variable components.

     The canonical Toro Chap. 3 convention:
       R_0  (lambda = u - c)    acoustic left-going
       R_1  (lambda = u)        entropy wave
       R_2  (lambda = u)        shear wave (tangential velocity)
       R_3  (lambda = u + c)    acoustic right-going
     Two-fold degeneracy at lambda = u is explicit.

  4. Left eigenvectors L_k: rows of R^{-1}.
     Orthogonality  L_k . R_l == delta_{kl}  verified pointwise.

  5. Characteristic decomposition closure:
       R diag(lambda) L == A_x
     verified entry-by-entry.

  6. For each eigenvector, A_x R_k == lambda_k R_k.

Code anchors:
  src/gpu/explicit/strang_device.cuh :: d_euler_flux_x  (only F_x used;
                                                         Jacobian is
                                                         not explicitly
                                                         assembled in
                                                         the kernel,
                                                         but the HLLC
                                                         solver uses
                                                         sound-speed
                                                         and S_star
                                                         that derive
                                                         from this
                                                         eigensystem)

Rule 4 note: every identity is pointwise strong-form; no weak-form
fallback in this section.
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
    banner("A3 - x-direction flux Jacobian eigensystem")

    # Symbols.
    rho = sp.Symbol("rho", positive=True)
    u = sp.Symbol("u", real=True)
    v = sp.Symbol("v", real=True)
    p = sp.Symbol("p", positive=True)

    # Sound speed c = sqrt(gamma p / rho).  Keep symbolic for clarity;
    # c itself is an independent symbol with the defining relation
    # c^2 = gamma p / rho.  We will substitute at verification time.
    c_sym = sp.Symbol("c", positive=True)
    c_expr = sp.sqrt(gamma * p / rho)

    # Conservative state U = (rho, m_x, m_y, E)  with  m_x = rho u,
    # m_y = rho v,  E = p/(gamma-1) + rho(u^2 + v^2)/2.
    mx, my, E = sp.symbols("m_x m_y E", real=True)

    # Flux F_x in terms of U.  Substitute primitives from U:
    #   rho = U[0], u = mx/rho, v = my/rho,
    #   p = (gamma - 1) (E - (mx^2 + my^2)/(2 rho)).
    U = sp.Matrix([rho, rho * u, rho * v,
                   p / (gamma - 1) + sp.Rational(1, 2) * rho * (u**2 + v**2)])

    # The flux is cleanest in primitive form; we then take
    # d F_x / d U  via chain rule by diff-ing F_x expressed as
    # a function of U's components.  Strategy: invert the U
    # definitions and substitute.
    rho_from_U = U[0]
    mx_from_U = U[1]
    my_from_U = U[2]
    E_from_U = U[3]

    # Express primitives (rho, u, v, p) symbolically via the U
    # components  (rho_c, mx_c, my_c, E_c):
    rho_c, mx_c, my_c, E_c = sp.symbols("rho_c m_xc m_yc E_c", real=True)
    u_of_U = mx_c / rho_c
    v_of_U = my_c / rho_c
    p_of_U = (gamma - 1) * (E_c - (mx_c**2 + my_c**2) / (2 * rho_c))

    # Flux F_x written purely in U-components:
    Fx_of_U = sp.Matrix([
        mx_c,                                                # rho u
        mx_c * u_of_U + p_of_U,                              # rho u^2 + p
        mx_c * v_of_U,                                       # rho u v
        (E_c + p_of_U) * u_of_U,                             # (E+p) u
    ])

    U_vec_c = sp.Matrix([rho_c, mx_c, my_c, E_c])
    A_x = Fx_of_U.jacobian(U_vec_c)
    A_x = sp.simplify(A_x)

    # ════════════════════════════════════════════════════════════
    # Eigenvalues.  Substitute the primitive form  u = mx_c/rho_c,
    # v = my_c/rho_c,  p = ... and evaluate at (rho, u, v, p).
    # ════════════════════════════════════════════════════════════
    # Rewrite A_x into (rho, u, v, p) for easier algebra.
    subs_prim = {
        rho_c: rho,
        mx_c: rho * u,
        my_c: rho * v,
        E_c: p / (gamma - 1) + sp.Rational(1, 2) * rho * (u**2 + v**2),
    }
    A_x_prim = sp.simplify(A_x.subs(subs_prim))

    # ════════════════════════════════════════════════════════════
    # Verify eigenvalues by characteristic polynomial.  The
    # eigenvalues of the x-flux Jacobian on a gamma-law gas are
    # {u-c, u, u, u+c} with c^2 = gamma p / rho.  Substitute
    # c^2 -> gamma*p/rho at the end.
    # ════════════════════════════════════════════════════════════
    I4 = sp.eye(4)
    char_poly_lambda = sp.Symbol("lam", real=True)
    char_det = sp.simplify((A_x_prim - char_poly_lambda * I4).det())
    # Expected:  (u - lam)^2 * ((u - lam)^2 - c^2)   with c^2 = gamma p / rho.
    expected = (
        (u - char_poly_lambda)**2
        * ((u - char_poly_lambda)**2 - gamma * p / rho)
    )
    assert_zero(sp.expand(char_det - expected),
                "A3-char-poly: det(A_x - lam I) = (u-lam)^2 ((u-lam)^2 - c^2)")

    # ════════════════════════════════════════════════════════════
    # Right eigenvectors R_k in conservative form.
    # Derived via standard Roe-style projection; expressions below
    # are from Toro §3.1.2 (gamma-law ideal gas).
    # ════════════════════════════════════════════════════════════
    h = gamma * p / ((gamma - 1) * rho) + sp.Rational(1, 2) * (u**2 + v**2)
    #   specific total enthalpy

    # Eigenvectors, as columns:
    #  R[:,0]: left-going acoustic  (lambda = u - c)
    #  R[:,1]: entropy              (lambda = u)
    #  R[:,2]: shear (tangential v) (lambda = u)
    #  R[:,3]: right-going acoustic (lambda = u + c)
    R_cols = [
        sp.Matrix([1, u - c_sym, v, h - u * c_sym]),          # left acoustic
        sp.Matrix([1, u, v, sp.Rational(1, 2) * (u**2 + v**2)]),  # entropy
        sp.Matrix([0, 0, 1, v]),                              # shear
        sp.Matrix([1, u + c_sym, v, h + u * c_sym]),          # right acoustic
    ]
    R = sp.Matrix.hstack(*R_cols)

    lambdas = [u - c_sym, u, u, u + c_sym]

    # Verify  A_x R_k == lambda_k R_k  after substituting c^2 = gamma p / rho.
    for k, (lam, Rk) in enumerate(zip(lambdas, R_cols)):
        # Substitute the c^2 -> gamma p / rho relation.
        residual = sp.expand(
            A_x_prim @ Rk - lam * Rk
        )
        residual = residual.subs(c_sym**2, gamma * p / rho)
        residual = sp.simplify(residual.subs(c_sym, c_expr))
        for i in range(4):
            assert_zero(
                residual[i],
                f"A3-eigvec A_x R_{k} = lam_{k} R_{k} component {i}",
            )

    # ════════════════════════════════════════════════════════════
    # Left eigenvectors:  L = R^{-1}.
    # sympy will happily invert a 4x4 symbolic matrix here; we then
    # check  L A_x == diag(lambda) L  and  L R == I.
    # ════════════════════════════════════════════════════════════
    L = sp.simplify(R.inv())
    LR = sp.simplify(L @ R)
    for i in range(4):
        for j in range(4):
            expected_val = 1 if i == j else 0
            assert_zero(
                sp.simplify(LR[i, j] - expected_val),
                f"A3-LR orthogonality L R = I at ({i},{j})",
            )

    # Characteristic decomposition:  R diag(lam) L == A_x.
    # Substitute c -> c_expr and simplify.
    LambdaDiag = sp.diag(*lambdas)
    recon = sp.simplify(R @ LambdaDiag @ L)
    recon = recon.subs(c_sym, c_expr)
    recon = sp.simplify(recon)
    for i in range(4):
        for j in range(4):
            diff = sp.simplify(recon[i, j] - A_x_prim[i, j])
            # Try trigsimp/radsimp if simplify leaves sqrts unresolved.
            if diff != 0:
                diff = sp.simplify(sp.radsimp(diff))
            assert_zero(
                diff,
                f"A3-char-decomp R Lam L = A_x at ({i},{j})",
            )

    # ════════════════════════════════════════════════════════════
    # LaTeX dump — the eigenvectors are the headline result.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Flux Jacobian in the x-direction",
        r"A_x(\mathbf{U}) \;\equiv\; \frac{\partial \mathbf{F}_x}{\partial \mathbf{U}}",
        label="eq:A3-Ax-def",
    )
    ld.add(
        "Characteristic polynomial",
        r"\det(A_x - \lambda I) \;=\; (u - \lambda)^{2}\bigl[(u - \lambda)^{2} - c^{2}\bigr], "
        r"\qquad c = \sqrt{\gamma p / \rho}",
        label="eq:A3-charpoly",
    )
    ld.add(
        "Eigenvalues",
        r"\{\lambda_k\}_{k=0}^{3} \;=\; \{\,u - c,\ u,\ u,\ u + c\,\}",
        label="eq:A3-eigvals",
    )
    ld.add(
        "Right eigenvectors (columns of R)",
        r"R_0 \;=\; \begin{pmatrix}1\\ u - c\\ v\\ h - u\,c\end{pmatrix},\quad "
        r"R_1 \;=\; \begin{pmatrix}1\\ u\\ v\\ \tfrac{1}{2}(u^{2}+v^{2})\end{pmatrix},\quad "
        r"R_2 \;=\; \begin{pmatrix}0\\ 0\\ 1\\ v\end{pmatrix},\quad "
        r"R_3 \;=\; \begin{pmatrix}1\\ u + c\\ v\\ h + u\,c\end{pmatrix}",
        label="eq:A3-right-eigvecs",
    )
    ld.add(
        "Specific total enthalpy (appears in acoustic eigenvectors)",
        r"h \;=\; \frac{\gamma\,p}{(\gamma - 1)\,\rho} + \tfrac{1}{2}(u^{2} + v^{2})",
        label="eq:A3-enthalpy",
    )
    ld.add(
        "Characteristic decomposition",
        r"A_x \;=\; R\,\mathrm{diag}(u - c,\ u,\ u,\ u + c)\,L, \qquad L = R^{-1}",
        label="eq:A3-diagonalisation",
    )
    ld.add(
        "Two-fold degeneracy at lambda = u",
        r"R_1\ \text{and}\ R_2\ \text{both satisfy}\ A_x R_k = u\,R_k; "
        r"\ R_1\ \text{carries density contrast (entropy), }\ R_2\ \text{carries tangential velocity (shear)}",
        label="eq:A3-degeneracy",
    )

    ld.write()
    print()
    print("All A3 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
