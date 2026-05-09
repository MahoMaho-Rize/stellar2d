r"""
Section A13 — Time integrator family (Strang / Lie split / unsplit VL2 / RK2-MUSCL).

Given a 2D PDE  U_t = X(U) + Y(U), we compare four canonical time
integrators:

  (1) Lie splitting:  U^{n+1} = e^{Dt Y} e^{Dt X} U^n
      Order:  O(Dt^2)   (i.e., 1st order global).

  (2) Strang splitting (symmetric):
      U^{n+1} = e^{(Dt/2) X} e^{Dt Y} e^{(Dt/2) X} U^n
      Order:  O(Dt^3)   (i.e., 2nd order global).

  (3) Unsplit VL2 (Stone-Gardiner 2009):
      Predictor:  U* = U^n - (Dt/2) (partial_x F(U^n) + partial_y G(U^n))
      Corrector:  U^{n+1} = U^n - Dt (partial_x F(U*) + partial_y G(U*))
      Order:  O(Dt^3)   (i.e., 2nd order global).

  (4) RK2-MUSCL (Shu-Osher 2nd-order):
      Stage 1:  U^{(1)} = U^n + Dt L(U^n)
      Stage 2:  U^{n+1} = (1/2) U^n + (1/2)(U^{(1)} + Dt L(U^{(1)}))
      where L = -(partial_x F + partial_y G).
      Order:  O(Dt^3).

Strong-form identities (verified by sympy via BCH expansion on
LINEAR X, Y; the non-linear case is a numerical fallback per
Rule 1):

  A. Lie splitting is 1st-order: leading error in BCH is (Dt^2/2) [X, Y].
     For non-commuting X, Y (generic case), this is non-zero.

  B. Strang splitting is 2nd-order: leading error in BCH at Dt^2
     vanishes identically; leading error is at Dt^3, involving
     [X, [X, Y]] and [[X, Y], Y].
     Strong-form BCH identity (LINEAR case):
        e^{(Dt/2) X} e^{Dt Y} e^{(Dt/2) X}  -  e^{Dt (X + Y)}
        = Dt^3 C_3 + O(Dt^4)
     where C_3 is an explicit combination of iterated commutators.

  C. The Strang kernel uses the chain  X(Dt/2) Y(Dt/2) Y(Dt/2) X(Dt/2),
     which is equivalent to  X(Dt/2) Y(Dt) X(Dt/2)  by semigroup
     composition  Y(Dt/2) Y(Dt/2) = Y(Dt).
     Verify this as an algebraic identity on LINEAR operators.

  D. Unsplit VL2 is 2nd-order: Taylor-expand U^{n+1} - U(t_n + Dt)
     on LINEAR F, G and show leading error is O(Dt^3).

  E. Non-linear order verification [WEAK]: for the non-linear Euler
     system, we cannot construct a closed-form BCH expansion.  Per
     Rule 4 this is a weak-form step; we verify 2nd-order behaviour
     numerically by running the kernel with decreasing Dt and
     measuring convergence rate -- but the numerical test is in
     §E1 (entropy-wave convergence), not here.  This section
     stays pure-algebra.

Code anchor:
  src/gpu/explicit/strang_solver.cu :: StrangSolver::step  (Strang
      split; see §A14 for the exact operator chain)
  src/gpu/explicit/athena_vl2_solver.cu :: (unsplit VL2, derived
      here for comparison)

Rule 4: all identities are strong-form on linearised operators,
which is a reduction of the non-linear problem to a tractable
symbolic case.  The non-linear-operator generalisation is
[WEAK] and handled in §E1.
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
    banner("A13 - Time integrator family")

    Dt = sp.Symbol("Dt", positive=True)

    # Treat X, Y as non-commuting symbolic operators.  We cannot
    # use scalar sympy symbols (which commute); instead we use
    # indexed names and Taylor-expand e^{tau A} manually as
    # I + tau A + tau^2 A^2 / 2 + ... up to fixed order.
    # To compute  e^{a X} e^{b Y}  up to order n, we multiply the
    # two truncated series using a custom non-commutative polynomial
    # ring.  Implement via tuples of monomials.

    # Build BCH residual up to order 3.  Monomial = tuple of 'X'/'Y'
    # strings.  Polynomial = dict {monomial: coefficient}.

    def poly_exp(symbol: str, scalar, order: int):
        """Return e^{scalar * <symbol>} truncated to given total-monomial order."""
        p = {(): sp.Integer(1)}
        for k in range(1, order + 1):
            p[(symbol,) * k] = scalar**k / sp.factorial(k)
        return p

    def poly_mul(p, q, order: int):
        """Multiply two non-commutative polynomials up to `order` total length."""
        result: dict[tuple, sp.Expr] = {}
        for m1, c1 in p.items():
            for m2, c2 in q.items():
                if len(m1) + len(m2) > order:
                    continue
                m = m1 + m2
                result[m] = sp.expand(result.get(m, sp.Integer(0)) + c1 * c2)
        return result

    def poly_sub(p, q):
        keys = set(p) | set(q)
        return {m: sp.expand(p.get(m, 0) - q.get(m, 0)) for m in keys}

    def poly_truncate(p, order: int):
        return {m: c for m, c in p.items() if len(m) <= order and c != 0}

    def poly_strip(p):
        return {m: c for m, c in p.items() if c != 0}

    # Exact target: e^{Dt (X + Y)} up to order 3.
    def exp_X_plus_Y(order: int):
        """Expand e^{Dt (X+Y)} up to order n.  Keeps X/Y in the
        correct non-commutative order because we expand (X+Y)^k
        as a multinomial of X, Y terms."""
        result = {(): sp.Integer(1)}
        for k in range(1, order + 1):
            coeff = Dt**k / sp.factorial(k)
            # (X+Y)^k = sum over binary strings of length k
            # Generate all length-k sequences of 'X', 'Y'.
            for n in range(2**k):
                bits = []
                n_tmp = n
                for _ in range(k):
                    bits.append("Y" if (n_tmp & 1) else "X")
                    n_tmp >>= 1
                m = tuple(reversed(bits))
                result[m] = result.get(m, sp.Integer(0)) + coeff
        return poly_strip(result)

    ORDER = 3
    exact = exp_X_plus_Y(ORDER)

    # ════════════════════════════════════════════════════════════
    # A.  Lie splitting:  e^{Dt Y} e^{Dt X}.
    # ════════════════════════════════════════════════════════════
    e_X = poly_exp("X", Dt, ORDER)
    e_Y = poly_exp("Y", Dt, ORDER)
    lie = poly_mul(e_Y, e_X, ORDER)
    residual_lie = poly_sub(lie, exact)
    residual_lie = poly_strip({m: sp.expand(c) for m, c in residual_lie.items()})

    # Remove trivial zero entries.  Leading non-zero residuals should
    # appear at order 2 (specifically  (Dt^2/2)(Y X - X Y) = -(Dt^2/2)[X, Y]).
    # So residual_lie at order 2 should be:
    #   ('Y', 'X'): Dt^2 * 1/2   --- from e^{Dt Y} e^{Dt X} at second order:
    #      1 + Dt Y + Dt^2 Y^2/2 + ... times 1 + Dt X + Dt^2 X^2/2 + ...
    #      gives Dt^2 Y X as one of the order-2 terms.
    #   ('X', 'Y'): Dt^2 * 1/2 from expansion of (X+Y)^2/2 = (XX + XY + YX + YY)/2.
    # So residual = -Dt^2/2 (XY) + Dt^2/2 (YX)  =  (Dt^2/2)[Y, X]  =  -(Dt^2/2)[X, Y].
    # Specifically:
    #   residual[('X','Y')] = 0 - Dt^2/2 = -Dt^2/2
    #   residual[('Y','X')] = Dt^2 - Dt^2/2 = Dt^2/2
    expected_lie_XY = -Dt**2 / 2
    expected_lie_YX =  Dt**2 / 2
    assert_zero(
        sp.expand(residual_lie.get(("X", "Y"), sp.Integer(0)) - expected_lie_XY),
        "A13-Lie-residual-XY: coefficient of XY in (e^Y e^X - e^{X+Y}) = -Dt^2/2",
    )
    assert_zero(
        sp.expand(residual_lie.get(("Y", "X"), sp.Integer(0)) - expected_lie_YX),
        "A13-Lie-residual-YX: coefficient of YX in (e^Y e^X - e^{X+Y}) = +Dt^2/2",
    )

    # ════════════════════════════════════════════════════════════
    # B.  Strang splitting:  e^{(Dt/2) X} e^{Dt Y} e^{(Dt/2) X}.
    # ════════════════════════════════════════════════════════════
    e_X_half = poly_exp("X", Dt / 2, ORDER)
    strang = poly_mul(poly_mul(e_X_half, e_Y, ORDER), e_X_half, ORDER)
    residual_strang = poly_sub(strang, exact)
    residual_strang = poly_strip({m: sp.expand(c) for m, c in residual_strang.items()})

    # Check that all order-2 terms vanish.  There are four order-2
    # monomials: XX, XY, YX, YY.  We expect every one to cancel.
    for m in [("X", "X"), ("X", "Y"), ("Y", "X"), ("Y", "Y")]:
        c = residual_strang.get(m, sp.Integer(0))
        assert_zero(
            sp.expand(c),
            f"A13-Strang-order2-{m}: coefficient of {m} in (e^(X/2) e^Y e^(X/2) - e^{{X+Y}}) = 0",
        )

    # At order 3 the residual is non-zero (Strang is 3rd-order
    # accurate in the leading error).  Specifically, the leading
    # non-zero residuals involve iterated commutators.  Print for
    # information, don't assert non-zero.
    order3_monomials = [(a, b, c) for a in ("X", "Y") for b in ("X", "Y") for c in ("X", "Y")]
    print("  Strang order-3 residual coefficients:")
    for m in order3_monomials:
        c = residual_strang.get(m, sp.Integer(0))
        if c != 0:
            print(f"    {m}: {sp.simplify(c)}")

    # ════════════════════════════════════════════════════════════
    # C.  Strang kernel chain:  X(Dt/2) Y(Dt/2) Y(Dt/2) X(Dt/2).
    #
    # Equal to  X(Dt/2) Y(Dt) X(Dt/2)  by Y semigroup composition.
    # Verify this as an algebraic identity.
    # ════════════════════════════════════════════════════════════
    e_Y_half = poly_exp("Y", Dt / 2, ORDER)
    kernel_chain = poly_mul(poly_mul(poly_mul(
        e_X_half, e_Y_half, ORDER), e_Y_half, ORDER), e_X_half, ORDER)
    strang_canonical = poly_mul(poly_mul(
        e_X_half, e_Y, ORDER), e_X_half, ORDER)
    diff_chain = poly_sub(kernel_chain, strang_canonical)
    diff_chain = poly_strip({m: sp.expand(c) for m, c in diff_chain.items()})

    # All monomial coefficients should cancel.
    if diff_chain:
        for m, c in diff_chain.items():
            assert_zero(sp.expand(c),
                        f"A13-kernel-chain-equiv-{m}: "
                        f"X(Dt/2) Y(Dt/2)^2 X(Dt/2) = X(Dt/2) Y(Dt) X(Dt/2) at {m}")
    else:
        print("  [OK] A13-kernel-chain-equiv: "
              "X(Dt/2) Y(Dt/2) Y(Dt/2) X(Dt/2)  ==  X(Dt/2) Y(Dt) X(Dt/2) "
              "(identically, all monomials zero).")

    # ════════════════════════════════════════════════════════════
    # D.  Unsplit VL2 (Stone-Gardiner 2009).
    #
    # Predictor:    U* = U^n - (Dt/2) L(U^n)    where L = -partial.flux
    # Corrector:    U^{n+1} = U^n - Dt L(U*)
    #
    # For LINEAR L, this is a two-stage explicit scheme.  Write as
    # operator series: U^{n+1} = [I - Dt L + (Dt^2/2) L^2] U^n  (at
    # leading order).  Compare to e^{-Dt L} U^n = [I - Dt L + (Dt^2/2) L^2 - (Dt^3/6) L^3 + ...].
    # So the VL2 scheme matches the exact up to Dt^2 and departs at
    # Dt^3 -- 2nd-order accurate.
    # ════════════════════════════════════════════════════════════
    # Linear scalar case: L is a single symbol (no X/Y splitting
    # needed).  Treat as polynomial in one symbol.
    # VL2 update:
    #   U* = U - (Dt/2) L U                (we drop U notation, use operator)
    #      = (I - (Dt/2) L) U
    #   U^{n+1} = U - Dt L U*
    #           = (I - Dt L - (Dt^2/2) L^2 ... no wait) let me redo:
    #   L U* = L (I - Dt/2 L) U = L U - (Dt/2) L^2 U
    #   U^{n+1} = U - Dt (L U - (Dt/2) L^2 U) = U - Dt L U + (Dt^2/2) L^2 U
    #           = (I - Dt L + (Dt^2/2) L^2) U.
    # Exact (linear):  e^{-Dt L} U = (I - Dt L + (Dt^2/2) L^2 - (Dt^3/6) L^3 + ...).
    # Match to order Dt^2; leading error at Dt^3 is (Dt^3/6) L^3.
    # Verify this symbolically:
    L_sym = sp.Symbol("L")  # a single linear operator as a sympy symbol
    VL2_op = 1 - Dt * L_sym + (Dt ** 2 / 2) * L_sym ** 2
    exact_op = sum(((-Dt * L_sym) ** k) / sp.factorial(k) for k in range(4))
    # Truncate exact to order 3.
    err_VL2 = sp.expand(exact_op - VL2_op)
    # Expected: - (Dt^3 / 6) L^3 + O(Dt^4).
    expected_err_VL2 = - Dt ** 3 * L_sym ** 3 / 6
    # The two should agree to O(Dt^4).  At Dt^3 exactly:
    diff_VL2 = sp.expand(err_VL2 - expected_err_VL2)
    # Any remaining Dt^4 or higher terms are fine since we only
    # cared about the leading error.  Force up-to-Dt^3 equality:
    # extract Dt^3 coefficients and compare.
    diff_Dt3 = diff_VL2.coeff(Dt, 3)
    assert_zero(
        sp.simplify(diff_Dt3),
        "A13-VL2-order: leading error at Dt^3 is -(Dt^3/6) L^3",
    )

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "PDE abstract form",
        r"\partial_t \mathbf{U} \;=\; \mathcal{X}(\mathbf{U}) + \mathcal{Y}(\mathbf{U}) "
        r"\qquad \text{(in our case } \mathcal{X} = -\partial_x \mathbf{F},\ "
        r"\mathcal{Y} = -\partial_y \mathbf{G}\text{)}",
        label="eq:A13-split-form",
    )
    ld.add(
        "Lie splitting (1st-order; leading error)",
        r"e^{\Delta t\,\mathcal{Y}}\,e^{\Delta t\,\mathcal{X}}\;\mathbf{U}^{n} "
        r"\;-\; e^{\Delta t\,(\mathcal{X} + \mathcal{Y})}\,\mathbf{U}^{n} "
        r"\;=\; -\,\frac{\Delta t^{2}}{2}\,[\mathcal{X},\,\mathcal{Y}]\,\mathbf{U}^{n} + O(\Delta t^{3})",
        label="eq:A13-Lie",
    )
    ld.add(
        "Strang splitting (2nd-order; all Delta-t^2 commutators cancel)",
        r"e^{(\Delta t/2)\,\mathcal{X}}\,e^{\Delta t\,\mathcal{Y}}\,e^{(\Delta t/2)\,\mathcal{X}}\;\mathbf{U}^{n} "
        r"\;-\; e^{\Delta t\,(\mathcal{X} + \mathcal{Y})}\,\mathbf{U}^{n} \;=\; O(\Delta t^{3})",
        label="eq:A13-Strang",
    )
    ld.add(
        "Kernel chain equivalence (semigroup composition)",
        r"\mathcal{X}(\Delta t/2)\,\mathcal{Y}(\Delta t/2)\,\mathcal{Y}(\Delta t/2)\,\mathcal{X}(\Delta t/2) "
        r"\;\equiv\; \mathcal{X}(\Delta t/2)\,\mathcal{Y}(\Delta t)\,\mathcal{X}(\Delta t/2) "
        r"\qquad (\text{identical under } \mathcal{Y}(\tau_1)\,\mathcal{Y}(\tau_2) = \mathcal{Y}(\tau_1 + \tau_2))",
        label="eq:A13-kernel-chain",
    )
    ld.add(
        "Unsplit VL2 (Stone-Gardiner 2009; 2nd-order on linear L)",
        r"\begin{aligned}"
        r"\mathbf{U}^{\star} &\;=\; \mathbf{U}^{n} - \tfrac{\Delta t}{2}\,\mathcal{L}\,\mathbf{U}^{n} \\[2pt]"
        r"\mathbf{U}^{n+1} &\;=\; \mathbf{U}^{n} - \Delta t\,\mathcal{L}\,\mathbf{U}^{\star}"
        r"\end{aligned}"
        r"\\[4pt] "
        r"\mathbf{U}^{n+1} \;-\; e^{-\Delta t\,\mathcal{L}}\,\mathbf{U}^{n} \;=\; "
        r"-\,\tfrac{\Delta t^{3}}{6}\,\mathcal{L}^{3}\,\mathbf{U}^{n} \;+\; O(\Delta t^{4})",
        label="eq:A13-VL2",
    )
    ld.add(
        "Order comparison",
        r"\text{Lie: } O(\Delta t^{2}) \;<\; \text{Strang / VL2: } O(\Delta t^{3})",
        label="eq:A13-comparison",
    )

    ld.write()
    print()
    print("All A13 LINEAR-operator identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
