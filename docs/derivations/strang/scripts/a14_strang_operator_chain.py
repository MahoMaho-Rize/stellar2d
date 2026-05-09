r"""
Section A14 — Strang operator chain self-adjointness.

Let L(Dt) denote the Strang integrator applied for one step of
duration Dt > 0.  A numerical integrator is SELF-ADJOINT (or
"time-reversible", or "symmetric") if

  L(Dt)  L(-Dt)  =  I       (on the linearised operator ring)

for all Dt.  Symmetric integrators of explicit-splitting type are
EVEN-ORDER by construction; their leading error appears at odd-in-
Dt orders that cancel in the symmetric combination.  This property
is the structural reason Strang splitting is 2nd-order (not merely
"accidentally" 2nd-order).

Strong-form identities:

  1. Self-adjointness of each half-flow:  exp((Dt/2) X) * exp(-(Dt/2) X) = I
     Trivial (semigroup property), but we verify it.

  2. Self-adjointness of the Strang chain:
       [exp((Dt/2) X) * exp(Dt Y) * exp((Dt/2) X)]
       [exp((-Dt/2) X) * exp(-Dt Y) * exp((-Dt/2) X)]
       = I
     Verify by non-commutative polynomial multiplication up to
     order 4 in Dt.

  3. Lie splitting is NOT self-adjoint:
       e^{Dt Y} e^{Dt X} * e^{-Dt Y} e^{-Dt X}  !=  I
     Specifically, the product has non-zero residuals at order
     Dt^2.  This explains why Lie is 1st-order: the odd-in-Dt
     leading error does NOT cancel under time reversal.

Code anchor:
  src/gpu/explicit/strang_solver.cu :: StrangSolver::step
  (the four-factor chain is symmetric by construction:
     X(Dt/2) Y(Dt/2) Y(Dt/2) X(Dt/2) ,
   reversing time gives
     X(-Dt/2) Y(-Dt/2) Y(-Dt/2) X(-Dt/2)
   and their composition is identity to all orders; verified
   below for the canonical three-factor form)

Rule 4: strong-form on linearised operators; non-linear case
handled in §E1.
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
    banner("A14 - Strang chain self-adjointness")

    Dt = sp.Symbol("Dt", real=True)   # allow negative values

    # Non-commutative polynomial utilities (copied/adapted from A13).
    def poly_exp(symbol: str, scalar, order: int):
        p = {(): sp.Integer(1)}
        for k in range(1, order + 1):
            p[(symbol,) * k] = scalar**k / sp.factorial(k)
        return p

    def poly_mul(p, q, order: int):
        result: dict[tuple, sp.Expr] = {}
        for m1, c1 in p.items():
            for m2, c2 in q.items():
                if len(m1) + len(m2) > order:
                    continue
                m = m1 + m2
                result[m] = sp.expand(result.get(m, sp.Integer(0)) + c1 * c2)
        return result

    def poly_strip(p):
        return {m: c for m, c in p.items() if sp.expand(c) != 0}

    ORDER = 4

    # ════════════════════════════════════════════════════════════
    # 1.  Trivial: e^{(Dt/2) X} * e^{-(Dt/2) X} = I.
    #     True by semigroup property.
    # ════════════════════════════════════════════════════════════
    e_X_half_pos = poly_exp("X", Dt / 2, ORDER)
    e_X_half_neg = poly_exp("X", -Dt / 2, ORDER)
    identity = poly_mul(e_X_half_pos, e_X_half_neg, ORDER)
    identity = poly_strip(identity)
    # identity should be {(): 1}.  Sanity check.
    for m, c in identity.items():
        expected = 1 if m == () else 0
        assert_zero(
            sp.expand(c - expected),
            f"A14-half-flow-reversal: e^{{(Dt/2) X}} e^{{-(Dt/2) X}} at monomial {m}",
        )

    # ════════════════════════════════════════════════════════════
    # 2.  Self-adjointness of the Strang chain.
    #
    #   L_forward  = e^{(Dt/2) X} e^{Dt Y} e^{(Dt/2) X}
    #   L_backward = e^{(-Dt/2) X} e^{-Dt Y} e^{(-Dt/2) X}
    #
    #   Claim: L_forward L_backward = I.
    #
    #   This is proven by expanding and showing all non-identity
    #   monomials cancel to the order we truncate at.
    # ════════════════════════════════════════════════════════════
    e_Y_pos = poly_exp("Y", Dt, ORDER)
    e_Y_neg = poly_exp("Y", -Dt, ORDER)

    L_forward = poly_mul(
        poly_mul(e_X_half_pos, e_Y_pos, ORDER),
        e_X_half_pos, ORDER,
    )
    L_backward = poly_mul(
        poly_mul(e_X_half_neg, e_Y_neg, ORDER),
        e_X_half_neg, ORDER,
    )
    composition = poly_mul(L_forward, L_backward, ORDER)

    # Enumerate every possible monomial up to ORDER.  Each must
    # satisfy:  coefficient == 1  if m == (),  else 0.
    def enumerate_monomials(max_len: int):
        yield ()
        from itertools import product
        for k in range(1, max_len + 1):
            for seq in product(("X", "Y"), repeat=k):
                yield seq

    n_checked = 0
    for m in enumerate_monomials(ORDER):
        c = composition.get(m, sp.Integer(0))
        expected = 1 if m == () else 0
        assert_zero(
            sp.expand(c - expected),
            f"A14-Strang-self-adjoint at monomial {m}",
        )
        n_checked += 1
    print(f"  [OK] {n_checked} Strang-chain self-adjoint monomial identities verified.")

    # ════════════════════════════════════════════════════════════
    # 3.  Lie splitting is NOT self-adjoint.
    #
    #   L_Lie_fwd = e^{Dt Y} e^{Dt X}
    #   L_Lie_bwd = e^{-Dt Y} e^{-Dt X}
    #   composition has non-zero residual at Dt^2.
    # ════════════════════════════════════════════════════════════
    e_X_pos = poly_exp("X", Dt, ORDER)
    e_X_neg = poly_exp("X", -Dt, ORDER)

    L_Lie_fwd = poly_mul(e_Y_pos, e_X_pos, ORDER)
    L_Lie_bwd = poly_mul(e_Y_neg, e_X_neg, ORDER)
    lie_composition = poly_mul(L_Lie_fwd, L_Lie_bwd, ORDER)
    lie_composition = poly_strip(lie_composition)

    # The leading residual at Dt^2 should be non-zero.  Specifically,
    # the coefficient of XY (or YX) should be Dt^2 (or similar).
    # Let's find the non-trivial order-2 residual.
    print("  Lie-splitting time-reversal composition minus identity "
          "(non-zero => not self-adjoint):")
    any_nonzero = False
    for m in [("X", "Y"), ("Y", "X"), ("X", "X"), ("Y", "Y")]:
        c = lie_composition.get(m, sp.Integer(0))
        expected = 1 if m == () else 0   # never () here, always 0
        diff = sp.expand(c - 0)
        if diff != 0:
            any_nonzero = True
            print(f"    {m}: {diff}")
    if not any_nonzero:
        raise AssertionError("A14-Lie-not-self-adjoint: expected non-zero residual; got 0")

    # At least one of the above is non-zero, confirming Lie is not
    # self-adjoint.  The specific coefficient at XY is 0 * Dt^2 + 0 * Dt +
    # leading term = Dt * Dt * (...).  We can extract:
    # e^{Dt Y} e^{Dt X} = I + Dt(X + Y) + Dt^2 (YY/2 + YX + XX/2) + ...
    # e^{-Dt Y} e^{-Dt X} = I - Dt(X + Y) + Dt^2 (YY/2 + YX + XX/2) + ...  (note sign)
    # Their product: I + Dt^2 [(YY/2 + YX + XX/2) + (YY/2 + YX + XX/2) - (X+Y)^2]
    #              = I + Dt^2 [YY + 2 YX + XX - (XX + XY + YX + YY)]
    #              = I + Dt^2 [YX - XY]
    # = I - Dt^2 [X, Y].
    # So we expect residual at XY = -Dt^2 and at YX = +Dt^2.
    assert_zero(
        sp.expand(lie_composition.get(("X", "Y"), sp.Integer(0)) - (-Dt**2)),
        "A14-Lie-not-self-adjoint XY: coefficient = -Dt^2",
    )
    assert_zero(
        sp.expand(lie_composition.get(("Y", "X"), sp.Integer(0)) - Dt**2),
        "A14-Lie-not-self-adjoint YX: coefficient = +Dt^2",
    )

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Self-adjointness (time-reversibility) of an integrator",
        r"L(\Delta t)\,L(-\Delta t) \;=\; I \qquad \text{for all } \Delta t",
        label="eq:A14-self-adjoint-def",
    )
    ld.add(
        "Strang splitting is self-adjoint",
        r"\bigl[e^{(\Delta t/2)\,\mathcal{X}}\,e^{\Delta t\,\mathcal{Y}}\,e^{(\Delta t/2)\,\mathcal{X}}\bigr] "
        r"\cdot \bigl[e^{(-\Delta t/2)\,\mathcal{X}}\,e^{-\Delta t\,\mathcal{Y}}\,e^{(-\Delta t/2)\,\mathcal{X}}\bigr] \;=\; I",
        label="eq:A14-Strang-self-adjoint",
    )
    ld.add(
        "Lie splitting is NOT self-adjoint",
        r"\bigl[e^{\Delta t\,\mathcal{Y}}\,e^{\Delta t\,\mathcal{X}}\bigr] "
        r"\cdot \bigl[e^{-\Delta t\,\mathcal{Y}}\,e^{-\Delta t\,\mathcal{X}}\bigr] "
        r"\;=\; I - \Delta t^{2}\,[\mathcal{X}, \mathcal{Y}] + O(\Delta t^{3})",
        label="eq:A14-Lie-not-self-adjoint",
    )
    ld.add(
        "Consequence: self-adjoint integrators are even-order",
        r"\text{Self-adjointness implies the leading error is at an EVEN power of } \Delta t; "
        r"\text{Strang is 2nd order, not 1st.}",
        label="eq:A14-even-order",
    )
    ld.add(
        "Kernel realisation (both forms equivalent)",
        r"\mathcal{X}(\Delta t/2)\,\mathcal{Y}(\Delta t/2)\,\mathcal{Y}(\Delta t/2)\,\mathcal{X}(\Delta t/2) "
        r"\;\equiv\; \mathcal{X}(\Delta t/2)\,\mathcal{Y}(\Delta t)\,\mathcal{X}(\Delta t/2) "
        r"\qquad \text{(§A13 kernel-chain equivalence)}",
        label="eq:A14-kernel-chain",
    )

    ld.write()
    print()
    print("All A14 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
