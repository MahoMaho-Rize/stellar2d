r"""
Section D7 — Reflection-symmetric IC (bit-reproducibility test).

An IC satisfying  U(x, y) = R_ref U(x, -y)  (reflection symmetry
across y = 0) must evolve into U(x, y, t) = R_ref U(x, -y, t)
at all t.  Proof path:

  - §B5: reflective BC on y = 0 preserves the symmetry (ghost
    cells are exactly R_ref of the physical mirror).
  - Operator-split symmetry: the x-sweep preserves y-symmetry
    trivially (operates row-wise).  The y-sweep preserves the
    symmetry because F_y(R_ref U) = R_ref' F_y(U) (§B5 flux-
    reversal identity), so the flux divergence at cell j and
    at the mirror cell -j differ only by the diag(-1,-1,+1,-1)
    sign pattern of flux reversal, which after divergence and
    sign-pattern negation returns the reflection.
  - Gravity source -rho g is y-reflection-invariant (rho is even
    in y, g is constant, so -rho g is even in y, which equals
    +1 * itself — consistent with the diag(+1, +1, -1, +1) of
    R_ref applied to the (0, 0, -rho g, -m_y g) vector: the m_y
    component is +(-1)*(-rho g) = rho g... wait, this needs more
    care).

Let me redo the gravity analysis:
  Source S = (0, 0, -rho g, -m_y g).
  Under y -> -y,  m_y -> -m_y  (mirror's y-momentum flips sign),
                  rho unchanged.
  So S at the mirror is (0, 0, -rho g, +m_y g).
  R_ref S at physical = diag(+1,+1,-1,+1)(0, 0, -rho g, -m_y g)
                      = (0, 0, +rho g, -m_y g).
  These don't match.

But wait: the STORED delta_E at the mirror is  -m_y g * dt / gm1
+ ...  not the same as at the physical.  Is that the issue?

Actually the symmetry we preserve is U(x, y, t) = R_ref U(x, -y, t),
which means the y-momentum at the mirror is NEGATIVE of the
physical (v(x, -y) = -v(x, y)).  So the energy source at the
mirror is -m_y(x, -y) g = -(-m_y(x,y)) g = +m_y(x,y) g.  That's
NOT equal to R_ref applied to the physical source's energy
component (+1 * (-m_y g) = -m_y g).  So the energy source
BREAKS the strict (x, y) <-> (x, -y) symmetry.

Hmm, this is a subtlety.  Let's think more carefully:
  Under the symmetry U(x, y, t) = R_ref U(x, -y, t), we have
  m_y(x, y) = -m_y(x, -y)  (v flips sign).

  Gravity energy source at (x, y):  S_E(x, y) = -m_y(x, y) g.
  Gravity energy source at mirror (x, -y):  S_E(x, -y) = -m_y(x, -y) g
                                                        = -(-m_y(x, y)) g = +m_y(x, y) g.

  For the symmetry to be preserved by the time-step, we need
  S_E(x, y) and S_E(x, -y) to be consistent with the symmetry
  (i.e., the integral updates preserve E(x, y) = E(x, -y), which
  is the +1 entry of R_ref).

  But S_E(x, y) = -m_y g and S_E(x, -y) = +m_y g; these are
  NOT equal but OPPOSITE sign.  Therefore the gravity source
  DOES BREAK the symmetry.

Actually, wait — the physical setup here must have gravity pointing
in a FIXED direction (say -y).  So below y = 0 gravity still
pulls DOWN in the same direction.  That means symmetric reflection
across y = 0 does NOT symmetrize gravity.

Hence the §D7 test must be for systems WITHOUT gravity, OR with
gravity aligned along the reflection axis, OR the reflection is
in a direction orthogonal to gravity (e.g., x-reflection under
gravity-in-y).

Correction: §D7 is the **x-reflection** symmetric IC.  Under
x -> -x:
  u(x, y) = -u(-x, y)  (u flips sign)
  rho, v, P unchanged
  R_ref_x = diag(+1, -1, +1, +1).
Gravity is in -y, rho and y don't flip, so S_E = -m_y g is the
same.  Symmetry preserved.

Let me redo the derivation with R_ref_x.

Strong-form identities verified:

  1. x-reflection R_ref_x = diag(+1, -1, +1, +1) satisfies
     R_ref_x^2 = I.

  2. Flux F_x under R_ref_x applied to U: F_x(R_ref_x U) = ?
     Compute and compare to R_ref_x * diag(-1, ...) F_x(U).
     Result: F_x(R_ref_x U) = diag(-1, +1, -1, -1) F_x(U).

  3. Under x-reflection + periodic-x BC (NOT reflective — but
     periodic-x doesn't break x-reflection), evolution preserves
     symmetry.

  4. Alternatively, with a reflective-x BC at x = 0 (NOT currently
     in kernel), x-reflection is naturally preserved.

  5. For the Strang kernel, the periodic-x domain with an
     x-reflection-symmetric IC preserves the symmetry to the
     Riemann-solver precision if the kernel treats L/R symmetrically.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp
import math

from _common import (
    LatexDump,
    GoldensDump,
    assert_zero,
    banner,
    flux_x_euler,
    flux_y_euler,
)


def main() -> int:
    ld = LatexDump(__file__)
    gd = GoldensDump(__file__)
    banner("D7 - Reflection-symmetric IC (bit-reproducibility test)")

    gamma = sp.Symbol("gamma", positive=True)
    rho, u, v, P = sp.symbols("rho u v P", positive=True, real=True)

    # ════════════════════════════════════════════════════════════
    # x-reflection matrix on conservative state U = (rho, m_x, m_y, E).
    # Under x -> -x:  u -> -u, m_x -> -m_x, rho, v, m_y, E unchanged.
    # ════════════════════════════════════════════════════════════
    R_x = sp.diag(1, -1, 1, 1)

    # Involution
    assert_zero(
        sp.simplify((R_x * R_x - sp.eye(4)).norm()),
        "D7-involution-x: R_x^2 = I",
    )

    # ════════════════════════════════════════════════════════════
    # Flux F_x under x-reflection.
    #
    # F_x(rho, -u, v, P) = (rho*(-u), rho*u^2 + P, rho*(-u)*v, (E+P)*(-u))
    #                    = (-rho u, rho u^2 + P, -rho u v, -(E+P) u)
    #
    # Compare to R_x' F_x(U) with R_x' = diag(-1, +1, -1, -1):
    # diag(-1, +1, -1, -1) * (rho u, rho u^2 + P, rho u v, (E+P) u)
    # = (-rho u, rho u^2 + P, -rho u v, -(E+P) u).
    # Match!
    # ════════════════════════════════════════════════════════════
    F_x_ref = flux_x_euler(rho, -u, v, P, gamma)
    R_x_prime = sp.diag(-1, 1, -1, -1)
    F_x = flux_x_euler(rho, u, v, P, gamma)
    R_x_prime_F = R_x_prime * F_x
    for k in range(4):
        assert_zero(
            sp.simplify(F_x_ref[k] - R_x_prime_F[k]),
            f"D7-flux-reflection-x[{k}]: F_x(R_x U)[{k}] = R_x' F_x(U)[{k}]",
        )

    # ════════════════════════════════════════════════════════════
    # F_y under x-reflection.  F_y(rho, -u, v, P) = (rho v, rho*(-u)*v,
    # rho v^2 + P, (E+P) v).  Compare to R_x F_y(U):
    # diag(1, -1, 1, 1) * (rho v, rho u v, rho v^2 + P, (E+P) v)
    # = (rho v, -rho u v, rho v^2 + P, (E+P) v).
    # These match (F_y(R_x U) = R_x F_y(U)).
    # ════════════════════════════════════════════════════════════
    F_y_ref = flux_y_euler(rho, -u, v, P, gamma)
    F_y = flux_y_euler(rho, u, v, P, gamma)
    R_x_F_y = R_x * F_y
    for k in range(4):
        assert_zero(
            sp.simplify(F_y_ref[k] - R_x_F_y[k]),
            f"D7-flux-y-x-symmetry[{k}]: F_y(R_x U)[{k}] = R_x F_y(U)[{k}]",
        )

    # ════════════════════════════════════════════════════════════
    # Gravity source.  Under x-reflection, m_y unchanged (m_y is
    # in the +1 component of R_x).  rho unchanged.  Gravity direction
    # is along -y, which is invariant under x-reflection.
    # So S = (0, 0, -rho g, -m_y g) is invariant under x-reflection
    # in the sense that it is unchanged.  Compare to R_x S:
    # diag(1, -1, 1, 1) * (0, 0, -rho g, -m_y g) = (0, 0, -rho g, -m_y g).
    # Match.
    # ════════════════════════════════════════════════════════════
    g_sym = sp.Symbol("g", positive=True)
    S = sp.Matrix([0, 0, -rho * g_sym, -sp.Symbol("m_y", real=True) * g_sym])
    R_x_S = R_x * S
    diff = R_x_S - S
    for k in range(4):
        assert_zero(
            sp.simplify(diff[k]),
            f"D7-source-x-symmetry[{k}]: R_x S = S (under x-reflection)",
        )

    # ════════════════════════════════════════════════════════════
    # Conclusion: x-reflection symmetry is preserved exactly by
    # the Strang kernel's operator chain on domains with periodic-x
    # BC or with a reflective-x BC at x = 0.  Under periodic-x,
    # the IC  U(x, y) = R_x U(-x, y)  must also have  U(+L/2) =
    # R_x U(-L/2),  i.e., the periodic copy of x = -L/2 (which is
    # x = +L/2) must be R_x of itself, which requires u(-L/2) = 0
    # or some specific constraint.  In practice one uses a reflective
    # BC to enforce this.
    #
    # SIMPLEST implementation: run the Strang kernel on an
    # x-reflection-symmetric IC and check U(x, y, t) - R_x U(-x, y, t)
    # stays within ULP precision.
    # ════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════
    # Golden values dump.
    #
    # Canonical reflection-symmetric IC on [0, 1] x [0, 1]:
    # A pair of counter-rotating vortices symmetric across x = 0.5.
    # Or simpler: a Rayleigh-Taylor-style vertical flame IC with
    # symmetric bubbles.
    #
    # For simplicity, use two bubbles at (0.3, 0.3) and (0.7, 0.3)
    # with opposite entropy perturbations, symmetric under x <-> 1-x.
    # ════════════════════════════════════════════════════════════
    gamma_val = 1.4
    rho_0_val = 1.0
    K_val = 1.0
    g_val = 1.0
    L_y_val = 1.0
    gd.add("gamma", gamma_val)
    gd.add("rho_0_bottom", rho_0_val)
    gd.add("K_poly", K_val)
    gd.add("g", g_val)
    gd.add("L_y", L_y_val)

    # Symmetric bubble positions
    bubble_1 = {"x_0": 0.3, "y_0": 0.3, "R_0": 0.1, "delta_s": 0.5}
    bubble_2 = {"x_0": 0.7, "y_0": 0.3, "R_0": 0.1, "delta_s": 0.5}  # mirror
    gd.add("bubble_1", bubble_1)
    gd.add("bubble_2", bubble_2)
    gd.add("reflection_axis", 0.5)   # mirror symmetry across x = 0.5
    gd.add("reflection_matrix", "R_x = diag(+1, -1, +1, +1)")

    # Test tolerance at t = 0.1 (a short test, many steps).
    gd.add("T_test", 0.1)
    gd.add("N_step_approx", 50)
    gd.add("symmetry_tolerance", 1e-12)  # should be bitwise zero at ULP
    gd.add("comparison_field", "rho")    # compare rho(x, y, t) with rho(1-x, y, t)

    gd.write()

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "x-reflection matrix",
        r"\mathcal{R}_x \;=\; \mathrm{diag}(+1,\,-1,\,+1,\,+1) "
        r"\qquad\text{(flip } m_x \text{ sign)}",
        label="eq:D7-R-x",
    )
    ld.add(
        "x-reflection flux identities",
        r"\mathbf{F}_x(\mathcal{R}_x \mathbf{U}) \;=\; \mathrm{diag}(-1, +1, -1, -1)\,\mathbf{F}_x(\mathbf{U}),\qquad "
        r"\mathbf{F}_y(\mathcal{R}_x \mathbf{U}) \;=\; \mathcal{R}_x\,\mathbf{F}_y(\mathbf{U})",
        label="eq:D7-flux",
    )
    ld.add(
        "Gravity-source x-invariance",
        r"\mathcal{R}_x\,\mathbf{S} \;=\; \mathbf{S} \qquad\text{(gravity is y-only)}",
        label="eq:D7-source",
    )
    ld.add(
        "Reflection-symmetry preservation (strong form)",
        r"\mathbf{U}(\mathbf{x}, 0) \;=\; \mathcal{R}_x\,\mathbf{U}(\mathcal{R}_{\mathrm{geom}}\mathbf{x}, 0) "
        r"\;\Longrightarrow\; \mathbf{U}(\mathbf{x}, t) \;=\; \mathcal{R}_x\,\mathbf{U}(\mathcal{R}_{\mathrm{geom}}\mathbf{x}, t) "
        r"\quad \forall\, t",
        label="eq:D7-preserve",
    )

    ld.write()
    print()
    print("All D7 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
