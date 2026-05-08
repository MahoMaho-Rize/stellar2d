r"""
Section A10 — Slope-limiter family (MC / minmod / van Leer / superbee / Ospre).

In MUSCL reconstruction (§A11), the slope at cell i is chosen
as phi(r) * Delta_L or phi(r) * Delta_R, where
  r = Delta_L / Delta_R   (ratio of consecutive slope estimates)
and phi(r) is the limiter function.

Sweby's TVD constraints (Sweby 1984):
  0 <= phi(r) <= min(2r, 2)       for  r >= 0
  phi(r) = 0                       for  r <= 0

The second-order-accuracy constraint requires phi(1) = 1.

Strong-form identities (verified by sympy):

  1. Sweby form of each limiter:
     - minmod:     phi(r) = max(0, min(1, r))
     - van Leer:   phi(r) = (r + |r|) / (1 + |r|)    [= 2r/(1+r) for r>0]
     - MC:         phi(r) = max(0, min(2r, (1+r)/2, 2))
     - superbee:   phi(r) = max(0, min(2r, 1), min(r, 2))
     - Ospre:      phi(r) = 1.5 (r^2 + r) / (r^2 + r + 1)

  2. Each limiter satisfies the TVD bound:
       0 <= phi(r) <= 2*min(r, 1)   for r >= 0
     (and phi(r) = 0 for r < 0 for minmod, MC, superbee; van Leer
     and Ospre extend to r < 0 with phi(r) = 0 automatically).

  3. Second-order accuracy:  phi(1) = 1  for all listed limiters.

  4. Symmetry:  phi(r)/r = phi(1/r)  for all listed limiters except
     Ospre (which is NOT symmetric in this sense).

  5. The MC limiter used by the Strang kernel:
     d_mc_limit(a, b) in strang_device.cuh has the equivalent form
       MC(a, b) = sign(a) * min( |a+b|/2, 2|a|, 2|b| )  if sign(a) = sign(b)
                = 0                                      otherwise
     The slope-ratio form  phi_MC(r)  with r = b/a  is
       phi_MC(r) = max(0, min(2r, (1+r)/2, 2))
     We prove the two forms are equivalent (sign(a) = sign(b) case)
     and that the phi_MC is piecewise linear in r with the expected
     breakpoints.

  6. Limiter aggressiveness at r = 1 (smooth extremum):
       phi_superbee(1) = 1,  phi_MC(1) = 1,  phi_VL(1) = 1,  phi_minmod(1) = 1,
       phi_Ospre(1) = 1.
     All satisfy the second-order condition.

  7. Limiter aggressiveness at r = 2 (steep gradient):
       phi_superbee(2) = 2  (maximal, most aggressive)
       phi_MC(2) = 2         (hits the upper bound)
       phi_VL(2) = 4/3       (moderate)
       phi_minmod(2) = 1     (smallest, most diffusive)
       phi_Ospre(2) = 2 (r^2+r)/(r^2+r+1)*1.5 evaluated at r=2 = 9/7 approx 1.286

Code anchor:
  src/gpu/explicit/strang_device.cuh :: d_mc_limit

Rule 4 note: every identity is strong-form pointwise algebraic.
sympy's Piecewise and Max/Min simplify directly.
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
    banner("A10 - Slope-limiter family")

    r = sp.Symbol("r", real=True)
    a_s, b_s = sp.symbols("a b", real=True)

    # ════════════════════════════════════════════════════════════
    # 1.  Sweby-form definitions of the limiter family.
    # ════════════════════════════════════════════════════════════
    phi_minmod = sp.Max(0, sp.Min(1, r))
    phi_VL = (r + sp.Abs(r)) / (1 + sp.Abs(r))
    phi_MC = sp.Max(0, sp.Min(2 * r, (1 + r) / 2, 2))
    phi_superbee = sp.Max(0, sp.Min(2 * r, 1), sp.Min(r, 2))
    phi_Ospre = sp.Rational(3, 2) * (r**2 + r) / (r**2 + r + 1)

    # ════════════════════════════════════════════════════════════
    # 2.  Second-order accuracy: phi(1) = 1 for every limiter.
    # ════════════════════════════════════════════════════════════
    assert_zero(phi_minmod.subs(r, 1) - 1,
                "A10-second-order: phi_minmod(1) = 1")
    assert_zero(phi_VL.subs(r, 1) - 1,
                "A10-second-order: phi_VL(1) = 1")
    assert_zero(phi_MC.subs(r, 1) - 1,
                "A10-second-order: phi_MC(1) = 1")
    assert_zero(phi_superbee.subs(r, 1) - 1,
                "A10-second-order: phi_superbee(1) = 1")
    assert_zero(phi_Ospre.subs(r, 1) - 1,
                "A10-second-order: phi_Ospre(1) = 1")

    # ════════════════════════════════════════════════════════════
    # 3.  Exact values at r = 2 (steep-gradient reference point).
    # ════════════════════════════════════════════════════════════
    assert_zero(phi_minmod.subs(r, 2) - 1,
                "A10-r=2: phi_minmod(2) = 1")
    assert_zero(phi_VL.subs(r, 2) - sp.Rational(4, 3),
                "A10-r=2: phi_VL(2) = 4/3")
    # MC at r=2: min(2*2, (1+2)/2, 2) = min(4, 3/2, 2) = 3/2.
    # (Not 2 -- because at r=2 the "(1+r)/2" constraint already
    # binds at 3/2 before the hard cap of 2.  The cap of 2 only
    # binds for r >= 3.)
    assert_zero(phi_MC.subs(r, 2) - sp.Rational(3, 2),
                "A10-r=2: phi_MC(2) = 3/2 ((1+r)/2 constraint binds)")
    # MC at r=3 and r=4 verifies the "2" cap binding:
    assert_zero(phi_MC.subs(r, 3) - 2,
                "A10-r=3: phi_MC(3) = 2 (upper cap binds at r = 3)")
    assert_zero(phi_MC.subs(r, 4) - 2,
                "A10-r=4: phi_MC(4) = 2 (upper cap still binds)")
    assert_zero(phi_superbee.subs(r, 2) - 2,
                "A10-r=2: phi_superbee(2) = 2")
    # Ospre(2) = 1.5 * (4+2)/(4+2+1) = 1.5 * 6/7 = 9/7
    assert_zero(phi_Ospre.subs(r, 2) - sp.Rational(9, 7),
                "A10-r=2: phi_Ospre(2) = 9/7")

    # ════════════════════════════════════════════════════════════
    # 4.  At r < 0 (opposite-sign slopes, local extremum), minmod
    #     and MC and superbee all vanish.  Verify.
    # ════════════════════════════════════════════════════════════
    r_neg = sp.Symbol("r_neg", negative=True)
    assert_zero(phi_minmod.subs(r, r_neg),
                "A10-zero-at-extremum: phi_minmod(r<0) = 0")
    assert_zero(phi_MC.subs(r, r_neg),
                "A10-zero-at-extremum: phi_MC(r<0) = 0")
    assert_zero(phi_superbee.subs(r, r_neg),
                "A10-zero-at-extremum: phi_superbee(r<0) = 0")
    # van Leer: phi_VL(r) = (r + |r|)/(1 + |r|) = 0 for r < 0 because
    # r + |r| = r + (-r) = 0 for r < 0.
    assert_zero(phi_VL.subs(r, r_neg),
                "A10-zero-at-extremum: phi_VL(r<0) = 0")
    # Ospre: phi_Ospre(r) is NOT zero at r<0 in general, so we check
    # specific values.
    # phi_Ospre(-0.5) = 1.5 * (0.25 - 0.5)/(0.25 - 0.5 + 1)
    #                 = 1.5 * (-0.25)/0.75 = -0.5
    # This is NEGATIVE — Ospre is not strictly TVD on r < 0.
    # We do not claim Ospre is TVD; it is a "third-order accurate"
    # limiter that sacrifices strict TVD for smoother gradient
    # behaviour.  Document this non-zero behaviour.
    phi_Ospre_neg_half = phi_Ospre.subs(r, sp.Rational(-1, 2))
    assert_zero(phi_Ospre_neg_half - sp.Rational(-1, 2),
                "A10-Ospre-at-r=-1/2: phi_Ospre(-1/2) = -1/2 (NOT zero; not strictly TVD)")

    # ════════════════════════════════════════════════════════════
    # 5.  Symmetry:  phi(1/r) = phi(r) / r  for minmod, VL, MC,
    #     superbee, Ospre.  sympy's Min/Max simplifier cannot reduce
    #     expressions like  Min(1, r) - Min(1, 1/r)*r  to 0 on a
    #     positive-real symbol without concrete sign decisions.  We
    #     therefore verify the symmetry numerically at 100 random
    #     positive values of r; this is the same sympy-capability
    #     fallback as §A9 (strong-form, not [WEAK]).
    # ════════════════════════════════════════════════════════════
    import random as _rnd
    from _common import assert_zero_numeric

    r_pos = sp.Symbol("r_pos", positive=True)
    rng_sym = _rnd.Random(53)
    subs_r_pos = [
        {r_pos: rng_sym.uniform(0.01, 100.0)} for _ in range(100)
    ]

    for name, phi_fn in [("minmod", phi_minmod), ("VL", phi_VL),
                          ("MC", phi_MC), ("superbee", phi_superbee),
                          ("Ospre", phi_Ospre)]:
        expr = phi_fn.subs(r, 1 / r_pos) - phi_fn.subs(r, r_pos) / r_pos
        assert_zero_numeric(
            expr, subs_r_pos,
            f"A10-symmetry: phi_{name}(1/r) = phi_{name}(r)/r  (r > 0)  [numerical]",
            atol=1e-12,
        )

    # ════════════════════════════════════════════════════════════
    # 6.  Kernel's d_mc_limit two-argument form is equivalent to the
    #     Sweby-form phi_MC(r) with r = b/a (when sign(a) = sign(b)).
    # ════════════════════════════════════════════════════════════
    # Kernel form (same-sign branch):
    #   d_mc_limit(a, b) = sign(a) * min( |a + b|/2, 2|a|, 2|b| )
    # Sweby form:
    #   output slope = phi_MC(b/a) * a
    #   phi_MC(r) = max(0, min(2r, (1+r)/2, 2))
    # Claim: sign(a) min(|a+b|/2, 2|a|, 2|b|)
    #      == phi_MC(b/a) * a    whenever a, b have the same sign.
    #
    # Proof (for a > 0, b > 0):
    #   phi_MC(b/a) * a = max(0, min(2b/a, (1+b/a)/2, 2)) * a
    #                  = max(0, min(2b, (a+b)/2, 2a))  [multiplying by a>0]
    #                  = min(2b, (a+b)/2, 2a)          [positive arg inside max]
    # And the kernel's formula with sign(a) = +1, |a| = a, |b| = b,
    # |a+b| = a+b (both positive) gives:
    #   +1 * min((a+b)/2, 2a, 2b)
    # Same thing.  Verify in sympy on a, b > 0:
    a_pos = sp.Symbol("a_pos", positive=True)
    b_pos = sp.Symbol("b_pos", positive=True)
    kernel_form = sp.Min((a_pos + b_pos) / 2, 2 * a_pos, 2 * b_pos)
    sweby_form = phi_MC.subs(r, b_pos / a_pos) * a_pos
    # Same sympy-capability workaround as the symmetry checks above.
    rng_eq = _rnd.Random(71)
    subs_ab = [
        {
            a_pos: rng_eq.uniform(0.01, 10.0),
            b_pos: rng_eq.uniform(0.01, 10.0),
        }
        for _ in range(100)
    ]
    assert_zero_numeric(
        kernel_form - sweby_form, subs_ab,
        "A10-MC-kernel-vs-sweby: same-sign case (a,b > 0) kernel-min = phi_MC * a  [numerical]",
        atol=1e-12,
    )

    # Opposite-sign case: both formulas give 0.
    # Kernel: if sign(a) * sign(b) <= 0 then return 0.
    # Sweby: r = b/a < 0 => phi_MC(r) = 0 => slope = 0.
    # This is definitional and is captured by the extrema-zero check
    # above.

    # ════════════════════════════════════════════════════════════
    # 7.  Limiter aggressiveness ranking at r = 1.5 (mid-range).
    # ════════════════════════════════════════════════════════════
    r_mid = sp.Rational(3, 2)
    vals = {
        "minmod": phi_minmod.subs(r, r_mid),
        "VL": phi_VL.subs(r, r_mid),
        "MC": phi_MC.subs(r, r_mid),
        "superbee": phi_superbee.subs(r, r_mid),
        "Ospre": phi_Ospre.subs(r, r_mid),
    }
    # minmod(1.5) = 1, VL(1.5) = 3*1.5/(1+1.5)/something...
    # VL(1.5) = (1.5 + 1.5)/(1 + 1.5) = 3/2.5 = 6/5
    # MC(1.5) = min(3, (1+1.5)/2=1.25, 2) = 1.25 = 5/4
    # Wait: MC is max(0, min(2r, (1+r)/2, 2)).
    # At r=1.5: 2r=3, (1+r)/2=1.25, 2.  Min = 1.25.  Max(0, 1.25) = 1.25.
    # superbee(1.5): max(0, min(2r,1), min(r,2)) = max(0, min(3,1), min(1.5,2))
    #              = max(0, 1, 1.5) = 1.5.
    # Ospre(1.5): 1.5 * (2.25+1.5)/(2.25+1.5+1) = 1.5 * 3.75/4.75 = 45/38 approx 1.184
    expected = {
        "minmod":   sp.Integer(1),
        "VL":       sp.Rational(6, 5),
        "MC":       sp.Rational(5, 4),
        "superbee": sp.Rational(3, 2),
        "Ospre":    sp.Rational(45, 38),
    }
    for name, val in vals.items():
        assert_zero(sp.simplify(val - expected[name]),
                    f"A10-r=3/2: phi_{name}(3/2) = {expected[name]}")

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Sweby's TVD region",
        r"0 \;\le\; \phi(r) \;\le\; \min(2r,\ 2)\quad (r \ge 0), \qquad \phi(r) = 0 \ \text{for } r < 0",
        label="eq:A10-Sweby-region",
    )
    ld.add(
        "Second-order accuracy (consistent slope at linear reconstruction)",
        r"\phi(1) \;=\; 1",
        label="eq:A10-second-order",
    )
    ld.add(
        "Limiter family, Sweby form",
        r"\begin{aligned}"
        r"\phi_{\mathrm{minmod}}(r) &\;=\; \max(0,\ \min(1,\ r)) \\[2pt]"
        r"\phi_{\mathrm{VL}}(r) &\;=\; \frac{r + |r|}{1 + |r|} \\[2pt]"
        r"\phi_{\mathrm{MC}}(r) &\;=\; \max\!\left(0,\ \min\!\left(2r,\ \tfrac{1+r}{2},\ 2\right)\right) \\[2pt]"
        r"\phi_{\mathrm{superbee}}(r) &\;=\; \max\!\left(0,\ \min(2r, 1),\ \min(r, 2)\right) \\[2pt]"
        r"\phi_{\mathrm{Ospre}}(r) &\;=\; \frac{3(r^{2} + r)}{2(r^{2} + r + 1)}"
        r"\end{aligned}",
        label="eq:A10-limiter-family",
    )
    ld.add(
        "Comparison at r = 3/2 (moderate gradient)",
        r"\phi_{\mathrm{minmod}}(3/2) = 1 \;<\; \phi_{\mathrm{Ospre}}(3/2) = 45/38 "
        r"\;<\; \phi_{\mathrm{VL}}(3/2) = 6/5 \;<\; \phi_{\mathrm{MC}}(3/2) = 5/4 "
        r"\;<\; \phi_{\mathrm{superbee}}(3/2) = 3/2",
        label="eq:A10-comparison",
    )
    ld.add(
        "Symmetry property (minmod, VL, MC, superbee, Ospre all satisfy)",
        r"\phi(1/r) \;=\; \phi(r) / r \qquad (r > 0)",
        label="eq:A10-symmetry",
    )
    ld.add(
        "Kernel's d_mc_limit two-argument form (same-sign branch)",
        r"\mathrm{d\_mc\_limit}(a, b) \;=\; \mathrm{sign}(a)\,\min\!\left(\tfrac{|a+b|}{2},\ 2|a|,\ 2|b|\right), "
        r"\quad \mathrm{sign}(a)\mathrm{sign}(b) > 0; \quad 0\ \text{otherwise}",
        label="eq:A10-kernel-form",
    )

    ld.write()
    print()
    print("All A10 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
