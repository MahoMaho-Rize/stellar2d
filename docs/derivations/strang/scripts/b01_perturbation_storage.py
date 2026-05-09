r"""
Section B1 — Perturbation storage bijection.

The Strang solver stores ``(delta_rho, m_x, m_y, delta_E)`` on top
of an isentropic HSE background ``(rho_bar(y), p_bar(y))``.  At every
device-side read site, the total-state arithmetic uses

    rho = delta_rho + rho_bar(y)
    P   = gm1 * [ E_tot - 0.5 * rho * (u^2 + v^2) ]
    E_tot = delta_E + p_bar(y) / (gamma - 1)

and the momentum components are stored in full because the background
momentum is zero (HSE is static).

Strong-form identities verified:

  1.  Round-trip  (rho, u, v, P)  ->  (delta_rho, m_x, m_y, delta_E)
       ->  (rho, u, v, P)  is the identity on the domain rho>0, P>0.

  2.  Total total-energy identity
         delta_E + p_bar/gm1  ==  P/gm1 + 0.5 rho (u^2 + v^2)
       reduces to  P = p_bar + (gm1)*[delta_E - 0.5 rho (u^2+v^2)]
       when the perturbation of pressure is extracted.

  3.  Zero-perturbation invariant:
         (delta_rho, m_x, m_y, delta_E) = (0, 0, 0, 0)
         <=>  (rho, u, v, P) = (rho_bar, 0, 0, p_bar).

  4.  The Jacobian  det d(delta_rho, m_x, m_y, delta_E) / d(rho, u, v, P)
       is strictly positive on the positivity domain (bijection is
       a smooth diffeomorphism on rho>0, P>0).

Code anchor:
  src/gpu/explicit/strang_device.cuh :: d_cons2prim
  src/gpu/explicit/strang_solver.cu  :: all sites that add d_rho_bar[j]
  src/gpu/explicit/strang_solver.cu  :: k_strang_init_bubble
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
    banner("B1 - Perturbation storage bijection")

    rho, u, v, P = sp.symbols("rho u v P", positive=True)
    gamma = sp.Symbol("gamma", positive=True)
    rho_bar = sp.Symbol("rho_bar", positive=True)   # bg density at this y
    p_bar   = sp.Symbol("p_bar",   positive=True)   # bg pressure at this y
    gm1 = gamma - 1

    # Forward map: primitive + bg -> perturbation storage.
    delta_rho = rho - rho_bar
    m_x       = rho * u
    m_y       = rho * v
    # E_tot = P/gm1 + 0.5 rho (u^2 + v^2);  delta_E = E_tot - p_bar/gm1.
    E_tot   = P / gm1 + sp.Rational(1, 2) * rho * (u**2 + v**2)
    delta_E = E_tot - p_bar / gm1

    # ════════════════════════════════════════════════════════════
    # 1.  Round-trip storage -> primitive.
    #
    # Reverse map: given (delta_rho, m_x, m_y, delta_E) and (rho_bar,
    # p_bar), reconstruct (rho, u, v, P).
    # ════════════════════════════════════════════════════════════
    rho_rt = delta_rho + rho_bar
    u_rt   = m_x / rho_rt
    v_rt   = m_y / rho_rt
    # E_tot = delta_E + p_bar/gm1; pressure from E_tot and kinetic:
    E_tot_rt = delta_E + p_bar / gm1
    P_rt     = gm1 * (E_tot_rt - sp.Rational(1, 2) * rho_rt * (u_rt**2 + v_rt**2))

    assert_zero(
        sp.simplify(rho_rt - rho),
        "B1-roundtrip-rho: delta_rho + rho_bar = rho",
    )
    assert_zero(
        sp.simplify(u_rt - u),
        "B1-roundtrip-u: m_x / rho = u",
    )
    assert_zero(
        sp.simplify(v_rt - v),
        "B1-roundtrip-v: m_y / rho = v",
    )
    assert_zero(
        sp.simplify(P_rt - P),
        "B1-roundtrip-P: gm1 (E_tot - KE) = P",
    )

    # ════════════════════════════════════════════════════════════
    # 2.  Perturbation-pressure split.
    #
    # delta_P := P - p_bar satisfies
    #   delta_P = gm1 * [delta_E - 0.5 rho (u^2 + v^2)].
    # This is the identity that justifies the solver's decision to
    # store delta_E instead of full E_tot.
    # ════════════════════════════════════════════════════════════
    delta_P = P - p_bar
    rhs = gm1 * (delta_E - sp.Rational(1, 2) * rho * (u**2 + v**2))
    assert_zero(
        sp.simplify(delta_P - rhs),
        "B1-pressure-perturbation: delta_P = gm1 [delta_E - KE]",
    )

    # ════════════════════════════════════════════════════════════
    # 3.  Zero-perturbation invariant (both directions).
    # ════════════════════════════════════════════════════════════
    # Forward: at (rho, u, v, P) = (rho_bar, 0, 0, p_bar),
    # the storage is (0, 0, 0, 0).
    zero_subs = {rho: rho_bar, u: 0, v: 0, P: p_bar}
    for sym, name in [
        (delta_rho, "delta_rho"),
        (m_x, "m_x"),
        (m_y, "m_y"),
        (delta_E, "delta_E"),
    ]:
        assert_zero(
            sp.simplify(sym.subs(zero_subs)),
            f"B1-zero-pert-forward-{name}: at HSE state, {name} = 0",
        )

    # Reverse: at (delta_rho, m_x, m_y, delta_E) = (0, 0, 0, 0),
    # (rho, u, v, P) = (rho_bar, 0, 0, p_bar).
    # Define local perturbation symbols so that the reverse map is
    # testable independently of the forward map.
    drho_s, mx_s, my_s, dE_s = sp.symbols(
        "delta_rho_s m_x_s m_y_s delta_E_s", real=True
    )
    rho_s = drho_s + rho_bar
    u_s   = mx_s / rho_s
    v_s   = my_s / rho_s
    E_s   = dE_s + p_bar / gm1
    P_s   = gm1 * (E_s - sp.Rational(1, 2) * rho_s * (u_s**2 + v_s**2))
    zero_pert = {drho_s: 0, mx_s: 0, my_s: 0, dE_s: 0}
    assert_zero(sp.simplify(rho_s.subs(zero_pert) - rho_bar),
                "B1-zero-pert-reverse-rho: rho = rho_bar")
    assert_zero(sp.simplify(u_s.subs(zero_pert)),
                "B1-zero-pert-reverse-u: u = 0")
    assert_zero(sp.simplify(v_s.subs(zero_pert)),
                "B1-zero-pert-reverse-v: v = 0")
    assert_zero(sp.simplify(P_s.subs(zero_pert) - p_bar),
                "B1-zero-pert-reverse-P: P = p_bar")

    # ════════════════════════════════════════════════════════════
    # 4.  Smooth-diffeomorphism Jacobian.
    #
    # The forward map  phi: (rho, u, v, P) -> (delta_rho, m_x, m_y, delta_E)
    # has Jacobian matrix
    #   d phi / d (rho, u, v, P)  =
    #     [[1,        0,       0,      0   ],
    #      [u,        rho,     0,      0   ],
    #      [v,        0,       rho,    0   ],
    #      [0.5(u^2+v^2), rho u, rho v, 1/gm1]]
    # Its determinant is rho^2 / gm1 > 0 on the positivity domain.
    # This is strictly the same determinant as in A2 (which computed
    # det d U / d W = rho^2/gm1) because the perturbation shift is
    # a pure translation in the first and fourth rows and does not
    # change the Jacobian.
    # ════════════════════════════════════════════════════════════
    phi = sp.Matrix([delta_rho, m_x, m_y, delta_E])
    W = sp.Matrix([rho, u, v, P])
    J = phi.jacobian(W)
    det_J = sp.simplify(J.det())
    expected_det = rho**2 / gm1
    assert_zero(
        sp.simplify(det_J - expected_det),
        "B1-jacobian-det: det d phi / d W = rho^2 / (gamma-1)",
    )

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Storage convention",
        r"\mathbf{U}_{\text{store}} \;=\; \bigl(\delta\rho,\; m_x,\; m_y,\; \delta E\bigr)^{\mathsf T} "
        r"\;=\; \bigl(\rho - \bar\rho(y),\;\rho u,\;\rho v,\;E_{\mathrm{tot}} - \bar p(y)/(\gamma-1)\bigr)^{\mathsf T}",
        label="eq:B1-store",
    )
    ld.add(
        "Inverse (decode before arithmetic)",
        r"\rho = \delta\rho + \bar\rho,\qquad "
        r"u = m_x/\rho,\qquad v = m_y/\rho,\qquad "
        r"P = (\gamma-1)\,\bigl[\delta E + \bar p/(\gamma-1)\bigr] - \tfrac{1}{2}\rho(u^2+v^2)",
        label="eq:B1-decode",
    )
    ld.add(
        "Pressure-perturbation identity",
        r"\delta P \;\equiv\; P - \bar p \;=\; (\gamma-1)\bigl[\delta E - \tfrac{1}{2}\rho(u^2+v^2)\bigr]",
        label="eq:B1-dP",
    )
    ld.add(
        "Zero-perturbation invariant",
        r"\delta\rho = m_x = m_y = \delta E = 0 "
        r"\;\Longleftrightarrow\; (\rho, u, v, P) = (\bar\rho,\,0,\,0,\,\bar p)",
        label="eq:B1-zero-pert",
    )
    ld.add(
        "Positive Jacobian (smooth bijection on rho>0, P>0)",
        r"\det\frac{\partial \mathbf{U}_{\text{store}}}{\partial \mathbf{W}} "
        r"\;=\; \frac{\rho^{2}}{\gamma-1} \;>\; 0",
        label="eq:B1-jacobian",
    )
    ld.write()
    print()
    print("All B1 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
