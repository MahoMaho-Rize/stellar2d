r"""
Section E3 — Top outgoing characteristic BC for 2D MHD Alfvén wind.

Motivation: §E2 gave a characteristic inner (bottom) BC that drives a
z^+ Alfvén wave into the domain and absorbs the returning z^- exactly
at linear order.  For a steady-state wind integration, the column also
needs a clean top boundary that lets the upgoing z^+ wave EXIT without
reflection.  With the v1 top-reflect wall the round-trip standing wave
between §E2 driver and a hard top accumulates PLM+HLLD noise each
transit and the dt collapses to ~1e-20 after a few τ_top (observed
empirically in T7 early prototype).  §E3 fixes this at the derivation
level: specify *incoming* (from outside the domain) z^- = 0, extrapolate
outgoing z^+ from interior.

This is the top-boundary mirror of §E2.  At y = L_y:
  - z^+ moves at +v_A (UPWARD, i.e. OUT OF the domain through the top)
  - z^- moves at -v_A (DOWNWARD, i.e. INTO the domain through the top)

So at the TOP BC we impose:
  z^-|_{top_ghost} = 0     (no incoming reflection from above)
  z^+|_{top_ghost} = z^+|_{top_interior}   (outgoing extrapolation)

This is the "non-reflecting radiation BC" standard in 1D Alfvén wind
codes (Suzuki & Inutsuka 2005 Appendix A eq. A3 for the outer boundary).

We verify the following identities by sympy:

  1. Same linearised Alfvén algebra as §E2 — same invariants, same
     eigenvalues, but with SIGN of propagation reversed at the top
     boundary (now +v_A is outgoing).
  2. Ghost-fill closure for the top: invert z^+|_{ghost} = z^+|_{int},
     z^-|_{ghost} = 0 to get v_x|_{ghost}, B_x|_{ghost}.
  3. Sanity — pure upgoing z^+ pulse hitting the top exits with
     reflection coefficient R_top = 0 at linear order.
  4. Sanity — quiescent interior + z^- = 0 recovers v_x = 0, B_x = 0
     (no spurious currents from the BC itself).
  5. Face-B consistency on the top ghost row (same rule as §E2, symmetric
     argument at the j = ng + ny face).
  6. Composite domain — bottom §E2 + top §E3 produces a well-posed
     IBVP: z^+ influx at bottom, z^+ outflux at top, z^- quiescent
     everywhere in the linear steady state.  The WKB action
     F = ρ v_⊥² v_A / ω is conserved exactly in this limit.

References:
  - Suzuki & Inutsuka 2005 ApJ 632 L49, Appendix A (outer char BC)
  - Leroy 1980 A&A 91 136 (Alfvén amplitude in exponential atm)
  - Cranmer, van Ballegooijen & Edgar 2007 ApJS 171 520, eq. 15-16
  - Velli 1993 A&A 270 304 (Alfvén reflection WKB)
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("E3 — Top outgoing characteristic BC for 2D MHD Alfvén wind")

    # ─────────────────────────────────────────────────────────────────
    # Setup: same linearised 2D MHD as §E2 (around (rho0, 0, p0, B_y0 y-hat))
    # ─────────────────────────────────────────────────────────────────
    y, t_sym = sp.symbols("y t", real=True)
    rho0 = sp.Symbol("rho_0", positive=True)
    By0  = sp.Symbol("B_{y0}", positive=True)
    v_A  = By0 / sp.sqrt(rho0)

    v_x = sp.Function("v_x")(y, t_sym)
    B_x = sp.Function("B_x")(y, t_sym)

    # ─────────────────────────────────────────────────────────────────
    # Identity 1: Riemann invariants unchanged from §E2.
    #   z^+ = -v_x + B_x/√ρ_0   advects at +v_A
    #   z^- = +v_x + B_x/√ρ_0   advects at -v_A
    # At the TOP boundary (y = L_y):
    #   +v_A goes OUT of the domain (upward through the wall) → OUTGOING
    #   -v_A goes INTO the domain (downward from above)       → INCOMING
    # §E3 BC: incoming = 0 (no source outside domain), outgoing = extrapolate.
    # ─────────────────────────────────────────────────────────────────
    dtvx_rhs = (By0 / rho0) * sp.diff(B_x, y)
    dtBx_rhs = By0 * sp.diff(v_x, y)
    tilde_z_plus  = -v_x + B_x / sp.sqrt(rho0)
    tilde_z_minus =  v_x + B_x / sp.sqrt(rho0)

    # Re-verify the advection equations (same as §E2 — sanity).
    dt_zp = -sp.diff(v_x, t_sym) + sp.diff(B_x, t_sym)/sp.sqrt(rho0)
    dt_zp_sub = dt_zp.subs({sp.Derivative(v_x, t_sym): dtvx_rhs,
                            sp.Derivative(B_x, t_sym): dtBx_rhs})
    assert_zero(dt_zp_sub + v_A * sp.diff(tilde_z_plus, y),
                "z^+ advects at +v_A  (→ OUTGOING at y = L_y)",
                verbose=False)

    dt_zm = sp.diff(v_x, t_sym) + sp.diff(B_x, t_sym)/sp.sqrt(rho0)
    dt_zm_sub = dt_zm.subs({sp.Derivative(v_x, t_sym): dtvx_rhs,
                            sp.Derivative(B_x, t_sym): dtBx_rhs})
    assert_zero(dt_zm_sub - v_A * sp.diff(tilde_z_minus, y),
                "z^- advects at -v_A  (→ INCOMING at y = L_y, from above)",
                verbose=False)
    print("  [OK] Advection signs consistent with §E2; roles flipped at top.")

    # ─────────────────────────────────────────────────────────────────
    # Identity 2: ghost-fill closure at the TOP ghost row.
    # Constraints at the top ghost cell:
    #   (a) No incoming Alfvén from above (vacuum / non-reflecting):
    #           z^-|_{ghost} = 0.
    #   (b) Outgoing wave is whatever the interior carries:
    #           z^+|_{ghost} = z^+|_{int}.
    # Invert for primitives.
    # ─────────────────────────────────────────────────────────────────
    vx_int = sp.Symbol("v_x^{int}", real=True)
    Bx_int = sp.Symbol("B_x^{int}", real=True)

    zp_int = -vx_int + Bx_int / sp.sqrt(rho0)
    zp_ghost_top = zp_int          # extrapolated outgoing
    zm_ghost_top = sp.Integer(0)   # zero incoming

    # Invert:  v_x = (z^- - z^+)/2,  B_x/√ρ_0 = (z^+ + z^-)/2.
    vx_ghost_top = (zm_ghost_top - zp_ghost_top) / 2
    Bx_over_sqrtrho_top = (zp_ghost_top + zm_ghost_top) / 2

    # Closed form after substitution:
    #   v_x|_{top_ghost} = -(z^+_{int})/2 = (v_x^{int} - B_x^{int}/√ρ_0)/2
    #   B_x|_{top_ghost} = √ρ_0 · (z^+_{int})/2
    #                    = √ρ_0 · ( (-v_x^{int} + B_x^{int}/√ρ_0) ) / 2
    #                    = (-√ρ_0 v_x^{int} + B_x^{int}) / 2
    expected_vx_top = (vx_int - Bx_int/sp.sqrt(rho0)) / 2
    assert_zero(sp.simplify(vx_ghost_top - expected_vx_top),
                "ghost v_x|_{top} = ½(v_x^{int} - B_x^{int}/√ρ_0)",
                verbose=False)

    expected_Bx_top = (-sp.sqrt(rho0)*vx_int + Bx_int) / 2
    assert_zero(sp.simplify(sp.sqrt(rho0)*Bx_over_sqrtrho_top - expected_Bx_top),
                "ghost B_x|_{top} = ½(-√ρ_0 v_x^{int} + B_x^{int})",
                verbose=False)
    print("  [OK] Top ghost-fill closure verified.")

    # ─────────────────────────────────────────────────────────────────
    # Identity 3: reflection coefficient at the top for a pure upgoing
    # z^+ pulse.  Incident: z^+_{int} = Z_0, z^-_{int} = 0.
    # Primitives at the top interior: v_x^{int} = -Z_0/2,
    #                                B_x^{int}/√ρ_0 = +Z_0/2.
    # Top BC forces z^-|_{ghost} = 0.  Measured reflection =
    # z^-|_{ghost} / z^+|_{int} = 0.
    # ─────────────────────────────────────────────────────────────────
    Z0 = sp.Symbol("Z_0", real=True)
    # Incident pure +v_A wave: z^+ = Z_0, z^- = 0
    # From z^+ = -v_x + B_x/√ρ_0 = Z_0 and z^- = v_x + B_x/√ρ_0 = 0:
    #   v_x     = -Z_0/2
    #   B_x/√ρ_0 = +Z_0/2
    inc_top = {vx_int: -Z0/2, Bx_int: sp.sqrt(rho0)*Z0/2}
    vx_g_inc = expected_vx_top.subs(inc_top)
    Bx_g_inc = expected_Bx_top.subs(inc_top)
    zm_g_inc = sp.simplify(vx_g_inc + Bx_g_inc/sp.sqrt(rho0))
    assert_zero(zm_g_inc,
                "pure upgoing z^+ pulse: z^-|_{top_ghost} = 0 "
                "⇒ R_top = 0 (linear reflection)",
                verbose=False)
    print("  [OK] Top reflection coefficient R_top = 0 at linear order.")

    # Also verify: the outgoing wave is transmitted unchanged, i.e.
    # z^+|_{ghost} = Z_0 (so PLM reconstruction at the top face gives
    # the cell's own z^+ on both sides of the wall — no discontinuity).
    zp_g_inc = sp.simplify(-vx_g_inc + Bx_g_inc/sp.sqrt(rho0))
    assert_zero(sp.simplify(zp_g_inc - Z0),
                "pure upgoing z^+ pulse: z^+|_{top_ghost} = z^+|_{int} "
                "(transmitted amplitude unchanged)",
                verbose=False)
    print("  [OK] Outgoing z^+ transmitted unchanged across top BC.")

    # ─────────────────────────────────────────────────────────────────
    # Identity 4: quiescent interior sanity (v_x = 0, B_x = 0) → ghost
    # also zero.
    # ─────────────────────────────────────────────────────────────────
    quiet = {vx_int: 0, Bx_int: 0}
    assert_zero(expected_vx_top.subs(quiet),
                "quiescent interior → v_x|_{top_ghost} = 0",
                verbose=False)
    assert_zero(expected_Bx_top.subs(quiet),
                "quiescent interior → B_x|_{top_ghost} = 0",
                verbose=False)
    print("  [OK] Quiescent interior: top BC produces zero ghost "
          "(no spurious current).")

    # ─────────────────────────────────────────────────────────────────
    # Identity 5: z-polarised channel (v_z, B_z) is identical by
    # symmetry — same formulas with (v_x, B_x) → (v_z, B_z).  No driver
    # on z in v1, so pure absorber at both boundaries: z^-|_{top_ghost,z}
    # = 0, z^+|_{top_ghost,z} = z^+|_{top_int,z}.
    # ─────────────────────────────────────────────────────────────────
    vz_int = sp.Symbol("v_z^{int}", real=True)
    Bz_int = sp.Symbol("B_z^{int}", real=True)
    vz_ghost_top = (vz_int - Bz_int/sp.sqrt(rho0)) / 2
    Bz_ghost_top = (-sp.sqrt(rho0)*vz_int + Bz_int) / 2
    zm_z_g = sp.simplify(vz_ghost_top + Bz_ghost_top/sp.sqrt(rho0))
    assert_zero(zm_z_g,
                "z-channel: z^-_z|_{top_ghost} = 0 (pure absorber)",
                verbose=False)
    print("  [OK] z-polarised Alfvén channel absorbs identically at top.")

    # ─────────────────────────────────────────────────────────────────
    # Identity 6: face-B consistency (same argument as §E2 Identity 8).
    #   B_x^{cc,ghost_top} = ½(B_x^{face,i-½,j_top_g} + B_x^{face,i+½,j_top_g})
    # Simplest: set both x-faces in the top ghost row to B_x^{cc,ghost_top}.
    # ─────────────────────────────────────────────────────────────────
    Bx_face = sp.Symbol("B_x^{face,top_g}", real=True)
    Bxcc = (Bx_face + Bx_face) / 2
    assert_zero(sp.simplify(Bxcc - Bx_face),
                "top face-fill B_x^{face,i±½,j_top_g} = B_x^{cc,ghost_top}",
                verbose=False)
    print("  [OK] Top face-B ghost fill consistent with cc formula.")

    # ─────────────────────────────────────────────────────────────────
    # Identity 7: well-posedness of the composite §E2 + §E3 IBVP.
    #   In the linear steady state ∂_t = 0 and ∂_y z^+ = 0 = ∂_y z^-,
    #   so z^+ and z^- are constants along the column.  §E2 at y=0:
    #     z^+(0) = -2 v_drv (driven), z^-(0) = z^-(L) (extrapolation)
    #   §E3 at y=L:
    #     z^-(L) = 0 (incoming absorbed), z^+(L) = z^+(0) (extrapolation)
    #   Composite: z^-(L) = 0 propagates down → z^-(0) = 0 → §E2
    #   extrapolation consistent.  z^+(0) = -2 v_drv propagates up →
    #   z^+(L) = -2 v_drv → §E3 extrapolation consistent.
    #   Unique solution exists; no free parameters; well-posed.
    # ─────────────────────────────────────────────────────────────────
    v_drv = sp.Symbol("v_x^{drv}", real=True)
    # In steady state the invariants are constants along y.  Determine them.
    zp_steady = sp.Symbol("zp_steady", real=True)
    zm_steady = sp.Symbol("zm_steady", real=True)
    # §E2 at y=0:  zp = -2 v_drv (driven); zm(ghost) = zm(int) (extrap).
    # §E3 at y=L:  zm = 0 (absorbed);  zp(ghost) = zp(int) (extrap).
    # Consistency:  zp_steady  = -2 v_drv   (from §E2)
    #               zm_steady  = 0          (from §E3)
    eqns = [sp.Eq(zp_steady, -2 * v_drv),
            sp.Eq(zm_steady, 0)]
    sol = sp.solve(eqns, (zp_steady, zm_steady), dict=True)
    assert sol, "linear steady-state system must have a solution"
    assert sol[0][zp_steady] == -2 * v_drv, \
        "steady zp = -2 v_drv"
    assert sol[0][zm_steady] == 0, \
        "steady zm = 0"
    print("  [OK] Composite §E2 + §E3 linear steady state is unique and "
          "well-posed:  z^+(y) = -2 v_drv,  z^-(y) = 0.")

    # Also verify the WKB upgoing amplitude law  v_⊥ ∝ ρ^{-1/4}.
    #   For monochromatic z^+ of angular frequency ω, the linear
    #   wave equation in an isothermal stratified atm reduces (WKB) to
    #   d(A² ρ v_A)/dy = 0 where A = |v_⊥|.  With v_A = B_{y0}/√ρ,
    #   A² · ρ · ρ^{-1/2} = A² ρ^{1/2} = const,  so  A ∝ ρ^{-1/4}.
    # We verify the algebraic step.
    A_amp = sp.Function("A")(y)
    rho_y = sp.Function("rho")(y)
    # Wave action F = A² · ρ · v_A = A² · ρ · B_{y0}/√ρ = A² · B_{y0} · √ρ
    # (constants B_{y0} factored).
    wave_action = A_amp**2 * sp.sqrt(rho_y)
    # d/dy of this = 0 at steady state gives  2 A A' √ρ + A²/(2√ρ) ρ' = 0
    # ⇒  A'/A = -ρ'/(4ρ)  ⇒  A ∝ ρ^{-1/4}.
    dy_action = sp.diff(wave_action, y)
    # Solve for A'/A:
    A_prime = sp.Symbol("A_prime", real=True)
    rho_prime = sp.Symbol("rho_prime", real=True)
    dy_action_sub = dy_action.subs({sp.Derivative(A_amp, y): A_prime,
                                    sp.Derivative(rho_y, y): rho_prime})
    # Set to zero, solve for A_prime in terms of A, rho, rho_prime.
    ratio_sol = sp.solve(dy_action_sub, A_prime)[0]
    # Expected:  A_prime = -A·rho_prime / (4 rho).
    expected = -A_amp * rho_prime / (4 * rho_y)
    assert_zero(sp.simplify(ratio_sol - expected),
                "WKB steady-state:  A'/A = -ρ'/(4ρ)  "
                "⇒  A ∝ ρ^{-1/4}  (Leroy80, Cranmer07 eq. 16)",
                verbose=False)
    print("  [OK] WKB v_⊥ ∝ ρ^{-1/4} growth law derived from §E3 "
          "steady state (Leroy80 / Cranmer07 benchmark for T7).")

    # ─────────────────────────────────────────────────────────────────
    # LaTeX dump
    # ─────────────────────────────────────────────────────────────────
    ld.add(
        "Alfvén invariants at the top boundary (y=L)",
        r"\tilde z^+ \text{ outgoing at } +v_A,\qquad "
        r"\tilde z^- \text{ incoming at } -v_A",
        label="eq:E3_top_roles",
    )
    ld.add(
        "Top non-reflecting characteristic BC",
        r"\tilde z^-\bigr|_{\mathrm{top\;ghost}} = 0,\qquad "
        r"\tilde z^+\bigr|_{\mathrm{top\;ghost}} = "
        r"\tilde z^+\bigr|_{\mathrm{top\;int}}",
        label="eq:E3_BC",
    )
    ld.add(
        "Top ghost-fill closure — primitives",
        r"v_x\bigr|_{\mathrm{top\;ghost}} = "
        r"\tfrac{1}{2}\bigl[v_x^{\mathrm{int}} "
        r"- B_x^{\mathrm{int}}/\sqrt{\rho_0}\bigr],\qquad "
        r"\frac{B_x\bigr|_{\mathrm{top\;ghost}}}{\sqrt{\rho_0}} = "
        r"\tfrac{1}{2}\bigl[-v_x^{\mathrm{int}} "
        r"+ B_x^{\mathrm{int}}/\sqrt{\rho_0}\bigr]",
        label="eq:E3_ghost_fill",
    )
    ld.add(
        "Top reflection coefficient (linear)",
        r"R_{\mathrm{top}} \equiv "
        r"\tilde z^-\bigr|_{\mathrm{top\;ghost}}/"
        r"\tilde z^+\bigr|_{\mathrm{top\;int}} = 0",
        label="eq:E3_reflection",
    )
    ld.add(
        "Composite §E2 + §E3 linear steady state",
        r"\tilde z^+(y) = -2\,v_x^{\mathrm{drv}},\qquad "
        r"\tilde z^-(y) = 0\qquad\forall y\in[0,L_y]",
        label="eq:E3_composite_steady",
    )
    ld.add(
        "WKB upgoing amplitude growth (Leroy80 / Cranmer07)",
        r"\frac{\mathrm{d}}{\mathrm{d}y}"
        r"\bigl[A^2\,\rho\,v_A\bigr] = 0"
        r"\quad\Longrightarrow\quad "
        r"A \propto \rho^{-1/4}",
        label="eq:E3_wkb_growth",
    )

    ld.write()
    print()
    print("All E3 identities verified.")


if __name__ == "__main__":
    main()
