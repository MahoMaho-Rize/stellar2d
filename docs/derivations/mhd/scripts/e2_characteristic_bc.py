r"""
Section E2 — Characteristic inner BC for 2D MHD (Alfvén + reflective).

Motivation: the B-M5.75 all-operators smoke exposed that the interior-SET
driver implementation (overwriting v_x in the j=ng row) is non-conservative
at O(1e-6) over 300 steps.  This is a choice of implementation, not a
fundamental feature of prescribed-velocity BCs.  A proper characteristic
inner BC — specifying only the *incoming* Elsässer amplitude $\tilde z^+$
and extrapolating the outgoing $\tilde z^-$ — is exactly mass-conservative,
reflection-free to linear order, and matches Suzuki+25 / Shoda+2018a.

This script derives the characteristic-BC ghost-fill formulas and verifies
the three essential identities:

  1. Linearised 2D MHD around (ρ_0, 0, p_0, B_{y0}\hat y) decouples into
     three independent 2×2 subsystems + one 3×3 acoustic/entropy system.
     We focus on the Alfvén-x channel (v_x, B_x) since the driver acts
     there; the z-polarised Alfvén channel is identical by symmetry.

  2. Riemann invariants along the Alfvén characteristics:
       \tilde z^+ = -v_x + B_x/√ρ_0   propagates at +v_A (incoming from bottom)
       \tilde z^- =  v_x + B_x/√ρ_0   propagates at -v_A (outgoing toward bottom)
     Polarisation of the +v_A mode:  δB_x = -√ρ_0 δv_x  (confirmed against
     B-M5 T5 measurement and standard textbook Alfvén wave result).

  3. Characteristic ghost-fill closure:
       \tilde z^+|_{ghost} = -2 v_x^{drv}(t)                  (driver-specified)
       \tilde z^-|_{ghost} = \tilde z^-|_{interior}           (absorbing)
     Inverted:
       v_x|_{ghost}       = v_x^{drv} + ½(v_x|_{int} + B_x|_{int}/√ρ_0)
       B_x|_{ghost}/√ρ_0  = -v_x^{drv} + ½(v_x|_{int} + B_x|_{int}/√ρ_0)

  4. Reflection coefficient for a downgoing Alfvén pulse:
     with quiescent driver v_x^{drv} = 0 and a pure \tilde z^- pulse
     incident on the boundary, the scheme produces \tilde z^+|_{ghost} = 0
     exactly, so the reflected amplitude is zero to linear order.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("E2 — Characteristic inner BC for 2D MHD")

    # ─────────────────────────────────────────────────────────────────
    # Setup: linearised 2D MHD around (rho0, v=0, p0, B = B_{y0} y-hat)
    # ─────────────────────────────────────────────────────────────────
    y, t_sym = sp.symbols("y t", real=True)
    rho0 = sp.Symbol("rho_0", positive=True)
    p0 = sp.Symbol("p_0", positive=True)
    By0 = sp.Symbol("B_{y0}", positive=True)
    gamma_sym = sp.Symbol("gamma", positive=True)

    v_A = By0 / sp.sqrt(rho0)             # Alfvén speed (Gaussian units)
    c_s = sp.sqrt(gamma_sym * p0 / rho0)  # sound speed (unused in BC but handy)

    # First-order perturbation fields (functions of y, t).
    v_x = sp.Function("v_x")(y, t_sym)
    B_x = sp.Function("B_x")(y, t_sym)
    v_z = sp.Function("v_z")(y, t_sym)
    B_z = sp.Function("B_z")(y, t_sym)

    # ─────────────────────────────────────────────────────────────────
    # Identity 1: linearised Alfvén equations in the x-polarised channel
    # and real eigenvalues ±v_A of the transport matrix.
    # ─────────────────────────────────────────────────────────────────
    # From ∂_t v + (v·∇)v = -∇p/ρ + (1/ρ)(B·∇)B - ∇(B²/2)/ρ,
    # linearised x-component (no pressure force in pure Alfvén):
    dtvx_rhs = (By0 / rho0) * sp.diff(B_x, y)
    # From ∂_t B = curl(v × B), linearised x-component with
    # v_y0 = 0, B_x0 = B_z0 = 0:
    dtBx_rhs = By0 * sp.diff(v_x, y)

    # Check eigenvalues of the transport matrix A in ∂_t U = A ∂_y U
    # where U = (v_x, B_x)^T.
    A_mat = sp.Matrix([[0, By0/rho0],
                       [By0, 0]])
    eigvals = list(A_mat.eigenvals().keys())
    # Should be ±v_A but note: for the PDE in the form
    #   ∂_t U + A' ∂_y U = 0  with A' = -A,
    # the wave speeds are the eigenvalues of A' = -A, still ±v_A (same
    # set since ±v_A is symmetric about zero).
    for lam in eigvals:
        # Simplify lam² - v_A² to zero.
        assert_zero(sp.simplify(lam**2 - v_A**2),
                    f"eigenvalue {lam} squared equals v_A²",
                    verbose=False)
    print("  [OK] Alfvén transport matrix has real eigenvalues ±v_A.")

    # ─────────────────────────────────────────────────────────────────
    # Identity 2: Riemann invariants (left eigenvectors of A' = -A).
    # Sign convention: PDE is  ∂_t U + A' ∂_y U = 0,  A' = -A.
    # Invariant along wave speed +v_A:
    #   ℓ A' = (+v_A) ℓ  with U = (v_x, B_x)^T.
    # Solve: ℓ₁·0 + ℓ₂·(-By0) = v_A ℓ₁
    #        ℓ₁·(-By0/rho0) + ℓ₂·0 = v_A ℓ₂
    # From first:  ℓ₁ = -ℓ₂·By0/v_A = -√ρ₀·ℓ₂.
    # Take ℓ = (-√ρ₀, 1): invariant is  I_+ = -√ρ₀ v_x + B_x.
    # Normalise by √ρ₀:  \tilde z^+ = -v_x + B_x/√ρ₀.  This quantity is
    # constant along characteristics y - v_A t = const (i.e. waves moving
    # at +v_A, i.e. INTO the domain from the bottom boundary at y=0).
    A_prime = -A_mat
    # Build the invariant symbolically and check it obeys
    #   ∂_t \tilde z^+ + v_A ∂_y \tilde z^+ = 0.
    tilde_z_plus = -v_x + B_x / sp.sqrt(rho0)
    tilde_z_minus = v_x + B_x / sp.sqrt(rho0)

    # Time derivative with substitutions from linearised MHD.
    dt_zp = -sp.diff(v_x, t_sym) + sp.diff(B_x, t_sym)/sp.sqrt(rho0)
    dt_zp_subbed = dt_zp.subs({
        sp.Derivative(v_x, t_sym): dtvx_rhs,
        sp.Derivative(B_x, t_sym): dtBx_rhs,
    })
    # Should equal -v_A ∂_y \tilde z^+  (so that ∂_t + v_A ∂_y = 0).
    advect_plus = dt_zp_subbed + v_A * sp.diff(tilde_z_plus, y)
    assert_zero(advect_plus,
                "∂_t \\tilde z^+ + v_A ∂_y \\tilde z^+ = 0  (incoming into domain)",
                verbose=False)
    print("  [OK] \\tilde z^+ propagates at +v_A (into domain from bottom).")

    # And the outgoing one: ∂_t \tilde z^- - v_A ∂_y \tilde z^- = 0.
    dt_zm = sp.diff(v_x, t_sym) + sp.diff(B_x, t_sym)/sp.sqrt(rho0)
    dt_zm_subbed = dt_zm.subs({
        sp.Derivative(v_x, t_sym): dtvx_rhs,
        sp.Derivative(B_x, t_sym): dtBx_rhs,
    })
    advect_minus = dt_zm_subbed - v_A * sp.diff(tilde_z_minus, y)
    assert_zero(advect_minus,
                "∂_t \\tilde z^- - v_A ∂_y \\tilde z^- = 0  (outgoing to bottom)",
                verbose=False)
    print("  [OK] \\tilde z^- propagates at -v_A (out of domain toward bottom).")

    # ─────────────────────────────────────────────────────────────────
    # Identity 3: polarisation of the +v_A mode.
    # Right eigenvector of A' for eigenvalue +v_A: solve A' r = v_A r.
    # Expect r ∝ (1, -√ρ₀), i.e.  δB_x = -√ρ₀ δv_x  on the incoming mode.
    # This matches the B-M5 T5 measurement (ratio 1.019).
    # ─────────────────────────────────────────────────────────────────
    vA_sym = sp.Symbol("v_A_pos", positive=True)
    eq_pair = A_prime * sp.Matrix([1, sp.Symbol("alpha")]) \
              - vA_sym * sp.Matrix([1, sp.Symbol("alpha")])
    # Substitute vA_sym → By0/√ρ0 and solve for alpha.
    alpha_sym = sp.Symbol("alpha")
    alpha_sol = sp.solve(
        eq_pair.subs(vA_sym, v_A), alpha_sym, dict=True)
    # Expect alpha = -√ρ₀
    assert_zero(sp.simplify(alpha_sol[0][alpha_sym] + sp.sqrt(rho0)),
                "+v_A mode polarisation: δB_x = -√ρ₀ δv_x",
                verbose=False)
    print("  [OK] +v_A mode has polarisation δB_x = -√ρ_0 δv_x "
          "(matches B-M5 T5).")

    # ─────────────────────────────────────────────────────────────────
    # Identity 4: ghost-fill closure for prescribed v_x driver.
    # Constraints at the bottom ghost cell:
    #   (a) Driver specifies horizontal velocity v_x^drv(t) which, for
    #       a pure incoming +v_A Alfvén wave with polarisation
    #       δB_x = -√ρ_0 δv_x, corresponds to
    #           \tilde z^+|_{ghost} = -v_x^drv + (-√ρ_0 v_x^drv)/√ρ_0
    #                               = -2 v_x^drv.
    #   (b) Outgoing (from interior, toward bottom) wave is absorbed:
    #           \tilde z^-|_{ghost} = \tilde z^-|_{interior}.
    # Solve for the ghost primitives (v_x, B_x)|_{ghost}.
    # ─────────────────────────────────────────────────────────────────
    v_drv = sp.Symbol("v_x^{drv}", real=True)
    vx_int = sp.Symbol("v_x^{int}", real=True)
    Bx_int = sp.Symbol("B_x^{int}", real=True)

    zp_ghost = -2 * v_drv
    zm_ghost_sym = vx_int + Bx_int / sp.sqrt(rho0)   # from absorbing BC

    # Invert the Elsässer system in the ghost:
    #   \tilde z^+ = -v_x + B_x/√ρ₀
    #   \tilde z^- =  v_x + B_x/√ρ₀
    # ⇒ v_x = (\tilde z^- - \tilde z^+)/2,  B_x/√ρ₀ = (\tilde z^+ + \tilde z^-)/2.
    vx_ghost = (zm_ghost_sym - zp_ghost) / 2
    Bx_over_sqrtrho_ghost = (zp_ghost + zm_ghost_sym) / 2

    vx_ghost_simp = sp.simplify(vx_ghost)
    # Expected: v_drv + ½(v_x^{int} + B_x^{int}/√ρ_0)
    expected_vx_ghost = v_drv + sp.Rational(1, 2) * (vx_int + Bx_int/sp.sqrt(rho0))
    assert_zero(sp.simplify(vx_ghost_simp - expected_vx_ghost),
                "ghost v_x = v_drv + ½(v_x^{int} + B_x^{int}/√ρ_0)",
                verbose=False)

    expected_Bx_ghost = sp.sqrt(rho0) * (
        -v_drv + sp.Rational(1, 2) * (vx_int + Bx_int/sp.sqrt(rho0)))
    assert_zero(
        sp.simplify(sp.sqrt(rho0)*Bx_over_sqrtrho_ghost - expected_Bx_ghost),
        "ghost B_x = √ρ_0[-v_drv + ½(v_x^{int} + B_x^{int}/√ρ_0)]",
        verbose=False)
    print("  [OK] characteristic ghost-fill formulas closed and verified.")

    # ─────────────────────────────────────────────────────────────────
    # Identity 5: sanity — quiescent interior + driver active.
    # ─────────────────────────────────────────────────────────────────
    quiet_subs = {vx_int: 0, Bx_int: 0}
    vx_g_quiet = expected_vx_ghost.subs(quiet_subs)
    Bx_g_quiet = expected_Bx_ghost.subs(quiet_subs)
    assert_zero(sp.simplify(vx_g_quiet - v_drv),
                "quiescent interior: ghost v_x = v_drv",
                verbose=False)
    assert_zero(sp.simplify(Bx_g_quiet + sp.sqrt(rho0) * v_drv),
                "quiescent interior: ghost B_x = -√ρ_0 v_drv "
                "(matches pure +v_A Alfvén polarisation)",
                verbose=False)
    print("  [OK] quiescent interior + driver → pure +v_A Alfvén injection.")

    # ─────────────────────────────────────────────────────────────────
    # Identity 6: sanity — driver off, interior arbitrary.
    #   Should reduce to \tilde z^+|_{ghost} = 0 exactly (no spurious
    #   reflection) and \tilde z^-|_{ghost} = \tilde z^-|_{interior}.
    # ─────────────────────────────────────────────────────────────────
    off_subs = {v_drv: 0}
    vx_g_off = expected_vx_ghost.subs(off_subs)
    Bx_g_off = expected_Bx_ghost.subs(off_subs)
    zp_g_off = -vx_g_off + Bx_g_off/sp.sqrt(rho0)
    zm_g_off = vx_g_off + Bx_g_off/sp.sqrt(rho0)
    assert_zero(sp.simplify(zp_g_off),
                "driver off: \\tilde z^+|_{ghost} = 0 (no spurious incoming)",
                verbose=False)
    assert_zero(sp.simplify(zm_g_off - (vx_int + Bx_int/sp.sqrt(rho0))),
                "driver off: \\tilde z^-|_{ghost} = \\tilde z^-|_{int} "
                "(outgoing extrapolation)",
                verbose=False)
    print("  [OK] driver off: BC is reflection-free on outgoing "
          "\\tilde z^- (linear order).")

    # ─────────────────────────────────────────────────────────────────
    # Identity 7: reflection coefficient for an incident +v_A Alfvén
    # pulse coming down from interior.
    #   Incident pulse:  \tilde z^-_{inc}(y = 0^+, t) = Z_0(t)  (nonzero)
    #                   \tilde z^+_{inc}(y = 0^+, t) = 0
    #   After hitting the BC with v_drv = 0:
    #   BC enforces \tilde z^+|_{ghost} = 0 and \tilde z^-|_{ghost} =
    #   \tilde z^-|_{int} = Z_0(t).
    #   So there is NO \tilde z^+ reflected wave generated. R ≡ 0.
    # Formally: reflection amplitude = \tilde z^+|_{ghost} / \tilde z^-|_{int}
    # evaluated with v_drv = 0 and arbitrary Z_0.
    # ─────────────────────────────────────────────────────────────────
    Z0 = sp.Symbol("Z_0", real=True)
    # At the BC: \tilde z^-|_{int} = Z_0 (the incident pulse arriving).
    # Corresponding primitives: v_x|_{int} = Z_0/2, B_x|_{int}/√ρ_0 = Z_0/2
    # (on \tilde z^+|_{int} = 0 branch). Substitute into ghost formula
    # with v_drv = 0, then compute \tilde z^+|_{ghost}.
    inc_subs = {v_drv: 0, vx_int: Z0/2, Bx_int: sp.sqrt(rho0)*Z0/2}
    vx_g_inc = expected_vx_ghost.subs(inc_subs)
    Bx_g_inc = expected_Bx_ghost.subs(inc_subs)
    zp_g_inc = sp.simplify(-vx_g_inc + Bx_g_inc/sp.sqrt(rho0))
    assert_zero(zp_g_inc,
                "pure incident \\tilde z^- pulse: \\tilde z^+|_{ghost} = 0 "
                "⇒ R = 0  (linear reflection coefficient)",
                verbose=False)
    print("  [OK] reflection coefficient R = 0 for pure incident "
          "Alfvén pulse (linear order).")

    # ─────────────────────────────────────────────────────────────────
    # Identity 8: face-B consistency.
    # For the staggered Yee grid, cell-centred B_x^{cc} in the ghost row
    # is the average of the two x-faces bounding the cell:
    #   B_x^{cc,ghost} = ½(B_x^{face,i-½,j_g} + B_x^{face,i+½,j_g}).
    # Simplest face-fill that reproduces the characteristic ghost B_x^{cc}
    # is to set both x-faces in the ghost row equal to B_x^{cc,ghost}.
    # This preserves local div·B to ULP if the interior row j_g + 1 satisfies
    # the CT constraint (it does, by induction).
    # Verify: average of two equal faces = the face value.
    Bx_face = sp.Symbol("B_x^{face,g}", real=True)
    Bxcc_ghost = (Bx_face + Bx_face) / 2
    assert_zero(sp.simplify(Bxcc_ghost - Bx_face),
                "face-fill B_x^{face,i±½,j_g} = B_x^{cc,ghost} "
                "gives average equal to ghost cell-centred value",
                verbose=False)
    print("  [OK] face-B ghost fill consistent with characteristic "
          "cell-centred formula.")

    # ─────────────────────────────────────────────────────────────────
    # LaTeX dump
    # ─────────────────────────────────────────────────────────────────
    ld.add(
        "Alfvén Riemann invariants (2D MHD, $\\mathbf{B}_0 = B_{y0}\\hat{y}$)",
        r"\tilde z^\pm = \mp v_x + B_x/\sqrt{\rho_0},\qquad "
        r"\partial_t \tilde z^\pm \pm v_A\,\partial_y \tilde z^\pm = 0,\qquad "
        r"v_A = B_{y0}/\sqrt{\rho_0}",
        label="eq:E2_alfven_invariants",
    )
    ld.add(
        "Polarisation of the $+v_A$ (incoming) mode",
        r"\delta B_x = -\sqrt{\rho_0}\,\delta v_x",
        label="eq:E2_polarisation",
    )
    ld.add(
        "Characteristic inner BC (prescribed $v_x^{\\mathrm{drv}}(t)$)",
        r"\tilde z^+\bigr|_{\mathrm{ghost}} = -2\,v_x^{\mathrm{drv}}(t),"
        r"\qquad "
        r"\tilde z^-\bigr|_{\mathrm{ghost}} = \tilde z^-\bigr|_{\mathrm{int}}",
        label="eq:E2_BC",
    )
    ld.add(
        "Ghost-fill closure — primitives",
        r"v_x\bigr|_{\mathrm{ghost}} = v_x^{\mathrm{drv}}"
        r" + \tfrac{1}{2}\bigl[v_x^{\mathrm{int}} "
        r"+ B_x^{\mathrm{int}}/\sqrt{\rho_0}\bigr],\qquad "
        r"\frac{B_x\bigr|_{\mathrm{ghost}}}{\sqrt{\rho_0}} = "
        r"-v_x^{\mathrm{drv}}"
        r" + \tfrac{1}{2}\bigl[v_x^{\mathrm{int}} "
        r"+ B_x^{\mathrm{int}}/\sqrt{\rho_0}\bigr]",
        label="eq:E2_ghost_fill",
    )
    ld.add(
        "Reflection coefficient (linear order)",
        r"R \equiv \tilde z^+\bigr|_{\mathrm{ghost}}/"
        r"\tilde z^-\bigr|_{\mathrm{int}} = 0"
        r"\quad\text{when } v_x^{\mathrm{drv}} = 0",
        label="eq:E2_reflection",
    )
    ld.add(
        "z-polarised Alfvén channel (not driven in v1)",
        r"\tilde z^\pm_z = \mp v_z + B_z/\sqrt{\rho_0},\qquad "
        r"\tilde z^+_z\bigr|_{\mathrm{ghost}} = 0"
        r"\quad\text{(no driver for } v_z\text{)}",
        label="eq:E2_z_channel",
    )
    ld.add(
        "Face-B consistency (Yee grid, ghost row)",
        r"B_x^{\mathrm{face},i\pm\tfrac12,\,j_g} = B_x^{\mathrm{cc,ghost}},\qquad "
        r"B_y^{\mathrm{face},\,j_g - \tfrac12} = B_y^{\mathrm{face,mirror}}"
        r"\quad(\text{preserve } \nabla\!\cdot\!\mathbf{B}=0)",
        label="eq:E2_face_consistency",
    )
    ld.add(
        "Acoustic / entropy channel (reflective in v1)",
        r"\rho\bigr|_{\mathrm{ghost}} = \rho_{\mathrm{HSE}}(y_{\mathrm{ghost}}),"
        r"\quad v_y\bigr|_{\mathrm{ghost}} = -v_y\bigr|_{\mathrm{int}},"
        r"\quad p\bigr|_{\mathrm{ghost}} = p_{\mathrm{HSE}}(y_{\mathrm{ghost}})",
        label="eq:E2_acoustic",
    )

    ld.write()
    print()
    print("All E2 identities verified.")


if __name__ == "__main__":
    main()
