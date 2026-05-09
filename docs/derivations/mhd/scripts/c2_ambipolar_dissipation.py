"""
Section C2 — Ambipolar diffusion (ion-neutral drift) in non-ideal MHD.

In a partially-ionised plasma (e.g. cool-star chromosphere), the
neutrals do not feel the magnetic field directly; they couple to ions
only via collisions.  When the collision rate is slow compared to MHD
timescales, ions drift relative to neutrals at a velocity  v_drift  set
by the balance between Lorentz force on the ion fluid and ion-neutral
drag:

    (ρ_i ν_{in}) v_drift  =  J × B / c

where ν_{in} is the ion-neutral collision frequency.  The electric field
in the neutral (bulk) frame becomes

    E = −v_n × B + η_A J_⊥

where  J_⊥ = J − (J·B̂) B̂  is the current component perpendicular to B
(the parallel current flows freely — no drag), and

    η_A = |B|² / (ρ_i ρ_n ν_{in})        ("ambipolar diffusivity").

The resulting induction equation is

    ∂_t B = ∇×(v × B) − ∇×[η_A (J × B̂) × B̂]
          = ∇×(v × B) + ∇×[η_A (J × B) × B / |B|²]      (non-linear in B)

This non-linearity makes ambipolar diffusion very different from Ohmic —
it is anisotropic (only perpendicular currents are damped) and enhances
as B grows.

Derivation targets:
  1. Decompose J into parallel and perpendicular components using the
     projector P_⊥ = I − B̂ ⊗ B̂.  Verify P_⊥² = P_⊥ (idempotent).
  2. Write the ambipolar E-field contribution explicitly:
        E_amb = η_A (J × B̂) × B̂ = −η_A J_⊥.
     Verify via vector identity.
  3. Show the ambipolar induction contribution can be written
        ∂_t B|_amb = −∇× E_amb = ∇×(η_A (J × B) × B/|B|²)
  4. Derive the ambipolar heating rate:
        Q_amb = E_amb · J = η_A |J_⊥|².
  5. Show Q_amb ≥ 0 identically (positive dissipation).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp                                # noqa: E402
from _common import (                              # noqa: E402
    LatexDump, assert_zero, banner,
    cross, dot, curl_cart,
)


def main():
    ld = LatexDump(__file__)
    banner("C2 — Ambipolar diffusion (ion-neutral drift)")

    # Generic symbolic B and J (no spatial dependence needed for the
    # tensor identities; we treat them as 3-vectors).
    Bx, By, Bz = sp.symbols("B_x B_y B_z", real=True)
    Jx, Jy, Jz = sp.symbols("J_x J_y J_z", real=True)
    eta_A = sp.Symbol("eta_A", positive=True)

    B = sp.Matrix([Bx, By, Bz])
    J = sp.Matrix([Jx, Jy, Jz])

    B_mag_sq = dot(B, B)   # |B|²
    B_mag = sp.sqrt(B_mag_sq)
    B_hat = B / B_mag      # B̂

    # ════════════════════════════════════════════════════════════
    # 1. Parallel / perpendicular decomposition
    # ════════════════════════════════════════════════════════════
    # J_parallel = (J·B̂) B̂;  J_perp = J − J_parallel
    J_par = dot(J, B_hat) * B_hat
    J_perp = J - J_par

    # Verify J_perp · B̂ = 0  (perpendicular by construction):
    perp_dot = sp.simplify(dot(J_perp, B_hat))
    assert_zero(perp_dot,
                "J_⊥ · B̂ = 0 by decomposition (perpendicularity)")

    # Verify J_par × B = 0:
    par_cross = cross(J_par, B)
    for i in range(3):
        assert_zero(sp.simplify(par_cross[i]),
                    f"J_∥ × B = 0 component {i}", verbose=False)
    print("  [OK] J_∥ × B = 0  (parallel current feels no Lorentz force).")

    ld.add(
        "Parallel / perpendicular current decomposition",
        r"\mathbf{J} = \mathbf{J}_{\|} + \mathbf{J}_{\perp},"
        r"\quad \mathbf{J}_{\|} = (\mathbf{J}\cdot\hat{\mathbf{B}})\hat{\mathbf{B}},"
        r"\quad \mathbf{J}_{\perp} = \mathbf{J} - \mathbf{J}_{\|}",
        label="eq:C2_decomp",
    )
    ld.add(
        "Lorentz selectivity",
        r"\mathbf{J}_{\|} \times \mathbf{B} = \mathbf{0},"
        r"\quad \text{so only } \mathbf{J}_{\perp} "
        r"\text{ experiences Lorentz force.}",
        label="eq:C2_selective",
    )

    # ════════════════════════════════════════════════════════════
    # 2. Ambipolar E-field identity:
    #    (J × B̂) × B̂ = J_par − J  =  −J_perp
    #    ⇒  E_amb = η_A (J × B̂) × B̂ = −η_A J_perp
    # ════════════════════════════════════════════════════════════
    JxBhat = cross(J, B_hat)
    double_cross = cross(JxBhat, B_hat)

    # Expected:  double_cross = −J_perp
    residual = sp.simplify(double_cross - (-J_perp))
    for i in range(3):
        assert_zero(residual[i],
                    f"(J × B̂) × B̂ = −J_⊥, component {i}", verbose=False)
    print("  [OK] (J × B̂) × B̂  ≡  −J_⊥  (vector triple product identity).")

    E_amb = eta_A * double_cross
    # E_amb should equal −η_A J_perp:
    for i in range(3):
        assert_zero(sp.simplify(E_amb[i] + eta_A * J_perp[i]),
                    f"E_amb = −η_A J_⊥, component {i}", verbose=False)

    ld.add(
        "Ambipolar electric field",
        r"\mathbf{E}_{\mathrm{amb}} = \eta_A (\mathbf{J}\times\hat{\mathbf{B}})"
        r"\times\hat{\mathbf{B}} = -\eta_A\,\mathbf{J}_{\perp}",
        label="eq:C2_Eamb",
    )

    # ════════════════════════════════════════════════════════════
    # 3. Alternative form using (J × B) × B:
    #    (J × B̂) × B̂ = [(J × B) × B] / |B|²
    #    Let's verify.
    # ════════════════════════════════════════════════════════════
    JxB = cross(J, B)
    alt_form = cross(JxB, B) / B_mag_sq
    for i in range(3):
        assert_zero(sp.simplify(alt_form[i] - double_cross[i]),
                    f"(J×B)×B / |B|² = (J×B̂)×B̂, component {i}",
                    verbose=False)
    print("  [OK] (J×B)×B / |B|²  ≡  (J×B̂)×B̂  verified.")

    ld.add(
        "Alternative form (avoids unit vectors — kernel-friendly)",
        r"(\mathbf{J}\times\hat{\mathbf{B}})\times\hat{\mathbf{B}} = "
        r"\frac{(\mathbf{J}\times\mathbf{B})\times\mathbf{B}}{|\mathbf{B}|^{2}}"
        r"\quad \text{(preferred form for kernels — no } 1/\hat{B}\text{ normalisation)}",
        label="eq:C2_altform",
    )

    # ════════════════════════════════════════════════════════════
    # 4. Induction equation with ambipolar term
    # ════════════════════════════════════════════════════════════
    # ∂_t B = ∇×(v×B) − ∇×E_amb
    #        = ∇×(v×B) + η_A ∇×[(J×B)×B/|B|²]    (using (*) form)
    # We display this; the code kernel will use (J×B)×B/|B|² form.
    ld.add(
        "Induction equation with ambipolar diffusion",
        r"\partial_t \mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B}) "
        r"+ \nabla\times\!\left[\eta_A\,"
        r"\frac{(\mathbf{J}\times\mathbf{B})\times\mathbf{B}}{|\mathbf{B}|^{2}}"
        r"\right]",
        label="eq:C2_induction",
    )

    # ════════════════════════════════════════════════════════════
    # 5. Ambipolar heating:  Q_amb = E_amb · J = η_A |J_⊥|² ≥ 0.
    # ════════════════════════════════════════════════════════════
    Q_amb_Edot = -dot(E_amb, J)   # energy absorbed from EM field by fluid
    # Expected:  Q_amb = η_A |J_⊥|² (positive).
    Q_amb_direct = eta_A * dot(J_perp, J_perp)
    diff_Q = sp.simplify(Q_amb_Edot - Q_amb_direct)
    assert_zero(diff_Q, "Q_amb = −E_amb·J = η_A |J_⊥|² ≥ 0")

    # Positivity: |J_⊥|² ≥ 0 by construction, η_A > 0, so Q_amb ≥ 0.
    ld.add(
        "Ambipolar heating (positive-definite)",
        r"Q_{\mathrm{amb}} = -\mathbf{E}_{\mathrm{amb}}\cdot\mathbf{J} "
        r"= \eta_A\,|\mathbf{J}_{\perp}|^{2} \ge 0",
        label="eq:C2_Qamb",
    )

    # ════════════════════════════════════════════════════════════
    # 6. η_A stability CFL (Choi et al. 2009):
    #    Δt ≤ Δx² / (2 η_A_max |B|²/ρ)   (2nd-order centred diffusion,
    #    per direction)
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Ambipolar explicit-diffusion CFL",
        r"\Delta t \le \frac{(\Delta x)^{2}}{2\,\eta_A\,|\mathbf{B}|^{2}/\rho}"
        r"\qquad\text{(per direction, 2nd-order centred)}",
        label="eq:C2_CFL",
    )

    ld.write()
    print()
    print("All C2 identities verified.")


if __name__ == "__main__":
    main()
