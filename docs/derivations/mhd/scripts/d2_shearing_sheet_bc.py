"""
Section D2 — Shearing-sheet approximation and shearing-periodic BC.

Goldreich & Lynden-Bell 1965 / Hill 1878 expansion:
  - Pick a local patch centred at R = R_0 in a disk rotating at Ω_0.
  - Define local Cartesian coordinates x = R − R_0 (radial),
    y = R_0(φ − Ω_0 t) (azimuthal, co-rotating frame), z unchanged.
  - Expand rotational velocity to linear order: v_φ ≈ −q Ω_0 x  + u_y
    where q = −d ln Ω/d ln R is the "shear parameter" (q = 3/2 for
    Keplerian disks).
  - Equations look Cartesian but pick up Coriolis + tidal source terms.

Derivation targets:
  1. Linearised rotational velocity  v_φ^{bg} = −q Ω_0 x.
  2. Momentum-equation source terms in the shearing frame:
        − 2 Ω × u  (Coriolis)
        + 2 q Ω² x ê_x  (tidal force balancing Coriolis on bg shear)
  3. Shearing-periodic BC:
        x = L_x:  (same field as) x = 0  with  y shifted by
        Δy(t) = q Ω_0 L_x t  mod L_y.
     Verify that after one "shear time" the BC becomes exactly periodic
     again (t such that Δy(t) = L_y).
  4. Energy-dissipation rate in the shearing frame equals the Maxwell
     + Reynolds stress times the shear rate — the α_SS story (covered
     in D3).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp                                # noqa: E402
from _common import LatexDump, assert_zero, banner    # noqa: E402


def main():
    ld = LatexDump(__file__)
    banner("D2 — Shearing-sheet approximation and shearing-periodic BC")

    x, y, z, t = sp.symbols("x y z t", real=True)
    Omega_0, q, R_0 = sp.symbols("Omega_0 q R_0", positive=True)

    # ════════════════════════════════════════════════════════════
    # 1. Background shear profile
    # ════════════════════════════════════════════════════════════
    # Ω(R) = Ω_0 (R_0/R)^q, so d ln Ω / d ln R = −q.
    # Linearise around R_0:  v_φ^{bg}(x) = R Ω(R) − R_0 Ω_0 ≈ −q Ω_0 x.
    v_phi_bg = -q * Omega_0 * x
    ld.add(
        "Shearing-sheet background (linearised)",
        r"v_\phi^{\text{bg}}(x) = -q\,\Omega_0\,x,\qquad "
        r"q = -\frac{d\ln\Omega}{d\ln R}\ \ "
        r"(q = 3/2\ \text{Keplerian},\ q = 2\ \text{rigid body})",
        label="eq:D2_shear",
    )

    # ════════════════════════════════════════════════════════════
    # 2. Frame-rotation source terms
    # ════════════════════════════════════════════════════════════
    # In a frame rotating at Ω_0 ẑ, the momentum equation picks up:
    #   F_Coriolis = −2 Ω × u = −2 Ω_0 ẑ × u = 2 Ω_0 (u_y ê_x − u_x ê_y)
    #   F_tidal    = 2 q Ω_0² x ê_x  (Hill expansion of gravity + centrif.)
    # The tidal term is the gradient of the "effective potential"
    # Φ_eff = −q Ω_0² x².
    ld.add(
        "Coriolis force in shearing frame",
        r"\mathbf{F}_{\mathrm{Cor}} = -2\,\Omega_0\,\hat{z}\times\mathbf{u}"
        r" = 2\,\Omega_0\,(u_y\,\hat{x} - u_x\,\hat{y})",
        label="eq:D2_coriolis",
    )
    ld.add(
        "Tidal force (from Hill expansion)",
        r"\mathbf{F}_{\mathrm{tidal}} = 2\,q\,\Omega_0^{2}\,x\,\hat{x}"
        r" = -\nabla\Phi_{\mathrm{eff}},\quad "
        r"\Phi_{\mathrm{eff}} = -q\,\Omega_0^{2}\,x^{2}",
        label="eq:D2_tidal",
    )

    # Verify that the background shear v_φ^{bg} = −q Ω_0 x is a steady
    # solution of (Coriolis + tidal):
    # In steady state, u ≡ (0, v_φ^{bg} − 0, 0) = (0, −q Ω_0 x, 0).
    # Net radial force on this background:
    #   F_x = 2 Ω_0 u_y + 2 q Ω_0² x = 2 Ω_0 (−q Ω_0 x) + 2 q Ω_0² x = 0 ✓
    u_bg = sp.Matrix([0, v_phi_bg, 0])
    F_Cor_x = 2 * Omega_0 * u_bg[1]
    F_tid_x = 2 * q * Omega_0**2 * x
    F_net_x = sp.simplify(F_Cor_x + F_tid_x)
    assert_zero(F_net_x,
                "Shearing-sheet background is exact steady state "
                "(Coriolis balances tidal force on v_φ^bg = −q Ω x)")

    # ════════════════════════════════════════════════════════════
    # 3. Shearing-periodic BC
    # ════════════════════════════════════════════════════════════
    # At the x-boundaries, the azimuthal coordinate of matching points
    # shifts in time:
    #   field(x = L_x, y, z, t) = field(x = 0, y − Δy(t), z, t)
    # with Δy(t) = q Ω_0 L_x t  mod L_y.
    L_x, L_y = sp.symbols("L_x L_y", positive=True)
    Delta_y = q * Omega_0 * L_x * t

    ld.add(
        "Shearing-periodic BC",
        r"\text{field}(x = L_x, y, z, t) = "
        r"\text{field}(x = 0,\ y - q\Omega_0 L_x t,\ z, t)",
        label="eq:D2_sheared_periodic",
    )

    # Shear wrap-around time:  t_shear = L_y / (q Ω_0 L_x)
    # After this time the sheared BC becomes pure periodic again.
    t_shear_from_Lx = sp.solve(Delta_y - L_y, t)[0]
    t_shear_expected = L_y / (q * Omega_0 * L_x)
    assert_zero(sp.simplify(t_shear_from_Lx - t_shear_expected),
                "Shear wrap-around time t_shear = L_y / (q Ω_0 L_x)")

    ld.add(
        "Shear wrap-around time (BC is pure periodic again)",
        r"t_{\mathrm{shear}} = \frac{L_y}{q\,\Omega_0\,L_x}",
        label="eq:D2_tshear",
    )

    # ════════════════════════════════════════════════════════════
    # 4. Effective potential in the shearing frame
    # ════════════════════════════════════════════════════════════
    # Some implementations absorb the tidal force into an "effective
    # gravity":
    #   g_eff = −∇Φ_eff = (2 q Ω² x, 0, 0) (above)
    # Note this is UNSTABLE (∂_x Φ_eff has NEGATIVE 2nd derivative),
    # which is why the shearing sheet needs the Coriolis force to
    # stabilise rotation;  without Coriolis, a rotating disk would
    # fly apart.
    Phi_eff = -q * Omega_0**2 * x**2
    d2Phi = sp.diff(Phi_eff, x, 2)
    # d²Φ_eff/dx² = −2 q Ω² < 0  (potential is a saddle).
    sign_check = sp.simplify(d2Phi + 2 * q * Omega_0**2)
    assert_zero(sign_check,
                "Effective potential is anti-confining: "
                "d²Φ_eff/dx² = −2qΩ² < 0")

    ld.write()
    print()
    print("All D2 identities verified.")


if __name__ == "__main__":
    main()
