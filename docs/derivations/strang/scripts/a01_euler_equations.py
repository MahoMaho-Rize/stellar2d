r"""
Section A1 — Compressible Euler equations (strong-form conservation laws).

Starting assumptions:
  (A1a) Compressible neutral gas; no viscosity, no conduction, no
        radiation, no external body-force except optional uniform
        gravity g in the -y direction (gravity is treated as a source
        in C1, not as part of the flux here).
  (A1b) Ideal EOS: p = (gamma - 1) * rho * e_int, where e_int is the
        specific internal energy.
  (A1c) Smooth solutions; strong-form identities are taken pointwise
        in (x, y, t).

Derivation targets (each verified by sympy):

  1. Mass conservation:
        partial_t rho + div(rho v) = 0
     Used to derive the material derivative of any specific quantity.

  2. Momentum conservation (divergence form, 2D):
        partial_t (rho u) + partial_x (rho u^2 + p) + partial_y (rho u v) = 0
        partial_t (rho v) + partial_x (rho u v)     + partial_y (rho v^2 + p) = 0
     These are the rows 1 and 2 of the flux Matrix used everywhere
     downstream.

  3. Total-energy conservation:
        partial_t E + partial_x [(E + p) u] + partial_y [(E + p) v] = 0
        E = p / (gamma - 1) + rho * (u^2 + v^2) / 2

  4. Primary identity that glues momentum and energy together:
        (E + p) v = rho (h + (u^2+v^2)/2) v
     where h = e_int + p / rho = gamma p / ((gamma-1) rho) is the
     specific enthalpy.  This is the form Athena++ and every Godunov
     code uses for the energy flux.

  5. Momentum material-derivative form:
        rho D_t u = -partial_x p
        rho D_t v = -partial_y p
     where D_t = partial_t + u partial_x + v partial_y.  This is
     derived from (1) and (2) by expanding the conservative momentum
     equation and subtracting u * continuity.

  6. Internal-energy equation:
        rho D_t e_int = -p * (partial_x u + partial_y v)
     derived from (1), (2), (3); corresponds to the first law of
     thermodynamics along a particle trajectory.  Important because
     this is the form that motivates the Riemann-solver star-region
     pressure algebra in A7/A8.

Every identity is strong-form, pointwise.  No test-function
integration appears.

Code anchors (for the markdown section):
  src/gpu/explicit/strang_device.cuh :: d_euler_flux_x
  src/gpu/explicit/strang_device.cuh :: d_euler_flux_y
  src/gpu/explicit/strang_device.cuh :: d_cons2prim
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp

import _common as C
from _common import (
    LatexDump,
    assert_zero,
    banner,
    div_cart_2d,
    gamma,
    t,
    x,
    y,
)


def main() -> int:
    ld = LatexDump(__file__)
    banner("A1 - Compressible Euler equations (strong form)")

    # ════════════════════════════════════════════════════════════
    # 1. Primary fields as smooth functions of (x, y, t).
    #    rho_f, p_f are positive; u_f, v_f are real.
    # ════════════════════════════════════════════════════════════
    rho_f = sp.Function("rho", positive=True)(x, y, t)
    p_f = sp.Function("p", positive=True)(x, y, t)
    u_f = sp.Function("u", real=True)(x, y, t)
    v_f = sp.Function("v", real=True)(x, y, t)

    # Total energy per unit volume: E = p/(gamma-1) + rho*(u^2+v^2)/2.
    KE_f = sp.Rational(1, 2) * rho_f * (u_f**2 + v_f**2)
    e_int_f = p_f / (rho_f * (gamma - 1))           # specific internal energy
    E_f = p_f / (gamma - 1) + KE_f

    # ════════════════════════════════════════════════════════════
    # 2. Mass conservation.
    #    Strong-form residual is simply
    #       partial_t rho + div(rho v) = 0
    #    By construction this is an assumption, not a derived identity;
    #    we store the residual as a sympy object so later sections
    #    can substitute it as zero when proving derived identities.
    # ════════════════════════════════════════════════════════════
    mass_residual = sp.diff(rho_f, t) + div_cart_2d(rho_f * u_f, rho_f * v_f)

    ld.add_equation(
        "Mass conservation (strong form)",
        sp.Derivative(rho_f, t) + sp.Symbol(r"\nabla\!\cdot\!(\rho\mathbf{v})"),
        sp.Integer(0),
        label="eq:A1-mass",
    )
    print("\n[A1.1] Mass residual (sympy symbolic object):")
    print("       ", mass_residual)

    # ════════════════════════════════════════════════════════════
    # 3. Momentum conservation (divergence form).
    # ════════════════════════════════════════════════════════════
    # x-momentum:  partial_t (rho u) + partial_x (rho u^2 + p) + partial_y (rho u v)
    mom_x_residual = (
        sp.diff(rho_f * u_f, t)
        + sp.diff(rho_f * u_f**2 + p_f, x)
        + sp.diff(rho_f * u_f * v_f, y)
    )
    # y-momentum:  partial_t (rho v) + partial_x (rho u v) + partial_y (rho v^2 + p)
    mom_y_residual = (
        sp.diff(rho_f * v_f, t)
        + sp.diff(rho_f * u_f * v_f, x)
        + sp.diff(rho_f * v_f**2 + p_f, y)
    )

    ld.add(
        "x-momentum conservation",
        r"\partial_t(\rho u) + \partial_x(\rho u^{2} + p) + \partial_y(\rho u v) = 0",
        label="eq:A1-mom-x",
    )
    ld.add(
        "y-momentum conservation",
        r"\partial_t(\rho v) + \partial_x(\rho u v) + \partial_y(\rho v^{2} + p) = 0",
        label="eq:A1-mom-y",
    )

    # ════════════════════════════════════════════════════════════
    # 4. Total-energy conservation.
    #    With  E = p/(gamma-1) + rho*(u^2+v^2)/2, the flux is (E+p)*(u, v).
    # ════════════════════════════════════════════════════════════
    energy_residual = (
        sp.diff(E_f, t)
        + sp.diff((E_f + p_f) * u_f, x)
        + sp.diff((E_f + p_f) * v_f, y)
    )

    ld.add(
        "Total-energy density",
        r"E \;=\; \frac{p}{\gamma - 1} \;+\; \tfrac{1}{2}\rho\,(u^{2} + v^{2})",
        label="eq:A1-E",
    )
    ld.add(
        "Total-energy conservation",
        r"\partial_t E + \partial_x\!\left[(E + p)\,u\right] + \partial_y\!\left[(E + p)\,v\right] = 0",
        label="eq:A1-energy",
    )

    # ════════════════════════════════════════════════════════════
    # 5. Energy-flux factorisation: (E + p) v_i = rho * (h + |v|^2/2) * v_i
    #    with h = e_int + p/rho = gamma p / ((gamma-1) rho).
    #
    #    This is the identity every Godunov energy-flux kernel relies
    #    on.  Verify as an algebraic equality independent of time.
    # ════════════════════════════════════════════════════════════
    h_f = e_int_f + p_f / rho_f
    flux_energy_x_form_A = (E_f + p_f) * u_f
    flux_energy_x_form_B = rho_f * (h_f + sp.Rational(1, 2) * (u_f**2 + v_f**2)) * u_f
    assert_zero(
        flux_energy_x_form_A - flux_energy_x_form_B,
        "A1-energy-flux-factorisation-x: (E+p) u  ==  rho (h + |v|^2/2) u",
    )

    flux_energy_y_form_A = (E_f + p_f) * v_f
    flux_energy_y_form_B = rho_f * (h_f + sp.Rational(1, 2) * (u_f**2 + v_f**2)) * v_f
    assert_zero(
        flux_energy_y_form_A - flux_energy_y_form_B,
        "A1-energy-flux-factorisation-y: (E+p) v  ==  rho (h + |v|^2/2) v",
    )

    ld.add(
        "Enthalpy and energy-flux factorisation",
        r"h \;\equiv\; e_{\mathrm{int}} + \frac{p}{\rho} = \frac{\gamma\,p}{(\gamma-1)\,\rho},"
        r"\qquad (E + p)\,v_i \;=\; \rho\!\left(h + \tfrac{1}{2}|\mathbf{v}|^{2}\right) v_i",
        label="eq:A1-enthalpy-flux",
    )

    # ════════════════════════════════════════════════════════════
    # 6. Momentum material-derivative form  rho D_t u  =  -partial_x p.
    #
    #    Derivation: expand the conservative x-momentum residual
    #        partial_t(rho u) + partial_x(rho u^2) + partial_y(rho u v) + partial_x p = 0
    #    and subtract u * (mass residual).  The result must simplify
    #    to   rho (partial_t u + u partial_x u + v partial_y u) + partial_x p.
    # ════════════════════════════════════════════════════════════
    Dt_u = sp.diff(u_f, t) + u_f * sp.diff(u_f, x) + v_f * sp.diff(u_f, y)
    Dt_v = sp.diff(v_f, t) + u_f * sp.diff(v_f, x) + v_f * sp.diff(v_f, y)

    # Conservative x-momentum residual - u * (mass residual)
    # should equal rho * D_t u + partial_x p (strong form).
    cons_minus_u_mass = mom_x_residual - u_f * mass_residual
    material_form_x = rho_f * Dt_u + sp.diff(p_f, x)
    assert_zero(
        cons_minus_u_mass - material_form_x,
        "A1-material-x: conservative momentum - u * continuity  ==  rho D_t u + p_x",
    )

    cons_minus_v_mass = mom_y_residual - v_f * mass_residual
    material_form_y = rho_f * Dt_v + sp.diff(p_f, y)
    assert_zero(
        cons_minus_v_mass - material_form_y,
        "A1-material-y: conservative momentum - v * continuity  ==  rho D_t v + p_y",
    )

    ld.add(
        "Momentum, material-derivative (non-conservative) form",
        r"\rho\,D_t u = -\partial_x p, \qquad "
        r"\rho\,D_t v = -\partial_y p, \qquad "
        r"D_t \;\equiv\; \partial_t + u\,\partial_x + v\,\partial_y",
        label="eq:A1-material-mom",
    )

    # ════════════════════════════════════════════════════════════
    # 7. Internal-energy equation  rho D_t e_int  =  -p * div(v).
    #
    #    Derivation: expand the total-energy residual, subtract
    #      u * (rho D_t u + p_x)
    #      v * (rho D_t v + p_y)
    #      [(u^2+v^2)/2] * (mass residual)
    #    The result must simplify to  rho D_t e_int + p * div(v).
    #
    #    This is the pointwise first-law: p * div(v) is the reversible
    #    compression/expansion work.
    # ════════════════════════════════════════════════════════════
    Dt_e_int = (
        sp.diff(e_int_f, t)
        + u_f * sp.diff(e_int_f, x)
        + v_f * sp.diff(e_int_f, y)
    )
    div_v = sp.diff(u_f, x) + sp.diff(v_f, y)

    # Strong-form reduction.
    #
    # Expand the energy residual by product rule:
    #   partial_t E         = e_int M_density_piece + rho D_t e_int
    #                         + (|v|^2/2) M_density_piece
    #                         + rho (u partial_t u + v partial_t v)
    #   partial_i[(E+p) v_i] = e_int div(rho v) + rho v . grad e_int
    #                         + (|v|^2/2) div(rho v) + rho v . grad(|v|^2/2)
    #                         + div(p v)
    # Combine and use D_t := partial_t + v . grad:
    #   energy_res = (e_int + |v|^2/2) M + rho D_t e_int
    #               + rho (u D_t u + v D_t v) + div(p v)
    # Now substitute rho D_t u = mom_x_res - u M - p_x (proved above),
    # similarly for y.  Expanding rho (u D_t u + v D_t v):
    #   = u mom_x_res + v mom_y_res - (u^2 + v^2) M - (u p_x + v p_y)
    # and div(p v) = (u p_x + v p_y) + p div v.  The (u p_x + v p_y)
    # terms cancel, leaving
    #   energy_res = (e_int - |v|^2/2) M + rho D_t e_int
    #               + u mom_x_res + v mom_y_res + p div v.
    # Therefore the reduction that isolates rho D_t e_int + p div v is
    #   energy_res - u mom_x_res - v mom_y_res - (e_int - |v|^2/2) M.
    reduction = (
        energy_residual
        - u_f * mom_x_residual
        - v_f * mom_y_residual
        - (e_int_f - sp.Rational(1, 2) * (u_f**2 + v_f**2)) * mass_residual
    )
    material_form_e = rho_f * Dt_e_int + p_f * div_v
    assert_zero(
        sp.expand(reduction - material_form_e),
        "A1-material-energy: total-energy - u Xmom - v Ymom + KE * mass  ==  rho D_t e_int + p div(v)",
    )

    ld.add(
        "Internal-energy evolution (first law along a particle trajectory)",
        r"\rho\,D_t e_{\mathrm{int}} = -\,p\,\nabla\!\cdot\!\mathbf{v}, \qquad "
        r"e_{\mathrm{int}} = \frac{p}{(\gamma - 1)\,\rho}",
        label="eq:A1-material-e",
    )

    # ════════════════════════════════════════════════════════════
    # 8. Compact conservative system (summary).
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Compact conservative form",
        r"\partial_t \mathbf{U} + \partial_x \mathbf{F}_x(\mathbf{U}) + "
        r"\partial_y \mathbf{F}_y(\mathbf{U}) = \mathbf{0}, \qquad "
        r"\mathbf{U} = (\rho,\,\rho u,\,\rho v,\,E)^{\!\top}",
        label="eq:A1-compact",
    )
    ld.add(
        "x-flux vector",
        r"\mathbf{F}_x(\mathbf{U}) = "
        r"\begin{pmatrix}\rho u\\ \rho u^{2} + p\\ \rho u v\\ (E + p)\,u\end{pmatrix}",
        label="eq:A1-Fx",
    )
    ld.add(
        "y-flux vector",
        r"\mathbf{F}_y(\mathbf{U}) = "
        r"\begin{pmatrix}\rho v\\ \rho u v\\ \rho v^{2} + p\\ (E + p)\,v\end{pmatrix}",
        label="eq:A1-Fy",
    )

    ld.write()
    print()
    print("All A1 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
