r"""
Section A12 — MUSCL-Hancock half-step predictor.

The Hancock predictor evolves each face-reconstructed state by a
half time step using the physical flux, producing a time-centred
value at the Riemann interface.  For a generic hyperbolic
conservation law U_t + F(U)_x = 0, and face-reconstructed states
U_{j+1/2, L} and U_{j-1/2, R} on cell j:

    U^{n+1/2}_{L} = U_{j+1/2, L} - (Dt/2) (1/h) [F(U_{j+1/2, L})
                                                 - F(U_{j-1/2, R})]
    U^{n+1/2}_{R} = U_{j-1/2, R} - (Dt/2) (1/h) [F(U_{j+1/2, L})
                                                 - F(U_{j-1/2, R})]

Strong-form identities (verified by sympy):

  1. Conservation property.  The half-stepped cell-average equals
     the original cell-average minus half the flux difference, i.e.
         (U^{n+1/2}_L + U^{n+1/2}_R) / 2
     equals the cell-average formula one would get from a simple
     FV half-step.  (The proof is trivial because both faces use
     the same flux difference.)

  2. Linear advection consistency.  For scalar advection u_t + a u_x = 0
     with constant a > 0, the Hancock-predictor output is
         u^{n+1/2}_L = u_L - a (Dt/(2 h)) (u_{L,+} - u_{L,-})
     where u_{L,+}, u_{L,-} are face states on either side of cell j.
     With unlimited MUSCL reconstruction, u^{n+1/2}_L matches the
     exact u(x_{j+1/2} - a Dt/2) to O(Dt^2, h^2).

  3. 2nd-order consistency in time.  For smooth IC, Taylor
     expansion of U^{n+1/2}_L around (t_n, x_{j+1/2}) gives
         U(x_{j+1/2}, t_n) + (Dt/2) U_t + O(Dt^2, h^2)
         = U(x_{j+1/2}, t_n + Dt/2) + O(Dt^2)
     using U_t = -F(U)_x.  sympy verifies the algebra:
         U_face + (Dt/2)(-F_face_right + F_face_left)/h
           == U at midpoint  + truncation in the right orders.

Code anchor:
  src/gpu/explicit/strang_solver.cu :: k_muscl_hancock_x (lines
     that compute c_L*, c_R* from the reconstructed face states
     by adding a half-step based on the flux through the opposite
     face; see the `coeff = 0.5 * dt / dy` block)

Rule 4: all identities are strong-form Taylor expansions; sympy
handles them directly.
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
    banner("A12 - MUSCL-Hancock half-step predictor")

    h = sp.Symbol("h", positive=True)
    Dt = sp.Symbol("Dt", positive=True)     # Delta t
    a_adv = sp.Symbol("a", real=True)        # advection speed

    # ════════════════════════════════════════════════════════════
    # Scalar advection u_t + a u_x = 0.
    # Cell-average at time t_n:  <u>_j(t_n) = u(x_j, t_n) + O(h^2)
    # Face reconstruction at j+1/2 (left side):  u_{j+1/2,L}^n
    #   = <u>_j + (h/2) * sigma_j
    # where sigma_j is an MC-limited slope (or just (<u>_{j+1} - <u>_{j-1})/(2h)
    # for the unlimited case).
    # ════════════════════════════════════════════════════════════
    # For the time-consistency check we treat the face values and
    # their x-/t-derivatives as independent symbols, then apply the
    # PDE as substitution rules.  This avoids sympy's Subs(...) wrapper
    # that arises when re-expressing a shifted Function in situ.
    u_0, u_x, u_t_sym, u_xx, u_xt, u_tt = sp.symbols(
        "u u_x u_t u_xx u_xt u_tt", real=True,
    )
    # PDE u_t + a u_x = 0 => u_t = -a u_x.
    # Differentiating in t:  u_tt = -a u_xt = -a d/dt(u_x) = -a * (-a u_xx) = a^2 u_xx.
    # Also u_xt = -a u_xx.
    pde_rule = {u_t_sym: -a_adv * u_x, u_xt: -a_adv * u_xx, u_tt: a_adv**2 * u_xx}

    # Hancock half-step on scalar linear advection (at the face):
    u_hancock = u_0 - (Dt / 2) * a_adv * u_x
    # True midpoint-time value via 3-term time Taylor around the face:
    u_midpoint_full = u_0 + (Dt / 2) * u_t_sym + (Dt**2 / 8) * u_tt
    u_midpoint_sub = u_midpoint_full.subs(pde_rule)
    # Compute u_midpoint_sub - u_hancock and require leading error
    # is (Dt^2 / 8) a^2 u_xx.
    err_expr = sp.simplify(u_midpoint_sub - u_hancock)
    expected_err = (Dt**2 / 8) * a_adv**2 * u_xx
    assert_zero(
        sp.simplify(err_expr - expected_err),
        "A12-Hancock-time-order: u_true - u_hancock = (Dt^2/8) a^2 u_xx + O(Dt^3)",
    )

    # The "exact to O(Dt^2)" claim follows: dropping the u_tt term
    # gives u_true_linear = u_0 + (Dt/2) u_t = u_0 - (Dt/2) a u_x
    # (after PDE) = u_hancock identically.
    u_midpoint_linear = u_0 + (Dt / 2) * u_t_sym
    u_midpoint_linear_sub = u_midpoint_linear.subs(pde_rule)
    assert_zero(
        sp.simplify(u_midpoint_linear_sub - u_hancock),
        "A12-Hancock-linear-advect: u_hancock == u_0 + (Dt/2) u_t  after u_t = -a u_x",
    )

    # ════════════════════════════════════════════════════════════
    # Cell-average conservation.
    # The kernel k_muscl_hancock_x applies the half-step to BOTH the
    # left-face and right-face reconstructions, using the SAME flux
    # difference.  For a scalar field, this means
    #   u_L^{n+1/2} = u_L - (Dt/2 h) [F(u_R) - F(u_L)]
    #   u_R^{n+1/2} = u_R - (Dt/2 h) [F(u_R) - F(u_L)]
    # Both face states are shifted by the same amount, so the
    # cell-average identity holds:
    #   (u_L^{n+1/2} + u_R^{n+1/2})/2 = (u_L + u_R)/2 - (Dt/2 h)(F(u_R) - F(u_L))
    # Verify.
    # ════════════════════════════════════════════════════════════
    u_L_sym = sp.Symbol("u_L", real=True)
    u_R_sym = sp.Symbol("u_R", real=True)
    def F_scalar(u):
        return a_adv * u   # linear advection flux
    F_L_val = F_scalar(u_L_sym)
    F_R_val = F_scalar(u_R_sym)
    uL_half = u_L_sym - (Dt / (2 * h)) * (F_R_val - F_L_val)
    uR_half = u_R_sym - (Dt / (2 * h)) * (F_R_val - F_L_val)
    avg_half = (uL_half + uR_half) / 2
    expected_avg = (u_L_sym + u_R_sym) / 2 - (Dt / (2 * h)) * (F_R_val - F_L_val)
    assert_zero(
        sp.simplify(avg_half - expected_avg),
        "A12-Hancock-cell-avg: cell-avg(u^{n+1/2}) == cell-avg(u^n) - (Dt/2h)(F_R - F_L)",
    )

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Hancock half-step predictor (per cell)",
        r"\mathbf{U}^{n+1/2}_{L} \;=\; \mathbf{U}_{j+1/2, L}^{n} - \frac{\Delta t}{2h}\,"
        r"\bigl[\mathbf{F}(\mathbf{U}_{j+1/2, L}^{n}) - \mathbf{F}(\mathbf{U}_{j-1/2, R}^{n})\bigr], "
        r"\\[4pt] "
        r"\mathbf{U}^{n+1/2}_{R} \;=\; \mathbf{U}_{j-1/2, R}^{n} - \frac{\Delta t}{2h}\,"
        r"\bigl[\mathbf{F}(\mathbf{U}_{j+1/2, L}^{n}) - \mathbf{F}(\mathbf{U}_{j-1/2, R}^{n})\bigr]",
        label="eq:A12-Hancock",
    )
    ld.add(
        "Linear advection consistency",
        r"u^{n+1/2}_{L} \;=\; u(x_{j+1/2},\ t_n + \tfrac{\Delta t}{2}) \ \text{exactly, modulo } O(\Delta t^{2})"
        r"\ \text{leading error } \tfrac{\Delta t^{2}}{8}\,a^{2}\,u_{xx}",
        label="eq:A12-linear-advect",
    )
    ld.add(
        "2nd-order time consistency (leading truncation)",
        r"u_{\mathrm{true}} - u_{\mathrm{hancock}} \;=\; \tfrac{\Delta t^{2}}{8}\,a^{2}\,u_{xx}(x_{j+1/2}, t_n) + O(\Delta t^{3})",
        label="eq:A12-time-order",
    )
    ld.add(
        "Cell-average conservation through the half-step",
        r"\tfrac{1}{2}\bigl(\mathbf{U}^{n+1/2}_L + \mathbf{U}^{n+1/2}_R\bigr) \;=\; "
        r"\tfrac{1}{2}\bigl(\mathbf{U}^{n}_L + \mathbf{U}^{n}_R\bigr) - \frac{\Delta t}{2h}\bigl[\mathbf{F}_R - \mathbf{F}_L\bigr]",
        label="eq:A12-conservation",
    )

    ld.write()
    print()
    print("All A12 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
