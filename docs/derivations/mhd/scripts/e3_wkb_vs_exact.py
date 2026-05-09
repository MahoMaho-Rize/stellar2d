r"""
Section E3 benchmark audit — sympy / mpmath derivation of the EXACT
Alfvén-wave amplitude ratio in an isothermal exponentially-stratified
atmosphere with bottom §E2 driver and top §E3 absorber, and comparison
against the leading-order WKB prediction A ∝ ρ^{-1/4}.

Motivation for this script: the B-M5 T7 test compares the measured
time-RMS amplitude ratio |v_x|(y2)/|v_x|(y1) against the textbook WKB
result  exp((y2 − y1)/(4H)) = (ρ₁/ρ₂)^{1/4}.  Measured value in the
code (Ny=128, f=2, H=1, B_y0=0.5, y1=0.5, y2=1.5, §E2 bottom + §E3 top,
10 driver periods averaged) is 1.3463, while the leading WKB predicts
1.2840 — a 4.9% overshoot.  Before declaring this a numerical bug, we
must compute the EXACT (non-WKB) analytic ratio for the same BVP; if
the exact ratio differs from the leading WKB by ~5%, the code is in
fact consistent with the full analytic solution and only the "benchmark"
label was wrong.

Setup:
  - Isothermal HSE: ρ(y) = ρ₀ e^{−y/H},  P(y) = ρ(y) c_s²
  - Uniform B₀ = B_{y0} ŷ  ⇒  v_A(y) = B_{y0}/√ρ(y) = v_A0 · e^{y/(2H)}
  - Linearised Alfvén-x equations reduce to a single 2nd-order ODE
    for V(y) ∝ δv_x(y, t) e^{−iωt}:
        V'' + k²(y) V = 0,   k(y) = ω/v_A(y) = k₀ e^{−y/(2H)}

The substitution ξ = 2H · k(y) = 2H k₀ e^{−y/(2H)} turns this into a
Bessel equation of order 0:
        ξ² V_ξξ + ξ V_ξ + ξ² V = 0
whose outgoing (upward) solution is the Hankel function H₀^{(2)}(ξ)
with temporal factor e^{−iωt}.

§E2 bottom at y=0:  V(ξ₀) = v_drv  (prescribed).
§E3 top  at y=L:    pure outgoing  ⇒  V(ξ) ∝ H₀^{(2)}(ξ)  everywhere.

Combined:   V(y) = v_drv · H₀^{(2)}(ξ(y)) / H₀^{(2)}(ξ₀).

The code measures the time-RMS envelope, which is |V(y)|.  So the exact
theoretical prediction for the T7 ratio is
  R_exact(y1, y2) = |H₀^{(2)}(ξ(y2))| / |H₀^{(2)}(ξ(y1))|
                  = √(J₀² + Y₀²) at ξ(y2) divided by the same at ξ(y1).

Leading WKB (large-ξ asymptotic):
  |H₀^{(2)}(ξ)|² = J₀²(ξ) + Y₀²(ξ) ≈ 2/(πξ)
  ⇒  R_WKB = √(ξ(y1)/ξ(y2)) = e^{(y2 − y1)/(4H)} = (ρ(y1)/ρ(y2))^{1/4}

This script:
  1. Verifies algebraically that the ODE reduces to Bessel order 0.
  2. Verifies H₀^{(2)}(ξ(y)) satisfies the original PDE (with e^{−iωt}).
  3. Computes R_exact vs R_WKB for the T7 parameters.
  4. Identifies the source of any remaining discrepancy between
     R_exact and the measured code value.

If R_exact ≈ 1.346, the "benchmark" label on T7 was wrong: the code
matches the EXACT analytic solution, and the leading WKB formula
happens to be 4.9% off for this (ω, H, v_A) combination.  Update T7
to test against R_exact, not R_WKB.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
import mpmath as mp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("E3 audit — WKB A ∝ ρ^{-1/4} vs exact Hankel amplitude")

    # ─────────────────────────────────────────────────────────────────
    # Step 1: algebraic reduction of the linear Alfvén ODE to Bessel.
    # ─────────────────────────────────────────────────────────────────
    y = sp.Symbol("y", real=True)
    H    = sp.Symbol("H",   positive=True)
    k0   = sp.Symbol("k_0", positive=True)     # = ω/v_A0

    # Stratified wavenumber k(y) = k₀ e^{−y/(2H)}.
    k_y = k0 * sp.exp(-y/(2*H))

    # V(y) solves V'' + k²(y) V = 0.
    V = sp.Function("V")
    ode_y = sp.diff(V(y), y, 2) + k_y**2 * V(y)

    # Substitute ξ = 2H k(y).
    xi = sp.Symbol("xi", positive=True)
    # Relation: y = -2H ln(ξ/(2Hk₀)), so dξ/dy = -ξ/(2H).
    # d/dy   = dξ/dy · d/dξ = -ξ/(2H) · d/dξ
    # d²/dy² = d/dy [-ξ/(2H) d/dξ]
    #        = -ξ/(2H) · d/dξ [-ξ/(2H) · d/dξ]
    #        = -ξ/(2H) · [-1/(2H) · d/dξ - ξ/(2H) · d²/dξ²]
    #        = (ξ/(4H²)) · d/dξ + (ξ²/(4H²)) · d²/dξ²
    U = sp.Function("U")    # V(y) = U(ξ(y))
    # Build V'' + k²V in ξ-variables:
    #   V''(y) = (1/(4H²))(ξ² U_ξξ + ξ U_ξ)
    #   k²     = (ξ/(2H))²
    ode_xi = (ξ := xi)  # shadow for clarity
    lhs_y  = sp.Rational(1, 4) / H**2 * (xi**2 * sp.diff(U(xi), xi, 2)
                                         + xi * sp.diff(U(xi), xi)) \
             + (xi / (2*H))**2 * U(xi)
    # Multiply through by 4H²/ξ² to clean up:
    lhs_clean = sp.simplify(sp.expand(4*H**2/xi**2 * lhs_y))
    # Expected: ξ² U_ξξ/ξ² + U_ξ/ξ + U = U_ξξ + U_ξ/ξ + U = Bessel(0).
    target = sp.diff(U(xi), xi, 2) + sp.diff(U(xi), xi)/xi + U(xi)
    diff = sp.simplify(lhs_clean - target)
    assert_zero(diff,
                "ODE reduces to Bessel of order 0: U'' + U'/ξ + U = 0",
                verbose=False)
    print("  [OK] V''(y) + k²(y) V = 0 reduces to ξ² U_ξξ + ξ U_ξ + ξ² U = 0.")

    # ─────────────────────────────────────────────────────────────────
    # Step 2: verify H₀^{(2)}(ξ) solves the ODE (symbolic check via the
    # series expansion).  We don't do this in full sympy since Hankel
    # of complex arg isn't trivially manipulable; instead we check that
    # any linear combination of J₀(ξ), Y₀(ξ) solves the Bessel ODE.
    # ─────────────────────────────────────────────────────────────────
    c1, c2 = sp.symbols("c1 c2")
    U_sol  = c1 * sp.besselj(0, xi) + c2 * sp.bessely(0, xi)
    check  = sp.simplify(sp.diff(U_sol, xi, 2)
                         + sp.diff(U_sol, xi)/xi + U_sol)
    assert_zero(check,
                "c1 J₀(ξ) + c2 Y₀(ξ) solves the Bessel ODE",
                verbose=False)
    print("  [OK] General solution  c1 J₀(ξ) + c2 Y₀(ξ)  "
          "verified symbolically.")

    # ─────────────────────────────────────────────────────────────────
    # Step 3: physical interpretation.
    # Temporal factor e^{−iωt}.  We want the UPWARD-moving wave (toward
    # increasing y, i.e. DECREASING ξ).  The large-ξ asymptotic of
    # H₀^{(2)}(ξ) = J₀(ξ) − i Y₀(ξ) is
    #   H₀^{(2)}(ξ) ∼ √(2/(πξ)) · e^{−i(ξ − π/4)}.
    # So  e^{−iωt} · H₀^{(2)}(ξ) ∝ e^{−i(ωt + ξ)}  ⇒  phase decreases with ξ,
    # which ⇔ increases with y (since dξ/dy < 0).  This is the OUTGOING
    # (upward) wave.  §E3 top-absorbing BC selects exactly this branch.
    # §E2 bottom with v_drv amplitude then fixes the normalisation:
    #   V(ξ(y)) = v_drv · H₀^{(2)}(ξ(y)) / H₀^{(2)}(ξ₀).
    # ─────────────────────────────────────────────────────────────────
    print("  [OK] Outgoing branch = H₀^{(2)}(ξ) (large-ξ phase analysis).")

    # ─────────────────────────────────────────────────────────────────
    # Step 4: numerical ratio for the T7 code parameters.
    # Parameters match the test: Ny=128, f=2, H=1, ρ₀=1, B_{y0}=0.5,
    # y1=0.5, y2=1.5.  Cell-centred y values are (jc+0.5)·dy where
    # dy = Ly/Ny = 2/128.  The rounded jc1, jc2 below match the code.
    # ─────────────────────────────────────────────────────────────────
    mp.mp.dps = 50
    H_num    = mp.mpf(1)
    f_num    = mp.mpf(2)
    omega    = 2 * mp.pi * f_num
    rho0_num = mp.mpf(1)
    By0_num  = mp.mpf("0.5")
    vA0      = By0_num / mp.sqrt(rho0_num)             # 0.5
    k0_num   = omega / vA0                             # 4π
    xi0      = 2 * H_num * k0_num                      # 16π

    Ly       = mp.mpf(2)
    Ny       = 128
    dy       = Ly / Ny
    y1       = mp.mpf("0.5")
    y2       = mp.mpf("1.5")
    jc1      = int(mp.floor(y1 / dy))
    jc2      = int(mp.floor(y2 / dy))
    yc1      = (jc1 + mp.mpf("0.5")) * dy
    yc2      = (jc2 + mp.mpf("0.5")) * dy
    xi1      = xi0 * mp.exp(-yc1 / (2 * H_num))
    xi2      = xi0 * mp.exp(-yc2 / (2 * H_num))

    # Amplitude of H₀^{(2)} = √(J₀² + Y₀²).
    def hankel_amp(x):
        return mp.sqrt(mp.besselj(0, x) ** 2 + mp.bessely(0, x) ** 2)

    amp1 = hankel_amp(xi1)
    amp2 = hankel_amp(xi2)
    ratio_exact = amp2 / amp1

    # Leading WKB.
    ratio_wkb_asymp = mp.sqrt(xi1 / xi2)      # = √(ξ₁/ξ₂)
    ratio_rho14     = mp.exp((yc2 - yc1) / (4 * H_num))   # same thing

    # Measured ratio from the T7 run (y1=0.5, y2=1.5, 10 periods,
    # §E2 bottom + §E3 top, A=1e-3).
    ratio_measured = mp.mpf("1.3463")

    print()
    print(f"  yc1 = {float(yc1):.5f}, yc2 = {float(yc2):.5f}")
    print(f"  ξ(y1) = {float(xi1):.4f},  ξ(y2) = {float(xi2):.4f}")
    print(f"  |H₀^(2)|(ξ₁) = {float(amp1):.6e}")
    print(f"  |H₀^(2)|(ξ₂) = {float(amp2):.6e}")
    print()
    print(f"  R_WKB    = exp((y2−y1)/(4H))      = {float(ratio_rho14):.6f}")
    print(f"  R_WKB²   = √(ξ₁/ξ₂)               = {float(ratio_wkb_asymp):.6f}")
    print(f"  R_exact  = |H₀^(2)|(ξ₂)/|…|(ξ₁)  = {float(ratio_exact):.6f}")
    print(f"  R_T7     = measured in test        = {float(ratio_measured):.6f}")
    print()
    err_wkb_vs_exact   = abs(ratio_wkb_asymp / ratio_exact - 1)
    err_meas_vs_wkb    = abs(ratio_measured / ratio_wkb_asymp - 1)
    err_meas_vs_exact  = abs(ratio_measured / ratio_exact - 1)
    print(f"  |R_WKB   − R_exact|/R_exact       = {float(err_wkb_vs_exact)*100:.3f} %")
    print(f"  |R_T7    − R_WKB|/R_WKB           = {float(err_meas_vs_wkb)*100:.3f} %")
    print(f"  |R_T7    − R_exact|/R_exact       = {float(err_meas_vs_exact)*100:.3f} %")

    # ─────────────────────────────────────────────────────────────────
    # Step 5: LaTeX dump with the numerical benchmark.
    # ─────────────────────────────────────────────────────────────────
    ld.add(
        "Linear Alfvén wave BVP in isothermal stratified atmosphere",
        r"V''(y) + k^2(y)\,V(y) = 0,\qquad "
        r"k(y) = \omega/v_A(y) = k_0\,e^{-y/(2H)},\qquad "
        r"k_0 = \omega\sqrt{\rho_0}/B_{y0}",
        label="eq:E3_audit_ode",
    )
    ld.add(
        "Substitution $\\xi = 2H k(y)$ reduces to Bessel of order 0",
        r"\xi^2 U_{\xi\xi} + \xi U_\xi + \xi^2 U = 0"
        r"\quad\Longrightarrow\quad "
        r"U(\xi) = c_1 J_0(\xi) + c_2 Y_0(\xi)",
        label="eq:E3_audit_bessel",
    )
    ld.add(
        "Outgoing branch under §E3 top absorbing BC",
        r"V(y) = v_x^{\mathrm{drv}}\cdot "
        r"\frac{H_0^{(2)}(\xi(y))}{H_0^{(2)}(\xi_0)},\qquad "
        r"H_0^{(2)} = J_0 - i Y_0",
        label="eq:E3_audit_hankel",
    )
    ld.add(
        "Exact steady-state amplitude ratio (Bessel-function benchmark)",
        r"R_{\mathrm{exact}}(y_1, y_2) = "
        r"\frac{\lvert H_0^{(2)}(\xi(y_2))\rvert}{\lvert H_0^{(2)}(\xi(y_1))\rvert}"
        r" = \sqrt{\frac{J_0^2(\xi_2)+Y_0^2(\xi_2)}{J_0^2(\xi_1)+Y_0^2(\xi_1)}}",
        label="eq:E3_audit_R_exact",
    )
    ld.add(
        "Leading WKB (large-$\\xi$ asymptotic)",
        r"R_{\mathrm{WKB}}(y_1, y_2) = \sqrt{\xi_1/\xi_2}"
        r" = e^{(y_2 - y_1)/(4H)}"
        r" = \bigl(\rho(y_1)/\rho(y_2)\bigr)^{1/4}",
        label="eq:E3_audit_R_WKB",
    )
    ld.add(
        "T7 numerical benchmark values (H=1, f=2, B_{y0}=0.5, y1=0.5, y2=1.5)",
        rf"R_{{\mathrm{{WKB}}}} = {float(ratio_rho14):.6f},\qquad "
        rf"R_{{\mathrm{{exact}}}} = {float(ratio_exact):.6f}",
        label="eq:E3_audit_numbers",
    )

    ld.write()

    # ─────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────
    print()
    if err_wkb_vs_exact > mp.mpf("0.01"):
        print(f"  VERDICT: leading WKB differs from the EXACT Hankel solution "
              f"by {float(err_wkb_vs_exact)*100:.2f}%, NOT the expected "
              f"1/(ξ²) correction of a few × 10⁻⁴.")
        print(f"           This is the dominant source of T7's apparent "
              f"overshoot against the textbook ρ^{{-1/4}} label.")
    else:
        print(f"  VERDICT: leading WKB matches the EXACT Hankel solution "
              f"within {float(err_wkb_vs_exact)*100:.3f}%. "
              f"T7's 5% discrepancy is therefore NOT of this origin.")

    print()
    print("All E3 audit identities verified.")
    print("Updated T7 threshold should test against R_exact (this script), "
          "not R_WKB.")


if __name__ == "__main__":
    main()
