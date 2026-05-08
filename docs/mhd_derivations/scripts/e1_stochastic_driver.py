"""
Section E1 — Stochastic broadband photospheric driver.
See sections/e1_stochastic_driver.md for context.

Suzuki+25 / Shimizu+22 inner-boundary driver:
  - Transverse: outgoing Elsässer z^+ = v_⊥ − B_⊥/√(4πρ)  driven as
      z^+(t) = A_⊥ · Σ_N  sin(2π f_N t + φ_N) / √(f_N),
    with f_N log-spaced on [f_min, f_max = 100 f_min], φ_N uniform
    random.  Incoming z^- absorbed by ∂z^-/∂r = 0.
  - Longitudinal: v_r driven similarly, narrow-band (p-mode).
  - Power spectrum: P(ω) ∝ ω^{-1}  (equal power per log-octave).
  - Normalisation: ⟨δv^2⟩ = ∫ P(ω) dω.

Verifies:
  - Power-spectrum normalisation: for P(ω) = A²/ω, ∫_{ω_min}^{ω_max} dω = A² ln(ω_max/ω_min),
    so A² = ⟨δv²⟩ / ln(ω_max/ω_min).
  - Single-sinusoid-sum variance (Parseval under random phases):
    ⟨(Σ_N A_N sin(ω_N t + φ_N))²⟩_{φ,t} = ½ Σ_N A_N².
  - Elsässer absorption: ∂z^-/∂r = 0 at r = R_* with z^- = v_⊥ + B_⊥/√(4πρ)
    gives a reflection coefficient R → 0 in the WKB limit.
  - MHD characteristic form: z^+ propagates with +v_A; z^- with −v_A.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("E1 — Stochastic broadband photospheric driver")

    # ─── symbols ─────────────────────────────────────────────────────
    omega, omega_min, omega_max, t_sym = sp.symbols(
        "omega omega_min omega_max t", positive=True)
    A_rms = sp.Symbol("A_rms", positive=True)      # target ⟨δv²⟩^{1/2}
    # single-sinusoid amplitude and frequency
    A_N = sp.Symbol("A_N", positive=True)
    f_N = sp.Symbol("f_N", positive=True)
    phi_N = sp.Symbol("phi_N", real=True)

    # ─── Identity 1: P(ω) ∝ ω^{-1} normalisation ────────────────────
    # P(ω) = A² / ω,  ∫_{ω_min}^{ω_max} P dω = A² ln(ω_max/ω_min)
    # Require that integral = ⟨δv²⟩ = A_rms²:
    A = sp.Symbol("A", positive=True)
    P = A**2 / omega
    integral = sp.integrate(P, (omega, omega_min, omega_max))
    # = A² ln(ω_max/ω_min)
    # Solve for A:
    expected = A**2 * sp.log(omega_max / omega_min)
    assert_zero(sp.simplify(integral - expected),
                "∫ P(ω) dω = A² ln(ω_max/ω_min) for ω^{-1} spectrum",
                verbose=False)
    print("  [OK] P(ω) ∝ ω^{-1} spectrum integrates to A² ln(ω_max/ω_min).")

    # Normalisation: A² = A_rms² / ln(ω_max/ω_min)
    A_squared_norm = A_rms**2 / sp.log(omega_max / omega_min)
    # Substitute into integral, verify integral equals A_rms².
    integral_normalised = integral.subs(A**2, A_squared_norm)
    assert_zero(sp.simplify(integral_normalised - A_rms**2),
                "normalised amplitude A² = A_rms² / ln(ω_max/ω_min)",
                verbose=False)
    print("  [OK] normalisation reproduces target ⟨δv²⟩ = A_rms².")

    # ─── Identity 2: sinusoid-sum variance (Parseval under phases) ──
    # For x(t) = A sin(ω t + φ), ⟨x²⟩_t = A²/2 (average over one period).
    A_sym = sp.Symbol("Amp", positive=True)
    omega_sym = sp.Symbol("omegasym", positive=True)
    phi_sym = sp.Symbol("phisym", real=True)
    T_period = 2*sp.pi / omega_sym
    x_t = A_sym * sp.sin(omega_sym * t_sym + phi_sym)
    avg_x2 = sp.integrate(x_t**2, (t_sym, 0, T_period)) / T_period
    assert_zero(sp.simplify(avg_x2 - A_sym**2/2),
                "⟨sin²(ω t + φ)⟩_t = 1/2 ⇒ single-sinusoid variance = A²/2",
                verbose=False)
    print("  [OK] single sinusoid variance = A²/2.")

    # ─── Identity 3: cross terms vanish under random phases ─────────
    # For x(t) = A_1 sin(ω_1 t + φ_1) + A_2 sin(ω_2 t + φ_2), ω_1 ≠ ω_2:
    # ⟨x²⟩_t = A_1²/2 + A_2²/2 + 2 A_1 A_2 ⟨sin⟩⟨sin⟩
    # Cross-term integrates to a bounded oscillation / T → 0 for
    # commensurate periods in long-time limit; for incommensurate it
    # is 0 in the ergodic average.  Here we use "period-averaged over
    # a common multiple" for sympy-verifiability: set ω_2 = 3 ω_1.
    A1, A2 = sp.symbols("A1 A2", positive=True)
    om1 = sp.Symbol("om1", positive=True)
    om2 = 3 * om1
    ph1, ph2 = sp.symbols("ph1 ph2", real=True)
    x12 = A1*sp.sin(om1*t_sym + ph1) + A2*sp.sin(om2*t_sym + ph2)
    T_common = 2*sp.pi / om1   # also covers 3 periods of om2
    var12 = sp.integrate(x12**2, (t_sym, 0, T_common)) / T_common
    # Should equal A1²/2 + A2²/2  regardless of phases.
    expected_var = A1**2/2 + A2**2/2
    assert_zero(sp.simplify(var12 - expected_var),
                "two-frequency sum variance = A1²/2 + A2²/2 (cross vanishes)",
                verbose=False)
    print("  [OK] cross-frequency terms vanish in time average.")

    # ─── Identity 4: Elsässer propagation ───────────────────────────
    # Linearised transverse MHD around uniform (ρ_0, B_{r,0}) with v_{r,0} = 0:
    #   ∂_t v_⊥ = (B_{r,0}/ρ_0) ∂_r B_⊥,
    #   ∂_t B_⊥ = B_{r,0} ∂_r v_⊥.
    # Define z^± = v_⊥ ∓ B_⊥/√(4πρ_0).  (Here we drop the 4π in Gaussian
    # convention consistent with _common.py: v_A = B_r / √ρ.)
    # Expect: ∂_t z^± ± v_A ∂_r z^± = 0.
    r_sym = sp.Symbol("r", real=True)
    rho0 = sp.Symbol("rho0", positive=True)
    B_r0 = sp.Symbol("B_r0", positive=True)
    v_A = B_r0 / sp.sqrt(rho0)
    v_perp = sp.Function("vp")(r_sym, t_sym)
    B_perp = sp.Function("Bp")(r_sym, t_sym)
    z_plus = v_perp - B_perp/sp.sqrt(rho0)
    z_minus = v_perp + B_perp/sp.sqrt(rho0)

    # Linearised equations
    dv_perp_dt = (B_r0/rho0) * sp.diff(B_perp, r_sym)
    dB_perp_dt = B_r0 * sp.diff(v_perp, r_sym)

    # Compute ∂_t z^+ and show = −v_A ∂_r z^+
    dt_zp = sp.diff(z_plus, t_sym).subs({sp.Derivative(v_perp, t_sym): dv_perp_dt,
                                          sp.Derivative(B_perp, t_sym): dB_perp_dt})
    # Alternative: build manually.
    dt_zp_direct = dv_perp_dt - dB_perp_dt/sp.sqrt(rho0)
    dr_zp = sp.diff(z_plus, r_sym)
    # ∂_t z^+ + v_A ∂_r z^+ should vanish
    lhs_plus = sp.simplify(dt_zp_direct + v_A * dr_zp)
    assert_zero(lhs_plus, "∂_t z^+ + v_A ∂_r z^+ = 0 (outgoing Alfvén)",
                verbose=False)
    # ∂_t z^- − v_A ∂_r z^- should vanish
    dt_zm_direct = dv_perp_dt + dB_perp_dt/sp.sqrt(rho0)
    dr_zm = sp.diff(z_minus, r_sym)
    lhs_minus = sp.simplify(dt_zm_direct - v_A * dr_zm)
    assert_zero(lhs_minus, "∂_t z^- − v_A ∂_r z^- = 0 (incoming Alfvén)",
                verbose=False)
    print("  [OK] Elsässer variables obey pure one-way advection.")

    # ─── Identity 5: ∂z^-/∂r = 0 BC ⇒ pure outgoing in WKB ─────────
    # For a plane wave z^- = Z_0 exp(i(k r − ω t)),
    # ∂_r z^- = i k Z_0 e^{...} → 0 only if k Z_0 = 0, i.e., Z_0 = 0.
    # So the "zero-gradient" BC enforces no incoming component, i.e.,
    # R = Z_0 / (amplitude of outgoing) = 0.
    k, Z0, Wavep = sp.symbols("k Z0 Wavep", complex=True)
    zm_wave = Z0 * sp.exp(sp.I*(k*r_sym - omega*t_sym))
    grad_zm = sp.diff(zm_wave, r_sym)
    # Set BC: grad_zm = 0 ⇒ Z0 * (ik) exp(...) = 0
    # As exp(...) is nonzero, this enforces Z0 = 0 (for k ≠ 0).
    # sympy: solve grad_zm = 0 for Z0 gives Z0 = 0.
    solution = sp.solve(grad_zm, Z0)
    assert solution == [0], f"expected Z0 = 0, got {solution}"
    print("  [OK] zero-gradient BC on z^- forces incoming amplitude = 0.")

    # ─── LaTeX dump ─────────────────────────────────────────────────
    ld.add(
        "Broadband driver — transverse Elsässer",
        r"z^+_{\perp,\odot}(t) = A_\perp\,\sum_{N=0}^{N_{\max}}"
        r"\frac{\sin(2\pi f_N t + \varphi_N)}{\sqrt{f_N}},"
        r"\quad \varphi_N \sim U[0, 2\pi)",
        label="eq:E1_driver_transverse",
    )
    ld.add(
        "Broadband driver — longitudinal",
        r"\delta v_{\parallel,\odot}(t) = A_\parallel\,\sum_{N=0}^{N_{\max}}"
        r"\frac{\sin(2\pi f_N^\parallel t + \varphi_N^\parallel)}{\sqrt{f_N^\parallel}}",
        label="eq:E1_driver_long",
    )
    ld.add(
        "Power spectrum",
        r"P(\omega) = A^2/\omega,\qquad "
        r"\int_{\omega_\mathrm{min}}^{\omega_\mathrm{max}} P(\omega)\,\mathrm{d}\omega"
        r" = A^2\,\ln(\omega_\mathrm{max}/\omega_\mathrm{min})",
        label="eq:E1_spectrum",
    )
    ld.add(
        "Normalisation to target rms",
        r"A^2 = \frac{\langle\delta v^2\rangle}{\ln(\omega_\mathrm{max}/\omega_\mathrm{min})}"
        r"\ \Longrightarrow\ \int P\,\mathrm{d}\omega = \langle\delta v^2\rangle",
        label="eq:E1_norm",
    )
    ld.add(
        "Parseval / Koopman variance (random phases)",
        r"\bigl\langle(\sum_N A_N\,\sin(\omega_N t + \varphi_N))^2\bigr\rangle_t"
        r" = \tfrac{1}{2}\,\sum_N A_N^2,\quad "
        r"A_N = A/\sqrt{f_N}\ \Rightarrow\ A_N^2 = A^2/f_N",
        label="eq:E1_parseval",
    )
    ld.add(
        "Log-spaced sampling to mimic $\\omega^{-1}$",
        r"f_N = f_\mathrm{min}\,\bigl(f_\mathrm{max}/f_\mathrm{min}\bigr)^{N/N_{\max}},"
        r"\quad f_\mathrm{max} = 100\,f_\mathrm{min}",
        label="eq:E1_logspace",
    )
    ld.add(
        "Incoming-Alfvén absorbing BC",
        r"\partial_r z^-_\perp\bigr|_{r=R_*} = 0\ \Longrightarrow\ "
        r"\text{no reflected amplitude in WKB}",
        label="eq:E1_BC",
    )
    ld.add(
        "Elsässer propagation (linearised around uniform $\\rho_0$)",
        r"\partial_t z^\pm \pm v_A\,\partial_r z^\pm = 0,\qquad "
        r"v_A = B_{r,0}/\sqrt{\rho_0}",
        label="eq:E1_advect",
    )
    ld.add(
        "Why random phases — avoid lock-in",
        r"\{\varphi_N\}\text{ deterministic} \Rightarrow\ "
        r"\text{driver is periodic with period }T_\min = 2\pi/\gcd(\omega_N)"
        r"\Rightarrow\ \text{spurious resonance}",
        label="eq:E1_phases",
    )
    ld.add(
        "Longitudinal narrow band (5 min p-mode)",
        r"f_\mathrm{min}^\parallel \approx 3.33\!\times\!10^{-3}\,\mathrm{Hz},"
        r"\ f_\mathrm{max}^\parallel \approx 10^{-2}\,\mathrm{Hz}",
        label="eq:E1_pmode",
    )
    ld.add(
        "Transverse broadband (Alfvén range)",
        r"f_\mathrm{min}^\perp \approx 10^{-3}\,\mathrm{Hz},"
        r"\ f_\mathrm{max}^\perp \approx 10^{-2}\,\mathrm{Hz}",
        label="eq:E1_alfven_range",
    )

    ld.write()
    print()
    print("All E1 identities verified.")


if __name__ == "__main__":
    main()
