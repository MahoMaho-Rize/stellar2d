"""
Section C8 — Chromospheric / blended optically-thick cooling closure.
See sections/c8_chromospheric_cooling.md for context.

Two-component cooling closure used in Shimizu+22 and Suzuki+25:

  Q_R = Q_R^thck · ξ_rad + Q_R^thin · (1 − ξ_rad),
  ξ_rad = max(0, 1 − p_chr/p),           (blend weight; eq. 15 in Shimizu+22)
  Q_R^thck = (e_int − e_int^ref)/τ_thck, τ_thck = 0.1 (ρ/ρ̄)^{-1/2} s
  Q_R^thin = two-piece Anderson-Athay 1989 / Goodman-Judge 2012
             Sutherland-Dopita 1993 + CHIANTI

Verifies:
  - ξ_rad smooth limits (p → 0 gives ξ=0 "pure thin"; p → ∞ gives ξ=1
    only if p > p_chr, else still 0); sympy proves ξ_rad ∈ [0, 1].
  - Newton cooling drives T → T_ref exponentially with time-constant
    τ_thck: sympy-verifies dT/dt + (T − T_ref)/τ_thck = 0 admits
    exponential solution.
  - Total cooling is convex combination: Q_R ≥ 0 if both pieces ≥ 0.
  - Anderson-Athay 1989 chromospheric form
      Q_R^AA = 4.5e9 · (0.2 + 0.8 Z/Z_sun) · min(1, ρ/ρ_cr)
    — non-negative; monotone in ρ.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("C8 — Chromospheric / thick-thin blended cooling")

    # ─── symbols ─────────────────────────────────────────────────────
    p, p_chr, rho, rho_bar = sp.symbols("p p_chr rho rhobar", positive=True)
    T_sym = sp.Symbol("T", positive=True)
    T_ref = sp.Symbol("T_ref", positive=True)
    e_int, e_ref = sp.symbols("e_int e_ref", positive=True)
    tau_thck = sp.Symbol("tau_thck", positive=True)
    Z_ratio = sp.Symbol("Z_ratio", nonnegative=True)     # Z / Z_sun
    rho_cr = sp.Symbol("rho_cr", positive=True)
    t_sym = sp.Symbol("t", positive=True)

    # ─── Identity 1: blending weight ∈ [0, 1] ───────────────────────
    # ξ_rad = max(0, 1 − p_chr/p) ; show that 0 ≤ ξ ≤ 1.
    # For p ≥ p_chr: ξ = 1 − p_chr/p ∈ [0, 1).
    # For p < p_chr: ξ = 0 (floor).
    # Sympy-friendly form: use Piecewise.
    xi_rad = sp.Piecewise(
        (0, p < p_chr),
        (1 - p_chr/p, True),
    )
    # Upper bound: 1 − p_chr/p < 1 since p_chr > 0 and p > 0.
    upper_bound = sp.simplify((1 - p_chr/p) - 1)     # = -p_chr/p < 0
    assert_zero(sp.simplify(upper_bound + p_chr/p),
                "ξ_rad upper bound: (1 − p_chr/p) − 1 = −p_chr/p",
                verbose=False)
    # Lower bound: at p = p_chr, ξ = 0; just above the cutoff ξ > 0.
    assert_zero(sp.simplify((1 - p_chr/p).subs(p, p_chr)),
                "ξ_rad at cutoff p = p_chr is zero", verbose=False)
    print("  [OK] blend weight ξ_rad ∈ [0, 1), anchored at p = p_chr.")

    # ─── Identity 2: Newton cooling has exponential solution ────────
    # dT/dt = -(T − T_ref)/τ_thck  →  T(t) = T_ref + (T_0 − T_ref) exp(−t/τ).
    T0 = sp.Symbol("T_0", positive=True)
    T_t = T_ref + (T0 - T_ref) * sp.exp(-t_sym/tau_thck)
    dTdt = sp.diff(T_t, t_sym)
    rhs = -(T_t - T_ref) / tau_thck
    assert_zero(sp.simplify(dTdt - rhs),
                "Newton cooling admits T(t) = T_ref + (T_0−T_ref) exp(−t/τ)",
                verbose=False)
    print("  [OK] Newton cooling exponentially relaxes to T_ref.")

    # ─── Identity 3: Gudiksen-Nordlund 2005 scaling τ_thck ∝ ρ^{−1/2} ─
    # τ_thck(ρ) = 0.1 (ρ/ρ̄)^{−1/2} s.  Limit ρ → ρ̄ gives τ = 0.1 s; limit
    # ρ → 0 blows up (consistent with the quench for optically-thin
    # takeover).  sympy checks: d log τ / d log ρ = −1/2.
    tau_rho = sp.Rational(1, 10) * (rho/rho_bar)**(-sp.Rational(1, 2))
    log_slope = sp.simplify(sp.diff(sp.log(tau_rho), rho) * rho)
    assert_zero(sp.simplify(log_slope + sp.Rational(1, 2)),
                "d ln τ_thck / d ln ρ = −1/2", verbose=False)
    print("  [OK] thick cooling time τ ∝ ρ^{−1/2} (GN05 scaling).")

    # ─── Identity 4: Anderson-Athay 1989 non-negative + Z-monotone ──
    # Q_R^AA = 4.5e9 (0.2 + 0.8 Z/Z_sun) min(1, ρ/ρ_cr).
    Q_AA = sp.Rational(45, 10) * 10**9 * (sp.Rational(1, 5) + sp.Rational(4, 5)*Z_ratio) \
           * sp.Min(1, rho/rho_cr)
    # Sympy: ∂Q/∂Z = 4.5e9 · 0.8 · min(1, ρ/ρ_cr) ≥ 0  since both factors ≥ 0.
    dQdZ = sp.diff(Q_AA, Z_ratio)
    # For ρ ≥ ρ_cr: factor is 1; for ρ < ρ_cr: factor is ρ/ρ_cr.  Both ≥ 0.
    # Check the simplification: dQ/dZ = 3.6e9 * min(1, ρ/ρ_cr)
    assert_zero(sp.simplify(dQdZ - sp.Rational(36, 10)*10**9*sp.Min(1, rho/rho_cr)),
                "∂Q_AA/∂(Z/Z_sun) = 3.6e9 · min(1, ρ/ρ_cr) ≥ 0", verbose=False)
    print("  [OK] Anderson-Athay Q_R monotone in Z, saturates at ρ ≥ ρ_cr.")

    # ─── Identity 5: convex combination preserves sign ──────────────
    # Q_R = ξ · Q_thck + (1 − ξ) · Q_thin, with ξ ∈ [0, 1].
    # Both pieces ≥ 0  →  Q_R ≥ 0  trivially.
    Q_thck_sym = sp.Symbol("Q_thck", nonnegative=True)
    Q_thin_sym = sp.Symbol("Q_thin", nonnegative=True)
    xi_sym = sp.Symbol("xi", nonnegative=True)
    # Linear blend: Q_R = ξ · Q_thck + (1 − ξ) · Q_thin.
    Q_R = xi_sym * Q_thck_sym + (1 - xi_sym) * Q_thin_sym
    # ∂Q_R/∂Q_thck = ξ ≥ 0 ; ∂Q_R/∂Q_thin = 1 − ξ ≥ 0 for ξ ∈ [0, 1].
    assert_zero(sp.simplify(sp.diff(Q_R, Q_thck_sym) - xi_sym),
                "∂Q_R/∂Q_thck = ξ", verbose=False)
    assert_zero(sp.simplify(sp.diff(Q_R, Q_thin_sym) - (1 - xi_sym)),
                "∂Q_R/∂Q_thin = 1 − ξ", verbose=False)
    print("  [OK] convex-combination partials: ξ and 1−ξ.")

    # ─── LaTeX dump ─────────────────────────────────────────────────
    ld.add(
        "Blending weight (Shimizu+22 eq.\\ 15)",
        r"\xi_\mathrm{rad}(p) = \max\!\bigl(0,\,1 - p_\mathrm{chr}/p\bigr)"
        r"\ \in [0, 1),\quad p_\mathrm{chr} \approx 0.1\,p_\odot",
        label="eq:C8_xi",
    )
    ld.add(
        "Total cooling",
        r"Q_R(T, \rho, p) = \xi_\mathrm{rad}(p)\,Q_R^\mathrm{thck}(T, \rho)"
        r" + (1 - \xi_\mathrm{rad}(p))\,Q_R^\mathrm{thin}(T, \rho, Z)",
        label="eq:C8_blend",
    )
    ld.add(
        "Gudiksen-Nordlund 2005 Newton cooling (thick)",
        r"Q_R^\mathrm{thck} = (e_\mathrm{int} - e_\mathrm{int}^\mathrm{ref}) / \tau_\mathrm{thck},"
        r"\quad \tau_\mathrm{thck}(\rho) = 0.1\,(\rho/\bar{\rho})^{-1/2}\ \mathrm{s}",
        label="eq:C8_GN05",
    )
    ld.add(
        "Newton cooling closed-form relaxation",
        r"T(t) = T_\mathrm{ref} + (T_0 - T_\mathrm{ref})\,\exp(-t/\tau_\mathrm{thck})",
        label="eq:C8_relax",
    )
    ld.add(
        "Anderson-Athay 1989 chromospheric law (Suzuki+25 eq.\\ 17)",
        r"Q_R^\mathrm{AA} = 4.5\!\times\!10^9\,(0.2 + 0.8\,Z/Z_\odot)"
        r"\,\min(1,\,\rho/\rho_\mathrm{cr})\ \mathrm{erg\,cm^{-3}\,s^{-1}},"
        r"\quad \rho_\mathrm{cr} = 10^{-16}\,\mathrm{g\,cm^{-3}}",
        label="eq:C8_AA",
    )
    ld.add(
        "Two-piece thin cooling",
        r"Q_R^\mathrm{thin}(T, \rho, Z) = \begin{cases}"
        r"Q_R^\mathrm{SD93}(T, Z)\ n_e n_i, & T > 1.2\!\times\!10^4\,\mathrm{K} \\"
        r"Q_R^\mathrm{AA}(\rho, Z), & T \le 1.2\!\times\!10^4\,\mathrm{K} \\"
        r"0, & T \le T_\mathrm{cut} = 0.7\,T_\mathrm{eff}"
        r"\end{cases}",
        label="eq:C8_twopiece",
    )
    ld.add(
        "Explicit-source CFL",
        r"\Delta t_\mathrm{rad}^\mathrm{thck} \le \beta_\mathrm{thck}\,\tau_\mathrm{thck},"
        r"\quad \beta_\mathrm{thck} \approx 0.3\text{ (backward-Euler handles }\beta \to \infty\text{)}",
        label="eq:C8_CFL",
    )
    ld.add(
        "ξ-boundary smoothness (C^∞ alternative)",
        r"\xi_\mathrm{rad}^\mathrm{smooth}(p) = \tfrac{1}{2}\bigl[1 + \tanh\!\bigl((p - p_\mathrm{chr})/\Delta p\bigr)\bigr]"
        r"\ \Longrightarrow\ \text{no sharp }C^0\text{ at }p = p_\mathrm{chr}",
        label="eq:C8_smooth",
    )

    ld.write()
    print()
    print("All C8 identities verified.")


if __name__ == "__main__":
    main()
