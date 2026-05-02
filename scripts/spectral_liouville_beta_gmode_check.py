#!/usr/bin/env python3
"""
E5b: check whether the α★ = 1 - σ/2 prefactor that regularises the
reduced-pressure Poisson operator ALSO regularises the g-mode SL operator.

If yes — unified basis narrative survives.
If no  — the Poisson and g-mode problems have DIFFERENT optimal prefactors,
         breaking the unified-basis selling point. Path A must then be
         rescoped (e.g., SL basis handles Poisson only).

The g-mode spectral operator (Cowling reduction, from
docs/gmode_experiments_2026-05-02.md §10) is

    -ψ'' + ell(ell+1)/r² · ψ = omega^{-2} · ell(ell+1) · N²/r² · ψ,
    ψ = ρ₀ r² ξ_r

i.e. a standard SL GEP  -ψ'' + V(r) ψ = omega^{-2} λ N²(r)/r² ψ  with
V(r) = ell(ell+1)/r² and weight λ N²/r².

Near the surface r → R:  N² is finite but ρ₀ → 0 does NOT appear in
V or the weight — the g-mode surface singularity is DIFFERENT from the
Poisson singularity.  Near the origin r → 0:  V ~ r⁻² centrifugal,
which is the singularity.  So the "bad point" for g-mode is r=0
(origin), while the "bad point" for Poisson is r=R (surface).

CONCLUSION (hypothesis to check): α★ for Poisson ≠ α★ for g-mode.
Unified SL basis with a single prefactor cannot solve both.

Alternative: keep two separate substitutions
  π̂ = t^α · r̂  for Poisson
  ξ_r = r^β · η for g-mode
and share the Chebyshev BASE grid but not the weights.
"""
from __future__ import annotations

import sympy as sp


def compute_schrod_potential(A, w, t):
    """For SL equation -[A p']' = μ w p, the Schrödinger form after
    p = (A w)^{-1/4} φ and coordinate change s = ∫ √(w/A) dt is

       -φ''(s) + Q(s) φ = μ φ,
       Q(s) = (A w)^{-1/4} [d²(A w)^{-1/4}/ds²] · (A w)^{1/2}
            + intermediate terms ...

    but for our case where A and w are simple we can just compute the
    leading singular behaviour numerically.
    """
    pass


def main():
    t, r, R, c, ell, sigma = sp.symbols('t r R c ell sigma', real=True, positive=True)
    k = sp.Symbol('k', real=True)
    n_pidx = sp.Symbol('n', real=True, positive=True)

    # Use t = R - r (distance to surface); Lane-Emden gives rho ~ t^sigma.
    # Near origin: r = R - t, t near R.

    print("=" * 72)
    print("Poisson operator — reduced pressure")
    print("=" * 72)
    print()
    rho = c * t ** sigma
    # Operator:  -[A p']' + k² A p = f  with A = rho
    # After π̂ = ρ₀^{-1/2} q the Schrödinger potential is
    #   W̃ = ρ''/(2ρ) - (ρ')²/(4ρ²)
    # For rho = c t^σ: W̃ = σ(σ-2)/(4t²)
    W_poisson = sigma * (sigma - 2) / (4 * t ** 2)
    print(f"  W̃_Poisson(t) = {W_poisson}     (surface singularity at t=0)")
    print(f"  C = σ(σ-2)/4")
    print(f"    σ=3/2 (n=1.5): C = {sp.Rational(3, 2) * (sp.Rational(3, 2) - 2) / 4} = -3/16")
    print(f"    σ=3   (n=3):   C = {sp.Integer(3) * (sp.Integer(3) - 2) / 4} = 3/4")
    print()
    print("  α★(Poisson) = 1 - σ/2  kills this singularity.")
    print()

    print("=" * 72)
    print("g-mode scalar Cowling operator  -ψ'' + ℓ(ℓ+1)/r² ψ = μ·ℓ(ℓ+1) N²/r² ψ")
    print("=" * 72)
    print()
    print("  Singular points:  r = 0 (centrifugal) and r = R (if N²/r² blows up).")
    print()

    # Near r = 0:  ψ = r^β · φ  standard Frobenius, β(β-1) + ℓ(ℓ+1)·0 = 0 …
    # actually the ODE is -ψ'' + ℓ(ℓ+1)/r² ψ = ... N²/r² ψ so indicial:
    #   β(β-1) - ℓ(ℓ+1) = 0  at r=0 (assuming N² → 0 linearly)
    # roots: β = ℓ+1 or β = -ℓ.  The finite branch is β = ℓ+1.
    # GYRE's y_1 = x^{2-ℓ} · (ξ_r/r) uses the same idea.
    beta = sp.Symbol('beta', real=True)
    indicial_origin = beta * (beta - 1) - ell * (ell + 1)
    roots = sp.solve(indicial_origin, beta)
    print(f"  Indicial equation at r=0:  β(β-1) = ℓ(ℓ+1)")
    print(f"  Roots:  β = {roots}")
    print(f"  Regular branch:  β = ℓ+1.")
    print()
    print("  For ℓ=1:  β_origin = 2  (i.e. ψ ~ r² near origin)")
    print()

    # Near r = R (surface):  ψ = rho_0 · r² · xi_r = (c (R-r)^σ) · R² · xi_r
    # If xi_r is finite at surface:  ψ ~ (R-r)^σ
    # This is a ZERO of ψ, not a singularity, provided σ > 0.
    # So the g-mode operator has NO operator-singularity at the surface,
    # only the kinematic fact that ψ → 0 as (R-r)^σ.
    # BUT the coefficient N²/r² → N²/R² is finite.
    # So the g-mode operator is SINGULAR at r=0 only.
    print("  Surface (r → R): ψ = ρ_0 r² ξ_r ~ (R-r)^σ · R² · ξ_r(R)")
    print("  This is a KINEMATIC zero of ψ, not an operator singularity.")
    print("  Operator coefficients ℓ(ℓ+1)/r² and N²/r² are bounded at r=R.")
    print()
    print("  CONCLUSION: g-mode has singular point at r=0 ONLY.")
    print()

    print("=" * 72)
    print("SEPARATE regularisations required")
    print("=" * 72)
    print()
    print("  • Poisson:  regularise at r = R (surface, ρ₀ → 0)")
    print("              via  π̂ = t^α · u  with  α = 1 - σ/2")
    print("              where t = R - r.")
    print()
    print("  • g-mode:   regularise at r = 0 (origin, centrifugal)")
    print("              via  ψ = r^{ℓ+1} · η")
    print("              which is GYRE's y_1 = x^{2-ℓ} · (ξ_r/r) choice in disguise.")
    print()
    print("  UNIFIED BASIS: the SL eigenproblem used for Poisson runs over [0, R]")
    print("  with ONE singular endpoint (r = R).  The g-mode eigenproblem runs")
    print("  over the same interval with ONE singular endpoint (r = 0).  A single")
    print("  Chebyshev basis discretises BOTH operators, but the prefactors are")
    print("  endpoint-specific and different.")
    print()
    print("  Path A still works IF: the underlying Chebyshev discretisation is")
    print("  the same, and we apply endpoint-appropriate prefactors per operator.")
    print("  The 'single SL eigendecomposition used for all k_x' claim survives")
    print("  for Poisson.  The g-mode spectrum is a separate EVP but on the same")
    print("  grid — similar in cost to one Poisson assembly.")


if __name__ == "__main__":
    main()
