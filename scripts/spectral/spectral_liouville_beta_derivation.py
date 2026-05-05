#!/usr/bin/env python3
"""
E5: symbolic derivation of β★ that kills the t⁻² singularity in the
    reduced-pressure Liouville potential after a second substitution
    q = t^β · q̃.

Background (see docs/reduced_pressure_liouville.md §3):
  - Anelastic reduced pressure π = p/ρ₀, operator ∇·(ρ₀ ∇π) = f.
  - Fourier-in-x: d/dy[ρ₀ dπ̂/dy] - k_x² ρ₀ π̂ = f̂.
  - First Liouville substitution  π̂ = ρ₀^{-1/2} q  turns the operator into
        q'' + W̃(y) q - k_x² q = √ρ₀ · f̂
    with  W̃ = ρ₀''/(2ρ₀) - (ρ₀')²/(4ρ₀²).
  - For Lane-Emden surface decay ρ₀ ~ t^σ  (σ = 1.5 for n=3/2; σ = 3 for n=3),
    W̃ ~ +C / t²  with C = σ(σ-2)/4 + σ·(other)... derived below.

Our task (E5): introduce q = t^β · q̃, compute the transformed potential
W̃_β(t), and find β★ such that t⁻² terms cancel entirely.  A clean β★ is
part of the theoretical Path A pitch.

NOTE the transform adds a first-derivative cross term; we must ensure the
second substitution keeps the equation self-adjoint (same Liouville canonical
form).  The clean way: transform the SL equation

    q'' + W̃ q - k_x² q = g

to a new SL equation in q̃ via q = t^β q̃.  The resulting equation contains
a q̃' term, which we absorb by writing everything as
(t^{2β} q̃)'' + (coefficients) = 0 ... — actually the cleanest
approach is to use a pair of substitutions:
    q = t^β · m(t) · q̃
where m(t) is chosen to absorb the q̃' term (standard Liouville-to-Schrödinger
reduction).  We do this symbolically.

Reference: Morse & Feshbach (1953) §6.3; Coddington-Levinson §4.
"""
from __future__ import annotations

import sympy as sp


def derive_W_tilde(rho_expr, t):
    """Reduced-pressure Liouville potential W̃ from ρ₀(t).

    From reduced_pressure_liouville.md §3.2:
        W̃(y) = ρ''/(2ρ) - (ρ')²/(4ρ²).
    """
    rho_p = sp.diff(rho_expr, t)
    rho_pp = sp.diff(rho_p, t)
    return rho_pp / (2 * rho_expr) - rho_p ** 2 / (4 * rho_expr ** 2)


def transform_W_with_prefactor(W_tilde, t, beta):
    """Derive the potential W̃_β(t) for q = t^β · q̃ under the SL equation

        q'' + W̃ q - k_x² q = 0.

    Write q = t^β m q̃ where m(t) is chosen to kill the q̃' term arising
    from the non-unit weight t^β.  For a pure power prefactor q = t^β q̃:
        q'  = β t^(β-1) q̃ + t^β q̃'
        q'' = β(β-1) t^(β-2) q̃ + 2β t^(β-1) q̃' + t^β q̃''
    Substituting into q'' + W̃ q - k² q = 0 and dividing by t^β:
        q̃'' + (2β/t) q̃' + [β(β-1)/t² + W̃ - k²] q̃ = 0

    This is NOT Schrödinger form.  The standard fix is to introduce
    q̃ = t^{-β} · ψ (i.e. go back!), or to use the Liouville canonical
    transformation  q̃ = t^{-β} · φ.  But we want a prefactor that
    SIMPLIFIES the equation.  The clean approach: rewrite in terms of
    the substitution  q = t^β · (something that brings us back to
    Schrödinger).

    Standard result: if we substitute q = h(t) φ with h chosen so that
    the original self-adjoint coefficient is absorbed, we get a new
    Schrödinger equation with shifted potential.  For our case with
    original equation already in canonical form (coefficient of q'' is
    1), the substitution q = t^β φ does NOT preserve canonical form;
    we need q = t^β · ψ and then let v = ψ / √(t^{2β}) ... this is
    circular.

    The RIGHT way.  The self-adjoint operator is -d²/dt² - W̃(t) acting
    on q.  Consider the change of coordinates OR the similarity
    transform operator U = t^β, giving

        U⁻¹ [-d²/dt² - W̃] U q̃ = μ q̃
        = [-d²/dt² - (2β/t) d/dt - β(β-1)/t² - W̃] q̃ = μ q̃

    This is first-order in d/dt (non-self-adjoint).  To recover self-
    adjoint form we'd need a *weighted* inner product, i.e. the
    eigenproblem becomes

        -[t^{2β} q̃']' - t^{2β} W̃ q̃ = μ t^{2β} q̃

    which IS self-adjoint in the weight w(t) = t^{2β}.  The operator
    -[t^{2β} q̃']' / t^{2β} has the same eigenvalues as -d²/dt² - W̃
    on q, but the *weighted* SL form is what we discretise.

    So the meaningful "W̃_β" in the Schrödinger sense is:
        Ŵ_β = W̃ + β(β-1)/t² + 2β·(t^{2β})'/(2·t^{2β}·t) · ...
    (messy first-order term)

    Cleanest derivation: apply the substitution q(t) = t^β ψ(t) and then
    re-Liouville-ise back to Schrödinger form by the standard recipe
    ψ = w(t)^{-1/2} φ, w = t^{2β}, giving a new Schrödinger equation
    for φ whose potential we print out.
    """
    # Variables
    q = sp.Function('q')(t)
    q_tilde = sp.Function('q_tilde')(t)
    phi = sp.Function('phi')(t)

    # Step 1: q = t^β q_tilde, weighted SL form with w = t^{2β}
    # Step 2: re-Liouville-ise q_tilde = t^{-β} φ to get Schrödinger form
    # Net effect: q = t^β · t^{-β} · φ = φ, so NO prefactor?? Let's
    # check via direct calculation.
    # Actually the standard trick: q = √w · φ where w = t^{2β}, i.e.
    # q = t^β φ.  Plug into canonical Schrödinger and compute new potential.

    # Substitute q = t^β · phi in q'' + W̃ q = 0
    q_sub = t ** beta * phi
    q_sub_p = sp.diff(q_sub, t)
    q_sub_pp = sp.diff(q_sub_p, t)

    # Original Schrödinger: q'' + W̃ q = 0 (modulo k² and eigenvalue)
    # After substitution, divide by t^β to get equation for φ:
    lhs = sp.expand(q_sub_pp + W_tilde * q_sub)
    lhs_phi = sp.simplify(lhs / t ** beta)
    # This has both φ, φ', φ'' terms — NOT Schrödinger form.
    return lhs_phi


def analyze_leading_singularity(W_tilde, t, rho_expr, sigma):
    """Compute the leading t⁻² coefficient of W̃ as t→0, given ρ₀ ~ t^σ."""
    W_series = sp.series(W_tilde, t, 0, 1).removeO()
    # Or: just compute W̃ * t² at t→0
    W_times_t2 = sp.simplify(W_tilde * t ** 2)
    leading = sp.limit(W_times_t2, t, 0)
    return leading


def main():
    t, beta, C, k = sp.symbols('t beta C k', real=True, positive=True)
    sigma = sp.Symbol('sigma', real=True, positive=True)

    # Test ρ₀ ~ c · t^σ near surface.
    c = sp.Symbol('c', real=True, positive=True)
    rho_generic = c * t ** sigma

    W_tilde = derive_W_tilde(rho_generic, t)
    W_tilde_simp = sp.simplify(W_tilde)
    print("=" * 70)
    print("Reduced-pressure Liouville potential W̃(t) for ρ₀ = c·t^σ :")
    print(f"  W̃(t) = {W_tilde_simp}")
    # Expected from reduced_pressure_liouville.md: W̃ = σ(σ-2)/(4 t²)
    # because ρ'/ρ = σ/t, ρ''/ρ = σ(σ-1)/t², so
    # W̃ = σ(σ-1)/(2t²) - σ²/(4t²) = (2σ²-2σ-σ²)/(4t²) = (σ²-2σ)/(4t²)
    C_expected = sigma * (sigma - 2) / 4
    leading_C = sp.simplify(W_tilde_simp * t ** 2)
    print(f"  leading t⁻² coefficient: C(σ) = {leading_C}")
    print(f"  check: σ(σ-2)/4 = {sp.simplify(C_expected)}")
    assert sp.simplify(leading_C - C_expected) == 0
    print()

    # Evaluate for specific polytropes:
    for n_val, sigma_val, label in [(sp.Rational(3, 2), sp.Rational(3, 2), "n=3/2 (Lane-Emden 3/2)"),
                                     (sp.Integer(3), sp.Integer(3), "n=3 (Lane-Emden 3)")]:
        C_val = sigma_val * (sigma_val - 2) / 4
        print(f"  {label}:  σ = {sigma_val},  C = σ(σ-2)/4 = {C_val}")
    print()

    # Now add the second substitution q = t^β q̃ and compute the new SL eqn.
    # Form: q'' + W̃ q = 0  →  (t^β q̃)'' + W̃ (t^β q̃) = 0
    # →  β(β-1) t^{β-2} q̃ + 2β t^{β-1} q̃' + t^β q̃''  +  W̃ t^β q̃ = 0
    # Divide by t^β:
    #   q̃'' + (2β/t) q̃' + [β(β-1)/t² + W̃] q̃ = 0
    # This has a q̃' term: it's NOT Schrödinger.  To return to Schrödinger,
    # use the Liouville trick q̃ = t^{-β} φ (canonical for weight t^{2β}):
    # Substituting q̃ = t^{-β} φ in the above cancels the q̃' but also
    # rebuilds β(β-1)/t² with opposite sign ... so q = t^β q̃ = t^β · t^{-β} φ = φ,
    # which is trivial and gives BACK the original equation.  This confirms:
    # a PURE power prefactor alone cannot eliminate the singularity.  We need
    # the LIOUVILLE substitution on a DIFFERENT underlying operator (i.e., with
    # a modified weight), OR we change the original ρ₀ → ρ₀·t^{-2β} effectively.
    #
    # CORRECT PATH: recognise that the SL equation
    #     -[A(t) p']' + B(t) p = μ w(t) p
    # has the Liouville-canonical form (Schrödinger) with weight w.  In our
    # case A = ρ₀, B = 0, w = ρ₀.  The standard Liouville transform  p = ρ₀^{-1/2} q
    # AND a coordinate stretching  s = ∫ (w/A)^{1/2} dt  together give
    # canonical Schrödinger.  For our case w=A=ρ₀ so ds/dt = 1 (no stretch).
    # The resulting potential is the W̃ we already have.  A second substitution
    # can only multiply by a non-singular factor, NOT kill singular behaviour.
    #
    # So: a plain q = t^β q̃ doesn't help.  What DOES help is modifying the
    # OUTER equation BEFORE Liouville-ising.  Specifically: substitute
    #   p̂ = t^α · r̂   (at the PRESSURE level, before any Liouville step)
    # so that r̂ is well-behaved at t=0.  Then Liouville-ise the resulting
    # equation for r̂.  The relevant α is chosen from the Frobenius analysis.

    # Re-do: start from  (ρ₀ π̂')' - k² ρ₀ π̂ = f̂
    # Substitute  π̂ = t^α · r̂.  Then r̂ obeys a different self-adjoint equation
    # with coefficient ρ₀ · t^{2α}.  Applying Liouville  r̂ = (ρ₀ t^{2α})^{-1/2} u
    # gives a Schrödinger equation with effective potential
    #   W̃_α(t) = W̃_ρ₀·t^{2α}  = (ρ₀·t^{2α})''/(2 ρ₀·t^{2α}) - ...'²/(...)²
    # If ρ₀ = c·t^σ  then ρ₀·t^{2α} = c·t^{σ+2α}, so the effective "σ" becomes
    # σ_eff = σ + 2α.  Leading t⁻² coefficient becomes
    #   C_eff = σ_eff(σ_eff - 2)/4 = (σ + 2α)(σ + 2α - 2)/4.
    # We want C_eff = 0  →  σ + 2α = 0  OR  σ + 2α = 2.
    #   α★₁ = -σ/2   (gives σ_eff = 0, no density left, degenerate)
    #   α★₂ = (2-σ)/2 = 1 - σ/2

    alpha = sp.Symbol('alpha', real=True)
    sigma_eff = sigma + 2 * alpha
    C_eff = sigma_eff * (sigma_eff - 2) / 4
    print("Second substitution  π̂ = t^α · r̂  (applied BEFORE Liouville):")
    print(f"  effective  σ_eff = σ + 2α = {sigma_eff}")
    print(f"  effective  C_eff = σ_eff(σ_eff-2)/4 = {sp.simplify(C_eff)}")
    print()
    print("Solve C_eff = 0:")
    sols = sp.solve(C_eff, alpha)
    print(f"  α★ = {sols}")
    print()

    # Evaluate the two branches for our two polytropes
    print("Concrete values of α★:")
    print(f"  {'Polytrope':<22} {'σ':>6}   {'α★₁ = -σ/2':>14}   {'α★₂ = (2-σ)/2':>16}")
    for lbl, sig in [("Lane-Emden n=3/2", sp.Rational(3, 2)),
                     ("Lane-Emden n=3",   sp.Integer(3))]:
        a1 = -sig / 2
        a2 = (2 - sig) / 2
        print(f"  {lbl:<22} {sig!s:>6}   {a1!s:>14}   {a2!s:>16}")
    print()

    # Check: α★₁ = -σ/2 makes ρ₀·t^{2α} = c (constant): the whole density
    # weight vanishes, and the equation becomes constant-coefficient — the
    # trivial limit.  That's not a productive substitution (we lose the
    # variable-coefficient structure we wanted to handle).
    #
    # α★₂ = 1 - σ/2 makes ρ₀·t^{2α} = c·t², effectively replacing the
    # Lane-Emden surface decay with a universal t² weight.  The resulting
    # equation still has k² coupling but the singularity at t=0 is GONE:
    # W̃_{α★₂} ≡ (ρ₀·t²)'' / (2·ρ₀·t²) - (...)²/(...)² evaluated at t→0
    # with ρ₀·t² = c·t² (independent of σ): W̃ = 0 to leading order,
    # O(1) corrections only.
    #
    # This is the clean result.  Verify by direct substitution.

    print("=" * 70)
    print("VERIFY: pick α = α★₂ and compute new W̃ directly.")
    for sig_v in [sp.Rational(3, 2), sp.Integer(3)]:
        alpha_star = (2 - sig_v) / 2
        rho_new = c * t ** (sig_v + 2 * alpha_star)  # = c·t²
        W_new = derive_W_tilde(rho_new, t)
        W_new_simp = sp.simplify(W_new)
        print(f"  σ = {sig_v}:  α★₂ = {alpha_star},  ρ₀·t^{{2α★}} = {sp.simplify(rho_new)}")
        print(f"    New W̃(t) = {W_new_simp}")
        # Multiply by t² and take limit
        W_times_t2 = sp.simplify(W_new_simp * t ** 2)
        C_leading = sp.limit(W_times_t2, t, 0)
        print(f"    Leading t⁻² coefficient:  C = {C_leading}")
        print()

    # Full check: compose π̂ = t^α · r̂  and  r̂ = (c·t²)^{-1/2} u = t^{-1}/√c · u
    # So π̂ = t^{α-1}/√c · u.  For n=3: α = -1/2, so π̂ = t^{-3/2}/√c · u.
    # That's a strong singular prefactor at t=0, pushing u to be regular
    # there.  The underlying u obeys Schrödinger with a smooth-at-t=0 W_eff.
    print("Total substitution chain:")
    print(r"  π̂(t) = t^α · r̂ = t^α · (ρ₀·t^{2α})^{-1/2} · u")
    print(r"       = t^α · (c·t²)^{-1/2} · u    [with α = α★₂]")
    print(r"       = t^{α-1}/√c · u")
    print()
    for sig_v in [sp.Rational(3, 2), sp.Integer(3)]:
        alpha_star = (2 - sig_v) / 2
        expo = alpha_star - 1
        print(f"  σ = {sig_v}:  π̂ = t^{{{expo}}} / √c · u    (α★ = {alpha_star})")
    print()

    # Now check g-mode version: same substitution, same α★, because the
    # singularity structure of the g-mode equation is the same rho₀-driven.

    print("=" * 70)
    print("SUMMARY")
    print(f"  α★₂ = 1 - σ/2  regularises the reduced-pressure Liouville")
    print(f"  potential at t = 0 for ANY polytropic index σ.")
    print(f"  For Lane-Emden n = 1.5 (σ = 3/2): α★ = 1/4.")
    print(f"  For Lane-Emden n = 3   (σ = 3):   α★ = -1/2.")


if __name__ == "__main__":
    main()
