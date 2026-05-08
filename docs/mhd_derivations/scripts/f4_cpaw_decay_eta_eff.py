"""
Section F4 — CPAW 2D long-time decay and η_eff extraction.

Goal: derive the **closed-form numerical-resistivity formula** that
lets us extract η_eff(N) from CPAW 2D long-time runs at multiple
resolutions.

The linearised Alfvén wave in 1D on a uniform background (§B2) obeys
  ∂_t z^± ± v_A ∂_r z^± = 0   (ideal MHD).
In the presence of finite resistivity η, both Elsässer variables
decay diffusively:
  ∂_t z^± ± v_A ∂_r z^± = η ∂_r² z^±.
Fourier mode exp(i(kr − ωt)) gives
  ω = v_A k − i η k²  ⇒  amplitude decays as exp(-η k² t).

For a 2-nd-order Godunov scheme the numerical resistivity scales as
  η_eff(h, k) = C · h² · v_A · k² / 2  (modified-equation analysis)
                                   of PLM + HLLD on linear Alfvén mode.
This is the formula we use to extract η_eff(N) from pairs of
long-time CPAW runs at different N.

Verifies:
  - Diffusive dispersion ω = v_A k − i η k² from linearised Alfvén +
    resistivity (sympy symbolically).
  - Amplitude exp(-η k² t) decays monotonically (η > 0).
  - Power-law scaling η_eff ∝ h² (for 2nd-order scheme): log-log slope
    of η_eff vs h is 2; sympy-verified via symbolic log differentiation.
  - Two-resolution inversion: given measured decay rates γ_N₁ and γ_N₂
    at resolutions N₁, N₂, the scheme order p = log(γ_N₁/γ_N₂) / log(N₂/N₁).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("F4 — CPAW 2D decay rate & η_eff extraction")

    # ─── Identity 1: resistive Alfvén dispersion ─────────────────────
    # Linearised equations on uniform (ρ₀, B_r0, v=0) background with
    # finite resistivity η:
    #   ∂_t v_⊥ = (B_r0/ρ₀) ∂_r B_⊥,
    #   ∂_t B_⊥ = B_r0 ∂_r v_⊥ + η ∂_r² B_⊥.
    # Plane-wave ansatz (v_⊥, B_⊥) = (V, B) exp(i(k r − ω t)) gives
    # the 2×2 dispersion matrix
    #   [-iω         -i k B_r0 / ρ₀ ] [V]
    #   [-i k B_r0    -iω + η k²    ] [B]   = 0.
    # Determinant-=0 condition:
    #   (-iω)(-iω + η k²) − (-i k B_r0 / ρ₀)(-i k B_r0) = 0
    #   ω² + iω η k² − k² v_A² = 0    (using v_A² = B_r0² / ρ₀)
    # Two branches:  ω = (-i η k² ± √(-η² k⁴ + 4 v_A² k²)) / 2
    # For η ≪ v_A / k (weak-diffusion limit):
    #   ω ≈ v_A k − (i/2) η k²    (outgoing)
    #   ω ≈ -v_A k − (i/2) η k²   (incoming)
    # Both decay at rate η k² / 2.
    # NOTE: The "(i/2)" factor absorbed into a redefined η is convention-
    # dependent; some authors drop the 1/2.  We track the factor.
    k, rho0, Br0, eta = sp.symbols("k rho_0 B_r0 eta", positive=True)
    omega = sp.Symbol("omega", complex=True)
    v_A = Br0 / sp.sqrt(rho0)
    # Dispersion from det = 0 with conventional sign choice
    # Use ansatz  ∂_t → -iω, ∂_r → i k
    #   (-iω) V   − (i k v_A²) B / v_A² · v_A² = 0  (momentum)
    #   (-iω) B − (i k B_r0) V   + η k² B = 0       (induction)
    # → expand:
    #   -iω V = i k v_A² B / (... careful)
    # Clean derivation:
    # Eq1: -iω V − (i k B_r0 / ρ₀) B = 0  ⇒ V = -(k B_r0 / (ω ρ₀)) B
    # Eq2: -iω B − i k B_r0 V + η k² B = 0
    # Substitute V:
    #   -iω B − i k B_r0 · (-(k B_r0 / (ω ρ₀))) B + η k² B = 0
    #   -iω B + i k² v_A² / ω · B + η k² B = 0
    #   (-iω² + i k² v_A² + η k² ω) B = 0
    # → ω² + i η k² ω − k² v_A² = 0
    dispersion = omega**2 + sp.I * eta * k**2 * omega - k**2 * v_A**2
    # Solve for ω
    sol = sp.solve(dispersion, omega)
    # Two roots; each should approach ±k v_A in η → 0 limit.
    for root in sol:
        limit_ideal = sp.simplify(root.subs(eta, 0))
        # Should be ±k v_A = ±k Br0/√ρ₀
        # simplify: Br0/sqrt(rho0) = v_A
        expected_plus = k * v_A
        expected_minus = -k * v_A
        d1 = sp.simplify(limit_ideal - expected_plus)
        d2 = sp.simplify(limit_ideal - expected_minus)
        assert d1 == 0 or d2 == 0, f"η=0 limit of {root} not ±v_A k, got {limit_ideal}"
    print("  [OK] ideal limit η → 0 recovers ω = ±v_A k.")

    # Weak-η expansion: pick the +v_A k root,
    #   ω = v_A k − (i/2) η k² + O(η²)
    # Extract leading imaginary part by series expansion.
    root_plus = [r for r in sol if sp.simplify(r.subs(eta, 0) - k*v_A) == 0][0]
    expansion = sp.series(root_plus, eta, 0, 2).removeO()
    # Should equal v_A k − (I/2) η k² (leading order)
    expected = k * v_A - sp.I * eta * k**2 / 2
    assert_zero(sp.simplify(expansion - expected),
                "weak-η expansion: ω ≈ v_A k − (i/2) η k²", verbose=False)
    print("  [OK] weak-η expansion: ω = v_A k − (i/2) η k² + O(η²).")

    # Amplitude:  A(t) = A_0 exp(-Im(ω) t) = A_0 exp(-(η k² / 2) t)
    # (monotone-decreasing for η > 0).
    Im_omega = sp.im(expansion)
    # For eta > 0, Im_omega should be negative (amplitude decays).
    # Im(v_A k - (i/2) η k²) = -(η k² / 2)
    assert_zero(sp.simplify(Im_omega - (-eta * k**2 / 2)),
                "Im(ω) = −η k²/2 (amplitude decay rate)", verbose=False)
    print("  [OK] amplitude decay rate γ_phys = η k² / 2.")

    # ─── Identity 2: η_eff for 2nd-order Godunov (dimensional) ───────
    # The modified-equation analysis of PLM + HLLD on linear Alfvén
    # modes gives
    #   η_eff(h) = C · h² · v_A
    # where C = O(1) is scheme-dependent.  The wavenumber dependence
    # factors out of the amplitude decay rate:
    #   γ_num(N) = η_eff(N) k² / 2 = (C · h² · v_A / 2) · k²
    # On a fixed domain of size L with k = k_0 fixed, h ∝ 1/N, so
    #   γ_num(N) ∝ N^{-2}.
    # This is the 2nd-order convergence signature we will extract from
    # multi-N runs.
    h, C_num, N_sym = sp.symbols("h C_num N", positive=True)
    eta_eff = C_num * h**2 * v_A
    # γ_num = η_eff k² / 2
    gamma_num = eta_eff * k**2 / 2
    # Check: d/dh (log γ_num) = 2/h  ⇒  log-log slope of γ vs h is 2.
    dloggamma_dlogh = sp.simplify(sp.diff(sp.log(gamma_num), h) * h)
    assert_zero(sp.simplify(dloggamma_dlogh - 2),
                "d log γ_num / d log h = 2 (2nd-order scheme)",
                verbose=False)
    print("  [OK] γ_num ∝ h² (2nd-order modified-equation signature).")

    # ─── Identity 3: two-resolution scheme-order inversion ─────────
    # Given measured γ_1 at N_1 and γ_2 at N_2 (with N_2 > N_1),
    # the effective scheme order is
    #   p = log(γ_1 / γ_2) / log(N_2 / N_1).
    # For an ideal 2nd-order solver p = 2.  This is the formula the
    # A4 test uses to validate the scheme's long-time η_eff scaling.
    gamma_1, gamma_2, N1, N2 = sp.symbols("gamma_1 gamma_2 N_1 N_2",
                                           positive=True)
    p = sp.log(gamma_1 / gamma_2) / sp.log(N2 / N1)
    # Consistency check: if γ_i = C / N_i^q, then p should give q.
    q = sp.Symbol("q", positive=True)
    gamma_1_ansatz = C_num / N1**q
    gamma_2_ansatz = C_num / N2**q
    p_evaluated = p.subs({gamma_1: gamma_1_ansatz, gamma_2: gamma_2_ansatz})
    assert_zero(sp.simplify(p_evaluated - q),
                "two-resolution order p = log(γ_1/γ_2) / log(N_2/N_1)",
                verbose=False)
    print("  [OK] two-resolution inversion formula recovers scheme order q.")

    # ─── Identity 4: weak-η validity bound ───────────────────────────
    # The weak-η expansion valid iff η k² ≪ v_A k, i.e., η k ≪ v_A.
    # For our CPAW IC v_A = 1, k = 2π/λ with λ = 1, so k = 2π, and the
    # criterion is η ≪ 1/(2π) ≈ 0.16.
    # At N=32 we will measure η_eff ~ 10⁻³, so the weak-η expansion is
    # well-justified.
    # Symbolic: dimensionless expansion parameter η k / v_A.
    epsilon_weak = eta * k / v_A
    print(f"  weak-η criterion:  η k / v_A ≪ 1  "
          f"(= {sp.simplify(epsilon_weak)})")

    # ─── LaTeX dump ─────────────────────────────────────────────────
    ld.add(
        "Resistive Alfvén dispersion",
        r"\omega^2 + i\,\eta\,k^2\,\omega - v_A^2 k^2 = 0,\qquad "
        r"v_A = B_{r,0}/\sqrt{\rho_0}",
        label="eq:F4_disp",
    )
    ld.add(
        "Weak-resistivity expansion (outgoing branch)",
        r"\omega = v_A k - \tfrac{i}{2}\,\eta\,k^2 + \mathcal{O}(\eta^2),\qquad "
        r"\varepsilon_{\mathrm{weak}} \equiv \eta k / v_A \ll 1",
        label="eq:F4_weak",
    )
    ld.add(
        "Amplitude decay rate",
        r"A(t) = A_0 \exp(-\gamma\,t),\qquad \gamma = \tfrac{1}{2}\,\eta\,k^2",
        label="eq:F4_decay",
    )
    ld.add(
        "Numerical resistivity (2nd-order Godunov modified-equation)",
        r"\eta_{\mathrm{eff}}(h) = C_{\mathrm{num}}\,h^2\,v_A,\qquad "
        r"\gamma_{\mathrm{num}}(N) = \tfrac{1}{2}\,C_{\mathrm{num}}\,h^2\,v_A\,k^2 "
        r"\propto N^{-2}",
        label="eq:F4_eta_eff",
    )
    ld.add(
        "Two-resolution scheme-order inversion",
        r"p = \frac{\log(\gamma_1/\gamma_2)}{\log(N_2/N_1)},\qquad "
        r"\text{expect } p \approx 2\text{ for 2nd-order scheme}",
        label="eq:F4_order",
    )
    ld.add(
        "Test-pass criteria (A4 CPAW long-time)",
        r"|p_{\text{measured}} - 2| < 0.3\quad \text{AND}\quad "
        r"A(t=50\,T)/A(0) \in [0.1, 1]\quad \text{for all }N",
        label="eq:F4_pass",
    )

    ld.write()
    print()
    print("All F4 identities verified.")


if __name__ == "__main__":
    main()
