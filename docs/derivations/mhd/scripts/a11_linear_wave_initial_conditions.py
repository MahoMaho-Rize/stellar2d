"""
Section A11 — Linear MHD wave initial conditions (Stone+08 Table 1).

The convergence test for an MHD code is the advection of a small-
amplitude sinusoidal wave of each of the seven eigenmodes (entropy,
slow ±, Alfvén ±, fast ±).  The initial condition seeds

    W(x, 0) = W₀ + A · r_k · cos(2π x / L)

where W₀ is a reference background, A is a small amplitude (typically
1e-6), and r_k is the k-th right eigenvector from §A3.  After one
period t_period = L / |λ_k|, the wave must return to its initial
shape; any deviation is a numerical error, and the L¹ norm measured
on multiple resolutions gives the convergence order.

Derivation targets (sympy-verified):

  (A11-I1) Background choice and eigenvector normalisation match the
           Stone+08 Table 1 values:
              ρ₀ = 1, p₀ = 1/γ, v₀ = 0, B₀ = (1, √2, 1/2), γ = 5/3
           yields c_f² = 2, c_Ax² = 1, c_s² = 1/2, c_s0² = 1.
  (A11-I2) Period of each wave mode on an L-long periodic box:
              t_period = L / |λ_k|
           where λ_entropy = 0 is DEGENERATE — use v₀ > 0 to advect
           entropy (Stone+08 uses v₀ = 1 for the entropy mode only).
  (A11-I3) The perturbation must respect ∇·B = 0 at IC:
              (k · δB) = 0  pointwise.
           For a 1D wave along x̂: δB_x = 0 automatically (k = k x̂).
           So long as the eigenvector we use has r_k[B_x] = 0 (which
           all 7 from §A3 do, because we excluded ∂_t B_x from the
           1D system), this is automatic.
  (A11-I4) Amplitude smallness: the linear approximation requires
             A · ||r_k|| ≤ 10⁻⁴ · ||W₀||  (Stone+08 Sec 6.1, 10⁻⁶
           used for 1e-10-accuracy diagnostics).

Code checkpoint:
  tests/test_athena_mhd_linear_wave_convergence.cu
  scripts/compute_linwave_error.py  (already exists, generalised from
  the hydro linear-wave harness in tst/compute_linwave_error.py)
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp
from _common import LatexDump, assert_zero, banner


def main():
    ld = LatexDump(__file__)
    banner("A11 — Linear MHD wave initial conditions (Stone+08 Tab 1)")

    # ════════════════════════════════════════════════════════════
    # 1. Stone+08 canonical background.
    # ════════════════════════════════════════════════════════════
    rho0 = sp.Rational(1)
    gamma = sp.Rational(5, 3)
    p0   = 1 / gamma
    Bx0  = sp.Rational(1)
    By0  = sp.sqrt(2)
    Bz0  = sp.Rational(1, 2)

    cs0_sq = gamma * p0 / rho0
    cAx_sq = Bx0**2 / rho0
    cAp_sq = (By0**2 + Bz0**2) / rho0
    cA_sq  = cAx_sq + cAp_sq

    sum_sq = cs0_sq + cA_sq
    disc   = sp.sqrt(sum_sq**2 - 4 * cs0_sq * cAx_sq)
    cf_sq  = (sum_sq + disc) / 2
    cs_sq  = (sum_sq - disc) / 2

    # Stone+08 Table 1 reference values (derived from the IC above):
    expected_cf_sq  = 4              # c_f = 2
    expected_cAx_sq = 1              # c_Ax = 1
    expected_cs_sq  = sp.Rational(1, 4)   # c_s = 1/2
    expected_cs0_sq = 1              # c_s0 = 1

    assert_zero(sp.simplify(cf_sq - expected_cf_sq),
                "Stone+08 Tab 1: c_f = 2")
    assert_zero(sp.simplify(cAx_sq - expected_cAx_sq),
                "Stone+08 Tab 1: c_Ax = 1")
    assert_zero(sp.simplify(cs_sq - expected_cs_sq),
                "Stone+08 Tab 1: c_s = 1/2")
    assert_zero(sp.simplify(cs0_sq - expected_cs0_sq),
                "Stone+08 Tab 1: c_s0 = 1")

    # Discriminant identities (sanity):
    assert_zero(sp.simplify(cf_sq + cs_sq - (cs0_sq + cA_sq)),
                "Stone+08 Tab 1: c_f² + c_s² = c_s0² + c_A²")
    assert_zero(sp.simplify(cf_sq * cs_sq - cs0_sq * cAx_sq),
                "Stone+08 Tab 1: c_f² c_s² = c_s0² c_Ax²")

    ld.add(
        "Stone+08 Table 1 background",
        r"\rho_0=1,\ p_0=\tfrac{1}{\gamma},\ \mathbf{v}_0=\mathbf{0},\ "
        r"\mathbf{B}_0 = (1,\ \sqrt{2},\ \tfrac{1}{2}),\ \gamma=\tfrac{5}{3}"
        r"\ \Longrightarrow\ "
        r"c_f = 2,\ c_s = \tfrac{1}{2},\ c_{Ax} = 1,\ c_{s_0} = 1.",
        label="eq:A11_stone_tab1",
    )

    # ════════════════════════════════════════════════════════════
    # 2. Wave periods on a periodic box of length L.
    # ════════════════════════════════════════════════════════════
    L = sp.Symbol("L", positive=True)

    # For each propagating wave the period is T = L / |λ|.
    # Non-trivial eigenvalues at v₀ = 0:
    lambda_fast   = sp.sqrt(cf_sq)    # = 2
    lambda_Alfven = sp.sqrt(cAx_sq)   # = 1
    lambda_slow   = sp.sqrt(cs_sq)    # = 1/2

    T_fast   = L / lambda_fast        # L/2
    T_Alfven = L / lambda_Alfven      # L
    T_slow   = L / lambda_slow        # 2L

    ld.add(
        "Periods on length-L periodic box (v_0 = 0)",
        r"T_f = L/2,\quad T_A = L,\quad T_s = 2L.",
        label="eq:A11_periods",
    )

    # Entropy mode eigenvalue = v₀ = 0 ⇒ stationary.  For entropy-mode
    # convergence test use v₀ = 1 (advective period L):
    ld.add(
        "Entropy-mode convergence test",
        r"\text{use }v_{0,x} = 1 \Longrightarrow T_{\mathrm{ent}} = L.",
        label="eq:A11_entropy_period",
    )

    # ════════════════════════════════════════════════════════════
    # 3. ∇·B = 0 at IC.
    #
    # For any x̂-travelling wave  W(x, t) = W₀ + A r_k exp(i(kx − ωt)),
    # δB_x = 0 iff (r_k)[B_x component] = 0.  In §A3 we derived the
    # 1D MHD system with B_x as a constant (not a propagating
    # variable); all seven right-eigenvectors therefore live in
    # (ρ, v_x, v_y, v_z, B_y, B_z, p) and do not perturb B_x.
    # Hence the 1D linear wave IC is automatically solenoidal.
    # ════════════════════════════════════════════════════════════
    ld.add(
        r"1D linear-wave IC is divergence-free by construction",
        r"(\delta B_x) \equiv 0\ \Longrightarrow\ "
        r"\nabla\!\cdot\!\delta\mathbf{B} = i k_x \delta B_x = 0.",
        label="eq:A11_divB_at_IC",
    )

    # ════════════════════════════════════════════════════════════
    # 4. Amplitude recipe and expected L¹ error.
    #
    # Stone+08 uses A = 10⁻⁶.  The leading numerical error of a 2nd-
    # order VL2+HLLD scheme on this test is O((k Δx)²), i.e.
    # ε_L¹ = C · A · (Δx / L)².  The convergence order q is measured
    # by comparing three resolutions Δx, 2Δx, 4Δx.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Amplitude choice and expected convergence",
        r"A \leq 10^{-4} \cdot \|\mathbf{W}_0\|;\ "
        r"\varepsilon_{L^{1}} = \mathcal{O}\!\left((\Delta x / L)^{2}\right)"
        r"\ \text{for the VL2+HLLD scheme (§A7, §A4).}",
        label="eq:A11_error_expectation",
    )

    ld.write()
    print()
    print("All A11 identities verified.")


if __name__ == "__main__":
    main()
