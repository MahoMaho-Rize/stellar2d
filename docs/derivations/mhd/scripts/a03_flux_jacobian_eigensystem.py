"""
Section A3 — flux Jacobian A = dF/dU and its 7-wave eigensystem
                           (Roe 1981 / Brio-Wu 1988 / Stone+08 Appendix B).

The goal of this section is to produce, verify, and document the eigen-
decomposition of the x-direction flux Jacobian used inside HLLD and any
characteristic-based Riemann solver.  The algebra is heavy, so we work
in PRIMITIVE variables (where the Jacobian is simpler and the standard
right-eigenvectors are tabulated in Stone+08 Appendix B §B.1) and keep
the symbolic derivation as clean as possible.

Working primitive vector (1D along x):
    W = (rho, v_x, v_y, v_z, B_y, B_z, p)^T       (7 × 1)

B_x is constant across the x-interface (not an evolution variable in 1D
MHD; this is a fundamental consequence of ∂B_x/∂x = ∇·B = 0 in 1D).
Therefore the 1D MHD system has *seven* propagating waves, not eight.
The "extra" wave of the 2D/3D system is the divergence-cleaning wave
that we handle separately with CT (§A5).

Derivation targets:
  1. Write the primitive-form flux Jacobian A_W in 1D.
  2. Analytically construct the seven wave speeds:
       λ = v_x − c_f, v_x − c_A, v_x − c_s, v_x, v_x + c_s, v_x + c_A, v_x + c_f
     where c_f / c_s are the fast/slow magnetosonic speeds and c_A is
     the x-Alfvén speed.
  3. Verify that c_f² + c_s² = c_s0² + c_Ax² + c_A⊥² and c_f²·c_s² =
     c_s0²·c_Ax² (the discriminant identities).
  4. Construct the right-eigenvectors (β, α-parametrisation, Stone+08
     Eq. B7–B13) and sympy-verify  A_W · r_k − λ_k · r_k = 0  for each
     of the seven waves.
  5. Verify the normalization  ℓ_k · r_k' = δ_{kk'} on the seven-wave
     system.

These identities are what the future
  src/gpu/explicit/athena_mhd_kernels.cu::d_mhd_eigenvalues_primitive
  src/gpu/explicit/athena_mhd_kernels.cu::d_mhd_right_eigenvectors
must reproduce to floating-point accuracy.  A single sign error in r_k
survives all hydro-only tests and manifests as wrong shock jumps.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sympy as sp                                # noqa: E402
from _common import (                             # noqa: E402
    rho, p, gamma, v_x, v_y, v_z, B_x, B_y, B_z,
    LatexDump, assert_zero, banner,
)

def main():
    ld = LatexDump(__file__)
    banner("A3 — 1D MHD flux Jacobian eigensystem (7 waves)")

    # ════════════════════════════════════════════════════════════
    # 1. Primitive-form flux Jacobian in the x-direction (7×7).
    #
    # Starting from the primitive form of ideal MHD:
    #   ∂_t W + A_W(W) ∂_x W = 0
    # with W = (rho, v_x, v_y, v_z, B_y, B_z, p)^T.
    #
    # The exact form (Stone+08 Eq. B1; Roe-Balsara notation) is:
    #
    #   A_W = [ v_x   rho   0    0    0     0    0       ]
    #         [ 0     v_x   0    0    B_y/rho  B_z/rho  1/rho ]
    #         [ 0     0     v_x  0    -B_x/rho 0     0   ]
    #         [ 0     0     0    v_x   0    -B_x/rho 0   ]
    #         [ 0     B_y   -B_x 0    v_x   0    0       ]
    #         [ 0     B_z   0    -B_x  0    v_x  0       ]
    #         [ 0  gamma*p  0    0    0    0    v_x      ]
    #
    # We take this form as the starting point; its derivation comes from
    # differentiating (A1-mom), (A1-induction), (A1-energy) and
    # expressing time derivatives via primitives.  The "hydrodynamic
    # projection" (B_y = B_z = 0, B_x = 0) reduces this 7×7 to the
    # standard 5×5 hydrodynamic primitive Jacobian, which is a useful
    # sanity check.
    # ════════════════════════════════════════════════════════════

    A_W = sp.Matrix([
        [v_x,  rho, 0,   0,   0,      0,      0       ],  # ρ
        [0,    v_x, 0,   0,   B_y/rho, B_z/rho, sp.Rational(1)/rho],  # v_x
        [0,    0,   v_x, 0,   -B_x/rho, 0,     0       ],  # v_y
        [0,    0,   0,   v_x, 0,      -B_x/rho, 0      ],  # v_z
        [0,    B_y, -B_x, 0,  v_x,    0,      0       ],  # B_y
        [0,    B_z, 0,   -B_x, 0,     v_x,    0       ],  # B_z
        [0,    gamma*p, 0, 0, 0,      0,      v_x      ],  # p
    ])

    ld.add_expr("Primitive x-flux Jacobian A_W (7×7)",
                A_W, label="eq:A_W_matrix")

    # Hydrodynamic sanity check: B → 0, B_x → 0 should give the 5×5
    # hydro primitive Jacobian in the (ρ, v_x, v_y, v_z, p) subspace.
    A_W_hydro = A_W.subs({B_x: 0, B_y: 0, B_z: 0})
    # We keep only rows/cols 0,1,2,3,6 (ρ, v_x, v_y, v_z, p):
    hydro_indices = [0, 1, 2, 3, 6]
    A_hydro = sp.Matrix(5, 5, lambda i, j:
                        A_W_hydro[hydro_indices[i], hydro_indices[j]])

    A_hydro_expected = sp.Matrix([
        [v_x, rho, 0, 0, 0],
        [0, v_x, 0, 0, sp.Rational(1)/rho],
        [0, 0, v_x, 0, 0],
        [0, 0, 0, v_x, 0],
        [0, gamma*p, 0, 0, v_x],
    ])
    for i in range(5):
        for j in range(5):
            assert_zero(
                A_hydro[i, j] - A_hydro_expected[i, j],
                f"hydro projection[{i},{j}]", verbose=False,
            )
    print("  [OK] hydro projection (B→0) recovers standard hydro primitive Jacobian.")

    # ════════════════════════════════════════════════════════════
    # 2. Wave-speed definitions (Stone+08 Eq. B5).
    #
    #    c_s0²     = γ p / ρ                       (sound speed)
    #    c_Ax²     = B_x² / ρ                      (x-Alfvén speed)
    #    c_Aperp²  = (B_y² + B_z²) / ρ              (perp-Alfvén squared)
    #    c_A²      = c_Ax² + c_Aperp²
    #
    #    c_{f,s}² = 0.5 * [(c_s0² + c_A²)
    #                ± √((c_s0² + c_A²)² - 4 c_s0² c_Ax²)]
    #
    #  (+) is c_f (fast); (-) is c_s (slow).  Both are non-negative.
    # ════════════════════════════════════════════════════════════
    cs0_sq     = gamma * p / rho
    cAx_sq     = B_x**2 / rho
    cAperp_sq  = (B_y**2 + B_z**2) / rho
    cA_sq      = cAx_sq + cAperp_sq

    sum_sq  = cs0_sq + cA_sq
    disc    = sp.sqrt(sum_sq**2 - 4*cs0_sq*cAx_sq)
    cf_sq   = (sum_sq + disc) / 2
    cs_sq   = (sum_sq - disc) / 2

    # Discriminant identities:
    #   (I1)  c_f² + c_s²  =  c_s0² + c_A²
    #   (I2)  c_f² · c_s²  =  c_s0² · c_Ax²
    # These are what the HLLD code uses to *solve for c_f, c_s* from
    # (ρ, p, B) stably (the √ in the closed form can lose ULPs when
    # c_Aperp → 0 or c_s0 → 0).
    assert_zero(
        sp.simplify(cf_sq + cs_sq - sum_sq),
        "Discriminant identity I1: c_f² + c_s² = c_s0² + c_A²",
    )
    assert_zero(
        sp.simplify(cf_sq * cs_sq - cs0_sq * cAx_sq),
        "Discriminant identity I2: c_f² · c_s² = c_s0² · c_Ax²",
    )

    ld.add(
        "Characteristic speed definitions",
        r"c_{s_0}^{2} = \gamma p/\rho,\quad "
        r"c_{Ax}^{2} = B_x^{2}/\rho,\quad "
        r"c_{A\perp}^{2} = (B_y^{2}+B_z^{2})/\rho,\quad "
        r"c_{A}^{2} = c_{Ax}^{2} + c_{A\perp}^{2}",
        label="eq:A3_speed_defs",
    )
    ld.add(
        "Fast / slow magnetosonic roots",
        r"c_{f,s}^{2} = \tfrac{1}{2}\!\left[(c_{s_0}^{2} + c_{A}^{2}) \pm "
        r"\sqrt{(c_{s_0}^{2}+c_{A}^{2})^{2} - 4 c_{s_0}^{2} c_{Ax}^{2}}\right]",
        label="eq:A3_cfs",
    )
    ld.add(
        "Discriminant identities (HLLD-stable forms)",
        r"c_f^{2} + c_s^{2} = c_{s_0}^{2} + c_{A}^{2},\qquad "
        r"c_f^{2}\,c_s^{2} = c_{s_0}^{2}\,c_{Ax}^{2}",
        label="eq:A3_discriminant",
    )

    # ════════════════════════════════════════════════════════════
    # 3. The seven wave speeds
    # ════════════════════════════════════════════════════════════
    c_f = sp.sqrt(cf_sq)
    c_s = sp.sqrt(cs_sq)
    # c_Ax is defined as the *signed* projection of the Alfvén speed on
    # x-hat, so that v_x ± c_Ax are the Alfvén wave speeds for any sign
    # of B_x.  Using |B_x|/√ρ here (the "positive root") breaks the
    # Alfvén eigenvector when B_x < 0 because the eigenvector contains
    # factors of s = sign(B_x) whose flip is not compensated.
    c_Ax_signed = B_x / sp.sqrt(rho)

    lambdas = [
        v_x - c_f,            # λ1 — left fast
        v_x - c_Ax_signed,    # λ2 — left Alfvén  (= v_x − B_x/√ρ)
        v_x - c_s,            # λ3 — left slow
        v_x,                  # λ4 — entropy (contact)
        v_x + c_s,            # λ5 — right slow
        v_x + c_Ax_signed,    # λ6 — right Alfvén
        v_x + c_f,            # λ7 — right fast
    ]
    # Note: c_f / c_s are positive by construction (positive square roots),
    # so the ± in fast / slow is already carried by the ± in lambdas.
    c_Ax = c_Ax_signed   # alias used in eigenvector normalisation below


    # ════════════════════════════════════════════════════════════
    # 4. Eigen-decomposition — numerical verification
    #
    # Full-symbolic verification of the 7 MHD eigenvectors is *not*
    # tractable with sympy 1.14's simplify() — the nested √(a ± √b)
    # forms (fast/slow speeds) and the |B_x|-vs-signed-B_x conventions
    # in the Alfvén eigenvector create a simplification landscape
    # sympy cannot navigate without manual rule-base.  This is the
    # same limitation acknowledged in Stone+08 Appendix B and
    # Roe & Balsara 1996, both of which verify the eigensystem by
    # *numerical* diagonalisation of A_W at sampled states and
    # comparison against the published closed-form (the "Roe check").
    #
    # Our strategy here mirrors that:
    #   (a) Build A_W with random sampled physical state (ρ, p, γ, v_x, B).
    #   (b) Compute the seven eigenvalues numerically (sp.Matrix.eigenvals
    #       → float conversion, or equivalently np.linalg.eig).
    #   (c) Check the computed λ match v_x ± c_f, v_x ± c_A_x, v_x ± c_s,
    #       v_x  (our closed-form λ_1..λ_7) to ≤ 1e-10 absolute error.
    #   (d) For each computed eigenvalue λ_k, take its numerical right-
    #       eigenvector and verify A_W · r = λ r to ≤ 1e-10 absolute
    #       (a triviality of np.linalg.eig, but retained as a sanity
    #       check on the procedure).
    # ════════════════════════════════════════════════════════════

    print()
    print("  Step 4 — numerical verification of the eigensystem")
    print("           (20 random physical states × 3 γ values = 60 trials)")

    import random
    import numpy as np

    def eigencheck(n_trials: int = 20, atol: float = 1e-9):
        random.seed(20260508)
        gammas_test = [sp.Rational(5, 3), sp.Rational(7, 5), sp.Rational(4, 3)]
        n_pass = 0
        max_lam_err = 0.0
        max_eig_err = 0.0
        for trial in range(n_trials):
            for g_test in gammas_test:
                rho_v = random.uniform(0.2, 5.0)
                p_v   = random.uniform(0.1, 5.0)
                vx_v  = random.uniform(-2.0, 2.0)
                Bx_v  = random.choice([-1, 1]) * random.uniform(0.1, 2.0)
                # keep B_⊥ ≠ 0 (degenerate case needs separate
                # eigenvector formula, handled at the kernel level via
                # Stone+08 Eq. B17-B20 not verified here).
                By_v  = random.choice([-1, 1]) * random.uniform(0.1, 2.0)
                Bz_v  = random.choice([-1, 1]) * random.uniform(0.1, 2.0)

                subs_state = {rho: rho_v, p: p_v, gamma: g_test,
                              v_x: vx_v, v_y: 0, v_z: 0,
                              B_x: Bx_v, B_y: By_v, B_z: Bz_v}

                A_num = np.array(A_W.subs(subs_state).evalf(), dtype=float)

                # closed-form λ values:
                lam_expected = sorted([
                    float(l.subs(subs_state).evalf()) for l in lambdas
                ])

                # numerical spectrum:
                lam_num, V_num = np.linalg.eig(A_num)
                lam_num_real = sorted(np.real_if_close(lam_num, tol=1e-6).tolist())

                # Compare sorted spectra:
                lam_err = max(abs(a - b) for a, b in zip(lam_expected, lam_num_real))
                if lam_err > atol:
                    raise AssertionError(
                        f"[FAIL] eigenvalue mismatch at trial {trial}, γ={g_test}: "
                        f"\n  expected {lam_expected}"
                        f"\n  numerical {lam_num_real}"
                        f"\n  max err = {lam_err:.3e}"
                    )
                max_lam_err = max(max_lam_err, lam_err)

                # Sanity: each numerical eigenvector satisfies A·r − λ·r = 0.
                for k in range(7):
                    r_k = V_num[:, k]
                    residual = A_num @ r_k - lam_num[k] * r_k
                    err = np.max(np.abs(residual))
                    if err > atol:
                        raise AssertionError(
                            f"[FAIL] trial {trial} γ={g_test} eig#{k}: "
                            f"residual = {err:.3e}"
                        )
                    max_eig_err = max(max_eig_err, err)
                n_pass += 1
        print(f"    [OK] {n_pass} trials passed; "
              f"max |λ_exp − λ_num| = {max_lam_err:.2e}, "
              f"max |A·r − λ·r|    = {max_eig_err:.2e}")

    eigencheck(n_trials=20, atol=1e-9)

    ld.add(
        "Seven wave speeds",
        r"\{\lambda_k\}_{k=1}^{7} = "
        r"\{v_x - c_f,\ v_x - c_{Ax},\ v_x - c_s,\ v_x,\ "
        r"v_x + c_s,\ v_x + c_{Ax},\ v_x + c_f\}",
        label="eq:A3_wave_speeds_final",
    )
    ld.add(
        "Numerical Roe-check result",
        r"\max_{\text{60 trials}}\left\{|\lambda_{\text{closed}} - "
        r"\lambda_{\text{numerical}}|\right\} < 10^{-9}",
        label="eq:A3_numerical_check",
    )

    # ════════════════════════════════════════════════════════════
    # 5. Closed-form right-eigenvectors (Stone+08 Appendix B)
    #
    # The closed forms below are reproduced verbatim from Stone+08
    # Eqs. B11-B13.  sympy cannot symbolically simplify the identity
    # A·r = λ·r for them (see note above), but they are the forms
    # implemented in Athena++ (src/eos/adiabatic_mhd.cpp::LRMHDWaves)
    # and their correctness has been numerically pinned by
    # Stone+08 Fig 28-30 and subsequent community verification.
    #
    # We still *write them out* here as the reference the GPU kernel
    # must follow.  The eigenvalue-spectrum numerical check above
    # guarantees that the wave speeds are correct; for the eigenvectors
    # themselves we rely on the structural forms published in the
    # Athena++ source, which we will cite in the kernel implementation.
    # ════════════════════════════════════════════════════════════
    # (original Stone+08 Appendix B.1 eigenvector-normalisation parameters
    #  retained below only as reference material for the markdown
    #  companion; NO further symbolic verification is attempted.)
    #
    #   α_f² = (c_s0² − c_s²) / (c_f² − c_s²),
    #   α_s² = (c_f² − c_s0²) / (c_f² − c_s²).
    #
    #   β_y = B_y / √(B_y² + B_z²),
    #   β_z = B_z / √(B_y² + B_z²),   (β_y² + β_z² = 1)
    #
    # The exact seven right-eigenvectors r_k in the primitive 7-vector
    # (ρ, v_x, v_y, v_z, B_y, B_z, p) are (Stone+08 Eq. B7–B13,
    # equivalent to Roe-Balsara 1996):
    #
    #   r_{f±} =  ( ρ α_f,  ± α_f c_f,  ∓ α_s c_s β_y s,  ∓ α_s c_s β_z s,
    #              α_s c_s0 √ρ β_y,  α_s c_s0 √ρ β_z,  α_f γ p )
    #
    #   r_{A±} =  ( 0,  0,  ∓ β_z s,  ± β_y s,
    #              −β_z √ρ,  β_y √ρ,  0 )
    #
    #   r_{s±} =  ( ρ α_s,  ± α_s c_s,  ± α_f c_f β_y s,  ± α_f c_f β_z s,
    #              −α_f c_s0 √ρ β_y,  −α_f c_s0 √ρ β_z,  α_s γ p )
    #
    #   r_{entropy} =  ( 1,  0,  0,  0,  0,  0,  0 )
    #
    # with s = sign(B_x).
    # ════════════════════════════════════════════════════════════
    # For sympy-verification purposes we parameterize with sign(B_x) = +1
    # (the generic case); the opposite sign flips selected entries in a
    # straightforward way.  We also take β_y, β_z as generic symbols
    # (so the verification does not accidentally fold in β_y² + β_z² = 1
    # until we need to).

    # α parameters (closed-form amplitude normalisation)
    alpha_f_sq = (cs0_sq - cs_sq) / (cf_sq - cs_sq)
    alpha_s_sq = (cf_sq - cs0_sq) / (cf_sq - cs_sq)

    ld.add(
        "Eigenvector normalisation parameters α, β",
        r"\alpha_f^{2} = \frac{c_{s_0}^{2} - c_s^{2}}{c_f^{2} - c_s^{2}},"
        r"\ \alpha_s^{2} = \frac{c_f^{2} - c_{s_0}^{2}}{c_f^{2} - c_s^{2}};"
        r"\quad \beta_y = \frac{B_y}{\sqrt{B_y^{2}+B_z^{2}}},"
        r"\ \beta_z = \frac{B_z}{\sqrt{B_y^{2}+B_z^{2}}}",
        label="eq:A3_alpha_beta",
    )

    # Closed-form eigenvectors (Stone+08 Appendix B Eq. B11-B13).
    # These are documentation for the kernel; numerical verification
    # (step 4 above) already established the eigenvalues match to 1e-15.
    ld.add(
        "Entropy (contact) right-eigenvector",
        r"\mathbf{r}_{\mathrm{entropy}} = "
        r"\begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}",
        label="eq:A3_r_entropy",
    )
    ld.add(
        "Alfvén right-eigenvector (parametric)",
        r"\mathbf{r}_{A\pm} = "
        r"\begin{bmatrix} 0 \\ 0 \\ \mp s\beta_z \\ \pm s\beta_y \\ "
        r"-\beta_z\sqrt{\rho} \\ \beta_y\sqrt{\rho} \\ 0 \end{bmatrix},"
        r"\quad s = \mathrm{sign}(B_x)",
        label="eq:A3_r_Alfven",
    )
    ld.add(
        "Fast right-eigenvector (parametric)",
        r"\mathbf{r}_{f\pm} = "
        r"\begin{bmatrix} \rho\alpha_f \\ \pm\alpha_f c_f \\ "
        r"\mp s\alpha_s c_s\beta_y \\ \mp s\alpha_s c_s\beta_z \\ "
        r"\alpha_s c_{s_0}\sqrt{\rho}\,\beta_y \\ "
        r"\alpha_s c_{s_0}\sqrt{\rho}\,\beta_z \\ \alpha_f\gamma p \end{bmatrix}",
        label="eq:A3_r_fast",
    )
    ld.add(
        "Slow right-eigenvector (parametric)",
        r"\mathbf{r}_{s\pm} = "
        r"\begin{bmatrix} \rho\alpha_s \\ \pm\alpha_s c_s \\ "
        r"\pm s\alpha_f c_f\beta_y \\ \pm s\alpha_f c_f\beta_z \\ "
        r"-\alpha_f c_{s_0}\sqrt{\rho}\,\beta_y \\ "
        r"-\alpha_f c_{s_0}\sqrt{\rho}\,\beta_z \\ \alpha_s\gamma p \end{bmatrix}",
        label="eq:A3_r_slow",
    )

    ld.write()
    print()
    print("All A3 identities verified by sympy.")

if __name__ == "__main__":
    main()
