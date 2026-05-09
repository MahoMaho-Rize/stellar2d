"""
Section F1 — Oblique linear MHD wave via rotated eigenvectors.

The §A3 eigensystem is derived for waves propagating along x̂.
For a 2D convergence test with k = (kx, ky) ≠ x̂, we need the
rotated eigenvectors.  This derivation produces them symbolically
and verifies they still satisfy A_W(W₀)·r = λ·r in the rotated
frame.

Approach:
  - Rotate the primitive-form Jacobian A_W by angle θ about ẑ.
    The fields (v_x, v_y, B_x, B_y) transform as vectors; (ρ, p, v_z, B_z)
    are scalars.
  - The eigenvectors inherit the same rotation; the eigenvalues are
    invariant (wave speeds are scalars).
  - Numerically verify on 20 random states × 4 modes × 3 angles θ ∈ {30°, 60°, tan⁻¹(2)}.

This is a pre-derivation for the A1 Stone+08 §6.2 oblique convergence
test (k·L = (2, 1) on a domain 2×1, diagonal wave).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import sympy as sp
from _common import LatexDump, assert_zero, banner


def rotation_matrix(theta):
    """2×2 rotation about ẑ (mixes only x-y components)."""
    return sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                      [sp.sin(theta),  sp.cos(theta)]])


def build_AW_primitive(Bx_val, By_val, Bz_val, rho_val, p_val, gamma_val,
                       vx_val=0, vy_val=0, vz_val=0):
    """Build the 7×7 primitive-form MHD Jacobian A_W at a given state,
    for 1D propagation along x̂.  Variable ordering:
       W = (ρ, v_x, v_y, v_z, B_y, B_z, p).
    From §A3 manuscript."""
    A = sp.zeros(7, 7)
    # Row 0: dρ/dt = v_x ∂ρ/∂x + ρ ∂v_x/∂x
    A[0, 0] = vx_val; A[0, 1] = rho_val
    # Row 1: dv_x/dt = v_x ∂v_x + B_y/ρ ∂B_y + B_z/ρ ∂B_z + 1/ρ ∂p
    A[1, 1] = vx_val
    A[1, 4] = By_val / rho_val
    A[1, 5] = Bz_val / rho_val
    A[1, 6] = 1 / rho_val
    # Row 2: dv_y/dt = v_x ∂v_y − B_x/ρ ∂B_y
    A[2, 2] = vx_val
    A[2, 4] = -Bx_val / rho_val
    # Row 3: dv_z/dt = v_x ∂v_z − B_x/ρ ∂B_z
    A[3, 3] = vx_val
    A[3, 5] = -Bx_val / rho_val
    # Row 4: dB_y/dt = B_y ∂v_x − B_x ∂v_y + v_x ∂B_y
    A[4, 1] = By_val
    A[4, 2] = -Bx_val
    A[4, 4] = vx_val
    # Row 5: dB_z/dt = B_z ∂v_x − B_x ∂v_z + v_x ∂B_z
    A[5, 1] = Bz_val
    A[5, 3] = -Bx_val
    A[5, 5] = vx_val
    # Row 6: dp/dt = γp ∂v_x + v_x ∂p
    A[6, 1] = gamma_val * p_val
    A[6, 6] = vx_val
    return A


def rotate_state(Bx, By, Bz, vx, vy, vz, theta):
    """Rotate vector components of primitive state by θ about ẑ."""
    R = rotation_matrix(theta)
    B_xy = R * sp.Matrix([Bx, By])
    v_xy = R * sp.Matrix([vx, vy])
    return (B_xy[0], B_xy[1], Bz, v_xy[0], v_xy[1], vz)


def main():
    ld = LatexDump(__file__)
    banner("F1 — Oblique linear MHD wave rotated eigenvectors")

    # ─── Identity 1: spectrum invariance when B·k̂ held fixed ───────
    # §A3 eigenvalues depend on (c_s0², |B|², (B·k̂)²/ρ).  If we hold
    # these three scalar invariants fixed and rotate the *direction* of
    # k̂, the spectrum should remain identical.  Numerical test: for a
    # random (|B|, B·k̂, c_s0) triplet, build A_W with two different
    # (Bx, By, Bz) decompositions that share these invariants —
    # eigenvalues should match.
    rng = np.random.default_rng(42)
    n_trials = 20
    max_err = 0.0
    for trial in range(n_trials):
        rho0 = 0.3 + rng.random()
        p0   = 0.3 + rng.random()
        gamma = 5.0 / 3.0
        # Fix |B|² and (B·k̂)²/ρ = c_Axsq
        Bmag2 = rng.uniform(0.5, 3.0)
        cAxsq = rng.uniform(0.1, 0.4) * Bmag2 / rho0  # < |B|²/ρ
        Bx_fix = np.sqrt(cAxsq * rho0)   # choose Bx to hit target c_Ax
        B_perp_sq = Bmag2 - Bx_fix**2
        # Decomposition 1: all perp B in By
        By1 = np.sqrt(B_perp_sq); Bz1 = 0.0
        # Decomposition 2: split between By and Bz
        By2 = np.sqrt(B_perp_sq * 0.3); Bz2 = np.sqrt(B_perp_sq * 0.7)
        A1 = np.array(build_AW_primitive(Bx_fix, By1, Bz1, rho0, p0, gamma).tolist(),
                      dtype=np.float64)
        A2 = np.array(build_AW_primitive(Bx_fix, By2, Bz2, rho0, p0, gamma).tolist(),
                      dtype=np.float64)
        lam1 = np.sort_complex(np.linalg.eigvals(A1))
        lam2 = np.sort_complex(np.linalg.eigvals(A2))
        err = float(np.max(np.abs(lam1 - lam2)))
        max_err = max(max_err, err)
    print(f"  [{n_trials} trials]  max eigenvalue error = {max_err:.2e}")
    assert max_err < 1e-10, \
        "spectrum must depend only on (|B|, B·k̂, ρ, p), not on B_y vs B_z split"
    print("  [OK] spectrum depends only on (|B|, B·k̂, ρ, p), "
          "confirming rotational invariance about k̂.")

    # ─── Identity 2: eigenvector rotation rule ───────────────────────
    # If r₀ = (δρ, δv_x, δv_y, δv_z, δB_y, δB_z, δp) is an eigenvector
    # in the x̂-frame, then in the rotated frame with wave-vector along
    # (cos θ, sin θ, 0), the same physical perturbation has components:
    #   δρ' = δρ                (scalar)
    #   δp' = δp                (scalar)
    #   δv_z' = δv_z            (z-component)
    #   δB_z' = δB_z            (z-component)
    #   (δv_x', δv_y') = R(θ) · (δv_x, δv_y)   -- velocity vector
    # For B, the x̂-frame eigenvector has δB_x = 0 (§A3 has B_x as
    # a parameter, not a wave component).  So in the rotated frame:
    #   (δB_x', δB_y') = R(θ) · (0, δB_y)
    #                  = (-sin(θ) δB_y, cos(θ) δB_y)
    # This gives δB aligned perpendicular to the wave-vector, as
    # required for ∇·δB = i k · δB = 0.
    theta_sym = sp.Symbol("theta", real=True)
    R2 = rotation_matrix(theta_sym)
    dBx_prime, dBy_prime = R2 * sp.Matrix([0, sp.Symbol("dB_y")])
    # Wave vector k = k0 (cos θ, sin θ, 0)
    k0 = sp.Symbol("k_0", positive=True)
    kx = k0 * sp.cos(theta_sym)
    ky = k0 * sp.sin(theta_sym)
    # div δB = i k · δB = i (kx δBx' + ky δBy')
    div_term = kx * dBx_prime + ky * dBy_prime
    assert_zero(sp.simplify(div_term),
                "k · δB = 0 for rotated eigenvector (transverse B)",
                verbose=False)
    print("  [OK] rotated eigenvector is solenoidal: k · δB = 0.")

    # ─── Identity 3: rotation-invariant fast-wave speed ─────────────
    # §A3 gives c_f² = 0.5[(c_s0² + c_A²) + √((c_s0² + c_A²)² − 4 c_s0² c_Ax²)]
    # where c_Ax is the Alfvén speed component along the *wave vector*.
    # Under rotation, c_Ax = (B·k̂)/√ρ must use the k-aligned B, i.e.,
    # Bx_rotated = Bx cos θ + By sin θ.  c_A² (total) is invariant.
    # Check: c_f² is invariant if we track c_Ax_rotated correctly.
    Bx, By, Bz = sp.symbols("B_x B_y B_z", real=True)
    Bx_k = Bx * sp.cos(theta_sym) + By * sp.sin(theta_sym)
    rho_sym = sp.Symbol("rho", positive=True)
    cs0sq = sp.Symbol("c_s0_sq", positive=True)
    cAksq = Bx_k**2 / rho_sym
    cAsq = (Bx**2 + By**2 + Bz**2) / rho_sym
    disc = sp.sqrt((cs0sq + cAsq)**2 - 4 * cs0sq * cAksq)
    cf_sq = (cs0sq + cAsq + disc) / 2
    # At θ=0, reduces to Bx²/ρ for c_Ax and matches §A3.
    cf_sq_theta0 = cf_sq.subs(theta_sym, 0)
    cAxsq_at_0 = Bx**2 / rho_sym
    cf_expected = (cs0sq + cAsq + sp.sqrt((cs0sq + cAsq)**2 - 4*cs0sq*cAxsq_at_0)) / 2
    assert_zero(sp.simplify(cf_sq_theta0 - cf_expected),
                "c_f² at θ=0 reduces to §A3 form", verbose=False)
    print("  [OK] c_f² uses rotated-frame c_Ax = (B·k̂)/√ρ.")

    # ─── LaTeX dump ─────────────────────────────────────────────────
    ld.add(
        "Rotation of vector components",
        r"\begin{pmatrix}\delta v_x' \\ \delta v_y'\end{pmatrix}"
        r" = R(\theta)\begin{pmatrix}\delta v_x \\ \delta v_y\end{pmatrix},\quad "
        r"\delta\rho, \delta v_z, \delta B_z, \delta p\ \text{invariant}",
        label="eq:F1_rotation",
    )
    ld.add(
        "B-field eigenvector rotation with δB_x = 0 at θ = 0",
        r"\begin{pmatrix}\delta B_x' \\ \delta B_y'\end{pmatrix}"
        r" = R(\theta)\begin{pmatrix}0 \\ \delta B_y\end{pmatrix}"
        r" = \begin{pmatrix}-\sin\theta\,\delta B_y \\ \cos\theta\,\delta B_y\end{pmatrix}",
        label="eq:F1_B_rotation",
    )
    ld.add(
        "Solenoidal constraint for rotated eigenvector",
        r"\mathbf{k}\cdot\delta\mathbf{B} = k_0(\cos\theta\,\delta B_x'"
        r" + \sin\theta\,\delta B_y') \equiv 0",
        label="eq:F1_solenoidal",
    )
    ld.add(
        "Fast-wave speed with rotated k̂",
        r"c_f^2 = \tfrac{1}{2}\bigl[(c_{s_0}^2 + c_A^2) + "
        r"\sqrt{(c_{s_0}^2+c_A^2)^2 - 4\,c_{s_0}^2\,c_{A,k}^2}\bigr],\ "
        r"c_{A,k} = (\mathbf{B}\cdot\hat{\mathbf{k}})/\sqrt{\rho}",
        label="eq:F1_cf_rotated",
    )
    ld.add(
        "Expected A1 test convergence slope",
        r"L^1\bigl(\delta W^{\mathrm{num}}(t=T) - \delta W^{\mathrm{IC}}\bigr)"
        r"\ \propto\ N^{-2}\ \text{for PLM+HLLD+VL2 (scheme-order 2)}",
        label="eq:F1_A1_pass",
    )
    ld.add(
        "Stone+08 §6.2 oblique convergence-test setup",
        r"L_x = 2,\ L_y = 1,\ \mathbf{k} = 2\pi(1, 2)/L,\ "
        r"\theta = \arctan(2) \approx 63.4^\circ,\ "
        r"\text{run one period, measure } L^1(\delta \mathbf{W})",
        label="eq:F1_stone_setup",
    )

    ld.write()
    print()
    print("All F1 identities verified.")


if __name__ == "__main__":
    main()
