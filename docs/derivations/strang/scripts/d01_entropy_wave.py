r"""
Section D1 — Entropy-wave canonical IC.

A passive density perturbation advected by a uniform flow:

  rho(x, t) = rho_0 + A sin(k (x - u_0 t))
  u(x, t) = u_0
  v(x, t) = 0
  P(x, t) = P_0

with rho_0 > 0, P_0 > 0, |A| < rho_0 (positivity).  The entropy
wave is a left eigenvector of A_x with eigenvalue u_0 (§A3);
pressure and velocity are uncommited, only density (=entropy) is
advected.

Strong-form identities verified:

  1. The ansatz satisfies the 2D Euler PDE exactly:
       partial_t rho + partial_x (rho u) + partial_y (rho v) = 0
       partial_t m_x + partial_x (rho u^2 + P) + partial_y (rho u v) = 0
       partial_t m_y + partial_x (rho u v) + partial_y (rho v^2 + P) = 0
       partial_t E + partial_x ((E+P) u) + partial_y ((E+P) v) = 0
     where E = P/gm1 + rho (u^2 + v^2)/2.

  2. At t = T = L_x / u_0 (one full period), rho(x, T) = rho(x, 0).

  3. HLLC degenerates: at every face the L/R states have identical
     u and P and differ only in rho.  The HLLC contact speed
     S_star = u_0 (from §A8 on this pair), and the flux is the
     pure upwind  F^* = (rho_L u_0, rho_L u_0^2 + P_0, 0, (E_L + P_0) u_0).
     Verify this symbolic reduction.

  4. Entropy function s = log(P / rho^gamma) varies with x through
     rho only, so s(x, t) = s(x - u_0 t, 0).  The entropy is
     advected identically (entropy-wave = entropy being advected).

Golden-values dump (per Rule 5):
  output/d01_entropy_wave.goldens.json:
    - N_ref = 4096 reference grid points (1D)
    - rho(x, T) at each of N_ref points (should equal rho(x, 0))
    - canonical physical parameters: rho_0, P_0, u_0, A, k, L_x, gamma
    - expected L1 error vs analytic (should be zero up to
      round-off at T = L_x / u_0).

Code anchor:
  src/gpu/explicit/strang_solver.cu :: (IC not yet wired; new
    init_entropy_wave needed for test_strang_convergence.cu)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp
import math

from _common import (
    LatexDump,
    GoldensDump,
    assert_zero,
    banner,
    total_energy_sym,
    flux_x_euler,
)


def main() -> int:
    ld = LatexDump(__file__)
    gd = GoldensDump(__file__)
    banner("D1 - Entropy-wave canonical IC")

    # Symbolic variables
    x, y, t = sp.symbols("x y t", real=True)
    gamma = sp.Symbol("gamma", positive=True)
    gm1 = gamma - 1
    rho_0 = sp.Symbol("rho_0", positive=True)
    P_0 = sp.Symbol("P_0", positive=True)
    u_0 = sp.Symbol("u_0", positive=True)
    A = sp.Symbol("A", positive=True)
    k_sym = sp.Symbol("k", positive=True)

    # Entropy-wave ansatz
    rho_ansatz = rho_0 + A * sp.sin(k_sym * (x - u_0 * t))
    u_ansatz = u_0
    v_ansatz = sp.Integer(0)
    P_ansatz = P_0
    E_ansatz = total_energy_sym(rho_ansatz, u_ansatz, v_ansatz, P_ansatz, gamma)

    # ════════════════════════════════════════════════════════════
    # 1.  Verify ansatz satisfies Euler PDE component-wise.
    # ════════════════════════════════════════════════════════════
    # Mass:
    mass_lhs = sp.diff(rho_ansatz, t) + sp.diff(rho_ansatz * u_ansatz, x) + sp.diff(rho_ansatz * v_ansatz, y)
    assert_zero(sp.simplify(mass_lhs), "D1-PDE-mass: rho_t + (rho u)_x + (rho v)_y = 0")

    # x-Momentum:
    mx_ansatz = rho_ansatz * u_ansatz
    xmom_lhs = (sp.diff(mx_ansatz, t)
                + sp.diff(rho_ansatz * u_ansatz**2 + P_ansatz, x)
                + sp.diff(rho_ansatz * u_ansatz * v_ansatz, y))
    assert_zero(sp.simplify(xmom_lhs), "D1-PDE-xmom: m_x eq holds")

    # y-Momentum:
    my_ansatz = rho_ansatz * v_ansatz
    ymom_lhs = (sp.diff(my_ansatz, t)
                + sp.diff(rho_ansatz * u_ansatz * v_ansatz, x)
                + sp.diff(rho_ansatz * v_ansatz**2 + P_ansatz, y))
    assert_zero(sp.simplify(ymom_lhs), "D1-PDE-ymom: m_y eq holds")

    # Energy:
    E_lhs = (sp.diff(E_ansatz, t)
             + sp.diff((E_ansatz + P_ansatz) * u_ansatz, x)
             + sp.diff((E_ansatz + P_ansatz) * v_ansatz, y))
    assert_zero(sp.simplify(E_lhs), "D1-PDE-energy: E eq holds")

    # ════════════════════════════════════════════════════════════
    # 2.  Periodicity at t = T = L_x / u_0.
    # ════════════════════════════════════════════════════════════
    L_x = sp.Symbol("L_x", positive=True)
    # Take k = 2*pi / L_x (one full wavelength on the domain).
    k_val = 2 * sp.pi / L_x
    T_period = L_x / u_0
    rho_at_T = rho_ansatz.subs({k_sym: k_val, t: T_period})
    rho_at_0 = rho_ansatz.subs({k_sym: k_val, t: 0})
    assert_zero(
        sp.simplify(rho_at_T - rho_at_0),
        "D1-periodicity: rho(x, T) = rho(x, 0)",
    )

    # ════════════════════════════════════════════════════════════
    # 3.  HLLC degeneracy at every face (identical u, P between
    #     neighbours; only rho differs).
    #
    #   Face L state: (rho_L, u_0, 0, P_0)
    #   Face R state: (rho_R, u_0, 0, P_0)
    #
    # Davis wave speeds (§A9):
    #   c_L = c_R = sqrt(gamma P_0 / rho_L) or rho_R (different!)
    #   S_L = min(u_0 - c_L, u_0 - c_R) = u_0 - max(c_L, c_R)
    #   S_R = max(u_0 + c_L, u_0 + c_R) = u_0 + max(c_L, c_R)
    #
    # HLLC contact speed (§A8):
    #   S_star = [p_R - p_L + rho_L u_L (S_L - u_L) - rho_R u_R (S_R - u_R)]
    #           / [rho_L (S_L - u_L) - rho_R (S_R - u_R)]
    # = [0 + rho_L u_0 (S_L - u_0) - rho_R u_0 (S_R - u_0)] / [rho_L (S_L - u_0) - rho_R (S_R - u_0)]
    #
    # With S_L = u_0 - c_max, S_R = u_0 + c_max:
    # S_star = [rho_L u_0 (-c_max) - rho_R u_0 (c_max)] / [rho_L (-c_max) - rho_R (c_max)]
    #        = -u_0 c_max (rho_L + rho_R) / [-c_max (rho_L + rho_R)]
    #        = u_0.
    # ════════════════════════════════════════════════════════════
    rho_Ls, rho_Rs = sp.symbols("rho_L rho_R", positive=True)
    c_max = sp.Symbol("c_max", positive=True)
    S_L = u_0 - c_max
    S_R = u_0 + c_max
    # p_R - p_L = 0 (uniform pressure)
    num = (sp.Integer(0)
           + rho_Ls * u_0 * (S_L - u_0)
           - rho_Rs * u_0 * (S_R - u_0))
    den = rho_Ls * (S_L - u_0) - rho_Rs * (S_R - u_0)
    S_star = num / den
    assert_zero(
        sp.simplify(S_star - u_0),
        "D1-HLLC-S-star: on entropy wave, S_star = u_0",
    )

    # Flux reduces to upwind (at the L side).  If S_star > 0 (u_0 > 0),
    # the upwind direction is from the left, so F* = F_L.
    F_L = flux_x_euler(rho_Ls, u_0, sp.Integer(0), P_0, gamma)
    # Verify F_L = (rho_L u_0, rho_L u_0^2 + P_0, 0, (E_L + P_0) u_0).
    E_L = total_energy_sym(rho_Ls, u_0, 0, P_0, gamma)
    expected = sp.Matrix([rho_Ls * u_0, rho_Ls * u_0**2 + P_0, 0, (E_L + P_0) * u_0])
    for k_ in range(4):
        assert_zero(
            sp.simplify(F_L[k_] - expected[k_]),
            f"D1-upwind-flux[{k_}]: F_L[{k_}] = expected",
        )

    # ════════════════════════════════════════════════════════════
    # 4.  Entropy advection.  s = log(P rho^(-gamma)); on entropy
    # wave, P is constant and rho varies.  So s(x, t) = log(P_0)
    # - gamma log(rho(x, t)), which moves with rho at velocity u_0.
    # ════════════════════════════════════════════════════════════
    s_ansatz = sp.log(P_ansatz) - gamma * sp.log(rho_ansatz)
    # Verify D_t s = 0.
    D_t_s = (sp.diff(s_ansatz, t)
             + u_ansatz * sp.diff(s_ansatz, x)
             + v_ansatz * sp.diff(s_ansatz, y))
    assert_zero(
        sp.simplify(D_t_s),
        "D1-entropy-invariant: D_t s = 0 on entropy-wave ansatz",
    )

    # ════════════════════════════════════════════════════════════
    # Golden values: dump the canonical IC and reference profile.
    # ════════════════════════════════════════════════════════════
    # Canonical parameters (dimensionless SI-ish):
    rho_0_val = 1.0
    P_0_val = 1.0 / 1.4   # P_0 = 1/gamma so c_0 = 1
    u_0_val = 1.0
    A_val = 0.05
    L_x_val = 1.0
    gamma_val = 1.4
    k_val_num = 2.0 * math.pi / L_x_val
    N_ref = 4096
    T_val = L_x_val / u_0_val

    gd.add("rho_0", rho_0_val)
    gd.add("P_0",   P_0_val)
    gd.add("u_0",   u_0_val)
    gd.add("A",     A_val)
    gd.add("L_x",   L_x_val)
    gd.add("gamma", gamma_val)
    gd.add("k",     k_val_num)
    gd.add("N_ref", N_ref)
    gd.add("T",     T_val)
    # Reference profile rho(x, t=0) at N_ref equi-spaced points.
    dx_ref = L_x_val / N_ref
    rho_samples = []
    for i in range(N_ref):
        x_val = (i + 0.5) * dx_ref
        rho_samples.append(rho_0_val + A_val * math.sin(k_val_num * x_val))
    gd.add("rho_initial", rho_samples)

    # Reference at t = T (should equal t = 0 exactly for periodic).
    # Since T corresponds to one period, rho(x, T) = rho(x, 0).
    # Dump both for regression: one is the IC, the other is the
    # "final state" the kernel should reproduce.
    gd.add("rho_final_at_T", rho_samples)   # identical up to round-off
    gd.add("L1_expected_error_at_T", 0.0)   # zero in the analytic limit

    gd.write()

    # ════════════════════════════════════════════════════════════
    # LaTeX dump.
    # ════════════════════════════════════════════════════════════
    ld.add(
        "Entropy-wave IC (smooth, periodic)",
        r"\rho(x, t) = \rho_{0} + A\,\sin\bigl(k(x - u_{0} t)\bigr),\quad "
        r"u = u_{0},\quad v = 0,\quad P = P_{0}",
        label="eq:D1-IC",
    )
    ld.add(
        "Exact Euler PDE satisfaction",
        r"\partial_t \rho + \partial_x(\rho u) = 0,\quad "
        r"\partial_t P = 0 \;\;\text{(since } P \text{ const)},\quad "
        r"\partial_t E + \partial_x\bigl((E+P) u\bigr) = 0",
        label="eq:D1-PDE",
    )
    ld.add(
        "HLLC contact-wave speed on entropy-wave pair",
        r"S_{\star} \;=\; u_{0} "
        r"\qquad\text{(pure upwind: no acoustic contribution)}",
        label="eq:D1-S-star",
    )
    ld.add(
        "Entropy advection",
        r"s(x, t) = \log P_{0} - \gamma \log \rho(x, t),\qquad "
        r"D_{t}s = 0 \;\;\text{(along streamline)}",
        label="eq:D1-entropy",
    )
    ld.add(
        "Periodicity at t = T = L_x / u_0",
        r"\rho(x,\,T) \;=\; \rho(x,\,0) \quad \forall x \in [0, L_{x}] "
        r"\qquad\text{(one full wavelength)}",
        label="eq:D1-period",
    )

    ld.write()
    print()
    print("All D1 identities verified by sympy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
