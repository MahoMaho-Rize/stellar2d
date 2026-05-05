#!/usr/bin/env python3
"""Path C benchmark (Python): compare v-space vs φ-space discrete linear
operators to check whether path C can close the operator-mismatch gap
for Lane-Emden g-mode TD.

Test strategy:
  For each variant (v-space / φ-space), assemble the discretised linear
  anelastic evolution operator L_TD acting on (state) → (state after one
  linear RK3-consistent step).  Measure how close the EVP eigenvector
  V_EVP is to an eigenvector of L_TD by computing the projection
  residual  ‖L_TD·V - λ·V‖ / ‖V‖  over one step.

  If path C eliminates the mismatch, the φ-space residual should drop to
  ~1e-10 (machine-precision level) for Lane-Emden, while v-space residual
  stays at ~1e-4 (matching the CUDA dev/step).

Key insight (path B analysis):
  The EVP is always a discretisation of the same continuous operator.
  What differs is the TD discrete operator: v-space uses (D, diag(ρ),
  diag(ρ'/ρ), diag(N²)) assembled physically;  φ-space uses (D², diag(N²))
  only.  Path C works iff the TD operator in φ-space has EVP eigenvector
  as an exact discrete eigenvector.

For the linear anelastic Boussinesq reduction we derive below, the
discrete TD operator acts on reduced state (V̂, B̂) (y-grid vectors):

  ∂t V̂ = -Π' - (ρ'/ρ)·Π + B̂
  ∂t B̂ = -N²·V̂
  continuity:  -k ρ Û + ∂y(ρ V̂) = 0   ⇒   Û = ∂y(ρ V̂) / (kρ)
  momentum x:  ∂t Û = -k Π
  Combining (derivative of continuity):
     k² ρ Π = ∂y( ρ [-Π' - (ρ'/ρ)Π + B̂] ) − 0
           = -∂y(ρΠ') - ∂y(ρ'Π) + ∂y(ρ B̂)
  Let Π solve:  ∂y(ρΠ') + (kρ')·Π ...

Simpler: collapse to single-variable EVP (standard derivation):
    ω² [-(ρV̂)'' + k² ρ V̂] = k² N² ρ V̂       ← v-space EVP
  Substituting φ = ρV̂:
    ω² [-φ'' + k² φ] = k² N² φ                ← φ-space EVP (ρ gone)

TD-operator analogue (assembles same LHS/RHS in discrete form):
  v-space:  L_v V = -D·diag(ρ)·D V + k² diag(ρ) V
            R_v V = k² diag(N²·ρ) V
           ⇒ TD matrix M_v = L_v⁻¹ R_v,  ω² eigenvalue of M_v

  φ-space:  L_φ = -D² + k²I
            R_φ = k² diag(N²)
           ⇒ M_φ = L_φ⁻¹ R_φ

Both yield ω² as eigenvalues of M_v or M_φ.  In continuum limit they
agree; in discrete they differ because D·diag(ρ)·D ≠ ρ·D²  (variable-coef
discrete Leibniz failure).

Central benchmark:
    Given IC V_EVP in v-space variables from L_v EVP:
    Does the v-space TD (which uses D·diag(ρ)·D discretely) preserve it?
    ⇔  Is V_EVP an exact eigenvector of M_v?     YES, by construction.
  So "v-space EVP IC in v-space TD" has zero residual — but this is NOT
  what CUDA does:  CUDA uses D, diag(ρ), diag(ρ'/ρ) SEPARATELY in different
  operations (`apply_dy`, `∂y(ρV̂)` for divergence, diag(ρ'/ρ) as buoyancy
  pre-factor, etc).  The effective TD discretisation is NOT L_v⁻¹ R_v.

So the real question: does the v-space CUDA TD effective linear operator
have V_EVP as eigenvector?  Empirically NO (dev/step ~6e-4).
Does the φ-space analogue of CUDA TD have φ_EVP as eigenvector?
That's what this script measures.
"""
from __future__ import annotations
import argparse
import numpy as np
import scipy.linalg


# ── CGL grid, D matrix, CC weights ─────────────────────────────────────
def cgl_grid(ny: int, Ly: float):
    N = ny - 1
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0)**np.arange(N + 1))
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    idx = np.arange(N, -1, -1)
    y = (1.0 + x[idx]) * Ly / 2.0
    D = (2.0 / Ly) * D[np.ix_(idx, idx)]
    return y, D


def cc_weights(ny: int, Ly: float):
    N = ny - 1
    w = np.zeros(N + 1)
    for k in range(N + 1):
        s = 0.0
        for j in range(1, N // 2 + 1):
            b = 2.0 if (2 * j != N) else 1.0
            s += b / (4.0 * j * j - 1) * np.cos(2.0 * j * k * np.pi / N)
        w[k] = (1.0 - s) * 2.0 / N
    w[0] /= 2.0; w[-1] /= 2.0
    return w[::-1] * Ly / 2.0


# ── Backgrounds ────────────────────────────────────────────────────────
def background_boussinesq(y, Ly, N2_val=1.0):
    return np.ones_like(y), np.full_like(y, N2_val)


def background_lane_emden(y, Ly, rho_cut=0.01):
    from scipy.integrate import solve_ivp
    def rhs(xi, st):
        th, dth = st
        return [dth, -max(th, 0.0)**1.5 - 2*dth/max(xi, 1e-8)]
    sol = solve_ivp(rhs, [1e-6, 4.0], [1.0, 0.0], dense_output=True,
                    events=lambda t, st: st[0], max_step=1e-3)
    xi_s = sol.t_events[0][0]
    xi = np.linspace(1e-6, xi_s, 4000)
    rho_fine = np.clip(sol.sol(xi)[0], 0, None)**1.5
    mask = rho_fine > rho_cut
    xi_lo, xi_hi = xi[mask].min(), xi[mask].max()
    xi_q = xi_lo + (y / Ly) * (xi_hi - xi_lo)
    rho = np.clip(np.interp(xi_q, xi, rho_fine), rho_cut, None)
    drho = np.gradient(rho, y, edge_order=2)
    N2 = np.maximum(-drho / rho, 0.0)
    return rho, N2


# ── EVP assemblies (single variable) ───────────────────────────────────
def evp_vspace(D, rho, N2, kx):
    """L_v V = -(D ρ D - k² ρ) V,  R_v V = k² N² ρ V.  ω² eigenvalue."""
    ny = len(rho)
    L = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
    R = kx**2 * np.diag(N2 * rho)
    intr = slice(1, ny - 1)
    lam, V = scipy.linalg.eig(R[intr, intr], L[intr, intr])
    lam = np.real(lam); V = np.real(V)
    mask = np.isfinite(lam) & (lam > 0)
    lam, V = lam[mask], V[:, mask]
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    V_full = np.zeros((ny, V.shape[1])); V_full[1:-1] = V
    return lam, V_full, L, R


def evp_phi_space(D, N2, kx, ny):
    """L_φ φ = -D² φ + k² φ,  R_φ φ = k² N² φ.  ω² eigenvalue."""
    D2 = D @ D
    L = -D2 + kx**2 * np.eye(ny)
    R = kx**2 * np.diag(N2)
    intr = slice(1, ny - 1)
    lam, C = scipy.linalg.eig(R[intr, intr], L[intr, intr])
    lam = np.real(lam); C = np.real(C)
    mask = np.isfinite(lam) & (lam > 0)
    lam, C = lam[mask], C[:, mask]
    order = np.argsort(lam)[::-1]
    lam, C = lam[order], C[:, order]
    phi_full = np.zeros((ny, C.shape[1])); phi_full[1:-1] = C
    return lam, phi_full, L, R


# ── TD "effective linear operator" models ──────────────────────────────
def td_operator_vspace_faithful_cuda(D, rho, N2, kx):
    """Emulate the discrete linear operator that CUDA TD implements in
    v-space, using the SAME set of primitive discrete ops as the real code:

    RHS_V = B
    RHS_B = -N² V
    After RK-substep update, Chorin project:
       div = ρ·(-k U) + ρ·DV + ρ'·V
       but U is eliminated via continuity, so for a Chorin-free linear
       analysis of the (V, B) subspace we assemble:

    Semi-discrete ODE in (V, B):
       (V')  = [ 0    I  ] (V)      + projection correction
       (B')    [-N²   0  ] (B)
    The projection step removes divergent component; for pure kx-mode its
    effect is equivalent to replacing V with V_proj = V - D⁻¹[div/(-k²ρ + D ρ D ρ⁻¹)] ...

    To keep the benchmark clean, we use the *semi-discrete* momentum-only
    operator and check that V_EVP is its eigenvector modulo projection.

    Actually the cleanest thing to measure is:
       Let Lv = -D ρ D + k² ρ I,   Rv = k² N² ρ I   (as assembled by CUDA EVP).
       The discrete operator governing linear evolution of (V, B) on
       interior nodes is (block):

          d/dt (V)  =  (       0         I   ) (V)
                (B)     (-diag(N²)       0   ) (B)
       coupled to momentum projection via:
          (V_post) = V - M_proj · RHS
       where M_proj is determined by continuity.

    For this benchmark, the *eigenvector* of the combined (V, B) system
    that oscillates as e^{iωt} with ω² = EVP solution satisfies
          (Lv - ω² / k² · Rv) V = 0     ← this is the EVP!

    So V_EVP IS by construction the discrete eigenvector of the
    assembled-operator form.  The empirical CUDA 6e-4 dev/step must come
    from CUDA NOT using the assembled form — instead using separate D,
    diag(ρ), diag(ρ'/ρ) operations whose product drifts from
    D·diag(ρ)·D due to discrete non-commutativity.

    The only faithful CUDA-equivalent operator assembly is:
       L_cuda = -D ∘ diag(ρ) ∘ D - diag(ρ'/ρ) ∘ [projection_op for Π]
              ... which couples back via Poisson ...

    The bottom line from path B: the 6e-4 is reality of CUDA's DISCRETE
    Leibniz rule failure.  Assembling it cleanly in Python just
    reproduces EITHER the EVP exactly (if we use L_v R_v) OR something
    different (if we assemble primitively).  Let's do BOTH and compare.
    """
    # Method A: assemble momentum+continuity as a single matrix equation
    #  (this IS the EVP, so V_EVP is exact eigenvector)
    ny = len(rho)
    L = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
    R = kx**2 * np.diag(N2 * rho)
    # EVP eigenvalue problem R V = ω² L V → implicit 2nd-order ODE for V
    return L, R


def td_operator_phi_space(D, N2, kx, ny):
    D2 = D @ D
    L = -D2 + kx**2 * np.eye(ny)
    R = kx**2 * np.diag(N2)
    return L, R


def td_operator_vspace_primitive(D, rho, N2, kx):
    """Emulate CUDA's PRIMITIVE discrete operator: assembles L by applying
    D and diag(ρ) as separate matrix-vector ops, which is EXACTLY the same
    as L = -D·diag(ρ)·D + k²diag(ρ).  So it's literally identical to the
    EVP L.  This confirms that the CUDA 6e-4 dev/step CANNOT come from
    the L matrix — it must come from the IC reconstruction step where
    V is computed on CGL nodes and then U = ∂y(ρV)/(kρ) involves one
    extra D application whose discrete (ρV)' may not equal ρV' + ρ'V
    exactly at round-off."""
    ny = len(rho)
    L = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
    R = kx**2 * np.diag(N2 * rho)
    return L, R


# ── Central benchmark: TD-operator vs EVP eigenvalue residual ──────────
def residual_metric(variant, D, rho, N2, kx):
    """Compute  ‖(R - ω² L) V_EVP‖ / ‖V_EVP‖  for the TD discrete operator
    pair (L, R) associated with the variant."""
    ny = len(rho)
    if variant == "vspace":
        lam, V_full, L, R = evp_vspace(D, rho, N2, kx)
        V = V_full[:, 0]
        om2 = lam[0]
        # Residual with THE SAME L, R that gave the EVP: zero by construction
        r_evp = (R - om2 * L) @ V
        r_evp_norm = np.linalg.norm(r_evp) / max(np.linalg.norm(V), 1e-30)

        # Residual with ALTERNATE discretisation: -D² + k² (ignoring ρ variability)
        D2 = D @ D
        L_alt = -D2 + kx**2 * np.eye(ny)
        R_alt = kx**2 * np.diag(N2)
        # In v-space, this should give large residual for Lane-Emden
        r_alt = (R_alt - om2 * L_alt) @ V
        r_alt_norm = np.linalg.norm(r_alt) / max(np.linalg.norm(V), 1e-30)
        return {"omega_sq": om2, "r_consistent": r_evp_norm,
                "r_phi_operator": r_alt_norm}
    else:  # phi
        lam, phi_full, L, R = evp_phi_space(D, N2, kx, ny)
        phi = phi_full[:, 0]
        om2 = lam[0]
        r_evp = (R - om2 * L) @ phi
        r_evp_norm = np.linalg.norm(r_evp) / max(np.linalg.norm(phi), 1e-30)
        # cross-check: apply vspace operator to φ (this is NOT expected to work)
        L_alt = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
        R_alt = kx**2 * np.diag(N2 * rho)
        r_alt = (R_alt - om2 * L_alt) @ phi
        r_alt_norm = np.linalg.norm(r_alt) / max(np.linalg.norm(phi), 1e-30)
        return {"omega_sq": om2, "r_consistent": r_evp_norm,
                "r_vspace_operator": r_alt_norm}


# ── Single-step discrete TD simulation (primitive Euler, no projection) ─
def simulate_vspace(D, rho, N2, kx, V0, B0, dt, n_steps, w_cc):
    """Advance (V, B) with ∂t V = B, ∂t B = -N²V (pressure omitted).
    Measures: if V0 is a true eigenvector of this semi-discrete system,
    it rotates in (V, B) plane without shape change → dev stays tiny.
    Shape drift means D-driven advection/divergence mismatch matters."""
    ny = len(V0)
    V, B = V0.copy(), B0.copy()
    devs = [0.0]
    V_IC = V0.copy()
    V_IC_norm = np.sum(w_cc * V_IC**2)
    for _ in range(n_steps):
        # explicit midpoint step to emulate RK3 roughly
        dV = B
        dB = -N2 * V
        V = V + dt * dV
        B = B + dt * dB
        V[0] = 0.0; V[-1] = 0.0
        a = np.sum(w_cc * V * V_IC) / max(V_IC_norm, 1e-30)
        r = V - a * V_IC
        devs.append(np.sqrt(np.sum(w_cc * r**2) / max(V_IC_norm, 1e-30)))
    return devs


def simulate_phi_space(D, rho, N2, kx, phi0, Bphi0, dt, n_steps, w_cc):
    ny = len(phi0)
    phi, Bphi = phi0.copy(), Bphi0.copy()
    devs = [0.0]
    phi_IC = phi0.copy()
    phi_IC_norm = np.sum(w_cc * phi_IC**2)
    for _ in range(n_steps):
        dphi = Bphi
        dBphi = -N2 * phi
        phi = phi + dt * dphi
        Bphi = Bphi + dt * dBphi
        phi[0] = 0.0; phi[-1] = 0.0
        a = np.sum(w_cc * phi * phi_IC) / max(phi_IC_norm, 1e-30)
        r = phi - a * phi_IC
        devs.append(np.sqrt(np.sum(w_cc * r**2) / max(phi_IC_norm, 1e-30)))
    return devs


# ── Main driver ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--kx_int", type=int, default=1)
    ap.add_argument("--n_steps", type=int, default=15)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--amp", type=float, default=1e-8)
    args = ap.parse_args()

    y, D = cgl_grid(args.ny, args.Ly)
    w_cc = cc_weights(args.ny, args.Ly)
    kx = args.kx_int * 2 * np.pi / args.Lx

    for bg_name, (rho, N2) in [
        ("boussinesq", background_boussinesq(y, args.Ly, N2_val=1.0)),
        ("lane_emden", background_lane_emden(y, args.Ly, rho_cut=0.01))]:

        print(f"\n{'═' * 72}")
        print(f"  Background: {bg_name},  ny={args.ny},  kx={kx:.4f}")
        print('═' * 72)

        # ── (A) Cross-consistency of discrete operators ──
        print("\n  (A) Residual ‖(R - ω²L) V_EVP‖ / ‖V_EVP‖  for each variant:")
        res_v   = residual_metric("vspace", D, rho, N2, kx)
        res_phi = residual_metric("phi",    D, rho, N2, kx)
        print(f"    v-space EVP ω²  = {res_v['omega_sq']:.6e}")
        print(f"      self-consistent residual    = {res_v['r_consistent']:.3e}")
        print(f"      residual under φ operator   = {res_v['r_phi_operator']:.3e}")
        print(f"    φ-space EVP ω²  = {res_phi['omega_sq']:.6e}")
        print(f"      self-consistent residual    = {res_phi['r_consistent']:.3e}")
        print(f"      residual under v operator   = {res_phi['r_vspace_operator']:.3e}")

        # ── (B) Oscillator-only TD simulation ──
        print("\n  (B) ∂t(V,B) = (B, -N²V) eigenmode IC, measure dev/step:")
        # v-space IC
        lam_v, V_full, _, _ = evp_vspace(D, rho, N2, kx)
        om2_v = lam_v[0]; V0 = V_full[:, 0] / np.max(np.abs(V_full[:, 0])) * args.amp
        B0 = -N2 * V0 / om2_v
        devs_v = simulate_vspace(D, rho, N2, kx, V0, B0, args.dt,
                                  args.n_steps, w_cc)
        # φ-space IC
        lam_phi, phi_full, _, _ = evp_phi_space(D, N2, kx, args.ny)
        om2_phi = lam_phi[0]
        phi0 = phi_full[:, 0] / np.max(np.abs(phi_full[:, 0])) * args.amp
        Bphi0 = -N2 * phi0 / om2_phi
        devs_phi = simulate_phi_space(D, rho, N2, kx, phi0, Bphi0, args.dt,
                                      args.n_steps, w_cc)

        rate_v   = (devs_v[args.n_steps]   - devs_v[1])   / (args.n_steps - 1)
        rate_phi = (devs_phi[args.n_steps] - devs_phi[1]) / (args.n_steps - 1)
        print(f"    v-space TD: ω²={om2_v:.4e}  "
              f"dev/step ≈ {rate_v:.3e}   "
              f"(after {args.n_steps} steps: {devs_v[args.n_steps]:.3e})")
        print(f"    φ-space TD: ω²={om2_phi:.4e}  "
              f"dev/step ≈ {rate_phi:.3e}   "
              f"(after {args.n_steps} steps: {devs_phi[args.n_steps]:.3e})")


if __name__ == "__main__":
    main()
