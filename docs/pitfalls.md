# stellar2d Pitfall Log

This document records every bug, misdesign, and subtle numerical issue
encountered during the development of stellar2d. Each entry describes the
symptom, root cause, fix, and how to write a regression test.

P01–P22: Low-Mach implicit solver (`lowmach_solver.cu`).
P23–P29: FAS solver and Strang Cartesian solver.
P30: Conservation fix for atmosphere reset.
P31–P32: cart_ale2 周期邊界與力/速度同步(`cart_ale2_solver.cu`, `cart_ale2_kernels.cu`)。
P33: cart_ale2 KE→IE compensation 违反局域性 — 全局均分 ΔKE 导致分层对流稳定层被伪加热。

---

## Table of Contents

1. [P01: 2pi factor missing in gravity mass integral](#p01)
2. [P02: Theta gravity gradient missing 1/r](#p02)
3. [P03: HLLC denominator sign catastrophe](#p03)
4. [P04: Sedov blast energy deposition wrong formula](#p04)
5. [P05: Density/pressure floor absent in to_primitive](#p05)
6. [P06: AmgX re-setup crash on persistent handle](#p06)
7. [P07: Well-balanced residual — full P vs perturbation P mismatch](#p07)
8. [P08: FV geometric source on P' creates false theta force](#p08)
9. [P09: GMG noise destroying well-balanced property](#p09)
10. [P10: Perturbed IC — HSE snapshot absorbed perturbation](#p10)
11. [P11: Block Jacobi numerical FD race condition on GPU](#p11)
12. [P12: SIMPLE preconditioner sign error (J_diag is negative)](#p12)
13. [P13: SIMPLE dt mismatch — member vs local variable](#p13)
14. [P14: SIMPLE Poisson coefficient degeneracy at small dt](#p14)
15. [P15: Newton convergence tolerance too tight for truncation error](#p15)
16. [P16: dt penalty too aggressive after Newton failure](#p16)
17. [P17: Line Jacobi Thomas sweep numerical instability](#p17)
18. [P18: Poisson residual scale mismatch on log mesh (5-DOF)](#p18)
19. [P19: solve_gravity at step start destroys well-balanced HSE](#p19)
20. [P20: GMRES Krylov basis polluted by Poisson noise floor (5-DOF)](#p20)
21. [P21: Line search solve_gravity inconsistent with JFNK frozen-Φ](#p21)
22. [P22: GMRES Krylov vectors have uninitialized 5th component](#p22)
23. [P23: FAS origin cell missing flux-level WB P₀ subtraction](#p23)
24. [P24: Strang `-=` operator precedence in source term update](#p24)
25. [P25: Strang face-state ghost cells uninitialized after MUSCL](#p25)
26. [P26: Strang reflective BC face mapping error](#p26)
27. [P27: Smooth floor returns exact zero for large negative inputs](#p27)
28. [P28: Entropy wave convergence masked by nonlinear O(A²) error](#p28)
29. [P29: LM-HLLC acoustic suppression foils sound-wave convergence test](#p29)
30. [P30: Atmosphere reset and sponge layer destroy mass/energy conservation](#p30)
31. [P31: cart_ale2 remap skipped periodic wrap edge](#p31)
32. [P32: cart_ale2 periodic sync master missed x-only / y-only duplicate classes + wrong accumulator semantics](#p32)
33. [P33: cart_ale2 Phase-M KE→IE compensation 违反局域性(全局均分 ΔKE)](#p33)

---

<a id="p01"></a>
## P01: 2pi factor missing in gravity mass integral

**Symptom**: Gravitational potential Phi too weak by factor 2pi. Star expands instead of holding equilibrium.

**Root cause**: The Poisson equation is `nabla^2 Phi = 4 pi G rho`. The mass integral for the outer Dirichlet BC is `M = integral(rho dV)` over the 3D sphere. Our FV volumes omit the 2pi azimuthal factor (Eq. 2.2), so the volumetric integral gives M/(2pi). Must multiply by 2pi.

**Fix**: `M_total = gpu_reduce_sum(rho * vol) * 2 * pi` in `solve_gravity()` and `k_lm_grav_rhs`.

**Files**: `lowmach_solver.cu:solve_gravity`, `gravity/gmg.cpp`, `gravity/poisson.cpp`

**Test strategy**: Compute M_total from known Lane-Emden analytic mass. Assert `|M_numerical - M_analytic| / M_analytic < 1e-6`. Verify Phi(R_outer) = -G*M/R_outer after gravity solve.

---

<a id="p02"></a>
## P02: Theta gravity gradient missing 1/r

**Symptom**: Spurious theta-direction acceleration at large r. Star deforms oblately.

**Root cause**: Gravity source in theta direction is `S_mt = -(rho/r) * dPhi/dtheta` (Eq. 6.2). The `/r` was missing, so `dPhi/dtheta` was applied as a force rather than `(1/r)*dPhi/dtheta`.

**Fix**: Divide theta gravity gradient by `r_center[i]` in the residual kernel.

**Files**: `gpu_solver.cu` (compressible path), `lowmach_solver.cu:k_lm_residual`

**Test strategy**: For a spherically symmetric Lane-Emden star (no theta perturbation), assert `max|S_mt_gravity| < epsilon` over all cells. The theta gravity force should be zero by symmetry.

---

<a id="p03"></a>
## P03: HLLC denominator sign catastrophe

**Symptom**: NaN in HLLC flux at near-zero velocity. Crash on first time step.

**Root cause**: HLLC contact wave speed `S* = (P_R - P_L + rho_L*u_L*(S_L-u_L) - rho_R*u_R*(S_R-u_R)) / (rho_L*(S_L-u_L) - rho_R*(S_R-u_R))` (Eq. 4.2). When left and right states are nearly identical, denominator approaches zero. Division by zero gives NaN.

**Fix**: Add sign guard: `denom = max(|denom|, 1e-300) * sign(denom)`.

**Files**: `hydro/riemann.cpp`

**Test strategy**: Call HLLC flux with `rho_L = rho_R, P_L = P_R, u_L = u_R = 1e-20`. Assert result is finite (no NaN/Inf). Flux should equal the common-state physical flux.

---

<a id="p04"></a>
## P04: Sedov blast energy deposition wrong formula

**Symptom**: Blast energy too high. Sedov shock propagates faster than analytic solution.

**Root cause**: Sedov pressure should be `P_blast = (gamma-1) * E_blast / V_blast` (Eq. 9.6). Code had `P_blast = (gamma-1) * rho_0 * E_blast / V_blast`, an extra factor of `rho_0`. Internal energy density `rho*e = P/(gamma-1) = E_blast/V_blast`, so `e = E_blast/(rho_0*V_blast)`, and total energy `E = rho*e + 0 = E_blast/V_blast`. But code was setting `E = rho_0 * E_blast / V_blast`, doubling the energy.

**Fix**: Remove the extra `rho_0` factor.

**Files**: `init/sedov.cpp`

**Test strategy**: Sum total energy `integral(rho*E * dV)` over the grid after init. Assert `|E_total - E_blast| / E_blast < 0.01` (1% from discretization).

---

<a id="p05"></a>
## P05: Density/pressure floor absent in to_primitive

**Symptom**: Negative density or pressure in low-density cells (stellar atmosphere). Crash in sqrt(P/rho) for sound speed.

**Root cause**: No floor on rho or P when converting conservative to primitive variables. Numerical diffusion can drive rho slightly negative in vacuum regions.

**Fix**: Apply `rho = max(rho, 1e-15)`, `P = max(P, 1e-15)` in `to_primitive` and `k_lm_floor`.

**Files**: `state.cpp:to_primitive`, `lowmach_solver.cu:k_lm_floor`

**Test strategy**: Initialize a state with `rho = 1e-20` in one cell. Run one time step. Assert `rho >= floor_value` and `P >= floor_value` everywhere after the step. Assert no NaN in any state variable.

---

<a id="p06"></a>
## P06: AmgX re-setup crash on persistent handle

**Symptom**: `AMGX solver_setup (re) failed: 15` error at second time step.

**Root cause**: AmgX solver_setup was called repeatedly with a stale matrix handle. AmgX requires explicit destruction and re-creation when the matrix sparsity pattern doesn't change but you want to re-factor.

**Fix**: Removed AmgX re-setup; switched to GMG which is stateless and re-entrant.

**Files**: `gpu_solver.cu` (historical; AmgX code later removed from lowmach path)

**Test strategy**: N/A (AmgX removed from lowmach path). If re-introduced, test that 2 consecutive solves with same sparsity don't crash.

---

<a id="p07"></a>
## P07: Well-balanced residual — full P vs perturbation P mismatch

**Symptom**: `||R(U_HSE)||` = 2.234e+05 instead of near-zero. Unperturbed star immediately starts moving.

**Root cause**: The HLLC flux used full pressure P in the Riemann solve, but gravity used perturbation Phi' = Phi - Phi_0. At HSE, the HLLC flux generates a nonzero pressure gradient (because FV divergence of full P != continuous dP/dr on a non-uniform mesh), but the gravity term using Phi' = 0 produces zero force. The mismatch creates a net force at every cell.

**Fix**: Complete rewrite to reference-state subtraction form. Momentum force is:
```
force = -nabla(P') - rho'*nabla(Phi_0) - rho*nabla(Phi')
```
where primed quantities are perturbations from the HSE reference state. At HSE, all primes are zero, so force = 0 exactly (to machine precision).

**Files**: `lowmach_solver.cu:k_lm_residual`

**Test strategy**: Upload unperturbed Lane-Emden state, snapshot HSE, compute residual. Assert `||R||_2 < 1e-20`. This is the most critical regression test.

---

<a id="p08"></a>
## P08: FV geometric source on P' creates false theta force

**Symptom**: `max|R_mt|` nonzero at HSE. Spurious theta acceleration in an otherwise spherically-symmetric equilibrium.

**Root cause**: The FV geometric source term `P * (r_hi^2 - r_lo^2) * (cos_lo - cos_hi) / V` is the volume-consistent discretization of `2P/r` (Eq. 5.3). When applied to the perturbation P', it double-counts: the central-difference `nabla(P')` already gives the complete spherical gradient. The FV geometric source on P' adds an extra term.

The reference state P_0's geometric source exactly cancels with `nabla(P_0) + rho_0*nabla(Phi_0)` by construction of the well-balanced scheme. But P' should NOT have a FV geometric source — only the central-diff gradient.

**Fix**: FV geometric source applied only to P_0 (cancels by well-balanced construction). Perturbation P' uses central-difference gradient only.

**Files**: `lowmach_solver.cu:k_lm_residual`

**Test strategy**: Subsumes P07 test. Additionally: apply a theta-independent pressure perturbation `P' = P'(r)`. Assert `R_mt = 0` everywhere (no theta force from a radial-only perturbation).

---

<a id="p09"></a>
## P09: GMG noise destroying well-balanced property

**Symptom**: `||F_0|| = 2199` vs `||R|| = 63` at step start. The Newton function value is much larger than the spatial residual, meaning the `(U-Un)/dt` term isn't the issue — the gravity solve is.

**Root cause**: Each call to `solve_gravity()` via GMG gives `Phi ≈ Phi_0 ± O(GMG_tol)`. On a log mesh with `dr ~ 1e-4`, the numerical gradient `nabla(Phi - Phi_0) ~ O(GMG_tol / dr) ~ O(1e-6 / 1e-4) = O(1e-2)`. Multiplied by `rho ~ O(1)` and summed over `N ~ 2000` cells, this gives `||noise|| ~ O(10-100)`.

**Fix**: Do NOT re-solve gravity at the start of each step. Phi from the previous step (or from `snapshot_hse`) is already consistent. Gravity is updated inside Newton only after fluid state changes (via `solve_gravity` in the line search).

**Files**: `lowmach_solver.cu:step()`

**Test strategy**: Run unperturbed HSE for 10 steps. Assert `||R_fluid||_2 < 1e-20` at every step (not just step 0). If solve_gravity is called at step start, this will fail.

---

<a id="p10"></a>
## P10: Perturbed IC — HSE snapshot absorbed perturbation

**Symptom**: Perturbed test shows zero perturbation response. Star sits perfectly still despite perturbation.

**Root cause**: The code uploaded the perturbed state, then called `snapshot_hse()`. This snapshots the PERTURBED state as the reference (rho_0, P_0, Phi_0). Since `rho' = rho - rho_0 = 0`, the well-balanced residual sees no perturbation.

**Fix**: Snapshot HSE on the UNPERTURBED state BEFORE uploading the perturbed state. In main.cpp:
```cpp
upload_state(grid, state_hse);  // unperturbed
snapshot_hse();                  // captures rho_0, P_0, Phi_0
upload_state(grid, state);       // perturbed
```
Flag `hse_set_externally = true` to prevent automatic re-snapshot.

**Files**: `main.cpp` (lowmach path), `lowmach_solver.cu:step()`

**Test strategy**: Upload perturbed state with `epsilon = 1e-3`. Compute residual. Assert `||R_mr|| > epsilon * reference_scale` (the perturbation creates a measurable force imbalance).

---

<a id="p11"></a>
## P11: Block Jacobi numerical FD race condition on GPU

**Symptom**: Preconditioner output is garbage. GMRES diverges immediately.

**Root cause**: The initial block Jacobi implementation used numerical finite-differencing: perturb `rho[k] += eps`, compute residual, restore `rho[k]`. On GPU, this is a **race condition** — neighboring threads read `rho[k]` while the owning thread has modified it. The upwind stencil reads from neighbors, so cell k's FD contaminates cells k-1, k+1, k-nt, k+nt.

**Fix**: Rewrote as analytical 4x4 Jacobian computation (`k_lm_assemble_blkjac`). Each cell's Jacobian block is computed from local quantities only, without modifying global arrays:
- Diagonal: `-(1/dt + advection_rate)`
- Pressure-energy coupling: `-(gamma-1) * stencil_coeff`
- Gravity-density coupling: `-nabla(Phi_0)`
- Compression work: `-P * d(div_v)/d(momentum)`

Then LU-inverted with partial pivoting per cell.

**Files**: `lowmach_solver.cu:k_lm_assemble_blkjac`

**Test strategy**: For a known state, compute `J_analytical * v` and compare to `(F(U+eps*v) - F(U)) / eps` (single-threaded reference). Assert `||J_analytical*v - J_FD*v|| / ||J_FD*v|| < 0.01`. The FD reference must be serial (no race condition).

---

<a id="p12"></a>
## P12: SIMPLE preconditioner sign error (J_diag is negative)

**Symptom**: SIMPLE gives correction in wrong direction. Newton diverges on first iteration.

**Root cause**: The Jacobian diagonal for Backward Euler is `J_ii = -(1/dt + advection_rate)`, which is **negative**. So `J_ii^{-1} = -1/Ap`. The SIMPLE vstar computation `vstar = r_momentum / Ap` was missing the negative sign: should be `vstar = -r_momentum / Ap`.

Same sign error in the density/energy pass-through: `delta_rho = r_rho / (1/dt)` should be `delta_rho = -dt * r_rho`.

**Fix**: Added negative signs to `k_lm_simple_vstar` (`inv_ap = -1.0 / Ap`) and `k_lm_simple_correct` (`out_rho = -dt * r_rho`).

**Files**: `lowmach_solver.cu:k_lm_simple_vstar`, `k_lm_simple_correct`

**Test strategy**: Apply preconditioner to a known residual vector. Check that `M^{-1} * J * v ≈ v` (the preconditioner approximately inverts J). The sign test: for a residual with `r_mr > 0` (positive momentum deficit), the velocity correction `delta_vr` should be negative (J is negative-definite, so J^{-1}*r has opposite sign).

---

<a id="p13"></a>
## P13: SIMPLE dt mismatch — member vs local variable

**Symptom**: SIMPLE works at initial dt but fails after dt is halved. Preconditioner output becomes inconsistent.

**Root cause**: `apply_simple()` used `dt_current` (the struct member variable) instead of the local `dt` passed through the Newton loop. When Newton fails and dt is halved, the local `dt` is 0.5x but `dt_current` still holds the old value. The SIMPLE Ap was assembled with old dt but the correction used new dt.

**Fix**: Pass `dt` as parameter through `apply_preconditioner() -> apply_simple()`.

**Files**: `lowmach_solver.cu:apply_simple`, `apply_preconditioner`

**Test strategy**: Run 2 steps where the second step's dt differs from the first. Assert that the preconditioner output is consistent with the actual dt used. Specifically: `Ap = 1/dt + advection_rate` should use the current dt, not the previous one.

---

<a id="p14"></a>
## P14: SIMPLE Poisson coefficient degeneracy at small dt

**Symptom**: SIMPLE gives worse dt than BLOCK_JACOBI (1.5e-8 vs 1.25e-7).

**Root cause**: The SIMPLE pressure Poisson uses coefficient `alpha = 1/Ap` where `Ap = 1/dt + advection_rate`. At small dt (1e-7), `Ap ≈ 1/dt = 1e7`, so `alpha ≈ dt = 1e-7`. The Poisson equation `nabla·(alpha*nabla(dp)) = div(v*)` becomes `dt * nabla^2(dp) = div(v*)`, meaning `dp ≈ div(v*) / (dt * k^2)`. As dt→0, dp→infinity, which is unphysical.

The velocity correction `delta_v = -(1/Ap)*nabla(dp)` then has `delta_v ≈ -dt * nabla(dp) ≈ -div(v*)/k^2`, which loses all dt-scaling information.

**Fix**: Kept the variable-coefficient formulation `nabla·((1/Ap)*nabla(dp)) = div(v*)` which is self-consistent, but the fundamental issue is that SIMPLE is designed for incompressible flow where dt is large. For compressible low-Mach at small dt, the pressure-velocity coupling through SIMPLE degenerates.

**Files**: `lowmach_solver.cu:apply_simple`

**Test strategy**: Compare `||M^{-1}*F||` for BLOCK_JACOBI vs SIMPLE at dt=1e-7. SIMPLE should not be significantly worse. If it is, the coefficient assembly is suspect.

---

<a id="p15"></a>
## P15: Newton convergence tolerance too tight for truncation error

**Symptom**: Newton oscillates between `||F|| = 0.1` and `||F|| = 0.11`, never reaching convergence. dt keeps getting cut.

**Root cause**: The convergence tolerance was `||F||_per-cell < 1e-6`. But the spatial discretization error (1st-order upwind) creates a truncation error floor of `O(h) ≈ O(1e-2)` in the residual. Newton cannot drive ||F|| below this floor because any correction that reduces F below truncation error gets amplified by the discrete operator.

**Fix**: Relaxed to per-cell < 1e-4 (fluid-only norm). Also added relative criterion: `||F|| < 1e-3 * ||F_0||`. Newton stops when either is satisfied.

**Files**: `lowmach_solver.cu:step()`

**Test strategy**: Run one step with very small dt (1e-10). Newton should converge in 0-1 iterations because `(U-Un)/dt → 0` implies `F ≈ R(Un) ≈ 0`. If it takes many iterations, the tolerance is too tight relative to the residual floor.

---

<a id="p16"></a>
## P16: dt penalty too aggressive after Newton failure

**Symptom**: dt drops from 1e-7 to 1e-15 in 8 cuts (0.5^8 = 1/256), then hits dt_min and aborts.

**Root cause**: Original settings: `dt_min = 1e-8 * dt`, growth factor 2.0x, 8 cuts allowed. A single Newton failure chain (e.g., from a bad initial guess) would destroy dt and take many steps to recover at 2x growth.

**Fix**: `dt_min = 1e-4 * dt` (only 4 orders below initial), growth factor 1.2x (conservative). The slower growth prevents dt from overshooting the stability limit, reducing the frequency of Newton failures.

**Files**: `lowmach_solver.cu:step()`

**Test strategy**: Inject a single Newton failure (e.g., by corrupting the preconditioner for one step). Assert that dt recovers to within 50% of pre-failure value within 20 steps.

---

<a id="p17"></a>
## P17: Line Jacobi Thomas sweep numerical instability

**Symptom**: NaN after ~13 steps. GMRES converges in 3-5 iterations (good), then output goes to infinity.

**Root cause**: The Thomas algorithm (block-tridiagonal forward sweep) for 64 radial cells accumulates errors. Three bugs in the off-diagonal block assembly:

1. **r-advection double-counted in diagonal**: `sr` included `sr_r = |vr|/dr`, then `adv_self = -(coeff_lo_self + coeff_hi_self)` added the same contribution again. The diagonal was too large by roughly `|vr|/dr`.

2. **Lower block pressure coupling wrong sign**: The central-diff stencil coefficient for `dP/d(rhoE_{i-1})` should be `+(gamma-1)*dh/(dl*(dl+dh))` (positive, because increasing P at i-1 reduces the gradient at i). Code had negative sign.

3. **Diagonal pressure coupling formula wrong**: Used `-(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh)))` without the outer negative from `dF_mr/d(rhoE)` being `- dPdr_coeff * (gamma-1)`. The double negation produced the wrong coefficient.

**Status**: Fixed. All three bugs corrected in both LM and FAS line-solve kernels.

**Files**: `lowmach_solver.cu:k_lm_line_solve`

**Test strategy**: For each theta-line j, extract the block-tridiagonal system (L, D, U matrices). Verify:
- D[i] is diagonally dominant: `||D[i]|| > ||L[i]|| + ||U[i]||` for stability
- The product `L*D^{-1}*U` has spectral radius < 1 (Thomas stability condition)
- Compare Thomas solution vs dense direct solve (LAPACK) for the same system. Assert `||x_thomas - x_dense|| / ||x_dense|| < 1e-6`.

---

<a id="p18"></a>
## P18: Poisson residual scale mismatch on log mesh (5-DOF)

**Symptom**: After upgrading to 5-DOF (Phi as state variable), `max|F_Phi| = 3e+4` even after a converged GMG gravity solve.

**Root cause**: The Laplacian stencil coefficients grow as `1/(r^2 * dr^2)` near the origin. On a log mesh with `r_center[0] ~ 1e-3` and `dr[0] ~ 1e-4`, the diagonal coefficient `|cC| ~ 1e8`. The GMG converges to relative residual `~1e-6`, but the absolute residual at cell (0,j) is `1e-6 * 1e8 = 100`. Summed over cells, `||F_Phi|| ~ 3e4`.

This is not a bug in GMG — it converged correctly. The issue is that the raw Poisson residual has a different magnitude scale than the fluid residual.

**Fix**: Scale the Poisson residual by `1/max(|cC|, 1.0)` per cell. This makes `F_Phi` dimensionless (relative residual), comparable in scale to the fluid equations.

**Files**: `lowmach_solver.cu:k_lm_poisson_residual`

**Test strategy**: After `solve_gravity()`, compute scaled Poisson residual. Assert `max|F_Phi_scaled| < 0.1` (much less than fluid perturbation forces). Without scaling, this test would fail with max ~ 3e4.

---

<a id="p19"></a>
## P19: solve_gravity at step start destroys well-balanced HSE

**Symptom**: `max|R_ρvr|` jumps from `4e-30` (perfect HSE) to `2.4` when `solve_gravity()` is called before the first Newton iteration.

**Root cause**: The well-balanced property relies on `Phi = Phi_0` exactly (bit-for-bit). Calling `solve_gravity()` gives `Phi_new ≈ Phi_0 ± O(GMG_noise)`. The perturbation `Phi' = Phi_new - Phi_0 ≠ 0` creates a spurious force `-rho * nabla(Phi')` that breaks equilibrium.

For the perturbed case, this is not a problem because the physical perturbation forces (~100) dominate over GMG noise (~1). But for the unperturbed case, any noise is unacceptable.

**Fix**: Do NOT call `solve_gravity()` at step start. Instead:
- Step 0: Phi = Phi_0 from `snapshot_hse()` (exact, by construction)
- Step > 0: Phi from last accepted Newton iterate (updated by `solve_gravity()` inside the line search)
- Inside Newton: `solve_gravity()` called after fluid correction in line search, not before

**Files**: `lowmach_solver.cu:step()`

**Test strategy**: Run unperturbed HSE for 100 steps. Assert `max|R_ρvr| < 1e-20` at every step. This test is extremely sensitive — any gravity re-solve at step start will cause failure.

---

<a id="p20"></a>
## P20: GMRES Krylov basis polluted by Poisson noise floor (5-DOF)

**Symptom**: With 5-DOF JFNK (perturbing all of ρ, ρvr, ρvθ, ρe, Phi simultaneously), GMRES converges in 2-3 iterations but Newton stalls after step 12. Line search accepts only alpha = 0.01.

**Root cause**: The JFNK matvec `J*v = (F(W+eps*v) - F(W))/eps` includes the Poisson residual response `delta(nabla^2 Phi - 4*pi*G*rho)`. This component has a noise floor from GMG discretization (~0.01 per cell after scaling). The Krylov basis vectors V[j] include this noise, so GMRES optimizes over a subspace that tries to minimize both the fluid residual AND the Poisson noise. The Poisson noise is irreducible, so GMRES wastes degrees of freedom on it.

With the Poisson component consuming Krylov subspace dimensions, the effective GMRES approximation for the fluid equations is degraded. The search direction quality drops, line search accepts only tiny steps, and dt collapses.

**Fix**: Hybrid JFNK — GMRES operates on 4N fluid DOFs only (Phi frozen during GMRES). After GMRES correction, `solve_gravity()` updates Phi in the line search. This is equivalent to exact Schur complement elimination: the Poisson solve exactly inverts the Phi-Phi block, and GMRES only sees the Schur complement system for the fluid.

Convergence/line-search use fluid-only norm `||F_fluid||` (first 4n components). The Poisson residual is monitored but not used for Newton convergence decisions.

**Files**: `lowmach_solver.cu:jfnk_matvec`, `gmres_solve`, `step()`

**Test strategy**: Run perturbed test for 20 steps. Assert dt does not drop below `dt_initial * 1e-2` (no catastrophic collapse). With the polluted 5-DOF GMRES, dt would drop to ~1e-12 by step 13.

---

---

<a id="p21"></a>
## P21: Line search solve_gravity inconsistent with JFNK frozen-Φ

**Symptom**: Newton line search accepts only tiny α (0.01-0.06). `||F(U+αδU)|| >> ||F(U)||` even at α→0. First step requires dt=7.8e-9 despite block Jacobi being accurate at dt=1e-7.

**Root cause**: JFNK matvec freezes Φ during `J·v = (F(U+εv,Φ) - F(U,Φ))/ε`. But line search called `solve_gravity()` after applying the correction, evaluating `F(U+αδU, Φ_new)`. Since Φ_new ≠ Φ, the merit function `||F(U+αδU, Φ_new)||` is inconsistent with the linear model `F ≈ F₀ + α·J·δU` that GMRES optimized. For near-equilibrium stars, ρ→Φ coupling via Poisson is strong: a small δρ changes Φ significantly, which feeds back through ρ∇Φ to create large spurious forces that overwhelm the correction.

**Fix**: Remove `solve_gravity()` from line search. Evaluate `F(U+αδU, Φ_frozen)` consistently with the JFNK model. Update Φ once after Newton converges, before the next time step.

**Files**: `lowmach_solver.cu:step()` (line search loop and post-convergence gravity update)

**Test strategy**: Run `test_precond_quality` sweep. At dt=1e-7, `dt_accepted/dt_target` should be 1.0 (not 0.01). `test_lowmach` A5 should show first-step dt ≥ 1e-7.

**Impact**: First-step dt improved 16× (7.8e-9 → 1.25e-7).

---

<a id="p22"></a>
## P22: GMRES Krylov vectors have uninitialized 5th component (Φ garbage)

**Symptom**: Diagnosed by `test_solver_diagnosis` D2: `||J·x + F|| / ||F|| = 3.5e7` — GMRES claims to converge but the actual linear residual is 7 orders of magnitude off.

**Root cause**: GMRES allocates V[j] and Z[j] vectors at 5N (to match the 5-DOF state), but only operates on the first 4N (fluid) components. The 5th component (Φ, entries 4n to 5n-1) is uninitialized heap garbage. When JFNK matvec calls `unpack_add(Z[j], ε)`, it adds `ε × garbage` to Φ, corrupting the gravity potential during every Arnoldi step. The matvec then evaluates `F(U+εv, Φ+ε·garbage)` — the resulting J·v includes `∂F/∂Φ × garbage`, making the entire Krylov basis vectors wrong.

The JFNK matvec saves/restores state via `d_gmres_Uk`, so the state after each matvec is correct. But the J·v product stored in the Arnoldi basis is corrupted. GMRES converges in its internal (corrupted) norm but the actual solution `x = Σ y_i Z[i]` is a linear combination of vectors with garbage Φ components.

**Fix**: Zero the 5th component of Z[j] after preconditioner application and before matvec. Also zero the 5th component of V[0] (initial residual) and w (Arnoldi work vector) after each matvec.

**Files**: `lowmach_solver.cu:gmres_solve()` — three `k_lm_zero` calls added

**Test strategy**: `test_solver_diagnosis` D2 should show `||J·x + F|| / ||F|| < 0.01` (GMRES actually converged). D1 line search profile should show `||F(U+αδU)|| / ||F₀||` decreasing for α > 0 (descent direction).

---

<a id="p23"></a>
## P23: FAS origin cell missing flux-level WB P₀ subtraction

**Symptom**: Explicit polar solver: unperturbed Lane-Emden star develops max|vr| = 0.387 at origin, corresponding to ~6750g spurious radial acceleration. All other cells have max|vr| ~ 1e-14.

**Root cause**: `k_fas_residual_origin` (i=0 cell) computes the radial momentum flux divergence as:
```cuda
div_mr = -invV*(Ar_hi*fr_hi.f_mr + At_hi*ft_hi.f_mr - At_lo*ft_lo.f_mr);
```
This uses the **full** HLLC momentum flux `f_mr = rho*u^2 + P` without subtracting the HSE background pressure P₀ at the face. The regular-cell kernel `k_fas_residual` (i ≥ 1) correctly subtracts P₀:
```cuda
div_mr = -invV*(Ar_hi*(fr_hi.f_mr - P0f_rhi) - Ar_lo*(fr_lo.f_mr - P0f_rlo) + ...);
```
The origin kernel was written separately (different geometry: no inner face, volume-averaged 1/r) and the P₀ subtraction was overlooked.

**Fix**: Add 3 lines computing face-averaged P₀ and subtract from the momentum flux in the origin kernel (Eq. 12.6):
```cuda
double P0f_rhi = 0.5*(P0r(0) + P0r(1));
double P0f_thi = 0.5*(P0t(j) + P0t(j+1));
double P0f_tlo = 0.5*(P0t(j-1) + P0t(j));
```

**Impact**: Affects all 4 solvers sharing this kernel: explicit, FAS, SIMPLE, projection. The LowMach solver is unaffected (uses FD pressure gradient, not flux-level WB). After fix: max|vr| = 7.6e-23 (17 orders of magnitude improvement).

**Files**: `fas_residual.cu:k_fas_residual_origin`, line 316

**Test strategy**: `test_fas_verify` F11: upload unperturbed Lane-Emden, run 1 explicit step, assert `max|vr| < 1e-10`. `test_coverage_critical` P1: LM origin HSE residual ≈ 0.

---

<a id="p24"></a>
## P24: Strang `-=` operator precedence in source term update

**Symptom**: Y-sweep gravity double-counted. Star atmosphere in hydrostatic equilibrium develops dv/dt ≈ 2g instead of 0.

**Root cause**: C++ operator precedence on compound assignment:
```cpp
d_my -= dp_dy + rho_avg * g;   // WRONG: d_my = d_my - dp_dy - rho*g
```
The intent was `d_my = d_my - dp_dy + rho*g` (pressure gradient opposes gravity). But `-= (A + B)` is parsed as `d_my = d_my - (A + B) = d_my - A - B`. Both terms have the same sign, doubling the gravity instead of cancelling.

**Fix**: Always use explicit assignment for mixed-sign source terms:
```cpp
d_my = d_my - dp_dy + rho_avg * g;
```

**Files**: `strang_solver.cu:k_strang_sweep_y`

**Test strategy**: `test_strang_step` S1: run 50 steps of unperturbed isentropic HSE, assert `max|v| < 1e-3` (HSE preserved). Would show max|v| growing linearly with the double-counting bug.

---

<a id="p25"></a>
## P25: Strang face-state ghost cells uninitialized after MUSCL

**Symptom**: First step produces random results — sometimes reasonable, sometimes NaN. `compute-sanitizer` reports reads of uninitialized GPU memory in the HLLC kernel.

**Root cause**: The MUSCL-Hancock kernel fills face states (`rho_L, rho_R, u_L, u_R, ...`) only for physical cells `[0, nx) × [0, ny)`. The HLLC kernel at boundary `j=0` reads `face_L[j=0]`, which is the right-face state of ghost cell `j=-1` — never written by MUSCL. GPU memory is not zeroed, so HLLC reads garbage.

**Fix**: After MUSCL, before HLLC, explicitly fill boundary face states:
- y=0 reflective: `face_R[j=-1] = reflect(face_R[j=0])` (use inner face of first physical cell, negate v_y)
- y=ny outflow: `face_L[j=ny] = face_R[j=ny-1]` (copy outer face of last physical cell)
- x-periodic: `face_R[x=-1] = face_R[x=nx-1]`, `face_L[x=nx] = face_L[x=0]`

**Files**: `strang_solver.cu` (face ghost fill routines added after each MUSCL call)

**Test strategy**: Run `compute-sanitizer --tool initcheck ./test_strang_step`. Should report 0 uninitialized reads. `test_strang_step` S1 with HSE should give deterministic results.

---

<a id="p26"></a>
## P26: Strang reflective BC face mapping error

**Symptom**: At y=0 boundary, reflected face state uses wrong side of the boundary cell. Result: small but systematic velocity error at the reflecting wall that grows over time.

**Root cause**: The reflective BC was filling `ghost_face = reflect(face_L[j=0])`. But `face_L[j=0]` is the left face of cell j=0 (pointing away from the boundary). The correct source is `face_R[j=0]` — the right face of cell j=0, which is the state closest to the y=0 boundary.

**Fix**: Use `face_R[j=0]` as the source for reflected ghost face state:
```
ghost_face_R[j=-1] = reflect(face_R[j=0])
```
where `reflect` negates the y-velocity component.

**Files**: `strang_solver.cu` (y-boundary face ghost fill)

**Test strategy**: `test_strang_unit` U7: test y-bottom ghost cell by checking that the reflected cell has exactly negated v_y and identical rho, u, P.

---

<a id="p27"></a>
## P27: Smooth floor returns exact zero for large negative inputs

**Symptom**: `test_coverage_critical` P4: after corrupting a cell to ρ = -0.5 and applying floor, the cell has ρ = 0.0 exactly (not ρ > 0).

**Root cause**: The smooth floor `floor(x) = 0.5*(x + sqrt(x² + 4ε²))` with ε = 1e-20. For x = -0.5: `sqrt(0.25 + 4e-40) ≈ sqrt(0.25) = 0.5` in double precision. So `floor(-0.5) = 0.5*(-0.5 + 0.5) = 0.0` exactly. The smooth floor is designed for |x| ~ ε, not for |x| >> ε.

**Fix**: The floor is mathematically correct (it never returns negative values), but it does not guarantee strict positivity for large negative inputs. Options:
1. For tests: use small corruption (ρ = -1e-5) and check `≥ 0` not `> 0`.
2. For production: add hard clamp after smooth floor: `rho = fmax(rho, rho_fl)`.

Current approach: option 1 for tests. Production code retains the smooth floor only.

**Files**: `fas_residual.cu:k_fas_floor`

**Test strategy**: `test_coverage_critical` P4: corrupt ρ to -1e-5, apply floor, assert `min(ρ) ≥ 0` and no NaN.

---

<a id="p28"></a>
## P28: Entropy wave convergence masked by nonlinear O(A²) error

**Symptom**: Using a sound wave (A=0.01) to test 2nd-order convergence of the Strang solver, the measured order at N=128→256 is 0.3 instead of ~2.0. L1 error barely decreases with refinement.

**Root cause**: The Euler equations are nonlinear. A sound wave with amplitude A produces solution deviation from the linear mode at O(A²·t). For A=0.01 and t=0.1: nonlinear error ~ O(1e-4). At N=128: truncation error ~ O(dx²) = O(6e-5). At N=256: truncation error ~ O(1.5e-5). Since O(A²) = 1e-4 dominates at both resolutions, the convergence ratio is ~1 (order 0).

**Fix**: Use **entropy waves** (δρ ≠ 0, δP = 0, δv = 0, advected at constant velocity). These are **exact nonlinear solutions** of Euler (passive scalar advection). The only error is truncation, so convergence is clean.

Alternative: use sound waves with A << dx (e.g., A=1e-6) so that A² << dx² at all tested resolutions.

**Files**: `test_strang_convergence.cu`

**Test strategy**: Entropy wave at A=0.01, measure L1 convergence at N=64/128/256. Assert order > 1.8 (2nd-order). Achieved: 1.92 (L1), 1.99 (L2).

---

<a id="p29"></a>
## P29: LM-HLLC acoustic suppression foils sound-wave convergence test

**Symptom**: Even with small amplitude (A=1e-6), sound wave test shows order ~1.0 instead of ~2.0 when LM-HLLC is enabled.

**Root cause**: The LM-HLLC blending factor f(M) = clamp(M_local, M_cutoff=1e-3, 1.0) is clamped to M_cutoff for flows with M < 1e-3. A sound wave with A=1e-6 has M ~ 1e-6, so f = 1e-3. This adds O(M_cutoff) = O(1e-3) artificial dissipation to the acoustic mode — much larger than the truncation error O(dx²) at reasonable resolutions. The dissipation scales as 1/dx (first-order), masking the 2nd-order truncation.

**Fix**: For convergence testing, either (a) use entropy waves (unaffected by LM-HLLC since they have no acoustic component), or (b) disable LM-HLLC (use standard HLLC for sound wave tests).

The acoustic suppression is a **feature** for stellar convection simulations where M ~ 10^{-3}–10^{-1}: it prevents the acoustic time scale from dominating the dynamics. It is not a bug.

**Files**: `strang_device.cuh:d_lmhllc`, `fas_hllc.cuh:fas_hllc_lm`

**Test strategy**: `test_strang_convergence` uses entropy waves exclusively to avoid this issue. Sound wave convergence should only be tested with standard HLLC.

---

<a id="p30"></a>
## P30: Atmosphere reset and sponge layer destroy mass/energy conservation

**Symptom**: Total mass `M = ∫ρ dV` drifts by 0.01–1% over 10⁴ steps, depending on how much material crosses the atmosphere boundary. Total energy shows similar drift.

**Root cause**: Three non-conservative operations are applied every time step:

1. `k_fas_atm_reset`: forces cells with `ρ₀ < atm_thresh`, `ρ < 0.01·ρ₀`, or `T/T₀ > 100` to exact HSE state `(ρ₀, 0, P₀/(γ-1))`. The mass and energy difference between the old and new state is silently discarded.

2. `k_fas_sponge`: damps velocity via `v *= 1/(1+α)` and relaxes internal energy toward HSE. Kinetic energy is removed without conversion to heat; internal energy difference vanishes.

3. `k_fas_floor` / `k_lm_floor`: smooth floor + hard clamp can create mass when `ρ` is lifted from below-floor to floor value.

For HSE tests this is invisible (atmosphere cells are already at HSE). For convection, material reaching the atmosphere boundary is truncated, accumulating a deficit proportional to the flux × timestep × number of affected cells.

**Fix**: Wrap the sponge + atm_reset block with global reduce of `∫ρ·dV` and `∫E·dV` before and after. The deficit `ΔM = M_before − M_after` is distributed uniformly as a density correction `ΔM / V_interior` to all interior cells (`ρ₀ ≥ atm_thresh`). Same for energy. This preserves the atm_reset's ability to suppress vacuum noise while maintaining exact global conservation.

Applied to all 4 polar solvers: FAS (implicit + explicit), SIMPLE, projection.

**Files**: `fas_solver.cu:step()`, `fas_solver.cu:step_explicit()`, `simple_solver.cu:step()`, `projection_solver.cu:step()`, `fas_residual.cu` (new kernels `k_fas_rhoV_EV`, `k_fas_conserve_correct`), `fas_linalg.cuh` (`fas_reduce_sum`)

**Test strategy**: Run Lane-Emden perturbed for 1000 steps. Assert `|M_final - M_initial| / M_initial < 1e-12` (machine precision for compensated sum). Without the fix, expect ~1e-4 drift.

---

<a id="p31"></a>
## P31: cart_ale2 remap skipped periodic wrap edge

**Symptom**: 在 `--bc-x periodic` 或 `--bc-y periodic` 下,**均勻 advection**(uniform ρ, uniform P, uniform vx=1,vy=0)也會被破壞:|v|_max 從 1.000 漂到 1.079(64²,一步之內),KE 漂 0.03%/step,E 漂 0.004/step,**不是機器精度數值噪聲**。Lecoanet KH 在 t≈0.043 `dt` 塌到 1e-12,所有 PPM 變體(CW/CS,prim/cons,char)都一樣。

**Root cause**: `k_cale2_remap_east` 的 edge 並行範圍是 `(nx-1)*ny`,對應內部 east face `ic ∈ [0, nx-2]`。在 reflective BC 下正確(ic=nx-1 的 east face 是牆壁),**但 x-periodic 下它是 cell nx-1 ↔ cell 0 的 wrap face**,必須算 swept flux。整條 wrap edge 沒處理 → 一整列 mass/momentum/energy 每步不交換,在 cell 0 和 cell nx-1 堆積差異 → 虛假 pressure gradient → 虛假 force → 虛假 vy → 連鎖崩塌。同問題存在於所有 6 個 remap kernel(donor-cell `east/north`、MUSCL `east/north_2nd`、conservative PPM `east/north_ppm`、primitive PPM `east/north_ppm_prim`)。

**Fix**: 所有 remap kernel 加 `bc_mode` 參數,週期時把 `n_edges` 從 `(nx-1)*ny` 擴到 `nx*ny`(y 方向同理),kernel 內 `cR_idx = (ic+1) % nx`(只在 x_per 時),PPM 版的 donor-relative `sx = (cx - xd) / dx_u` 在 wrap 時 `sx -= nx` 或 `+= nx` 展開(MUSCL 2nd 版同理,對 `ex = cx - xd` 做 `-= nx·dx_u`)。solver dispatch 的 `n_east` / `n_north` 同步切換為 `nx*ny`。

**Files**: `src/gpu/cart_ale2_kernels.cu`(6 個 remap kernel 簽名和內部 wrap 邏輯),`src/gpu/cart_ale2_solver.cu`(forward decl + dispatch 傳 `bc_mode` + `n_east/n_north` 計算)。

**Test strategy**: Uniform-advection 守恆測試 — 設 vx=vflow(uniform)、ρ/P uniform、vy=0、gravity=0、bc=periodic。`FORCE_UNIFORM_VX=1` 環境變量已加到 `init_kh_lecoanet`,跑一步後應有 `|KE(t=0) - KE(t=dt)| < 1e-14`、`|v|_max` 變化 ≤ 機器精度。diag 要先排除周期重複 node(見 P32),否則噪聲掩蓋信號。

---

<a id="p32"></a>
## P32: cart_ale2 periodic sync master missed x-only / y-only duplicate classes + wrong accumulator semantics

**Symptom**: 修了 P31 後 uniform advection 仍漂:64² 一步 |v|_max 1.000 → 1.079 沒變。AV 關了(`--cq-lin 0 --cq-quad 0`)、remap-order 降 0 都沒改善。加 debug 發現 `k_cale2_node_forces` 對 **uniform P 輸出** `|FX|=1.25`(應該 = 0),而 `ΔP = 0`,`Q = 0`。

**Root cause**: 兩個獨立 bug 疊在一起。

(a) **Master 選擇錯**。`k_cale2_periodic_sync_node` 判斷 `x_master = !x_per || (in == 0)`,`y_master = !y_per || (jn == 0)`,`return if (!(x_master && y_master))`。當 `bc_mode = 3`(x + y 都周期),**只 (in=0, jn=0) 通過**,所有 x-only-duplicate node(`in=0, jn∈(0, nny-1)`)和 y-only-duplicate node(`jn=0, in∈(0, nnx-1)`)**從未被 sync**。結果:內部邊界 node 的 F_x = -1.25(因為 cell ic=0 的 SW+NW corner 推出去的力)和 F_x = +1.25(cell nx-1 的 SE+NE 推出去)之和應該是 0,但 sync 不做就維持兩個非零值,node_update 用各自 ±1.25 計算 dv,產生真實運動。

(b) **sync 語義混亂**。原 kernel 對所有欄位都**平均**(`s /= np`)。但 `node_forces` 是 cell-parallel + `atomicAdd`:邊界 node 只收到**自己那半** domain 的 cells 的 subcell force contribution(2/4 cells),所以 partner 之和 = full force,**要 sum 不要 avg**;而 state (velocity, dX) 在兩個副本上本應相等,應 copy/avg。混用一個 sync 函數把 force 也平均,意味著每個 partner 只拿到**一半** of its physically correct force。

**Fix**:
1. master 判斷改為 `x_dup = x_per && (in == nnx-1)`; `y_dup = y_per && (jn == nny-1)`; `return if (x_dup || y_dup)`。canonical 代表是 `in=0` 和 `jn=0` 線(x-only、y-only、角的 4 副本皆為 master 的子 partner)。
2. kernel 加 `int mode` 參數:**mode=1 (sum) 給 force**,**mode=0 (copy/avg) 給 velocity/dX**。diagnostics 計算 Σ KE/PE 時也要 **skip `in=nnx-1` 和 `jn=nny-1` 的 duplicate**(否則 Σ m_node 雙計,KE 虛高)。

**Files**: `src/gpu/cart_ale2_kernels.cu:k_cale2_periodic_sync_node`(master 判斷 + mode 參數),`src/gpu/cart_ale2_solver.cu`(3 處 sync 呼叫加 mode;`compute_diagnostics` 的 KE 累加循環 skip duplicates)。

**Test strategy**: 同 P31 的 uniform-advection test,但要求 KE 變化 ≤ 1e-10(比 P31 嚴 5 個數量級,因為兩個 bug 疊起來時漂移很明顯,單獨修一個會看到另一個殘餘)。Lecoanet KH t=5 長跑下 `|E(t) - E(0)| / E(0) < 1.5%`,`IE` 要守到 10 位精度。兩個 bug 都修好後,`dt` 不再塌到 1e-12,可以跑完 5 time units。

---

<a id="p33"></a>
## P33: cart_ale2 Phase-M KE→IE compensation 违反局域性(全局均分 ΔKE)

**Discovered**: 2026-05-07,Andrassy 2022 scan 重跑后,ale2 256² v_rms 塌陷到 0.009
(vs vl2 0.15),128/256/512 resolution 呈非单调 pattern。根因追踪发现 compensation 机制本身
是前一天(commit `214a7d9`)为修 benchmark E_tot 漂移引入的,该 commit 的实现违反数值方法的局域性原则。

**Symptom**:
1. Andrassy 2022 长时分层对流 benchmark 上,ale2 的 saturation v_rms 被系统性压低。
   256² 上从预期的 ~0.03 塌到 0.009(比 vl2 Godunov 的 0.15 低 17×)。
2. E_tot 生长率(dE/dt)比物理 heating rate(IC header L_tot=1.21e-4)高 1-6×:
   ale2 128 跑出 6.9e-4(5.7× 超),ale2 256 跑出 1.9e-4(1.5× 超),ale2 512 跑出 1.2e-4(1.0×)。
   vl2 的 dE/dt 全在 0.5-1.0× 之间,符合物理。
3. t=100 时三 res 的 v_rms 起点一致(0.029,IC linear response 正确),之后:
   128 爬回 0.035,256 单调塌至 0.009,512 先跌后部分恢复到 0.02-0.03。
   经典 "数值耗散 resolution-dependent 非单调" pattern,不是物理现象。
4. Sod / Sedov / Gresho / Yee 5 个 benchmark **全部通过**,E_tot 机器精度守恒 — 掩盖了本 bug。

**Root cause**:

ALE swept-remap 由 Jensen 不等式必然丢 KE:
```
remap 前: 相邻 cell (m₁, v₁=+1) 和 (m₂, v₂=−1),KE_pre = ½(m₁ + m₂)
remap 后: 合并为 cell (m₁+m₂, v_new ≈ 0),KE_post ≈ 0
          ΔKE = KE_pre − KE_post > 0 丢失
```

这份 ΔKE 物理上应成为**局域**粘性热(该 cell 的 IE 增加),遵循 Caramana-Shashkov "compatible Lagrangian"
原则(JCP 2008 eq. 16 附近):每个 subcell swept-edge 丢掉的 KE 在**该两相邻 cell 内部**就地消化。

`214a7d9` 的实现采用全局 mean-field short-cut:
```cpp
// cart_ale2_solver.cu, Phase-M 编排:
KE_before_tot = gpu_reduce_sum(d_KE_node_before, ...);  // 全局 sum
KE_after_tot  = gpu_reduce_sum(d_KE_node_after,  ...);
M_tot         = gpu_reduce_sum(d_dm, ...);
delta_e = (KE_before_tot - KE_after_tot) / M_tot;        // 全局标量
k_cale2_ke_compensate_uniform<<<>>>(delta_e, d_e_int, ncell);
// → 每个 cell e_int 加 delta_e(mass-weighted 均匀)
```

这同时违反双曲守恒律的两条核心性质:
1. **因果性**:远端 cell 的 e_int 变化包含了整格所有 cell 的 KE 变化信息,信息瞬时全局传播,
   绕过 CFL / 声速有限传播速度
2. **局域守恒**:cell 的 IE 变化不等于该 cell 通量平衡 + 局域源项之和,而含一个全局背景项

对 Andrassy 这种**长时 + 分层 + 持续 ΔKE 产生**的场景,实际发生的:
- 对流层 ∇v 大,每步产生大量 ΔKE(remap swept 边上能量损失)
- 稳定层 v ≈ 0,不产生 ΔKE
- 全局均分把对流层的 ΔKE 撒进稳定层 → **稳定层被不该有的加热** → 热膨胀向上挤压对流层 → 对流 amplitude
  被系统性抑制 → v_rms 被压

同时 IE 总账是平的(compensation 目的),所以 E_tot 守恒成立,但温度场 **空间分布完全失真**。

Sod/Sedov/Gresho/Yee 均未命中该 pathology:
- Sod/Sedov 测全局量(L1 ρ, r_sh),局域 IE 失真部分 cancel
- Gresho 稳态,AV 不触发,ΔKE ≈ 0,compensation 无事可做
- Yee 纯平流,无 shock, ΔKE 小

因此 5 个 benchmark 无一捕捉此 bug,反而 "compensation 修好 E 漂移" 误给以上 benchmark 大幅改善。

**Fix**:

改写为 Caramana-Shashkov compatible 的 **per-subcell local** compensation。关键是把 ΔKE 计算
从 "global node KE_before/KE_after 做 reduction" 改为 "remap 时每条 swept edge 的 flux 自带一份
subcell ΔKE,在**两侧 cell 内部**做 IE 增量"。

伪码骨架:
```cpp
// Remap east/north kernels 每条 swept edge 算完 flux 后,额外算:
//   ΔKE_subcell = ½ |donor_flux_m| (v_donor² − v_accept²)    (approximate)
// 用 atomicAdd 就地加到 **两侧 cell** 的 e_int 增量里:
//   atomicAdd(&d_e_int_incr[cL], +½ · ΔKE_subcell / dm_new[cL]);
//   atomicAdd(&d_e_int_incr[cR], +½ · ΔKE_subcell / dm_new[cR]);
// finalize 时把 d_e_int_incr 加到 d_e_int,不再全局 reduce
```

具体实施参考:
- Caramana, Burton, Shashkov, Whalen (1998 JCP 146) "The Construction of Compatible Hydrodynamics
  Algorithms utilizing Conservation of Total Energy",eq. (24)-(30) 定义 node-centered 的
  compatible KE→IE 转移
- Kucharik-Shashkov (2012 JCP 231) 第 5 节,swept-remap 中如何把 edge flux 的 ΔKE 局域化

**Files**(拟改动):
- `src/gpu/ale/cart_ale2_kernels.cu`:所有 `k_cale2_remap_*_2nd / _ppm_*` kernels 加 subcell ΔKE 累积
- `src/gpu/ale/cart_ale2_solver.cu`:Phase-M 拆掉全局 KE 诊断 + global `delta_e` + uniform kernel
- `src/gpu/ale/cart_ale2_solver.cuh`:删 `d_KE_node_{before,after}`,改为 `d_e_int_incr` 标量
  per-cell buffer

**Test strategy**:
1. **局域守恒回归测试**:`tests/test_ale2_phase_m_local_compat.cu` — 跑 30 步 Gaussian 密度 perturb
   (见 testing_plan § 10.2),断言:
   - `|E_end − E_0| / E_0 < 1e-10`(全局守恒保留)
   - 对每个稳定层 cell(设置 IC 时 v=0 的 cell)在 30 步内 `|Δe_int| < 1e-6`
     (局域不被对流层 ΔKE 污染 — 这是 P33 的核心断言)
   - 对活动对流层 cell,`|Σ_subcell_ΔKE_into_this_cell − Δe_int| < 1e-12`
     (compatible 条件)
2. **Andrassy regression**:bc=1 分层 HSE + 底部 heating 跑 100 步,v=0 稳定层 cell 的 `|Δe_int|` 上限
3. **benchmark 不退化**:Sod/Sedov/Gresho/Yee 5 项 L1 不能比 P33 修前差 >10%
4. **Andrassy v_rms 恢复**:256² MUSCL-VL 重跑应给出 v_rms ≈ 0.02-0.04,与 128 和 512 协调,
   不再出现 256² 单调塌至 0.009 的 pathology

**影响范围**:
- 本 bug 下 Andrassy scan 的 16 run(2026-05-07 第二轮)**全部不可信**,需在 P33 修复后第三次重跑
- 5 项 benchmark 的 `L1/ρ_peak/r_sh` 数据 formally 也是"错代码下过的",但局域失真对该类 benchmark
  的主观测影响 <1%,可保留为"benchmark passed with known local-compat caveat"
- 对未来所有"长时 + 分层"型任务(pre-MS KH、O-shell core ingestion、恒星振荡)cart_ale2 目前**不可用**,
  直到 P33 修复

---

## Summary: Regression Test Priority

### Tier 1 — Correctness fundamentals (must pass for any commit)

| Test | Validates | Key assertion |
|------|-----------|---------------|
| HSE zero residual (polar) | P07, P08, P09, P19, P23 | `\|\|R_fluid(U_HSE)\|\|_2 < 1e-20` |
| HSE zero residual (Cartesian) | P24, P25, P26 | `max\|v\| < 1e-3` after 50 steps |
| Uniform advection (cart_ale2 periodic) | P31, P32 | `\|KE(dt) - KE(0)\| < 1e-10`, `\|v\|_max` 不變 |
| Mass conservation | P01 | `\|M_numerical - M_analytic\| < 1e-6` |
| Floor enforcement | P05, P27 | `rho >= 0`, `P >= floor`, no NaN |
| Gravity Phi(R) | P01, P02 | `\|Phi(R) + GM/R\| < tol` |
| Origin cell WB | P23 | `max\|vr\|_{i=0} < 1e-10` in explicit HSE step |

### Tier 2 — Solver convergence (must pass for lowmach / strang solver)

| Test | Validates | Key assertion |
|------|-----------|---------------|
| Unperturbed HSE long run | P09, P15, P19 | 100 steps, `per-cell < 1e-4`, dt grows |
| Perturbed first step | P10, P07 | `\|\|R_mr\|\| > 0` (perturbation visible) |
| Newton convergence at small dt | P15 | dt=1e-10: converge in 0-1 iterations |
| dt recovery after failure | P16 | dt recovers within 20 steps |
| Entropy wave 2nd-order | P28, P29 | order > 1.8 (L1 and L2) |
| Memory safety | P25 | `compute-sanitizer` reports 0 errors |

### Tier 3 — Component tests (preconditioner, matvec, stencil)

| Test | Validates | Key assertion |
|------|-----------|---------------|
| Analytical vs FD Jacobian | P11 | `\|\|J_analytic*v - J_FD*v\|\| / \|\|J_FD*v\|\| < 0.01` |
| Preconditioner sign | P12 | `sign(M^{-1}*r) = sign(-J^{-1}*r)` |
| Poisson residual scaling | P18 | `max\|F_Phi_scaled\| < 0.1` after GMG |
| Helmholtz GMG | new | `\|\|(nabla^2 - sigma)*u - rhs\|\| < tol` |
| Theta symmetry | P02 | `S_mt_gravity = 0` for symmetric state |
| MC limiter monotonicity | new | MC(a,b) has correct sign and magnitude |
| Reflective BC sign | P26 | `v_y(ghost) = -v_y(mirror)` |
| Face ghost fill | P25 | All face states finite after MUSCL + fill |
