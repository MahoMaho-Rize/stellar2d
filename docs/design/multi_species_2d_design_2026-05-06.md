# Multi-species passive tracer extension — 2D solver families

**Date**: 2026-05-06
**Scope**: Route A (Andrassy 2022 2D benchmark) prerequisite — add one or more passive scalars X(species) advected by the velocity field in three solvers: `anelastic_sl`, `cart_ale2`, `lowmach`.
**Non-goals**: α-chain source terms (Route C), Route B Boussinesq buoyancy (separate solver). No chemistry here — pure conservative advection only.

---

## 1. Why a passive tracer first

1. **Cheapest possible composition infrastructure** — no EOS dependence, no source terms, decouples from Route C α-chain risk.
2. **Directly measurable diagnostic** — Andrassy 2022 reports entrainment rate via a composition-like tracer; their "FV" (fractional volume) is mathematically a passive scalar.
3. **Route A needs it** regardless of α-chain decisions; Route C inherits it.
4. **Smoke-testable** — pure advection has analytic reference (rotating Gaussian, L2 decay rate) that doesn't need realistic IC.

## 2. Unified interface across three solvers

Define one C++ concept:

```cpp
// each solver adds (per physical cell):
double X[N_SPECIES];  // mass fraction, sum X_i = 1 enforced post-step
```

For Route A we start with **N_SPECIES = 1** (single tracer marking upper stable layer). This is enough for entrainment rate measurement. Extension to 6 (match `alpha_net`) is deferred.

Three design constraints from CLAUDE.md + project memory:

- ✅ **GPU first**, CPU never;
- ✅ **No mocks / no "pre-wired for future species"** — one scalar, one implementation path;
- ✅ **Anelastic_sl W advection bug precedent**: in Fourier space, derivatives are exact but **nonlinear products (u·∇X) must be done in physical space** then FFT'd back. Same for tracer.

## 3. anelastic_sl — spectral solver

### 3.1 Current state vector

From `src/gpu/spectral/anelastic_sl_solver.cuh`:

- `d_V(y, kx)` — horizontal velocity, Fourier-in-x, CGL-in-y. Size `ncplx = nh·ny`.
- `d_W(y, kx)` — vertical velocity.
- `d_pihat(y, kx)` — pressure correction.
- Background: `d_rho(y)`, `d_rho_sqrt_inv(y)`, `d_N2(y)` — all 1D in y.

### 3.2 Where X slots in

- Add `d_X(y, kx)` — complex, size `ncplx`.
- **Linear term**: NONE — X has no linear restoring force. Skip the Path-D per-kx matrix entirely.
- **Nonlinear step**: in `nonlinear_deriv()` (currently computes u·∇V + buoyancy for V, W), extend with u·∇X in physical space:
  1. IFFT X_hat → X(x, y);
  2. Compute ∂X/∂x and ∂X/∂y in physical space (finite difference in y using CGL `d_Dy_row`, spectral in x via ikx multiplication in Fourier space);
  3. Form u·X and w·X, take divergence(∂_x(uX) + ∂_y(wX)) — same pattern as V advection **minus** the buoyancy source;
  4. FFT back.
- **Strang split**: add X to the nonlinear half only. Linear half is identity on X.
- **kx=0 mean**: unlike V (which is zeroed at kx=0 by anelastic mass conservation), X keeps its kx=0 column — that's the horizontally averaged composition. Do not project it out.
- **Galerkin V_K projection**: do NOT apply to X. The projection was needed for V because incompressibility couples V_0 to higher V_k; X has no such constraint. This matches the "W advection bug" precedent where W was incorrectly passively advected; here X *is* passive and gets plain advection.

### 3.3 Complexity

**Moderate** — ~200 LOC. Main work: (a) allocate `d_X`, `d_Xhat_rhs` scratch; (b) factor out the existing `nonlinear_deriv` divergence-of-(u·field) pattern into a helper that takes any scalar field; (c) add X to `upload_state` / `download_state`; (d) add H_im contribution (zero for X — it's pure advection, no Hamiltonian term).

### 3.4 Gotchas

- **Tracer bounds**: pure spectral advection in Fourier will produce Gibbs overshoots at sharp interfaces (standard spectral phenomenon). For Andrassy 2022 smooth initial interface this is fine; for a sharp step IC, add a weak spectral filter (existing `Q` filter in solver can be reused).
- **kx=0 bookkeeping**: the zero-wavenumber column is the horizontal mean ⟨X⟩(y,t), which *is* the Andrassy 2022 entrainment diagnostic. Make sure it's not accidentally zeroed.
- **CGL y-derivative for X**: reuse `d_Dy_row` — same differentiation matrix used for V, W.

### 3.5 Test problem (smoke)

Non-rotating, N²=0, zero buoyancy perturbation, V=W=0, prescribe X(y) = tanh((y-y_int)/δ). After a specified divergence-free velocity (e.g. steady shear), X must remain unchanged; ∫X²dV conserved to round-off (pure advection property).

## 4. cart_ale2 — ALE compressible

### 4.1 Current state vector

From `src/gpu/ale/cart_ale2_solver.cuh`:

- `d_rho`, `d_P`, `d_cs` (cell center primitive / thermodynamic);
- `d_rho_dens = dm/V₀`, `d_rhoE_dens = dm·e_int/V₀` (conservative density = mass per reference volume);
- Node velocities in x/y;
- Swept remap temp: `d_rho_sx`, `d_rho_sy`, `d_rhoE_sx`, `d_rhoE_sy` (per-edge mass/energy swept fluxes);
- Per-face L/R reconstruction buffers: `d_rho_xL/xR/yD/yU`, `d_rhoE_xL/xR/yD/yU`.

**No existing tracer / species slot. NVAR is implicit (2 remapped scalars: mass and energy). No n_vars abstraction.**

### 4.2 Where X slots in

cart_ale2 remaps conservative densities with either donor-cell or PPM. To add X:

1. **Per-species conservative density**: `d_rhoX_dens[N_SPECIES]` = dm·X / V₀.
2. **Per-species reconstruction buffers**: `d_rhoX_{xL,xR,yD,yU}[N_SPECIES]` and swept flux buffers `d_rhoX_{sx,sy}[N_SPECIES]`.
3. **Remap kernel extension**: existing MUSCL/PPM swept-remap kernels for `rhoE_dens` are the exact template — same stencil, same limiter, same face loop, same `bc_mode` handling. Copy-paste + rename to `rhoX_dens`, no new physics.
4. **Lagrangian phase**: tracer does not feel pressure, does not appear in momentum equation. So Lagrangian phase is identity on X (mass moves with mesh, X per-unit-mass is invariant). Divide by new dm after rezone to recover X.
5. **Bounds**: clamp 0 ≤ X ≤ 1 post-remap. Donor-cell is monotone so no clamp needed for order-1; MUSCL/PPM with minmod is monotone for convex advection; CW limiter can overshoot — clamp there.

### 4.3 Complexity

**Moderate-to-hard** — ~400 LOC. cart_ale2 is the **largest and most pitfall-ridden** of the three (see CLAUDE.md P30, P31 on periodic BC sync_node mode). Each new conserved variable has to thread through Lagrangian phase, reconstruction, flux, remap, BC, diagnostics. The copy-paste template is clean but the combinatorics (4 edge buffers × 2 remap orders × 2 limiters × N_SPECIES) need care.

### 4.4 Gotchas

- **Periodic BC sync_node mode**: CLAUDE.md P31 — state variables use `mode=0` (copy), forces use `mode=1` (sum). X is a state variable → mode=0. But tracer *flux* at edges needs the same sum-at-boundary treatment as mass flux.
- **n_edges for periodic wrap**: CLAUDE.md P30 — periodic BCs need `n_edges = nx*ny`, not `(nx-1)*ny`. Applies to X swept flux the same way.
- **Bounds preservation**: PPM-CW can overshoot. If we use the PPM path (default for cart_ale2), clamp X post-remap. Record max overshoot as diagnostic.
- **Energy-X coupling for future**: if later α-chain adds source ε(X, T), the species-energy coupling is operator-split (Strang): advect X → update e_int from ε·dt in cell-local ODE. For now, keep them decoupled.

### 4.5 Test problem (smoke)

Same as anelastic_sl §3.5 — prescribed divergence-free velocity, initial tanh X profile, ∫X²dV should monotonically *decrease* (numerical diffusion from limiter) but dm·X (total tracer mass) must be conserved to machine precision.

## 5. lowmach — JFNK implicit

### 5.1 Current state vector

From `src/gpu/implicit_lowmach/lowmach_solver.h`:

- `d_rho`, `d_mr`, `d_mtheta`, `d_rhoE` — 4 unknowns per cell in JFNK vector.
- `d_rho0`, `d_P0`, `d_phi0` — HSE reference for well-balancing.
- `d_scale_R` — MUSIC row scaling, size `4*n`.

### 5.2 Where X slots in — two paths

**Path 1 — JFNK-coupled (hard)**:
- Extend state to 5 unknowns: `(rho, mr, mtheta, rhoE, rhoX)`.
- Update `jfnk_matvec`, residual, preconditioner.
- Scale `d_scale_R` to `5*n`.
- SIMPLE / block-Schur preconditioner extensions non-trivial.
- ~600 LOC + debugging the JFNK convergence (MEMORY warns about existing JFNK stability issues).

**Path 2 — Operator-split (recommended)**:
- JFNK step as-is solves for (rho, mr, mtheta, rhoE).
- Separate explicit step for X: advance `d_rhoX` by `rho·u·∇X` using donor-cell or MUSCL on the updated velocity field from JFNK.
- ~100 LOC, no JFNK surgery.
- Slight loss of implicitness for X, but X has CFL similar to advective flow — explicit is fine at lowmach CFL limits.

**Decision: Path 2 first.** If we later need tight X-u coupling (e.g. α-chain heating feedback), revisit.

### 5.3 Complexity

**Moderate** (Path 2) — ~100 LOC, mostly duplicating the existing mr advection scheme for a scalar. **Hard** (Path 1) — several weeks + JFNK convergence risk. Do Path 2.

### 5.4 Gotchas

- Operator-split may violate mass-X conservation at the `1e-13` level per step. Acceptable for O(10k) step runs; document it.
- Lowmach preconditioner's row-scaling assumes 4 equations per cell. Path 2 bypasses this (X is outside JFNK), no change needed.
- Project memory flags existing JFNK convergence issues — touching the residual (Path 1) is tempting rabbit hole, refuse.

## 6. Testing strategy

Each solver gets:

1. **Conservation test** (pure advection, solid-body rotation): ∫ρX dV conserved to 1e-12 after N steps.
2. **Order-of-accuracy test**: Gaussian blob advected through a constant velocity for T_end, L2 error vs exact solution at 3 resolutions. Expected: spectral for anelastic_sl (machine precision), 2nd for cart_ale2 PPM, 2nd for cart_ale2 donor, 2nd for lowmach Path 2 MUSCL.
3. **Bounds test**: initialize X ∈ {0, 1} sharp jump, verify max overshoot after N steps is below threshold (0% for anelastic_sl with filter, <1% for cart_ale2 PPM-CW, 0% for donor + minmod).
4. **Cross-solver comparison**: same IC + same velocity field → compare ⟨X⟩(y,t) profiles. Must agree in the resolved regime.

## 7. Implementation sequence (M1, 3 weeks)

Week 1:
- anelastic_sl: add d_X + nonlinear advection + upload/download + smoke test 1 (pass conservation).
- cart_ale2: add d_rhoX_dens + donor-cell remap only + smoke test 1.

Week 2:
- cart_ale2: add MUSCL/PPM remap for X + bounds clamp + test 3.
- lowmach: Path 2 explicit advection + test 1.

Week 3:
- All three: test 2 (accuracy) at 3 resolutions.
- Cross-solver comparison (test 4) on rotating velocity IC.
- Write smoke-test driver `tests/test_tracer_advection.cu`.

Zero Andrassy IC work this M1 — that's M2.

## 8. Deliverables for M1

- Code: 3 solvers each passing the 4 tests.
- Regression suite: `tests/test_tracer_advection.cu` added to CMake.
- Doc: smoke test results (L2 errors at 3 resolutions, 3 solvers) in `docs/projects/shell_merger/tracer_smoke_2026-06.md`.
- No Andrassy IC, no α-chain, no Boussinesq — just passive advection infrastructure.

## 9. Open questions for user

1. `N_SPECIES` at compile time or runtime? Compile-time template is fastest for GPU; runtime is more flexible but slower. Route A needs 1, Route C needs 6. **Suggest**: template `<int NS>`, instantiate for NS=1 and NS=6 in .cu file.
2. Start with all 3 solvers or staggered? **Suggest**: anelastic_sl + cart_ale2 in parallel week 1; lowmach only if time permits (lowest priority — MUSIC occupies this niche and we may drop it from Route A per §8 of scope doc).
3. Any concerns about adding CGL y-derivative to a non-vertical-velocity variable in anelastic_sl? The `d_Dy_row` matrix is non-symmetric but that was only an issue for the Liouville eigenproblem, not for advection. Should be fine.

---

**References**
- `CLAUDE.md` (root) — solver asset rules, cart_ale2 periodic BC pitfalls P30/P31.
- `docs/design/anelastic_SL_spectral_design.md` — anelastic_sl state vector definition.
- `docs/design/cart_ale2_design.md` — ALE remap architecture.
- `src/physics/alpha_network.h` — shape of species state for future Route C hookup (`X[N_SPEC]` with N_SPEC=6).
- `docs/projects/shell_merger/shell_merger_scope_2026-05-06.md` §7.2 — M1 target.
