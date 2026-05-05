# radial1d Implicit Time Integration Plan

**Date**: 2026-05-02
**Goal**: Let `dt` grow from ~1 s (explicit acoustic CFL) to ~1e8 s (MESA Henyey scale) so the solver can reach the KH timescale (~1e15 s) and observe pp-chain ignition.
**Scope**: Design-only; no implementation.

---

## 1. Current state (what we have)

### radial1d (explicit RK2 Lagrangian, 1D)
- Files: `src/gpu/radial1d_solver.{cu,cuh}`, `src/gpu/radial1d_kernels.cuh`.
- State: faces (r, v, M, g) + zones (dm, Vol, rho, e_int, P, Pvsc, rhoE, X, Y).
- Per step (`Radial1DSolver::step`, ~100 LoC):
  1. `compute_dt()` (acoustic + compression).
  2. `enclosed_mass`, `gravity`, save prev state, `artificial_viscosity`.
  3. RK2 stage 1: momentum → position → primitives → energy → primitives.
  4. RK2 stage 2: same, then average prev/curr.
  5. Operator-split: `nuclear_pp` (+species) and `apply_radiation_diffusion`.
- dt bound: acoustic `dr/(|v|+cs)`; solar polytrope in physical units: `cs` at center ~5e7 cm/s, `dr` a few 1e8 cm → `dt~1` s.

### cart_impl (full BE + JFNK, 2D Cartesian, reference)
- Files: `src/gpu/cart_impl_solver.{cu,cuh}`, `cart_impl_jfnk.cu`, `cart_impl_residual.cu`.
- Residual `R(U)`: HLLC fluxes + MUSCL + gravity source, well-balanced `hse_defect` subtraction.
- JFNK: `F(U) = (U−Uⁿ)/dt − (R(U)−R_hse)`; `J·v ≈ (F(U+ε v̂) − F(Uⁿ))/ε`.
- FGMRES(30) + CGS2, Viallet eq 72 scaling, identity preconditioner.
- Proven at 128² with Ma 0→2e-2, long-time stable.

### Why ~1 s explicit dt is the bottleneck
pp rate at solar center ~2e−3 erg/g/s → thermal timescale `c_v T/ε_pp ~ 1e15` s. Even with exact mass conservation, explicit needs ~1e15 steps. MESA reaches 1e7–1e8 s by making hydrodynamics implicit.

---

## 2. Evaluation of three routes

### Option 1 — BE only on energy / sources
Replace operator-split nuclear + radiation with implicit per-zone updates. Leave hydro explicit.
- **Does not unlock `dt~1e8` s** — hydro still on acoustic CFL.
- Cost: small (~200 LoC). Risk: low.
- Verdict: solves wrong problem.

### Option 2 — IMEX
Explicit hydro + implicit sources (Kupka 2020 A-MUSIC style).
- **Still bounded by acoustic CFL**.
- Cost: moderate. Risk: low.
- Verdict: does not meet 1e8 s target.

### Option 3 — Full BE + JFNK on radial1d (RECOMMENDED)
`F(U) = U − Uⁿ − dt · R(U)` solved by Newton-Krylov.
State per zone: `U_k = (v_k, r_k, e_k, [X_k])` (3–4 fields).
- **The only route that reaches `dt~1e8` s**. Proven by MESA/Henyey + cart_impl.
- Cost: 2–3 weeks / ~1500–2000 LoC, significantly reduced by cart_impl port.
- Risk: medium. 1D Lagrangian is friendlier than 2D log-spherical — κ(J) ≈ O(N) per field.
- Verdict: **this is the path**.

---

## 3. Key architectural decisions for Option 3

### 3.1 Residual `R(U)`
```
R_r[k]   = v[k]
R_v[k]   = -A_k (XP_k − XP_{k−1}) / dm̄_k  − g_k
R_e[k]   = -(P_k + Pvsc_k) · (dV/dt)_k / dm_k  + ε_pp,k  + Q_rad,k
R_X[k]   = -ε_pp,k / q_burn
```
Invariants: `r[0]=0`, `v[0]=0`, `dm_k` fixed, ghost `P_surf_floor` at `k=nz`.
Jacobian: 3-zone stencil in `r,v,e`; zero-bandwidth in `X`. Bandwidth per block ≈ 9.

### 3.2 Well-balanced HSE subtraction
Precompute `R(U_hse)` once at `snapshot_hse()`; subtract in `F`. Stored as `d_R_hse`.

### 3.3 Packed state vector
`d_U` size `n_fields · nz`. Faces use `k=1..nz` only (r[0]=v[0]=0 pinned).
`n_fields=3` (v, r, e), +1 if species_enabled.

### 3.4 JFNK matvec (exact port of cart_impl)
Unit-normalize v̂, `α = eps/‖v‖`, perturb U, recompute F, restore.

### 3.5 FGMRES(30) + CGS2
Verbatim port of `cart_impl_jfnk.cu`. `N_dof` ~ few thousand, double precision.

### 3.6 Viallet scaling (critical at Ma~1e-3)
- face `v`: `ρ_bar · max(|v|, α·cs)`
- face `r`: `dm̄/(4π r²)`  (or `ρ_bar · cs`)
- zone `e`: `ρ · cs²`
- zone `X`: `ρ` (or unit)

### 3.7 Preconditioning
Start with **identity**. If GMRES stalls at high dt: **block-tridiagonal Thomas sweep** (exactly Henyey). Port from `use_line_precond_y` shape in cart_impl.

### 3.8 Newton convergence
Port `newton_solve`: abs tol + rel half-drop + stall + dt halving. `newton_max_iter=15`, `gmres_tol=1e-3`, `GMRES_K=30`.

### 3.9 dt selection
Change-fraction control: `dt_{n+1} = dt_n · min_k(0.1 · |U_k| / |ΔU_k|)`.
Growth cap ≤ 2×. Floor/ceil `[1e-3, 1e10]`. Retry with halve on failure.

### 3.10 Operator-split vs coupled sources
- **Radiation**: keep operator-split but switch to BE tridiagonal (replaces subcycle). Can fold later.
- **Nuclear + species**: in `R(U)` from the start (per-zone, easy Jacobian).

---

## 4. Staged implementation plan

### Phase 4.0 — scaffolding (1 day)
- New `radial1d_implicit.cu` + `step_implicit()` entry (keep explicit `step()` for regression).
- Alloc implicit scratch: `d_U, d_Un, d_Ubak, d_R, d_F, d_Fk, d_R_hse, d_scale_*`, GMRES basis.
- Pack/unpack kernels.
- `compute_R_implicit()` — single hardest kernel. Start with `n_fields=3` (v, r, e).

**Checkpoint 4.0**: `step_implicit(dt=1 s)` reproduces `step(dt=1 s)` to 1e-6.

### Phase 4.1 — JFNK + GMRES port (1–2 days) — FIRST CRITICAL MILESTONE
- Port `jfnk_matvec`, `apply_precond=identity`, `gmres_solve`, `newton_solve`.
- Port `compute_F` with `R − R_hse`.
- Enable Viallet L/R scaling.

**Checkpoint 4.1**: Solar polytrope IC, HSE, no perturb, no nuclear, no radiation. `dt=1e4 s` (~1e4× explicit) for 10 steps.
- Newton converges ≤5 iters
- ‖v‖_max ≤ 1e-3 · cs
- Mass conserved 1e-12
- Total energy drift < 1e-6

If Newton diverges: inspect scaling, try tridiagonal preconditioner, debug Jacobian FD vs hand-written tangent on 4-zone toy.

### Phase 4.2 — Push dt (1 day)
- Change-fraction dt control.
- Sweep dt=1e4 → 1e8 s.
- Add block-tridiagonal preconditioner if GMRES stalls.

**Checkpoint 4.2**: `dt=1e8 s` stable for 100 steps on polytrope.

### Phase 4.3 — Reintroduce physics (1–2 days)
- Nuclear source + dX in `R(U)`.
- Radiation BE tridiagonal substep.

**Checkpoint 4.3**: nuclear + radiation on, dt→1e8, integrate to 1e10 s. Expect core heating + X depletion.

### Phase 4.4 — Full KH run
Integrate to 1e12–1e15 s (may need dt growth to 1e10 s post-thermal-equilibrium).

---

## 5. Risk register

| Risk | Mitigation |
|-----|-----------|
| JFNK diverges at dt=1e4 (scaling wrong) | Hand-written tangent on 3-zone; FD-vs-analytic debug harness. |
| Lagrangian faces cross during Newton | Line search; backtrack if Δr → negative Vol. |
| `P_surf_floor` pathological at large dt | `R(U_hse)` subtraction including ghost. |
| Radiation BE coupling stiff | Fall back to current subcycle if BE unstable. |
| Species mass loss at large dt | Clamp X∈[0,1] + X+Y+Z=1 post-Newton. |
| GMRES stalls at dt=1e8 | Block-tridiagonal line-implicit preconditioner (Henyey). |

---

## 6. Recommendation

**Go Option 3, staged 4.0 → 4.4.** Phase 4.1 checkpoint (`dt=1e4 s` stable, 10 steps) is the first hard gate. cart_impl's JFNK/GMRES/CGS2/scaling is ~95% drop-in for the 1D Lagrangian problem — working dt=1e4 s is plausibly hours of coding away once the residual kernel is correct.
