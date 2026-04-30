# Lag2D Solver: 2D Axisymmetric Lagrangian-Eulerian Hybrid — Design Document

**Status**: Design phase (pre-implementation)
**Date**: 2026-04-30
**Branch**: radial-only-mode
**Scope**: New GPU solver module, non-invasive to existing FAS / LowMach / SIMPLE / projection / Strang code.

---

## Motivation

The existing 2D FAS/explicit solvers exhibit the following pathologies on radially-symmetric test cases (`lane_emden_perturbed` with `--mesh mass`):
- Energy drift ~9%/t=10 (baseline 2D explicit)
- Central cell runaway collapse
- Outermost cell internal energy eroded to KE → P = 0 or negative
- θ-symmetry broken by conditional branches (`atm_reset` hot_vacuum, floor mixing)

Investigating MESA's RSP (Radial Stellar Pulsations, `/home/kiriko/mesa-ref/star/private/rsp_step.f90`) revealed well-proven mechanisms for the 1D case:
- **Lagrangian mass coordinates** (exact mass conservation, dm_k fixed)
- **Tscharnuter-Winkler artificial viscosity** (shock-triggered, CQ/V·(Δv - ZSH√(PV))²)
- **Compression-limited dt** (dt ≤ fraction · Δr / Δv)
- **Pinned center** (r=0, v_center=0)
- **Surface pressure floor** (grey atmosphere or HSE reference value)

We built `Radial1DSolver` (`src/gpu/radial1d_solver.cu[h]`) implementing these. Results on `lane_emden_perturbed`, 256 zones, t=1000 (730k steps):
- Mass conservation: **ΔM/M₀ = 0** (machine precision, exact)
- Energy drift: **ΔE/|E₀| = 8e-4** (vs baseline 9e-2, ~100× improvement)
- Mach peak: 0.064 (vs baseline 234, then 10^13)
- dt stable at 1.37e-3 (vs baseline 7e-4)

Goal: Port these mechanisms into a full 2D solver that can handle non-axisymmetric perturbations (bubbles, convection) while preserving the stability benefits on radially-symmetric IC.

---

## Architecture Choice: Lagrangian-Eulerian Hybrid

| Dimension | Type | Rationale |
|-----------|------|-----------|
| Radial $r$ | **Lagrangian** | Mass conservation; stable origin; MESA's proven approach |
| Polar $\theta$ | **Eulerian** (fixed $\theta_j$) | Allows non-axisymmetric flow; standard HLLC applies |

The mesh is a "pancake": angular bins $\theta_j$ are fixed, but radial shell positions $r_k(\theta_j, t)$ evolve. For each $\theta_j$ column, the radial structure is Lagrangian (1D RSP-like); columns couple via Eulerian θ-direction fluxes.

**Trade-off accepted**: $dm_{k,j}$ is *not* strictly invariant — θ-fluxes move mass between angular columns. Total mass $\sum_{k,j} dm_{k,j}$ is conserved (symmetric flux), but individual cell masses change. This is the price of handling non-axisymmetric flow without fully-Lagrangian 2D grid.

---

## Variable Layout

### Face-Centered (radial faces, size `(nz+1) × nt`)
| Variable | Staggering | Purpose |
|----------|------------|---------|
| `r[k, j]` | r-face | radial face position (can vary with j after perturbation) |
| `v_r[k, j]` | r-face | radial velocity at face (Lagrangian momentum variable) |
| `M[k, j]` | r-face | enclosed mass (for gravity) |
| `gr[k, j]` | r-face | radial gravity = G·M/r² |

### Zone-Centered (cells, size `nz × nt`)
| Variable | Purpose |
|----------|---------|
| `dm[k, j]` | zone mass (evolves via θ-flux) |
| `Vol[k, j]` | zone volume |
| `rho[k, j]` | density = dm/Vol |
| `e_int[k, j]` | specific internal energy |
| `v_theta[k, j]` | cell-centered θ-velocity |
| `P[k, j]` | gas pressure (EOS) |
| `Pvsc_r[k, j]` | TW viscosity from r-direction compression |
| `Pvsc_t[k, j]` | TW viscosity from θ-direction compression |

### HSE Reference (zone-centered)
| Variable | Purpose |
|----------|---------|
| `rho0[k, j]`, `P0[k, j]`, `r0[k, j]` | Initial HSE state (for diagnostics, floors) |

### Derived / Scratch
- `dt_cell[k, j]`: per-cell dt (reduction)
- `dm_shell[k]`: total shell mass summed over j (diagnostic/init sanity check)

---

## Geometry (axisymmetric spherical)

Cell volume (exact integral):
$$V_{k,j} = \tfrac{1}{3}(r_{k+1,j}^3 - r_{k,j}^3)(\cos\theta_{j-\tfrac{1}{2}} - \cos\theta_{j+\tfrac{1}{2}})$$

Radial face area:
$$A^r_{k,j} = r_{k,j}^2 (\cos\theta_{j-\tfrac{1}{2}} - \cos\theta_{j+\tfrac{1}{2}})$$

Theta face area:
$$A^\theta_{k,j+\tfrac{1}{2}} = \tfrac{1}{2}(r_{k+1,\cdot}^2 - r_{k,\cdot}^2) \sin\theta_{j+\tfrac{1}{2}}$$

(At θ-faces we average r from both sides.)

---

## Discrete Equations

### Stage 1 & 3: Radial Lagrangian (half-step)

For each column $j$ independently:

**Momentum** (face k, zone centered above = zone k, below = zone k-1):
$$\frac{v_r^{n+1/2} - v_r^{n}}{\Delta t/2} = -\frac{A^r_{k,j} (XP_{k,j} - XP_{k-1,j})}{\overline{dm}_{k,j}} - g_{k,j}$$

with $XP = P + Pvsc_r$, $\overline{dm}_{k,j} = \tfrac{1}{2}(dm_{k-1,j} + dm_{k,j})$.

**Position**: $r_{k,j}^{n+1/2} = r_{k,j}^n + (\Delta t/2) v_{r,k,j}^{n+1/2}$

**Geometry update**: recompute $V_{k,j}, A^r_{k,j}$ from new $r$.

**Energy** (zone-centered, adiabatic compression):
$$e_{\text{int}}^{n+1/2} = e_{\text{int}}^n - (P + Pvsc_r) \cdot \frac{V^{n+1/2} - V^n}{dm_{k,j}}$$

**BC**: $v_r[0, j] = 0$ pinned; $P_{ghost}$ at surface = `P_surf_floor`.

### Stage 2: θ-Direction Eulerian (full step)

For each zone $(k,j)$, compute HLLC fluxes at θ-faces $j \pm 1/2$. Update:

$$\begin{bmatrix} dm \\ dm \cdot v_r \\ dm \cdot v_\theta \\ dm \cdot e_{\text{tot}} \end{bmatrix}_{k,j}^{n+1} = (\cdots)^n - \Delta t \left( A^\theta_{j+\tfrac{1}{2}} F^\theta_{j+\tfrac{1}{2}} - A^\theta_{j-\tfrac{1}{2}} F^\theta_{j-\tfrac{1}{2}} \right)$$

where $F^\theta$ is the standard HLLC flux using primitives $(\rho, v_r, v_\theta, P)$.

**Key symmetry property**: When state is θ-uniform and $v_\theta \equiv 0$:
- MUSCL reconstruction produces L = R at every θ-face (no slope)
- HLLC flux is just $F(\text{center state})$
- $A^\theta_{j+\tfrac{1}{2}} F - A^\theta_{j-\tfrac{1}{2}} F$ is **not** zero (areas differ via $\sin\theta$) — but the flux is $\rho v_\theta (=0)$ for mass and $P$ for $mθ$
- Need to carefully verify: θ-symmetric IC → zero θ-contribution to all time-evolved variables

**BC (θ-direction)**:
- $\theta = 0$ (north pole): reflective, $v_\theta \to -v_\theta$
- $\theta = \pi$ (south pole): reflective

### Strang Splitting Time Integration

$$U^{n+1} = L_r^{\Delta t/2} \circ L_\theta^{\Delta t} \circ L_r^{\Delta t/2} \; U^n$$

2nd-order accurate when each sub-operator is ≥2nd-order.

---

## Boundary Conditions

### Inner (r = 0)
- $r_{0,j} = 0$ pinned (all j)
- $v_r[0, j] = 0$ pinned (reflective)
- No ghost cell needed for momentum (face is at origin)

### Outer (r = r_surface)
- Ghost zone pressure above surface: $P_{\text{ghost}} = P_{\text{surf\_floor}} = P_0[nz-1]$ (initial surface HSE pressure)
- Surface face can move freely

### θ = 0 (north pole) and θ = π (south pole)
- Standard reflective: $v_\theta \to -v_\theta$, other fields copy
- Ghost cells in θ-direction populated symmetrically

---

## CFL / dt Control

dt is the minimum of:

1. **Acoustic CFL (radial)**: $dt_r = \text{CFL} \cdot \min_{k,j} \frac{\Delta r_{k,j}}{|v_r| + c_s}$
2. **Acoustic CFL (theta)**: $dt_\theta = \text{CFL} \cdot \min_{k,j} \frac{r_{k,j} \Delta\theta_j}{|v_\theta| + c_s}$
3. **Compression limit (Lagrangian)**: $dt_c = \text{frac} \cdot \min \frac{\Delta r}{\max(0, v_r[k] - v_r[k+1])}$ (per column j)

Note: θ-CFL near center ($r \to 0$) becomes severe. Mitigation: reuse `angular_avg` trick (merge innermost shells) or central v-damping.

---

## Module Structure

```
src/gpu/
├── lag2d_solver.cuh       (~200 lines)  Main struct, public API
├── lag2d_solver.cu        (~400 lines)  init, step (Strang), destroy, diagnostics
├── lag2d_kernels.cuh      (~800 lines)  All device kernel declarations + inline helpers
├── lag2d_radial.cu        (~300 lines)  Stage 1/3: radial Lagrangian update kernels
├── lag2d_theta.cu         (~400 lines)  Stage 2: θ-direction HLLC kernels
├── lag2d_init.cu          (~200 lines)  Lane-Emden → 2D Lagrangian shell layout
└── lag2d_output.cu        (~150 lines)  VTK output (compatible with existing renderer)
```

**Total estimate**: ~2500 lines.

All files are new — **existing code is not modified** (only `CMakeLists.txt` adds the new sources, and `main.cpp` gets a new `--solver lag2d` dispatch branch).

---

## Implementation Plan

### Phase C1: Symmetric-IC Validation (Week 1)

Goal: Demonstrate that the 2D solver, when given a radially-symmetric IC, produces results **numerically identical to `Radial1DSolver`** and preserves θ-symmetry to machine precision.

Steps:
1. Data structures + init (Lane-Emden → (nz, nt) arrays, all j identical)
2. Stage 1 radial Lagrangian kernels (adapted from radial1d)
3. Stage 2 θ-HLLC kernels
4. Strang splitting orchestrator
5. VTK output compatible with `render_video_fast.py`
6. Validation:
   - Run `lane_emden` (HSE) and `lane_emden_perturbed` with symmetric IC
   - Compare time-series (mass, energy, max|v|) against `Radial1DSolver` — must match to relative 1e-10 or better
   - Verify θ-standard-deviation of ρ, v_r stays at machine epsilon

### Phase C2: Non-Symmetric Tests (Week 2)

7. Hot bubble test (`bubble` test case, off-axis perturbation)
8. Polar axis (θ=0, π) boundary treatment verification
9. Surface treatment under strong oscillations
10. Quantitative comparison with current 2D FAS on bubble: stability, computational cost, accuracy

### Phase C3: Optimization & Robustness (Week 3+, optional)

- GMG-based gravity (replace current 1D angle-averaged gravity with full 2D Poisson)
- Implicit Newton time-stepping (following MESA RSP pattern, for large-dt stability)
- Mesh reconnection if θ-flux causes $r_{k,j}$ to become non-monotonic

---

## Key Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| θ-flux + Lagrangian r coupling breaks conservation | Rigorous flux-form discretization; validate against radial1d on symmetric IC |
| dt limited by θ-CFL at origin | Angular averaging (reuse FAS `n_angular_avg`) or central damping |
| HSE not preserved exactly under Strang splitting | Well-balanced flux-level subtraction of HSE pressure (copy from fas_residual.cu:250) |
| $r_{k,j}$ becomes non-monotonic in θ | Rare; can add monotonicity floor; mesh reconnection (Phase C3) |
| Debugging 2D harder than 1D | Phase C1 validation requirement: must reproduce radial1d to 1e-10 before proceeding |

---

## Validation Targets (Phase C1)

Run `lag2d` and `radial1d` with identical `lane_emden_perturbed`, nr=256, tend=10:

| Metric | Target |
|--------|--------|
| $\Delta M / M_0$ | < 1e-12 (should be 0 by construction) |
| $\Delta E / \|E_0\|$ | < 1e-3 (match radial1d) |
| θ-relative-stddev of $\rho$ | < 1e-12 (machine epsilon) |
| Max |v_θ| | < 1e-12 (should be 0) |
| Runtime | within 3× of radial1d (2D extra cost) |

---

## Open Design Questions (to decide before implementation)

1. **Time-integration order in Stage 1**: RK2 within each radial half-step, or just forward Euler (since the half-step is already small)?
   - Radial1d uses RK2. For Strang splitting with overall 2nd-order accuracy, each sub-op ≥ 2nd-order is sufficient but not necessary. FE sub-op + Strang = Godunov-like 1st order. **Decision**: use RK2 in each half-step to preserve 2nd order.

2. **θ-MUSCL reconstruction**: include it from Phase C1, or use 1st-order donor-cell?
   - 1st-order θ reconstruction is fine on symmetric IC (reduces to zero contribution). For bubble case, we need 2nd-order. **Decision**: implement MUSCL from Phase C1, test symmetric case first to catch bugs.

3. **Gravity**: 1D angle-averaged (like FAS) or full 2D GMG Poisson?
   - On radially-symmetric IC both give identical result. Full 2D GMG is more accurate for bubble. **Decision**: start with 1D angle-averaged (simpler, matches radial1d for C1 validation); upgrade in C3 if needed.

4. **$dm_{k,j}$ update under θ-flux**: explicit mass accumulation or implicit via flux-form $\rho$ update?
   - Flux-form $\rho$ update is standard; then $dm_{k,j} = \rho_{k,j} V_{k,j}$ after each step. **Decision**: use flux-form for all conserved quantities; $dm$ is a derived diagnostic.

---

## Status Checklist

- [x] MESA RSP studied (`rsp_step.f90`)
- [x] `Radial1DSolver` implemented and validated (t=1000 stable, ΔM=0, ΔE/|E0|=8e-4)
- [x] Design document written (this file)
- [ ] Data structure + init + CMake integration
- [ ] Stage 1 (radial Lagrangian) kernels
- [ ] Stage 2 (θ-HLLC) kernels
- [ ] Strang splitting orchestration
- [ ] VTK output
- [ ] Phase C1 validation: symmetric IC matches radial1d

---

## Reference: Radial1DSolver Benchmarks

For comparison in Phase C1 validation. Both solvers same IC (`lane_emden_perturbed`, nr=256, perturb=1e-3).

Run: `runs/lane_emden_perturbed_256x256_20260430_214343/diagnostics.csv`, tend=1000, 730k steps:

```
step      t             dt         Mass                  KE          IE         PE          E_tot         ΔM/M0      ΔE/|E0|     Mach     |v|max
1500      2.053e+00     1.37e-03   3.0263820114000e+00   2.3e-06     2.415e+00  -4.822e+00  -2.40699e+00  0.0e+00    0.0e+00     0.0323   9.29e-03
91500     1.252e+02     1.37e-03   3.0263820114000e+00   2.9e-07     2.414e+00  -4.821e+00  -2.40747e+00  0.0e+00    -2.0e-04    0.0087   2.60e-03
366000    5.010e+02     1.37e-03   3.0263820114000e+00   2.5e-08     2.411e+00  -4.820e+00  -2.40833e+00  0.0e+00    -5.6e-04    0.0003   1.66e-04
729000    9.980e+02     1.37e-03   3.0263820114000e+00   3.4e-09     2.412e+00  -4.820e+00  -2.40818e+00  0.0e+00    -4.9e-04    0.0004   1.04e-04
```

These are Phase C1 validation targets for `lag2d` on symmetric IC.
