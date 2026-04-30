# wb2d_solver: Well-Balanced 2D Eulerian Solver with MESA-Inspired Stabilization

**Status**: Design (pre-implementation)
**Date**: 2026-04-30
**Branch**: radial-only-mode
**Scope**: New GPU module, non-invasive to existing FAS/LowMach/SIMPLE/projection/Strang code.

---

## Motivation

Our existing 2D FAS/explicit solvers exhibit known pathologies on radially-symmetric test cases — baseline energy drift ~9%/t=10, central runaway, surface P→0, conditional-branch θ-symmetry breakage. These are *symptoms of Eulerian stabilization missing MESA-style mechanisms*, not fundamental defects of the Eulerian approach.

`Radial1DSolver` (MESA-RSP port) achieves ΔM=0 and ΔE/|E₀|=8e-4 at t=1000 on the same IC by combining:
- Tscharnuter-Winkler shock-triggered artificial viscosity
- Compression-limited dt
- Physical surface P floor (not conditional-reset)
- Pinned origin

This module ports those mechanisms into a **dualistic 2D Eulerian formulation** where r and θ are treated symmetrically.

Lagrangian-Eulerian hybrid (Plan C) was considered and deferred: production 2D Lagrangian codes all require ALE infrastructure (ReALE, HYDRA, FLAG; ~40 person-year estimate for production-grade ALE). Dukowicz-Meltz (1992) proves that any 2D flow with vorticity tangles a pure Lagrangian mesh in finite time ≈ 2π/|ω|. For our stellar-pulsation use case with vortical perturbations (bubble, convection), Eulerian with well-balanced discretization is the mainstream and robust choice (Käppeli-Mishra 2016; MUSIC, SLH, PROMPI all use it).

---

## Design Principles

1. **Reuse existing infrastructure aggressively.** Grid, State, EOS, HLLC, MUSCL, GMG Poisson, Lane-Emden init — all existing. We write *only* the new pieces.
2. **Two directions treated identically.** r and θ flux computation, viscosity, reconstruction share the same kernels and code paths. No "radial special-case" vs "theta special-case" asymmetry.
3. **Explicit RK2 first.** Get the well-balanced + MESA-stabilized Eulerian working correctly. Add implicit in a later phase if dt becomes a bottleneck.
4. **No conditional branches that break θ-symmetry.** All stabilization mechanisms (viscosity, dt limit, floor) act uniformly, not based on cell-state-dependent predicates.

---

## Architectural Choices

| Choice | Selection | Rationale |
|---|---|---|
| Time integration | **Explicit RK2** | Simple, correct first. dt can be upgraded to implicit later. |
| Grid | **mesh=mass (hybrid mass-shell)** | Matches radial1d geometry exactly for direct comparison. |
| Gravity | **2D Poisson via GMG** | Existing `GmgGpu` is reused; well-balanced is straightforward. |
| Viscosity | **TW in both r and θ** | Symmetric; standard MESA discretization extended. |
| Well-balancing | **Flux-level HSE subtraction** | Existing FAS mechanism reused (P0r, rho0r lambdas). |
| dt limit | **min(acoustic, compression)** | Both directions contribute; compression limiter prevents origin runaway. |

---

## Variable Layout

All quantities in the standard `(nr, ntheta)` grid with ghost cells (same as Grid class).

### Conservative (cell-centered, size `nr × ntheta`)
| Variable | Meaning |
|---|---|
| `rho[k]` | density |
| `mr[k]` | radial momentum = ρ v_r |
| `mt[k]` | polar momentum = ρ v_θ |
| `rhoE[k]` | total energy density = ρ(e + ½v²) |

### HSE reference (copied from initial state)
| Variable | Meaning |
|---|---|
| `rho0[flat]`, `P0[flat]` | initial HSE profile (flat = i*ntheta + j) |
| `hse_defect[4*flat]` | pre-computed R(U₀), subtracted from residual |

### Stability buffers
| Variable | Meaning |
|---|---|
| `Pvsc_r[flat]`, `Pvsc_t[flat]` | TW viscosity in r and θ (added to pressure at flux-face) |
| `dt_cell[flat]` | per-cell dt, min-reduced |

### Gravity
| Variable | Meaning |
|---|---|
| `phi[flat]` | gravitational potential (solved by GMG) |
| `gr[flat]`, `gt[flat]` | gravity components at cell center |

---

## Core Equations (Eulerian, conservation form)

Let $U = (\rho, m_r, m_\theta, \rho E)$ on spherical coordinates $(r, \theta)$.

### Residual

$$R_{ij}(U) = -\frac{1}{V_{ij}}\left[A^r_{i+\tfrac{1}{2},j} \mathcal{F}^r_{i+\tfrac{1}{2},j} - A^r_{i-\tfrac{1}{2},j} \mathcal{F}^r_{i-\tfrac{1}{2},j} + A^\theta_{i,j+\tfrac{1}{2}} \mathcal{F}^\theta_{i,j+\tfrac{1}{2}} - A^\theta_{i,j-\tfrac{1}{2}} \mathcal{F}^\theta_{i,j-\tfrac{1}{2}}\right] + S_{ij}^{\text{geom}} + S_{ij}^{\text{grav}}$$

### Flux with WB and TW viscosity

At face $i+\tfrac{1}{2}, j$ (radial):
1. MUSCL reconstruction using **perturbation variables**: $\delta \rho = \rho - \rho_0$, $\delta P = P - P_0$, keep $v_r, v_\theta$ raw.
2. Reconstruct L and R face-states, add back $\rho_0, P_0$ at face.
3. HLLC on reconstructed states.
4. **Subtract $P_0^{\text{face}}$ from momentum flux** (flux-level WB):
   $$\mathcal{F}^r_{m_r} \leftarrow \mathcal{F}^r_{m_r} - P_0^{\text{face}}$$
5. Add viscous pressure: replace $P$ in HLLC input with $P + Q^r$ where $Q^r$ is the TW viscosity at that cell.

Same pattern for θ direction.

### Geometric source

$$S_{m_r}^{\text{geom}} = \rho v_\theta^2 / r, \quad S_{m_\theta}^{\text{geom}} = -\rho v_r v_\theta / r$$

(For θ-symmetric flow these vanish exactly.)

### Gravity source (WB form)

Let $g_r$ come from $-\nabla \phi$ where $\phi$ is solved via 2D Poisson (GMG). WB form:
$$S_{m_r}^{\text{grav}} = \rho' g_0 + \rho g'$$

where $\rho' = \rho - \rho_0$, $g_0 = g(\rho_0)$, $g' = g - g_0$. At HSE, $\rho' = 0$ and $g' = 0$ so source = 0.

### Tscharnuter-Winkler artificial viscosity (per cell, per direction)

**Radial**:
$$Q^r_{ij} = \begin{cases} (C_Q/V_{ij})(\Delta v_r - Z_{SH}\sqrt{P_{ij} V_{ij}})^2 & \Delta v_r > Z_{SH}\sqrt{P V} \\ 0 & \text{otherwise} \end{cases}$$

where $\Delta v_r = v_{r,\text{face\_inner}} - v_{r,\text{face\_outer}} > 0$ means radial compression.

**Polar**:
$$Q^\theta_{ij} = \begin{cases} (C_Q/V_{ij})(\Delta v_\theta - Z_{SH}\sqrt{P V})^2 & \Delta v_\theta > Z_{SH}\sqrt{P V} \\ 0 & \text{otherwise} \end{cases}$$

where $\Delta v_\theta = v_{\theta,\text{face\_north}} - v_{\theta,\text{face\_south}}$.

For cell-centered $v_\theta$: compute from neighbor differences $(v_{\theta,j-1} - v_{\theta,j+1})/2$ or similar.

These act additively in HLLC flux: the face-pressure in the Riemann problem is $P_L + Q$, $P_R + Q$ (use max(Q_L, Q_R) to stay single-valued at the face).

### dt Limit

$$\Delta t = \text{CFL} \cdot \min_{ij}\left( \frac{\Delta r_i}{|v_r|+c_s}, \frac{r_i \Delta\theta_j}{|v_\theta|+c_s}, f_{\text{comp}} \cdot \frac{\Delta r_i}{\max(0, \Delta v_r)}, f_{\text{comp}} \cdot \frac{r_i \Delta\theta_j}{\max(0, \Delta v_\theta)} \right)$$

with $f_{\text{comp}} \sim 0.1$ (per MESA RSP convention).

### Surface P Floor

Before computing fluxes, enforce:
$$P_{ij} \geq P_{\text{floor},ij}, \quad P_{\text{floor},ij} = 0.1 \cdot P_0[\text{flat}]$$

applied uniformly to all cells (no condition on `evacuated` or `hot_vacuum`). This replaces the existing `k_fas_atm_reset` mechanism without introducing θ-asymmetry.

### Central v_r Damping

Reuse existing `k_fas_central_damp` as-is. It is conservative (removes KE from rhoE) and applied uniformly across all j → θ-symmetric.

---

## Reused Components (no modification)

| Component | Location | Usage |
|---|---|---|
| `Grid` (mesh_mass, geometry) | `src/grid.*` | Mesh initialization |
| `State`, `EOS` | `src/state.*`, `src/eos.h` | Data layout |
| `init_lane_emden*` | `src/init/lane_emden.*` | Initial conditions |
| HLLC implementation | `src/gpu/fas_hllc.cuh` | `fas_hllc`, `fas_hllc_lm` |
| MUSCL reconstruction | `src/gpu/fas_hllc.cuh` | `fas_recon`, `fas_limit` |
| Index helpers | `src/gpu/fas_common.cuh` | `fas_idx`, `CUDA_CHECK` |
| GMG Poisson | `src/gpu/gmg_gpu.*` | Gravity solve |
| VTK output | `src/io/output.cpp` | Uses `State`, no modification needed |
| Lane-Emden solver | `src/init/lane_emden.cpp` | ODE + interpolation |

**Reused stability kernels**:
- `k_fas_central_damp` (conservative, safe to reuse)
- `k_fas_sponge` (optional, same as FAS)

---

## New Files (all in `src/gpu/`)

| File | Est. LOC | Purpose |
|---|---|---|
| `wb2d_solver.cuh` | ~120 | Struct, API declarations |
| `wb2d_solver.cu` | ~250 | init, step, destroy, diagnostics, snapshot_hse |
| `wb2d_kernels.cu` | ~400 | All device kernels (residual, TW viscosity, CFL, floor) |

**Total**: ~770 lines (vs 2500 estimated for the earlier Lagrangian-Eulerian hybrid).

No new `.cuh` for kernels — declarations inline in `wb2d_kernels.cu`, with forward decls in `wb2d_solver.cuh` used by `wb2d_solver.cu`.

---

## Struct

```cpp
struct Wb2DSolver {
    // State (device arrays, all size total_with_ghost = (nr+2*ng)*(nt+2*ng))
    double *d_rho, *d_mr, *d_mt, *d_rhoE;

    // HSE reference (size nr*nt, no ghost)
    double *d_rho0, *d_P0, *d_gr0;
    double *d_hse_defect;       // 4*nr*nt

    // TW viscosity (cell-centered, size nr*nt)
    double *d_Pvsc_r, *d_Pvsc_t;

    // Gravity
    double *d_phi;              // (nr*nt), solved by GMG
    double *d_gr, *d_gt;        // (nr*nt), cell-centered

    // Scratch
    double *d_dt_cell;          // (nr*nt)
    double *d_res;              // 4*nr*nt
    double *d_Un;               // 4*nr*nt (for RK2 save)

    // Grid geometry (uploaded from Grid class)
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;

    // GMG instance for Poisson
    GmgGpu gravity_gmg;

    // Physical / numerical parameters
    double gamma, G_const, cfl;
    double CQ = 2.0;              // TW viscosity coefficient
    double ZSH = 0.1;             // TW shock threshold
    double comp_dt_frac = 0.1;    // compression dt fraction
    double P_floor_frac = 0.1;    // P >= P_floor_frac * P0
    double central_damp_r = 0.0;  // (optional) central damping radius
    int limiter_type = 0;         // 0=minmod, 1=vanleer, 2=MC

    int nr, nt, ng;
    int step_count = 0;
    double dt_current = 0.0;
    bool hse_set = false;

    // Lifecycle
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void destroy();

    // I/O
    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);
    void snapshot_hse();

    // Integration
    double step(double t, double t_end);
    double compute_cfl_dt();
};
```

---

## Time Step (RK2)

```
step(t, t_end):
    apply_floor()               # P >= P_floor; ρ >= ρ_floor
    fill_ghost_cells()
    solve_gravity()             # GMG Poisson using current ρ
    compute_cfl_dt()            # dt = min(acoustic_r, acoustic_t, comp_r, comp_t)

    save U^n to d_Un

    # --- Stage 1 ---
    compute_tw_viscosity()      # Q^r, Q^θ from current state
    compute_residual()          # HLLC with WB + viscosity in both r and θ
    subtract_hse_defect()       # R(U) -= R(U₀)
    U = U^n + dt * R

    apply_floor()
    fill_ghost_cells()
    solve_gravity()

    # --- Stage 2 ---
    compute_tw_viscosity()
    compute_residual()
    subtract_hse_defect()
    U = U + dt * R              # (U is now U^n + dt*R1 + dt*R2)

    # RK2 average
    U = 0.5 * (U^n + U)

    apply_floor()
    central_damp_vr (optional)

    step_count++
    return dt
```

Differences from existing FAS `step_explicit`:
- **No atm_reset** (replaced by uniform floor)
- **TW viscosity added** at flux level
- **compression dt limit** included in CFL computation
- **2D Poisson gravity** instead of 1D angular-averaged (though we could use 1D too for direct radial1d comparison)

---

## Residual Kernel Design (critical piece)

The residual kernel is the heart. Design approach:

**Option 1**: Write a new kernel `k_wb2d_residual` that's a near-copy of `k_fas_residual` but with TW viscosity added and no hard-coded `radial_only`. **Preferred.** Keeps module independent.

**Option 2**: Add a `use_tw_viscosity` flag to `k_fas_residual`. Invasive; rejected.

The new kernel:
```
for each cell (i, j):
    # Reconstruct using MUSCL on δρ = ρ - ρ₀, δP = P - P₀
    # Radial face i+1/2
    wL, wR = reconstruct_r(i, j)
    F_r_hi = hllc_r(wL, wR, P_eff = P + Q^r)  # viscosity adds to pressure
    F_r_hi.mr -= 0.5*(P0[i,j] + P0[i+1,j])    # flux-level WB

    F_r_lo = similar for face i-1/2

    # Theta face j+1/2
    ...

    # Geometric source (uses cell-centered v_r, v_θ)
    S_mr_geom = ρ v_θ² / r
    S_mt_geom = -ρ v_r v_θ / r

    # Gravity (WB form)
    S_grav_r = ρ' g_r0 + ρ g_r'

    # Assemble residual
    R_rho = -div(F.rho)
    R_mr  = -div(F.mr)  + S_mr_geom + S_grav_r
    R_mt  = -div(F.mt)  + S_mt_geom + S_grav_t
    R_E   = -div(F.E)   + ρ v_r g_r + ρ v_θ g_θ

    write to res[4*flat + {0,1,2,3}]
```

---

## Implementation Plan (1 day target)

| Step | Est. time | Content |
|---|---|---|
| 1 | 20 min | `wb2d_solver.cuh`: struct + API |
| 2 | 30 min | `wb2d_solver.cu`: init/destroy/upload/download (copy from FAS) |
| 3 | 30 min | `snapshot_hse` + `compute_hse_defect` (adapt from FAS) |
| 4 | 45 min | TW viscosity kernels (r and θ) |
| 5 | 1.5 hr | Main residual kernel (HLLC + WB + viscosity) |
| 6 | 30 min | CFL kernel (acoustic + compression) |
| 7 | 30 min | Floor kernel (P >= P_floor uniform) |
| 8 | 30 min | `step()` orchestration (RK2, floor, ghost, gravity) |
| 9 | 20 min | CMakeLists + main.cpp integration |
| 10 | 20 min | First build + smoke test |
| 11 | 2 hr | Debug, HSE test, Lane-Emden perturbed test |

**Total**: ~7-8 hours of focused work. Realistic for 1 day.

---

## Validation Plan

### Test 1: HSE stability (unperturbed Lane-Emden)
```
./stellar2d --solver wb2d --test lane_emden --nr 128 --ntheta 128 --tend 10 --mesh mass
```
Expect: max|v| < 1e-6, ΔM/M < 1e-12, ΔE/|E₀| < 1e-6

### Test 2: Radial perturbation vs radial1d
```
./stellar2d --solver wb2d --test lane_emden_perturbed --nr 256 --ntheta 64 --tend 10 --mesh mass
./stellar2d --solver radial1d --test lane_emden_perturbed --nr 256 --tend 10
```
Expect: mass conservation comparable (flux-form ~1e-10, not exact 0); energy drift within 2× of radial1d; Mach and |v| profiles qualitatively matching at similar t.

### Test 3: θ-symmetry preservation
Any symmetric IC should maintain max|θ-stddev(ρ)|/mean(ρ) < 1e-12 for all t.

### Test 4: Bubble (non-symmetric)
```
./stellar2d --solver wb2d --test bubble --nr 128 --ntheta 128 --tend 1 --mesh mass
```
Expect: stable evolution, no runaway; qualitative bubble rise pattern.

---

## Known Limitations (acknowledged up front)

1. **Not as accurate on purely-radial problems as radial1d.** Flux-form conservation is 1e-10 vs 0; explicit RK2 more diffusive than Lagrangian PdV.
2. **Subject to acoustic CFL.** Implicit version can come later if needed.
3. **Origin still has geometric singularity.** Mitigated by central_damp + compression-dt, but not eliminated like in Lagrangian.
4. **Well-balancing is flux-level.** Not as perfect as structural WB (Käppeli-Mishra higher-order), but good enough for the current problem class.

---

## Future Extensions (not in this phase)

- **Implicit JFNK time-stepping** (reuse FAS infrastructure as preconditioner — potentially novel?)
- **Higher-order WB** (Käppeli-Mishra 2016; Berberich et al. 2021)
- **Full 2D Lagrangian ALE** (long-term, 6+ months)
- **Well-balanced treatment of ionization zones** (for full stellar applications)

---

## Checklist

- [ ] Design doc (this file) — done
- [ ] `wb2d_solver.cuh`
- [ ] `wb2d_solver.cu`
- [ ] `wb2d_kernels.cu`
- [ ] CMakeLists entry
- [ ] main.cpp --solver wb2d branch
- [ ] Build clean
- [ ] HSE test passes
- [ ] Perturbed test matches radial1d within 2×
- [ ] Bubble test stable
