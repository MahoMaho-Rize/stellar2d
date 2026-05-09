# ALE2D Solver: Phased Design Document

**Status**: Design (pre-implementation)
**Date**: 2026-04-30
**Branch**: radial-only-mode
**Scope**: Full 2D Lagrangian/ALE solver — new module, non-invasive.

---

## Executive Summary

Pure 2D Eulerian with MESA-style stabilization (wb2d) is confirmed insufficient:
it survives ~2× longer with angular_avg + pole_avg + sponge + central_damp, but
still blows up at t≈2 on `lane_emden_perturbed`. The fundamental issue is
geometric focusing at r→0 that Eulerian grids cannot escape. ALE
(Arbitrary Lagrangian-Eulerian) is the standard industry answer.

This document defines a **phased build** of a staggered quadrilateral ALE solver.
Each phase delivers a validated milestone; later phases add capability on top.

| Phase | Scope | LOC | Target |
|---|---|---|---|
| **A1** | Pure Lagrangian structured (i,j) quad mesh | ~2000 | Match radial1d on symmetric IC |
| **A2** | Caramana compatible hydro (exact E conservation) | +800 | ΔE/\|E\| < 1e-3 at t=100 |
| **A3** | Rezone + radial-only remap | +1500 | Long-time radial IC stable |
| **A4** | Local mesh reconnection (edge/node merging) | +2000 | Weak-vorticity IC (bubble) stable |
| **A5** | Full Voronoi/ReALE unstructured mesh | +3000 | Strong vorticity / convection |

Total at A4: ~6000 LOC, estimated 4-6 weeks. A5 is production research (6+ months).

**A1-A2 is the near-term target**. Do not start A3+ until A2 is validated.

---

## Why ALE (vs Eulerian or pure Lagrangian)

| Approach | Mass | Origin | 2D vorticity |
|---|---|---|---|
| Eulerian (current) | ~1e-7 drift | Geometric runaway | Fine |
| Pure Lagrangian (rigid mesh) | Exact | Clean | **Tangles in finite time** (Dukowicz-Meltz 1992) |
| ALE (Lagrangian + rezone) | Exact if done right | Clean | Rezone prevents tangling |

ALE = Lagrangian for the physics step, then rezone+remap when mesh quality degrades.
A1-A2 skip rezone/remap (treats only radially-symmetric or slowly-deforming cases);
A3 adds simple radial remap; A4+ adds full reconnection.

---

## Geometry: Staggered Quadrilateral Mesh

**Topology**: preserved from current `Grid` — (nr, nt) structured (i,j) connectivity.
**Positions**: nodes can move freely in (r, θ) space.

### Node-Centered (faces of the current grid)
Size `(nr+1) × (nt+1)` nodes.

| Variable | Symbol | Purpose |
|---|---|---|
| `x[i,j]` | r·sin θ | Cartesian X position (Lagrangian-varying) |
| `y[i,j]` | r·cos θ | Cartesian Y position |
| `vx[i,j]` | | node velocity in X |
| `vy[i,j]` | | node velocity in Y |

Using Cartesian internally avoids θ-wraparound singularities at the pole and makes
finite-volume geometry integrals exact for quadrilateral cells.

### Cell-Centered
Size `nr × nt` cells. Cell (i,j) is bounded by nodes (i,j), (i+1,j), (i+1,j+1), (i,j+1).

| Variable | Purpose |
|---|---|
| `dm[i,j]` | zone mass (exactly constant in A1) |
| `Vol[i,j]` | zone volume (recomputed each step) |
| `rho[i,j]` | density = dm/Vol |
| `e_int[i,j]` | specific internal energy |
| `P[i,j]` | gas pressure from EOS |
| `Q[i,j]` | artificial viscosity (scalar, Caramana or TW) |

### HSE Reference
| Variable | Purpose |
|---|---|
| `rho0[i,j]`, `P0[i,j]`, `e0[i,j]` | initial HSE (for diagnostics + surface floor) |

### Axisymmetric Geometry (r-φ rotation)

The physical problem is 2D axisymmetric — we simulate one (r, θ) plane and treat
it as a surface of revolution around the symmetry axis. Cell volume (exact
integration for a revolved quadrilateral):

$$V = \pi \cdot \oint (x^2 - \langle x \rangle^2) \, dy \approx \pi \sum_k x_k x_{k+1} (y_{k+1} - y_k)$$

(sum over the 4 edges of the quadrilateral).

Face "area" for flux computation is not needed in pure Lagrangian — fluxes are
replaced by subcell forces. See §Equations.

---

## Equations (A1: pure Lagrangian)

### Mass (A1: trivially conserved)
$$\frac{d m_{ij}}{dt} = 0$$

Every cell's mass is fixed; density evolves via volume change: `ρ = dm/V`.

### Momentum (node-based)
Per node (i,j):
$$m_{\text{node}} \frac{d \mathbf{v}}{dt} = \sum_{c \in \mathrm{adj}(i,j)} \mathbf{F}_{c \to \text{node}}$$

where the 4 adjacent cells each contribute a subcell pressure force:
$$\mathbf{F}_{c \to n} = -(P_c + Q_c) \cdot \mathbf{A}_{c,n}$$

and $\mathbf{A}_{c,n}$ is the outward-normal area vector of the subcell face opposite the node.
For a quadrilateral with 4 nodes, the subcell "corner" force is the negative pressure
times half the sum of the two adjacent edge-normal area vectors (standard Wilkins
discretization).

Node mass $m_{\text{node}}$ is 1/4 of adjacent cell mass sum (uniformly distributed).

Gravity force adds pointwise: $\mathbf{F}_g = m_{\text{node}} \cdot \mathbf{g}$
where g is computed from 1D angle-averaged shell mass (reuse from FAS).

### Energy (A1: non-compatible, fixed in A2)
$$\frac{d e_{\text{int},c}}{dt} = -(P_c + Q_c) \frac{1}{V_c} \frac{dV_c}{dt}$$

This is the adiabatic PdV form. **Not exactly E-conserving** in A1 — the work done
on nodes doesn't exactly equal minus the work done by cells due to staggering.
A2 fixes this via Caramana's compatible hydro (total E error → machine zero).

### Artificial Viscosity (A1: scalar Tscharnuter-Winkler)

Per cell:
$$Q_c = \begin{cases} (C_Q / V_c)(\Delta v - Z_{SH} \sqrt{P V})^2 & \Delta v > Z_{SH} \sqrt{P V} \\ 0 & \text{otherwise}\end{cases}$$

where $\Delta v = -(1/V) dV/dt$ is the volume compression rate. This directly
generalizes the 1D radial1d formula. A2 may upgrade to Caramana-Shashkov 2001
tensor viscosity.

### Time Step
$$\Delta t = \mathrm{CFL} \cdot \min_c \left( \frac{L_c}{|v|+c_s}, \frac{f_{\text{comp}}}{|\dot V / V|} \right)$$

where $L_c$ is the minimum edge length of cell c.

Additional **geometric limit**: if any cell's projected new volume would be negative,
reduce dt. (In A1 with moderate perturbations this doesn't trigger; in A3+ it's
the trigger for rezone.)

### Boundary Conditions

- **Inner (i=0 nodes)**: pinned at r=0 (vx=vy=0). Innermost nodes don't move.
- **Outer (i=nr nodes)**: free surface. Pressure beyond = `P_surf_floor` (HSE value).
- **θ=0, π nodes (j=0, nt)**: constrained to the symmetry axis (x=0).
  The node can move in y but not x. This is a reflective constraint.

---

## Time Integration (A1: Predictor-Corrector RK2)

Standard Heun:
```
Stage 1: U* = Uⁿ + dt · R(Uⁿ)
Stage 2: Uⁿ⁺¹ = ½(Uⁿ + U* + dt·R(U*))
```

Where U is the full state (nodes + cells) and R is:
- Node: force-to-acceleration (→ dv/dt), velocity-to-position (→ dr/dt)
- Cell: volume recompute, adiabatic e update

No implicit solve in A1. Explicit CFL.

---

## Module Structure (A1)

```
src/gpu/
├── ale2d_solver.cuh       (~150 LOC)  Struct + API
├── ale2d_solver.cu        (~400 LOC)  init/step/destroy/diagnostics
├── ale2d_geom.cu          (~300 LOC)  volume/area/edge kernels
├── ale2d_kernels.cu       (~600 LOC)  physics kernels (force, PdV, Q, CFL)
└── ale2d_init.cu          (~200 LOC)  Lane-Emden → 2D quad mesh
```

Total A1 estimate: ~1650 LOC. (~2-3 hours of typing, plus debugging.)

---

## Struct (A1)

```cpp
struct Ale2DSolver {
    // Node arrays, size (nr+1)*(nt+1)
    double *d_x, *d_y;
    double *d_vx, *d_vy;
    double *d_fx, *d_fy;        // accumulated node force
    double *d_mnode;             // node mass (constant)
    double *d_x_prev, *d_y_prev; // RK2 save
    double *d_vx_prev, *d_vy_prev;

    // Cell arrays, size nr*nt
    double *d_dm;                // zone mass (constant)
    double *d_Vol, *d_Vol_prev;
    double *d_rho, *d_e_int, *d_e_prev;
    double *d_P, *d_Q;

    // HSE reference
    double *d_rho0, *d_P0, *d_e0;

    // Gravity (1D angle-averaged for A1)
    double *d_shell_mass, *d_gr;

    // Scratch
    double *d_dt_cell;

    int nr, nt;  // physical cells
    // (no ghost cells — node-centered physical BCs handle edges)

    double gamma, G_const, cfl;
    double CQ = 2.0, ZSH = 0.1;
    double comp_dt_frac = 0.25;
    double P_surf_floor = 0.0;

    int step_count = 0;
    double dt_current = 0.0;
    bool hse_set = false;

    // API
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void init_lane_emden(double rho_c, double K, double n);
    void apply_perturbation(double amplitude);
    void snapshot_hse();
    void destroy();

    double step(double t, double t_end);
    double compute_dt();
    Diagnostics compute_diagnostics();

    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);
    void write_vtk(const char* path);  // StructuredGrid with moving nodes
};
```

---

## Validation Plan (A1)

### Test 1: HSE (unperturbed)
```
./stellar2d run --solver ale2d --test lane_emden --nr 128 --ntheta 32 --tend 100
```
Expect: nodes stationary to relative 1e-10; ΔM = 0 exactly; ΔE/\|E\| < 1e-6.

### Test 2: Radial perturbation (vs radial1d)
```
./stellar2d run --solver ale2d --test lane_emden_perturbed --nr 256 --ntheta 32 --tend 10
```
Expect:
- ΔM = 0 (exact by construction)
- ΔE/\|E\| within 2× of radial1d (~2e-3 at t=10)
- θ-symmetry preserved to machine epsilon (max σ_θ(ρ)/mean(ρ) < 1e-12)

### Test 3: θ-symmetry preservation
Any θ-symmetric IC should stay θ-symmetric bit-for-bit.

### Test 4: Long-time stability
```
--tend 100
```
Expect: no mesh tangling for radial-only perturbations (would tangle only if
vortical motion develops, which it shouldn't from pure radial IC).

---

## Known Issues & Future Phases

### A1 limitations (accepted)
1. **Energy is not exactly conserved** — PdV + momentum work balance has O(h²) error.
   Fixed in A2 with Caramana compatible discretization.
2. **No mesh quality control** — if amplitude is large enough, cells can become
   badly skewed or inverted. Fixed in A3 (rezone) and A4 (reconnection).
3. **Gravity is 1D angle-averaged** — fine for axisymmetric problems, wrong for
   bubbles. Upgrade to 2D Poisson on Lagrangian mesh is an A3+ topic.

### A2: Caramana compatible hydro
Replace cell-centered $P dV/dt$ with summation over subcell force · node velocity.
Guarantees exact discrete E conservation: $\dot E_{\text{total}} = 0$ to machine
precision. ~800 LOC change to energy update + diagnostics.

### A3: Rezone + Remap (when mesh quality degrades)
- Monitor cell Jacobian / aspect ratio per step
- When below threshold, rezone to a smoother mesh (Winslow smoother)
- Remap conservatively: intersection-based exact polygon remap (NOT linear
  interpolation — that would dissipate).
- For A3, limit to **radial rezone only** (θ-nodes stay fixed).

### A4: Local reconnection
Per Margolin-Shashkov: when an edge becomes too short or a cell too degenerate,
collapse the edge (merging two nodes) or swap diagonals. This is where the code
starts to feel like an ALE research code.

### A5: Unstructured ReALE
Full Voronoi tessellation with reconnection. ~3-6 months of research-level
engineering.

---

## Implementation Plan (A1, 3-5 days target)

| Day | Step | Content |
|---|---|---|
| 1 AM | Struct + alloc | ale2d_solver.cuh/.cu skeleton |
| 1 PM | Geometry kernels | volume, edge lengths, node masses |
| 2 AM | Lane-Emden init | Lay out Lagrangian quads from equimass shells |
| 2 PM | Momentum kernel | Subcell corner forces (Wilkins style) |
| 3 AM | Energy + Q kernels | PdV + scalar TW viscosity |
| 3 PM | RK2 orchestration | step() wiring + dt reduction |
| 4 AM | Pole constraints + BC | x=0 for polar nodes, free surface |
| 4 PM | First build + HSE test | |
| 5 | Radial perturbation vs radial1d | Validation to ΔM=0, match 1D |

Aggressive but doable. If the quad-geometry kernel turns into a trap, expect +2 days.

---

## Checklist

- [x] Design doc (this file) — done
- [ ] Struct + alloc
- [ ] Geometry (volume, edge, node mass)
- [ ] Lane-Emden layout on Lagrangian quads
- [ ] Momentum kernel (subcell corner force)
- [ ] Energy + Q kernels
- [ ] RK2 orchestration
- [ ] Pole + surface BCs
- [ ] HSE test
- [ ] Radial perturbation match vs radial1d
- [ ] VTK writer for moving mesh

---

## Key References

- **Caramana, Shashkov, Whalen 1998**, "Formulations of Artificial Viscosity
  for Multi-Dimensional Shock Wave Computations", JCP 144, 70.
- **Caramana, Shashkov 1998**, "Elimination of Artificial Grid Distortion and
  Hourglass-type Motions by Means of Lagrangian Subzonal Masses and Pressures",
  JCP 142, 521. (Compatible hydro — basis for A2.)
- **Dukowicz, Meltz 1992**, "Vorticity Errors in Multidimensional Lagrangian
  Codes", JCP 99, 115. (Why pure Lagrangian tangles.)
- **Margolin, Shashkov 2003**, "Second-order sign-preserving conservative
  interpolation (remapping) on general grids", JCP 184, 266. (Basis for A3.)
- **Loubère, Maire, Váchal et al. 2010**, "ReALE: A reconnection-based
  arbitrary-Lagrangian-Eulerian method", JCP 229, 4724. (A4-A5.)
