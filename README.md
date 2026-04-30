# stellar2d

A 2D axisymmetric compressible Euler + self-gravity solver for proof-of-concept
stellar convection simulations.  All computation runs on GPU (CUDA).

## Physical Model

The code solves the compressible Euler equations coupled with self-gravity via
a Poisson equation.  Two coordinate systems are supported:

| Coordinate | Grid | Solvers |
|---|---|---|
| Axisymmetric spherical $(r,\theta)$ | Log-radial + uniform polar | explicit, FAS, SIMPLE, projection, lowmach |
| 2D Cartesian $(x,y)$ | Uniform | strang |

All discrete equations are documented in [docs/equations.md](docs/equations.md);
physics computations in the source are annotated with the corresponding equation
number where applicable.

- Ideal gas EOS ($\gamma = 5/3$)
- MUSCL reconstruction + HLLC Riemann solver (second-order Godunov method)
- Well-balanced (WB) discretisation: reference-state subtraction for HSE preservation
- Low-Mach HLLC (LM-HLLC): acoustic-speed blending $f(M)$ for buoyancy-driven flows
- Self-gravity: GPU geometric multigrid (GMG) Poisson solver, or 1D radial mass integral
- Time integration: RK2 (explicit), Backward Euler + JFNK (lowmach), Strang splitting (strang)

## Solvers

### `--solver explicit` (default for polar grid)

Second-order Godunov on the polar grid.  MUSCL-Hancock + HLLC, RK2 time
integration.  CFL-limited by sound speed.  Well-balanced geometric source terms
(Eq. 5.3--5.4).  Self-gravity via GPU GMG Poisson solve.

### `--solver fas`

FAS (Full Approximation Scheme) nonlinear multigrid on the polar grid.
Block-Jacobi smoothing, V-cycle with restrict/prolongate transfers.  Intended
for implicit time stepping but currently limited by smoother convergence at
large CFL.

### `--solver simple`

SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) pressure-correction
on the polar grid.  Momentum predictor + pressure Poisson + velocity corrector
inner loop, with outer time-stepping.

### `--solver projection`

Fractional-step pressure projection on the polar grid.  Explicit advection step
followed by pressure Poisson projection to enforce a discrete divergence
constraint.

### `--solver lowmach`

Fully implicit low-Mach solver using JFNK (Jacobian-Free Newton-Krylov) with
right-preconditioned FGMRES.  Well-balanced residual with reference-state
subtraction (Eq. 10.5).  Preconditioner options: block-Jacobi, line-Jacobi,
physics-based (PBP with Schur complement pressure Poisson, Eq. 11.1--11.6).
1D radial gravity (angle-averaged density, cumulative mass integral, Eq. 10.4).

### `--solver strang` *(Cartesian grid)*

Strang-splitting MUSCL-Hancock on a uniform Cartesian grid.  Splitting:
$X(\Delta t/2) \to Y(\Delta t/2) \to Y(\Delta t/2) \to X(\Delta t/2)$.
MC limiter, LM-HLLC Riemann solver, well-balanced y-sweep with perturbation
storage ($\rho' = \rho - \bar\rho$, $E' = E - \bar E$).  See
[docs/equations.md §13](docs/equations.md) for details.

## Code Structure

```
src/
├── grid.h/cpp                  Mesh: log-radial + uniform polar, precomputed geometry
├── state.h/cpp                 Conservative <-> primitive variable conversion
├── eos.h                       Ideal gas equation of state
├── main.cpp                    CLI, init, solver dispatch, time-stepping loop
├── hydro/
│   ├── reconstruct.h/cpp       MUSCL reconstruction (minmod / van Leer limiters)
│   ├── riemann.h/cpp           HLLC Riemann solver (CPU reference)
│   ├── flux.h/cpp              Flux divergence + geometric source + gravity source
│   └── integrate.h/cpp         RK2 time integration + CFL condition
├── gravity/
│   ├── poisson.h/cpp           Poisson matrix assembly (CSR, 5-point spherical stencil)
│   ├── gmg.h/cpp               CPU geometric multigrid (V-cycle, red-black Gauss-Seidel)
│   └── amgx_solver.h/cpp       AmgX wrapper (USE_AMGX) / CPU Jacobi fallback
├── bc/boundary.h/cpp           BCs: reflecting centre, axis symmetry, outflow
├── io/output.h/cpp             VTK output + conservation diagnostics
├── init/
│   ├── lane_emden.h/cpp        Lane-Emden polytropic equilibrium + bubble perturbation
│   ├── sedov.h/cpp             Sedov point-source explosion
│   ├── jeans.h/cpp             Jeans gravitational instability
│   └── evrard.h/cpp            Evrard cold gas sphere collapse
└── gpu/
    ├── strang_solver.cu/cuh    Strang Cartesian solver (splits, sweeps, BCs, VTK)
    ├── strang_device.cuh       Device functions: MC limiter, LM-HLLC, HSE helpers
    ├── fas_solver.cu/cuh       FAS nonlinear multigrid (V-cycle, time stepping)
    ├── fas_residual.cu         FAS/explicit residual + HLLC + ghost cell kernels
    ├── fas_hllc.cuh            HLLC + minmod MUSCL for polar grid (device)
    ├── fas_common.cuh          Index helpers, FasLevel struct
    ├── fas_smoothers.cu        Block-Jacobi + SIMPLE smoothers for FAS
    ├── fas_multigrid.cu        Restrict / prolongate / V-cycle
    ├── fas_linalg.cuh          GPU linear algebra (reductions, preconditioner kernels)
    ├── simple_solver.cu/cuh    SIMPLE pressure-correction solver
    ├── projection_solver.cu/cuh  Pressure projection solver
    ├── lowmach_solver.cu/h     JFNK low-Mach implicit solver (main driver)
    ├── lm_residual.cu          Low-Mach well-balanced residual kernel
    ├── lm_precond.cu           Preconditioners: block-Jacobi, line-Jacobi, PBP
    ├── lm_krylov.cu            FGMRES + JFNK matvec + Newton loop
    ├── lm_common.cuh           Shared device helpers for low-Mach solver
    ├── gmg_gpu.cu/cuh          GPU geometric multigrid Poisson / Helmholtz solver
    └── gpu_solver.cu/h         AmgX-backed compressible solver (requires USE_AMGX)

tests/
├── test_strang_init.cu         Strang: HSE balance, bubble init, mass, VTK (8 checks)
├── test_strang_muscl.cu        MC limiter, WB y-sweep reconstruction (11 checks)
├── test_strang_hllc.cu         LM-HLLC: uniform/Sod/LM/antisymmetry/HSE (10 checks)
├── test_strang_step.cu         Strang integration: HSE, mass, buoyancy, CFL (9 checks)
├── test_strang_unit.cu         EOS, ghost cells, HLLC edge cases (12 checks)
├── test_strang_convergence.cu  Entropy wave 2nd-order convergence (3 checks)
├── test_fas_verify.cu          Polar: minmod, recon, HLLC, ghost, HSE, convergence (19 checks)
├── test_coverage_critical.cu   LM origin, GMG, FAS restrict/floor/CFL, solvers, init (16 checks)
├── test_lowmach.cu             JFNK solver: HSE, Newton convergence, dt growth
├── test_solver_diagnosis.cu    GMRES quality, line search profile, matvec accuracy
├── test_precond_quality.cu     Preconditioner comparison sweep
├── test_unit.cpp               CPU: grid geometry, EOS, minmod, HLLC (unit tests)
├── test_exact.cpp              CPU: Lane-Emden analytic comparison
├── test_pitfalls.cpp           CPU: regression tests for historical bugs (P01-P05)
└── (Python)                    pytest: conservation, symmetry, convergence, HSE
```

## Building

Requirements: C++17 compiler, CMake >= 3.18, CUDA Toolkit.

```bash
mkdir build_gpu && cd build_gpu

# GPU build (all solvers)
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON
make -j$(nproc)

# Optional: with AmgX for --solver compressible
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON -DAMGX_DIR=/path/to/amgx
make -j$(nproc)
```

CPU-only builds are possible but only support the original hydro/ path (no GPU
solvers, no FAS, no low-Mach):

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Running

```bash
# Polar grid — Lane-Emden HSE (explicit solver, default)
./stellar2d --test lane_emden --nr 128 --ntheta 64 --tend 1.0

# Polar grid — bubble convection (explicit)
./stellar2d --test bubble --solver explicit --nr 128 --ntheta 64 --tend 0.3

# Polar grid — implicit low-Mach
./stellar2d --test bubble --solver lowmach --nr 64 --ntheta 32 --tend 0.1

# Polar grid — SIMPLE / projection / FAS
./stellar2d --test bubble --solver simple --nr 64 --ntheta 32 --tend 0.1
./stellar2d --test bubble --solver projection --nr 64 --ntheta 32 --tend 0.1
./stellar2d --test bubble --solver fas --nr 64 --ntheta 32 --tend 0.1

# Cartesian grid — Strang splitting (not yet integrated into main.cpp)
# Run via test suite: ./test_strang_step

# Full option list
./stellar2d --test <case> --solver <solver> --nr <N> --ntheta <N> \
    --tend <T> --cfl <C> --output-interval <N> \
    --precond <pc> --bubble-mode <entropy|pressure> \
    --limiter <minmod|vanleer> --no-sponge --lm-hllc
```

| Option | Values | Default |
|---|---|---|
| `--test` | `lane_emden`, `lane_emden_perturbed`, `bubble`, `sedov`, `jeans`, `evrard` | `lane_emden` |
| `--solver` | `explicit`, `fas`, `simple`, `projection`, `lowmach`, `compressible`* | `compressible` |
| `--precond` | `block_jacobi`, `line_jacobi`, `simple`, `pbp`, `none` | `line_jacobi` |
| `--bubble-mode` | `pressure`, `entropy` | `pressure` |
| `--limiter` | `minmod`, `vanleer` | `minmod` |
| `--lm-hllc` | (flag) enable LM-HLLC acoustic blending | off |
| `--no-sponge` | (flag) disable velocity sponge layer | off |

\* `compressible` requires AmgX.

Output is VTK format (`output_XXXX.vtk`), viewable in ParaView.

## Testing

```bash
cd build_gpu

# Run all GPU tests
for t in test_strang_init test_strang_muscl test_strang_hllc \
         test_strang_step test_strang_unit test_strang_convergence \
         test_fas_verify test_coverage_critical; do
    ./$t
done

# Or via CTest
ctest --output-on-failure

# Memory safety (NVIDIA compute-sanitizer)
compute-sanitizer --tool memcheck ./test_strang_step
```

Current test status: **88 checks across 8 GPU test suites, all passing.**

| Suite | Checks | Coverage |
|---|---|---|
| `test_strang_init` | 8 | HSE balance, bubble init, mass, pressure, VTK |
| `test_strang_muscl` | 11 | MC limiter (7 cases), WB y-sweep (Deltarho/DeltaP/Deltav), face bounds |
| `test_strang_hllc` | 10 | Uniform, Sod, LM-blend, antisymmetry, HSE-column, contact |
| `test_strang_step` | 9 | HSE preservation, mass conservation, buoyancy, CFL |
| `test_strang_unit` | 12 | EOS roundtrip, MC edge cases, ghost x/y/outflow, HLLC contact/vacuum |
| `test_strang_convergence` | 3 | Entropy wave 2nd-order: order 1.92 (L1), 1.99 (L2) |
| `test_fas_verify` | 19 | Polar minmod, MUSCL recon, HLLC (6 cases), ghost r/theta/pole, explicit HSE, self-convergence |
| `test_coverage_critical` | 16 | LM origin residual, GMG Poisson, FAS restrict/floor/CFL, SIMPLE/projection smoke, Jeans/Evrard/Bubble init |

## Documentation

- [docs/equations.md](docs/equations.md) -- Authoritative equation reference (sections 1--14)
- [docs/pitfalls.md](docs/pitfalls.md) -- Bug log with root cause analysis (P01--P29)
- [docs/provenance.md](docs/provenance.md) -- Figure/data traceability convention

## Equation-Code Traceability

This project enforces a strict correspondence between equations and source code:

1. [docs/equations.md](docs/equations.md) serves as the authoritative equation reference.
2. Every physics computation in the source is annotated with `// Eq. (X.Y)`.
3. Any discrepancy between the code and the reference document constitutes a bug.
