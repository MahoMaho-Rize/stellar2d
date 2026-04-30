# stellar2d

A 2D axisymmetric compressible Euler + self-gravity solver for stellar convection simulations.

Two coordinate systems are supported:

| Coordinate | Grid | Solvers |
|---|---|---|
| Axisymmetric spherical $(r,\theta)$ | Log-radial + uniform polar | explicit, FAS, SIMPLE, projection, lowmach |
| 2D Cartesian $(x,y)$ | Uniform | strang |

## Quick Start

All operations go through the unified entry script `stellar2d.py`:

```bash
# 1. Build (CPU-only)
./stellar2d.py build

# 2. Run a simulation
./stellar2d.py run --test lane_emden --nr 128 --ntheta 64 --tend 1.0

# 3. Visualize results
./stellar2d.py plot runs/lane_emden_128x64_*/ --gif

# 4. Start interactive web viewer
./stellar2d.py viewer
```

For GPU builds (enables FAS, LowMach, SIMPLE, projection solvers):

```bash
./stellar2d.py build --gpu
./stellar2d.py run --test bubble --solver fas --nr 128 --ntheta 64
```

## Physical Model

The code solves the compressible Euler equations coupled with self-gravity via
a Poisson equation. All discrete equations are documented in
[docs/equations.md](docs/equations.md); physics computations in the source are
annotated with the corresponding equation number where applicable.

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

### Preconditioners (for `lowmach` / `fas`)

Set via `--precond`: `none`, `block_jacobi`, `simple`, `line_jacobi` (default), `block_schur`, `combined`, `pbp`.

## Test Cases

| Case | `--test` flag | Description |
|---|---|---|
| Lane-Emden equilibrium | `lane_emden` | Polytropic star ($n=1.5$), hydrostatic, zero velocity |
| Lane-Emden perturbed | `lane_emden_perturbed` | Equilibrium + 0.1% density perturbation, oscillation modes |
| Hot bubble | `bubble` | Entropy bubble in Lane-Emden star, buoyant convection |
| Sedov blast | `sedov` | Point-source explosion in uniform medium |
| Jeans instability | `jeans` | Uniform medium + perturbation, gravitational collapse |
| Evrard collapse | `evrard` | Cold isothermal gas sphere, gravitational collapse |

## CLI Reference

### `stellar2d.py build`

```
./stellar2d.py build [--gpu] [--amgx-dir PATH] [--debug]
```

| Flag | Description |
|---|---|
| `--gpu` | Enable CUDA (builds into `build/`; CPU goes to `build-cpu/`) |
| `--amgx-dir PATH` | Path to AmgX installation (enables compressible GPU solver) |
| `--debug` | Debug build with `-O0` and address sanitizer |

### `stellar2d.py run`

```
./stellar2d.py run [--test CASE] [--solver SOLVER] [--precond PRECOND]
                   [--mesh {log,equimass}] [--nr N] [--ntheta N]
                   [--tend T] [--cfl C] [--output-interval N]
```

| Option | Values | Default |
|---|---|---|
| `--test` | `lane_emden`, `lane_emden_perturbed`, `bubble`, `sedov`, `jeans`, `evrard` | `lane_emden` |
| `--solver` | `explicit`, `fas`, `simple`, `projection`, `lowmach`, `compressible`\* | `compressible` |
| `--precond` | `block_jacobi`, `line_jacobi`, `simple`, `pbp`, `none` | `line_jacobi` |
| `--mesh` | `log`, `equimass` | `log` |
| `--nr` | integer | `128` |
| `--ntheta` | integer | `64` |
| `--tend` | float | `1.0` |
| `--cfl` | float | `0.4` |
| `--output-interval` | integer | `100` |
| `--bubble-mode` | `pressure`, `entropy` | `pressure` |
| `--limiter` | `minmod`, `vanleer` | `minmod` |
| `--lm-hllc` | (flag) enable LM-HLLC acoustic blending | off |
| `--no-sponge` | (flag) disable velocity sponge layer | off |

\* `compressible` requires AmgX.

Output goes to `runs/<test>_<nr>x<nt>_<timestamp>/` with `output_XXXX.vtk` files.

### `stellar2d.py plot`

```
./stellar2d.py plot <run_dir> [--gif]
```

Renders every VTK frame as a 3-panel PNG (density, velocity+quiver, Mach). Add `--gif` to assemble an animated GIF.

### `stellar2d.py animate`

```
./stellar2d.py animate <run_dir> [-o FILE] [--skip N] [--fps N]
```

Generates a publication-quality GIF with mirrored 2D density map and radial profile.

### `stellar2d.py mach`

```
./stellar2d.py mach <run_dir> [--plot]
```

Prints per-frame Mach statistics table. Add `--plot` to save `mach_history.png`.

### `stellar2d.py viewer`

```
./stellar2d.py viewer [--port PORT]
```

Starts the React + Three.js web viewer (Vite dev server). Supports 2D heatmap, radial profile, and 3D star views. Load VTK files via drag-and-drop or by entering a run directory path.

### `stellar2d.py test`

```
./stellar2d.py test [--gpu] [--slow]
```

Runs the full test suite: C++ unit/exact/pitfall tests + Python integration tests.

| Flag | Description |
|---|---|
| `--gpu` | Also run GPU-specific tests (`test_lowmach`, `test_mini`) |
| `--slow` | Include slow convergence tests |

### `stellar2d.py clean`

Removes `build/` and `build-cpu/` directories.

## Typical Workflows

**Hydrostatic stability check:**
```bash
./stellar2d.py build
./stellar2d.py run --test lane_emden --nr 64 --ntheta 32 --tend 5.0
./stellar2d.py mach runs/lane_emden_64x32_*/ --plot
```

**Bubble convection (GPU):**
```bash
./stellar2d.py build --gpu
./stellar2d.py run --test bubble --solver fas --nr 128 --ntheta 64 --tend 0.5
./stellar2d.py plot runs/bubble_128x64_*/ --gif
```

**Low-Mach stellar oscillation (GPU):**
```bash
./stellar2d.py build --gpu
./stellar2d.py run --test lane_emden_perturbed --solver lowmach --nr 64 --ntheta 32 --tend 5.0
./stellar2d.py animate runs/lane_emden_perturbed_64x32_*/
```

## Code Structure

```
stellar2d/
├── stellar2d.py                Unified CLI entry point
├── CMakeLists.txt              Build system (CPU / GPU / AmgX options)
├── src/
│   ├── main.cpp                CLI, init, solver dispatch, time-stepping loop
│   ├── grid.h/cpp              Mesh: log-radial + uniform polar, precomputed geometry
│   ├── state.h/cpp             Conservative <-> primitive variable conversion
│   ├── eos.h                   Ideal gas equation of state
│   ├── hydro/
│   │   ├── reconstruct.h/cpp   MUSCL reconstruction (minmod / van Leer limiters)
│   │   ├── riemann.h/cpp       HLLC Riemann solver (CPU reference)
│   │   ├── flux.h/cpp          Flux divergence + geometric source + gravity source
│   │   └── integrate.h/cpp     RK2 time integration + CFL condition
│   ├── gravity/
│   │   ├── poisson.h/cpp       Poisson matrix assembly (CSR, 5-point spherical stencil)
│   │   ├── gmg.h/cpp           CPU geometric multigrid (V-cycle, red-black Gauss-Seidel)
│   │   └── amgx_solver.h/cpp   AmgX wrapper (USE_AMGX) / CPU Jacobi fallback
│   ├── bc/boundary.h/cpp       BCs: reflecting centre, axis symmetry, outflow
│   ├── io/output.h/cpp         VTK output + conservation diagnostics
│   ├── init/
│   │   ├── lane_emden.h/cpp    Lane-Emden polytropic equilibrium + bubble perturbation
│   │   ├── sedov.h/cpp         Sedov point-source explosion
│   │   ├── jeans.h/cpp         Jeans gravitational instability
│   │   └── evrard.h/cpp        Evrard cold gas sphere collapse
│   └── gpu/
│       ├── strang_solver.cu/cuh    Strang Cartesian solver (splits, sweeps, BCs, VTK)
│       ├── strang_device.cuh       Device functions: MC limiter, LM-HLLC, HSE helpers
│       ├── fas_solver.cu/cuh       FAS nonlinear multigrid (V-cycle, time stepping)
│       ├── fas_residual.cu         FAS/explicit residual + HLLC + ghost cell kernels
│       ├── fas_hllc.cuh            HLLC + minmod MUSCL for polar grid (device)
│       ├── fas_common.cuh          Index helpers, FasLevel struct
│       ├── fas_smoothers.cu        Block-Jacobi + SIMPLE smoothers for FAS
│       ├── fas_multigrid.cu        Restrict / prolongate / V-cycle
│       ├── fas_linalg.cuh          GPU linear algebra (reductions, preconditioner kernels)
│       ├── simple_solver.cu/cuh    SIMPLE pressure-correction solver
│       ├── projection_solver.cu/cuh  Pressure projection solver
│       ├── lowmach_solver.cu/h     JFNK low-Mach implicit solver (main driver)
│       ├── lm_residual.cu          Low-Mach well-balanced residual kernel
│       ├── lm_precond.cu           Preconditioners: block-Jacobi, line-Jacobi, PBP
│       ├── lm_krylov.cu            FGMRES + JFNK matvec + Newton loop
│       ├── lm_common.cuh           Shared device helpers for low-Mach solver
│       ├── gmg_gpu.cu/cuh          GPU geometric multigrid Poisson / Helmholtz solver
│       └── gpu_solver.cu/h         AmgX-backed compressible solver (requires USE_AMGX)
├── config/
│   ├── amgx.json               AmgX Poisson solver config
│   └── amgx_precond.json       AmgX block-AMG preconditioner config
├── frontend/                   React + Three.js web viewer
│   └── src/
│       ├── App.tsx             Main app: file/server loading, view switching
│       ├── Heatmap2D.tsx       2D pcolormesh heatmap
│       ├── RadialProfile.tsx   Radial line-plot viewer
│       ├── Star3D.tsx          3D surface-of-revolution viewer
│       └── lib/vtk-parser.ts   VTK file parser
├── scripts/
│   ├── plot_frames.py          3-panel frame renderer
│   ├── animate_evolution.py    Publication GIF generator
│   └── check_mach.py           Mach number analysis
├── tests/
│   ├── test_strang_init.cu         Strang: HSE balance, bubble init, mass, VTK (8 checks)
│   ├── test_strang_muscl.cu        MC limiter, WB y-sweep reconstruction (11 checks)
│   ├── test_strang_hllc.cu         LM-HLLC: uniform/Sod/LM/antisymmetry/HSE (10 checks)
│   ├── test_strang_step.cu         Strang integration: HSE, mass, buoyancy, CFL (9 checks)
│   ├── test_strang_unit.cu         EOS, ghost cells, HLLC edge cases (12 checks)
│   ├── test_strang_convergence.cu  Entropy wave 2nd-order convergence (3 checks)
│   ├── test_fas_verify.cu          Polar: minmod, recon, HLLC, ghost, HSE, convergence (19 checks)
│   ├── test_coverage_critical.cu   LM origin, GMG, FAS restrict/floor/CFL, solvers, init (16 checks)
│   ├── test_unit.cpp               CPU: grid geometry, EOS, minmod, HLLC (unit tests)
│   ├── test_exact.cpp              CPU: Lane-Emden analytic comparison
│   ├── test_pitfalls.cpp           CPU: regression tests for historical bugs (P01-P05)
│   └── (Python)                    pytest: conservation, symmetry, convergence, HSE
└── docs/
    ├── equations.md            Equation reference (authoritative)
    ├── pitfalls.md             Known pitfalls and fixes
    └── provenance.md           Output provenance conventions
```

## Building (Manual)

If you prefer not to use `stellar2d.py`:

Requirements: C++17 compiler, CMake >= 3.18. GPU builds need CUDA Toolkit. AmgX is optional.

```bash
# CPU-only (only supports hydro/ path — no GPU solvers, no FAS, no low-Mach)
mkdir build-cpu && cd build-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# GPU (all solvers: FAS, LowMach, SIMPLE, projection)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON
make -j$(nproc)

# GPU + AmgX (enables compressible JFNK solver)
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON -DAMGX_DIR=/path/to/amgx
make -j$(nproc)
```

## Testing

```bash
# All fast tests (C++ + Python)
./stellar2d.py test

# Include GPU tests
./stellar2d.py test --gpu

# Include slow convergence tests
./stellar2d.py test --slow

# Or manually:
cd build-cpu && ctest              # C++ tests via CTest
pytest                             # Python integration tests
pytest -m slow                     # Convergence tests
pytest --update-baselines          # Regenerate regression baselines

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

## License

See [LICENSE](LICENSE).
