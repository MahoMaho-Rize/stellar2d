# stellar2d

2D axisymmetric compressible Euler + self-gravity Poisson solver for stellar evolution simulations.

Solves the Euler equations in spherical $(r, \theta)$ coordinates coupled with a Poisson equation for self-gravity. Supports explicit RK2 (CPU), fully-implicit low-Mach Newton-GMRES (GPU), and FAS nonlinear multigrid (GPU) solvers.

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

For GPU builds (enables FAS and LowMach solvers):

```bash
./stellar2d.py build --gpu
./stellar2d.py run --test bubble --solver fas --nr 128 --ntheta 64
```

## Solvers

| Solver | `--solver` flag | Backend | Method | Requires |
|---|---|---|---|---|
| **Explicit** | `compressible` (default) | CPU | RK2 + HLLC + GMG Poisson | C++17 |
| **Compressible JFNK** | `compressible` | GPU | Backward Euler + FGMRES(60) + AmgX | `--gpu --amgx-dir` |
| **Low-Mach** | `lowmach` | GPU | Backward Euler + Newton-GMRES(120) | `--gpu` |
| **FAS Multigrid** | `fas` | GPU | Backward Euler + FAS V-cycle + GMRES(3) | `--gpu` |

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

| Flag | Default | Description |
|---|---|---|
| `--test` | `lane_emden` | Test case (see table above) |
| `--solver` | `compressible` | Solver backend |
| `--precond` | `line_jacobi` | Preconditioner for implicit solvers |
| `--mesh` | `log` | Mesh type: `log` (logarithmic radial) or `equimass` |
| `--nr` | `128` | Radial cells |
| `--ntheta` | `64` | Polar cells |
| `--tend` | `1.0` | End time |
| `--cfl` | `0.4` | CFL number |
| `--output-interval` | `100` | Steps between VTK snapshots |

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

## Physical Model

The code solves the compressible Euler equations in axisymmetric spherical coordinates $(r, \theta)$. All equations are documented in [docs/equations.md](docs/equations.md); every physics computation in source code is annotated with the corresponding equation number (`// Eq. (X.Y)`).

- Ideal gas EOS ($\gamma = 5/3$)
- MUSCL reconstruction + HLLC Riemann solver (second-order Godunov)
- Well-balanced geometric source terms (volume-consistent form; Eq. 5.3--5.4)
- Gravity: GPU geometric multigrid / AmgX AMG / CPU GMG V-cycle
- Time integration: RK2 explicit, or Backward Euler implicit (Newton-GMRES / FAS)

## Code Structure

```
stellar2d/
├── stellar2d.py                Unified CLI entry point
├── CMakeLists.txt              Build system (CPU / GPU / AmgX options)
├── src/
│   ├── main.cpp                Entry: CLI parsing, solver dispatch, time loop
│   ├── grid.h/cpp              Mesh: log-radial + equimass, precomputed geometry
│   ├── state.h/cpp             Conservative <-> primitive variable conversion
│   ├── eos.h                   Ideal gas equation of state
│   ├── hydro/
│   │   ├── reconstruct.h/cpp   MUSCL reconstruction (minmod / van Leer)
│   │   ├── riemann.h/cpp       HLLC Riemann solver
│   │   ├── flux.h/cpp          Flux divergence + geometric/gravity source terms
│   │   └── integrate.h/cpp     RK2 time integration + CFL
│   ├── gravity/
│   │   ├── poisson.h/cpp       Poisson matrix assembly (CSR, 5-point stencil)
│   │   ├── gmg.h/cpp           CPU geometric multigrid V-cycle
│   │   └── amgx_solver.h/cpp   AmgX GPU wrapper (optional)
│   ├── bc/boundary.h/cpp       BCs: reflecting centre, axis symmetry, outflow
│   ├── io/output.h/cpp         VTK output + diagnostics CSV
│   ├── init/
│   │   ├── lane_emden.h/cpp    Lane-Emden polytropic equilibrium + bubble
│   │   ├── sedov.h/cpp         Sedov blast wave
│   │   ├── jeans.h/cpp         Jeans gravitational instability
│   │   └── evrard.h/cpp        Evrard cold gas sphere
│   └── gpu/
│       ├── fas_solver.*        FAS nonlinear multigrid (V-cycle, GMRES smoother)
│       ├── lowmach_solver.*    Low-Mach Newton-GMRES (JFNK)
│       ├── gpu_solver.*        Compressible JFNK + AmgX
│       └── gmg_gpu.*           GPU geometric multigrid (Poisson / Helmholtz)
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
│   ├── test_conservation.py    Mass & energy conservation bounds
│   ├── test_hse.py             HSE stability checks
│   ├── test_symmetry.py        IC symmetry verification
│   ├── test_convergence.py     Grid convergence (slow)
│   ├── test_regression.py      Golden-value regression
│   ├── test_unit.cpp           C++ unit tests
│   ├── test_exact.cpp          Physics exact-solution tests
│   └── test_pitfalls.cpp       Pitfall regression tests
└── docs/
    ├── equations.md            Equation reference (authoritative)
    ├── pitfalls.md             Known pitfalls and fixes
    └── provenance.md           Output provenance conventions
```

## Building (Manual)

If you prefer not to use `stellar2d.py`:

```bash
# CPU-only
mkdir build-cpu && cd build-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# GPU (FAS + LowMach, no AmgX required)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON
make -j$(nproc)

# GPU + AmgX (enables compressible JFNK solver)
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON -DAMGX_DIR=/path/to/amgx
make -j$(nproc)
```

Requirements: C++17 compiler, CMake >= 3.18. GPU builds need CUDA Toolkit. AmgX is optional.

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
```

## License

See [LICENSE](LICENSE).
