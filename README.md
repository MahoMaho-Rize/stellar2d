# stellar2d

A GPU-accelerated 2D compressible Euler + self-gravity solver for stellar
convection and related astrophysical flows. Multiple solvers share the same
CLI entry point; each has its own intended use domain.

Quick doc links:

- Overall manual: [docs/manual.md](docs/manual.md)
- Local MESA opacity workflow: [docs/mesa_opacity_workflow.md](docs/mesa_opacity_workflow.md)

## Coordinate systems

| Coordinate | Grid | Solver families |
|---|---|---|
| Axisymmetric spherical $(r, \theta)$ | Log-radial × uniform polar | `explicit`, `fas`, `simple`, `projection`, `lowmach`, `radial1d`, `ale2d`, `wb2d`, `compressible` |
| 2D Cartesian $(x, y)$ | Uniform | `strang` (tests-only), `cart_lag`, `cart_ale`, `cart_ale2` |

## Build

Requirements: C++17, CMake ≥ 3.18. GPU solvers need CUDA Toolkit 12+.
AmgX is optional (required only for the `compressible` JFNK solver).

```bash
# CPU-only (explicit polar hydro + tests, no GPU solvers)
mkdir build-cpu && cd build-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# GPU (default: all active GPU solvers)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON
make -j$(nproc)

# GPU + AmgX (enables the legacy `compressible` JFNK solver)
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON -DAMGX_DIR=/path/to/amgx
make -j$(nproc)
```

The wrapper script `stellar2d.py` provides convenience subcommands
(`build`, `run`, `plot`, `animate`, `mach`, `test`, `clean`). It does
not expose newer flags (`--ppm`, `--bc-x`, ...) — for full control use
`./build/stellar2d ...` directly.

## Solvers

Selected via `--solver <name>`. Table lists intended use and key references.
See [docs/equations.md](docs/equations.md) for the authoritative math and
[CLAUDE.md](CLAUDE.md) for the asset-preservation policy.

| Solver | Grid | Status | Use for |
|---|---|---|---|
| `explicit` | polar | ✅ stable | RK2 HLLC + MUSCL, baseline for polar tests |
| `fas` | polar | ✅ stable | Nonlinear multigrid, implicit polar time stepping |
| `simple` | polar | ✅ stable | SIMPLE pressure correction (low-Mach-ish) |
| `projection` | polar | ✅ stable | Fractional-step pressure projection |
| `lowmach` | polar | ✅ stable | JFNK fully implicit, stellar low-Mach regime |
| `radial1d` | 1D radial | ✅ stable | MESA-style 1D, baseline for ALE radial regression |
| `compressible` | polar | ⚠️ requires AmgX | Legacy HLLC + AmgX Poisson |
| `wb2d` | polar | ⚠️ perturbation dies ~t=2 | Well-balanced Euler family, kept as reference |
| `ale2d` | polar | ⚠️ axisymmetric hoop bug | Axisymmetric Caramana; see `docs/ale_hoop_stress_fix.md` |
| `cart_lag` | Cartesian | ⚠️ HSE dt degenerates | Pure Lagrangian reference (hourglass mode over long HSE) |
| `cart_ale` | Cartesian | ✅ stable | Cartesian ALE baseline (reflective walls only) |
| `cart_ale2` | Cartesian | ✅ stable | **cart_ale + periodic BC + PPM + char projection.** Intended for compressible stellar convection. See [docs/cart_ale2_design.md](docs/cart_ale2_design.md). |

`strang` exists in `src/gpu/strang_solver.*` and is exercised by the
Strang test suite; it is not currently wired into `main.cpp` dispatch.

## Test cases

Selected via `--test <name>`. Not every test case runs on every solver.

| `--test` | Description | Common solvers |
|---|---|---|
| `lane_emden` | Polytropic star ($n = 1.5$), equilibrium | `explicit`, `fas`, `lowmach`, `radial1d` |
| `lane_emden_perturbed` | Lane-Emden + 0.1% density perturbation (oscillation modes) | `explicit`, `fas`, `lowmach` |
| `bubble` | Entropy bubble in Lane-Emden star (buoyant convection) | `fas`, `simple`, `lowmach` |
| `sedov` | Point-source blast, uniform ambient | `explicit`, `fas` |
| `jeans` | Uniform medium + perturbation, gravitational collapse | `explicit`, `fas` |
| `evrard` | Cold isothermal gas sphere, gravitational collapse | `explicit`, `fas` |
| `hse` | 2D Cartesian hydrostatic polytrope | `cart_lag`, `cart_ale`, `cart_ale2` |
| `hse_perturbed` | HSE + density perturbation | `cart_ale`, `cart_ale2` |
| `hse_bubble` | HSE + Gaussian bubble overlay(s) | `cart_ale`, `cart_ale2` |
| `sod` | 1D Sod tube in 2D Cartesian | `cart_ale`, `cart_ale2` |
| `kh_shear` | Classic dual shear band KH (density contrast) | `cart_ale`, `cart_ale2` |
| `kh_lecoanet` | Athena iprob=4 canonical KH (dual tanh layers, periodic) | `cart_ale2` only |

## Core CLI flags

All dispatch flags; some are solver-specific (ignored when not applicable).

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--test <name>` | string | `lane_emden` | See test case table |
| `--solver <name>` | string | `compressible` | See solver table |
| `--nr <int>` | int | `128` | x (or radial) resolution |
| `--ntheta <int>` | int | `64` | y (or polar) resolution |
| `--tend <float>` | float | `1.0` | End time |
| `--cfl <float>` | float | `0.4` | CFL number |
| `--gamma <float>` | float | varies | Adiabatic index |
| `--output-interval <int>` | int | `100` | step interval for VTK (legacy) |
| `--vtk-interval <int>` | int | (follows output-interval) | cart_ale{,2}: step-indexed VTK |
| `--vtk-dt <float>` | float | off | cart_ale{,2}: write VTK at physical time intervals |
| `--diag-interval <int>` | int | (follows output-interval) | cart_ale{,2}: diagnostics + CSV cadence |
| `--mesh <name>` | `log` / `equimass` | `log` | Polar grid only |
| `--limiter <name>` | `minmod` / `vanleer` | `minmod` | HLLC MUSCL limiter |
| `--perturb <float>` | float | `1e-3` | Perturbation amplitude for IC |
| `--radial-only` | flag | off | Enforce $v_\theta = 0$ (explicit/FAS) |
| `--r-inner <float>` | float | auto | Override inner radius |

### `cart_lag` / `cart_ale` / `cart_ale2` specific

| Flag | Default | Notes |
|---|---|---|
| `--remap-order <1\|2>` | 2 | cart_ale{,2}: donor-cell or MUSCL |
| `--remap-limiter <name>` | `vanleer` | `minmod` / `vanleer` / `mc` |
| `--cq-lin <float>` | 0.5 | Caramana AV linear coefficient |
| `--cq-quad <float>` | 2.0 | Caramana AV quadratic coefficient |
| `--shear-aware-av` | off | Reduce $Q$ in shear-dominated cells |
| `--bubble` | off | IC: `hse_bubble` with CLI bubble params |
| `--bubble-mode <name>` | `pressure` | `pressure` / `entropy` |
| `--bubble-xc/yc/rb/alpha/beta <float>` | — | Bubble position / radius / ρ-amp / P-amp |

### `cart_ale2` additional

| Flag | Default | Notes |
|---|---|---|
| `--bc-x <name>` | `reflect` | `reflect` / `periodic` |
| `--bc-y <name>` | `reflect` | `reflect` / `periodic` |
| `--ppm` | off | Use PPM instead of MUSCL for 2nd-order remap |
| `--ppm-limiter <name>` | `cs` | `cs` (Colella-Sekora) / `cw` (Colella-Woodward) |
| `--ppm-space <name>` | `prim` | `prim` (primitive) / `cons` (conservative) |
| `--ppm-char` / `--no-ppm-char` | on | Characteristic-variable projection |
| `--kh-k <int>` | 0 | KH mode number override (0 = IC default) |
| `--frame-buffer` | off | VRAM-buffered frame capture (high-rate I/O) |
| `--frame-headroom-mb <int>` | 1024 | VRAM to leave free when sizing buffer |

### `lowmach` specific

| Flag | Values | Default |
|---|---|---|
| `--precond <name>` | `block_jacobi`, `line_jacobi`, `simple`, `pbp`, `none`, `block_schur`, `combined` | `line_jacobi` |
| `--lm-hllc` | flag | off (enable acoustic blending) |
| `--no-sponge` | flag | off (disable velocity sponge layer) |

Run `./build/stellar2d --help` equivalent: there is none. Read
`src/main.cpp` lines 160–250 for the full argument parser.

## Quick-start recipes

### HSE regression (cart_ale2)
```bash
./build/stellar2d --solver cart_ale2 --test hse \
                  --nr 64 --ntheta 64 --remap-order 2 --ppm --tend 0.5
```
Expect `E` conserved to ~10 digits.

### Canonical KH (Lecoanet / Athena iprob=4)
```bash
./build/stellar2d --solver cart_ale2 --test kh_lecoanet \
                  --nr 256 --ntheta 512 --cfl 0.3 \
                  --bc-x periodic --bc-y periodic \
                  --remap-order 2 --ppm \
                  --tend 5 --vtk-dt 0.025 --frame-buffer
python scripts/render_kh.py runs/kh_lecoanet_256x512_*
python scripts/spectrum_kh.py runs/kh_lecoanet_256x512_*
```

### Stellar oscillation (low-Mach JFNK)
```bash
./build/stellar2d --solver lowmach --test lane_emden_perturbed \
                  --nr 64 --ntheta 32 --tend 5.0
```

### Buoyant bubble (FAS multigrid)
```bash
./build/stellar2d --solver fas --test bubble --nr 128 --ntheta 64 --tend 0.5
```

## Visualization scripts

Each live in `scripts/`. Most take a run directory as the first argument
and write an MP4/PNG next to the VTK files.

| Script | For | Notes |
|---|---|---|
| `render_cart_ale.py` | cart_ale{,2} 3-panel (ρ, entropy δs/s₀, Mach) | Generic cart output |
| `render_kh.py` | cart_ale{,2} KH (ω, \|v\|) | Vorticity from velocity VECTORS, streamlines coming |
| `spectrum_kh.py` | cart_ale{,2} KE spectrum E(k) | Kraichnan reference lines |
| `render_bubble.py` | Polar bubble evolution | 3D-like mirrored view |
| `render_cartesian.py` | Polar → Cartesian interpolation | Publication plots |
| `render_video_fast.py` | Polar density+entropy+mach → MP4 | Default polar renderer |

## Tests

C++ unit + GPU integration tests via CTest:
```bash
cd build && ctest -j$(nproc)
```

Python integration tests (slow tests gated):
```bash
pytest                 # fast tests
pytest -m slow         # convergence tests
```

The `stellar2d.py test` wrapper also runs a curated subset:
```bash
./stellar2d.py test         # C++ unit + fast Python
./stellar2d.py test --gpu   # also GPU-specific
./stellar2d.py test --slow  # include slow convergence
```

Current GPU test status (`test_strang_*`, `test_fas_*`, `test_coverage_critical`):
**88 checks across 8 suites, all passing** at the Strang/FAS/lowmach layer.
The cart_ale{,2} stack does not yet have a dedicated ctest target —
uniform-advection periodic conservation check (P30/P31 regression)
should be added next.

## Code structure

```
stellar2d/
├── CMakeLists.txt               Build system (CPU / GPU / AmgX options)
├── stellar2d.py                 Optional CLI wrapper (build/run/plot/test/clean)
├── src/
│   ├── main.cpp                 CLI parser + solver dispatch + time-stepping loop
│   ├── grid.{h,cpp}             Log-radial × polar mesh, precomputed geometry
│   ├── state.{h,cpp}            Conservative ↔ primitive conversion
│   ├── eos.h                    Ideal-gas EOS (Eq. 1.2–1.3)
│   ├── hydro/                   CPU polar hydrodynamics
│   │   ├── reconstruct.{h,cpp}  MUSCL (minmod / van Leer)
│   │   ├── riemann.{h,cpp}      HLLC (CPU reference)
│   │   ├── flux.{h,cpp}         Flux divergence + geometric + gravity source
│   │   └── integrate.{h,cpp}    RK2 + CFL
│   ├── gravity/
│   │   ├── poisson.{h,cpp}      CSR Poisson matrix (5-point spherical stencil)
│   │   ├── gmg.{h,cpp}          CPU geometric multigrid (V-cycle, red-black GS)
│   │   └── amgx_solver.{h,cpp}  AmgX wrapper (USE_AMGX) or Jacobi fallback
│   ├── bc/boundary.{h,cpp}      Reflective centre + axis + outflow BCs
│   ├── io/output.{h,cpp}        VTK write + conservation diagnostics
│   ├── init/                    Initial conditions
│   │   ├── lane_emden.{h,cpp}   Polytrope equilibrium + bubble perturbation
│   │   ├── sedov.{h,cpp}        Point-source blast
│   │   ├── jeans.{h,cpp}        Jeans gravitational instability
│   │   └── evrard.{h,cpp}       Evrard cold sphere collapse
│   └── gpu/                     GPU solvers (USE_GPU=ON)
│       ├── strang_solver.{cu,cuh}     Strang Cartesian (tests-only, not dispatched)
│       ├── strang_device.cuh          Device helpers: MC limiter, LM-HLLC, HSE
│       ├── fas_solver.{cu,cuh}        FAS nonlinear multigrid driver
│       ├── fas_residual.cu            FAS residual + HLLC + ghosts
│       ├── fas_smoothers.cu           Block-Jacobi + SIMPLE smoothers
│       ├── fas_multigrid.cu           Restrict / prolongate / V-cycle
│       ├── fas_{hllc,common,linalg}.cuh   Shared FAS device utilities
│       ├── simple_solver.{cu,cuh}     SIMPLE pressure correction
│       ├── projection_solver.{cu,cuh} Fractional-step pressure projection
│       ├── lowmach_solver.{cu,h}      JFNK low-Mach driver
│       ├── lm_residual.cu             Well-balanced low-Mach residual
│       ├── lm_precond.cu              Block-Jacobi / line-Jacobi / PBP preconditioners
│       ├── lm_krylov.cu               FGMRES + JFNK matvec + Newton loop
│       ├── lm_common.cuh              Shared low-Mach device helpers
│       ├── gmg_gpu.{cu,cuh}           GPU Poisson/Helmholtz multigrid
│       ├── radial1d_solver.{cu,cuh}   1D radial MESA-style solver
│       ├── radial1d_kernels.cuh       Radial kernels
│       ├── wb2d_solver.{cu,cuh}       Well-balanced 2D polar (Eulerian)
│       ├── wb2d_kernels.cu            wb2d device kernels
│       ├── ale2d_solver.{cu,cuh}      Axisymmetric ALE (Caramana)
│       ├── ale2d_kernels.cu           ale2d device kernels
│       ├── cart_lag_solver.{cu,cuh}   Cartesian pure Lagrangian (reference baseline)
│       ├── cart_lag_kernels.cu        cart_lag device kernels
│       ├── cart_ale_solver.{cu,cuh}   Cartesian ALE (reflective walls, baseline)
│       ├── cart_ale_kernels.cu        cart_ale device kernels
│       ├── cart_ale2_solver.{cu,cuh}  Cartesian ALE + periodic BC + PPM (§15–§17)
│       ├── cart_ale2_kernels.cu       cart_ale2 device kernels
│       └── gpu_solver.{cu,h}          AmgX-backed compressible solver (USE_AMGX)
├── config/
│   ├── amgx.json                AmgX Poisson solver config
│   └── amgx_precond.json        AmgX block-AMG preconditioner config
├── scripts/
│   ├── plot_frames.py           Polar 3-panel frame renderer
│   ├── animate_evolution.py     Polar publication-quality GIF
│   ├── check_mach.py            Polar Mach number history
│   ├── render_video_fast.py     Polar ρ + entropy + Mach → MP4
│   ├── render_bubble.py         Polar bubble mirrored view
│   ├── render_cartesian.py      Polar → Cartesian interpolation
│   ├── render_cart_ale.py       cart_ale{,2} 3-panel (ρ, δs/s₀, Mach)
│   ├── render_kh.py             cart_ale{,2} KH (vorticity + \|v\|)
│   ├── spectrum_kh.py           cart_ale{,2} 2D kinetic-energy spectrum E(k)
│   ├── make_bubble_gif.py       Bubble GIF helper
│   └── bench_precond.sh         Preconditioner benchmark harness
├── tests/
│   ├── test_unit.cpp                      Grid geometry, EOS, minmod, HLLC (CPU)
│   ├── test_exact.cpp                     Lane-Emden analytic comparison
│   ├── test_pitfalls.cpp                  Regression tests for P01–P05
│   ├── test_strang_init.cu                HSE balance, bubble, mass, VTK (8 checks)
│   ├── test_strang_muscl.cu               MC limiter + WB y-sweep (11 checks)
│   ├── test_strang_hllc.cu                LM-HLLC variants (10 checks)
│   ├── test_strang_step.cu                HSE + mass + buoyancy + CFL (9 checks)
│   ├── test_strang_unit.cu                EOS, ghosts, HLLC edge cases (12 checks)
│   ├── test_strang_convergence.cu         Entropy-wave 2nd-order convergence (3 checks)
│   ├── test_fas_verify.cu                 Polar recon + HLLC + ghost + HSE (19 checks)
│   ├── test_fas_diagnose_hse.cu           FAS HSE diagnostic
│   ├── test_coverage_critical.cu          Critical-path coverage (16 checks)
│   ├── test_lowmach.cu                    Low-Mach JFNK smoke
│   ├── test_solver_diagnosis.cu           Low-Mach Newton/Krylov diagnosis
│   ├── test_precond_quality.cu            Preconditioner quality comparison
│   ├── test_newton_tuning.cu              JFNK Newton tolerance sweep
│   ├── test_eps_sweep.cu                  FD Jacobian epsilon sweep
│   ├── test_mini.cu                       Minimal GPU smoke
│   ├── test_{hse,conservation,convergence,symmetry,regression}.py  pytest integrations
│   ├── plot_lane_emden.py                 Verification plot generator
│   ├── provenance.py                      Filename / footer helpers (see docs/provenance.md)
│   └── conftest.py                        pytest fixtures
├── frontend/                              React + Three.js VTK viewer
│   └── src/
│       ├── main.tsx                       Vite entry
│       ├── App.tsx                        Top-level routing + file loading
│       ├── components/
│       │   ├── Heatmap2D.tsx              2D pcolormesh view
│       │   ├── RadialProfile.tsx          Radial line view
│       │   ├── Star3D.tsx                 3D surface-of-revolution view
│       │   └── Colorbar.tsx               Shared colorbar widget
│       ├── hooks/                         (reserved)
│       └── lib/
│           ├── vtk-parser.ts              Legacy VTK reader
│           ├── colormap.ts                Colormap utilities
│           └── types.ts                   Shared TypeScript types
└── docs/
    ├── equations.md                       §1–§17 authoritative equation reference
    ├── equations.pdf                      Rendered PDF
    ├── pitfalls.md                        P01–P31 bug log
    ├── cart_ale2_design.md                cart_ale2 design + KH benchmark + scope
    ├── cart_ale_progress_2026-05-01.md    cart_ale development journal
    ├── cart_ale_2nd_order_remap_2026-05-02.md  2nd-order remap progress
    ├── ale_design.md                      ale2d axisymmetric design
    ├── ale_hoop_stress_fix.md             ale2d hoop-stress patch plan
    ├── ale_rezone_design.md               ALE rezone design (cart_lag → cart_ale)
    ├── lag2d_design.md                    Lag2d legacy notes
    ├── wb2d_design.md                     wb2d design notes
    └── provenance.md                      Figure/data naming conventions
```

## Documentation

- [docs/equations.md](docs/equations.md) — Authoritative equation reference (§1–§17)
- [docs/pitfalls.md](docs/pitfalls.md) — Bug log with root cause (P01–P31)
- [docs/cart_ale2_design.md](docs/cart_ale2_design.md) — cart_ale2 design + KH benchmark results + scope policy
- [docs/ale_design.md](docs/ale_design.md) — Axisymmetric ale2d design notes
- [docs/ale_hoop_stress_fix.md](docs/ale_hoop_stress_fix.md) — ale2d hoop-stress patch plan
- [docs/ale_rezone_design.md](docs/ale_rezone_design.md) — Original ALE rezone strategy (cart_lag → cart_ale)
- [docs/cart_ale_progress_2026-05-01.md](docs/cart_ale_progress_2026-05-01.md) — cart_ale day-1 journal
- [docs/cart_ale_2nd_order_remap_2026-05-02.md](docs/cart_ale_2nd_order_remap_2026-05-02.md) — 2nd-order remap progress
- [docs/lag2d_design.md](docs/lag2d_design.md) — Lag2d legacy notes
- [docs/wb2d_design.md](docs/wb2d_design.md) — wb2d design notes
- [docs/provenance.md](docs/provenance.md) — Figure/data filename and footer convention
- [CLAUDE.md](CLAUDE.md) — Solver asset-preservation rules (read first before modifying existing solvers)

## Equation-code traceability

Every numbered equation in `docs/equations.md` is the authoritative
specification. Source comments of the form `// Eq. (X.Y)` point at the
corresponding equation. If code disagrees with the document, the code
has a bug.

Current annotation coverage: 117 references in CPU sources (§1–§9),
78 references across GPU sources (§12 FAS, §13 Strang, §14 LM-HLLC,
§15–§17 cart_lag / cart_ale / cart_ale2). Additional GPU solvers
(`simple_solver`, `projection_solver`, `lm_krylov`, `lm_precond`,
`ale2d`, `wb2d`, `radial1d`, `gmg_gpu`) are progressively being
annotated against their corresponding sections — missing annotations
are a backlog item, not a sign of divergence.

## License

See [LICENSE](LICENSE).
