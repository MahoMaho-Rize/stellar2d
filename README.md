# stellar2d

A GPU-accelerated 2D compressible Euler + self-gravity solver for stellar
convection and related astrophysical flows. Multiple solvers share the same
CLI entry point; each has its own intended use domain.

## Coordinate systems

| Coordinate | Grid | Solver families |
|---|---|---|
| Axisymmetric spherical $(r, \theta)$ | Log-radial × uniform polar | `explicit`, `fas`, `fas2`, `simple`, `projection`, `lowmach`, `radial1d`, `ale2d`, `wb2d`, `compressible` |
| 2D Cartesian $(x, y)$ | Uniform | `strang`, `cart_lag`, `cart_ale`, `cart_ale2`, `athena_vl2` |
| 2D Cartesian $(x, y)$ periodic spectral | `pseudo_spectral`, `sph2d_spectral`, `anelastic_sl` |

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
See [docs/projects/spectral_liouville/equations.md](docs/projects/spectral_liouville/equations.md) for the authoritative math and
[CLAUDE.md](CLAUDE.md) for the asset-preservation policy.

| Solver | Grid | Status | Use for |
|---|---|---|---|
| `explicit` | polar | ✅ stable | RK2 HLLC + MUSCL, baseline for polar tests |
| `fas` / `fas2` | polar | ✅ stable | Nonlinear multigrid, implicit polar time stepping |
| `simple` | polar | ✅ stable | SIMPLE pressure correction (low-Mach-ish) |
| `projection` | polar | ✅ stable | Fractional-step pressure projection |
| `lowmach` | polar | ✅ stable | JFNK fully implicit, stellar low-Mach regime |
| `radial1d` | 1D radial | ✅ stable | 1D Lagrangian implicit hydro; testbed for 2D ALE Newton-Krylov. **Not a stellar evolution code** (cannot do KH contraction, see [CLAUDE.md](CLAUDE.md) radial1d scope) |
| `compressible` | polar | ⚠️ requires AmgX | Legacy HLLC + AmgX Poisson |
| `wb2d` | polar | ⚠️ perturbation dies ~t=2 | Well-balanced Euler family, kept as reference |
| `ale2d` | polar | ⚠️ axisymmetric hoop bug | Axisymmetric Caramana; see `docs/design/ale_hoop_stress_fix.md` |
| `cart_lag` | Cartesian | ⚠️ HSE dt degenerates | Pure Lagrangian reference (hourglass mode over long HSE) |
| `cart_ale` | Cartesian | ✅ stable | Cartesian ALE baseline (reflective walls only) |
| `cart_ale2` | Cartesian | ✅ stable | **cart_ale + periodic BC + PPM + char projection.** Intended for compressible stellar convection. linwave convergence ~0.9-order (Lagrangian rebuild diffuses smooth modes). Gresho/Yee need `--shear-aware-av --rebuild-order 1`. See [docs/design/cart_ale2_design.md](docs/design/cart_ale2_design.md) |
| `athena_vl2` | Cartesian | ✅ stable | GPU port of Athena++ vl2 predictor-corrector (PLM + HLLC). **linwave 2.03-order** (cleanest convergence in the repo). Hard-coded x-periodic BC → not suitable for shock tubes. First choice for smooth-flow convergence studies |
| `pseudo_spectral` | 2D periodic Cartesian | ⚠️ Taylor-Green test currently FAILS (pre-existing regression) | 2D incompressible Navier-Stokes: cuFFT + IFRK3 + 2/3 dealias. Target for KH turbulence + forced turbulence |
| `anelastic_sl` | plane-parallel spectral | ✅ stable | Anelastic spectral-Liouville for stratified subsonic convection |
| `sph2d_spectral` | thin spherical shell | ✅ stable | 2D spherical spectral (Rossby waves, shallow shell turbulence) |
| `strang` | Cartesian | ✅ stable | Full 2D explicit Strang split HLLC+MUSCL; standard baseline |

### Solver selection by use case (2026-05-07 quantitative guidance)

Based on the Phase 2 `tst/` convergence data (linwave, Sod, entropy wave):

| Use case | Recommended | Alternate | Avoid |
|---|---|---|---|
| Stellar convection / long-time stratified HSE | `cart_ale2` | `cart_ale` (reflect only) | `athena_vl2` (y-reflect not robust) |
| Andrassy 2022 O-shell benchmark | `cart_ale2` | `athena_vl2` (for CPU↔GPU cross-validation) | `cart_lag` (hourglass mode) |
| 2D incompressible turbulence, KH spectrum | `pseudo_spectral` *(after regression fix)* | — | `cart_ale2` (ν_eff too high, k^{-10} spectrum) |
| Sod / shock tube 1D verification | `cart_ale2` w/ `--bc-x reflect` | `strang` | `athena_vl2` (x-periodic hard-coded) |
| Linear wave convergence study | `athena_vl2` (2.03-order verified) | `strang` | `cart_ale2` (~0.9-order on smooth flow) |
| Smooth scalar advection | `athena_vl2` | `pseudo_spectral` | `cart_ale2` (rebuild amplifies v modes) |
| Gresho stationary vortex | `cart_ale2` + `--shear-aware-av --rebuild-order 1` | — | `cart_ale2` defaults (vφ→0.2 in 1 s) |
| Yee isentropic vortex | `cart_ale2` + `--ppm` at N≥256, manual scan only | — | `cart_ale2` defaults (blows up at t≳5) |
| Pulsation / acoustic response in stellar atmosphere | `lowmach` (JFNK) | `radial1d` (1D cap) | — |
| pre-MS KH contraction / any τ_KH-scale evolution | Use MESA/KEPLER/GENEC output as IC | — | `radial1d` (architecturally cannot do this) |

## Test cases

Selected via `--test <name>`. Not every test case runs on every solver.

| `--test` | Description | Common solvers |
|---|---|---|
| `lane_emden` | Polytropic star ($n = 1.5$), equilibrium | `explicit`, `fas`, `lowmach`, `radial1d` |
| `lane_emden_perturbed` | Lane-Emden + 0.1% density perturbation (oscillation modes) | `explicit`, `fas`, `lowmach` |
| `bubble` | Entropy bubble in Lane-Emden star (buoyant convection) | `fas`, `simple`, `lowmach` |
| `sedov` | Point-source blast, uniform ambient | `explicit`, `fas` |
| `sedov2d` | 2D cylindrical Sedov blast (Cartesian) | `cart_ale2` |
| `noh` | 2D Noh implosion (strong AV test) | `cart_ale2` |
| `gresho` | Stationary vortex, AV trigger test | `cart_ale2` |
| `yee_vortex` | Yee-Vinokur-Djomehri isentropic vortex round-trip | `cart_ale2` |
| `jeans` | Uniform medium + perturbation, gravitational collapse | `explicit`, `fas` |
| `evrard` | Cold isothermal gas sphere, gravitational collapse | `explicit`, `fas` |
| `hse` | 2D Cartesian hydrostatic polytrope | `cart_lag`, `cart_ale`, `cart_ale2` |
| `hse_perturbed` | HSE + density perturbation | `cart_ale`, `cart_ale2` |
| `hse_bubble` | HSE + Gaussian bubble overlay(s) | `cart_ale`, `cart_ale2` |
| `sod` | 1D Sod tube in 2D Cartesian | `cart_ale2`, `athena_vl2` (note: vl2 x-periodic pollutes Sod) |
| `kh_shear` | Classic dual shear band KH (density contrast) | `cart_ale`, `cart_ale2` |
| `kh_lecoanet` | Athena iprob=4 canonical KH (dual tanh layers, periodic) | `cart_ale2` only |
| `shear_mode` | Linear shear-mode decay (T3 ν_eff characterization) | `cart_ale2`, `athena_vl2` |
| `entropy_wave` | Smooth entropy wave advection (T1 convergence) | `cart_ale2`, `athena_vl2` |
| `acoustic_wave` | Right-running acoustic eigenmode (linwave convergence) | `cart_ale2`, `athena_vl2` |
| `andrassy2022` | Idealized O-shell convection benchmark | `cart_ale2`, `athena_vl2` |
| `local_convection` | MESA-envelope slab stratified convection | `cart_ale2` |

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

### Test-case IC parameters (2026-05-07 `tst/` framework)

| Flag | Applies to | Notes |
|---|---|---|
| `--compute-error` | any benchmark with a C++ `compute_*_error` | Solver writes `<testname>-errors.dat` at t_end |
| `--ewave-{rho0,P0,u0,A,k,periods}` | `entropy_wave` | δρ = A·ρ₀·sin(kx), advected at u₀ |
| `--awave-{rho0,P0,A,k,periods}` | `acoustic_wave` | Right-acoustic eigenmode, c₀ = √(γP₀/ρ₀) |
| `--shear-{V0,k,rho,P}` | `shear_mode` | Linear shear-mode decay for ν_eff probe |
| `--rebuild-order <0\|1>` | cart_ale2 | Node-velocity rebuild order (see cart_ale2 notes) |
| `--shear-aware-av` | cart_ale2 | Reduce Q in shear-dominated cells (needed for Gresho) |

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
python scripts/render/render_kh.py runs/kh_lecoanet_256x512_*
python scripts/pseudo_spectral/spectrum_kh.py runs/kh_lecoanet_256x512_*
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

Three-layer test organisation (2026-05-07 refactor, see
[docs/design/testing_infrastructure_plan_2026-05-07.md](docs/design/testing_infrastructure_plan_2026-05-07.md)):

1. **`tests/test_*.cu`** — in-process CUDA unit / regression locks
   (Strang, FAS, lowmach, cart_ale2 Phase 1 regression guards).
2. **`tst/test_*/test_*_gpu.py`** — end-to-end pytest suite. Each test
   launches `build/stellar2d --compute-error`, which writes a schema-
   documented `<testname>-errors.dat`; pytest reads it and asserts L1 /
   convergence-ratio thresholds. **No analytic solvers in Python.**
3. **`tests/test_*.py`** — legacy Python tests (CPU polar hydro).

### Running

```bash
cd build

# Fast CI bucket (< 20 s) — Phase 1 CUDA locks + pytest_gpu_fast suite
ctest -L fast

# Everything (includes slow and scan-marker pytest tests if any)
ctest -j$(nproc)

# Just the pytest end-to-end suite
cd ../tst && STELLAR2D_BIN=../build/stellar2d pytest -m fast -v
```

### Current coverage (2026-05-07)

**`ctest -L fast`**: 23 tests, 20 passing in ~16 s.
3 pre-existing failures unrelated to the Phase 2 migration work:

| Failing | Nature |
|---|---|
| `lowmach_s_e_regression` | S_E = ρv·g leak ratio = 20 (expected ≲ 1e-6) |
| `fas_verify` | FAS HSE residual verify below tolerance |
| `pseudo_spectral_taylor_green` | Taylor-Green analytic decay regression |

**`tst/` pytest suite** (16 tests, all passing in ~16 s):

| benchmark | cart_ale2 | athena_vl2 |
|---|---|---|
| entropy_wave | ✅ 2 tests | ✅ 2 tests |
| sod (Toro Riemann in `src/gpu/common/sod_exact.h`) | ✅ 2 tests | ✅ 2 tests |
| gresho | ✅ 2 tests (stationary-vortex L1) | — |
| yee_vortex | ✅ 2 tests (short-t smoke) | — |
| acoustic_wave (linwave) | ✅ 2 tests | ✅ 2 tests **(2.03-order verified)** |

**`tests/` CUDA unit tests**: Strang (6), FAS (2), lowmach family (7),
Helm/Dual (5), cart_ale2 Phase 1 regression (4: `hse_stratified_reflect`,
`phase_m_compensation`, `symmetry`, `athena_vl2_sod`).

See [docs/tests_ale2/README.md](docs/tests_ale2/README.md) for the
standard verification runbook (Sod / Sedov / Noh / Gresho / Yee).

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

- [docs/projects/spectral_liouville/equations.md](docs/projects/spectral_liouville/equations.md) — Authoritative equation reference (§1–§17)
- [docs/pitfalls.md](docs/pitfalls.md) — Bug log with root cause (P01–P31)
- [docs/design/cart_ale2_design.md](docs/design/cart_ale2_design.md) — cart_ale2 design + KH benchmark results + scope policy
- [docs/design/ale_design.md](docs/design/ale_design.md) — Axisymmetric ale2d design notes
- [docs/design/ale_hoop_stress_fix.md](docs/design/ale_hoop_stress_fix.md) — ale2d hoop-stress patch plan
- [docs/design/ale_rezone_design.md](docs/design/ale_rezone_design.md) — Original ALE rezone strategy (cart_lag → cart_ale)
- [docs/cart_ale_progress_2026-05-01.md](docs/cart_ale_progress_2026-05-01.md) — cart_ale day-1 journal
- [docs/cart_ale_2nd_order_remap_2026-05-02.md](docs/cart_ale_2nd_order_remap_2026-05-02.md) — 2nd-order remap progress
- [docs/design/lag2d_design.md](docs/design/lag2d_design.md) — Lag2d legacy notes
- [docs/design/wb2d_design.md](docs/design/wb2d_design.md) — wb2d design notes
- [docs/provenance.md](docs/provenance.md) — Figure/data filename and footer convention
- [docs/design/testing_infrastructure_plan_2026-05-07.md](docs/design/testing_infrastructure_plan_2026-05-07.md) — Test architecture (Phase 1/2 done, Phase 3/4 backlog)
- [docs/design/testing_scheme_characterization_2026-05-07.md](docs/design/testing_scheme_characterization_2026-05-07.md) — ν_eff, convergence order, and scheme-type classification across all hydro solvers
- [docs/sessions/session_journal_2026-05-07_phase2_testing.md](docs/sessions/session_journal_2026-05-07_phase2_testing.md) — Phase 1+2 implementation journal + discovered solver issues + solver selection guide
- [tst/README.md](tst/README.md) — `tst/` pytest framework + compute_error pattern
- [CLAUDE.md](CLAUDE.md) — Solver asset-preservation rules (read first before modifying existing solvers)

## Equation-code traceability

Every numbered equation in `docs/projects/spectral_liouville/equations.md` is the authoritative
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
