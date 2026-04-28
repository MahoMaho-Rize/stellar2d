# stellar2d

A 2D axisymmetric compressible Euler + self-gravity Poisson solver for proof-of-concept stellar evolution simulations.

## Physical Model

The code solves the compressible Euler equations in axisymmetric spherical coordinates $(r, \theta)$, coupled with a Poisson equation for self-gravity. All discrete equations are documented in [docs/equations.md](docs/equations.md); every physics computation in the source code is annotated with the corresponding equation number.

- Pluggable EOS interface with `ideal` and `ideal_rad` implementations
- Default CPU EOS: ideal gas + radiation pressure
- MUSCL reconstruction + HLLC Riemann solver (second-order Godunov method)
- Well-balanced geometric source term discretisation (volume-consistent form; see Eq. 5.3–5.4)
- Poisson equation: NVIDIA AmgX GPU solver / CPU Jacobi fallback
- RK2 (Heun's method) explicit time integration

## Code Structure

```
src/
├── grid.h/cpp              Mesh: logarithmic radial + uniform polar, precomputed geometry
├── state.h/cpp             Conservative ↔ primitive variable conversion
├── eos.h                   EOS interface + ideal / ideal+radiation implementations
├── hydro/
│   ├── reconstruct.h/cpp   MUSCL reconstruction (minmod / van Leer limiters)
│   ├── riemann.h/cpp       HLLC Riemann solver
│   ├── flux.h/cpp          Flux divergence + geometric source terms + gravity source terms
│   └── integrate.h/cpp     RK2 time integration + CFL condition
├── gravity/
│   ├── poisson.h/cpp       Poisson matrix assembly (CSR, 5-point spherical stencil)
│   └── amgx_solver.h/cpp   AmgX wrapper (USE_AMGX) / CPU Jacobi fallback
├── bc/boundary.h/cpp       Boundary conditions: reflecting centre, axis symmetry, outflow
├── io/output.h/cpp         VTK output + conservation diagnostics
├── init/
│   ├── lane_emden.h/cpp    Lane–Emden polytropic equilibrium
│   ├── sedov.h/cpp         Sedov point-source explosion
│   ├── jeans.h/cpp         Jeans gravitational instability
│   └── evrard.h/cpp        Evrard cold gas sphere collapse
└── main.cpp                Main loop: initialisation → RK2 time stepping → output
```

## Building

Requirements: C++17 compiler, CMake ≥ 3.18. Optional: CUDA Toolkit + NVIDIA AmgX.

```bash
mkdir build && cd build

# CPU-only (Jacobi fallback for Poisson solve)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# With AmgX GPU acceleration
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_AMGX=ON -DAMGX_DIR=/path/to/amgx
make -j$(nproc)
```

## Running

```bash
# Lane–Emden hydrostatic equilibrium with ideal+radiation EOS (default)
./stellar2d --test lane_emden --nr 128 --ntheta 64 --tend 1.0

# Compare against the original gamma-law ideal gas EOS
./stellar2d --test lane_emden --eos ideal --nr 128 --ntheta 64 --tend 1.0

# Sedov blast wave
./stellar2d --test sedov --nr 256 --ntheta 128 --tend 0.1

# Full option list
./stellar2d --test <case> --eos <ideal|ideal_rad> --nr <N> --ntheta <N> --tend <T> --cfl <C> --output-interval <N>
```

Available test cases: `lane_emden`, `lane_emden_perturbed`, `sedov`, `jeans`, `evrard`.

Notes:
- `--mu` controls the mean molecular weight used by both EOS implementations.
- `--radiation-a` controls the dimensionless radiation-pressure coefficient for `ideal_rad`.
- GPU solvers currently remain gamma-law only; use `--eos ideal` for GPU runs.

Output is written in VTK format (`output_XXXX.vtk`) and can be visualised with ParaView.

## Equation–Code Traceability

This project enforces a strict correspondence between equations and source code:

1. [docs/equations.md](docs/equations.md) serves as the authoritative equation reference.
2. Every physics computation in the source is annotated with `// Eq. (X.Y)`.
3. Any discrepancy between the code and the reference document constitutes a bug.
