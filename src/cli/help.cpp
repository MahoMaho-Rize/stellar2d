#include "cli/help.h"

#include <cstdio>

#ifndef STELLAR2D_GIT_HASH
#define STELLAR2D_GIT_HASH "unknown"
#endif

#ifndef STELLAR2D_BUILD_DATE
#define STELLAR2D_BUILD_DATE "unknown"
#endif

// The help block below is hard-coded.  Keep formatting consistent so users
// can grep for flag names.  Tier C will regenerate this from SolverSpec.
//
// Conventions:
//   - Column 1 (2 spaces): indent of a flag name
//   - Flag name padded to ~24 chars before the description
//   - `[default]` in square brackets at end of description
//   - Grouped by: Core / IO / Physics / IC / Test IC / Solver-specific / Misc
//
// Solver-specific groups flag which solver each section applies to.
// Flags that are solver-specific but currently silently accepted by any
// solver are listed under their owning solver; a note at the top warns that
// Tier A does not yet enforce solver-flag compatibility.

void print_help() {
    std::puts(
R"(stellar2d — 2D axisymmetric compressible Euler + self-gravity (+ MHD, low-Mach, spectral)

Usage:
  stellar2d [FLAGS]             Run a simulation (current, flat-CLI form)
  stellar2d --help              Show this help and exit
  stellar2d --version           Print version and exit

Notes:
  - Unknown flags are a HARD ERROR with a did-you-mean suggestion.
  - Many solver-specific flags are still silently accepted by other
    solvers in Tier A.  Tier C will enforce solver/flag compatibility;
    see docs/design/cli_unification_plan_2026-05-09.md.
  - Each run writes runs/<name>/config.dump.txt for reproducibility.
    Tier B will upgrade this to config.toml with --config reload support.

Available solvers (see docs/design/*_design.md for each):
  compressible   (default)        Flagship compressible Euler + AmgX (USE_AMGX build)
  strang                          2D explicit Strang split HLLC+MUSCL (baseline)
  wb2d                            Well-balanced 2D (perturbed t~2 unstable)
  lowmach                         JFNK low-Mach (HLLC variants)
  fas / fas2 / simple / projection  Low-Mach implicit family
  cart_lag / cart_ale / cart_impl Lagrangian / ALE / implicit Cartesian
  cart_ale2                       Cart ALE + periodic BC + PPM (convection workhorse)
  athena_vl2 / athena_mhd         Athena++ vl2 port + MHD port
  radial1d                        1D Lagrangian JFNK hydro testbed
  pseudo_spectral                 2D incompressible NS (cuFFT)
  anelastic_sl                    2D Boussinesq / anelastic spectral
  sph2d_spectral                  2D thin-shell spectral
  ale2d                           axisymmetric ALE (hoop stress bug — see design doc)

Core flags:
  --solver <name>           Solver to use                        [compressible]
  --test <case>             Test case name                       [lane_emden]
  --nr <int>                Radial / x-direction grid points     [128]
  --ntheta <int>            Theta / y-direction grid points      [64]
  --cfl <float>             CFL number                           [0.4]
  --tend <float>            End time (code units)                [1.0]
  --gamma <float>           Adiabatic index                      [5/3]
  --G <float>               Gravitational constant               [1.0]
  --mesh <type>             log | linear | mass                  [log]
  --limiter <name>          vanleer | mc | minmod                [minmod]
  --run-base <path>         Output root directory                [runs]
  --perturb <amp>           Initial δρ/ρ for perturbed IC        [1e-3]

IO / diagnostics:
  --output-interval <N>     Step cadence for CSV diagnostics     [100]
  --vtk-interval <N>        Step cadence for VTK dump            [0]
  --vtk-dt <float>          Time-based VTK cadence (overrides)   [0]
  --frame-buffer            Buffer VTK frames in VRAM            [off]
  --frame-headroom-mb <N>   VRAM headroom when buffering         [1024]
  --diag-interval <N>       Diagnostic cadence (0 = follow CSV)  [0]
  --compute-error           Emit <test>-errors.dat at t_end      [off]

EOS / physics:
  --eos <type>              ideal | ideal_rad | helmholtz        [ideal]
  --eos-mu <float>          Mean molecular weight                [1.0]
  --eos-rad-a <float>       Radiation constant (code units)      [0.1]
  --helm-table <path>       Helmholtz EOS binary
  --helm-abar <float>       Abar for Helm EOS                    [1.28]
  --helm-zbar <float>       Zbar for Helm EOS                    [1.13]

Nuclear burning (radial1d + cart_impl):
  --nuclear                 Enable pp-chain burn
  --nuc-x <float>           Hydrogen fraction                    [0.7]
  --nuc-y <float>           Helium fraction                      [0.28]
  --nuc-scale <float>       Rate multiplier                      [1.0]
  --nuc-q <erg/g>           Energy per burn                      [6.4e18]
  --nuc-t-floor <K>         Temperature floor                    [1e6]
  --nuc-t-scale <float>     Temperature scale                    [1.0]
  --nuc-compress <frac>     Dynamic scale: ε·dt/(cv·T) ≤ frac    [0]
  --species                 Track per-species abundance          [off]

Radiation:
  --radiation               Enable radiation pressure            [off]
  --rad-c <float>           Reduced speed of light               [100.0]
  --rad-T-phot-floor <K>    Photospheric T floor                 [0]

Opacity tables (radial1d + implicit solvers):
  --kap                     Enable MESA opacity lookup           [off]
  --kap-dir <path>          Table directory                      [third_party/mesa_kap]
  --kap-Z <float>           Metallicity                          [0.02]
  --kap-highT <family>      High-T table family                  [gs98]
  --kap-lowT <family>       Low-T table family                   [lowT_fa05_gs98]
  --kap-logT-lo-end <T>     Low table upper logT                 [3.9]
  --kap-logT-hi-start <T>   High table lower logT                [4.1]

MLT convection (radial1d):
  --mlt                     Enable MLT convection                [off]
  --mlt-alpha <float>       Mixing length / H_p                  [1.5]

Initial conditions (radial):
  --ic-solar                Use physical cgs solar Lane-Emden
  --ic-rho-c <ρ>            Central density override             [test default]
  --ic-rstar <cm>           Stellar radius override              [derive from K]
  --ic-n-poly <n>           Polytropic index                     [1.5]
  --ic-mesa <path>          MESA IC file                         [(none)]
  --ic-mesa-seed-T          Seed (e,P) from Helm(ρ,T_MESA)
  --ic-mesa-atm-zones <N>   Hybrid outer log-zoning              [0 = equal-mass]
  --atm-split <N>           Operator-split rad outer N zones     [0]
  --rich-profile            Emit T, κ, ∇_ad, ... per zone

Initial conditions (cart_ale2 local_convection):
  --ic-slab <file>          Slab IC (6-col plain text from build_ic.py)
  --slab-perturb <amp>      Entropy seed amplitude               [0.01]
  --slab-seed-k <int>       Horizontal seed mode                 [4]

Boundary / radial-only:
  --no-sponge               Disable sponge layer                 [off]
  --radial-only             Enforce v_theta=0, skip θ work       [off]
  --r-inner <r>             Inner radial cut                     [auto]

Test-case IC parameters (activate for the matching --test case):
  --perturb                                lane_emden_perturbed δρ/ρ
  --bubble "xc,yc,rb,α,β"                  hse_bubble (repeatable)
  --bubble-xc / -yc / -rb / -alpha / -beta hse_bubble (single-bubble shortcut)
  --bubble-mode pressure|entropy           hse_bubble perturbation mode
  --ewave-rho0 / -P0 / -u0 / -A / -k / -periods    entropy_wave
  --awave-rho0 / -P0 / -A / -k / -periods  acoustic_wave (Athena++ linwave)
  --shear-V0 / -k / -rho / -P              shear_mode (ν_eff probe)
  --cart-ale2-kh-k <int>                   kh_lecoanet mode number
  --andrassy-amp / -seed / -noise          andrassy2022 O-shell
  --heat-flux / -lsun / -bot-R / -bot-frac local_convection bottom heating
  --cool-tau / -top-frac                   local_convection Newton cooling

Solver.cart_ale / cart_ale2 (remap + AV controls):
  --remap-order <1|2>       1 = donor-cell, 2 = MUSCL/PPM        [2]
  --remap-limiter <name>    minmod | vanleer | mc                [vanleer]
  --cq-lin <float>          Caramana AV linear coefficient       [0.5]
  --cq-quad <float>         Caramana AV quadratic coefficient    [2.0]
  --shear-aware-av          Reduce Q in shear-dominated cells    [off]
  --rebuild-order <0|1>     Node-velocity rebuild order          [0]

Solver.cart_ale2 (adds to cart_ale flags):
  --bc-x {reflect|periodic} X-direction BC                       [reflect]
  --bc-y {reflect|periodic} Y-direction BC                       [reflect]
  --ppm                     PPM-in-remap (default OFF -> MUSCL)
  --ppm-limiter <cs|cw>     PPM limiter                          [cs]
  --ppm-space <prim|cons>   PPM reconstruction space             [prim]
  --no-ppm-char             Skip characteristic-variable project (prim space only)
  --ppm-char                Enable characteristic-variable project (default)
  --trace-cells "ic,jc;..." Per-cell cumulative tracer           [(none)]
  --trace-step-cap <N>      Max step rows per pick cell          [0 = off]

Solver.athena_vl2:
  --athena-xorder <1|2>     1 = donor-cell, 2 = PLM              [2]
  --athena-limiter <name>   vanleer (0) | minmod (1)             [vanleer]

Solver.radial1d implicit (activated by --implicit):
  --implicit                Use BE + JFNK instead of explicit RK2
  --dt-implicit <float>     Fixed dt                             [acoustic CFL]
  --dt-implicit-scale <f>   Multiplier on CFL dt                 [1.0]
  --no-viallet              Disable Viallet L/R scaling          [off]
  --precond-tridiag         Block-tridiag preconditioner         [off]
  --jfnk-autodiff           Exact J·v via Dual<1> AD             [off]
  --no-rhse                 Diagnostic: skip R_hse subtraction   [off]
  --newton-tol <float>      Override Newton ||F|| tol            [1e-8]
  --hse-resnap <N>          Re-snapshot R_hse every N steps      [0]
  --dt-thermal-frac <f>     dt ≤ f · IE/L_surf                   [0]
  --dt-mach-cap <f>         Shrink dt when max Mach > f          [0]

Solver.lowmach:
  --precond <type>          line_jacobi | pbp | block | ...      [line_jacobi]
  --lm-hllc                 Shortcut for --hllc lm               (Rieper LM-HLLC)
  --lhllc                   Shortcut for --hllc lhllc            (Minoshima LHLLC)
  --hllc {standard|lm|lhllc}HLLC variant                         [standard]

Solver.pseudo_spectral:
  See docs/design/pseudo_spectral_design_2026-05-01.md.
  --ps-nu <float>           Kinematic viscosity                  [1e-4]
  --ps-Lx / --ps-Ly         Domain lengths                       [1.0]
  --ps-vshear <float>       KH base flow speed                   [0.5]
  --ps-k <int>              KH perturbation mode                 [4]
  --ps-explicit             Force all-explicit SSP-RK3           [off]
  --ps-adv-only             Use advective form (skew default)    [off]
  --ps-forcing-eps <ε>      Stochastic forcing energy injection  [0]
  --ps-forcing-kf <int>     Forcing central mode                 [32]
  --ps-forcing-dk <int>     Forcing half-width                   [1]
  --ps-forcing-seed <u64>   RNG seed                             [0x5a5a5a5a]
  --ps-forcing-host-rng     Force host mt19937 + D2H             [off]
  --ps-drag <α>             Linear drag -α·ω                     [0]
  --ps-hyper <p>            Hyperviscosity power                 [1]
  --ps-conservative         Use conservative (rotational) form   [off]
  --ps-batched-fft          Enable batched FFT pipeline          [off]
  --ps-pi                   PI controller adaptive dt            [off]
  --ps-tg-k <int>           Taylor-Green wavenumber              [2]
  --ps-vm-gamma / -sigma / -dist   vortex_merger knobs
  --ps-ckpt-every <N>       Checkpoint every N steps             [0]
  --ps-resume <path>        Resume from checkpoint               [(none)]

Solver.sph2d_spectral:
  See docs/design/sph_spectral_roadmap.md.
  --sph-R <float>           Sphere radius                        [1.0]
  --sph-Omega <float>       Rotation rate                        [1.0]
  --sph-nu <float>          Kinematic viscosity                  [1e-4]
  --sph-Lmax <int>          Truncation (0 = auto)                [0]
  --sph-drag <α>            Linear drag                          [0]
  --sph-hyper <p>           Hyperviscosity power                 [1]
  --sph-pi                  PI controller adaptive dt            [off]
  --sph-rossby-l/-m/-amp    Rossby-wave IC knobs
  --sph-forcing-eps/-lmin/-lmax/-seed  Stochastic forcing knobs
  --sph-ckpt-every <N>      Checkpoint every N steps             [0]
  --sph-resume <path>       Resume from checkpoint               [(none)]

For design rationale of this Tier A hardening and Tier B/C roadmap see
  docs/design/cli_unification_plan_2026-05-09.md
)"
    );
}

void print_version() {
    std::printf("stellar2d %s (built %s)\n",
                STELLAR2D_GIT_HASH, STELLAR2D_BUILD_DATE);
}
