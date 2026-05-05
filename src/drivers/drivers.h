#pragma once

#include "cli/options.h"
#include "sim/setup.h"

// Each solver branch formerly inlined into main() is exposed as a free
// function here. Drivers take mutable references to cfg (some mutate it,
// e.g. radial1d), the SimContext (grid/state/eos/tables loaded by
// setup_simulation), and the time-step counters (so the finalization
// block in main() can report the final step count).
//
// Return 0 on successful completion, non-zero on fatal solver error. The
// value is propagated straight to main()'s return code.

#ifdef USE_GPU
int run_radial1d        (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_projection      (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_simple          (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_fas2            (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_fas             (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_cart_lag        (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_cart_ale        (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_cart_impl       (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_cart_ale2       (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_pseudo_spectral (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_anelastic_sl    (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_sph2d_spectral  (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_ale2d           (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_wb2d            (SimConfig& cfg, SimContext& ctx, double& t, int& step);
int run_lowmach         (SimConfig& cfg, SimContext& ctx, double& t, int& step);
#ifdef USE_AMGX
int run_compressible    (SimConfig& cfg, SimContext& ctx, double& t, int& step);
#endif
#endif
