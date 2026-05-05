#pragma once

#include "grid.h"
#include "state.h"
#include "eos.h"
#include "cli/options.h"

#ifdef USE_GPU
#include "physics/helmholtz_eos.cuh"
#include "physics/opacity_table.cuh"
#endif

#include <string>

// Runtime state built once from SimConfig, then passed by reference to the
// per-solver driver. The config itself is held by value so the driver can read
// the CLI flags; setup may mutate cfg when the mesh type implies derived
// quantities (R_outer, r_inner, M_core, no_sponge).
struct SimContext {
    Grid grid;
    State state;
    EOS eos;
    std::string run_dir;

#ifdef USE_GPU
    // Helmholtz table lives through the whole run; destroyed after the
    // solver loop via SimContext::destroy_tables().
    HelmholtzTable helm_tbl;
    bool helm_loaded = false;

    // Optional MESA kap table pair (low-T + high-T stitched at logT≈4);
    // only loaded if cfg.kap_use_table is true.
    KapTable kap_tbl_lowT;
    KapTable kap_tbl_highT;
    bool kap_loaded = false;

    // Must be called before returning from main; frees device memory.
    void destroy_tables();
#endif
};

// Build a SimContext from cfg. Mutates cfg: sets derived R_outer for
// lane_emden family, sets r_inner / M_core / no_sponge when mass-shell mesh
// is in use. Returns 0 on success, non-zero on fatal error (value is
// suitable for returning from main()).
int setup_simulation(SimConfig& cfg, SimContext& ctx);
