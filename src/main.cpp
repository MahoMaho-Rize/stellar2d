// stellar2d — 2D Axisymmetric Euler + Self-Gravity
//
// Top-level entry point. Responsibilities, in order:
//   1. install SIGINT/SIGTERM handler so the time loop can bail cleanly
//   2. parse argv into SimConfig                       (cli/options.cpp)
//   3. build grid / state / EOS / tables / IC          (sim/setup.cpp)
//   4. dispatch to the selected solver driver          (drivers/<name>.cpp)
//   5. dump output_final.vtk + destroy tables          (below)
//
// Each solver owns its own time loop, diagnostics, VTK cadence, and exit
// code in src/drivers/<name>.cpp; add a new solver by writing one driver
// file + adding a `} else if (cfg.solver_type == "newname") { ... }`
// branch here + a prototype in drivers/drivers.h.

#include "grid.h"
#include "state.h"
#include "eos.h"
#include "io/output.h"

#include "cli/options.h"
#include "sim/helpers.h"
#include "sim/setup.h"
#include "sim/run_loop.h"
#include "drivers/drivers.h"

#include <csignal>
#include <cstdio>
#include <string>

// g_interrupted is defined in sim/run_loop.cpp (GPU) or drivers/cpu.cpp (CPU).
static void handle_sigint(int) { g_interrupted = 1; }

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    SimConfig cfg;
    if (int rc = parse_cli(argc, argv, cfg); rc != 0) return rc;

    SimContext ctx;
    if (int rc = setup_simulation(cfg, ctx); rc != 0) return rc;

    double t = 0.0;
    int step = 0;
    std::printf("Starting time integration...\n");

#ifdef USE_GPU
    if      (cfg.solver_type == "radial1d")        { if (int rc = run_radial1d       (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "projection")      { if (int rc = run_projection     (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "simple")          { if (int rc = run_simple         (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "fas2")            { if (int rc = run_fas2           (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "fas" ||
             cfg.solver_type == "explicit")        { if (int rc = run_fas            (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "cart_lag")        { if (int rc = run_cart_lag       (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "cart_ale")        { if (int rc = run_cart_ale       (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "cart_impl")       { if (int rc = run_cart_impl      (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "cart_ale2")       { if (int rc = run_cart_ale2      (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "athena_vl2")      { if (int rc = run_athena_vl2     (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "pseudo_spectral") { if (int rc = run_pseudo_spectral(cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "anelastic_sl")    { if (int rc = run_anelastic_sl   (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "sph2d_spectral")  { if (int rc = run_sph2d_spectral (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "ale2d")           { if (int rc = run_ale2d          (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "wb2d")            { if (int rc = run_wb2d           (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "lowmach")         { if (int rc = run_lowmach        (cfg, ctx, t, step); rc != 0) return rc; }
    else {
#ifdef USE_AMGX
        if (int rc = run_compressible(cfg, ctx, t, step); rc != 0) return rc;
#else
        std::fprintf(stderr, "ERROR: --solver compressible requires AmgX. "
                     "Rebuild with -DAMGX_DIR=/path/to/amgx, or use --solver lowmach.\n");
        return 1;
#endif
    }
#else
    if (int rc = run_cpu(cfg, ctx, t, step); rc != 0) return rc;
#endif

    Diagnostics diag = compute_diagnostics(ctx.grid, ctx.state, cfg.gamma);
    std::printf("Final: step %d  t = %.6e  M = %.10e  E = %.10e\n",
                step, t, diag.total_mass, diag.total_energy);
    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_final.vtk", ctx.run_dir.c_str());
        write_vtk(path, ctx.grid, ctx.state, cfg.gamma);
    }

#ifdef USE_GPU
    ctx.destroy_tables();
#endif

    std::printf("Done.\n");
    return 0;
}
