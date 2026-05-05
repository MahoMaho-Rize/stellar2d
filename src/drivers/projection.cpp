#include "drivers/drivers.h"

#ifdef USE_GPU
#include "projection_solver.cuh"
#include "sim/run_loop.h"

int run_projection(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    ProjSolver proj;
    proj.init(ctx.grid, ctx.eos, cfg.G, cfg.cfl);
    if (cfg.no_sponge) proj.sponge_kappa = 0.0;
    configure_mass_mesh(cfg, proj);
    snapshot_hse_if_needed(cfg, ctx, proj);
    proj.upload_state(ctx.grid, ctx.state);

    SolverOps ops;
    ops.step = [&](double t_, double te) { return proj.step(t_, te); };
    ops.download = [&](const Grid& g, State& s, double) { proj.download_state(g, s); };
    ops.destroy = [&]() { proj.destroy(); };
    run_time_loop(cfg, ctx, t, step, ops);
    return 0;
}
#endif
