#include "drivers/drivers.h"

#ifdef USE_GPU
#include "simple_solver.cuh"
#include "sim/run_loop.h"

int run_simple(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    SimpleSolver sim;
    sim.init(ctx.grid, ctx.eos, cfg.G, cfg.cfl);
    if (cfg.no_sponge) sim.sponge_kappa = 0.0;
    configure_mass_mesh(cfg, sim);
    snapshot_hse_if_needed(cfg, ctx, sim);
    sim.upload_state(ctx.grid, ctx.state);

    SolverOps ops;
    ops.step = [&](double t_, double te) { return sim.step(t_, te); };
    ops.download = [&](const Grid& g, State& s, double) { sim.download_state(g, s); };
    ops.destroy = [&]() { sim.destroy(); };
    run_time_loop(cfg, ctx, t, step, ops);
    return 0;
}
#endif
