#include "drivers/drivers.h"

#ifdef USE_GPU
#include "fas2_solver.cuh"
#include "io/output.h"
#include "sim/run_loop.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int run_fas2(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== fas2: experimental fork of FAS for low-Mach robustness =====
    // Clone of FasSolver with incremental fixes:
    //   - CGS2 Gram-Schmidt (Knoll-Keyes 2004)
    //   - Unit-normalize v in JFNK matvec (Trilinos NOX)
    //   - Viallet 2016 eq 72 asymmetric L/R scaling
    //   - Line-implicit-in-r preconditioner (cherry-pick from line-jacobi-precond)
    FasSolver2 fas;
    fas.use_simple_smoother = (cfg.precond != "block_jacobi");
    // fas2-specific: --precond line_r toggles line-implicit-in-r JFNK preconditioner
    fas.use_line_precond_r = (cfg.precond == "line_r");
    fas.limiter_type = static_cast<int>(cfg.limiter);
    fas.hllc_variant = cfg.hllc_variant;
    fas.radial_only = cfg.radial_only;
    fas.init(ctx.grid, ctx.eos, cfg.G, cfg.cfl);
    if (cfg.radial_only)
        std::printf("fas2 radial-only mode\n");
    if (cfg.no_sponge) fas.sponge_kappa = 0.0;
    configure_mass_mesh(cfg, fas);
    if (cfg.mesh_type == "mass" && cfg.r_inner <= 0)
        fas.central_damp_r = 0.15 * cfg.R_outer;
    snapshot_hse_if_needed(cfg, ctx, fas);
    fas.upload_state(ctx.grid, ctx.state);

    FasLevel2& fl = fas.levels[0];
    int snap_size = fl.total;
    int max_snaps = static_cast<int>(cfg.t_end / (cfg.output_interval * 1e-5)) + 100;
    long long bytes_per_snap = 4LL * snap_size * sizeof(double);
    size_t mem_free = 0, mem_total = 0;
    cudaMemGetInfo(&mem_free, &mem_total);
    long long max_bytes = static_cast<long long>(mem_free) / 2;
    if (max_snaps > max_bytes / bytes_per_snap)
        max_snaps = static_cast<int>(max_bytes / bytes_per_snap);
    if (max_snaps < 1) max_snaps = 1;

    double* d_snap_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_snap_buf, (long long)max_snaps * 4 * snap_size * sizeof(double)));
    std::vector<double> snap_times, snap_dts;
    std::vector<int> snap_steps;
    int n_snaps = 0;

    SolverOps ops;
    ops.progress_interval = 2000;
    ops.step = [&](double t_, double te) { return fas.step(t_, te); };
    ops.download = [&](const Grid&, State&, double dt_val) {
        if (n_snaps >= max_snaps) return;
        long long off = (long long)n_snaps * 4 * snap_size;
        CUDA_CHECK(cudaMemcpy(d_snap_buf + off, fl.d_rho, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_snap_buf + off + snap_size, fl.d_mr, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 2*snap_size, fl.d_mt, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 3*snap_size, fl.d_rhoE, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
        snap_times.push_back(t);
        snap_dts.push_back(dt_val);
        snap_steps.push_back(step);
        n_snaps++;
    };
    ops.destroy = [&]() {
        if (g_interrupted)
            std::printf("\nInterrupted at step %d, t=%.6e. ", step, t);
        std::printf("Writing %d snapshots from GPU...\n", n_snaps);
        std::vector<double> h_buf(4LL * snap_size);
        for (int s = 0; s < n_snaps; ++s) {
            long long off = (long long)s * 4 * snap_size;
            CUDA_CHECK(cudaMemcpy(h_buf.data(), d_snap_buf + off, 4*snap_size*sizeof(double), cudaMemcpyDeviceToHost));
            for (int ii = 0; ii < fl.nr; ++ii)
                for (int jj = 0; jj < fl.nt; ++jj) {
                    int k = ctx.grid.idx(ii, jj);
                    int kg = (ii + fl.ng) * (fl.nt + 2*fl.ng) + (jj + fl.ng);
                    ctx.state.rho[k] = h_buf[kg];
                    ctx.state.mr[k] = h_buf[snap_size + kg];
                    ctx.state.mtheta[k] = h_buf[2*snap_size + kg];
                    ctx.state.E[k] = h_buf[3*snap_size + kg];
                }
            Diagnostics diag = compute_diagnostics(ctx.grid, ctx.state, cfg.gamma);
            std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                        snap_steps[s], snap_times[s], snap_dts[s], diag.total_mass, diag.total_energy);
            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", ctx.run_dir.c_str(), s + 1);
            write_vtk(fname, ctx.grid, ctx.state, cfg.gamma);
        }
        cudaFree(d_snap_buf);
        fas.download_state(ctx.grid, ctx.state);
        fas.destroy();
    };
    run_time_loop(cfg, ctx, t, step, ops);
    return 0;
}
#endif
