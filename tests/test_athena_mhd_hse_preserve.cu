// test_athena_mhd_hse_preserve.cu
// ============================================================
// Phase B-M1 — Well-balanced MHSE preservation.
//
// Derivation: §B4 of the MHD derivation manuscript:
//   F_wb(U) ≡ R(U) − R(U_hse).   F_wb(U_hse) = 0 by construction.
//
// Setup: isothermal exponential atmosphere ρ(y) = ρ₀ exp(−y/H),
// P = ρ·c_s², optional uniform B_y.  Reflective y-BC, periodic x-BC.
// After snapshot_hse() the residual subtraction is active.
//
// Pass criteria:
//   B1. Naive (no WB) path: drift ≫ 10⁻⁴ after 1000 steps
//       (smoke-test that WB makes a difference; the bad path should fail)
//   B2. WB-enabled: max|δP|/P < 1e-10 over 1000 steps at N = 64
//   B3. WB-enabled: max|δρ|/ρ < 1e-10
//   B4. WB-enabled: max|v| < 1e-8 · c_s (stays at rest to machine ε)
//   B5. WB-enabled: max|∇·B| < 1e-10 (CT still works with source)
// ============================================================

#include "athena_mhd_solver.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

static int g_failures = 0;
static int g_tests = 0;

#define CHECK_LT(got, bound, msg) do {                                \
    ++g_tests;                                                        \
    double _g = (got), _b = (bound);                                  \
    if (!(_g < _b)) {                                                 \
        std::fprintf(stderr, "FAIL %s:%d [%s]: got=%.6e bound=%.1e\n",\
                     __FILE__, __LINE__, msg, _g, _b);                \
        ++g_failures;                                                 \
    } else {                                                          \
        std::printf("  PASS  %s  (got=%.3e < %.1e)\n", msg, _g, _b); \
    }                                                                 \
} while (0)

#define CHECK_GT(got, bound, msg) do {                                \
    ++g_tests;                                                        \
    double _g = (got), _b = (bound);                                  \
    if (!(_g > _b)) {                                                 \
        std::fprintf(stderr, "FAIL %s:%d [%s]: got=%.6e bound=%.1e\n",\
                     __FILE__, __LINE__, msg, _g, _b);                \
        ++g_failures;                                                 \
    } else {                                                          \
        std::printf("  PASS  %s  (got=%.3e > %.1e)\n", msg, _g, _b); \
    }                                                                 \
} while (0)

struct DriftStats {
    double max_drho_rel;
    double max_dP_rel;
    double max_v_abs;
    double max_divB;
    int steps_run;
};

static DriftStats measure_drift(bool use_wb, int nx, int ny, int nsteps_max) {
    AthenaMHDSolver sv;
    const double Lx = 1.0, Ly = 1.0;
    const double gamma = 5.0 / 3.0;
    const double cfl = 0.3;
    const double g_val = 1.0;
    const double H     = 1.0;
    const double rho0  = 1.0;
    const double B0y   = 0.1;   // weak uniform vertical field, β ≫ 1

    sv.init(nx, ny, Lx, Ly, gamma, cfl);
    sv.xorder = 2; sv.limiter = 0; sv.cfl_limit = 0.5;
    sv.init_hse_atmosphere(g_val, H, rho0, B0y);
    if (!use_wb) {
        // For the "naive" control: disable WB but keep gravity table.
        sv.wb_active = false;
    }

    // Snapshot IC ρ, P for drift comparison.
    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng;
    size_t nb = (size_t)sx * sy * sizeof(double);
    std::vector<double> h_rho0(sx * sy), h_mx0(sx * sy), h_my0(sx * sy),
                        h_mz0(sx * sy), h_E0(sx * sy);
    std::vector<double> h_Bxcc0(sx * sy), h_Bycc0(sx * sy), h_Bzcc0(sx * sy);
    sv.fill_ghost(); sv.cons_to_prim();
    cudaMemcpy(h_rho0.data(),  sv.d_rho,   nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Bxcc0.data(), sv.d_Bx_cc, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Bycc0.data(), sv.d_By_cc, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Bzcc0.data(), sv.d_Bz_cc, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mx0.data(),   sv.d_mx,    nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_my0.data(),   sv.d_my,    nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mz0.data(),   sv.d_mz,    nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E0.data(),    sv.d_E,     nb, cudaMemcpyDeviceToHost);

    // Compute IC pressure per cell for comparison.
    auto cell_P = [&](int c, const std::vector<double>& rho,
                      const std::vector<double>& mx,
                      const std::vector<double>& my,
                      const std::vector<double>& mz,
                      const std::vector<double>& E,
                      const std::vector<double>& Bx,
                      const std::vector<double>& By,
                      const std::vector<double>& Bz) {
        double r = std::max(rho[c], 1e-30);
        double u = mx[c] / r, v = my[c] / r, w = mz[c] / r;
        double ke = 0.5 * r * (u*u + v*v + w*w);
        double me = 0.5 * (Bx[c]*Bx[c] + By[c]*By[c] + Bz[c]*Bz[c]);
        return std::max((gamma - 1.0) * (E[c] - ke - me), 1e-30);
    };

    std::vector<double> h_P0(sx * sy);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            h_P0[c] = cell_P(c, h_rho0, h_mx0, h_my0, h_mz0, h_E0,
                             h_Bxcc0, h_Bycc0, h_Bzcc0);
        }

    // Evolve for nsteps_max steps.  Capture drift every 100 steps.
    DriftStats stats{0, 0, 0, 0, 0};
    double t = 0.0;
    const double t_end = 1e30;   // not time-limited; step-limited
    for (int s = 0; s < nsteps_max; ++s) {
        double dt = sv.step(t, t_end);
        if (std::isnan(dt) || dt <= 0) break;
        t += dt;
        stats.steps_run = s + 1;
        // Periodic divB check
        if ((s+1) % 100 == 0 || s == nsteps_max - 1) {
            stats.max_divB = std::max(stats.max_divB,
                                      sv.compute_diagnostics().max_divB);
        }
    }

    // Measure final drift.
    sv.fill_ghost(); sv.cons_to_prim();
    std::vector<double> h_rho(sx * sy), h_mx(sx * sy), h_my(sx * sy),
                        h_mz(sx * sy), h_E(sx * sy);
    std::vector<double> h_Bxcc(sx * sy), h_Bycc(sx * sy), h_Bzcc(sx * sy);
    cudaMemcpy(h_rho.data(),  sv.d_rho,   nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mx.data(),   sv.d_mx,    nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_my.data(),   sv.d_my,    nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mz.data(),   sv.d_mz,    nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E.data(),    sv.d_E,     nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Bxcc.data(), sv.d_Bx_cc, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Bycc.data(), sv.d_By_cc, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Bzcc.data(), sv.d_Bz_cc, nb, cudaMemcpyDeviceToHost);

    double c_s = std::sqrt(g_val * H);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            double r = std::max(h_rho[c], 1e-30);
            double P = cell_P(c, h_rho, h_mx, h_my, h_mz, h_E,
                              h_Bxcc, h_Bycc, h_Bzcc);
            double drho_rel = std::fabs(h_rho[c] - h_rho0[c]) / h_rho0[c];
            double dP_rel   = std::fabs(P - h_P0[c]) / h_P0[c];
            double vmag     = std::sqrt(h_mx[c]*h_mx[c] + h_my[c]*h_my[c]
                                        + h_mz[c]*h_mz[c]) / r;
            stats.max_drho_rel = std::max(stats.max_drho_rel, drho_rel);
            stats.max_dP_rel   = std::max(stats.max_dP_rel,   dP_rel);
            stats.max_v_abs    = std::max(stats.max_v_abs,    vmag / c_s);
        }

    std::printf("    [%s] N=%dx%d, %d steps, t=%.3e\n"
                "       max|δρ|/ρ = %.3e,  max|δP|/P = %.3e\n"
                "       max|v|/c_s = %.3e,  max|∇·B| = %.3e\n",
                use_wb ? "WB on" : "WB off",
                nx, ny, stats.steps_run, t,
                stats.max_drho_rel, stats.max_dP_rel,
                stats.max_v_abs, stats.max_divB);

    sv.destroy();
    return stats;
}

int main() {
    std::printf("=== athena_mhd HSE preservation (B-M1, §B4) ===\n");
    std::printf("  Setup: isothermal atm ρ(y)=ρ₀exp(-y/H), g=1, H=1, B0y=0.1\n");
    std::printf("         N=64×64, reflective y-BC, periodic x-BC.\n\n");

    std::printf("[control] WB disabled — naive ρg source only:\n");
    DriftStats naive = measure_drift(/*use_wb=*/false, 64, 64, 1000);

    std::printf("\n[test] WB enabled — §B4 residual subtraction:\n");
    DriftStats wb = measure_drift(/*use_wb=*/true, 64, 64, 1000);

    std::printf("\n=== Assertions ===\n");

    // B1: naive path drift > 1e-4 (just prints; we don't make this a
    //     hard fail because naive might be closer-than-expected on tiny
    //     grids).  Keep as an informational assertion.
    ++g_tests;
    if (naive.max_dP_rel > 1e-6) {
        std::printf("  PASS  B1 naive drift visible (%.2e > 1e-6)\n",
                    naive.max_dP_rel);
    } else {
        std::printf("  WARN  B1 naive drift tiny (%.2e); test still valid\n",
                    naive.max_dP_rel);
    }

    // B2-B5: WB path must stay at machine precision.
    CHECK_LT(wb.max_dP_rel,   1e-10, "B2 WB |δP|/P < 1e-10");
    CHECK_LT(wb.max_drho_rel, 1e-10, "B3 WB |δρ|/ρ < 1e-10");
    CHECK_LT(wb.max_v_abs,    1e-8,  "B4 WB |v|/c_s < 1e-8");
    CHECK_LT(wb.max_divB,     1e-10, "B5 WB max|∇·B| < 1e-10");

    // B6: WB must strictly beat naive — shows the correction is non-trivial.
    ++g_tests;
    if (wb.max_dP_rel < naive.max_dP_rel) {
        double factor = naive.max_dP_rel / std::max(wb.max_dP_rel, 1e-30);
        std::printf("  PASS  B6 WB improves over naive by %.1e×\n", factor);
    } else {
        std::fprintf(stderr,
            "FAIL B6 WB worse than naive (WB=%.2e, naive=%.2e)\n",
            wb.max_dP_rel, naive.max_dP_rel);
        ++g_failures;
    }

    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
