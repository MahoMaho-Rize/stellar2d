// test_ale2_phase_m_compensation.cu
// ============================================================
// Regression lock for the 2026-05-07 P33 local-compatibility fix
// (commit 25e3cfc).  The Phase-M remap's KE→IE compensation was
// previously done on a global mean-field basis, which violated
// per-cell locality and drifted under non-uniform advection.
//
// The per-cell Jensen #2 compensation now ensures E_total is
// conserved to machine precision on smooth non-uniform IC
// advection, even when ρ has O(1) contrast across the grid.
//
// IC: ρ = ρ_base · [1 + α · exp(-r²/σ²)] with
//       ρ_base=1, α=0.3 (so peak ρ=1.3), σ=0.1,
//       centred at (0.5, 0.5) on [0,1]².
//     P = 1 uniform (isobaric perturbation).
//     v = (0.5, 0.3) uniform (non-aligned advection).
//     g = 0, bc = fully periodic.
//
// Checks:
//   U1. |E_end − E_0| / E_0 < 1e-10  (P33 core property)
//   U2. |M_end − M_0| / M_0 < 1e-12
//   U3. max_v stays finite and within 10 % of IC |v|
//
// On a pre-fix codebase (global mean-field compensation), U1
// drifts to ~1e-4 within 30 steps for this IC.
// ============================================================

#include "cart_ale2_solver.cuh"
#include "gpu_common.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

// Forward decl of the node-mass builder in cart_ale2_kernels.cu — we need
// to rebuild d_mnode after we overwrite d_dm with a Gaussian profile.
__global__ void k_cale2_node_mass(const double*, double*, int, int, int);

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

// Upload a Gaussian-ρ, uniform-v IC bypassing the built-in init_*
// methods.  We read d_X, d_Y, d_Vol from the solver (already built
// by init()) and write dm, e_int, vx, vy directly.
static void upload_gaussian_ic(CartAle2Solver& sol,
                               double rho_base, double alpha,
                               double sigma, double P0,
                               double vx0, double vy0) {
    const int nx = sol.nx, ny = sol.ny;
    const int nnode_x = sol.nnode_x, nnode_y = sol.nnode_y;
    const int nnode = sol.nnode;
    const int ncell = sol.ncell;
    const double gamma = sol.gamma;

    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), sol.d_X, nnode*sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), sol.d_Y, nnode*sizeof(double),
                          cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), sol.d_Vol, ncell*sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::vector<double> h_dm(ncell), h_e(ncell);
    const double xc_dom = 0.5, yc_dom = 0.5;
    const double inv_sig2 = 1.0 / (sigma * sigma);
    for (int ic = 0; ic < nx; ++ic) {
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic * ny + jc;
            int I[4] = {
                ic * nnode_y + jc,
                (ic + 1) * nnode_y + jc,
                (ic + 1) * nnode_y + (jc + 1),
                ic * nnode_y + (jc + 1)
            };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double dx = Xc - xc_dom, dy = Yc - yc_dom;
            double gauss = std::exp(-(dx*dx + dy*dy) * inv_sig2);
            double rho = rho_base * (1.0 + alpha * gauss);
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = P0 / ((gamma - 1.0) * rho);
        }
    }
    CUDA_CHECK(cudaMemcpy(sol.d_dm,    h_dm.data(), ncell*sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sol.d_e_int, h_e.data(),  ncell*sizeof(double),
                          cudaMemcpyHostToDevice));
    std::vector<double> h_vX(nnode, vx0), h_vY(nnode, vy0);
    CUDA_CHECK(cudaMemcpy(sol.d_vX, h_vX.data(), nnode*sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sol.d_vY, h_vY.data(), nnode*sizeof(double),
                          cudaMemcpyHostToDevice));

    // Rebuild d_mnode from the new d_dm (same call every init_* does).
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(sol.d_dm, sol.d_mnode,
                                             sol.nx, sol.ny, sol.bc_mode);
    CUDA_CHECK(cudaGetLastError());
}

int main() {
    std::printf("=== cart_ale2 Phase-M per-cell KE→IE compensation (P33 lock) ===\n\n");

    const int nx = 64, ny = 64;
    const double Lx = 1.0, Ly = 1.0;
    const double gamma = 5.0 / 3.0;
    const double cfl   = 0.3;
    const int nsteps = 30;

    CartAle2Solver sol;
    sol.init(nx, ny, Lx, Ly, gamma, cfl);
    sol.bc_mode = 3;   // fully periodic
    sol.g_y     = 0.0;
    upload_gaussian_ic(sol,
                       /*rho_base=*/1.0, /*alpha=*/0.3, /*sigma=*/0.1,
                       /*P0=*/1.0, /*vx0=*/0.5, /*vy0=*/0.3);

    auto d0 = sol.compute_diagnostics();
    const double E0 = d0.total_E;
    const double M0 = d0.total_mass;
    const double v0 = d0.max_v;
    std::printf("  IC: E0=%.10e  M0=%.10e  max_v=%.6e\n", E0, M0, v0);

    double t = 0.0;
    const double t_end_cap = 1e9;
    for (int s = 0; s < nsteps; ++s) {
        double dt = sol.step(t, t_end_cap);
        t += dt;
    }
    auto de = sol.compute_diagnostics();
    std::printf("  after %d steps: E_end=%.10e  M_end=%.10e  max_v=%.6e\n",
                nsteps, de.total_E, de.total_mass, de.max_v);

    double rel_E = std::fabs(de.total_E    - E0) / std::fabs(E0);
    double rel_M = std::fabs(de.total_mass - M0) / std::fabs(M0);
    double rel_v = std::fabs(de.max_v      - v0) / std::fabs(v0);

    CHECK_LT(rel_E, 1e-10, "U1: |ΔE|/E0 < 1e-10 (P33 core)");
    CHECK_LT(rel_M, 1e-12, "U2: |ΔM|/M0 < 1e-12");
    CHECK_LT(rel_v, 0.10,  "U3: max_v within 10 % of IC |v|");

    sol.destroy();

    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
