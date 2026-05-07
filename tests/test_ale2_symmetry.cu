// test_ale2_symmetry.cu
// ============================================================
// Regression lock for direction-dependent bugs in the swept-remap
// and node-rebuild paths (history: the 2026-05-07 `de242a8`
// ex-sign bug that only affected x-direction mass transport).
//
// Protocol:
//   Run A: Gaussian ρ blob centred at (0.3, 0.5) with v = (0.5, 0)
//          advect for 50 steps under bc=3 (fully periodic, g=0).
//   Run B: identical solver but with IC rotated 90° — blob at
//          (0.5, 0.3) and v = (0, 0.5).  Advect same steps.
//
// Because the flow is symmetric under the 90° swap (x, vx, ic)
// ↔ (y, vy, jc), every global diagnostic should match between A
// and B to the same tolerance we hold on mass conservation.
//
// Checks:
//   S1. |KE_A - KE_B| / KE_A < 1e-10
//   S2. |IE_A - IE_B| / IE_A < 1e-10
//   S3. |M_A  - M_B|  / M_A  < 1e-12
//   S4. |max_v_A - max_v_B| / max_v_A < 1e-10
// ============================================================

#include "cart_ale2_solver.cuh"
#include "gpu_common.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

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

static void upload_blob_ic(CartAle2Solver& sol,
                           double xc_blob, double yc_blob,
                           double sigma, double alpha,
                           double P0, double vx0, double vy0) {
    const int nx = sol.nx, ny = sol.ny;
    const int nnode_y = sol.nnode_y, nnode = sol.nnode;
    const int ncell = sol.ncell;
    const double gamma = sol.gamma;
    const double rho_base = 1.0;

    std::vector<double> h_X(nnode), h_Y(nnode), h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_X.data(),   sol.d_X,   nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(),   sol.d_Y,   nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), sol.d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_dm(ncell), h_e(ncell);
    const double inv_sig2 = 1.0 / (sigma * sigma);
    for (int ic = 0; ic < nx; ++ic) {
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic * ny + jc;
            int I[4] = {
                ic * nnode_y + jc, (ic + 1) * nnode_y + jc,
                (ic + 1) * nnode_y + (jc + 1), ic * nnode_y + (jc + 1)
            };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double dx = Xc - xc_blob, dy = Yc - yc_blob;
            double gauss = std::exp(-(dx*dx + dy*dy) * inv_sig2);
            double rho = rho_base * (1.0 + alpha * gauss);
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = P0 / ((gamma - 1.0) * rho);
        }
    }
    CUDA_CHECK(cudaMemcpy(sol.d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sol.d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> h_vX(nnode, vx0), h_vY(nnode, vy0);
    CUDA_CHECK(cudaMemcpy(sol.d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sol.d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(sol.d_dm, sol.d_mnode,
                                             sol.nx, sol.ny, sol.bc_mode);
    CUDA_CHECK(cudaGetLastError());
}

static CartAle2Solver::Diagnostics
run_case(double xc, double yc, double vx, double vy, int nsteps) {
    const int nx = 64, ny = 64;
    const double Lx = 1.0, Ly = 1.0;
    const double gamma = 1.4;
    const double cfl = 0.3;
    CartAle2Solver sol;
    sol.init(nx, ny, Lx, Ly, gamma, cfl);
    sol.bc_mode = 3;     // fully periodic
    sol.g_y     = 0.0;
    upload_blob_ic(sol, xc, yc, /*sigma=*/0.08, /*alpha=*/0.3,
                   /*P0=*/1.0, vx, vy);

    double t = 0.0;
    const double t_end_cap = 1e9;
    for (int s = 0; s < nsteps; ++s) {
        double dt = sol.step(t, t_end_cap);
        t += dt;
    }
    auto d = sol.compute_diagnostics();
    sol.destroy();
    return d;
}

int main() {
    std::printf("=== cart_ale2 x/y 90° symmetry lock ===\n\n");

    auto A = run_case(/*xc=*/0.3, /*yc=*/0.5,
                      /*vx=*/0.5, /*vy=*/0.0, /*nsteps=*/50);
    auto B = run_case(/*xc=*/0.5, /*yc=*/0.3,
                      /*vx=*/0.0, /*vy=*/0.5, /*nsteps=*/50);

    std::printf("  A (x-advection): M=%.10e KE=%.6e IE=%.10e max_v=%.6e\n",
                A.total_mass, A.total_KE, A.total_internal_E, A.max_v);
    std::printf("  B (y-advection): M=%.10e KE=%.6e IE=%.10e max_v=%.6e\n",
                B.total_mass, B.total_KE, B.total_internal_E, B.max_v);

    auto rel = [](double a, double b) {
        return std::fabs(a - b) / std::max(std::fabs(a), 1e-30);
    };

    CHECK_LT(rel(A.total_KE, B.total_KE),           1e-10, "S1: KE symmetry");
    CHECK_LT(rel(A.total_internal_E, B.total_internal_E), 1e-10, "S2: IE symmetry");
    CHECK_LT(rel(A.total_mass, B.total_mass),        1e-12, "S3: M symmetry");
    CHECK_LT(rel(A.max_v, B.max_v),                   1e-10, "S4: max_v symmetry");

    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
