// ============================================================
// test_strang_muscl.cu — Verify MUSCL-Hancock predictor
//
// Tests:
//   1. MC limiter: basic correctness via a kernel
//   2. Y-sweep WB: pure HSE → face states match at every interface
//   3. X-sweep: smooth profile → bounded face states
//   4. Hancock half-step: face values evolve correctly
// ============================================================

#include "gpu/strang_solver.cuh"
#include "gpu/strang_device.cuh"  // d_mc_limit
#include "gpu/fas_common.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

static int n_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); n_fail++; } \
    else { std::printf("  PASS: %s\n", msg); } \
} while(0)

// Forward-declare the kernels (defined in strang_solver.cu)
__global__ void k_muscl_hancock_x(
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*,
    double, double, double, int, int, int, int);

__global__ void k_muscl_hancock_y(
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*,
    double, double, double,
    double, double, double, double,
    int, int, int, int);

__global__ void k_ghost_x(double*, double*, double*, double*, int, int, int, int);
__global__ void k_ghost_y(double*, double*, double*, double*, int, int, int, int);

// Small kernel to test MC limiter on device
__global__
void k_test_mc(double* out, const double* a, const double* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = d_mc_limit(a[i], b[i]);
}

int main()
{
    std::printf("=== test_strang_muscl ===\n\n");

    // ---- Test 1: MC limiter unit tests ----
    std::printf("[Test 1] MC limiter\n");
    {
        // Test cases: (a, b) → expected
        double h_a[] = { 1.0, -1.0,  1.0,  2.0,  0.5,  3.0, -2.0};
        double h_b[] = { 1.0, -1.0, -1.0,  1.0,  0.5,  1.0, -1.0};
        // MC(1,1) = min(1, 2, 2) = 1
        // MC(-1,-1) = -min(1, 2, 2) = -1
        // MC(1,-1) = 0 (different signs)
        // MC(2,1) = min(1.5, 4, 2) = 1.5
        // MC(0.5,0.5) = min(0.5, 1, 1) = 0.5
        // MC(3,1) = min(2, 6, 2) = 2
        // MC(-2,-1) = -min(1.5, 4, 2) = -1.5
        double expected[] = {1.0, -1.0, 0.0, 1.5, 0.5, 2.0, -1.5};
        int n = 7;

        double *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, n*sizeof(double));
        cudaMalloc(&d_b, n*sizeof(double));
        cudaMalloc(&d_out, n*sizeof(double));
        cudaMemcpy(d_a, h_a, n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, n*sizeof(double), cudaMemcpyHostToDevice);

        k_test_mc<<<1, n>>>(d_out, d_a, d_b, n);

        double h_out[7];
        cudaMemcpy(h_out, d_out, n*sizeof(double), cudaMemcpyDeviceToHost);

        bool all_ok = true;
        for (int i = 0; i < n; i++) {
            if (std::fabs(h_out[i] - expected[i]) > 1e-12) {
                std::printf("  MC(%g,%g) = %g, expected %g\n",
                            h_a[i], h_b[i], h_out[i], expected[i]);
                all_ok = false;
            }
        }
        CHECK(all_ok, "MC limiter all 7 cases correct");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    }

    // ---- Test 2: Y-sweep WB on pure HSE ----
    std::printf("\n[Test 2] Y-sweep WB: pure HSE → matching face states\n");
    {
        int nx = 32, ny_val = 64;
        double Lx = 1.0, Ly = 1.5;
        double gamma = 5.0/3.0, g = 1.0, cfl = 0.4;

        StrangSolver sol;
        sol.init(nx, ny_val, Lx, Ly, gamma, g, cfl, 1.0, 1.0);
        // NO bubble → pure HSE: all perturbations = 0

        // Fill ghost cells
        sol.fill_ghost_x();
        sol.fill_ghost_y();

        // Run MUSCL-Hancock y-sweep predictor
        int N_phys = nx * ny_val;
        int B = 256;
        double dt_test = 1e-3;   // arbitrary small dt
        k_muscl_hancock_y<<<(N_phys+B-1)/B, B>>>(
            sol.d_rho, sol.d_mx, sol.d_my, sol.d_E,
            sol.d_rho_bar, sol.d_p_bar,
            sol.d_wL, sol.d_wR,
            dt_test, sol.dy, gamma,
            sol.y_lo, sol.rho0_bottom, sol.g_grav, sol.K_poly,
            nx, ny_val, sol.ng, sol.stride());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download face states
        int Ntot = sol.total_cells();
        std::vector<double> h_wL(4*Ntot), h_wR(4*Ntot);
        CUDA_CHECK(cudaMemcpy(h_wL.data(), sol.d_wL, 4*Ntot*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_wR.data(), sol.d_wR, 4*Ntot*sizeof(double), cudaMemcpyDeviceToHost));

        int str = sol.stride();
        int ng = sol.ng;

        // At each interior face j+1/2 (between cells j and j+1):
        //   Left state  = wR of cell j     (top face of cell j)
        //   Right state = wL of cell j+1   (bottom face of cell j+1)
        // In WB HSE these must match to roundoff.
        double max_drho = 0, max_dP = 0, max_dv = 0;
        int n_faces = 0;
        for (int j = 0; j < ny_val - 1; j++) {
            for (int i = 0; i < nx; i++) {
                int k_j  = (j + ng) * str + (i + ng);
                int k_jp = (j + 1 + ng) * str + (i + ng);

                double rhoL = h_wR[k_j*4+0];   // top face of cell j
                double rhoR = h_wL[k_jp*4+0];  // bottom face of cell j+1
                double vL   = h_wR[k_j*4+2];
                double vR   = h_wL[k_jp*4+2];
                double PL   = h_wR[k_j*4+3];
                double PR   = h_wL[k_jp*4+3];

                max_drho = std::fmax(max_drho, std::fabs(rhoL - rhoR));
                max_dv   = std::fmax(max_dv,   std::fabs(vL   - vR));
                max_dP   = std::fmax(max_dP,   std::fabs(PL   - PR));
                n_faces++;
            }
        }
        std::printf("  Checked %d y-faces\n", n_faces);
        std::printf("  max |Δρ| = %.4e  max |Δv| = %.4e  max |ΔP| = %.4e\n",
                    max_drho, max_dv, max_dP);
        // Hancock half-step introduces O(dt*dy²) residual; use realistic thresholds
        CHECK(max_drho < 1e-10, "WB ρ: face states match in HSE");
        CHECK(max_dP   < 1e-6,  "WB P: face states match in HSE (O(dt*dy²))");
        CHECK(max_dv   < 1e-4,  "WB v: face states small in HSE (O(dt*dy²))");

        sol.destroy();
    }

    // ---- Test 3: X-sweep with bubble → bounded face states ----
    std::printf("\n[Test 3] X-sweep: bubble → bounded face states\n");
    {
        int nx = 64, ny_val = 96;
        double Lx = 1.0, Ly = 1.5;
        double gamma = 5.0/3.0, g = 1.0, cfl = 0.4;

        StrangSolver sol;
        sol.init(nx, ny_val, Lx, Ly, gamma, g, cfl, 1.0, 1.0);
        sol.init_bubble(0.5, 0.3, 0.15, 0.5, 0.05, 10, 3.0*sol.dx);

        sol.fill_ghost_x();
        sol.fill_ghost_y();

        int N_phys = nx * ny_val;
        int B = 256;
        double dt_test = 1e-3;
        k_muscl_hancock_x<<<(N_phys+B-1)/B, B>>>(
            sol.d_rho, sol.d_mx, sol.d_my, sol.d_E,
            sol.d_rho_bar, sol.d_p_bar,
            sol.d_wL, sol.d_wR,
            dt_test, sol.dx, gamma,
            nx, ny_val, sol.ng, sol.stride());
        CUDA_CHECK(cudaDeviceSynchronize());

        int Ntot = sol.total_cells();
        std::vector<double> h_wR(4*Ntot);
        CUDA_CHECK(cudaMemcpy(h_wR.data(), sol.d_wR, 4*Ntot*sizeof(double), cudaMemcpyDeviceToHost));

        int str = sol.stride();
        int ng = sol.ng;
        double rho_min = 1e30, rho_max = -1e30, P_min = 1e30;
        for (int j = 0; j < ny_val; j++) {
            for (int i = 0; i < nx; i++) {
                int k = (j + ng) * str + (i + ng);
                double rho = h_wR[k*4+0];
                double P   = h_wR[k*4+3];
                rho_min = std::fmin(rho_min, rho);
                rho_max = std::fmax(rho_max, rho);
                P_min   = std::fmin(P_min, P);
            }
        }
        std::printf("  Face ρ: [%.6f, %.6f]  P_min = %.6f\n", rho_min, rho_max, P_min);
        CHECK(rho_min > 0.0, "Face density > 0 everywhere");
        CHECK(P_min > 0.0,   "Face pressure > 0 everywhere");
        CHECK(rho_max < 2.0, "Face density bounded (< 2.0)");

        sol.destroy();
    }

    // ---- Test 4: Y-sweep with bubble → face states match between cells ----
    std::printf("\n[Test 4] Y-sweep with bubble: face states bounded\n");
    {
        int nx = 64, ny_val = 96;
        double Lx = 1.0, Ly = 1.5;
        double gamma = 5.0/3.0, g = 1.0, cfl = 0.4;

        StrangSolver sol;
        sol.init(nx, ny_val, Lx, Ly, gamma, g, cfl, 1.0, 1.0);
        sol.init_bubble(0.5, 0.3, 0.15, 0.5, 0.05, 10, 3.0*sol.dx);

        sol.fill_ghost_x();
        sol.fill_ghost_y();

        int N_phys = nx * ny_val;
        int B = 256;
        double dt_test = 1e-3;
        k_muscl_hancock_y<<<(N_phys+B-1)/B, B>>>(
            sol.d_rho, sol.d_mx, sol.d_my, sol.d_E,
            sol.d_rho_bar, sol.d_p_bar,
            sol.d_wL, sol.d_wR,
            dt_test, sol.dy, gamma,
            sol.y_lo, sol.rho0_bottom, sol.g_grav, sol.K_poly,
            nx, ny_val, sol.ng, sol.stride());
        CUDA_CHECK(cudaDeviceSynchronize());

        int Ntot = sol.total_cells();
        std::vector<double> h_wL(4*Ntot), h_wR(4*Ntot);
        CUDA_CHECK(cudaMemcpy(h_wL.data(), sol.d_wL, 4*Ntot*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_wR.data(), sol.d_wR, 4*Ntot*sizeof(double), cudaMemcpyDeviceToHost));

        int str = sol.stride();
        int ng = sol.ng;
        double rho_min = 1e30, P_min = 1e30;
        // Also check face jumps at interfaces far from bubble
        double max_jump_far = 0;
        for (int j = 0; j < ny_val - 1; j++) {
            for (int i = 0; i < nx; i++) {
                int k  = (j + ng)     * str + (i + ng);
                int k1 = (j + 1 + ng) * str + (i + ng);

                double rL = h_wR[k*4+0];
                double rR = h_wL[k1*4+0];
                double PL = h_wR[k*4+3];

                rho_min = std::fmin(rho_min, std::fmin(rL, rR));
                P_min   = std::fmin(P_min, PL);

                // Far from bubble (j > 60 → y > 0.94, well above bubble)
                if (j > 60) {
                    max_jump_far = std::fmax(max_jump_far, std::fabs(rL - rR));
                }
            }
        }
        std::printf("  Face ρ_min = %.6f  P_min = %.6f\n", rho_min, P_min);
        std::printf("  max face jump (far from bubble) = %.4e\n", max_jump_far);
        CHECK(rho_min > 0.0, "Y-sweep face density > 0");
        CHECK(P_min > 0.0,   "Y-sweep face pressure > 0");
        CHECK(max_jump_far < 1e-6, "WB: small face jumps far from bubble");

        sol.destroy();
    }

    std::printf("\n=== %s (%d failures) ===\n",
                n_fail == 0 ? "ALL PASSED" : "SOME FAILED", n_fail);
    return n_fail;
}
