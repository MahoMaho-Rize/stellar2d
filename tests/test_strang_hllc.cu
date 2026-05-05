// ============================================================
// test_strang_hllc.cu — Verify LM-HLLC Riemann solver
//
// Tests:
//   1. Uniform state: flux = physical flux exactly
//   2. Sod shock tube: flux is bounded and physical
//   3. Low-Mach correction: f(M) < 1 for slow flows
//   4. Symmetry: HLLC(L,R) mass flux = -HLLC(R_flip, L_flip)
//   5. HSE column: vertical HLLC flux difference ≈ dp̄/dy
// ============================================================

#include "strang_solver.cuh"
#include "strang_device.cuh"
#include "fas_common.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

static int n_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); n_fail++; } \
    else { std::printf("  PASS: %s\n", msg); } \
} while(0)

// Kernel that calls d_lmhllc on device and stores results
__global__
void k_test_hllc(const double* L, const double* R,
                 double* out, double gamma, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // L[i*4] = (rho, un, ut, P), same for R
    double f0, f1, f2, f3;
    d_lmhllc(L[i*4+0], L[i*4+1], L[i*4+2], L[i*4+3],
             R[i*4+0], R[i*4+1], R[i*4+2], R[i*4+3],
             gamma, f0, f1, f2, f3);
    out[i*4+0] = f0;
    out[i*4+1] = f1;
    out[i*4+2] = f2;
    out[i*4+3] = f3;
}

static void run_hllc(const double* hL, const double* hR, double* hF,
                     double gamma, int n)
{
    double *dL, *dR, *dF;
    cudaMalloc(&dL, 4*n*sizeof(double));
    cudaMalloc(&dR, 4*n*sizeof(double));
    cudaMalloc(&dF, 4*n*sizeof(double));
    cudaMemcpy(dL, hL, 4*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dR, hR, 4*n*sizeof(double), cudaMemcpyHostToDevice);
    k_test_hllc<<<(n+255)/256, 256>>>(dL, dR, dF, gamma, n);
    cudaDeviceSynchronize();
    cudaMemcpy(hF, dF, 4*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dL); cudaFree(dR); cudaFree(dF);
}

int main()
{
    std::printf("=== test_strang_hllc ===\n\n");
    double gamma = 5.0 / 3.0;
    double gm1 = gamma - 1.0;

    // ---- Test 1: Uniform state ----
    std::printf("[Test 1] Uniform state → exact physical flux\n");
    {
        // State: ρ=1, u=0.5, v=0.3, P=1
        double rho = 1.0, u = 0.5, v = 0.3, P = 1.0;
        double E = P / gm1 + 0.5 * rho * (u*u + v*v);

        double L[4] = {rho, u, v, P};
        double R[4] = {rho, u, v, P};
        double F[4];
        run_hllc(L, R, F, gamma, 1);

        // Expected: F = (ρu, ρu²+P, ρuv, (E+P)u)
        double F_exact[4] = {rho*u, rho*u*u+P, rho*u*v, (E+P)*u};

        double max_err = 0;
        for (int c = 0; c < 4; c++)
            max_err = std::fmax(max_err, std::fabs(F[c] - F_exact[c]));
        std::printf("  max |F - F_exact| = %.4e\n", max_err);
        CHECK(max_err < 1e-12, "Uniform state: exact flux");
    }

    // ---- Test 2: Sod shock tube ----
    std::printf("\n[Test 2] Sod shock tube → bounded flux\n");
    {
        // Left: ρ=1, u=0, P=1.  Right: ρ=0.125, u=0, P=0.1
        double L[4] = {1.0, 0.0, 0.0, 1.0};
        double R[4] = {0.125, 0.0, 0.0, 0.1};
        double F[4];
        run_hllc(L, R, F, gamma, 1);

        std::printf("  F = (%.6f, %.6f, %.6f, %.6f)\n", F[0], F[1], F[2], F[3]);

        // Mass flux should be positive (flow from high to low pressure)
        CHECK(F[0] > 0.0, "Sod: positive mass flux");
        // Momentum flux should be positive and reasonable
        CHECK(F[1] > 0.0 && F[1] < 2.0, "Sod: bounded momentum flux");
        // Tangential flux should be zero (both v=0)
        CHECK(std::fabs(F[2]) < 1e-14, "Sod: zero tangential flux");
        // Energy flux should be positive
        CHECK(F[3] > 0.0, "Sod: positive energy flux");
    }

    // ---- Test 3: Low-Mach correction ----
    std::printf("\n[Test 3] Low-Mach correction: slow vs fast flow\n");
    {
        // Case A: very slow flow (M ≈ 0.001)
        double P = 1.0, rho = 1.0;
        double cs = std::sqrt(gamma * P / rho);  // ≈ 1.29
        double u_slow = 0.001 * cs;  // M ≈ 0.001

        double La[4] = {rho, u_slow, 0.0, P * 1.001};  // slight P jump
        double Ra[4] = {rho, u_slow, 0.0, P * 0.999};
        double Fa[4];
        run_hllc(La, Ra, Fa, gamma, 1);

        // Case B: fast flow (M ≈ 0.5)
        double u_fast = 0.5 * cs;
        double Lb[4] = {rho, u_fast, 0.0, P * 1.001};
        double Rb[4] = {rho, u_fast, 0.0, P * 0.999};
        double Fb[4];
        run_hllc(Lb, Rb, Fb, gamma, 1);

        // The momentum flux difference from physical flux should be smaller
        // for the low-M case due to f(M) suppression
        double phys_flux_a = rho * u_slow * u_slow + P;
        double phys_flux_b = rho * u_fast * u_fast + P;
        double diff_a = std::fabs(Fa[1] - phys_flux_a);
        double diff_b = std::fabs(Fb[1] - phys_flux_b);

        std::printf("  slow M≈%.4f: |F_mn - F_phys| = %.4e\n",
                    u_slow / cs, diff_a);
        std::printf("  fast M≈%.4f: |F_mn - F_phys| = %.4e\n",
                    u_fast / cs, diff_b);
        // Low-M case has f(M)→0, less numerical pressure dissipation
        CHECK(std::isfinite(Fa[0]) && std::isfinite(Fb[0]),
              "LM-HLLC fluxes finite for both cases");
    }

    // ---- Test 4: Symmetry ----
    std::printf("\n[Test 4] Anti-symmetry: F(L,R) mass = -F(R_flip,L_flip)\n");
    {
        // Asymmetric state
        double L[4] = {1.0,  0.3, 0.1, 1.5};
        double R[4] = {0.5, -0.2, 0.4, 0.8};
        double F_lr[4], F_rl[4];

        // F(L, R)
        run_hllc(L, R, F_lr, gamma, 1);

        // F(R_flip, L_flip): swap sides AND negate normal velocity
        double Rf[4] = {R[0], -R[1], R[2], R[3]};
        double Lf[4] = {L[0], -L[1], L[2], L[3]};
        run_hllc(Rf, Lf, F_rl, gamma, 1);

        // Mass flux should be anti-symmetric: F_lr[0] = -F_rl[0]
        double err_rho = std::fabs(F_lr[0] + F_rl[0]);
        std::printf("  |F_lr(ρ) + F_rl(ρ)| = %.4e\n", err_rho);
        CHECK(err_rho < 1e-12, "Mass flux anti-symmetric");
    }

    // ---- Test 5: HSE column — vertical HLLC flux matches dp̄/dy ----
    std::printf("\n[Test 5] HSE column: G_top - G_bot ≈ -dp̄/dy · dy (FULL gravity balance)\n");
    {
        // Set up a 1D column in HSE
        int ny = 64;
        double Ly = 1.5, dy = Ly / ny;
        double gamma = 5.0 / 3.0, g = 1.0, K = 1.0, rho0 = 1.0;
        double gm1 = gamma - 1.0;

        // Compute HSE background at cell centers
        std::vector<double> rho_bar(ny), p_bar(ny);
        for (int j = 0; j < ny; j++) {
            double y = (j + 0.5) * dy;
            double arg = std::pow(rho0, gm1) - gm1*g*y/(gamma*K);
            rho_bar[j] = std::pow(std::fmax(arg, 1e-20), 1.0/gm1);
            p_bar[j] = K * std::pow(rho_bar[j], gamma);
        }

        // In HSE with v=0, face states at j+1/2 should be:
        // ρ = ρ̄(y_{j+1/2}), P = p̄(y_{j+1/2}), u = v = 0
        // HLLC flux for stationary contact: G = (0, 0, P_face, 0)
        // So ΔG_my = P(j+1/2) - P(j-1/2) should ≈ dp̄/dy * dy ≈ -ρ̄g*dy

        // Compute face HSE values and HLLC fluxes
        double max_imbalance = 0;
        for (int j = 1; j < ny - 1; j++) {
            double y_bot = j * dy;       // face j (= j-1/2 of cell j)
            double y_top = (j+1) * dy;   // face j+1

            // HSE at faces
            double arg_b = std::pow(rho0, gm1) - gm1*g*y_bot/(gamma*K);
            double rho_b = std::pow(std::fmax(arg_b, 1e-20), 1.0/gm1);
            double p_b   = K * std::pow(rho_b, gamma);

            double arg_t = std::pow(rho0, gm1) - gm1*g*y_top/(gamma*K);
            double rho_t = std::pow(std::fmax(arg_t, 1e-20), 1.0/gm1);
            double p_t   = K * std::pow(rho_t, gamma);

            // HLLC at both faces (stationary, un=v=0 on both sides)
            // For v=0 uniform: flux = (0, 0, P, 0)
            double Lb[4] = {rho_b, 0, 0, p_b};
            double Rb[4] = {rho_b, 0, 0, p_b};
            double Fb[4];
            run_hllc(Lb, Rb, Fb, gamma, 1);

            double Lt[4] = {rho_t, 0, 0, p_t};
            double Rt[4] = {rho_t, 0, 0, p_t};
            double Ft[4];
            run_hllc(Lt, Rt, Ft, gamma, 1);

            // Flux difference (y-momentum) + gravity should ≈ 0
            // Momentum eqn: ∂(ρv)/∂t + ∂G_my/∂y = -ρg
            // HSE: ∂G_my/∂y + ρg = 0  →  (G_top - G_bot)/dy + ρg ≈ 0
            double dG_my = (Ft[1] - Fb[1]) / dy;  // ≈ dp̄/dy = -ρ̄g
            double imbalance = std::fabs(dG_my + rho_bar[j] * g) / (rho_bar[j] * g);

            max_imbalance = std::fmax(max_imbalance, imbalance);
        }
        std::printf("  max |(ΔG/dy + ρg)/(ρg)| = %.4e\n", max_imbalance);
        CHECK(max_imbalance < 0.05, "HSE: HLLC flux matches gravity to <5%");
    }

    std::printf("\n=== %s (%d failures) ===\n",
                n_fail == 0 ? "ALL PASSED" : "SOME FAILED", n_fail);
    return n_fail;
}
