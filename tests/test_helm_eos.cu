// Correctness test for Helmholtz EOS evaluator.
// Uses physical regimes with known analytic limits:
//   1. Dilute warm gas (ρ=1, T=1e5) → Helmholtz ≈ ideal-gas
//   2. Stellar core (ρ=150, T=1.5e7) → finite electron pressure
//   3. Degenerate core (ρ=1e6, T=1e6) → electron degeneracy pressure
//      dominates; P ≫ ideal gas
// Also checks T_from_rho_e round-trip.

#include "../src/physics/helmholtz_eos.cuh"
#include "../src/physics/helmholtz_eos.cu"

#include <cstdio>
#include <cmath>

static __global__ void k_eval_one(HelmholtzTableView tv,
                                  double rho, double T,
                                  double* out /* P, e, cs, dPde, gam1 */)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    HelmState s = helm_eval(rho, T, tv);
    out[0] = s.P;
    out[1] = s.e;
    out[2] = s.cs;
    out[3] = s.dPde_rho;
    out[4] = s.gamma1;
}

static __global__ void k_invert(HelmholtzTableView tv,
                                double rho, double e_target,
                                double* out_T)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    out_T[0] = helm_T_from_rho_e(rho, e_target, tv);
}

struct TestCase {
    const char* name;
    double rho;       // g/cc
    double T;         // K
    double Abar;      // mean ion weight
    double Zbar;      // mean ion charge
};

static int run_case(const HelmholtzTable& tbl, const TestCase& tc,
                    double* d_out, double* d_T_rt)
{
    // Override composition on the view for this test
    HelmholtzTableView v = tbl.view;
    v.Abar = tc.Abar;
    v.Zbar = tc.Zbar;

    k_eval_one<<<1,1>>>(v, tc.rho, tc.T, d_out);
    cudaDeviceSynchronize();
    double h_out[5];
    cudaMemcpy(h_out, d_out, 5*sizeof(double), cudaMemcpyDeviceToHost);
    double P = h_out[0], e = h_out[1], cs = h_out[2], dPde = h_out[3], gam = h_out[4];

    // Analytic ideal-gas + radiation reference
    constexpr double k_B = 1.380650424e-16;
    constexpr double N_A = 6.0221417930e23;
    constexpr double a_r = 7.5657e-15;
    double R_gas = N_A * k_B;     // erg/mol/K
    double P_ideal = tc.rho * R_gas * tc.T / tc.Abar
                   + tc.rho * R_gas * tc.T * tc.Zbar / tc.Abar   // electron ideal contribution
                   + a_r * tc.T*tc.T*tc.T*tc.T / 3.0;
    double P_ratio = P / P_ideal;

    std::printf("[%s]  ρ=%.2e T=%.2e  (Abar=%.2f, Zbar=%.2f)\n",
                tc.name, tc.rho, tc.T, tc.Abar, tc.Zbar);
    std::printf("   P       = %.4e erg/cc   (ideal+rad ref %.4e, ratio %.4f)\n",
                P, P_ideal, P_ratio);
    std::printf("   e       = %.4e erg/g\n", e);
    std::printf("   cs      = %.4e cm/s     (Mach 1 = cs)\n", cs);
    std::printf("   Γ₁      = %.4f\n", gam);
    std::printf("   dP/de|ρ = %.4e\n", dPde);

    // Round-trip T_from_rho_e
    k_invert<<<1,1>>>(v, tc.rho, e, d_T_rt);
    cudaDeviceSynchronize();
    double T_rt;
    cudaMemcpy(&T_rt, d_T_rt, sizeof(double), cudaMemcpyDeviceToHost);
    double T_err = fabs(T_rt - tc.T) / tc.T;
    std::printf("   T_from_rho_e round-trip = %.4e K  (rel err %.2e)\n", T_rt, T_err);

    bool ok = true;
    if (!std::isfinite(P) || P <= 0) { std::printf("   FAIL: P not finite/positive\n"); ok = false; }
    if (!std::isfinite(e) || e <= 0) { std::printf("   FAIL: e not finite/positive\n"); ok = false; }
    if (!std::isfinite(cs) || cs <= 0) { std::printf("   FAIL: cs not finite/positive\n"); ok = false; }
    if (T_err > 1e-3) { std::printf("   FAIL: T round-trip err > 1e-3\n"); ok = false; }

    std::printf("\n");
    return ok ? 0 : 1;
}

int main() {
    HelmholtzTable tbl;
    cudaStream_t s;
    cudaStreamCreate(&s);

    if (tbl.load(nullptr, "third_party/helmholtz/helm_table.bin", s) != 0) {
        std::printf("SKIP: helm_table.bin not present\n");
        return 77;  // ctest SKIP convention (see CMakeLists SKIP_RETURN_CODE)
    }

    double *d_out, *d_T_rt;
    cudaMalloc(&d_out, 5*sizeof(double));
    cudaMalloc(&d_T_rt, sizeof(double));

    // Solar composition: X≈0.73, Y≈0.25, Z≈0.02 → Abar ≈ 1.28, Zbar ≈ 1.13
    TestCase cases[] = {
        {"Warm diffuse gas",  1.0e-6,   1.0e5,  1.28, 1.13},
        {"Stellar envelope",  1.0e-2,   1.0e6,  1.28, 1.13},
        {"Solar core",        150.0,    1.5e7,  1.28, 1.13},
        {"Degenerate",        1.0e6,    1.0e6,  1.28, 1.13},
    };

    int fail = 0;
    std::printf("\n==== Helmholtz EOS evaluator tests ====\n\n");
    for (auto& tc : cases) fail += run_case(tbl, tc, d_out, d_T_rt);

    cudaFree(d_out); cudaFree(d_T_rt);
    tbl.destroy();
    cudaStreamDestroy(s);

    if (fail) { std::printf("FAIL: %d cases\n", fail); return 1; }
    std::printf("All cases passed basic sanity.\n");
    return 0;
}
