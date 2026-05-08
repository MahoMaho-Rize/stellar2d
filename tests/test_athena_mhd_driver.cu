// test_athena_mhd_driver.cu
// ============================================================
// Phase B-M5 — §E1 stochastic broadband driver.
//
// Derivation: §E1 (Suzuki+25 inner-boundary driver):
//   v_x(t, j=ng) = Σ_N A_N sin(2π f_N t + φ_N)
// with amplitude normalisation Σ_N A_N²/2 = A_rms².
//
// Tests:
//   E1-T1  driver_on = false → zero effect (regression of B-M1 WB).
//   E1-T2  driver on, HSE atm, measure ⟨v_x²⟩_t on j=ng row over
//          many samples; must equal A_rms² within statistical noise.
//   E1-T3  driver on but IC v_x = 0, 1 VL2 step,
//          apply_driver(t=0) sets v_x = Σ A_N sin(φ_N) — compare with
//          host-computed waveform to machine precision.
//   E1-T4  driver deterministic with seed: two identical-seed runs
//          give bit-identical v_x history.
// ============================================================

#include "athena_mhd_solver.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <random>

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

// Read the j=ng row's v_x = m_x / ρ at time after apply_driver.
static void read_vx_row(AthenaMHDSolver& sv, std::vector<double>& out) {
    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng;
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell), h_mx(ncell);
    cudaMemcpy(h_rho.data(), sv.d_rho, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mx.data(),  sv.d_mx,  (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    out.resize(sv.nx);
    for (int ic = 0; ic < sv.nx; ++ic) {
        int c = (ic + ng) * sy + ng;
        out[ic] = h_mx[c] / std::max(h_rho[c], 1e-30);
    }
}

// --------------------------------------------------------------------
// E1-T1: driver off — B-M1 WB must still hold.
// --------------------------------------------------------------------
static void test_T1_driver_off_preserves_wb() {
    std::printf("\n[E1-T1] driver off — HSE WB unchanged\n");
    AthenaMHDSolver sv;
    sv.init(32, 32, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0;
    sv.init_hse_atmosphere(1.0, 1.0, 1.0, 0.0);
    // driver_on = false by default.
    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng, ncell = sx * sy;
    std::vector<double> h_rho0(ncell), h_E0(ncell);
    sv.fill_ghost(); sv.cons_to_prim();
    cudaMemcpy(h_rho0.data(), sv.d_rho, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E0.data(),   sv.d_E,   (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    double t = 0.0;
    for (int s = 0; s < 200; ++s) {
        double dt = sv.step(t, 1e30);
        if (!(dt > 0)) break;
        sv.apply_driver(t + dt);   // no-op since driver_on=false
        t += dt;
    }
    std::vector<double> h_rho(ncell), h_E(ncell);
    cudaMemcpy(h_rho.data(), sv.d_rho, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E.data(),   sv.d_E,   (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    double max_drho = 0.0, max_dE = 0.0;
    for (int ic = 0; ic < sv.nx; ++ic)
        for (int jc = 0; jc < sv.ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            max_drho = std::max(max_drho,
                std::fabs(h_rho[c] - h_rho0[c]) / std::max(h_rho0[c], 1e-30));
            max_dE = std::max(max_dE,
                std::fabs(h_E[c] - h_E0[c]) / std::max(h_E0[c], 1e-30));
        }
    std::printf("    200 steps, t=%.3e; max|δρ|/ρ=%.3e  max|δE|/E=%.3e\n",
                t, max_drho, max_dE);
    CHECK_LT(max_drho, 1e-10, "E1-T1a: driver off → δρ/ρ < 1e-10");
    CHECK_LT(max_dE,   1e-10, "E1-T1b: driver off → δE/E < 1e-10");
    sv.destroy();
}

// --------------------------------------------------------------------
// E1-T2: driver on, measure ⟨v_x²⟩_t on j=ng row, compare with A_rms².
//   Collect samples on a dense time grid, independently of VL2 step
//   (just pure apply_driver on frozen atm).
// --------------------------------------------------------------------
static void test_T2_power_normalisation() {
    std::printf("\n[E1-T2] ⟨v_x²⟩_t matches A_rms² (Parseval)\n");
    AthenaMHDSolver sv;
    sv.init(32, 32, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0;
    sv.init_hse_atmosphere(1.0, 1.0, 1.0, 0.0);

    const double A_rms = 0.1;
    const double f_min = 1.0, f_max = 100.0;
    const int N_modes = 32;
    sv.init_stochastic_driver(A_rms, f_min, f_max, N_modes, /*seed=*/42u);

    // Sample v_x at many t ∈ [0, T_long], T_long ≫ 1/f_min.
    const int N_samples = 2000;
    double T_long = 20.0 / f_min;          // 20 periods of slowest mode
    double dt_samp = T_long / N_samples;
    double sum_v2 = 0.0;
    int count = 0;
    std::vector<double> vx_row(sv.nx);
    for (int s = 0; s < N_samples; ++s) {
        double t = (s + 0.5) * dt_samp;
        sv.apply_driver(t);
        read_vx_row(sv, vx_row);
        // Average over x (all cells see same driver).
        for (double v : vx_row) {
            sum_v2 += v * v;
            ++count;
        }
    }
    double mean_v2 = sum_v2 / (double)count;
    double target = A_rms * A_rms;
    double rel_err = std::fabs(mean_v2 - target) / target;
    std::printf("    ⟨v_x²⟩ = %.6e  target = %.6e  rel err %.3e\n",
                mean_v2, target, rel_err);
    // Stat noise on N samples ~ 1/√N for gaussian ≈ 2%.  Relax to 10%
    // for broadband Parseval — single-mode convergence is slower.
    CHECK_LT(rel_err, 1e-1, "E1-T2: ⟨v_x²⟩ within 10% of A_rms²");
    sv.destroy();
}

// --------------------------------------------------------------------
// E1-T3: apply_driver(t=0) bitwise matches host waveform Σ A sin(φ).
// --------------------------------------------------------------------
static void test_T3_host_device_match() {
    std::printf("\n[E1-T3] apply_driver(t) matches host Σ A sin(2πft+φ)\n");
    AthenaMHDSolver sv;
    sv.init(32, 32, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0;
    sv.init_hse_atmosphere(1.0, 1.0, 1.0, 0.0);

    const double A_rms = 0.05;
    const double f_min = 1.0, f_max = 100.0;
    const int N_modes = 16;
    const unsigned seed = 12345u;
    sv.init_stochastic_driver(A_rms, f_min, f_max, N_modes, seed);

    // Regenerate mode table identically on host (same seed, same
    // log-spacing, same per-mode amp formula as in solver).
    std::vector<double> h_f(N_modes), h_amp(N_modes), h_phi(N_modes);
    double ln_ratio = std::log(f_max / f_min);
    double per_mode_amp = A_rms * std::sqrt(2.0 / (double)N_modes);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 2.0 * M_PI);
    for (int n = 0; n < N_modes; ++n) {
        double log_f = std::log(f_min) + ln_ratio * (n + 0.5) / (double)N_modes;
        h_f[n] = std::exp(log_f);
        h_amp[n] = per_mode_amp;
        h_phi[n] = uniform(rng);
    }
    double t_test = 0.123456;
    double vx_host = 0.0;
    for (int n = 0; n < N_modes; ++n) {
        vx_host += h_amp[n] * std::sin(2.0 * M_PI * h_f[n] * t_test + h_phi[n]);
    }

    sv.apply_driver(t_test);
    std::vector<double> vx_dev;
    read_vx_row(sv, vx_dev);
    double max_err = 0.0;
    for (double v : vx_dev)
        max_err = std::max(max_err, std::fabs(v - vx_host));
    std::printf("    t=%.6f  v_x host=%.6e  max |dev−host| = %.3e\n",
                t_test, vx_host, max_err);
    CHECK_LT(max_err, 1e-12, "E1-T3: device waveform matches host to ULP");
    sv.destroy();
}

// --------------------------------------------------------------------
// E1-T4: seed reproducibility — two runs with same seed agree bit-wise.
// --------------------------------------------------------------------
static void test_T4_seed_reproducibility() {
    std::printf("\n[E1-T4] seed reproducibility — identical runs match ULP\n");
    std::vector<double> series[2];
    for (int run = 0; run < 2; ++run) {
        AthenaMHDSolver sv;
        sv.init(32, 32, 1.0, 1.0, 5.0/3.0, 0.3);
        sv.xorder = 2; sv.limiter = 0;
        sv.init_hse_atmosphere(1.0, 1.0, 1.0, 0.0);
        sv.init_stochastic_driver(0.1, 1.0, 100.0, 32, /*seed=*/777u);
        std::vector<double> row;
        std::vector<double>& s = series[run];
        for (int k = 0; k < 20; ++k) {
            double t = 0.01 * k;
            sv.apply_driver(t);
            read_vx_row(sv, row);
            s.push_back(row[0]);    // first cell, all same within a row
        }
        sv.destroy();
    }
    double max_diff = 0.0;
    for (size_t k = 0; k < series[0].size(); ++k)
        max_diff = std::max(max_diff, std::fabs(series[0][k] - series[1][k]));
    std::printf("    20 samples, max |run0 − run1| = %.3e\n", max_diff);
    CHECK_LT(max_diff, 1e-14,
             "E1-T4: seed-identical runs agree bit-wise");
}

int main() {
    std::printf("=== athena_mhd stochastic driver (B-M5, §E1) ===\n");
    test_T1_driver_off_preserves_wb();
    test_T2_power_normalisation();
    test_T3_host_device_match();
    test_T4_seed_reproducibility();
    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
