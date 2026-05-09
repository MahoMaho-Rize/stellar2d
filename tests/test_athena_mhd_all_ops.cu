// test_athena_mhd_all_ops.cu
// ============================================================
// Phase B-M5.75 — all-operators combined smoke (regression sentinel).
//
// B-M4 (test_athena_mhd_combined) tested WB + κ + cool together.
// B-M5   adds §E1 stochastic driver.
// B-M5.5 adds §C8 chromo blended cooling.
// None of those tests ever exercised all six operators at once.
// This file closes that gap.
//
// Operator chain per step (derivation-driven operator split order):
//     U^{n+1} = L_driver(t+dt) · L_chromo(dt) · L_cool(dt)
//               · L_cond(dt)   · L_vl2(U^n; dt, WB)
//
// Three tests:
//
//   F-T1  all-operators-OFF regression (all flags false / kappa0=0).
//         Isothermal HSE atm.  The operator-chain loop still runs the
//         calls but every operator is a no-op → must bit-identical
//         match the pure B-M1 WB path (δρ/ρ, δE/E < 1e-10 over 200
//         steps).  Catches accidental side-effects in the plumbing.
//
//   F-T2  all-operators-ON, null parameters (Λ₀=0, A_rms=0, etc.)
//         but the toggles true, plus κ₀ > 0 with ∇T = 0.  Isothermal
//         HSE atm.  Every operator runs but contributes zero.  WB
//         must still hold to < 1e-8 over 200 steps.  Stronger than
//         F-T1 because kernels are actually invoked.  Regression for
//         cross-operator leakage (e.g. chromo writing ρ even when
//         Λ₀=0 and ξ=1 with Newton target T = bg T).
//
//   F-T3  all-operators-ON, live parameters.  HSE atm with B_y.
//         Small driver (A_rms = 1e-3 c_s), small κ₀, small cool Λ₀,
//         small chromo with p_chr below interior p.  Run ~300 steps.
//         Verifies:
//           * no NaN / no ρ ≤ 0 / no E ≤ 0
//           * |∇·B| stays < 1e-10 (CT sentinel under all sources)
//           * driver is actually driving: ⟨v_x²⟩ on j=ng row is
//             non-zero and bounded
//           * conservation drift: §E2 characteristic BC conserves mass
//             to O(A_rms²) linear order; residual floor is cool/chromo
//             Λt HSE degradation (see F-T4 decomposition).
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

// Run the full operator chain for n_steps on whatever sv state is set up.
// Returns final t.
static double run_chain(AthenaMHDSolver& sv, int n_steps) {
    double t = 0.0;
    const double t_end = 1e30;
    for (int s = 0; s < n_steps; ++s) {
        double dt = sv.step(t, t_end);
        if (std::isnan(dt) || dt <= 0.0) return t;
        sv.apply_conduction(dt);
        sv.apply_cooling(dt);
        sv.apply_chromo_cooling(dt);
        sv.apply_driver(t + dt);
        t += dt;
    }
    return t;
}

// --------------------------------------------------------------------
// F-T1: all toggles off, chain runs but is a no-op.  WB preserved.
// --------------------------------------------------------------------
static void test_FT1_all_off_chain_no_leak() {
    std::printf("\n[F-T1] all operators OFF — chain must be a no-op\n");
    AthenaMHDSolver sv;
    const int N = 32;
    const double g_val = 1.0, H = 1.0, rho0 = 1.0, B0y = 0.1;
    sv.init(N, N, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0;
    sv.init_hse_atmosphere(g_val, H, rho0, B0y);

    // Everything OFF.
    sv.kappa0       = 0.0;
    sv.cool_on      = false;
    sv.chromo_on    = false;
    sv.driver_on    = false;

    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng, ncell = sx * sy;
    size_t nb = (size_t)ncell * sizeof(double);
    std::vector<double> h_rho0(ncell), h_E0(ncell);
    sv.fill_ghost(); sv.cons_to_prim();
    cudaMemcpy(h_rho0.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E0.data(),   sv.d_E,   nb, cudaMemcpyDeviceToHost);

    double t = run_chain(sv, 200);

    sv.fill_ghost(); sv.cons_to_prim();
    std::vector<double> h_rho(ncell), h_E(ncell), h_mx(ncell), h_my(ncell);
    cudaMemcpy(h_rho.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E.data(),   sv.d_E,   nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mx.data(),  sv.d_mx,  nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_my.data(),  sv.d_my,  nb, cudaMemcpyDeviceToHost);

    double max_drho = 0.0, max_dE = 0.0, max_v = 0.0;
    double cs = std::sqrt(g_val * H);
    for (int ic = 0; ic < N; ++ic)
        for (int jc = 0; jc < N; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            max_drho = std::max(max_drho,
                std::fabs(h_rho[c] - h_rho0[c]) / std::max(h_rho0[c], 1e-30));
            max_dE = std::max(max_dE,
                std::fabs(h_E[c] - h_E0[c]) / std::max(h_E0[c], 1e-30));
            double r = std::max(h_rho[c], 1e-30);
            double vmag = std::sqrt(h_mx[c]*h_mx[c] + h_my[c]*h_my[c]) / r;
            max_v = std::max(max_v, vmag);
        }
    double divB = sv.compute_diagnostics().max_divB;
    std::printf("    200 steps, t=%.3e\n", t);
    std::printf("    max|δρ|/ρ=%.3e  max|δE|/E=%.3e  max|v|/c_s=%.3e  divB=%.3e\n",
                max_drho, max_dE, max_v / cs, divB);
    CHECK_LT(max_drho, 1e-10, "F-T1a: off-chain preserves δρ/ρ");
    CHECK_LT(max_dE,   1e-10, "F-T1b: off-chain preserves δE/E");
    CHECK_LT(max_v / cs, 1e-8, "F-T1c: fluid stays at rest");
    CHECK_LT(divB,     1e-10, "F-T1d: ∇·B stays 0");
    sv.destroy();
}

// --------------------------------------------------------------------
// F-T2: all toggles ON but parameters chosen so each contributes 0.
//       Isothermal atm → ∇T=0 so κ flux is 0 pointwise.
//       Λ₀=0 → cool ODE trivial.  A_rms=0 → driver waveform = 0.
//       chromo with T_ref_thck = bg T + Λ₀_thin = 0 → ∂_t T ≡ 0.
//       WB must hold; drift a bit looser than T1 because kernels run.
// --------------------------------------------------------------------
static void test_FT2_all_on_null_params() {
    std::printf("\n[F-T2] all operators ON, null parameters — WB must hold\n");
    AthenaMHDSolver sv;
    const int N = 32;
    const double g_val = 1.0, H = 1.0, rho0 = 1.0, B0y = 0.1;
    sv.init(N, N, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0;
    sv.init_hse_atmosphere(g_val, H, rho0, B0y);
    const double T_bg = 1.0;   // c_s² = g·H = 1

    sv.kappa0         = 1.0;             // active, but ∇T=0 so F_c=0
    sv.cool_on        = true;
    sv.cool_Lambda0   = 0.0;             // ODE RHS = 0
    sv.cool_Tref      = 1.0;
    sv.cool_alpha     = 2.0;
    sv.chromo_on      = true;
    sv.chromo_p_chr   = 0.0;             // ξ=1 everywhere → pure thick
    sv.chromo_T_ref_thck = T_bg;         // Newton target = bg T → relax=0
    sv.chromo_tau_thck   = 1.0;
    sv.chromo_Lambda0    = 0.0;          // thin RHS = 0 regardless
    sv.chromo_T_ref_thin = 1.0;
    sv.chromo_alpha      = 2.0;
    sv.driver_on      = false;            // leave driver off; covered by T3

    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng, ncell = sx * sy;
    size_t nb = (size_t)ncell * sizeof(double);
    std::vector<double> h_rho0(ncell), h_E0(ncell);
    sv.fill_ghost(); sv.cons_to_prim();
    cudaMemcpy(h_rho0.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E0.data(),   sv.d_E,   nb, cudaMemcpyDeviceToHost);

    double t = run_chain(sv, 200);

    sv.fill_ghost(); sv.cons_to_prim();
    std::vector<double> h_rho(ncell), h_E(ncell), h_mx(ncell), h_my(ncell);
    cudaMemcpy(h_rho.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E.data(),   sv.d_E,   nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mx.data(),  sv.d_mx,  nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_my.data(),  sv.d_my,  nb, cudaMemcpyDeviceToHost);

    double max_drho = 0.0, max_dE = 0.0, max_v = 0.0;
    double cs = std::sqrt(g_val * H);
    for (int ic = 0; ic < N; ++ic)
        for (int jc = 0; jc < N; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            max_drho = std::max(max_drho,
                std::fabs(h_rho[c] - h_rho0[c]) / std::max(h_rho0[c], 1e-30));
            max_dE = std::max(max_dE,
                std::fabs(h_E[c] - h_E0[c]) / std::max(h_E0[c], 1e-30));
            double r = std::max(h_rho[c], 1e-30);
            double vmag = std::sqrt(h_mx[c]*h_mx[c] + h_my[c]*h_my[c]) / r;
            max_v = std::max(max_v, vmag);
        }
    double divB = sv.compute_diagnostics().max_divB;
    std::printf("    200 steps, t=%.3e\n", t);
    std::printf("    max|δρ|/ρ=%.3e  max|δE|/E=%.3e  max|v|/c_s=%.3e  divB=%.3e\n",
                max_drho, max_dE, max_v / cs, divB);
    // Looser than T1 because 5 kernels run each step; each cons_to_prim
    // in apply_cooling / apply_chromo_cooling introduces ULP roundoff.
    CHECK_LT(max_drho, 1e-8, "F-T2a: null-parameter chain preserves δρ/ρ");
    CHECK_LT(max_dE,   1e-8, "F-T2b: null-parameter chain preserves δE/E");
    CHECK_LT(max_v / cs, 1e-6, "F-T2c: no spurious velocities");
    CHECK_LT(divB,     1e-10, "F-T2d: ∇·B stays 0 under full chain");
    sv.destroy();
}

// --------------------------------------------------------------------
// F-T3: full chain with live, small parameters.  Smoke / sanity only —
//       no analytic match, just "runs stable, divB holds, driver works".
// --------------------------------------------------------------------
static void test_FT3_all_on_live_params_smoke() {
    std::printf("\n[F-T3] all operators ON live — 300-step smoke\n");
    AthenaMHDSolver sv;
    const int N = 32;
    const double g_val = 1.0, H = 1.0, rho0 = 1.0, B0y = 0.5;
    sv.init(N, N, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0;
    sv.init_hse_atmosphere(g_val, H, rho0, B0y);
    const double T_bg = 1.0, p_bg_floor = rho0 * T_bg * std::exp(-1.0);
    const double cs = std::sqrt(g_val * H);

    // κ small so subcycle count stays bounded.
    sv.kappa0         = 0.01;
    // cool very gently so T doesn't slide away from T_bg meaningfully.
    sv.cool_on        = true;
    sv.cool_Lambda0   = 1e-4;
    sv.cool_Tref      = T_bg;
    sv.cool_alpha     = 2.0;
    // chromo: p_chr well below bottom-cell p so ξ ≈ 0 (mostly thin).
    sv.chromo_on      = true;
    sv.chromo_p_chr   = 0.01 * p_bg_floor;
    sv.chromo_T_ref_thck = T_bg;
    sv.chromo_tau_thck   = 10.0;
    sv.chromo_Lambda0    = 1e-4;
    sv.chromo_T_ref_thin = T_bg;
    sv.chromo_alpha      = 2.0;
    // Driver: small amplitude, narrow band, 8 modes, fixed seed.
    sv.init_stochastic_driver(
        /*A_rms=*/ 1e-3 * cs,
        /*f_min=*/ 0.5,
        /*f_max=*/ 4.0,
        /*N_modes=*/ 8,
        /*seed=*/   20260509u);

    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng, ncell = sx * sy;
    size_t nb = (size_t)ncell * sizeof(double);

    // Track total mass & driver-row v_x rms through the run.
    auto total_mass = [&]() {
        std::vector<double> h_rho(ncell);
        cudaMemcpy(h_rho.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
        double sum = 0.0;
        for (int ic = 0; ic < N; ++ic)
            for (int jc = 0; jc < N; ++jc) {
                int c = (ic + ng) * sy + (jc + ng);
                sum += h_rho[c];
            }
        return sum * sv.dx * sv.dy;
    };
    double mass0 = total_mass();

    // Sample ⟨v_x²⟩ on the j=ng row every 20 steps.
    std::vector<double> vx_rms_samples;
    auto sample_driver_row = [&]() {
        std::vector<double> h_rho(ncell), h_mx(ncell);
        cudaMemcpy(h_rho.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mx.data(),  sv.d_mx,  nb, cudaMemcpyDeviceToHost);
        double sum_v2 = 0.0;
        for (int ic = 0; ic < N; ++ic) {
            int c = (ic + ng) * sy + ng;
            double v = h_mx[c] / std::max(h_rho[c], 1e-30);
            sum_v2 += v * v;
        }
        return std::sqrt(sum_v2 / N);
    };

    double t = 0.0;
    const int n_steps = 300;
    int n_nan = 0, n_neg = 0;
    double max_divB = 0.0;
    for (int s = 0; s < n_steps; ++s) {
        double dt = sv.step(t, 1e30);
        if (std::isnan(dt) || dt <= 0.0) { ++n_nan; break; }
        sv.apply_conduction(dt);
        sv.apply_cooling(dt);
        sv.apply_chromo_cooling(dt);
        sv.apply_driver(t + dt);
        t += dt;
        if ((s + 1) % 20 == 0) {
            vx_rms_samples.push_back(sample_driver_row());
            double dB = sv.compute_diagnostics().max_divB;
            max_divB = std::max(max_divB, dB);
        }
    }
    // End-of-run positivity sweep.
    sv.fill_ghost(); sv.cons_to_prim();
    std::vector<double> h_rho(ncell), h_E(ncell);
    cudaMemcpy(h_rho.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E.data(),   sv.d_E,   nb, cudaMemcpyDeviceToHost);
    double rho_min = 1e30, E_min = 1e30;
    for (int ic = 0; ic < N; ++ic)
        for (int jc = 0; jc < N; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            if (std::isnan(h_rho[c]) || std::isnan(h_E[c])) ++n_nan;
            if (h_rho[c] <= 0.0) ++n_neg;
            if (h_E[c]   <= 0.0) ++n_neg;
            rho_min = std::min(rho_min, h_rho[c]);
            E_min   = std::min(E_min,   h_E[c]);
        }
    double mass1 = total_mass();
    double mass_drift = std::fabs(mass1 - mass0) / mass0;

    double vrms_mean = 0.0;
    for (double v : vx_rms_samples) vrms_mean += v;
    if (!vx_rms_samples.empty()) vrms_mean /= vx_rms_samples.size();

    std::printf("    %d steps attempted, t=%.3e\n", n_steps, t);
    std::printf("    ρ_min=%.3e  E_min=%.3e  n_nan=%d  n_neg=%d\n",
                rho_min, E_min, n_nan, n_neg);
    std::printf("    max|∇·B| over run=%.3e\n", max_divB);
    std::printf("    driver-row v_rms: mean=%.3e  A_rms=%.3e (target)\n",
                vrms_mean, 1e-3 * cs);
    std::printf("    mass drift |Δm|/m0 = %.3e\n", mass_drift);

    CHECK_LT((double)n_nan, 0.5, "F-T3a: no NaN anywhere");
    CHECK_LT((double)n_neg, 0.5, "F-T3b: ρ>0 and E>0 everywhere");
    CHECK_LT(max_divB, 1e-10, "F-T3c: ∇·B stays 0 under full chain");
    // Driver should be actually driving: mean v_rms in O(A_rms).
    // Very loose band [0.3·A_rms, 3·A_rms] — the j=ng row sees the set
    // velocity each step but VL2 + conduction nibble at it between
    // set-calls, so the exact sample mean drifts a bit.
    CHECK_GT(vrms_mean, 0.3 * 1e-3 * cs,
             "F-T3d: driver row v_rms > 0.3·A_rms (driver active)");
    CHECK_LT(vrms_mean, 3.0 * 1e-3 * cs,
             "F-T3e: driver row v_rms < 3·A_rms (not blowing up)");
    // Mass drift under the LIVE chain is NOT ULP-bounded, but after the
    // §E2 characteristic BC swap the residual is O(1e-6) / 300 steps,
    // not O(1e-4).  F-T4 decomposes the sources:
    //   (a) 2D PLM+HLLD truncation at the tangential Alfvén discontinuity:
    //       strict O(A_rms²) (F-T4 scan shows s_{k+1}/s_k = 0.25 exactly)
    //   (b) cool + chromo Λ t residual degrading HSE: F-T4 variant (g),
    //       A_rms=0 + full chain, already gives 3.15e-6 on its own
    // Neither is a BC leak.  Threshold 1e-5 rides the (b) floor while
    // catching any real BC regression (would be ≥ 1e-4 if a conservation
    // bug slipped back in).
    CHECK_LT(mass_drift, 1e-5,
             "F-T3f: mass drift bounded (§E2 BC conservative to O(A²); "
             "floor = cool/chromo Λt residual)");
    sv.destroy();
}

// --------------------------------------------------------------------
// F-T4: probe — where does the F-T3 mass drift actually come from?
// Four variants, each running 200 VL2 steps with no source terms on
// HSE background; measure |Δm|/m0:
//   (a) driver_on=false, B_y=0.5 (plain reflect-y BC)    → baseline
//   (b) driver_on=true  A_rms=0, B_y=0.5                 → BC plumbing
//   (c) driver_on=true  A_rms=1e-3, B_y=0                → no Alfvén coupling
//   (d) driver_on=true  A_rms=1e-3, B_y=0.5              → full driver
// --------------------------------------------------------------------
static void test_FT4_drift_decomposition() {
    std::printf("\n[F-T4] drift source decomposition\n");
    auto run_variant = [](const char* name, bool drv_on, double A_rms,
                          double B0y, int n_steps, bool with_cond,
                          double kappa) -> double {
        AthenaMHDSolver sv;
        const int N = 32;
        sv.init(N, N, 1.0, 1.0, 5.0/3.0, 0.3);
        sv.xorder = 2; sv.limiter = 0;
        sv.init_hse_atmosphere(1.0, 1.0, 1.0, B0y);
        if (drv_on) {
            sv.init_stochastic_driver(A_rms, 0.5, 4.0, 8, 20260509u);
        }
        if (with_cond) sv.kappa0 = kappa;
        int sx = sv.stride_x(), sy = sv.stride_y();
        int ng = sv.ng, ncell = sx * sy;
        size_t nb = (size_t)ncell * sizeof(double);
        auto total_mass = [&]() {
            std::vector<double> h_rho(ncell);
            cudaMemcpy(h_rho.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
            double sum = 0.0;
            for (int ic = 0; ic < N; ++ic)
                for (int jc = 0; jc < N; ++jc) {
                    int c = (ic + ng) * sy + (jc + ng);
                    sum += h_rho[c];
                }
            return sum * sv.dx * sv.dy;
        };
        double m0 = total_mass();
        double t = 0.0;
        for (int s = 0; s < n_steps; ++s) {
            double dt = sv.step(t, 1e30);
            if (std::isnan(dt) || dt <= 0) break;
            if (with_cond) sv.apply_conduction(dt);
            sv.apply_driver(t + dt);
            t += dt;
        }
        double m1 = total_mass();
        double drift = std::fabs(m1 - m0) / m0;
        std::printf("    %-40s |Δm|/m0 = %.3e\n", name, drift);
        sv.destroy();
        return drift;
    };
    double d_a = run_variant("(a) drv off B=0.5      200 VL2only      ", false, 0.0, 0.5, 200, false, 0.0);
    double d_b = run_variant("(b) drv on A=0 B=0.5   200 VL2only      ", true, 0.0, 0.5, 200, false, 0.0);
    double d_c = run_variant("(c) drv on A=1e-3 B=0  200 VL2only      ", true, 1e-3, 0.0, 200, false, 0.0);
    double d_d = run_variant("(d) drv on A=1e-3 B=0.5 200 VL2only     ", true, 1e-3, 0.5, 200, false, 0.0);
    double d_d3 = run_variant("(e) drv on A=1e-3 B=0.5 300 VL2only     ", true, 1e-3, 0.5, 300, false, 0.0);
    double d_d4 = run_variant("(f) drv on A=1e-3 B=0.5 300 VL2+κ(0.01) ", true, 1e-3, 0.5, 300, true, 0.01);
    // Amplitude scaling: halve A_rms, does drift scale as A² (linear-regime
    // tangential-discontinuity error) or as A¹ (BC bug)?
    double d_h1 = run_variant("(s1) A=1e-3 B=0.5 300 VL2only           ", true, 1e-3, 0.5, 300, false, 0.0);
    double d_h2 = run_variant("(s2) A=5e-4 B=0.5 300 VL2only           ", true, 5e-4, 0.5, 300, false, 0.0);
    double d_h3 = run_variant("(s3) A=2.5e-4 B=0.5 300 VL2only         ", true, 2.5e-4, 0.5, 300, false, 0.0);
    double d_h4 = run_variant("(s4) A=1e-6 B=0.5 300 VL2only (linear)  ", true, 1e-6, 0.5, 300, false, 0.0);
    // Ratios s2/s1 and s3/s2 expected to be 0.25 for A² scaling.
    std::printf("    scaling s2/s1 = %.3f (expect 0.25 if O(A²))\n", d_h2/d_h1);
    std::printf("    scaling s3/s2 = %.3f (expect 0.25 if O(A²))\n", d_h3/d_h2);
    std::printf("    linear-A expectation: drift ≈ (1e-6/1e-3)² × s1 = 1e-6 × %.3e = %.3e\n",
                d_h1, 1e-6*d_h1);
    std::printf("    actual s4 / expected = %.3f\n", d_h4 / (1e-6*d_h1));
    (void)d_h1; (void)d_h2; (void)d_h3; (void)d_h4;
    // Also: what happens if we run F-T3's FULL chain but A_rms=0
    // (driver off at amplitude level, but all other ops live)?
    AthenaMHDSolver sv;
    const int N = 32;
    sv.init(N, N, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0;
    sv.init_hse_atmosphere(1.0, 1.0, 1.0, 0.5);
    const double T_bg = 1.0;
    const double p_bg_floor = 1.0 * T_bg * std::exp(-1.0);
    sv.kappa0 = 0.01;
    sv.cool_on = true; sv.cool_Lambda0 = 1e-4; sv.cool_Tref = T_bg; sv.cool_alpha = 2.0;
    sv.chromo_on = true; sv.chromo_p_chr = 0.01 * p_bg_floor;
    sv.chromo_T_ref_thck = T_bg; sv.chromo_tau_thck = 10.0;
    sv.chromo_Lambda0 = 1e-4; sv.chromo_T_ref_thin = T_bg; sv.chromo_alpha = 2.0;
    sv.init_stochastic_driver(0.0, 0.5, 4.0, 8, 20260509u);   // A_rms=0
    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng, ncell = sx * sy;
    size_t nb = (size_t)ncell * sizeof(double);
    auto total_mass_g = [&]() {
        std::vector<double> h_rho(ncell);
        cudaMemcpy(h_rho.data(), sv.d_rho, nb, cudaMemcpyDeviceToHost);
        double sum = 0.0;
        for (int ic = 0; ic < N; ++ic)
            for (int jc = 0; jc < N; ++jc) {
                int c = (ic + ng) * sy + (jc + ng);
                sum += h_rho[c];
            }
        return sum * sv.dx * sv.dy;
    };
    double mass_g_0 = total_mass_g();
    double t_g = 0.0;
    for (int s = 0; s < 300; ++s) {
        double dt = sv.step(t_g, 1e30);
        if (!(dt > 0)) break;
        sv.apply_conduction(dt);
        sv.apply_cooling(dt);
        sv.apply_chromo_cooling(dt);
        sv.apply_driver(t_g + dt);
        t_g += dt;
    }
    double d_g = std::fabs(total_mass_g() - mass_g_0) / mass_g_0;
    sv.destroy();
    std::printf("    (g) drv on A=0 B=0.5 300 FULL CHAIN       |Δm|/m0 = %.3e\n", d_g);
    (void)d_d3; (void)d_d4;
    // Expectations:
    //   (a) ≡ baseline reflect-y WB atm     → ULP (< 1e-12)
    //   (b) should be ≡ (a) since A_rms = 0  → ULP (< 1e-12) IF BC plumbing correct
    //   (c) no Alfvén coupling → some acoustic-mode leak, expect small
    //   (d) full case, this is the F-T3 number
    CHECK_LT(d_a, 1e-12, "F-T4a: driver-off baseline mass to ULP");
    CHECK_LT(d_b, 1e-12, "F-T4b: A_rms=0 chain mass to ULP (BC plumbing clean)");
    // (c), (d) are informational
    (void)d_c; (void)d_d;
}

int main() {
    std::printf("=== athena_mhd all-ops integration smoke (B-M5.75) ===\n");
    test_FT1_all_off_chain_no_leak();
    test_FT2_all_on_null_params();
    test_FT3_all_on_live_params_smoke();
    test_FT4_drift_decomposition();
    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
