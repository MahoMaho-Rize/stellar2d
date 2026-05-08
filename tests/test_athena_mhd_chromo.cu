// test_athena_mhd_chromo.cu
// ============================================================
// Phase B-M5.5 — §C8 chromospheric blended cooling.
//
// Blend:  Q_R = ξ·Q_thck + (1-ξ)·Q_thin,  ξ = max(0, 1 - p_chr/p)
//   thick: Newton relaxation  dT/dt = -(T - T_ref_thck)/τ_thck
//   thin:  Townsend §C7 power law  dT/dt = -C T^α,
//          C = (γ-1) ρ Λ₀ / T_ref_thin^α
// Operator-split per cell (O(dt²) splitting error):
//   T_a = T_ref_thck + (T0 - T_ref_thck)·exp(-ξ·dt/τ_thck)
//   T_b = Townsend-closed-form on T_a with C scaled by (1-ξ)
//
// Four tests:
//   E1-T1  pure thin limit   (ξ=0)   — match §C7 Townsend analytic
//   E1-T2  pure thick limit  (ξ=1)   — match exponential relaxation
//   E1-T3  blend intermediate (ξ=0.5) — match host-side split reference
//   E1-T4  ρ / mom / B_f invariant    — passive with respect to all else
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

// ------------------------------------------------------------
// Seed uniform (ρ, T, B) periodic domain, v = 0.  Copied from
// test_athena_mhd_cooling.cu.
// ------------------------------------------------------------
static void seed_uniform(AthenaMHDSolver& sv, int N,
                         double rho0, double T0,
                         double Bx = 0.0, double By = 0.0) {
    sv.init(N, N, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0; sv.cfl_limit = 0.5;
    sv.x_bc = 0; sv.y_bc = 0;
    sv.x_lo = 0.0; sv.x_hi = 1.0;
    sv.y_lo = 0.0; sv.y_hi = 1.0;
    sv.dx = 1.0 / (double)N;
    sv.dy = 1.0 / (double)N;

    int sx = sv.stride_x(), sy = sv.stride_y();
    int ncell = sx * sy;
    int nfx = sv.total_fx(), nfy = sv.total_fy();
    double P0 = rho0 * T0;
    double gm1 = sv.gamma - 1.0;
    double me = 0.5 * (Bx * Bx + By * By);

    std::vector<double> h_rho(ncell, rho0);
    std::vector<double> h_mx(ncell, 0.0), h_my(ncell, 0.0),
                        h_mz(ncell, 0.0), h_Bz(ncell, 0.0);
    std::vector<double> h_E(ncell, P0 / gm1 + me);
    std::vector<double> h_Bxf(nfx, Bx), h_Byf(nfy, By);
    size_t nb = (size_t)ncell * sizeof(double);
    cudaMemcpy(sv.d_rho, h_rho.data(), nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_mx,  h_mx.data(),  nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_my,  h_my.data(),  nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_mz,  h_mz.data(),  nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_E,   h_E.data(),   nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Bz_cc, h_Bz.data(), nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
               cudaMemcpyHostToDevice);
}

// Read cell (ng, ng) primitive T.
static double read_T_interior(AthenaMHDSolver& sv) {
    sv.fill_ghost();
    sv.cons_to_prim();
    int sx = sv.stride_x(), sy = sv.stride_y();
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell), h_P(ncell);
    cudaMemcpy(h_rho.data(), sv.d_w_rho, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P.data(), sv.d_w_P, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    int c = sv.ng * sy + sv.ng;
    return h_P[c] / std::max(h_rho[c], 1e-30);
}

// Townsend-analytic closed form (single segment).
static double townsend_analytic(double T0, double C, double alpha,
                                double t, double Tfloor = 1e-6) {
    if (std::fabs(1.0 - alpha) < 1e-12) {
        double T = T0 * std::exp(-C * t);
        return (T < Tfloor) ? Tfloor : T;
    }
    double one_minus_a = 1.0 - alpha;
    double base = std::pow(T0, one_minus_a) - C * one_minus_a * t;
    if (base <= 0.0) return Tfloor;
    double T = std::pow(base, 1.0 / one_minus_a);
    return (T < Tfloor) ? Tfloor : T;
}

// Host-side reference for the full §C8 blend (matches kernel exactly).
static double chromo_ref(double T0, double rho, double p, double dt,
                         double gm1, double p_chr,
                         double T_ref_thck, double tau_thck,
                         double Lambda0, double T_ref_thin, double alpha,
                         double Tfloor) {
    double xi;
    if (p <= p_chr) xi = 0.0; else xi = 1.0 - p_chr / p;
    double T_a;
    if (tau_thck > 0.0)
        T_a = T_ref_thck + (T0 - T_ref_thck) * std::exp(-xi * dt / tau_thck);
    else
        T_a = T0;
    double one_minus_xi = 1.0 - xi;
    double T_b;
    if (one_minus_xi > 0.0) {
        double C = one_minus_xi * gm1 * rho * Lambda0
                   * std::pow(T_ref_thin, -alpha);
        double one_minus_a = 1.0 - alpha;
        if (std::fabs(one_minus_a) < 1e-12) {
            T_b = T_a * std::exp(-C * dt);
        } else {
            double base = std::pow(T_a, one_minus_a) - C * one_minus_a * dt;
            if (base <= 0.0) T_b = Tfloor;
            else             T_b = std::pow(base, 1.0 / one_minus_a);
        }
    } else {
        T_b = T_a;
    }
    if (T_b < Tfloor) T_b = Tfloor;
    if (T_b > T0)     T_b = T0;
    return T_b;
}

// --------------------------------------------------------------------
// E1-T1  pure thin limit (ξ ≡ 0).  Reduces to §C7 Townsend closed form.
// --------------------------------------------------------------------
static void test_T1_pure_thin_limit() {
    std::printf("\n[E1-T1] pure thin limit ξ=0 → matches §C7 Townsend\n");
    const int N = 16;
    const double rho0 = 1.0, T0 = 1.0;
    const double alpha = 2.0;
    const double Lambda0 = 1.0, T_ref_thin = 1.0;
    // p_chr >> p = 1 forces ξ = 0 everywhere.
    AthenaMHDSolver sv;
    seed_uniform(sv, N, rho0, T0);
    sv.chromo_on         = true;
    sv.chromo_p_chr      = 1.0e10;
    sv.chromo_T_ref_thck = 0.0;
    sv.chromo_tau_thck   = 1.0;   // irrelevant when ξ=0
    sv.chromo_Lambda0    = Lambda0;
    sv.chromo_T_ref_thin = T_ref_thin;
    sv.chromo_alpha      = alpha;
    sv.chromo_Tfloor     = 1e-6;

    double gm1 = sv.gamma - 1.0;
    double C = gm1 * rho0 * Lambda0 * std::pow(T_ref_thin, -alpha);
    // aim for ~30% cooling
    double t_end = (1.0 - std::pow(0.7, 1.0 - alpha))
                   * std::pow(T0, 1.0 - alpha) / (C * (1.0 - alpha));
    if (t_end < 0.0) t_end = -t_end;
    sv.apply_chromo_cooling(t_end);
    double T_num = read_T_interior(sv);
    double T_an  = townsend_analytic(T0, C, alpha, t_end);
    double rel_err = std::fabs(T_num - T_an) / std::max(T_an, 1e-30);
    std::printf("    t=%.4g  T_num=%.12e  T_an=%.12e  rel=%.3e\n",
                t_end, T_num, T_an, rel_err);
    CHECK_LT(rel_err, 1e-14, "E1-T1: pure-thin limit matches §C7 to ULP");
    sv.destroy();
}

// --------------------------------------------------------------------
// E1-T2  pure thick limit (ξ ≈ 1).  Newton exponential relaxation.
// --------------------------------------------------------------------
static void test_T2_pure_thick_limit() {
    std::printf("\n[E1-T2] pure thick limit ξ≈1 → exponential relaxation\n");
    const int N = 16;
    const double rho0 = 1.0, T0 = 1.0;
    const double T_ref_thck = 0.5, tau_thck = 1.0;
    // p_chr << p → ξ = 1 − 1e-10 ≈ 1
    AthenaMHDSolver sv;
    seed_uniform(sv, N, rho0, T0);
    sv.chromo_on         = true;
    sv.chromo_p_chr      = 1.0e-10;
    sv.chromo_T_ref_thck = T_ref_thck;
    sv.chromo_tau_thck   = tau_thck;
    sv.chromo_Lambda0    = 0.0;    // zero out thin contribution entirely
    sv.chromo_T_ref_thin = 1.0;
    sv.chromo_alpha      = 2.0;
    sv.chromo_Tfloor     = 1e-6;

    double t_end = 0.5;    // half τ → partial relaxation
    sv.apply_chromo_cooling(t_end);
    double T_num = read_T_interior(sv);
    // With p0 = rho0*T0 = 1, p_chr = 1e-10 → ξ = 1 − 1e-10.
    double xi = 1.0 - 1e-10 / (rho0 * T0);
    double T_an = T_ref_thck
                  + (T0 - T_ref_thck) * std::exp(-xi * t_end / tau_thck);
    double rel_err = std::fabs(T_num - T_an) / std::max(std::fabs(T_an), 1e-30);
    std::printf("    t=%.4g  T_num=%.12e  T_an=%.12e  rel=%.3e\n",
                t_end, T_num, T_an, rel_err);
    CHECK_LT(rel_err, 1e-14,
             "E1-T2: pure-thick limit matches Newton exponential to ULP");
    sv.destroy();
}

// --------------------------------------------------------------------
// E1-T3  blend intermediate (ξ=0.5).  Match host-side split reference.
// --------------------------------------------------------------------
static void test_T3_blend_intermediate() {
    std::printf("\n[E1-T3] blend ξ=0.5 → match host split reference\n");
    const int N = 16;
    const double rho0 = 1.0, T0 = 1.0;
    // p = rho0*T0 = 1, p_chr = 0.5 → ξ = 1 − 0.5 = 0.5
    const double p_chr = 0.5;
    const double T_ref_thck = 0.2, tau_thck = 2.0;
    const double Lambda0 = 1.0, T_ref_thin = 1.0, alpha = 2.0;

    AthenaMHDSolver sv;
    seed_uniform(sv, N, rho0, T0);
    sv.chromo_on         = true;
    sv.chromo_p_chr      = p_chr;
    sv.chromo_T_ref_thck = T_ref_thck;
    sv.chromo_tau_thck   = tau_thck;
    sv.chromo_Lambda0    = Lambda0;
    sv.chromo_T_ref_thin = T_ref_thin;
    sv.chromo_alpha      = alpha;
    sv.chromo_Tfloor     = 1e-6;

    double dt = 0.3;
    sv.apply_chromo_cooling(dt);
    double T_num = read_T_interior(sv);
    double gm1 = sv.gamma - 1.0;
    double p0 = rho0 * T0;
    double T_ref = chromo_ref(T0, rho0, p0, dt, gm1, p_chr,
                              T_ref_thck, tau_thck,
                              Lambda0, T_ref_thin, alpha, 1e-6);
    double rel_err = std::fabs(T_num - T_ref) / std::max(std::fabs(T_ref), 1e-30);
    std::printf("    dt=%g  T_num=%.15e  T_ref=%.15e  rel=%.3e\n",
                dt, T_num, T_ref, rel_err);
    CHECK_LT(rel_err, 1e-14,
             "E1-T3: blend ξ=0.5 matches host-side split to ULP");
    sv.destroy();
}

// --------------------------------------------------------------------
// E1-T4  passive: ρ / momentum / B_f / ∇·B all invariant.
// --------------------------------------------------------------------
static void test_T4_passive_to_other_fields() {
    std::printf("\n[E1-T4] chromo cooling only changes E (ρ/mom/B invariant)\n");
    const int N = 16;
    const double rho0 = 1.0, T0 = 1.0;
    const double Bx0 = 0.3, By0 = 0.7;
    AthenaMHDSolver sv;
    seed_uniform(sv, N, rho0, T0, Bx0, By0);
    sv.chromo_on         = true;
    sv.chromo_p_chr      = 0.5;
    sv.chromo_T_ref_thck = 0.2;
    sv.chromo_tau_thck   = 1.0;
    sv.chromo_Lambda0    = 1.0;
    sv.chromo_T_ref_thin = 1.0;
    sv.chromo_alpha      = 2.0;
    sv.chromo_Tfloor     = 1e-6;

    auto d0 = sv.compute_diagnostics();
    double dt = 0.3;
    sv.apply_chromo_cooling(dt);
    auto d1 = sv.compute_diagnostics();

    double dMass = std::fabs(d1.total_mass - d0.total_mass) /
                   std::max(std::fabs(d0.total_mass), 1e-30);
    double dME = std::fabs(d1.total_ME - d0.total_ME) /
                 std::max(std::fabs(d0.total_ME), 1e-30);
    double dKE = std::fabs(d1.total_KE - d0.total_KE);    // was 0
    std::printf("    rel Δmass=%.3e  rel ΔME=%.3e  ΔKE=%.3e  divB=%.3e\n",
                dMass, dME, dKE, d1.max_divB);
    CHECK_LT(dMass, 1e-14, "E1-T4: mass conserved exactly");
    CHECK_LT(dME,   1e-14, "E1-T4: magnetic energy frozen");
    CHECK_LT(dKE,   1e-14, "E1-T4: v=0 stays v=0");
    CHECK_LT(d1.max_divB, 1e-10, "E1-T4: ∇·B untouched");
    sv.destroy();
}

int main() {
    std::printf("=== athena_mhd chromospheric blended cooling (B-M5.5, §C8) ===\n");
    test_T1_pure_thin_limit();
    test_T2_pure_thick_limit();
    test_T3_blend_intermediate();
    test_T4_passive_to_other_fields();
    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
