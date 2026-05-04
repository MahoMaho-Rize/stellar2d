// Day-3 unit test: Radial1DSolver's alpha-burn kernel must agree with a
// direct CPU call to alpha_net::advance_substep to <1% per-zone.
//
// Setup: 4 zones each initialized to (He=0.5, O=0.5), varying (ρ, T).
// After 1 ms the kernel should produce the same X and (eps added to e_int)
// as advance_substep on the same (ρ, T, dt).
//
// Mass conservation is also checked (Σ X = 1 preserved).

#include "../src/gpu/radial1d_solver.cuh"
#include "../src/physics/alpha_network.h"

// Kernel defined in radial1d_kernels.cuh (emitted via radial1d_solver.cu TU).
extern __global__ void k_rad1d_alpha_burn(
    double*, double*, const double*, double, int, EOS, double, double*);
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    constexpr int nz = 4;
    constexpr int NS = alpha_net::N_SPEC;

    // Use ideal EOS but in CGS-aware mu so that temperature_from_rho_e
    // returns Kelvin: T_K = e / cv = e * mu * (γ-1) / (R_gas with k_B/m_u).
    // For the code's dimensionless ideal EOS, cv = 1/(mu·(γ-1)).  To get
    // T in Kelvin from a CGS e [erg/g], need mu = m_u / k_B (seconds are
    // irrelevant here — this is purely a dimensional map).  The practical
    // choice: set e = T · cv_cgs = T · k_B / ((γ-1) · μ_phys · m_u), and
    // set EOS.mu so that cv_eos = cv_cgs numerically: mu_eos = 1/cv_cgs.
    constexpr double k_B = 1.380649e-16;
    constexpr double m_u = 1.660539e-24;
    double mu_phys = 28.0;   // A-averaged for mostly-O composition
    double cv_cgs = k_B / ((5.0/3.0 - 1.0) * mu_phys * m_u);

    Radial1DSolver r1d;
    r1d.init(nz, 5.0/3.0, 6.674e-8, 0.4);
    r1d.use_eos = true;
    r1d.eos.type = static_cast<int>(EosType::IDEAL);
    r1d.eos.gamma = 5.0/3.0;
    // cv_eos = 1/(mu·(γ-1)). Want cv_eos = cv_cgs ⇒ mu = 1/((γ-1)·cv_cgs).
    r1d.eos.mu = 1.0 / ((5.0/3.0 - 1.0) * cv_cgs);
    r1d.alpha_burn_T_min = 1.0e6;   // allow all zones to burn

    // Four test states: one cold (below floor), three hot.
    double rho_test[nz]  = {1e4,   1e6,   1e7,   1e8};
    double T_test[nz]    = {5e5,   2e9,   3e9,   3.5e9};

    // Initial composition: (He=0.5, O=0.5) in every zone.
    std::vector<double> X0(nz * NS, 0.0);
    for (int k = 0; k < nz; ++k) {
        X0[k * NS + alpha_net::HE4] = 0.5;
        X0[k * NS + alpha_net::O16] = 0.5;
    }
    r1d.init_species_alpha(X0.data());

    // Seed e so that temperature_from_rho_e(rho, e) == T_test.
    std::vector<double> rho_h(nz), e_h(nz);
    for (int k = 0; k < nz; ++k) {
        rho_h[k] = rho_test[k];
        e_h[k] = T_test[k] * cv_cgs;
    }
    cudaMemcpy(r1d.lev.d_rho, rho_h.data(), nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(r1d.lev.d_e_int, e_h.data(), nz * sizeof(double), cudaMemcpyHostToDevice);

    // Reference: run advance_substep on CPU for each zone.
    double dt = 1.0e-3;  // 1 ms
    std::vector<double> X_ref(nz * NS), eps_ref(nz, 0.0);
    for (int k = 0; k < nz; ++k) {
        double Xk[NS];
        for (int s = 0; s < NS; ++s) Xk[s] = X0[k * NS + s];
        if (T_test[k] >= r1d.alpha_burn_T_min) {
            eps_ref[k] = alpha_net::advance_substep(Xk, rho_test[k], T_test[k], dt);
        }
        for (int s = 0; s < NS; ++s) X_ref[k * NS + s] = Xk[s];
    }

    // Launch the kernel.
    int B = 64;
    k_rad1d_alpha_burn<<<(nz+B-1)/B, B>>>(
        r1d.lev.d_e_int, r1d.d_X_spec, r1d.lev.d_rho, dt,
        nz, r1d.eos, r1d.alpha_burn_T_min, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> X_gpu;
    r1d.download_species_alpha(X_gpu);
    std::vector<double> e_gpu(nz);
    cudaMemcpy(e_gpu.data(), r1d.lev.d_e_int, nz * sizeof(double), cudaMemcpyDeviceToHost);

    int fail = 0;
    std::printf("Day-3 alpha burn unit test (dt=%.1e s):\n", dt);
    std::printf("  cv_cgs=%.3e, mu_eos=%.3e\n", cv_cgs, r1d.eos.mu);
    for (int k = 0; k < nz; ++k) {
        double T_back = r1d.eos.temperature_from_rho_e(rho_h[k], e_h[k]);
        std::printf("    seed k=%d: e_h=%.3e, T_back=%.3e (want %.3e)\n",
                    k, e_h[k], T_back, T_test[k]);
    }
    for (int k = 0; k < nz; ++k) {
        double max_dX = 0.0;
        for (int s = 0; s < NS; ++s) {
            double dX = std::fabs(X_gpu[k * NS + s] - X_ref[k * NS + s]);
            if (dX > max_dX) max_dX = dX;
        }
        double e_expect = e_h[k] + eps_ref[k];
        double rel_e = std::fabs(e_gpu[k] - e_expect)
                       / std::fmax(std::fabs(e_expect), 1e-30);
        // Σ X = 1 check
        double sX = 0.0;
        for (int s = 0; s < NS; ++s) sX += X_gpu[k * NS + s];
        std::printf("  k=%d rho=%.1e T=%.1e: max|dX|=%.3e  rel_e=%.3e  ΣX=%.6f  eps=%.3e erg/g\n",
                    k, rho_test[k], T_test[k], max_dX, rel_e, sX, eps_ref[k]);
        // advance_substep is adaptive forward-Euler; GPU <-> CPU can disagree
        // at high T due to reduction-order differences in floating-point.
        // 1e-3 tolerance is well inside the 1% accuracy the kernel itself
        // claims at T₉ ≥ 3 (stiff regime).
        if (max_dX > 1e-3)  { std::printf("    FAIL: species mismatch > 1e-3\n"); ++fail; }
        if (rel_e > 1e-3)   { std::printf("    FAIL: e_int mismatch > 1e-3\n"); ++fail; }
        if (std::fabs(sX - 1.0) > 1e-8) { std::printf("    FAIL: ΣX drift > 1e-8\n"); ++fail; }
    }

    r1d.destroy();
    if (fail) { std::printf("FAIL (%d)\n", fail); return 1; }
    std::printf("PASS\n");
    return 0;
}
