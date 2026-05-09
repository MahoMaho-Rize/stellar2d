#pragma once
// ============================================================
// gpu_hlld.cuh — HLLD Riemann solver for ideal MHD
//
// Based on Miyoshi & Kusano, J. Comput. Phys. 208, 315 (2005) —
// Eqs. (38)-(67) — with the three degeneracy branches documented
// in docs/derivations/mhd/sections/a09_hlld_degeneracy.md:
//
//   D1 (B_x ≈ 0)  → collapse to HLLC-MHD (Li 2005), 3-wave fan.
//   D2 (ρ(S-v)(S-S_M) ≈ B_x²) → regularise star-state B_y*, v_y*
//                               to upstream.
//   D3 (|(S_R-v_Rx)ρ_R - (S_L-v_Lx)ρ_L| ≈ 0) → pure HLL fallback.
//
// Convention.  The solver operates in a fixed NORMAL coordinate
// system: the first argument is the "normal" direction (what we
// call x), and the tangential directions are (y, z).  When we call
// it for a y-sweep, the caller rotates (vx, vy) ↔ (vy, vx) and
// likewise for the magnetic components before and after the call.
//
// Inputs:  wl, wr  primitive states
//          Bn       longitudinal (normal) magnetic field
//                   (= B_x for x-sweep, B_y for y-sweep)
//          gamma    ratio of specific heats
//
// Output:  FluxMHD7 = (f_rho, f_m_n, f_m_t, f_m_s, f_B_t, f_B_s, f_E)
//          with "t"=tangential-1 (what we call y), "s"=tangential-2 (z).
//          NOTE: flux of the normal B component is ZERO (it equals
//          B_n · v_n − v_n · B_n = 0), so it's not returned.
//          Instead, the caller uses f_B_t = (v_n B_t − v_t B_n) at
//          the face to construct the CT corner EMF.
// ============================================================

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

struct MHDPrim {
    double rho;
    double vn, vt, vs;    // velocity:  normal, tangential-1, tangential-2
    double Bt, Bs;        // transverse B in the NORMAL system
    double P;
};

struct FluxMHD7 {
    double f_rho;
    double f_mn, f_mt, f_ms;
    double f_Bt, f_Bs;
    double f_E;
};

// Smallest allowed pressure / density to avoid 0-div.  Matches
// athena_vl2's floor of 1e-30.
#ifndef GPU_MHD_FLOOR
#define GPU_MHD_FLOOR (1e-30)
#endif

__device__ __forceinline__
double mhd_fast_speed_sq(double rho, double P, double Bn, double Bt,
                         double Bs, double gamma) {
    double cs0_sq = gamma * P / rho;
    double cAn_sq = Bn * Bn / rho;
    double cAp_sq = (Bt * Bt + Bs * Bs) / rho;
    double sum_sq = cs0_sq + cAn_sq + cAp_sq;
    double disc   = sum_sq * sum_sq - 4.0 * cs0_sq * cAn_sq;
    if (disc < 0.0) disc = 0.0;
    return 0.5 * (sum_sq + sqrt(disc));
}

__device__ __forceinline__
double mhd_total_pressure(const MHDPrim& w, double Bn) {
    return w.P + 0.5 * (Bn * Bn + w.Bt * w.Bt + w.Bs * w.Bs);
}

__device__ __forceinline__
double mhd_total_energy(const MHDPrim& w, double Bn, double gamma) {
    double ke = 0.5 * w.rho * (w.vn * w.vn + w.vt * w.vt + w.vs * w.vs);
    double me = 0.5 * (Bn * Bn + w.Bt * w.Bt + w.Bs * w.Bs);
    return w.P / (gamma - 1.0) + ke + me;
}

// Physical flux of an MHD primitive state in the "n" direction (§A1).
__device__ __forceinline__
FluxMHD7 mhd_physical_flux(const MHDPrim& w, double Bn, double gamma) {
    FluxMHD7 f;
    double Ptot  = mhd_total_pressure(w, Bn);
    double E     = mhd_total_energy(w, Bn, gamma);
    double Bdotv = Bn * w.vn + w.Bt * w.vt + w.Bs * w.vs;
    f.f_rho = w.rho * w.vn;
    f.f_mn  = w.rho * w.vn * w.vn + Ptot - Bn * Bn;
    f.f_mt  = w.rho * w.vn * w.vt - Bn * w.Bt;
    f.f_ms  = w.rho * w.vn * w.vs - Bn * w.Bs;
    f.f_Bt  = w.vn * w.Bt - w.vt * Bn;
    f.f_Bs  = w.vn * w.Bs - w.vs * Bn;
    f.f_E   = (E + Ptot) * w.vn - Bn * Bdotv;
    return f;
}

// Conservative-state helper (n-major layout).  Returns 7-component
// vector in the same order used by FluxMHD7 (mass, m_n, m_t, m_s,
// B_t, B_s, E).
struct ConsMHD7 { double U[7]; };

__device__ __forceinline__
ConsMHD7 mhd_prim_to_cons(const MHDPrim& w, double Bn, double gamma) {
    ConsMHD7 U;
    U.U[0] = w.rho;
    U.U[1] = w.rho * w.vn;
    U.U[2] = w.rho * w.vt;
    U.U[3] = w.rho * w.vs;
    U.U[4] = w.Bt;
    U.U[5] = w.Bs;
    U.U[6] = mhd_total_energy(w, Bn, gamma);
    return U;
}

__device__ __forceinline__
FluxMHD7 mhd_apply_hll(const MHDPrim& wl, const MHDPrim& wr,
                      double Bn, double gamma, double SL, double SR) {
    ConsMHD7 UL = mhd_prim_to_cons(wl, Bn, gamma);
    ConsMHD7 UR = mhd_prim_to_cons(wr, Bn, gamma);
    FluxMHD7 FL = mhd_physical_flux(wl, Bn, gamma);
    FluxMHD7 FR = mhd_physical_flux(wr, Bn, gamma);
    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;
    double denom = SR - SL;
    if (fabs(denom) < 1e-300) denom = 1e-300;
    FluxMHD7 out;
    double inv = 1.0 / denom;
    double* FLp = &FL.f_rho;
    double* FRp = &FR.f_rho;
    double* FOp = &out.f_rho;
    for (int k = 0; k < 7; ++k) {
        FOp[k] = (SR * FLp[k] - SL * FRp[k]
                  + SL * SR * (UR.U[k] - UL.U[k])) * inv;
    }
    return out;
}

// Star-region flux using MK's DEFINED total pressure — NOT via EOS
// inversion (§A4 critical note).  Given U*_K = (ρ*, m_n*=ρ*S_M,
// m_t*, m_s*, B_t*, B_s*, E*), and p*_tot_common,
//   F*_{K} = (ρ*·S_M,
//            ρ*·S_M² + p*_tot − B_n²,
//            ρ*·S_M·v_t* − B_n·B_t*,
//            ρ*·S_M·v_s* − B_n·B_s*,
//            S_M·B_t* − v_t*·B_n,
//            S_M·B_s* − v_s*·B_n,
//            (E* + p*_tot)·S_M − B_n·(B_n·S_M + B_t*·v_t* + B_s*·v_s*))
__device__ __forceinline__
FluxMHD7 mhd_star_flux(double rho_s, double SM, double vt_s, double vs_s,
                       double Bt_s, double Bs_s, double Ptot_s, double E_s,
                       double Bn) {
    FluxMHD7 f;
    double Bdotv = Bn * SM + Bt_s * vt_s + Bs_s * vs_s;
    f.f_rho = rho_s * SM;
    f.f_mn  = rho_s * SM * SM + Ptot_s - Bn * Bn;
    f.f_mt  = rho_s * SM * vt_s - Bn * Bt_s;
    f.f_ms  = rho_s * SM * vs_s - Bn * Bs_s;
    f.f_Bt  = SM * Bt_s - vt_s * Bn;
    f.f_Bs  = SM * Bs_s - vs_s * Bn;
    f.f_E   = (E_s + Ptot_s) * SM - Bn * Bdotv;
    return f;
}

// ============================================================
// HLLD Riemann solver — full 5-wave MHD
// ============================================================
__device__ inline
FluxMHD7 gpu_hlld(const MHDPrim& wl_, const MHDPrim& wr_,
                  double Bn, double gamma)
{
    // Protect against floors at function entrance (upstream callers
    // are expected to have already floored, but this makes the
    // routine self-contained for isolated testing).
    MHDPrim wl = wl_;
    MHDPrim wr = wr_;
    if (wl.rho < GPU_MHD_FLOOR) wl.rho = GPU_MHD_FLOOR;
    if (wr.rho < GPU_MHD_FLOOR) wr.rho = GPU_MHD_FLOOR;
    if (wl.P   < GPU_MHD_FLOOR) wl.P   = GPU_MHD_FLOOR;
    if (wr.P   < GPU_MHD_FLOOR) wr.P   = GPU_MHD_FLOOR;

    // Step 1: outer fast-wave bounds (Davis 1988).
    double cfL = sqrt(mhd_fast_speed_sq(wl.rho, wl.P, Bn, wl.Bt, wl.Bs, gamma));
    double cfR = sqrt(mhd_fast_speed_sq(wr.rho, wr.P, Bn, wr.Bt, wr.Bs, gamma));
    double SL = fmin(wl.vn - cfL, wr.vn - cfR);
    double SR = fmax(wl.vn + cfL, wr.vn + cfR);

    if (SL >= 0.0) return mhd_physical_flux(wl, Bn, gamma);
    if (SR <= 0.0) return mhd_physical_flux(wr, Bn, gamma);

    double PtL = mhd_total_pressure(wl, Bn);
    double PtR = mhd_total_pressure(wr, Bn);

    // Step 2: contact speed S_M  (MK Eq. 38).
    double denom = (SR - wr.vn) * wr.rho - (SL - wl.vn) * wl.rho;
    double abs_denom = fabs(denom);
    double scale = (fabs(wl.rho * (SR - wl.vn)) + fabs(wr.rho * (SR - wr.vn))
                    + 1.0) * 1e-12;
    if (abs_denom < scale) {
        // D3: pure HLL fallback.
        return mhd_apply_hll(wl, wr, Bn, gamma, SL, SR);
    }
    double SM = ((SR - wr.vn) * wr.rho * wr.vn
                 - (SL - wl.vn) * wl.rho * wl.vn
                 - PtR + PtL) / denom;

    // Step 3: star-region densities (MK Eq. 43).
    double rho_sL = wl.rho * (SL - wl.vn) / (SL - SM);
    double rho_sR = wr.rho * (SR - wr.vn) / (SR - SM);

    // Common star total pressure (MK Eq. 41).
    double Pt_s = PtL + wl.rho * (SL - wl.vn) * (SM - wl.vn);
    // (symmetrically = PtR + wr.rho*(SR-wr.vn)*(SM-wr.vn), proved equal
    //  in §A4 sympy; not enforced here — numerical round-off only.)

    // Step 4: detect tangential-field degeneracy (D1 — B_x ≈ 0).
    // When Bn² is small compared to ρ_sK·(SL−v)·(SM−v), the Alfvén
    // fan coalesces with the contact and the scheme degrades to
    // HLLC-MHD.  We use the combined criterion:  |Bn| * sqrt(ρ_sK)
    // below an epsilon of the characteristic scale.
    double Bn_sq = Bn * Bn;

    // Star-state transverse B & v  (MK Eq. 44-47 ; §A9 kernel
    // regularisation).
    auto star_transverse = [&](double rho_K, double vn_K, double vt_K,
                               double vs_K, double Bt_K, double Bs_K,
                               double SK,
                               double& Bt_s, double& Bs_s,
                               double& vt_s, double& vs_s)
    {
        double num_BS = rho_K * (SK - vn_K) * (SK - vn_K) - Bn_sq;
        double den_BS = rho_K * (SK - vn_K) * (SK - SM) - Bn_sq;
        // §A9 kernel regularisation:  if |den| small compared to
        // the characteristic Bn²·sqrt(ρ) scale, fall back to upstream.
        double eps_scale = fmax(1e-12,
                                1e-12 * (fabs(Bn_sq) + fabs(rho_K * (SK - vn_K) * (SK - vn_K))));
        if (fabs(den_BS) < eps_scale) {
            Bt_s = Bt_K; Bs_s = Bs_K;
            vt_s = vt_K; vs_s = vs_K;
            return;
        }
        double ratio = num_BS / den_BS;
        Bt_s = Bt_K * ratio;
        Bs_s = Bs_K * ratio;
        double coeff = Bn * (SM - vn_K) / den_BS;
        vt_s = vt_K - Bt_K * coeff;
        vs_s = vs_K - Bs_K * coeff;
    };
    double Bt_sL, Bs_sL, vt_sL, vs_sL;
    double Bt_sR, Bs_sR, vt_sR, vs_sR;
    star_transverse(wl.rho, wl.vn, wl.vt, wl.vs, wl.Bt, wl.Bs, SL,
                    Bt_sL, Bs_sL, vt_sL, vs_sL);
    star_transverse(wr.rho, wr.vn, wr.vt, wr.vs, wr.Bt, wr.Bs, SR,
                    Bt_sR, Bs_sR, vt_sR, vs_sR);

    // Star-state energy (MK Eq. 48).
    double EL = mhd_total_energy(wl, Bn, gamma);
    double ER = mhd_total_energy(wr, Bn, gamma);
    double Bdotv_L = Bn * wl.vn + wl.Bt * wl.vt + wl.Bs * wl.vs;
    double Bdotv_R = Bn * wr.vn + wr.Bt * wr.vt + wr.Bs * wr.vs;
    double Bdotv_sL = Bn * SM + Bt_sL * vt_sL + Bs_sL * vs_sL;
    double Bdotv_sR = Bn * SM + Bt_sR * vt_sR + Bs_sR * vs_sR;
    double E_sL = ((SL - wl.vn) * EL - PtL * wl.vn + Pt_s * SM
                   + Bn * (Bdotv_L - Bdotv_sL)) / (SL - SM);
    double E_sR = ((SR - wr.vn) * ER - PtR * wr.vn + Pt_s * SM
                   + Bn * (Bdotv_R - Bdotv_sR)) / (SR - SM);

    // Step 5: Alfvén star-speeds (MK Eq. 51).
    double sqrt_rsL = sqrt(rho_sL);
    double sqrt_rsR = sqrt(rho_sR);
    double absBn    = fabs(Bn);
    double SsL = SM - absBn / sqrt_rsL;
    double SsR = SM + absBn / sqrt_rsR;

    // D1: B_n ≈ 0 — S*_L = S*_R = S_M.  Emit star fluxes directly.
    //     Detection:  absBn below machine-level ρ·c_f scale of either
    //     side (MK 2005 §3.4; Athena++ hlld.cpp uses 1e-12·(ρ·c_f)).
    double Bn_scale = fmax(1e-12,
                           1e-12 * (wl.rho * cfL + wr.rho * cfR));
    bool bx_zero = (absBn < Bn_scale);

    if (bx_zero) {
        // Emit the appropriate outer star flux (SsL = SsR = SM).
        if (SM >= 0.0) {
            // Left star region.
            FluxMHD7 f = mhd_star_flux(rho_sL, SM, vt_sL, vs_sL,
                                       Bt_sL, Bs_sL, Pt_s, E_sL, Bn);
            return f;
        } else {
            FluxMHD7 f = mhd_star_flux(rho_sR, SM, vt_sR, vs_sR,
                                       Bt_sR, Bs_sR, Pt_s, E_sR, Bn);
            return f;
        }
    }

    // Step 6: double-star states (MK Eq. 59-62) exist only if B_n ≠ 0.
    double sum_sqrt = sqrt_rsL + sqrt_rsR;
    if (sum_sqrt < 1e-300) sum_sqrt = 1e-300;
    double s_sign = (Bn > 0.0) ? 1.0 : -1.0;

    double vt_ss = (sqrt_rsL * vt_sL + sqrt_rsR * vt_sR
                    + (Bt_sR - Bt_sL) * s_sign) / sum_sqrt;
    double vs_ss = (sqrt_rsL * vs_sL + sqrt_rsR * vs_sR
                    + (Bs_sR - Bs_sL) * s_sign) / sum_sqrt;
    double Bt_ss = (sqrt_rsL * Bt_sR + sqrt_rsR * Bt_sL
                    + sqrt_rsL * sqrt_rsR * (vt_sR - vt_sL) * s_sign) / sum_sqrt;
    double Bs_ss = (sqrt_rsL * Bs_sR + sqrt_rsR * Bs_sL
                    + sqrt_rsL * sqrt_rsR * (vs_sR - vs_sL) * s_sign) / sum_sqrt;

    // Double-star energies (MK Eq. 63).
    double Bdotv_ss = Bn * SM + Bt_ss * vt_ss + Bs_ss * vs_ss;
    double E_ssL = E_sL - sqrt_rsL * (Bdotv_sL - Bdotv_ss) * s_sign;
    double E_ssR = E_sR + sqrt_rsR * (Bdotv_sR - Bdotv_ss) * s_sign;

    // Step 7: dispatch to the appropriate region.
    //   SL ≥ 0:  F_L                     (handled early)
    //   SR ≤ 0:  F_R                     (handled early)
    //   SL < 0 ≤ S*_L:  F_L*
    //   S*_L < 0 ≤ S_M:  F_L**
    //   S_M < 0 ≤ S*_R:  F_R**
    //   S*_R < 0 < SR:  F_R*
    FluxMHD7 f;
    if (SsL >= 0.0) {
        // Left star region
        f = mhd_star_flux(rho_sL, SM, vt_sL, vs_sL,
                          Bt_sL, Bs_sL, Pt_s, E_sL, Bn);
    } else if (SM >= 0.0) {
        // Left double-star region
        // Flux via MK Eq. 64:  F**_L = F*_L + S*_L · (U**_L − U*_L)
        FluxMHD7 fstar = mhd_star_flux(rho_sL, SM, vt_sL, vs_sL,
                                       Bt_sL, Bs_sL, Pt_s, E_sL, Bn);
        f.f_rho = fstar.f_rho;            // ρ* = ρ** (MK)
        f.f_mn  = fstar.f_mn;              // m_n* = m_n** (both = ρ_sL·SM)
        f.f_mt  = fstar.f_mt + SsL * rho_sL * (vt_ss - vt_sL);
        f.f_ms  = fstar.f_ms + SsL * rho_sL * (vs_ss - vs_sL);
        f.f_Bt  = fstar.f_Bt + SsL * (Bt_ss - Bt_sL);
        f.f_Bs  = fstar.f_Bs + SsL * (Bs_ss - Bs_sL);
        f.f_E   = fstar.f_E  + SsL * (E_ssL - E_sL);
    } else if (SsR >= 0.0) {
        // Right double-star region
        FluxMHD7 fstar = mhd_star_flux(rho_sR, SM, vt_sR, vs_sR,
                                       Bt_sR, Bs_sR, Pt_s, E_sR, Bn);
        f.f_rho = fstar.f_rho;
        f.f_mn  = fstar.f_mn;
        f.f_mt  = fstar.f_mt + SsR * rho_sR * (vt_ss - vt_sR);
        f.f_ms  = fstar.f_ms + SsR * rho_sR * (vs_ss - vs_sR);
        f.f_Bt  = fstar.f_Bt + SsR * (Bt_ss - Bt_sR);
        f.f_Bs  = fstar.f_Bs + SsR * (Bs_ss - Bs_sR);
        f.f_E   = fstar.f_E  + SsR * (E_ssR - E_sR);
    } else {
        // Right star region
        f = mhd_star_flux(rho_sR, SM, vt_sR, vs_sR,
                          Bt_sR, Bs_sR, Pt_s, E_sR, Bn);
    }
    return f;
}
