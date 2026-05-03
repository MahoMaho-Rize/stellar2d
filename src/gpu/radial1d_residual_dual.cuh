#pragma once

// Dual<N>-templated residual for radial1d.
//
// Mirrors the static __global__ k_r1di_residual in radial1d_implicit.cu but
// operates on Dual<N>-valued state, producing Dual<N>-valued residuals so
// that a single pass gives F plus all N directional derivatives (for exact
// J·v matvec via Dual<1>, or block-tridiag PC build via Dual<9>).
//
// Inputs are "lifted" packed state (U as Dual<N>[3·nz]). Face-centered arrays
// r, v and zone-centered e_int, rho, P, Pvsc are reconstructed here — the
// Dual path does NOT use lev.d_{r,v,e_int,rho,P,Pvsc} (those are scalar).
//
// EOS path: Helmholtz only. For IDEAL / IDEAL_RAD we fall back to direct
// gamma-law since EOS::pressure is not Dual-aware. PRE_MS not yet supported.
//
// Nuclear pp included inline when nuclear_on (composition-independent form).

#include "../physics/dual.cuh"
#include "../physics/helmholtz_eos_dual.cuh"
#include "../physics/nuclear_pp.h"
#include "../physics/opacity.h"
#include "../eos.h"

namespace dualR {

// Radiation + convection params bundled for residual. Mirrors the inputs to
// k_rad1d_be_assemble: grey Rosseland κ via OpacityParams, Stefan-Boltzmann
// surface BC. `enabled` gates the whole contribution.
struct RadParams {
    int  enabled        = 0;        // 0 ⇒ skip rad term entirely
    double a_rad        = 7.5657e-15;
    double c_light      = 2.998e10;
    double sigma_sb     = 5.6704e-5; // = c·a/4
    // Minimum photospheric T used in surface Stefan BC. Prevents runaway
    // cooling below the Helm table floor (1000 K) which physically would
    // be arrested by H⁻ opacity blow-up. 0 = no floor.
    double T_phot_floor = 0.0;
    OpacityParams opa;
    // MLT convection: Picard-lagged K_conv per zone (nullptr ⇒ no MLT).
    // F_conv_face = A_face · K_face · (T_L − T_R) / Δr_zc, linear in T.
    // K_face = 0.5·(K_conv[L] + K_conv[R]). Treated as scalar (not Dual)
    // since K is lagged between Newton solves.
    const double* K_conv = nullptr;
};

// PI constants (match radial1d_kernels.cuh)
static constexpr double PI4_D  = 12.566370614359172;
static constexpr double PI43_D = 4.188790204786391;

// ε_pp for Dual<N>: inline the formula from nuclear_pp.h with dual math so
// derivatives flow through.
template <int N>
__host__ __device__ inline dual::Dual<N> pp_eps_dual(const dual::Dual<N>& rho,
                                                     const dual::Dual<N>& T,
                                                     const NuclearPPParams& p)
{
    using dual::pow; using dual::exp;
    double T_K_v = T.v * p.T_scale;
    if (T_K_v < p.T_floor) return dual::Dual<N>(0.0);
    if (p.X_hydrogen <= 0.0) return dual::Dual<N>(0.0);
    dual::Dual<N> T_K = T * p.T_scale;
    dual::Dual<N> T9 = T_K * 1e-9;
    if (T9.v < 1e-6) return dual::Dual<N>(0.0);
    dual::Dual<N> T9_13 = pow(T9,  1.0 / 3.0);
    dual::Dual<N> T9_m23= pow(T9, -2.0 / 3.0);
    double exp_arg_v = -3.381 / T9_13.v;
    if (exp_arg_v < -700.0) return dual::Dual<N>(0.0);
    dual::Dual<N> X = dual::Dual<N>(p.X_hydrogen);
    dual::Dual<N> eps = 2.57e4 * X * X * rho * T9_m23 * exp(-3.381 / T9_13);
    return eps * p.epsilon_scale;
}

// Dual pressure from (rho, e). Helm branch: Newton-invert T from e, then
// helm_eval_dual at that T. IDEAL path: analytic.
template <int N>
__host__ __device__ inline dual::Dual<N> pressure_dual(const dual::Dual<N>& rho,
                                                       const dual::Dual<N>& e,
                                                       const EOS& eos)
{
#ifdef USE_GPU
    if (eos.type == (int)EosType::HELMHOLTZ) {
        // Use implicit function theorem to lift T, then helm_eval_dual.
        dual::Dual<N> T = helm_T_from_rho_e_dual<N>(rho, e, eos.helm);
        HelmStateDual<N> s = helm_eval_dual<N>(rho, T, eos.helm);
        return s.P;
    }
#endif
    // IDEAL: P = (γ - 1) ρ e
    double gm1 = eos.gamma - 1.0;
    return gm1 * rho * e;
}

// Same but return T too (for nuclear).
template <int N>
__host__ __device__ inline void primitives_dual(const dual::Dual<N>& rho,
                                                const dual::Dual<N>& e,
                                                const EOS& eos,
                                                dual::Dual<N>& P_out,
                                                dual::Dual<N>& T_out)
{
#ifdef USE_GPU
    if (eos.type == (int)EosType::HELMHOLTZ) {
        T_out = helm_T_from_rho_e_dual<N>(rho, e, eos.helm);
        HelmStateDual<N> s = helm_eval_dual<N>(rho, T_out, eos.helm);
        P_out = s.P;
        return;
    }
#endif
    // IDEAL fallback: T = e / cv, P = (γ − 1) ρ e
    double cv = (1.0 / eos.mu) / (eos.gamma - 1.0);
    if (cv < 1e-30) cv = 1e-30;
    T_out = e * (1.0 / cv);
    P_out = (eos.gamma - 1.0) * rho * e;
}

// Rosseland radiation flux across a single face between zones L and R.
// Returns the Dual-valued luminosity flowing from L to R (positive = outward).
//
// L_face = 4π r_face² · D · a · (T_L⁴ - T_R⁴) / Δr_zc
// with D = c / (3 κ ρ_face). Opacity is Picard-lagged (evaluated at scalar
// T_L.v, ρ_face.v) so the Dual gradient comes only from T_L⁴, T_R⁴, r_face.
// Δr_zc is the zone-center spacing — scalar since r enters only via face
// positions, and differentiating Δr_zc through r would couple more broadly;
// the standard choice in BE-rad assembly treats it as geometric lag too.
template <int N>
__host__ __device__ inline dual::Dual<N> rad_face_L_dual(
    const dual::Dual<N>& rho_L_d, const dual::Dual<N>& T_L_d,
    const dual::Dual<N>& rho_R_d, const dual::Dual<N>& T_R_d,
    const dual::Dual<N>& r_face_d, double dr_zc,
    const RadParams& rad,
    double K_face = 0.0)   // Picard-lagged MLT conductivity at this face
{
    using dual::Dual;
    double rho_face_v = 0.5 * (rho_L_d.v + rho_R_d.v);
    double T_face_v   = 0.5 * (T_L_d.v   + T_R_d.v);
    double kap = grey_opacity(rho_face_v > 1e-30 ? rho_face_v : 1e-30,
                              T_face_v   > 1.0   ? T_face_v   : 1.0, rad.opa);
    if (!(kap > 1e-30)) kap = 1e-30;
    double D_v = rad.c_light / (3.0 * kap * (rho_face_v > 1e-30 ? rho_face_v : 1e-30));
    // A_face = 4π r²
    Dual<N> A_face = PI4_D * r_face_d * r_face_d;
    // T⁴_L - T⁴_R (Dual). Clamp tiny values.
    Dual<N> TL2 = T_L_d * T_L_d;
    Dual<N> TL4 = TL2 * TL2;
    Dual<N> TR2 = T_R_d * T_R_d;
    Dual<N> TR4 = TR2 * TR2;
    double dr = dr_zc > 1e-30 ? dr_zc : 1e-30;
    Dual<N> L = A_face * (D_v * rad.a_rad / dr) * (TL4 - TR4);
    // MLT convective flux (Picard-lagged K). Linear in T.
    //   F_conv = A_face · K_face · (T_L − T_R) / Δr_zc
    if (K_face > 0.0) {
        L = L + A_face * (K_face / dr) * (T_L_d - T_R_d);
    }
    return L;
}

// Compute residual R = R(U) for one zone k and write face-v, face-r, zone-e
// components. Face arrays (r, v) are indexed k = 0..nz; this helper produces
// the three R packed components for zone/face index k.
//
// U layout (as Dual<N>):
//   U[0 .. nz-1]     = v_face[1..nz]
//   U[nz .. 2nz-1]   = r_face[1..nz]
//   U[2nz .. 3nz-1]  = e_zone[0..nz-1]
// v_face[0] = 0, r_face[0] = 0 (pinned).
//
// Outputs: R_out[k], R_out[nz+k], R_out[2nz+k].
//
// dm, g_face are scalar double (input data; not differentiated through).
// Note: g_face depends on r through g = GM/r²; since r is one of our
// unknowns, strictly we should differentiate g_face through r. But r only
// appears at the face — only g[k+1] affects R_v at face k+1.
//
// Simpler approach: precompute g as Dual<N> here, consistent with r-dual.
template <int N>
__host__ __device__ inline void residual_zone_dual(
    int k, int nz,
    const dual::Dual<N>* U,          // packed state (3·nz)
    const double* dm,                // nz zone masses (scalar)
    double G_const, double P_surf_floor,
    double CQ, double ZSH,
    const EOS& eos,
    const NuclearPPParams& npars,
    int nuclear_on,
    const RadParams& rad,
    dual::Dual<N>& R_v_out,
    dual::Dual<N>& R_r_out,
    dual::Dual<N>& R_e_out)
{
    using dual::fmax; using dual::sqrt;
    using dual::Dual;

    // --- Reconstruct face-v, face-r, zone-e from U ---
    auto v_face = [&](int kf) -> Dual<N> {
        if (kf == 0) return Dual<N>(0.0);
        if (kf >= 1 && kf <= nz) return U[kf - 1];  // v_face[kf] stored at U[kf-1]
        // should not happen
        return Dual<N>(0.0);
    };
    auto r_face = [&](int kf) -> Dual<N> {
        if (kf == 0) return Dual<N>(0.0);
        if (kf >= 1 && kf <= nz) return U[nz + kf - 1];
        return Dual<N>(0.0);
    };
    auto e_zone = [&](int j) -> Dual<N> {
        return U[2*nz + j];
    };

    int kf = k + 1;

    // --- Zone primitives for zone k ---
    Dual<N> rL = r_face(k);
    Dual<N> rR = r_face(k+1);
    Dual<N> Vk = PI43_D * (rR*rR*rR - rL*rL*rL);
    if (Vk.v < 1e-30) Vk = Dual<N>(1e-30);
    Dual<N> rho_k = Dual<N>(dm[k]) / Vk;
    Dual<N> e_k = e_zone(k);
    if (e_k.v < 1e-30) e_k = Dual<N>(1e-30);
    Dual<N> P_k, T_k;
    primitives_dual<N>(rho_k, e_k, eos, P_k, T_k);

    // --- Artificial viscosity for zone k (Tscharnuter-Winkler) ---
    Dual<N> vL = v_face(k);
    Dual<N> vR = v_face(k+1);
    Dual<N> dv = vL - vR;
    Dual<N> Pk_safe = fmax(P_k, 1e-30);
    Dual<N> Vk_safe = fmax(Vk, 1e-30);
    Dual<N> sqrt_PV = sqrt(Pk_safe * Vk_safe);
    Dual<N> thresh  = ZSH * sqrt_PV;
    Dual<N> Pvsc_k(0.0);
    if (dv.v > thresh.v) {
        Dual<N> excess = dv - thresh;
        Pvsc_k = (CQ / Vk_safe) * excess * excess;
    }

    // --- Zone primitives for zone k-1 and k+1 (needed for R_v at face k+1) ---
    // For R_v at face kf = k+1:
    //   XP_above = P[kf] + Pvsc[kf]   (zone kf if < nz, else ghost)
    //   XP_below = P[kf-1] + Pvsc[kf-1] = P_k + Pvsc_k   (zone k)
    //
    // We need zone (kf = k+1) primitives. Recompute inline (small cost
    // relative to total F eval).
    Dual<N> XP_above, XP_below, dm_bar_d;
    if (kf == nz) {
        XP_above = Dual<N>(P_surf_floor);
        XP_below = P_k + Pvsc_k;
        dm_bar_d = Dual<N>(0.5 * dm[kf-1]);
    } else {
        int j = kf;  // zone j = kf
        Dual<N> rLj = r_face(j);
        Dual<N> rRj = r_face(j+1);
        Dual<N> Vj = PI43_D * (rRj*rRj*rRj - rLj*rLj*rLj);
        if (Vj.v < 1e-30) Vj = Dual<N>(1e-30);
        Dual<N> rho_j = Dual<N>(dm[j]) / Vj;
        Dual<N> e_j   = e_zone(j);
        if (e_j.v < 1e-30) e_j = Dual<N>(1e-30);
        Dual<N> P_j, T_j;
        primitives_dual<N>(rho_j, e_j, eos, P_j, T_j);

        Dual<N> vLj = v_face(j);
        Dual<N> vRj = v_face(j+1);
        Dual<N> dvj = vLj - vRj;
        Dual<N> Pj_safe = fmax(P_j, 1e-30);
        Dual<N> Vj_safe = fmax(Vj, 1e-30);
        Dual<N> sqrt_PVj = sqrt(Pj_safe * Vj_safe);
        Dual<N> threshj  = ZSH * sqrt_PVj;
        Dual<N> Pvsc_j(0.0);
        if (dvj.v > threshj.v) {
            Dual<N> excess = dvj - threshj;
            Pvsc_j = (CQ / Vj_safe) * excess * excess;
        }
        XP_above = P_j + Pvsc_j;
        XP_below = P_k + Pvsc_k;
        dm_bar_d = Dual<N>(0.5 * (dm[kf-1] + dm[kf]));
    }

    // Gravity at face kf: g = G M[kf] / r[kf]²
    // M[kf] is a prefix sum of dm — scalar, independent of U.
    double M_kf = 0.0;
    for (int j = 0; j < kf; ++j) M_kf += dm[j];
    Dual<N> rk = r_face(kf);
    Dual<N> rk2 = rk * rk;
    if (rk2.v < 1e-40) rk2 = Dual<N>(1e-40);
    Dual<N> g_kf = (G_const * M_kf) / rk2;
    Dual<N> Ak = PI4_D * rk2;

    R_v_out = (-1.0) * Ak * (XP_above - XP_below) / fmax(dm_bar_d, 1e-30) - g_kf;

    // R_r at face kf
    R_r_out = v_face(kf);

    // R_e at zone k
    Dual<N> dVdt = PI4_D * (rR*rR*vR - rL*rL*vL);
    Dual<N> Pt = P_k + Pvsc_k;
    R_e_out = (-1.0) * Pt * dVdt / fmax(Dual<N>(dm[k]), 1e-30);

    // Nuclear source (zone k)
    if (nuclear_on) {
        Dual<N> rho_safe = fmax(rho_k, 1e-30);
        Dual<N> eps = pp_eps_dual<N>(rho_safe, T_k, npars);
        R_e_out = R_e_out + eps;
    }

    // --- Radiation diffusion source for zone k ---
    // d(ρe·V)/dt_rad = L_in - L_out  ⇒ per-mass rate (L_in - L_out) / dm_k.
    // L at inner face (index k, between zones k-1 and k):
    //   L_in = 4π r_k² · D · a · (T_{k-1}⁴ - T_k⁴) / Δr_zc,  D = c/(3κρ_face)
    // L at outer face (index k+1):
    //   Interior: same form with T_k, T_{k+1}
    //   Surface (k == nz-1): L = 4π r_surf² · σ · T_k⁴ (Stefan-Boltzmann)
    if (rad.enabled) {
        Dual<N> L_in(0.0);
        Dual<N> L_out(0.0);

        // Inner face (k). For k=0 there is no inner face (center is
        // reflective); treat L_in = 0.
        if (k >= 1) {
            // Zone k-1 primitives
            Dual<N> rLm = r_face(k-1);
            Dual<N> rRm = r_face(k);
            Dual<N> Vm  = PI43_D * (rRm*rRm*rRm - rLm*rLm*rLm);
            if (Vm.v < 1e-30) Vm = Dual<N>(1e-30);
            Dual<N> rho_m = Dual<N>(dm[k-1]) / Vm;
            Dual<N> e_m   = e_zone(k-1);
            if (e_m.v < 1e-30) e_m = Dual<N>(1e-30);
            Dual<N> P_m, T_m;
            primitives_dual<N>(rho_m, e_m, eos, P_m, T_m);

            // zone-center spacing: rc_k - rc_{k-1} ≈ 0.5(r_{k+1}-r_{k-1})
            double rc_lo = 0.5 * (r_face(k-1).v + r_face(k).v);
            double rc_hi = 0.5 * (r_face(k).v   + r_face(k+1).v);
            double dr_zc = rc_hi - rc_lo;
            double K_face = 0.0;
            if (rad.K_conv != nullptr) {
                K_face = 0.5 * (rad.K_conv[k-1] + rad.K_conv[k]);
            }
            L_in = rad_face_L_dual<N>(rho_m, T_m, rho_k, T_k,
                                       r_face(k), dr_zc, rad, K_face);
        }

        // Outer face (k+1)
        if (k < nz - 1) {
            // Zone k+1 primitives
            Dual<N> rLj = r_face(k+1);
            Dual<N> rRj = r_face(k+2);
            Dual<N> Vj  = PI43_D * (rRj*rRj*rRj - rLj*rLj*rLj);
            if (Vj.v < 1e-30) Vj = Dual<N>(1e-30);
            Dual<N> rho_j = Dual<N>(dm[k+1]) / Vj;
            Dual<N> e_j   = e_zone(k+1);
            if (e_j.v < 1e-30) e_j = Dual<N>(1e-30);
            Dual<N> P_j, T_j;
            primitives_dual<N>(rho_j, e_j, eos, P_j, T_j);

            double rc_lo = 0.5 * (r_face(k).v   + r_face(k+1).v);
            double rc_hi = 0.5 * (r_face(k+1).v + r_face(k+2).v);
            double dr_zc = rc_hi - rc_lo;
            double K_face = 0.0;
            if (rad.K_conv != nullptr) {
                K_face = 0.5 * (rad.K_conv[k] + rad.K_conv[k+1]);
            }
            L_out = rad_face_L_dual<N>(rho_k, T_k, rho_j, T_j,
                                        r_face(k+1), dr_zc, rad, K_face);
        } else {
            // Surface: photospheric (τ=2/3) BC. Integrate τ inward from the
            // outer boundary:  τ(j) = Σ_{i=j..nz-1} κ(ρ_i,T_i) · ρ_i · Δr_i.
            // The photosphere sits at the zone where τ first exceeds 2/3;
            // linearly interpolate in τ between that zone and its neighbor.
            // L_out = 4π r² σ T_phot⁴.
            //
            // Opacity and ρ are Picard-lagged (scalar); T_phot is computed
            // as Dual so the gradient reaches T_k when the photosphere
            // coincides with zone k, and T_{k-1}, T_{k-2}, ... when it sits
            // deeper. In practice for pre-MS stars the photosphere is one
            // or two zones below the last zone, so the Dual coupling reaches
            // those neighbors.
            const double tau_target = 2.0 / 3.0;
            double tau_acc = 0.0;
            int phot_zone = 0;   // zone index where τ first exceeds target
            double tau_below = 0.0;
            // Walk from outermost zone inward
            for (int j = nz - 1; j >= 0; --j) {
                double rho_j_v;
                double T_j_v;
                double r_j_v   = r_face(j).v;
                double r_jp1_v = r_face(j+1).v;
                double dr = r_jp1_v - r_j_v;
                if (dr < 0.0) dr = 0.0;
                if (j == k) {
                    rho_j_v = rho_k.v;
                    T_j_v   = T_k.v;
                } else {
                    double V = (4.188790204786391) * (r_jp1_v*r_jp1_v*r_jp1_v
                                                    - r_j_v*r_j_v*r_j_v);
                    if (V < 1e-30) V = 1e-30;
                    rho_j_v = dm[j] / V;
                    double e_j_v = e_zone(j).v;
                    if (e_j_v < 1e-30) e_j_v = 1e-30;
                    T_j_v = eos.temperature_from_rho_e(
                        rho_j_v > 1e-30 ? rho_j_v : 1e-30, e_j_v);
                }
                double kap = grey_opacity(rho_j_v > 1e-30 ? rho_j_v : 1e-30,
                                          T_j_v > 1.0 ? T_j_v : 1.0, rad.opa);
                double dtau = kap * rho_j_v * dr;
                if (tau_acc + dtau >= tau_target) {
                    phot_zone = j;
                    tau_below = tau_acc;
                    tau_acc = tau_acc + dtau;
                    break;
                }
                tau_acc += dtau;
                phot_zone = j;
                tau_below = tau_acc;
            }
            // Use T of the photospheric zone as T_eff. For now (grey, no
            // interpolation), just take Dual<N> T at that zone — this keeps
            // the gradient correct.
            Dual<N> T_phot(0.0);
            if (phot_zone == k) {
                T_phot = T_k;
            } else {
                // Reconstruct Dual T at phot_zone
                Dual<N> rLp = r_face(phot_zone);
                Dual<N> rRp = r_face(phot_zone + 1);
                Dual<N> Vp  = PI43_D * (rRp*rRp*rRp - rLp*rLp*rLp);
                if (Vp.v < 1e-30) Vp = Dual<N>(1e-30);
                Dual<N> rho_p = Dual<N>(dm[phot_zone]) / Vp;
                Dual<N> e_p   = e_zone(phot_zone);
                if (e_p.v < 1e-30) e_p = Dual<N>(1e-30);
                Dual<N> P_p;
                primitives_dual<N>(rho_p, e_p, eos, P_p, T_phot);
            }
            Dual<N> rs = r_face(k+1);
            Dual<N> A_surf = PI4_D * rs * rs;
            // Soft floor: T_eff⁴ = T_phot⁴ + T_floor⁴. Preserves non-zero
            // gradient even below the floor, so Newton still couples to the
            // surface. Physically: photospheric H⁻ opacity self-regulation
            // prevents T_eff from dropping below ~3000 K for pre-MS stars.
            Dual<N> T2 = T_phot * T_phot;
            Dual<N> T4 = T2 * T2;
            double T_floor = rad.T_phot_floor;
            if (T_floor > 0.0) {
                double f2 = T_floor * T_floor;
                double f4 = f2 * f2;
                T4 = T4 + f4;
            }
            L_out = A_surf * (rad.sigma_sb * T4);
        }

        Dual<N> eps_rad = (L_in - L_out) / fmax(Dual<N>(dm[k]), 1e-30);
        R_e_out = R_e_out + eps_rad;
    }
}

}  // namespace dualR
