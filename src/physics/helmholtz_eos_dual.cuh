#pragma once

// Dual<N> forward-mode AD variant of helm_eval.
//
// Mirrors src/physics/helmholtz_eos.cuh but threads Dual<N> through all
// continuous operations (ρ, T, derived scalars). Table indices remain
// integer (determined by ρ.v, T.v); we assume derivatives do not probe
// across cell boundaries (a valid assumption for small FD-scale AD seeds
// and for the JFNK matvec use case).
//
// Usage:
//   dual::Dual<N> rho_d = dual::Dual<N>::seed(rho, seed_rho_idx);
//   dual::Dual<N> T_d   = dual::Dual<N>::seed(T,   seed_T_idx);
//   HelmStateDual<N> s  = helm_eval_dual<N>(rho_d, T_d, tv);
//   // s.P.g[k] = ∂P/∂(seed_k), etc.

#include "helmholtz_eos.cuh"
#include "dual.cuh"

#ifdef __CUDACC__
#define HEDHD __host__ __device__ inline
#else
#define HEDHD inline
#endif

template <int N>
struct HelmStateDual {
    dual::Dual<N> P;
    dual::Dual<N> e;
    dual::Dual<N> cs;
    dual::Dual<N> dPde_rho;
    dual::Dual<N> gamma1;
    dual::Dual<N> grada;
    dual::Dual<N> chiT;
    dual::Dual<N> chiRho;
    dual::Dual<N> cV;
    dual::Dual<N> cP;
};

// Biquintic Hermite interpolation with Dual T/ρ weights.
// Mirrors helm_h5 structure but with Dual<N> weights. fi[] tabulated values
// stay as scalar doubles (promoted via implicit Dual constructor).
template <int N>
HEDHD dual::Dual<N> helm_h5_dual(const double fi[36],
                                 const dual::Dual<N>& w0t, const dual::Dual<N>& w1t, const dual::Dual<N>& w2t,
                                 const dual::Dual<N>& w0mt, const dual::Dual<N>& w1mt, const dual::Dual<N>& w2mt,
                                 const dual::Dual<N>& w0d,  const dual::Dual<N>& w1d,  const dual::Dual<N>& w2d,
                                 const dual::Dual<N>& w0md, const dual::Dual<N>& w1md, const dual::Dual<N>& w2md)
{
    return fi[ 0]*w0d*w0t + fi[ 1]*w0md*w0t + fi[ 2]*w0d*w0mt + fi[ 3]*w0md*w0mt
         + fi[ 4]*w0d*w1t + fi[ 5]*w0md*w1t + fi[ 6]*w0d*w1mt + fi[ 7]*w0md*w1mt
         + fi[ 8]*w0d*w2t + fi[ 9]*w0md*w2t + fi[10]*w0d*w2mt + fi[11]*w0md*w2mt
         + fi[12]*w1d*w0t + fi[13]*w1md*w0t + fi[14]*w1d*w0mt + fi[15]*w1md*w0mt
         + fi[16]*w2d*w0t + fi[17]*w2md*w0t + fi[18]*w2d*w0mt + fi[19]*w2md*w0mt
         + fi[20]*w1d*w1t + fi[21]*w1md*w1t + fi[22]*w1d*w1mt + fi[23]*w1md*w1mt
         + fi[24]*w2d*w1t + fi[25]*w2md*w1t + fi[26]*w2d*w1mt + fi[27]*w2md*w1mt
         + fi[28]*w1d*w2t + fi[29]*w1md*w2t + fi[30]*w1d*w2mt + fi[31]*w1md*w2mt
         + fi[32]*w2d*w2t + fi[33]*w2md*w2t + fi[34]*w2d*w2mt + fi[35]*w2md*w2mt;
}

template <int N>
HEDHD dual::Dual<N> helm_h3_dual(const double fi[16],
                                 const dual::Dual<N>& w0t,  const dual::Dual<N>& w1t,
                                 const dual::Dual<N>& w0mt, const dual::Dual<N>& w1mt,
                                 const dual::Dual<N>& w0d,  const dual::Dual<N>& w1d,
                                 const dual::Dual<N>& w0md, const dual::Dual<N>& w1md)
{
    return fi[ 0]*w0d*w0t + fi[ 1]*w0md*w0t + fi[ 2]*w0d*w0mt + fi[ 3]*w0md*w0mt
         + fi[ 4]*w0d*w1t + fi[ 5]*w0md*w1t + fi[ 6]*w0d*w1mt + fi[ 7]*w0md*w1mt
         + fi[ 8]*w1d*w0t + fi[ 9]*w1md*w0t + fi[10]*w1d*w0mt + fi[11]*w1md*w0mt
         + fi[12]*w1d*w1t + fi[13]*w1md*w1t + fi[14]*w1d*w1mt + fi[15]*w1md*w1mt;
}

// Quintic / cubic Hermite basis — Dual overloads.
// These are identical polynomials to the scalar helm_psi* but operate on Dual.
// Inline them directly.
template <int N> HEDHD dual::Dual<N> helm_psi0_d (const dual::Dual<N>& z) { return z*z*z * (z*(-6.0*z + 15.0) - 10.0) + 1.0; }
template <int N> HEDHD dual::Dual<N> helm_dpsi0_d(const dual::Dual<N>& z) { return z*z   * (z*(-30.0*z + 60.0) - 30.0); }
template <int N> HEDHD dual::Dual<N> helm_ddpsi0_d(const dual::Dual<N>& z){ return z     * (z*(-120.0*z + 180.0) - 60.0); }
template <int N> HEDHD dual::Dual<N> helm_psi1_d (const dual::Dual<N>& z) { return z     * (z*z*(z*(-3.0*z + 8.0) - 6.0) + 1.0); }
template <int N> HEDHD dual::Dual<N> helm_dpsi1_d(const dual::Dual<N>& z) { return z*z   * (z*(-15.0*z + 32.0) - 18.0) + 1.0; }
template <int N> HEDHD dual::Dual<N> helm_ddpsi1_d(const dual::Dual<N>& z){ return z     * (z*(-60.0*z + 96.0) - 36.0); }
template <int N> HEDHD dual::Dual<N> helm_psi2_d (const dual::Dual<N>& z) { return 0.5*z*z * (z*(z*(-z + 3.0) - 3.0) + 1.0); }
template <int N> HEDHD dual::Dual<N> helm_dpsi2_d(const dual::Dual<N>& z) { return 0.5*z   * (z*(z*(-5.0*z + 12.0) - 9.0) + 2.0); }
template <int N> HEDHD dual::Dual<N> helm_ddpsi2_d(const dual::Dual<N>& z){ return 0.5     * (z*(z*(-20.0*z + 36.0) - 18.0) + 2.0); }
template <int N> HEDHD dual::Dual<N> helm_xpsi0_d (const dual::Dual<N>& z){ return z*z * (2.0*z - 3.0) + 1.0; }
template <int N> HEDHD dual::Dual<N> helm_xdpsi0_d(const dual::Dual<N>& z){ return z * (6.0*z - 6.0); }
template <int N> HEDHD dual::Dual<N> helm_xpsi1_d (const dual::Dual<N>& z){ return z * (z * (z - 2.0) + 1.0); }
template <int N> HEDHD dual::Dual<N> helm_xdpsi1_d(const dual::Dual<N>& z){ return z * (3.0*z - 4.0) + 1.0; }

// Full Dual helm_eval. See helm_eval in helmholtz_eos.cuh for comments; this
// mirrors exactly with Dual replacing double for continuous state.
template <int N>
HEDHD HelmStateDual<N> helm_eval_dual(const dual::Dual<N>& rho_in, const dual::Dual<N>& T_in,
                                      const HelmholtzTableView& tv)
{
    using dual::Dual;
    using dual::sqrt;
    using dual::pow;
    using dual::log10;
    using dual::fmax;
    using dual::fmin;
    using dual::fabs;
    using namespace helm_consts;

    // Saturate to grid bounds (derivatives of saturated range are zero)
    Dual<N> Trun   = T_in;
    Dual<N> rhorun = rho_in;
    if (Trun.v   < 1e3)  Trun   = Dual<N>(1e3);
    if (Trun.v   > 1e13) Trun   = Dual<N>(1e13);
    if (rhorun.v < 1e-12) rhorun = Dual<N>(1e-12);
    if (rhorun.v > 1e15)  rhorun = Dual<N>(1e15);

    double Abar = tv.Abar > 1e-30 ? tv.Abar : 1.0;
    double Zbar = tv.Zbar > 1e-30 ? tv.Zbar : 1.0;
    double ytot1 = 1.0 / Abar;
    double ye    = ytot1 * Zbar;
    if (ye < 1e-16) ye = 1e-16;

    Dual<N> deni  = 1.0 / rhorun;
    Dual<N> tempi = 1.0 / Trun;
    Dual<N> kT    = k_B * Trun;

    // Radiation
    Dual<N> T2 = Trun * Trun;
    Dual<N> T4 = T2 * T2;
    Dual<N> prad = (a_rad / 3.0) * T4;
    Dual<N> erad = 3.0 * prad * deni;
    Dual<N> dpraddT   = 4.0 * prad * tempi;
    Dual<N> dpraddrho = Dual<N>(0.0);
    Dual<N> deraddrho = -erad * deni;
    Dual<N> deraddT   = 3.0 * dpraddT * deni;

    // Ions (ideal gas)
    Dual<N> xni       = (N_A * ytot1) * rhorun;
    Dual<N> dxnidrho  = Dual<N>(N_A * ytot1);
    Dual<N> pion      = xni * kT;
    Dual<N> dpiondrho = dxnidrho * kT;
    Dual<N> dpiondT   = xni * Dual<N>(k_B);
    Dual<N> eion      = 1.5 * pion * deni;
    Dual<N> deiondrho = (1.5 * dpiondrho - eion) * deni;
    Dual<N> deiondT   = 1.5 * dpiondT * deni;

    // Electrons: table lookup. Indices stay integer (use .v).
    Dual<N> din     = ye * rhorun;
    Dual<N> log_T   = log10(Trun);
    Dual<N> log_din = log10(din);

    int jat = (int)((log_T.v - tv.tlo) * tv.dti);
    if (jat < 0) jat = 0;
    if (jat > HELM_JMAX - 2) jat = HELM_JMAX - 2;
    int iat = (int)((log_din.v - tv.dlo) * tv.ddi);
    if (iat < 0) iat = 0;
    if (iat > HELM_IMAX - 2) iat = HELM_IMAX - 2;

    double t_j   = ::pow(10.0, tv.tlo + (double)jat * tv.dt);
    double t_jp1 = ::pow(10.0, tv.tlo + (double)(jat + 1) * tv.dt);
    double d_i   = ::pow(10.0, tv.dlo + (double)iat * tv.dd);
    double d_ip1 = ::pow(10.0, tv.dlo + (double)(iat + 1) * tv.dd);
    double dt_j  = t_jp1 - t_j;
    double dd_i  = d_ip1 - d_i;
    double dti_j = 1.0 / dt_j;
    double ddi_i = 1.0 / dd_i;
    double dt2_j = dt_j * dt_j;
    double dd2_i = dd_i * dd_i;
    double dt2i_j = dti_j * dti_j;

    // Normalized positions (Dual)
    Dual<N> xt = (Trun - Dual<N>(t_j)) * dti_j;
    Dual<N> xd = (din  - Dual<N>(d_i)) * ddi_i;
    if (xt.v < 0.0) xt = Dual<N>(0.0);
    if (xd.v < 0.0) xd = Dual<N>(0.0);
    Dual<N> mxt = 1.0 - xt;
    Dual<N> mxd = 1.0 - xd;

    Dual<N> si0t  = helm_psi0_d(xt);
    Dual<N> si1t  = helm_psi1_d(xt) * dt_j;
    Dual<N> si2t  = helm_psi2_d(xt) * dt2_j;
    Dual<N> si0mt = helm_psi0_d(mxt);
    Dual<N> si1mt = Dual<N>(-1.0) * helm_psi1_d(mxt) * dt_j;
    Dual<N> si2mt = helm_psi2_d(mxt) * dt2_j;

    Dual<N> si0d  = helm_psi0_d(xd);
    Dual<N> si1d  = helm_psi1_d(xd) * dd_i;
    Dual<N> si2d  = helm_psi2_d(xd) * dd2_i;
    Dual<N> si0md = helm_psi0_d(mxd);
    Dual<N> si1md = Dual<N>(-1.0) * helm_psi1_d(mxd) * dd_i;
    Dual<N> si2md = helm_psi2_d(mxd) * dd2_i;

    Dual<N> dsi0t  = helm_dpsi0_d(xt)  * dti_j;
    Dual<N> dsi1t  = helm_dpsi1_d(xt);
    Dual<N> dsi2t  = helm_dpsi2_d(xt)  * dt_j;
    Dual<N> dsi0mt = Dual<N>(-1.0) * helm_dpsi0_d(mxt) * dti_j;
    Dual<N> dsi1mt = helm_dpsi1_d(mxt);
    Dual<N> dsi2mt = Dual<N>(-1.0) * helm_dpsi2_d(mxt) * dt_j;

    Dual<N> dsi0d  = helm_dpsi0_d(xd)  * ddi_i;
    Dual<N> dsi1d  = helm_dpsi1_d(xd);
    Dual<N> dsi2d  = helm_dpsi2_d(xd)  * dd_i;
    Dual<N> dsi0md = Dual<N>(-1.0) * helm_dpsi0_d(mxd) * ddi_i;
    Dual<N> dsi1md = helm_dpsi1_d(mxd);
    Dual<N> dsi2md = Dual<N>(-1.0) * helm_dpsi2_d(mxd) * dd_i;

    double fi[36];
    int c00 = helm_idx2d(jat,   iat);
    int c10 = helm_idx2d(jat,   iat + 1);
    int c01 = helm_idx2d(jat+1, iat);
    int c11 = helm_idx2d(jat+1, iat + 1);
    fi[ 0] = tv.f[c00];    fi[ 1] = tv.f[c10];    fi[ 2] = tv.f[c01];    fi[ 3] = tv.f[c11];
    fi[ 4] = tv.ft[c00];   fi[ 5] = tv.ft[c10];   fi[ 6] = tv.ft[c01];   fi[ 7] = tv.ft[c11];
    fi[ 8] = tv.ftt[c00];  fi[ 9] = tv.ftt[c10];  fi[10] = tv.ftt[c01];  fi[11] = tv.ftt[c11];
    fi[12] = tv.fd[c00];   fi[13] = tv.fd[c10];   fi[14] = tv.fd[c01];   fi[15] = tv.fd[c11];
    fi[16] = tv.fdd[c00];  fi[17] = tv.fdd[c10];  fi[18] = tv.fdd[c01];  fi[19] = tv.fdd[c11];
    fi[20] = tv.fdt[c00];  fi[21] = tv.fdt[c10];  fi[22] = tv.fdt[c01];  fi[23] = tv.fdt[c11];
    fi[24] = tv.fddt[c00]; fi[25] = tv.fddt[c10]; fi[26] = tv.fddt[c01]; fi[27] = tv.fddt[c11];
    fi[28] = tv.fdtt[c00]; fi[29] = tv.fdtt[c10]; fi[30] = tv.fdtt[c01]; fi[31] = tv.fdtt[c11];
    fi[32] = tv.fddtt[c00];fi[33] = tv.fddtt[c10];fi[34] = tv.fddtt[c01];fi[35] = tv.fddtt[c11];

    Dual<N> free_e = helm_h5_dual<N>(fi,
                          si0t, si1t, si2t, si0mt, si1mt, si2mt,
                          si0d, si1d, si2d, si0md, si1md, si2md);
    Dual<N> df_d = helm_h5_dual<N>(fi,
                          si0t, si1t, si2t, si0mt, si1mt, si2mt,
                          dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);
    Dual<N> df_t = helm_h5_dual<N>(fi,
                          dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt,
                          si0d, si1d, si2d, si0md, si1md, si2md);
    Dual<N> df_dt = helm_h5_dual<N>(fi,
                           dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt,
                           dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

    // Bicubic weights for dpdf
    Dual<N> xsi0t  = helm_xpsi0_d(xt);
    Dual<N> xsi1t  = helm_xpsi1_d(xt)  * dt_j;
    Dual<N> xsi0mt = helm_xpsi0_d(mxt);
    Dual<N> xsi1mt = Dual<N>(-1.0) * helm_xpsi1_d(mxt) * dt_j;
    Dual<N> xsi0d  = helm_xpsi0_d(xd);
    Dual<N> xsi1d  = helm_xpsi1_d(xd)  * dd_i;
    Dual<N> xsi0md = helm_xpsi0_d(mxd);
    Dual<N> xsi1md = Dual<N>(-1.0) * helm_xpsi1_d(mxd) * dd_i;

    double fi16[16];
    fi16[ 0] = tv.dpdf[c00];   fi16[ 1] = tv.dpdf[c10];   fi16[ 2] = tv.dpdf[c01];   fi16[ 3] = tv.dpdf[c11];
    fi16[ 4] = tv.dpdft[c00];  fi16[ 5] = tv.dpdft[c10];  fi16[ 6] = tv.dpdft[c01];  fi16[ 7] = tv.dpdft[c11];
    fi16[ 8] = tv.dpdfd[c00];  fi16[ 9] = tv.dpdfd[c10];  fi16[10] = tv.dpdfd[c01];  fi16[11] = tv.dpdfd[c11];
    fi16[12] = tv.dpdfdt[c00]; fi16[13] = tv.dpdfdt[c10]; fi16[14] = tv.dpdfdt[c01]; fi16[15] = tv.dpdfdt[c11];
    Dual<N> dpepdd = helm_h3_dual<N>(fi16,
                            xsi0t, xsi1t, xsi0mt, xsi1mt,
                            xsi0d, xsi1d, xsi0md, xsi1md);
    if (dpepdd.v < 1e-30) dpepdd = Dual<N>(1e-30);
    dpepdd = dpepdd * ye;

    Dual<N> din2 = din * din;
    Dual<N> pele   = din2 * df_d;
    Dual<N> dpepdT = din2 * df_dt;

    Dual<N> sele = Dual<N>(-1.0) * df_t * ye;
    Dual<N> eele = ye * free_e + Trun * sele;

    // Total
    Dual<N> P_gas = pion + pele;
    Dual<N> e_gas = eion + eele;
    Dual<N> P_tot = prad + P_gas;
    Dual<N> e_tot = erad + e_gas;

    Dual<N> dP_dT_gas   = dpiondT + dpepdT;
    Dual<N> dP_drho_gas = dpiondrho + dpepdd;

    // d_e/d_T for electrons — derived from Helmholtz second derivatives like in scalar version.
    // From scalar: deepdT = Trun * dsepdT_scalar;  dsepdT_scalar = -df_tt * ye.
    // We need df_tt here (skipped above via a hack). Compute it properly:
    Dual<N> ddsi0t  = helm_ddpsi0_d(xt)  * dt2i_j;
    Dual<N> ddsi1t  = helm_ddpsi1_d(xt)  * dti_j;
    Dual<N> ddsi2t  = helm_ddpsi2_d(xt);
    Dual<N> ddsi0mt = helm_ddpsi0_d(mxt) * dt2i_j;
    Dual<N> ddsi1mt = Dual<N>(-1.0) * helm_ddpsi1_d(mxt) * dti_j;
    Dual<N> ddsi2mt = helm_ddpsi2_d(mxt);
    Dual<N> df_tt = helm_h5_dual<N>(fi,
                                    ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
                                    si0d, si1d, si2d, si0md, si1md, si2md);
    Dual<N> dsepdT_real = Dual<N>(-1.0) * df_tt * ye;
    Dual<N> deepdT = Trun * dsepdT_real;

    Dual<N> de_dT_gas = deiondT + deepdT;

    Dual<N> dP_dT_tot   = dpraddT + dP_dT_gas;
    Dual<N> dP_drho_tot = dpraddrho + dP_drho_gas;
    Dual<N> de_dT_tot   = deraddT + de_dT_gas;
    Dual<N> cv_tot      = de_dT_tot;

    Dual<N> chiT = (Trun / P_tot) * dP_dT_tot;
    Dual<N> chiD = (rhorun / P_tot) * dP_drho_tot;
    Dual<N> xfac = (P_tot * deni) * chiT / (Trun * cv_tot);
    Dual<N> gam1 = chiT * xfac + chiD;
    if (gam1.v < 1.0 + 1e-8) gam1 = Dual<N>(1.0 + 1e-8);
    Dual<N> cs = sqrt(gam1 * P_tot * deni);

    Dual<N> dPde_rho = (cv_tot.v > 1e-30)
                      ? (dP_dT_tot / cv_tot)
                      : (dP_dT_tot / Dual<N>(1e-30));

    Dual<N> gam3_m1 = (cv_tot.v > 1e-30 && chiD.v > 1e-30)
                     ? (P_tot * deni) * chiT / (Trun * cv_tot * chiD)
                     : Dual<N>(0.0);
    Dual<N> grada = gam3_m1 / gam1;
    Dual<N> cP_val = cv_tot + ((chiD.v > 1e-30)
                               ? (P_tot * chiT * gam3_m1) / (rhorun * Trun * chiD)
                               : Dual<N>(0.0));

    HelmStateDual<N> s;
    s.P = P_tot;
    s.e = e_tot;
    s.cs = cs;
    s.dPde_rho = dPde_rho;
    s.gamma1 = gam1;
    s.grada  = grada;
    s.chiT   = chiT;
    s.chiRho = chiD;
    s.cV     = cv_tot;
    s.cP     = cP_val;
    return s;
}

// Dual T_from_rho_e: given a converged scalar T_0 found via the non-Dual
// helm_T_from_rho_e, "lift" to Dual via one implicit-function-theorem step.
// Input:  rho_d (Dual), e_target (Dual), T_0 scalar (from scalar Newton)
// Output: Dual<N> T with value T_0 and gradients from ρ, e.
//
// Implicit equation F(T, ρ, e_tgt) = e_helm(ρ, T) − e_tgt = 0.
// At convergence (F.v≈0), by implicit function theorem:
//   dT/dρ    = −(∂e_helm/∂ρ) / (∂e_helm/∂T)
//   dT/de    =  1 / (∂e_helm/∂T)
// We obtain these automatically from Dual arithmetic: evaluate e_helm_dual at
// rho_d, Dual(T_0) — gradient carries ∂e/∂ρ · dρ (T_0 is constant). Then:
//   residual = e_target − e_helm_dual.e    (value ≈ 0, grads = de_tgt − ∂e/∂ρ dρ)
//   T_dual   = T_0 + residual / cv         (cv = ∂e/∂T, scalar ≈ cv_tot.v)
template <int N>
HEDHD dual::Dual<N> helm_T_from_rho_e_dual(const dual::Dual<N>& rho_d,
                                           const dual::Dual<N>& e_target,
                                           const HelmholtzTableView& tv,
                                           double T_guess = -1.0)
{
    // Scalar Newton — exactly helm_T_from_rho_e but explicit here so we can
    // re-use the converged T_0 and the cv at T_0.
    double T = (T_guess > 0.0) ? T_guess : 1e4;
    for (int it = 0; it < 40; ++it) {
        HelmState s = helm_eval(rho_d.v, T, tv);
        double f = s.e - e_target.v;
        if (std::fabs(f) < 1e-10 * std::fabs(e_target.v) + 1e-20) break;
        double dT_h = 1e-4 * T + 1.0;
        HelmState sp = helm_eval(rho_d.v, T + dT_h, tv);
        double df = (sp.e - s.e) / dT_h;
        if (std::fabs(df) < 1e-30) df = 1e-30;
        double T_new = T - f / df;
        if (T_new < 0.5 * T) T_new = 0.5 * T;
        if (T_new > 2.0 * T) T_new = 2.0 * T;
        if (T_new < 1.0)     T_new = 1.0;
        T = T_new;
    }
    // Lift to Dual at converged T_0 = T
    dual::Dual<N> T0_const(T);
    HelmStateDual<N> sd = helm_eval_dual<N>(rho_d, T0_const, tv);
    // residual: e_target − e_helm(rho_d, T0)
    dual::Dual<N> residual = e_target - sd.e;
    double cv = sd.cV.v;
    if (cv < 1e-30) cv = 1e-30;
    dual::Dual<N> T_d = residual / cv;
    T_d.v = T;  // value is the scalar converged T
    return T_d;
}

#undef HEDHD
