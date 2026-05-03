#pragma once

// Helmholtz free-energy EOS (Timmes & Swesty 2000, cococubed.com).
//
// Public-domain community-standard EOS covering partial degenerate electrons,
// radiation, and ideal ions. Sits alongside our other EOS types (IDEAL,
// IDEAL_RAD, PRE_MS) as the high-fidelity option for stellar interiors.
//
// Table layout (following Timmes):
//   log10 T  grid, imax=211  (T = 1e3 … 1e13 K)
//   log10 ρ  grid, jmax=71   (ρ = 1e-12 … 1e15 g/cc)
//   9 tabulated functions of (log T, log ρ):
//     f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt
//   plus two 1D tables (positron, coulomb) — skipped in this minimal port.
//
// Biquintic Hermite interpolation in (log T, log ρ) gives C² continuous
// pressure + its derivatives — essential for JFNK Newton stability.
//
// GPU placement:
//   Tables are malloc'd via cudaMalloc and pinned to L2 persisting cache
//   (sm_80+; on sm_89 RTX 40-series we get up to 44 MB persisting). The
//   5 MB Helmholtz table stays in L2 across every BE solve, giving us
//   ~50× DRAM bandwidth for JFNK GMRES iterations.

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define HE_HD __host__ __device__
#else
#define HE_HD
#endif

// Resolution of the cococubed table.
// NOTE Timmes convention: i indexes log ρ (inner loop), j indexes log T
// (outer loop). Two resolutions ship with helmholtz.tbz:
//   "standard"    : imax=271, jmax=101    (file ~15 MB)
//   "twice dense" : imax=541, jmax=201    (file ~60 MB, current default)
// We hard-code the twice-dense resolution since that is what the
// distributed helm_table.dat provides.
static constexpr int HELM_IMAX = 541;   // log ρ points (inner)
static constexpr int HELM_JMAX = 201;   // log T points (outer)

// POD view of table arrays on device. Passed by value to kernels.
// Pointers live in GPU memory; lifetime owned by HelmholtzTable (below).
//
// Layout: each 2D table stored row-major as [j * IMAX + i] where
//   i ∈ [0, IMAX) indexes log ρ (inner loop on read)
//   j ∈ [0, JMAX) indexes log T (outer loop on read)
// This matches Timmes' Fortran helm_table.dat file order directly.
struct HelmholtzTableView {
    // ---- Table 1: Helmholtz free energy (9 entries per grid point) ----
    const double* f;       // free energy
    const double* fd;      // ∂f/∂ρ
    const double* ft;      // ∂f/∂T
    const double* fdd;     // ∂²f/∂ρ²
    const double* ftt;     // ∂²f/∂T²
    const double* fdt;     // ∂²f/∂ρ ∂T
    const double* fddt;    // ∂³f/∂ρ² ∂T
    const double* fdtt;    // ∂³f/∂ρ ∂T²
    const double* fddtt;   // ∂⁴f/∂ρ² ∂T²
    // ---- Table 2: dP/dρ and its derivatives (4 entries) ----
    const double* dpdf;
    const double* dpdfd;
    const double* dpdft;
    const double* dpdfdt;
    // ---- Table 3: electron chemical potential (4 entries) ----
    const double* ef;
    const double* efd;
    const double* eft;
    const double* efdt;
    // ---- Table 4: electron-positron number density (4 entries) ----
    const double* xf;
    const double* xfd;
    const double* xft;
    const double* xfdt;

    // Grid bookkeeping — Timmes helmholtz.f90 uses:
    //   log T ∈ [3, 13]  with jmax points (outer index j)
    //   log ρ ∈ [-12, 15] with imax points (inner index i)
    double tlo, thi, dt, dti;   // log10 T grid (j axis)
    double dlo, dhi, dd, ddi;   // log10 ρ grid (i axis)

    // Composition. Helmholtz is composition-agnostic; Abar/Zbar only appear
    // in the ion + coulomb parts we layer on top of the electron table.
    // For pure ionized H:  Abar=1, Zbar=1.  Solar ~0.73X+0.25Y+Z:
    // Abar≈1.25, Zbar≈1.14.
    double Abar;
    double Zbar;
};

// Host-side owner that allocates + uploads + pins in L2.
struct HelmholtzTable {
    double* d_data = nullptr;   // single allocation holding all 9 tables
    size_t  bytes  = 0;
    HelmholtzTableView view;
    cudaStream_t stream = 0;    // stream used for L2 persisting window

    // Load the cococubed ASCII "helm_table.dat" (or binary .bin), upload to
    // GPU, and set up L2 persisting cache window on `stream`. Returns 0 on
    // success, non-zero otherwise.
    int load(const char* ascii_path, const char* bin_cache_path, cudaStream_t s = 0);

    // Reset L2 persisting config + free device memory.
    void destroy();
};

// Device-side interpolation helpers.
//
// Biquintic interpolation per Timmes "helmeos.f" uses 36 coefficients; the
// minimal reduction is to use the bicubic Hermite form with (f, fd, ft, fdt)
// at the 4 corners of the enclosing rectangle.  That gives C¹ continuous
// result. For C² we'd need fdd/ftt/fddtt too.
//
// First pass (this file): bicubic Hermite only. Biquintic upgrade kept for
// after baseline correctness is proven.
HE_HD inline int helm_index_log_T(const HelmholtzTableView& tv, double log_T) {
    int i = (int)((log_T - tv.tlo) * tv.dti);
    if (i < 0) i = 0;
    if (i > HELM_IMAX - 2) i = HELM_IMAX - 2;
    return i;
}
HE_HD inline int helm_index_log_rho(const HelmholtzTableView& tv, double log_rho) {
    int j = (int)((log_rho - tv.dlo) * tv.ddi);
    if (j < 0) j = 0;
    if (j > HELM_JMAX - 2) j = HELM_JMAX - 2;
    return j;
}
// Row-major index: j (log T) is the outer axis, i (log ρ) inner.
HE_HD inline int helm_idx2d(int j, int i) { return j * HELM_IMAX + i; }

// ============================================================================
// Quintic Hermite basis functions (Timmes 2000 eqs. 12-14).
// psi0 matches value, psi1 matches 1st derivative, psi2 matches 2nd deriv.
// All evaluated at z ∈ [0, 1] — the normalized position within the cell.
// ============================================================================
HE_HD inline double helm_psi0(double z)   { return z*z*z * (z*(-6.0*z + 15.0) - 10.0) + 1.0; }
HE_HD inline double helm_dpsi0(double z)  { return z*z   * (z*(-30.0*z + 60.0) - 30.0); }
HE_HD inline double helm_ddpsi0(double z) { return z     * (z*(-120.0*z + 180.0) - 60.0); }

HE_HD inline double helm_psi1(double z)   { return z     * (z*z*(z*(-3.0*z + 8.0) - 6.0) + 1.0); }
HE_HD inline double helm_dpsi1(double z)  { return z*z   * (z*(-15.0*z + 32.0) - 18.0) + 1.0; }
HE_HD inline double helm_ddpsi1(double z) { return z     * (z*(-60.0*z + 96.0) - 36.0); }

HE_HD inline double helm_psi2(double z)   { return 0.5*z*z * (z*(z*(-z + 3.0) - 3.0) + 1.0); }
HE_HD inline double helm_dpsi2(double z)  { return 0.5*z   * (z*(z*(-5.0*z + 12.0) - 9.0) + 2.0); }
HE_HD inline double helm_ddpsi2(double z) { return 0.5     * (z*(z*(-20.0*z + 36.0) - 18.0) + 2.0); }

// Cubic Hermite basis (used for dpdf, ef, xf — which only carry ∂/∂log T,
// ∂/∂log ρ, cross — no second derivatives on the table).
HE_HD inline double helm_xpsi0(double z)  { return z*z * (2.0*z - 3.0) + 1.0; }
HE_HD inline double helm_xdpsi0(double z) { return z * (6.0*z - 6.0); }
HE_HD inline double helm_xpsi1(double z)  { return z * (z * (z - 2.0) + 1.0); }
HE_HD inline double helm_xdpsi1(double z) { return z * (3.0*z - 4.0) + 1.0; }

// ============================================================================
// Biquintic Hermite interpolation (36-term expansion, Timmes eq. 15).
//
// Inputs:
//   fi[36]     : stacked table values at the 4 corners of the enclosing cell
//   w0t, w1t, w2t, w0mt, w1mt, w2mt : T-direction weight functions
//   w0d, w1d, w2d, w0md, w1md, w2md : ρ-direction weight functions
// Output: single scalar (either free energy or one of its derivatives,
//   depending on which weight set is passed).
// ============================================================================
HE_HD inline double helm_h5(const double fi[36],
                            double w0t, double w1t, double w2t,
                            double w0mt, double w1mt, double w2mt,
                            double w0d, double w1d, double w2d,
                            double w0md, double w1md, double w2md)
{
    return fi[0]*w0d*w0t + fi[1]*w0md*w0t + fi[2]*w0d*w0mt + fi[3]*w0md*w0mt
         + fi[4]*w0d*w1t + fi[5]*w0md*w1t + fi[6]*w0d*w1mt + fi[7]*w0md*w1mt
         + fi[8]*w0d*w2t + fi[9]*w0md*w2t + fi[10]*w0d*w2mt + fi[11]*w0md*w2mt
         + fi[12]*w1d*w0t + fi[13]*w1md*w0t + fi[14]*w1d*w0mt + fi[15]*w1md*w0mt
         + fi[16]*w2d*w0t + fi[17]*w2md*w0t + fi[18]*w2d*w0mt + fi[19]*w2md*w0mt
         + fi[20]*w1d*w1t + fi[21]*w1md*w1t + fi[22]*w1d*w1mt + fi[23]*w1md*w1mt
         + fi[24]*w2d*w1t + fi[25]*w2md*w1t + fi[26]*w2d*w1mt + fi[27]*w2md*w1mt
         + fi[28]*w1d*w2t + fi[29]*w1md*w2t + fi[30]*w1d*w2mt + fi[31]*w1md*w2mt
         + fi[32]*w2d*w2t + fi[33]*w2md*w2t + fi[34]*w2d*w2mt + fi[35]*w2md*w2mt;
}

// Bicubic Hermite (16-term).
HE_HD inline double helm_h3(const double fi[16],
                            double w0t, double w1t, double w0mt, double w1mt,
                            double w0d, double w1d, double w0md, double w1md)
{
    return fi[0]*w0d*w0t + fi[1]*w0md*w0t + fi[2]*w0d*w0mt + fi[3]*w0md*w0mt
         + fi[4]*w0d*w1t + fi[5]*w0md*w1t + fi[6]*w0d*w1mt + fi[7]*w0md*w1mt
         + fi[8]*w1d*w0t + fi[9]*w1md*w0t + fi[10]*w1d*w0mt + fi[11]*w1md*w0mt
         + fi[12]*w1d*w1t + fi[13]*w1md*w1t + fi[14]*w1d*w1mt + fi[15]*w1md*w1mt;
}

// ============================================================================
// Physical constants (cgs). Matches Timmes const.dek.
// ============================================================================
namespace helm_consts {
    constexpr double k_B   = 1.380650424e-16;     // erg/K
    constexpr double N_A   = 6.0221417930e23;     // Avogadro
    constexpr double h_P   = 6.6260689633e-27;    // Planck
    constexpr double m_u   = 1.66053878283e-24;   // atomic mass unit
    constexpr double c     = 2.99792458e10;       // cm/s
    constexpr double a_rad = 7.5657e-15;          // radiation constant erg/cm³/K⁴
    constexpr double sigma = 5.6704e-5;           // Stefan-Boltzmann
    constexpr double q_e   = 4.8032042712e-10;    // electron charge (esu)
    constexpr double PI    = 3.141592653589793;
}

// ============================================================================
// Full Helmholtz EOS evaluation at (ρ, T) given composition (Abar, Zbar).
//
// Returns the full thermodynamic state needed by stellar hydro:
//   P       = pressure
//   e       = specific internal energy
//   cs      = adiabatic sound speed
//   dP/de|ρ = needed for Jacobian / JFNK
//
// Reference: Timmes & Swesty 2000 (public helmeos.f90 source, cococubed.com).
//
// Structure:
//   1. Radiation: prad = a T⁴/3, erad = 3 prad/ρ
//   2. Ions: ideal gas with Abar
//   3. Electrons: table lookup via biquintic Hermite on (f, fd, ft, ...)
//   4. Coulomb: Yakovlev-Shalybkov uniform background correction (small;
//      we skip in the first pass — introduces ~1% error for white-dwarf
//      interiors but ~0.01% for pre-MS / MS stars)
//   5. Total = sum of all four
// ============================================================================
struct HelmState {
    double P;          // total pressure
    double e;          // total specific internal energy
    double cs;         // adiabatic sound speed
    double dPde_rho;   // ∂P/∂e at fixed ρ — needed for Jacobian scaling
    double gamma1;     // Γ₁ = (∂ln P / ∂ln ρ)_S
    // ∇_ad = (∂ ln T / ∂ ln P)_S, a.k.a. "grada" in MESA. Derived from the
    // textbook identity ∇_ad = (Γ₃ − 1) / Γ₁ with
    //   Γ₃ − 1 = (P · χ_T) / (ρ · T · c_V · χ_ρ)     (Cox & Giuli §9.17)
    // so it automatically reflects radiation + ion + electron contributions.
    double grada;
    // Additional thermodynamic derivatives consumers may want (MLT, sound
    // speed decomposition, Jacobian builds). Filled at essentially zero
    // cost since helm_eval already has them in local scope.
    double chiT;       // χ_T = (∂ln P / ∂ln T)_ρ
    double chiRho;     // χ_ρ = (∂ln P / ∂ln ρ)_T
    double cV;         // specific heat at constant volume [erg/g/K]
    double cP;         // specific heat at constant pressure [erg/g/K]
};

HE_HD inline HelmState helm_eval(double rho, double T,
                                 const HelmholtzTableView& tv)
{
    using namespace helm_consts;

    // ---- Saturate inputs to grid bounds ----
    double Trun = T;
    double rhorun = rho;
    if (Trun   < 1e3)  Trun   = 1e3;
    if (Trun   > 1e13) Trun   = 1e13;
    if (rhorun < 1e-12) rhorun = 1e-12;
    if (rhorun > 1e15)  rhorun = 1e15;

    double Abar = tv.Abar > 1e-30 ? tv.Abar : 1.0;
    double Zbar = tv.Zbar > 1e-30 ? tv.Zbar : 1.0;
    double ytot1 = 1.0 / Abar;
    double ye    = ytot1 * Zbar;
    if (ye < 1e-16) ye = 1e-16;

    double deni  = 1.0 / rhorun;
    double tempi = 1.0 / Trun;
    double kT    = k_B * Trun;

    // ---- Radiation ----
    double T2 = Trun * Trun;
    double T4 = T2 * T2;
    double prad = a_rad * T4 / 3.0;
    double erad = 3.0 * prad * deni;
    double dpraddT = 4.0 * prad * tempi;
    double dpraddrho = 0.0;
    double deraddrho = -erad * deni;
    double deraddT   = 3.0 * dpraddT * deni;

    // ---- Ions (ideal gas) ----
    double xni     = N_A * ytot1 * rhorun;
    double dxnidrho= N_A * ytot1;
    double pion    = xni * kT;
    double dpiondrho = dxnidrho * kT;
    double dpiondT   = xni * k_B;
    double eion      = 1.5 * pion * deni;
    double deiondrho = (1.5 * dpiondrho - eion) * deni;
    double deiondT   = 1.5 * dpiondT * deni;

    // ---- Electrons: table lookup ----
    double din = ye * rhorun;
    double log_T   = log10(Trun);
    double log_din = log10(din);

    // Index into grid: j = T axis, i = ρ axis
    int jat = (int)((log_T - tv.tlo) * tv.dti);
    if (jat < 0) jat = 0;
    if (jat > HELM_JMAX - 2) jat = HELM_JMAX - 2;
    int iat = (int)((log_din - tv.dlo) * tv.ddi);
    if (iat < 0) iat = 0;
    if (iat > HELM_IMAX - 2) iat = HELM_IMAX - 2;

    // Grid values at corners (linear in log space, but table stores values at
    // literal T and ρ values, so we need the corners in real units).
    double t_j    = pow(10.0, tv.tlo + (double)jat * tv.dt);
    double t_jp1  = pow(10.0, tv.tlo + (double)(jat + 1) * tv.dt);
    double d_i    = pow(10.0, tv.dlo + (double)iat * tv.dd);
    double d_ip1  = pow(10.0, tv.dlo + (double)(iat + 1) * tv.dd);
    double dt_j   = t_jp1 - t_j;
    double dd_i   = d_ip1 - d_i;
    double dti_j  = 1.0 / dt_j;
    double ddi_i  = 1.0 / dd_i;
    double dt2_j  = dt_j * dt_j;
    double dd2_i  = dd_i * dd_i;
    double dt2i_j = dti_j * dti_j;

    // Normalized positions
    double xt  = (Trun - t_j) * dti_j;
    double xd  = (din  - d_i) * ddi_i;
    if (xt < 0.0) xt = 0.0;
    if (xd < 0.0) xd = 0.0;
    double mxt = 1.0 - xt;
    double mxd = 1.0 - xd;

    // Quintic weights + derivatives
    double si0t  = helm_psi0(xt);
    double si1t  = helm_psi1(xt)  * dt_j;
    double si2t  = helm_psi2(xt)  * dt2_j;
    double si0mt = helm_psi0(mxt);
    double si1mt = -helm_psi1(mxt) * dt_j;
    double si2mt = helm_psi2(mxt) * dt2_j;

    double si0d  = helm_psi0(xd);
    double si1d  = helm_psi1(xd)  * dd_i;
    double si2d  = helm_psi2(xd)  * dd2_i;
    double si0md = helm_psi0(mxd);
    double si1md = -helm_psi1(mxd) * dd_i;
    double si2md = helm_psi2(mxd) * dd2_i;

    double dsi0t  = helm_dpsi0(xt)  * dti_j;
    double dsi1t  = helm_dpsi1(xt);
    double dsi2t  = helm_dpsi2(xt)  * dt_j;
    double dsi0mt = -helm_dpsi0(mxt) * dti_j;
    double dsi1mt = helm_dpsi1(mxt);
    double dsi2mt = -helm_dpsi2(mxt) * dt_j;

    double dsi0d  = helm_dpsi0(xd)  * ddi_i;
    double dsi1d  = helm_dpsi1(xd);
    double dsi2d  = helm_dpsi2(xd)  * dd_i;
    double dsi0md = -helm_dpsi0(mxd) * ddi_i;
    double dsi1md = helm_dpsi1(mxd);
    double dsi2md = -helm_dpsi2(mxd) * dd_i;

    double ddsi0t  = helm_ddpsi0(xt)  * dt2i_j;
    double ddsi1t  = helm_ddpsi1(xt)  * dti_j;
    double ddsi2t  = helm_ddpsi2(xt);
    double ddsi0mt = helm_ddpsi0(mxt) * dt2i_j;
    double ddsi1mt = -helm_ddpsi1(mxt) * dti_j;
    double ddsi2mt = helm_ddpsi2(mxt);

    // Load fi[36] from the 9 f-family tables at (iat, jat), (iat+1, jat),
    // (iat, jat+1), (iat+1, jat+1).
    // Matching the Fortran loop exactly:
    //   fi(1..4)   = f at 4 corners
    //   fi(5..8)   = ft
    //   fi(9..12)  = ftt
    //   fi(13..16) = fd
    //   fi(17..20) = fdd
    //   fi(21..24) = fdt
    //   fi(25..28) = fddt
    //   fi(29..32) = fdtt
    //   fi(33..36) = fddtt
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

    double free = helm_h5(fi,
                          si0t, si1t, si2t, si0mt, si1mt, si2mt,
                          si0d, si1d, si2d, si0md, si1md, si2md);
    double df_d = helm_h5(fi,
                          si0t, si1t, si2t, si0mt, si1mt, si2mt,
                          dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);
    double df_t = helm_h5(fi,
                          dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt,
                          si0d, si1d, si2d, si0md, si1md, si2md);
    double df_tt = helm_h5(fi,
                           ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
                           si0d, si1d, si2d, si0md, si1md, si2md);
    double df_dt = helm_h5(fi,
                           dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt,
                           dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

    // Bicubic weights for dpdf
    double xsi0t  = helm_xpsi0(xt);
    double xsi1t  = helm_xpsi1(xt)  * dt_j;
    double xsi0mt = helm_xpsi0(mxt);
    double xsi1mt = -helm_xpsi1(mxt) * dt_j;
    double xsi0d  = helm_xpsi0(xd);
    double xsi1d  = helm_xpsi1(xd)  * dd_i;
    double xsi0md = helm_xpsi0(mxd);
    double xsi1md = -helm_xpsi1(mxd) * dd_i;

    // dpdf table
    double fi16[16];
    fi16[ 0] = tv.dpdf[c00];   fi16[ 1] = tv.dpdf[c10];   fi16[ 2] = tv.dpdf[c01];   fi16[ 3] = tv.dpdf[c11];
    fi16[ 4] = tv.dpdft[c00];  fi16[ 5] = tv.dpdft[c10];  fi16[ 6] = tv.dpdft[c01];  fi16[ 7] = tv.dpdft[c11];
    fi16[ 8] = tv.dpdfd[c00];  fi16[ 9] = tv.dpdfd[c10];  fi16[10] = tv.dpdfd[c01];  fi16[11] = tv.dpdfd[c11];
    fi16[12] = tv.dpdfdt[c00]; fi16[13] = tv.dpdfdt[c10]; fi16[14] = tv.dpdfdt[c01]; fi16[15] = tv.dpdfdt[c11];
    double dpepdd = helm_h3(fi16,
                            xsi0t, xsi1t, xsi0mt, xsi1mt,
                            xsi0d, xsi1d, xsi0md, xsi1md);
    if (dpepdd < 1e-30) dpepdd = 1e-30;
    dpepdd *= ye;

    // Electron-positron thermodynamics (from Helmholtz free energy derivatives)
    double din2 = din * din;
    double pele   = din2 * df_d;
    double dpepdT = din2 * df_dt;

    double ye2 = ye * ye;
    double sele   = -df_t * ye;
    double dsepdT = -df_tt * ye;

    double eele   = ye * free + Trun * sele;
    double deepdT = Trun * dsepdT;
    double deepdrho = ye2 * df_d + Trun * (-df_dt * ye2);

    // ---- Total ----
    double P_gas = pion + pele;
    double e_gas = eion + eele;
    double P_tot = prad + P_gas;
    double e_tot = erad + e_gas;

    // Derivatives needed for Γ₁ / cs (Timmes uses chit, chid, cv, gam1)
    double dP_dT_gas    = dpiondT + dpepdT;
    double dP_drho_gas  = dpiondrho + dpepdd;
    double de_dT_gas    = deiondT + deepdT;

    double dP_dT_tot   = dpraddT + dP_dT_gas;
    double dP_drho_tot = dpraddrho + dP_drho_gas;
    double de_dT_tot   = deraddT + de_dT_gas;
    double cv_tot      = de_dT_tot;

    double chiT = (Trun / P_tot) * dP_dT_tot;
    double chiD = (rhorun / P_tot) * dP_drho_tot;
    double xfac = (P_tot * deni) * chiT / (Trun * cv_tot);
    double gam1 = chiT * xfac + chiD;
    if (gam1 < 1.0 + 1e-8) gam1 = 1.0 + 1e-8;
    double cs = sqrt(gam1 * P_tot * deni);

    // ∂P/∂e|ρ needed for Jacobian — chain via T: dP/de|ρ = (dP/dT)/(de/dT)
    double dPde_rho = dP_dT_tot / (cv_tot > 1e-30 ? cv_tot : 1e-30);

    // ∇_ad via Cox & Giuli textbook identity
    //   Γ_3 − 1 = (P · χ_T) / (ρ · T · c_V · χ_ρ)
    //   ∇_ad   = (Γ_3 − 1) / Γ_1
    double gam3_m1 = (cv_tot > 1e-30 && chiD > 1e-30)
                     ? (P_tot * deni) * chiT / (Trun * cv_tot * chiD)
                     : 0.0;
    double grada_val = gam3_m1 / gam1;
    // c_P = c_V + P·χ_T·(Γ_3 − 1) / (ρ·T·χ_ρ)  ← Mayer's relation generalised
    double cP_val = cv_tot
                  + (chiD > 1e-30
                     ? (P_tot * chiT * gam3_m1) / (rhorun * Trun * chiD)
                     : 0.0);

    HelmState s;
    s.P = P_tot;
    s.e = e_tot;
    s.cs = cs;
    s.dPde_rho = dPde_rho;
    s.gamma1 = gam1;
    s.grada = grada_val;
    s.chiT = chiT;
    s.chiRho = chiD;
    s.cV = cv_tot;
    s.cP = cP_val;
    return s;
}

// ============================================================================
// Invert helm_eval for T given (ρ, e) via Newton iteration.
// Needed because hydro evolves e, but lookup is on T.
// ============================================================================
HE_HD inline double helm_T_from_rho_e(double rho, double e_target,
                                      const HelmholtzTableView& tv,
                                      double T_guess = -1.0)
{
    if (T_guess <= 0.0) {
        // Ideal-ion+electron estimate: e ≈ 3·(N_A·k_B/Abar)·T for full
        // ionization, e ≈ 1.5·(N_A·k_B/Abar)·T for neutral. Use the
        // full-ionization coefficient (smaller T_guess) so we start below
        // and Newton monotonically increases.
        //   N_A·k_B = 8.314e7 erg/mol/K, Abar~1.3 → coeff ≈ 1.9e8 erg/g/K
        double T_est = e_target / 1.9e8;
        if (T_est < 500.0)   T_est = 500.0;
        if (T_est > 1e8) T_est = 1e8;
        T_guess = T_est;
    }
    // The Helm table is grid-defined over [1e3, 1e13] K. Below T=1e3 the
    // interpolator clamps to the edge row, so e(T) is flat and Newton
    // cannot drive f to zero — it just halves T forever toward the 1 K
    // floor. Bracket T at [T_min, T_max] of the table and accept the
    // saturated value when we hit the bound.
    double T_min = pow(10.0, tv.tlo);
    double T_max = pow(10.0, tv.tlo + (HELM_JMAX - 1) * tv.dt);
    double T = T_guess;
    if (T < T_min) T = T_min;
    if (T > T_max) T = T_max;
    for (int it = 0; it < 60; ++it) {
        HelmState s = helm_eval(rho, T, tv);
        double f = s.e - e_target;
        if (fabs(f) < 1e-10 * fabs(e_target) + 1e-20) break;
        // de/dT = c_V. Use the analytic cv returned by helm_eval, NOT a
        // finite-difference on e. The biquintic Hermite table is on a
        // log-T grid with spacing ~0.05 decade ≈ 0.12·T, so a small
        // linear dT (O(1)) falls inside the same interpolation cell and
        // produces a zero df.
        double df = s.cV;
        if (!(df > 1e-30)) df = 1e-30;
        double T_new = T - f / df;
        if (T_new < 0.5 * T) T_new = 0.5 * T;
        if (T_new > 2.0 * T) T_new = 2.0 * T;
        if (T_new < T_min)   T_new = T_min;
        if (T_new > T_max)   T_new = T_max;
        if (T == T_new) break;      // saturated at grid edge; can't improve
        T = T_new;
    }
    return T;
}

#undef HE_HD
