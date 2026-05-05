#pragma once

#include "eos.h"

struct FPrim { double rho, vr, vt, P; };
struct FFlux4 { double f_rho, f_mr, f_mt, f_E; };

// limiter_type: 0=minmod, 1=van_leer, 2=MC
// Eq. (3.1): minmod limiter.
__device__ __forceinline__
double gpu_minmod(double a, double b) {
    return (a*b <= 0.0) ? 0.0 : (fabs(a) < fabs(b) ? a : b);
}

// Eq. (3.2): van Leer harmonic limiter.
__device__ __forceinline__
double gpu_van_leer(double a, double b) {
    return (a*b <= 0.0) ? 0.0 : 2.0*a*b/(a+b);
}

__device__ __forceinline__
double gpu_mc(double a, double b) {
    if (a*b <= 0.0) return 0.0;
    double c = 0.5*(a+b);
    double s = (a > 0.0) ? 1.0 : -1.0;
    return s * fmin(fmin(fabs(c), 2.0*fabs(a)), 2.0*fabs(b));
}

__device__ __forceinline__
double gpu_limit(double a, double b, int lim_type) {
    if (lim_type == 2) return gpu_mc(a, b);
    if (lim_type == 1) return gpu_van_leer(a, b);
    return gpu_minmod(a, b);
}

// Eq. (3.3-3.5): MUSCL reconstruction at face i+1/2.
__device__ __forceinline__
void gpu_recon(double vm1, double v0, double vp1, double vp2,
               double& L, double& R, int lim_type = 0) {
    L = v0   + 0.5 * gpu_limit(v0 - vm1, vp1 - v0, lim_type);
    R = vp1  - 0.5 * gpu_limit(vp1 - v0, vp2 - vp1, lim_type);
}

__device__
inline FFlux4 gpu_hllc(FPrim wl, FPrim wr, EOS eos, bool radial) {
    double rhol=wl.rho, rhor=wr.rho, pl=wl.P, pr=wr.P;
    double ul = radial ? wl.vr : wl.vt;
    double ur = radial ? wr.vr : wr.vt;
    double vtl = radial ? wl.vt : wl.vr;
    double vtr = radial ? wr.vt : wr.vr;

    double cl = eos.sound_speed(rhol, pl), cr = eos.sound_speed(rhor, pr);
    double sl = fmin(ul-cl, ur-cr), sr = fmax(ul+cl, ur+cr);     // Eq. (4.1)
    double denom = rhol*(sl-ul) - rhor*(sr-ur);
    if (fabs(denom) < 1e-300) denom = -1e-300;
    // Eq. (4.2): contact wave speed S*.
    double s_star = (pr-pl + rhol*ul*(sl-ul) - rhor*ur*(sr-ur)) / denom;

    auto phys_flux = [&](double rho, double u, double vt, double p) -> FFlux4 {
        double E = rho * eos.internal_energy(rho, p) + 0.5*rho*(u*u + vt*vt);
        FFlux4 f;
        f.f_rho = rho*u;
        if (radial) { f.f_mr = rho*u*u+p; f.f_mt = rho*u*vt; }
        else        { f.f_mr = rho*u*vt;  f.f_mt = rho*u*u+p; }
        f.f_E = (E+p)*u;
        return f;
    };

    auto star_flux = [&](double rho, double u, double vt, double p,
                         double sk, FFlux4 fk) -> FFlux4 {
        double E = rho * eos.internal_energy(rho, p) + 0.5*rho*(u*u+vt*vt);
        double ratio = rho*(sk-u)/(sk-s_star);
        double E_star = ratio*(E/rho + (s_star-u)*(s_star + p/(rho*(sk-u))));
        FFlux4 f;
        f.f_rho = fk.f_rho + sk*(ratio-rho);
        if (radial) {
            f.f_mr = fk.f_mr + sk*(ratio*s_star - rho*u);
            f.f_mt = fk.f_mt + sk*(ratio*vt - rho*vt);
        } else {
            f.f_mr = fk.f_mr + sk*(ratio*vt - rho*vt);
            f.f_mt = fk.f_mt + sk*(ratio*s_star - rho*u);
        }
        f.f_E = fk.f_E + sk*(E_star-E);
        return f;
    };

    if (sl >= 0.0) return phys_flux(rhol,ul,vtl,pl);
    else if (s_star >= 0.0) { auto fl=phys_flux(rhol,ul,vtl,pl); return star_flux(rhol,ul,vtl,pl,sl,fl); }
    else if (sr >= 0.0) { auto fr=phys_flux(rhor,ur,vtr,pr); return star_flux(rhor,ur,vtr,pr,sr,fr); }
    else return phys_flux(rhor,ur,vtr,pr);
} // end gpu_hllc(EOS)

// Backward-compatible overload for ideal-gas callers (wb2d etc.) that pass a bare γ.
__device__
inline FFlux4 gpu_hllc(FPrim wl, FPrim wr, double gamma, bool radial) {
    return gpu_hllc(wl, wr, EOS::ideal(gamma), radial);
}

// Low-Mach corrected HLLC (Rieper 2011): reduces pressure dissipation when M→0.
// Scales pressure jump by local Mach number to prevent O(1/M) viscosity.
__device__
inline FFlux4 gpu_hllc_lm(FPrim wl, FPrim wr, EOS eos, bool radial) {
    double rhol=wl.rho, rhor=wr.rho, pl=wl.P, pr=wr.P;
    double ul = radial ? wl.vr : wl.vt;
    double ur = radial ? wr.vr : wr.vt;
    double vtl = radial ? wl.vt : wl.vr;
    double vtr = radial ? wr.vt : wr.vr;

    double cl = eos.sound_speed(rhol, pl), cr = eos.sound_speed(rhor, pr);
    double c_avg = 0.5*(cl+cr);
    double M_face = fmax(fabs(ul), fabs(ur)) / fmax(c_avg, 1e-30); // Eq. (14.1)
    double f = fmin(1.0, M_face);                                   // Eq. (14.2)

    // Eq. (14.4a): modified left/right pressures P_{L,mod}, P_{R,mod}.
    double p_avg = 0.5*(pl+pr);
    double p_diff = 0.5*f*(pl-pr);
    double pl_mod = p_avg + p_diff;
    double pr_mod = p_avg - p_diff;

    double sl = fmin(ul - cl, ur - cr);                   // Eq. (4.1)
    double sr = fmax(ul + cl, ur + cr);                   // Eq. (4.1)
    double denom = rhol*(sl-ul) - rhor*(sr-ur);
    if (fabs(denom) < 1e-300) denom = -1e-300;
    // Eq. (14.4b): LM-HLLC contact wave (FAS form).
    double s_star = (pr_mod-pl_mod + rhol*ul*(sl-ul) - rhor*ur*(sr-ur)) / denom;

    auto phys_flux = [&](double rho, double u, double vt, double p) -> FFlux4 {
        double E = rho * eos.internal_energy(rho, p) + 0.5*rho*(u*u + vt*vt);
        FFlux4 ff;
        ff.f_rho = rho*u;
        if (radial) { ff.f_mr = rho*u*u+p; ff.f_mt = rho*u*vt; }
        else        { ff.f_mr = rho*u*vt;  ff.f_mt = rho*u*u+p; }
        ff.f_E = (E+p)*u;
        return ff;
    };

    auto star_flux = [&](double rho, double u, double vt, double p,
                         double sk, FFlux4 fk) -> FFlux4 {
        double E = rho * eos.internal_energy(rho, p) + 0.5*rho*(u*u+vt*vt);
        double ratio = rho*(sk-u)/(sk-s_star);
        double E_star = ratio*(E/rho + (s_star-u)*(s_star + p/(rho*(sk-u))));
        FFlux4 ff;
        ff.f_rho = fk.f_rho + sk*(ratio-rho);
        if (radial) {
            ff.f_mr = fk.f_mr + sk*(ratio*s_star - rho*u);
            ff.f_mt = fk.f_mt + sk*(ratio*vt - rho*vt);
        } else {
            ff.f_mr = fk.f_mr + sk*(ratio*vt - rho*vt);
            ff.f_mt = fk.f_mt + sk*(ratio*s_star - rho*u);
        }
        ff.f_E = fk.f_E + sk*(E_star-E);
        return ff;
    };

    if (sl >= 0.0) return phys_flux(rhol,ul,vtl,pl);
    else if (s_star >= 0.0) { auto fl=phys_flux(rhol,ul,vtl,pl); return star_flux(rhol,ul,vtl,pl,sl,fl); }
    else if (sr >= 0.0) { auto fr=phys_flux(rhor,ur,vtr,pr); return star_flux(rhor,ur,vtr,pr,sr,fr); }
    else return phys_flux(rhor,ur,vtr,pr);
} // end gpu_hllc_lm(EOS)

__device__
inline FFlux4 gpu_hllc_lm(FPrim wl, FPrim wr, double gamma, bool radial) {
    return gpu_hllc_lm(wl, wr, EOS::ideal(gamma), radial);
}

// Minoshima 2021 Low-dissipation HLLC (LHLLC).
// Core low-Mach fix: phi = χ(2-χ), χ = min(1, |v|_max / c_max) weights the
// velocity jump in the contact pressure, vanishing as M→0.
// Also uses PVRS middle state + ql/qr nonlinear wave correction (Toro 10.5.2).
// Carbuncle-cure shock detector (th) omitted — would require tangential stencil.
__device__
inline FFlux4 gpu_lhllc(FPrim wl, FPrim wr, EOS eos, bool radial) {
    double rhol=wl.rho, rhor=wr.rho, pl=wl.P, pr=wr.P;
    double ul  = radial ? wl.vr : wl.vt;
    double ur  = radial ? wr.vr : wr.vt;
    double vtl = radial ? wl.vt : wl.vr;
    double vtr = radial ? wr.vt : wr.vr;

    double cl = eos.sound_speed(rhol, pl);
    double cr = eos.sound_speed(rhor, pr);
    double vsql = ul*ul + vtl*vtl;
    double vsqr = ur*ur + vtr*vtr;

    // Step 2: PVRS middle state (Toro 10.5.2) for non-linear wave strength.
    double rhoa = 0.5*(rhol + rhor);
    double ca   = 0.5*(cl + cr);
    double pmid = 0.5*(pl + pr + (ul - ur)*rhoa*ca);
    if (pmid < 1e-30) pmid = 1e-30;

    // Step 3: Shock strength correction ql, qr.
    // For IDEAL_RAD, gamma1 is an approximation (config γ); shock branches
    // activate only when pmid > pk, where the correction is second-order anyway.
    double g1 = eos.gamma;
    double ql = (pmid <= pl) ? 1.0 :
                sqrt(1.0 + (g1 + 1.0)/(2.0*g1) * (pmid/pl - 1.0));
    double qr = (pmid <= pr) ? 1.0 :
                sqrt(1.0 + (g1 + 1.0)/(2.0*g1) * (pmid/pr - 1.0));

    // Step 4: Wave speeds with non-linear correction.
    double al = ul - cl*ql;
    double ar = ur + cr*qr;
    double bp = (ar > 0.0) ? ar : 1e-30;
    double bm = (al < 0.0) ? al : -1e-30;

    // Step 5: Contact speed am and pressure cp.
    double vxl = al - ul;
    double vxr = ar - ur;
    double ml  = rhol * vxl;
    double mr  = rhor * vxr;
    double dm  = mr - ml;
    if (fabs(dm) < 1e-300) dm = (dm >= 0) ? 1e-300 : -1e-300;

    // Low-Mach weight: phi = χ(2-χ), χ = min(1, |v|_max / c_max).
    double cmax = fmax(cl, cr);
    double chi  = fmin(1.0, sqrt(fmax(vsql, vsqr)) / fmax(cmax, 1e-30));
    double phi  = chi * (2.0 - chi);

    // Contact velocity (no shock detector; th=1).
    double am = (mr*ur - ml*ul - (pr - pl)) / dm;

    // Contact pressure — low-Mach fix: phi weights the velocity jump.
    double cp = (mr*pl - ml*pr + phi*mr*ml*(ur - ul)) / dm;
    if (cp < 0.0) cp = 0.0;

    // Step 6: L/R fluxes along bm, bp.
    double uxl = ul - bm;
    double uxr = ur - bp;
    double El  = rhol * eos.internal_energy(rhol, pl) + 0.5*rhol*vsql;
    double Er  = rhor * eos.internal_energy(rhor, pr) + 0.5*rhor*vsqr;

    FFlux4 fl, fr;
    fl.f_rho = rhol * uxl;
    fr.f_rho = rhor * uxr;
    if (radial) {
        fl.f_mr = rhol*ul*uxl + pl;
        fr.f_mr = rhor*ur*uxr + pr;
        fl.f_mt = rhol*vtl*uxl;
        fr.f_mt = rhor*vtr*uxr;
    } else {
        fl.f_mr = rhol*vtl*uxl;
        fr.f_mr = rhor*vtr*uxr;
        fl.f_mt = rhol*ul*uxl + pl;
        fr.f_mt = rhor*ur*uxr + pr;
    }
    fl.f_E = El*uxl + pl*ul;
    fr.f_E = Er*uxr + pr*ur;

    // Step 8: Flux weights along bm, bp.
    double sl, sr, sm;
    if (am >= 0.0) {
        double denom = am - bm;
        if (fabs(denom) < 1e-300) denom = 1e-300;
        sl =  am / denom;
        sr =  0.0;
        sm = -bm / denom;
    } else {
        double denom = bp - am;
        if (fabs(denom) < 1e-300) denom = 1e-300;
        sl =  0.0;
        sr = -am / denom;
        sm =  bp / denom;
    }

    // Step 9: Combine with contact flux.
    FFlux4 f;
    f.f_rho = sl*fl.f_rho + sr*fr.f_rho;
    if (radial) {
        f.f_mr = sl*fl.f_mr + sr*fr.f_mr + sm*cp;
        f.f_mt = sl*fl.f_mt + sr*fr.f_mt;
    } else {
        f.f_mr = sl*fl.f_mr + sr*fr.f_mr;
        f.f_mt = sl*fl.f_mt + sr*fr.f_mt + sm*cp;
    }
    f.f_E = sl*fl.f_E + sr*fr.f_E + sm*cp*am;
    return f;
} // end gpu_lhllc(EOS)

__device__
inline FFlux4 gpu_lhllc(FPrim wl, FPrim wr, double gamma, bool radial) {
    return gpu_lhllc(wl, wr, EOS::ideal(gamma), radial);
}

// Unified dispatch — variant: 0=standard HLLC, 1=Rieper LM-HLLC, 2=Minoshima LHLLC.
__device__ __forceinline__
FFlux4 gpu_hllc_dispatch(FPrim wl, FPrim wr, EOS eos, bool radial, int variant) {
    if (variant == 2) return gpu_lhllc(wl, wr, eos, radial);
    if (variant == 1) return gpu_hllc_lm(wl, wr, eos, radial);
    return gpu_hllc(wl, wr, eos, radial);
}

__device__ __forceinline__
FFlux4 gpu_hllc_dispatch(FPrim wl, FPrim wr, double gamma, bool radial, int variant) {
    if (variant == 2) return gpu_lhllc(wl, wr, gamma, radial);
    if (variant == 1) return gpu_hllc_lm(wl, wr, gamma, radial);
    return gpu_hllc(wl, wr, gamma, radial);
}
