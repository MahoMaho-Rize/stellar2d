#pragma once

struct FPrim { double rho, vr, vt, P; };
struct FFlux4 { double f_rho, f_mr, f_mt, f_E; };

// limiter_type: 0=minmod, 1=van_leer, 2=MC
__device__ __forceinline__
double fas_minmod(double a, double b) {
    return (a*b <= 0.0) ? 0.0 : (fabs(a) < fabs(b) ? a : b);
}

__device__ __forceinline__
double fas_van_leer(double a, double b) {
    return (a*b <= 0.0) ? 0.0 : 2.0*a*b/(a+b);
}

__device__ __forceinline__
double fas_mc(double a, double b) {
    if (a*b <= 0.0) return 0.0;
    double c = 0.5*(a+b);
    double s = (a > 0.0) ? 1.0 : -1.0;
    return s * fmin(fmin(fabs(c), 2.0*fabs(a)), 2.0*fabs(b));
}

__device__ __forceinline__
double fas_limit(double a, double b, int lim_type) {
    if (lim_type == 2) return fas_mc(a, b);
    if (lim_type == 1) return fas_van_leer(a, b);
    return fas_minmod(a, b);
}

__device__ __forceinline__
void fas_recon(double vm1, double v0, double vp1, double vp2,
               double& L, double& R, int lim_type = 0) {
    L = v0   + 0.5 * fas_limit(v0 - vm1, vp1 - v0, lim_type);
    R = vp1  - 0.5 * fas_limit(vp1 - v0, vp2 - vp1, lim_type);
}

__device__
inline FFlux4 fas_hllc(FPrim wl, FPrim wr, double gamma, bool radial) {
    double rhol=wl.rho, rhor=wr.rho, pl=wl.P, pr=wr.P;
    double ul = radial ? wl.vr : wl.vt;
    double ur = radial ? wr.vr : wr.vt;
    double vtl = radial ? wl.vt : wl.vr;
    double vtr = radial ? wr.vt : wr.vr;

    double cl = sqrt(gamma*pl/rhol), cr = sqrt(gamma*pr/rhor);
    double sl = fmin(ul-cl, ur-cr), sr = fmax(ul+cl, ur+cr);
    double denom = rhol*(sl-ul) - rhor*(sr-ur);
    if (fabs(denom) < 1e-300) denom = -1e-300;
    double s_star = (pr-pl + rhol*ul*(sl-ul) - rhor*ur*(sr-ur)) / denom;

    auto phys_flux = [&](double rho, double u, double vt, double p) -> FFlux4 {
        double E = p/(gamma-1.0) + 0.5*rho*(u*u + vt*vt);
        FFlux4 f;
        f.f_rho = rho*u;
        if (radial) { f.f_mr = rho*u*u+p; f.f_mt = rho*u*vt; }
        else        { f.f_mr = rho*u*vt;  f.f_mt = rho*u*u+p; }
        f.f_E = (E+p)*u;
        return f;
    };

    auto star_flux = [&](double rho, double u, double vt, double p,
                         double sk, FFlux4 fk) -> FFlux4 {
        double E = p/(gamma-1.0) + 0.5*rho*(u*u+vt*vt);
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
}

// Low-Mach corrected HLLC (Rieper 2011): reduces pressure dissipation when M→0.
// Scales pressure jump by local Mach number to prevent O(1/M) viscosity.
__device__
inline FFlux4 fas_hllc_lm(FPrim wl, FPrim wr, double gamma, bool radial) {
    double rhol=wl.rho, rhor=wr.rho, pl=wl.P, pr=wr.P;
    double ul = radial ? wl.vr : wl.vt;
    double ur = radial ? wr.vr : wr.vt;
    double vtl = radial ? wl.vt : wl.vr;
    double vtr = radial ? wr.vt : wr.vr;

    double cl = sqrt(gamma*pl/rhol), cr = sqrt(gamma*pr/rhor);
    double c_avg = 0.5*(cl+cr);
    double M_face = fmax(fabs(ul), fabs(ur)) / fmax(c_avg, 1e-30);
    double f = fmin(1.0, M_face);

    // Scale pressure difference and wave speeds
    double p_avg = 0.5*(pl+pr);
    double p_diff = 0.5*f*(pl-pr);
    double pl_mod = p_avg + p_diff;
    double pr_mod = p_avg - p_diff;

    double sl = fmin(ul - cl, ur - cr);
    double sr = fmax(ul + cl, ur + cr);
    double denom = rhol*(sl-ul) - rhor*(sr-ur);
    if (fabs(denom) < 1e-300) denom = -1e-300;
    double s_star = (pr_mod-pl_mod + rhol*ul*(sl-ul) - rhor*ur*(sr-ur)) / denom;

    auto phys_flux = [&](double rho, double u, double vt, double p) -> FFlux4 {
        double E = p/(gamma-1.0) + 0.5*rho*(u*u + vt*vt);
        FFlux4 ff;
        ff.f_rho = rho*u;
        if (radial) { ff.f_mr = rho*u*u+p; ff.f_mt = rho*u*vt; }
        else        { ff.f_mr = rho*u*vt;  ff.f_mt = rho*u*u+p; }
        ff.f_E = (E+p)*u;
        return ff;
    };

    auto star_flux = [&](double rho, double u, double vt, double p,
                         double sk, FFlux4 fk) -> FFlux4 {
        double E = p/(gamma-1.0) + 0.5*rho*(u*u+vt*vt);
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
}
