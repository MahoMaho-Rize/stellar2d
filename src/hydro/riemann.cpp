#include "riemann.h"
#include <cmath>
#include <algorithm>

static Flux4 hllc_solve(const PrimitiveVars& wl, const PrimitiveVars& wr,
                         double gamma, bool radial_dir) {
    double rhol = wl.rho, rhor = wr.rho;
    double pl = wl.P, pr = wr.P;
    double ul = radial_dir ? wl.vr : wl.vtheta;
    double ur = radial_dir ? wr.vr : wr.vtheta;
    double vt_l = radial_dir ? wl.vtheta : wl.vr;
    double vt_r = radial_dir ? wr.vtheta : wr.vr;

    double cl = std::sqrt(gamma * pl / rhol);
    double cr = std::sqrt(gamma * pr / rhor);

    double sl = std::min(ul - cl, ur - cr);
    double sr = std::max(ul + cl, ur + cr);

    double s_star = (pr - pl + rhol * ul * (sl - ul) - rhor * ur * (sr - ur))
                    / (rhol * (sl - ul) - rhor * (sr - ur));

    auto compute_flux = [&](double rho, double u, double vt, double p,
                            double ke) -> Flux4 {
        double e_total = p / ((gamma - 1.0)) + rho * ke;
        Flux4 f;
        f.f_rho = rho * u;
        if (radial_dir) {
            f.f_mr = rho * u * u + p;
            f.f_mtheta = rho * u * vt;
        } else {
            f.f_mr = rho * u * vt;
            f.f_mtheta = rho * u * u + p;
        }
        f.f_E = (e_total + p) * u;
        return f;
    };

    auto compute_hllc_flux = [&](double rho, double u, double vt, double p,
                                  double s_k, Flux4 fk) -> Flux4 {
        double ke = 0.5 * (u * u + vt * vt);
        double e_total = p / (gamma - 1.0) + rho * ke;
        double ratio = rho * (s_k - u) / (s_k - s_star);

        double rho_star = ratio;
        double E_star = ratio * (e_total / rho + (s_star - u) * (s_star + p / (rho * (s_k - u))));

        Flux4 f;
        f.f_rho = fk.f_rho + s_k * (rho_star - rho);
        if (radial_dir) {
            double mr_star = ratio * s_star;
            double mt_star = ratio * vt;
            f.f_mr = fk.f_mr + s_k * (mr_star - rho * u);
            f.f_mtheta = fk.f_mtheta + s_k * (mt_star - rho * vt);
        } else {
            double mr_star = ratio * vt;
            double mt_star = ratio * s_star;
            f.f_mr = fk.f_mr + s_k * (mr_star - rho * vt);
            f.f_mtheta = fk.f_mtheta + s_k * (mt_star - rho * u);
        }
        f.f_E = fk.f_E + s_k * (E_star - e_total);
        return f;
    };

    double ke_l = 0.5 * (ul * ul + vt_l * vt_l);
    double ke_r = 0.5 * (ur * ur + vt_r * vt_r);

    if (sl >= 0.0) {
        return compute_flux(rhol, ul, vt_l, pl, ke_l);
    } else if (s_star >= 0.0) {
        Flux4 fl = compute_flux(rhol, ul, vt_l, pl, ke_l);
        return compute_hllc_flux(rhol, ul, vt_l, pl, sl, fl);
    } else if (sr >= 0.0) {
        Flux4 fr = compute_flux(rhor, ur, vt_r, pr, ke_r);
        return compute_hllc_flux(rhor, ur, vt_r, pr, sr, fr);
    } else {
        return compute_flux(rhor, ur, vt_r, pr, ke_r);
    }
}

Flux4 hllc_flux_r(const PrimitiveVars& wl, const PrimitiveVars& wr, double gamma) {
    return hllc_solve(wl, wr, gamma, true);
}

Flux4 hllc_flux_theta(const PrimitiveVars& wl, const PrimitiveVars& wr, double gamma) {
    return hllc_solve(wl, wr, gamma, false);
}
