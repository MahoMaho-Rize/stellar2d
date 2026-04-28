#include "riemann.h"
#include <cmath>
#include <algorithm>

static Flux4 physical_flux(const PrimitiveVars& w, const EOS& eos, bool radial_dir) {
    double rho = w.rho;
    double u = radial_dir ? w.vr : w.vtheta;
    double vt = radial_dir ? w.vtheta : w.vr;
    double p = w.P;
    double ke = 0.5 * (u * u + vt * vt);
    double e_total = rho * (eos.internal_energy_from_rho_p(rho, p) + ke); // Eq. (4.3), generalized EOS

    Flux4 f;
    f.f_rho = rho * u;                                 // Eq. (4.4)
    if (radial_dir) {
        f.f_mr = rho * u * u + p;                      // Eq. (4.4): radial momentum flux
        f.f_mtheta = rho * u * vt;                     // Eq. (4.4): tangential momentum flux
    } else {
        f.f_mr = rho * u * vt;                         // Eq. (4.4): radial momentum flux (theta face)
        f.f_mtheta = rho * u * u + p;                  // Eq. (4.4): theta momentum flux (theta face)
    }
    f.f_E = (e_total + p) * u;                         // Eq. (4.4): energy flux
    return f;
}

static Flux4 hllc_solve(const PrimitiveVars& wl, const PrimitiveVars& wr,
                        const EOS& eos, bool radial_dir) {
    double rhol = wl.rho, rhor = wr.rho;
    double pl = wl.P, pr = wr.P;
    double ul = radial_dir ? wl.vr : wl.vtheta;       // Eq. (4.4): normal velocity
    double ur = radial_dir ? wr.vr : wr.vtheta;
    double vt_l = radial_dir ? wl.vtheta : wl.vr;     // Eq. (4.4): tangential velocity
    double vt_r = radial_dir ? wr.vtheta : wr.vr;

    double cl = eos.sound_speed(rhol, pl);             // Eq. (1.3), generalized EOS
    double cr = eos.sound_speed(rhor, pr);             // Eq. (1.3), generalized EOS

    double sl = std::min(ul - cl, ur - cr);            // Eq. (4.1)
    double sr = std::max(ul + cl, ur + cr);            // Eq. (4.1)

    // Eq. (4.2)
    double denom = rhol * (sl - ul) - rhor * (sr - ur);
    if (std::fabs(denom) < 1e-300) denom = -1e-300;
    double s_star = (pr - pl + rhol * ul * (sl - ul) - rhor * ur * (sr - ur)) / denom;

    auto compute_hllc_flux = [&](double rho, double u, double vt, double p,
                                  double s_k, Flux4 fk) -> Flux4 {
        double ke = 0.5 * (u * u + vt * vt);
        double e_total = rho * (eos.internal_energy_from_rho_p(rho, p) + ke); // Eq. (4.3), generalized EOS
        double ratio = rho * (s_k - u) / (s_k - s_star); // Eq. (4.5): rho*_K

        // Eq. (4.6): E*_K
        double E_star = ratio * (e_total / rho + (s_star - u) * (s_star + p / (rho * (s_k - u))));

        // Eq. (4.7): F*_K = F_K + S_K * (U*_K - U_K)
        Flux4 f;
        f.f_rho = fk.f_rho + s_k * (ratio - rho);
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

    // Eq. (4.7): select flux based on wave structure
    if (sl >= 0.0) {
        return physical_flux(wl, eos, radial_dir);
    } else if (s_star >= 0.0) {
        Flux4 fl = physical_flux(wl, eos, radial_dir);
        return compute_hllc_flux(rhol, ul, vt_l, pl, sl, fl);
    } else if (sr >= 0.0) {
        Flux4 fr = physical_flux(wr, eos, radial_dir);
        return compute_hllc_flux(rhor, ur, vt_r, pr, sr, fr);
    } else {
        return physical_flux(wr, eos, radial_dir);
    }
}

Flux4 hllc_flux_r(const PrimitiveVars& wl, const PrimitiveVars& wr, const EOS& eos) {
    return hllc_solve(wl, wr, eos, true);
}

Flux4 hllc_flux_theta(const PrimitiveVars& wl, const PrimitiveVars& wr, const EOS& eos) {
    return hllc_solve(wl, wr, eos, false);
}

Flux4 rusanov_flux_theta(const PrimitiveVars& wl, const PrimitiveVars& wr, const EOS& eos) {
    Flux4 fl = physical_flux(wl, eos, false);
    Flux4 fr = physical_flux(wr, eos, false);

    double cl = eos.sound_speed(wl.rho, wl.P);
    double cr = eos.sound_speed(wr.rho, wr.P);
    double alpha = std::max(std::abs(wl.vtheta) + cl, std::abs(wr.vtheta) + cr);

    double el = wl.rho * (eos.internal_energy_from_rho_p(wl.rho, wl.P)
                        + 0.5 * (wl.vr * wl.vr + wl.vtheta * wl.vtheta));
    double er = wr.rho * (eos.internal_energy_from_rho_p(wr.rho, wr.P)
                        + 0.5 * (wr.vr * wr.vr + wr.vtheta * wr.vtheta));

    Flux4 f;
    f.f_rho = 0.5 * (fl.f_rho + fr.f_rho) - 0.5 * alpha * (wr.rho - wl.rho);
    f.f_mr = 0.5 * (fl.f_mr + fr.f_mr) - 0.5 * alpha * (wr.rho * wr.vr - wl.rho * wl.vr);
    f.f_mtheta = 0.5 * (fl.f_mtheta + fr.f_mtheta) - 0.5 * alpha * (wr.rho * wr.vtheta - wl.rho * wl.vtheta);
    f.f_E = 0.5 * (fl.f_E + fr.f_E) - 0.5 * alpha * (er - el);
    return f;
}
