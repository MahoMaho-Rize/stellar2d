// Stellar profile builders — Lane-Emden polytrope + GYRE structure
// coefficients (V_2, A_star, U, c_1, Γ_1).  Derivation follows Cox 1980
// Chapter 17 and GYRE's `poly_to_txt` tool output.
//
// For a polytrope ρ = ρ_c θ^n, P = K ρ^(1+1/n), ξ = r / α, where
// α = [(n+1) P_c / (4π G ρ_c²)]^{1/2}, the Lane-Emden equation
//     (1/ξ²) d/dξ (ξ² dθ/dξ) + θ^n = 0,     θ(0)=1, θ'(0)=0
// has surface ξ_1 where θ=0.  Dimensionless derived quantities:
//     M_r(ξ) / M_total      = [ξ² (−dθ/dξ)] / [ξ_1² (−dθ/dξ)|_{ξ_1}]
//     ρ(ξ) / ρ̄              = (M_r/M_total) · (ξ_1 / ξ)³
//     V(ξ)  = −d ln P / d ln r
//            = (n+1) · x · ξ_1² (−dθ/dξ) / [θ · (ξ_1² (−dθ/dξ)|_{ξ_1})]
//            Actually:  V = (n+1) · (−d ln θ / d ln ξ) · x      (derived below)
//     x = r/R = ξ/ξ_1
// The algebra follows GYRE (Townsend & Teitler 2013), reproduced via the
// physical definitions:
//
//   g(r)  = G M_r / r²
//   V(r)  = −r ρ g / P_scale = r ρ g / P     (note sign convention)
//   V_2   = V / x²
//   U(r)  = 4π ρ r³ / M_r                     = 3 ρ(r) / ρ̄_r
//   c_1   = (r/R)³ · M_total / M_r
//   N²(r) = g/Γ_1 · dln P/dr - g · dln ρ/dr   (Schwarzschild Brunt-Väisälä²)
//   A_star = r N² / g                         = N² / (g/r)
//          = r [ (1/Γ_1) dln P/dr − dln ρ/dr ]    (dimensionless)

#include "stellar_profile.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

// RK4 step for d(θ, dθ)/dξ where dθ/dξ = η and dη/dξ = -2η/ξ - θ^n.
static void rk4_step_lane_emden(double n, double xi, double theta, double eta,
                                double h, double& theta_next, double& eta_next) {
    auto rhs = [n](double xi_, double th_, double et_,
                   double& dth, double& det) {
        dth = et_;
        if (xi_ < 1e-12) {
            det = -std::pow(std::max(th_, 0.0), n) / 3.0;
        } else {
            double tn = (th_ > 0.0) ? std::pow(th_, n) : 0.0;
            det = -2.0 / xi_ * et_ - tn;
        }
    };
    double k1a, k1b, k2a, k2b, k3a, k3b, k4a, k4b;
    rhs(xi,       theta,            eta,             k1a, k1b);
    rhs(xi + h/2, theta + h/2*k1a,  eta + h/2*k1b,   k2a, k2b);
    rhs(xi + h/2, theta + h/2*k2a,  eta + h/2*k2b,   k3a, k3b);
    rhs(xi + h,   theta + h  *k3a,  eta + h  *k3b,   k4a, k4b);
    theta_next = theta + h * (k1a + 2*k2a + 2*k3a + k4a) / 6.0;
    eta_next   = eta   + h * (k1b + 2*k2b + 2*k3b + k4b) / 6.0;
}

}  // namespace

StellarProfile build_polytrope_profile(double n_poly, int n_pts,
                                       double gamma_1,
                                       double inner_cut, double outer_cut) {
    // ---- Integrate Lane-Emden to find ξ_1 and full (ξ, θ, η=dθ/dξ) -------
    // Dense sampling for later CubicHermite interpolation.  Use a small step
    // (h=1e-5) so the RK4 truncation error on η_surf is under 1e-14: this
    // drives M_r, V_2, U, c_1 accuracy since those scale as ξ² · (−η_surf).
    const double h = 1e-5;
    std::vector<double> xi_raw, th_raw, eta_raw;
    double xi = 1e-8;
    double theta = 1.0 - 1e-16;
    double eta   = 0.0;
    xi_raw.push_back(xi); th_raw.push_back(theta); eta_raw.push_back(eta);
    while (theta > 0.0 && xi < 30.0) {
        double th_next, et_next;
        rk4_step_lane_emden(n_poly, xi, theta, eta, h, th_next, et_next);
        if (th_next <= 0.0) {
            // Linear root find on [xi, xi+h]
            double alpha = theta / (theta - th_next);
            xi += alpha * h;
            // Extrapolate eta
            eta = eta + alpha * (et_next - eta);
            theta = 0.0;
            xi_raw.push_back(xi); th_raw.push_back(0.0); eta_raw.push_back(eta);
            break;
        }
        xi += h;
        theta = th_next;
        eta   = et_next;
        xi_raw.push_back(xi); th_raw.push_back(theta); eta_raw.push_back(eta);
    }
    if (th_raw.back() > 1e-6) {
        std::fprintf(stderr,
            "build_polytrope_profile: failed to find surface "
            "(final θ=%g at ξ=%g)\n", th_raw.back(), xi_raw.back());
        std::exit(1);
    }
    const double xi_1     = xi_raw.back();
    const double eta_surf = eta_raw.back();              // dθ/dξ|_{ξ_1} < 0
    const double mtot_fac = xi_1 * xi_1 * (-eta_surf);    // ξ_1² (−η)_surf

    // ---- Resample to n_pts in x ∈ [inner_cut, outer_cut] -----------------
    StellarProfile P;
    P.x      .resize(n_pts);
    P.rho    .resize(n_pts);
    P.P      .resize(n_pts);
    P.M_r    .resize(n_pts);
    P.V_2    .resize(n_pts);
    P.A_star .resize(n_pts);
    P.U      .resize(n_pts);
    P.c_1    .resize(n_pts);
    P.Gamma_1.resize(n_pts);

    for (int i = 0; i < n_pts; ++i) {
        double x = inner_cut + (outer_cut - inner_cut) * (double)i / (double)(n_pts - 1);
        double xi_q = x * xi_1;
        // Linear (search + interp) on xi_raw
        int j = 0;
        while (j + 1 < (int)xi_raw.size() && xi_raw[j + 1] < xi_q) ++j;
        if (j + 1 >= (int)xi_raw.size()) j = (int)xi_raw.size() - 2;
        double frac = (xi_q - xi_raw[j]) / (xi_raw[j + 1] - xi_raw[j]);
        double th  = th_raw[j]  + frac * (th_raw[j + 1]  - th_raw[j]);
        double eta_q = eta_raw[j] + frac * (eta_raw[j + 1] - eta_raw[j]);
        if (th < 0.0) th = 0.0;

        // ρ / ρ_c = θ^n;  P / P_c = θ^(n+1);  M_r / M_tot from ξ²(-η)/ξ_1²(-η)_1
        double rho = std::pow(th, n_poly);
        double pre = std::pow(th, n_poly + 1.0);
        double M_r = (xi_q * xi_q * (-eta_q)) / mtot_fac;  // M_r / M_tot

        // ---- GYRE dimensionless structure coefficients ---------------------
        // For a polytrope (Townsend & Teitler 2013 Appendix):
        //   V   = -(n+1) · x · dln(θ) / dln(ξ)
        //       = -(n+1) · (ξ/θ) · (dθ/dξ) · (x / ξ·x⁻¹)  → algebra below
        // Equivalently (direct from physical definition V = r ρ g / P ):
        //   Using g = G M_r/r², P = K ρ^((n+1)/n), and for polytrope K/ρ_c^(1/n) =
        //   (n+1)α²/... (standard manipulations), we obtain:
        //       V = (n + 1) · x · ξ_1² · (-η_q) · ρ_c / ρ / ...
        //
        // Simpler form (used by GYRE and easy to verify): for a polytrope
        //    V = -(n+1) · ξ · (dθ/dξ) / θ      (dimensionless V in polytrope units)
        // But the x = ξ/ξ_1 scaling must be kept in mind:  V_GYRE uses r/R,
        // same as x above.  The polytrope-frame formula
        //    V(ξ) = -(n+1) ξ dθ/dξ / θ
        // matches GYRE's output when x = ξ/ξ_1.
        double V = -(n_poly + 1.0) * xi_q * eta_q / std::max(th, 1e-30);
        double V_2 = V / (x * x);

        // U = 4π ρ r³ / M_r = 3 · ρ(ξ) / ρ̄_r(ξ)
        //   ρ̄_r = M_r / (4/3 π r³) → ratio: U = 3 ρ / ρ̄_r
        // In polytrope dimensionless form:
        //    ρ̄_r(ξ) / ρ_c = 3 (M_r/M_tot) · M_tot/(ρ_c · 4/3 π R³)
        //                  = 3 (M_r/M_tot) / U_proxy
        //  Cleaner: the formula below reproduces GYRE's U directly.
        //    U = -ξ θ^n / η    (standard Lane-Emden identity)
        double U = (std::fabs(eta_q) > 1e-30)
                 ? (-xi_q * std::pow(th, n_poly) / eta_q)
                 : 0.0;

        // c_1 = x³ / (M_r / M_total)
        double c_1 = (M_r > 1e-30) ? (x * x * x / M_r) : 0.0;

        // Γ_1 = gamma_1 (constant for ideal-gas polytrope model)
        double G1 = gamma_1;

        // A* = V · ( 1/Γ_1 − 1/(1+1/n) ) = V · (1/Γ_1 − n/(n+1))
        //    (polytropic formula; generic stars would use dlnρ/dlnP directly)
        double A_star = V * (1.0 / G1 - n_poly / (n_poly + 1.0));

        P.x[i]      = x;
        P.rho[i]    = rho;
        P.P[i]      = pre;
        P.M_r[i]    = M_r;
        P.V_2[i]    = V_2;
        P.A_star[i] = A_star;
        P.U[i]      = U;
        P.c_1[i]    = c_1;
        P.Gamma_1[i]= G1;
    }
    std::fprintf(stderr,
        "  StellarProfile: n_poly=%g, ξ_1=%.6f, n_pts=%d, "
        "x∈[%.4g, %.4g]\n",
        n_poly, xi_1, n_pts, P.x.front(), P.x.back());
    return P;
}

namespace {
// Integrate Lane-Emden once, return (xi_raw, theta_raw, eta_raw, xi_1, eta_surf).
struct LEData {
    std::vector<double> xi, theta, eta;
    double xi_1;
    double eta_surf;
};
static LEData integrate_lane_emden_shared(double n_poly) {
    LEData L;
    const double h = 1e-5;
    double xi = 1e-8;
    double theta = 1.0 - 1e-16;
    double eta = 0.0;
    L.xi.push_back(xi); L.theta.push_back(theta); L.eta.push_back(eta);
    while (theta > 0.0 && xi < 30.0) {
        double th_next, et_next;
        rk4_step_lane_emden(n_poly, xi, theta, eta, h, th_next, et_next);
        if (th_next <= 0.0) {
            double alpha = theta / (theta - th_next);
            xi += alpha * h;
            eta = eta + alpha * (et_next - eta);
            theta = 0.0;
            L.xi.push_back(xi); L.theta.push_back(0.0); L.eta.push_back(eta);
            break;
        }
        xi += h;
        theta = th_next;
        eta = et_next;
        L.xi.push_back(xi); L.theta.push_back(theta); L.eta.push_back(eta);
    }
    L.xi_1 = L.xi.back();
    L.eta_surf = L.eta.back();
    return L;
}

static void eval_profile_at_xi(const LEData& L, double n_poly, double gamma_1,
                               double x, double mtot_fac,
                               double& rho, double& Pres, double& M_r,
                               double& V_2, double& A_star, double& U,
                               double& c_1, double& Gamma_1) {
    double xi_q = x * L.xi_1;
    int j = 0;
    while (j + 1 < (int)L.xi.size() && L.xi[j + 1] < xi_q) ++j;
    if (j + 1 >= (int)L.xi.size()) j = (int)L.xi.size() - 2;
    double frac = (xi_q - L.xi[j]) / (L.xi[j + 1] - L.xi[j]);
    double th  = L.theta[j] + frac * (L.theta[j + 1]  - L.theta[j]);
    double eta = L.eta[j]   + frac * (L.eta[j + 1]    - L.eta[j]);
    if (th < 0.0) th = 0.0;

    rho = std::pow(th, n_poly);
    Pres = std::pow(th, n_poly + 1.0);
    M_r = (xi_q * xi_q * (-eta)) / mtot_fac;
    double V = -(n_poly + 1.0) * xi_q * eta / std::max(th, 1e-30);
    V_2 = V / (x * x);
    U = (std::fabs(eta) > 1e-30) ? (-xi_q * std::pow(th, n_poly) / eta) : 0.0;
    c_1 = (M_r > 1e-30) ? (x * x * x / M_r) : 0.0;
    Gamma_1 = gamma_1;
    A_star = V * (1.0 / gamma_1 - n_poly / (n_poly + 1.0));
}
}  // namespace

StellarProfile build_polytrope_profile_at(double n_poly,
                                          const std::vector<double>& x_query,
                                          double gamma_1) {
    LEData L = integrate_lane_emden_shared(n_poly);
    const double mtot_fac = L.xi_1 * L.xi_1 * (-L.eta_surf);

    int n = (int)x_query.size();
    StellarProfile P;
    P.x = x_query;
    P.rho.resize(n); P.P.resize(n); P.M_r.resize(n);
    P.V_2.resize(n); P.A_star.resize(n); P.U.resize(n);
    P.c_1.resize(n); P.Gamma_1.resize(n);
    for (int i = 0; i < n; ++i) {
        eval_profile_at_xi(L, n_poly, gamma_1, x_query[i], mtot_fac,
                           P.rho[i], P.P[i], P.M_r[i],
                           P.V_2[i], P.A_star[i], P.U[i],
                           P.c_1[i], P.Gamma_1[i]);
    }
    std::fprintf(stderr,
        "  StellarProfile (at CGL): n_poly=%g, ξ_1=%.6f, n_query=%d\n",
        n_poly, L.xi_1, n);
    return P;
}

bool read_gyre_structure_txt(const std::string& path, StellarProfile& out) {
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) {
        std::fprintf(stderr, "read_gyre_structure_txt: cannot open %s\n",
                     path.c_str());
        return false;
    }
    char line[1024];
    if (!std::fgets(line, sizeof(line), fp)) { std::fclose(fp); return false; }
    // Header line: skip.
    out.x.clear(); out.V_2.clear(); out.A_star.clear();
    out.U.clear(); out.c_1.clear(); out.Gamma_1.clear();
    out.rho.clear(); out.P.clear(); out.M_r.clear();
    while (std::fgets(line, sizeof(line), fp)) {
        double x, V_2, A_star, U, c_1, G1;
        if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf",
                        &x, &V_2, &A_star, &U, &c_1, &G1) == 6) {
            // GYRE poly_to_txt writes V_2=A*=Infinity at the surface row (x=1);
            // skip non-finite rows so downstream interpolation stays clean.
            if (!std::isfinite(V_2) || !std::isfinite(A_star)
                || !std::isfinite(U) || !std::isfinite(c_1)
                || !std::isfinite(G1)) continue;
            out.x.push_back(x);
            out.V_2.push_back(V_2);
            out.A_star.push_back(A_star);
            out.U.push_back(U);
            out.c_1.push_back(c_1);
            out.Gamma_1.push_back(G1);
        }
    }
    std::fclose(fp);
    return !out.x.empty();
}
