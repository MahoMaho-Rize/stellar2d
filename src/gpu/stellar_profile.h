#pragma once

#include <string>
#include <vector>

// Stellar profile representation and builders for g-mode benchmarks.
//
// Minimal set (for Exp I/J/K vs GYRE): the 5 GYRE dimensionless structure
// coefficients
//
//   V_2    = V / x²,     V = -d ln P / d ln r = r ρ g / P
//   A_star = r / Γ_1 · ( d ln ρ / d r - (1/Γ_1) · d ln P / dr )
//          =  ( d ln ρ / d ln r ) − ( 1/Γ_1 ) · ( d ln P / d ln r )      (x→0 finite)
//   U      = d ln M_r / d ln r = 4π ρ r³ / M_r
//   c_1    = (r/R)³ · M/M_r
//   Γ_1    = (d ln P / d ln ρ)_s      (adiabatic index)
//
// on a dimensionless x = r/R grid.  These are the inputs to GYRE's
// oscillation equation (see Cox 1980; Unno+89; gyre/src/eqns/ad/A_t.inc).
//
// We also keep the raw (r, ρ, P, M_r) so N² etc. can be recomputed if needed.
struct StellarProfile {
    std::vector<double> x;        // r / R (dimensionless, ascending, excludes r=0, r=R)
    std::vector<double> rho;      // ρ / ρ_c    (dimensionless)
    std::vector<double> P;        // P / P_c
    std::vector<double> M_r;      // M_r / M_total
    std::vector<double> V_2;      // V / x²
    std::vector<double> A_star;
    std::vector<double> U;
    std::vector<double> c_1;
    std::vector<double> Gamma_1;

    int n_points() const { return (int)x.size(); }
};

// Build a Lane-Emden polytrope profile (n = polytropic index) with the 5
// GYRE dimensionless structure coefficients, sampled at n_pts roughly
// uniformly in ξ.  Uses RK4 to integrate Lane-Emden, high-rtol (≤1e-12).
//
//   gamma_1 : adiabatic index (distinct from polytropic n; for ideal gas 5/3)
//
// n_poly = 3 gives the Eddington standard model (matches GYRE's poly3.txt).
StellarProfile build_polytrope_profile(double n_poly,
                                       int n_pts = 5000,
                                       double gamma_1 = 5.0 / 3.0,
                                       double inner_cut = 1e-4,
                                       double outer_cut = 0.9999);

// Evaluate the polytrope profile at an arbitrary set of x-points (sorted
// ascending).  The Lane-Emden ODE is integrated once at high precision;
// structure coefficients are evaluated directly at each requested x (no
// interpolation between samples).  This is what Exp K really wants
// because CGL nodes are non-uniform.
StellarProfile build_polytrope_profile_at(double n_poly,
                                          const std::vector<double>& x_query,
                                          double gamma_1 = 5.0 / 3.0);

// Read a 6-column space-separated GYRE poly-style file:
//   header (1 line) + rows of  x  V_2  A*  U  c_1  Γ_1
// (Exactly the format of `poly_to_txt` output.)  Fills V_2, A_star, U, c_1,
// Gamma_1 from the file; rho, P, M_r left empty.
bool read_gyre_structure_txt(const std::string& path, StellarProfile& out);
