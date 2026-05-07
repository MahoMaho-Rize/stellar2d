#pragma once

// ============================================================
// sod_exact.h — shared exact Toro/Sod 1D Riemann solver for
// the standard Sod shock tube (ρL=1, PL=1, ρR=0.125, PR=0.1,
// γ=1.4, v=0, split at x=x0). Used by cart_ale2 and athena_vl2
// compute_sod_error(). Header-only, host-side, inlined so there
// is no separate .cu TU to add to CMakeLists.
//
// Reference: Toro "Riemann Solvers and Numerical Methods for
// Fluid Dynamics" (2009), §4.3, star-region pressure iteration.
// Matches the python port that was in scripts/tests_ale2/sod_compare.py;
// this commit deprecates that port.
// ============================================================

#include <cmath>
#include <cstddef>

namespace sod_exact {

struct Params {
    double gamma = 1.4;
    double rhoL  = 1.0;   double pL = 1.0;   double uL = 0.0;
    double rhoR  = 0.125; double pR = 0.1;   double uR = 0.0;
    double x0    = 0.5;   // interface position
};

// Bracketed bisection on the star-region pressure p*.
inline double solve_pstar(const Params& P) {
    const double g = P.gamma;
    const double cL = std::sqrt(g * P.pL / P.rhoL);
    const double cR = std::sqrt(g * P.pR / P.rhoR);

    auto fK = [&](double p, double pK, double rhoK, double cK) -> double {
        if (p > pK) {
            double A = 2.0 / ((g + 1.0) * rhoK);
            double B = (g - 1.0) / (g + 1.0) * pK;
            return (p - pK) * std::sqrt(A / (p + B));
        }
        return 2.0 * cK / (g - 1.0) *
               (std::pow(p / pK, (g - 1.0) / (2.0 * g)) - 1.0);
    };
    auto F = [&](double p) {
        return fK(p, P.pL, P.rhoL, cL) + fK(p, P.pR, P.rhoR, cR) +
               (P.uR - P.uL);
    };
    double lo = 1e-9;
    double hi = 10.0 * std::fmax(P.pL, P.pR);
    for (int i = 0; i < 200; ++i) {
        double pm = 0.5 * (lo + hi);
        if (F(pm) > 0.0) hi = pm; else lo = pm;
    }
    return 0.5 * (lo + hi);
}

// Evaluate ρ(x, t) on the self-similar fan; x passed absolute.
inline double rho_at(const Params& P, double x, double t) {
    if (t <= 0.0) {
        return (x < P.x0) ? P.rhoL : P.rhoR;
    }
    const double g   = P.gamma;
    const double cL  = std::sqrt(g * P.pL / P.rhoL);
    const double cR  = std::sqrt(g * P.pR / P.rhoR);
    const double pS  = solve_pstar(P);

    // Fan variable s = (x - x0) / t.
    const double s = (x - P.x0) / t;

    // Star-region velocity.
    auto fK = [&](double p, double pK, double rhoK, double cK) -> double {
        if (p > pK) {
            double A = 2.0 / ((g + 1.0) * rhoK);
            double B = (g - 1.0) / (g + 1.0) * pK;
            return (p - pK) * std::sqrt(A / (p + B));
        }
        return 2.0 * cK / (g - 1.0) *
               (std::pow(p / pK, (g - 1.0) / (2.0 * g)) - 1.0);
    };
    const double uS = 0.5 * (P.uL + P.uR) +
                      0.5 * (fK(pS, P.pR, P.rhoR, cR) -
                             fK(pS, P.pL, P.rhoL, cL));

    if (s < uS) {
        // Left side of contact.
        if (pS > P.pL) {
            // Left shock.
            const double SL = P.uL - cL * std::sqrt(
                (g + 1.0) / (2.0 * g) * pS / P.pL +
                (g - 1.0) / (2.0 * g));
            if (s < SL) return P.rhoL;
            return P.rhoL * (pS / P.pL + (g - 1.0) / (g + 1.0)) /
                            ((g - 1.0) / (g + 1.0) * pS / P.pL + 1.0);
        }
        // Left rarefaction fan.
        const double cLs = cL * std::pow(pS / P.pL, (g - 1.0) / (2.0 * g));
        const double head = P.uL - cL;
        const double tail = uS    - cLs;
        if (s < head) return P.rhoL;
        if (s > tail) {
            return P.rhoL * std::pow(pS / P.pL, 1.0 / g);
        }
        const double c_fan = 2.0 / (g + 1.0) *
                             (cL + (g - 1.0) / 2.0 * (P.uL - s));
        return P.rhoL * std::pow(c_fan / cL, 2.0 / (g - 1.0));
    }
    // Right side of contact.
    if (pS > P.pR) {
        const double SR = P.uR + cR * std::sqrt(
            (g + 1.0) / (2.0 * g) * pS / P.pR +
            (g - 1.0) / (2.0 * g));
        if (s > SR) return P.rhoR;
        return P.rhoR * (pS / P.pR + (g - 1.0) / (g + 1.0)) /
                        ((g - 1.0) / (g + 1.0) * pS / P.pR + 1.0);
    }
    const double cRs = cR * std::pow(pS / P.pR, (g - 1.0) / (2.0 * g));
    const double head = P.uR + cR;
    const double tail = uS    + cRs;
    if (s > head) return P.rhoR;
    if (s < tail) {
        return P.rhoR * std::pow(pS / P.pR, 1.0 / g);
    }
    const double c_fan = 2.0 / (g + 1.0) *
                         (cR - (g - 1.0) / 2.0 * (P.uR - s));
    return P.rhoR * std::pow(c_fan / cR, 2.0 / (g - 1.0));
}

}  // namespace sod_exact
