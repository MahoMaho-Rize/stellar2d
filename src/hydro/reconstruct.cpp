#include "reconstruct.h"
#include <algorithm>
#include <cmath>

double minmod(double a, double b) { // Eq. (3.1)
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

double van_leer(double a, double b) { // Eq. (3.2)
    if (a * b <= 0.0) return 0.0;
    return 2.0 * a * b / (a + b);
}

double mc_limiter(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    double c = 0.5 * (a + b);
    double s = (a > 0.0) ? 1.0 : -1.0;
    return s * std::min({std::abs(c), 2.0 * std::abs(a), 2.0 * std::abs(b)});
}

double apply_limiter(double a, double b, Limiter lim) {
    switch (lim) {
        case Limiter::VAN_LEER: return van_leer(a, b);
        case Limiter::MC:       return mc_limiter(a, b);
        default:                return minmod(a, b);
    }
}

static void reconstruct_component(double vm1, double v0, double vp1, double vp2,
                                  Limiter lim, double& left, double& right) {
    double slope_l = v0 - vm1;
    double slope_r = vp1 - v0;
    double slope0 = apply_limiter(slope_l, slope_r, lim); // Eq. (3.3)
    left = v0 + 0.5 * slope0;                             // Eq. (3.4)

    double slope_l2 = vp1 - v0;
    double slope_r2 = vp2 - vp1;
    double slope1 = apply_limiter(slope_l2, slope_r2, lim); // Eq. (3.3)
    right = vp1 - 0.5 * slope1;                             // Eq. (3.5)
}

ReconstructedPair muscl_reconstruct_r(
    const PrimitiveVars& wim1, const PrimitiveVars& wi,
    const PrimitiveVars& wip1, const PrimitiveVars& wip2,
    Limiter lim)
{
    ReconstructedPair rp;
    reconstruct_component(wim1.rho, wi.rho, wip1.rho, wip2.rho, lim, rp.left.rho, rp.right.rho);
    reconstruct_component(wim1.vr, wi.vr, wip1.vr, wip2.vr, lim, rp.left.vr, rp.right.vr);
    reconstruct_component(wim1.vtheta, wi.vtheta, wip1.vtheta, wip2.vtheta, lim, rp.left.vtheta, rp.right.vtheta);
    reconstruct_component(wim1.P, wi.P, wip1.P, wip2.P, lim, rp.left.P, rp.right.P);
    return rp;
}

ReconstructedPair muscl_reconstruct_theta(
    const PrimitiveVars& wjm1, const PrimitiveVars& wj,
    const PrimitiveVars& wjp1, const PrimitiveVars& wjp2,
    Limiter lim)
{
    return muscl_reconstruct_r(wjm1, wj, wjp1, wjp2, lim);
}
