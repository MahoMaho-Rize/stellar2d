#include "state.h"
#include <cmath>

void State::allocate(const Grid& grid) {
    size = grid.total_with_ghost();
    rho.assign(size, 0.0);
    mr.assign(size, 0.0);
    mtheta.assign(size, 0.0);
    E.assign(size, 0.0);
    phi.assign(grid.total_cells(), 0.0);
}

PrimitiveVars State::to_primitive(int k, double gamma) const {
    PrimitiveVars w;
    w.rho = std::fmax(rho[k], 1e-20);
    double inv_rho = 1.0 / w.rho;
    w.vr = mr[k] * inv_rho;                              // Eq. (1.1): m_r = rho * v_r
    w.vtheta = mtheta[k] * inv_rho;                      // Eq. (1.1): m_theta = rho * v_theta
    double ke = 0.5 * (w.vr * w.vr + w.vtheta * w.vtheta); // Eq. (1.1): kinetic energy
    double e_int = E[k] * inv_rho - ke;                  // Eq. (1.1): E = e + ke
    w.P = std::fmax((gamma - 1.0) * w.rho * e_int, 1e-30); // Eq. (1.2)
    return w;
}

void State::from_primitive(int k, const PrimitiveVars& w, double gamma) {
    rho[k] = w.rho;
    mr[k] = w.rho * w.vr;                                // Eq. (1.1)
    mtheta[k] = w.rho * w.vtheta;                        // Eq. (1.1)
    double ke = 0.5 * (w.vr * w.vr + w.vtheta * w.vtheta); // Eq. (1.1)
    double e_int = w.P / ((gamma - 1.0) * w.rho);        // Eq. (1.2) inverted
    E[k] = w.rho * (e_int + ke);                          // Eq. (1.1): rho * E
}

ConservedVars State::get_conserved(int k) const {
    return {rho[k], mr[k], mtheta[k], E[k]};
}

void State::set_conserved(int k, const ConservedVars& u) {
    rho[k] = u.rho;
    mr[k] = u.mr;
    mtheta[k] = u.mtheta;
    E[k] = u.E;
}

void State::copy_from(const State& other) {
    rho = other.rho;
    mr = other.mr;
    mtheta = other.mtheta;
    E = other.E;
    phi = other.phi;
}
