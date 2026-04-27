#include "state.h"

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
    w.rho = rho[k];
    double inv_rho = 1.0 / w.rho;
    w.vr = mr[k] * inv_rho;
    w.vtheta = mtheta[k] * inv_rho;
    double ke = 0.5 * (w.vr * w.vr + w.vtheta * w.vtheta);
    double e_int = E[k] * inv_rho - ke;
    w.P = (gamma - 1.0) * w.rho * e_int;
    return w;
}

void State::from_primitive(int k, const PrimitiveVars& w, double gamma) {
    rho[k] = w.rho;
    mr[k] = w.rho * w.vr;
    mtheta[k] = w.rho * w.vtheta;
    double ke = 0.5 * (w.vr * w.vr + w.vtheta * w.vtheta);
    double e_int = w.P / ((gamma - 1.0) * w.rho);
    E[k] = w.rho * (e_int + ke);
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
