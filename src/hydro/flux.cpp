#include "flux.h"
#include "reconstruct.h"
#include <cmath>

void FluxAccumulator::allocate(int n) {
    dU_rho.resize(n);
    dU_mr.resize(n);
    dU_mtheta.resize(n);
    dU_E.resize(n);
}

void FluxAccumulator::zero() {
    std::fill(dU_rho.begin(), dU_rho.end(), 0.0);
    std::fill(dU_mr.begin(), dU_mr.end(), 0.0);
    std::fill(dU_mtheta.begin(), dU_mtheta.end(), 0.0);
    std::fill(dU_E.begin(), dU_E.end(), 0.0);
}

void compute_flux_divergence(
    const Grid& grid, const State& state, const EOS& eos,
    FluxAccumulator& acc, Limiter lim)
{
    int nr = grid.nr, nt = grid.ntheta;
    double gamma = eos.gamma;

    auto W = [&](int i, int j) -> PrimitiveVars {
        int k = grid.idx(i, j);
        return state.to_primitive(k, gamma);
    };

    // Radial fluxes: faces i+1/2 for i = 0..nr
    std::vector<Flux4> flux_r((nr + 1) * nt);

    for (int i = 0; i <= nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            auto rp = muscl_reconstruct_r(W(i - 2, j), W(i - 1, j), W(i, j), W(i + 1, j), lim);
            flux_r[i * nt + j] = hllc_flux_r(rp.left, rp.right, gamma);
        }
    }

    // Theta fluxes: faces j+1/2 for j = 0..ntheta
    std::vector<Flux4> flux_t(nr * (nt + 1));

    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j <= nt; ++j) {
            auto rp = muscl_reconstruct_theta(W(i, j - 2), W(i, j - 1), W(i, j), W(i, j + 1), lim);
            flux_t[i * (nt + 1) + j] = hllc_flux_theta(rp.left, rp.right, gamma);
        }
    }

    // Accumulate -div(F)/V
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int flat = i * nt + j;
            double invV = 1.0 / grid.cell_volume[flat];

            double Ar_hi = grid.area_r[(i + 1) * nt + j];
            double Ar_lo = grid.area_r[i * nt + j];
            const Flux4& fr_hi = flux_r[(i + 1) * nt + j];
            const Flux4& fr_lo = flux_r[i * nt + j];

            double At_hi = grid.area_theta[i * (nt + 1) + j + 1];
            double At_lo = grid.area_theta[i * (nt + 1) + j];
            const Flux4& ft_hi = flux_t[i * (nt + 1) + j + 1];
            const Flux4& ft_lo = flux_t[i * (nt + 1) + j];

            acc.dU_rho[flat] = -invV * (Ar_hi * fr_hi.f_rho - Ar_lo * fr_lo.f_rho
                                       + At_hi * ft_hi.f_rho - At_lo * ft_lo.f_rho);
            acc.dU_mr[flat] = -invV * (Ar_hi * fr_hi.f_mr - Ar_lo * fr_lo.f_mr
                                      + At_hi * ft_hi.f_mr - At_lo * ft_lo.f_mr);
            acc.dU_mtheta[flat] = -invV * (Ar_hi * fr_hi.f_mtheta - Ar_lo * fr_lo.f_mtheta
                                          + At_hi * ft_hi.f_mtheta - At_lo * ft_lo.f_mtheta);
            acc.dU_E[flat] = -invV * (Ar_hi * fr_hi.f_E - Ar_lo * fr_lo.f_E
                                     + At_hi * ft_hi.f_E - At_lo * ft_lo.f_E);
        }
    }
}

void add_geometric_source(
    const Grid& grid, const State& state, const EOS& eos,
    FluxAccumulator& acc)
{
    int nr = grid.nr, nt = grid.ntheta;
    double gamma = eos.gamma;

    for (int i = 0; i < nr; ++i) {
        double r = grid.r_center[i];
        double inv_r = 1.0 / r;
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            int flat = i * nt + j;
            PrimitiveVars w = state.to_primitive(k, gamma);

            double cot_theta = std::cos(grid.theta_center[j]) / std::sin(grid.theta_center[j]);

            // S_mr = rho*vtheta^2/r + 2P/r
            acc.dU_mr[flat] += w.rho * w.vtheta * w.vtheta * inv_r + 2.0 * w.P * inv_r;
            // S_mtheta = P*cot(theta)/r - rho*vr*vtheta/r
            acc.dU_mtheta[flat] += w.P * cot_theta * inv_r - w.rho * w.vr * w.vtheta * inv_r;
        }
    }
}

void add_gravity_source(
    const Grid& grid, const State& state,
    FluxAccumulator& acc)
{
    int nr = grid.nr, nt = grid.ntheta;

    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            int flat = i * nt + j;

            // Gradient of phi via central differences on cell-centered phi
            double dphi_dr, dphi_dtheta;

            if (i > 0 && i < nr - 1) {
                dphi_dr = (state.phi[(i + 1) * nt + j] - state.phi[(i - 1) * nt + j])
                          / (grid.r_center[i + 1] - grid.r_center[i - 1]);
            } else if (i == 0) {
                dphi_dr = (state.phi[1 * nt + j] - state.phi[0 * nt + j])
                          / (grid.r_center[1] - grid.r_center[0]);
            } else {
                dphi_dr = (state.phi[(nr - 1) * nt + j] - state.phi[(nr - 2) * nt + j])
                          / (grid.r_center[nr - 1] - grid.r_center[nr - 2]);
            }

            if (j > 0 && j < nt - 1) {
                dphi_dtheta = (state.phi[i * nt + j + 1] - state.phi[i * nt + j - 1])
                              / (grid.theta_center[j + 1] - grid.theta_center[j - 1]);
            } else if (j == 0) {
                dphi_dtheta = (state.phi[i * nt + 1] - state.phi[i * nt + 0])
                              / (grid.theta_center[1] - grid.theta_center[0]);
            } else {
                dphi_dtheta = (state.phi[i * nt + nt - 1] - state.phi[i * nt + nt - 2])
                              / (grid.theta_center[nt - 1] - grid.theta_center[nt - 2]);
            }

            double rho_val = state.rho[k];
            double r = grid.r_center[i];

            // -rho * dPhi/dr
            acc.dU_mr[flat] -= rho_val * dphi_dr;
            // -rho/r * dPhi/dtheta
            acc.dU_mtheta[flat] -= rho_val / r * dphi_dtheta;
            // -rho * v . grad(Phi)
            double vr = state.mr[k] / rho_val;
            double vt = state.mtheta[k] / rho_val;
            acc.dU_E[flat] -= rho_val * (vr * dphi_dr + vt / r * dphi_dtheta);
        }
    }
}
