#include "flux.h"
#include "reconstruct.h"
#include "../parallel.h"
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

    auto W = [&](int i, int j) -> PrimitiveVars {
        int k = grid.idx(i, j);
        return state.to_primitive(k, eos);
    };

    // Radial fluxes at faces i+1/2, for i = 0..nr
    std::vector<Flux4> flux_r((nr + 1) * nt);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i <= nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            // Eq. (3.4), (3.5): MUSCL reconstruction at face i+1/2
            auto rp = muscl_reconstruct_r(W(i - 2, j), W(i - 1, j), W(i, j), W(i + 1, j), lim);
            // Eq. (4.7): HLLC numerical flux
            flux_r[i * nt + j] = hllc_flux_r(rp.left, rp.right, eos);
        }
    }

    // Theta fluxes at faces j+1/2, for j = 0..ntheta
    std::vector<Flux4> flux_t(nr * (nt + 1));

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j <= nt; ++j) {
            ReconstructedPair rp;
            bool polar_cap = (j <= 1 || j >= nt - 1);
            if (polar_cap) {
                // Near the polar axis, fall back to first-order theta reconstruction.
                // This sacrifices local accuracy in a tiny cap region to suppress
                // axis-aligned overshoots from high-order reconstruction.
                rp.left = W(i, j - 1);
                rp.right = W(i, j);
            } else {
                // Eq. (3.4), (3.5): MUSCL reconstruction at face j+1/2
                rp = muscl_reconstruct_theta(W(i, j - 2), W(i, j - 1), W(i, j), W(i, j + 1), lim);
            }
            // Near the polar axis, use a more diffusive theta flux to damp
            // coordinate-singularity-driven spikes.
            flux_t[i * (nt + 1) + j] = polar_cap
                ? rusanov_flux_theta(rp.left, rp.right, eos)
                : hllc_flux_theta(rp.left, rp.right, eos);
        }
    }

    // Eq. (2.5): accumulate -div(F)/V
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int flat = i * nt + j;
            double invV = 1.0 / grid.cell_volume[flat]; // Eq. (2.2)

            double Ar_hi = grid.area_r[(i + 1) * nt + j]; // Eq. (2.3)
            double Ar_lo = grid.area_r[i * nt + j];       // Eq. (2.3)
            const Flux4& fr_hi = flux_r[(i + 1) * nt + j];
            const Flux4& fr_lo = flux_r[i * nt + j];

            double At_hi = grid.area_theta[i * (nt + 1) + j + 1]; // Eq. (2.4)
            double At_lo = grid.area_theta[i * (nt + 1) + j];     // Eq. (2.4)
            const Flux4& ft_hi = flux_t[i * (nt + 1) + j + 1];
            const Flux4& ft_lo = flux_t[i * (nt + 1) + j];

            // Eq. (2.5): dU/dt = -(1/V) * sum(A * F_hat)
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

namespace {

double compute_p_geom_theta_for_row(
    const Grid& grid, const State& state, const EOS& eos,
    int i, int j_eval)
{
    int k_eval = grid.idx(i, j_eval);
    int flat_eval = i * grid.ntheta + j_eval;
    PrimitiveVars w_eval = state.to_primitive(k_eval, eos);

    double r2_hi = grid.r_face[i + 1] * grid.r_face[i + 1];
    double r2_lo = grid.r_face[i] * grid.r_face[i];
    double r2_diff_half = 0.5 * (r2_hi - r2_lo);
    double dsin = std::sin(grid.theta_face[j_eval]) - std::sin(grid.theta_face[j_eval + 1]);
    double vol = grid.cell_volume[flat_eval];
    return w_eval.P * r2_diff_half * dsin / vol;
}

} // namespace

void add_geometric_source(
    const Grid& grid, const State& state, const EOS& eos,
    FluxAccumulator& acc, PolarThetaGeomMode polar_theta_geom_mode)
{
    int nr = grid.nr, nt = grid.ntheta;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < nr; ++i) {
        double r = grid.r_center[i];
        double inv_r = 1.0 / r;
        double r2_hi = grid.r_face[i + 1] * grid.r_face[i + 1];
        double r2_lo = grid.r_face[i] * grid.r_face[i];

        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            int flat = i * nt + j;
            PrimitiveVars w = state.to_primitive(k, eos);
            double vol = grid.cell_volume[flat]; // Eq. (2.2)

            // Eq. (5.3): volume-consistent 2P/r source (well-balanced)
            double dcos = std::cos(grid.theta_face[j]) - std::cos(grid.theta_face[j + 1]);
            double P_geom_r = w.P * (r2_hi - r2_lo) * dcos / vol;

            // Eq. (5.1): S_mr = rho*vtheta^2/r + <2P/r>
            acc.dU_mr[flat] += w.rho * w.vtheta * w.vtheta * inv_r + P_geom_r;

            // Eq. (5.4): volume-consistent P*cot(theta)/r source (well-balanced)
            double P_geom_theta = 0.0;
            bool use_adjacent_source =
                (polar_theta_geom_mode == PolarThetaGeomMode::ADJACENT && nt >= 3);
            if (polar_theta_geom_mode == PolarThetaGeomMode::ZERO &&
                (j == 0 || j == nt - 1)) {
                P_geom_theta = 0.0;
            } else if (use_adjacent_source && j == 0) {
                P_geom_theta = compute_p_geom_theta_for_row(grid, state, eos, i, 1);
            } else if (use_adjacent_source && j == nt - 1) {
                P_geom_theta = compute_p_geom_theta_for_row(grid, state, eos, i, nt - 2);
            } else {
                double r2_diff_half = (r2_hi - r2_lo) * 0.5;
                double dsin = std::sin(grid.theta_face[j]) - std::sin(grid.theta_face[j + 1]);
                P_geom_theta = w.P * r2_diff_half * dsin / vol;
            }

            // Eq. (5.2): S_mtheta = <P*cot(theta)/r> - rho*vr*vtheta/r
            acc.dU_mtheta[flat] += P_geom_theta - w.rho * w.vr * w.vtheta * inv_r;
        }
    }
}

void add_gravity_source(
    const Grid& grid, const State& state,
    FluxAccumulator& acc)
{
    int nr = grid.nr, nt = grid.ntheta;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            int flat = i * nt + j;

            // Eq. (6.9a), (6.9b): face-based gravity gradient, radial
            double g_lo_r = 0.0; // Eq. (6.7): Neumann dPhi/dr=0 at r=0
            if (i > 0) {
                g_lo_r = (state.phi[i * nt + j] - state.phi[(i - 1) * nt + j])
                         / (grid.r_center[i] - grid.r_center[i - 1]); // Eq. (6.9a)
            }
            double g_hi_r = 0.0;
            if (i < nr - 1) {
                g_hi_r = (state.phi[(i + 1) * nt + j] - state.phi[i * nt + j])
                         / (grid.r_center[i + 1] - grid.r_center[i]); // Eq. (6.9a)
            } else {
                // At outer boundary: use Dirichlet Phi = -G*M/R, Eq. (6.6)
                // phi at i=nr-1 is already the Dirichlet value from Poisson solve
                // one-sided: use interior gradient
                if (nr >= 2) {
                    g_hi_r = (state.phi[(nr - 1) * nt + j] - state.phi[(nr - 2) * nt + j])
                             / (grid.r_center[nr - 1] - grid.r_center[nr - 2]);
                }
            }

            // Eq. (6.9b): weighted average to cell center
            double dr_lo = (i > 0) ? (grid.r_center[i] - grid.r_center[i - 1]) : 1.0;
            double dr_hi = (i < nr - 1) ? (grid.r_center[i + 1] - grid.r_center[i]) : dr_lo;
            double dphi_dr = (dr_hi * g_lo_r + dr_lo * g_hi_r) / (dr_lo + dr_hi);

            // Eq. (6.10a), (6.10b): face-based gravity gradient, theta
            double g_lo_t = 0.0; // Eq. (6.8): Neumann dPhi/dtheta=0 at theta=0
            if (j > 0) {
                g_lo_t = (state.phi[i * nt + j] - state.phi[i * nt + j - 1])
                         / (grid.theta_center[j] - grid.theta_center[j - 1]); // Eq. (6.10a)
            }
            double g_hi_t = 0.0; // Eq. (6.8): Neumann dPhi/dtheta=0 at theta=pi
            if (j < nt - 1) {
                g_hi_t = (state.phi[i * nt + j + 1] - state.phi[i * nt + j])
                         / (grid.theta_center[j + 1] - grid.theta_center[j]); // Eq. (6.10a)
            }

            // Eq. (6.10b): weighted average to cell center
            double dt_lo = (j > 0) ? (grid.theta_center[j] - grid.theta_center[j - 1]) : 1.0;
            double dt_hi = (j < nt - 1) ? (grid.theta_center[j + 1] - grid.theta_center[j]) : dt_lo;
            double dphi_dtheta = (dt_hi * g_lo_t + dt_lo * g_hi_t) / (dt_lo + dt_hi);

            double rho_val = state.rho[k];
            double r = grid.r_center[i];

            acc.dU_mr[flat] -= rho_val * dphi_dr;                                // Eq. (6.1)
            acc.dU_mtheta[flat] -= rho_val / r * dphi_dtheta;                    // Eq. (6.2)
            double vr = state.mr[k] / rho_val;
            double vt = state.mtheta[k] / rho_val;
            acc.dU_E[flat] -= rho_val * (vr * dphi_dr + vt / r * dphi_dtheta);  // Eq. (6.3)
        }
    }
}
