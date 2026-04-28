#include "gmg.h"
#include "../parallel.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

// Standalone RHS builder — replaces PoissonMatrix::set_rhs for the GMG path
void compute_poisson_rhs(const Grid& grid, const std::vector<double>& rho_cells,
                         double G, std::vector<double>& rhs) {
    int nr = grid.nr, nt = grid.ntheta;
    rhs.resize(nr * nt);

    double M_total = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:M_total)
#endif
    for (int k = 0; k < nr * nt; ++k)
        M_total += rho_cells[k] * grid.cell_volume[k];
    M_total *= 2.0 * M_PI;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int flat = i * nt + j;
            if (i == nr - 1)
                rhs[flat] = -G * M_total / grid.r_center[i]; // Eq. (6.6) Dirichlet
            else
                rhs[flat] = 4.0 * M_PI * G * rho_cells[flat]; // Eq. (1.8)
        }
    }
}

// Build geometric data for a multigrid level given its face positions
void PoissonGMG::build_level(int l, const std::vector<double>& rf, const std::vector<double>& tf) {
    Level& lev = levels_[l];
    lev.nr = static_cast<int>(rf.size()) - 1;
    lev.nt = static_cast<int>(tf.size()) - 1;
    int nr = lev.nr, nt = lev.nt;

    lev.r_face = rf;
    lev.theta_face = tf;
    lev.r_center.resize(nr);
    lev.dr.resize(nr);
    lev.theta_center.resize(nt);
    lev.dtheta.resize(nt);
    lev.sin_theta_face.resize(nt + 1);
    lev.sin_theta_center.resize(nt);

    for (int i = 0; i < nr; ++i) {
        lev.r_center[i] = 0.5 * (rf[i] + rf[i + 1]);
        lev.dr[i] = rf[i + 1] - rf[i];
    }
    for (int j = 0; j <= nt; ++j)
        lev.sin_theta_face[j] = std::sin(tf[j]);
    for (int j = 0; j < nt; ++j) {
        lev.theta_center[j] = 0.5 * (tf[j] + tf[j + 1]);
        lev.dtheta[j] = tf[j + 1] - tf[j];
        lev.sin_theta_center[j] = std::sin(lev.theta_center[j]);
    }

    lev.cell_volume.resize(nr * nt);
    for (int i = 0; i < nr; ++i) {
        double r3h = rf[i + 1] * rf[i + 1] * rf[i + 1];
        double r3l = rf[i] * rf[i] * rf[i];
        for (int j = 0; j < nt; ++j) {
            double dcos = std::cos(tf[j]) - std::cos(tf[j + 1]);
            lev.cell_volume[i * nt + j] = (r3h - r3l) / 3.0 * dcos;
        }
    }

    int n = nr * nt;
    lev.phi.assign(n, 0.0);
    lev.rhs.assign(n, 0.0);
    lev.res.assign(n, 0.0);
}

// Stencil coefficients for the discrete Laplacian (Eq. 6.4, 6.5)
// L*phi = cW*phi[i-1,j] + cE*phi[i+1,j] + cS*phi[i,j-1] + cN*phi[i,j+1] + cC*phi[i,j]
void PoissonGMG::stencil_coeffs(const Level& lev, int i, int j,
                                double& cW, double& cE, double& cS, double& cN, double& cC) const {
    int nr = lev.nr, nt = lev.nt;
    double ri = lev.r_center[i];
    double ri2 = ri * ri;

    cW = cE = cS = cN = 0.0;
    cC = 0.0;

    // Outer boundary: identity row (Dirichlet, Eq. 6.6)
    if (i == nr - 1) {
        cC = 1.0;
        return;
    }

    // Radial stencil (Eq. 6.4)
    if (i > 0) {
        double r_lo = lev.r_face[i];
        double dr_lo = lev.r_center[i] - lev.r_center[i - 1];
        cW = r_lo * r_lo / (ri2 * lev.dr[i] * dr_lo);
        cC -= cW;
    }
    if (i < nr - 1) {
        double r_hi = lev.r_face[i + 1];
        double dr_hi = lev.r_center[i + 1] - lev.r_center[i];
        cE = r_hi * r_hi / (ri2 * lev.dr[i] * dr_hi);
        cC -= cE;
    }

    // Theta stencil (Eq. 6.5)
    double sin_j = lev.sin_theta_center[j];
    double dth = lev.dtheta[j];

    if (j > 0) {
        double sin_lo = lev.sin_theta_face[j];
        double dtheta_lo = lev.theta_center[j] - lev.theta_center[j - 1];
        cS = sin_lo / (ri2 * sin_j * dth * dtheta_lo);
        cC -= cS;
    }
    if (j < nt - 1) {
        double sin_hi = lev.sin_theta_face[j + 1];
        double dtheta_hi = lev.theta_center[j + 1] - lev.theta_center[j];
        cN = sin_hi / (ri2 * sin_j * dth * dtheta_hi);
        cC -= cN;
    }
}

void PoissonGMG::init(const Grid& grid) {
    int nr = grid.nr, nt = grid.ntheta;

    // Count levels: coarsen by 2x until min(nr,nt) < 4
    n_levels_ = 1;
    int cr = nr, ct = nt;
    while (cr >= 4 && ct >= 4) {
        cr /= 2;
        ct /= 2;
        n_levels_++;
    }
    levels_.resize(n_levels_);

    // Build finest level from the actual grid
    std::vector<double> rf(grid.r_face.begin(), grid.r_face.end());
    std::vector<double> tf(grid.theta_face.begin(), grid.theta_face.end());
    build_level(0, rf, tf);

    // Build coarser levels by merging pairs of cells
    for (int l = 1; l < n_levels_; ++l) {
        const Level& fine = levels_[l - 1];
        int fnr = fine.nr, fnt = fine.nt;
        int cnr = fnr / 2, cnt = fnt / 2;

        std::vector<double> crf(cnr + 1), ctf(cnt + 1);
        for (int i = 0; i <= cnr; ++i)
            crf[i] = fine.r_face[2 * i];
        for (int j = 0; j <= cnt; ++j)
            ctf[j] = fine.theta_face[2 * j];

        build_level(l, crf, ctf);
    }
}

void PoissonGMG::smooth(int l, int n_iters) {
    Level& lev = levels_[l];
    int nr = lev.nr, nt = lev.nt;

    // Theta-line Gauss-Seidel: for each radial index i, solve the
    // tridiagonal system along theta implicitly. This handles the
    // tight theta-coupling near r=0 on log-stretched grids.
    std::vector<double> a(nt), b(nt), c(nt), d(nt);

    for (int iter = 0; iter < n_iters; ++iter) {
        for (int i = 0; i < nr; ++i) {
            if (i == nr - 1) {
                for (int j = 0; j < nt; ++j)
                    lev.phi[i * nt + j] = lev.rhs[i * nt + j];
                continue;
            }

            for (int j = 0; j < nt; ++j) {
                double cW, cE, cS, cN, cC;
                stencil_coeffs(lev, i, j, cW, cE, cS, cN, cC);

                // RHS: original rhs minus radial off-diagonal contributions
                double rhs_j = lev.rhs[i * nt + j];
                if (i > 0)     rhs_j -= cW * lev.phi[(i - 1) * nt + j];
                if (i < nr-1)  rhs_j -= cE * lev.phi[(i + 1) * nt + j];

                // Tridiagonal: a[j]*phi[j-1] + b[j]*phi[j] + c[j]*phi[j+1] = d[j]
                a[j] = (j > 0) ? cS : 0.0;
                b[j] = cC;
                c[j] = (j < nt - 1) ? cN : 0.0;
                d[j] = rhs_j;
            }

            // Thomas algorithm (tridiagonal solve)
            for (int j = 1; j < nt; ++j) {
                double m = a[j] / b[j - 1];
                b[j] -= m * c[j - 1];
                d[j] -= m * d[j - 1];
            }
            lev.phi[i * nt + (nt - 1)] = d[nt - 1] / b[nt - 1];
            for (int j = nt - 2; j >= 0; --j)
                lev.phi[i * nt + j] = (d[j] - c[j] * lev.phi[i * nt + (j + 1)]) / b[j];
        }
    }
}

void PoissonGMG::compute_residual(int l) {
    Level& lev = levels_[l];
    int nr = lev.nr, nt = lev.nt;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = i * nt + j;
            if (i == nr - 1) {
                lev.res[k] = lev.rhs[k] - lev.phi[k]; // Dirichlet
                continue;
            }
            double cW, cE, cS, cN, cC;
            stencil_coeffs(lev, i, j, cW, cE, cS, cN, cC);

            double Lphi = cC * lev.phi[k];
            if (i > 0)     Lphi += cW * lev.phi[(i - 1) * nt + j];
            if (i < nr-1)  Lphi += cE * lev.phi[(i + 1) * nt + j];
            if (j > 0)     Lphi += cS * lev.phi[i * nt + (j - 1)];
            if (j < nt-1)  Lphi += cN * lev.phi[i * nt + (j + 1)];

            lev.res[k] = lev.rhs[k] - Lphi;
        }
    }
}

// Full-weighting restriction: volume-weighted average of 2x2 fine cells.
// Dirichlet rows (i=nr-1 on fine → ic=cnr-1 on coarse) are set to zero
// since the correction at the boundary must be zero.
void PoissonGMG::restrict_level(int fine, int coarse) {
    const Level& fl = levels_[fine];
    Level& cl = levels_[coarse];
    int cnt = cl.nt, fnt = fl.nt;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int ic = 0; ic < cl.nr; ++ic) {
        for (int jc = 0; jc < cl.nt; ++jc) {
            if (ic == cl.nr - 1) {
                cl.rhs[ic * cnt + jc] = 0.0;
                continue;
            }
            int if0 = 2 * ic, jf0 = 2 * jc;

            double wsum = 0.0, vsum = 0.0;
            for (int di = 0; di < 2; ++di) {
                for (int dj = 0; dj < 2; ++dj) {
                    int fi = if0 + di, fj = jf0 + dj;
                    if (fi >= fl.nr - 1) continue; // skip Dirichlet rows
                    int kf = fi * fnt + fj;
                    double v = fl.cell_volume[kf];
                    wsum += fl.res[kf] * v;
                    vsum += v;
                }
            }
            cl.rhs[ic * cnt + jc] = (vsum > 0.0) ? wsum / vsum : 0.0;
        }
    }

    std::fill(cl.phi.begin(), cl.phi.end(), 0.0);
}

// Piecewise-constant prolongation and correction.
// Skip Dirichlet boundary rows — correction must be zero there.
void PoissonGMG::prolongate_and_correct(int coarse, int fine) {
    const Level& cl = levels_[coarse];
    Level& fl = levels_[fine];
    int fnt = fl.nt;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int ic = 0; ic < cl.nr; ++ic) {
        for (int jc = 0; jc < cl.nt; ++jc) {
            double e = cl.phi[ic * cl.nt + jc];
            for (int di = 0; di < 2; ++di) {
                int fi = 2 * ic + di;
                if (fi == fl.nr - 1) continue; // Dirichlet: no correction
                for (int dj = 0; dj < 2; ++dj)
                    fl.phi[fi * fnt + (2 * jc + dj)] += e;
            }
        }
    }
}

void PoissonGMG::vcycle(int l) {
    if (l == n_levels_ - 1) {
        smooth(l, 200);
        return;
    }

    smooth(l, NU1);
    compute_residual(l);
    restrict_level(l, l + 1);
    vcycle(l + 1);
    prolongate_and_correct(l + 1, l);
    smooth(l, NU2);
}

void PoissonGMG::solve(const double* rhs, double* phi, int max_cycles, double tol) {
    Level& finest = levels_[0];
    int n = finest.nr * finest.nt;

    for (int i = 0; i < n; ++i) {
        finest.phi[i] = phi[i];
        finest.rhs[i] = rhs[i];
    }

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        vcycle(0);

        compute_residual(0);
        double norm = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(max:norm)
#endif
        for (int i = 0; i < n; ++i)
            norm = std::max(norm, std::abs(finest.res[i]));

        if (norm < tol) break;
    }

    for (int i = 0; i < n; ++i)
        phi[i] = finest.phi[i];
}
