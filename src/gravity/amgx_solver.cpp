#include "amgx_solver.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef USE_AMGX

void AmgxPoissonSolver::init(const std::string& config_file) {
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg_, config_file.c_str()));
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc_, cfg_));
    AMGX_SAFE_CALL(AMGX_matrix_create(&A_, rsrc_, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&b_, rsrc_, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&x_, rsrc_, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_solver_create(&solver_, rsrc_, AMGX_mode_dDDI, cfg_));
    initialized_ = true;
}

void AmgxPoissonSolver::setup(const PoissonMatrix& mat) {
    n_ = mat.n;
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(
        A_, mat.n, mat.nnz, 1, 1,
        mat.row_ptr.data(), mat.col_idx.data(), mat.values.data(), nullptr));
    AMGX_SAFE_CALL(AMGX_solver_setup(solver_, A_));
}

void AmgxPoissonSolver::solve(const double* rhs, double* phi) {
    AMGX_SAFE_CALL(AMGX_vector_upload(b_, n_, 1, rhs));
    AMGX_SAFE_CALL(AMGX_vector_upload(x_, n_, 1, phi));
    AMGX_SAFE_CALL(AMGX_solver_solve(solver_, b_, x_));
    AMGX_SAFE_CALL(AMGX_vector_download(x_, phi));
}

void AmgxPoissonSolver::destroy() {
    if (!initialized_) return;
    AMGX_solver_destroy(solver_);
    AMGX_matrix_destroy(A_);
    AMGX_vector_destroy(b_);
    AMGX_vector_destroy(x_);
    AMGX_resources_destroy(rsrc_);
    AMGX_config_destroy(cfg_);
    AMGX_finalize();
    initialized_ = false;
}

#else // CPU fallback: simple Jacobi iteration

void AmgxPoissonSolver::init(const std::string& /*config_file*/) {
    initialized_ = true;
}

void AmgxPoissonSolver::setup(const PoissonMatrix& mat) {
    n_ = mat.n;
    mat_ = &mat;
}

void AmgxPoissonSolver::solve(const double* rhs, double* phi) {
    if (!mat_) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr, "WARNING: AmgX not available, using CPU Jacobi fallback\n");
            warned = true;
        }
        return;
    }

    // Simple Jacobi iteration as CPU fallback
    std::vector<double> phi_new(n_);
    for (int i = 0; i < n_; ++i) phi_new[i] = phi[i];

    for (int iter = 0; iter < 500; ++iter) {
        double max_diff = 0.0;
        for (int i = 0; i < n_; ++i) {
            double diag = 0.0;
            double off_diag_sum = 0.0;
            for (int jj = mat_->row_ptr[i]; jj < mat_->row_ptr[i + 1]; ++jj) {
                if (mat_->col_idx[jj] == i) {
                    diag = mat_->values[jj];
                } else {
                    off_diag_sum += mat_->values[jj] * phi[mat_->col_idx[jj]];
                }
            }
            if (std::abs(diag) > 1e-30) {
                phi_new[i] = (rhs[i] - off_diag_sum) / diag;
            }
            max_diff = std::max(max_diff, std::abs(phi_new[i] - phi[i]));
        }
        for (int i = 0; i < n_; ++i) phi[i] = phi_new[i];
        if (max_diff < 1e-10) break;
    }
}

void AmgxPoissonSolver::destroy() {
    initialized_ = false;
}

#endif
