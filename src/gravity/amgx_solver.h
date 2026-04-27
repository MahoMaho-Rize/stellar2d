#pragma once

#include "poisson.h"
#include <string>

#ifdef USE_AMGX
#include <amgx_c.h>
#endif

class AmgxPoissonSolver {
public:
    void init(const std::string& config_file);
    void setup(const PoissonMatrix& mat);
    void solve(const double* rhs, double* phi);
    void destroy();

private:
    int n_ = 0;
    bool initialized_ = false;
    const PoissonMatrix* mat_ = nullptr;

#ifdef USE_AMGX
    AMGX_config_handle cfg_;
    AMGX_resources_handle rsrc_;
    AMGX_matrix_handle A_;
    AMGX_vector_handle b_;
    AMGX_vector_handle x_;
    AMGX_solver_handle solver_;
#endif
};
