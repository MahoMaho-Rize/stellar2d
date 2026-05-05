#pragma once

struct GmgLevel {
    int nr, nt;

    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_sin_theta_face, *d_sin_theta_center;
    double *d_cell_volume;

    double *d_phi, *d_rhs, *d_res;
    double *d_phi_tmp;
    double *d_alpha; // variable coefficient (1/ρ for pressure solve)
    double *d_sigma; // Helmholtz mass term (for ∇²u - σu = f)
};

struct GmgGpu {
    void init(int nr, int nt,
              const double* h_r_face, const double* h_theta_face);

    // Constant-coefficient Poisson: ∇²φ = f  (gravity)
    void solve(double* d_rhs, double* d_phi,
               int max_cycles = 50, double tol = 1e-6);

    // Variable-coefficient Poisson: ∇·(α∇p) = f  (pressure projection)
    void solve_varcoeff(double* d_alpha_finest, double* d_rhs, double* d_phi,
                        int max_cycles = 50, double tol = 1e-6);

    // Helmholtz: ∇²u - σ(x)·u = f  (Schur complement for gravity coupling)
    void solve_helmholtz(double* d_sigma_finest, double* d_rhs, double* d_phi,
                         int max_cycles = 50, double tol = 1e-6);

    void destroy();

    int n_levels = 0;
    GmgLevel levels[12];

    static constexpr int NU1 = 5, NU2 = 5;

    void compute_residual(int l);
    double reduce_absmax(int l);

private:
    void build_level(int l, int nr, int nt,
                     const double* h_rf, const double* h_tf);

    void smooth(int l, int n_iters);
    void vcycle(int l);

    void smooth_var(int l, int n_iters);
    void compute_residual_var(int l);
    void restrict_alpha(int fine, int coarse);
    void vcycle_var(int l);

    void smooth_helm(int l, int n_iters);
    void compute_residual_helm(int l);
    void restrict_sigma(int fine, int coarse);
    void vcycle_helm(int l);

    void restrict_level(int fine, int coarse);
    void prolongate_and_correct(int coarse, int fine);

    int h_res_buf_size = 0;
};
