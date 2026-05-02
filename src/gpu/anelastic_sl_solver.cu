// 2D anelastic / Boussinesq solver with SL spectral Poisson.
// See anelastic_sl_solver.cuh for design notes.
//
// Phase 1a scope: CPU precompute of SL basis (Chebyshev collocation + LAPACK
// via cuSOLVER in host mode), device-side buffer allocation.  No time stepping
// yet.  Sanity check: W̃ at CGL nodes should match the reduced-pressure Liouville
// potential verified in scripts/reduced_pressure_chebyshev.py.

#include "anelastic_sl_solver.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(_e));                                \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)
#endif

#ifndef CUFFT_CHECK
#define CUFFT_CHECK(call)                                                        \
    do {                                                                         \
        cufftResult _e = (call);                                                 \
        if (_e != CUFFT_SUCCESS) {                                               \
            std::fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__,  \
                         (int)_e);                                               \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t _e = (call);                                              \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, \
                         (int)_e);                                               \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)
#endif

#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call)                                                     \
    do {                                                                         \
        cusolverStatus_t _e = (call);                                            \
        if (_e != CUSOLVER_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__,\
                         (int)_e);                                               \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)
#endif

// ============================================================================
// Host helpers: Chebyshev-Gauss-Lobatto grid, differentiation matrix D,
// Clenshaw-Curtis quadrature weights.  Follows Trefethen, "Spectral Methods
// in MATLAB" (2000), chapters 6 and 12.
// ============================================================================
namespace {

// CGL nodes x_k = cos(kπ/N), k = 0..N, on [-1, 1] (descending order).
// Differentiation matrix D (N+1)×(N+1).  Row-major.
static void cheb_diffmat(int N, std::vector<double>& x, std::vector<double>& D) {
    x.resize(N + 1);
    D.assign((N + 1) * (N + 1), 0.0);

    if (N == 0) { x[0] = 1.0; D[0] = 0.0; return; }

    for (int k = 0; k <= N; ++k) x[k] = std::cos(M_PI * k / N);

    std::vector<double> c(N + 1, 1.0);
    c[0] = 2.0; c[N] = 2.0;
    for (int k = 0; k <= N; ++k) c[k] *= ((k & 1) ? -1.0 : 1.0);

    for (int i = 0; i <= N; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j <= N; ++j) {
            double val;
            if (i == j) {
                val = 0.0;  // will fix after
            } else {
                val = (c[i] / c[j]) / (x[i] - x[j]);
            }
            D[i * (N + 1) + j] = val;
            row_sum += val;
        }
        D[i * (N + 1) + i] = -row_sum;
    }
}

// Clenshaw-Curtis weights on CGL grid, Trefethen eq (12.8).  Sum = 2.
static void cc_weights(int N, std::vector<double>& w) {
    w.assign(N + 1, 0.0);
    if (N == 0) { w[0] = 2.0; return; }
    std::vector<double> theta(N + 1);
    for (int k = 0; k <= N; ++k) theta[k] = M_PI * k / N;

    std::vector<double> v(N - 1, 1.0);
    for (int k = 2; k < N; k += 2) {
        double denom = (double)(k * k - 1);
        for (int j = 0; j < N - 1; ++j)
            v[j] -= 2.0 * std::cos(k * theta[j + 1]) / denom;
    }
    if (N % 2 == 0) {
        double denom = (double)(N * N - 1);
        for (int j = 0; j < N - 1; ++j)
            v[j] -= std::cos(N * theta[j + 1]) / denom;
    }
    for (int j = 1; j < N; ++j) w[j] = 2.0 * v[j - 1] / N;
    w[0] = w[N] = 1.0 / (double)(N * N - 1 + (N % 2));
}

// Reduced-pressure Liouville potential W̃ = (ρ')²/(4ρ²) − ρ''/(2ρ)
// evaluated on CGL nodes.  Uses D1 for spectral derivatives (ρ is smooth on
// CGL away from the rho→0 truncation).
static void compute_W_tilde_cgl(
    const std::vector<double>& y,     // CGL grid (ascending)
    const std::vector<double>& rho,   // ρ_0 at y
    const std::vector<double>& D1,    // (N+1)×(N+1) row-major scaled differentiation matrix
    std::vector<double>& W)
{
    int N = (int)y.size();
    W.resize(N);
    std::vector<double> drho(N, 0.0), d2rho(N, 0.0);

    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += D1[i * N + j] * rho[j];
        drho[i] = s;
    }
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += D1[i * N + j] * drho[j];
        d2rho[i] = s;
    }
    for (int i = 0; i < N; ++i) {
        double r = rho[i];
        double dr = drho[i], ddr = d2rho[i];
        W[i] = dr * dr / (4.0 * r * r) - ddr / (2.0 * r);
    }
}

// Lane-Emden n = 3/2 integrator (RK4, simple).  Returns (r/R, ρ/ρ_c) pairs
// at n_pts roughly uniform in xi up to the first zero xi_1.
static void lane_emden_n32(std::vector<double>& r_norm,
                           std::vector<double>& rho_c,
                           int n_pts = 8000) {
    auto rhs = [](double xi, double theta, double dtheta,
                  double& d_theta, double& d_dtheta) {
        d_theta = dtheta;
        if (xi < 1e-10) {
            d_dtheta = -std::pow(std::max(theta, 0.0), 1.5) / 3.0;
        } else {
            double tp = theta > 0.0 ? std::pow(theta, 1.5) : 0.0;
            d_dtheta = -2.0 / xi * dtheta - tp;
        }
    };

    double xi = 1e-6;
    double theta = 1.0 - 1e-12;
    double dtheta = 0.0;
    double h = 1e-3;

    std::vector<double> xs, thetas;
    xs.push_back(xi); thetas.push_back(theta);

    while (theta > 0.0 && xi < 10.0) {
        double k1a, k1b, k2a, k2b, k3a, k3b, k4a, k4b;
        rhs(xi,       theta,              dtheta,              k1a, k1b);
        rhs(xi + h/2, theta + h/2 * k1a,  dtheta + h/2 * k1b,  k2a, k2b);
        rhs(xi + h/2, theta + h/2 * k2a,  dtheta + h/2 * k2b,  k3a, k3b);
        rhs(xi + h,   theta + h   * k3a,  dtheta + h   * k3b,  k4a, k4b);
        double dth  = (k1a + 2*k2a + 2*k3a + k4a) / 6.0;
        double ddth = (k1b + 2*k2b + 2*k3b + k4b) / 6.0;
        double theta_new = theta + h * dth;
        if (theta_new <= 0.0) {
            // linear root-find on [xi, xi+h]
            double alpha = theta / (theta - theta_new);
            xi += alpha * h;
            theta = 0.0;
            xs.push_back(xi); thetas.push_back(0.0);
            break;
        }
        xi += h;
        theta = theta_new;
        dtheta += h * ddth;
        xs.push_back(xi); thetas.push_back(theta);
    }
    double xi_1 = xs.back();

    // Resample to n_pts ascending with xi_1 as upper bound (exclusive).
    r_norm.resize(n_pts);
    rho_c.resize(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double xi_q = xi_1 * ((double)i / (double)(n_pts - 1)) * 0.9995 + 1e-5;
        // linear search for bracket
        int j = 0;
        while (j + 1 < (int)xs.size() && xs[j + 1] < xi_q) ++j;
        double frac = (xi_q - xs[j]) / (xs[j + 1] - xs[j] + 1e-30);
        double th   = thetas[j] + frac * (thetas[j + 1] - thetas[j]);
        if (th < 0.0) th = 0.0;
        r_norm[i] = xi_q / xi_1;
        rho_c[i]  = std::pow(th, 1.5);
    }
}

}  // namespace

// ============================================================================
// AnelasticSLSolver implementation
// ============================================================================

void AnelasticSLSolver::free_all() {
    auto free_ptr = [](double*& p){ if (p) { cudaFree(p); p = nullptr; } };
    auto free_cptr = [](cufftDoubleComplex*& p){ if (p) { cudaFree(p); p = nullptr; } };
    free_ptr(d_u); free_ptr(d_v); free_ptr(d_omega); free_ptr(d_rhs_pi);
    free_ptr(d_rho); free_ptr(d_rho_sqrt_inv);
    free_ptr(d_Psi); free_ptr(d_mu); free_ptr(d_cc_weights);
    free_cptr(d_fhat); free_cptr(d_ghat); free_cptr(d_Ghat);
    free_cptr(d_Qhat); free_cptr(d_qhat); free_cptr(d_pihat);
    free_ptr(d_kx);
    free_ptr(d_reduce);
    if (plan_r2c_x) { cufftDestroy(plan_r2c_x); plan_r2c_x = 0; }
    if (plan_c2r_x) { cufftDestroy(plan_c2r_x); plan_c2r_x = 0; }
    if (cublas)     { cublasDestroy(cublas); cublas = nullptr; }
}

// Build CGL grid on [0, Ly], compute ρ_0 from requested profile,
// build scaled differentiation matrix, compute W̃, solve the SL eigenproblem
// on the host using cuSOLVER dense symmetric eigensolver.
void AnelasticSLSolver::set_background(const std::string& kind, double rho_cut) {
    if (ny == 0) {
        std::fprintf(stderr, "AnelasticSL: set_background called before init (ny=0).\n");
        std::exit(1);
    }
    int N = ny - 1;  // CGL (N+1) points
    std::vector<double> x_cheb, D_raw;
    cheb_diffmat(N, x_cheb, D_raw);

    // Map CGL x ∈ [1, -1] descending → y ∈ [0, Ly] ascending.
    // y_k = (1 - x_k) * Ly / 2.  Note D on y = -2/Ly * D (chain rule).
    h_y_cgl.resize(ny);
    std::vector<double> perm(ny);
    for (int k = 0; k <= N; ++k) perm[k] = k;
    // Already descending, so reverse → ascending.
    std::vector<int> idx(ny);
    for (int k = 0; k <= N; ++k) idx[k] = N - k;

    std::vector<double> y_asc(ny), D_scaled((size_t)ny * ny);
    for (int k = 0; k < ny; ++k)
        y_asc[k] = (1.0 - x_cheb[idx[k]]) * Ly / 2.0;
    h_y_cgl = y_asc;

    // Permute D to ascending order and apply chain rule scaling.
    double scale = -2.0 / Ly;  // dy/dx = -Ly/2, so d/dy = -2/Ly * d/dx
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            D_scaled[(size_t)i * ny + j] = scale * D_raw[(size_t)idx[i] * (N + 1) + idx[j]];
        }
    }

    // Compute rho_0 at y nodes.
    h_rho.resize(ny);
    if (kind == "boussinesq") {
        for (int k = 0; k < ny; ++k) h_rho[k] = 1.0;
    } else if (kind == "lane_emden_1_5") {
        std::vector<double> r_fine, rho_fine;
        lane_emden_n32(r_fine, rho_fine);
        // Find surface truncation r_hi where rho_fine crosses rho_cut.
        int i_lo = 0, i_hi = (int)r_fine.size() - 1;
        for (int i = 0; i < (int)r_fine.size(); ++i) {
            if (rho_fine[i] > rho_cut) { i_lo = i; break; }
        }
        for (int i = (int)r_fine.size() - 1; i >= 0; --i) {
            if (rho_fine[i] > rho_cut) { i_hi = i; break; }
        }
        double r_lo = r_fine[i_lo], r_hi = r_fine[i_hi];
        // Linear interp: map y ∈ [0, Ly] to r ∈ [r_lo, r_hi].
        for (int k = 0; k < ny; ++k) {
            double frac = y_asc[k] / Ly;
            double r = r_lo + frac * (r_hi - r_lo);
            // Bisection on r_fine
            int j = 0;
            while (j + 1 < (int)r_fine.size() && r_fine[j + 1] < r) ++j;
            double a = (r - r_fine[j]) / (r_fine[j + 1] - r_fine[j] + 1e-30);
            h_rho[k] = rho_fine[j] + a * (rho_fine[j + 1] - rho_fine[j]);
            if (h_rho[k] < rho_cut) h_rho[k] = rho_cut;
        }
    } else {
        std::fprintf(stderr, "AnelasticSL: unknown background '%s'.\n", kind.c_str());
        std::exit(1);
    }

    // W̃ on CGL.
    compute_W_tilde_cgl(h_y_cgl, h_rho, D_scaled, h_W_tilde);

    // Clenshaw-Curtis weights on y (scale from [-1,1] to [0, Ly]).
    std::vector<double> w_raw;
    cc_weights(N, w_raw);
    h_cc_weights.resize(ny);
    for (int k = 0; k < ny; ++k)
        h_cc_weights[k] = w_raw[idx[k]] * Ly / 2.0;

    // ── SL eigenproblem on interior nodes (Dirichlet BCs): ────────────
    //   A := -D²_int - diag(W̃_int),   A ψ = μ ψ.
    // Build D² = D_scaled @ D_scaled then strip endpoints.
    int M = ny - 2;
    std::vector<double> D2((size_t)ny * ny, 0.0);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            double s = 0.0;
            for (int k = 0; k < ny; ++k)
                s += D_scaled[(size_t)i * ny + k] * D_scaled[(size_t)k * ny + j];
            D2[(size_t)i * ny + j] = s;
        }
    }
    std::vector<double> A((size_t)M * M, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double val = -D2[(size_t)(i + 1) * ny + (j + 1)];
            if (i == j) val -= h_W_tilde[i + 1];
            A[(size_t)i * M + j] = val;
        }
    }
    // Symmetrise (the interior D² is not exactly symmetric in collocation;
    // the SL eigenproblem is formally self-adjoint so we average).
    for (int i = 0; i < M; ++i) {
        for (int j = i + 1; j < M; ++j) {
            double avg = 0.5 * (A[(size_t)i * M + j] + A[(size_t)j * M + i]);
            A[(size_t)i * M + j] = avg;
            A[(size_t)j * M + i] = avg;
        }
    }

    // Dense symmetric eigensolver on device via cuSOLVER.
    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    double* d_A = nullptr;
    double* d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * (size_t)M * M));
    CUDA_CHECK(cudaMalloc(&d_W, sizeof(double) * (size_t)M));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * (size_t)M * M,
                          cudaMemcpyHostToDevice));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
        solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        M, d_A, M, d_W, &lwork));
    double* d_work = nullptr; int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(double) * lwork));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDsyevd(
        solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        M, d_A, M, d_W, d_work, lwork, d_info));
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::fprintf(stderr, "AnelasticSL: cusolverDnDsyevd info=%d\n", info);
        std::exit(1);
    }

    std::vector<double> mu_all(M), V_all((size_t)M * M);
    CUDA_CHECK(cudaMemcpy(mu_all.data(), d_W, sizeof(double) * (size_t)M,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V_all.data(), d_A, sizeof(double) * (size_t)M * M,
                          cudaMemcpyDeviceToHost));  // column-major eigenvectors

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    cusolverDnDestroy(solver);

    // Build Psi (ny × n_modes) column-major, with Dirichlet zeros at endpoints.
    int Nm = std::min(n_modes, M);
    h_mu.assign(mu_all.begin(), mu_all.begin() + Nm);
    h_Psi.assign((size_t)ny * Nm, 0.0);
    for (int m = 0; m < Nm; ++m) {
        for (int i = 0; i < M; ++i) {
            h_Psi[(size_t)m * ny + (i + 1)] = V_all[(size_t)m * M + i];
        }
    }
    // Normalize with Clenshaw-Curtis inner product: ⟨ψ_m, ψ_m⟩_cc = 1.
    for (int m = 0; m < Nm; ++m) {
        double s = 0.0;
        for (int i = 0; i < ny; ++i) {
            double v = h_Psi[(size_t)m * ny + i];
            s += h_cc_weights[i] * v * v;
        }
        double nrm = std::sqrt(std::max(s, 1e-300));
        for (int i = 0; i < ny; ++i)
            h_Psi[(size_t)m * ny + i] /= nrm;
    }
    n_modes = Nm;  // shrink if we asked for more than available

    std::fprintf(stderr,
        "  AnelasticSL background '%s': ny=%d, n_modes=%d, |W̃|_max=%.3e, μ_0=%.4e, μ_%d=%.4e\n",
        kind.c_str(), ny, n_modes,
        [&]{ double m=0; for (double v : h_W_tilde) m = std::max(m, std::abs(v)); return m; }(),
        h_mu[0], n_modes - 1, h_mu.back());
}

void AnelasticSLSolver::init(int nx_, int ny_, int n_modes_,
                             double Lx_, double Ly_,
                             double nu_, double cfl_) {
    nx = nx_; ny = ny_; n_modes = n_modes_;
    Lx = Lx_; Ly = Ly_;
    nu = nu_; cfl = cfl_;
    nh = nx / 2 + 1;
    ncell = nx * ny;
    ncplx = nh * ny;
    dx = Lx / nx;
    dy_ref = Ly / (ny - 1);

    // Physical fields (y-major: double precision, ny × nx column-major)
    CUDA_CHECK(cudaMalloc(&d_u,       sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_v,       sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_omega,   sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_rhs_pi,  sizeof(double) * ncell));

    CUDA_CHECK(cudaMalloc(&d_rho,          sizeof(double) * ny));
    CUDA_CHECK(cudaMalloc(&d_rho_sqrt_inv, sizeof(double) * ny));
    CUDA_CHECK(cudaMalloc(&d_cc_weights,   sizeof(double) * ny));

    CUDA_CHECK(cudaMalloc(&d_kx, sizeof(double) * nh));

    CUDA_CHECK(cudaMalloc(&d_fhat,  sizeof(cufftDoubleComplex) * ncplx));
    CUDA_CHECK(cudaMalloc(&d_ghat,  sizeof(cufftDoubleComplex) * ncplx));
    CUDA_CHECK(cudaMalloc(&d_Qhat,  sizeof(cufftDoubleComplex) * (size_t)nh * n_modes));
    CUDA_CHECK(cudaMalloc(&d_Ghat,  sizeof(cufftDoubleComplex) * (size_t)nh * n_modes));
    CUDA_CHECK(cudaMalloc(&d_qhat,  sizeof(cufftDoubleComplex) * ncplx));
    CUDA_CHECK(cudaMalloc(&d_pihat, sizeof(cufftDoubleComplex) * ncplx));

    // cuFFT R2C/C2R along x, batched over ny rows.
    // layout: d_u is (ny × nx) row-major in y, so each row is contiguous in x.
    //         In cuFFT terms: n = nx, batch = ny, istride = 1, idist = nx.
    int rank = 1;
    int n[1] = { nx };
    int inembed[1] = { nx };
    int onembed[1] = { nh };
    CUFFT_CHECK(cufftPlanMany(&plan_r2c_x, rank, n,
                              inembed, 1, nx,
                              onembed, 1, nh,
                              CUFFT_D2Z, ny));
    CUFFT_CHECK(cufftPlanMany(&plan_c2r_x, rank, n,
                              onembed, 1, nh,
                              inembed, 1, nx,
                              CUFFT_Z2D, ny));

    // cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&cublas));

    // k_x array
    std::vector<double> h_kx(nh);
    for (int k = 0; k < nh; ++k) h_kx[k] = 2.0 * M_PI * k / Lx;
    CUDA_CHECK(cudaMemcpy(d_kx, h_kx.data(), sizeof(double) * nh,
                          cudaMemcpyHostToDevice));

    dt_current = 0.0;
    step_count = 0;

    std::fprintf(stderr,
        "  AnelasticSL init: %dx%d (nh=%d, cells=%d, cplx=%d), L=(%g,%g), ν=%g, cfl=%g\n",
        nx, ny, nh, ncell, ncplx, Lx, Ly, nu, cfl);
}

// ============================================================================
// The stubs below are placeholders; Phase 1b will implement them.
// ============================================================================

void AnelasticSLSolver::init_zero() {
    CUDA_CHECK(cudaMemset(d_u,     0, sizeof(double) * ncell));
    CUDA_CHECK(cudaMemset(d_v,     0, sizeof(double) * ncell));
    CUDA_CHECK(cudaMemset(d_omega, 0, sizeof(double) * ncell));
}

void AnelasticSLSolver::init_kh_shear(double /*vshear*/, double /*amp*/, int /*k*/) {
    // TODO Phase 1b
    init_zero();
}

double AnelasticSLSolver::step() {
    // TODO Phase 1b
    return 0.0;
}

void AnelasticSLSolver::sl_poisson_solve() {
    // TODO Phase 1a: implement 7-step pipeline here.
}

void AnelasticSLSolver::download_uv(std::vector<double>& h_u, std::vector<double>& h_v) {
    h_u.resize(ncell); h_v.resize(ncell);
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, sizeof(double) * ncell, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, sizeof(double) * ncell, cudaMemcpyDeviceToHost));
}

void AnelasticSLSolver::download_omega(std::vector<double>& h_omega) {
    h_omega.resize(ncell);
    CUDA_CHECK(cudaMemcpy(h_omega.data(), d_omega, sizeof(double) * ncell,
                          cudaMemcpyDeviceToHost));
}
