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

#include <algorithm>
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

// ── Small local kernel: pack real A into complex for cuSOLVER Xgeev. ───
static __global__ void k_real_to_cplx_local(cufftDoubleComplex* dst,
                                             const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i].x = src[i]; dst[i].y = 0.0; }
}

// ── Kernel declarations from anelastic_sl_kernels.cu ───────────────────
extern "C" {
    __global__ void k_weight_fhat_inplace(cufftDoubleComplex*, const cufftDoubleComplex*,
                                          const double*, int, int);
    __global__ void k_diag_divide_sl(cufftDoubleComplex*, const cufftDoubleComplex*,
                                     const double*, const double*, int, int);
    __global__ void k_normalize(double*, int, double);
    __global__ void k_mult_ikx_out(cufftDoubleComplex*, const cufftDoubleComplex*,
                                   const double*, double, int, int);
    __global__ void k_mult_mkx2_out(cufftDoubleComplex*, const cufftDoubleComplex*,
                                    const double*, double, int, int);
    __global__ void k_fma_product(double*, double, const double*, const double*, int);
    __global__ void k_fma_scalar(double*, double, const double*, int);
    __global__ void k_rk3_combine(double*, const double*, const double*, const double*,
                                  double, double, double, int);
    __global__ void k_sub_inplace(double*, const double*, int);
    __global__ void k_add_out(double*, const double*, const double*, int);
    __global__ void k_compute_omega(double*, const double*, const double*, int);
    __global__ void k_zero_y_boundary(double*, int, int);
    __global__ void k_max_abs_pass1(const double*, int, double*);
    __global__ void k_dealias_x_inplace(cufftDoubleComplex*, int, int, int);
    __global__ void k_add_buoyancy(double*, const double*, int);
    __global__ void k_sub_N2_v(double*, const double*, const double*, int, int);
    __global__ void k_fma_row(double*, double, const double*, const double*, int, int);
}

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
    free_ptr(d_u); free_ptr(d_v); free_ptr(d_omega); free_ptr(d_rhs_pi); free_ptr(d_pi);
    free_ptr(d_u_orig); free_ptr(d_v_orig); free_ptr(d_b_orig); free_ptr(d_b);
    free_ptr(d_rhs_u);  free_ptr(d_rhs_v); free_ptr(d_rhs_b);
    free_ptr(d_scratch);
    free_ptr(d_Dy);
    free_ptr(d_rho); free_ptr(d_rho_sqrt_inv);
    free_ptr(d_rho_prime); free_ptr(d_N2);
    free_cptr(d_Psi_fwd); free_cptr(d_Psi_inv);
    free_ptr(d_mu); free_ptr(d_cc_weights);
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

    // Map CGL x ∈ [1, -1] descending → y ∈ [0, Ly] ascending via y = (1 + x) · Ly/2.
    //   x_cheb[0] = +1 → y = Ly (want ascending, so reverse)
    //   x_cheb[N] = -1 → y = 0
    // idx[k] = N - k reorders so y_asc[0] = 0, y_asc[N] = Ly.
    h_y_cgl.resize(ny);
    std::vector<int> idx(ny);
    for (int k = 0; k <= N; ++k) idx[k] = N - k;

    std::vector<double> y_asc(ny), D_scaled((size_t)ny * ny);
    for (int k = 0; k < ny; ++k)
        y_asc[k] = (1.0 + x_cheb[idx[k]]) * Ly / 2.0;  // ascending [0, Ly]
    h_y_cgl = y_asc;

    // Chain rule: y = (1 + x) · Ly / 2  ⇒  dy/dx = Ly/2  ⇒  d/dy = (2/Ly) d/dx.
    double scale = 2.0 / Ly;
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
    } else if (kind == "stratified_n2") {
        // Uniform ρ₀ = 1 with a separate, user-supplied constant N² (stored
        // in the `rho_cut` argument for this kind).  This is the analytically
        // tractable Boussinesq-stratified limit: g-modes have the exact
        // dispersion ω² = N²·k_x²/(k_x² + k_y²).
        for (int k = 0; k < ny; ++k) h_rho[k] = 1.0;
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

    // Upload D_scaled to device for the apply_dy DGEMM path.  With F stored
    // row-major (ny × nx), its col-major reinterpretation is (nx × ny) with
    // lda = nx and col-major entry C_cm(ix, jy) == row-major F[jy*nx + ix].
    // We want (∂y F)[jy, ix] = Σ_{jy'} D(jy, jy') · F[jy', ix], which in
    // col-major is C = A · M_cm where
    //   A     = F_cm        of shape (nx × ny),
    //   M_cm  = the (ny × ny) matrix whose col-major entry M_cm(jy', jy) = D(jy, jy').
    // Using cublasDgemm with OP_N on both and lda_B = ny then multiplies by
    //   B_cm(jy', jy) = D(jy, jy')     (note: jy is the COLUMN of B_cm)
    // which is exactly what we store: d_Dy[jy' + jy*ny] = D(jy, jy').
    // In other words, the col-major d_Dy IS the row-major D transposed.  We
    // store D_scaled (which is row-major) directly:
    //   d_Dy[jy' + jy*ny] = D_scaled[jy*ny + jy']     (values match, layout differs)
    // Host-side row-major copy (for 2D EVP assembly etc.)
    h_Dy_row.assign(D_scaled.begin(), D_scaled.end());

    std::vector<double> h_Dy_col((size_t)ny * ny);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            h_Dy_col[(size_t)i + (size_t)j * ny] = D_scaled[(size_t)i * ny + j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_Dy, h_Dy_col.data(),
                          sizeof(double) * (size_t)ny * ny,
                          cudaMemcpyHostToDevice));

    // Min Δy on the non-uniform CGL grid — used for convective CFL.
    dy_min = Ly;
    for (int k = 1; k < ny; ++k) {
        double dy = h_y_cgl[k] - h_y_cgl[k - 1];
        if (dy < dy_min) dy_min = dy;
    }

    // ρ₀'(y) on CGL = D · ρ₀  (used by anelastic continuity and N²).
    h_rho_prime.assign(ny, 0.0);
    for (int i = 0; i < ny; ++i) {
        double s = 0.0;
        for (int j = 0; j < ny; ++j) s += D_scaled[(size_t)i * ny + j] * h_rho[j];
        h_rho_prime[i] = s;
    }
    CUDA_CHECK(cudaMemcpy(d_rho_prime, h_rho_prime.data(),
                          sizeof(double) * ny, cudaMemcpyHostToDevice));

    // Brunt-Väisälä N²(y) for anelastic dynamics.  For a stratified ρ₀(y)
    // in a uniform gravity g·ê_y the polytropic N² = g · (−ρ₀'/ρ₀ − g/c_s²).
    // For the Phase 1d minimum we use a simple proxy: N² = g · max(0, −ρ₀'/ρ₀)
    // (convectively stable where ρ decreases with height).  g=1 by default.
    // Boussinesq background sets N² ≡ 0.
    is_anelastic = (kind != "boussinesq");
    h_N2.assign(ny, 0.0);
    if (kind == "stratified_n2") {
        // Constant N² supplied via the rho_cut argument (reused as N²-value
        // channel for this branch; ρ₀ is uniform so no surface cutoff needed).
        for (int i = 0; i < ny; ++i) h_N2[i] = rho_cut;
    } else if (is_anelastic) {
        // Lane-Emden proxy: N² ≈ g·max(0, −ρ'/ρ) (convectively stable where ρ↓).
        const double g_const = 1.0;
        for (int i = 0; i < ny; ++i) {
            double dlnrho = h_rho_prime[i] / std::max(h_rho[i], 1e-12);
            double n2 = -g_const * dlnrho;
            h_N2[i] = std::max(0.0, n2);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_N2, h_N2.data(),
                          sizeof(double) * ny, cudaMemcpyHostToDevice));

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
    // Build A in COLUMN-MAJOR layout for cuSOLVER: A_cm[i + j*M] = A(i, j).
    std::vector<double> A_cm((size_t)M * M, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double val = -D2[(size_t)(i + 1) * ny + (j + 1)];
            if (i == j) val -= h_W_tilde[i + 1];
            A_cm[(size_t)i + (size_t)j * M] = val;
        }
    }

    // Use cusolverDnXgeev (non-symmetric eigensolver) — the Chebyshev
    // collocation operator is self-adjoint under the CC-weighted inner
    // product but NOT in Euclidean, so we cannot use dsyevd.
    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    cusolverDnParams_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    double* d_A = nullptr;
    cuDoubleComplex* d_W = nullptr;         // complex eigenvalues
    cuDoubleComplex* d_VR = nullptr;         // right eigenvectors (complex)
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * (size_t)M * M));
    CUDA_CHECK(cudaMalloc(&d_W, sizeof(cuDoubleComplex) * (size_t)M));
    CUDA_CHECK(cudaMalloc(&d_VR, sizeof(cuDoubleComplex) * (size_t)M * M));
    CUDA_CHECK(cudaMemcpy(d_A, A_cm.data(), sizeof(double) * (size_t)M * M,
                          cudaMemcpyHostToDevice));

    size_t work_dev = 0, work_host = 0;
    // Note: cusolverDnXgeev requires A, W, VL, VR, computeType all CUDA_C_64F.
    // Convert the real A into complex (real part only) via a small kernel.
    cuDoubleComplex* d_Ac = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Ac, sizeof(cuDoubleComplex) * (size_t)M * M));
    {
        int block = 256;
        int total = M * M;
        int grid = (total + block - 1) / block;
        k_real_to_cplx_local<<<grid, block>>>(d_Ac, d_A, total);
    }

    CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
        solver, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
        (int64_t)M,
        CUDA_C_64F, d_Ac, (int64_t)M,
        CUDA_C_64F, d_W,
        CUDA_C_64F, nullptr, (int64_t)M,    // VL unused
        CUDA_C_64F, d_VR,    (int64_t)M,
        CUDA_C_64F, &work_dev, &work_host));
    void* d_work = nullptr; void* h_work = nullptr;
    if (work_dev) CUDA_CHECK(cudaMalloc(&d_work, work_dev));
    if (work_host) h_work = std::malloc(work_host);
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXgeev(
        solver, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
        (int64_t)M,
        CUDA_C_64F, d_Ac, (int64_t)M,
        CUDA_C_64F, d_W,
        CUDA_C_64F, nullptr, (int64_t)M,
        CUDA_C_64F, d_VR,    (int64_t)M,
        CUDA_C_64F, d_work, work_dev, h_work, work_host, d_info));
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::fprintf(stderr, "AnelasticSL: cusolverDnXgeev info=%d\n", info);
        std::exit(1);
    }

    std::vector<cuDoubleComplex> W_cplx(M), VR_cplx((size_t)M * M);
    CUDA_CHECK(cudaMemcpy(W_cplx.data(), d_W, sizeof(cuDoubleComplex) * M,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(VR_cplx.data(), d_VR, sizeof(cuDoubleComplex) * (size_t)M * M,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Ac));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_VR));
    CUDA_CHECK(cudaFree(d_info));
    if (d_work) cudaFree(d_work);
    if (h_work) std::free(h_work);
    cusolverDnDestroyParams(params);
    cusolverDnDestroy(solver);

    // Take real parts (complex components should be ≤ 1e-10 for a well-
    // posed SL operator).  Warn if any imaginary component is large.
    std::vector<double> mu_all(M), V_all((size_t)M * M);
    double max_imag = 0.0;
    for (int i = 0; i < M; ++i) {
        mu_all[i] = W_cplx[i].x;
        max_imag = std::max(max_imag, std::abs(W_cplx[i].y));
    }
    for (size_t k = 0; k < (size_t)M * M; ++k) {
        V_all[k] = VR_cplx[k].x;
        max_imag = std::max(max_imag, std::abs(VR_cplx[k].y));
    }
    if (max_imag > 1e-6) {
        std::fprintf(stderr,
            "  AnelasticSL WARNING: eigenpair has significant imag part (%.3e).\n",
            max_imag);
    }

    // Sort by real part (ascending).
    std::vector<int> sort_idx(M);
    for (int i = 0; i < M; ++i) sort_idx[i] = i;
    std::sort(sort_idx.begin(), sort_idx.end(),
              [&](int a, int b){ return mu_all[a] < mu_all[b]; });
    std::vector<double> mu_sorted(M), V_sorted((size_t)M * M);
    for (int i = 0; i < M; ++i) {
        mu_sorted[i] = mu_all[sort_idx[i]];
        for (int j = 0; j < M; ++j)
            V_sorted[(size_t)j + (size_t)i * M] =
                V_all[(size_t)j + (size_t)sort_idx[i] * M];
    }
    mu_all = mu_sorted;
    V_all = V_sorted;

    // Build Psi (ny × n_modes) column-major, with Dirichlet zeros at endpoints.
    int Nm = n_modes;
    if (Nm > M) {
        std::fprintf(stderr,
            "  AnelasticSL: requested n_modes=%d exceeds interior points M=%d; "
            "reduce n_modes or increase ny.\n", Nm, M);
        std::exit(1);
    }
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

    // ── Upload to device (init() must have been called already) ─────────
    if (d_rho == nullptr) {
        std::fprintf(stderr,
            "  AnelasticSL: set_background called before init(); buffers not allocated.\n");
        std::exit(1);
    }

    // ρ_0 and 1/√ρ_0 on CGL grid
    std::vector<double> h_rho_sqrt_inv(ny);
    for (int k = 0; k < ny; ++k) h_rho_sqrt_inv[k] = 1.0 / std::sqrt(h_rho[k]);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(),
                          sizeof(double) * ny, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho_sqrt_inv, h_rho_sqrt_inv.data(),
                          sizeof(double) * ny, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cc_weights, h_cc_weights.data(),
                          sizeof(double) * ny, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, h_mu.data(),
                          sizeof(double) * n_modes, cudaMemcpyHostToDevice));

    // CC-normalized spectral-Galerkin basis (matches scripts/reduced_pressure_chebyshev.py).
    // Note: the h_Psi currently contains cuSOLVER's Euclidean-orthonormal eigenvectors.
    // Re-normalize to CC inner product: <ψ_m, ψ_n>_cc = Σ_j w_cc,j ψ_m[j] ψ_n[j] = δ_mn.
    // (Already done above, but the Euclidean normalisation happened in eigh.)
    for (int m = 0; m < n_modes; ++m) {
        double s = 0.0;
        for (int i = 0; i < ny; ++i) {
            double v = h_Psi[(size_t)m * ny + i];
            s += h_cc_weights[i] * v * v;
        }
        double nrm = std::sqrt(std::max(s, 1e-300));
        for (int i = 0; i < ny; ++i)
            h_Psi[(size_t)m * ny + i] /= nrm;
    }

    // Psi_fwd = diag(w_cc) · Psi (forward: G_n = Σ_j w_cc,j · ψ_n[j] · g[j]).
    // Psi_inv = Psi.
    std::vector<cufftDoubleComplex> h_Psi_fwd((size_t)ny * n_modes);
    std::vector<cufftDoubleComplex> h_Psi_inv((size_t)ny * n_modes);
    for (int m = 0; m < n_modes; ++m) {
        for (int i = 0; i < ny; ++i) {
            double psi = h_Psi[(size_t)m * ny + i];
            size_t off = (size_t)m * ny + i;
            h_Psi_inv[off].x = psi;
            h_Psi_inv[off].y = 0.0;
            h_Psi_fwd[off].x = h_cc_weights[i] * psi;
            h_Psi_fwd[off].y = 0.0;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_Psi_fwd, h_Psi_fwd.data(),
                          sizeof(cufftDoubleComplex) * (size_t)ny * n_modes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Psi_inv, h_Psi_inv.data(),
                          sizeof(cufftDoubleComplex) * (size_t)ny * n_modes,
                          cudaMemcpyHostToDevice));
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

    // Physical fields (row-major ny × nx, y slow, x fast).
    CUDA_CHECK(cudaMalloc(&d_u,       sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_v,       sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_omega,   sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_rhs_pi,  sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_pi,      sizeof(double) * ncell));

    // RK3 scratch buffers for primitive-variable projection time stepping.
    CUDA_CHECK(cudaMalloc(&d_u_orig,  sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_v_orig,  sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_b_orig,  sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_b,       sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_rhs_u,   sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_rhs_v,   sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_rhs_b,   sizeof(double) * ncell));
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(double) * ncell));
    CUDA_CHECK(cudaMemset(d_b, 0, sizeof(double) * ncell));

    // Chebyshev differentiation matrix on [0, Ly], col-major ny × ny.
    CUDA_CHECK(cudaMalloc(&d_Dy,      sizeof(double) * (size_t)ny * ny));

    CUDA_CHECK(cudaMalloc(&d_rho,          sizeof(double) * ny));
    CUDA_CHECK(cudaMalloc(&d_rho_sqrt_inv, sizeof(double) * ny));
    CUDA_CHECK(cudaMalloc(&d_rho_prime,    sizeof(double) * ny));
    CUDA_CHECK(cudaMalloc(&d_N2,           sizeof(double) * ny));
    CUDA_CHECK(cudaMalloc(&d_cc_weights,   sizeof(double) * ny));
    CUDA_CHECK(cudaMemset(d_N2, 0, sizeof(double) * ny));   // Boussinesq default

    CUDA_CHECK(cudaMalloc(&d_kx, sizeof(double) * nh));

    CUDA_CHECK(cudaMalloc(&d_fhat,  sizeof(cufftDoubleComplex) * ncplx));
    CUDA_CHECK(cudaMalloc(&d_ghat,  sizeof(cufftDoubleComplex) * ncplx));
    CUDA_CHECK(cudaMalloc(&d_Qhat,  sizeof(cufftDoubleComplex) * (size_t)nh * n_modes));
    CUDA_CHECK(cudaMalloc(&d_Ghat,  sizeof(cufftDoubleComplex) * (size_t)nh * n_modes));
    CUDA_CHECK(cudaMalloc(&d_qhat,  sizeof(cufftDoubleComplex) * ncplx));
    CUDA_CHECK(cudaMalloc(&d_pihat, sizeof(cufftDoubleComplex) * ncplx));
    CUDA_CHECK(cudaMalloc(&d_Psi_fwd, sizeof(cufftDoubleComplex) * (size_t)ny * n_modes));
    CUDA_CHECK(cudaMalloc(&d_Psi_inv, sizeof(cufftDoubleComplex) * (size_t)ny * n_modes));
    CUDA_CHECK(cudaMalloc(&d_mu,      sizeof(double) * n_modes));

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

void AnelasticSLSolver::init_kh_shear(double vshear, double amp, int k) {
    // KH Boussinesq with Dirichlet velocity BC (u=v=0 on y-walls).  This is
    // needed for compatibility with the SL-Poisson pressure projection: ψ_n
    // all vanish at y=0, Ly, so the projection can only kill divergence that
    // is itself expressible in the Dirichlet basis.  We therefore multiply
    // the tanh shear profile by a smooth taper T(y) that is 1 over most of
    // the interior and 0 at the walls.  The KH dynamics live in the interior
    // where T ≈ 1; the wall taper only affects a thin boundary layer.
    //   u(x,y) = vshear · T(y) · [tanh((y-y_lo)/δ) - tanh((y-y_hi)/δ) - 1]
    //   v(x,y) = amp · T(y) · sin(kx·x) · (G_lo + G_hi)
    //   T(y) = sin²(π · y / Ly)   (smooth, 0 at walls, 1 at Ly/2, 1 order zero)
    double delta   = std::fmax(8.0 * dy_ref, 0.02 * Ly);
    double y_lo    = 0.25 * Ly;
    double y_hi    = 0.75 * Ly;
    double sigma   = 0.05 * Ly;
    double kx_phys = k * 2.0 * M_PI / Lx;

    std::vector<double> h_u(ncell), h_v(ncell);
    for (int jy = 0; jy < ny; ++jy) {
        double y = h_y_cgl[jy];
        double T = std::sin(M_PI * y / Ly);
        T = T * T;                                       // sin² wall taper
        double t1 = std::tanh((y - y_lo) / delta);
        double t2 = std::tanh((y - y_hi) / delta);
        double u_y = T * vshear * (t1 - t2 - 1.0);
        double G1 = std::exp(-(y - y_lo) * (y - y_lo) / (sigma * sigma));
        double G2 = std::exp(-(y - y_hi) * (y - y_hi) / (sigma * sigma));
        double G  = T * (G1 + G2);
        for (int ix = 0; ix < nx; ++ix) {
            double x = ix * dx;
            double sn = std::sin(kx_phys * x);
            int kidx = jy * nx + ix;
            h_u[kidx] = u_y;
            h_v[kidx] = amp * sn * G;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), sizeof(double) * ncell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), sizeof(double) * ncell, cudaMemcpyHostToDevice));

    dt_current = 0.0;
    step_count = 0;
    // Project IC onto divergence-free subspace — ∂v/∂x is nonzero from the
    // perturbation, so ∇·u ≠ 0 at t=0 before projection.
    project_div_free();
    std::fprintf(stderr,
        "  AnelasticSL KH Boussinesq: |vx|=%g, amp=%g, k=%d (δ=%.3g, σ=%.3g)\n",
        vshear, amp, k, delta, sigma);
}

// ── Phase 1d IC: internal-gravity-wave eigenmode seed ───────────────────
// Seeds u = v = 0, b = amp · sin(k_x · 2π · x/Lx) · sin(k_y · π · y/Ly).
// Linearised about ρ₀ = 1, constant N², this is the exact eigenmode of
// the incompressible Boussinesq-stratified system with analytic frequency
//     ω² = N² · k_x² / (k_x² + k_y²)
// where k_x (per-x) = k_x_int · 2π/Lx  and  k_y = k_y_int · π/Ly
// (sin on [0, Ly] matches Dirichlet wall BC).
void AnelasticSLSolver::init_gmode_pulsation(double amp, int k_y) {
    // Debug mode (env ANSL_PROJECT_TEST=1): seed v = sin(k_x x)·sin(k_y π y/Ly)
    // instead of b, then run a single projection to check the analytic
    // correction  v → v · k_x²/(k_x² + k_y²).
    const bool projection_test = (std::getenv("ANSL_PROJECT_TEST") != nullptr);
    std::vector<double> h_b(ncell, 0.0);
    std::vector<double> h_v(ncell, 0.0);
    const double kx_phys = 2.0 * M_PI / Lx;    // x-wavenumber: k_x_int = 1
    for (int jy = 0; jy < ny; ++jy) {
        double y = h_y_cgl[jy];
        double shape = std::sin((double)k_y * M_PI * y / Ly);
        for (int ix = 0; ix < nx; ++ix) {
            double x = ix * dx;
            double f = amp * shape * std::sin(kx_phys * x);
            if (projection_test) h_v[jy * nx + ix] = f;
            else                 h_b[jy * nx + ix] = f;
        }
    }
    CUDA_CHECK(cudaMemset(d_u, 0, sizeof(double) * ncell));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), sizeof(double) * ncell,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), sizeof(double) * ncell,
                          cudaMemcpyHostToDevice));
    dt_current = 0.0;
    step_count = 0;
    if (projection_test) {
        // Capture v BEFORE projection, run projection, compute expected ratio
        std::vector<double> v_before(ncell), v_after(ncell);
        CUDA_CHECK(cudaMemcpy(v_before.data(), d_v, sizeof(double) * ncell,
                              cudaMemcpyDeviceToHost));
        project_div_free();
        CUDA_CHECK(cudaMemcpy(v_after.data(), d_v, sizeof(double) * ncell,
                              cudaMemcpyDeviceToHost));
        double ky_phys = k_y * M_PI / Ly;
        double expected_ratio = kx_phys * kx_phys /
                                (kx_phys * kx_phys + ky_phys * ky_phys);
        // Probe at (jy_mid, ix_mid) where sin is largest
        int jy = ny / 2, ix = nx / 4;
        double vb = v_before[jy * nx + ix];
        double va = v_after[jy * nx + ix];
        double actual_ratio = va / vb;
        std::fprintf(stderr,
            "  [PROJECT TEST] v_before=%.6e, v_after=%.6e, ratio=%.6f, expected=%.6f, diff=%.3e\n",
            vb, va, actual_ratio, expected_ratio,
            std::fabs(actual_ratio - expected_ratio));
    } else {
        project_div_free();
    }
    std::fprintf(stderr,
        "  AnelasticSL gmode IC: amp=%g, k_y=%d, kx_int=1, anelastic=%d%s\n",
        amp, k_y, (int)is_anelastic,
        projection_test ? ", PROJECTION TEST" : "");
}

double AnelasticSLSolver::probe_v_center() {
    // Sample v at (jy = ny/2, ix = nx/4).  Single-cell D2H copy (slow-ish
    // but called infrequently).
    int jy = ny / 2, ix = nx / 4;
    int off = jy * nx + ix;
    double val = 0.0;
    CUDA_CHECK(cudaMemcpy(&val, d_v + off, sizeof(double), cudaMemcpyDeviceToHost));
    return val;
}

void AnelasticSLSolver::download_b(std::vector<double>& h_b) {
    h_b.resize(ncell);
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, sizeof(double) * ncell, cudaMemcpyDeviceToHost));
}

// ── Chebyshev D1 on arbitrary interval, ascending nodes ────────────────
static void cheb_D1_interval(int N, double a, double b,
                             std::vector<double>& x_asc,
                             std::vector<double>& D1_row) {
    // Trefethen-descending
    std::vector<double> x(N + 1), D((N + 1) * (N + 1));
    for (int k = 0; k <= N; ++k) x[k] = std::cos(M_PI * k / (double)N);
    auto c = [N](int k) {
        double v = 1.0; if (k == 0 || k == N) v = 2.0;
        return ((k & 1) ? -1.0 : 1.0) * v;
    };
    for (int i = 0; i <= N; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j <= N; ++j) {
            double v;
            if (i == j) v = 0.0;
            else v = (c(i) / c(j)) / (x[i] - x[j]);
            D[(size_t)i * (N + 1) + j] = v;
            if (i != j) row_sum += v;
        }
        D[(size_t)i * (N + 1) + i] = -row_sum;
    }
    // Reorder ascending; also scale for [a, b]: d/dy = (2/(b-a)) d/dx
    double scale = 2.0 / (b - a);
    x_asc.resize(N + 1);
    D1_row.resize((size_t)(N + 1) * (N + 1));
    for (int i = 0; i <= N; ++i) {
        int ii = N - i;
        x_asc[i] = a + (1.0 + x[ii]) * (b - a) / 2.0;
    }
    for (int i = 0; i <= N; ++i) {
        int ii = N - i;
        for (int j = 0; j <= N; ++j) {
            int jj = N - j;
            D1_row[(size_t)i * (N + 1) + j] = scale * D[(size_t)ii * (N + 1) + jj];
        }
    }
}

// ── Exp K: 4-var full GYRE-compatible g-mode EVP on CGL ─────────────────
void AnelasticSLSolver::solve_gmode_full_chebyshev(
        const std::vector<double>& x_nodes,
        const std::vector<double>& V_2,
        const std::vector<double>& U,
        const std::vector<double>& A_star,
        const std::vector<double>& c_1,
        const std::vector<double>& Gamma_1,
        int ell,
        int n_modes_out,
        std::vector<double>& omega_sq_out,
        std::vector<double>& eigvecs_y1_out) {
    const int Nr = (int)x_nodes.size();
    if (Nr < 4) { std::fprintf(stderr, "solve_gmode_full_chebyshev: Nr too small\n"); std::exit(1); }
    const int N = Nr - 1;
    const int size = 4 * Nr;
    const double lam = (double)ell * (ell + 1.0);

    // Build D1 on [x_nodes[0], x_nodes.back()]
    std::vector<double> x_chk, D1_row;
    cheb_D1_interval(N, x_nodes.front(), x_nodes.back(), x_chk, D1_row);
    // sanity: x_chk should equal x_nodes
    double max_x_diff = 0.0;
    for (int i = 0; i < Nr; ++i) {
        double d = std::fabs(x_chk[i] - x_nodes[i]);
        if (d > max_x_diff) max_x_diff = d;
    }
    if (max_x_diff > 1e-10) {
        std::fprintf(stderr,
            "solve_gmode_full_chebyshev: x_nodes not matching CGL grid "
            "(max_diff=%.3e). Use cheb_D1_interval output.\n", max_x_diff);
    }

    // V_g = V_2 · x² / Γ_1 ;  A_iso = A* · (alpha_gam if A*>0 else 1) — alpha_gam=1
    std::vector<double> V_g(Nr), A_iso(Nr);
    for (int i = 0; i < Nr; ++i) {
        V_g[i]   = V_2[i] * x_nodes[i] * x_nodes[i] / Gamma_1[i];
        A_iso[i] = A_star[i];       // alpha_gam=1 ⇒ A_iso = A*
    }

    // Build P, Q row-major (size × size), later transpose to col-major.
    // Row index convention:
    //   eq1: row r in [0, Nr)         (y_1 equation)
    //   eq2: row r in [Nr, 2Nr)       (y_2)
    //   eq3: row r in [2Nr, 3Nr)      (y_3)
    //   eq4: row r in [3Nr, 4Nr)      (y_4)
    // iy1(n)=n, iy2(n)=Nr+n, iy3(n)=2Nr+n, iy4(n)=3Nr+n.
    auto iy1 = [&](int n) { return n; };
    auto iy2 = [&](int n) { return Nr + n; };
    auto iy3 = [&](int n) { return 2 * Nr + n; };
    auto iy4 = [&](int n) { return 3 * Nr + n; };
    std::vector<double> P_rm((size_t)size * size, 0.0);
    std::vector<double> Q_rm((size_t)size * size, 0.0);
    auto Pref = [&](int r, int c) -> double& { return P_rm[(size_t)r * size + c]; };
    auto Qref = [&](int r, int c) -> double& { return Q_rm[(size_t)r * size + c]; };

    // xD1_row[i, j] = x[i] * D1[i, j]
    auto xD1 = [&](int i, int j) { return x_nodes[i] * D1_row[(size_t)i * Nr + j]; };

    // eq1: P[r, :Nr] = xD1[i, :] - diag(V_g-ell-1);  P[r, Nr:2Nr] += diag(V_g)
    //       Q[r, iy2(i)] += λ/c_1;  Q[r, iy3(i)] += λ/c_1
    for (int i = 0; i < Nr; ++i) {
        int r = i;
        for (int j = 0; j < Nr; ++j) {
            Pref(r, iy1(j)) = xD1(i, j);
        }
        Pref(r, iy1(i)) -= (V_g[i] - (double)ell - 1.0);
        Pref(r, iy2(i)) += V_g[i];
        Qref(r, iy2(i)) += lam / c_1[i];
        Qref(r, iy3(i)) += lam / c_1[i];
    }
    // eq2
    for (int i = 0; i < Nr; ++i) {
        int r = Nr + i;
        Pref(r, iy1(i)) = c_1[i];
        Qref(r, iy1(i)) = A_iso[i];
        for (int j = 0; j < Nr; ++j) {
            Qref(r, iy2(j)) = xD1(i, j);
        }
        Qref(r, iy2(i)) -= (A_star[i] - U[i] + 3.0 - (double)ell);
        Qref(r, iy4(i)) += 1.0;
    }
    // eq3
    for (int i = 0; i < Nr; ++i) {
        int r = 2 * Nr + i;
        for (int j = 0; j < Nr; ++j) {
            Qref(r, iy3(j)) = xD1(i, j);
        }
        Qref(r, iy3(i)) -= (3.0 - U[i] - (double)ell);
        Qref(r, iy4(i)) -= 1.0;
    }
    // eq4
    for (int i = 0; i < Nr; ++i) {
        int r = 3 * Nr + i;
        Qref(r, iy1(i)) -= U[i] * A_star[i];
        Qref(r, iy2(i)) -= U[i] * V_g[i];
        Qref(r, iy3(i)) -= lam;
        for (int j = 0; j < Nr; ++j) {
            Qref(r, iy4(j)) = xD1(i, j);
        }
        Qref(r, iy4(i)) -= (2.0 - U[i] - (double)ell);
    }

    // BCs (overwrite rows)
    // IB1 (has ω²):  c_1(0) ω² y_1(0) - ell y_2(0) - ell y_3(0) = 0
    //               ⇒ ω² [c_1 y_1] = ell y_2 + ell y_3
    {
        int r = 0;
        for (int c = 0; c < size; ++c) { Pref(r, c) = 0.0; Qref(r, c) = 0.0; }
        Pref(r, iy1(0)) = c_1[0];
        Qref(r, iy2(0)) = (double)ell;
        Qref(r, iy3(0)) = (double)ell;
    }
    // IB2: ell y_3(0) - y_4(0) = 0
    {
        int r = 2 * Nr;
        for (int c = 0; c < size; ++c) { Pref(r, c) = 0.0; Qref(r, c) = 0.0; }
        Qref(r, iy3(0)) = (double)ell;
        Qref(r, iy4(0)) = -1.0;
    }
    // OB1: y_1(R) - y_2(R) = 0
    {
        int r = Nr - 1;
        for (int c = 0; c < size; ++c) { Pref(r, c) = 0.0; Qref(r, c) = 0.0; }
        Qref(r, iy1(Nr - 1)) =  1.0;
        Qref(r, iy2(Nr - 1)) = -1.0;
    }
    // OB2: U(R) y_1(R) + (ell+1) y_3(R) + y_4(R) = 0
    {
        int r = 4 * Nr - 1;
        for (int c = 0; c < size; ++c) { Pref(r, c) = 0.0; Qref(r, c) = 0.0; }
        Qref(r, iy1(Nr - 1)) = U[Nr - 1];
        Qref(r, iy3(Nr - 1)) = (double)(ell + 1);
        Qref(r, iy4(Nr - 1)) = 1.0;
    }

    // Transpose to col-major for cuSOLVER.
    std::vector<double> P_cm((size_t)size * size), Q_cm((size_t)size * size);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            P_cm[(size_t)i + (size_t)j * size] = P_rm[(size_t)i * size + j];
            Q_cm[(size_t)i + (size_t)j * size] = Q_rm[(size_t)i * size + j];
        }

    // ---- GPU: P^-1 Q via getrf/getrs, then Xgeev on M = P^-1 Q ------------
    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    cusolverDnParams_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    // For Exp K the P matrix has zero rows (eq3, eq4 are algebraic
    // constraints without ω²), so P is singular.  Instead factor Q and
    // solve Q · M = P for M = Q⁻¹ P.  The eigenvalues λ of M then
    // satisfy Q u = ω² P u  ⇒  (Q⁻¹ P) u = (1/ω²) u, i.e. ω² = 1/λ.
    double *d_P = nullptr, *d_Q = nullptr;
    int *d_ipiv = nullptr, *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_P, sizeof(double) * (size_t)size * size));
    CUDA_CHECK(cudaMalloc(&d_Q, sizeof(double) * (size_t)size * size));
    CUDA_CHECK(cudaMalloc(&d_ipiv, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_P, P_cm.data(), sizeof(double) * (size_t)size * size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q_cm.data(), sizeof(double) * (size_t)size * size,
                          cudaMemcpyHostToDevice));

    int getrf_ws = 0;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(solver, size, size, d_Q, size, &getrf_ws));
    double* d_getrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_getrf_work, sizeof(double) * getrf_ws));
    CUSOLVER_CHECK(cusolverDnDgetrf(solver, size, size, d_Q, size,
                                    d_getrf_work, d_ipiv, d_info));
    int h_info_v = 0;
    CUDA_CHECK(cudaMemcpy(&h_info_v, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info_v != 0) {
        std::fprintf(stderr, "solve_gmode_full_chebyshev: getrf(Q) info=%d\n", h_info_v);
        std::exit(1);
    }
    CUSOLVER_CHECK(cusolverDnDgetrs(solver, CUBLAS_OP_N, size, size,
                                    d_Q, size, d_ipiv, d_P, size, d_info));
    // d_P now holds M = Q⁻¹ P (col-major).  Eigenvalues λ give ω² = 1/λ.

    // Pack to complex and Xgeev
    cuDoubleComplex *d_Mc = nullptr, *d_W = nullptr, *d_VR = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Mc, sizeof(cuDoubleComplex) * (size_t)size * size));
    CUDA_CHECK(cudaMalloc(&d_W,  sizeof(cuDoubleComplex) * size));
    CUDA_CHECK(cudaMalloc(&d_VR, sizeof(cuDoubleComplex) * (size_t)size * size));
    int block = 256;
    int total = size * size;
    int grid = (total + block - 1) / block;
    k_real_to_cplx_local<<<grid, block>>>(d_Mc, d_P, total);

    size_t work_device = 0, work_host = 0;
    CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
        solver, params,
        CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
        (int64_t)size,
        CUDA_C_64F, d_Mc, (int64_t)size,
        CUDA_C_64F, d_W,
        CUDA_C_64F, nullptr, (int64_t)size,
        CUDA_C_64F, d_VR,    (int64_t)size,
        CUDA_C_64F, &work_device, &work_host));
    void *d_work = nullptr, *h_work = nullptr;
    if (work_device) CUDA_CHECK(cudaMalloc(&d_work, work_device));
    if (work_host) h_work = std::malloc(work_host);
    CUSOLVER_CHECK(cusolverDnXgeev(
        solver, params,
        CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
        (int64_t)size,
        CUDA_C_64F, d_Mc, (int64_t)size,
        CUDA_C_64F, d_W,
        CUDA_C_64F, nullptr, (int64_t)size,
        CUDA_C_64F, d_VR,    (int64_t)size,
        CUDA_C_64F, d_work, work_device, h_work, work_host, d_info));
    CUDA_CHECK(cudaMemcpy(&h_info_v, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info_v != 0) {
        std::fprintf(stderr, "solve_gmode_full_chebyshev: Xgeev info=%d\n", h_info_v);
        std::exit(1);
    }

    std::vector<cuDoubleComplex> h_W(size);
    std::vector<cuDoubleComplex> h_VR((size_t)size * size);
    CUDA_CHECK(cudaMemcpy(h_W.data(),  d_W,  sizeof(cuDoubleComplex) * size,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_VR.data(), d_VR,
                          sizeof(cuDoubleComplex) * (size_t)size * size,
                          cudaMemcpyDeviceToHost));

    // Filter real positive eigenvalues.  M = Q⁻¹ P has eigenvalues λ with
    // ω² = 1/λ.  True g-modes have λ ~ O(0.1–1); the P-singular directions
    // produce either λ ≈ 0 (ω² → ∞ ghost) or NaN.  Threshold 1e-6 keeps
    // everything physically reasonable ( ω² < 1e6 ).
    struct Cand { double omega_sq; int idx; };
    std::vector<Cand> cand;
    for (int k = 0; k < size; ++k) {
        double re = h_W[k].x;
        double im = h_W[k].y;
        if (!std::isfinite(re)) continue;
        if (std::fabs(im) > 1e-6 * (std::fabs(re) + 1e-30)) continue;
        if (re <= 1e-6) continue;                 // drop ghost modes (λ ≈ 0)
        double omega_sq = 1.0 / re;
        if (!std::isfinite(omega_sq) || omega_sq > 1e6) continue;
        cand.push_back({omega_sq, k});
    }

    // Propagation-cavity g-mode classification
    std::vector<double> N2_profile(Nr), L2_profile(Nr);
    for (int i = 0; i < Nr; ++i) {
        N2_profile[i] = A_star[i] / std::max(c_1[i], 1e-30);
        if (V_2[i] > 0.0) {
            double x4 = std::pow(x_nodes[i], 4);
            L2_profile[i] = lam * Gamma_1[i] / (V_2[i] * x4);
        } else {
            L2_profile[i] = 1e30;
        }
    }
    auto classify = [&](const double* y1, double mu_val) {
        double total = 0.0, in_g = 0.0, in_p = 0.0;
        for (int i = 0; i < Nr; ++i) {
            double w = x_nodes[i] * x_nodes[i] * x_nodes[i] * y1[i] * y1[i];
            total += w;
            bool g = (N2_profile[i] > mu_val) && (L2_profile[i] > mu_val);
            bool p = (N2_profile[i] < mu_val) && (L2_profile[i] < mu_val);
            if (g) in_g += w;
            if (p) in_p += w;
        }
        double g_frac = (total > 0) ? in_g / total : 0.0;
        double p_frac = (total > 0) ? in_p / total : 0.0;
        return std::make_pair(g_frac, p_frac);
    };
    const double p_frac_cut = 0.05;

    // Filter by classifier (optional — ANSL_SKIP_CLASSIFY disables it for debugging)
    std::vector<Cand> kept;
    if (std::getenv("ANSL_SKIP_CLASSIFY")) {
        kept = cand;
    } else {
        for (const auto& c : cand) {
            int col = c.idx;
            std::vector<double> y1(Nr);
            for (int i = 0; i < Nr; ++i) {
                y1[i] = h_VR[(size_t)i + (size_t)col * size].x;
            }
            auto [gf, pf] = classify(y1.data(), c.omega_sq);
            if (pf < p_frac_cut) kept.push_back(c);
        }
    }
    std::fprintf(stderr,
        "  [Exp K] eigenvalues: total=%d, valid_real_pos=%zu, kept=%zu\n",
        size, cand.size(), kept.size());

    // Sort descending ω², dedup close ones
    std::sort(kept.begin(), kept.end(),
              [](const Cand& a, const Cand& b){ return a.omega_sq > b.omega_sq; });
    std::vector<Cand> dedup;
    for (const auto& c : kept) {
        if (!dedup.empty()) {
            double last = dedup.back().omega_sq;
            if (std::fabs(c.omega_sq - last) / std::fabs(last + 1e-30) < 1e-4) continue;
        }
        dedup.push_back(c);
        if ((int)dedup.size() >= n_modes_out) break;
    }

    int n_out = (int)dedup.size();
    omega_sq_out.resize(n_out);
    eigvecs_y1_out.assign((size_t)Nr * n_out, 0.0);
    for (int k = 0; k < n_out; ++k) {
        omega_sq_out[k] = dedup[k].omega_sq;
        int col = dedup[k].idx;
        for (int i = 0; i < Nr; ++i) {
            eigvecs_y1_out[(size_t)i + (size_t)k * Nr] =
                h_VR[(size_t)i + (size_t)col * size].x;
        }
    }

    // Cleanup
    std::free(h_work);
    if (d_work) cudaFree(d_work);
    cudaFree(d_Mc); cudaFree(d_W); cudaFree(d_VR);
    cudaFree(d_getrf_work);
    cudaFree(d_P); cudaFree(d_Q); cudaFree(d_ipiv); cudaFree(d_info);
    cusolverDnDestroyParams(params);
    cusolverDnDestroy(solver);
}

// ── 2D anelastic g-mode EVP (single k_x, cuSOLVER Xgeev) ────────────────
// Builds scalar operator  A v = ω² B v  with
//     A = k_x² diag(N² ρ₀)         (diagonal, ny-2)
//     B = -D diag(ρ₀) D + k_x² diag(ρ₀)   (dense, ny-2)
// on interior CGL nodes.  Solves via M = B⁻¹ A followed by cusolverDnXgeev.
// All steps on device (getrf, getrs, Xgeev).  Host only for light indexing.
void AnelasticSLSolver::compute_2d_gmode_evp(double kx_phys,
                                             int n_modes_out,
                                             std::vector<double>& omega_sq_out,
                                             std::vector<double>& v_modes_out) {
    if (h_Dy_row.empty()) {
        std::fprintf(stderr,
            "compute_2d_gmode_evp: call set_background() first.\n");
        std::exit(1);
    }
    const int M_int = ny - 2;
    const double kx2 = kx_phys * kx_phys;

    // ---------- Host-side assembly (col-major for cuBLAS/cuSOLVER) ---------
    // We must build B on the FULL ny × ny grid, then slice to interior.
    // Otherwise the discrete operator -∂y(ρ∂y) loses its coupling to the
    // boundary nodes (Python reference does exactly this full-then-slice).
    //
    // h_Dy_row is row-major:  D_full[i,j] = h_Dy_row[i*ny + j].
    // Build T_full = diag(ρ)·D_full  and  B_full = -D·T + k²·diag(ρ)
    // in row-major to match h_Dy_row, then slice interior into col-major.

    std::vector<double> T_full((size_t)ny * ny, 0.0);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            T_full[(size_t)i * ny + j] = h_rho[i] * h_Dy_row[(size_t)i * ny + j];
        }
    }
    std::vector<double> B_full((size_t)ny * ny, 0.0);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            double s = 0.0;
            for (int k = 0; k < ny; ++k) {
                s += h_Dy_row[(size_t)i * ny + k] * T_full[(size_t)k * ny + j];
            }
            B_full[(size_t)i * ny + j] = -s;
        }
        B_full[(size_t)i * ny + i] += kx2 * h_rho[i];
    }

    // Slice interior (row-major) and store col-major for cuSOLVER/cuBLAS.
    std::vector<double> B_cm((size_t)M_int * M_int, 0.0);
    for (int i = 0; i < M_int; ++i) {
        for (int j = 0; j < M_int; ++j) {
            // col-major B_cm(i, j) = B_full(i+1, j+1)
            B_cm[(size_t)i + (size_t)j * M_int] =
                B_full[(size_t)(i + 1) * ny + (j + 1)];
        }
    }

    // A is diagonal: A = k_x² N² ρ  on interior nodes (col-major dense).
    std::vector<double> A_cm((size_t)M_int * M_int, 0.0);
    for (int i = 0; i < M_int; ++i) {
        A_cm[(size_t)i + (size_t)i * M_int] = kx2 * h_N2[i + 1] * h_rho[i + 1];
    }

    // ---------- Solve B · M = A for M  (cuSOLVER DnDgetrf + DnDgetrs) ------
    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    double *d_B = nullptr, *d_A = nullptr;
    int *d_ipiv = nullptr, *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B,    sizeof(double) * (size_t)M_int * M_int));
    CUDA_CHECK(cudaMalloc(&d_A,    sizeof(double) * (size_t)M_int * M_int));
    CUDA_CHECK(cudaMalloc(&d_ipiv, sizeof(int) * M_int));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_B, B_cm.data(),
                          sizeof(double) * (size_t)M_int * M_int,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, A_cm.data(),
                          sizeof(double) * (size_t)M_int * M_int,
                          cudaMemcpyHostToDevice));


    int getrf_work_size = 0;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(solver, M_int, M_int,
                                               d_B, M_int, &getrf_work_size));
    double* d_getrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_getrf_work, sizeof(double) * getrf_work_size));
    CUSOLVER_CHECK(cusolverDnDgetrf(solver, M_int, M_int, d_B, M_int,
                                    d_getrf_work, d_ipiv, d_info));
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::fprintf(stderr,
            "compute_2d_gmode_evp: getrf info=%d (B not invertible).\n", h_info);
        std::exit(1);
    }
    CUSOLVER_CHECK(cusolverDnDgetrs(solver, CUBLAS_OP_N, M_int, M_int,
                                    d_B, M_int, d_ipiv, d_A, M_int, d_info));
    // d_A now holds M = B⁻¹ A  (col-major, M_int × M_int).

    // ---------- Solve standard EVP  M u = λ u  with cusolverDnXgeev -------
    cusolverDnParams_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    cuDoubleComplex *d_Mc = nullptr, *d_W = nullptr, *d_VR = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Mc, sizeof(cuDoubleComplex) * (size_t)M_int * M_int));
    CUDA_CHECK(cudaMalloc(&d_W,  sizeof(cuDoubleComplex) * M_int));
    CUDA_CHECK(cudaMalloc(&d_VR, sizeof(cuDoubleComplex) * (size_t)M_int * M_int));

    // Pack real M into complex (Im=0).  Reuse k_real_to_cplx_local from file.
    int block = 256;
    int total = M_int * M_int;
    int grid = (total + block - 1) / block;
    k_real_to_cplx_local<<<grid, block>>>(d_Mc, d_A, total);

    size_t work_device = 0, work_host = 0;
    CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
        solver, params,
        CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
        (int64_t)M_int,
        CUDA_C_64F, d_Mc, (int64_t)M_int,
        CUDA_C_64F, d_W,
        CUDA_C_64F, nullptr, (int64_t)M_int,    // VL unused
        CUDA_C_64F, d_VR,    (int64_t)M_int,
        CUDA_C_64F, &work_device, &work_host));
    void *d_work = nullptr, *h_work = nullptr;
    if (work_device) CUDA_CHECK(cudaMalloc(&d_work, work_device));
    if (work_host) h_work = std::malloc(work_host);
    CUSOLVER_CHECK(cusolverDnXgeev(
        solver, params,
        CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
        (int64_t)M_int,
        CUDA_C_64F, d_Mc, (int64_t)M_int,
        CUDA_C_64F, d_W,
        CUDA_C_64F, nullptr, (int64_t)M_int,
        CUDA_C_64F, d_VR,    (int64_t)M_int,
        CUDA_C_64F, d_work, work_device, h_work, work_host, d_info));
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::fprintf(stderr,
            "compute_2d_gmode_evp: Xgeev info=%d.\n", h_info);
        std::exit(1);
    }

    // Download eigenvalues + eigenvectors, filter real-positive.
    std::vector<cuDoubleComplex> h_W(M_int);
    std::vector<cuDoubleComplex> h_VR((size_t)M_int * M_int);
    CUDA_CHECK(cudaMemcpy(h_W.data(),  d_W,  sizeof(cuDoubleComplex) * M_int,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_VR.data(), d_VR,
                          sizeof(cuDoubleComplex) * (size_t)M_int * M_int,
                          cudaMemcpyDeviceToHost));

    struct EigPair { double omega_sq; int idx; };
    std::vector<EigPair> good;
    int n_dropped_imag = 0, n_dropped_neg = 0;
    double max_imag_ratio = 0.0;
    for (int k = 0; k < M_int; ++k) {
        double re = h_W[k].x;
        double im = h_W[k].y;
        if (!std::isfinite(re)) continue;
        double ratio = std::fabs(im) / (std::fabs(re) + 1e-30);
        if (ratio > max_imag_ratio) max_imag_ratio = ratio;
        if (ratio > 1e-8) { ++n_dropped_imag; continue; }
        if (re <= 0.0) { ++n_dropped_neg; continue; }
        good.push_back({re, k});
    }
    std::fprintf(stderr,
        "  [2D EVP] total=%d, kept=%zu, dropped imag=%d neg=%d, max_imag_ratio=%.3e\n",
        M_int, good.size(), n_dropped_imag, n_dropped_neg, max_imag_ratio);
    std::sort(good.begin(), good.end(),
              [](const EigPair& a, const EigPair& b) {
                  return a.omega_sq > b.omega_sq;
              });
    int n_out = std::min(n_modes_out, (int)good.size());
    omega_sq_out.resize(n_out);
    v_modes_out.assign((size_t)M_int * n_out, 0.0);
    for (int k = 0; k < n_out; ++k) {
        omega_sq_out[k] = good[k].omega_sq;
        int col = good[k].idx;
        for (int i = 0; i < M_int; ++i) {
            v_modes_out[(size_t)i + (size_t)k * M_int] =
                h_VR[(size_t)i + (size_t)col * M_int].x;
        }
    }

    // Cleanup
    std::free(h_work);
    cudaFree(d_work);
    cudaFree(d_Mc); cudaFree(d_W); cudaFree(d_VR);
    cudaFree(d_getrf_work);
    cudaFree(d_B); cudaFree(d_A); cudaFree(d_ipiv); cudaFree(d_info);
    cusolverDnDestroyParams(params);
    cusolverDnDestroy(solver);
}

// ── GPU-resident max|f| reduction ───────────────────────────────────────
static double gpu_max_abs(const double* d_in, int n, double* d_scratch_blocks,
                          int& cached_blocks) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (cached_blocks < grid) cached_blocks = grid;
    k_max_abs_pass1<<<grid, block>>>(d_in, n, d_scratch_blocks);
    std::vector<double> h(grid);
    CUDA_CHECK(cudaMemcpy(h.data(), d_scratch_blocks,
                          sizeof(double) * grid, cudaMemcpyDeviceToHost));
    double m = 0.0;
    for (int i = 0; i < grid; ++i) if (h[i] > m) m = h[i];
    return m;
}

// ── apply_dy: dst[jy, ix] = Σ_jy' D(jy, jy') src[jy', ix]. ──────────────
// Row-major (ny × nx) physical buffers ↔ col-major (nx × ny) GEMM view:
//   buf[jy*nx + ix]  ↔  buf_cm(ix, jy).
// d_Dy uploaded as col-major (ny × ny) with d_Dy[i + j*ny] = D(i, j).
// We want dst_cm(ix, jy) = Σ_jy' D(jy, jy') src_cm(ix, jy').  That matches
// C = A · B^T with A = src_cm, B = d_Dy (so B^T(jy', jy) = d_Dy(jy, jy')
// = D(jy, jy')).  Hence OP_N on A, OP_T on B.
static void apply_dy(cublasHandle_t h, const double* d_src, double* d_dst,
                     const double* d_D, int nx, int ny) {
    const double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(
        h, CUBLAS_OP_N, CUBLAS_OP_T,
        nx, ny, ny,
        &one,
        d_src, nx,     // A: src_cm (nx × ny), lda = nx
        d_D,   ny,     // B: d_Dy  (ny × ny), lda = ny,  op=T
        &zero,
        d_dst, nx));   // C: dst_cm (nx × ny), lda = nx
}

// ── Evaluate du/dt, dv/dt (advection + viscosity, NO pressure). ─────────
//   ∂t u = -(u ∂_x u + v ∂_y u) + ν (∂_xx u + ∂_yy u)
//   ∂t v = -(u ∂_x v + v ∂_y v) + ν (∂_xx v + ∂_yy v)
// Pressure projection is handled by the caller after adding dt·rhs.
void AnelasticSLSolver::compute_rhs_uv(const double* dU, const double* dV,
                                       double* dRU, double* dRV) {
    int block = 256;
    int grid1d = (ncell + block - 1) / block;
    dim3 b2(32, 8);
    dim3 g_ny_nh((nh + b2.x - 1) / b2.x, (ny + b2.y - 1) / b2.y);

    const double inv_nx = 1.0 / (double)nx;

    // zero the RHS buffers
    CUDA_CHECK(cudaMemsetAsync(dRU, 0, sizeof(double) * ncell));
    CUDA_CHECK(cudaMemsetAsync(dRV, 0, sizeof(double) * ncell));

    // ───────── u-equation ─────────
    // FFT u → d_fhat (spectral u)
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_x, const_cast<double*>(dU), d_fhat));
    k_dealias_x_inplace<<<g_ny_nh, b2>>>(d_fhat, ny, nh, (2 * (nh - 1)) / 3);

    // -u·∂x u: compute ∂x u in physical space (d_scratch), then dRU -= u·∂x u
    k_mult_ikx_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));      // d_scratch = ∂x u
    k_fma_product<<<grid1d, block>>>(dRU, -1.0, dU, d_scratch, ncell);

    // +ν·∂xx u: ∂xx u = IFFT(-kx² û)
    k_mult_mkx2_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));      // d_scratch = ∂xx u
    k_fma_scalar<<<grid1d, block>>>(dRU, nu, d_scratch, ncell);

    // ∂y u and ∂yy u via Chebyshev D (apply D twice).
    apply_dy(cublas, dU, d_scratch, d_Dy, nx, ny);                 // d_scratch = ∂y u
    k_fma_product<<<grid1d, block>>>(dRU, -1.0, dV, d_scratch, ncell); // -v·∂y u
    apply_dy(cublas, d_scratch, d_rhs_pi, d_Dy, nx, ny);           // d_rhs_pi = ∂yy u (scratch-borrow)
    k_fma_scalar<<<grid1d, block>>>(dRU, nu, d_rhs_pi, ncell);     // +ν·∂yy u

    // ───────── v-equation ─────────
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_x, const_cast<double*>(dV), d_fhat));
    k_dealias_x_inplace<<<g_ny_nh, b2>>>(d_fhat, ny, nh, (2 * (nh - 1)) / 3);

    // -u·∂x v
    k_mult_ikx_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));      // d_scratch = ∂x v
    k_fma_product<<<grid1d, block>>>(dRV, -1.0, dU, d_scratch, ncell);

    // +ν·∂xx v
    k_mult_mkx2_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));
    k_fma_scalar<<<grid1d, block>>>(dRV, nu, d_scratch, ncell);

    // ∂y v and ∂yy v
    apply_dy(cublas, dV, d_scratch, d_Dy, nx, ny);                 // d_scratch = ∂y v
    k_fma_product<<<grid1d, block>>>(dRV, -1.0, dV, d_scratch, ncell);
    apply_dy(cublas, d_scratch, d_rhs_pi, d_Dy, nx, ny);
    k_fma_scalar<<<grid1d, block>>>(dRV, nu, d_rhs_pi, ncell);

    // ───────── Anelastic extras: buoyancy in v-eq + b equation ────
    if (is_anelastic) {
        // +b · ê_y to v-momentum
        k_add_buoyancy<<<grid1d, block>>>(dRV, d_b, ncell);

        // ∂t b = -(u · ∇) b - N²(y) · v
        CUDA_CHECK(cudaMemsetAsync(d_rhs_b, 0, sizeof(double) * ncell));

        // Spectral ∂x b → d_scratch; dRB -= u · ∂x b
        CUFFT_CHECK(cufftExecD2Z(plan_r2c_x, d_b, d_fhat));
        k_dealias_x_inplace<<<g_ny_nh, b2>>>(d_fhat, ny, nh, (2 * (nh - 1)) / 3);
        k_mult_ikx_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
        CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));
        k_fma_product<<<grid1d, block>>>(d_rhs_b, -1.0, dU, d_scratch, ncell);

        // Chebyshev ∂y b → d_scratch; dRB -= v · ∂y b
        apply_dy(cublas, d_b, d_scratch, d_Dy, nx, ny);
        k_fma_product<<<grid1d, block>>>(d_rhs_b, -1.0, dV, d_scratch, ncell);

        // −N²(y) · v (Brunt-Väisälä restoring force)
        k_sub_N2_v<<<g_ny_nh, b2>>>(d_rhs_b, dV, d_N2, nx, ny);
    }
}

// ── Project (u, v) onto divergence-free subspace. ───────────────────────
//   (1) RHS = ∂x u + ∂y v  (written into d_rhs_pi, row-major ncell)
//   (2) sl_poisson_solve() reads d_rhs_pi, writes π into d_pi,
//       solving ∇·(ρ ∇π) = RHS.  For Boussinesq (ρ=1) this is ∇²π = RHS.
//   (3) u -= ∂x π ;  v -= ∂y π.
void AnelasticSLSolver::project_div_free() {
    int block = 256;
    int grid1d = (ncell + block - 1) / block;
    dim3 b2(32, 8);
    dim3 g_ny_nh((nh + b2.x - 1) / b2.x, (ny + b2.y - 1) / b2.y);
    const double inv_nx = 1.0 / (double)nx;

    // ∂x u → d_scratch
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_x, d_u, d_fhat));
    k_dealias_x_inplace<<<g_ny_nh, b2>>>(d_fhat, ny, nh, (2 * (nh - 1)) / 3);
    k_mult_ikx_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));     // d_scratch = ∂x u

    // ∂y v → d_rhs_pi
    apply_dy(cublas, d_v, d_rhs_pi, d_Dy, nx, ny);                // d_rhs_pi = ∂y v

    // d_rhs_pi = ∂x u + ∂y v  (divergence of the unprojected velocity).
    // For anelastic (ρ₀(y) non-uniform): RHS = ρ₀·(∂x u + ∂y v) + ρ₀'·v
    //   = ∇·(ρ₀ u) when projected so the SL-Poisson solve ∇·(ρ₀ ∇π) = RHS
    //   produces a π whose ∇π cancels the mass flux divergence exactly.
    k_fma_scalar<<<grid1d, block>>>(d_rhs_pi, 1.0, d_scratch, ncell);
    if (is_anelastic) {
        // Scale divergence by ρ₀(y) and add ρ₀'·v along y.
        // Use d_scratch as temp: d_scratch = (∂x u + ∂y v)·ρ₀ - (∂x u + ∂y v)
        // Easier path: rebuild RHS from scratch for clarity.
        CUDA_CHECK(cudaMemsetAsync(d_rhs_pi, 0, sizeof(double) * ncell));
        // term 1: ρ₀ · ∂x u  (d_scratch currently holds ∂x u)
        k_fma_row<<<g_ny_nh, b2>>>(d_rhs_pi, 1.0, d_scratch, d_rho, nx, ny);
        // term 2: ρ₀ · ∂y v
        apply_dy(cublas, d_v, d_scratch, d_Dy, nx, ny);           // d_scratch = ∂y v
        k_fma_row<<<g_ny_nh, b2>>>(d_rhs_pi, 1.0, d_scratch, d_rho, nx, ny);
        // term 3: ρ₀' · v
        k_fma_row<<<g_ny_nh, b2>>>(d_rhs_pi, 1.0, d_v, d_rho_prime, nx, ny);
    }

    // Solve ∇·(ρ ∇π) = RHS  (reduced-pressure form; for ρ=1, plain ∇²π = RHS).
    // For our solver the SL pipeline already solves with a -1/(μ+kx²) factor
    // i.e. ∇²π + RHS = 0,  so passing RHS = ∂x u + ∂y v yields π satisfying
    // ∇²π = -(∂x u + ∂y v).  Correction: u ← u − ∂x π gives
    //   ∇·u_new = ∇·u + ∇²π = ∇·u - ∇·u = 0.  ✓
    sl_poisson_solve();

    // ∂x π → d_scratch, subtract from u
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_x, d_pi, d_fhat));
    k_dealias_x_inplace<<<g_ny_nh, b2>>>(d_fhat, ny, nh, (2 * (nh - 1)) / 3);
    k_mult_ikx_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));
    k_sub_inplace<<<grid1d, block>>>(d_u, d_scratch, ncell);

    // ∂y π → d_scratch, subtract from v;  then enforce v|_wall = 0 (Dirichlet).
    apply_dy(cublas, d_pi, d_scratch, d_Dy, nx, ny);
    k_sub_inplace<<<grid1d, block>>>(d_v, d_scratch, ncell);

    // Dirichlet velocity BC: u = v = 0 at y = 0, Ly.  Required for
    // compatibility with the SL-Poisson Dirichlet-π projection: if u_tan
    // were non-zero at the wall, ∂x u_tan would contribute to ∇·u at the
    // wall where the SL basis cannot resolve it (leaving residual div).
    int grid_bdy = (nx + 255) / 256;
    k_zero_y_boundary<<<grid_bdy, 256>>>(d_u, nx, ny);
    k_zero_y_boundary<<<grid_bdy, 256>>>(d_v, nx, ny);
}

// ── Shu-Osher RK3 (primitive-variable + projection). ────────────────────
// Standard low-storage 3-stage form:
//   u^(1) = u^n + dt R(u^n)                      ;  project
//   u^(2) = 3/4 u^n + 1/4 (u^(1) + dt R(u^(1)))  ;  project
//   u^{n+1} = 1/3 u^n + 2/3 (u^(2) + dt R(u^(2))) ; project
double AnelasticSLSolver::step() {
    int block = 256;
    int grid1d = (ncell + block - 1) / block;

    // ── CFL-limited dt ──
    if (!d_reduce) {
        // Reserve ~enough slots for the two-pass max|·| reductions.
        reduce_blocks = (ncell + block - 1) / block;
        CUDA_CHECK(cudaMalloc(&d_reduce, sizeof(double) * reduce_blocks));
    }
    double umax = gpu_max_abs(d_u, ncell, d_reduce, reduce_blocks);
    double vmax = gpu_max_abs(d_v, ncell, d_reduce, reduce_blocks);
    double dt_adv = cfl / std::max(umax / dx + vmax / std::max(dy_min, 1e-30), 1e-30);
    // Explicit viscous CFL (RK3, Chebyshev clustered grid).  Stability needs
    //   ν · dt · (kx_max² + μ_max) ≲ 2.51  (RK3 stability on imaginary axis).
    // kx_max = π/dx;  μ_max ≈ largest SL eigenvalue = (2 N²/Ly)² / 4 (Trefethen).
    // Use the conservative estimate dt_visc ≲ 0.5·dy_min² / ν (matches the FD
    // rule of thumb and is dominated by the near-boundary cluster).
    double dt_visc = (nu > 0.0) ? 0.5 * dy_min * dy_min / nu : 1e30;
    double dt = std::min({dt_max, dt_adv, dt_visc});
    dt = std::max(dt_min, dt);
    dt_current = dt;


    // Snapshot y_n
    CUDA_CHECK(cudaMemcpy(d_u_orig, d_u, sizeof(double) * ncell, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_orig, d_v, sizeof(double) * ncell, cudaMemcpyDeviceToDevice));
    if (is_anelastic) {
        CUDA_CHECK(cudaMemcpy(d_b_orig, d_b, sizeof(double) * ncell,
                              cudaMemcpyDeviceToDevice));
    }

    // ── Stage 1 : y^(1) = y_n + dt·R(y_n) ──
    compute_rhs_uv(d_u, d_v, d_rhs_u, d_rhs_v);
    k_fma_scalar<<<grid1d, block>>>(d_u, dt, d_rhs_u, ncell);
    k_fma_scalar<<<grid1d, block>>>(d_v, dt, d_rhs_v, ncell);
    if (is_anelastic) k_fma_scalar<<<grid1d, block>>>(d_b, dt, d_rhs_b, ncell);
    project_div_free();

    // ── Stage 2 : y^(2) = 3/4 y_n + 1/4 (y^(1) + dt·R(y^(1))) ──
    compute_rhs_uv(d_u, d_v, d_rhs_u, d_rhs_v);
    k_rk3_combine<<<grid1d, block>>>(d_u, d_u_orig, d_u, d_rhs_u,
                                     0.75, 0.25, dt, ncell);
    k_rk3_combine<<<grid1d, block>>>(d_v, d_v_orig, d_v, d_rhs_v,
                                     0.75, 0.25, dt, ncell);
    if (is_anelastic)
        k_rk3_combine<<<grid1d, block>>>(d_b, d_b_orig, d_b, d_rhs_b,
                                         0.75, 0.25, dt, ncell);
    project_div_free();

    // ── Stage 3 : y_{n+1} = 1/3 y_n + 2/3 (y^(2) + dt·R(y^(2))) ──
    compute_rhs_uv(d_u, d_v, d_rhs_u, d_rhs_v);
    k_rk3_combine<<<grid1d, block>>>(d_u, d_u_orig, d_u, d_rhs_u,
                                     1.0 / 3.0, 2.0 / 3.0, dt, ncell);
    k_rk3_combine<<<grid1d, block>>>(d_v, d_v_orig, d_v, d_rhs_v,
                                     1.0 / 3.0, 2.0 / 3.0, dt, ncell);
    if (is_anelastic)
        k_rk3_combine<<<grid1d, block>>>(d_b, d_b_orig, d_b, d_rhs_b,
                                         1.0 / 3.0, 2.0 / 3.0, dt, ncell);
    project_div_free();

    ++step_count;
    return dt;
}


// 7-step SL-Poisson pipeline (reduced-pressure form):
//   in:  d_rhs_pi (ny × nx physical RHS tilde_f)
//   out: d_pi    (ny × nx physical reduced pressure π)
//
// Steps:
//   1. FFT_x:      f(x,y)   → f̂(kx,y)         [d_rhs_pi  → d_fhat]
//   2. weight:     ĝ(kx,y)  = (1/√ρ) · f̂      [d_fhat    → d_ghat, with d_rho_sqrt_inv]
//   3. fwd SL:     Ĝ(kx,n)  = Σ_y ψ_n(y)·w_cc(y)·ĝ(kx,y)
//                           = Psi_fwd^T · ĝ   [d_ghat    → d_Ghat]
//   4. diag solve: Q̂(kx,n)  = -Ĝ / (μ_n + kx²) [d_Ghat   → d_Qhat]
//   5. inv SL:     q̂(kx,y)  = Σ_n ψ_n(y) · Q̂(kx,n)
//                           = Psi_inv · Q̂    [d_Qhat    → d_qhat]
//   6. weight:     π̂(kx,y)  = (1/√ρ) · q̂      [d_qhat    → d_pihat]
//   7. IFFT_x:     π(x,y)   ← π̂              [d_pihat   → d_pi, scale 1/nx]
void AnelasticSLSolver::sl_poisson_solve() {
    // Step 1: FFT in x  (d_rhs_pi is the physical RHS of size ncell)
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_x, d_rhs_pi, d_fhat));

    // Layout helper: d_fhat, d_ghat are (ny × nh) row-major complex.
    // cuBLAS ZGEMM treats them as column-major (nh × ny).
    dim3 block2d(32, 8);
    dim3 grid_ny_nh((nh + block2d.x - 1) / block2d.x,
                    (ny + block2d.y - 1) / block2d.y);
    dim3 grid_nm_nh((nh + block2d.x - 1) / block2d.x,
                    (n_modes + block2d.y - 1) / block2d.y);

    // Step 2: weight by 1/√ρ_0 along y  (d_fhat → d_ghat)
    k_weight_fhat_inplace<<<grid_ny_nh, block2d>>>(
        d_ghat, d_fhat, d_rho_sqrt_inv, ny, nh);

    // Step 3: forward SL transform.
    // Want:  Ghat[n, kx] = Σ_y Psi_fwd[y, n] * ghat[y, kx]    (n_modes × nh)
    //
    // In col-major view: ghat = (nh × ny), lda = nh
    //                    Psi_fwd = (ny × n_modes), col-major, lda = ny
    //                    Ghat = (nh × n_modes), lda = nh
    // We need Ghat = ghat · Psi_fwd    (nh × ny) * (ny × n_modes) → (nh × n_modes)
    // i.e. op(A) = N, op(B) = N, m = nh, n = n_modes, k = ny.
    cuDoubleComplex one  = { 1.0, 0.0 };
    cuDoubleComplex zero = { 0.0, 0.0 };
    CUBLAS_CHECK(cublasZgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        nh, n_modes, ny,
        &one,
        d_ghat, nh,                 // A: (nh × ny) col-major, lda=nh
        d_Psi_fwd, ny,              // B: (ny × n_modes) col-major, lda=ny
        &zero,
        d_Ghat, nh));               // C: (nh × n_modes) col-major, lda=nh
    // Storage: d_Ghat is (n_modes × nh) in row-major == (nh × n_modes) in col-major. ✓

    // Step 4: diagonal divide  Qhat[n, kx] = -Ghat[n, kx] / (mu[n] + kx²)
    k_diag_divide_sl<<<grid_nm_nh, block2d>>>(
        d_Qhat, d_Ghat, d_mu, d_kx, n_modes, nh);

    // Step 5: inverse SL transform.
    // Want:  qhat[y, kx] = Σ_n Psi_inv[y, n] * Qhat[n, kx]    (ny × nh)
    //
    // In col-major view: Qhat = (nh × n_modes), lda = nh
    //                    Psi_inv = (ny × n_modes), col-major, lda = ny
    //                    qhat = (nh × ny), lda = nh
    // We need qhat = Qhat · Psi_inv^T   (nh × n_modes) * (n_modes × ny) → (nh × ny)
    // i.e. op(A) = N (Qhat nh × n_modes), op(B) = T (Psi_inv^T n_modes × ny).
    CUBLAS_CHECK(cublasZgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        nh, ny, n_modes,
        &one,
        d_Qhat, nh,                 // A: (nh × n_modes) col-major, lda=nh
        d_Psi_inv, ny,              // B: (ny × n_modes) col-major (= (n_modes×ny)^T), lda=ny
        &zero,
        d_qhat, nh));               // C: (nh × ny) col-major, lda=nh

    // Step 6: weight by 1/√ρ_0  (d_qhat → d_pihat)
    k_weight_fhat_inplace<<<grid_ny_nh, block2d>>>(
        d_pihat, d_qhat, d_rho_sqrt_inv, ny, nh);

    // Step 7: IFFT_x with cuFFT (unnormalised).
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_pihat, d_pi));

    // Normalise by 1/nx (cuFFT does not scale C2R output).
    int block1d = 256;
    int grid1d = (ncell + block1d - 1) / block1d;
    k_normalize<<<grid1d, block1d>>>(d_pi, ncell, 1.0 / (double)nx);
}

// Manufactured-solution self-test.
//   pi_exact(x, y) = sin(k_x · x) · sin(π (y / Ly))
//   RHS (for reduced-pressure form div(ρ ∇π)):
//     f = ρ · (∂xx + ∂yy) π + ρ' · ∂y π
//       = ρ · (-kx² + d²_y) [sin(kx·x) · sin(π y/Ly)]
//         + ρ' · (π/Ly) cos(π y/Ly) · sin(kx·x)
// Returns the L2 error on π (with boundary zeros excluded where appropriate).
double AnelasticSLSolver::manufactured_test() {
    if (nx == 0 || ny == 0 || n_modes == 0) {
        std::fprintf(stderr, "AnelasticSL: manufactured_test requires init+set_background.\n");
        return -1.0;
    }

    const double kx_test = 2.0 * M_PI * 2.0 / Lx;  // k = 2
    std::vector<double> h_pi_exact(ncell), h_rhs(ncell);

    // Precompute analytic derivatives of phi(y) = sin(π y/Ly) on CGL grid
    std::vector<double> phi(ny), dphi_dy(ny), d2phi_dy2(ny);
    for (int jy = 0; jy < ny; ++jy) {
        double y = h_y_cgl[jy];
        double eta = y / Ly;
        phi[jy]      =  std::sin(M_PI * eta);
        dphi_dy[jy]  =  (M_PI / Ly) * std::cos(M_PI * eta);
        d2phi_dy2[jy] = -(M_PI / Ly) * (M_PI / Ly) * phi[jy];
    }
    // Compute ρ' on CGL grid (finite diff for sanity; set_background stored ρ).
    // Use central differences on (possibly non-uniform) grid.
    std::vector<double> drho(ny, 0.0);
    for (int jy = 1; jy < ny - 1; ++jy) {
        double hL = h_y_cgl[jy]   - h_y_cgl[jy - 1];
        double hR = h_y_cgl[jy + 1] - h_y_cgl[jy];
        drho[jy] = (h_rho[jy + 1] * hL * hL - h_rho[jy - 1] * hR * hR
                    + h_rho[jy] * (hR * hR - hL * hL))
                   / (hL * hR * (hL + hR));
    }
    drho[0]    = (h_rho[1]     - h_rho[0])     / (h_y_cgl[1]     - h_y_cgl[0]);
    drho[ny-1] = (h_rho[ny-1]  - h_rho[ny-2])  / (h_y_cgl[ny-1]  - h_y_cgl[ny-2]);

    // Physical grid uses ascending CGL for y, uniform for x (x_i = i·dx).
    for (int jy = 0; jy < ny; ++jy) {
        double rho_y  = h_rho[jy];
        double drho_y = drho[jy];
        double phi_y  = phi[jy];
        double dphi_y = dphi_dy[jy];
        double d2phi_y = d2phi_dy2[jy];
        for (int ix = 0; ix < nx; ++ix) {
            double x = ix * dx;
            double sn = std::sin(kx_test * x);
            double cs = std::cos(kx_test * x);  // unused but kept for clarity
            (void)cs;
            int k = jy * nx + ix;
            h_pi_exact[k] = sn * phi_y;
            // f = ρ·(-kx²·sn·phi_y + sn·d2phi_y) + ρ'·sn·dphi_y
            //   = sn · [ρ·(-kx²·phi_y + d2phi_y) + ρ'·dphi_y]
            h_rhs[k] = sn * (rho_y * (-kx_test * kx_test * phi_y + d2phi_y)
                             + drho_y * dphi_y);
        }
    }

    // Upload RHS, run pipeline, download π.
    CUDA_CHECK(cudaMemcpy(d_rhs_pi, h_rhs.data(),
                          sizeof(double) * ncell, cudaMemcpyHostToDevice));
    sl_poisson_solve();

    std::vector<double> h_pi(ncell);
    CUDA_CHECK(cudaMemcpy(h_pi.data(), d_pi,
                          sizeof(double) * ncell, cudaMemcpyDeviceToHost));

    // L2 error (exclude y boundary points where Dirichlet BC is imposed).
    double num = 0.0, den = 0.0;
    int count = 0;
    double max_pi = 0.0, max_exact = 0.0, max_diff = 0.0;
    for (int jy = 1; jy < ny - 1; ++jy) {
        double w = h_cc_weights[jy];
        for (int ix = 0; ix < nx; ++ix) {
            int k = jy * nx + ix;
            double d = h_pi[k] - h_pi_exact[k];
            num += w * d * d;
            den += w * h_pi_exact[k] * h_pi_exact[k];
            ++count;
            max_pi   = std::max(max_pi,   std::abs(h_pi[k]));
            max_exact= std::max(max_exact,std::abs(h_pi_exact[k]));
            max_diff = std::max(max_diff, std::abs(d));
        }
    }
    double err_L2 = std::sqrt(num / std::max(den, 1e-300));
    double err_abs = std::sqrt(num / (double)count);
    // Ratio at a sample point near the centre: diagnoses constant scaling errors.
    // Pick a point where sin(kx x) is large: kx=4π, x=1/8 -> 4π/8=π/2 -> sin=1.
    int jy_c = ny / 2, ix_c = nx / 8;
    double pi_c = h_pi[jy_c * nx + ix_c];
    double ex_c = h_pi_exact[jy_c * nx + ix_c];
    std::fprintf(stderr,
        "  AnelasticSL manufactured test: kx_mode=2, ny=%d, n_modes=%d, Ly=%g\n"
        "    rel err_L2 = %.3e,  abs err_L2 = %.3e\n"
        "    |π|_max = %.4e,  |π_exact|_max = %.4e,  |diff|_max = %.3e\n"
        "    sample (jy=%d,ix=%d): π=%.6e, exact=%.6e, ratio=%.4f\n",
        ny, n_modes, Ly, err_L2, err_abs,
        max_pi, max_exact, max_diff,
        jy_c, ix_c, pi_c, ex_c, pi_c / (ex_c + 1e-30));
    return err_L2;
}

void AnelasticSLSolver::download_uv(std::vector<double>& h_u, std::vector<double>& h_v) {
    h_u.resize(ncell); h_v.resize(ncell);
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, sizeof(double) * ncell, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, sizeof(double) * ncell, cudaMemcpyDeviceToHost));
}

void AnelasticSLSolver::download_divergence(std::vector<double>& h_div) {
    dim3 b2(32, 8);
    dim3 g_ny_nh((nh + b2.x - 1) / b2.x, (ny + b2.y - 1) / b2.y);
    int block = 256;
    int grid1d = (ncell + block - 1) / block;
    const double inv_nx = 1.0 / (double)nx;

    CUFFT_CHECK(cufftExecD2Z(plan_r2c_x, d_u, d_fhat));
    k_mult_ikx_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));      // ∂x u

    apply_dy(cublas, d_v, d_rhs_pi, d_Dy, nx, ny);                 // ∂y v

    k_fma_scalar<<<grid1d, block>>>(d_rhs_pi, 1.0, d_scratch, ncell);  // ∂x u + ∂y v

    h_div.resize(ncell);
    CUDA_CHECK(cudaMemcpy(h_div.data(), d_rhs_pi,
                          sizeof(double) * ncell, cudaMemcpyDeviceToHost));
}

void AnelasticSLSolver::download_omega(std::vector<double>& h_omega) {
    // ω = ∂v/∂x − ∂u/∂y.  Compute on device into d_omega, then download.
    int block = 256;
    int grid1d = (ncell + block - 1) / block;
    dim3 b2(32, 8);
    dim3 g_ny_nh((nh + b2.x - 1) / b2.x, (ny + b2.y - 1) / b2.y);
    const double inv_nx = 1.0 / (double)nx;

    // ∂v/∂x via FFT
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_x, d_v, d_fhat));
    k_mult_ikx_out<<<g_ny_nh, b2>>>(d_ghat, d_fhat, d_kx, inv_nx, ny, nh);
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_x, d_ghat, d_scratch));   // d_scratch = ∂v/∂x

    // ∂u/∂y via Chebyshev D (store into d_rhs_pi as scratch)
    apply_dy(cublas, d_u, d_rhs_pi, d_Dy, nx, ny);              // d_rhs_pi = ∂u/∂y

    k_compute_omega<<<grid1d, block>>>(d_omega, d_scratch, d_rhs_pi, ncell);

    h_omega.resize(ncell);
    CUDA_CHECK(cudaMemcpy(h_omega.data(), d_omega, sizeof(double) * ncell,
                          cudaMemcpyDeviceToHost));
}
