// CartImplSolver orchestration: init, destroy, HSE, upload/download, step.
// Residual kernels are in cart_impl_residual.cu.
// JFNK Newton-Krylov is in cart_impl_jfnk.cu.

#include "cart_impl_solver.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// Kernel forward decls (implemented in cart_impl_residual.cu)
__global__ void k_ci_residual(
    const double* rho, const double* mx, const double* my, const double* rhoE,
    const double* rho0, const double* P0,
    double* res,
    int nx, int ny, int ng, double dx, double dy, double g_y,
    EOS eos, double atm_thresh, int lim_type, int hllc_variant);

__global__ void k_ci_compute_F(double* F,
    const double* rho, const double* mx, const double* my, const double* rhoE,
    const double* Un, const double* hse_defect,
    const double* R_pre, double inv_dt,
    int nx, int ny, int ng);

__global__ void k_ci_apply_bc_y_hse(double* rho, double* mx, double* my, double* rhoE,
    const double* rho0, const double* P0,
    EOS eos, int nx, int ny, int ng);

__global__ void k_ci_apply_bc_x_periodic(double* rho, double* mx, double* my, double* rhoE,
    int nx, int ny, int ng);

__global__ void k_ci_floor(double* rho, double* mx, double* my, double* rhoE,
    const double* rho0, EOS eos,
    double atm_thresh, int nx, int ny, int ng);

__global__ void k_ci_pack_state(double* out,
    const double* rho, const double* mx, const double* my, const double* rhoE,
    int nx, int ny, int ng);

__global__ void k_ci_unpack_state(const double* in,
    double* rho, double* mx, double* my, double* rhoE,
    int nx, int ny, int ng);

__global__ void k_ci_cfl(const double* rho, const double* mx, const double* my,
    const double* rhoE, double* out,
    int nx, int ny, int ng, EOS eos, double dx, double dy, double atm_thresh);

// --- Utility helpers (host+device) ---
static __host__ __device__ __forceinline__
int ci_idx(int i, int j, int ny, int ng) {
    return (i + ng) * (ny + 2*ng) + (j + ng);
}

static __host__ __device__ __forceinline__
int ci_flat(int i, int j, int ny) { return i * ny + j; }

// ---------------------------------------------------------------------
// Host-side device utility kernels (shared)
// ---------------------------------------------------------------------
__global__ void k_ci_copy(double* dst, const double* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}
__global__ void k_ci_axpy(double* y, double a, const double* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] += a * x[i];
}
__global__ void k_ci_scale(double* x, double a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= a;
}
__global__ void k_ci_neg(double* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = -x[i];
}
__global__ void k_ci_dot_partial(const double* a, const double* b, double* partial, int N) {
    extern __shared__ double sd[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double s = 0.0;
    while (i < N) { s += a[i]*b[i]; i += blockDim.x * gridDim.x; }
    sd[tid] = s;
    __syncthreads();
    for (int off = blockDim.x/2; off > 0; off >>= 1) {
        if (tid < off) sd[tid] += sd[tid + off];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sd[0];
}

double gpu_dot_ci(const double* d_a, const double* d_b, int N) {
    int B = 256;
    int nb = (N + B - 1) / B;
    if (nb > 256) nb = 256;
    static double* d_partial = nullptr;
    static int alloc_n = 0;
    if (alloc_n < nb) {
        if (d_partial) cudaFree(d_partial);
        cudaMalloc(&d_partial, nb * sizeof(double));
        alloc_n = nb;
    }
    k_ci_dot_partial<<<nb, B, B*sizeof(double)>>>(d_a, d_b, d_partial, N);
    std::vector<double> h(nb);
    cudaMemcpy(h.data(), d_partial, nb*sizeof(double), cudaMemcpyDeviceToHost);
    double s = 0;
    for (int i = 0; i < nb; ++i) s += h[i];
    return s;
}

// ---------------------------------------------------------------------
// Viallet eq 72 scaling (fas2 fix 3, ported) — cart_impl variant
// ---------------------------------------------------------------------
__global__ void k_ci_build_scaling(
    const double* rho, const double* mx, const double* my,
    const double* rho0, const double* P0,
    double* L, double* R, double* invL,
    EOS eos, double alpha1, double alpha2,
    int nx, int ny, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int k = ci_idx(i, j, ny, ng);
    int fk = ci_flat(i, j, ny);

    double r0 = fmax(rho0[fk], 1e-30);
    double p0 = fmax(P0[fk], 1e-30);
    double cs = sqrt(eos.gamma * p0 / r0);
    double rho_k = fmax(rho[k], 1e-30);
    double vx = mx[k] / rho_k;
    double vy = my[k] / rho_k;
    double vmag = sqrt(vx*vx + vy*vy);

    // L = ρ·diag(1, max(|v|,α1·cs), max(|v|,α1·cs), cs²)
    // R =   diag(1, max(|v|,α2·cs), max(|v|,α2·cs), cs²/ρ)
    double Lm = r0 * fmax(vmag, alpha1 * cs);
    double Rm =       fmax(vmag, alpha2 * cs);
    double Le = r0 * cs * cs;
    double Re =      cs * cs;

    L[4*fk+0] = r0;   R[4*fk+0] = 1.0;     invL[4*fk+0] = 1.0 / r0;
    L[4*fk+1] = Lm;   R[4*fk+1] = Rm;      invL[4*fk+1] = 1.0 / Lm;
    L[4*fk+2] = Lm;   R[4*fk+2] = Rm;      invL[4*fk+2] = 1.0 / Lm;
    L[4*fk+3] = Le;   R[4*fk+3] = Re;      invL[4*fk+3] = 1.0 / Le;
}

__global__ void k_ci_scale_by_diag(double* x, const double* D, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= D[i];
}

// ---------------------------------------------------------------------
// init / destroy
// ---------------------------------------------------------------------
void CartImplSolver::init(int nx_in, int ny_in, double Lx_in, double Ly_in,
                          const EOS& eos_in, double gamma_in,
                          double g_y_in, double cfl)
{
    nx = nx_in; ny = ny_in; ng = 2;
    Lx = Lx_in; Ly = Ly_in;
    dx = Lx / nx; dy = Ly / ny;
    stride = ny + 2*ng;
    total = (nx + 2*ng) * stride;
    phys = nx * ny;
    eos = eos_in;
    gamma = gamma_in;
    g_y = g_y_in;
    cfl_num = cfl;

    alloc_all();

    std::fprintf(stderr,
        "CartImplSolver: %dx%d cells, box=[0,%g]x[0,%g], dx=%g dy=%g, γ=%g, g_y=%g, CFL=%g\n",
        nx, ny, Lx, Ly, dx, dy, gamma, g_y, cfl);
    std::fprintf(stderr,
        "  line-y precond: %s, Viallet scaling: %s (α1=%.1e, α2=%.1e)\n",
        use_line_precond_y ? "ON" : "OFF",
        use_music_scaling ? "ON" : "OFF",
        music_alpha1, music_alpha2);
}

void CartImplSolver::alloc_all() {
    auto mal = [](double** p, size_t nbytes) {
        CUDA_CHECK(cudaMalloc(p, nbytes));
        CUDA_CHECK(cudaMemset(*p, 0, nbytes));
    };
    size_t sz_total = total * sizeof(double);
    size_t sz_phys4 = 4 * phys * sizeof(double);
    size_t sz_phys  = phys * sizeof(double);

    mal(&d_rho,   sz_total);
    mal(&d_mx,    sz_total);
    mal(&d_my,    sz_total);
    mal(&d_rhoE,  sz_total);
    mal(&d_Un,    sz_phys4);
    mal(&d_Ubak,  sz_phys4);

    mal(&d_rho0,  sz_phys);
    mal(&d_P0,    sz_phys);
    mal(&d_hse_defect, sz_phys4);

    mal(&d_res,   sz_phys4);
    mal(&d_Fk,    sz_phys4);

    mal(&d_scale_L,    sz_phys4);
    mal(&d_scale_R,    sz_phys4);
    mal(&d_scale_invL, sz_phys4);

    for (int k = 0; k <= GMRES_K; ++k) mal(&d_V[k], sz_phys4);
    for (int k = 0; k <  GMRES_K; ++k) mal(&d_Z[k], sz_phys4);
    mal(&d_gmres_w, sz_phys4);
}

void CartImplSolver::destroy() {
    auto f = [](double* p) { if (p) cudaFree(p); };
    f(d_rho); f(d_mx); f(d_my); f(d_rhoE); f(d_Un); f(d_Ubak);
    f(d_rho0); f(d_P0); f(d_hse_defect);
    f(d_res); f(d_Fk);
    f(d_scale_L); f(d_scale_R); f(d_scale_invL);
    for (int k = 0; k <= GMRES_K; ++k) f(d_V[k]);
    for (int k = 0; k <  GMRES_K; ++k) f(d_Z[k]);
    f(d_gmres_w);
    std::memset(this, 0, sizeof(*this));
}

// ---------------------------------------------------------------------
// IC: HSE polytrope (2D Cartesian, constant g_y)
// ρ(y) = ρ_b · θ(y)^n, P = K·ρ^γ, where θ = 1 - y/Ly, n = 1/(γ-1)
// Perturbation(adiabatic):δρ/ρ = amp·sin(π y/Ly)·cos(k_x x) with k_x=2π/Lx
// ---------------------------------------------------------------------
void CartImplSolver::init_hse_polytrope(double rho_base, double amp) {
    double n_poly = 1.0 / (gamma - 1.0);
    // K: choose so that hydrostatic closure is consistent.
    // For polytrope: dP/dy = -ρ·g ⇒ K·γ·ρ^{γ-1}·(dρ/dy) = -ρ·g
    // with ρ = ρ_b·θⁿ gives K = (γ-1)/γ · g · Ly · ρ_b^{γ-1}
    double K = (gamma - 1.0) / gamma * g_y * Ly * std::pow(rho_base, gamma - 1.0);
    double rho_floor = 1e-6 * rho_base;

    std::vector<double> h_rho(total, rho_floor), h_mx(total, 0.0),
                        h_my(total, 0.0), h_rhoE(total, 0.0);

    for (int i = 0; i < nx; ++i) {
        double xc = (i + 0.5) * dx;
        for (int j = 0; j < ny; ++j) {
            double yc = (j + 0.5) * dy;
            double theta_val = std::max(1.0 - yc / Ly, 1e-10);
            double rho = std::max(rho_base * std::pow(theta_val, n_poly), rho_floor);
            double P   = K * std::pow(rho, gamma);

            if (amp != 0.0) {
                double kx = 2.0 * M_PI / Lx;
                double delta = amp * std::sin(M_PI * yc / Ly) * std::cos(kx * xc);
                rho *= (1.0 + delta);
                P   *= (1.0 + gamma * delta);  // adiabatic
            }
            double e = P / ((gamma - 1.0) * rho);
            int k = ci_idx(i, j, ny, ng);
            h_rho[k]  = rho;
            h_mx[k]   = 0.0;
            h_my[k]   = 0.0;
            h_rhoE[k] = rho * e;   // zero velocity
        }
    }
    CUDA_CHECK(cudaMemcpy(d_rho,  h_rho.data(),  total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,   h_mx.data(),   total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,   h_my.data(),   total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhoE, h_rhoE.data(), total*sizeof(double), cudaMemcpyHostToDevice));

    std::fprintf(stderr,
        "  CartImpl HSE polytrope: ρ_b=%g, g=%g, K=%g, Ly=%g, perturb=%g\n",
        rho_base, g_y, K, Ly, amp);
}

// ---------------------------------------------------------------------
// snapshot_hse — capture d_rho, P (from rhoE) as reference
// Also precompute R_hse (well-balanced defect)
// ---------------------------------------------------------------------
void CartImplSolver::snapshot_hse() {
    std::vector<double> h_rho(total), h_rhoE(total);
    CUDA_CHECK(cudaMemcpy(h_rho.data(),  d_rho,  total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhoE.data(), d_rhoE, total*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> rho0(phys), P0(phys);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            int flat = ci_flat(i, j, ny);
            int k = ci_idx(i, j, ny, ng);
            rho0[flat] = h_rho[k];
            P0[flat]   = (gamma - 1.0) * h_rhoE[k];  // v=0 so E = ρe = P/(γ-1)
        }
    CUDA_CHECK(cudaMemcpy(d_rho0, rho0.data(), phys*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P0,   P0.data(),   phys*sizeof(double), cudaMemcpyHostToDevice));
    double rho_max = *std::max_element(rho0.begin(), rho0.end());
    atm_rho_thresh = 1e-4 * rho_max;

    // Apply BC (use HSE rho/P for y-ghosts) so residual on HSE is well-defined.
    apply_bc();

    // Pre-compute HSE defect: R_wb(U_hse) on current state
    dim3 B2d(16, 16);
    dim3 G2d((nx+15)/16, (ny+15)/16);
    k_ci_residual<<<G2d, B2d>>>(d_rho, d_mx, d_my, d_rhoE, d_rho0, d_P0,
                                 d_hse_defect,
                                 nx, ny, ng, dx, dy, g_y,
                                 eos, atm_rho_thresh, limiter_type, hllc_variant);
    cudaError_t err_r = cudaGetLastError();
    if (err_r != cudaSuccess) std::fprintf(stderr, "residual launch: %s\n", cudaGetErrorString(err_r));
    CUDA_CHECK(cudaDeviceSynchronize());

    hse_set = true;
    std::fprintf(stderr, "  CartImpl HSE snapshot: ρ_max=%g, atm=%g\n", rho_max, atm_rho_thresh);
}

// ---------------------------------------------------------------------
// BC: x-periodic, y-HSE Dirichlet on ghost cells
// ---------------------------------------------------------------------
void CartImplSolver::apply_bc() {
    int B = 128;
    int n_x = ny + 2*ng;
    k_ci_apply_bc_x_periodic<<<(n_x + B - 1) / B, B>>>(
        d_rho, d_mx, d_my, d_rhoE, nx, ny, ng);
    cudaError_t err_x = cudaGetLastError();
    if (err_x != cudaSuccess) {
        std::fprintf(stderr, "BC x-periodic launch failed: %s\n", cudaGetErrorString(err_x));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    k_ci_apply_bc_y_hse<<<(nx + B - 1) / B, B>>>(
        d_rho, d_mx, d_my, d_rhoE, d_rho0, d_P0, eos, nx, ny, ng);
    cudaError_t err_y = cudaGetLastError();
    if (err_y != cudaSuccess) {
        std::fprintf(stderr, "BC y-hse launch failed: %s\n", cudaGetErrorString(err_y));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CartImplSolver::apply_floor() {
    dim3 B2d(16, 16);
    dim3 G2d((nx+15)/16, (ny+15)/16);
    k_ci_floor<<<G2d, B2d>>>(d_rho, d_mx, d_my, d_rhoE, d_rho0, eos,
                              atm_rho_thresh, nx, ny, ng);
}

// ---------------------------------------------------------------------
// compute_F: F(U) = (U-Uⁿ)/dt - (R(U) - R_hse)
// writes d_res.
// ---------------------------------------------------------------------
void CartImplSolver::compute_F(double inv_dt) {
    apply_bc();
    dim3 B2d(16, 16);
    dim3 G2d((nx+15)/16, (ny+15)/16);
    k_ci_residual<<<G2d, B2d>>>(d_rho, d_mx, d_my, d_rhoE, d_rho0, d_P0,
                                 d_res,
                                 nx, ny, ng, dx, dy, g_y,
                                 eos, atm_rho_thresh, limiter_type, hllc_variant);
    k_ci_compute_F<<<G2d, B2d>>>(d_res,
                                  d_rho, d_mx, d_my, d_rhoE,
                                  d_Un, d_hse_defect,
                                  d_res, inv_dt,
                                  nx, ny, ng);
}

double CartImplSolver::residual_norm() {
    int N = 4 * phys;
    double s2 = gpu_dot_ci(d_res, d_res, N);
    return std::sqrt(s2 / (double)N);
}

// ---------------------------------------------------------------------
// compute_cfl_dt — |vx|+cs / dx, |vy|+cs / dy
// ---------------------------------------------------------------------
double CartImplSolver::compute_cfl_dt() {
    static double* d_out = nullptr;
    static int alloc_sz = 0;
    int n = phys;
    int B = 256;
    int nb = (n + B - 1) / B;
    if (alloc_sz < nb) {
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_out, nb * sizeof(double));
        alloc_sz = nb;
    }
    k_ci_cfl<<<nb, B, B*sizeof(double)>>>(d_rho, d_mx, d_my, d_rhoE, d_out,
                         nx, ny, ng, eos, dx, dy, atm_rho_thresh);
    std::vector<double> h_out(nb);
    cudaMemcpy(h_out.data(), d_out, nb*sizeof(double), cudaMemcpyDeviceToHost);
    double inv_dt_max = 0.0;
    for (double v : h_out) if (v > inv_dt_max) inv_dt_max = v;
    if (inv_dt_max <= 0.0) return 1e-3;
    return cfl_num / inv_dt_max;
}

// ---------------------------------------------------------------------
// step — save Uⁿ, pick dt, Newton solve
// ---------------------------------------------------------------------
double CartImplSolver::step(double t, double t_end) {
    if (!hse_set) snapshot_hse();
    apply_floor();
    apply_bc();

    double dt = compute_cfl_dt();
    if (dt_current > 0.0 && dt > 1.5 * dt_current) dt = 1.5 * dt_current;
    if (t + dt > t_end) dt = t_end - t;

    // Save Uⁿ: pack 4-field state into flat phys×4 buffer
    dim3 B2d(16, 16);
    dim3 G2d((nx+15)/16, (ny+15)/16);
    k_ci_pack_state<<<G2d, B2d>>>(d_Un, d_rho, d_mx, d_my, d_rhoE, nx, ny, ng);

    int max_cuts = 5;
    double dt_try = dt;
    for (int cut = 0; cut <= max_cuts; ++cut) {
        // Restore state from Un before each attempt
        if (cut > 0) {
            k_ci_unpack_state<<<G2d, B2d>>>(d_Un, d_rho, d_mx, d_my, d_rhoE, nx, ny, ng);
        }
        int newton_iters = newton_solve(dt_try);
        if (newton_iters >= 0) {
            dt_current = dt_try;
            step_count++;
            return dt_try;
        }
        dt_try *= 0.5;
    }
    // All cuts failed: rollback to Un
    k_ci_unpack_state<<<G2d, B2d>>>(d_Un, d_rho, d_mx, d_my, d_rhoE, nx, ny, ng);
    std::fprintf(stderr, "  step %d: rollback to Un (all cuts failed)\n", step_count);
    step_count++;
    return dt;
}

// ---------------------------------------------------------------------
// Host download helper
// ---------------------------------------------------------------------
void CartImplSolver::download_host(std::vector<double>& rho,
                                    std::vector<double>& mx,
                                    std::vector<double>& my,
                                    std::vector<double>& E)
{
    std::vector<double> h_rho(total), h_mx(total), h_my(total), h_E(total);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho,  total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mx.data(),  d_mx,   total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_my.data(),  d_my,   total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_E.data(),   d_rhoE, total*sizeof(double), cudaMemcpyDeviceToHost));

    rho.resize(phys); mx.resize(phys); my.resize(phys); E.resize(phys);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            int fk = ci_flat(i, j, ny);
            int k  = ci_idx(i, j, ny, ng);
            rho[fk] = h_rho[k];
            mx[fk]  = h_mx[k];
            my[fk]  = h_my[k];
            E[fk]   = h_E[k];
        }
}

CartImplSolver::Diag CartImplSolver::compute_diagnostics() {
    std::vector<double> rho, mx, my, E;
    download_host(rho, mx, my, E);
    double dA = dx * dy;
    double M = 0, Etot = 0, maxv = 0, maxM = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            int fk = i*ny + j;
            double r = std::max(rho[fk], 1e-30);
            double vx = mx[fk]/r, vy = my[fk]/r;
            double v = std::sqrt(vx*vx + vy*vy);
            double ie = std::max(E[fk] - 0.5*r*(vx*vx+vy*vy), 1e-30);
            double P = (gamma - 1.0) * ie;
            double cs = std::sqrt(std::max(gamma * P / r, 1e-30));
            M    += rho[fk] * dA;
            Etot += E[fk]   * dA;
            if (v > maxv) maxv = v;
            double Mach = v / cs;
            if (Mach > maxM) maxM = Mach;
        }
    return { M, Etot, maxv, maxM };
}

void CartImplSolver::write_vtk(const char* filename) {
    std::vector<double> rho, mx, my, E;
    download_host(rho, mx, my, E);

    std::FILE* fp = std::fopen(filename, "w");
    if (!fp) return;
    int nnx = nx + 1, nny = ny + 1;
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "cart_impl 2D Cartesian implicit solver\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\n");
    std::fprintf(fp, "DIMENSIONS %d %d 1\n", nnx, nny);
    std::fprintf(fp, "POINTS %d double\n", nnx * nny);
    for (int jn = 0; jn < nny; ++jn) {
        double y = Ly * (double)jn / (double)(nny - 1);
        for (int in = 0; in < nnx; ++in) {
            double x = Lx * (double)in / (double)(nnx - 1);
            std::fprintf(fp, "%.10e %.10e %.10e\n", x, y, 0.0);
        }
    }
    std::fprintf(fp, "CELL_DATA %d\n", phys);

    // density
    std::fprintf(fp, "SCALARS density double 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            std::fprintf(fp, "%.10e\n", rho[i*ny + j]);

    // pressure
    std::fprintf(fp, "SCALARS pressure double 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int fk = i*ny + j;
            double r = std::max(rho[fk], 1e-30);
            double vx = mx[fk]/r, vy = my[fk]/r;
            double ie = std::max(E[fk] - 0.5*r*(vx*vx+vy*vy), 1e-30);
            double P = (gamma - 1.0) * ie;
            std::fprintf(fp, "%.10e\n", P);
        }

    // velocity (cell-centered: vx, vy, 0)
    std::fprintf(fp, "VECTORS velocity double\n");
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int fk = i*ny + j;
            double r = std::max(rho[fk], 1e-30);
            std::fprintf(fp, "%.10e %.10e %.10e\n", mx[fk]/r, my[fk]/r, 0.0);
        }

    // mach
    std::fprintf(fp, "SCALARS mach double 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int fk = i*ny + j;
            double r = std::max(rho[fk], 1e-30);
            double vx = mx[fk]/r, vy = my[fk]/r;
            double v = std::sqrt(vx*vx + vy*vy);
            double ie = std::max(E[fk] - 0.5*r*(vx*vx+vy*vy), 1e-30);
            double P = (gamma - 1.0) * ie;
            double cs = std::sqrt(std::max(gamma * P / r, 1e-30));
            std::fprintf(fp, "%.10e\n", v / cs);
        }
    std::fclose(fp);
}
