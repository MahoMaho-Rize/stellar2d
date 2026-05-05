// Cartesian implicit solver JFNK Newton-Krylov.
// Borrows fas2 fixes: CGS2 Gram-Schmidt, unit-normalize v̂, Viallet scaling.
// Preconditioner: identity (for now — Cartesian uniform mesh has κ(J) ~ O(N),
// much better than the 10⁷ of spherical log mesh,identity 通常夠用).
//
// Reuses host-side dot via gpu_dot_ci.

#include "cart_impl_solver.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// Kernel forward decls
extern __global__ void k_ci_copy(double* dst, const double* src, int N);
extern __global__ void k_ci_axpy(double* y, double a, const double* x, int N);
extern __global__ void k_ci_scale(double* x, double a, int N);
extern __global__ void k_ci_neg(double* x, int N);
extern __global__ void k_ci_build_scaling(
    const double* rho, const double* mx, const double* my,
    const double* rho0, const double* P0,
    double* L, double* R, double* invL,
    EOS eos, double alpha1, double alpha2,
    int nx, int ny, int ng);
extern __global__ void k_ci_scale_by_diag(double* x, const double* D, int N);
extern __global__ void k_ci_pack_state(double* out,
    const double* rho, const double* mx, const double* my, const double* rhoE,
    int nx, int ny, int ng);
extern __global__ void k_ci_unpack_state(const double* in,
    double* rho, double* mx, double* my, double* rhoE,
    int nx, int ny, int ng);

extern double gpu_dot_ci(const double* a, const double* b, int N);
static double gpu_norm_ci(const double* a, int N) { return std::sqrt(gpu_dot_ci(a, a, N)); }

// ---------------------------------------------------------------------
// Perturb kernel: U += alpha · Rv (Rv is 4-packed perturbation vector)
// ---------------------------------------------------------------------
__global__ void k_ci_perturb(
    double* rho, double* mx, double* my, double* rhoE,
    const double* Rv, double alpha,
    int nx, int ny, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int k = (i+ng) * (ny+2*ng) + (j+ng);
    int fk = i*ny + j;
    rho[k]  += alpha * Rv[4*fk+0];
    mx[k]   += alpha * Rv[4*fk+1];
    my[k]   += alpha * Rv[4*fk+2];
    rhoE[k] += alpha * Rv[4*fk+3];
}

__global__ void k_ci_state_update(
    double* rho, double* mx, double* my, double* rhoE,
    const double* dU, double alpha,
    int nx, int ny, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int k = (i+ng) * (ny+2*ng) + (j+ng);
    int fk = i*ny + j;
    rho[k]  += alpha * dU[4*fk+0];
    mx[k]   += alpha * dU[4*fk+1];
    my[k]   += alpha * dU[4*fk+2];
    rhoE[k] += alpha * dU[4*fk+3];
}

// ---------------------------------------------------------------------
// JFNK matvec: J·(R·v) with Viallet scaling, unit-normalize v̂
// Returns invL · J · (R·v).
// ---------------------------------------------------------------------
void CartImplSolver::jfnk_matvec(const double* d_v, double* d_Jv, double inv_dt) {
    int N4 = 4 * phys;
    int B = 256;
    dim3 B2d(16,16);
    dim3 G2d((nx+15)/16, (ny+15)/16);

    // Step 1: w = R · v(if scaling on)—— use d_gmres_w as temp,then swap
    const double* v_for_perturb = d_v;
    if (use_music_scaling) {
        k_ci_copy<<<(N4+B-1)/B, B>>>(d_gmres_w, d_v, N4);
        k_ci_scale_by_diag<<<(N4+B-1)/B, B>>>(d_gmres_w, d_scale_R, N4);
        v_for_perturb = d_gmres_w;
    }

    // Unit-normalize v̂: avoid amplifying round-off noise from Arnoldi basis
    double norm_U = gpu_norm_ci(d_Un, N4);
    double norm_v = gpu_norm_ci(v_for_perturb, N4);
    double norm_v_floor = 1e-15 * (1.0 + norm_U);
    if (norm_v < norm_v_floor) {
        CUDA_CHECK(cudaMemset(d_Jv, 0, N4*sizeof(double)));
        return;
    }
    double eps_hat = std::sqrt(1e-15) * (1.0 + norm_U);
    double alpha_v = eps_hat / norm_v;

    // Save state into d_Ubak
    k_ci_pack_state<<<G2d, B2d>>>(d_Ubak, d_rho, d_mx, d_my, d_rhoE, nx, ny, ng);

    // Perturb: U += alpha_v · (R·v)
    k_ci_perturb<<<G2d, B2d>>>(d_rho, d_mx, d_my, d_rhoE,
                                v_for_perturb, alpha_v, nx, ny, ng);
    apply_floor();
    apply_bc();

    // F(U + eps_hat · v̂_scaled)
    compute_F(inv_dt);
    // d_res now has F(U+ε·v̂)
    // J·(R·v) = (F(U+ε·v̂) − F_k) · ‖R·v‖ / eps_hat
    k_ci_copy<<<(N4+B-1)/B, B>>>(d_Jv, d_res, N4);
    k_ci_axpy<<<(N4+B-1)/B, B>>>(d_Jv, -1.0, d_Fk, N4);
    double scale_Jv = norm_v / eps_hat;
    k_ci_scale<<<(N4+B-1)/B, B>>>(d_Jv, scale_Jv, N4);

    // Apply invL: invL · J · (R·v) — this is the scaled operator
    if (use_music_scaling) {
        k_ci_scale_by_diag<<<(N4+B-1)/B, B>>>(d_Jv, d_scale_invL, N4);
    }

    // Restore state
    k_ci_unpack_state<<<G2d, B2d>>>(d_Ubak, d_rho, d_mx, d_my, d_rhoE, nx, ny, ng);
    apply_bc();
}

// ---------------------------------------------------------------------
// Preconditioner M⁻¹
// Start simple: identity (for Cartesian uniform mesh κ(J) ~ O(N),
// likely adequate). Can upgrade to line-implicit-y later if needed.
// ---------------------------------------------------------------------
void CartImplSolver::apply_precond(const double* d_v, double* d_Mv, double /*inv_dt*/) {
    int N4 = 4 * phys, B = 256;
    // M⁻¹ = I
    k_ci_copy<<<(N4+B-1)/B, B>>>(d_Mv, d_v, N4);
}

// ---------------------------------------------------------------------
// FGMRES with CGS2 Gram-Schmidt + Viallet-scaled operator
// Solves M⁻¹·J·δX = −F, then back-scales δU = R·δX
// ---------------------------------------------------------------------
int CartImplSolver::gmres_solve(double* d_x, const double* d_b,
                                 double inv_dt, double tol, int max_iter)
{
    int N = 4 * phys, B = 256;
    int m = std::min(max_iter, (int)GMRES_K);
    std::vector<double> H((m+1)*m, 0.0);
    std::vector<double> cs(m), sn(m), g(m+1, 0.0);

    // r₀ = -b (solving J·δ = -F, so Krylov RHS is -F)
    k_ci_copy<<<(N+B-1)/B, B>>>(d_V[0], d_b, N);
    k_ci_scale<<<(N+B-1)/B, B>>>(d_V[0], -1.0, N);
    double beta = gpu_norm_ci(d_V[0], N);
    if (beta < 1e-30) return 0;
    k_ci_scale<<<(N+B-1)/B, B>>>(d_V[0], 1.0/beta, N);
    g[0] = beta;

    int j;
    for (j = 0; j < m; ++j) {
        // z_j = M⁻¹ · v_j
        apply_precond(d_V[j], d_Z[j], inv_dt);
        // w = J · z_j (scaled)
        jfnk_matvec(d_Z[j], d_gmres_w, inv_dt);

        // CGS2 Gram-Schmidt: two passes of MGS
        for (int i = 0; i <= j; ++i) {
            H[i*m+j] = gpu_dot_ci(d_gmres_w, d_V[i], N);
            k_ci_axpy<<<(N+B-1)/B, B>>>(d_gmres_w, -H[i*m+j], d_V[i], N);
        }
        for (int i = 0; i <= j; ++i) {
            double h_corr = gpu_dot_ci(d_gmres_w, d_V[i], N);
            H[i*m+j] += h_corr;
            k_ci_axpy<<<(N+B-1)/B, B>>>(d_gmres_w, -h_corr, d_V[i], N);
        }
        H[(j+1)*m+j] = gpu_norm_ci(d_gmres_w, N);

        if (H[(j+1)*m+j] < 1e-30) { j++; break; }
        k_ci_copy<<<(N+B-1)/B, B>>>(d_V[j+1], d_gmres_w, N);
        k_ci_scale<<<(N+B-1)/B, B>>>(d_V[j+1], 1.0/H[(j+1)*m+j], N);

        // Givens rotations
        for (int i = 0; i < j; ++i) {
            double h1 = H[i*m+j], h2 = H[(i+1)*m+j];
            H[i*m+j]     =  cs[i]*h1 + sn[i]*h2;
            H[(i+1)*m+j] = -sn[i]*h1 + cs[i]*h2;
        }
        double h1 = H[j*m+j], h2 = H[(j+1)*m+j];
        double t = std::sqrt(h1*h1 + h2*h2);
        cs[j] = h1/t; sn[j] = h2/t;
        H[j*m+j] = t; H[(j+1)*m+j] = 0.0;
        g[j+1] = -sn[j]*g[j]; g[j] = cs[j]*g[j];

        if (std::fabs(g[j+1]) < tol * beta) { j++; break; }
    }

    // Back-substitution
    std::vector<double> y(j);
    for (int i = j-1; i >= 0; --i) {
        y[i] = g[i];
        for (int kk = i+1; kk < j; ++kk)
            y[i] -= H[i*m+kk] * y[kk];
        y[i] /= H[i*m+i];
    }

    CUDA_CHECK(cudaMemset(d_x, 0, N*sizeof(double)));
    for (int i = 0; i < j; ++i)
        k_ci_axpy<<<(N+B-1)/B, B>>>(d_x, y[i], d_Z[i], N);

    return j;
}

// ---------------------------------------------------------------------
// Newton outer loop — returns iteration count on success, -1 on failure
// ---------------------------------------------------------------------
int CartImplSolver::newton_solve(double dt) {
    int N = 4 * phys, B = 256;
    double inv_dt = 1.0 / dt;
    dim3 B2d(16, 16);
    dim3 G2d((nx+15)/16, (ny+15)/16);

    double init_res = -1.0;
    for (int it = 0; it < newton_max_iter; ++it) {
        // Build scaling from current state (must happen each Newton iter)
        if (use_music_scaling) {
            k_ci_build_scaling<<<G2d, B2d>>>(
                d_rho, d_mx, d_my, d_rho0, d_P0,
                d_scale_L, d_scale_R, d_scale_invL,
                eos, music_alpha1, music_alpha2, nx, ny, ng);
        }

        // F(U_k) cached for matvec
        compute_F(inv_dt);
        k_ci_copy<<<(N+B-1)/B, B>>>(d_Fk, d_res, N);
        double res_norm = residual_norm();
        if (it == 0) init_res = res_norm;
        if (it == 0 && res_norm < newton_tol) return 0;
        if (!std::isfinite(res_norm)) {
            std::fprintf(stderr, "  newton: NaN residual at iter %d\n", it);
            return -1;
        }

        // Scale RHS: d_res ← invL · F
        if (use_music_scaling) {
            k_ci_scale_by_diag<<<(N+B-1)/B, B>>>(d_res, d_scale_invL, N);
        }

        // Solve J·δX = -F (in scaled space)
        int gm_iters = gmres_solve(d_gmres_w, d_res, inv_dt, gmres_tol, gmres_max_iter);

        // Back-scale: δU = R · δX
        if (use_music_scaling) {
            k_ci_scale_by_diag<<<(N+B-1)/B, B>>>(d_gmres_w, d_scale_R, N);
        }

        // State update: U ← U + δU
        k_ci_state_update<<<G2d, B2d>>>(d_rho, d_mx, d_my, d_rhoE,
                                         d_gmres_w, 1.0, nx, ny, ng);
        apply_floor();
        apply_bc();

        // Check convergence: post-update residual
        compute_F(inv_dt);
        double new_res = residual_norm();
        if (!std::isfinite(new_res)) {
            std::fprintf(stderr, "  newton iter %d: NaN after update\n", it);
            return -1;
        }
        if (step_count < 2 && it < 3) {
            std::fprintf(stderr,
                "  newton iter %d: ||F||=%.3e → %.3e, GMRES=%d\n",
                it, res_norm, new_res, gm_iters);
        }
        // Converge if absolute below newton_tol, OR relative below 0.5 (half-drop),
        // OR stalled (2 iters with < 5% change).
        bool abs_ok = (new_res < newton_tol);
        bool rel_ok = (init_res > 0.0 && new_res < 0.5 * init_res);
        bool stalled = (it >= 2 && res_norm > 0.0 && std::fabs(new_res - res_norm) < 0.05 * res_norm);
        if (abs_ok || rel_ok || stalled) {
            if (step_count < 3 || step_count % 500 == 0) {
                std::fprintf(stderr,
                    "  newton converged in %d iters (||F||: %.3e → %.3e, GMRES=%d)\n",
                    it+1, init_res, new_res, gm_iters);
            }
            return it + 1;
        }
    }
    std::fprintf(stderr, "  newton: max_iter %d reached (||F|| still high)\n",
                 newton_max_iter);
    return -1;
}
