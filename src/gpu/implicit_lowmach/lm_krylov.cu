// LowMach Krylov solver: JFNK matrix-free Jacobian-vector product + FGMRES

#include "lowmach_solver.h"
#include "lm_common.cuh"

// ========================= JFNK matvec ============================
// Pure 4-DOF: GMRES on fluid (ρ, ρvr, ρvθ, ρe).
// Gravity g(r) is recomputed from ρ in compute_residual, so the
// Jacobian-vector product J·v naturally includes ∂g/∂ρ coupling.

void LowMachSolver::jfnk_matvec(const double* d_v, double* d_Jv, double dt) {
    int n = nr*nt, N4 = 4*n, B = 256;

    double norm_v = sqrt(gpu_dot(d_v, d_v, d_work_a, d_work_b, N4));
    if (norm_v < 1e-30) { k_lm_zero<<<(N4+B-1)/B,B>>>(d_Jv, N4); return; }

    double norm_U = sqrt(gpu_dot(d_Un, d_Un, d_work_a, d_work_b, N4));
    double eps_fd = sqrt(1e-15) * (1.0 + norm_U) / norm_v;

    k_lm_pack<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_gmres_Uk, nr,nt,ng);

    k_lm_unpack_add<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_v, eps_fd, nr,nt,ng);
    apply_floor();

    // compute_residual recomputes g(r) from perturbed ρ, so ∂g/∂ρ is captured
    compute_residual(d_residual);
    k_lm_compute_F<<<(n+B-1)/B,B>>>(d_Jv, d_residual,
        d_rho, d_mr, d_mtheta, d_rhoE, d_Un, 1.0/dt, nr, nt, ng);

    k_lm_axpy<<<(N4+B-1)/B,B>>>(d_Jv, -1.0, d_Fk, N4);
    k_lm_scale<<<(N4+B-1)/B,B>>>(d_Jv, 1.0/eps_fd, N4);

    k_lm_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_gmres_Uk, nr,nt,ng);
}

// ========================= FGMRES ================================
// Right-preconditioned FGMRES in physical space.
// GMRES minimizes ||F + J·δU||₂ directly.
// L scaling is used only for convergence check and Newton merit function,
// not inside the Krylov basis — this keeps the preconditioner consistent.

int LowMachSolver::gmres_solve(double* d_x, const double* d_b, double dt,
                                double tol, int max_iter) {
    int n = nr*nt, N = 4*n, B = 256;
    int m = std::min(max_iter, (int)GMRES_RESTART);

    std::vector<double> H((m+1)*m, 0.0);
    std::vector<double> cs(m), sn(m), g(m+1, 0.0);

    k_lm_copy<<<(N+B-1)/B,B>>>(d_gmres_V[0], d_b, N);
    k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[0], -1.0, N);
    k_lm_zero<<<(n+B-1)/B,B>>>(d_gmres_V[0] + N, n);

    double beta = sqrt(gpu_dot(d_gmres_V[0], d_gmres_V[0], d_work_a, d_work_b, N));
    if (beta < 1e-30) return 0;
    k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[0], 1.0/beta, N);
    g[0] = beta;

    int j;
    for (j = 0; j < m; ++j) {
        apply_preconditioner(d_gmres_V[j], d_gmres_Z[j], dt);
        k_lm_zero<<<(n+B-1)/B,B>>>(d_gmres_Z[j] + N, n);

        jfnk_matvec(d_gmres_Z[j], d_gmres_w, dt);
        k_lm_zero<<<(n+B-1)/B,B>>>(d_gmres_w + N, n);

        for (int i = 0; i <= j; ++i) {
            H[i*m+j] = gpu_dot(d_gmres_w, d_gmres_V[i], d_work_a, d_work_b, N);
            k_lm_axpy<<<(N+B-1)/B,B>>>(d_gmres_w, -H[i*m+j], d_gmres_V[i], N);
        }
        H[(j+1)*m+j] = sqrt(gpu_dot(d_gmres_w, d_gmres_w, d_work_a, d_work_b, N));

        if (H[(j+1)*m+j] < 1e-30) { j++; break; }
        k_lm_copy<<<(N+B-1)/B,B>>>(d_gmres_V[j+1], d_gmres_w, N);
        k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[j+1], 1.0/H[(j+1)*m+j], N);

        for (int i = 0; i < j; ++i) {
            double h1 = H[i*m+j], h2 = H[(i+1)*m+j];
            H[i*m+j]     =  cs[i]*h1 + sn[i]*h2;
            H[(i+1)*m+j] = -sn[i]*h1 + cs[i]*h2;
        }
        double h1 = H[j*m+j], h2 = H[(j+1)*m+j];
        double t = sqrt(h1*h1 + h2*h2);
        cs[j] = h1/t; sn[j] = h2/t;
        H[j*m+j] = t; H[(j+1)*m+j] = 0.0;
        g[j+1] = -sn[j]*g[j]; g[j] = cs[j]*g[j];

        if (fabs(g[j+1]) < tol * beta) { j++; break; }
    }

    std::vector<double> y(j);
    for (int i = j-1; i >= 0; --i) {
        y[i] = g[i];
        for (int kk = i+1; kk < j; ++kk)
            y[i] -= H[i*m+kk] * y[kk];
        y[i] /= H[i*m+i];
    }

    k_lm_zero<<<(N+B-1)/B,B>>>(d_x, N);
    for (int i = 0; i < j; ++i)
        k_lm_axpy<<<(N+B-1)/B,B>>>(d_x, y[i], d_gmres_Z[i], N);

    return j;
}
