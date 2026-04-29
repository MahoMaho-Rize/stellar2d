// FAS smoothers: block-Jacobi, SIMPLE pressure correction, GMRES(k)

#include "fas_solver.cuh"
#include "fas_linalg.cuh"
#include <cmath>
#include <vector>

// ========================= Block Jacobi smoother ========================

__global__
void k_fas_assemble_blkjac(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* gr0,
    double* blk_inv,
    int nr, int nt, int ng, double gam, double inv_dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = fas_idx(i,j,nt,ng);

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double KE_c = 0.5 * rho_c * (vr_c*vr_c + vt_c*vt_c);
    double P_c = fmax((gam-1.0)*(rhoE[k] - KE_c), 1e-30);
    double cs = sqrt(gam * P_c / rho_c);
    double r = (i == 0 && r_face[1] > 1e-30) ? (2.0/3.0)*r_face[1] : r_center[i];
    double invV = 1.0 / vol[flat];

    // Factor 2: MUSCL reconstruction doubles the effective stencil width vs donor-cell
    double sr = 2.0 * ((fabs(vr_c)+cs)/dr[i] + (fabs(vt_c)+cs)/(r*dtheta[j]));
    // For total energy E: P = (γ-1)(E - ½ρv²), so ∂P/∂E = (γ-1)
    double dP_drhoE = gam - 1.0;

    double dPdr_coeff = 0.0;
    if (i > 0 && i < nr-1) {
        double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
        dPdr_coeff = -(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh)));
    } else if (i == 0 && nr > 1) {
        dPdr_coeff = -1.0/(r_center[1]-r_center[0]);
    } else if (i == nr-1 && nr >= 2) {
        dPdr_coeff = 1.0/(r_center[nr-1]-r_center[nr-2]);
    }

    double dPdt_coeff = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        dPdt_coeff = -(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh))) / r;
    }

    double g0_val = gr0[i];

    double Ar_hi=ar[(i+1)*nt+j], Ar_lo=ar[i*nt+j];
    double At_hi=at[i*(nt+1)+j+1], At_lo=at[i*(nt+1)+j];
    double ddivv_dmr = invV * 0.5/rho_c * (Ar_hi + Ar_lo);
    double ddivv_dmt = invV * 0.5/rho_c * (At_hi + At_lo);

    double J[16];
    for (int q=0; q<16; q++) J[q] = 0.0;
    J[0]  = -(inv_dt + sr);
    J[5]  = -(inv_dt + sr);
    J[10] = -(inv_dt + sr);
    J[15] = -(inv_dt + sr);
    J[1*4+3] = -dPdr_coeff * dP_drhoE;
    J[2*4+3] = -dPdt_coeff * dP_drhoE;
    J[1*4+0] += g0_val;
    // Total energy: R_E = -∇·((E+P)v) + ρv·g
    // ∂R_E/∂(mr) ≈ -(E+P)/ρ · ∂(∇·(ρv))/∂(mr) ≈ signal speed contribution (in sr)
    // ∂R_E/∂E = -(γ)·∇·v (from dP/dE = γ-1 in (E+P)u flux)
    // Simplified: coupling via spectral radius is already in sr on diagonal
    // Keep off-diagonal small to maintain diagonal dominance

    // 4×4 inversion via Gauss-Jordan
    double A[16]; for(int q=0;q<16;q++) A[q]=J[q];
    double inv_m[16]; for(int q=0;q<16;q++) inv_m[q]=(q/4==q%4)?1.0:0.0;

    for(int col=0;col<4;col++){
        double mx=fabs(A[col*4+col]); int mi=col;
        for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
        if(mi!=col){
            for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
            for(int q=0;q<4;q++){double t=inv_m[col*4+q];inv_m[col*4+q]=inv_m[mi*4+q];inv_m[mi*4+q]=t;}
        }
        double d=A[col*4+col]; if(fabs(d)<1e-30) d=(d>=0?1e-30:-1e-30);
        for(int row=col+1;row<4;row++){
            double m=A[row*4+col]/d;
            for(int q=col;q<4;q++) A[row*4+q]-=m*A[col*4+q];
            for(int q=0;q<4;q++) inv_m[row*4+q]-=m*inv_m[col*4+q];
        }
    }
    for(int c=0;c<4;c++){
        for(int row=3;row>=0;row--){
            double s=inv_m[row*4+c];
            for(int q=row+1;q<4;q++) s-=A[row*4+q]*inv_m[q*4+c];
            inv_m[row*4+c]=s/A[row*4+row];
        }
    }

    double* B = &blk_inv[flat*16];
    for(int q=0;q<16;q++) B[q]=inv_m[q];
}

// ========================= SIMPLE smoother kernels ========================

__global__
void k_fas_mom_diag(const double* rho, const double* mr, const double* mt,
                    const double* rhoE,
                    const double* dr, const double* rc, const double* rf,
                    const double* dtheta,
                    double* Ap, int nr, int nt, int ng,
                    double inv_dt, double gam) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = fas_idx(i,j,nt,ng);
    double rho_c = fmax(rho[k], 1e-20);
    double vr = fabs(mr[k]/rho_c), vt = fabs(mt[k]/rho_c);
    double KE = 0.5 * rho_c * (vr*vr + vt*vt);
    double P = fmax((gam - 1.0) * (rhoE[k] - KE), 1e-30);
    double cs = sqrt(gam * P / rho_c);
    double r_eff = (i == 0 && rf[1] > 1e-30) ? (2.0/3.0)*rf[1] : rc[i];
    Ap[flat] = inv_dt + (vr + cs)/dr[i] + (vt + cs)/(r_eff*dtheta[j]);
}

__global__
void k_fas_vstar(const double* F, const double* Ap, const double* rho0,
                 double atm_thresh, double* vr_s, double* vt_s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (rho0[i] < atm_thresh) { vr_s[i] = 0; vt_s[i] = 0; return; }
    double inv_ap = 1.0 / Ap[i];
    vr_s[i] = -F[n + i] * inv_ap;
    vt_s[i] = -F[2*n + i] * inv_ap;
}

__global__
void k_fas_div(const double* vr_s, const double* vt_s,
               const double* vol, const double* ar, const double* at,
               double* div_out, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    double invV = 1.0 / vol[flat];
    double vr_lo = (i>0)    ? 0.5*(vr_s[flat]+vr_s[(i-1)*nt+j]) : 0.0;
    double vr_hi = (i<nr-1) ? 0.5*(vr_s[flat]+vr_s[(i+1)*nt+j]) : 0.0;
    double vt_lo = (j>0)    ? 0.5*(vt_s[flat]+vt_s[i*nt+j-1])   : 0.0;
    double vt_hi = (j<nt-1) ? 0.5*(vt_s[flat]+vt_s[i*nt+j+1])   : 0.0;
    div_out[flat] = invV*(ar[(i+1)*nt+j]*vr_hi - ar[i*nt+j]*vr_lo
                         + at[i*(nt+1)+j+1]*vt_hi - at[i*(nt+1)+j]*vt_lo);
}

__global__
void k_fas_prhs(const double* div_v, double* rhs, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    rhs[flat] = (flat/nt == nr-1) ? 0.0 : div_v[flat];
}

__global__
void k_fas_inv_ap(const double* Ap, double* alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) alpha[i] = 1.0 / Ap[i];
}

// SIMPLE correction: momentum via Poisson-corrected velocity, ρ/E via block-inverse
__global__
void k_fas_simple_correct(
    double* rho, double* mr, double* mt, double* rhoE,
    const double* F, const double* blk_inv,
    const double* vr_s, const double* vt_s,
    const double* dp, const double* Ap,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* rho0, double atm_thresh,
    int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (rho0[flat] < atm_thresh) return;
    int n = nr*nt;
    int i = flat/nt, j = flat%nt;
    int k = fas_idx(i,j,nt,ng);
    double rho_c = fmax(rho[k], 1e-20);

    // Pressure gradient of δp
    int im = flat - nt*(int)(i>0), ip = flat + nt*(int)(i<nr-1);
    double dp_dr = 0.0;
    if (i > 0 && i < nr-1) {
        double dl=r_center[i]-r_center[i-1], dh=r_center[i+1]-r_center[i];
        dp_dr = (dh*(dp[flat]-dp[im])/dl + dl*(dp[ip]-dp[flat])/dh)/(dl+dh);
    } else if (i==0 && nr>1) dp_dr = (dp[ip]-dp[flat])/(r_center[1]-r_center[0]);
    else if (i==nr-1 && nr>=2) dp_dr = (dp[flat]-dp[im])/(r_center[nr-1]-r_center[nr-2]);

    double r_eff = (i == 0 && r_face[1] > 1e-30) ? (2.0/3.0)*r_face[1] : r_center[i];
    double dp_dt_r = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        dp_dt_r = (dh*(dp[flat]-dp[i*nt+j-1])/dl + dl*(dp[i*nt+j+1]-dp[flat])/dh)
                  / (r_eff*(dl+dh));
    }

    double inv_ap = 1.0 / Ap[flat];
    double dvr = vr_s[flat] - inv_ap * dp_dr;
    double dvt = vt_s[flat] - inv_ap * dp_dt_r;

    // Under-relaxed momentum correction (0.3): reduces conflict between
    // SIMPLE's central-diff ∇δp and HLLC's Riemann-fan pressure
    mr[k]  -= rho_c * dvr;
    mt[k]  -= rho_c * dvt;

    const double* B = &blk_inv[flat*16];
    double f0 = F[flat], f1 = F[n+flat], f2 = F[2*n+flat], f3 = F[3*n+flat];
    rho[k]  -= B[0]*f0 + B[1]*f1 + B[2]*f2 + B[3]*f3;
    rhoE[k] -= B[12]*f0 + B[13]*f1 + B[14]*f2 + B[15]*f3;
}

// ========================= SIMPLE smoother ========================

void FasSolver::assemble_smoother(int l, double g0_over_dt) {
    FasLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    k_fas_assemble_blkjac<<<(n+B-1)/B,B>>>(
        lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_cell_volume, lev.d_area_r, lev.d_area_theta,
        lev.d_r_center, lev.d_r_face, lev.d_theta_face,
        lev.d_dr, lev.d_dtheta,
        lev.d_gr0,
        lev.d_blk_inv,
        lev.nr, lev.nt, lev.ng, gamma, g0_over_dt);
    k_fas_mom_diag<<<(n+B-1)/B,B>>>(
        lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_dr, lev.d_r_center, lev.d_r_face, lev.d_dtheta,
        lev.d_Ap, lev.nr, lev.nt, lev.ng, g0_over_dt, gamma);
}

// Block Jacobi smooth: U ← U - ω · J⁻¹_diag · F(U)
__global__
void k_fas_smooth_blkjac(
    double* rho, double* mr, double* mt, double* rhoE,
    const double* F, const double* blk_inv,
    const double* rho0, double atm_thresh,
    double omega, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (rho0[flat] < atm_thresh) return;
    int n = nr*nt;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    const double* B = &blk_inv[flat*16];
    double f0 = F[flat], f1 = F[n+flat], f2 = F[2*n+flat], f3 = F[3*n+flat];
    rho[k]  -= omega * (B[0]*f0 + B[1]*f1 + B[2]*f2 + B[3]*f3);
    mr[k]   -= omega * (B[4]*f0 + B[5]*f1 + B[6]*f2 + B[7]*f3);
    mt[k]   -= omega * (B[8]*f0 + B[9]*f1 + B[10]*f2 + B[11]*f3);
    rhoE[k] -= omega * (B[12]*f0 + B[13]*f1 + B[14]*f2 + B[15]*f3);
}

void FasSolver::smooth(int l, double dt, double g0_over_dt, int n_iters) {
    FasLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;

    constexpr int KK = FasLevel::GMRES_K;  // = 3
    int n4 = 4 * n;

    for (int it = 0; it < n_iters; ++it) {
        // ===== Right-preconditioned GMRES(k) on J·δU = -F(U) =====
        // Preconditioner M⁻¹ = block-Jacobi inverse (already computed).
        // Jacobian-vector product via finite differences:
        //   J·z ≈ (F(U+ε·z) - F(U)) / ε

        // r₀ = -F(U) (= RHS of Newton system)
        compute_F(l, g0_over_dt);
        k_fas_copy<<<(n4+255)/256,256>>>(lev.d_gmres_V[0], lev.d_res, n4);
        k_fas_scale<<<(n4+255)/256,256>>>(lev.d_gmres_V[0], -1.0, n4);

        double beta = gpu_norm(lev.d_gmres_V[0], n4);
        if (beta < 1e-14) break;
        k_fas_scale<<<(n4+255)/256,256>>>(lev.d_gmres_V[0], 1.0/beta, n4);

        // Hessenberg matrix H (KK+1 x KK)
        double H[(KK+1)*KK];
        for (int q = 0; q < (KK+1)*KK; ++q) H[q] = 0.0;

        int k_actual = KK;
        for (int j = 0; j < KK; ++j) {
            // z_j = M⁻¹ · v_j
            k_fas_precond<<<(n+255)/256,256>>>(
                lev.d_gmres_V[j], lev.d_blk_inv, lev.d_gmres_Z[j], n4);

            // w = J · z_j via finite difference
            // Save current state for FD Jacobian-vector product
            k_fas_pack_flat<<<(n+255)/256,256>>>(
                lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
                lev.d_gmres_Ubak, lev.nr, lev.nt, lev.ng);

            double z_norm = gpu_norm(lev.d_gmres_Z[j], n4);
            double eps_fd = 1e-7 * std::fmax(1.0, beta) / std::fmax(z_norm, 1e-30);

            // U_pert = U + eps·z_j
            k_fas_perturb<<<(n+255)/256,256>>>(
                lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
                lev.d_gmres_Z[j], eps_fd, lev.nr, lev.nt, lev.ng);
            compute_F(l, g0_over_dt);  // F(U+ε·z) stored in d_res

            // w = (F(U+ε·z) - F(U)) / ε  →  stored in V[j+1]
            // But F(U) = -r₀·beta·... no, we need to save F(U).
            // Actually: V[0] = -F(U)/||F||, so F(U) = -beta·V[0] (only at j=0 before
            // V[0] was overwritten).  Simpler: recompute F(U) after restoring state.

            // Store F_pert in V[j+1] temporarily
            k_fas_copy<<<(n4+255)/256,256>>>(lev.d_gmres_V[j+1], lev.d_res, n4);

            // Restore state
            k_fas_unpack_flat<<<(n+255)/256,256>>>(
                lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
                lev.d_gmres_Ubak, lev.nr, lev.nt, lev.ng);
            launch_ghost(l);

            // Recompute F(U) at original state
            compute_F(l, g0_over_dt);
            // V[j+1] = (F_pert - F(U)) / eps
            k_fas_axpy_v<<<(n4+255)/256,256>>>(lev.d_gmres_V[j+1], -1.0, lev.d_res, n4);
            k_fas_scale<<<(n4+255)/256,256>>>(lev.d_gmres_V[j+1], 1.0/eps_fd, n4);

            // Modified Gram-Schmidt
            for (int i = 0; i <= j; ++i) {
                double hij = gpu_dot(lev.d_gmres_V[j+1], lev.d_gmres_V[i], n4);
                H[i*KK + j] = hij;
                k_fas_axpy_v<<<(n4+255)/256,256>>>(lev.d_gmres_V[j+1], -hij, lev.d_gmres_V[i], n4);
            }
            double hjj = gpu_norm(lev.d_gmres_V[j+1], n4);
            H[(j+1)*KK + j] = hjj;
            if (hjj < 1e-14) { k_actual = j + 1; break; }
            k_fas_scale<<<(n4+255)/256,256>>>(lev.d_gmres_V[j+1], 1.0/hjj, n4);
        }

        // Solve least-squares: min ||β·e₁ - H·y||
        double g[KK+1];
        g[0] = beta;
        for (int i = 1; i <= KK; ++i) g[i] = 0.0;

        // Givens rotations
        double cs[KK], sn[KK];
        for (int j = 0; j < k_actual; ++j) {
            for (int i = 0; i < j; ++i) {
                double tmp = cs[i]*H[i*KK+j] + sn[i]*H[(i+1)*KK+j];
                H[(i+1)*KK+j] = -sn[i]*H[i*KK+j] + cs[i]*H[(i+1)*KK+j];
                H[i*KK+j] = tmp;
            }
            double a = H[j*KK+j], b = H[(j+1)*KK+j];
            double r = std::sqrt(a*a + b*b);
            cs[j] = a / std::fmax(r, 1e-30);
            sn[j] = b / std::fmax(r, 1e-30);
            H[j*KK+j] = r;
            H[(j+1)*KK+j] = 0.0;
            double tmp = cs[j]*g[j] + sn[j]*g[j+1];
            g[j+1] = -sn[j]*g[j] + cs[j]*g[j+1];
            g[j] = tmp;
        }

        // Back-substitution
        double y[KK];
        for (int i = k_actual - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int j = i + 1; j < k_actual; ++j) y[i] -= H[i*KK+j]*y[j];
            y[i] /= std::fmax(std::fabs(H[i*KK+i]), 1e-30);
        }

        // Update: U += Σ y_j · z_j (damped)
        for (int j = 0; j < k_actual; ++j) {
            k_fas_perturb<<<(n+255)/256,256>>>(
                lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
                lev.d_gmres_Z[j], OMEGA * y[j], lev.nr, lev.nt, lev.ng);
        }
        apply_floor(l);

        // ===== SIMPLE: pressure correction (elliptic mode) =====
        compute_F(l, g0_over_dt);
        if (use_simple_smoother) {
            k_fas_vstar<<<(n+B-1)/B,B>>>(lev.d_res, lev.d_Ap,
                lev.d_rho0, atm_rho_thresh,
                lev.d_vr_s, lev.d_vt_s, n);
            k_fas_div<<<(n+B-1)/B,B>>>(lev.d_vr_s, lev.d_vt_s,
                lev.d_cell_volume, lev.d_area_r, lev.d_area_theta,
                lev.d_div_s, lev.nr, lev.nt);
            k_fas_inv_ap<<<(n+B-1)/B,B>>>(lev.d_Ap, lev.d_inv_Ap, n);
            k_fas_prhs<<<(n+B-1)/B,B>>>(lev.d_div_s, lev.d_poisson_rhs, lev.nr, lev.nt);
            CUDA_CHECK(cudaMemset(lev.d_dp, 0, n*sizeof(double)));
            lev.pressure_gmg.solve_varcoeff(lev.d_inv_Ap, lev.d_poisson_rhs, lev.d_dp, 2, 1e-2);
            k_fas_simple_correct<<<(n+B-1)/B,B>>>(
                lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
                lev.d_res, lev.d_blk_inv,
                lev.d_vr_s, lev.d_vt_s,
                lev.d_dp, lev.d_Ap,
                lev.d_r_center, lev.d_r_face, lev.d_theta_face,
                lev.d_rho0, atm_rho_thresh,
                lev.nr, lev.nt, lev.ng);
        }
        apply_floor(l);

        if (l == 0 && sponge_r_start < sponge_r_top) {
            k_fas_sponge<<<(n+B-1)/B,B>>>(
                lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
                lev.d_rho0, lev.d_P0, lev.d_r_center,
                sponge_r_start, sponge_r_top, sponge_kappa, dt,
                1.0/(gamma-1.0),
                lev.nr, lev.nt, lev.ng);
        }
    }
}
