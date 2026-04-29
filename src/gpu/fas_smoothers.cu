// FAS smoothers: block-Jacobi, SIMPLE pressure correction, GMRES(k)

#include "fas_solver.cuh"
#include "fas_linalg.cuh"
#include <cmath>
#include <vector>
#include <algorithm>

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

// ========================= Line-Jacobi preconditioner ========================
// For each θ-line (fixed j), solve the 4×4 block-tridiagonal system along r.
// Diagonal: -(1/dt + spectral_radius) + pressure-gravity coupling
// Off-diagonal: upwind advection + pressure gradient from neighbors
// Adapted from LowMach k_lm_line_solve for total energy E = ρe + ½ρv².
// Key difference: P = (γ-1)(E - ½ρv²), so ∂P/∂E = (γ-1) as in block-Jacobi.
// One GPU block per θ-line; sequential Thomas algorithm on thread 0.

__global__
void k_fas_line_solve(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* gr0,
    const double* v_in, double* Mv_out,
    int nr, int nt, int ng, double gamma, double inv_dt) {
    int j = blockIdx.x;
    if (j >= nt) return;

    extern __shared__ double smem[];
    double* L   = smem;               // nr * 16
    double* D   = L + nr * 16;        // nr * 16
    double* U_b = D + nr * 16;        // nr * 16
    double* rhs_s = U_b + nr * 16;    // nr * 4
    double* x_s = rhs_s + nr * 4;     // nr * 4

    int tid = threadIdx.x;
    int n = nr * nt;

    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        int k = fas_idx(i, j, nt, ng);

        double rho_c = fmax(rho[k], 1e-20);
        double vr_c = mr[k] / rho_c;
        double vt_c = mt[k] / rho_c;
        double KE_c = 0.5 * rho_c * (vr_c*vr_c + vt_c*vt_c);
        double P_c = fmax((gamma - 1.0) * (rhoE[k] - KE_c), 1e-30);
        double r = (i == 0 && r_face[1] > 1e-30) ? (2.0/3.0)*r_face[1] : r_center[i];
        double invV = 1.0 / vol[flat];

        double cs = sqrt(gamma * P_c / rho_c);
        double Ar_hi = ar[(i+1)*nt+j], Ar_lo = ar[i*nt+j];

        double sr = 2.0 * ((fabs(vr_c)+cs)/dr[i] + (fabs(vt_c)+cs)/(r*dtheta[j]));
        double diag_val = -(inv_dt + sr);

        double vf_lo = 0.0, vf_hi = 0.0;
        if (i > 0) {
            double rl = fmax(rho[fas_idx(i-1,j,nt,ng)], 1e-20);
            vf_lo = 0.5 * (mr[fas_idx(i-1,j,nt,ng)] / rl + vr_c);
        }
        if (i < nr - 1) {
            double rr = fmax(rho[fas_idx(i+1,j,nt,ng)], 1e-20);
            vf_hi = 0.5 * (vr_c + mr[fas_idx(i+1,j,nt,ng)] / rr);
        }

        // Lower block (coupling to i-1)
        double* Lb = &L[i * 16];
        for (int q = 0; q < 16; q++) Lb[q] = 0.0;
        if (i > 0) {
            double coeff = (vf_lo >= 0.0 ? vf_lo : 0.0) * Ar_lo * invV;
            Lb[0]  = coeff;
            Lb[5]  = coeff;
            Lb[10] = coeff;
            Lb[15] = coeff;
            // Pressure coupling to i-1: dR_mr/d(E_{i-1}) from ∇P
            double dl = r_center[i] - r_center[i-1];
            double dh = (i < nr-1) ? r_center[i+1] - r_center[i] : dl;
            double pcoeff = -(gamma - 1.0) * dh / (dl * (dl + dh));
            Lb[1*4+3] += pcoeff;
        }

        // Upper block (coupling to i+1)
        double* Ub = &U_b[i * 16];
        for (int q = 0; q < 16; q++) Ub[q] = 0.0;
        if (i < nr - 1) {
            double coeff = (vf_hi < 0.0 ? -vf_hi : 0.0) * Ar_hi * invV;
            Ub[0]  = coeff;
            Ub[5]  = coeff;
            Ub[10] = coeff;
            Ub[15] = coeff;
            // Pressure coupling to i+1
            double dh = r_center[i+1] - r_center[i];
            double dl = (i > 0) ? r_center[i] - r_center[i-1] : dh;
            double pcoeff = -(gamma - 1.0) * dl / (dh * (dl + dh));
            Ub[1*4+3] += pcoeff;
        }

        // Diagonal block
        double* Db = &D[i * 16];
        for (int q = 0; q < 16; q++) Db[q] = 0.0;
        Db[0]  = diag_val;
        Db[5]  = diag_val;
        Db[10] = diag_val;
        Db[15] = diag_val;

        // Pressure-energy coupling: dR_mr/dE (total energy)
        // dP/dE = (γ-1), same central-difference stencil as block-Jacobi
        double dP_drhoE = gamma - 1.0;
        if (i > 0 && i < nr-1) {
            double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
            double pcoeff = -(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh)));
            Db[1*4+3] = -pcoeff * dP_drhoE;
        } else if (i == 0 && nr > 1) {
            Db[1*4+3] = (gamma-1.0) / (r_center[1]-r_center[0]);
        } else if (i == nr-1 && nr >= 2) {
            Db[1*4+3] = -(gamma-1.0) / (r_center[nr-1]-r_center[nr-2]);
        }

        // Gravity-density coupling: dR_mr/d(rho) += g₀(r)
        Db[1*4+0] += gr0[i];

        // Advection self-coupling (from upwind flux at both faces)
        double coeff_lo_self = (vf_lo < 0.0 ? -vf_lo : 0.0) * Ar_lo * invV;
        double coeff_hi_self = (vf_hi >= 0.0 ? vf_hi : 0.0) * Ar_hi * invV;
        double adv_self = -(coeff_lo_self + coeff_hi_self);
        Db[0]  += adv_self;
        Db[5]  += adv_self;
        Db[10] += adv_self;
        Db[15] += adv_self;

        // Energy-momentum coupling: P·∇·v work term
        double At_hi_l = at[i*(nt+1)+j+1], At_lo_l = at[i*(nt+1)+j];
        double ddivv_dmr = invV * 0.5/rho_c * (Ar_hi + Ar_lo);
        double ddivv_dmt = invV * 0.5/rho_c * (At_hi_l + At_lo_l);
        Db[3*4+1] = -P_c * ddivv_dmr;
        Db[3*4+2] = -P_c * ddivv_dmt;

        // RHS
        rhs_s[i*4+0] = v_in[flat];
        rhs_s[i*4+1] = v_in[n + flat];
        rhs_s[i*4+2] = v_in[2*n + flat];
        rhs_s[i*4+3] = v_in[3*n + flat];
    }
    __syncthreads();

    // Block Thomas forward sweep (thread 0)
    if (tid == 0) {
        for (int i = 0; i < nr; i++) {
            double* Db = &D[i*16];
            double* rb = &rhs_s[i*4];

            if (i > 0) {
                double* Lb = &L[i*16];
                double* Dp = &D[(i-1)*16];
                double* Up = &U_b[(i-1)*16];

                // inv(D_{i-1}) via 4×4 Gauss-Jordan
                double inv_D[16];
                {
                    double A[16];
                    for(int q=0;q<16;q++) A[q]=Dp[q];
                    for(int q=0;q<16;q++) inv_D[q]=(q/4==q%4)?1.0:0.0;
                    for(int col=0;col<4;col++){
                        double mx=fabs(A[col*4+col]); int mi=col;
                        for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
                        if(mi!=col){
                            for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
                            for(int q=0;q<4;q++){double t=inv_D[col*4+q];inv_D[col*4+q]=inv_D[mi*4+q];inv_D[mi*4+q]=t;}
                        }
                        double d=A[col*4+col]; if(fabs(d)<1e-30) d=1e-30;
                        for(int row=col+1;row<4;row++){
                            double m=A[row*4+col]/d;
                            for(int q=col;q<4;q++) A[row*4+q]-=m*A[col*4+q];
                            for(int q=0;q<4;q++) inv_D[row*4+q]-=m*inv_D[col*4+q];
                        }
                    }
                    for(int c=0;c<4;c++){
                        for(int row=3;row>=0;row--){
                            double s=inv_D[row*4+c];
                            for(int q=row+1;q<4;q++) s-=A[row*4+q]*inv_D[q*4+c];
                            inv_D[row*4+c]=s/A[row*4+row];
                        }
                    }
                }

                // M = L_i * inv(D_{i-1})
                double M[16];
                for(int r=0;r<4;r++)
                    for(int c=0;c<4;c++){
                        double s=0;
                        for(int q=0;q<4;q++) s+=Lb[r*4+q]*inv_D[q*4+c];
                        M[r*4+c]=s;
                    }

                // D_i -= M * U_{i-1}
                for(int r=0;r<4;r++)
                    for(int c=0;c<4;c++){
                        double s=0;
                        for(int q=0;q<4;q++) s+=M[r*4+q]*Up[q*4+c];
                        Db[r*4+c]-=s;
                    }

                // rhs_i -= M * rhs_{i-1}
                double* rp = &rhs_s[(i-1)*4];
                for(int r=0;r<4;r++){
                    double s=0;
                    for(int q=0;q<4;q++) s+=M[r*4+q]*rp[q];
                    rb[r]-=s;
                }
            }
        }

        // Back-substitution
        for (int i = nr-1; i >= 0; i--) {
            double* Db = &D[i*16];
            double* rb = &rhs_s[i*4];
            double* xb = &x_s[i*4];

            if (i < nr-1) {
                double* Ub = &U_b[i*16];
                double* xp = &x_s[(i+1)*4];
                for(int r=0;r<4;r++){
                    double s=0;
                    for(int q=0;q<4;q++) s+=Ub[r*4+q]*xp[q];
                    rb[r]-=s;
                }
            }

            // Solve D_i * x_i = rb (4×4 Gaussian elimination)
            double A[16]; for(int q=0;q<16;q++) A[q]=Db[q];
            double b4[4]; for(int q=0;q<4;q++) b4[q]=rb[q];
            for(int col=0;col<4;col++){
                double mx=fabs(A[col*4+col]); int mi=col;
                for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
                if(mi!=col){
                    for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
                    {double t=b4[col];b4[col]=b4[mi];b4[mi]=t;}
                }
                double d=A[col*4+col]; if(fabs(d)<1e-30) d=1e-30;
                for(int row=col+1;row<4;row++){
                    double m=A[row*4+col]/d;
                    for(int q=col;q<4;q++) A[row*4+q]-=m*A[col*4+q];
                    b4[row]-=m*b4[col];
                }
            }
            for(int row=3;row>=0;row--){
                double s=b4[row];
                for(int q=row+1;q<4;q++) s-=A[row*4+q]*b4[q];
                xb[row]=s/A[row*4+row];
            }
        }
    }
    __syncthreads();

    // Write output
    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        Mv_out[flat]       = x_s[i*4+0];
        Mv_out[n + flat]   = x_s[i*4+1];
        Mv_out[2*n + flat] = x_s[i*4+2];
        Mv_out[3*n + flat] = x_s[i*4+3];
    }
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

    for (int it = 0; it < n_iters; ++it) {
        // ===== Block-Jacobi relaxation: U -= ω · J⁻¹_diag · F(U) =====
        compute_F(l, g0_over_dt);
        k_fas_smooth_blkjac<<<(n+B-1)/B,B>>>(
            lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
            lev.d_res, lev.d_blk_inv,
            lev.d_rho0, atm_rho_thresh,
            OMEGA, lev.nr, lev.nt, lev.ng);
        apply_floor(l);

        // ===== SIMPLE: pressure correction (elliptic mode) =====
        if (use_simple_smoother) {
            compute_F(l, g0_over_dt);
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
            apply_floor(l);
        }

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
