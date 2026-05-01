// LowMach preconditioners: block-Jacobi, SIMPLE, line-Jacobi, PBP, block-Schur

#include "lowmach_solver.h"
#include "lm_common.cuh"
#include "fas_common.cuh"
#include <algorithm>

// ========================= Variable scaling =============================
// Two separate scaling systems:
//
// 1. Clamp scale (MESA): max(1, |U|) — limits Newton correction magnitude
__global__
void k_lm_compute_scale(const double* Un, double* scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) scale[i] = fmax(1.0, fabs(Un[i]));
}

// 2. MUSIC L/R scaling (Viallet 2016): balances GMRES residual/unknown norms
//    R (unknowns): what "1 unit" of correction means
//    L (residuals): what "1 unit" of residual means
//    GMRES solves (L⁻¹·J·R)·x̃ = -L⁻¹·F, then δU = R·x̃
__global__
void k_lm_compute_music_scale(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    double* scR, double* scL,
    int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int n = nr*nt;
    int i = flat/nt, j = flat%nt;
    int k = (i + ng) * (nt + 2*ng) + (j + ng);

    double rho_c = fmax(rho[k], 1e-20);
    double e_c = rhoE[k] / rho_c;
    double P = fmax((gamma - 1.0) * rho_c * e_c, 1e-30);
    double cs = sqrt(gamma * P / rho_c);
    double vr = mr[k] / rho_c, vt = mt[k] / rho_c;

    double a1 = 1e-5, a2 = 1.0;

    // Right scaling (unknowns)
    scR[flat]       = fmax(rho_c, 1.0);
    scR[n + flat]   = fmax(fabs(mr[k]),  rho_c * a1 * cs);
    scR[2*n + flat] = fmax(fabs(mt[k]),  rho_c * a1 * cs);
    scR[3*n + flat] = fmax(fabs(rhoE[k]), 1.0);

    // Left scaling (residuals) — floor at 1.0 to avoid division by tiny atmosphere values
    scL[flat]       = fmax(rho_c, 1.0);
    scL[n + flat]   = fmax(rho_c * fmax(fabs(vr), a2 * cs), 1.0);
    scL[2*n + flat] = fmax(rho_c * fmax(fabs(vt), a2 * cs), 1.0);
    scL[3*n + flat] = fmax(fabs(rhoE[k]), 1.0);
}

// Elementwise v[i] *= s[i]
__global__ void k_lm_emul(double* v, const double* s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= s[i];
}
// Elementwise v[i] /= s[i]
__global__ void k_lm_ediv(double* v, const double* s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] /= s[i];
}

// Clamp correction: |delta[i]| <= max_rel * scale[i]
// ========================= Block-diagonal Jacobi preconditioner ====
// Numerically assemble the TRUE 4×4 diagonal block of J = dF/dU for
// each cell by column-wise finite differencing of the residual.
// This captures ALL intra-cell coupling: pressure-energy, gravity-density, etc.
// Then LU-invert each 4×4 block on the GPU.

// Assemble 4×4 Jacobian diagonal block analytically.
// J = dF/dU where F = R(U) - (U-Un)/dt.
// The diagonal block captures how each cell's F depends on its OWN state.
//
// Key couplings:
//   dF_mr/d(rhoE) = -dP'/dr·d(P)/d(rhoE) = -(γ-1)/Δr  (pressure-energy)
//   dF_rhoE/d(mr) = -P·d(div_v)/d(mr)                   (compression work)
//   dF_mr/d(rho)  = -ρ'·dΦ₀/dr → captured via ρ dependence
// Plus the time derivative: -1/dt on diagonal.
//
// We compute this analytically instead of numerically to avoid
// the race condition of in-place FD on global arrays.

__global__
void k_lm_assemble_blkjac(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* gr0,
    double* blk, double* blk_J,
    int nr, int nt, int ng, double gamma, double inv_dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double e_c = rhoE[k] / rho_c;
    double P_c = fmax((gamma-1.0)*rho_c*e_c, 1e-30);
    double r = r_center[i];
    double invV = 1.0 / vol[flat];

    // Advection rate (upwind diagonal contribution)
    double sr_r = fabs(vr_c) / dr[i];
    double sr_t = fabs(vt_c) / (r * dtheta[j]);
    double sr = sr_r + sr_t;

    // Exact central-diff self-coefficient: b = (dh-dl)/(dl*dh)
    double dPdr_self = 0.0;
    if (i > 0 && i < nr-1) {
        double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
        dPdr_self = (dh - dl) / (dl * dh);
    } else if (i == 0 && nr > 1) {
        dPdr_self = -1.0/(r_center[1]-r_center[0]);
    } else if (i == nr-1 && nr >= 2) {
        dPdr_self = 1.0/(r_center[nr-1]-r_center[nr-2]);
    }

    double dPdt_self = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        dPdt_self = (dh - dl) / (dl * dh * r);
    }

    double dP_drhoE = gamma - 1.0;
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

    J[1*4+3] = dPdr_self * dP_drhoE;
    J[2*4+3] = dPdt_self * dP_drhoE;

    // 1D gravity-density coupling: dF_mr/d(rho) += g₀(r) (from +ρ'·g₀ term)
    J[1*4+0] += g0_val;   // dF_mr/d(rho)
    // No θ gravity component with 1D gravity

    // Compression work coupling: dF_rhoE/d(mr) = -P * d(div_v)/d(mr)
    J[3*4+1] = -P_c * ddivv_dmr;  // dF_rhoE/d(mr)
    J[3*4+2] = -P_c * ddivv_dmt;  // dF_rhoE/d(mt)

    // 4×4 LU with partial pivoting → invert
    double A[16]; for(int q=0;q<16;q++) A[q]=J[q];
    double inv_m[16]; for(int q=0;q<16;q++) inv_m[q]=(q/4==q%4)?1.0:0.0;

    for(int col=0;col<4;col++){
        double mx=fabs(A[col*4+col]); int mi=col;
        for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
        if(mi!=col){
            for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
            for(int q=0;q<4;q++){double t=inv_m[col*4+q];inv_m[col*4+q]=inv_m[mi*4+q];inv_m[mi*4+q]=t;}
        }
        double d=A[col*4+col];
        if(fabs(d)<1e-30) d=(d>=0?1e-30:-1e-30);
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

    double* BJ = &blk_J[flat*16];
    for(int q=0;q<16;q++) BJ[q]=J[q];

    double* B = &blk[flat*16];
    for(int q=0;q<16;q++) B[q]=inv_m[q];
}

// Apply block-diagonal preconditioner: Mv[cell] = B[cell] * v[cell]
__global__
void k_lm_apply_blkjac(const double* blk, const double* v, double* Mv,
                        int n) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n) return;
    const double* B = &blk[cell * 16];
    double v0 = v[cell], v1 = v[n+cell], v2 = v[2*n+cell], v3 = v[3*n+cell];
    Mv[cell]     = B[0]*v0 + B[1]*v1 + B[2]*v2 + B[3]*v3;
    Mv[n+cell]   = B[4]*v0 + B[5]*v1 + B[6]*v2 + B[7]*v3;
    Mv[2*n+cell] = B[8]*v0 + B[9]*v1 + B[10]*v2 + B[11]*v3;
    Mv[3*n+cell] = B[12]*v0 + B[13]*v1 + B[14]*v2 + B[15]*v3;
}

void LowMachSolver::assemble_block_jacobi(double dt) {
    int n = nr*nt, B = 256;
    k_lm_assemble_blkjac<<<(n+B-1)/B,B>>>(
        d_rho, d_mr, d_mtheta, d_rhoE,
        d_cell_volume, d_area_r, d_area_theta,
        d_r_center, d_r_face, d_theta_face,
        d_dr, d_dtheta,
        d_gr0,
        d_blk_diag, d_blk_J, nr, nt, ng, gamma, 1.0/dt);
}

// ========================= SIMPLE preconditioner ===================
// Given residual vector r = (r_ρ, r_mr, r_mt, r_ρe):
//
// Step 1: Approximate momentum solve (diagonal):
//   δvr* = r_mr / Ap,  δvt* = r_mt / Ap
//   where Ap = 1/dt + upwind_rate (per cell)
//
// Step 2: Pressure Poisson (GMG):
//   ∇·(dt/ρ · ∇δp) = ∇·(δv*)    [constant-coeff approx: dt/ρ ≈ dt/ρ₀]
//   Actually use: ∇²δp = (ρ/dt) · ∇·(δv*)  with same Laplacian as gravity
//
// Step 3: Velocity correction:
//   δvr = δvr* - (dt/ρ) ∂δp/∂r
//   δvt = δvt* - (dt/ρ) (1/r)∂δp/∂θ
//
// Step 4: Pass through density and energy:
//   δρ  = r_ρ / (1/dt)  = dt * r_ρ
//   δρe = r_ρe / (1/dt) = dt * r_ρe

__global__
void k_lm_simple_mom_diag(const double* rho, const double* mr, const double* mt,
                          const double* rhoE,
                          const double* dr, const double* rc, const double* dtheta,
                          double* Ap, int nr, int nt, int ng,
                          double inv_dt, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);
    double rho_c = fmax(rho[k], 1e-20);
    double vr = fabs(mr[k]/rho_c), vt = fabs(mt[k]/rho_c);
    double P = fmax((gamma - 1.0) * rhoE[k], 1e-30);
    double cs = sqrt(gamma * P / rho_c);
    Ap[flat] = inv_dt + (vr + cs)/dr[i] + (vt + cs)/(rc[i]*dtheta[j]);
}

// Step 1: δv* = -r_momentum / Ap  (negative because J_diag = -Ap)
__global__
void k_lm_simple_vstar(const double* r_mr, const double* r_mt,
                       const double* Ap, double* vr_s, double* vt_s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double inv_ap = -1.0 / Ap[i];
    vr_s[i] = r_mr[i] * inv_ap;
    vt_s[i] = r_mt[i] * inv_ap;
}

// Step 2: compute divergence of δv* using FV divergence theorem
// div = (1/V) Σ(A_face · v_face), where v_face = average of neighbors
__global__
void k_lm_simple_div(const double* vr_s, const double* vt_s,
                     const double* vol, const double* ar, const double* at,
                     double* div_out, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    double invV = 1.0 / vol[flat];

    // Radial face velocities (average of neighbors, zero at boundaries)
    double vr_lo = (i > 0)    ? 0.5*(vr_s[(i-1)*nt+j] + vr_s[i*nt+j]) : 0.0;
    double vr_hi = (i < nr-1) ? 0.5*(vr_s[i*nt+j] + vr_s[(i+1)*nt+j]) : 0.0;

    // Theta face velocities
    double vt_lo = (j > 0)    ? 0.5*(vt_s[i*nt+j-1] + vt_s[i*nt+j]) : 0.0;
    double vt_hi = (j < nt-1) ? 0.5*(vt_s[i*nt+j] + vt_s[i*nt+j+1]) : 0.0;

    double Ar_hi = ar[(i+1)*nt+j], Ar_lo = ar[i*nt+j];
    double At_hi = at[i*(nt+1)+j+1], At_lo = at[i*(nt+1)+j];

    div_out[flat] = invV*(Ar_hi*vr_hi - Ar_lo*vr_lo + At_hi*vt_hi - At_lo*vt_lo);
}

// Step 2b: compute α = 1/Ap for variable-coefficient Poisson
__global__
void k_lm_simple_alpha(const double* Ap, double* alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) alpha[i] = 1.0 / Ap[i];
}

// Poisson RHS = div(v*), with Dirichlet δp=0 at outer boundary
__global__
void k_lm_simple_prhs(const double* div_v, double* rhs, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    rhs[flat] = div_v[flat] * (double)(flat/nt < nr-1);
}

// Step 3: correct velocity and assemble output
// out_ρ  = -dt * r_ρ
// out_mr = δvr* - (1/Ap) * ∂δp/∂r
// out_mt = δvt* - (1/Ap) * (1/r)∂δp/∂θ
// out_ρe = -dt * r_ρe
__global__
void k_lm_simple_correct(
    const double* r_rho, const double* r_mr, const double* r_mt, const double* r_rhoE,
    const double* Ap,
    const double* vr_s, const double* vt_s,
    const double* dp,
    const double* rc, const double* theta_face,
    double* out,
    int nr, int nt, double dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;

    // Pressure gradient of correction δp
    double dp_dr = 0.0;
    if (i > 0 && i < nr-1) {
        double dl = rc[i]-rc[i-1], dh = rc[i+1]-rc[i];
        double gl = (dp[flat]-dp[(i-1)*nt+j])/dl;
        double gr = (dp[(i+1)*nt+j]-dp[flat])/dh;
        dp_dr = (dh*gl+dl*gr)/(dl+dh);
    } else if (i == 0 && nr > 1)
        dp_dr = (dp[1*nt+j]-dp[0])/( rc[1]-rc[0]);
    else if (i == nr-1 && nr >= 2)
        dp_dr = (dp[(nr-1)*nt+j]-dp[(nr-2)*nt+j])/(rc[nr-1]-rc[nr-2]);

    double dp_dt_r = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m = 0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c = 0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p = 0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl = tc_c-tc_m, dh = tc_p-tc_c;
        dp_dt_r = ((dh*(dp[flat]-dp[i*nt+j-1])/dl + dl*(dp[i*nt+j+1]-dp[flat])/dh))
                  / (rc[i]*(dl+dh));
    }

    double inv_ap = 1.0 / Ap[flat];
    out[flat]       = -dt * r_rho[flat];
    out[n + flat]   = vr_s[flat] - inv_ap * dp_dr;
    out[2*n + flat] = vt_s[flat] - inv_ap * dp_dt_r;
    out[3*n + flat] = -dt * r_rhoE[flat];
}

// ========================= Line-Jacobi preconditioner ==============
// For each θ-line (fixed j), solve the block-tridiagonal system along r.
// Each cell has a 4×4 diagonal block (same as block Jacobi) plus
// 4×4 off-diagonal blocks from the radial upwind + pressure stencil.
// One GPU block per θ-line; sequential Thomas algorithm within the block.

__global__
void k_lm_line_solve(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* gr0,
    const double* v_in, double* Mv_out,
    int nr, int nt, int ng, double gamma, double inv_dt) {
    int j = blockIdx.x;  // one block per θ-line
    if (j >= nt) return;

    // Shared memory for the Thomas algorithm: L[nr], D[nr], U[nr], rhs[nr]
    // Each is a 4×4 matrix stored as 16 doubles (row-major).
    extern __shared__ double smem[];
    double* L = smem;                      // nr * 16
    double* D = L + nr * 16;              // nr * 16
    double* U = D + nr * 16;              // nr * 16
    double* rhs_s = U + nr * 16;          // nr * 4
    double* x_s = rhs_s + nr * 4;         // nr * 4

    int tid = threadIdx.x;
    int n = nr * nt;

    // Step 1: Each thread assembles its row's L, D, U blocks
    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        int k = d_idx(i, j, nt, ng);

        double rho_c = fmax(rho[k], 1e-20);
        double vr_c = mr[k] / rho_c;
        double vt_c = mt[k] / rho_c;
        double P_c = fmax((gamma - 1.0) * rhoE[k], 1e-30);
        double r = r_center[i];
        double invV = 1.0 / vol[flat];

        double cs = sqrt(gamma * P_c / rho_c);
        double Ar_hi = ar[(i+1)*nt+j], Ar_lo = ar[i*nt+j];

        // Transverse (θ) spectral radius only; r-direction is captured
        // structurally by L/D/U off-diagonals + adv_self
        double sr_transverse = (fabs(vt_c)+cs)/(r*dtheta[j]);
        double diag_val = -(inv_dt + sr_transverse);

        double vf_lo = 0.0, vf_hi = 0.0;
        if (i > 0) {
            double rl = fmax(rho[d_idx(i-1,j,nt,ng)], 1e-20);
            vf_lo = 0.5 * (mr[d_idx(i-1,j,nt,ng)] / rl + vr_c);
        }
        if (i < nr - 1) {
            double rr = fmax(rho[d_idx(i+1,j,nt,ng)], 1e-20);
            vf_hi = 0.5 * (vr_c + mr[d_idx(i+1,j,nt,ng)] / rr);
        }

        // Lower block (coupling to i-1): from upwind flux at face (i-1,i)
        double* Lb = &L[i * 16];
        for (int q = 0; q < 16; q++) Lb[q] = 0.0;
        if (i > 0) {
            double coeff = (vf_lo >= 0.0 ? vf_lo : 0.0) * Ar_lo * invV;
            // Advection: dR_c/dU_{i-1} contributes +coeff to diagonal entries
            // (upwind from left means flux carries q_{i-1})
            Lb[0] = coeff;   // dR_rho/d(rho_{i-1})
            Lb[5] = coeff;   // dR_mr/d(mr_{i-1})
            Lb[10] = coeff;  // dR_mt/d(mt_{i-1})
            Lb[15] = coeff;  // dR_rhoE/d(rhoE_{i-1})
        }

        // Upper block (coupling to i+1)
        double* Ub = &U[i * 16];
        for (int q = 0; q < 16; q++) Ub[q] = 0.0;
        if (i < nr - 1) {
            double coeff = (vf_hi < 0.0 ? -vf_hi : 0.0) * Ar_hi * invV;
            Ub[0] = coeff;
            Ub[5] = coeff;
            Ub[10] = coeff;
            Ub[15] = coeff;

            // Pressure coupling to i+1: central-diff stencil coeff c = dl/(dh*(dl+dh))
            // dR_mr/dP_{i+1} = -c  (gradient enters with minus sign)
            // dF_mr/dP_{i+1} = +c  (F = -R)
            // dF_mr/d(rhoE_{i+1}) = (γ-1) * dl/(dh*(dl+dh))
            double dh = r_center[i+1] - r_center[i];
            double dl = (i > 0) ? r_center[i] - r_center[i-1] : dh;
            Ub[1*4+3] += (gamma - 1.0) * dl / (dh * (dl + dh));
        }

        // Diagonal block: full 4×4 from block Jacobi + corrections
        double* Db = &D[i * 16];
        // Start from the block Jacobi diagonal already computed in d_blk_diag
        // BUT d_blk_diag stores J⁻¹, not J. We need J itself here.
        // Recompute J diagonal inline (same as k_lm_assemble_blkjac but not inverted).

        for (int q = 0; q < 16; q++) Db[q] = 0.0;
        Db[0] = diag_val;   // dF_rho/d(rho)
        Db[5] = diag_val;   // dF_mr/d(mr)
        Db[10] = diag_val;  // dF_mt/d(mt)
        Db[15] = diag_val;  // dF_rhoE/d(rhoE)

        // Pressure-energy coupling: dF_mr/d(rhoE_i)
        // Central-diff self-coefficient b = (dh-dl)/(dl*dh)
        // dF_mr/dP_i = -(-b) = b, so dF_mr/d(rhoE_i) = (γ-1)*(dh-dl)/(dl*dh)
        if (i > 0 && i < nr-1) {
            double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
            Db[1*4+3] = (gamma-1.0) * (dh - dl) / (dl * dh);
        } else if (i == 0 && nr > 1) {
            Db[1*4+3] = (gamma-1.0) / (r_center[1]-r_center[0]);
        } else if (i == nr-1 && nr >= 2) {
            Db[1*4+3] = -(gamma-1.0) / (r_center[nr-1]-r_center[nr-2]);
        }

        // 1D gravity-density coupling: dF_mr/d(rho) += g₀(r)
        Db[1*4+0] += gr0[i];

        // Lower block pressure coupling to i-1:
        // Central-diff stencil coeff a = -dh/(dl*(dl+dh))
        // dF_mr/dP_{i-1} = -(-a) = a (negative)
        // dF_mr/d(rhoE_{i-1}) = -(γ-1)*dh/(dl*(dl+dh))
        if (i > 0) {
            double dl = r_center[i] - r_center[i-1];
            double dh = (i < nr-1) ? r_center[i+1] - r_center[i] : dl;
            Lb[1*4+3] += -(gamma - 1.0) * dh / (dl * (dl + dh));
        }

        // Advection correction to diagonal from self-flux
        double coeff_lo_self = (vf_lo < 0.0 ? -vf_lo : 0.0) * Ar_lo * invV;
        double coeff_hi_self = (vf_hi >= 0.0 ? vf_hi : 0.0) * Ar_hi * invV;
        double adv_self = -(coeff_lo_self + coeff_hi_self);
        Db[0] += adv_self;
        Db[5] += adv_self;
        Db[10] += adv_self;
        Db[15] += adv_self;

        // Compression work coupling: dF_rhoE/d(mr), dF_rhoE/d(mt)
        double ddivv_dmr = invV * 0.5/rho_c * (Ar_hi + Ar_lo);
        double At_hi_l = at[i*(nt+1)+j+1], At_lo_l = at[i*(nt+1)+j];
        double ddivv_dmt = invV * 0.5/rho_c * (At_hi_l + At_lo_l);
        Db[3*4+1] = -P_c * ddivv_dmr;
        Db[3*4+2] = -P_c * ddivv_dmt;

        // RHS: input vector
        rhs_s[i*4+0] = v_in[flat];
        rhs_s[i*4+1] = v_in[n + flat];
        rhs_s[i*4+2] = v_in[2*n + flat];
        rhs_s[i*4+3] = v_in[3*n + flat];
    }
    __syncthreads();

    // Step 2: Block Thomas forward sweep (sequential, thread 0 only)
    if (tid == 0) {
        // Forward elimination
        for (int i = 0; i < nr; i++) {
            double* Db = &D[i*16];
            double* rb = &rhs_s[i*4];

            if (i > 0) {
                double* Lb = &L[i*16];
                double* Dp = &D[(i-1)*16]; // D_{i-1} after elimination = upper triangular

                // Compute multiplier M = L_i * D_{i-1}^{-1}
                double inv_D[16];
                mat4_invert(Dp, inv_D);
                double M[16];
                mat4_mul(Lb, inv_D, M);
                double MU[16];
                mat4_mul(M, &U[(i-1)*16], MU);
                for (int q = 0; q < 16; q++) Db[q] -= MU[q];
                double* rp = &rhs_s[(i-1)*4];
                for (int r = 0; r < 4; r++) {
                    double s = 0;
                    for (int q = 0; q < 4; q++) s += M[r*4+q] * rp[q];
                    rb[r] -= s;
                }
            }
        }
        for (int i = nr-1; i >= 0; i--) {
            double* rb = &rhs_s[i*4];
            double* xb = &x_s[i*4];
            if (i < nr-1) {
                double* Ub = &U[i*16];
                double* xp = &x_s[(i+1)*4];
                for (int r = 0; r < 4; r++) {
                    double s = 0;
                    for (int q = 0; q < 4; q++) s += Ub[r*4+q] * xp[q];
                    rb[r] -= s;
                }
            }
            for (int q = 0; q < 4; q++) xb[q] = rb[q];
            mat4_solve(&D[i*16], xb);
        }
    }
    __syncthreads();

    // Step 3: Write output
    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        Mv_out[flat]       = x_s[i*4+0];
        Mv_out[n + flat]   = x_s[i*4+1];
        Mv_out[2*n + flat] = x_s[i*4+2];
        Mv_out[3*n + flat] = x_s[i*4+3];
    }
}

// θ-direction block-tridiagonal line solve: one block per radial line.
// Mirror of k_lm_line_solve but sweeps along θ at fixed r.
__global__
void k_lm_line_solve_theta(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* gr0,
    const double* v_in, double* Mv_out,
    int nr, int nt, int ng, double gamma, double inv_dt) {
    int i = blockIdx.x;  // one block per r-line
    if (i >= nr) return;

    extern __shared__ double smem[];
    double* L = smem;                 // nt * 16
    double* D = L + nt * 16;         // nt * 16
    double* U = D + nt * 16;         // nt * 16
    double* rhs_s = U + nt * 16;     // nt * 4
    double* x_s = rhs_s + nt * 4;    // nt * 4

    int tid = threadIdx.x;
    int n = nr * nt;
    double r = r_center[i];

    for (int j = tid; j < nt; j += blockDim.x) {
        int flat = i * nt + j;
        int k = d_idx(i, j, nt, ng);

        double rho_c = fmax(rho[k], 1e-20);
        double vr_c = mr[k] / rho_c;
        double vt_c = mt[k] / rho_c;
        double P_c = fmax((gamma - 1.0) * rhoE[k], 1e-30);
        double invV = 1.0 / vol[flat];
        double cs = sqrt(gamma * P_c / rho_c);

        double At_hi = at[i*(nt+1)+j+1], At_lo = at[i*(nt+1)+j];

        // Transverse (r) spectral radius only; θ-direction captured by L/D/U + adv_self
        double sr_transverse = (fabs(vr_c)+cs)/dr[i];
        double diag_val = -(inv_dt + sr_transverse);

        // θ-direction face velocities for upwind
        double vf_lo = 0.0, vf_hi = 0.0;
        if (j > 0) {
            double rl = fmax(rho[d_idx(i,j-1,nt,ng)], 1e-20);
            vf_lo = 0.5 * (mt[d_idx(i,j-1,nt,ng)] / rl + vt_c);
        }
        if (j < nt - 1) {
            double rr = fmax(rho[d_idx(i,j+1,nt,ng)], 1e-20);
            vf_hi = 0.5 * (vt_c + mt[d_idx(i,j+1,nt,ng)] / rr);
        }

        // Lower block (j-1)
        double* Lb = &L[j * 16];
        for (int q = 0; q < 16; q++) Lb[q] = 0.0;
        if (j > 0) {
            double coeff = (vf_lo >= 0.0 ? vf_lo : 0.0) * At_lo * invV;
            Lb[0] = coeff; Lb[5] = coeff; Lb[10] = coeff; Lb[15] = coeff;

            // Pressure coupling: dF_mt/d(rhoE_{j-1}) from (1/r)∂P'/∂θ
            double tc_m = 0.5*(theta_face[j-1]+theta_face[j]);
            double tc_c = 0.5*(theta_face[j]+theta_face[j+1]);
            double dl = tc_c - tc_m;
            double dh = (j < nt-1) ? 0.5*(theta_face[j+1]+theta_face[j+2]) - tc_c : dl;
            double pcoeff = -(gamma - 1.0) * dh / (r * dl * (dl + dh));
            Lb[2*4+3] += pcoeff;
        }

        // Upper block (j+1)
        double* Ub = &U[j * 16];
        for (int q = 0; q < 16; q++) Ub[q] = 0.0;
        if (j < nt - 1) {
            double coeff = (vf_hi < 0.0 ? -vf_hi : 0.0) * At_hi * invV;
            Ub[0] = coeff; Ub[5] = coeff; Ub[10] = coeff; Ub[15] = coeff;

            double tc_c = 0.5*(theta_face[j]+theta_face[j+1]);
            double tc_p = 0.5*(theta_face[j+1]+theta_face[j+2]);
            double dh = tc_p - tc_c;
            double dl = (j > 0) ? tc_c - 0.5*(theta_face[j-1]+theta_face[j]) : dh;
            double pcoeff = -(gamma - 1.0) * dl / (r * dh * (dl + dh));
            Ub[2*4+3] += pcoeff;
        }

        // Diagonal block
        double* Db = &D[j * 16];
        for (int q = 0; q < 16; q++) Db[q] = 0.0;
        Db[0] = diag_val; Db[5] = diag_val; Db[10] = diag_val; Db[15] = diag_val;

        // θ pressure gradient diagonal: dF_mt/d(rhoE_j)
        // Self-coefficient b = (dh-dl)/(dl*dh), divided by r for θ-gradient
        // dF_mt/d(rhoE_j) = (γ-1)*(dh-dl)/(dl*dh*r)
        if (j > 0 && j < nt-1) {
            double tc_m = 0.5*(theta_face[j-1]+theta_face[j]);
            double tc_c = 0.5*(theta_face[j]+theta_face[j+1]);
            double tc_p = 0.5*(theta_face[j+1]+theta_face[j+2]);
            double dl = tc_c-tc_m, dh = tc_p-tc_c;
            Db[2*4+3] = (gamma-1.0) * (dh - dl) / (dl * dh * r);
        }

        // Advection self-coupling
        double coeff_lo_self = (vf_lo < 0.0 ? -vf_lo : 0.0) * At_lo * invV;
        double coeff_hi_self = (vf_hi >= 0.0 ? vf_hi : 0.0) * At_hi * invV;
        double adv_self = -(coeff_lo_self + coeff_hi_self);
        Db[0] += adv_self; Db[5] += adv_self; Db[10] += adv_self; Db[15] += adv_self;

        // 1D gravity-density coupling: dF_mr/d(rho) += g₀(r)
        Db[1*4+0] += gr0[i];

        // r-direction pressure gradient diagonal (same as r-line)
        double dPdr_coeff = 0.0;
        if (i > 0 && i < nr-1) {
            double dl_r = r_center[i]-r_center[i-1], dh_r = r_center[i+1]-r_center[i];
            dPdr_coeff = -(dh_r/(dl_r*(dl_r+dh_r)) + dl_r/(dh_r*(dl_r+dh_r)));
        } else if (i == 0 && nr > 1) {
            dPdr_coeff = -1.0/(r_center[1]-r_center[0]);
        } else if (i == nr-1 && nr >= 2) {
            dPdr_coeff = 1.0/(r_center[nr-1]-r_center[nr-2]);
        }
        Db[1*4+3] += -dPdr_coeff * (gamma - 1.0);

        // Compression work coupling
        double Ar_hi_l = ar[(i+1)*nt+j], Ar_lo_l = ar[i*nt+j];
        double ddivv_dmr = invV * 0.5/rho_c * (Ar_hi_l + Ar_lo_l);
        double ddivv_dmt = invV * 0.5/rho_c * (At_hi + At_lo);
        Db[3*4+1] = -P_c * ddivv_dmr;
        Db[3*4+2] = -P_c * ddivv_dmt;

        rhs_s[j*4+0] = v_in[flat];
        rhs_s[j*4+1] = v_in[n + flat];
        rhs_s[j*4+2] = v_in[2*n + flat];
        rhs_s[j*4+3] = v_in[3*n + flat];
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < nt; j++) {
            double* Db = &D[j*16];
            double* rb = &rhs_s[j*4];
            if (j > 0) {
                double inv_D[16];
                mat4_invert(&D[(j-1)*16], inv_D);
                double M[16];
                mat4_mul(&L[j*16], inv_D, M);
                double MU[16];
                mat4_mul(M, &U[(j-1)*16], MU);
                for (int q = 0; q < 16; q++) Db[q] -= MU[q];
                double* rp = &rhs_s[(j-1)*4];
                for (int r = 0; r < 4; r++) {
                    double s = 0;
                    for (int q = 0; q < 4; q++) s += M[r*4+q] * rp[q];
                    rb[r] -= s;
                }
            }
        }
        for (int j = nt-1; j >= 0; j--) {
            double* rb = &rhs_s[j*4];
            double* xb = &x_s[j*4];
            if (j < nt-1) {
                double* Ub = &U[j*16];
                double* xp = &x_s[(j+1)*4];
                for (int r = 0; r < 4; r++) {
                    double s = 0;
                    for (int q = 0; q < 4; q++) s += Ub[r*4+q] * xp[q];
                    rb[r] -= s;
                }
            }
            for (int q = 0; q < 4; q++) xb[q] = rb[q];
            mat4_solve(&D[j*16], xb);
        }
    }
    __syncthreads();

    for (int j = tid; j < nt; j += blockDim.x) {
        int flat = i * nt + j;
        Mv_out[flat]       = x_s[j*4+0];
        Mv_out[n + flat]   = x_s[j*4+1];
        Mv_out[2*n + flat] = x_s[j*4+2];
        Mv_out[3*n + flat] = x_s[j*4+3];
    }
}

// ========================= 2-DOF momentum-only r-line solve ===========
// Solves only the (mr, mt) block along r, WITHOUT pressure coupling.
// 2×2 blocks: diagonal = -(1/dt + sr), off-diagonal = upwind advection.
// Pressure gradient is excluded — it enters through the Schur complement.
// One GPU block per θ-line.
__global__
void k_lm_mom_line_solve(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar,
    const double* r_center, const double* dr, const double* dtheta,
    const double* v_in_mr, const double* v_in_mt,
    double* out_vr, double* out_vt,
    int nr, int nt, int ng, double inv_dt, double gamma) {
    int j = blockIdx.x;
    if (j >= nt) return;

    extern __shared__ double smem[];
    // 2×2 blocks: L[nr*4], D[nr*4], U[nr*4], rhs[nr*2], x[nr*2]
    double* L = smem;
    double* D = L + nr * 4;
    double* U = D + nr * 4;
    double* rhs_s = U + nr * 4;
    double* x_s = rhs_s + nr * 2;

    int tid = threadIdx.x;
    int n = nr * nt;

    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        int k = d_idx(i, j, nt, ng);
        double rho_c = fmax(rho[k], 1e-20);
        double vr_c = mr[k] / rho_c;
        double vt_c = mt[k] / rho_c;
        double r = r_center[i];
        double invV = 1.0 / vol[flat];

        double P = fmax((gamma - 1.0) * rhoE[k], 1e-30);
        double cs = sqrt(gamma * P / rho_c);
        double sr = (fabs(vr_c)+cs)/dr[i] + (fabs(vt_c)+cs)/(r*dtheta[j]);
        double Ar_hi = ar[(i+1)*nt+j], Ar_lo = ar[i*nt+j];

        double vf_lo = 0.0, vf_hi = 0.0;
        if (i > 0) {
            double rl = fmax(rho[d_idx(i-1,j,nt,ng)], 1e-20);
            vf_lo = 0.5 * (mr[d_idx(i-1,j,nt,ng)] / rl + vr_c);
        }
        if (i < nr - 1) {
            double rr = fmax(rho[d_idx(i+1,j,nt,ng)], 1e-20);
            vf_hi = 0.5 * (vr_c + mr[d_idx(i+1,j,nt,ng)] / rr);
        }

        // Lower 2×2
        double* Lb = &L[i * 4];
        Lb[0] = Lb[1] = Lb[2] = Lb[3] = 0.0;
        if (i > 0) {
            double coeff = (vf_lo >= 0.0 ? vf_lo : 0.0) * Ar_lo * invV;
            Lb[0] = coeff; Lb[3] = coeff;
        }

        // Upper 2×2
        double* Ub = &U[i * 4];
        Ub[0] = Ub[1] = Ub[2] = Ub[3] = 0.0;
        if (i < nr - 1) {
            double coeff = (vf_hi < 0.0 ? -vf_hi : 0.0) * Ar_hi * invV;
            Ub[0] = coeff; Ub[3] = coeff;
        }

        // Diagonal 2×2
        double* Db = &D[i * 4];
        double diag_val = -(inv_dt + sr);
        double coeff_lo_self = (vf_lo < 0.0 ? -vf_lo : 0.0) * Ar_lo * invV;
        double coeff_hi_self = (vf_hi >= 0.0 ? vf_hi : 0.0) * Ar_hi * invV;
        double adv_self = -(coeff_lo_self + coeff_hi_self);
        Db[0] = diag_val + adv_self;
        Db[1] = 0.0;
        Db[2] = 0.0;
        Db[3] = diag_val + adv_self;

        // RHS: input residual for mr, mt
        rhs_s[i*2+0] = v_in_mr[flat];
        rhs_s[i*2+1] = v_in_mt[flat];
    }
    __syncthreads();

    // 2×2 block Thomas (thread 0)
    if (tid == 0) {
        for (int i = 0; i < nr; i++) {
            double* Db = &D[i*4];
            double* rb = &rhs_s[i*2];
            if (i > 0) {
                double* Lb = &L[i*4];
                double* Dp = &D[(i-1)*4];
                double* Up = &U[(i-1)*4];
                // inv(D_{i-1}) for 2×2
                double det = Dp[0]*Dp[3] - Dp[1]*Dp[2];
                if (fabs(det) < 1e-30) det = 1e-30;
                double id = 1.0/det;
                double inv_D[4] = {Dp[3]*id, -Dp[1]*id, -Dp[2]*id, Dp[0]*id};
                // M = L * inv(D)
                double M[4];
                M[0] = Lb[0]*inv_D[0] + Lb[1]*inv_D[2];
                M[1] = Lb[0]*inv_D[1] + Lb[1]*inv_D[3];
                M[2] = Lb[2]*inv_D[0] + Lb[3]*inv_D[2];
                M[3] = Lb[2]*inv_D[1] + Lb[3]*inv_D[3];
                // D -= M * U
                Db[0] -= M[0]*Up[0] + M[1]*Up[2];
                Db[1] -= M[0]*Up[1] + M[1]*Up[3];
                Db[2] -= M[2]*Up[0] + M[3]*Up[2];
                Db[3] -= M[2]*Up[1] + M[3]*Up[3];
                // rhs -= M * rhs_{i-1}
                double* rp = &rhs_s[(i-1)*2];
                rb[0] -= M[0]*rp[0] + M[1]*rp[1];
                rb[1] -= M[2]*rp[0] + M[3]*rp[1];
            }
        }
        for (int i = nr-1; i >= 0; i--) {
            double* Db = &D[i*4];
            double* rb = &rhs_s[i*2];
            double* xb = &x_s[i*2];
            if (i < nr-1) {
                double* Ub = &U[i*4];
                double* xp = &x_s[(i+1)*2];
                rb[0] -= Ub[0]*xp[0] + Ub[1]*xp[1];
                rb[1] -= Ub[2]*xp[0] + Ub[3]*xp[1];
            }
            double det = Db[0]*Db[3] - Db[1]*Db[2];
            if (fabs(det) < 1e-30) det = 1e-30;
            double id = 1.0/det;
            xb[0] = (Db[3]*rb[0] - Db[1]*rb[1]) * id;
            xb[1] = (Db[0]*rb[1] - Db[2]*rb[0]) * id;
        }
    }
    __syncthreads();

    // Output: predicted velocity δvr, δvt (divide momentum correction by ρ)
    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        int k = d_idx(i, j, nt, ng);
        double rho_c = fmax(rho[k], 1e-20);
        out_vr[flat] = x_s[i*2+0] / rho_c;
        out_vt[flat] = x_s[i*2+1] / rho_c;
    }
}

// ========================= PBP assembly kernel =======================
// Assemble full 4-DOF output from PBP components.
// Momentum: corrected velocity from line solve + pressure Poisson.
// ρ: point Jacobi (no off-diagonal coupling to momentum in row 0).
// Energy: forward substitution using predicted momentum:
//   δE = (r_E - J_Em_r·δmr - J_Em_t·δmt) / J_EE
// This makes the energy output consistent with the velocity prediction,
// eliminating the main source of GMRES iterations at large dt.
__global__
void k_lm_pbp_assemble(
    const double* rho_state, const double* blk_diag, const double* blk_J,
    const double* v_in,      // original 4n RHS
    const double* vr_pred, const double* vt_pred,  // predicted velocities
    const double* dp, const double* Ap,
    const double* grad_r_wm, const double* grad_r_wp,
    const double* grad_t_wm, const double* grad_t_wp,
    double* Mv_out,
    int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int n = nr*nt;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i, j, nt, ng);
    double rho_c = fmax(rho_state[k], 1e-20);

    // ρ: full block-inverse row 0
    const double* Binv = &blk_diag[flat*16];
    double r0 = v_in[flat], r1 = v_in[n+flat], r2 = v_in[2*n+flat], r3 = v_in[3*n+flat];
    Mv_out[flat] = Binv[0]*r0 + Binv[1]*r1 + Binv[2]*r2 + Binv[3]*r3;

    // Branchless pressure gradient using precomputed stencil weights
    int im = flat - nt * (int)(i > 0);
    int ip = flat + nt * (int)(i < nr-1);
    double dp_dr = grad_r_wm[flat]*(dp[im] - dp[flat]) + grad_r_wp[flat]*(dp[ip] - dp[flat]);

    int jm = flat - (int)(j > 0);
    int jp = flat + (int)(j < nt-1);
    double dp_dt_r = grad_t_wm[flat]*(dp[jm] - dp[flat]) + grad_t_wp[flat]*(dp[jp] - dp[flat]);

    double inv_ap = 1.0 / Ap[flat];
    double dvr = vr_pred[flat] - inv_ap * dp_dr;
    double dvt = vt_pred[flat] - inv_ap * dp_dt_r;

    // Momentum output = ρ · corrected velocity
    double dmr = rho_c * dvr;
    double dmt = rho_c * dvt;
    Mv_out[n + flat]   = dmr;
    Mv_out[2*n + flat] = dmt;

    // Energy: full block-inverse row 3
    Mv_out[3*n + flat] = Binv[12]*r0 + Binv[13]*r1 + Binv[14]*r2 + Binv[15]*r3;
}

// Velocity-only pressure correction: vr -= (1/Ap)*∂δp/∂r, vt -= (1/Ap)*(1/r)*∂δp/∂θ
// Branchless: uses precomputed gradient stencil weights.
__global__
void k_lm_simple_vcorr(double* vr_io, double* vt_io,
                       const double* dp, const double* Ap,
                       const double* grad_r_wm, const double* grad_r_wp,
                       const double* grad_t_wm, const double* grad_t_wp,
                       int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;

    int im = flat - nt * (int)(i > 0);
    int ip = flat + nt * (int)(i < nr-1);
    double dp_dr = grad_r_wm[flat]*(dp[im] - dp[flat]) + grad_r_wp[flat]*(dp[ip] - dp[flat]);

    int jm = flat - (int)(j > 0);
    int jp = flat + (int)(j < nt-1);
    double dp_dt_r = grad_t_wm[flat]*(dp[jm] - dp[flat]) + grad_t_wp[flat]*(dp[jp] - dp[flat]);

    double inv_ap = 1.0 / Ap[flat];
    vr_io[flat] -= inv_ap * dp_dr;
    vt_io[flat] -= inv_ap * dp_dt_r;
}

void LowMachSolver::assemble_simple(double dt) {
    int n = nr*nt, B = 256;
    k_lm_simple_mom_diag<<<(n+B-1)/B,B>>>(
        d_rho, d_mr, d_mtheta, d_rhoE,
        d_dr, d_r_center, d_dtheta,
        d_Ap, nr, nt, ng, 1.0/dt, gamma);
}

void LowMachSolver::apply_simple(const double* d_v, double* d_Mv, double dt) {
    int n = nr*nt, B = 256;

    // Step 1: v* = r_momentum / Ap
    k_lm_simple_vstar<<<(n+B-1)/B,B>>>(
        d_v + n, d_v + 2*n, d_Ap, d_simple_vr_s, d_simple_vt_s, n);

    // Step 2: div(v*)
    k_lm_simple_div<<<(n+B-1)/B,B>>>(
        d_simple_vr_s, d_simple_vt_s,
        d_cell_volume, d_area_r, d_area_theta,
        d_simple_div, nr, nt);

    // Compute α = 1/Ap for variable-coefficient Poisson
    k_lm_simple_alpha<<<(n+B-1)/B,B>>>(d_Ap, d_inv_rho, n);

    // Poisson RHS = div(v*), Dirichlet δp=0 at boundary
    k_lm_simple_prhs<<<(n+B-1)/B,B>>>(d_simple_div, d_rhs_poisson, nr, nt);

    // Solve ∇·(α∇δp) = div(v*), where α = 1/Ap
    CUDA_CHECK(cudaMemset(d_simple_p, 0, n*sizeof(double)));
    gmg_pressure.solve_varcoeff(d_inv_rho, d_rhs_poisson, d_simple_p, 10, 1e-3);

    // Step 3: correct velocities with (1/Ap)∇δp, assemble full output
    k_lm_simple_correct<<<(n+B-1)/B,B>>>(
        d_v, d_v + n, d_v + 2*n, d_v + 3*n,
        d_Ap,
        d_simple_vr_s, d_simple_vt_s,
        d_simple_p,
        d_r_center, d_theta_face,
        d_Mv,
        nr, nt, dt);
}

// Assemble Schur complement σ(x) = 4πGρ / Ap
// where Ap = 1/dt + upwind_rate (from block Jacobi diagonal)
__global__
void k_lm_assemble_sigma(const double* rho, const double* blk_diag,
                         double* sigma, int n, int ng, int nt, double G) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    double rho_c = fmax(rho[d_idx(flat/nt, flat%nt, nt, ng)], 1e-20);
    // Ap ≈ |J_{rho,rho}| = |blk_diag inverse's (0,0) entry|
    // But we need J_diag not J⁻¹. The block Jacobi diagonal value was -(1/dt+sr).
    // We stored the INVERSE. Extract Ap from the (0,0) element of blk inverse:
    //   blk_inv[0][0] ≈ -1/Ap for diagonally-dominant system.
    // So Ap ≈ -1/blk_inv[0][0].
    double inv00 = blk_diag[flat * 16]; // (0,0) element of J⁻¹
    double Ap = (fabs(inv00) > 1e-30) ? fabs(1.0 / inv00) : 1e30;
    sigma[flat] = 4.0 * M_PI * G * rho_c / Ap;
}

// Precompute S(x) = max(|cC|, 1) for Poisson residual scaling.
// Matches the stencil in k_lm_poisson_residual exactly.
__global__
void k_lm_poisson_scale(
    const double* r_face, const double* r_center, const double* dr,
    const double* dtheta,
    const double* sin_tf, const double* sin_tc,
    double* scale_out,
    int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;

    if (i == nr-1) { scale_out[flat] = 1.0; return; }

    double ri = r_center[i], ri2 = ri*ri, dri = dr[i];
    double cC = 0.0;
    if (i > 0) {
        double rl = r_face[i];
        cC -= rl*rl / (ri2 * dri * (r_center[i] - r_center[i-1]));
    }
    if (i < nr-1) {
        double rh = r_face[i+1];
        cC -= rh*rh / (ri2 * dri * (r_center[i+1] - r_center[i]));
    }
    double sj = sin_tc[j], dtj = dtheta[j];
    if (j > 0) cC -= sin_tf[j] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j-1]));
    if (j < nt-1) cC -= sin_tf[j+1] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j+1]));

    scale_out[flat] = fmax(fabs(cC), 1.0);
}

// Elementwise: out[i] = a[i] * b[i]
__global__
void k_lm_mul(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// Elementwise: out[i] = a[i] / b[i]
__global__
void k_lm_div(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

void LowMachSolver::assemble_schur_sigma(double dt) {
    int n = nr*nt, B = 256;
    k_lm_assemble_sigma<<<(n+B-1)/B,B>>>(
        d_rho, d_blk_diag, d_sigma_schur, n, ng, nt, G_const);
    k_lm_poisson_scale<<<(n+B-1)/B,B>>>(
        d_r_face, d_r_center, d_dr, d_dtheta,
        d_sin_theta_face, d_sin_theta_center,
        d_poisson_scale, nr, nt);
}

void LowMachSolver::apply_block_schur(const double* d_v, double* d_Mv, double dt) {
    int n = nr*nt, B = 256;

    // Step 1: Fluid part — block Jacobi on first 4N
    k_lm_apply_blkjac<<<(n+B-1)/B,B>>>(d_blk_diag, d_v, d_Mv, n);

    // Step 2: Build Schur RHS in physical (unscaled) space.
    // v₅ is scaled: v₅_scaled = (∇²Φ - 4πGρ)/S
    // Need physical: v₅_phys = v₅_scaled * S
    // J_Φρ = -4πG, so coupling term = (-4πG)·δρ
    // But δρ comes from block Jacobi applied to *scaled* input, so it's physical.
    // RHS = v₅_phys - J_Φρ·δρ = S·v₅ + 4πG·δρ
    k_lm_mul<<<(n+B-1)/B,B>>>(d_v + 4*n, d_poisson_scale, d_schur_rhs, n);
    k_lm_axpy<<<(n+B-1)/B,B>>>(d_schur_rhs, 4.0*M_PI*G_const, d_Mv, n);

    // Step 3: Solve (∇² - σ)δΦ = RHS with Helmholtz GMG
    CUDA_CHECK(cudaMemset(d_Mv + 4*n, 0, n*sizeof(double)));
    gmg_schur.solve_helmholtz(d_sigma_schur, d_schur_rhs, d_Mv + 4*n, 10, 1e-4);
}

void LowMachSolver::apply_preconditioner(const double* d_v, double* d_Mv, double dt) {
    int n = nr*nt, B = 256;

    if (precond_type == PrecondType::PBP) {
        // Hybrid PBP: 2-DOF momentum line solve + GMG Poisson + 4×4 block-inverse
        //   1. 2-DOF momentum r-line solve → ṽr, ṽt
        //   2. GMG Poisson: ∇·(α∇δp) = div(ṽ), global pressure correction
        //   3. Assembly: ρ,E via full block-inverse rows; momentum via Poisson correction
        double inv_dt = 1.0 / dt;

        // Step 1: 2-DOF momentum-only r-line solve
        {
            int smem = nr * (3*4 + 2*2) * sizeof(double);
            int threads = std::min(nr, 256);
            k_lm_mom_line_solve<<<nt, threads, smem>>>(
                d_rho, d_mr, d_mtheta, d_rhoE,
                d_cell_volume, d_area_r,
                d_r_center, d_dr, d_dtheta,
                d_v + n, d_v + 2*n,
                d_simple_vr_s, d_simple_vt_s,
                nr, nt, ng, inv_dt, gamma);
        }

        // Step 2: Pressure Poisson
        k_lm_simple_div<<<(n+B-1)/B,B>>>(
            d_simple_vr_s, d_simple_vt_s,
            d_cell_volume, d_area_r, d_area_theta,
            d_simple_div, nr, nt);

        k_lm_simple_alpha<<<(n+B-1)/B,B>>>(d_Ap, d_inv_rho, n);
        k_lm_simple_prhs<<<(n+B-1)/B,B>>>(d_simple_div, d_rhs_poisson, nr, nt);

        CUDA_CHECK(cudaMemset(d_simple_p, 0, n*sizeof(double)));
        gmg_pressure.solve_varcoeff(d_inv_rho, d_rhs_poisson, d_simple_p, 3, 1e-2);

        // Step 3: Assemble (full block-inverse for ρ,E; Poisson-corrected momentum)
        k_lm_pbp_assemble<<<(n+B-1)/B,B>>>(
            d_rho, d_blk_diag, d_blk_J,
            d_v,
            d_simple_vr_s, d_simple_vt_s,
            d_simple_p, d_Ap,
            d_grad_r_wm, d_grad_r_wp, d_grad_t_wm, d_grad_t_wp,
            d_Mv,
            nr, nt, ng, gamma);
    } else if (precond_type == PrecondType::BLOCK_SCHUR) {
        apply_block_schur(d_v, d_Mv, dt);
    } else if (precond_type == PrecondType::LINE_JACOBI) {
        int smem = nr * (3*16 + 2*4) * sizeof(double);
        int threads = std::min(nr, 256);
        k_lm_line_solve<<<nt, threads, smem>>>(
            d_rho, d_mr, d_mtheta, d_rhoE,
            d_cell_volume, d_area_r, d_area_theta,
            d_r_center, d_r_face, d_theta_face,
            d_dr, d_dtheta, d_gr0,
            d_v, d_Mv,
            nr, nt, ng, gamma, 1.0/dt);
    } else if (precond_type == PrecondType::SIMPLE) {
        apply_simple(d_v, d_Mv, dt);
    } else if (precond_type == PrecondType::BLOCK_JACOBI) {
        k_lm_apply_blkjac<<<(n+B-1)/B,B>>>(d_blk_diag, d_v, d_Mv, n);
    } else {
        k_lm_copy<<<(4*n+B-1)/B,B>>>(d_Mv, d_v, 4*n);
    }
}

