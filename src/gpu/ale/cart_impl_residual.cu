// Cartesian implicit solver residual kernels.
// 2D Cartesian, cell-centered (ρ, ρvx, ρvy, E) + 2 ghost cells.
//
// Residual (R(U), before subtracting hse_defect):
//   ∂U/∂t + ∂F_x/∂x + ∂F_y/∂y = S(U)
//   S = (0, 0, -ρ·g_y, -ρ·vy·g_y)   (constant downward gravity)
//
// Fluxes use HLLC dispatch (reuse gpu_hllc_dispatch from gpu_hllc.cuh).
// MUSCL reconstruction on primitives (ρ, vx, vy, P).

#include "cart_impl_solver.cuh"
#include "gpu_hllc.cuh"

static __host__ __device__ __forceinline__
int ci_idx(int i, int j, int ny, int ng) {
    return (i + ng) * (ny + 2*ng) + (j + ng);
}
static __host__ __device__ __forceinline__
int ci_flat(int i, int j, int ny) { return i * ny + j; }

// ---------------------------------------------------------------------
// BC kernels
// ---------------------------------------------------------------------
// x-periodic: copy rightmost ng physical cells into left ghosts, leftmost into right
__global__ void k_ci_apply_bc_x_periodic(
    double* rho, double* mx, double* my, double* rhoE,
    int nx, int ny, int ng)
{
    int jp = blockIdx.x * blockDim.x + threadIdx.x;  // j in [-ng, ny+ng)
    if (jp >= ny + 2*ng) return;
    int j = jp - ng;
    int s = ny + 2*ng;

    // left ghost cells i = -ng .. -1 ← physical (nx-ng) .. (nx-1)
    for (int g = 1; g <= ng; ++g) {
        int L = (-g + ng) * s + (j + ng);
        int src = (nx - g + ng) * s + (j + ng);
        rho[L]  = rho[src];
        mx[L]   = mx[src];
        my[L]   = my[src];
        rhoE[L] = rhoE[src];
    }
    // right ghost cells i = nx .. nx+ng-1 ← physical 0 .. ng-1
    for (int g = 0; g < ng; ++g) {
        int R = (nx + g + ng) * s + (j + ng);
        int src = (g + ng) * s + (j + ng);
        rho[R]  = rho[src];
        mx[R]   = mx[src];
        my[R]   = my[src];
        rhoE[R] = rhoE[src];
    }
}

// y-HSE Dirichlet: extrapolate HSE state into y-ghosts (zero velocity, ρ=ρ_edge, P=P_edge)
// For simplicity, clone first/last physical row values (extrapolation by copy is OK
// for the outer-wall-like BC on our convection problem).
__global__ void k_ci_apply_bc_y_hse(
    double* rho, double* mx, double* my, double* rhoE,
    const double* rho0, const double* P0,
    EOS eos, int nx, int ny, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    int s = ny + 2*ng;

    // Bottom ghosts: j = -ng..-1 ← clone HSE reference at j=0
    // We use the HSE reference rho0/P0 (at j=0) so ghost is exactly stationary.
    for (int g = 1; g <= ng; ++g) {
        int jg = -g;
        int k = (i + ng) * s + (jg + ng);
        int fk = ci_flat(i, 0, ny);
        double r0 = rho0[fk], p0 = P0[fk];
        rho[k]  = r0;
        mx[k]   = 0.0;
        my[k]   = 0.0;
        rhoE[k] = r0 * eos.internal_energy(r0, p0);  // v=0
    }
    // Top ghosts: j = ny..ny+ng-1 ← clone HSE reference at j=ny-1
    for (int g = 0; g < ng; ++g) {
        int jg = ny + g;
        int k = (i + ng) * s + (jg + ng);
        int fk = ci_flat(i, ny-1, ny);
        double r0 = rho0[fk], p0 = P0[fk];
        rho[k]  = r0;
        mx[k]   = 0.0;
        my[k]   = 0.0;
        rhoE[k] = r0 * eos.internal_energy(r0, p0);
    }
}

// ---------------------------------------------------------------------
// Atmosphere floor
// ---------------------------------------------------------------------
__global__ void k_ci_floor(
    double* rho, double* mx, double* my, double* rhoE,
    const double* rho0, EOS eos,
    double atm_thresh, int nx, int ny, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int k = ci_idx(i, j, ny, ng);
    int fk = ci_flat(i, j, ny);
    double floor_rho = fmax(atm_thresh, 1e-10 * rho0[fk]);
    if (rho[k] < floor_rho) {
        // Reset to atmosphere: keep HSE, zero velocity
        rho[k]  = rho0[fk];
        mx[k]   = 0.0;
        my[k]   = 0.0;
        // Keep existing rhoE unless it's also degenerate; simplest is reset to HSE energy
        // But we don't have P0 here; skip — upstream snapshot keeps it consistent.
    }
    // Energy positivity
    double r = fmax(rho[k], 1e-30);
    double ke = 0.5 * (mx[k]*mx[k] + my[k]*my[k]) / r;
    double ie = rhoE[k] - ke;
    if (ie <= 0.0) {
        rhoE[k] = ke + 1e-12 * r;  // minimal internal energy
    }
}

// ---------------------------------------------------------------------
// CFL estimate
// ---------------------------------------------------------------------
__global__ void k_ci_cfl(
    const double* rho, const double* mx, const double* my, const double* rhoE,
    double* out,
    int nx, int ny, int ng, EOS eos, double dx, double dy, double atm_thresh)
{
    extern __shared__ double sd[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny;
    double m = 0.0;
    int step = blockDim.x * gridDim.x;
    for (int g = gid; g < N; g += step) {
        int i = g / ny;
        int j = g % ny;
        int k = ci_idx(i, j, ny, ng);
        double r = fmax(rho[k], atm_thresh);
        if (r < atm_thresh) continue;
        double vx = mx[k] / r;
        double vy = my[k] / r;
        double ie = fmax(rhoE[k] - 0.5*r*(vx*vx+vy*vy), 1e-30);
        double P = (eos.gamma - 1.0) * ie;
        double cs = sqrt(fmax(eos.gamma * P / r, 1e-30));
        double inv = (fabs(vx) + cs) / dx + (fabs(vy) + cs) / dy;
        if (inv > m) m = inv;
    }
    sd[tid] = m;
    __syncthreads();
    for (int off = blockDim.x/2; off > 0; off >>= 1) {
        if (tid < off) sd[tid] = fmax(sd[tid], sd[tid+off]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sd[0];
}

// Launcher wrapper (extern "C"-less — fine since called from same TU signatures)
// Actually we need explicit shared mem size; add host wrapper in solver.cu.
// But we launch directly in solver.cu; add shared-mem wrapper here:
__global__ void k_ci_cfl_launch_stub() {}  // marker only

// ---------------------------------------------------------------------
// Pack / unpack state (phys-sized 4-field buffer <-> ghost-cell arrays)
// ---------------------------------------------------------------------
__global__ void k_ci_pack_state(
    double* out,
    const double* rho, const double* mx, const double* my, const double* rhoE,
    int nx, int ny, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int k = ci_idx(i, j, ny, ng);
    int fk = ci_flat(i, j, ny);
    out[4*fk+0] = rho[k];
    out[4*fk+1] = mx[k];
    out[4*fk+2] = my[k];
    out[4*fk+3] = rhoE[k];
}

__global__ void k_ci_unpack_state(
    const double* in,
    double* rho, double* mx, double* my, double* rhoE,
    int nx, int ny, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int k = ci_idx(i, j, ny, ng);
    int fk = ci_flat(i, j, ny);
    rho[k]  = in[4*fk+0];
    mx[k]   = in[4*fk+1];
    my[k]   = in[4*fk+2];
    rhoE[k] = in[4*fk+3];
}

// ---------------------------------------------------------------------
// Compute F(U) = (U-Uⁿ)/dt − (R(U) − R_hse)
// Note: residual kernel writes R(U) into the target buffer; this kernel
// overwrites it with F in-place (safe because we pass different pointers
// in practice, but kernel tolerates alias).
// ---------------------------------------------------------------------
__global__ void k_ci_compute_F(
    double* F,
    const double* rho, const double* mx, const double* my, const double* rhoE,
    const double* Un, const double* hse_defect,
    const double* R_pre, double inv_dt,
    int nx, int ny, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int k = ci_idx(i, j, ny, ng);
    int fk = ci_flat(i, j, ny);

    double u0 = rho[k],  u1 = mx[k], u2 = my[k], u3 = rhoE[k];
    double un0 = Un[4*fk+0], un1 = Un[4*fk+1], un2 = Un[4*fk+2], un3 = Un[4*fk+3];
    double r0 = R_pre[4*fk+0], r1 = R_pre[4*fk+1], r2 = R_pre[4*fk+2], r3 = R_pre[4*fk+3];
    double h0 = hse_defect[4*fk+0], h1 = hse_defect[4*fk+1], h2 = hse_defect[4*fk+2], h3 = hse_defect[4*fk+3];

    // F = (U-Uⁿ)/dt − (R − R_hse)
    F[4*fk+0] = (u0 - un0) * inv_dt - (r0 - h0);
    F[4*fk+1] = (u1 - un1) * inv_dt - (r1 - h1);
    F[4*fk+2] = (u2 - un2) * inv_dt - (r2 - h2);
    F[4*fk+3] = (u3 - un3) * inv_dt - (r3 - h3);
}

// ---------------------------------------------------------------------
// MUSCL reconstruction helper
// ---------------------------------------------------------------------
__device__ __forceinline__
void ci_recon(double vm1, double v0, double vp1, double vp2,
              double& L, double& R, int lim) {
    gpu_recon(vm1, v0, vp1, vp2, L, R, lim);
}

// Convert conserved → primitive at ghost-aware index
__device__ __forceinline__
FPrim ci_prim(const double* rho, const double* mx, const double* my,
              const double* rhoE, EOS eos, int k)
{
    double r = fmax(rho[k], 1e-30);
    double vx = mx[k] / r;
    double vy = my[k] / r;
    double ke = 0.5 * r * (vx*vx + vy*vy);
    double ie = fmax(rhoE[k] - ke, 1e-30);
    double P = (eos.gamma - 1.0) * ie;
    FPrim w; w.rho = r; w.vr = vx; w.vt = vy; w.P = P;
    return w;
}

// ---------------------------------------------------------------------
// Main residual kernel: computes R(U) at each interior cell.
// R writes 4-tuple per cell into res[4*fk..4*fk+3].
// Flux sign convention: dU/dt = -divF + S ⇒ R = -divF + S.
// ---------------------------------------------------------------------
__global__ void k_ci_residual(
    const double* rho, const double* mx, const double* my, const double* rhoE,
    const double* rho0, const double* P0,
    double* res,
    int nx, int ny, int ng, double dx, double dy, double g_y,
    EOS eos, double atm_thresh, int lim_type, int hllc_variant)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int k  = ci_idx(i,   j,   ny, ng);
    int fk = ci_flat(i, j, ny);

    // Grab 4-stencil in x and y
    int kxm1 = ci_idx(i-1, j,   ny, ng);
    int kxp1 = ci_idx(i+1, j,   ny, ng);
    int kxp2 = ci_idx(i+2, j,   ny, ng);
    int kxm2 = ci_idx(i-2, j,   ny, ng);
    int kym1 = ci_idx(i,   j-1, ny, ng);
    int kyp1 = ci_idx(i,   j+1, ny, ng);
    int kyp2 = ci_idx(i,   j+2, ny, ng);
    int kym2 = ci_idx(i,   j-2, ny, ng);

    FPrim w_xm2 = ci_prim(rho,mx,my,rhoE,eos,kxm2);
    FPrim w_xm1 = ci_prim(rho,mx,my,rhoE,eos,kxm1);
    FPrim w_0   = ci_prim(rho,mx,my,rhoE,eos,k);
    FPrim w_xp1 = ci_prim(rho,mx,my,rhoE,eos,kxp1);
    FPrim w_xp2 = ci_prim(rho,mx,my,rhoE,eos,kxp2);
    FPrim w_ym2 = ci_prim(rho,mx,my,rhoE,eos,kym2);
    FPrim w_ym1 = ci_prim(rho,mx,my,rhoE,eos,kym1);
    FPrim w_yp1 = ci_prim(rho,mx,my,rhoE,eos,kyp1);
    FPrim w_yp2 = ci_prim(rho,mx,my,rhoE,eos,kyp2);

    // ----- x faces: i-1/2 (L) and i+1/2 (R) -----
    // At i+1/2: reconstruct from cells (i-1, i, i+1, i+2)
    FPrim wL_xp, wR_xp, wL_xm, wR_xm;
    #define REC_X(face_L, face_R, a, b, c, d) \
        ci_recon(a.rho, b.rho, c.rho, d.rho, face_L.rho, face_R.rho, lim_type); \
        ci_recon(a.vr,  b.vr,  c.vr,  d.vr,  face_L.vr,  face_R.vr,  lim_type); \
        ci_recon(a.vt,  b.vt,  c.vt,  d.vt,  face_L.vt,  face_R.vt,  lim_type); \
        ci_recon(a.P,   b.P,   c.P,   d.P,   face_L.P,   face_R.P,   lim_type);
    REC_X(wL_xp, wR_xp, w_xm1, w_0,   w_xp1, w_xp2);
    REC_X(wL_xm, wR_xm, w_xm2, w_xm1, w_0,   w_xp1);
    #undef REC_X

    // HLLC x-flux: radial=true means use .vr as normal (here vx == vr for x-direction)
    // We pass radial=true for x, and the struct field vr=vx, vt=vy.
    FFlux4 F_xp = gpu_hllc_dispatch(wL_xp, wR_xp, eos, true, hllc_variant);
    FFlux4 F_xm = gpu_hllc_dispatch(wL_xm, wR_xm, eos, true, hllc_variant);

    // ----- y faces: j-1/2 and j+1/2 -----
    // For y direction: we want normal=vy. Use radial=false so gpu_hllc treats
    // .vt as normal. Pass swapped prim: vr<->vt (we need to remap).
    // Simpler: swap vr/vt in y-direction prims.
    auto yprim = [](FPrim w) { FPrim o; o.rho=w.rho; o.vr=w.vt; o.vt=w.vr; o.P=w.P; return o; };
    FPrim yw_ym2 = yprim(w_ym2), yw_ym1 = yprim(w_ym1), yw_0 = yprim(w_0),
          yw_yp1 = yprim(w_yp1), yw_yp2 = yprim(w_yp2);

    FPrim wLy_p, wRy_p, wLy_m, wRy_m;
    #define REC_Y(face_L, face_R, a, b, c, d) \
        ci_recon(a.rho, b.rho, c.rho, d.rho, face_L.rho, face_R.rho, lim_type); \
        ci_recon(a.vr,  b.vr,  c.vr,  d.vr,  face_L.vr,  face_R.vr,  lim_type); \
        ci_recon(a.vt,  b.vt,  c.vt,  d.vt,  face_L.vt,  face_R.vt,  lim_type); \
        ci_recon(a.P,   b.P,   c.P,   d.P,   face_L.P,   face_R.P,   lim_type);
    REC_Y(wLy_p, wRy_p, yw_ym1, yw_0,   yw_yp1, yw_yp2);
    REC_Y(wLy_m, wRy_m, yw_ym2, yw_ym1, yw_0,   yw_yp1);
    #undef REC_Y

    // HLLC y-flux (radial=true here too, since after swap .vr holds vy)
    FFlux4 F_yp_s = gpu_hllc_dispatch(wLy_p, wRy_p, eos, true, hllc_variant);
    FFlux4 F_ym_s = gpu_hllc_dispatch(wLy_m, wRy_m, eos, true, hllc_variant);
    // Unswap flux: (f_rho, f_mr, f_mt, f_E) in swapped frame → true f for (ρ, ρvx, ρvy, E):
    // swapped: vr=vy so f_mr here is momentum in swapped-vr = true ρvy flux
    //          f_mt is ρvy·vx (original ρvx component in y-flux)
    FFlux4 F_yp; F_yp.f_rho = F_yp_s.f_rho; F_yp.f_mr = F_yp_s.f_mt; F_yp.f_mt = F_yp_s.f_mr; F_yp.f_E = F_yp_s.f_E;
    FFlux4 F_ym; F_ym.f_rho = F_ym_s.f_rho; F_ym.f_mr = F_ym_s.f_mt; F_ym.f_mt = F_ym_s.f_mr; F_ym.f_E = F_ym_s.f_E;

    // -----  R = -divF + S  -----
    double invdx = 1.0 / dx;
    double invdy = 1.0 / dy;
    double r_cell = fmax(rho[k], 1e-30);
    double vy_cell = my[k] / r_cell;

    // Well-balanced: subtract hydrostatic pressure gradient in y-momentum
    // dP/dy_hse = -ρ0·g. At cell center: dP₀/dy ≈ (P0[j+1] - P0[j-1])/(2dy)
    // But easier: evaluate the y-pressure gradient of HSE flux and subtract.
    // Simpler well-balanced: include gravity source only on (ρ - ρ0) part of momentum
    // to avoid spurious forcing from discrete HSE imbalance.
    double rho_here = rho[k];
    double rho0_here = rho0[fk];

    double S_mx = 0.0;
    double S_my = -(rho_here - rho0_here) * g_y;  // residual form: only perturbation feels g
    double S_E  = -(rho_here - rho0_here) * vy_cell * g_y;
    // Note: for full HSE preservation via residual,需要 (rho, rho0) 一致 + 壓力梯度 well-balanced.
    // 這裡 ρ=ρ0 + δρ:先只拉擾動部分,HSE 自動平衡。

    res[4*fk+0] = -(F_xp.f_rho - F_xm.f_rho)*invdx - (F_yp.f_rho - F_ym.f_rho)*invdy;
    res[4*fk+1] = -(F_xp.f_mr  - F_xm.f_mr)*invdx  - (F_yp.f_mr  - F_ym.f_mr)*invdy + S_mx;
    res[4*fk+2] = -(F_xp.f_mt  - F_xm.f_mt)*invdx  - (F_yp.f_mt  - F_ym.f_mt)*invdy + S_my;
    res[4*fk+3] = -(F_xp.f_E   - F_xm.f_E)*invdx   - (F_yp.f_E   - F_ym.f_E)*invdy  + S_E;
}
