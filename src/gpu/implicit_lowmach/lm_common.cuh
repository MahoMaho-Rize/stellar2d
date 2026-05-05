#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); std::exit(1); \
    } \
} while(0)

#ifdef __CUDACC__
__device__ __forceinline__
int d_idx(int i, int j, int nt, int ng) {
    return (i + ng) * (nt + 2 * ng) + (j + ng);
}

__global__ void k_lm_dot(const double* a, const double* b, double* out, int n);
__global__ void k_lm_reduce_sum(const double* in, double* out, int n);
__global__ void k_lm_axpy(double* y, double a, const double* x, int n);
__global__ void k_lm_scale(double* x, double a, int n);
__global__ void k_lm_copy(double* dst, const double* src, int n);
__global__ void k_lm_zero(double* x, int n);
__global__ void k_lm_pack(const double* rho, const double* mr, const double* mt,
                           const double* rhoE, const double* phi,
                           double* out, int nr, int nt, int ng);
__global__ void k_lm_unpack_set(double* rho, double* mr, double* mt,
                                 double* rhoE, double* phi,
                                 const double* in, int nr, int nt, int ng);
__global__ void k_lm_unpack_add(double* rho, double* mr, double* mt,
                                 double* rhoE, double* phi,
                                 const double* delta, double alpha,
                                 int nr, int nt, int ng);
__global__ void k_lm_compute_F(double* F, const double* R,
                                const double* rho, const double* mr,
                                const double* mt, const double* rhoE,
                                const double* Un, double inv_dt,
                                int nr, int nt, int ng);
__global__ void k_lm_apply_blkjac(const double* blk_inv, const double* v, double* Mv, int n);

__global__ void k_lm_ediv(double* v, const double* s, int n);
__global__ void k_lm_compute_scale(const double* Un, double* scale, int n);
__global__ void k_lm_compute_music_scale(const double* rho, const double* mr,
    const double* mt, const double* rhoE, double* sR, double* sL,
    int nr, int nt, int ng, double gam);
__global__ void k_lm_emul(double* v, const double* s, int n);

__global__ void k_lm_snapshot_hse(const double* rho, const double* rhoE, const double* phi,
    double* rho0, double* P0, double* phi0, int nr, int nt, int ng, double gamma);

__global__ void k_lm_residual(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, double gam, double atm_thresh, int use_wellbalance);

__global__ void k_lm_residual_origin(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, double gam, double atm_thresh, int use_wellbalance);

__global__ void k_lm_floor(double* rho, double* mr, double* mt, double* rhoE,
    int nr, int nt, int ng, double gamma, double rho_fl, double P_fl);

__global__ void k_lm_sponge(double* rho, double* mr, double* mt, double* rhoE,
    const double* rho0, const double* P0, const double* r_center,
    double r_sp, double r_tp, double kappa_max, double dt_s, double gam_minus1_inv,
    int nr, int nt, int ng);

__global__ void k_lm_cfl(const double* rho, const double* mr, const double* mt,
    const double* rhoE, const double* dr, const double* rc, const double* dtheta,
    const double* rho0, double* out, int nr, int nt, int ng, double gam, double atm_thresh);
__global__ void k_lm_simple_div(const double* vr, const double* vt,
                                 const double* vol, const double* ar, const double* at,
                                 double* div_out, int nr, int nt);
__global__ void k_lm_simple_alpha(const double* Ap, double* alpha, int n);
__global__ void k_lm_simple_prhs(const double* div_v, double* rhs, int nr, int nt);
#endif

static inline double gpu_dot(const double* a, const double* b, double* wa, double* wb, int n) {
    int B = 256, nb = (n+B-1)/B;
    k_lm_dot<<<nb,B,B*sizeof(double)>>>(a, b, wa, n);
    double *src = wa, *dst = wb;
    int cur = nb;
    while (cur > 1) {
        int nb2 = (cur+B-1)/B;
        k_lm_reduce_sum<<<nb2,B,B*sizeof(double)>>>(src, dst, cur);
        cur = nb2; double* t = src; src = dst; dst = t;
    }
    double val;
    CUDA_CHECK(cudaMemcpy(&val, src, sizeof(double), cudaMemcpyDeviceToHost));
    return val;
}
