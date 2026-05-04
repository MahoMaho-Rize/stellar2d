#pragma once

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); std::exit(1); \
    } \
} while(0)

#ifdef __CUDACC__
#include "../eos.h"

__device__ __forceinline__
int fas_idx(int i, int j, int nt, int ng) {
    return (i + ng) * (nt + 2 * ng) + (j + ng);
}

// Forward declarations for kernels shared across TUs
__global__ void k_fas_sponge(double* rho, double* mr, double* mt, double* rhoE,
    const double* rho0, const double* P0, const double* r_center,
    double r_sp, double r_tp, double kappa_max, double dt_s,
    double gam_minus1_inv, int nr, int nt, int ng);

__global__ void k_fas_compute_F(double* F, const double* R,
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* fas_rhs, double inv_dt, int nr, int nt, int ng);

// Residual kernels (needed by explicit stepper)
// radial_only=1: skip theta-face HLLC + MUSCL; set mt-channel residual to 0
__global__ void k_fas_residual(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, EOS eos, double atm_thresh,
    int use_wellbalance, int lim_type, int hllc_variant, int radial_only);

__global__ void k_fas_residual_origin(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, EOS eos, double atm_thresh,
    int use_wellbalance, int lim_type, int hllc_variant, int radial_only);

__global__ void k_fas_cfl(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* dr, const double* r_center, const double* dtheta,
    const double* rho0, double* out,
    int nr, int nt, int ng, EOS eos, double atm_thresh,
    int n_angular_avg, int radial_only);

// Zero the theta-momentum channel everywhere (used to enforce v_theta=0 invariant)
__global__ void k_fas_zero_mt(double* mt, int nr, int nt, int ng);

__global__ void k_fas_atm_reset(double* rho, double* mr, double* mt, double* rhoE,
    const double* rho0, const double* P0,
    double atm_thresh, EOS eos,
    int nr, int nt, int ng, int strict_atm_only);

__global__ void k_fas_ghost_r_out_hse(double* rho, double* mr, double* mt, double* rhoE,
    const double* rho0, const double* P0,
    EOS eos,
    int nr, int nt, int ng);

__global__ void k_fas_angular_avg(double* rho, double* mr, double* mt, double* rhoE,
    const double* vol,
    int n_avg, int nr, int nt, int ng);

__global__ void k_fas_pole_avg(double* rho, double* mr, double* mt, double* rhoE,
    const double* vol,
    int n_pole, int nr, int nt, int ng);

__global__ void k_fas_central_damp(double* mr, double* rhoE,
    const double* rho, const double* r_center,
    double r_damp, double alpha,
    int nr, int nt, int ng);

__global__ void k_fas_rhoV_EV(const double*, const double*, const double*,
    double*, double*, int, int, int);

__global__ void k_fas_conserve_correct(double*, double*,
    const double*, double, double, double, int, int, int);

// 4×4 Gauss-Jordan inverse: A → inv_out (row-major, partial pivoting)
__device__ __forceinline__
void mat4_invert(const double* __restrict__ src, double* __restrict__ inv_out) {
    double A[16], I[16];
    for (int q = 0; q < 16; q++) { A[q] = src[q]; I[q] = (q/4 == q%4) ? 1.0 : 0.0; }
    for (int col = 0; col < 4; col++) {
        double mx = fabs(A[col*4+col]); int mi = col;
        for (int row = col+1; row < 4; row++)
            if (fabs(A[row*4+col]) > mx) { mx = fabs(A[row*4+col]); mi = row; }
        if (mi != col) {
            for (int q = 0; q < 4; q++) { double t = A[col*4+q]; A[col*4+q] = A[mi*4+q]; A[mi*4+q] = t; }
            for (int q = 0; q < 4; q++) { double t = I[col*4+q]; I[col*4+q] = I[mi*4+q]; I[mi*4+q] = t; }
        }
        double d = A[col*4+col]; if (fabs(d) < 1e-30) d = (d >= 0 ? 1e-30 : -1e-30);
        for (int row = col+1; row < 4; row++) {
            double m = A[row*4+col] / d;
            for (int q = col; q < 4; q++) A[row*4+q] -= m * A[col*4+q];
            for (int q = 0; q < 4; q++) I[row*4+q] -= m * I[col*4+q];
        }
    }
    for (int c = 0; c < 4; c++)
        for (int row = 3; row >= 0; row--) {
            double s = I[row*4+c];
            for (int q = row+1; q < 4; q++) s -= A[row*4+q] * I[q*4+c];
            I[row*4+c] = s / A[row*4+row];
        }
    for (int q = 0; q < 16; q++) inv_out[q] = I[q];
}

// 4×4 solve: A·x = b (in-place, b overwritten with x)
__device__ __forceinline__
void mat4_solve(const double* __restrict__ src, double* __restrict__ b) {
    double A[16];
    for (int q = 0; q < 16; q++) A[q] = src[q];
    for (int col = 0; col < 4; col++) {
        double mx = fabs(A[col*4+col]); int mi = col;
        for (int row = col+1; row < 4; row++)
            if (fabs(A[row*4+col]) > mx) { mx = fabs(A[row*4+col]); mi = row; }
        if (mi != col) {
            for (int q = 0; q < 4; q++) { double t = A[col*4+q]; A[col*4+q] = A[mi*4+q]; A[mi*4+q] = t; }
            double t = b[col]; b[col] = b[mi]; b[mi] = t;
        }
        double d = A[col*4+col]; if (fabs(d) < 1e-30) d = 1e-30;
        for (int row = col+1; row < 4; row++) {
            double m = A[row*4+col] / d;
            for (int q = col; q < 4; q++) A[row*4+q] -= m * A[col*4+q];
            b[row] -= m * b[col];
        }
    }
    for (int row = 3; row >= 0; row--) {
        double s = b[row];
        for (int q = row+1; q < 4; q++) s -= A[row*4+q] * b[q];
        b[row] = s / A[row*4+row];
    }
}

// 4×4 matrix multiply: C = A * B (row-major)
__device__ __forceinline__
void mat4_mul(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double s = 0;
            for (int q = 0; q < 4; q++) s += A[r*4+q] * B[q*4+c];
            C[r*4+c] = s;
        }
}
#endif
