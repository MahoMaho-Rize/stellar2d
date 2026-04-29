#pragma once

#include "fas_common.cuh"
#include <cmath>
#include <vector>

static inline double gpu_dot(const double* a, const double* b, int n) {
    std::vector<double> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n*sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < n; ++i) s += ha[i]*hb[i];
    return s;
}

static inline double gpu_norm(const double* x, int n) {
    return std::sqrt(gpu_dot(x, x, n));
}

__global__
inline void k_fas_scale(double* x, double a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

__global__
inline void k_fas_copy(double* dst, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__
inline void k_fas_axpy(double* y, double a, const double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__
inline void k_fas_axpy_v(double* y, double a, const double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__
inline void k_fas_pack_flat(const double* rho, const double* mr,
                            const double* mt, const double* rhoE,
                            double* out, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    int n = nr*nt;
    out[flat] = rho[k]; out[n+flat] = mr[k];
    out[2*n+flat] = mt[k]; out[3*n+flat] = rhoE[k];
}

__global__
inline void k_fas_unpack_flat(double* rho, double* mr,
                              double* mt, double* rhoE,
                              const double* in, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    int n = nr*nt;
    rho[k] = in[flat]; mr[k] = in[n+flat];
    mt[k] = in[2*n+flat]; rhoE[k] = in[3*n+flat];
}

__global__
inline void k_fas_precond(const double* v, const double* blk_inv,
                          double* z, int n4) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = n4 / 4;
    if (flat >= n) return;
    const double* B = &blk_inv[flat*16];
    double v0 = v[flat], v1 = v[n+flat], v2 = v[2*n+flat], v3 = v[3*n+flat];
    z[flat]     = B[0]*v0 + B[1]*v1 + B[2]*v2 + B[3]*v3;
    z[n+flat]   = B[4]*v0 + B[5]*v1 + B[6]*v2 + B[7]*v3;
    z[2*n+flat] = B[8]*v0 + B[9]*v1 + B[10]*v2 + B[11]*v3;
    z[3*n+flat] = B[12]*v0 + B[13]*v1 + B[14]*v2 + B[15]*v3;
}

__global__
inline void k_fas_perturb(double* rho, double* mr, double* mt, double* rhoE,
                          const double* z, double eps, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    int n = nr*nt;
    rho[k]  += eps * z[flat];
    mr[k]   += eps * z[n+flat];
    mt[k]   += eps * z[2*n+flat];
    rhoE[k] += eps * z[3*n+flat];
}
