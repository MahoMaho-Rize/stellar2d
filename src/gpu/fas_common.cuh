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
#endif
