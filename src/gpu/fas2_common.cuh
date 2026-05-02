#pragma once

// fas2_common: k_fas2_* forward declarations for the fas2 solver family.
// Reuses fas_common.cuh for shared utilities (fas_idx, CUDA_CHECK,
// mat4_invert/solve/mul), since those are identical across fas / fas2.
#include "fas_common.cuh"

#ifdef __CUDACC__

__global__ void k_fas2_sponge(double* rho, double* mr, double* mt, double* rhoE,
    const double* rho0, const double* P0, const double* r_center,
    double r_sp, double r_tp, double kappa_max, double dt_s,
    double gam_minus1_inv, int nr, int nt, int ng);

__global__ void k_fas2_compute_F(double* F, const double* R,
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* fas_rhs, double inv_dt, int nr, int nt, int ng);

__global__ void k_fas2_residual(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, EOS eos, double atm_thresh,
    int use_wellbalance, int lim_type, int hllc_variant, int radial_only);

__global__ void k_fas2_residual_origin(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, EOS eos, double atm_thresh,
    int use_wellbalance, int lim_type, int hllc_variant, int radial_only);

__global__ void k_fas2_cfl(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* dr, const double* r_center, const double* dtheta,
    const double* rho0, double* out,
    int nr, int nt, int ng, EOS eos, double atm_thresh,
    int n_angular_avg, int radial_only);

__global__ void k_fas2_zero_mt(double* mt, int nr, int nt, int ng);

__global__ void k_fas2_atm_reset(double* rho, double* mr, double* mt, double* rhoE,
    const double* rho0, const double* P0,
    double atm_thresh, EOS eos,
    int nr, int nt, int ng, int strict_atm_only);

__global__ void k_fas2_ghost_r_out_hse(double* rho, double* mr, double* mt, double* rhoE,
    const double* rho0, const double* P0,
    EOS eos,
    int nr, int nt, int ng);

__global__ void k_fas2_angular_avg(double* rho, double* mr, double* mt, double* rhoE,
    const double* vol,
    int n_avg, int nr, int nt, int ng);

__global__ void k_fas2_pole_avg(double* rho, double* mr, double* mt, double* rhoE,
    const double* vol,
    int n_pole, int nr, int nt, int ng);

__global__ void k_fas2_central_damp(double* mr, double* rhoE,
    const double* rho, const double* r_center,
    double r_damp, double alpha,
    int nr, int nt, int ng);

__global__ void k_fas2_rhoV_EV(const double*, const double*, const double*,
    double*, double*, int, int, int);

__global__ void k_fas2_conserve_correct(double*, double*,
    const double*, double, double, double, int, int, int);

// Viallet 2016 eq 72 scaling (fas2 fix 3/4)
__global__ void k_fas2_build_scaling(
    const double* rho, const double* mr, const double* mt,
    const double* rho0, const double* P0,
    double* L, double* R, double* invL,
    EOS eos, double alpha1, double alpha2,
    int nr, int nt, int ng);

__global__ void k_fas2_scale_by_diag(double* d_x, const double* d_D, int N);

#endif
