#include "gmg_gpu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)

// =================================================================
// Device stencil: constant-coefficient Laplacian  ∇²φ
// =================================================================

__device__ __forceinline__
void d_stencil(int i, int j, int nr, int nt,
               const double* r_face, const double* r_center, const double* dr,
               const double* dtheta,
               const double* sin_tf, const double* sin_tc,
               double& cW, double& cE, double& cS, double& cN, double& cC) {
    cW = cE = cS = cN = cC = 0.0;
    if (i == nr - 1) { cC = 1.0; return; }

    double ri = r_center[i], ri2 = ri * ri, dri = dr[i];

    if (i > 0) {
        double rl = r_face[i];
        cW = rl * rl / (ri2 * dri * (r_center[i] - r_center[i - 1]));
        cC -= cW;
    }
    if (i < nr - 1) {
        double rh = r_face[i + 1];
        cE = rh * rh / (ri2 * dri * (r_center[i + 1] - r_center[i]));
        cC -= cE;
    }

    double sj = sin_tc[j], dtj = dtheta[j];
    if (j > 0) {
        cS = sin_tf[j] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j - 1]));
        cC -= cS;
    }
    if (j < nt - 1) {
        cN = sin_tf[j + 1] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j + 1]));
        cC -= cN;
    }
}

// =================================================================
// Device stencil: variable-coefficient  ∇·(α∇p)
// α is cell-centered; face values by harmonic mean
// =================================================================

__device__ __forceinline__
double harmonic2(double a, double b) {
    return 2.0 * a * b / (a + b + 1e-300);
}

__device__ __forceinline__
void d_stencil_var(int i, int j, int nr, int nt,
                   const double* r_face, const double* r_center, const double* dr,
                   const double* dtheta,
                   const double* sin_tf, const double* sin_tc,
                   const double* alpha,
                   double& cW, double& cE, double& cS, double& cN, double& cC) {
    cW = cE = cS = cN = cC = 0.0;
    if (i == nr - 1) { cC = 1.0; return; }

    double ri = r_center[i], ri2 = ri * ri, dri = dr[i];
    double ai = alpha[i * nt + j];

    if (i > 0) {
        double af = harmonic2(alpha[(i - 1) * nt + j], ai);
        double rl = r_face[i];
        cW = af * rl * rl / (ri2 * dri * (r_center[i] - r_center[i - 1]));
        cC -= cW;
    }
    if (i < nr - 1) {
        double af = harmonic2(alpha[(i + 1) * nt + j], ai);
        double rh = r_face[i + 1];
        cE = af * rh * rh / (ri2 * dri * (r_center[i + 1] - r_center[i]));
        cC -= cE;
    }

    double sj = sin_tc[j], dtj = dtheta[j];
    if (j > 0) {
        double af = harmonic2(alpha[i * nt + (j - 1)], ai);
        cS = af * sin_tf[j] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j - 1]));
        cC -= cS;
    }
    if (j < nt - 1) {
        double af = harmonic2(alpha[i * nt + (j + 1)], ai);
        cN = af * sin_tf[j + 1] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j + 1]));
        cC -= cN;
    }
}

// =================================================================
// Theta-line tridiagonal smoother (constant-coefficient)
// =================================================================

__global__
void k_line_smooth(double* phi, const double* rhs,
                   int nr, int nt,
                   const double* r_face, const double* r_center, const double* dr,
                   const double* dtheta,
                   const double* sin_tf, const double* sin_tc) {
    int i = blockIdx.x;
    if (i >= nr) return;
    if (i == nr - 1) {
        for (int j = threadIdx.x; j < nt; j += blockDim.x)
            phi[i * nt + j] = rhs[i * nt + j];
        return;
    }

    extern __shared__ double smem[];
    double* sa = smem;
    double* sb = sa + nt;
    double* sc = sb + nt;
    double* sd = sc + nt;

    for (int j = threadIdx.x; j < nt; j += blockDim.x) {
        double cW, cE, cS, cN, cC;
        d_stencil(i, j, nr, nt, r_face, r_center, dr, dtheta, sin_tf, sin_tc,
                  cW, cE, cS, cN, cC);
        double rj = rhs[i * nt + j];
        if (i > 0)    rj -= cW * phi[(i - 1) * nt + j];
        if (i < nr-1) rj -= cE * phi[(i + 1) * nt + j];
        sa[j] = (j > 0) ? cS : 0.0;
        sb[j] = cC;
        sc[j] = (j < nt - 1) ? cN : 0.0;
        sd[j] = rj;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = 1; j < nt; ++j) {
            double m = sa[j] / sb[j - 1];
            sb[j] -= m * sc[j - 1];
            sd[j] -= m * sd[j - 1];
        }
        sd[nt - 1] /= sb[nt - 1];
        for (int j = nt - 2; j >= 0; --j)
            sd[j] = (sd[j] - sc[j] * sd[j + 1]) / sb[j];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < nt; j += blockDim.x)
        phi[i * nt + j] = sd[j];
}

// =================================================================
// Theta-line tridiagonal smoother (variable-coefficient)
// =================================================================

__global__
void k_line_smooth_var(double* phi, const double* rhs, const double* alpha,
                       int nr, int nt,
                       const double* r_face, const double* r_center, const double* dr,
                       const double* dtheta,
                       const double* sin_tf, const double* sin_tc) {
    int i = blockIdx.x;
    if (i >= nr) return;
    if (i == nr - 1) {
        for (int j = threadIdx.x; j < nt; j += blockDim.x)
            phi[i * nt + j] = rhs[i * nt + j];
        return;
    }

    extern __shared__ double smem[];
    double* sa = smem;
    double* sb = sa + nt;
    double* sc = sb + nt;
    double* sd = sc + nt;

    for (int j = threadIdx.x; j < nt; j += blockDim.x) {
        double cW, cE, cS, cN, cC;
        d_stencil_var(i, j, nr, nt, r_face, r_center, dr, dtheta, sin_tf, sin_tc,
                      alpha, cW, cE, cS, cN, cC);
        double rj = rhs[i * nt + j];
        if (i > 0)    rj -= cW * phi[(i - 1) * nt + j];
        if (i < nr-1) rj -= cE * phi[(i + 1) * nt + j];
        sa[j] = (j > 0) ? cS : 0.0;
        sb[j] = cC;
        sc[j] = (j < nt - 1) ? cN : 0.0;
        sd[j] = rj;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = 1; j < nt; ++j) {
            double m = sa[j] / sb[j - 1];
            sb[j] -= m * sc[j - 1];
            sd[j] -= m * sd[j - 1];
        }
        sd[nt - 1] /= sb[nt - 1];
        for (int j = nt - 2; j >= 0; --j)
            sd[j] = (sd[j] - sc[j] * sd[j + 1]) / sb[j];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < nt; j += blockDim.x)
        phi[i * nt + j] = sd[j];
}

// =================================================================
// Residual kernels
// =================================================================

__global__
void k_residual(const double* phi, const double* rhs, double* res,
                int nr, int nt,
                const double* r_face, const double* r_center, const double* dr,
                const double* dtheta,
                const double* sin_tf, const double* sin_tc) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr * nt) return;
    int i = flat / nt, j = flat % nt;
    if (i == nr - 1) { res[flat] = rhs[flat] - phi[flat]; return; }

    double cW, cE, cS, cN, cC;
    d_stencil(i, j, nr, nt, r_face, r_center, dr, dtheta, sin_tf, sin_tc,
              cW, cE, cS, cN, cC);
    double Lp = cC * phi[flat];
    if (i > 0)     Lp += cW * phi[(i - 1) * nt + j];
    if (i < nr-1)  Lp += cE * phi[(i + 1) * nt + j];
    if (j > 0)     Lp += cS * phi[i * nt + (j - 1)];
    if (j < nt-1)  Lp += cN * phi[i * nt + (j + 1)];
    res[flat] = rhs[flat] - Lp;
}

__global__
void k_residual_var(const double* phi, const double* rhs, const double* alpha,
                    double* res,
                    int nr, int nt,
                    const double* r_face, const double* r_center, const double* dr,
                    const double* dtheta,
                    const double* sin_tf, const double* sin_tc) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr * nt) return;
    int i = flat / nt, j = flat % nt;
    if (i == nr - 1) { res[flat] = rhs[flat] - phi[flat]; return; }

    double cW, cE, cS, cN, cC;
    d_stencil_var(i, j, nr, nt, r_face, r_center, dr, dtheta, sin_tf, sin_tc,
                  alpha, cW, cE, cS, cN, cC);
    double Lp = cC * phi[flat];
    if (i > 0)     Lp += cW * phi[(i - 1) * nt + j];
    if (i < nr-1)  Lp += cE * phi[(i + 1) * nt + j];
    if (j > 0)     Lp += cS * phi[i * nt + (j - 1)];
    if (j < nt-1)  Lp += cN * phi[i * nt + (j + 1)];
    res[flat] = rhs[flat] - Lp;
}

// =================================================================
// Helmholtz stencil: ∇²u - σ(x)·u
// Same as constant-coefficient Laplacian but with -σ on diagonal
// =================================================================

__global__
void k_line_smooth_helm(double* phi, const double* rhs, const double* sigma,
                        int nr, int nt,
                        const double* r_face, const double* r_center, const double* dr,
                        const double* dtheta,
                        const double* sin_tf, const double* sin_tc) {
    int i = blockIdx.x;
    if (i >= nr) return;
    if (i == nr - 1) {
        for (int j = threadIdx.x; j < nt; j += blockDim.x)
            phi[i * nt + j] = rhs[i * nt + j];
        return;
    }

    extern __shared__ double smem[];
    double* sa = smem;
    double* sb = sa + nt;
    double* sc = sb + nt;
    double* sd = sc + nt;

    for (int j = threadIdx.x; j < nt; j += blockDim.x) {
        double cW, cE, cS, cN, cC;
        d_stencil(i, j, nr, nt, r_face, r_center, dr, dtheta, sin_tf, sin_tc,
                  cW, cE, cS, cN, cC);
        cC -= sigma[i * nt + j];
        double rj = rhs[i * nt + j];
        if (i > 0)    rj -= cW * phi[(i - 1) * nt + j];
        if (i < nr-1) rj -= cE * phi[(i + 1) * nt + j];
        sa[j] = (j > 0) ? cS : 0.0;
        sb[j] = cC;
        sc[j] = (j < nt - 1) ? cN : 0.0;
        sd[j] = rj;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = 1; j < nt; ++j) {
            double m = sa[j] / sb[j - 1];
            sb[j] -= m * sc[j - 1];
            sd[j] -= m * sd[j - 1];
        }
        sd[nt - 1] /= sb[nt - 1];
        for (int j = nt - 2; j >= 0; --j)
            sd[j] = (sd[j] - sc[j] * sd[j + 1]) / sb[j];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < nt; j += blockDim.x)
        phi[i * nt + j] = sd[j];
}

__global__
void k_residual_helm(const double* phi, const double* rhs, const double* sigma,
                     double* res,
                     int nr, int nt,
                     const double* r_face, const double* r_center, const double* dr,
                     const double* dtheta,
                     const double* sin_tf, const double* sin_tc) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr * nt) return;
    int i = flat / nt, j = flat % nt;
    if (i == nr - 1) { res[flat] = rhs[flat] - phi[flat]; return; }

    double cW, cE, cS, cN, cC;
    d_stencil(i, j, nr, nt, r_face, r_center, dr, dtheta, sin_tf, sin_tc,
              cW, cE, cS, cN, cC);
    cC -= sigma[flat];
    double Lp = cC * phi[flat];
    if (i > 0)     Lp += cW * phi[(i - 1) * nt + j];
    if (i < nr-1)  Lp += cE * phi[(i + 1) * nt + j];
    if (j > 0)     Lp += cS * phi[i * nt + (j - 1)];
    if (j < nt-1)  Lp += cN * phi[i * nt + (j + 1)];
    res[flat] = rhs[flat] - Lp;
}

// Restrict σ: volume-weighted average over 2×2 fine cells
__global__
void k_restrict_sigma(const double* fine_sigma, const double* fine_vol,
                      double* coarse_sigma,
                      int cnr, int cnt, int fnr, int fnt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr * cnt) return;
    int ic = flat / cnt, jc = flat % cnt;
    int if0 = 2 * ic, jf0 = 2 * jc;

    double wsum = 0.0, vsum = 0.0;
    for (int di = 0; di < 2; ++di)
        for (int dj = 0; dj < 2; ++dj) {
            int kf = (if0 + di) * fnt + (jf0 + dj);
            double v = fine_vol[kf];
            wsum += fine_sigma[kf] * v;
            vsum += v;
        }
    coarse_sigma[flat] = (vsum > 0.0) ? wsum / vsum : 0.0;
}

// =================================================================
// Restriction: volume-weighted, Dirichlet-aware
// =================================================================

__global__
void k_restrict(const double* fine_res, const double* fine_vol,
                double* coarse_rhs,
                int cnr, int cnt, int fnr, int fnt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr * cnt) return;
    int ic = flat / cnt, jc = flat % cnt;
    if (ic == cnr - 1) { coarse_rhs[flat] = 0.0; return; }

    int if0 = 2 * ic, jf0 = 2 * jc;
    double wsum = 0.0, vsum = 0.0;
    for (int di = 0; di < 2; ++di) {
        int fi = if0 + di;
        if (fi >= fnr - 1) continue;
        for (int dj = 0; dj < 2; ++dj) {
            int kf = fi * fnt + (jf0 + dj);
            double v = fine_vol[kf];
            wsum += fine_res[kf] * v;
            vsum += v;
        }
    }
    coarse_rhs[flat] = (vsum > 0.0) ? wsum / vsum : 0.0;
}

// Restrict α: harmonic mean over 2×2 fine cells
__global__
void k_restrict_alpha(const double* fine_alpha, double* coarse_alpha,
                      int cnr, int cnt, int fnt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr * cnt) return;
    int ic = flat / cnt, jc = flat % cnt;
    int if0 = 2 * ic, jf0 = 2 * jc;

    double inv_sum = 0.0;
    for (int di = 0; di < 2; ++di)
        for (int dj = 0; dj < 2; ++dj)
            inv_sum += 1.0 / (fine_alpha[(if0 + di) * fnt + (jf0 + dj)] + 1e-300);
    coarse_alpha[flat] = 4.0 / (inv_sum + 1e-300);
}

// =================================================================
// Prolongation: piecewise-constant, skip Dirichlet
// =================================================================

__global__
void k_prolongate(double* fine_phi, const double* coarse_phi,
                  int cnr, int cnt, int fnr, int fnt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr * cnt) return;
    int ic = flat / cnt, jc = flat % cnt;
    double e = coarse_phi[flat];
    for (int di = 0; di < 2; ++di) {
        int fi = 2 * ic + di;
        if (fi == fnr - 1) continue;
        for (int dj = 0; dj < 2; ++dj)
            fine_phi[fi * fnt + (2 * jc + dj)] += e;
    }
}

// =================================================================
// Max-norm reduction
// =================================================================

__global__
void k_absmax(const double* arr, double* block_max, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? fabs(arr[idx]) : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = sdata[0];
}

// =================================================================
// Host: level allocation / construction
// =================================================================

static void alloc_level_arrays(GmgLevel& lev) {
    int nr = lev.nr, nt = lev.nt, n = nr * nt;
    CUDA_CHECK(cudaMalloc(&lev.d_r_face, (nr + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_r_center, nr * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_dr, nr * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_theta_face, (nt + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_theta_center, nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_dtheta, nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_sin_theta_face, (nt + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_sin_theta_center, nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_cell_volume, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_phi, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_rhs, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_res, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_phi_tmp, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_alpha, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_sigma, n * sizeof(double)));
}

void GmgGpu::build_level(int l, int nr, int nt,
                         const double* h_rf, const double* h_tf) {
    GmgLevel& lev = levels[l];
    lev.nr = nr; lev.nt = nt;
    alloc_level_arrays(lev);

    std::vector<double> rc(nr), dr_v(nr);
    for (int i = 0; i < nr; ++i) {
        rc[i] = 0.5 * (h_rf[i] + h_rf[i + 1]);
        dr_v[i] = h_rf[i + 1] - h_rf[i];
    }
    CUDA_CHECK(cudaMemcpy(lev.d_r_face, h_rf, (nr+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_r_center, rc.data(), nr*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_dr, dr_v.data(), nr*sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> tc(nt), dt_v(nt), stf(nt+1), stc(nt);
    for (int j = 0; j <= nt; ++j) stf[j] = std::sin(h_tf[j]);
    for (int j = 0; j < nt; ++j) {
        tc[j] = 0.5 * (h_tf[j] + h_tf[j+1]);
        dt_v[j] = h_tf[j+1] - h_tf[j];
        stc[j] = std::sin(tc[j]);
    }
    CUDA_CHECK(cudaMemcpy(lev.d_theta_face, h_tf, (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_theta_center, tc.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_dtheta, dt_v.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_sin_theta_face, stf.data(), (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_sin_theta_center, stc.data(), nt*sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> vol(nr * nt);
    for (int i = 0; i < nr; ++i) {
        double r3h = h_rf[i+1]*h_rf[i+1]*h_rf[i+1], r3l = h_rf[i]*h_rf[i]*h_rf[i];
        for (int j = 0; j < nt; ++j)
            vol[i*nt+j] = (r3h - r3l) / 3.0 * (std::cos(h_tf[j]) - std::cos(h_tf[j+1]));
    }
    CUDA_CHECK(cudaMemcpy(lev.d_cell_volume, vol.data(), nr*nt*sizeof(double), cudaMemcpyHostToDevice));

    int n = nr * nt;
    CUDA_CHECK(cudaMemset(lev.d_phi, 0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_rhs, 0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_res, 0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_phi_tmp, 0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_alpha, 0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_sigma, 0, n*sizeof(double)));
}

void GmgGpu::init(int nr, int nt,
                  const double* h_r_face, const double* h_theta_face) {
    n_levels = 1;
    int cr = nr, ct = nt;
    while (cr >= 4 && ct >= 4) { cr /= 2; ct /= 2; n_levels++; }

    build_level(0, nr, nt, h_r_face, h_theta_face);

    std::vector<double> rf(h_r_face, h_r_face + nr + 1);
    std::vector<double> tf(h_theta_face, h_theta_face + nt + 1);
    for (int l = 1; l < n_levels; ++l) {
        int fnr = (int)rf.size() - 1, fnt = (int)tf.size() - 1;
        int cnr = fnr / 2, cnt = fnt / 2;
        std::vector<double> crf(cnr+1), ctf(cnt+1);
        for (int i = 0; i <= cnr; ++i) crf[i] = rf[2*i];
        for (int j = 0; j <= cnt; ++j) ctf[j] = tf[2*j];
        build_level(l, cnr, cnt, crf.data(), ctf.data());
        rf = crf; tf = ctf;
    }
    h_res_buf_size = nr * nt;
}

// =================================================================
// Smooth / residual / V-cycle: constant-coefficient
// =================================================================

void GmgGpu::smooth(int l, int n_iters) {
    GmgLevel& lev = levels[l];
    size_t smem = 4 * lev.nt * sizeof(double);
    int threads = min(128, lev.nt);
    for (int it = 0; it < n_iters; ++it)
        k_line_smooth<<<lev.nr, threads, smem>>>(
            lev.d_phi, lev.d_rhs, lev.nr, lev.nt,
            lev.d_r_face, lev.d_r_center, lev.d_dr,
            lev.d_dtheta, lev.d_sin_theta_face, lev.d_sin_theta_center);
}

void GmgGpu::compute_residual(int l) {
    GmgLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    k_residual<<<(n+B-1)/B, B>>>(lev.d_phi, lev.d_rhs, lev.d_res,
        lev.nr, lev.nt,
        lev.d_r_face, lev.d_r_center, lev.d_dr,
        lev.d_dtheta, lev.d_sin_theta_face, lev.d_sin_theta_center);
}

void GmgGpu::vcycle(int l) {
    if (l == n_levels - 1) { smooth(l, 200); return; }
    smooth(l, NU1);
    compute_residual(l);
    restrict_level(l, l + 1);
    vcycle(l + 1);
    prolongate_and_correct(l + 1, l);
    smooth(l, NU2);
}

// =================================================================
// Smooth / residual / V-cycle: variable-coefficient
// =================================================================

void GmgGpu::smooth_var(int l, int n_iters) {
    GmgLevel& lev = levels[l];
    size_t smem = 4 * lev.nt * sizeof(double);
    int threads = min(128, lev.nt);
    for (int it = 0; it < n_iters; ++it)
        k_line_smooth_var<<<lev.nr, threads, smem>>>(
            lev.d_phi, lev.d_rhs, lev.d_alpha, lev.nr, lev.nt,
            lev.d_r_face, lev.d_r_center, lev.d_dr,
            lev.d_dtheta, lev.d_sin_theta_face, lev.d_sin_theta_center);
}

void GmgGpu::compute_residual_var(int l) {
    GmgLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    k_residual_var<<<(n+B-1)/B, B>>>(lev.d_phi, lev.d_rhs, lev.d_alpha, lev.d_res,
        lev.nr, lev.nt,
        lev.d_r_face, lev.d_r_center, lev.d_dr,
        lev.d_dtheta, lev.d_sin_theta_face, lev.d_sin_theta_center);
}

void GmgGpu::restrict_alpha(int fine, int coarse) {
    GmgLevel& fl = levels[fine], &cl = levels[coarse];
    int cn = cl.nr * cl.nt, B = 256;
    k_restrict_alpha<<<(cn+B-1)/B, B>>>(fl.d_alpha, cl.d_alpha, cl.nr, cl.nt, fl.nt);
}

void GmgGpu::vcycle_var(int l) {
    if (l == n_levels - 1) { smooth_var(l, 200); return; }
    smooth_var(l, NU1);
    compute_residual_var(l);
    restrict_level(l, l + 1);
    restrict_alpha(l, l + 1);
    vcycle_var(l + 1);
    prolongate_and_correct(l + 1, l);
    smooth_var(l, NU2);
}

// =================================================================
// Smooth / residual / V-cycle: Helmholtz (∇² - σ)
// =================================================================

void GmgGpu::smooth_helm(int l, int n_iters) {
    GmgLevel& lev = levels[l];
    size_t smem = 4 * lev.nt * sizeof(double);
    int threads = min(128, lev.nt);
    for (int it = 0; it < n_iters; ++it)
        k_line_smooth_helm<<<lev.nr, threads, smem>>>(
            lev.d_phi, lev.d_rhs, lev.d_sigma, lev.nr, lev.nt,
            lev.d_r_face, lev.d_r_center, lev.d_dr,
            lev.d_dtheta, lev.d_sin_theta_face, lev.d_sin_theta_center);
}

void GmgGpu::compute_residual_helm(int l) {
    GmgLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    k_residual_helm<<<(n+B-1)/B, B>>>(lev.d_phi, lev.d_rhs, lev.d_sigma, lev.d_res,
        lev.nr, lev.nt,
        lev.d_r_face, lev.d_r_center, lev.d_dr,
        lev.d_dtheta, lev.d_sin_theta_face, lev.d_sin_theta_center);
}

void GmgGpu::restrict_sigma(int fine, int coarse) {
    GmgLevel& fl = levels[fine], &cl = levels[coarse];
    int cn = cl.nr * cl.nt, B = 256;
    k_restrict_sigma<<<(cn+B-1)/B, B>>>(fl.d_sigma, fl.d_cell_volume,
                                         cl.d_sigma, cl.nr, cl.nt, fl.nr, fl.nt);
}

void GmgGpu::vcycle_helm(int l) {
    if (l == n_levels - 1) { smooth_helm(l, 200); return; }
    smooth_helm(l, NU1);
    compute_residual_helm(l);
    restrict_level(l, l + 1);
    restrict_sigma(l, l + 1);
    vcycle_helm(l + 1);
    prolongate_and_correct(l + 1, l);
    smooth_helm(l, NU2);
}

// =================================================================
// Restriction / Prolongation (shared by both paths)
// =================================================================

void GmgGpu::restrict_level(int fine, int coarse) {
    GmgLevel& fl = levels[fine], &cl = levels[coarse];
    int cn = cl.nr * cl.nt, B = 256;
    k_restrict<<<(cn+B-1)/B, B>>>(fl.d_res, fl.d_cell_volume, cl.d_rhs,
                                   cl.nr, cl.nt, fl.nr, fl.nt);
    CUDA_CHECK(cudaMemset(cl.d_phi, 0, cn * sizeof(double)));
}

void GmgGpu::prolongate_and_correct(int coarse, int fine) {
    GmgLevel& cl = levels[coarse], &fl = levels[fine];
    int cn = cl.nr * cl.nt, B = 256;
    k_prolongate<<<(cn+B-1)/B, B>>>(fl.d_phi, cl.d_phi, cl.nr, cl.nt, fl.nr, fl.nt);
}

// =================================================================
// Convergence check
// =================================================================

double GmgGpu::reduce_absmax(int l) {
    GmgLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256, nb = (n+B-1)/B;

    k_absmax<<<nb, B, B*sizeof(double)>>>(lev.d_res, lev.d_phi_tmp, n);

    int cur = nb;
    double* src = lev.d_phi_tmp;
    double* dst = lev.d_res; // safe — will be recomputed
    while (cur > 1) {
        int nb2 = (cur+B-1)/B;
        k_absmax<<<nb2, B, B*sizeof(double)>>>(src, dst, cur);
        cur = nb2;
        double* tmp = src; src = dst; dst = tmp;
    }
    double norm;
    CUDA_CHECK(cudaMemcpy(&norm, src, sizeof(double), cudaMemcpyDeviceToHost));
    return norm;
}

// =================================================================
// Public solve interfaces
// =================================================================

void GmgGpu::solve(double* d_rhs_in, double* d_phi_inout,
                   int max_cycles, double tol) {
    GmgLevel& finest = levels[0];
    int n = finest.nr * finest.nt;
    CUDA_CHECK(cudaMemcpy(finest.d_phi, d_phi_inout, n*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(finest.d_rhs, d_rhs_in, n*sizeof(double), cudaMemcpyDeviceToDevice));

    for (int cyc = 0; cyc < max_cycles; ++cyc) {
        vcycle(0);
        if ((cyc & 1) == 1 || cyc == max_cycles - 1) {
            compute_residual(0);
            if (reduce_absmax(0) < tol) break;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_phi_inout, finest.d_phi, n*sizeof(double), cudaMemcpyDeviceToDevice));
}

void GmgGpu::solve_varcoeff(double* d_alpha_finest, double* d_rhs_in,
                             double* d_phi_inout,
                             int max_cycles, double tol) {
    GmgLevel& finest = levels[0];
    int n = finest.nr * finest.nt;
    CUDA_CHECK(cudaMemcpy(finest.d_phi, d_phi_inout, n*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(finest.d_rhs, d_rhs_in, n*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(finest.d_alpha, d_alpha_finest, n*sizeof(double), cudaMemcpyDeviceToDevice));

    for (int cyc = 0; cyc < max_cycles; ++cyc) {
        vcycle_var(0);
        if ((cyc & 1) == 1 || cyc == max_cycles - 1) {
            compute_residual_var(0);
            if (reduce_absmax(0) < tol) break;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_phi_inout, finest.d_phi, n*sizeof(double), cudaMemcpyDeviceToDevice));
}

void GmgGpu::solve_helmholtz(double* d_sigma_finest, double* d_rhs_in,
                              double* d_phi_inout,
                              int max_cycles, double tol) {
    GmgLevel& finest = levels[0];
    int n = finest.nr * finest.nt;
    CUDA_CHECK(cudaMemcpy(finest.d_phi, d_phi_inout, n*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(finest.d_rhs, d_rhs_in, n*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(finest.d_sigma, d_sigma_finest, n*sizeof(double), cudaMemcpyDeviceToDevice));

    for (int cyc = 0; cyc < max_cycles; ++cyc) {
        vcycle_helm(0);
        if ((cyc & 1) == 1 || cyc == max_cycles - 1) {
            compute_residual_helm(0);
            if (reduce_absmax(0) < tol) break;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_phi_inout, finest.d_phi, n*sizeof(double), cudaMemcpyDeviceToDevice));
}

void GmgGpu::destroy() {
    for (int l = 0; l < n_levels; ++l) {
        GmgLevel& lev = levels[l];
        cudaFree(lev.d_r_face); cudaFree(lev.d_r_center); cudaFree(lev.d_dr);
        cudaFree(lev.d_theta_face); cudaFree(lev.d_theta_center); cudaFree(lev.d_dtheta);
        cudaFree(lev.d_sin_theta_face); cudaFree(lev.d_sin_theta_center);
        cudaFree(lev.d_cell_volume);
        cudaFree(lev.d_phi); cudaFree(lev.d_rhs); cudaFree(lev.d_res);
        cudaFree(lev.d_phi_tmp); cudaFree(lev.d_alpha); cudaFree(lev.d_sigma);
    }
    n_levels = 0;
}
