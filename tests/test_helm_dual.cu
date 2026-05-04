// Verify helm_eval_dual<N> derivatives match central finite differences
// on the scalar helm_eval. Runs on GPU (Helm tables live in device memory).

#include "../src/physics/helmholtz_eos.cuh"
#include "../src/physics/helmholtz_eos_dual.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---- Device kernels ----
struct GridPt { double rho, T; };
struct ScalarOut {
    double P, e, cs, gam1, cV, grada;
};
struct DualOut {
    double P_v, P_grho, P_gT;
    double e_v, e_grho, e_gT;
    double cs_v, gam1_v, cV_v, grada_v;
};

__global__ void k_eval_scalar(const GridPt* pts, int n, ScalarOut* out,
                              HelmholtzTableView tv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    HelmState s = helm_eval(pts[i].rho, pts[i].T, tv);
    out[i].P    = s.P;
    out[i].e    = s.e;
    out[i].cs   = s.cs;
    out[i].gam1 = s.gamma1;
    out[i].cV   = s.cV;
    out[i].grada= s.grada;
}

__global__ void k_eval_dual(const GridPt* pts, int n, DualOut* out,
                            HelmholtzTableView tv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dual::Dual<2> rho_d = dual::Dual<2>::seed(pts[i].rho, 0);
    dual::Dual<2> T_d   = dual::Dual<2>::seed(pts[i].T,   1);
    HelmStateDual<2> sd = helm_eval_dual<2>(rho_d, T_d, tv);
    out[i].P_v    = sd.P.v;
    out[i].P_grho = sd.P.g[0];
    out[i].P_gT   = sd.P.g[1];
    out[i].e_v    = sd.e.v;
    out[i].e_grho = sd.e.g[0];
    out[i].e_gT   = sd.e.g[1];
    out[i].cs_v   = sd.cs.v;
    out[i].gam1_v = sd.gamma1.v;
    out[i].cV_v   = sd.cV.v;
    out[i].grada_v= sd.grada.v;
}

// Central-difference probe on scalar kernel
__global__ void k_eval_fd(const GridPt* base, int n, double h_rho_frac, double h_T_frac,
                          ScalarOut* rho_plus, ScalarOut* rho_minus,
                          ScalarOut* T_plus,   ScalarOut* T_minus,
                          HelmholtzTableView tv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double rho = base[i].rho, T = base[i].T;
    double h_rho = h_rho_frac * rho + 1e-12;
    double h_T   = h_T_frac   * T   + 1e-3;
    HelmState srp = helm_eval(rho + h_rho, T, tv);
    HelmState srm = helm_eval(rho - h_rho, T, tv);
    HelmState stp = helm_eval(rho, T + h_T, tv);
    HelmState stm = helm_eval(rho, T - h_T, tv);
    rho_plus [i] = { srp.P, srp.e, srp.cs, srp.gamma1, srp.cV, srp.grada };
    rho_minus[i] = { srm.P, srm.e, srm.cs, srm.gamma1, srm.cV, srm.grada };
    T_plus   [i] = { stp.P, stp.e, stp.cs, stp.gamma1, stp.cV, stp.grada };
    T_minus  [i] = { stm.P, stm.e, stm.cs, stm.gamma1, stm.cV, stm.grada };
}

static bool close_rel(double a, double b, double rtol) {
    double s = std::fabs(a) + std::fabs(b);
    if (s < 1e-300) return true;
    return std::fabs(a - b) / s < rtol;
}

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 2; }} while (0)

int main() {
    HelmholtzTable tbl;
    if (tbl.load(nullptr, "third_party/helmholtz/helm_table.bin", 0) != 0) {
        std::printf("SKIP: helm_table.bin not present\n");
        return 0;
    }
    HelmholtzTableView tv = tbl.view;
    tv.Abar = 1.3;
    tv.Zbar = 1.1;

    std::vector<GridPt> h_pts = {
        {1e-6,  1e4},
        {1e-3,  1e5},
        {1e-1,  1e6},
        {1.0,   1e6},
        {10.0,  5e6},
        {100.0, 1.5e7},
        {150.0, 1.55e7},
        {1e3,   1e8},
    };
    int n = (int)h_pts.size();

    GridPt* d_pts;
    ScalarOut *d_s, *d_rp, *d_rm, *d_tp, *d_tm;
    DualOut* d_d;
    CUDA_OK(cudaMalloc(&d_pts, n*sizeof(GridPt)));
    CUDA_OK(cudaMalloc(&d_s,   n*sizeof(ScalarOut)));
    CUDA_OK(cudaMalloc(&d_d,   n*sizeof(DualOut)));
    CUDA_OK(cudaMalloc(&d_rp,  n*sizeof(ScalarOut)));
    CUDA_OK(cudaMalloc(&d_rm,  n*sizeof(ScalarOut)));
    CUDA_OK(cudaMalloc(&d_tp,  n*sizeof(ScalarOut)));
    CUDA_OK(cudaMalloc(&d_tm,  n*sizeof(ScalarOut)));

    CUDA_OK(cudaMemcpy(d_pts, h_pts.data(), n*sizeof(GridPt), cudaMemcpyHostToDevice));

    int B = 64, G = (n + B - 1) / B;
    k_eval_scalar<<<G, B>>>(d_pts, n, d_s, tv);
    k_eval_dual  <<<G, B>>>(d_pts, n, d_d, tv);
    k_eval_fd    <<<G, B>>>(d_pts, n, 1e-5, 1e-5, d_rp, d_rm, d_tp, d_tm, tv);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<ScalarOut> h_s(n), h_rp(n), h_rm(n), h_tp(n), h_tm(n);
    std::vector<DualOut>   h_d(n);
    CUDA_OK(cudaMemcpy(h_s.data(),  d_s,  n*sizeof(ScalarOut), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_d.data(),  d_d,  n*sizeof(DualOut),   cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_rp.data(), d_rp, n*sizeof(ScalarOut), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_rm.data(), d_rm, n*sizeof(ScalarOut), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_tp.data(), d_tp, n*sizeof(ScalarOut), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_tm.data(), d_tm, n*sizeof(ScalarOut), cudaMemcpyDeviceToHost));

    int fail = 0, total = 0;
    for (int i = 0; i < n; ++i) {
        double rho = h_pts[i].rho, T = h_pts[i].T;
        double h_rho = 1e-5 * rho + 1e-12;
        double h_T   = 1e-5 * T   + 1e-3;

        // Value checks
        auto val_match = [&](const char* lab, double a, double b, double rtol=1e-10) {
            if (!close_rel(a, b, rtol)) {
                std::fprintf(stderr, "FAIL  %s  rho=%.1e T=%.1e  AD=%.6e  scalar=%.6e\n",
                             lab, rho, T, a, b);
                ++fail;
            }
            ++total;
        };
        val_match("P",    h_d[i].P_v,    h_s[i].P);
        val_match("e",    h_d[i].e_v,    h_s[i].e);
        val_match("cs",   h_d[i].cs_v,   h_s[i].cs);
        val_match("gam1", h_d[i].gam1_v, h_s[i].gam1);
        val_match("cV",   h_d[i].cV_v,   h_s[i].cV);
        val_match("grada",h_d[i].grada_v,h_s[i].grada);

        // FD derivatives
        double dPdrho_fd = (h_rp[i].P - h_rm[i].P) / (2.0 * h_rho);
        double dPdT_fd   = (h_tp[i].P - h_tm[i].P) / (2.0 * h_T);
        double dedrho_fd = (h_rp[i].e - h_rm[i].e) / (2.0 * h_rho);
        double dedT_fd   = (h_tp[i].e - h_tm[i].e) / (2.0 * h_T);

        auto cmp = [&](const char* lab, double ad, double fd, double rtol=1e-3) {
            if (!close_rel(ad, fd, rtol)) {
                std::fprintf(stderr,
                    "FAIL  %s  rho=%.1e T=%.1e  AD=%.6e  FD=%.6e  rel=%.2e\n",
                    lab, rho, T, ad, fd,
                    std::fabs(ad - fd) / (std::fabs(ad) + std::fabs(fd) + 1e-300));
                ++fail;
            }
            ++total;
        };
        cmp("dP/drho", h_d[i].P_grho, dPdrho_fd);
        cmp("dP/dT",   h_d[i].P_gT,   dPdT_fd);
        cmp("de/drho", h_d[i].e_grho, dedrho_fd);
        cmp("de/dT",   h_d[i].e_gT,   dedT_fd);
    }

    cudaFree(d_pts);
    cudaFree(d_s); cudaFree(d_d);
    cudaFree(d_rp); cudaFree(d_rm); cudaFree(d_tp); cudaFree(d_tm);
    tbl.destroy();

    if (fail == 0) {
        std::printf("helm_eval_dual: all %d checks passed (%d grid points)\n", total, n);
        return 0;
    }
    std::printf("helm_eval_dual: %d of %d FAIL\n", fail, total);
    return 1;
}
