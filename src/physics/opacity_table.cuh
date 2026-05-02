#pragma once

// MESA-derived Rosseland-mean opacity table (Type-1, single-family).
//
// Mirrors the Helmholtz EOS pattern:
//   - KapTable             host-side owner + GPU allocation + L2 pinning
//   - KapTableView         POD passed by value into kernels (pointers live
//                          in device memory)
//   - kap_eval / kap_eval_Xslice trilinear interpolation in (X, logT, logR)
//
// Table layout on disk (see scripts/convert_mesa_kap.py for the writer):
//
//   128 B header (magic "KAPv1" + family[48] + Z + n_X + n_logT + n_logR
//                 + logT_min/max + logR_min/max + reserved)
//   float64 payload:
//     X       [n_X]
//     logT    [n_logT]
//     logR    [n_logR]
//     log_kap [n_X, n_logT, n_logR]    (X-outer, logR-fastest)
//
// Opacity is stored as log10 κ_R [cm²/g]. The independent variable is
//   logR = log10 ρ − 3 · log10 T + 18
// where ρ is in g/cc and T in Kelvin. Converting to ρ involves:
//   logR = logρ_cgs − 3·logT + 18
//
// GPU placement: we pin the `log_kap` payload into L2 persisting cache the
// same way we do for the Helmholtz table. Each `(family, Z)` binary is at
// most ~1.9 MB (oplib_mb22 has 30 X × 213 T × 39 R), so ten of them still
// fit in the 44 MB L2 persisting budget on sm_89.

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define KAP_HD __host__ __device__
#else
#define KAP_HD
#endif

// Magic tag that starts every KAPv1 binary.
static constexpr char KAP_MAGIC[9] = "KAPv1\0\0\0";

// Maximum expected table dimensions (used only for a loader sanity check).
static constexpr int KAP_MAX_X    = 40;
static constexpr int KAP_MAX_LOGT = 256;
static constexpr int KAP_MAX_LOGR = 64;

// POD description of the table as seen by device kernels. Pointers refer
// to device memory owned by `KapTable` below.
struct KapTableView {
    const double* X;           // (n_X,)        composition axis (H mass fraction)
    const double* logT;        // (n_logT,)     temperature axis
    const double* logR;        // (n_logR,)     R = logρ − 3 logT + 18 axis
    const double* log_kap;     // (n_X, n_logT, n_logR)   log10 κ_R [cm²/g]

    int n_X;
    int n_logT;
    int n_logR;

    double Z;                  // metal mass fraction for this table

    // Cached bounds. The logR grid is uniformly spaced for every MESA kap
    // file we've seen, so we keep the stride/inverse-stride constants and
    // use O(1) direct indexing there. The logT grid is piecewise-uniform
    // (MESA uses finer steps in the H-ionisation band) and the X grid is
    // non-uniform, so both use bisection.
    double logT_min, logT_max;
    double logR_min, logR_max, logR_d, logR_di;
};

// Host-side owner. Holds a single device allocation that contains
// [X][logT][logR][log_kap]; the view fields above point into it.
struct KapTable {
    double* d_data = nullptr;
    std::size_t bytes = 0;
    cudaStream_t stream = 0;
    char family[49] = {0};    // zero-terminated copy of the file's family tag

    KapTableView view = {};

    // Load one KAPv1 binary. Returns 0 on success.
    int load(const char* bin_path, cudaStream_t s = 0);

    // Free device memory + reset L2 window.
    void destroy();
};

// Row-major 3D index: (iX, jT, iR) with R the innermost / fastest axis.
KAP_HD inline int kap_idx3d(const KapTableView& kv, int iX, int jT, int iR) {
    return (iX * kv.n_logT + jT) * kv.n_logR + iR;
}

// Bisection on a sorted ascending array of length n. Returns i such that
// arr[i] <= x < arr[i+1], clamped to [0, n-2]. Used for X and logT.
KAP_HD inline int kap_bisect(const double* arr, int n, double x) {
    if (x <= arr[0]) return 0;
    if (x >= arr[n - 1]) return n - 2;
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] <= x) lo = mid;
        else               hi = mid;
    }
    return lo;
}

// logT grid is piecewise-uniform on MESA files — bisect.
KAP_HD inline int kap_index_logT(const KapTableView& kv, double logT) {
    return kap_bisect(kv.logT, kv.n_logT, logT);
}
// logR is uniformly spaced — O(1).
KAP_HD inline int kap_index_logR(const KapTableView& kv, double logR) {
    int i = (int)((logR - kv.logR_min) * kv.logR_di);
    if (i < 0) i = 0;
    if (i > kv.n_logR - 2) i = kv.n_logR - 2;
    return i;
}
// X grid is non-uniform (solar sets use {0, 0.1, 0.2, 0.35, 0.5, 0.7, …}).
KAP_HD inline int kap_index_X(const KapTableView& kv, double X) {
    return kap_bisect(kv.X, kv.n_X, X);
}

// --- Bilinear lookup on a single X slice --------------------------------
// Returns log10 κ_R for the given (logT, logR), interpolating bilinearly
// on the enclosing cell at X slice iX.
KAP_HD inline double kap_eval_log_slice(const KapTableView& kv,
                                        int iX, double logT, double logR) {
    int jT = kap_index_logT(kv, logT);
    int iR = kap_index_logR(kv, logR);
    double T0 = kv.logT[jT], T1 = kv.logT[jT + 1];
    double t = (logT - T0) / (T1 - T0);
    double r = (logR - (kv.logR_min + iR * kv.logR_d)) * kv.logR_di;
    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
    if (r < 0.0) r = 0.0; else if (r > 1.0) r = 1.0;
    double k00 = kv.log_kap[kap_idx3d(kv, iX, jT,     iR    )];
    double k01 = kv.log_kap[kap_idx3d(kv, iX, jT,     iR + 1)];
    double k10 = kv.log_kap[kap_idx3d(kv, iX, jT + 1, iR    )];
    double k11 = kv.log_kap[kap_idx3d(kv, iX, jT + 1, iR + 1)];
    double k0  = (1.0 - r) * k00 + r * k01;
    double k1  = (1.0 - r) * k10 + r * k11;
    return (1.0 - t) * k0 + t * k1;
}

// --- Trilinear lookup in (X, logT, logR) --------------------------------
// Returns κ_R [cm²/g] (not log10). Used by the hydro hot loop.
KAP_HD inline double kap_eval(const KapTableView& kv,
                              double X, double logT, double logR) {
    int iX = kap_index_X(kv, X);
    double Xlo = kv.X[iX];
    double Xhi = kv.X[iX + 1];
    double xt  = (X - Xlo) / (Xhi - Xlo);
    if (xt < 0.0) xt = 0.0; else if (xt > 1.0) xt = 1.0;

    double lk0 = kap_eval_log_slice(kv, iX,     logT, logR);
    double lk1 = kap_eval_log_slice(kv, iX + 1, logT, logR);
    double lk  = (1.0 - xt) * lk0 + xt * lk1;
#ifdef __CUDA_ARCH__
    return exp10(lk);
#else
    return pow(10.0, lk);
#endif
}

// Convenience wrapper that takes (rho [g/cc], T [K]) instead of (logT, logR).
KAP_HD inline double kap_eval_rho_T(const KapTableView& kv,
                                    double X, double rho, double T) {
    double logT = log10(T);
    double logR = log10(rho) - 3.0 * logT + 18.0;
    return kap_eval(kv, X, logT, logR);
}

#undef KAP_HD
