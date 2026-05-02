#pragma once

// Helmholtz free-energy EOS (Timmes & Swesty 2000, cococubed.com).
//
// Public-domain community-standard EOS covering partial degenerate electrons,
// radiation, and ideal ions. Sits alongside our other EOS types (IDEAL,
// IDEAL_RAD, PRE_MS) as the high-fidelity option for stellar interiors.
//
// Table layout (following Timmes):
//   log10 T  grid, imax=211  (T = 1e3 … 1e13 K)
//   log10 ρ  grid, jmax=71   (ρ = 1e-12 … 1e15 g/cc)
//   9 tabulated functions of (log T, log ρ):
//     f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt
//   plus two 1D tables (positron, coulomb) — skipped in this minimal port.
//
// Biquintic Hermite interpolation in (log T, log ρ) gives C² continuous
// pressure + its derivatives — essential for JFNK Newton stability.
//
// GPU placement:
//   Tables are malloc'd via cudaMalloc and pinned to L2 persisting cache
//   (sm_80+; on sm_89 RTX 40-series we get up to 44 MB persisting). The
//   5 MB Helmholtz table stays in L2 across every BE solve, giving us
//   ~50× DRAM bandwidth for JFNK GMRES iterations.

#include <cstddef>
#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define HE_HD __host__ __device__
#else
#define HE_HD
#endif

// Resolution of the cococubed table.
// NOTE Timmes convention: i indexes log ρ (inner loop), j indexes log T
// (outer loop). Two resolutions ship with helmholtz.tbz:
//   "standard"    : imax=271, jmax=101    (file ~15 MB)
//   "twice dense" : imax=541, jmax=201    (file ~60 MB, current default)
// We hard-code the twice-dense resolution since that is what the
// distributed helm_table.dat provides.
static constexpr int HELM_IMAX = 541;   // log ρ points (inner)
static constexpr int HELM_JMAX = 201;   // log T points (outer)

// POD view of table arrays on device. Passed by value to kernels.
// Pointers live in GPU memory; lifetime owned by HelmholtzTable (below).
//
// Layout: each 2D table stored row-major as [j * IMAX + i] where
//   i ∈ [0, IMAX) indexes log ρ (inner loop on read)
//   j ∈ [0, JMAX) indexes log T (outer loop on read)
// This matches Timmes' Fortran helm_table.dat file order directly.
struct HelmholtzTableView {
    // ---- Table 1: Helmholtz free energy (9 entries per grid point) ----
    const double* f;       // free energy
    const double* fd;      // ∂f/∂ρ
    const double* ft;      // ∂f/∂T
    const double* fdd;     // ∂²f/∂ρ²
    const double* ftt;     // ∂²f/∂T²
    const double* fdt;     // ∂²f/∂ρ ∂T
    const double* fddt;    // ∂³f/∂ρ² ∂T
    const double* fdtt;    // ∂³f/∂ρ ∂T²
    const double* fddtt;   // ∂⁴f/∂ρ² ∂T²
    // ---- Table 2: dP/dρ and its derivatives (4 entries) ----
    const double* dpdf;
    const double* dpdfd;
    const double* dpdft;
    const double* dpdfdt;
    // ---- Table 3: electron chemical potential (4 entries) ----
    const double* ef;
    const double* efd;
    const double* eft;
    const double* efdt;
    // ---- Table 4: electron-positron number density (4 entries) ----
    const double* xf;
    const double* xfd;
    const double* xft;
    const double* xfdt;

    // Grid bookkeeping — Timmes helmholtz.f90 uses:
    //   log T ∈ [3, 13]  with jmax points (outer index j)
    //   log ρ ∈ [-12, 15] with imax points (inner index i)
    double tlo, thi, dt, dti;   // log10 T grid (j axis)
    double dlo, dhi, dd, ddi;   // log10 ρ grid (i axis)

    // Composition. Helmholtz is composition-agnostic; Abar/Zbar only appear
    // in the ion + coulomb parts we layer on top of the electron table.
    // For pure ionized H:  Abar=1, Zbar=1.  Solar ~0.73X+0.25Y+Z:
    // Abar≈1.25, Zbar≈1.14.
    double Abar;
    double Zbar;
};

// Host-side owner that allocates + uploads + pins in L2.
struct HelmholtzTable {
    double* d_data = nullptr;   // single allocation holding all 9 tables
    size_t  bytes  = 0;
    HelmholtzTableView view;
    cudaStream_t stream = 0;    // stream used for L2 persisting window

    // Load the cococubed ASCII "helm_table.dat" (or binary .bin), upload to
    // GPU, and set up L2 persisting cache window on `stream`. Returns 0 on
    // success, non-zero otherwise.
    int load(const char* ascii_path, const char* bin_cache_path, cudaStream_t s = 0);

    // Reset L2 persisting config + free device memory.
    void destroy();
};

// Device-side interpolation helpers.
//
// Biquintic interpolation per Timmes "helmeos.f" uses 36 coefficients; the
// minimal reduction is to use the bicubic Hermite form with (f, fd, ft, fdt)
// at the 4 corners of the enclosing rectangle.  That gives C¹ continuous
// result. For C² we'd need fdd/ftt/fddtt too.
//
// First pass (this file): bicubic Hermite only. Biquintic upgrade kept for
// after baseline correctness is proven.
HE_HD inline int helm_index_log_T(const HelmholtzTableView& tv, double log_T) {
    int i = (int)((log_T - tv.tlo) * tv.dti);
    if (i < 0) i = 0;
    if (i > HELM_IMAX - 2) i = HELM_IMAX - 2;
    return i;
}
HE_HD inline int helm_index_log_rho(const HelmholtzTableView& tv, double log_rho) {
    int j = (int)((log_rho - tv.dlo) * tv.ddi);
    if (j < 0) j = 0;
    if (j > HELM_JMAX - 2) j = HELM_JMAX - 2;
    return j;
}
// Row-major index: j (log T) is the outer axis, i (log ρ) inner.
HE_HD inline int helm_idx2d(int j, int i) { return j * HELM_IMAX + i; }

// Placeholder scalar pressure — full Helmholtz physics will land in a
// follow-up commit. For now: returns analytic ideal-gas + radiation so the
// Helmholtz-enabled code path stays equivalent to --eos ideal_rad while we
// build up table plumbing + L2 pinning.
HE_HD inline double helm_pressure_stub(double rho, double T, double Abar, double radiation_a) {
    // R_gas / Abar · ρ T  +  a T⁴ / 3
    double Rgas_over_A = 1.0 / (Abar > 1e-30 ? Abar : 1e-30);
    return rho * Rgas_over_A * T + radiation_a * T * T * T * T / 3.0;
}

#undef HE_HD
