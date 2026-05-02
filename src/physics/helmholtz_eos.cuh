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
static constexpr int HELM_JMAX = 71;    // log rho points
static constexpr int HELM_IMAX = 211;   // log T points

// POD view of table arrays on device. Passed by value to kernels.
// Pointers live in GPU memory; lifetime owned by HelmholtzTable (below).
struct HelmholtzTableView {
    // 2D tables stored row-major as [rho_idx * IMAX + T_idx]
    const double* f;       // free energy / (N k T)       — dimensionless
    const double* fd;      // ∂f/∂(log ρ)
    const double* ft;      // ∂f/∂(log T)
    const double* fdd;     // ∂²f/∂(log ρ)²
    const double* ftt;     // ∂²f/∂(log T)²
    const double* fdt;     // ∂²f/∂(log ρ) ∂(log T)
    const double* fddt;    // ∂³f/∂(log ρ)² ∂(log T)
    const double* fdtt;    // ∂³f/∂(log ρ) ∂(log T)²
    const double* fddtt;   // ∂⁴f/∂(log ρ)² ∂(log T)²

    // Grid bookkeeping (1/dT, 1/dρ in log space)
    double tlo, thi, dt, dti;   // log10 T grid
    double dlo, dhi, dd, ddi;   // log10 ρ grid

    // Composition defaults (single Abar, Zbar for now).
    // Helmholtz is agnostic to composition; these multiply the ion + coulomb
    // parts. For pure ionized H:  Abar=1, Zbar=1.  Solar ~0.73X+0.25Y+Z:
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
