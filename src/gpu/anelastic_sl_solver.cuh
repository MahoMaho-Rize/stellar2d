#pragma once

#include <cufft.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <vector>

// 2D anelastic / Boussinesq solver with Sturm-Liouville (SL) spectral
// diagonalisation of the variable-density pressure Poisson equation.
//
// Geometry: periodic in x, Dirichlet (bounded) in y.  Domain [0, Lx] × [0, Ly].
//
// Pressure PDE (reduced-pressure form, π = p / ρ_0):
//
//     ∇ · (ρ_0(y) ∇π) = ∇ · (ρ_0 F),
//
// discretised as
//
//     FFT_x → weight by ρ_0^{-1/2} → DGEMM(Ψ^T) → divide by (μ_n + k_x²)
//         → DGEMM(Ψ) → weight by ρ_0^{-1/2} → IFFT_x.
//
// The SL basis {(μ_n, ψ_n)} diagonalises T = d²/dy² + W̃(y) once and for all,
// where W̃ = (ρ_0')² / (4 ρ_0²) − ρ_0'' / (2 ρ_0).
//
// For the Boussinesq limit (ρ_0 = const), W̃ ≡ 0 and the SL basis degenerates
// to sin(n π y / L_y), i.e.\ a Fourier basis.  This is a sanity-check test.
//
// Design: brand-new struct, brand-new file.  Reuses ideas (IFRK3, VRAM frame
// buffer, R2C-in-x) from pseudo_spectral_solver but shares no state.

struct AnelasticSLSolver {
    // ── Grid ─────────────────────────────────────────────────────────────
    int nx = 0;              // x-direction grid points (periodic)
    int ny = 0;              // y-direction grid points (includes both Dirichlet endpoints)
    int nh = 0;              // nx/2 + 1, R2C-in-x complex samples
    int ncell = 0;           // nx * ny
    int ncplx = 0;           // nh * ny  (y is the slow axis for GEMM efficiency)
    int n_modes = 0;         // number of SL modes retained for the Poisson solve

    double Lx = 1.0, Ly = 1.0;
    double dx = 0.0, dy_ref = 0.0;   // dy_ref is only a reference spacing (non-uniform CGL)

    // ── Physics ──────────────────────────────────────────────────────────
    double nu        = 1e-4;       // kinematic viscosity
    double cfl       = 0.5;
    double dt_max    = 5e-3;
    double dt_min    = 1e-12;
    double dt_current = 0.0;
    int    hyper_p   = 1;

    // ── Background stratification ρ_0(y) (host-side reference) ──────────
    //   Uploaded to device as d_rho, d_rho_sqrt_inv (= 1/√ρ_0), d_W (Liouville).
    std::vector<double> h_y_cgl;       // Chebyshev-Gauss-Lobatto nodes on [0, Ly]  (ascending)
    std::vector<double> h_rho;         // ρ_0 at CGL nodes
    std::vector<double> h_W_tilde;     // reduced-pressure Liouville potential
    std::vector<double> h_mu;          // SL eigenvalues μ_n (ascending)
    std::vector<double> h_Psi;         // (ny, n_modes) column-major, CGL-grid eigenfunctions
    std::vector<double> h_cc_weights;  // Clenshaw-Curtis quadrature weights
    std::vector<double> h_Dy_row;      // D (row-major, ny × ny) on [0, Ly] — kept for 2D EVP assembly
    std::vector<double> h_rho_prime;   // ρ₀'(y) on CGL nodes
    std::vector<double> h_N2;          // N²(y) on CGL nodes (0 for Boussinesq)

    // ── Device-side buffers ──────────────────────────────────────────────
    // Physical fields (row-major ny × nx: y is slow, x is fast — consistent
    // with cuFFT R2C-in-x layout and with DGEMM on d_Dy along y).
    double* d_u         = nullptr;   // velocity_x
    double* d_v         = nullptr;   // velocity_y
    double* d_omega     = nullptr;   // vorticity (derived on demand)
    double* d_rhs_pi    = nullptr;   // RHS of Poisson for π (temporary)

    // RK3 scratch (primitive-variable Shu-Osher: need u_orig + rhs_u + rhs_v)
    double* d_u_orig    = nullptr;   // y_n snapshot of u at start of step
    double* d_v_orig    = nullptr;
    double* d_b_orig    = nullptr;   // buoyancy snapshot (Phase 1d)
    double* d_b         = nullptr;   // buoyancy perturbation (anelastic)
    double* d_rhs_u     = nullptr;
    double* d_rhs_v     = nullptr;
    double* d_rhs_b     = nullptr;
    double* d_scratch   = nullptr;   // ∂y* scratch (size ncell)

    // Anelastic background on CGL y-grid.
    double* d_rho_prime = nullptr;   // ∂ρ₀/∂y on CGL nodes (ny,)
    double* d_N2        = nullptr;   // Brunt-Väisälä² on CGL nodes (ny,)
    bool    is_anelastic = false;    // set by set_background if ρ₀ non-uniform

    // Chebyshev differentiation matrix on [0, Ly] (ny × ny, col-major for DGEMM).
    // Uploaded once in set_background, reused every RK substep.
    double* d_Dy = nullptr;

    // Weight vectors (ny)
    double* d_rho               = nullptr;
    double* d_rho_sqrt_inv      = nullptr;  // 1/√ρ_0 on CGL nodes

    // SL basis (device).  Stored as ZGEMM-compatible complex matrices
    // (imaginary part zero) to keep the Poisson pipeline branch-free.
    cufftDoubleComplex* d_Psi_fwd = nullptr;  // (ny, n_modes) col-major, diag(w_cc) @ Psi
    cufftDoubleComplex* d_Psi_inv = nullptr;  // (ny, n_modes) col-major, Psi (plain)
    double* d_mu  = nullptr;   // (n_modes,)
    double* d_cc_weights = nullptr;  // (ny,) Clenshaw-Curtis weights (kept for diagnostics)

    // Spectral buffers (complex, R2C in x), layout matches cuFFT output:
    //   index offset jy*nh + kx  (row-major: y slow, kx fast)
    cufftDoubleComplex* d_fhat       = nullptr;  // ny × nh RHS after FFT_x
    cufftDoubleComplex* d_ghat       = nullptr;  // ny × nh weighted RHS
    cufftDoubleComplex* d_Ghat       = nullptr;  // n_modes × nh coefficients
    cufftDoubleComplex* d_Qhat       = nullptr;  // n_modes × nh after diag divide
    cufftDoubleComplex* d_qhat       = nullptr;  // ny × nh after inverse SL
    cufftDoubleComplex* d_pihat      = nullptr;  // ny × nh weighted again (= π̂(kx, y))
    double* d_pi                     = nullptr;  // ny × nx physical result

    // k_x array (nh,)
    double* d_kx = nullptr;

    // ── Handles ──────────────────────────────────────────────────────────
    cufftHandle plan_r2c_x = 0;   // batched 1D R2C along x, batch = ny
    cufftHandle plan_c2r_x = 0;   // batched 1D C2R along x
    cublasHandle_t cublas = nullptr;

    // ── Diagnostics & frame buffer (VRAM pool, binary VTK flush) ─────────
    double* d_reduce = nullptr;
    int reduce_blocks = 0;

    int step_count = 0;

    // Min dy on the non-uniform CGL grid (for CFL).  Set in set_background.
    double dy_min = 0.0;

    // ── API ──────────────────────────────────────────────────────────────
    void init(int nx_, int ny_, int n_modes_,
              double Lx_, double Ly_,
              double nu_, double cfl_);
    void free_all();
    ~AnelasticSLSolver() { free_all(); }

    // Background profile setup.  Call before init_*.
    //   kind = "boussinesq" (ρ_0 ≡ 1) | "lane_emden_1_5" (Lane-Emden n = 3/2)
    //   rho_cut: surface truncation for Lane-Emden (ignored otherwise)
    void set_background(const std::string& kind, double rho_cut = 0.01);

    // Test-case initial conditions (zero velocity unless otherwise noted)
    void init_zero();
    void init_kh_shear(double vshear, double amp, int k);
    // Lane-Emden n=3 g-mode pulsation: zero u, v drawn from a low-k sinusoid
    // in y (approximates the n_g=1 mode shape without full eigensolve).
    void init_gmode_pulsation(double amp, int k_y);

    // Main API
    double step();
    void download_uv(std::vector<double>& h_u, std::vector<double>& h_v);
    void download_omega(std::vector<double>& h_omega);
    void download_y(std::vector<double>& y_out) const { y_out = h_y_cgl; }

    // SL-Poisson solve.  Reads from d_rhs_pi (ny × nx physical RHS) and writes
    // the solution π (for reduced-pressure form) to d_pi.
    void sl_poisson_solve();

    // Phase 1c time-stepping helpers (primitive-variable + projection).
    // For Phase 1d (anelastic), compute_rhs_uv also fills d_rhs_b with
    //   ∂t b = -(u·∇) b - N² v
    // and adds the buoyancy force b·ê_y to ∂t v.
    void compute_rhs_uv(const double* dU, const double* dV,
                        double* dRU, double* dRV);
    void project_div_free();

    // Download ∇·u (∂x u + ∂y v) to host for diagnostics.  Uses d_scratch.
    void download_divergence(std::vector<double>& h_div);

    // Phase 1d probe: integrate v(x=Lx/2, y=Ly/2) for the pulsation frequency.
    // Returns a single double sampled at the middle grid node.
    double probe_v_center();

    void download_b(std::vector<double>& h_b);

    // ────────────────────────────────────────────────────────────────────
    // 2D anelastic g-mode EVP  (Phase 1e foundation)
    // Solves, for a single x-mode k_x (physical wavenumber, not index):
    //     A v = ω² B v,      A = k_x² diag(N² ρ₀),
    //     B = -∂_y (ρ₀ ∂_y) + k_x² diag(ρ₀),
    // on interior CGL nodes (Dirichlet v = 0 at both walls).
    //
    // Requires init() + set_background() already called so that D_scaled,
    // ρ₀(y), N²(y) are all in place.
    //
    // Outputs are host-side:
    //   omega_sq  : descending ω²  (length n_modes)
    //   v_modes   : (ny-2, n_modes) column-major, real eigenvectors on
    //               interior CGL nodes (extend by zero at walls to reconstruct)
    //
    // Uses cuSOLVER's 64-bit API (cusolverDnXgeev) on M = B⁻¹ A.  Host
    // inversion keeps the GPU path to a single Xgeev of a complex matrix.
    void compute_2d_gmode_evp(double kx_phys,
                              int n_modes,
                              std::vector<double>& omega_sq,
                              std::vector<double>& v_modes);

    // ────────────────────────────────────────────────────────────────────
    // Exp K: 4-var full GYRE-compatible g-mode EVP on CGL grid
    // (see scripts/gmode_exp_k_chebyshev_full.py for derivation)
    //
    // Inputs are per-node stellar structure coefficients (all length Nr),
    // ell is the spherical harmonic degree.
    // Builds a (4Nr × 4Nr) generalised EVP  Q u = ω² P u, solves via
    // M = P⁻¹ Q + cusolverDnXgeev, filters g-modes by propagation-cavity
    // classification (p_frac < 0.05), and returns up to n_modes ω²
    // in descending order (n_g=1 first).
    //
    // Uses its own CGL D matrix built from x_nodes[0], x_nodes.back() — does
    // NOT reuse AnelasticSLSolver's [0, Ly] D.  That's intentional:
    // the g-mode problem is posed on a radial x=r/R ∈ [inner_cut, outer_cut].
    void solve_gmode_full_chebyshev(
        const std::vector<double>& x_nodes,       // CGL x points (Nr,)
        const std::vector<double>& V_2,           // (Nr,)
        const std::vector<double>& U,
        const std::vector<double>& A_star,
        const std::vector<double>& c_1,
        const std::vector<double>& Gamma_1,
        int ell,
        int n_modes_out,
        std::vector<double>& omega_sq,
        std::vector<double>& eigvecs_y1);         // y_1(x) for each mode

    // Manufactured-solution self-test (Phase 1b).  Constructs a known π_exact,
    // computes the analytic RHS, runs the pipeline, measures L2 error.
    // Returns err_L2 and prints a breakdown.  Works for Boussinesq and
    // Lane-Emden backgrounds.
    double manufactured_test();
};
