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
    std::vector<double> h_dy_ds;       // y'(s) = dy/ds at CGL nodes (1.0 for identity map)
    std::vector<double> h_s_cgl;       // CGL nodes in the stretched coordinate s ∈ [0, Ly]
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
    double* d_rho_prime_over_rho = nullptr;  // ρ₀'/ρ₀ on CGL nodes (ny,)
    double* d_N2        = nullptr;   // Brunt-Väisälä² on CGL nodes (ny,)
    bool    is_anelastic = false;    // set by set_background if ρ₀ non-uniform

    // Chebyshev differentiation matrix on [0, Ly] (ny × ny, col-major for DGEMM).
    // Uploaded once in set_background, reused every RK substep.
    double* d_Dy = nullptr;

    // ── y-direction spectral filter (optional) ───────────────────────────
    // Boyd-Vandeven exponential filter applied in y after every step:
    //   σ(n) = 1                             if n ≤ n_cut
    //        = exp(-α · ((n − n_cut)/(N − n_cut))^s)   otherwise
    // The filter is a pre-compiled (ny × ny) real matrix Q that maps a
    // physical field on CGL nodes to itself, stored col-major for DGEMM.
    //
    // Three basis variants (operator-convergence comparison, Phase 1e):
    //   "cheb" : Q = F_T⁻¹ · diag(σ) · F_T     (standard Chebyshev-T DCT-I)
    //   "sl"   : Q = Ψ · diag(σ) · diag(w_cc) · Ψᵀ  (Dirichlet SL basis,
    //            which is also the pressure-projection basis)
    //   "evp"  : Q = V_R · diag(σ) · V_R⁻¹     (2D g-mode EVP eigenvectors
    //            at kx_phys = filter_evp_kx; these are the eigenvectors
    //            that the EVP IC uses, so filtering in this basis keeps
    //            each n_g-mode's amplitude an exact invariant — the truly
    //            "operator-aligned" filter)
    //
    // Activated when filter_alpha > 0 (set via ANSL_FILTER_ALPHA env var).
    enum class FilterBasis { CHEB, SL, EVP };
    // Coordinate map s∈[0,Ly] → y∈[0,Ly] for SL regularisation near walls.
    // "identity" : y = s (no change, legacy path)
    // "tanh"     : y(s) = Ly/2 + (Ly/2) · tanh(β(2s/Ly − 1)) / tanh(β),
    //              geometric stretch that clusters nodes near both walls
    //              independently of ρ₀
    // "logrho"   : y(s) = ρ₀⁻¹(exp(−s)), requires monotone ρ₀(y); makes W̃
    //              bounded for Lane-Emden surface-type singularities
    enum class CoordMap { IDENTITY, TANH, LOGRHO };
    CoordMap coord_map = CoordMap::IDENTITY;
    double   coord_beta = 2.0;  // tanh stretch strength

    double* d_y_filter = nullptr;
    FilterBasis filter_basis = FilterBasis::CHEB;
    double  filter_alpha  = 0.0;     // 0 disables the filter
    int     filter_s      = 16;
    double  filter_cut_frac = 2.0 / 3.0;  // n_cut = floor(N · cut_frac)
    double  filter_evp_kx = 0.0;     // kx_phys for EVP-basis filter (only used if basis=EVP)

    // ── Eigenmode-deviation diagnostic (Phase 1e) ───────────────────────
    // If set, stores the IC eigenmode shape V_EVP(y) · sin(kx·x) on interior
    // nodes; step() then computes ‖v − (V·c(t))‖ / ‖V‖ where c(t) is the
    // L²-projection onto the IC direction.  Quantifies scattering without
    // FFT windowing artefacts.
    std::vector<double> h_eigmode_v;     // (ncell,) IC v field
    double  eigmode_norm = 0.0;
    double  eigmode_omega = 0.0;

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

    // Seed an exact 2D g-mode eigenmode as IC.  Calls compute_2d_gmode_evp
    // internally for (kx_phys = kx_int·2π/Lx), picks the n_g-th eigenvalue
    // (1-based, n_g=1 is largest ω²), reconstructs v(x,y) = V(y)·sin(kx·x),
    // then enforces ∇·(ρ₀ u) = 0 → u(x,y) = (1/(ρ₀ kx))·∂y(ρ₀ V)·cos(kx·x),
    // and b(x,y) = -(N²/ω²)·v(x,y) (linear eigenmode balance).  Returns the
    // EVP ω² so caller can compare probe FFT frequency.
    //
    // Assumes stratified_n2 background (constant ρ₀=1, constant N²); under
    // those assumptions the Boussinesq dispersion ω² = N²·k_x²/(k_x²+k_y²)
    // must be reproduced to machine precision by the EVP.
    double init_gmode_eigenmode(int kx_int, int n_g, double amp);

    // Main API
    double step();
    void download_uv(std::vector<double>& h_u, std::vector<double>& h_v);
    void download_omega(std::vector<double>& h_omega);
    void download_y(std::vector<double>& y_out) const { y_out = h_y_cgl; }

    // SL-Poisson solve.  Reads from d_rhs_pi (ny × nx physical RHS) and writes
    // the solution π (for reduced-pressure form) to d_pi.
    void sl_poisson_solve();

    // Apply y-direction spectral filter in-place (ny × nx row-major field).
    // No-op if d_y_filter is null (filter_alpha == 0).
    void apply_y_filter(double* d_field);

    // Eigenmode-deviation diagnostic:  deviation = ‖v − a(t)·V‖ / ‖V‖
    // where a(t) = ⟨v, V⟩ / ⟨V, V⟩ (L² projection onto IC direction).
    // Returns NaN if the IC was not set via init_gmode_eigenmode.
    double eigmode_deviation();

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
    // q-space (reduced-pressure) 2D g-mode EVP — operator-consistent path
    //
    // Derivation (docs/qspace_reduced_pressure_algebra_2026-05-03.md):
    // Substituting φ = ρ₀·V̂ into the usual v-space EVP
    //     -(ρ₀ V̂)'' + k² ρ₀ V̂ = k² N² ρ₀ V̂ / ω²
    // yields a density-free Helmholtz eigenproblem
    //     -φ''(y) + k² φ(y) = (k² N²(y) / ω²) φ(y),
    // with Dirichlet φ(0) = φ(L_y) = 0. The natural basis is pure Fourier
    //     ϕ_n(y) = √(2/L_y) sin(nπy/L_y),  μ_n = (nπ/L_y)².
    //
    // Projecting onto ϕ_n (sampling with Clenshaw-Curtis weights on the same
    // CGL grid as the TD pipeline) collapses to the generalised EVP
    //     (diag(μ + k²) − (k²/ω²) H) c = 0,
    //     H_{nm} = Σ_k w_cc[k] · ϕ_n(y_k) · N²(y_k) · ϕ_m(y_k).
    // This is n_modes × n_modes (much smaller than ny × ny), and the matrix
    // coefficients contain no ρ₀ — so it matches exactly what the Boussinesq
    // SL-Poisson pipeline sees when the background happens to be uniform.
    //
    // Outputs on host:
    //   omega_sq : descending ω² (length n_modes)
    //   phi_modes: (ny, n_modes) row-major, φ(y) = ρ₀ V̂ reconstructed on the
    //              CGL grid (interior nodes are non-zero, walls forced to 0).
    //              To recover physical V̂(y), divide row-wise by ρ₀(y).
    // Requires init() + set_background() already called.
    void compute_2d_gmode_evp_qspace(double kx_phys,
                                     int n_modes,
                                     std::vector<double>& omega_sq,
                                     std::vector<double>& phi_modes);

    // ────────────────────────────────────────────────────────────────────
    // SL-basis q-space EVP — the truly operator-consistent path
    //
    // Galerkin projection of  -φ'' + k² φ = (k² N²/ω²) φ,  φ = ρ₀·V̂
    // onto the SL basis {ψ_n} already used by sl_poisson_solve().  Using
    // -ψ_n'' = μ_n ψ_n - W̃ ψ_n, expanding φ = Σ c_n ψ_n gives
    //     (diag(μ + k²) − W̃_matrix) c = (k²/ω²) H c
    // with  W̃_{nm} = ⟨ψ_n, W̃ ψ_m⟩_cc,  H_{nm} = ⟨ψ_n, N² ψ_m⟩_cc.
    // Because the TD SL-Poisson pipeline uses the same {ψ_n}, μ_n, w_cc,
    // the EVP eigenvector is an exact invariant direction of the
    // linear TD operator up to time-splitting error.
    //
    // Outputs phi_modes in V̂-space already (caller doesn't divide by ρ₀):
    //   for each mode k, V̂(y) = (Σ_n c_n ψ_n(y)) / ρ₀(y).
    void compute_2d_gmode_evp_qspace_sl(double kx_phys,
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
