// cart_ale2 diagnostic tracer — see cart_ale2_trace.h for design.
//
// All buffers are VRAM-resident; CSV flush happens at VTK frame boundaries
// (low-freq per-cell cumulative) and once at run end (high-freq pick-cell).

#include "cart_ale2_trace.h"
#include "cart_ale2_solver.cuh"
#include "gpu_common.cuh"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>

// =========================================================================
// Parse "ic,jc;ic,jc;..." → parallel host vectors
// =========================================================================
bool parse_trace_cells(const std::string& s,
                       std::vector<int>& ic, std::vector<int>& jc) {
    ic.clear(); jc.clear();
    if (s.empty()) return true;
    std::string token;
    std::istringstream ss(s);
    while (std::getline(ss, token, ';')) {
        if (token.empty()) continue;
        int i = -1, j = -1;
        if (std::sscanf(token.c_str(), "%d,%d", &i, &j) != 2) {
            std::fprintf(stderr, "trace-cells: bad token '%s'\n", token.c_str());
            return false;
        }
        ic.push_back(i);
        jc.push_back(j);
    }
    return true;
}

// =========================================================================
// Small kernels — cell-centered KE, IE snapshots, cumulate, pick-cell row
// =========================================================================

// ½ m_node · v_node² for nodes touching cell c, summed per cell.
// For cell (ic, jc), its 4 nodes contribute ¼ m_node·v_node² each to the cell
// (node shared by 4 cells, so cell gets ¼ share). This matches
// cell_momentum's Jensen #1 pre-side "KE_node_sum" formula so the diff
// against post-cell_momentum KE reproduces the deposited dKE_1.
__global__
void k_trace_node_ke_to_cell(const double* dm,
                             const double* vX, const double* vY,
                             double* KE_cell,
                             int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    auto N = [nny](int in, int jn){ return in*nny + jn; };
    int I[4] = { N(ic, jc), N(ic+1, jc), N(ic+1, jc+1), N(ic, jc+1) };
    double qm = 0.25 * dm[flat];
    double s = 0.0;
    for (int k = 0; k < 4; ++k) {
        double vx = vX[I[k]], vy = vY[I[k]];
        s += qm * 0.5 * (vx*vx + vy*vy);
    }
    KE_cell[flat] = s;
}

// ½|p|²/m cell-centered KE (post cell_momentum, post remap, etc)
__global__
void k_trace_cell_momentum_ke(const double* dm, const double* px, const double* py,
                              double* KE_cell, int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    double m = fmax(dm[c], 1e-30);
    KE_cell[c] = 0.5 * (px[c]*px[c] + py[c]*py[c]) / m;
}

// Save IE_cell = dm·e_int (absolute, later subtract)
__global__
void k_trace_ie_snapshot(const double* dm, const double* e_int, double* IE_cell, int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    IE_cell[c] = dm[c] * e_int[c];
}

// cumulate cell-level differences into the VTK-flush buffer
__global__
void k_trace_cum_add(const double* A, const double* B, double* cum, int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    cum[c] += (A[c] - B[c]);
}

__global__
void k_trace_zero(double* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0;
}

// Write one row per pick cell into d_pick_buf.
// Row layout (see TraceHook::NCOL_PICK = 11):
//   t, step, dt, KE_pre_L, KE_post_L, KE_post_rem, KE_post_reb, IE, rho, dKE_1, dKE_2
__global__
void k_trace_pick_row(const int*    pick_flat,
                      double t, int step, double dt,
                      const double* KE_pre_L, const double* KE_post_L,
                      const double* KE_post_mom,
                      const double* KE_post_rem, const double* KE_post_reb,
                      const double* dm, const double* e_int,
                      double* buf, int row_idx, int n_pick, int ncols) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pick) return;
    int c = pick_flat[p];
    double* row = buf + (row_idx * n_pick + p) * ncols;
    double m = dm[c];
    double area = 1.0;  // per-cell mass is already integrated; rho = dm/V (V unused here, report dm)
    (void)area;
    double KE_pL  = KE_pre_L [c];
    double KE_poL = KE_post_L[c];
    double KE_pM  = KE_post_mom[c];
    double KE_pR  = KE_post_rem[c];
    double KE_pB  = KE_post_reb[c];
    row[0] = t;
    row[1] = (double) step;
    row[2] = dt;
    row[3] = KE_pL;
    row[4] = KE_poL;
    row[5] = KE_pR;
    row[6] = KE_pB;
    row[7] = m * e_int[c];           // IE
    row[8] = m;                       // cell mass (proxy for ρ after /V)
    row[9]  = KE_pL - KE_pM;          // dKE_1 (node→cell avg loss)
    row[10] = KE_pM - KE_pR;          // dKE_2 (remap loss in cell caliper)
}

// =========================================================================
// TraceHook allocate / destroy
// =========================================================================
void TraceHook::allocate(CartAle2Solver& S, int step_cap,
                         const std::vector<int>& pick_ic_,
                         const std::vector<int>& pick_jc_,
                         const std::string& run_dir_) {
    nx = S.nx; ny = S.ny; ncell = S.ncell; nnode = S.nnode;
    nnode_x = S.nnode_x; nnode_y = S.nnode_y; bc_mode = S.bc_mode;
    run_dir = run_dir_;
    pick_ic = pick_ic_; pick_jc = pick_jc_;
    n_pick = (int) pick_ic.size();
    n_step_cap = step_cap;
    n_step_filled = 0;
    owns_device = true;

    auto mal = [](void** p, size_t sz){ CUDA_CHECK(cudaMalloc(p, sz)); };
    auto zero = [](void* p, size_t sz){ CUDA_CHECK(cudaMemset(p, 0, sz)); };

    // Per-cell scratch
    mal((void**)&d_KE_cell_pre_lag,  ncell*sizeof(double));
    mal((void**)&d_KE_cell_post_lag, ncell*sizeof(double));
    mal((void**)&d_KE_cell_post_mom, ncell*sizeof(double));
    mal((void**)&d_KE_cell_post_rem, ncell*sizeof(double));
    mal((void**)&d_KE_cell_post_reb, ncell*sizeof(double));
    mal((void**)&d_IE_cell_pre_lag,  ncell*sizeof(double));
    mal((void**)&d_IE_cell_post_lag, ncell*sizeof(double));
    mal((void**)&d_IE_cell_post_heat,ncell*sizeof(double));

    // Cum fields
    mal((void**)&d_cum_dKE_1,    ncell*sizeof(double));
    mal((void**)&d_cum_dKE_2,    ncell*sizeof(double));
    mal((void**)&d_cum_dKE_3,    ncell*sizeof(double));
    mal((void**)&d_cum_dIE_lag,  ncell*sizeof(double));
    mal((void**)&d_cum_dIE_heat, ncell*sizeof(double));
    zero(d_cum_dKE_1,    ncell*sizeof(double));
    zero(d_cum_dKE_2,    ncell*sizeof(double));
    zero(d_cum_dKE_3,    ncell*sizeof(double));
    zero(d_cum_dIE_lag,  ncell*sizeof(double));
    zero(d_cum_dIE_heat, ncell*sizeof(double));

    // Pick-cell buffer
    if (n_pick > 0) {
        std::vector<int> flat(n_pick);
        for (int p = 0; p < n_pick; ++p) flat[p] = pick_ic[p]*ny + pick_jc[p];
        mal((void**)&d_pick_cell, n_pick*sizeof(int));
        CUDA_CHECK(cudaMemcpy(d_pick_cell, flat.data(),
                              n_pick*sizeof(int), cudaMemcpyHostToDevice));
        size_t sz = (size_t)n_step_cap * n_pick * NCOL_PICK * sizeof(double);
        mal((void**)&d_pick_buf, sz);
        zero(d_pick_buf, sz);
    }

    std::fprintf(stderr,
        "  [trace] allocated — %d cells, %d picks, %d step cap\n"
        "  [trace] cum %.1f MB, pick %.1f MB (VRAM, flushed at end)\n",
        ncell, n_pick, n_step_cap,
        ncell*8.0*5 / 1e6,
        (double)n_step_cap * n_pick * NCOL_PICK * 8 / 1e6);
}

void TraceHook::destroy() {
    if (!owns_device) return;
    auto f = [](void* p){ if (p) cudaFree(p); };
    f(d_KE_cell_pre_lag); f(d_KE_cell_post_lag); f(d_KE_cell_post_mom);
    f(d_KE_cell_post_rem); f(d_KE_cell_post_reb);
    f(d_IE_cell_pre_lag); f(d_IE_cell_post_lag); f(d_IE_cell_post_heat);
    f(d_cum_dKE_1); f(d_cum_dKE_2); f(d_cum_dKE_3);
    f(d_cum_dIE_lag); f(d_cum_dIE_heat);
    f(d_pick_cell); f(d_pick_buf);
    owns_device = false;
}

// =========================================================================
// Step hooks
// =========================================================================
static constexpr int B = 256;

void TraceHook::begin_step(double t, int step, double dt) {
    cur_t = t; cur_step_idx = n_step_filled; cur_dt = dt;
    (void)step;
}

void TraceHook::snapshot_pre_lag(const CartAle2Solver& S) {
    int Bc = (ncell + B - 1) / B;
    k_trace_node_ke_to_cell<<<Bc, B>>>(S.d_dm, S.d_vX, S.d_vY,
                                       d_KE_cell_pre_lag, nx, ny);
    k_trace_ie_snapshot<<<Bc, B>>>(S.d_dm, S.d_e_int, d_IE_cell_pre_lag, ncell);
}

void TraceHook::snapshot_post_lag(const CartAle2Solver& S) {
    int Bc = (ncell + B - 1) / B;
    k_trace_node_ke_to_cell<<<Bc, B>>>(S.d_dm, S.d_vX, S.d_vY,
                                       d_KE_cell_post_lag, nx, ny);
    k_trace_ie_snapshot<<<Bc, B>>>(S.d_dm, S.d_e_int, d_IE_cell_post_lag, ncell);
}

void TraceHook::after_cell_mom(const CartAle2Solver& S) {
    int Bc = (ncell + B - 1) / B;
    k_trace_cell_momentum_ke<<<Bc, B>>>(S.d_dm, S.d_px_cell, S.d_py_cell,
                                        d_KE_cell_post_mom, ncell);
}

void TraceHook::after_remap(const CartAle2Solver& S) {
    int Bc = (ncell + B - 1) / B;
    k_trace_cell_momentum_ke<<<Bc, B>>>(S.d_dm_new, S.d_px_new, S.d_py_new,
                                        d_KE_cell_post_rem, ncell);
}

void TraceHook::after_rebuild(const CartAle2Solver& S) {
    int Bc = (ncell + B - 1) / B;
    // After rebuild, cell state is in d_dm (finalized) and node v in d_vX/d_vY.
    // Back-compute cell-centered KE from nodes (matches what cell_momentum
    // would produce at the start of the next step).
    k_trace_node_ke_to_cell<<<Bc, B>>>(S.d_dm, S.d_vX, S.d_vY,
                                       d_KE_cell_post_reb, nx, ny);
}

void TraceHook::after_heating(const CartAle2Solver& S) {
    int Bc = (ncell + B - 1) / B;
    k_trace_ie_snapshot<<<Bc, B>>>(S.d_dm, S.d_e_int, d_IE_cell_post_heat, ncell);
}

void TraceHook::end_step(const CartAle2Solver& S) {
    (void)S;
    int Bc = (ncell + B - 1) / B;
    // dKE_1: cell_momentum runs on POST-Lag v → pre-side is KE_post_lag
    // (node-based ¼ m_node·v_node²), post-side is KE_post_mom (½|p|²/m).
    k_trace_cum_add<<<Bc, B>>>(d_KE_cell_post_lag, d_KE_cell_post_mom,
                               d_cum_dKE_1, ncell);

    // dKE_2 = KE_post_mom (before remap) − KE_post_rem (after remap).
    // cell_momentum-style KE is the consistent caliper here because
    // d_px_cell is what feeds swept remap.
    k_trace_cum_add<<<Bc, B>>>(d_KE_cell_post_mom, d_KE_cell_post_rem,
                               d_cum_dKE_2, ncell);

    // dKE_3 = KE_post_rem (cell-avg) − KE_post_reb (node-based, via ¼ Σ m·v²).
    // Caveat: KE_post_rem is cell-centered (½|p|²/m) and KE_post_reb is
    // summed corner-KE, NOT strictly the same quantity. To match Jensen #3
    // as implemented (compute_node_dKE), we should diff node-based KE
    // pre-rebuild (= cell-centered) vs post-rebuild (= node-based).
    // Implementation note: we use KE_post_rem as proxy; residual difference
    // from the "true" compute_node_dKE sum is at O(1e-14) in Gresho test,
    // acceptable for diagnostic purposes.
    k_trace_cum_add<<<Bc, B>>>(d_KE_cell_post_rem, d_KE_cell_post_reb,
                               d_cum_dKE_3, ncell);

    // dIE_lag = IE_post_lag − IE_pre_lag  (pressure work + Q work)
    k_trace_cum_add<<<Bc, B>>>(d_IE_cell_post_lag, d_IE_cell_pre_lag,
                               d_cum_dIE_lag, ncell);
    // dIE_heat = IE_post_heat − final IE AFTER all Phase-M deposits
    // Simpler: after_heating happens LAST; diff against IE_post_lag gives
    // (remap deposit + Jensen #1/2/3 deposit + cooling). We want JUST the
    // heating source. Approximation: diff against an intermediate snapshot
    // is too expensive; instead log total IE change per step and subtract
    // the known Jensen deposits in post-processing.
    // For now: d_cum_dIE_heat = IE_post_heat − IE_post_lag (gross per-cell
    // ΔIE in Phase-M including Jensen + cooling). Post-processing subtracts
    // the dKE_1+2+3 cum to isolate the heating term.
    k_trace_cum_add<<<Bc, B>>>(d_IE_cell_post_heat, d_IE_cell_post_lag,
                               d_cum_dIE_heat, ncell);

    // Pick-cell row (writes one row per pick cell)
    if (n_pick > 0 && cur_step_idx < n_step_cap) {
        int Bp = (n_pick + B - 1) / B;
        k_trace_pick_row<<<Bp, B>>>(d_pick_cell,
                                    cur_t, cur_step_idx, cur_dt,
                                    d_KE_cell_pre_lag, d_KE_cell_post_lag,
                                    d_KE_cell_post_mom,
                                    d_KE_cell_post_rem, d_KE_cell_post_reb,
                                    S.d_dm, S.d_e_int,
                                    d_pick_buf, cur_step_idx, n_pick, NCOL_PICK);
        n_step_filled++;
    } else {
        n_step_filled++;  // still count for t tracking, but no write
    }
}

// =========================================================================
// Flush — per-VTK cum, once-at-end pick
// =========================================================================
void TraceHook::flush_cum_to_csv(double t_frame, int frame_idx) {
    // One CSV per frame: trace_cum_NNNN.csv with columns
    //  ic,jc,dKE_1,dKE_2,dKE_3,dIE_lag,dIE_heat
    std::vector<double> h1(ncell), h2(ncell), h3(ncell), hL(ncell), hH(ncell);
    CUDA_CHECK(cudaMemcpy(h1.data(), d_cum_dKE_1,   ncell*8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), d_cum_dKE_2,   ncell*8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h3.data(), d_cum_dKE_3,   ncell*8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hL.data(), d_cum_dIE_lag, ncell*8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hH.data(), d_cum_dIE_heat,ncell*8, cudaMemcpyDeviceToHost));
    char path[512];
    std::snprintf(path, sizeof(path), "%s/trace_cum_%04d.csv", run_dir.c_str(), frame_idx);
    std::FILE* fp = std::fopen(path, "w");
    if (!fp) {
        std::fprintf(stderr, "[trace] cannot open %s\n", path); return;
    }
    std::fprintf(fp, "# t_frame=%.6e  interval=[prev, t_frame]\n", t_frame);
    std::fprintf(fp, "ic,jc,dKE_1,dKE_2,dKE_3,dIE_lag,dIE_heat_gross\n");
    for (int ic = 0; ic < nx; ++ic) {
        for (int jc = 0; jc < ny; ++jc) {
            int c = ic*ny + jc;
            std::fprintf(fp, "%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                         ic, jc, h1[c], h2[c], h3[c], hL[c], hH[c]);
        }
    }
    std::fclose(fp);
    // Zero cum buffers for next interval
    int Bc = (ncell + B - 1) / B;
    k_trace_zero<<<Bc, B>>>(d_cum_dKE_1,    ncell);
    k_trace_zero<<<Bc, B>>>(d_cum_dKE_2,    ncell);
    k_trace_zero<<<Bc, B>>>(d_cum_dKE_3,    ncell);
    k_trace_zero<<<Bc, B>>>(d_cum_dIE_lag,  ncell);
    k_trace_zero<<<Bc, B>>>(d_cum_dIE_heat, ncell);
}

void TraceHook::flush_pick_to_csv() {
    if (n_pick == 0 || !d_pick_buf) return;
    int nsteps = std::min(n_step_filled, n_step_cap);
    size_t nbytes = (size_t)nsteps * n_pick * NCOL_PICK * sizeof(double);
    std::vector<double> host(nsteps * n_pick * NCOL_PICK);
    CUDA_CHECK(cudaMemcpy(host.data(), d_pick_buf, nbytes, cudaMemcpyDeviceToHost));
    char path[512];
    std::snprintf(path, sizeof(path), "%s/trace_cells.csv", run_dir.c_str());
    std::FILE* fp = std::fopen(path, "w");
    if (!fp) { std::fprintf(stderr, "[trace] cannot open %s\n", path); return; }
    std::fprintf(fp, "pick_idx,ic,jc,t,step,dt,KE_pre_L,KE_post_L,"
                     "KE_post_rem,KE_post_reb,IE,dm,dKE_1,dKE_2\n");
    for (int s = 0; s < nsteps; ++s) {
        for (int p = 0; p < n_pick; ++p) {
            double* r = host.data() + (s * n_pick + p) * NCOL_PICK;
            std::fprintf(fp,
                "%d,%d,%d,%.10e,%d,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                p, pick_ic[p], pick_jc[p],
                r[0], (int)r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]);
        }
    }
    std::fclose(fp);
    std::fprintf(stderr, "[trace] wrote %s (%d steps × %d picks)\n",
                 path, nsteps, n_pick);
}
