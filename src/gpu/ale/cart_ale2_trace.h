#pragma once

// Diagnostic tracer hook for cart_ale2 (energy / KE-deposit accounting).
//
// Intended for ALE dissipation studies — lets us observe how KE is lost
// through the 3 Jensen averaging operations in a step:
//   #1 cell_momentum (node v → cell v)
//   #2 remap_finalize_cells (swept-transport mass/momentum → new cell v)
//   #3 compute_node_dKE (cell v → node v)
// plus the Lagrangian-phase IE update (pressure work).
//
// The hook is entirely OPT-IN — cart_ale2_solver calls these functions
// only if solver->trace != nullptr. Zero numerical change when the hook
// is absent; the purpose is purely observability.
//
// All buffers live on-device and flush ONCE at run end (VRAM-buffered,
// matching cart_ale2's own frame buffer approach — no I/O in the hot loop).
//
// Data flow per step:
//   begin_step        → stamp (t, step, dt) for the current row
//   snapshot_pre_lag  → record ½Σ m_node·v_node² BEFORE Lagrangian kick
//   snapshot_post_lag → record AFTER energy_update (Phase-L done)
//   after_cell_mom    → record ½|p_cell|²/m_cell (cell-centered KE, used for #1 check)
//   after_remap       → record post-remap cell KE (used for #2 check)
//   after_rebuild     → record post-rebuild node KE (used for #3 check)
//   end_step          → cumulate per-cell dKE fields, advance pick-cell row

#include <vector>
#include <string>

struct CartAle2Solver;  // forward

struct TraceHook {
    // --- Static config ---
    int nx = 0, ny = 0;
    int ncell = 0;
    int nnode = 0;
    int nnode_x = 0, nnode_y = 0;
    int bc_mode = 0;

    // --- Pick-cell high-freq trace (VRAM ring) ---
    // Each step writes 1 row per pick cell with ncols scalars. Buffer sized
    // once at begin_run based on n_step_cap. Flushed once to CSV at end_run.
    int n_pick = 0;                       // # pick cells
    std::vector<int> pick_ic, pick_jc;    // host: pick cell (ic, jc)
    int*    d_pick_cell = nullptr;        // device: flat cell indices [n_pick]
    int     n_step_cap = 0;               // max steps before buffer overflow
    int     n_step_filled = 0;            // actual steps filled (≤ cap)
    static constexpr int NCOL_PICK = 11;
    // Columns:  0=t 1=step 2=dt 3=KE_pre_L 4=KE_post_L
    //           5=KE_post_rem 6=KE_post_reb 7=IE 8=rho 9=dKE_1 10=dKE_2
    //   (dKE_3 is reconstructable from KE_post_rem−KE_post_reb; we track
    //    the 3 components explicitly in the full-field cum buffer.)
    double* d_pick_buf = nullptr;         // device: [n_step_cap * n_pick * NCOL_PICK]

    // --- Full-field cumulative (per-cell dKE/dIE by type) ---
    // Cleared at each VTK flush. Each field is accumulated in-place over
    // the VTK-interval of steps, giving per-cell time-integrated dissipation
    // between snapshots. Lets us see the SPATIAL distribution.
    double* d_cum_dKE_1   = nullptr;  // Jensen #1 per-cell heat deposit
    double* d_cum_dKE_2   = nullptr;  // Jensen #2
    double* d_cum_dKE_3   = nullptr;  // Jensen #3
    double* d_cum_dIE_lag = nullptr;  // Phase-L (P·dV + Q·div work)
    double* d_cum_dIE_heat= nullptr;  // bottom heating / cooling (apply_cooling)
    double* d_cum_steps   = nullptr;  // step count in this interval (scalar on GPU, ok as single cell)

    // --- Per-step scratch buffers ---
    // Keep previous-stage KE for diffing into cum / pick:
    double* d_KE_cell_pre_lag  = nullptr;  // ½|p|²/m BEFORE Lagrangian kick
    double* d_KE_cell_post_lag = nullptr;  // AFTER Phase-L
    double* d_KE_cell_post_mom = nullptr;  // AFTER cell_momentum (= cell-avg KE)
    double* d_KE_cell_post_rem = nullptr;  // AFTER swept remap finalize
    double* d_KE_cell_post_reb = nullptr;  // AFTER rebuild + compute_node_dKE
    double* d_IE_cell_pre_lag  = nullptr;  // dm·e_int BEFORE Lagrangian
    double* d_IE_cell_post_lag = nullptr;  // AFTER Phase-L
    double* d_IE_cell_post_heat= nullptr;  // AFTER apply_cooling

    // Current step metadata (host copies, uploaded per step)
    double cur_t = 0.0, cur_dt = 0.0;
    int    cur_step_idx = 0;  // row in pick-buf

    // --- Output ---
    std::string run_dir;
    bool owns_device = false;

    // --- Lifecycle ---
    void allocate(CartAle2Solver& S, int step_cap,
                  const std::vector<int>& pick_ic_,
                  const std::vector<int>& pick_jc_,
                  const std::string& run_dir_);
    void destroy();

    // --- Hooks called from cart_ale2_solver.cu step() ---
    void begin_step(double t, int step, double dt);
    void snapshot_pre_lag (const CartAle2Solver& S);  // node v, dm, e_int
    void snapshot_post_lag(const CartAle2Solver& S);  // after energy_update
    void after_cell_mom   (const CartAle2Solver& S);  // uses d_px/py_cell
    void after_remap      (const CartAle2Solver& S);  // uses d_px_new/py_new
    void after_rebuild    (const CartAle2Solver& S);  // uses post-rebuild vX/vY
    void after_heating    (const CartAle2Solver& S);  // after apply_cooling
    void end_step         (const CartAle2Solver& S);  // cumulate + advance row

    // --- Per-VTK-frame and end-of-run output ---
    void flush_cum_to_csv(double t_frame, int frame_idx);  // D2H the cum fields
    void flush_pick_to_csv();                               // single D2H at run end
};

// Parse "--trace-cells ic,jc;ic,jc;..." into two parallel vectors.
bool parse_trace_cells(const std::string& s,
                       std::vector<int>& ic, std::vector<int>& jc);
