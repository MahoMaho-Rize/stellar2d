#include "sim/helpers.h"

#include "init/lane_emden.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

void extract_density(const Grid& grid, const State& state, std::vector<double>& rho_cells) {
    int nr = grid.nr, nt = grid.ntheta;
    rho_cells.resize(nr * nt);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            rho_cells[i * nt + j] = state.rho[grid.idx(i, j)];
}

double compute_lane_emden_R_star(double n_poly, double K_poly, double rho_c, double G) {
    auto sol = solve_lane_emden(n_poly);
    double alpha2 = (n_poly + 1.0) * K_poly
                    * std::pow(rho_c, 1.0 / n_poly - 1.0)
                    / (4.0 * M_PI * G);
    double alpha = std::sqrt(alpha2);
    return alpha * sol.xi_1;
}

double compute_lane_emden_R_outer(double n_poly, double K_poly, double rho_c, double G) {
    return compute_lane_emden_R_star(n_poly, K_poly, rho_c, G) * 1.1;
}

void print_progress(double t, double t_end, int step, double dt,
                    std::timespec& t_start) {
    double frac = t / t_end;
    int pct = static_cast<int>(frac * 100.0);
    if (pct > 100) pct = 100;

    int bar_width = 30;
    int filled = static_cast<int>(frac * bar_width);

    std::timespec t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double elapsed = (t_now.tv_sec - t_start.tv_sec)
                   + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;

    double eta = (frac > 1e-6) ? elapsed / frac * (1.0 - frac) : 0.0;

    std::fprintf(stderr, "\r  [");
    for (int i = 0; i < bar_width; ++i)
        std::fputc(i < filled ? '#' : '.', stderr);
    std::fprintf(stderr, "] %3d%%  step %-8d  t=%.3e  dt=%.2e  elapsed %.0fs  ETA %.0fs  ",
                 pct, step, t, dt, elapsed, eta);
    std::fflush(stderr);
}

std::string make_run_dir(const SimConfig& cfg) {
    std::time_t now = std::time(nullptr);
    std::tm* lt = std::localtime(&now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", lt);

    char dirname[512];
    std::snprintf(dirname, sizeof(dirname), "%s/%s_%dx%d_%s",
                  cfg.run_base.c_str(), cfg.test_case.c_str(),
                  cfg.nr, cfg.ntheta, ts);

    mkdir(cfg.run_base.c_str(), 0755);
    mkdir(dirname, 0755);
    return std::string(dirname);
}
