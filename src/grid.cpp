#include "grid.h"
#include <algorithm>
#include <numeric>

void Grid::init(int nr_, int ntheta_, double R_outer_, double alpha_, int ng_) {
    nr = nr_;
    ntheta = ntheta_;
    R_outer = R_outer_;
    log_alpha = alpha_;
    ng = ng_;
    mesh_type = MeshType::LOG;

    build_radial_mesh();
    build_theta_mesh();
    compute_geometry();
}

void Grid::init_uniform(int nr_, int ntheta_, double R_outer_, int ng_) {
    nr = nr_;
    ntheta = ntheta_;
    R_outer = R_outer_;
    log_alpha = 1.0;
    ng = ng_;
    mesh_type = MeshType::UNIFORM;

    build_uniform_radial_mesh();
    build_theta_mesh();
    compute_geometry();
}

void Grid::init_equimass(int nr_, int ntheta_, double R_outer_,
                         std::function<double(double)> rho_func, int ng_) {
    nr = nr_;
    ntheta = ntheta_;
    R_outer = R_outer_;
    log_alpha = 1.0;
    ng = ng_;
    mesh_type = MeshType::EQUIMASS;

    build_equimass_radial_mesh(rho_func);
    build_theta_mesh();
    compute_geometry();
}

void Grid::build_uniform_radial_mesh() {
    r_face.resize(nr + 1);
    r_center.resize(nr);
    dr.resize(nr);

    for (int i = 0; i <= nr; ++i)
        r_face[i] = R_outer * static_cast<double>(i) / nr;

    for (int i = 0; i < nr; ++i) {
        r_center[i] = 0.5 * (r_face[i] + r_face[i + 1]);
        dr[i] = r_face[i + 1] - r_face[i];
    }
}

void Grid::build_radial_mesh() {
    r_face.resize(nr + 1);
    r_center.resize(nr);
    dr.resize(nr);

    for (int i = 0; i <= nr; ++i) {
        double xi = static_cast<double>(i) / nr;
        r_face[i] = R_outer * std::pow(xi, log_alpha); // Eq. (2.1)
    }

    for (int i = 0; i < nr; ++i) {
        r_center[i] = 0.5 * (r_face[i] + r_face[i + 1]);
        dr[i] = r_face[i + 1] - r_face[i];
    }
}

void Grid::build_equimass_radial_mesh(std::function<double(double)> rho_func) {
    // Strategy: equidistribution with w(r) = |dρ/dr| + ρ/H_ρ_global
    //
    // |dρ/dr| concentrates cells where density changes rapidly (core-envelope
    // boundary, surface). The floor term ρ/H prevents empty regions from getting
    // no cells at all, while still weighting by density so the core gets good
    // resolution.
    //
    // This gives: fine cells at center (steep gradient), fine cells at surface
    // (steep gradient), moderate cells in between.

    const int nfine = 20000;
    double dr_fine = R_outer / nfine;

    // Evaluate density on fine grid
    std::vector<double> rho_fine(nfine + 1);
    for (int i = 0; i <= nfine; ++i)
        rho_fine[i] = rho_func(i * dr_fine);

    // Compute weight = |dρ/dr| + floor
    // Floor = ρ_c / (nr * some_factor) ensures no cell is wider than ~R/nr
    double rho_c_val = rho_fine[0];
    double floor_w = rho_c_val / (2.0 * nr);

    std::vector<double> w_cumul(nfine + 1, 0.0);
    for (int i = 0; i < nfine; ++i) {
        double grad = std::abs(rho_fine[i + 1] - rho_fine[i]) / dr_fine;
        double rho_local = 0.5 * (rho_fine[i] + rho_fine[i + 1]);
        double r_mid = (i + 0.5) * dr_fine;
        // 1/r² term ensures outer vacuum region still gets cells
        double geom_floor = rho_c_val * R_outer * R_outer / ((r_mid + 0.01 * R_outer) * (r_mid + 0.01 * R_outer) + R_outer * R_outer) * 0.3;
        double w = grad + floor_w + 0.1 * rho_local + geom_floor;
        w_cumul[i + 1] = w_cumul[i] + w * dr_fine;
    }

    double W_total = w_cumul[nfine];

    r_face.resize(nr + 1);
    r_center.resize(nr);
    dr.resize(nr);

    r_face[0] = 0.0;
    r_face[nr] = R_outer;

    for (int i = 1; i < nr; ++i) {
        double w_target = W_total * static_cast<double>(i) / nr;
        auto it = std::lower_bound(w_cumul.begin(), w_cumul.end(), w_target);
        int idx = static_cast<int>(it - w_cumul.begin());
        if (idx <= 0) idx = 1;
        if (idx >= nfine) idx = nfine - 1;
        double frac = (w_target - w_cumul[idx - 1]) / (w_cumul[idx] - w_cumul[idx - 1] + 1e-30);
        r_face[i] = ((idx - 1) + frac) * dr_fine;
    }

    for (int i = 0; i < nr; ++i) {
        r_center[i] = 0.5 * (r_face[i] + r_face[i + 1]);
        dr[i] = r_face[i + 1] - r_face[i];
    }
}

void Grid::init_mass_shell(int nr_, int ntheta_, double R_star_,
                           std::function<double(double)> rho_func,
                           double r_inner_, int ng_) {
    nr = nr_;
    ntheta = ntheta_;
    R_outer = R_star_;
    R_inner = r_inner_;
    log_alpha = 1.0;
    ng = ng_;
    mesh_type = MeshType::MASS_SHELL;

    build_mass_shell_radial_mesh(rho_func, r_inner_);
    build_theta_mesh();
    compute_geometry();
}

void Grid::build_mass_shell_radial_mesh(std::function<double(double)> rho_func,
                                         double r_inner_) {
    const int nfine = 40000;
    double dr_fine = R_outer / nfine;
    double r_blend = 0.15 * R_outer;

    // Hybrid weight: w(r) = ρ(r) · max(r², r_blend²)
    // r < r_blend: w ≈ ρ·r_blend² (uniform-like)
    // r > r_blend: w = ρ·r² (equal-mass shells)
    std::vector<double> w_cumul(nfine + 1, 0.0);
    for (int i = 0; i < nfine; ++i) {
        double r = (i + 0.5) * dr_fine;
        double rho = rho_func(r);
        double r2_eff = std::max(r * r, r_blend * r_blend);
        w_cumul[i + 1] = w_cumul[i] + rho * r2_eff * dr_fine;
    }
    double W_total = w_cumul[nfine];

    r_face.resize(nr + 1);
    r_center.resize(nr);
    dr.resize(nr);

    r_face[0] = r_inner_;
    r_face[nr] = R_outer;

    double W_start = 0.0;
    if (r_inner_ > 0.0) {
        int i_inner = static_cast<int>(r_inner_ / dr_fine);
        if (i_inner > nfine) i_inner = nfine;
        W_start = w_cumul[i_inner];
    }
    double W_shell = W_total - W_start;

    for (int i = 1; i < nr; ++i) {
        double w_target = W_start + W_shell * static_cast<double>(i) / nr;
        auto it = std::lower_bound(w_cumul.begin(), w_cumul.end(), w_target);
        int idx = static_cast<int>(it - w_cumul.begin());
        if (idx <= 0) idx = 1;
        if (idx >= nfine) idx = nfine - 1;
        double frac = (w_target - w_cumul[idx - 1]) /
                      (w_cumul[idx] - w_cumul[idx - 1] + 1e-30);
        r_face[i] = ((idx - 1) + frac) * dr_fine;
    }

    for (int i = 0; i < nr; ++i) {
        r_center[i] = 0.5 * (r_face[i] + r_face[i + 1]);
        dr[i] = r_face[i + 1] - r_face[i];
    }
}

void Grid::build_theta_mesh() {
    theta_face.resize(ntheta + 1);
    theta_center.resize(ntheta);
    dtheta.resize(ntheta);

    for (int j = 0; j <= ntheta; ++j) {
        theta_face[j] = M_PI * static_cast<double>(j) / ntheta;
    }

    for (int j = 0; j < ntheta; ++j) {
        theta_center[j] = 0.5 * (theta_face[j] + theta_face[j + 1]);
        dtheta[j] = theta_face[j + 1] - theta_face[j];
    }
}

void Grid::compute_geometry() {
    cell_volume.resize(nr * ntheta);
    area_r.resize((nr + 1) * ntheta);
    area_theta.resize(nr * (ntheta + 1));

    for (int i = 0; i < nr; ++i) {
        double r3_hi = r_face[i + 1] * r_face[i + 1] * r_face[i + 1];
        double r3_lo = r_face[i] * r_face[i] * r_face[i];
        for (int j = 0; j < ntheta; ++j) {
            double dcos = std::cos(theta_face[j]) - std::cos(theta_face[j + 1]);
            cell_volume[i * ntheta + j] = (r3_hi - r3_lo) / 3.0 * dcos; // Eq. (2.2)
        }
    }

    for (int i = 0; i <= nr; ++i) {
        double r2 = r_face[i] * r_face[i];
        for (int j = 0; j < ntheta; ++j) {
            double dcos = std::cos(theta_face[j]) - std::cos(theta_face[j + 1]);
            area_r[i * ntheta + j] = r2 * dcos; // Eq. (2.3)
        }
    }

    for (int i = 0; i < nr; ++i) {
        double r2_diff = (r_face[i + 1] * r_face[i + 1] - r_face[i] * r_face[i]) * 0.5;
        for (int j = 0; j <= ntheta; ++j) {
            area_theta[i * (ntheta + 1) + j] = std::sin(theta_face[j]) * r2_diff; // Eq. (2.4)
        }
        // sin(π) ≈ 1.2e-16 in IEEE 754 — force exact zero at both poles
        area_theta[i * (ntheta + 1) + 0] = 0.0;
        area_theta[i * (ntheta + 1) + ntheta] = 0.0;
    }
}
