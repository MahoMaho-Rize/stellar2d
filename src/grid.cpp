#include "grid.h"

void Grid::init(int nr_, int ntheta_, double R_outer_, double alpha_, int ng_) {
    nr = nr_;
    ntheta = ntheta_;
    R_outer = R_outer_;
    log_alpha = alpha_;
    ng = ng_;

    build_radial_mesh();
    build_theta_mesh();
    compute_geometry();
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
    }
}
