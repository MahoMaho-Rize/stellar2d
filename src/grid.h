#pragma once

#include <vector>
#include <cmath>
#include <functional>

enum class MeshType { LOG, EQUIMASS };

struct Grid {
    int nr, ntheta;
    int ng; // ghost cells per side

    double R_outer;
    double log_alpha; // radial stretch exponent
    MeshType mesh_type = MeshType::LOG;

    // Cell interface positions (size nr+1, ntheta+1 for physical cells)
    std::vector<double> r_face;     // r_{i+1/2}, i = 0..nr
    std::vector<double> theta_face; // theta_{j+1/2}, j = 0..ntheta

    // Cell center positions
    std::vector<double> r_center;     // i = 0..nr-1
    std::vector<double> theta_center; // j = 0..ntheta-1

    // Cell widths
    std::vector<double> dr; // dr_i = r_{i+1/2} - r_{i-1/2}
    std::vector<double> dtheta;

    // Precomputed geometric quantities
    std::vector<double> cell_volume;  // V_{ij}, flat array [i*ntheta + j]
    std::vector<double> area_r;      // A^r_{i+1/2,j}, [(nr+1)*ntheta]
    std::vector<double> area_theta;  // A^theta_{i,j+1/2}, [nr*(ntheta+1)]

    int total_cells() const { return nr * ntheta; }
    int stride() const { return ntheta + 2 * ng; }
    int total_with_ghost() const { return (nr + 2 * ng) * stride(); }

    int idx(int i, int j) const {
        return (i + ng) * stride() + (j + ng);
    }

    void init(int nr_, int ntheta_, double R_outer_, double alpha_, int ng_ = 2);

    // Equimass mesh: rho_func(r) returns density at radius r
    void init_equimass(int nr_, int ntheta_, double R_outer_,
                       std::function<double(double)> rho_func, int ng_ = 2);

private:
    void build_radial_mesh();
    void build_equimass_radial_mesh(std::function<double(double)> rho_func);
    void build_theta_mesh();
    void compute_geometry();
};
