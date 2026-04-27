#include "output.h"
#include <fstream>
#include <cmath>
#include <cstdio>

Diagnostics compute_diagnostics(const Grid& grid, const State& state, double gamma) {
    Diagnostics diag = {};
    int nr = grid.nr, nt = grid.ntheta;

    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            int flat = i * nt + j;
            double vol = grid.cell_volume[flat];
            double rho = state.rho[k];

            diag.total_mass += rho * vol;

            double vr = state.mr[k] / rho;
            double vt = state.mtheta[k] / rho;
            double ke = 0.5 * rho * (vr * vr + vt * vt);
            diag.kinetic_energy += ke * vol;

            double e_total = state.E[k];
            double e_int = e_total - ke;
            diag.thermal_energy += e_int * vol;

            diag.gravitational_energy += 0.5 * rho * state.phi[flat] * vol;
        }
    }

    diag.total_energy = diag.kinetic_energy + diag.thermal_energy + diag.gravitational_energy;
    return diag;
}

void write_vtk(const std::string& filename, const Grid& grid, const State& state, double gamma) {
    int nr = grid.nr, nt = grid.ntheta;
    int npts = (nr + 1) * (nt + 1);
    int ncells = nr * nt;

    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "stellar2d output\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << (nt + 1) << " " << (nr + 1) << " 1\n";
    out << "POINTS " << npts << " double\n";

    for (int i = 0; i <= nr; ++i) {
        double r = grid.r_face[i];
        for (int j = 0; j <= nt; ++j) {
            double theta = grid.theta_face[j];
            double x = r * std::sin(theta);
            double z = r * std::cos(theta);
            out << x << " " << 0.0 << " " << z << "\n";
        }
    }

    out << "CELL_DATA " << ncells << "\n";

    out << "SCALARS density double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            out << state.rho[grid.idx(i, j)] << "\n";

    out << "SCALARS pressure double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            double rho = state.rho[k];
            double vr = state.mr[k] / rho;
            double vt = state.mtheta[k] / rho;
            double ke = 0.5 * (vr * vr + vt * vt);
            double e_int = state.E[k] / rho - ke;
            double P = (gamma - 1.0) * rho * e_int;
            out << P << "\n";
        }
    }

    out << "SCALARS phi double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            out << state.phi[i * nt + j] << "\n";

    out << "VECTORS velocity double\n";
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            double rho = state.rho[k];
            double vr = state.mr[k] / rho;
            double vt = state.mtheta[k] / rho;
            double theta = grid.theta_center[j];
            double vx = vr * std::sin(theta) + vt * std::cos(theta);
            double vz = vr * std::cos(theta) - vt * std::sin(theta);
            out << vx << " " << 0.0 << " " << vz << "\n";
        }
    }
}
