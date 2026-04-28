#include "output.h"
#include "../parallel.h"
#include <fstream>
#include <cmath>
#include <cstdio>

Diagnostics compute_diagnostics(const Grid& grid, const State& state) {
    Diagnostics diag = {};
    int nr = grid.nr, nt = grid.ntheta;
    double total_mass = 0.0;
    double kinetic_energy = 0.0;
    double thermal_energy = 0.0;
    double gravitational_energy = 0.0;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction(+:total_mass,kinetic_energy,thermal_energy,gravitational_energy)
#endif
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            int flat = i * nt + j;
            double vol = grid.cell_volume[flat];
            double rho = state.rho[k];

            total_mass += rho * vol;

            double vr = state.mr[k] / rho;
            double vt = state.mtheta[k] / rho;
            double ke = 0.5 * rho * (vr * vr + vt * vt);
            kinetic_energy += ke * vol;

            double e_total = state.E[k];
            double e_int = e_total - ke;
            thermal_energy += e_int * vol;

            gravitational_energy += 0.5 * rho * state.phi[flat] * vol;
        }
    }

    diag.total_mass = total_mass;
    diag.kinetic_energy = kinetic_energy;
    diag.thermal_energy = thermal_energy;
    diag.gravitational_energy = gravitational_energy;
    diag.total_energy = diag.kinetic_energy + diag.thermal_energy + diag.gravitational_energy;
    return diag;
}

void write_vtk(const std::string& filename, const Grid& grid, const State& state, const EOS& eos) {
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
            PrimitiveVars w = state.to_primitive(k, eos);
            out << w.P << "\n";
        }
    }

    out << "SCALARS temperature double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            double rho = state.rho[k];
            double vr = state.mr[k] / rho;
            double vt = state.mtheta[k] / rho;
            double ke = 0.5 * (vr * vr + vt * vt);
            double e_int = std::fmax(state.E[k] / rho - ke, 1e-30);
            out << eos.temperature_from_rho_e(rho, e_int) << "\n";
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
