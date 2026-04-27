#pragma once

#include "grid.h"
#include <vector>

struct PrimitiveVars {
    double rho;
    double vr;
    double vtheta;
    double P;
};

struct ConservedVars {
    double rho;
    double mr;     // rho * vr
    double mtheta; // rho * vtheta
    double E;      // rho * (e + 0.5*(vr^2 + vtheta^2))
};

struct State {
    int size;

    std::vector<double> rho;
    std::vector<double> mr;
    std::vector<double> mtheta;
    std::vector<double> E;

    std::vector<double> phi; // gravitational potential

    void allocate(const Grid& grid);

    PrimitiveVars to_primitive(int k, double gamma) const;
    void from_primitive(int k, const PrimitiveVars& w, double gamma);

    ConservedVars get_conserved(int k) const;
    void set_conserved(int k, const ConservedVars& u);

    void copy_from(const State& other);
};
