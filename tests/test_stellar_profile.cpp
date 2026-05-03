// Tests C++ StellarProfile build against Python reference CSV.
// Reference: /tmp/poly3_ref.csv (from scripts/build_poly3_python.py)
//
// Usage:
//   python scripts/build_poly3_python.py
//   ./build/test_stellar_profile /tmp/poly3_ref.csv
#include "gpu/stellar_profile.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool load_python_ref(const std::string& path, StellarProfile& ref) {
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char line[2048];
    ref.x.clear(); ref.rho.clear(); ref.P.clear(); ref.M_r.clear();
    ref.V_2.clear(); ref.A_star.clear(); ref.U.clear();
    ref.c_1.clear(); ref.Gamma_1.clear();
    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        double x, rho, P, M_r, V_2, A_star, U, c_1, G1;
        if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                        &x, &rho, &P, &M_r, &V_2, &A_star, &U, &c_1, &G1) == 9) {
            ref.x.push_back(x); ref.rho.push_back(rho); ref.P.push_back(P);
            ref.M_r.push_back(M_r); ref.V_2.push_back(V_2);
            ref.A_star.push_back(A_star); ref.U.push_back(U);
            ref.c_1.push_back(c_1); ref.Gamma_1.push_back(G1);
        }
    }
    std::fclose(fp);
    return !ref.x.empty();
}

static double max_rel_err(const std::vector<double>& a,
                          const std::vector<double>& b,
                          double floor = 1e-30) {
    double m = 0.0;
    int n = (int)std::min(a.size(), b.size());
    for (int i = 0; i < n; ++i) {
        double denom = std::max(std::fabs(b[i]), floor);
        double r = std::fabs(a[i] - b[i]) / denom;
        if (r > m) m = r;
    }
    return m;
}

int main(int argc, char** argv) {
    std::string ref_path = (argc >= 2) ? argv[1] : "/tmp/poly3_ref.csv";
    StellarProfile ref;
    if (!load_python_ref(ref_path, ref)) return 1;

    int n = ref.n_points();
    std::printf("Reference loaded: %d rows, x ∈ [%.4g, %.4g]\n",
                n, ref.x.front(), ref.x.back());

    double inner = ref.x.front();
    double outer = ref.x.back();
    StellarProfile cpp = build_polytrope_profile(3.0, n, 5.0/3.0, inner, outer);

    // Compare each field.
    double err_rho  = max_rel_err(cpp.rho,  ref.rho);
    double err_P    = max_rel_err(cpp.P,    ref.P);
    double err_M_r  = max_rel_err(cpp.M_r,  ref.M_r);
    double err_V_2  = max_rel_err(cpp.V_2,  ref.V_2);
    double err_U    = max_rel_err(cpp.U,    ref.U);
    double err_c_1  = max_rel_err(cpp.c_1,  ref.c_1);
    // A* can cross zero → absolute tolerance
    double max_abs_Astar = 0.0;
    for (double v : ref.A_star) max_abs_Astar = std::max(max_abs_Astar, std::fabs(v));
    double err_Astar = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = std::fabs(cpp.A_star[i] - ref.A_star[i]) /
                   std::max(max_abs_Astar, 1e-30);
        if (d > err_Astar) err_Astar = d;
    }

    auto report = [](const char* name, double e, double tol) {
        const char* tag = (e < tol) ? "OK  " : "FAIL";
        std::printf("  [%s] %-8s rel/abs err = %.3e  (tol %.3e)\n",
                    tag, name, e, tol);
    };
    report("rho",   err_rho,  1e-6);
    report("P",     err_P,    1e-6);
    report("M_r",   err_M_r,  1e-5);
    report("V_2",   err_V_2,  1e-5);
    report("U",     err_U,    1e-5);
    report("c_1",   err_c_1,  1e-5);
    report("A_star",err_Astar,1e-5);

    bool all_ok = err_rho < 1e-6 && err_P < 1e-6 && err_M_r < 1e-5
               && err_V_2 < 1e-5 && err_U < 1e-5 && err_c_1 < 1e-5
               && err_Astar < 1e-5;
    return all_ok ? 0 : 1;
}
