// Day-1 round-trip test for Radial1DSolver's 13-species α-chain buffer.
//
// Allocate a minimal solver, upload a deterministic X_spec[nz][13] pattern
// via init_species_alpha(), download it via download_species_alpha(), and
// verify byte-exact round-trip.
//
// No hydro step runs; this tests only the upload/download plumbing added
// in Phase D Day 1.
//
// Build (CMake target test_radial1d_alpha13) — see tests/CMakeLists.txt.
// Run: ./build/test_radial1d_alpha13

#include "../src/gpu/radial1d_solver.cuh"
#include "../src/physics/alpha_network.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    constexpr int nz = 32;
    constexpr int NS = alpha_net::N_SPEC;

    Radial1DSolver r1d;
    r1d.init(nz, /*gamma=*/5.0/3.0, /*G=*/6.674e-8, /*cfl=*/0.4);

    // Deterministic pattern: X[k][s] = (k + 1) * 1e-3 + s * 1e-5.
    // Not normalised; we only care about bit-exact round-trip.
    std::vector<double> X_in(static_cast<size_t>(nz) * NS);
    for (int k = 0; k < nz; ++k) {
        for (int s = 0; s < NS; ++s) {
            X_in[static_cast<size_t>(k) * NS + s] =
                (k + 1) * 1.0e-3 + s * 1.0e-5;
        }
    }

    r1d.init_species_alpha(X_in.data());

    std::vector<double> X_out;
    r1d.download_species_alpha(X_out);

    if (X_out.size() != X_in.size()) {
        std::printf("FAIL: size mismatch (in=%zu out=%zu)\n",
                    X_in.size(), X_out.size());
        r1d.destroy();
        return 1;
    }

    double max_err = 0.0;
    int n_bad = 0;
    for (size_t i = 0; i < X_in.size(); ++i) {
        double err = std::fabs(X_in[i] - X_out[i]);
        if (err > max_err) max_err = err;
        if (err > 0.0) ++n_bad;
    }

    std::printf("Radial1D alpha13 round-trip: nz=%d, NS=%d, n_bad=%d, max_err=%.3e\n",
                nz, NS, n_bad, max_err);

    // Also verify mode flags got set.
    if (r1d.species_mode != Radial1DSolver::SPEC_ALPHA13) {
        std::printf("FAIL: species_mode did not switch to ALPHA13\n");
        r1d.destroy();
        return 2;
    }
    if (!r1d.species_enabled) {
        std::printf("FAIL: species_enabled not set\n");
        r1d.destroy();
        return 3;
    }

    r1d.destroy();

    if (max_err > 0.0) {
        std::printf("FAIL: round-trip not bit-exact\n");
        return 4;
    }
    std::printf("PASS\n");
    return 0;
}
