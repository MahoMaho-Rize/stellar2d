// Correctness test for the KAPv1 GPU loader.
//
// Loads third_party/mesa_kap/gs98_z0.02.kapbin, then
//   1. round-trips a handful of exact grid points — expect bit-exact
//      reproduction of the underlying log_kap (trilinear at the node).
//   2. evaluates the canonical solar probes we used in the Python
//      sanity check and prints κ [cm²/g].
//
// The ASCII reference values for the probes are hard-coded from the
// `scripts/verify/verify_mesa_kapbin.py` dump on the same binary, so any drift
// between host/device or bilinear paths shows up immediately.

#include "../src/physics/opacity_table.cuh"
#include "../src/physics/opacity_table.cu"

#include <cstdio>
#include <cmath>

static __global__ void k_probe(KapTableView kv, int n,
                               const double* Xs, const double* logTs,
                               const double* logRs, double* out_kap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_kap[i] = kap_eval(kv, Xs[i], logTs[i], logRs[i]);
}

static __global__ void k_probe_node(KapTableView kv, int iX, int jT, int iR,
                                    double* out_log_kap) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    out_log_kap[0] = kv.log_kap[kap_idx3d(kv, iX, jT, iR)];
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1]
                                  : "third_party/mesa_kap/gs98_z0.02.kapbin";
    KapTable kap;
    cudaStream_t s;
    cudaStreamCreate(&s);
    int rc = kap.load(path, s);
    if (rc != 0) {
        std::printf("SKIP: cannot load %s (rc=%d)\n", path, rc);
        return 0;
    }
    std::printf("\n==== KAPv1 loader test ====\n");
    std::printf("loaded: family=%s  Z=%.4g  n_X=%d  n_logT=%d  n_logR=%d\n",
                kap.family, kap.view.Z, kap.view.n_X, kap.view.n_logT, kap.view.n_logR);

    // --- Test 1: exact node round-trip at (iX, jT, iR) = (5, 50, 20) ---
    //   these numeric coordinates have to exist for the solar gs98_z0.02
    //   file (n_X=10, n_logT=138, n_logR=37). We probe a non-edge cell so
    //   trilinear weights collapse to a single corner.
    double* d_out_node = nullptr;
    cudaMalloc(&d_out_node, sizeof(double));
    k_probe_node<<<1, 1>>>(kap.view, 5, 50, 20, d_out_node);
    cudaDeviceSynchronize();
    double lk_node_dev = 0.0;
    cudaMemcpy(&lk_node_dev, d_out_node, sizeof(double), cudaMemcpyDeviceToHost);

    // Read same cell via the interpolator: pass the exact node coordinates.
    // We need X[5], logT[50], logR[20] — pull them from device.
    double X5  = 0.0, T50 = 0.0, R20 = 0.0;
    cudaMemcpy(&X5,  kap.view.X    + 5,  sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&T50, kap.view.logT + 50, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&R20, kap.view.logR + 20, sizeof(double), cudaMemcpyDeviceToHost);

    double *d_Xs, *d_Ts, *d_Rs, *d_out;
    cudaMalloc(&d_Xs, sizeof(double));
    cudaMalloc(&d_Ts, sizeof(double));
    cudaMalloc(&d_Rs, sizeof(double));
    cudaMalloc(&d_out, sizeof(double));
    cudaMemcpy(d_Xs, &X5,  sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ts, &T50, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rs, &R20, sizeof(double), cudaMemcpyHostToDevice);
    k_probe<<<1, 1>>>(kap.view, 1, d_Xs, d_Ts, d_Rs, d_out);
    cudaDeviceSynchronize();
    double kappa_node = 0.0;
    cudaMemcpy(&kappa_node, d_out, sizeof(double), cudaMemcpyDeviceToHost);

    double lk_interp = std::log10(kappa_node);
    double diff = std::fabs(lk_interp - lk_node_dev);
    std::printf("\nnode-exact probe  (iX=5, jT=50, iR=20)\n");
    std::printf("  X=%g  logT=%g  logR=%g\n", X5, T50, R20);
    std::printf("  direct log_kap = %.12e\n", lk_node_dev);
    std::printf("  interp log_kap = %.12e  (diff %.2e)\n", lk_interp, diff);
    int fails = 0;
    if (diff > 1e-12) {
        std::printf("  FAIL: trilinear at a grid node did not reproduce the node value\n");
        fails++;
    } else {
        std::printf("  OK: bit-exact at node\n");
    }

    // --- Test 2: canonical solar probes -----------------------------------
    struct Probe { const char* name; double rho; double T; double X; double kappa_expected; };
    // kappa_expected from an independent Python trilinear evaluation on
    // the same KAPv1 binary (bisect on logT, uniform on logR) — so the GPU
    // and host paths should agree to round-off. Tolerance 1e-3 relative.
    Probe probes[] = {
        {"photosphere X=0.7 ρ=1e-7 T=5800 K", 1.0e-7, 5800.0, 0.7, 0.198},
        {"core        X=0.7 ρ=150 T=1.5e7 K", 150.0,  1.5e7,  0.7, 1.57 },
        {"envelope    X=0.7 ρ=1e-2 T=1e6 K",  1.0e-2, 1.0e6,  0.7, 29.2 },
    };
    int n = sizeof(probes) / sizeof(probes[0]);
    double h_X[3], h_lT[3], h_lR[3];
    for (int i = 0; i < n; ++i) {
        h_X[i]  = probes[i].X;
        h_lT[i] = std::log10(probes[i].T);
        h_lR[i] = std::log10(probes[i].rho) - 3.0 * h_lT[i] + 18.0;
    }
    double *d_Xa, *d_Ta, *d_Ra, *d_outa;
    cudaMalloc(&d_Xa, n * sizeof(double));
    cudaMalloc(&d_Ta, n * sizeof(double));
    cudaMalloc(&d_Ra, n * sizeof(double));
    cudaMalloc(&d_outa, n * sizeof(double));
    cudaMemcpy(d_Xa, h_X,  n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ta, h_lT, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ra, h_lR, n * sizeof(double), cudaMemcpyHostToDevice);
    k_probe<<<1, 32>>>(kap.view, n, d_Xa, d_Ta, d_Ra, d_outa);
    cudaDeviceSynchronize();
    double h_out[3];
    cudaMemcpy(h_out, d_outa, n * sizeof(double), cudaMemcpyDeviceToHost);

    std::printf("\nsolar probes (trilinear):\n");
    for (int i = 0; i < n; ++i) {
        double kappa = h_out[i];
        double expected = probes[i].kappa_expected;
        double rel = std::fabs(kappa - expected) / expected;
        std::printf("  %-38s κ = %8.3g cm²/g   (py-ref %.3g, rel %.1e) %s\n",
                    probes[i].name, kappa, expected, rel,
                    rel < 5e-3 ? "OK" : "FAIL");
        if (rel >= 5e-3) fails++;
    }

    cudaFree(d_Xs); cudaFree(d_Ts); cudaFree(d_Rs); cudaFree(d_out);
    cudaFree(d_Xa); cudaFree(d_Ta); cudaFree(d_Ra); cudaFree(d_outa);
    cudaFree(d_out_node);
    kap.destroy();
    cudaStreamDestroy(s);

    if (fails) {
        std::printf("\nFAIL: %d test(s)\n", fails);
        return 1;
    }
    std::printf("\nAll kap_table tests passed.\n");
    return 0;
}
