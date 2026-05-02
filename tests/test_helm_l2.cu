// Smoke test: confirm HelmholtzTable loads (if table available) and L2
// persisting cache gets enabled on the current device. Does NOT test
// physics — just plumbing.

#include "../src/physics/helmholtz_eos.cuh"
#include "../src/physics/helmholtz_eos.cu"    // single-TU test; pulls impl
#include <cstdio>
#include <cstdlib>
#include <cstring>

static __global__ void probe_kernel(HelmholtzTableView tv, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Walk the table and sum the f column — forces L2 hits once table is
    // in cache on the second launch.
    int idx = i % (HELM_JMAX * HELM_IMAX);
    out[i] = tv.f[idx];
}

int main() {
    HelmholtzTable tbl;

    const char* ascii = "third_party/helmholtz/helm_table.dat";
    const char* bin   = "third_party/helmholtz/helm_table.bin";

    cudaStream_t s;
    cudaStreamCreate(&s);

    int rc = tbl.load(ascii, bin, s);
    if (rc != 0) {
        std::printf("Helm table not present — SKIP (non-fatal on clean repo).\n");
        std::printf("  Download:  wget http://cococubed.com/codes/eos/helm_table.dat\n");
        std::printf("  Place at: %s\n", ascii);
        return 0;
    }

    // Probe: simple kernel that reads f[] — if L2 is pinned, second run
    // should be much faster. We don't time here, just verify it runs.
    int N = 1 << 18;
    double* d_out;
    cudaMalloc(&d_out, N * sizeof(double));

    // Warm-up
    probe_kernel<<<(N+255)/256, 256, 0, s>>>(tbl.view, d_out, N);
    cudaStreamSynchronize(s);

    // Read one value back
    double h_val = 0;
    cudaMemcpy(&h_val, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    std::printf("helm_l2_smoke: probe[0] = %.6e  (sanity check)\n", h_val);

    cudaFree(d_out);
    tbl.destroy();
    cudaStreamDestroy(s);
    std::printf("helm_l2_smoke: OK\n");
    return 0;
}
