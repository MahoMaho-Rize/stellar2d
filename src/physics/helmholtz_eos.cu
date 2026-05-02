#include "helmholtz_eos.cuh"
#include "../gpu/fas_common.cuh"  // CUDA_CHECK

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// ASCII → binary conversion.
//
// The cococubed "helm_table.dat" file format (verified against helmeos.f):
//   Loop j = 1..jmax:        (log rho index)
//     Loop i = 1..imax:      (log T index)
//       read 9 doubles:  f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt
//
// Grid is implicit (not stored): log T ∈ [3.0, 13.0], log ρ ∈ [-12, 15].
// dt = (thi-tlo)/(imax-1), dd = (dhi-dlo)/(jmax-1).
//
// We cache the parsed table as raw little-endian float64 in bin_cache_path
// so subsequent launches skip the ASCII parse (~100 ms → 5 ms).
// ---------------------------------------------------------------------
static bool helm_parse_ascii(const char* path, std::vector<double>& out_concat) {
    std::FILE* fp = std::fopen(path, "r");
    if (!fp) return false;

    const int NFIELDS = 9;
    const size_t NENT  = (size_t)HELM_JMAX * HELM_IMAX;
    out_concat.assign(NFIELDS * NENT, 0.0);
    // Layout in out_concat: field_k * NENT + (j*IMAX + i)

    // f is the outer (j=1..jmax, i=1..imax) order; Timmes reads 9 values
    // per record. We pipe straight through.
    for (int j = 0; j < HELM_JMAX; ++j) {
        for (int i = 0; i < HELM_IMAX; ++i) {
            double v[9];
            int n = std::fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                                &v[0],&v[1],&v[2],&v[3],&v[4],&v[5],&v[6],&v[7],&v[8]);
            if (n != 9) {
                std::fprintf(stderr, "helm_parse_ascii: short read at j=%d i=%d (%d of 9)\n", j, i, n);
                std::fclose(fp);
                return false;
            }
            size_t ij = (size_t)j * HELM_IMAX + i;
            for (int k = 0; k < 9; ++k) out_concat[(size_t)k * NENT + ij] = v[k];
        }
    }
    std::fclose(fp);
    return true;
}

static bool helm_read_bin(const char* path, std::vector<double>& out_concat) {
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    const size_t NENT = (size_t)HELM_JMAX * HELM_IMAX;
    out_concat.assign(9 * NENT, 0.0);
    size_t got = std::fread(out_concat.data(), sizeof(double), 9 * NENT, fp);
    std::fclose(fp);
    return got == 9 * NENT;
}

static bool helm_write_bin(const char* path, const std::vector<double>& v) {
    std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;
    size_t wrote = std::fwrite(v.data(), sizeof(double), v.size(), fp);
    std::fclose(fp);
    return wrote == v.size();
}

int HelmholtzTable::load(const char* ascii_path, const char* bin_cache_path, cudaStream_t s) {
    const size_t NENT = (size_t)HELM_JMAX * HELM_IMAX;
    std::vector<double> host_concat;

    // Try binary cache first.
    bool loaded = false;
    if (bin_cache_path && helm_read_bin(bin_cache_path, host_concat)) {
        std::fprintf(stderr, "helmholtz: loaded binary cache %s (%.2f MB)\n",
                     bin_cache_path, host_concat.size() * 8.0 / 1048576.0);
        loaded = true;
    }
    if (!loaded) {
        if (!ascii_path) {
            std::fprintf(stderr,
                "helmholtz: no cache and no ASCII path given. Download\n"
                "  http://cococubed.com/codes/eos/helm_table.dat → third_party/helmholtz/\n");
            return 1;
        }
        if (!helm_parse_ascii(ascii_path, host_concat)) {
            std::fprintf(stderr, "helmholtz: ASCII parse failed (%s)\n", ascii_path);
            return 2;
        }
        std::fprintf(stderr, "helmholtz: parsed ASCII %s (%d×%d = %zu × 9 doubles)\n",
                     ascii_path, HELM_JMAX, HELM_IMAX, NENT);
        if (bin_cache_path) helm_write_bin(bin_cache_path, host_concat);
    }

    // Allocate + upload contiguous 9·NENT doubles.
    bytes = host_concat.size() * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, host_concat.data(), bytes, cudaMemcpyHostToDevice));

    // Wire up table view pointers.
    view.f     = d_data + 0 * NENT;
    view.fd    = d_data + 1 * NENT;
    view.ft    = d_data + 2 * NENT;
    view.fdd   = d_data + 3 * NENT;
    view.ftt   = d_data + 4 * NENT;
    view.fdt   = d_data + 5 * NENT;
    view.fddt  = d_data + 6 * NENT;
    view.fdtt  = d_data + 7 * NENT;
    view.fddtt = d_data + 8 * NENT;

    // Fixed cococubed grid
    view.tlo = 3.0; view.thi = 13.0;
    view.dt  = (view.thi - view.tlo) / (HELM_IMAX - 1);
    view.dti = 1.0 / view.dt;
    view.dlo = -12.0; view.dhi = 15.0;
    view.dd  = (view.dhi - view.dlo) / (HELM_JMAX - 1);
    view.ddi = 1.0 / view.dd;

    // Composition: will be set by caller (or left at pure H default).
    view.Abar = 1.0;
    view.Zbar = 1.0;

    // ---- L2 persisting cache window ----
    // sm_80+ (A100 / Ada / Hopper) supports pinning a memory range into
    // L2 so every access bypasses DRAM. Our table is 5 MB; persisting cap
    // on RTX 4080 Super is 44 MB, so we fit easily.
    stream = s;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.persistingL2CacheMaxSize > 0 && bytes <= (size_t)prop.accessPolicyMaxWindowSize) {
        // Set the global persisting L2 size (carve out the budget).
        size_t persist_size = bytes;
        if (persist_size > (size_t)prop.persistingL2CacheMaxSize)
            persist_size = prop.persistingL2CacheMaxSize;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size));

        // Configure access policy window on the target stream.
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.base_ptr  = static_cast<void*>(d_data);
        attr.accessPolicyWindow.num_bytes = bytes;
        attr.accessPolicyWindow.hitRatio  = 1.0;       // 100 % of window = persisting
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        cudaError_t err = cudaStreamSetAttribute(stream,
            cudaStreamAttributeAccessPolicyWindow, &attr);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "helmholtz: cudaStreamSetAttribute failed: %s (non-fatal, table still usable)\n",
                         cudaGetErrorString(err));
        } else {
            std::fprintf(stderr,
                "helmholtz: pinned %.2f MB to L2 persisting cache (cap %.1f MB, device max %.1f MB)\n",
                bytes / 1048576.0, persist_size / 1048576.0,
                prop.persistingL2CacheMaxSize / 1048576.0);
        }
    } else {
        std::fprintf(stderr, "helmholtz: L2 persisting unavailable on this device (cap=%.1f MB, needed=%.2f MB)\n",
                     prop.persistingL2CacheMaxSize / 1048576.0, bytes / 1048576.0);
    }

    return 0;
}

void HelmholtzTable::destroy() {
    if (stream) {
        // Reset the access policy window and clear the persisting budget.
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        cudaCtxResetPersistingL2Cache();
    }
    if (d_data) cudaFree(d_data);
    d_data = nullptr;
    bytes = 0;
}
