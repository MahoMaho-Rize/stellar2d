#include "helmholtz_eos.cuh"
#include "../gpu/fas_common.cuh"  // CUDA_CHECK

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Binary table loader. The ASCII helm_table.dat must be preprocessed by
// tools/helm_convert into a 64-byte-header + 21 × imax × jmax float64
// payload (~17 MB for twice-dense resolution). See tools/helm_convert.cpp
// for the format spec.
// ---------------------------------------------------------------------

int HelmholtzTable::load(const char* /*ascii_path*/, const char* bin_cache_path, cudaStream_t s) {
    if (!bin_cache_path) {
        std::fprintf(stderr, "helmholtz: no binary path given\n");
        return 1;
    }
    std::FILE* fp = std::fopen(bin_cache_path, "rb");
    if (!fp) {
        std::fprintf(stderr,
            "helmholtz: cannot open %s — run tools/helm_convert first:\n"
            "  tools/helm_convert third_party/helmholtz/helm_table.dat "
            "third_party/helmholtz/helm_table.bin\n", bin_cache_path);
        return 2;
    }

    // --- Read + validate 64-byte header ---
    char header[64];
    if (std::fread(header, 1, 64, fp) != 64) {
        std::fprintf(stderr, "helmholtz: short header read\n");
        std::fclose(fp); return 3;
    }
    if (std::memcmp(header, "HELMv1\0\0", 8) != 0) {
        std::fprintf(stderr, "helmholtz: bad magic (expected HELMv1)\n");
        std::fclose(fp); return 4;
    }
    int32_t imax, jmax;
    std::memcpy(&imax, header + 8, 4);
    std::memcpy(&jmax, header + 12, 4);
    if (imax != HELM_IMAX || jmax != HELM_JMAX) {
        std::fprintf(stderr, "helmholtz: dim mismatch (file %dx%d, code %dx%d)\n",
                     imax, jmax, HELM_IMAX, HELM_JMAX);
        std::fclose(fp); return 5;
    }
    double tlo, thi, dlo, dhi;
    std::memcpy(&tlo, header + 16, 8);
    std::memcpy(&thi, header + 24, 8);
    std::memcpy(&dlo, header + 32, 8);
    std::memcpy(&dhi, header + 40, 8);

    // --- Read payload: 21 fields × imax × jmax doubles ---
    const int N_FIELDS = 21;
    const size_t NENT  = (size_t)HELM_IMAX * HELM_JMAX;
    std::vector<double> host_concat((size_t)N_FIELDS * NENT);
    size_t got = std::fread(host_concat.data(), sizeof(double), host_concat.size(), fp);
    std::fclose(fp);
    if (got != host_concat.size()) {
        std::fprintf(stderr, "helmholtz: short payload read (%zu / %zu)\n",
                     got, host_concat.size());
        return 6;
    }
    std::fprintf(stderr, "helmholtz: loaded %s (%.2f MB, %d fields × %dx%d)\n",
                 bin_cache_path, host_concat.size() * 8.0 / 1048576.0,
                 N_FIELDS, HELM_IMAX, HELM_JMAX);

    // --- Allocate + upload ---
    bytes = host_concat.size() * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, host_concat.data(), bytes, cudaMemcpyHostToDevice));

    // --- Wire up table view pointers (layout matches helm_convert output) ---
    // Fields 0..8: f family
    view.f     = d_data + 0 * NENT;
    view.fd    = d_data + 1 * NENT;
    view.ft    = d_data + 2 * NENT;
    view.fdd   = d_data + 3 * NENT;
    view.ftt   = d_data + 4 * NENT;
    view.fdt   = d_data + 5 * NENT;
    view.fddt  = d_data + 6 * NENT;
    view.fdtt  = d_data + 7 * NENT;
    view.fddtt = d_data + 8 * NENT;
    // Fields 9..12: dpdf family
    view.dpdf   = d_data +  9 * NENT;
    view.dpdfd  = d_data + 10 * NENT;
    view.dpdft  = d_data + 11 * NENT;
    view.dpdfdt = d_data + 12 * NENT;
    // Fields 13..16: ef family
    view.ef   = d_data + 13 * NENT;
    view.efd  = d_data + 14 * NENT;
    view.eft  = d_data + 15 * NENT;
    view.efdt = d_data + 16 * NENT;
    // Fields 17..20: xf family
    view.xf   = d_data + 17 * NENT;
    view.xfd  = d_data + 18 * NENT;
    view.xft  = d_data + 19 * NENT;
    view.xfdt = d_data + 20 * NENT;

    // Grid bookkeeping (from header, not hard-coded)
    view.tlo = tlo; view.thi = thi;
    view.dt  = (thi - tlo) / (HELM_JMAX - 1);
    view.dti = 1.0 / view.dt;
    view.dlo = dlo; view.dhi = dhi;
    view.dd  = (dhi - dlo) / (HELM_IMAX - 1);
    view.ddi = 1.0 / view.dd;

    // Composition: caller can override.
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
