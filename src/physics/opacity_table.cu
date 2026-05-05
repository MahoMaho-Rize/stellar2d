#include "opacity_table.cuh"
#include "gpu_common.cuh"   // CUDA_CHECK

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// KAPv1 binary loader.
//
// Produced by scripts/mesa/convert_mesa_kap.py — refer to that module for the
// full layout. We re-derive the payload offsets here so the loader is
// self-contained.
//
// Header (128 bytes):
//   0   magic[8]         "KAPv1\0\0\0"
//   8   family[48]       UTF-8, zero-padded
//   56  Z               float64
//   64  n_X             int32
//   68  n_logT          int32
//   72  n_logR          int32
//   76  pad             int32
//   80  logT_min        float64
//   88  logT_max        float64
//   96  logR_min        float64
//   104 logR_max        float64
//   112 reserved[16]
// ---------------------------------------------------------------------
int KapTable::load(const char* bin_path, cudaStream_t s) {
    if (!bin_path) {
        std::fprintf(stderr, "kap_table: no binary path given\n");
        return 1;
    }
    std::FILE* fp = std::fopen(bin_path, "rb");
    if (!fp) {
        std::fprintf(stderr,
            "kap_table: cannot open %s — run scripts/mesa/convert_mesa_kap.py first\n",
            bin_path);
        return 2;
    }

    char header[128];
    if (std::fread(header, 1, sizeof(header), fp) != sizeof(header)) {
        std::fprintf(stderr, "kap_table: %s: short header read\n", bin_path);
        std::fclose(fp); return 3;
    }
    if (std::memcmp(header, KAP_MAGIC, 8) != 0) {
        std::fprintf(stderr, "kap_table: %s: bad magic (expected KAPv1)\n", bin_path);
        std::fclose(fp); return 4;
    }
    std::memcpy(family, header + 8, 48);
    family[48] = '\0';

    double Z_file;
    int32_t n_X, n_logT, n_logR, pad;
    double logT_min, logT_max, logR_min, logR_max;
    std::memcpy(&Z_file,   header + 56, 8);
    std::memcpy(&n_X,      header + 64, 4);
    std::memcpy(&n_logT,   header + 68, 4);
    std::memcpy(&n_logR,   header + 72, 4);
    std::memcpy(&pad,      header + 76, 4);
    std::memcpy(&logT_min, header + 80, 8);
    std::memcpy(&logT_max, header + 88, 8);
    std::memcpy(&logR_min, header + 96, 8);
    std::memcpy(&logR_max, header + 104, 8);

    if (n_X <= 1 || n_logT < 2 || n_logR < 2
        || n_X > KAP_MAX_X
        || n_logT > KAP_MAX_LOGT
        || n_logR > KAP_MAX_LOGR) {
        std::fprintf(stderr,
            "kap_table: %s: dims out of range (n_X=%d, n_logT=%d, n_logR=%d; max %d/%d/%d)\n",
            bin_path, n_X, n_logT, n_logR,
            KAP_MAX_X, KAP_MAX_LOGT, KAP_MAX_LOGR);
        std::fclose(fp); return 5;
    }

    const std::size_t n_axis = (std::size_t)n_X + n_logT + n_logR;
    const std::size_t n_kap  = (std::size_t)n_X * n_logT * n_logR;
    const std::size_t n_tot  = n_axis + n_kap;

    std::vector<double> host(n_tot);
    if (std::fread(host.data(), sizeof(double), n_tot, fp) != n_tot) {
        std::fprintf(stderr, "kap_table: %s: short payload read\n", bin_path);
        std::fclose(fp); return 6;
    }
    std::fclose(fp);

    // Upload single contiguous block so L2 persisting window is tight.
    bytes = n_tot * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, host.data(), bytes, cudaMemcpyHostToDevice));

    // Wire up the device view.
    view.X       = d_data;
    view.logT    = d_data + n_X;
    view.logR    = d_data + n_X + n_logT;
    view.log_kap = d_data + n_X + n_logT + n_logR;

    view.n_X    = n_X;
    view.n_logT = n_logT;
    view.n_logR = n_logR;
    view.Z      = Z_file;

    view.logT_min = logT_min;
    view.logT_max = logT_max;

    view.logR_min = logR_min;
    view.logR_max = logR_max;
    view.logR_d   = (logR_max - logR_min) / (double)(n_logR - 1);
    view.logR_di  = 1.0 / view.logR_d;

    // L2 persisting — same recipe as helmholtz_eos.cu.
    stream = s;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.persistingL2CacheMaxSize > 0
        && bytes <= (std::size_t)prop.accessPolicyMaxWindowSize) {
        std::size_t persist_size = bytes;
        if (persist_size > (std::size_t)prop.persistingL2CacheMaxSize)
            persist_size = prop.persistingL2CacheMaxSize;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size));

        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.base_ptr  = static_cast<void*>(d_data);
        attr.accessPolicyWindow.num_bytes = bytes;
        attr.accessPolicyWindow.hitRatio  = 1.0;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        cudaError_t err = cudaStreamSetAttribute(stream,
            cudaStreamAttributeAccessPolicyWindow, &attr);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "kap_table: cudaStreamSetAttribute failed: %s (non-fatal)\n",
                cudaGetErrorString(err));
        } else {
            std::fprintf(stderr,
                "kap_table: %s (family=%s, Z=%.4g, %dx%dx%d=%.2f KB) pinned to L2\n",
                bin_path, family, Z_file, n_X, n_logT, n_logR,
                bytes / 1024.0);
        }
    } else {
        std::fprintf(stderr,
            "kap_table: L2 persisting unavailable on this device "
            "(cap=%.1f MB, needed=%.2f KB)\n",
            prop.persistingL2CacheMaxSize / 1048576.0, bytes / 1024.0);
    }

    return 0;
}

void KapTable::destroy() {
    if (stream) {
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        cudaCtxResetPersistingL2Cache();
    }
    if (d_data) cudaFree(d_data);
    d_data = nullptr;
    bytes = 0;
    view = {};
}
