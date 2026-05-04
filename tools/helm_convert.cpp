// Standalone ASCII → binary converter for Timmes Helmholtz EOS table.
//
// Usage: helm_convert <ascii_path> <bin_path>
//
// Reads helm_table.dat (Fortran free-format ASCII) and emits a compact
// little-endian float64 binary with a 64-byte magic+metadata header:
//
//   Offset  Size  Field
//   0       8     magic = "HELMv1\0\0"
//   8       4     imax (int32, little-endian)
//   12      4     jmax (int32, little-endian)
//   16      8     tlo  (float64 log10 T low)
//   24      8     thi  (float64 log10 T high)
//   32      8     dlo  (float64 log10 ρ low)
//   40      8     dhi  (float64 log10 ρ high)
//   48      16    reserved (zeros)
//   64      ...   N_total float64 values
//
// Table layout in binary (contiguous):
//   1) f family    — 9 arrays of imax*jmax doubles
//   2) dpdf family — 4 arrays
//   3) ef family   — 4 arrays
//   4) xf family   — 4 arrays
//
// Total size for "twice dense" (imax=541, jmax=201):
//   (9+4+4+4) × 541×201 × 8 = 17.9 MB

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

static constexpr int IMAX = 541;   // log ρ points (inner loop)
static constexpr int JMAX = 201;   // log T points (outer loop)
static constexpr double TLO = 3.0, THI = 13.0;
static constexpr double DLO = -12.0, DHI = 15.0;

static bool read_doubles(std::FILE* fp, double* out, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::fscanf(fp, "%lf", &out[i]) != 1) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <ascii_path> <bin_path>\n", argv[0]);
        return 1;
    }
    const char* ascii_path = argv[1];
    const char* bin_path   = argv[2];

    std::FILE* fp = std::fopen(ascii_path, "r");
    if (!fp) { std::perror(ascii_path); return 2; }

    const size_t N = (size_t)IMAX * JMAX;

    // Storage:  f(9) + dpdf(4) + ef(4) + xf(4) = 21 fields
    const int N_FIELDS = 21;
    std::vector<double> data(N_FIELDS * N, 0.0);

    // Helper: offset into `data` for field f, grid (i, j).
    auto idx = [&](int field, int i, int j) -> size_t {
        return (size_t)field * N + (size_t)j * IMAX + i;
    };

    // -------- Table 1: f + 8 derivatives (9 values per grid point) --------
    std::fprintf(stderr, "reading f table (9 × %dx%d)...\n", IMAX, JMAX);
    for (int j = 0; j < JMAX; ++j) {
        for (int i = 0; i < IMAX; ++i) {
            double v[9];
            if (!read_doubles(fp, v, 9)) {
                std::fprintf(stderr, "short read in f table at i=%d j=%d\n", i, j);
                return 3;
            }
            for (int k = 0; k < 9; ++k) data[idx(k, i, j)] = v[k];
        }
    }

    // -------- Table 2: dpdf (4 values) --------
    std::fprintf(stderr, "reading dpdf table (4 × %dx%d)...\n", IMAX, JMAX);
    for (int j = 0; j < JMAX; ++j) {
        for (int i = 0; i < IMAX; ++i) {
            double v[4];
            if (!read_doubles(fp, v, 4)) {
                std::fprintf(stderr, "short read in dpdf table at i=%d j=%d\n", i, j);
                return 4;
            }
            for (int k = 0; k < 4; ++k) data[idx(9 + k, i, j)] = v[k];
        }
    }

    // -------- Table 3: ef (electron chemical potential, 4 values) --------
    std::fprintf(stderr, "reading ef table (4 × %dx%d)...\n", IMAX, JMAX);
    for (int j = 0; j < JMAX; ++j) {
        for (int i = 0; i < IMAX; ++i) {
            double v[4];
            if (!read_doubles(fp, v, 4)) {
                std::fprintf(stderr, "short read in ef table at i=%d j=%d\n", i, j);
                return 5;
            }
            for (int k = 0; k < 4; ++k) data[idx(13 + k, i, j)] = v[k];
        }
    }

    // -------- Table 4: xf (electron-positron number, 4 values) --------
    std::fprintf(stderr, "reading xf table (4 × %dx%d)...\n", IMAX, JMAX);
    for (int j = 0; j < JMAX; ++j) {
        for (int i = 0; i < IMAX; ++i) {
            double v[4];
            if (!read_doubles(fp, v, 4)) {
                std::fprintf(stderr, "short read in xf table at i=%d j=%d\n", i, j);
                return 6;
            }
            for (int k = 0; k < 4; ++k) data[idx(17 + k, i, j)] = v[k];
        }
    }
    std::fclose(fp);

    // -------- Write binary with header --------
    std::FILE* bp = std::fopen(bin_path, "wb");
    if (!bp) { std::perror(bin_path); return 7; }

    // 64-byte header (16 bytes used + 48 reserved)
    char header[64] = {0};
    std::memcpy(header, "HELMv1\0\0", 8);
    int32_t imax32 = IMAX, jmax32 = JMAX;
    std::memcpy(header + 8, &imax32, 4);
    std::memcpy(header + 12, &jmax32, 4);
    double tlo = TLO, thi = THI, dlo = DLO, dhi = DHI;
    std::memcpy(header + 16, &tlo, 8);
    std::memcpy(header + 24, &thi, 8);
    std::memcpy(header + 32, &dlo, 8);
    std::memcpy(header + 40, &dhi, 8);
    std::fwrite(header, 1, 64, bp);

    size_t wrote = std::fwrite(data.data(), sizeof(double), data.size(), bp);
    std::fclose(bp);
    if (wrote != data.size()) {
        std::fprintf(stderr, "short write: %zu / %zu\n", wrote, data.size());
        return 8;
    }

    double mb = (64 + wrote * 8) / 1048576.0;
    std::fprintf(stderr, "wrote %s  (%.2f MB, %d fields × %dx%d)\n",
                 bin_path, mb, N_FIELDS, IMAX, JMAX);
    return 0;
}
