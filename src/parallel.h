#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

inline bool stellar2d_openmp_enabled() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

inline int stellar2d_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
