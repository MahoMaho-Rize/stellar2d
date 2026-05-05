cmake_minimum_required(VERSION 3.18)

option(USE_GPU "Enable CUDA GPU solver (geometric multigrid on device)" OFF)

if(USE_GPU)
    project(stellar2d LANGUAGES CXX CUDA)
else()
    project(stellar2d LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    src/grid.cpp
    src/state.cpp
    src/hydro/reconstruct.cpp
    src/hydro/riemann.cpp
    src/hydro/flux.cpp
    src/hydro/integrate.cpp
    src/gravity/gmg.cpp
    src/bc/boundary.cpp
    src/io/output.cpp
    src/init/lane_emden.cpp
    src/init/sedov.cpp
    src/init/jeans.cpp
    src/init/evrard.cpp
    src/cli/options.cpp
    src/sim/helpers.cpp
    src/sim/setup.cpp
    src/sim/run_loop.cpp
    src/drivers/projection.cpp
    src/drivers/simple.cpp
    src/drivers/ale2d.cpp
    src/drivers/wb2d.cpp
    src/drivers/lowmach.cpp
    src/drivers/cart_lag.cpp
    src/drivers/cart_ale.cpp
    src/drivers/cart_impl.cpp
    src/drivers/cart_ale2.cpp
    src/drivers/fas.cpp
    src/drivers/fas2.cpp
    src/drivers/radial1d.cpp
    src/drivers/pseudo_spectral.cpp
    src/drivers/anelastic_sl.cpp
    src/drivers/sph2d_spectral.cpp
    src/drivers/cpu.cpp
    src/main.cpp
)

if(USE_GPU)
    list(APPEND SOURCES
        # --- implicit / low-Mach / FAS family -------------------------------
        src/gpu/implicit_lowmach/fas_residual.cu
        src/gpu/implicit_lowmach/fas_smoothers.cu
        src/gpu/implicit_lowmach/fas_multigrid.cu
        src/gpu/implicit_lowmach/fas_solver.cu
        src/gpu/implicit_lowmach/fas2_residual.cu
        src/gpu/implicit_lowmach/fas2_smoothers.cu
        src/gpu/implicit_lowmach/fas2_multigrid.cu
        src/gpu/implicit_lowmach/fas2_solver.cu
        src/gpu/implicit_lowmach/simple_solver.cu
        src/gpu/implicit_lowmach/projection_solver.cu
        src/gpu/implicit_lowmach/lm_residual.cu
        src/gpu/implicit_lowmach/lm_precond.cu
        src/gpu/implicit_lowmach/lm_krylov.cu
        src/gpu/implicit_lowmach/lowmach_solver.cu
        # --- explicit 2D Euler ----------------------------------------------
        src/gpu/explicit/strang_solver.cu
        src/gpu/explicit/wb2d_solver.cu
        src/gpu/explicit/wb2d_kernels.cu
        # --- common device utilities ----------------------------------------
        src/gpu/common/gmg_gpu.cu
        # --- Lagrangian / ALE family ----------------------------------------
        src/gpu/ale/ale2d_solver.cu
        src/gpu/ale/ale2d_kernels.cu
        src/gpu/ale/cart_lag_solver.cu
        src/gpu/ale/cart_lag_kernels.cu
        src/gpu/ale/cart_ale_solver.cu
        src/gpu/ale/cart_ale_kernels.cu
        src/gpu/ale/cart_ale2_solver.cu
        src/gpu/ale/cart_ale2_kernels.cu
        src/gpu/ale/cart_impl_solver.cu
        src/gpu/ale/cart_impl_residual.cu
        src/gpu/ale/cart_impl_jfnk.cu
        # --- spectral family ------------------------------------------------
        src/gpu/spectral/pseudo_spectral_solver.cu
        src/gpu/spectral/pseudo_spectral_kernels.cu
        src/gpu/spectral/anelastic_sl_solver.cu
        src/gpu/spectral/anelastic_sl_kernels.cu
        src/gpu/spectral/sph_transforms.cu
        src/gpu/spectral/sph2d_spectral_solver.cu
        src/gpu/spectral/sph2d_spectral_kernels.cu
        # --- 1D Lagrangian implicit -----------------------------------------
        src/gpu/radial1d/radial1d_solver.cu
        src/gpu/radial1d/radial1d_implicit.cu
        src/gpu/radial1d/stellar_profile.cpp
        # --- physics (shared, not under gpu/) -------------------------------
        src/physics/helmholtz_eos.cu
        src/physics/opacity_table.cu
    )
    set(LOWMACH_SOURCES
        src/gpu/implicit_lowmach/lm_residual.cu
        src/gpu/implicit_lowmach/lm_precond.cu
        src/gpu/implicit_lowmach/lm_krylov.cu
        src/gpu/implicit_lowmach/lowmach_solver.cu
        src/gpu/common/gmg_gpu.cu
    )
    if(AMGX_DIR)
        list(APPEND SOURCES src/gpu/common/gpu_solver.cu)
    endif()
endif()

add_executable(stellar2d ${SOURCES})

# Make every gpu/ subdirectory available on the include path, so cross-directory
# #include "gpu_common.cuh" / "gmg_gpu.cuh" / "sph_transforms.cuh" etc. keep
# working after the 2026-05-05 layout reorg (see chore/repo-layout).
set(GPU_INCLUDE_DIRS
    src
    src/gpu/common
    src/gpu/explicit
    src/gpu/implicit_lowmach
    src/gpu/ale
    src/gpu/spectral
    src/gpu/radial1d
)
target_include_directories(stellar2d PRIVATE ${GPU_INCLUDE_DIRS})

set(AMGX_DIR "" CACHE PATH "Path to AmgX installation")

if(USE_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(stellar2d PRIVATE USE_GPU)

    set_target_properties(stellar2d PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()

    target_link_libraries(stellar2d PRIVATE
        CUDA::cudart
        CUDA::cufft
        CUDA::cublas
        CUDA::cusolver
    )

    # AmgX is optional: only needed for --solver compressible (GpuSolver)
    if(AMGX_DIR)
        target_compile_definitions(stellar2d PRIVATE USE_AMGX)
        target_include_directories(stellar2d PRIVATE ${AMGX_DIR}/include)
        target_link_directories(stellar2d PRIVATE ${AMGX_DIR}/lib)
        target_link_libraries(stellar2d PRIVATE amgx CUDA::cusparse)
    endif()
endif()

# ── Tests ────────────────────────────────────────────────────────────
enable_testing()

set(TEST_COMMON_SOURCES
    src/grid.cpp
    src/state.cpp
    src/hydro/reconstruct.cpp
    src/hydro/riemann.cpp
    src/hydro/flux.cpp
    src/hydro/integrate.cpp
    src/gravity/gmg.cpp
    src/bc/boundary.cpp
    src/init/lane_emden.cpp
)

# Host-only tests (no CUDA)
add_executable(test_unit tests/test_unit.cpp ${TEST_COMMON_SOURCES})
target_include_directories(test_unit PRIVATE ${GPU_INCLUDE_DIRS})
add_test(NAME unit COMMAND test_unit)

add_executable(test_exact tests/test_exact.cpp ${TEST_COMMON_SOURCES})
target_include_directories(test_exact PRIVATE ${GPU_INCLUDE_DIRS})
add_test(NAME exact COMMAND test_exact)

add_executable(test_pitfalls tests/test_pitfalls.cpp ${TEST_COMMON_SOURCES}
    src/init/sedov.cpp)
target_include_directories(test_pitfalls PRIVATE ${GPU_INCLUDE_DIRS})
add_test(NAME pitfalls COMMAND test_pitfalls)

# Python integration tests (if pytest is available)
find_program(PYTEST_EXECUTABLE pytest)
if(PYTEST_EXECUTABLE)
    add_test(NAME pytest_fast
        COMMAND ${PYTEST_EXECUTABLE} -m "not slow" --tb=short -q
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_options(stellar2d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:-O3 -march=native>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:-g -O0 -fsanitize=address>>
)

target_link_options(stellar2d PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)

# ── GPU tests ────────────────────────────────────────────────────────
# Helper: register one CUDA test executable with the usual boilerplate
# (GPU_INCLUDE_DIRS, USE_GPU define, CUDA_STANDARD 17, CUDA_SEPARABLE_COMPILATION,
# CUDA::cudart). Pass NO_CTEST to skip add_test() for diagnostic-only binaries.
#
# Usage:
#   add_gpu_test(<name> <source.cu> [EXTRA_SOURCES ...] [NO_CTEST])
function(add_gpu_test name source)
    cmake_parse_arguments(PARSE_ARGV 2 T "NO_CTEST" "" "EXTRA_SOURCES")
    add_executable(test_${name} ${source} ${T_EXTRA_SOURCES})
    target_include_directories(test_${name} PRIVATE ${GPU_INCLUDE_DIRS})
    target_compile_definitions(test_${name} PRIVATE USE_GPU)
    set_target_properties(test_${name} PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_${name} PRIVATE CUDA::cudart)
    if(NOT T_NO_CTEST)
        add_test(NAME ${name} COMMAND test_${name})
    endif()
endfunction()

if(USE_GPU)
    set(FAS_GPU_SOURCES
        src/gpu/implicit_lowmach/fas_residual.cu
        src/gpu/implicit_lowmach/fas_smoothers.cu
        src/gpu/implicit_lowmach/fas_multigrid.cu
        src/gpu/implicit_lowmach/fas_solver.cu
        src/gpu/implicit_lowmach/simple_solver.cu
        src/gpu/implicit_lowmach/projection_solver.cu
        src/gpu/common/gmg_gpu.cu
    )

    # lowmach-family tests (some are diagnostic-only, no CTest)
    add_gpu_test(lowmach            tests/test_lowmach.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${LOWMACH_SOURCES})
    add_gpu_test(solver_diagnosis   tests/test_solver_diagnosis.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${LOWMACH_SOURCES} NO_CTEST)
    add_gpu_test(precond_quality    tests/test_precond_quality.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${LOWMACH_SOURCES} NO_CTEST)
    add_gpu_test(newton_tuning      tests/test_newton_tuning.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${LOWMACH_SOURCES} NO_CTEST)
    add_gpu_test(eps_sweep          tests/test_eps_sweep.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${LOWMACH_SOURCES} NO_CTEST)
    add_gpu_test(mini               tests/test_mini.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${LOWMACH_SOURCES} NO_CTEST)

    # Strang explicit-solver tests
    add_gpu_test(strang_init        tests/test_strang_init.cu
                 EXTRA_SOURCES src/gpu/explicit/strang_solver.cu)
    add_gpu_test(strang_muscl       tests/test_strang_muscl.cu
                 EXTRA_SOURCES src/gpu/explicit/strang_solver.cu)
    add_gpu_test(strang_hllc        tests/test_strang_hllc.cu
                 EXTRA_SOURCES src/gpu/explicit/strang_solver.cu)
    add_gpu_test(strang_step        tests/test_strang_step.cu
                 EXTRA_SOURCES src/gpu/explicit/strang_solver.cu)
    add_gpu_test(strang_unit        tests/test_strang_unit.cu
                 EXTRA_SOURCES src/gpu/explicit/strang_solver.cu)
    add_gpu_test(strang_convergence tests/test_strang_convergence.cu
                 EXTRA_SOURCES src/gpu/explicit/strang_solver.cu)

    # FAS multigrid tests
    add_gpu_test(fas_verify         tests/test_fas_verify.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${FAS_GPU_SOURCES})
    add_gpu_test(fas_diagnose       tests/test_fas_diagnose_hse.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${FAS_GPU_SOURCES} NO_CTEST)

    # Standalone CUDA unit tests (no extra solver sources)
    add_gpu_test(helm_l2            tests/test_helm_l2.cu)
    add_gpu_test(helm_eos           tests/test_helm_eos.cu)
    add_gpu_test(dual               tests/test_dual.cu)
    add_gpu_test(kap_table          tests/test_kap_table.cu)
    add_gpu_test(helm_dual          tests/test_helm_dual.cu
                 EXTRA_SOURCES src/physics/helmholtz_eos.cu)
    add_gpu_test(residual_dual      tests/test_residual_dual.cu
                 EXTRA_SOURCES src/physics/helmholtz_eos.cu)

    # Coverage omnibus
    add_gpu_test(coverage_critical  tests/test_coverage_critical.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${FAS_GPU_SOURCES}
                               ${LOWMACH_SOURCES}
                               src/init/jeans.cpp src/init/evrard.cpp)

    # Stellar profile (Lane-Emden + GYRE structure coefficients) — host-only, no CUDA
    add_executable(test_stellar_profile tests/test_stellar_profile.cpp
        src/gpu/radial1d/stellar_profile.cpp)
    target_include_directories(test_stellar_profile PRIVATE ${GPU_INCLUDE_DIRS})
endif()
