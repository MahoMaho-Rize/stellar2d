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
    src/drivers/athena_vl2.cpp
    src/drivers/athena_mhd.cpp
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
        src/gpu/explicit/athena_vl2_solver.cu
        src/gpu/explicit/athena_vl2_kernels.cu
        src/gpu/explicit/athena_mhd_solver.cu
        src/gpu/explicit/athena_mhd_kernels.cu
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
        src/gpu/ale/cart_ale2_trace.cu
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
    # End-to-end GPU pytest suite (tst/ framework, Phase 2 onwards).
    # Needs the built stellar2d binary; STELLAR2D_BIN env points at it.
    add_test(NAME pytest_gpu_fast
        COMMAND ${PYTEST_EXECUTABLE} -m fast --tb=short -q
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tst)
    set_tests_properties(pytest_gpu_fast PROPERTIES
        LABELS "fast"
        ENVIRONMENT "STELLAR2D_BIN=${CMAKE_BINARY_DIR}/stellar2d")
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
        # Default label `fast` (< 30 s).  Callers can re-label via
        # set_tests_properties(${name} PROPERTIES LABELS "slow") to mark
        # longer tests for exclusion from the `ctest -L fast` filter.
        set_tests_properties(${name} PROPERTIES LABELS "fast")
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
    # Regression lock for 2026-05-07 S_E = ρv·g removal (P32)
    add_gpu_test(lowmach_s_e_regression  tests/test_lowmach_s_e_regression.cu
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
    # Linear acoustic wave (complements entropy_wave: exercises HLLC's
    # compressible characteristics + pressure-gradient coupling).
    add_gpu_test(strang_linwave_convergence
                 tests/test_strang_linwave_convergence.cu
                 EXTRA_SOURCES src/gpu/explicit/strang_solver.cu)

    # FAS multigrid tests
    add_gpu_test(fas_verify         tests/test_fas_verify.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${FAS_GPU_SOURCES})
    add_gpu_test(fas_diagnose       tests/test_fas_diagnose_hse.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES} ${FAS_GPU_SOURCES} NO_CTEST)

    # Pseudo-spectral (2D incompressible NS) Taylor-Green analytic-decay test.
    # Links cufft via the stellar2d target's CUDA::cufft; replicate here.
    add_gpu_test(pseudo_spectral_taylor_green
                 tests/test_pseudo_spectral_taylor_green.cu
                 EXTRA_SOURCES src/gpu/spectral/pseudo_spectral_solver.cu
                               src/gpu/spectral/pseudo_spectral_kernels.cu)
    target_link_libraries(test_pseudo_spectral_taylor_green PRIVATE
        CUDA::cufft CUDA::cublas)

    # Pseudo-spectral inviscid KE+enstrophy conservation on a multi-mode
    # IC (double-shear layer).  Complements the single-mode TG test by
    # exercising the nonlinear convection kernel: with ν=0 and skew-
    # symmetric convection the scheme is bit-exact conservative, so we
    # lock |ΔKE/KE|<1e-10 and |ΔΩ/Ω|<1e-9 over ~60 steps.  Regressions in
    # skew-symmetric cancellation, dealias sign, or fused-kernel ordering
    # break this immediately.
    add_gpu_test(pseudo_spectral_inviscid_conserv
                 tests/test_pseudo_spectral_inviscid_conserv.cu
                 EXTRA_SOURCES src/gpu/spectral/pseudo_spectral_solver.cu
                               src/gpu/spectral/pseudo_spectral_kernels.cu)
    target_link_libraries(test_pseudo_spectral_inviscid_conserv PRIVATE
        CUDA::cufft CUDA::cublas)

    # anelastic_sl g-mode EVP convergence (Boussinesq + constant N²,
    # analytic dispersion ω² = N² kx²/(kx² + (n_g π/Ly)²)).
    add_gpu_test(anelastic_sl_gmode_convergence
                 tests/test_anelastic_sl_gmode_convergence.cu
                 EXTRA_SOURCES src/gpu/spectral/anelastic_sl_solver.cu
                               src/gpu/spectral/anelastic_sl_kernels.cu)
    target_link_libraries(test_anelastic_sl_gmode_convergence PRIVATE
        CUDA::cufft CUDA::cublas CUDA::cusolver)

    # anelastic_sl time-domain propagator test: 10 periods of a pure
    # g-mode; locks Hamiltonian conservation + phase accuracy.  This
    # is the first TD-path regression on anelastic_sl.
    add_gpu_test(anelastic_sl_td_gmode
                 tests/test_anelastic_sl_td_gmode.cu
                 EXTRA_SOURCES src/gpu/spectral/anelastic_sl_solver.cu
                               src/gpu/spectral/anelastic_sl_kernels.cu)
    target_link_libraries(test_anelastic_sl_td_gmode PRIVATE
        CUDA::cufft CUDA::cublas CUDA::cusolver)

    # anelastic_sl nonlinear path regression: at amp → 0 the Strang
    # split NL integrator should recover the exp propagator.  Locks
    # the full advection block + Galerkin projection + DC zero path.
    add_gpu_test(anelastic_sl_td_nonlinear
                 tests/test_anelastic_sl_td_nonlinear.cu
                 EXTRA_SOURCES src/gpu/spectral/anelastic_sl_solver.cu
                               src/gpu/spectral/anelastic_sl_kernels.cu)
    target_link_libraries(test_anelastic_sl_td_nonlinear PRIVATE
        CUDA::cufft CUDA::cublas CUDA::cusolver)

    # cart_ale2 periodic uniform-advection regression (P30/P31 lock)
    add_gpu_test(cart_ale2_uniform_advect
                 tests/test_cart_ale2_uniform_advect.cu
                 EXTRA_SOURCES src/gpu/ale/cart_ale2_solver.cu
                               src/gpu/ale/cart_ale2_kernels.cu
                               src/gpu/ale/cart_ale2_trace.cu)

    # Phase 1 (testing_infrastructure_plan) regression locks
    add_gpu_test(ale2_hse_stratified_reflect
                 tests/test_ale2_hse_stratified_reflect.cu
                 EXTRA_SOURCES src/gpu/ale/cart_ale2_solver.cu
                               src/gpu/ale/cart_ale2_kernels.cu
                               src/gpu/ale/cart_ale2_trace.cu)
    add_gpu_test(ale2_phase_m_compensation
                 tests/test_ale2_phase_m_compensation.cu
                 EXTRA_SOURCES src/gpu/ale/cart_ale2_solver.cu
                               src/gpu/ale/cart_ale2_kernels.cu
                               src/gpu/ale/cart_ale2_trace.cu)
    add_gpu_test(ale2_symmetry
                 tests/test_ale2_symmetry.cu
                 EXTRA_SOURCES src/gpu/ale/cart_ale2_solver.cu
                               src/gpu/ale/cart_ale2_kernels.cu
                               src/gpu/ale/cart_ale2_trace.cu)
    add_gpu_test(athena_vl2_sod
                 tests/test_athena_vl2_sod.cu
                 EXTRA_SOURCES src/gpu/explicit/athena_vl2_solver.cu
                               src/gpu/explicit/athena_vl2_kernels.cu)
    add_gpu_test(athena_mhd_field_loop
                 tests/test_athena_mhd_field_loop.cu
                 EXTRA_SOURCES src/gpu/explicit/athena_mhd_solver.cu
                               src/gpu/explicit/athena_mhd_kernels.cu)
    add_gpu_test(athena_mhd_brio_wu
                 tests/test_athena_mhd_brio_wu.cu
                 EXTRA_SOURCES src/gpu/explicit/athena_mhd_solver.cu
                               src/gpu/explicit/athena_mhd_kernels.cu)
    add_gpu_test(athena_mhd_linwave_convergence
                 tests/test_athena_mhd_linwave_convergence.cu
                 EXTRA_SOURCES src/gpu/explicit/athena_mhd_solver.cu
                               src/gpu/explicit/athena_mhd_kernels.cu)
    add_gpu_test(athena_mhd_benchmarks
                 tests/test_athena_mhd_benchmarks.cu
                 EXTRA_SOURCES src/gpu/explicit/athena_mhd_solver.cu
                               src/gpu/explicit/athena_mhd_kernels.cu)

    # Standalone CUDA unit tests (no extra solver sources)
    add_gpu_test(helm_l2            tests/test_helm_l2.cu)
    add_gpu_test(helm_eos           tests/test_helm_eos.cu)
    add_gpu_test(dual               tests/test_dual.cu)
    add_gpu_test(kap_table          tests/test_kap_table.cu)
    add_gpu_test(helm_dual          tests/test_helm_dual.cu
                 EXTRA_SOURCES src/physics/helmholtz_eos.cu)
    # Tests above load third_party tables (helm_table.bin, *.kapbin) that are
    # not in-tree. When the table is missing they exit 77 → ctest SKIPPED
    # instead of PASSED, so a missing fixture doesn't masquerade as coverage.
    set_tests_properties(helm_l2 helm_eos kap_table helm_dual
                         PROPERTIES SKIP_RETURN_CODE 77)
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

    # radial1d HSE static-stability regression: Lane-Emden n=3/2 + γ=5/3 is
    # exact static HSE; locks that the explicit RK2 path preserves it.
    add_gpu_test(radial1d_hse_static
                 tests/test_radial1d_hse_static.cu
                 EXTRA_SOURCES ${TEST_COMMON_SOURCES}
                               src/gpu/radial1d/radial1d_solver.cu
                               src/gpu/radial1d/radial1d_implicit.cu
                               src/gpu/radial1d/stellar_profile.cpp
                               src/physics/helmholtz_eos.cu
                               src/physics/opacity_table.cu)
    target_link_libraries(test_radial1d_hse_static PRIVATE CUDA::cusolver)
endif()
