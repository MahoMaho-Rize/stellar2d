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
    src/main.cpp
)

if(USE_GPU)
    list(APPEND SOURCES
        src/gpu/fas_residual.cu
        src/gpu/fas_smoothers.cu
        src/gpu/fas_multigrid.cu
        src/gpu/fas_solver.cu
        src/gpu/fas2_residual.cu
        src/gpu/fas2_smoothers.cu
        src/gpu/fas2_multigrid.cu
        src/gpu/fas2_solver.cu
        src/gpu/simple_solver.cu
        src/gpu/projection_solver.cu
        src/gpu/strang_solver.cu
        src/gpu/lm_residual.cu
        src/gpu/lm_precond.cu
        src/gpu/lm_krylov.cu
        src/gpu/lowmach_solver.cu
        src/gpu/gmg_gpu.cu
        src/gpu/radial1d_solver.cu
        src/gpu/radial1d_implicit.cu
        src/physics/helmholtz_eos.cu
        src/gpu/wb2d_solver.cu
        src/gpu/wb2d_kernels.cu
        src/gpu/ale2d_solver.cu
        src/gpu/ale2d_kernels.cu
        src/gpu/cart_lag_solver.cu
        src/gpu/cart_lag_kernels.cu
        src/gpu/cart_ale_solver.cu
        src/gpu/cart_ale_kernels.cu
        src/gpu/cart_ale2_solver.cu
        src/gpu/cart_ale2_kernels.cu
        src/gpu/cart_impl_solver.cu
        src/gpu/cart_impl_residual.cu
        src/gpu/cart_impl_jfnk.cu
        src/gpu/pseudo_spectral_solver.cu
        src/gpu/pseudo_spectral_kernels.cu
    )
    set(LOWMACH_SOURCES
        src/gpu/lm_residual.cu
        src/gpu/lm_precond.cu
        src/gpu/lm_krylov.cu
        src/gpu/lowmach_solver.cu
        src/gpu/gmg_gpu.cu
    )
    if(AMGX_DIR)
        list(APPEND SOURCES src/gpu/gpu_solver.cu)
    endif()
endif()

add_executable(stellar2d ${SOURCES})

target_include_directories(stellar2d PRIVATE src)

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
    )

    # AmgX is optional: only needed for --solver compressible (GpuSolver)
    if(AMGX_DIR)
        target_compile_definitions(stellar2d PRIVATE USE_AMGX)
        target_include_directories(stellar2d PRIVATE ${AMGX_DIR}/include)
        target_link_directories(stellar2d PRIVATE ${AMGX_DIR}/lib)
        target_link_libraries(stellar2d PRIVATE amgx CUDA::cublas CUDA::cusparse CUDA::cusolver)
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

add_executable(test_unit tests/test_unit.cpp ${TEST_COMMON_SOURCES})
target_include_directories(test_unit PRIVATE src)
add_test(NAME unit COMMAND test_unit)

add_executable(test_exact tests/test_exact.cpp ${TEST_COMMON_SOURCES})
target_include_directories(test_exact PRIVATE src)
add_test(NAME exact COMMAND test_exact)

add_executable(test_pitfalls tests/test_pitfalls.cpp ${TEST_COMMON_SOURCES}
    src/init/sedov.cpp)
target_include_directories(test_pitfalls PRIVATE src)
add_test(NAME pitfalls COMMAND test_pitfalls)

# GPU lowmach solver tests (no AmgX dependency)
if(USE_GPU)
    add_executable(test_lowmach tests/test_lowmach.cu
        ${TEST_COMMON_SOURCES}
        ${LOWMACH_SOURCES}
    )
    target_include_directories(test_lowmach PRIVATE src)
    target_compile_definitions(test_lowmach PRIVATE USE_GPU)
    set_target_properties(test_lowmach PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_lowmach PRIVATE CUDA::cudart)
    add_test(NAME lowmach COMMAND test_lowmach)

    add_executable(test_solver_diagnosis tests/test_solver_diagnosis.cu
        ${TEST_COMMON_SOURCES}
        ${LOWMACH_SOURCES}
    )
    target_include_directories(test_solver_diagnosis PRIVATE src)
    target_compile_definitions(test_solver_diagnosis PRIVATE USE_GPU)
    set_target_properties(test_solver_diagnosis PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_solver_diagnosis PRIVATE CUDA::cudart)

    add_executable(test_precond_quality tests/test_precond_quality.cu
        ${TEST_COMMON_SOURCES}
        ${LOWMACH_SOURCES}
    )
    target_include_directories(test_precond_quality PRIVATE src)
    target_compile_definitions(test_precond_quality PRIVATE USE_GPU)
    set_target_properties(test_precond_quality PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_precond_quality PRIVATE CUDA::cudart)

    add_executable(test_newton_tuning tests/test_newton_tuning.cu
        ${TEST_COMMON_SOURCES}
        ${LOWMACH_SOURCES}
    )
    target_include_directories(test_newton_tuning PRIVATE src)
    target_compile_definitions(test_newton_tuning PRIVATE USE_GPU)
    set_target_properties(test_newton_tuning PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_newton_tuning PRIVATE CUDA::cudart)

    add_executable(test_eps_sweep tests/test_eps_sweep.cu
        ${TEST_COMMON_SOURCES}
        ${LOWMACH_SOURCES}
    )
    target_include_directories(test_eps_sweep PRIVATE src)
    target_compile_definitions(test_eps_sweep PRIVATE USE_GPU)
    set_target_properties(test_eps_sweep PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_eps_sweep PRIVATE CUDA::cudart)
endif()

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

if(USE_GPU)
    add_executable(test_mini tests/test_mini.cu ${TEST_COMMON_SOURCES} ${LOWMACH_SOURCES})
    target_include_directories(test_mini PRIVATE src)
    target_compile_definitions(test_mini PRIVATE USE_GPU)
    set_target_properties(test_mini PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_mini PRIVATE CUDA::cudart)

    add_executable(test_strang_init tests/test_strang_init.cu
        src/gpu/strang_solver.cu)
    target_include_directories(test_strang_init PRIVATE src)
    target_compile_definitions(test_strang_init PRIVATE USE_GPU)
    set_target_properties(test_strang_init PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_strang_init PRIVATE CUDA::cudart)
    add_test(NAME strang_init COMMAND test_strang_init)

    add_executable(test_strang_muscl tests/test_strang_muscl.cu
        src/gpu/strang_solver.cu)
    target_include_directories(test_strang_muscl PRIVATE src)
    target_compile_definitions(test_strang_muscl PRIVATE USE_GPU)
    set_target_properties(test_strang_muscl PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_strang_muscl PRIVATE CUDA::cudart)
    add_test(NAME strang_muscl COMMAND test_strang_muscl)

    add_executable(test_strang_hllc tests/test_strang_hllc.cu
        src/gpu/strang_solver.cu)
    target_include_directories(test_strang_hllc PRIVATE src)
    target_compile_definitions(test_strang_hllc PRIVATE USE_GPU)
    set_target_properties(test_strang_hllc PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_strang_hllc PRIVATE CUDA::cudart)
    add_test(NAME strang_hllc COMMAND test_strang_hllc)

    add_executable(test_strang_step tests/test_strang_step.cu
        src/gpu/strang_solver.cu)
    target_include_directories(test_strang_step PRIVATE src)
    target_compile_definitions(test_strang_step PRIVATE USE_GPU)
    set_target_properties(test_strang_step PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_strang_step PRIVATE CUDA::cudart)
    add_test(NAME strang_step COMMAND test_strang_step)

    add_executable(test_strang_unit tests/test_strang_unit.cu
        src/gpu/strang_solver.cu)
    target_include_directories(test_strang_unit PRIVATE src)
    target_compile_definitions(test_strang_unit PRIVATE USE_GPU)
    set_target_properties(test_strang_unit PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_strang_unit PRIVATE CUDA::cudart)
    add_test(NAME strang_unit COMMAND test_strang_unit)

    add_executable(test_strang_convergence tests/test_strang_convergence.cu
        src/gpu/strang_solver.cu)
    target_include_directories(test_strang_convergence PRIVATE src)
    target_compile_definitions(test_strang_convergence PRIVATE USE_GPU)
    set_target_properties(test_strang_convergence PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_strang_convergence PRIVATE CUDA::cudart)
    add_test(NAME strang_convergence COMMAND test_strang_convergence)

    set(FAS_GPU_SOURCES
        src/gpu/fas_residual.cu
        src/gpu/fas_smoothers.cu
        src/gpu/fas_multigrid.cu
        src/gpu/fas_solver.cu
        src/gpu/simple_solver.cu
        src/gpu/projection_solver.cu
        src/gpu/gmg_gpu.cu
    )
    add_executable(test_fas_verify tests/test_fas_verify.cu
        ${TEST_COMMON_SOURCES}
        ${FAS_GPU_SOURCES})
    target_include_directories(test_fas_verify PRIVATE src)
    target_compile_definitions(test_fas_verify PRIVATE USE_GPU)
    set_target_properties(test_fas_verify PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_fas_verify PRIVATE CUDA::cudart)
    add_test(NAME fas_verify COMMAND test_fas_verify)

    add_executable(test_fas_diagnose tests/test_fas_diagnose_hse.cu
        ${TEST_COMMON_SOURCES}
        ${FAS_GPU_SOURCES})
    target_include_directories(test_fas_diagnose PRIVATE src)
    target_compile_definitions(test_fas_diagnose PRIVATE USE_GPU)
    set_target_properties(test_fas_diagnose PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_fas_diagnose PRIVATE CUDA::cudart)

    add_executable(test_helm_l2 tests/test_helm_l2.cu)
    target_include_directories(test_helm_l2 PRIVATE src)
    target_compile_definitions(test_helm_l2 PRIVATE USE_GPU)
    set_target_properties(test_helm_l2 PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_helm_l2 PRIVATE CUDA::cudart)
    add_test(NAME helm_l2 COMMAND test_helm_l2)

    add_executable(test_helm_eos tests/test_helm_eos.cu)
    target_include_directories(test_helm_eos PRIVATE src)
    target_compile_definitions(test_helm_eos PRIVATE USE_GPU)
    set_target_properties(test_helm_eos PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_helm_eos PRIVATE CUDA::cudart)
    add_test(NAME helm_eos COMMAND test_helm_eos)

    add_executable(test_coverage_critical tests/test_coverage_critical.cu
        ${TEST_COMMON_SOURCES}
        ${FAS_GPU_SOURCES}
        ${LOWMACH_SOURCES}
        src/init/jeans.cpp
        src/init/evrard.cpp)
    target_include_directories(test_coverage_critical PRIVATE src)
    target_compile_definitions(test_coverage_critical PRIVATE USE_GPU)
    set_target_properties(test_coverage_critical PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_coverage_critical PRIVATE CUDA::cudart)
    add_test(NAME coverage_critical COMMAND test_coverage_critical)
endif()
