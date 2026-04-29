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
        src/gpu/simple_solver.cu
        src/gpu/lm_residual.cu
        src/gpu/lm_precond.cu
        src/gpu/lm_krylov.cu
        src/gpu/lowmach_solver.cu
        src/gpu/gmg_gpu.cu
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
endif()
