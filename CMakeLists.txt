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
        src/gpu/gpu_solver.cu
        src/gpu/lowmach_solver.cu
        src/gpu/gmg_gpu.cu
    )
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

    if(AMGX_DIR)
        target_include_directories(stellar2d PRIVATE ${AMGX_DIR}/include)
        target_link_directories(stellar2d PRIVATE ${AMGX_DIR}/lib)
    else()
        target_include_directories(stellar2d PRIVATE ${CMAKE_SOURCE_DIR}/../AmgX/include)
        target_link_directories(stellar2d PRIVATE ${CMAKE_SOURCE_DIR}/../AmgX/build)
    endif()

    target_link_libraries(stellar2d PRIVATE
        amgx
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::cusolver
    )
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
