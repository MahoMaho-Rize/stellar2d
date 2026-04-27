cmake_minimum_required(VERSION 3.18)
project(stellar2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_AMGX "Enable NVIDIA AmgX Poisson solver" OFF)
set(AMGX_DIR "" CACHE PATH "Path to AmgX installation")

set(SOURCES
    src/grid.cpp
    src/state.cpp
    src/hydro/reconstruct.cpp
    src/hydro/riemann.cpp
    src/hydro/flux.cpp
    src/hydro/integrate.cpp
    src/gravity/poisson.cpp
    src/gravity/amgx_solver.cpp
    src/bc/boundary.cpp
    src/io/output.cpp
    src/init/lane_emden.cpp
    src/init/sedov.cpp
    src/init/jeans.cpp
    src/init/evrard.cpp
    src/main.cpp
)

add_executable(stellar2d ${SOURCES})

target_include_directories(stellar2d PRIVATE src)

if(USE_AMGX)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(stellar2d PRIVATE USE_AMGX)

    if(AMGX_DIR)
        target_include_directories(stellar2d PRIVATE ${AMGX_DIR}/include)
        target_link_directories(stellar2d PRIVATE ${AMGX_DIR}/lib)
    else()
        # Try to find AmgX from the adjacent directory
        target_include_directories(stellar2d PRIVATE ${CMAKE_SOURCE_DIR}/../AmgX/include)
        target_link_directories(stellar2d PRIVATE ${CMAKE_SOURCE_DIR}/../AmgX/build)
    endif()

    target_link_libraries(stellar2d PRIVATE amgx CUDA::cudart)
endif()

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_options(stellar2d PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-g -O0 -fsanitize=address>
)

target_link_options(stellar2d PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)
