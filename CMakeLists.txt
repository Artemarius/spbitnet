cmake_minimum_required(VERSION 3.24)
project(spbitnet LANGUAGES CXX CUDA)

# ---- C++ / CUDA standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architecture: RTX 3060 = SM 8.6
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ---- Include paths ----
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---- cuSPARSELt (optional) ----
option(SPBITNET_USE_CUSPARSELT "Enable cuSPARSELt backend" ON)

if(SPBITNET_USE_CUSPARSELT AND DEFINED CUSPARSELT_DIR)
    set(CUSPARSELT_INCLUDE_DIR "${CUSPARSELT_DIR}/include")
    set(CUSPARSELT_LIB_DIR "${CUSPARSELT_DIR}/lib64")
    if(EXISTS "${CUSPARSELT_INCLUDE_DIR}/cusparseLt.h")
        message(STATUS "cuSPARSELt found at ${CUSPARSELT_DIR}")
        add_compile_definitions(SPBITNET_HAS_CUSPARSELT)
        include_directories(${CUSPARSELT_INCLUDE_DIR})
        link_directories(${CUSPARSELT_LIB_DIR})
    else()
        message(WARNING "cuSPARSELt not found at ${CUSPARSELT_DIR} — building without it")
        set(SPBITNET_USE_CUSPARSELT OFF)
    endif()
else()
    message(STATUS "cuSPARSELt disabled or CUSPARSELT_DIR not set — building without it")
    set(SPBITNET_USE_CUSPARSELT OFF)
endif()

# ---- CUDA kernel library ----
add_library(spbitnet_kernels STATIC
    src/kernels/ternary_pack.cu
)
target_include_directories(spbitnet_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spbitnet_kernels PUBLIC cudart)

# ---- Main inference binary ----
add_executable(spbitnet_infer
    src/main.cpp
)
target_link_libraries(spbitnet_infer PRIVATE
    spbitnet_kernels
)
if(SPBITNET_USE_CUSPARSELT)
    target_link_libraries(spbitnet_infer PRIVATE cusparseLt cusparse)
endif()

# ---- Benchmark binary (placeholder) ----
# add_executable(spbitnet_bench
#     benchmarks/bench_kernels.cu
# )

# ---- Google Test ----
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(spbitnet_tests
    tests/test_ternary_pack.cu
)
target_link_libraries(spbitnet_tests PRIVATE gtest_main spbitnet_kernels)
add_test(NAME spbitnet_tests COMMAND spbitnet_tests)
