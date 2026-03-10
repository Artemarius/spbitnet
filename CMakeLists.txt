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

if(SPBITNET_USE_CUSPARSELT)
    # Try user-specified CUSPARSELT_DIR first, then system apt paths
    set(_CUSPARSELT_SEARCH_PATHS "")
    if(DEFINED CUSPARSELT_DIR)
        list(APPEND _CUSPARSELT_SEARCH_PATHS "${CUSPARSELT_DIR}/include"
                                              "${CUSPARSELT_DIR}/lib64")
    endif()
    # System apt package: libcusparselt0-dev-cuda-12
    list(APPEND _CUSPARSELT_SEARCH_PATHS "/usr/include/libcusparseLt/12"
                                          "/usr/lib/x86_64-linux-gnu/libcusparseLt/12")

    # Find header
    find_path(CUSPARSELT_INCLUDE_DIR cusparseLt.h PATHS ${_CUSPARSELT_SEARCH_PATHS} NO_DEFAULT_PATH)
    # Find library
    find_library(CUSPARSELT_LIBRARY NAMES cusparseLt PATHS ${_CUSPARSELT_SEARCH_PATHS} NO_DEFAULT_PATH)

    if(CUSPARSELT_INCLUDE_DIR AND CUSPARSELT_LIBRARY)
        message(STATUS "cuSPARSELt found: ${CUSPARSELT_INCLUDE_DIR}, ${CUSPARSELT_LIBRARY}")
        add_compile_definitions(SPBITNET_HAS_CUSPARSELT)
        include_directories(${CUSPARSELT_INCLUDE_DIR})
    else()
        message(WARNING "cuSPARSELt not found — building without it")
        set(SPBITNET_USE_CUSPARSELT OFF)
    endif()
endif()

# ---- CUDA kernel library ----
set(KERNEL_SOURCES
    src/kernels/ternary_pack.cu
    src/kernels/sparse_ternary.cu
)
if(SPBITNET_USE_CUSPARSELT)
    list(APPEND KERNEL_SOURCES src/cusparselt_backend.cu)
endif()

add_library(spbitnet_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(spbitnet_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spbitnet_kernels PUBLIC cudart)
if(SPBITNET_USE_CUSPARSELT)
    target_link_libraries(spbitnet_kernels PUBLIC ${CUSPARSELT_LIBRARY} cusparse)
endif()

# ---- Main inference binary ----
add_executable(spbitnet_infer
    src/main.cpp
)
target_include_directories(spbitnet_infer PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(spbitnet_infer PRIVATE
    spbitnet_kernels
)
# cuSPARSELt already linked via spbitnet_kernels

# ---- Benchmark binary ----
add_executable(spbitnet_bench
    benchmarks/bench_kernels.cu
)
target_link_libraries(spbitnet_bench PRIVATE spbitnet_kernels cublas)

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

add_executable(spbitnet_sparse_tests
    tests/test_sparse_ternary.cu
)
target_link_libraries(spbitnet_sparse_tests PRIVATE gtest_main spbitnet_kernels)
add_test(NAME spbitnet_sparse_tests COMMAND spbitnet_sparse_tests)

if(SPBITNET_USE_CUSPARSELT)
    add_executable(spbitnet_cusparselt_tests
        tests/test_cusparselt.cu
    )
    target_link_libraries(spbitnet_cusparselt_tests PRIVATE gtest_main spbitnet_kernels cublas)
    add_test(NAME spbitnet_cusparselt_tests COMMAND spbitnet_cusparselt_tests)
endif()
