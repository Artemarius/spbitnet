#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace spbitnet {

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

inline void print_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("CUDA devices: %d\n\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  SMs: %d\n", prop.multiProcessorCount);
        printf("  Global Memory: %.1f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("  Memory Bandwidth: %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
        printf("  L2 Cache: %.0f KB\n", prop.l2CacheSize / 1024.0);
        printf("  Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);

        // Check for Sparse Tensor Core support (Ampere+ = CC >= 8.0)
        bool has_sparse_tc = (prop.major > 8) || (prop.major == 8 && prop.minor >= 0);
        printf("  Sparse Tensor Cores: %s (CC %d.%d %s 8.0)\n",
               has_sparse_tc ? "YES" : "NO",
               prop.major, prop.minor,
               has_sparse_tc ? ">=" : "<");

        printf("\n");
    }
}

inline void print_vram_usage(const char* label) {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    double free_mb = free_bytes / (1024.0 * 1024.0);
    double total_mb = total_bytes / (1024.0 * 1024.0);
    double used_mb = total_mb - free_mb;
    printf("[VRAM %s] %.1f / %.1f MB used (%.1f MB free)\n",
           label, used_mb, total_mb, free_mb);
}

} // namespace spbitnet
