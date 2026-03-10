#include "spbitnet/cuda_utils.h"
#include "spbitnet/ternary_tensor.h"
#include "spbitnet/ternary_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// ---------------------------------------------------------------------------
// Benchmark harness: measure GPU kernel time via CUDA events
// ---------------------------------------------------------------------------

struct BenchResult {
    float median_us;
    float min_us;
    float max_us;
};

// Run a GPU kernel `warmup` times, then `iters` times, return timing stats.
template<typename Func>
BenchResult bench_gpu(Func&& fn, int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(iters);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms * 1000.0f;  // convert to microseconds
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());

    BenchResult r;
    r.median_us = times[iters / 2];
    r.min_us    = times[0];
    r.max_us    = times[iters - 1];
    return r;
}

// ---------------------------------------------------------------------------
// cuBLAS INT8 GEMV baseline
// Uses cublasGemmEx with INT8 inputs, INT32 output, compute in INT32.
// GEMV is GEMM with N=1.
// ---------------------------------------------------------------------------

struct CublasGemvContext {
    cublasHandle_t handle;
    int8_t*  d_A;     // weight matrix [rows x cols], col-major for cuBLAS
    int8_t*  d_x;     // input vector [cols x 1]
    int32_t* d_y;     // output vector [rows x 1]
    int rows;
    int cols;
};

void cublas_int8_gemv(const CublasGemvContext& ctx) {
    // y = A * x, where A is [rows x cols], x is [cols x 1], y is [rows x 1]
    // cuBLAS uses column-major: we store A as [rows x cols] row-major,
    // which is [cols x rows] column-major, so we compute y = A^T * x
    // in column-major notation. But since we want y = A * x with A row-major,
    // we use: C = alpha * op(A) * op(B) + beta * C
    // with op(A) = CUBLAS_OP_T (transpose), A stored col-major = our row-major.
    const int32_t alpha = 1;
    const int32_t beta  = 0;

    cublasStatus_t status = cublasGemmEx(
        ctx.handle,
        CUBLAS_OP_T,            // transpose A (row-major -> col-major)
        CUBLAS_OP_N,            // B (x) as-is
        ctx.rows,               // m = output rows
        1,                      // n = 1 (GEMV)
        ctx.cols,               // k = inner dimension
        &alpha,
        ctx.d_A, CUDA_R_8I, ctx.cols,   // A: [cols x rows] in col-major = [rows x cols] row-major
        ctx.d_x, CUDA_R_8I, ctx.cols,   // B: [cols x 1]
        &beta,
        ctx.d_y, CUDA_R_32I, ctx.rows,  // C: [rows x 1]
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GemmEx failed: %d\n", static_cast<int>(status));
    }
}

// ---------------------------------------------------------------------------
// Benchmark dimensions: representative of BitNet model layers
// ---------------------------------------------------------------------------

struct BenchDim {
    int rows;
    int cols;
    const char* label;
};

static const BenchDim bench_dims[] = {
    // BitNet-b1.58-large (729M) dimensions
    {2048,  2048,  "2048x2048  (attn proj)"},
    {5632,  2048,  "5632x2048  (FFN up)"},
    {2048,  5632,  "2048x5632  (FFN down)"},

    // BitNet-b1.58-2B-4T (2.4B) dimensions
    {2560,  2560,  "2560x2560  (attn proj)"},
    {6912,  2560,  "6912x2560  (FFN up)"},
    {2560,  6912,  "2560x6912  (FFN down)"},

    // Stress test
    {4096,  4096,  "4096x4096  (large square)"},
    {8192,  2560,  "8192x2560  (wide FFN)"},
};

static constexpr int NUM_DIMS = sizeof(bench_dims) / sizeof(bench_dims[0]);

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

int main() {
    printf("spbitnet — Dense Ternary GEMV vs cuBLAS INT8 GEMV Benchmark\n");
    printf("=============================================================\n\n");

    spbitnet::print_device_info();
    spbitnet::print_vram_usage("pre-benchmark");
    printf("\n");

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasStatus_t cstat = cublasCreate(&cublas_handle);
    if (cstat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS init failed: %d\n", static_cast<int>(cstat));
        return 1;
    }

    constexpr int WARMUP = 20;
    constexpr int ITERS  = 100;

    std::mt19937 rng(42);

    // Table header
    printf("%-24s  %12s  %12s  %12s  %10s\n",
           "Dimension", "Ternary(us)", "cuBLAS(us)", "Speedup", "BW Ratio");
    printf("%-24s  %12s  %12s  %12s  %10s\n",
           "------------------------", "------------", "------------",
           "------------", "----------");

    for (int d = 0; d < NUM_DIMS; ++d) {
        const int rows = bench_dims[d].rows;
        const int cols = bench_dims[d].cols;
        const char* label = bench_dims[d].label;

        // Generate random ternary weights {-1, 0, +1}
        std::vector<int8_t> h_weights(rows * cols);
        std::uniform_int_distribution<int> dist(-1, 1);
        for (auto& w : h_weights) w = static_cast<int8_t>(dist(rng));

        // Random input vector
        std::vector<int8_t> h_x(cols);
        std::uniform_int_distribution<int> xdist(-5, 5);
        for (auto& v : h_x) v = static_cast<int8_t>(xdist(rng));

        // --- Ternary GEMV setup ---
        auto tensor = spbitnet::TernaryTensor::pack_from_int8(h_weights.data(), rows, cols);

        uint32_t* d_packed = nullptr;
        int8_t*   d_x_tern = nullptr;
        int32_t*  d_y_tern = nullptr;

        const size_t packed_bytes = tensor.packed_size() * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
        CUDA_CHECK(cudaMalloc(&d_x_tern, cols * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_y_tern, rows * sizeof(int32_t)));

        CUDA_CHECK(cudaMemcpy(d_packed, tensor.packed_data(), packed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x_tern, h_x.data(), cols * sizeof(int8_t), cudaMemcpyHostToDevice));

        // --- cuBLAS INT8 GEMV setup ---
        int8_t*  d_A_cublas = nullptr;
        int8_t*  d_x_cublas = nullptr;
        int32_t* d_y_cublas = nullptr;

        CUDA_CHECK(cudaMalloc(&d_A_cublas, static_cast<size_t>(rows) * cols * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_x_cublas, cols * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_y_cublas, rows * sizeof(int32_t)));

        CUDA_CHECK(cudaMemcpy(d_A_cublas, h_weights.data(),
                              static_cast<size_t>(rows) * cols * sizeof(int8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x_cublas, h_x.data(), cols * sizeof(int8_t), cudaMemcpyHostToDevice));

        CublasGemvContext ctx;
        ctx.handle = cublas_handle;
        ctx.d_A    = d_A_cublas;
        ctx.d_x    = d_x_cublas;
        ctx.d_y    = d_y_cublas;
        ctx.rows   = rows;
        ctx.cols   = cols;

        // --- Correctness check: verify both produce same result ---
        CUDA_CHECK(cudaMemset(d_y_tern, 0, rows * sizeof(int32_t)));
        spbitnet::ternary_gemv_gpu(d_packed, d_x_tern, d_y_tern, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        cublas_int8_gemv(ctx);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int32_t> y_tern(rows), y_cublas(rows);
        CUDA_CHECK(cudaMemcpy(y_tern.data(), d_y_tern, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y_cublas.data(), d_y_cublas, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));

        int mismatches = 0;
        for (int i = 0; i < rows; ++i) {
            if (y_tern[i] != y_cublas[i]) ++mismatches;
        }
        if (mismatches > 0) {
            printf("WARNING: %s — %d/%d mismatches!\n", label, mismatches, rows);
        }

        // --- Benchmark ---
        BenchResult tern_result = bench_gpu([&]() {
            spbitnet::ternary_gemv_gpu(d_packed, d_x_tern, d_y_tern, rows, cols);
        }, WARMUP, ITERS);

        BenchResult cublas_result = bench_gpu([&]() {
            cublas_int8_gemv(ctx);
        }, WARMUP, ITERS);

        // Memory bandwidth: ternary reads packed_bytes + cols*1 + writes rows*4
        // cuBLAS reads rows*cols*1 + cols*1 + writes rows*4
        double tern_bytes   = static_cast<double>(packed_bytes) + cols + rows * 4;
        double cublas_bytes  = static_cast<double>(rows) * cols + cols + rows * 4;
        double bw_ratio = cublas_bytes / tern_bytes;

        float speedup = cublas_result.median_us / tern_result.median_us;

        printf("%-24s  %9.1f     %9.1f     %9.2fx     %7.1fx\n",
               label,
               tern_result.median_us,
               cublas_result.median_us,
               speedup,
               bw_ratio);

        // Cleanup
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_x_tern));
        CUDA_CHECK(cudaFree(d_y_tern));
        CUDA_CHECK(cudaFree(d_A_cublas));
        CUDA_CHECK(cudaFree(d_x_cublas));
        CUDA_CHECK(cudaFree(d_y_cublas));
    }

    printf("\n");
    printf("Legend:\n");
    printf("  Ternary(us)  — Our dense ternary GEMV (2-bit packed, add/sub only)\n");
    printf("  cuBLAS(us)   — cuBLAS INT8 GEMM with n=1 (full INT8 weights, Tensor Cores)\n");
    printf("  Speedup      — cuBLAS / Ternary (>1 means ternary is faster)\n");
    printf("  BW Ratio     — Memory read ratio: cuBLAS bytes / ternary bytes (~4x expected)\n");
    printf("\n");

    spbitnet::print_vram_usage("post-benchmark");

    cublasDestroy(cublas_handle);
    return 0;
}
