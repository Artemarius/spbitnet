#include "spbitnet/cuda_utils.h"
#include "spbitnet/ternary_tensor.h"
#include "spbitnet/ternary_kernels.h"
#include "spbitnet/sparse_ternary_tensor.h"
#include "spbitnet/sparse_ternary_kernels.h"
#ifdef SPBITNET_HAS_CUSPARSELT
#include "spbitnet/cusparselt_backend.h"
#endif

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

// ---------------------------------------------------------------------------
// Benchmark harness: measure GPU kernel time via CUDA events
// ---------------------------------------------------------------------------

struct BenchResult {
    float median_us;
    float min_us;
    float max_us;
};

template<typename Func>
BenchResult bench_gpu(Func&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
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
        times[i] = ms * 1000.0f;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());

    return {times[iters / 2], times[0], times[iters - 1]};
}

// ---------------------------------------------------------------------------
// cuBLAS INT8 GEMV/GEMM helper
// ---------------------------------------------------------------------------

void cublas_int8_gemm(cublasHandle_t handle,
                      const int8_t* d_A, const int8_t* d_B, int32_t* d_C,
                      int m, int n, int k) {
    // A is row-major [m,k] = col-major [k,m] transposed
    // B is row-major [k,n] = col-major [n,k] transposed
    // C is row-major [m,n] = col-major [n,m] transposed
    // Using the identity: C_row = A_row * B_row  <==>  C_col^T = B_col^T * A_col^T
    // So we compute C^T = B^T * A^T in col-major.
    const int32_t alpha = 1;
    const int32_t beta  = 0;

    cublasGemmEx(
        handle,
        CUBLAS_OP_T,    // op(A_stored) = transpose (row-major → col-major)
        CUBLAS_OP_N,    // op(B_stored) = no-transpose (B is a vector for GEMV)
        m, n, k,
        &alpha,
        d_A, CUDA_R_8I, k,    // A stored row-major, lda=k
        d_B, CUDA_R_8I, k,    // B stored as [k x n], ldb=k
        &beta,
        d_C, CUDA_R_32I, m,   // C stored as [m x n], ldc=m
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT);
}

// ---------------------------------------------------------------------------
// CPU reference sparse GEMV for correctness check
// ---------------------------------------------------------------------------
static void cpu_sparse_gemv_ref(const spbitnet::SparseTernaryTensor& tensor,
                                const int8_t* x, int32_t* y) {
    const int rows = tensor.rows();
    const int cols = tensor.cols();
    std::vector<int8_t> dense(static_cast<size_t>(rows) * cols);
    tensor.unpack_to_int8(dense.data());
    for (int r = 0; r < rows; ++r) {
        int32_t sum = 0;
        for (int c = 0; c < cols; ++c) {
            sum += static_cast<int32_t>(dense[r * cols + c]) *
                   static_cast<int32_t>(x[c]);
        }
        y[r] = sum;
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
    {2048,  2048,  "2048x2048  (attn proj)"},
    {5632,  2048,  "5632x2048  (FFN up)"},
    {2048,  5632,  "2048x5632  (FFN down)"},
    {2560,  2560,  "2560x2560  (attn proj)"},
    {6912,  2560,  "6912x2560  (FFN up)"},
    {2560,  6912,  "2560x6912  (FFN down)"},
    {4096,  4096,  "4096x4096  (large square)"},
    {8192,  2560,  "8192x2560  (wide FFN)"},
};
static constexpr int NUM_DIMS = sizeof(bench_dims) / sizeof(bench_dims[0]);

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

int main() {
    printf("spbitnet — Kernel Comparison Benchmark (Phase 4)\n");
    printf("=================================================\n\n");

    spbitnet::print_device_info();
    spbitnet::print_vram_usage("pre-benchmark");

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    constexpr int WARMUP = 20;
    constexpr int ITERS  = 100;
    std::mt19937 rng(42);

    // =====================================================================
    // Section 1: GEMV (n=1) — Dense Ternary vs Sparse Ternary vs cuBLAS
    // This is the critical path for autoregressive inference.
    // cuSPARSELt CANNOT do GEMV (requires n >= 16 for INT8 Tensor Cores).
    // =====================================================================

    printf("\n=== GEMV (n=1) — Autoregressive Inference ===\n");
    printf("cuSPARSELt is excluded: INT8 SpMMA requires n >= 16.\n\n");

    printf("%-24s  %10s  %10s  %10s  %10s  %8s\n",
           "Dimension", "Dense(us)", "Sparse(us)", "cuBLAS(us)", "S/cuBLAS", "BW Ratio");
    printf("%-24s  %10s  %10s  %10s  %10s  %8s\n",
           "------------------------", "----------", "----------",
           "----------", "----------", "--------");

    for (int d = 0; d < NUM_DIMS; ++d) {
        const int rows = bench_dims[d].rows;
        const int cols = bench_dims[d].cols;
        const char* label = bench_dims[d].label;

        std::vector<int8_t> h_weights(static_cast<size_t>(rows) * cols);
        std::uniform_int_distribution<int> dist(-1, 1);
        for (auto& w : h_weights) w = static_cast<int8_t>(dist(rng));

        std::vector<int8_t> h_x(cols);
        std::uniform_int_distribution<int> xdist(-5, 5);
        for (auto& v : h_x) v = static_cast<int8_t>(xdist(rng));

        // Dense ternary setup
        auto tensor = spbitnet::TernaryTensor::pack_from_int8(h_weights.data(), rows, cols);
        uint32_t* d_packed = nullptr;
        int8_t*   d_x_tern = nullptr;
        int32_t*  d_y_tern = nullptr;
        CUDA_CHECK(cudaMalloc(&d_packed, tensor.packed_size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_x_tern, cols));
        CUDA_CHECK(cudaMalloc(&d_y_tern, rows * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_packed, tensor.packed_data(),
                              tensor.packed_size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x_tern, h_x.data(), cols, cudaMemcpyHostToDevice));

        // Sparse ternary setup
        std::vector<uint8_t> mask(static_cast<size_t>(rows) * cols, 0);
        for (size_t i = 0; i < mask.size(); i += 4) {
            int idx[4] = {0, 1, 2, 3};
            std::shuffle(idx, idx + 4, rng);
            mask[i + idx[0]] = 1;
            mask[i + idx[1]] = 1;
        }
        auto sparse = spbitnet::SparseTernaryTensor::pack_from_dense(
            h_weights.data(), mask.data(), rows, cols);
        const int ms = static_cast<int>(sparse.meta_row_stride());
        const int vs = static_cast<int>(sparse.values_row_stride());
        const size_t mb = sparse.meta_size() * sizeof(uint32_t);
        const size_t vb = sparse.values_size() * sizeof(uint32_t);

        uint32_t *d_meta, *d_vals;
        int8_t *d_x_sp;
        int32_t *d_y_sp;
        CUDA_CHECK(cudaMalloc(&d_meta, mb));
        CUDA_CHECK(cudaMalloc(&d_vals, vb));
        CUDA_CHECK(cudaMalloc(&d_x_sp, cols));
        CUDA_CHECK(cudaMalloc(&d_y_sp, rows * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_meta, sparse.meta_data(), mb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals, sparse.values_data(), vb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x_sp, h_x.data(), cols, cudaMemcpyHostToDevice));

        // cuBLAS setup
        int8_t *d_A_cb, *d_x_cb;
        int32_t *d_y_cb;
        CUDA_CHECK(cudaMalloc(&d_A_cb, static_cast<size_t>(rows) * cols));
        CUDA_CHECK(cudaMalloc(&d_x_cb, cols));
        CUDA_CHECK(cudaMalloc(&d_y_cb, rows * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_A_cb, h_weights.data(),
                              static_cast<size_t>(rows) * cols, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x_cb, h_x.data(), cols, cudaMemcpyHostToDevice));

        // Correctness check: sparse vs CPU
        CUDA_CHECK(cudaMemset(d_y_sp, 0, rows * sizeof(int32_t)));
        spbitnet::sparse_ternary_gemv_gpu(d_meta, d_vals, d_x_sp, d_y_sp,
                                           rows, cols, ms, vs);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int32_t> cpu_y(rows), gpu_y(rows);
        cpu_sparse_gemv_ref(sparse, h_x.data(), cpu_y.data());
        CUDA_CHECK(cudaMemcpy(gpu_y.data(), d_y_sp, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
        int mismatch = 0;
        for (int i = 0; i < rows; ++i) if (gpu_y[i] != cpu_y[i]) ++mismatch;
        if (mismatch > 0)
            printf("WARNING: %s — %d/%d mismatches!\n", label, mismatch, rows);

        // Benchmark
        auto r_dense = bench_gpu([&]() {
            spbitnet::ternary_gemv_gpu(d_packed, d_x_tern, d_y_tern, rows, cols);
        }, WARMUP, ITERS);

        auto r_sparse = bench_gpu([&]() {
            spbitnet::sparse_ternary_gemv_gpu(d_meta, d_vals, d_x_sp, d_y_sp,
                                               rows, cols, ms, vs);
        }, WARMUP, ITERS);

        auto r_cublas = bench_gpu([&]() {
            cublas_int8_gemm(cublas_handle, d_A_cb, d_x_cb, d_y_cb, rows, 1, cols);
        }, WARMUP, ITERS);

        double sp_bytes = static_cast<double>(mb + vb) + cols + rows * 4;
        double cb_bytes = static_cast<double>(rows) * cols + cols + rows * 4;

        printf("%-24s  %8.1f    %8.1f    %8.1f    %8.2fx  %7.1fx\n",
               label, r_dense.median_us, r_sparse.median_us, r_cublas.median_us,
               r_cublas.median_us / r_sparse.median_us,
               cb_bytes / sp_bytes);

        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_x_tern));
        CUDA_CHECK(cudaFree(d_y_tern));
        CUDA_CHECK(cudaFree(d_meta));
        CUDA_CHECK(cudaFree(d_vals));
        CUDA_CHECK(cudaFree(d_x_sp));
        CUDA_CHECK(cudaFree(d_y_sp));
        CUDA_CHECK(cudaFree(d_A_cb));
        CUDA_CHECK(cudaFree(d_x_cb));
        CUDA_CHECK(cudaFree(d_y_cb));
    }

#ifdef SPBITNET_HAS_CUSPARSELT
    // =====================================================================
    // Section 2: GEMM (n=16,32) — cuSPARSELt vs cuBLAS dense
    // cuSPARSELt uses Sparse Tensor Cores with 2:4 compressed INT8.
    // cuBLAS uses dense INT8 Tensor Cores.
    // This shows where hardware sparsity helps for batched inference.
    // =====================================================================

    printf("\n=== GEMM (batched) — cuSPARSELt vs cuBLAS ===\n");
    printf("cuSPARSELt: 2:4 sparse INT8 via Sparse Tensor Cores\n");
    printf("cuBLAS: dense INT8 via Tensor Cores\n\n");

    int batch_sizes[] = {16, 32};
    constexpr int NUM_BATCHES = 2;

    for (int bi = 0; bi < NUM_BATCHES; ++bi) {
        const int n = batch_sizes[bi];
        printf("--- Batch size n=%d ---\n", n);
        printf("%-24s  %10s  %10s  %10s\n",
               "Dimension", "cuBLAS(us)", "cSpLt(us)", "Speedup");
        printf("%-24s  %10s  %10s  %10s\n",
               "------------------------", "----------", "----------", "----------");

        for (int d = 0; d < NUM_DIMS; ++d) {
            const int m = bench_dims[d].rows;
            const int k = bench_dims[d].cols;

            std::vector<int8_t> h_A(static_cast<size_t>(m) * k);
            std::uniform_int_distribution<int> wdist(-1, 1);
            for (auto& w : h_A) w = static_cast<int8_t>(wdist(rng));

            std::vector<int8_t> h_B(static_cast<size_t>(k) * n);
            std::uniform_int_distribution<int> xdist(-5, 5);
            for (auto& v : h_B) v = static_cast<int8_t>(xdist(rng));

            // cuBLAS dense GEMM
            int8_t *d_A_cb, *d_B_cb;
            int32_t *d_C_cb;
            CUDA_CHECK(cudaMalloc(&d_A_cb, static_cast<size_t>(m) * k));
            CUDA_CHECK(cudaMalloc(&d_B_cb, static_cast<size_t>(k) * n));
            CUDA_CHECK(cudaMalloc(&d_C_cb, static_cast<size_t>(m) * n * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpy(d_A_cb, h_A.data(), static_cast<size_t>(m) * k,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B_cb, h_B.data(), static_cast<size_t>(k) * n,
                                  cudaMemcpyHostToDevice));

            auto r_cublas = bench_gpu([&]() {
                cublas_int8_gemm(cublas_handle, d_A_cb, d_B_cb, d_C_cb, m, n, k);
            }, WARMUP, ITERS);

            // cuSPARSELt sparse GEMM
            int8_t *d_A_sp;
            CUDA_CHECK(cudaMalloc(&d_A_sp, static_cast<size_t>(m) * k));
            CUDA_CHECK(cudaMemcpy(d_A_sp, h_A.data(), static_cast<size_t>(m) * k,
                                  cudaMemcpyHostToDevice));

            // B must be column-major for cuSPARSELt
            std::vector<int8_t> h_B_col(h_B.size());
            for (int r = 0; r < k; ++r)
                for (int c = 0; c < n; ++c)
                    h_B_col[r + c * k] = h_B[r * n + c];

            int8_t *d_B_sp;
            int32_t *d_C_sp;
            CUDA_CHECK(cudaMalloc(&d_B_sp, static_cast<size_t>(k) * n));
            CUDA_CHECK(cudaMalloc(&d_C_sp, static_cast<size_t>(m) * n * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpy(d_B_sp, h_B_col.data(), static_cast<size_t>(k) * n,
                                  cudaMemcpyHostToDevice));

            spbitnet::CuSparseLtGemm csplt(m, k, n);
            csplt.prepare(d_A_sp);

            auto r_csplt = bench_gpu([&]() {
                csplt.execute(d_B_sp, d_C_sp);
            }, WARMUP, ITERS);

            float speedup = r_cublas.median_us / r_csplt.median_us;
            printf("%-24s  %8.1f    %8.1f    %8.2fx\n",
                   bench_dims[d].label,
                   r_cublas.median_us, r_csplt.median_us, speedup);

            CUDA_CHECK(cudaFree(d_A_cb));
            CUDA_CHECK(cudaFree(d_B_cb));
            CUDA_CHECK(cudaFree(d_C_cb));
            CUDA_CHECK(cudaFree(d_A_sp));
            CUDA_CHECK(cudaFree(d_B_sp));
            CUDA_CHECK(cudaFree(d_C_sp));
        }
        printf("\n");
    }
#endif

    printf("Legend:\n");
    printf("  Dense(us)    — Dense ternary GEMV (2-bit packed, add/sub only)\n");
    printf("  Sparse(us)   — Sparse ternary GEMV (2:4 sparsity, 1.5 bits/weight)\n");
    printf("  cuBLAS(us)   — cuBLAS INT8 (dense weights, Tensor Cores)\n");
#ifdef SPBITNET_HAS_CUSPARSELT
    printf("  cSpLt(us)    — cuSPARSELt INT8 SpMMA (2:4 sparse, Sparse Tensor Cores)\n");
    printf("  S/cuBLAS     — cuBLAS / Sparse (>1 = our kernel wins)\n");
    printf("  Speedup      — cuBLAS / cuSPARSELt (>1 = sparsity helps)\n");
#else
    printf("  S/cuBLAS     — cuBLAS / Sparse (>1 = our kernel wins)\n");
    printf("  BW Ratio     — Memory read ratio: cuBLAS / sparse\n");
#endif
    printf("\n");
    spbitnet::print_vram_usage("post-benchmark");

    cublasDestroy(cublas_handle);
    return 0;
}
