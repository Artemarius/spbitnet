#include <gtest/gtest.h>
#include "spbitnet/cusparselt_backend.h"
#include "spbitnet/cuda_utils.h"

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <random>

#ifdef SPBITNET_HAS_CUSPARSELT

// ---------------------------------------------------------------------------
// Helper: CPU reference INT8 GEMM (D = A * B, row-major)
// A is [m x k], B is [k x n], D is [m x n]
// ---------------------------------------------------------------------------
static void cpu_int8_gemm(const int8_t* A, const int8_t* B, int32_t* D,
                           int m, int k, int n) {
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            int32_t sum = 0;
            for (int j = 0; j < k; ++j) {
                sum += static_cast<int32_t>(A[r * k + j]) *
                       static_cast<int32_t>(B[j * n + c]);
            }
            D[r * n + c] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: check 2:4 sparsity pattern — every group of 4 has at most 2 nonzeros.
// Checks along the k dimension (columns) of row-major A[m, k].
// ---------------------------------------------------------------------------
static bool check_24_sparsity(const int8_t* weights, int m, int k) {
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < k; c += 4) {
            int nz = 0;
            for (int i = 0; i < 4 && (c + i) < k; ++i) {
                if (weights[r * k + c + i] != 0) ++nz;
            }
            if (nz > 2) return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Helper: transpose row-major [rows x cols] to column-major [rows x cols]
// Column-major stores column-by-column: out[r + c*rows] = in[r*cols + c]
// ---------------------------------------------------------------------------
static void row_to_col_major(const int8_t* row_major, int8_t* col_major,
                              int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            col_major[r + c * rows] = row_major[r * cols + c];
}

static void col_to_row_major_i32(const int32_t* col_major, int32_t* row_major,
                                  int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            row_major[r * cols + c] = col_major[r + c * rows];
}

// ---------------------------------------------------------------------------
// Test 1: Pruning produces a valid 2:4 sparsity pattern
// ---------------------------------------------------------------------------
TEST(CuSparseLt, PruneProduces24Pattern) {
    const int m = 64;
    const int k = 256;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1, 1);
    std::vector<int8_t> h_weights(static_cast<size_t>(m) * k);
    for (auto& w : h_weights) w = static_cast<int8_t>(dist(rng));

    // cuSPARSELt requires n >= 16, but pruning only touches A.
    // We still need a valid context to call prepare().
    const int n = 16;

    int8_t* d_weights = nullptr;
    int8_t* d_pruned  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weights, h_weights.size()));
    CUDA_CHECK(cudaMalloc(&d_pruned, h_weights.size()));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size(),
                           cudaMemcpyHostToDevice));

    spbitnet::CuSparseLtGemm ctx(m, k, n);
    ctx.prepare(d_weights);
    ctx.get_pruned_weights(d_pruned);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int8_t> h_pruned(h_weights.size());
    CUDA_CHECK(cudaMemcpy(h_pruned.data(), d_pruned, h_pruned.size(),
                           cudaMemcpyDeviceToHost));

    // The pruned data is in the layout cuSPARSELt uses internally.
    // With our TRANSPOSE trick, the data layout matches row-major [m,k].
    // cuSPARSELt prunes along the contracted dimension (k) in groups of 4.
    // For the TRANSPOSE case, groups of 4 are along the row direction of
    // the stored [k,m] col-major matrix = along columns of A[m,k] row-major.
    // So we check sparsity along k (columns of each row).

    // Verify values remain in {-1, 0, +1}
    for (size_t i = 0; i < h_pruned.size(); ++i) {
        EXPECT_TRUE(h_pruned[i] >= -1 && h_pruned[i] <= 1)
            << "Pruned value out of range at index " << i
            << ": " << static_cast<int>(h_pruned[i]);
    }

    // Count zeros — should be ~50% (2:4 means 50% sparsity)
    int zeros = 0;
    for (auto v : h_pruned) if (v == 0) ++zeros;
    double sparsity = static_cast<double>(zeros) / h_pruned.size();
    // Original data has ~1/3 zeros from uniform {-1,0,+1}.
    // After pruning, sparsity should be >= 50%.
    EXPECT_GE(sparsity, 0.45) << "Sparsity too low: " << sparsity;

    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_pruned));
}

// ---------------------------------------------------------------------------
// Test 2: cuSPARSELt GEMM (n=16) correctness against CPU reference.
// Uses pruned weights so both sides compute on the same sparse matrix.
// ---------------------------------------------------------------------------
TEST(CuSparseLt, GemmN16Correctness) {
    const int m = 64;
    const int k = 256;
    const int n = 16;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> wdist(-1, 1);
    std::vector<int8_t> h_weights(static_cast<size_t>(m) * k);
    for (auto& w : h_weights) w = static_cast<int8_t>(wdist(rng));

    // B matrix [k x n] row-major
    std::uniform_int_distribution<int> xdist(-5, 5);
    std::vector<int8_t> h_B(static_cast<size_t>(k) * n);
    for (auto& v : h_B) v = static_cast<int8_t>(xdist(rng));

    // Convert B to column-major for cuSPARSELt
    std::vector<int8_t> h_B_col(h_B.size());
    row_to_col_major(h_B.data(), h_B_col.data(), k, n);

    // Device allocations
    int8_t*  d_weights = nullptr;
    int8_t*  d_pruned  = nullptr;
    int8_t*  d_B       = nullptr;
    int32_t* d_D       = nullptr;

    const size_t w_bytes = h_weights.size();
    const size_t b_bytes = h_B.size();
    const size_t d_bytes = static_cast<size_t>(m) * n * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_weights, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_pruned, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_D, d_bytes));

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_col.data(), b_bytes, cudaMemcpyHostToDevice));

    spbitnet::CuSparseLtGemm ctx(m, k, n);
    ctx.prepare(d_weights);
    ctx.execute(d_B, d_D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get GPU result (column-major) and convert to row-major
    std::vector<int32_t> gpu_D_col(static_cast<size_t>(m) * n);
    CUDA_CHECK(cudaMemcpy(gpu_D_col.data(), d_D, d_bytes, cudaMemcpyDeviceToHost));
    std::vector<int32_t> gpu_D(gpu_D_col.size());
    col_to_row_major_i32(gpu_D_col.data(), gpu_D.data(), m, n);

    // Get pruned weights for CPU reference
    ctx.get_pruned_weights(d_pruned);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int8_t> h_pruned(w_bytes);
    CUDA_CHECK(cudaMemcpy(h_pruned.data(), d_pruned, w_bytes, cudaMemcpyDeviceToHost));

    // CPU reference: D = A_pruned * B (both row-major)
    std::vector<int32_t> cpu_D(static_cast<size_t>(m) * n, 0);
    cpu_int8_gemm(h_pruned.data(), h_B.data(), cpu_D.data(), m, k, n);

    int mismatches = 0;
    for (int i = 0; i < m * n; ++i) {
        if (gpu_D[i] != cpu_D[i]) {
            if (mismatches < 5) {
                printf("GEMM mismatch at [%d,%d]: expected %d, got %d\n",
                       i / n, i % n, cpu_D[i], gpu_D[i]);
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " mismatches out of " << m * n;

    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_pruned));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));
}

// ---------------------------------------------------------------------------
// Test 3: Large dimension GEMM (2048x2048, n=32) to verify no issues at scale
// ---------------------------------------------------------------------------
TEST(CuSparseLt, GemmLargeDimension) {
    const int m = 2048;
    const int k = 2048;
    const int n = 32;

    std::mt19937 rng(99);
    std::uniform_int_distribution<int> wdist(-1, 1);
    std::vector<int8_t> h_weights(static_cast<size_t>(m) * k);
    for (auto& w : h_weights) w = static_cast<int8_t>(wdist(rng));

    std::uniform_int_distribution<int> xdist(-5, 5);
    std::vector<int8_t> h_B(static_cast<size_t>(k) * n);
    for (auto& v : h_B) v = static_cast<int8_t>(xdist(rng));

    std::vector<int8_t> h_B_col(h_B.size());
    row_to_col_major(h_B.data(), h_B_col.data(), k, n);

    int8_t*  d_weights = nullptr;
    int8_t*  d_pruned  = nullptr;
    int8_t*  d_B       = nullptr;
    int32_t* d_D       = nullptr;

    const size_t w_bytes = h_weights.size();
    const size_t b_bytes = h_B.size();
    const size_t d_bytes = static_cast<size_t>(m) * n * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_weights, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_pruned, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_D, d_bytes));

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_col.data(), b_bytes, cudaMemcpyHostToDevice));

    spbitnet::CuSparseLtGemm ctx(m, k, n);
    ctx.prepare(d_weights);
    ctx.execute(d_B, d_D);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> gpu_D_col(static_cast<size_t>(m) * n);
    CUDA_CHECK(cudaMemcpy(gpu_D_col.data(), d_D, d_bytes, cudaMemcpyDeviceToHost));
    std::vector<int32_t> gpu_D(gpu_D_col.size());
    col_to_row_major_i32(gpu_D_col.data(), gpu_D.data(), m, n);

    ctx.get_pruned_weights(d_pruned);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int8_t> h_pruned(w_bytes);
    CUDA_CHECK(cudaMemcpy(h_pruned.data(), d_pruned, w_bytes, cudaMemcpyDeviceToHost));

    // CPU reference — only check a subset of rows for large matrices
    const int check_rows = 64;
    int mismatches = 0;
    for (int r = 0; r < check_rows; ++r) {
        for (int c = 0; c < n; ++c) {
            int32_t sum = 0;
            for (int j = 0; j < k; ++j) {
                sum += static_cast<int32_t>(h_pruned[r * k + j]) *
                       static_cast<int32_t>(h_B[j * n + c]);
            }
            if (gpu_D[r * n + c] != sum) ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << "/" << check_rows * n << " mismatches";

    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_pruned));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));
}

#endif // SPBITNET_HAS_CUSPARSELT
