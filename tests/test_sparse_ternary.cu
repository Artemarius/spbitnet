#include <gtest/gtest.h>
#include "spbitnet/sparse_ternary_tensor.h"
#include "spbitnet/sparse_ternary_kernels.h"
#include "spbitnet/ternary_tensor.h"
#include "spbitnet/cuda_utils.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Helper: CPU reference sparse GEMV
// Unpacks the sparse tensor to dense int8, then computes y = W * x.
// ---------------------------------------------------------------------------
static void cpu_sparse_gemv(const spbitnet::SparseTernaryTensor& tensor,
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
// Test 1: Pack from dense with a known mask, unpack, verify exact roundtrip.
// ---------------------------------------------------------------------------
TEST(SparseTernary, PackFromDenseRoundtrip) {
    const int rows = 4;
    const int cols = 32;
    const int total = rows * cols;

    // Build deterministic ternary weights: repeating {+1, -1, +1, -1, 0, +1, -1, 0, ...}
    std::vector<int8_t> weights(total);
    for (int i = 0; i < total; ++i) {
        int mod = i % 4;
        if (mod == 0)      weights[i] = +1;
        else if (mod == 1) weights[i] = -1;
        else if (mod == 2) weights[i] = +1;
        else               weights[i] = -1;
    }

    // Build 2:4 mask: for each group of 4, keep positions 0 and 2
    std::vector<uint8_t> mask(total);
    for (int i = 0; i < total; i += 4) {
        mask[i + 0] = 1;
        mask[i + 1] = 0;
        mask[i + 2] = 1;
        mask[i + 3] = 0;
    }

    auto tensor = spbitnet::SparseTernaryTensor::pack_from_dense(
        weights.data(), mask.data(), rows, cols);

    EXPECT_EQ(tensor.rows(), rows);
    EXPECT_EQ(tensor.cols(), cols);

    // Unpack and verify
    std::vector<int8_t> unpacked(total);
    tensor.unpack_to_int8(unpacked.data());

    for (int i = 0; i < total; ++i) {
        int pos_in_group = i % 4;
        if (pos_in_group == 0 || pos_in_group == 2) {
            // Kept positions: should match original weight
            EXPECT_EQ(unpacked[i], weights[i])
                << "Kept position mismatch at index " << i;
        } else {
            // Pruned positions: must be zero
            EXPECT_EQ(unpacked[i], 0)
                << "Pruned position not zero at index " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Pack with automatic 2:4 pruning from float weights with known
// magnitudes, so the selection is deterministic.
// ---------------------------------------------------------------------------
TEST(SparseTernary, PackWithPruning) {
    const int rows = 2;
    const int cols = 8;

    // For each group of 4, the two largest-magnitude values should be kept.
    // Group 0 of row 0: [0.1, -0.9, 0.8, -0.2]
    //   |vals|: [0.1, 0.9, 0.8, 0.2] -> keep indices 1 (|-0.9|) and 2 (|0.8|)
    //   Quantize: -0.9 -> -1 (< -0.5), 0.8 -> +1 (> 0.5)
    //   Expected: [0, -1, +1, 0]
    //
    // Group 1 of row 0: [-0.7, 0.3, -0.6, 0.05]
    //   |vals|: [0.7, 0.3, 0.6, 0.05] -> keep indices 0 (0.7) and 2 (0.6)
    //   Quantize: -0.7 -> -1, -0.6 -> -1
    //   Expected: [-1, 0, -1, 0]
    //
    // Group 0 of row 1: [0.95, -0.1, 0.02, -0.85]
    //   |vals|: [0.95, 0.1, 0.02, 0.85] -> keep indices 0 (0.95) and 3 (0.85)
    //   Quantize: 0.95 -> +1, -0.85 -> -1
    //   Expected: [+1, 0, 0, -1]
    //
    // Group 1 of row 1: [-0.4, 0.9, -0.8, 0.3]
    //   |vals|: [0.4, 0.9, 0.8, 0.3] -> keep indices 1 (0.9) and 2 (0.8)
    //   Quantize: 0.9 -> +1, -0.8 -> -1
    //   Expected: [0, +1, -1, 0]

    std::vector<float> latent = {
        0.1f, -0.9f, 0.8f, -0.2f,   -0.7f, 0.3f, -0.6f, 0.05f,   // row 0
        0.95f, -0.1f, 0.02f, -0.85f, -0.4f, 0.9f, -0.8f, 0.3f    // row 1
    };

    auto tensor = spbitnet::SparseTernaryTensor::pack_with_pruning(
        latent.data(), rows, cols);

    EXPECT_EQ(tensor.rows(), rows);
    EXPECT_EQ(tensor.cols(), cols);

    std::vector<int8_t> unpacked(rows * cols);
    tensor.unpack_to_int8(unpacked.data());

    // Row 0
    int8_t expected_row0[] = { 0, -1, +1, 0,   -1, 0, -1, 0 };
    for (int c = 0; c < cols; ++c) {
        EXPECT_EQ(unpacked[0 * cols + c], expected_row0[c])
            << "Row 0, col " << c;
    }

    // Row 1
    int8_t expected_row1[] = { +1, 0, 0, -1,   0, +1, -1, 0 };
    for (int c = 0; c < cols; ++c) {
        EXPECT_EQ(unpacked[1 * cols + c], expected_row1[c])
            << "Row 1, col " << c;
    }
}

// ---------------------------------------------------------------------------
// Test 3: Verify compression ratio is approximately 75% of dense packed size.
// Dense ternary: 2 bits/weight -> ceil(N/16) uint32 per row
// Sparse ternary: 1.5 bits/weight -> meta + values arrays
// ---------------------------------------------------------------------------
TEST(SparseTernary, CompressionRatio) {
    const int rows = 256;
    const int cols = 2048;
    const int total = rows * cols;

    // Create dummy weights and mask (all +1, mask keeps positions 0,1)
    std::vector<int8_t> weights(total, 1);
    std::vector<uint8_t> mask(total, false);
    for (int i = 0; i < total; i += 4) {
        mask[i + 0] = 1;
        mask[i + 1] = 1;
    }

    auto sparse_tensor = spbitnet::SparseTernaryTensor::pack_from_dense(
        weights.data(), mask.data(), rows, cols);

    // Sparse size in uint32 words
    const size_t sparse_words = sparse_tensor.meta_size() + sparse_tensor.values_size();
    const size_t sparse_bytes = sparse_words * sizeof(uint32_t);

    // Dense ternary: 2 bits/weight, 16 weights per uint32
    auto dense_tensor = spbitnet::TernaryTensor::pack_from_int8(weights.data(), rows, cols);
    const size_t dense_bytes = dense_tensor.packed_size() * sizeof(uint32_t);

    // Compression ratio should be approximately 75% (1.5/2.0)
    double ratio = static_cast<double>(sparse_bytes) / static_cast<double>(dense_bytes);

    // Allow some tolerance for rounding/padding effects
    EXPECT_GT(ratio, 0.70) << "Compression ratio too low: " << ratio;
    EXPECT_LT(ratio, 0.80) << "Compression ratio too high: " << ratio;
}

// ---------------------------------------------------------------------------
// Test 4: GPU unpack must match CPU unpack
// ---------------------------------------------------------------------------
TEST(SparseTernaryGPU, UnpackMatchesCPU) {
    const int rows = 8;
    const int cols = 64;
    const int total = rows * cols;

    // Build deterministic ternary weights with 2:4 mask
    std::vector<int8_t> weights(total);
    std::vector<uint8_t> mask(total, false);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> wdist(-1, 1);

    for (int i = 0; i < total; i += 4) {
        // Assign random ternary weights
        for (int j = 0; j < 4; ++j)
            weights[i + j] = static_cast<int8_t>(wdist(rng));

        // Randomly pick 2 of 4 positions to keep
        int idx[4] = {0, 1, 2, 3};
        std::shuffle(idx, idx + 4, rng);
        mask[i + idx[0]] = 1;
        mask[i + idx[1]] = 1;
    }

    auto tensor = spbitnet::SparseTernaryTensor::pack_from_dense(
        weights.data(), mask.data(), rows, cols);

    // --- CPU reference unpack ---
    std::vector<int8_t> cpu_out(total);
    tensor.unpack_to_int8(cpu_out.data());

    // --- GPU unpack ---
    const int meta_stride = static_cast<int>(tensor.meta_row_stride());
    const int vals_stride = static_cast<int>(tensor.values_row_stride());
    const size_t meta_bytes = tensor.meta_size() * sizeof(uint32_t);
    const size_t vals_bytes = tensor.values_size() * sizeof(uint32_t);

    uint32_t* d_meta   = nullptr;
    uint32_t* d_values = nullptr;
    int8_t*   d_out    = nullptr;

    CUDA_CHECK(cudaMalloc(&d_meta, meta_bytes));
    CUDA_CHECK(cudaMalloc(&d_values, vals_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(int8_t)));

    CUDA_CHECK(cudaMemcpy(d_meta, tensor.meta_data(), meta_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, tensor.values_data(), vals_bytes, cudaMemcpyHostToDevice));

    spbitnet::sparse_ternary_unpack_gpu(d_meta, d_values, d_out,
                                         rows, cols, meta_stride, vals_stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int8_t> gpu_out(total);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, total * sizeof(int8_t), cudaMemcpyDeviceToHost));

    // Compare
    for (int i = 0; i < total; ++i) {
        EXPECT_EQ(gpu_out[i], cpu_out[i])
            << "CPU vs GPU unpack mismatch at index " << i;
    }

    CUDA_CHECK(cudaFree(d_meta));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// Test 5: GPU sparse GEMV correctness against CPU reference (moderate size)
// ---------------------------------------------------------------------------
TEST(SparseTernaryGPU, GemvCorrectness) {
    const int rows = 64;
    const int cols = 256;
    const int total = rows * cols;

    // Build random ternary weights with random 2:4 mask
    std::vector<int8_t> weights(total);
    std::vector<uint8_t> mask(total, false);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> wdist(-1, 1);

    for (int i = 0; i < total; i += 4) {
        for (int j = 0; j < 4; ++j)
            weights[i + j] = static_cast<int8_t>(wdist(rng));

        int idx[4] = {0, 1, 2, 3};
        std::shuffle(idx, idx + 4, rng);
        mask[i + idx[0]] = 1;
        mask[i + idx[1]] = 1;
    }

    auto tensor = spbitnet::SparseTernaryTensor::pack_from_dense(
        weights.data(), mask.data(), rows, cols);

    // Random input vector
    std::vector<int8_t> h_x(cols);
    std::uniform_int_distribution<int> xdist(-5, 5);
    for (auto& v : h_x) v = static_cast<int8_t>(xdist(rng));

    // --- CPU reference GEMV ---
    std::vector<int32_t> cpu_y(rows);
    cpu_sparse_gemv(tensor, h_x.data(), cpu_y.data());

    // --- GPU GEMV ---
    const int meta_stride = static_cast<int>(tensor.meta_row_stride());
    const int vals_stride = static_cast<int>(tensor.values_row_stride());
    const size_t meta_bytes = tensor.meta_size() * sizeof(uint32_t);
    const size_t vals_bytes = tensor.values_size() * sizeof(uint32_t);

    uint32_t* d_meta   = nullptr;
    uint32_t* d_values = nullptr;
    int8_t*   d_x      = nullptr;
    int32_t*  d_y      = nullptr;

    CUDA_CHECK(cudaMalloc(&d_meta, meta_bytes));
    CUDA_CHECK(cudaMalloc(&d_values, vals_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_meta, tensor.meta_data(), meta_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, tensor.values_data(), vals_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), cols * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, rows * sizeof(int32_t)));

    spbitnet::sparse_ternary_gemv_gpu(d_meta, d_values, d_x, d_y,
                                       rows, cols, meta_stride, vals_stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> gpu_y(rows);
    CUDA_CHECK(cudaMemcpy(gpu_y.data(), d_y, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Compare element-by-element
    for (int r = 0; r < rows; ++r) {
        EXPECT_EQ(gpu_y[r], cpu_y[r])
            << "Sparse GEMV mismatch at row " << r
            << " (expected " << cpu_y[r] << ", got " << gpu_y[r] << ")";
    }

    CUDA_CHECK(cudaFree(d_meta));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// ---------------------------------------------------------------------------
// Test 6: GPU sparse GEMV at large dimension (2048x2048) to verify no
// issues at scale (thread count, warp reductions, memory alignment).
// ---------------------------------------------------------------------------
TEST(SparseTernaryGPU, GemvLargeDimension) {
    const int rows = 2048;
    const int cols = 2048;
    const int total = rows * cols;

    // Build random ternary weights with random 2:4 mask
    std::vector<int8_t> weights(total);
    std::vector<uint8_t> mask(total, false);
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> wdist(-1, 1);

    for (int i = 0; i < total; i += 4) {
        for (int j = 0; j < 4; ++j)
            weights[i + j] = static_cast<int8_t>(wdist(rng));

        int idx[4] = {0, 1, 2, 3};
        std::shuffle(idx, idx + 4, rng);
        mask[i + idx[0]] = 1;
        mask[i + idx[1]] = 1;
    }

    auto tensor = spbitnet::SparseTernaryTensor::pack_from_dense(
        weights.data(), mask.data(), rows, cols);

    // Random input vector
    std::vector<int8_t> h_x(cols);
    std::uniform_int_distribution<int> xdist(-5, 5);
    for (auto& v : h_x) v = static_cast<int8_t>(xdist(rng));

    // --- CPU reference GEMV ---
    std::vector<int32_t> cpu_y(rows);
    cpu_sparse_gemv(tensor, h_x.data(), cpu_y.data());

    // --- GPU GEMV ---
    const int meta_stride = static_cast<int>(tensor.meta_row_stride());
    const int vals_stride = static_cast<int>(tensor.values_row_stride());
    const size_t meta_bytes = tensor.meta_size() * sizeof(uint32_t);
    const size_t vals_bytes = tensor.values_size() * sizeof(uint32_t);

    uint32_t* d_meta   = nullptr;
    uint32_t* d_values = nullptr;
    int8_t*   d_x      = nullptr;
    int32_t*  d_y      = nullptr;

    CUDA_CHECK(cudaMalloc(&d_meta, meta_bytes));
    CUDA_CHECK(cudaMalloc(&d_values, vals_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_meta, tensor.meta_data(), meta_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, tensor.values_data(), vals_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), cols * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, rows * sizeof(int32_t)));

    spbitnet::sparse_ternary_gemv_gpu(d_meta, d_values, d_x, d_y,
                                       rows, cols, meta_stride, vals_stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> gpu_y(rows);
    CUDA_CHECK(cudaMemcpy(gpu_y.data(), d_y, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Compare element-by-element
    for (int r = 0; r < rows; ++r) {
        EXPECT_EQ(gpu_y[r], cpu_y[r])
            << "Large GEMV mismatch at row " << r
            << " (expected " << cpu_y[r] << ", got " << gpu_y[r] << ")";
    }

    CUDA_CHECK(cudaFree(d_meta));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}
