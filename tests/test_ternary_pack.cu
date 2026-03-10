#include <gtest/gtest.h>
#include "spbitnet/ternary_tensor.h"
#include "spbitnet/ternary_kernels.h"
#include "spbitnet/cuda_utils.h"
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Test 1: encode_ternary for representative float values
// ---------------------------------------------------------------------------

TEST(TernaryEncode, BasicValues) {
    // Encoding: 00=0, 01=+1, 10=-1

    // Clear positives (> 0.5) -> 01
    EXPECT_EQ(spbitnet::encode_ternary(1.0f),  0x01);
    EXPECT_EQ(spbitnet::encode_ternary(0.9f),  0x01);

    // Clear negatives (< -0.5) -> 10
    EXPECT_EQ(spbitnet::encode_ternary(-1.0f), 0x02);
    EXPECT_EQ(spbitnet::encode_ternary(-0.6f), 0x02);

    // Near-zero (abs <= 0.5) -> 00
    EXPECT_EQ(spbitnet::encode_ternary(0.0f),  0x00);
    EXPECT_EQ(spbitnet::encode_ternary(0.3f),  0x00);
}

// ---------------------------------------------------------------------------
// Test 2: Pack int8 ternary values, unpack, verify exact round-trip
// ---------------------------------------------------------------------------

TEST(TernaryTensor, PackUnpackRoundtrip) {
    const int rows = 4;
    const int cols = 32;
    const int total = rows * cols;

    // Build a deterministic pattern of {-1, 0, +1}
    std::vector<int8_t> src(total);
    for (int i = 0; i < total; ++i) {
        int mod = i % 3;
        if (mod == 0) src[i] =  0;
        if (mod == 1) src[i] = +1;
        if (mod == 2) src[i] = -1;
    }

    auto tensor = spbitnet::TernaryTensor::pack_from_int8(src.data(), rows, cols);

    EXPECT_EQ(tensor.rows(), rows);
    EXPECT_EQ(tensor.cols(), cols);

    // Unpack and compare
    std::vector<int8_t> dst(total);
    tensor.unpack_to_int8(dst.data());

    for (int i = 0; i < total; ++i) {
        EXPECT_EQ(dst[i], src[i]) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 3: Pack from float, verify get() returns expected ternary values
// ---------------------------------------------------------------------------

TEST(TernaryTensor, PackFromFloat) {
    // 2 rows x 5 logical values, padded internally to 16-aligned
    const int rows = 2;
    const int cols = 5;

    //                             +1    -1    0     0    -1
    //                             +1    +1    0     0    -1
    std::vector<float> src = {
         1.0f, -1.0f,  0.0f,  0.3f, -0.7f,   // row 0
         0.9f,  0.8f,  0.1f, -0.2f, -0.6f     // row 1
    };

    auto tensor = spbitnet::TernaryTensor::pack_from_float(src.data(), rows, cols);

    EXPECT_EQ(tensor.rows(), rows);
    EXPECT_EQ(tensor.cols(), cols);

    // Expected ternary values after quantization (threshold 0.5)
    int8_t expected[2][5] = {
        { +1, -1,  0,  0, -1 },   // row 0
        { +1, +1,  0,  0, -1 }    // row 1
    };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            EXPECT_EQ(tensor.get(r, c), expected[r][c])
                << "Mismatch at (" << r << ", " << c << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Non-multiple-of-16 column count — padding must not corrupt data
// ---------------------------------------------------------------------------

TEST(TernaryTensor, NonMultipleOf16Cols) {
    const int rows = 3;
    const int cols = 13;  // not a multiple of 16
    const int total = rows * cols;

    std::vector<int8_t> src(total);
    for (int i = 0; i < total; ++i) {
        int mod = i % 5;
        if (mod == 0)      src[i] = +1;
        else if (mod == 1)  src[i] = -1;
        else if (mod == 2)  src[i] =  0;
        else if (mod == 3)  src[i] = +1;
        else                src[i] = -1;
    }

    auto tensor = spbitnet::TernaryTensor::pack_from_int8(src.data(), rows, cols);

    EXPECT_EQ(tensor.rows(), rows);
    EXPECT_EQ(tensor.cols(), cols);

    // packed_row_stride = ceil(13 / 16) = 1 uint32 per row
    EXPECT_EQ(tensor.packed_row_stride(), static_cast<size_t>(1));

    // Unpack and verify only the logical (rows * cols) region
    // unpack_to_int8 writes with stride = cols (logical, not padded)
    std::vector<int8_t> dst(rows * cols, 0x7F);  // sentinel fill

    tensor.unpack_to_int8(dst.data());

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            EXPECT_EQ(dst[r * cols + c], src[r * cols + c])
                << "Mismatch at (" << r << ", " << c << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5: GPU unpack must match CPU unpack
// ---------------------------------------------------------------------------

TEST(TernaryGPU, UnpackMatchesCPU) {
    const int rows = 4;
    const int cols = 32;
    const int total = rows * cols;

    // Build source data
    std::vector<int8_t> src(total);
    for (int i = 0; i < total; ++i) {
        int mod = i % 3;
        if (mod == 0) src[i] =  0;
        if (mod == 1) src[i] = +1;
        if (mod == 2) src[i] = -1;
    }

    auto tensor = spbitnet::TernaryTensor::pack_from_int8(src.data(), rows, cols);

    // --- CPU reference unpack ---
    std::vector<int8_t> cpu_out(total);
    tensor.unpack_to_int8(cpu_out.data());

    // --- GPU unpack ---
    const size_t packed_bytes = tensor.packed_size() * sizeof(uint32_t);

    uint32_t* d_packed = nullptr;
    int8_t*   d_out    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(int8_t)));

    CUDA_CHECK(cudaMemcpy(d_packed, tensor.packed_data(),
                           packed_bytes, cudaMemcpyHostToDevice));

    spbitnet::ternary_unpack_gpu(d_packed, d_out,
                                  static_cast<size_t>(total), nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int8_t> gpu_out(total);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out,
                           total * sizeof(int8_t), cudaMemcpyDeviceToHost));

    // Compare
    for (int i = 0; i < total; ++i) {
        EXPECT_EQ(gpu_out[i], cpu_out[i])
            << "CPU vs GPU mismatch at index " << i;
    }

    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// Test 6: GPU GEMV correctness against CPU reference
// ---------------------------------------------------------------------------

TEST(TernaryGPU, GemvCorrectness) {
    const int rows = 8;
    const int cols = 32;

    // Build ternary weight matrix (row-major)
    std::vector<int8_t> weights(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            int mod = (r + c) % 5;
            if (mod == 0)      weights[idx] = +1;
            else if (mod == 1) weights[idx] = -1;
            else if (mod == 2) weights[idx] =  0;
            else if (mod == 3) weights[idx] = +1;
            else               weights[idx] = -1;
        }
    }

    // All-ones input vector (makes expected output = row-sum of weights)
    std::vector<int8_t> x(cols, 1);

    // --- CPU reference: y = W * x ---
    std::vector<int32_t> cpu_y(rows, 0);
    for (int r = 0; r < rows; ++r) {
        int32_t sum = 0;
        for (int c = 0; c < cols; ++c) {
            sum += static_cast<int32_t>(weights[r * cols + c]) *
                   static_cast<int32_t>(x[c]);
        }
        cpu_y[r] = sum;
    }

    // --- Pack weights ---
    auto tensor = spbitnet::TernaryTensor::pack_from_int8(weights.data(), rows, cols);

    // --- GPU GEMV ---
    const size_t packed_bytes = tensor.packed_size() * sizeof(uint32_t);

    uint32_t* d_packed  = nullptr;
    int8_t*   d_x       = nullptr;
    int32_t*  d_y       = nullptr;

    CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_packed, tensor.packed_data(),
                           packed_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(),
                           cols * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, rows * sizeof(int32_t)));

    spbitnet::ternary_gemv_gpu(d_packed, d_x, d_y, rows, cols, nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> gpu_y(rows);
    CUDA_CHECK(cudaMemcpy(gpu_y.data(), d_y,
                           rows * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Compare
    for (int r = 0; r < rows; ++r) {
        EXPECT_EQ(gpu_y[r], cpu_y[r])
            << "GEMV mismatch at row " << r
            << " (expected " << cpu_y[r] << ", got " << gpu_y[r] << ")";
    }

    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}
