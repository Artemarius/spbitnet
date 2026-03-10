#include "spbitnet/cuda_utils.h"
#include "spbitnet/inference_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace spbitnet;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<half> to_half(const std::vector<float>& v) {
    std::vector<half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        h[i] = __float2half(v[i]);
    return h;
}

static std::vector<float> from_half(const std::vector<half>& h) {
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); ++i)
        v[i] = __half2float(h[i]);
    return v;
}

template <typename T>
static T* upload(const T* data, size_t count) {
    T* d;
    CUDA_CHECK(cudaMalloc(&d, count * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, data, count * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
static std::vector<T> download(const T* d, size_t count) {
    std::vector<T> h(count);
    CUDA_CHECK(cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

TEST(InferenceKernels, RMSNorm) {
    const int size = 64;
    const float eps = 1e-5f;

    // Input: [1, 2, 3, ..., 64] scaled down
    std::vector<float> h_input(size), h_weight(size);
    for (int i = 0; i < size; ++i) {
        h_input[i]  = (i + 1.0f) / size;
        h_weight[i] = 1.0f;  // identity weight
    }

    // CPU reference
    float sum_sq = 0.0f;
    for (auto x : h_input) sum_sq += x * x;
    float rms_scale = 1.0f / sqrtf(sum_sq / size + eps);
    std::vector<float> expected(size);
    for (int i = 0; i < size; ++i)
        expected[i] = h_input[i] * rms_scale * h_weight[i];

    auto d_input  = upload(to_half(h_input).data(), size);
    auto d_weight = upload(h_weight.data(), size);
    half* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));

    rms_norm_gpu(d_input, d_weight, d_output, size, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = from_half(download(d_output, size));
    for (int i = 0; i < size; ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-2f)
            << "mismatch at i=" << i;

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
}

TEST(InferenceKernels, RMSNormLargeVector) {
    const int size = 2560;  // BitNet hidden_size
    const float eps = 1e-5f;

    std::vector<float> h_input(size), h_weight(size);
    srand(42);
    for (int i = 0; i < size; ++i) {
        h_input[i]  = (rand() % 2000 - 1000) / 1000.0f;
        h_weight[i] = (rand() % 1000 + 500) / 1000.0f;
    }

    float sum_sq = 0.0f;
    for (auto x : h_input) sum_sq += x * x;
    float rms_scale = 1.0f / sqrtf(sum_sq / size + eps);

    auto d_input  = upload(to_half(h_input).data(), size);
    auto d_weight = upload(h_weight.data(), size);
    half* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));

    rms_norm_gpu(d_input, d_weight, d_output, size, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = from_half(download(d_output, size));
    for (int i = 0; i < size; ++i) {
        float expected = h_input[i] * rms_scale * h_weight[i];
        EXPECT_NEAR(result[i], expected, 5e-2f)
            << "mismatch at i=" << i;
    }

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
}

// ---------------------------------------------------------------------------
// Absmax quantize + dequantize roundtrip
// ---------------------------------------------------------------------------

TEST(InferenceKernels, AbsmaxQuantizeDequantize) {
    const int size = 128;
    const float gamma = 0.5f;

    std::vector<float> h_input(size);
    srand(123);
    for (int i = 0; i < size; ++i)
        h_input[i] = (rand() % 2000 - 1000) / 500.0f;  // [-2, 2]

    // Upload
    auto d_input = upload(to_half(h_input).data(), size);
    int8_t* d_quant;
    int32_t* d_int_out;
    float* d_absmax;
    half* d_output;
    CUDA_CHECK(cudaMalloc(&d_quant, size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_int_out, size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_absmax, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));

    absmax_quantize_gpu(d_input, d_quant, d_absmax, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check absmax
    float h_absmax;
    CUDA_CHECK(cudaMemcpy(&h_absmax, d_absmax, sizeof(float), cudaMemcpyDeviceToHost));
    float expected_absmax = 0.0f;
    for (auto x : h_input) expected_absmax = std::max(expected_absmax, fabsf(x));
    EXPECT_NEAR(h_absmax, expected_absmax, 0.05f);

    // Check quantized values
    auto h_quant = download(d_quant, size);
    for (int i = 0; i < size; ++i) {
        float expected_q = roundf(h_input[i] * 127.0f / expected_absmax);
        expected_q = std::max(-128.0f, std::min(127.0f, expected_q));
        EXPECT_NEAR(static_cast<float>(h_quant[i]), expected_q, 1.5f)
            << "quant mismatch at i=" << i;
    }

    // Simulate GEMV with identity: int_out[i] = quant[i] (as if W=identity ternary)
    // For a simple roundtrip test, just copy quant to int32
    std::vector<int32_t> h_int32(size);
    for (int i = 0; i < size; ++i) h_int32[i] = h_quant[i];
    CUDA_CHECK(cudaMemcpy(d_int_out, h_int32.data(), size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    dequantize_gpu(d_int_out, d_output, d_absmax, gamma, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = from_half(download(d_output, size));
    for (int i = 0; i < size; ++i) {
        // Expected: quant[i] * gamma * absmax / 127
        float expected = h_quant[i] * gamma * h_absmax / 127.0f;
        EXPECT_NEAR(result[i], expected, 0.02f)
            << "dequant mismatch at i=" << i;
    }

    cudaFree(d_input); cudaFree(d_quant); cudaFree(d_int_out);
    cudaFree(d_absmax); cudaFree(d_output);
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

TEST(InferenceKernels, RoPE) {
    const int num_heads = 2;
    const int head_dim = 8;
    const int total = num_heads * head_dim;
    const int pos = 3;
    const float theta = 10000.0f;

    std::vector<float> h_input(total);
    for (int i = 0; i < total; ++i) h_input[i] = (i + 1.0f);

    auto d_vec = upload(to_half(h_input).data(), total);
    rope_gpu(d_vec, num_heads, head_dim, pos, theta);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = from_half(download(d_vec, total));

    // CPU reference
    for (int h = 0; h < num_heads; ++h) {
        for (int p = 0; p < head_dim / 2; ++p) {
            float freq  = 1.0f / powf(theta, (float)(2 * p) / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            int idx = h * head_dim + 2 * p;
            float x0 = h_input[idx];
            float x1 = h_input[idx + 1];
            float exp0 = x0 * cos_a - x1 * sin_a;
            float exp1 = x0 * sin_a + x1 * cos_a;

            EXPECT_NEAR(result[idx], exp0, 0.05f)
                << "RoPE mismatch at head=" << h << " pair=" << p << " dim0";
            EXPECT_NEAR(result[idx + 1], exp1, 0.05f)
                << "RoPE mismatch at head=" << h << " pair=" << p << " dim1";
        }
    }

    cudaFree(d_vec);
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

TEST(InferenceKernels, Softmax) {
    const int num_heads = 2;
    const int max_seq = 16;
    const int seq_len = 4;

    std::vector<float> h_scores(num_heads * max_seq, 0.0f);
    // Head 0: [1, 2, 3, 4, ...]
    // Head 1: [4, 3, 2, 1, ...]
    for (int j = 0; j < seq_len; ++j) {
        h_scores[0 * max_seq + j] = j + 1.0f;
        h_scores[1 * max_seq + j] = seq_len - j;
    }

    auto d_scores = upload(h_scores.data(), num_heads * max_seq);
    softmax_gpu(d_scores, num_heads, max_seq, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = download(d_scores, num_heads * max_seq);

    // Check each head sums to 1 and probabilities are correct
    for (int h = 0; h < num_heads; ++h) {
        // CPU reference
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; ++j)
            max_val = std::max(max_val, h_scores[h * max_seq + j]);
        float sum = 0.0f;
        std::vector<float> expected(seq_len);
        for (int j = 0; j < seq_len; ++j) {
            expected[j] = expf(h_scores[h * max_seq + j] - max_val);
            sum += expected[j];
        }
        for (int j = 0; j < seq_len; ++j) expected[j] /= sum;

        float result_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            EXPECT_NEAR(result[h * max_seq + j], expected[j], 1e-5f)
                << "softmax mismatch h=" << h << " j=" << j;
            result_sum += result[h * max_seq + j];
        }
        EXPECT_NEAR(result_sum, 1.0f, 1e-5f);
    }

    cudaFree(d_scores);
}

// ---------------------------------------------------------------------------
// Residual add
// ---------------------------------------------------------------------------

TEST(InferenceKernels, ResidualAdd) {
    const int size = 256;
    std::vector<float> ha(size), hb(size);
    for (int i = 0; i < size; ++i) {
        ha[i] = i * 0.1f;
        hb[i] = -i * 0.05f + 1.0f;
    }

    auto d_a = upload(to_half(ha).data(), size);
    auto d_b = upload(to_half(hb).data(), size);
    half* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, size * sizeof(half)));

    residual_add_gpu(d_a, d_b, d_out, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = from_half(download(d_out, size));
    for (int i = 0; i < size; ++i)
        EXPECT_NEAR(result[i], ha[i] + hb[i], 0.05f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// ReLU² + multiply
// ---------------------------------------------------------------------------

TEST(InferenceKernels, ReLU2Mul) {
    const int size = 64;
    std::vector<float> h_gate(size), h_up(size);
    for (int i = 0; i < size; ++i) {
        h_gate[i] = (i - 32) * 0.1f;   // mix of positive and negative
        h_up[i]   = 1.0f + i * 0.01f;
    }

    auto d_gate = upload(to_half(h_gate).data(), size);
    auto d_up   = upload(to_half(h_up).data(), size);
    half* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, size * sizeof(half)));

    relu2_mul_gpu(d_gate, d_up, d_out, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = from_half(download(d_out, size));
    for (int i = 0; i < size; ++i) {
        float relu = std::max(0.0f, h_gate[i]);
        float expected = relu * relu * h_up[i];
        EXPECT_NEAR(result[i], expected, 0.1f)
            << "relu2_mul mismatch at i=" << i;
    }

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// Half GEMV
// ---------------------------------------------------------------------------

TEST(InferenceKernels, HalfGEMV) {
    const int rows = 32;
    const int cols = 64;

    std::vector<float> h_W(rows * cols), h_x(cols);
    srand(42);
    for (int i = 0; i < rows * cols; ++i)
        h_W[i] = (rand() % 2000 - 1000) / 1000.0f;
    for (int i = 0; i < cols; ++i)
        h_x[i] = (rand() % 2000 - 1000) / 1000.0f;

    // CPU reference
    std::vector<float> expected(rows, 0.0f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            expected[r] += h_W[r * cols + c] * h_x[c];

    auto d_W = upload(to_half(h_W).data(), rows * cols);
    auto d_x = upload(to_half(h_x).data(), cols);
    float* d_y;
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));

    half_gemv_gpu(d_W, d_x, d_y, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = download(d_y, rows);
    for (int r = 0; r < rows; ++r)
        EXPECT_NEAR(result[r], expected[r], 0.5f)
            << "gemv mismatch at row=" << r;

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y);
}

// ---------------------------------------------------------------------------
// Attention (scores + softmax + output) integration
// ---------------------------------------------------------------------------

TEST(InferenceKernels, AttentionSingleHead) {
    const int num_heads = 1;
    const int num_kv_heads = 1;
    const int head_dim = 4;
    const int max_seq = 8;
    const int seq_len = 3;

    // Q: (1, 4)
    std::vector<float> h_Q = {1.0f, 0.0f, 1.0f, 0.0f};

    // K_cache: (1, max_seq, 4) — first 3 positions filled
    std::vector<float> h_K(max_seq * head_dim, 0.0f);
    // pos 0: [1, 0, 0, 0]
    h_K[0*head_dim + 0] = 1.0f;
    // pos 1: [0, 1, 0, 1]
    h_K[1*head_dim + 1] = 1.0f; h_K[1*head_dim + 3] = 1.0f;
    // pos 2: [1, 0, 1, 0]
    h_K[2*head_dim + 0] = 1.0f; h_K[2*head_dim + 2] = 1.0f;

    // V_cache: same layout
    std::vector<float> h_V(max_seq * head_dim, 0.0f);
    // pos 0: [10, 0, 0, 0]
    h_V[0*head_dim + 0] = 10.0f;
    // pos 1: [0, 20, 0, 0]
    h_V[1*head_dim + 1] = 20.0f;
    // pos 2: [0, 0, 30, 0]
    h_V[2*head_dim + 2] = 30.0f;

    auto d_Q = upload(to_half(h_Q).data(), num_heads * head_dim);
    auto d_K = upload(to_half(h_K).data(), max_seq * head_dim);
    auto d_V = upload(to_half(h_V).data(), max_seq * head_dim);

    float* d_scores;
    half* d_out;
    CUDA_CHECK(cudaMalloc(&d_scores, num_heads * max_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, num_heads * head_dim * sizeof(half)));

    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    attention_scores_gpu(d_Q, d_K, d_scores,
                         num_heads, num_kv_heads, head_dim,
                         max_seq, seq_len, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto scores = download(d_scores, num_heads * max_seq);
    // Q=[1,0,1,0], K0=[1,0,0,0] → dot=1, K1=[0,1,0,1] → dot=0, K2=[1,0,1,0] → dot=2
    // scaled by 1/sqrt(4)=0.5
    EXPECT_NEAR(scores[0], 0.5f, 1e-3f);
    EXPECT_NEAR(scores[1], 0.0f, 1e-3f);
    EXPECT_NEAR(scores[2], 1.0f, 1e-3f);

    softmax_gpu(d_scores, num_heads, max_seq, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    scores = download(d_scores, num_heads * max_seq);
    float sum = scores[0] + scores[1] + scores[2];
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
    // score[2] should be highest (raw=1.0), score[1] lowest (raw=0.0)
    EXPECT_GT(scores[2], scores[0]);
    EXPECT_GT(scores[0], scores[1]);

    attention_output_gpu(d_scores, d_V, d_out,
                         num_heads, num_kv_heads, head_dim,
                         max_seq, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto out = from_half(download(d_out, num_heads * head_dim));
    // out = scores[0]*V[0] + scores[1]*V[1] + scores[2]*V[2]
    float exp0 = scores[0] * 10.0f;
    float exp1 = scores[1] * 20.0f;
    float exp2 = scores[2] * 30.0f;
    EXPECT_NEAR(out[0], exp0, 0.1f);
    EXPECT_NEAR(out[1], exp1, 0.1f);
    EXPECT_NEAR(out[2], exp2, 0.1f);
    EXPECT_NEAR(out[3], 0.0f, 0.1f);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_scores); cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// Scatter KV
// ---------------------------------------------------------------------------

TEST(InferenceKernels, ScatterKV) {
    const int num_kv_heads = 2;
    const int head_dim = 4;
    const int max_seq = 8;
    const int pos = 3;

    // src: (2, 4) = [1,2,3,4, 5,6,7,8]
    std::vector<float> h_src = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> h_cache(num_kv_heads * max_seq * head_dim, 0.0f);

    auto d_src = upload(to_half(h_src).data(), num_kv_heads * head_dim);
    auto d_cache = upload(to_half(h_cache).data(),
                          num_kv_heads * max_seq * head_dim);

    scatter_kv_gpu(d_src, d_cache, num_kv_heads, head_dim, max_seq, pos);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = from_half(download(d_cache,
                                     num_kv_heads * max_seq * head_dim));

    // Head 0, pos 3: [1, 2, 3, 4]
    for (int d = 0; d < head_dim; ++d)
        EXPECT_NEAR(result[0 * max_seq * head_dim + pos * head_dim + d],
                    h_src[d], 1e-3f);

    // Head 1, pos 3: [5, 6, 7, 8]
    for (int d = 0; d < head_dim; ++d)
        EXPECT_NEAR(result[1 * max_seq * head_dim + pos * head_dim + d],
                    h_src[head_dim + d], 1e-3f);

    // Other positions should be zero
    EXPECT_NEAR(result[0], 0.0f, 1e-6f);

    cudaFree(d_src); cudaFree(d_cache);
}
