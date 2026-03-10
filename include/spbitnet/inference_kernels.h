#pragma once

/// \file inference_kernels.h
/// \brief CUDA kernel host wrappers for transformer inference ops.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace spbitnet {

/// RMSNorm: output[i] = input[i] * weight[i] * rsqrt(mean(input^2) + eps).
/// Single-block kernel; input/output are float16, weight is float32.
void rms_norm_gpu(const half* input, const float* weight, half* output,
                  int size, float eps, cudaStream_t stream = nullptr);

/// Absmax INT8 quantization: output[i] = round(input[i] * 127 / max(|input|)).
/// Writes absmax value to *d_absmax for later dequantization.
void absmax_quantize_gpu(const half* input, int8_t* output, float* d_absmax,
                         int size, cudaStream_t stream = nullptr);

/// Dequantize INT32 accumulator to float16: out[i] = int_in[i] * gamma * (*d_absmax) / 127.
/// d_absmax is a device pointer written by absmax_quantize_gpu on the same stream.
void dequantize_gpu(const int32_t* input, half* output, const float* d_absmax,
                    float gamma, int size, cudaStream_t stream = nullptr);

/// Apply RoPE in-place to a (num_heads, head_dim) half vector.
void rope_gpu(half* vec, int num_heads, int head_dim, int pos, float theta,
              cudaStream_t stream = nullptr);

/// Scatter a contiguous (num_kv_heads, head_dim) vector into KV cache
/// at the given position.  Cache layout: (num_kv_heads, max_seq_len, head_dim).
void scatter_kv_gpu(const half* src, half* cache, int num_kv_heads, int head_dim,
                    int max_seq_len, int pos, cudaStream_t stream = nullptr);

/// Attention scores: score[h][j] = dot(Q[h], K_cache[kv_h, j]) * scale.
/// Q: (num_heads, head_dim).  K_cache: (num_kv_heads, max_seq_len, head_dim).
/// scores: (num_heads, max_seq_len) — only first seq_len entries filled.
void attention_scores_gpu(const half* Q, const half* K_cache, float* scores,
                          int num_heads, int num_kv_heads, int head_dim,
                          int max_seq_len, int seq_len, float scale,
                          cudaStream_t stream = nullptr);

/// In-place softmax over the first seq_len entries per head.
/// scores: (num_heads, max_seq_len).
void softmax_gpu(float* scores, int num_heads, int max_seq_len, int seq_len,
                 cudaStream_t stream = nullptr);

/// Attention output: out[h][d] = sum_j score[h][j] * V_cache[kv_h, j, d].
/// output: (num_heads, head_dim).  V_cache: (num_kv_heads, max_seq_len, head_dim).
void attention_output_gpu(const float* scores, const half* V_cache, half* output,
                          int num_heads, int num_kv_heads, int head_dim,
                          int max_seq_len, int seq_len,
                          cudaStream_t stream = nullptr);

/// Fused ReLU² + element-wise multiply: out[i] = max(0, gate[i])^2 * up[i].
void relu2_mul_gpu(const half* gate, const half* up, half* output, int size,
                   cudaStream_t stream = nullptr);

/// Element-wise residual add: output[i] = a[i] + b[i].
void residual_add_gpu(const half* a, const half* b, half* output, int size,
                      cudaStream_t stream = nullptr);

/// Float16 GEMV: y[i] = dot(W[i], x).
/// W: (rows, cols) half, x: (cols) half, y: (rows) float.
/// Warp-per-row parallelism.
void half_gemv_gpu(const half* W, const half* x, float* y, int rows, int cols,
                   cudaStream_t stream = nullptr);

}  // namespace spbitnet
