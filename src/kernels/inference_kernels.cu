#include "spbitnet/cuda_utils.h"
#include "spbitnet/inference_kernels.h"

namespace spbitnet {

// ---------------------------------------------------------------------------
// RMSNorm: y[i] = x[i] * w[i] * rsqrt(mean(x^2) + eps)
//
// Single-block kernel (one normalization at a time for autoregressive).
// Block of 256 threads handles up to ~8K elements.
// Computation in float32 for accuracy; input/output float16.
// ---------------------------------------------------------------------------
__global__ void rms_norm_kernel(const half* __restrict__ input,
                                const float* __restrict__ weight,
                                half* __restrict__ output,
                                int size, float eps) {
    __shared__ float shared_sum[8];  // max 8 warps in 256 threads

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    // Local sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = __half2float(input[i]);
        local_sum += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    if (lane == 0) shared_sum[warp] = local_sum;
    __syncthreads();

    // Cross-warp reduction (first warp only)
    if (warp == 0) {
        float val = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) shared_sum[0] = val;
    }
    __syncthreads();

    float scale = rsqrtf(shared_sum[0] / static_cast<float>(size) + eps);

    // Apply normalization
    for (int i = tid; i < size; i += blockDim.x) {
        float val = __half2float(input[i]);
        output[i] = __float2half(val * scale * weight[i]);
    }
}

// ---------------------------------------------------------------------------
// Absmax INT8 quantization
//
// Single-block kernel.  Computes max(|x|), then quantizes:
//   q[i] = round(x[i] * 127 / absmax), clamped to [-128, 127].
// Writes absmax to *d_absmax for the dequant kernel.
// ---------------------------------------------------------------------------
__global__ void absmax_quantize_kernel(const half* __restrict__ input,
                                       int8_t* __restrict__ output,
                                       float* __restrict__ d_absmax,
                                       int size) {
    __shared__ float shared_max[8];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    // Local absmax
    float local_max = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = fabsf(__half2float(input[i]));
        local_max = fmaxf(local_max, val);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));

    if (lane == 0) shared_max[warp] = local_max;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < num_warps) ? shared_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        if (lane == 0) {
            shared_max[0] = val;
            *d_absmax = val;
        }
    }
    __syncthreads();

    float absmax = shared_max[0];
    float scale  = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;

    for (int i = tid; i < size; i += blockDim.x) {
        float val = __half2float(input[i]) * scale;
        int q = __float2int_rn(val);
        q = max(-128, min(127, q));
        output[i] = static_cast<int8_t>(q);
    }
}

// ---------------------------------------------------------------------------
// Absmax reduction only (no quantization)
// Writes max(|input|) to d_absmax. Used by fused BitLinear path.
// ---------------------------------------------------------------------------
__global__ void absmax_reduce_kernel(const half* __restrict__ input,
                                      float* __restrict__ d_absmax,
                                      int size) {
    __shared__ float shared_max[8];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float local_max = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = fabsf(__half2float(input[i]));
        local_max = fmaxf(local_max, val);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));

    if (lane == 0) shared_max[warp] = local_max;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < num_warps) ? shared_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        if (lane == 0) *d_absmax = val;
    }
}

// ---------------------------------------------------------------------------
// Dequantize INT32 → float16
//
// out[i] = (float)int_in[i] * gamma * (*d_absmax) / 127.0
// d_absmax is written by absmax_quantize on the same stream (ordering OK).
// ---------------------------------------------------------------------------
__global__ void dequantize_kernel(const int32_t* __restrict__ input,
                                  half* __restrict__ output,
                                  const float* __restrict__ d_absmax,
                                  float gamma, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float scale = gamma * (*d_absmax) / 127.0f;
    output[i] = __float2half(static_cast<float>(input[i]) * scale);
}

// ---------------------------------------------------------------------------
// Rotary Positional Embeddings (RoPE)
//
// Applied in-place to a (num_heads, head_dim) vector.
// For each dimension pair (2i, 2i+1):
//   freq_i = 1 / (theta ^ (2i / head_dim))
//   angle  = pos * freq_i
//   (x0', x1') = (x0*cos - x1*sin, x0*sin + x1*cos)
// ---------------------------------------------------------------------------
__global__ void rope_kernel(half* __restrict__ vec,
                            int num_heads, int head_dim,
                            int pos, float theta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_heads * (head_dim / 2);
    if (tid >= total_pairs) return;

    int head = tid / (head_dim / 2);
    int pair = tid % (head_dim / 2);

    float freq  = 1.0f / powf(theta, static_cast<float>(2 * pair) / static_cast<float>(head_dim));
    float angle = static_cast<float>(pos) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int idx = head * head_dim + 2 * pair;
    float x0 = __half2float(vec[idx]);
    float x1 = __half2float(vec[idx + 1]);

    vec[idx]     = __float2half(x0 * cos_a - x1 * sin_a);
    vec[idx + 1] = __float2half(x0 * sin_a + x1 * cos_a);
}

// ---------------------------------------------------------------------------
// Scatter KV to cache
//
// Src: (num_kv_heads, head_dim) contiguous.
// Cache: (num_kv_heads, max_seq_len, head_dim).
// Copies each head's vector to the correct position in the cache.
// ---------------------------------------------------------------------------
__global__ void scatter_kv_kernel(const half* __restrict__ src,
                                  half* __restrict__ cache,
                                  int num_kv_heads, int head_dim,
                                  int max_seq_len, int pos) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    if (tid >= total) return;

    int h = tid / head_dim;
    int d = tid % head_dim;
    cache[static_cast<size_t>(h) * max_seq_len * head_dim
        + static_cast<size_t>(pos) * head_dim + d] = src[tid];
}

// ---------------------------------------------------------------------------
// Attention scores
//
// score[h][j] = dot(Q[h], K_cache[kv_h, j]) * scale
// Each warp computes one (head, position) dot product.
// Grid: (ceil(seq_len / warps_per_block), num_heads).
// ---------------------------------------------------------------------------
__global__ void attention_scores_kernel(const half* __restrict__ Q,
                                        const half* __restrict__ K_cache,
                                        float* __restrict__ scores,
                                        int num_heads, int num_kv_heads,
                                        int head_dim, int max_seq_len,
                                        int seq_len, float scale) {
    const int lane          = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_blk = blockDim.x >> 5;

    const int head = blockIdx.y;
    const int j    = blockIdx.x * warps_per_blk + warp_in_block;
    if (j >= seq_len) return;

    // GQA mapping: Q head → KV head
    const int kv_head = head * num_kv_heads / num_heads;

    const half* q_head = Q + head * head_dim;
    const half* k_pos  = K_cache
                       + static_cast<size_t>(kv_head) * max_seq_len * head_dim
                       + static_cast<size_t>(j) * head_dim;

    float acc = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
        acc += __half2float(q_head[d]) * __half2float(k_pos[d]);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0) {
        scores[static_cast<size_t>(head) * max_seq_len + j] = acc * scale;
    }
}

// ---------------------------------------------------------------------------
// Softmax (in-place, stable)
//
// One block per head, 256 threads.
// Three-pass: max → exp → normalize.
// ---------------------------------------------------------------------------
__global__ void softmax_kernel(float* __restrict__ scores,
                               int max_seq_len, int seq_len) {
    __shared__ float shared_val[8];

    const int head = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float* hs = scores + static_cast<size_t>(head) * max_seq_len;

    // --- Pass 1: find max ---
    float local_max = -1e30f;
    for (int j = tid; j < seq_len; j += blockDim.x)
        local_max = fmaxf(local_max, hs[j]);

    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    if (lane == 0) shared_val[warp] = local_max;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < num_warps) ? shared_val[lane] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        if (lane == 0) shared_val[0] = val;
    }
    __syncthreads();
    float max_val = shared_val[0];

    // --- Pass 2: exp and sum ---
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float val = expf(hs[j] - max_val);
        hs[j] = val;
        local_sum += val;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    if (lane == 0) shared_val[warp] = local_sum;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < num_warps) ? shared_val[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) shared_val[0] = val;
    }
    __syncthreads();

    // --- Pass 3: normalize ---
    float inv_sum = 1.0f / shared_val[0];
    for (int j = tid; j < seq_len; j += blockDim.x)
        hs[j] *= inv_sum;
}

// ---------------------------------------------------------------------------
// Attention output
//
// out[h][d] = sum_j score[h][j] * V_cache[kv_h, j, d]
// One block per head, head_dim threads (one per output dim).
// TODO Phase 6: increase parallelism for long sequences.
// ---------------------------------------------------------------------------
__global__ void attention_output_kernel(const float* __restrict__ scores,
                                        const half* __restrict__ V_cache,
                                        half* __restrict__ output,
                                        int num_heads, int num_kv_heads,
                                        int head_dim, int max_seq_len,
                                        int seq_len) {
    const int head = blockIdx.x;
    const int d    = threadIdx.x;
    if (d >= head_dim) return;

    const int kv_head = head * num_kv_heads / num_heads;
    const float* hs   = scores + static_cast<size_t>(head) * max_seq_len;
    const half*  v_kv  = V_cache
                       + static_cast<size_t>(kv_head) * max_seq_len * head_dim;

    float acc = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        acc += hs[j] * __half2float(v_kv[static_cast<size_t>(j) * head_dim + d]);
    }

    output[head * head_dim + d] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// Fused ReLU² + element-wise multiply
// out[i] = max(0, gate[i])^2 * up[i]
// ---------------------------------------------------------------------------
__global__ void relu2_mul_kernel(const half* __restrict__ gate,
                                 const half* __restrict__ up,
                                 half* __restrict__ output,
                                 int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float g    = __half2float(gate[i]);
    float u    = __half2float(up[i]);
    float relu = fmaxf(0.0f, g);
    output[i]  = __float2half(relu * relu * u);
}

// ---------------------------------------------------------------------------
// Fused ReLU² + multiply with float32 output (avoids float16 overflow)
//
// relu²(gate) * up can exceed float16 max (65504) before sub-normalization
// brings values back into range.  This variant keeps the output in float32
// so that the subsequent RMSNorm (or float-to-half conversion) can handle it.
// ---------------------------------------------------------------------------
__global__ void relu2_mul_f32_kernel(const half* __restrict__ gate,
                                     const half* __restrict__ up,
                                     float* __restrict__ output,
                                     int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float g    = __half2float(gate[i]);
    float u    = __half2float(up[i]);
    float relu = fmaxf(0.0f, g);
    output[i]  = relu * relu * u;
}

// ---------------------------------------------------------------------------
// RMSNorm with float32 input: y[i] = x[i] * w[i] * rsqrt(mean(x^2) + eps)
//
// Used after relu2_mul_f32 to normalize large intermediate MLP activations
// back into float16 range.  Input is float32, output is float16.
// ---------------------------------------------------------------------------
__global__ void rms_norm_f32in_kernel(const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      half* __restrict__ output,
                                      int size, float eps) {
    __shared__ float shared_sum[8];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = input[i];
        local_sum += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    if (lane == 0) shared_sum[warp] = local_sum;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) shared_sum[0] = val;
    }
    __syncthreads();

    float scale = rsqrtf(shared_sum[0] / static_cast<float>(size) + eps);

    for (int i = tid; i < size; i += blockDim.x) {
        float val = input[i];
        output[i] = __float2half(val * scale * weight[i]);
    }
}

// ---------------------------------------------------------------------------
// Float32 to float16 conversion
// ---------------------------------------------------------------------------
__global__ void float_to_half_kernel(const float* __restrict__ input,
                                     half* __restrict__ output,
                                     int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    output[i] = __float2half(input[i]);
}

// ---------------------------------------------------------------------------
// Residual add: output[i] = a[i] + b[i]
// ---------------------------------------------------------------------------
__global__ void residual_add_kernel(const half* __restrict__ a,
                                    const half* __restrict__ b,
                                    half* __restrict__ output,
                                    int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    output[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// ---------------------------------------------------------------------------
// Float16 warp-per-row GEMV
//
// y[row] = dot(W[row], x)
// Each warp handles one row.  Accumulates in float32.
// Used for lm_head logits (W is float16, output float32).
// ---------------------------------------------------------------------------
__global__ void half_gemv_kernel(const half* __restrict__ W,
                                 const half* __restrict__ x,
                                 float* __restrict__ y,
                                 int rows, int cols) {
    const int lane    = threadIdx.x & 31;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= rows) return;

    const half* row = W + static_cast<size_t>(warp_id) * cols;
    float acc = 0.0f;

    // Vectorized float4 loads: 16 bytes = 8 halfs per load instruction.
    // Reduces load instruction count 8x vs scalar half loads.
    const int cols_v = cols >> 3;  // cols / 8
    const float4* row_v = reinterpret_cast<const float4*>(row);
    const float4* x_v   = reinterpret_cast<const float4*>(x);

    for (int j = lane; j < cols_v; j += 32) {
        float4 w4 = row_v[j];
        float4 x4 = x_v[j];
        const half* wp = reinterpret_cast<const half*>(&w4);
        const half* xp = reinterpret_cast<const half*>(&x4);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc += __half2float(wp[k]) * __half2float(xp[k]);
        }
    }

    // Handle remainder elements (cols not divisible by 8)
    const int tail_start = cols_v << 3;
    for (int j = tail_start + lane; j < cols; j += 32) {
        acc += __half2float(row[j]) * __half2float(x[j]);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0) {
        y[warp_id] = acc;
    }
}

// ===========================================================================
// Host launch wrappers
// ===========================================================================

void rms_norm_gpu(const half* input, const float* weight, half* output,
                  int size, float eps, cudaStream_t stream) {
    rms_norm_kernel<<<1, 256, 0, stream>>>(input, weight, output, size, eps);
    CUDA_CHECK(cudaGetLastError());
}

void absmax_quantize_gpu(const half* input, int8_t* output, float* d_absmax,
                         int size, cudaStream_t stream) {
    absmax_quantize_kernel<<<1, 256, 0, stream>>>(input, output, d_absmax, size);
    CUDA_CHECK(cudaGetLastError());
}

void absmax_reduce_gpu(const half* input, float* d_absmax, int size,
                       cudaStream_t stream) {
    absmax_reduce_kernel<<<1, 256, 0, stream>>>(input, d_absmax, size);
    CUDA_CHECK(cudaGetLastError());
}

void dequantize_gpu(const int32_t* input, half* output, const float* d_absmax,
                    float gamma, int size, cudaStream_t stream) {
    constexpr int block = 256;
    int grid = (size + block - 1) / block;
    dequantize_kernel<<<grid, block, 0, stream>>>(input, output, d_absmax, gamma, size);
    CUDA_CHECK(cudaGetLastError());
}

void rope_gpu(half* vec, int num_heads, int head_dim, int pos, float theta,
              cudaStream_t stream) {
    int total_pairs = num_heads * (head_dim / 2);
    constexpr int block = 256;
    int grid = (total_pairs + block - 1) / block;
    rope_kernel<<<grid, block, 0, stream>>>(vec, num_heads, head_dim, pos, theta);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_kv_gpu(const half* src, half* cache, int num_kv_heads, int head_dim,
                    int max_seq_len, int pos, cudaStream_t stream) {
    int total = num_kv_heads * head_dim;
    constexpr int block = 256;
    int grid = (total + block - 1) / block;
    scatter_kv_kernel<<<grid, block, 0, stream>>>(
        src, cache, num_kv_heads, head_dim, max_seq_len, pos);
    CUDA_CHECK(cudaGetLastError());
}

void attention_scores_gpu(const half* Q, const half* K_cache, float* scores,
                          int num_heads, int num_kv_heads, int head_dim,
                          int max_seq_len, int seq_len, float scale,
                          cudaStream_t stream) {
    constexpr int block = 256;  // 8 warps per block
    int warps_per_block = block / 32;
    dim3 grid((seq_len + warps_per_block - 1) / warps_per_block, num_heads);
    attention_scores_kernel<<<grid, block, 0, stream>>>(
        Q, K_cache, scores, num_heads, num_kv_heads, head_dim,
        max_seq_len, seq_len, scale);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_gpu(float* scores, int num_heads, int max_seq_len, int seq_len,
                 cudaStream_t stream) {
    softmax_kernel<<<num_heads, 256, 0, stream>>>(scores, max_seq_len, seq_len);
    CUDA_CHECK(cudaGetLastError());
}

void attention_output_gpu(const float* scores, const half* V_cache, half* output,
                          int num_heads, int num_kv_heads, int head_dim,
                          int max_seq_len, int seq_len,
                          cudaStream_t stream) {
    // One block per head, head_dim threads per block
    // Round up to next multiple of 32 for warp alignment
    int threads = ((head_dim + 31) / 32) * 32;
    attention_output_kernel<<<num_heads, threads, 0, stream>>>(
        scores, V_cache, output, num_heads, num_kv_heads, head_dim,
        max_seq_len, seq_len);
    CUDA_CHECK(cudaGetLastError());
}

void relu2_mul_gpu(const half* gate, const half* up, half* output, int size,
                   cudaStream_t stream) {
    constexpr int block = 256;
    int grid = (size + block - 1) / block;
    relu2_mul_kernel<<<grid, block, 0, stream>>>(gate, up, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void relu2_mul_f32_gpu(const half* gate, const half* up, float* output, int size,
                       cudaStream_t stream) {
    constexpr int block = 256;
    int grid = (size + block - 1) / block;
    relu2_mul_f32_kernel<<<grid, block, 0, stream>>>(gate, up, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void rms_norm_f32in_gpu(const float* input, const float* weight, half* output,
                        int size, float eps, cudaStream_t stream) {
    rms_norm_f32in_kernel<<<1, 256, 0, stream>>>(input, weight, output, size, eps);
    CUDA_CHECK(cudaGetLastError());
}

void float_to_half_gpu(const float* input, half* output, int size,
                       cudaStream_t stream) {
    constexpr int block = 256;
    int grid = (size + block - 1) / block;
    float_to_half_kernel<<<grid, block, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void residual_add_gpu(const half* a, const half* b, half* output, int size,
                      cudaStream_t stream) {
    constexpr int block = 256;
    int grid = (size + block - 1) / block;
    residual_add_kernel<<<grid, block, 0, stream>>>(a, b, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void half_gemv_gpu(const half* W, const half* x, float* y, int rows, int cols,
                   cudaStream_t stream) {
    constexpr int block = 256;  // 8 warps per block
    int warps_needed  = rows;
    int threads_total = warps_needed * 32;
    int grid = (threads_total + block - 1) / block;

    half_gemv_kernel<<<grid, block, 0, stream>>>(W, x, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace spbitnet
