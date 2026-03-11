#include "spbitnet/cuda_utils.h"
#include "spbitnet/sparse_ternary_kernels.h"

namespace spbitnet {

// ---------------------------------------------------------------------------
// Compressed Sparse-Ternary format (2:4 structured sparsity)
//
// Ref: Zhang et al., "Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly
//      to Semi-Structured Sparsity" (arXiv:2603.05168, 2026), Section 3.2.
// Ref: Mishra et al., "Accelerating Sparse Deep Neural Networks"
//      (arXiv:2104.08378, 2021) — NVIDIA 2:4 structured sparsity.
//
// Meta array: 4-bit position bitmap per group of 4, 8 groups per uint32.
//   Group g's bitmap: (meta_row[g/8] >> ((g%8)*4)) & 0xF
//   Exactly 2 bits set in each bitmap.
//
// Values array: 2-bit sign pair per group, 16 groups per uint32.
//   Group g's signs: (vals_row[g/16] >> ((g%16)*2)) & 0x3
//   bit 0 (sign_lo) = sign of lower-positioned non-zero  (0=+1, 1=-1)
//   bit 1 (sign_hi) = sign of higher-positioned non-zero (0=+1, 1=-1)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Bitmap → (pos0, pos1) lookup table
// Only 6 valid 2:4 bitmaps exist. Invalid entries map to (0,0).
// Indexed by 4-bit bitmap value (0..15).
// ---------------------------------------------------------------------------
struct Pos2 {
    int8_t p0;  // lower-positioned non-zero (first set bit)
    int8_t p1;  // higher-positioned non-zero (second set bit)
};

__device__ __constant__ Pos2 kBitmapToPos[16] = {
    {0, 0},  //  0 = 0b0000 — invalid
    {0, 0},  //  1 = 0b0001 — invalid (only 1 bit set)
    {0, 0},  //  2 = 0b0010 — invalid
    {0, 1},  //  3 = 0b0011
    {0, 0},  //  4 = 0b0100 — invalid
    {0, 2},  //  5 = 0b0101
    {1, 2},  //  6 = 0b0110
    {0, 0},  //  7 = 0b0111 — invalid (3 bits set)
    {0, 0},  //  8 = 0b1000 — invalid
    {0, 3},  //  9 = 0b1001
    {1, 3},  // 10 = 0b1010
    {0, 0},  // 11 = 0b1011 — invalid
    {2, 3},  // 12 = 0b1100
    {0, 0},  // 13 = 0b1101 — invalid
    {0, 0},  // 14 = 0b1110 — invalid
    {0, 0},  // 15 = 0b1111 — invalid
};

// ---------------------------------------------------------------------------
// Kernel: sparse_ternary_unpack_kernel
// Each thread unpacks one group of 4 elements from sparse format to dense.
// Decodes the 4-bit bitmap to find non-zero positions, then applies signs.
// ---------------------------------------------------------------------------
__global__ void sparse_ternary_unpack_kernel(const uint32_t* __restrict__ meta,
                                              const uint32_t* __restrict__ values,
                                              int8_t* __restrict__ out,
                                              int rows, int cols,
                                              int meta_row_stride,
                                              int values_row_stride) {
    const int groups_per_row = cols / 4;
    const int total_groups = rows * groups_per_row;
    const int gid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (gid >= total_groups) return;

    const int row = gid / groups_per_row;
    const int g   = gid % groups_per_row;

    // Pointer to this row's meta and values data
    const uint32_t* meta_row = meta + static_cast<size_t>(row) * meta_row_stride;
    const uint32_t* vals_row = values + static_cast<size_t>(row) * values_row_stride;

    // Extract 4-bit bitmap for this group
    const uint32_t bitmap = (meta_row[g / 8] >> ((g % 8) * 4)) & 0xF;

    // Extract 2-bit signs for this group
    const uint32_t signs = (vals_row[g / 16] >> ((g % 16) * 2)) & 0x3;

    // Look up the two non-zero positions
    const Pos2 pos = kBitmapToPos[bitmap];

    // Output base index for this group
    const size_t base = static_cast<size_t>(row) * cols + g * 4;

    // Write all 4 positions: zero first, then fill non-zeros
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[base + i] = 0;
    }

    // Lower-positioned non-zero: sign_lo = bit 0
    const int8_t val0 = (signs & 1u) ? static_cast<int8_t>(-1) : static_cast<int8_t>(+1);
    out[base + pos.p0] = val0;

    // Higher-positioned non-zero: sign_hi = bit 1
    const int8_t val1 = (signs & 2u) ? static_cast<int8_t>(-1) : static_cast<int8_t>(+1);
    out[base + pos.p1] = val1;
}

// ---------------------------------------------------------------------------
// Kernel: sparse_ternary_gemv_kernel
// Sparse ternary GEMV: y = W_sparse * x
//
// Warp-per-row parallelism: each warp (32 threads) cooperatively processes
// one output row. Each lane handles a strided subset of groups, accumulates
// a partial sum, then the warp reduces via __shfl_down_sync.
//
// This avoids multiplications entirely — ternary weights mean we only
// add or subtract input activations. Combined with 2:4 sparsity, each
// group touches only 2 of 4 input elements.
// ---------------------------------------------------------------------------
__global__ void sparse_ternary_gemv_kernel(const uint32_t* __restrict__ meta,
                                            const uint32_t* __restrict__ values,
                                            const int8_t* __restrict__ x,
                                            int32_t* __restrict__ y,
                                            int rows, int cols,
                                            int meta_row_stride,
                                            int values_row_stride) {
    const int lane = threadIdx.x % 32;
    const int warp_id = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int row = warp_id;  // 1 warp per row

    if (row >= rows) return;

    const int groups_per_row = cols / 4;

    // Pointer to this row's meta and values data
    const uint32_t* meta_row = meta + static_cast<size_t>(row) * meta_row_stride;
    const uint32_t* vals_row = values + static_cast<size_t>(row) * values_row_stride;

    int32_t acc = 0;

    for (int g = lane; g < groups_per_row; g += 32) {
        // Extract 4-bit bitmap for this group
        const uint32_t bitmap = (meta_row[g / 8] >> ((g % 8) * 4)) & 0xF;

        // Extract 2-bit signs for this group
        const uint32_t signs = (vals_row[g / 16] >> ((g % 16) * 2)) & 0x3;

        // Look up the two non-zero positions from constant LUT
        const Pos2 pos = kBitmapToPos[bitmap];

        const int base_col = g * 4;

        // Load the two input activations at non-zero positions
        const int32_t x0 = static_cast<int32_t>(x[base_col + pos.p0]);
        const int32_t x1 = static_cast<int32_t>(x[base_col + pos.p1]);

        // Accumulate: sign_lo (bit 0) for lower-positioned, sign_hi (bit 1) for higher
        acc += (signs & 1u) ? -x0 : x0;
        acc += (signs & 2u) ? -x1 : x1;
    }

    // Warp-level reduction using shuffle intrinsics
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Lane 0 writes the final result
    if (lane == 0) {
        y[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_sparse_bitlinear_kernel
// Fused implementation of the BitLinear layer from Ma et al. (2024):
//   y = dequant(W_sparse * quant(x))
// Combines absmax quantization + sparse ternary GEMV + dequantization
// into a single kernel. Requires d_absmax pre-computed by absmax_reduce_gpu.
//
// Each warp quantizes x on-the-fly, accumulates in int32, and dequantizes
// the result. Eliminates the intermediate d_quant_ and d_int_out_ buffers
// and saves 1 kernel launch per BitLinear layer.
// ---------------------------------------------------------------------------
__global__ void fused_sparse_bitlinear_kernel(
        const uint32_t* __restrict__ meta,
        const uint32_t* __restrict__ values,
        const half* __restrict__ x,
        half* __restrict__ output,
        const float* __restrict__ d_absmax,
        float gamma,
        int rows, int cols,
        int meta_row_stride,
        int values_row_stride) {
    const int lane = threadIdx.x % 32;
    const int warp_id = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int row = warp_id;

    if (row >= rows) return;

    // Read absmax (written by absmax_reduce_gpu on same stream)
    const float absmax = *d_absmax;
    const float quant_scale = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;

    const int groups_per_row = cols / 4;
    const uint32_t* meta_row = meta + static_cast<size_t>(row) * meta_row_stride;
    const uint32_t* vals_row = values + static_cast<size_t>(row) * values_row_stride;

    int32_t acc = 0;

    for (int g = lane; g < groups_per_row; g += 32) {
        const uint32_t bitmap = (meta_row[g / 8] >> ((g % 8) * 4)) & 0xF;
        const uint32_t signs = (vals_row[g / 16] >> ((g % 16) * 2)) & 0x3;
        const Pos2 pos = kBitmapToPos[bitmap];
        const int base_col = g * 4;

        // Quantize x on-the-fly: half → int8 (same math as absmax_quantize)
        float f0 = __half2float(x[base_col + pos.p0]) * quant_scale;
        float f1 = __half2float(x[base_col + pos.p1]) * quant_scale;
        int32_t x0 = max(-128, min(127, __float2int_rn(f0)));
        int32_t x1 = max(-128, min(127, __float2int_rn(f1)));

        acc += (signs & 1u) ? -x0 : x0;
        acc += (signs & 2u) ? -x1 : x1;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Dequantize and write: out = acc * gamma * absmax / 127
    if (lane == 0) {
        float dequant_scale = gamma * absmax / 127.0f;
        output[row] = __float2half(static_cast<float>(acc) * dequant_scale);
    }
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------

void sparse_ternary_unpack_gpu(const uint32_t* meta, const uint32_t* values,
                                int8_t* out, int rows, int cols,
                                int meta_row_stride, int values_row_stride,
                                cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    const int groups_per_row = cols / 4;
    const int total_groups = rows * groups_per_row;
    constexpr int block_size = 256;
    const int grid_size = (total_groups + block_size - 1) / block_size;

    sparse_ternary_unpack_kernel<<<grid_size, block_size, 0, stream>>>(
        meta, values, out, rows, cols, meta_row_stride, values_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

void sparse_ternary_gemv_gpu(const uint32_t* meta, const uint32_t* values,
                              const int8_t* x, int32_t* y,
                              int rows, int cols,
                              int meta_row_stride, int values_row_stride,
                              cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    // 256 threads per block = 8 warps per block = 8 rows per block
    constexpr int block_size = 256;
    const int warps_needed = rows;  // 1 warp per row
    const int threads_needed = warps_needed * 32;
    const int grid_size = (threads_needed + block_size - 1) / block_size;

    sparse_ternary_gemv_kernel<<<grid_size, block_size, 0, stream>>>(
        meta, values, x, y, rows, cols, meta_row_stride, values_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

void fused_sparse_bitlinear_gpu(const uint32_t* meta, const uint32_t* values,
                                 const half* x, half* output,
                                 const float* d_absmax, float gamma,
                                 int rows, int cols,
                                 int meta_row_stride, int values_row_stride,
                                 cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    constexpr int block_size = 256;
    const int warps_needed = rows;
    const int threads_needed = warps_needed * 32;
    const int grid_size = (threads_needed + block_size - 1) / block_size;

    fused_sparse_bitlinear_kernel<<<grid_size, block_size, 0, stream>>>(
        meta, values, x, output, d_absmax, gamma,
        rows, cols, meta_row_stride, values_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace spbitnet
