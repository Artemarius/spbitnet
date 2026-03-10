#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

namespace spbitnet {

// Unpack sparse ternary format to dense int8 array on GPU.
// meta: device pointer to position bitmaps (4-bit per group, 8 groups per uint32)
// values: device pointer to sign bits (2-bit per group, 16 groups per uint32)
// out: device pointer to output int8 array (rows * cols elements, row-major)
// rows, cols: matrix dimensions (cols must be multiple of 4)
// meta_row_stride: uint32 words per row in meta array
// values_row_stride: uint32 words per row in values array
void sparse_ternary_unpack_gpu(const uint32_t* meta, const uint32_t* values,
                                int8_t* out, int rows, int cols,
                                int meta_row_stride, int values_row_stride,
                                cudaStream_t stream = nullptr);

// Sparse ternary GEMV: y = W_sparse * x (warp-per-row parallelism)
// meta: device pointer to position bitmaps
// values: device pointer to sign bits
// x: device pointer to input vector (cols elements, int8)
// y: device pointer to output vector (rows elements, int32 accumulators)
// rows, cols: matrix dimensions (cols must be multiple of 4)
// meta_row_stride: uint32 words per row in meta array
// values_row_stride: uint32 words per row in values array
void sparse_ternary_gemv_gpu(const uint32_t* meta, const uint32_t* values,
                              const int8_t* x, int32_t* y,
                              int rows, int cols,
                              int meta_row_stride, int values_row_stride,
                              cudaStream_t stream = nullptr);

} // namespace spbitnet
