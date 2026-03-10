#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

namespace spbitnet {

// Unpack packed ternary uint32 array to int8 array on GPU.
// packed: device pointer to packed data (num_packed uint32_t elements)
// out: device pointer to output int8 array (num_weights elements)
// num_weights: total number of weights
void ternary_unpack_gpu(const uint32_t* packed, int8_t* out, size_t num_weights,
                        cudaStream_t stream = nullptr);

// Dense ternary GEMV: y = W_ternary * x
// packed_weights: device ptr, packed ternary matrix (rows x cols, row-major, rows padded to 16)
// x: device ptr, input vector (cols elements, int8)
// y: device ptr, output vector (rows elements, int32 accumulators)
// rows, cols: matrix dimensions
void ternary_gemv_gpu(const uint32_t* packed_weights, const int8_t* x, int32_t* y,
                      int rows, int cols, cudaStream_t stream = nullptr);

} // namespace spbitnet
