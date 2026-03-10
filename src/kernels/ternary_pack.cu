#include "spbitnet/cuda_utils.h"
#include "spbitnet/ternary_kernels.h"

namespace spbitnet {

// ---------------------------------------------------------------------------
// Ternary encoding: 2 bits per weight, LSB-first packing
//   00 = 0, 01 = +1, 10 = -1, 11 = reserved (treated as 0)
// 16 weights per uint32_t: weight 0 in bits [1:0], weight 15 in bits [31:30]
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Kernel: ternary_unpack_kernel
// Each thread unpacks one uint32_t (= 16 weights) into 16 int8 values.
// Boundary check: the last uint32 may encode padding beyond num_weights.
// ---------------------------------------------------------------------------
__global__ void ternary_unpack_kernel(const uint32_t* __restrict__ packed,
                                      int8_t* __restrict__ out,
                                      size_t num_weights) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t base = tid * 16;

    // Early exit if this thread's entire group is beyond num_weights
    if (base >= num_weights) return;

    const uint32_t word = packed[tid];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const size_t idx = base + i;
        if (idx >= num_weights) return;

        const uint32_t code = (word >> (i * 2)) & 0x3;

        // Decode: 01 -> +1, 10 -> -1, else -> 0
        int8_t val = 0;
        if (code == 0x1) {
            val = 1;
        } else if (code == 0x2) {
            val = -1;
        }
        out[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel: ternary_gemv_kernel
// Each thread computes one output element y[row].
// For that row, iterate over packed uint32 words, extract 2-bit codes,
// and accumulate: +1 -> +x[col], -1 -> -x[col], 0 -> skip.
// ---------------------------------------------------------------------------
__global__ void ternary_gemv_kernel(const uint32_t* __restrict__ packed_weights,
                                    const int8_t* __restrict__ x,
                                    int32_t* __restrict__ y,
                                    int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const int packed_row_stride = (cols + 15) / 16;
    const uint32_t* row_ptr = packed_weights + static_cast<size_t>(row) * packed_row_stride;

    int32_t acc = 0;
    int col = 0;

    for (int w = 0; w < packed_row_stride; ++w) {
        const uint32_t word = row_ptr[w];

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (col >= cols) break;

            const uint32_t code = (word >> (i * 2)) & 0x3;

            // Decode: 01 -> +1, 10 -> -1, else -> skip
            if (code == 0x1) {
                acc += static_cast<int32_t>(x[col]);
            } else if (code == 0x2) {
                acc -= static_cast<int32_t>(x[col]);
            }
            ++col;
        }
    }

    y[row] = acc;
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------

void ternary_unpack_gpu(const uint32_t* packed, int8_t* out, size_t num_weights,
                        cudaStream_t stream) {
    if (num_weights == 0) return;

    const size_t num_packed = (num_weights + 15) / 16;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((num_packed + block_size - 1) / block_size);

    ternary_unpack_kernel<<<grid_size, block_size, 0, stream>>>(packed, out, num_weights);
    CUDA_CHECK(cudaGetLastError());
}

void ternary_gemv_gpu(const uint32_t* packed_weights, const int8_t* x, int32_t* y,
                      int rows, int cols, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    constexpr int block_size = 256;
    const int grid_size = (rows + block_size - 1) / block_size;

    ternary_gemv_kernel<<<grid_size, block_size, 0, stream>>>(packed_weights, x, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace spbitnet
