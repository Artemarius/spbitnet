#ifdef SPBITNET_HAS_CUSPARSELT

#include "spbitnet/cusparselt_backend.h"
#include "spbitnet/cuda_utils.h"

#include <cstdio>
#include <cstdlib>

namespace spbitnet {

// ---------------------------------------------------------------------------
// cuSPARSELt error checking macro
// ---------------------------------------------------------------------------
#define CUSPARSELT_CHECK(call)                                                  \
    do {                                                                         \
        cusparseStatus_t err = (call);                                          \
        if (err != CUSPARSE_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuSPARSELt error at %s:%d — %s (%d)\n",           \
                    __FILE__, __LINE__, cusparseLtGetErrorString(err),           \
                    static_cast<int>(err));                                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Constructor: set up all cuSPARSELt descriptors and plan.
//
// Computes: D[m,n] = A[m,k] * B[k,n]   (caller's row-major view)
//
// cuSPARSELt requires column-major. Row-major A[m,k] with ld=k has the
// same memory layout as column-major [k,m] with ld=k. We declare the
// structured matrix as [k,m] col-major and use opA=TRANSPOSE to recover
// the effective [m,k] shape. No data transposition needed.
//
// For GEMV (n=1), vectors are layout-agnostic — the caller passes
// contiguous arrays directly.
// ---------------------------------------------------------------------------
CuSparseLtGemm::CuSparseLtGemm(int m, int k, int n)
    : m_(m), k_(k), n_(n) {

    if (n < 16) {
        fprintf(stderr, "CuSparseLtGemm: n=%d but cuSPARSELt INT8 requires n >= 16 "
                "(Tensor Core tile constraint). Use sparse_ternary_gemv_gpu for GEMV.\n", n);
        exit(EXIT_FAILURE);
    }

    CUSPARSELT_CHECK(cusparseLtInit(&handle_));

    // Matrix A: structured sparse.
    // Declared as [k, m] column-major with ld=k.
    // With opA=TRANSPOSE, effective shape is [m, k].
    // This matches row-major [m, k] data with stride k.
    CUSPARSELT_CHECK(cusparseLtStructuredDescriptorInit(
        &handle_, &mat_a_,
        static_cast<int64_t>(k),   // rows (in col-major view)
        static_cast<int64_t>(m),   // cols (in col-major view)
        static_cast<int64_t>(k),   // ld = k
        16,                         // alignment (bytes)
        CUDA_R_8I,
        CUSPARSE_ORDER_COL,
        CUSPARSELT_SPARSITY_50_PERCENT));

    // Matrix B: dense [k, n] column-major, ld=k
    CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(
        &handle_, &mat_b_,
        static_cast<int64_t>(k), static_cast<int64_t>(n),
        static_cast<int64_t>(k),   // ld
        16,
        CUDA_R_8I,
        CUSPARSE_ORDER_COL));

    // Matrix C/D: dense [m, n] column-major, ld=m
    CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(
        &handle_, &mat_c_,
        static_cast<int64_t>(m), static_cast<int64_t>(n),
        static_cast<int64_t>(m),
        16,
        CUDA_R_32I,
        CUSPARSE_ORDER_COL));

    // Matmul: D = alpha * op(A) * op(B) + beta * C
    // opA = TRANSPOSE: A^T is [m,k], B is [k,n], D is [m,n]
    CUSPARSELT_CHECK(cusparseLtMatmulDescriptorInit(
        &handle_, &matmul_,
        CUSPARSE_OPERATION_TRANSPOSE,       // opA
        CUSPARSE_OPERATION_NON_TRANSPOSE,   // opB
        &mat_a_, &mat_b_, &mat_c_, &mat_c_,
        CUSPARSE_COMPUTE_32I));

    // Algorithm selection
    CUSPARSELT_CHECK(cusparseLtMatmulAlgSelectionInit(
        &handle_, &alg_sel_, &matmul_,
        CUSPARSELT_MATMUL_ALG_DEFAULT));

    // Create plan
    CUSPARSELT_CHECK(cusparseLtMatmulPlanInit(
        &handle_, &plan_, &matmul_, &alg_sel_));

    // Query workspace size
    CUSPARSELT_CHECK(cusparseLtMatmulGetWorkspace(
        &handle_, &plan_, &workspace_size_));

    if (workspace_size_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace_, workspace_size_));
    }
}

CuSparseLtGemm::~CuSparseLtGemm() {
    if (d_pruned_)     cudaFree(d_pruned_);
    if (d_compressed_) cudaFree(d_compressed_);
    if (d_workspace_)  cudaFree(d_workspace_);

    cusparseLtMatmulPlanDestroy(&plan_);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel_);
    cusparseLtMatDescriptorDestroy(&mat_c_);
    cusparseLtMatDescriptorDestroy(&mat_b_);
    cusparseLtMatDescriptorDestroy(&mat_a_);
    cusparseLtDestroy(&handle_);
}

// ---------------------------------------------------------------------------
// Prune weights to 2:4 pattern and compress.
// d_weights: device ptr to row-major INT8 [m x k] (= col-major [k x m]).
// ---------------------------------------------------------------------------
void CuSparseLtGemm::prepare(const int8_t* d_weights, cudaStream_t stream) {
    const size_t weight_bytes = static_cast<size_t>(m_) * k_ * sizeof(int8_t);

    if (!d_pruned_) {
        CUDA_CHECK(cudaMalloc(&d_pruned_, weight_bytes));
    }

    // Prune: apply 2:4 sparsity (magnitude-based, strip algorithm)
    CUSPARSELT_CHECK(cusparseLtSpMMAPrune(
        &handle_, &matmul_,
        d_weights, d_pruned_,
        CUSPARSELT_PRUNE_SPMMA_STRIP,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify pruning is valid (optional validation step)
    int* d_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_valid, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_valid, 0, sizeof(int)));
    CUSPARSELT_CHECK(cusparseLtSpMMAPruneCheck(
        &handle_, &matmul_, d_pruned_, d_valid, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int is_valid = 0;
    CUDA_CHECK(cudaMemcpy(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_valid));

    if (is_valid != 0) {
        fprintf(stderr, "WARNING: cuSPARSELt prune check returned %d (expected 0)\n",
                is_valid);
    }

    // Get compressed size
    size_t compressed_size = 0;
    size_t compressed_buffer_size = 0;
    CUSPARSELT_CHECK(cusparseLtSpMMACompressedSize(
        &handle_, &plan_, &compressed_size, &compressed_buffer_size));

    if (d_compressed_) cudaFree(d_compressed_);
    CUDA_CHECK(cudaMalloc(&d_compressed_, compressed_size));

    void* d_compress_buffer = nullptr;
    if (compressed_buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_compress_buffer, compressed_buffer_size));
    }

    CUSPARSELT_CHECK(cusparseLtSpMMACompress(
        &handle_, &plan_,
        d_pruned_, d_compressed_, d_compress_buffer,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (d_compress_buffer) cudaFree(d_compress_buffer);

    prepared_ = true;
}

// ---------------------------------------------------------------------------
// Execute SpMMA: D = 1 * A^T * B + 0 * C
// d_B: contiguous INT8 vector/matrix. For GEMV, just k elements.
// d_D: contiguous INT32 output. For GEMV, just m elements.
// ---------------------------------------------------------------------------
void CuSparseLtGemm::execute(const int8_t* d_B, int32_t* d_D,
                               cudaStream_t stream) {
    if (!prepared_) {
        fprintf(stderr, "CuSparseLtGemm::execute() called before prepare()\n");
        exit(EXIT_FAILURE);
    }

    // cuSPARSELt requires float alpha/beta even for CUSPARSE_COMPUTE_32I
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUSPARSELT_CHECK(cusparseLtMatmul(
        &handle_, &plan_,
        &alpha,
        d_compressed_,  // A (compressed)
        d_B,            // B (dense)
        &beta,
        d_D,            // C (unused with beta=0, but must be valid ptr)
        d_D,            // D (output)
        d_workspace_,
        &stream, 1));
}

void CuSparseLtGemm::get_pruned_weights(int8_t* d_out, cudaStream_t stream) const {
    if (!d_pruned_) {
        fprintf(stderr, "CuSparseLtGemm::get_pruned_weights() — not prepared\n");
        exit(EXIT_FAILURE);
    }
    const size_t bytes = static_cast<size_t>(m_) * k_ * sizeof(int8_t);
    CUDA_CHECK(cudaMemcpyAsync(d_out, d_pruned_, bytes,
                                cudaMemcpyDeviceToDevice, stream));
}

} // namespace spbitnet

#endif // SPBITNET_HAS_CUSPARSELT
