#pragma once

#ifdef SPBITNET_HAS_CUSPARSELT

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cusparseLt.h>

namespace spbitnet {

// ---------------------------------------------------------------------------
// CuSparseLtGemm — RAII wrapper for cuSPARSELt 2:4 sparse INT8 GEMM/GEMV.
//
// Manages the full lifecycle: handle, descriptors, matmul plan, pruning,
// compression, workspace, and execution.
//
// Usage:
//   CuSparseLtGemm ctx(m, k, n);        // D[m,n] = A[m,k] * B[k,n]
//   ctx.prepare(d_int8_weights);         // prune to 2:4 + compress
//   ctx.execute(d_B, d_D);              // run SpMMA
//
// For GEMV: set n=1.
// Weights (A) are the sparse matrix. B is dense input, D is dense output.
// Compute type: INT32 accumulation.
// ---------------------------------------------------------------------------

class CuSparseLtGemm {
public:
    // m: output rows (weight rows)
    // k: inner dimension (weight cols)
    // n: output cols — must be >= 16 (cuSPARSELt INT8 Tensor Core constraint).
    //    cuSPARSELt cannot do GEMV (n=1). For GEMV, use our custom sparse
    //    ternary kernel instead. This is a hardware limitation of SpMMA tiles.
    CuSparseLtGemm(int m, int k, int n);
    ~CuSparseLtGemm();

    // Non-copyable, non-movable (owns GPU resources)
    CuSparseLtGemm(const CuSparseLtGemm&) = delete;
    CuSparseLtGemm& operator=(const CuSparseLtGemm&) = delete;

    // Prune d_weights to 2:4 pattern and compress for SpMMA.
    // d_weights: device ptr to INT8 row-major [m x k] matrix.
    // After this call, the compressed weights are stored internally.
    void prepare(const int8_t* d_weights, cudaStream_t stream = nullptr);

    // Execute: D = A_compressed * B (alpha=1, beta=0)
    // d_B: device ptr to INT8 [k x n] (column-major for n>1, contiguous for n=1)
    // d_D: device ptr to INT32 [m x n] output
    void execute(const int8_t* d_B, int32_t* d_D, cudaStream_t stream = nullptr);

    // Retrieve the pruned (but uncompressed) weights for correctness checks.
    // d_out: device ptr to INT8 [m x k], must be pre-allocated.
    // Only valid after prepare() has been called.
    void get_pruned_weights(int8_t* d_out, cudaStream_t stream = nullptr) const;

    int m() const { return m_; }
    int k() const { return k_; }
    int n() const { return n_; }

private:
    int m_, k_, n_;
    bool prepared_ = false;

    cusparseLtHandle_t            handle_;
    cusparseLtMatDescriptor_t     mat_a_;     // structured (sparse)
    cusparseLtMatDescriptor_t     mat_b_;     // dense
    cusparseLtMatDescriptor_t     mat_c_;     // dense (output)
    cusparseLtMatmulDescriptor_t  matmul_;
    cusparseLtMatmulAlgSelection_t alg_sel_;
    cusparseLtMatmulPlan_t        plan_;

    int8_t*  d_pruned_     = nullptr;  // [m x k] pruned weights (kept for debugging)
    void*    d_compressed_ = nullptr;  // compressed sparse format
    void*    d_workspace_  = nullptr;  // matmul workspace
    size_t   workspace_size_ = 0;
};

} // namespace spbitnet

#endif // SPBITNET_HAS_CUSPARSELT
