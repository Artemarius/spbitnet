#pragma once

/// \file inference.h
/// \brief Transformer inference engine for sparse ternary BitNet models.

#include "spbitnet/model.h"

#include <cuda_fp16.h>
#include <vector>

namespace spbitnet {

/// Runs autoregressive inference on a loaded sparse-ternary BitNet model.
///
/// Processes one token at a time, maintaining a KV cache across calls.
/// Each forward() call returns device-side logits for the next token.
///
/// Data flow per BitLinear layer:
///   1. Absmax quantize:  half → int8  (with per-token scale)
///   2. Sparse ternary GEMV:  int8 → int32
///   3. Dequantize:  int32 → half  (scale = gamma * absmax / 127)
class InferenceEngine {
public:
    explicit InferenceEngine(const Model& model);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /// Forward pass for one token (autoregressive).
    /// Returns device pointer to logits (vocab_size floats).
    const float* forward(int token_id);

    /// Reset KV cache and sequence position for a new sequence.
    void reset();

    int seq_len() const { return seq_len_; }

private:
    const Model& model_;
    const ModelConfig& cfg_;
    int seq_len_ = 0;

    // KV cache per layer: (num_kv_heads, max_seq_len, head_dim), float16
    std::vector<half*> d_k_cache_;
    std::vector<half*> d_v_cache_;

    // Scratch buffers (all device pointers)
    half*    d_x_;              // current hidden state (hidden_size)
    half*    d_residual_;       // residual connection (hidden_size)
    half*    d_q_;              // Q projection / temp (hidden_size)
    half*    d_kv_;             // K or V temp (num_kv_heads * head_dim)
    half*    d_gate_;           // gate projection (intermediate_size)
    half*    d_up_;             // up projection (intermediate_size)
    int8_t*  d_quant_;          // quantized activations (max dim)
    int32_t* d_int_out_;        // integer GEMV output (max dim)
    float*   d_absmax_;         // absmax scale (1 float)
    float*   d_logits_;         // output logits (vocab_size)
    float*   d_attn_scores_;    // attention scores (num_heads * max_seq_len)

    /// BitLinear: quantize → sparse ternary GEMV → dequantize.
    void bitlinear(const half* input, int input_size,
                   const SparseLinear& layer, half* output);
};

}  // namespace spbitnet
