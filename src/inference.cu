#include "spbitnet/inference.h"
#include "spbitnet/cuda_utils.h"
#include "spbitnet/inference_kernels.h"
#include "spbitnet/sparse_ternary_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace spbitnet {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

InferenceEngine::InferenceEngine(const Model& model)
    : model_(model), cfg_(model.config()) {

    const int H       = cfg_.hidden_size;
    const int n_kv    = cfg_.num_key_value_heads;
    const int hd      = cfg_.head_dim;
    const int kv_dim  = n_kv * hd;
    const int inter   = cfg_.intermediate_size;
    const int max_seq = cfg_.max_position_embeddings;
    const int vocab   = cfg_.vocab_size;
    const int n_layers = cfg_.num_hidden_layers;

    // --- KV caches ---
    size_t kv_bytes = static_cast<size_t>(n_kv) * max_seq * hd * sizeof(half);
    d_k_cache_.resize(n_layers);
    d_v_cache_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        CUDA_CHECK(cudaMalloc(&d_k_cache_[l], kv_bytes));
        CUDA_CHECK(cudaMalloc(&d_v_cache_[l], kv_bytes));
        CUDA_CHECK(cudaMemset(d_k_cache_[l], 0, kv_bytes));
        CUDA_CHECK(cudaMemset(d_v_cache_[l], 0, kv_bytes));
    }

    // --- Scratch buffers ---
    CUDA_CHECK(cudaMalloc(&d_x_,        H * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_residual_,  H * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_q_,         H * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_kv_,        kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gate_,      inter * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_up_,        inter * sizeof(half)));

    int max_dim = std::max(H, inter);
    CUDA_CHECK(cudaMalloc(&d_quant_,     max_dim * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_int_out_,   max_dim * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_absmax_,    sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_,    vocab * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_scores_,
        static_cast<size_t>(cfg_.num_attention_heads) * max_seq * sizeof(float)));

    // Memory summary
    size_t total = n_layers * 2 * kv_bytes;
    total += static_cast<size_t>(H) * sizeof(half) * 3;         // x, residual, q
    total += static_cast<size_t>(kv_dim) * sizeof(half);         // kv
    total += static_cast<size_t>(inter) * sizeof(half) * 2;     // gate, up
    total += static_cast<size_t>(max_dim) * (sizeof(int8_t) + sizeof(int32_t));
    total += sizeof(float);
    total += static_cast<size_t>(vocab) * sizeof(float);
    total += static_cast<size_t>(cfg_.num_attention_heads) * max_seq * sizeof(float);

    printf("InferenceEngine: %.1f MB allocated (%.1f MB KV cache, %.1f MB scratch)\n",
           total / (1024.0 * 1024.0),
           n_layers * 2 * kv_bytes / (1024.0 * 1024.0),
           (total - n_layers * 2 * kv_bytes) / (1024.0 * 1024.0));
}

InferenceEngine::~InferenceEngine() {
    for (auto p : d_k_cache_) cudaFree(p);
    for (auto p : d_v_cache_) cudaFree(p);
    cudaFree(d_x_);
    cudaFree(d_residual_);
    cudaFree(d_q_);
    cudaFree(d_kv_);
    cudaFree(d_gate_);
    cudaFree(d_up_);
    cudaFree(d_quant_);
    cudaFree(d_int_out_);
    cudaFree(d_absmax_);
    cudaFree(d_logits_);
    cudaFree(d_attn_scores_);
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void InferenceEngine::reset() {
    seq_len_ = 0;
    size_t kv_bytes = static_cast<size_t>(cfg_.num_key_value_heads)
                    * cfg_.max_position_embeddings
                    * cfg_.head_dim * sizeof(half);
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        CUDA_CHECK(cudaMemset(d_k_cache_[l], 0, kv_bytes));
        CUDA_CHECK(cudaMemset(d_v_cache_[l], 0, kv_bytes));
    }
}

// ---------------------------------------------------------------------------
// BitLinear:  quantize → sparse ternary GEMV → dequantize
// ---------------------------------------------------------------------------

void InferenceEngine::bitlinear(const half* input, int input_size,
                                const SparseLinear& layer, half* output) {
    // 1. Absmax quantize: half → int8
    absmax_quantize_gpu(input, d_quant_, d_absmax_, input_size);

    // 2. Sparse ternary GEMV: int8 → int32
    sparse_ternary_gemv_gpu(layer.d_meta, layer.d_values,
                            d_quant_, d_int_out_,
                            layer.rows, layer.cols,
                            layer.meta_stride, layer.values_stride);

    // 3. Dequantize: int32 → half  (scale = gamma * absmax / 127)
    dequantize_gpu(d_int_out_, output, d_absmax_, layer.gamma, layer.rows);
}

// ---------------------------------------------------------------------------
// Forward pass (one token, autoregressive)
// ---------------------------------------------------------------------------

const float* InferenceEngine::forward(int token_id) {
    const int pos     = seq_len_;
    const int H       = cfg_.hidden_size;
    const int hd      = cfg_.head_dim;
    const int n_heads = cfg_.num_attention_heads;
    const int n_kv    = cfg_.num_key_value_heads;
    const int inter   = cfg_.intermediate_size;
    const int max_seq = cfg_.max_position_embeddings;
    const int seq_len = pos + 1;  // tokens seen so far (including this one)

    // =====================================================================
    // 1. Embedding lookup
    // =====================================================================
    CUDA_CHECK(cudaMemcpy(
        d_x_,
        model_.embed_tokens() + static_cast<size_t>(token_id) * H,
        H * sizeof(half),
        cudaMemcpyDeviceToDevice));

    // =====================================================================
    // 2. Transformer layers
    // =====================================================================
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        const auto& layer = model_.layer(l);

        // ----- Save residual -----
        CUDA_CHECK(cudaMemcpy(d_residual_, d_x_, H * sizeof(half),
                              cudaMemcpyDeviceToDevice));

        // ----- Pre-attention RMSNorm -----
        // Read from d_residual_ (saved input), write to d_x_
        rms_norm_gpu(d_residual_, layer.d_input_layernorm,
                     d_x_, H, cfg_.rms_norm_eps);

        // ----- Q, K, V projections (BitLinear) -----
        bitlinear(d_x_, H, layer.q_proj, d_q_);    // → (n_heads * hd)
        bitlinear(d_x_, H, layer.k_proj, d_kv_);   // → (n_kv * hd)

        // Apply RoPE to Q and K
        rope_gpu(d_q_,  n_heads, hd, pos, cfg_.rope_theta);
        rope_gpu(d_kv_, n_kv,    hd, pos, cfg_.rope_theta);

        // Store K in cache
        scatter_kv_gpu(d_kv_, d_k_cache_[l], n_kv, hd, max_seq, pos);

        // V projection — reuses d_kv_ (K already copied to cache)
        bitlinear(d_x_, H, layer.v_proj, d_kv_);   // → (n_kv * hd)

        // Store V in cache
        scatter_kv_gpu(d_kv_, d_v_cache_[l], n_kv, hd, max_seq, pos);

        // ----- Attention -----
        float attn_scale = 1.0f / sqrtf(static_cast<float>(hd));

        // Q · K^T / sqrt(d_k) for all heads and all positions 0..pos
        attention_scores_gpu(d_q_, d_k_cache_[l], d_attn_scores_,
                             n_heads, n_kv, hd, max_seq, seq_len, attn_scale);

        // Softmax
        softmax_gpu(d_attn_scores_, n_heads, max_seq, seq_len);

        // Weighted sum of V → d_x_ (hidden_size = n_heads * hd)
        attention_output_gpu(d_attn_scores_, d_v_cache_[l], d_x_,
                             n_heads, n_kv, hd, max_seq, seq_len);

        // ----- Output projection -----
        bitlinear(d_x_, H, layer.o_proj, d_q_);  // d_q_ as temp

        // SubLN: attention sub-normalization (before residual add)
        if (layer.d_attn_sub_norm) {
            rms_norm_gpu(d_q_, layer.d_attn_sub_norm, d_q_, H, cfg_.rms_norm_eps);
        }

        // Residual add
        residual_add_gpu(d_q_, d_residual_, d_x_, H);

        // ----- Save residual for MLP block -----
        CUDA_CHECK(cudaMemcpy(d_residual_, d_x_, H * sizeof(half),
                              cudaMemcpyDeviceToDevice));

        // ----- Post-attention RMSNorm -----
        rms_norm_gpu(d_residual_, layer.d_post_attention_layernorm,
                     d_x_, H, cfg_.rms_norm_eps);

        // ----- MLP (gated with ReLU²) -----
        bitlinear(d_x_, H, layer.gate_proj, d_gate_);  // → (inter)
        bitlinear(d_x_, H, layer.up_proj,   d_up_);    // → (inter)

        // ReLU²(gate) * up → float32 buffer (avoids float16 overflow).
        // relu²(x) can produce values >> 65504 before sub-norm normalizes them.
        // Safe to reuse d_int_out_ here — it's only used inside bitlinear().
        float* d_mlp_f32 = reinterpret_cast<float*>(d_int_out_);
        relu2_mul_f32_gpu(d_gate_, d_up_, d_mlp_f32, inter);

        // SubLN: FFN sub-normalization (float32 input → float16 output)
        if (layer.d_ffn_sub_norm) {
            rms_norm_f32in_gpu(d_mlp_f32, layer.d_ffn_sub_norm, d_gate_,
                               inter, cfg_.rms_norm_eps);
        } else {
            float_to_half_gpu(d_mlp_f32, d_gate_, inter);
        }

        // Down projection
        bitlinear(d_gate_, inter, layer.down_proj, d_q_);  // → (H)

        // Residual add
        residual_add_gpu(d_q_, d_residual_, d_x_, H);
    }

    // =====================================================================
    // 3. Final RMSNorm
    // =====================================================================
    rms_norm_gpu(d_x_, model_.final_norm(), d_residual_, H, cfg_.rms_norm_eps);

    // =====================================================================
    // 4. LM head: logits = lm_head @ hidden_state
    // =====================================================================
    half_gemv_gpu(model_.lm_head(), d_residual_, d_logits_,
                  cfg_.vocab_size, H);

    seq_len_++;
    return d_logits_;
}

}  // namespace spbitnet
