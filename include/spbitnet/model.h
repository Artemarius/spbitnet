#pragma once

/// \file model.h
/// \brief Model loader for spbitnet sparse ternary format.
///
/// Loads the output of convert_model.py: config.json, manifest.json,
/// and binary weight files.  All weights are uploaded to GPU memory
/// during load.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>

namespace spbitnet {

// ---------------------------------------------------------------------------
// Model configuration (parsed from config.json)
// ---------------------------------------------------------------------------
struct ModelConfig {
    std::string model_type;
    int hidden_size        = 0;
    int num_hidden_layers  = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int intermediate_size  = 0;
    int vocab_size         = 0;
    int max_position_embeddings = 0;
    float rms_norm_eps     = 1e-5f;
    float rope_theta       = 10000.0f;
    int head_dim           = 0;
    std::string hidden_act;
    bool tie_word_embeddings = false;
    int bos_token_id       = 1;
    int eos_token_id       = 2;
};

// ---------------------------------------------------------------------------
// GPU weight structures
// ---------------------------------------------------------------------------

/// Sparse ternary linear layer weights on GPU.
/// Corresponds to one BitLinear projection (e.g. q_proj).
struct SparseLinear {
    uint32_t* d_meta   = nullptr;   ///< Position bitmaps (GPU)
    uint32_t* d_values = nullptr;   ///< Sign pairs (GPU)
    float gamma        = 0.0f;      ///< Absmean dequantization scale
    int rows           = 0;         ///< Output dimension
    int cols           = 0;         ///< Input dimension
    int meta_stride    = 0;         ///< uint32 words per row in meta
    int values_stride  = 0;         ///< uint32 words per row in values

    /// Total GPU bytes used by this layer.
    size_t gpu_bytes() const {
        return static_cast<size_t>(rows) * meta_stride * sizeof(uint32_t)
             + static_cast<size_t>(rows) * values_stride * sizeof(uint32_t);
    }
};

/// All weights for a single transformer layer.
struct TransformerLayer {
    // Attention projections (sparse ternary)
    SparseLinear q_proj;
    SparseLinear k_proj;
    SparseLinear v_proj;
    SparseLinear o_proj;

    // MLP projections (sparse ternary)
    SparseLinear gate_proj;
    SparseLinear up_proj;
    SparseLinear down_proj;

    // RMSNorm weights (GPU, float32)
    float* d_input_layernorm         = nullptr;  ///< (hidden_size,)
    float* d_post_attention_layernorm = nullptr;  ///< (hidden_size,)
};

// ---------------------------------------------------------------------------
// Model class — owns all GPU memory for weights
// ---------------------------------------------------------------------------

class Model {
public:
    /// Load a model from a directory produced by convert_model.py.
    /// Allocates GPU memory and copies all weights.
    ///
    /// \param model_dir  Path to the model directory (contains config.json,
    ///                    manifest.json, weights/).
    /// \throws std::runtime_error on file I/O or GPU allocation errors.
    static Model load(const std::string& model_dir);

    ~Model();

    // Move-only (GPU resources)
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // --- Accessors ---
    const ModelConfig& config() const { return config_; }
    const std::vector<TransformerLayer>& layers() const { return layers_; }
    const TransformerLayer& layer(int i) const { return layers_[i]; }

    /// Token embedding table (GPU, float16).  Shape: (vocab_size, hidden_size).
    const half* embed_tokens() const { return d_embed_tokens_; }

    /// Final RMSNorm weight (GPU, float32).  Shape: (hidden_size,).
    const float* final_norm() const { return d_final_norm_; }

    /// LM head weight (GPU, float16).  Shape: (vocab_size, hidden_size).
    /// Returns embed_tokens() when weights are tied.
    const half* lm_head() const {
        return d_lm_head_ ? d_lm_head_ : d_embed_tokens_;
    }

    /// Total GPU memory used by all weights (bytes).
    size_t total_gpu_bytes() const { return total_gpu_bytes_; }

    /// Print a summary of loaded weights.
    void print_summary() const;

private:
    Model() = default;

    void free_gpu();

    ModelConfig config_;
    std::vector<TransformerLayer> layers_;

    half*  d_embed_tokens_ = nullptr;
    float* d_final_norm_   = nullptr;
    half*  d_lm_head_      = nullptr;   // nullptr if tied

    size_t total_gpu_bytes_ = 0;
};

}  // namespace spbitnet
