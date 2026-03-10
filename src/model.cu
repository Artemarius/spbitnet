/// \file model.cu
/// \brief Model loader implementation — parses JSON config/manifest,
///        reads binary weight files, uploads to GPU.

#include "spbitnet/model.h"
#include "spbitnet/cuda_utils.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace spbitnet {

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------

static std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    if (!f) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return data;
}

static json read_json_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    json j;
    f >> j;
    return j;
}

// ---------------------------------------------------------------------------
// GPU upload helpers
// ---------------------------------------------------------------------------

/// Allocate GPU memory and copy host data.  Returns device pointer.
template <typename T>
static T* upload_to_gpu(const void* host_data, size_t count, size_t& total) {
    size_t nbytes = count * sizeof(T);
    T* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, nbytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, host_data, nbytes, cudaMemcpyHostToDevice));
    total += nbytes;
    return d_ptr;
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

static ModelConfig parse_config(const json& j) {
    ModelConfig c;
    c.model_type             = j.value("model_type", "bitnet");
    c.hidden_size            = j.at("hidden_size").get<int>();
    c.num_hidden_layers      = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads    = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads    = j.value("num_key_value_heads", c.num_attention_heads);
    c.intermediate_size      = j.at("intermediate_size").get<int>();
    c.vocab_size             = j.at("vocab_size").get<int>();
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.rms_norm_eps           = j.value("rms_norm_eps", 1e-5f);
    c.rope_theta             = j.value("rope_theta", 10000.0f);
    c.head_dim               = j.value("head_dim", c.hidden_size / c.num_attention_heads);
    c.hidden_act             = j.value("hidden_act", "relu2");
    c.tie_word_embeddings    = j.value("tie_word_embeddings", false);
    c.bos_token_id           = j.value("bos_token_id", 1);
    c.eos_token_id           = j.value("eos_token_id", 2);
    return c;
}

// ---------------------------------------------------------------------------
// Weight loading from manifest
// ---------------------------------------------------------------------------

/// Load a sparse ternary weight from its manifest entry.
static SparseLinear load_sparse_linear(
    const std::string& model_dir,
    const json& entry,
    size_t& total_gpu
) {
    SparseLinear w;

    auto shape = entry.at("shape").get<std::vector<int>>();
    w.rows          = shape[0];
    w.cols          = shape[1];
    w.meta_stride   = entry.at("meta_stride").get<int>();
    w.values_stride = entry.at("values_stride").get<int>();
    w.gamma         = entry.at("gamma").get<float>();

    std::string meta_path   = model_dir + "/" + entry.at("meta_file").get<std::string>();
    std::string values_path = model_dir + "/" + entry.at("values_file").get<std::string>();

    // Load meta array
    auto meta_data = read_binary_file(meta_path);
    size_t meta_count = static_cast<size_t>(w.rows) * w.meta_stride;
    if (meta_data.size() != meta_count * sizeof(uint32_t)) {
        throw std::runtime_error(
            "Meta file size mismatch: " + meta_path +
            " (expected " + std::to_string(meta_count * 4) +
            ", got " + std::to_string(meta_data.size()) + ")"
        );
    }
    w.d_meta = upload_to_gpu<uint32_t>(meta_data.data(), meta_count, total_gpu);

    // Load values array
    auto values_data = read_binary_file(values_path);
    size_t values_count = static_cast<size_t>(w.rows) * w.values_stride;
    if (values_data.size() != values_count * sizeof(uint32_t)) {
        throw std::runtime_error(
            "Values file size mismatch: " + values_path +
            " (expected " + std::to_string(values_count * 4) +
            ", got " + std::to_string(values_data.size()) + ")"
        );
    }
    w.d_values = upload_to_gpu<uint32_t>(values_data.data(), values_count, total_gpu);

    return w;
}

/// Load a float32 tensor from binary file, upload to GPU.
static float* load_float32_tensor(
    const std::string& path,
    size_t expected_elements,
    size_t& total_gpu
) {
    auto data = read_binary_file(path);
    if (data.size() != expected_elements * sizeof(float)) {
        throw std::runtime_error(
            "Float32 file size mismatch: " + path +
            " (expected " + std::to_string(expected_elements * 4) +
            ", got " + std::to_string(data.size()) + ")"
        );
    }
    return upload_to_gpu<float>(data.data(), expected_elements, total_gpu);
}

/// Load a float16 tensor from binary file, upload to GPU as half.
static half* load_float16_tensor(
    const std::string& path,
    size_t expected_elements,
    size_t& total_gpu
) {
    auto data = read_binary_file(path);
    if (data.size() != expected_elements * sizeof(half)) {
        throw std::runtime_error(
            "Float16 file size mismatch: " + path +
            " (expected " + std::to_string(expected_elements * 2) +
            ", got " + std::to_string(data.size()) + ")"
        );
    }
    return upload_to_gpu<half>(data.data(), expected_elements, total_gpu);
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

Model Model::load(const std::string& model_dir) {
    Model m;

    // 1. Parse config.json
    printf("Loading model from %s ...\n", model_dir.c_str());
    auto config_json = read_json_file(model_dir + "/config.json");
    m.config_ = parse_config(config_json);

    const auto& cfg = m.config_;
    printf("  Architecture: %s\n", cfg.model_type.c_str());
    printf("  hidden_size=%d, layers=%d, heads=%d, kv_heads=%d\n",
           cfg.hidden_size, cfg.num_hidden_layers,
           cfg.num_attention_heads, cfg.num_key_value_heads);
    printf("  intermediate_size=%d, vocab_size=%d, head_dim=%d\n",
           cfg.intermediate_size, cfg.vocab_size, cfg.head_dim);

    // 2. Parse manifest.json
    auto manifest = read_json_file(model_dir + "/manifest.json");
    const auto& weights = manifest.at("weights");

    // 3. Load embedding
    if (weights.contains("embed_tokens")) {
        const auto& e = weights.at("embed_tokens");
        std::string file = model_dir + "/" + e.at("file").get<std::string>();
        size_t count = static_cast<size_t>(cfg.vocab_size) * cfg.hidden_size;
        printf("  Loading embed_tokens (%d x %d, float16) ...\n",
               cfg.vocab_size, cfg.hidden_size);
        m.d_embed_tokens_ = load_float16_tensor(file, count, m.total_gpu_bytes_);
    }

    // 4. Load final norm
    if (weights.contains("norm")) {
        const auto& e = weights.at("norm");
        std::string file = model_dir + "/" + e.at("file").get<std::string>();
        printf("  Loading final norm (%d, float32) ...\n", cfg.hidden_size);
        m.d_final_norm_ = load_float32_tensor(
            file, cfg.hidden_size, m.total_gpu_bytes_
        );
    }

    // 5. Load LM head (or mark as tied)
    if (weights.contains("lm_head")) {
        const auto& e = weights.at("lm_head");
        if (e.value("dtype", "") == "tied") {
            printf("  lm_head: tied to embed_tokens\n");
            m.d_lm_head_ = nullptr;  // use embed_tokens
        } else {
            std::string file = model_dir + "/" + e.at("file").get<std::string>();
            size_t count = static_cast<size_t>(cfg.vocab_size) * cfg.hidden_size;
            printf("  Loading lm_head (%d x %d, float16) ...\n",
                   cfg.vocab_size, cfg.hidden_size);
            m.d_lm_head_ = load_float16_tensor(file, count, m.total_gpu_bytes_);
        }
    }

    // 6. Load transformer layers
    m.layers_.resize(cfg.num_hidden_layers);

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& layer = m.layers_[i];
        std::string prefix = "layers." + std::to_string(i);
        printf("  Loading layer %d/%d ...\r", i + 1, cfg.num_hidden_layers);
        fflush(stdout);

        // Norm weights
        {
            std::string name = prefix + ".input_layernorm";
            if (weights.contains(name)) {
                const auto& e = weights.at(name);
                std::string file = model_dir + "/" + e.at("file").get<std::string>();
                layer.d_input_layernorm = load_float32_tensor(
                    file, cfg.hidden_size, m.total_gpu_bytes_
                );
            }
        }
        {
            std::string name = prefix + ".post_attention_layernorm";
            if (weights.contains(name)) {
                const auto& e = weights.at(name);
                std::string file = model_dir + "/" + e.at("file").get<std::string>();
                layer.d_post_attention_layernorm = load_float32_tensor(
                    file, cfg.hidden_size, m.total_gpu_bytes_
                );
            }
        }

        // Sparse ternary projections
        auto load_proj = [&](const std::string& proj_name, SparseLinear& dest) {
            std::string name = prefix + "." + proj_name;
            if (weights.contains(name)) {
                dest = load_sparse_linear(model_dir, weights.at(name), m.total_gpu_bytes_);
            }
        };

        load_proj("self_attn.q_proj", layer.q_proj);
        load_proj("self_attn.k_proj", layer.k_proj);
        load_proj("self_attn.v_proj", layer.v_proj);
        load_proj("self_attn.o_proj", layer.o_proj);
        load_proj("mlp.gate_proj",    layer.gate_proj);
        load_proj("mlp.up_proj",      layer.up_proj);
        load_proj("mlp.down_proj",    layer.down_proj);
    }

    printf("  Loading layer %d/%d ... done\n",
           cfg.num_hidden_layers, cfg.num_hidden_layers);

    m.print_summary();
    return m;
}

// ---------------------------------------------------------------------------
// Cleanup and utilities
// ---------------------------------------------------------------------------

void Model::free_gpu() {
    if (d_embed_tokens_) { cudaFree(d_embed_tokens_); d_embed_tokens_ = nullptr; }
    if (d_final_norm_)   { cudaFree(d_final_norm_);   d_final_norm_ = nullptr; }
    if (d_lm_head_)      { cudaFree(d_lm_head_);      d_lm_head_ = nullptr; }

    for (auto& layer : layers_) {
        auto free_sparse = [](SparseLinear& s) {
            if (s.d_meta)   { cudaFree(s.d_meta);   s.d_meta = nullptr; }
            if (s.d_values) { cudaFree(s.d_values); s.d_values = nullptr; }
        };
        free_sparse(layer.q_proj);
        free_sparse(layer.k_proj);
        free_sparse(layer.v_proj);
        free_sparse(layer.o_proj);
        free_sparse(layer.gate_proj);
        free_sparse(layer.up_proj);
        free_sparse(layer.down_proj);

        if (layer.d_input_layernorm) {
            cudaFree(layer.d_input_layernorm);
            layer.d_input_layernorm = nullptr;
        }
        if (layer.d_post_attention_layernorm) {
            cudaFree(layer.d_post_attention_layernorm);
            layer.d_post_attention_layernorm = nullptr;
        }
    }
    layers_.clear();
    total_gpu_bytes_ = 0;
}

Model::~Model() {
    free_gpu();
}

Model::Model(Model&& other) noexcept
    : config_(std::move(other.config_))
    , layers_(std::move(other.layers_))
    , d_embed_tokens_(other.d_embed_tokens_)
    , d_final_norm_(other.d_final_norm_)
    , d_lm_head_(other.d_lm_head_)
    , total_gpu_bytes_(other.total_gpu_bytes_)
{
    other.d_embed_tokens_ = nullptr;
    other.d_final_norm_   = nullptr;
    other.d_lm_head_      = nullptr;
    other.total_gpu_bytes_ = 0;
}

Model& Model::operator=(Model&& other) noexcept {
    if (this != &other) {
        free_gpu();
        config_          = std::move(other.config_);
        layers_          = std::move(other.layers_);
        d_embed_tokens_  = other.d_embed_tokens_;
        d_final_norm_    = other.d_final_norm_;
        d_lm_head_       = other.d_lm_head_;
        total_gpu_bytes_ = other.total_gpu_bytes_;

        other.d_embed_tokens_ = nullptr;
        other.d_final_norm_   = nullptr;
        other.d_lm_head_      = nullptr;
        other.total_gpu_bytes_ = 0;
    }
    return *this;
}

void Model::print_summary() const {
    printf("\n  Model loaded:\n");
    printf("    Layers:     %d\n", config_.num_hidden_layers);
    printf("    GPU memory: %.1f MB\n",
           static_cast<double>(total_gpu_bytes_) / 1e6);

    // Break down by category
    size_t embed_bytes = 0;
    if (d_embed_tokens_) {
        embed_bytes = static_cast<size_t>(config_.vocab_size)
                    * config_.hidden_size * sizeof(half);
    }
    size_t lm_head_bytes = 0;
    if (d_lm_head_) {
        lm_head_bytes = static_cast<size_t>(config_.vocab_size)
                      * config_.hidden_size * sizeof(half);
    }
    size_t norm_bytes = 0;
    size_t sparse_bytes = 0;
    for (const auto& layer : layers_) {
        if (layer.d_input_layernorm)
            norm_bytes += config_.hidden_size * sizeof(float);
        if (layer.d_post_attention_layernorm)
            norm_bytes += config_.hidden_size * sizeof(float);
        sparse_bytes += layer.q_proj.gpu_bytes();
        sparse_bytes += layer.k_proj.gpu_bytes();
        sparse_bytes += layer.v_proj.gpu_bytes();
        sparse_bytes += layer.o_proj.gpu_bytes();
        sparse_bytes += layer.gate_proj.gpu_bytes();
        sparse_bytes += layer.up_proj.gpu_bytes();
        sparse_bytes += layer.down_proj.gpu_bytes();
    }
    if (d_final_norm_)
        norm_bytes += config_.hidden_size * sizeof(float);

    printf("    Embedding:  %.1f MB\n", embed_bytes / 1e6);
    printf("    Sparse:     %.1f MB\n", sparse_bytes / 1e6);
    printf("    Norms:      %.3f MB\n", norm_bytes / 1e6);
    if (lm_head_bytes > 0) {
        printf("    LM head:    %.1f MB\n", lm_head_bytes / 1e6);
    } else {
        printf("    LM head:    tied with embedding\n");
    }
    printf("\n");
}

}  // namespace spbitnet
