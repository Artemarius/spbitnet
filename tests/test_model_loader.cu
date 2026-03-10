/// \file test_model_loader.cu
/// \brief Integration test: create a tiny synthetic model on disk,
///        then load it with the C++ Model loader.

#include "spbitnet/model.h"
#include "spbitnet/cuda_utils.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

/// Write a raw binary file from a vector.
template <typename T>
void write_bin(const std::string& path, const std::vector<T>& data) {
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open()) << "Cannot open " << path;
    f.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(T));
}

/// Pack a single group of 4 ternary weights into meta (4-bit bitmap)
/// and values (2-bit sign pair).  Mirrors Python pack_sparse_ternary.
void pack_group(const int8_t* w, const bool* mask,
                uint32_t& bitmap, uint32_t& signs) {
    bitmap = 0;
    signs  = 0;
    int nz = 0;
    for (int i = 0; i < 4; ++i) {
        if (mask[i]) {
            bitmap |= (1u << i);
            if (w[i] < 0) signs |= (1u << nz);
            ++nz;
        }
    }
}

/// Create a tiny synthetic model directory for testing.
/// Config: hidden_size=8, layers=1, heads=1, kv_heads=1,
///         intermediate=16, vocab=32.
std::string create_test_model(const std::string& base_dir) {
    std::string dir = base_dir + "/test_model";
    fs::create_directories(dir + "/weights");

    constexpr int H = 8;            // hidden_size
    constexpr int V = 32;           // vocab_size
    constexpr int I = 16;           // intermediate_size
    constexpr int NUM_LAYERS = 1;

    // --- config.json ---
    json config;
    config["model_type"]             = "bitnet";
    config["hidden_size"]            = H;
    config["num_hidden_layers"]      = NUM_LAYERS;
    config["num_attention_heads"]    = 1;
    config["num_key_value_heads"]    = 1;
    config["intermediate_size"]      = I;
    config["vocab_size"]             = V;
    config["max_position_embeddings"] = 64;
    config["rms_norm_eps"]           = 1e-5;
    config["rope_theta"]             = 10000.0;
    config["head_dim"]               = H;
    config["hidden_act"]             = "relu2";
    config["tie_word_embeddings"]    = true;
    config["bos_token_id"]           = 0;
    config["eos_token_id"]           = 1;

    {
        std::ofstream f(dir + "/config.json");
        f << config.dump(2);
    }

    // --- Helper: create sparse ternary files ---
    // Generates random ternary weights with 2:4 mask, packs to binary.
    auto make_sparse = [&](const std::string& name, int rows, int cols) {
        // Simple deterministic pattern: alternating +1, -1, 0, 0
        int groups_per_row = cols / 4;
        int meta_stride  = (groups_per_row + 7) / 8;
        int values_stride = (groups_per_row + 15) / 16;

        std::vector<uint32_t> meta(rows * meta_stride, 0);
        std::vector<uint32_t> values(rows * values_stride, 0);

        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < groups_per_row; ++g) {
                // Positions 0 and 1 are non-zero: +1, -1
                int8_t w[4] = {1, -1, 0, 0};
                bool mask[4] = {true, true, false, false};

                uint32_t bm, sg;
                pack_group(w, mask, bm, sg);

                int mw = g / 8;
                int mb = (g % 8) * 4;
                meta[r * meta_stride + mw] |= (bm << mb);

                int vw = g / 16;
                int vb = (g % 16) * 2;
                values[r * values_stride + vw] |= (sg << vb);
            }
        }

        std::string base = dir + "/weights/" + name;
        write_bin(base + ".meta", meta);
        write_bin(base + ".values", values);

        return json{
            {"meta_file",     "weights/" + name + ".meta"},
            {"values_file",   "weights/" + name + ".values"},
            {"dtype",         "sparse_ternary"},
            {"shape",         {rows, cols}},
            {"meta_stride",   meta_stride},
            {"values_stride", values_stride},
            {"gamma",         0.5},
            {"meta_nbytes",   static_cast<int>(meta.size() * 4)},
            {"values_nbytes", static_cast<int>(values.size() * 4)},
        };
    };

    // --- Helper: create float tensor file ---
    auto make_float16 = [&](const std::string& name, int count) {
        std::vector<uint16_t> data(count, 0x3C00);  // 1.0 in float16
        std::string path = dir + "/weights/" + name + ".bin";
        write_bin(path, data);
        return json{
            {"file",   "weights/" + name + ".bin"},
            {"dtype",  "float16"},
            {"nbytes", static_cast<int>(count * 2)},
        };
    };

    auto make_float32 = [&](const std::string& name, int count) {
        std::vector<float> data(count, 1.0f);
        std::string path = dir + "/weights/" + name + ".bin";
        write_bin(path, data);
        return json{
            {"file",   "weights/" + name + ".bin"},
            {"dtype",  "float32"},
            {"shape",  {count}},
            {"nbytes", static_cast<int>(count * 4)},
        };
    };

    // --- manifest.json ---
    json manifest;
    manifest["format_version"] = 1;

    auto& w = manifest["weights"];

    // Embedding
    auto embed_entry = make_float16("embed_tokens", V * H);
    embed_entry["shape"] = {V, H};
    w["embed_tokens"] = embed_entry;

    // Final norm
    w["norm"] = make_float32("norm", H);

    // LM head (tied)
    w["lm_head"] = {{"tied_to", "embed_tokens"}, {"dtype", "tied"}};

    // Layer 0
    std::string pfx = "layers.0";
    w[pfx + ".input_layernorm"]          = make_float32(pfx + ".input_layernorm", H);
    w[pfx + ".post_attention_layernorm"] = make_float32(pfx + ".post_attention_layernorm", H);

    w[pfx + ".self_attn.q_proj"] = make_sparse(pfx + ".self_attn.q_proj", H, H);
    w[pfx + ".self_attn.k_proj"] = make_sparse(pfx + ".self_attn.k_proj", H, H);
    w[pfx + ".self_attn.v_proj"] = make_sparse(pfx + ".self_attn.v_proj", H, H);
    w[pfx + ".self_attn.o_proj"] = make_sparse(pfx + ".self_attn.o_proj", H, H);
    w[pfx + ".mlp.gate_proj"]    = make_sparse(pfx + ".mlp.gate_proj", I, H);
    w[pfx + ".mlp.up_proj"]      = make_sparse(pfx + ".mlp.up_proj", I, H);
    w[pfx + ".mlp.down_proj"]    = make_sparse(pfx + ".mlp.down_proj", H, I);

    {
        std::ofstream f(dir + "/manifest.json");
        f << manifest.dump(2);
    }

    return dir;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

class ModelLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use system temp directory
        const char* tmp = std::getenv("TMPDIR");
        if (!tmp) tmp = "/tmp";
        tmp_dir_ = std::string(tmp) + "/spbitnet_test_model";
        fs::remove_all(tmp_dir_);
        model_dir_ = create_test_model(tmp_dir_);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir_);
    }

    std::string tmp_dir_;
    std::string model_dir_;
};

TEST_F(ModelLoaderTest, LoadsConfigCorrectly) {
    auto model = spbitnet::Model::load(model_dir_);

    EXPECT_EQ(model.config().hidden_size, 8);
    EXPECT_EQ(model.config().num_hidden_layers, 1);
    EXPECT_EQ(model.config().num_attention_heads, 1);
    EXPECT_EQ(model.config().num_key_value_heads, 1);
    EXPECT_EQ(model.config().intermediate_size, 16);
    EXPECT_EQ(model.config().vocab_size, 32);
    EXPECT_EQ(model.config().head_dim, 8);
    EXPECT_EQ(model.config().hidden_act, "relu2");
    EXPECT_TRUE(model.config().tie_word_embeddings);
}

TEST_F(ModelLoaderTest, EmbeddingLoaded) {
    auto model = spbitnet::Model::load(model_dir_);

    EXPECT_NE(model.embed_tokens(), nullptr);
    // lm_head should return embed_tokens when tied
    EXPECT_EQ(model.lm_head(), model.embed_tokens());
}

TEST_F(ModelLoaderTest, NormWeightsLoaded) {
    auto model = spbitnet::Model::load(model_dir_);

    EXPECT_NE(model.final_norm(), nullptr);
    EXPECT_NE(model.layer(0).d_input_layernorm, nullptr);
    EXPECT_NE(model.layer(0).d_post_attention_layernorm, nullptr);
}

TEST_F(ModelLoaderTest, SparseWeightsLoaded) {
    auto model = spbitnet::Model::load(model_dir_);
    const auto& layer = model.layer(0);

    // Check all projections are loaded
    EXPECT_NE(layer.q_proj.d_meta, nullptr);
    EXPECT_NE(layer.q_proj.d_values, nullptr);
    EXPECT_EQ(layer.q_proj.rows, 8);
    EXPECT_EQ(layer.q_proj.cols, 8);
    EXPECT_FLOAT_EQ(layer.q_proj.gamma, 0.5f);

    EXPECT_NE(layer.gate_proj.d_meta, nullptr);
    EXPECT_EQ(layer.gate_proj.rows, 16);  // intermediate_size
    EXPECT_EQ(layer.gate_proj.cols, 8);   // hidden_size

    EXPECT_NE(layer.down_proj.d_meta, nullptr);
    EXPECT_EQ(layer.down_proj.rows, 8);   // hidden_size
    EXPECT_EQ(layer.down_proj.cols, 16);  // intermediate_size
}

TEST_F(ModelLoaderTest, GpuMemoryAccounted) {
    auto model = spbitnet::Model::load(model_dir_);

    // Should have allocated some GPU memory
    EXPECT_GT(model.total_gpu_bytes(), 0u);
}

TEST_F(ModelLoaderTest, MoveSemantics) {
    auto model1 = spbitnet::Model::load(model_dir_);
    size_t bytes1 = model1.total_gpu_bytes();
    EXPECT_GT(bytes1, 0u);

    // Move construction
    auto model2 = std::move(model1);
    EXPECT_EQ(model2.total_gpu_bytes(), bytes1);
    EXPECT_EQ(model1.total_gpu_bytes(), 0u);
}
