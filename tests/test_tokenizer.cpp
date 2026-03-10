#include "spbitnet/tokenizer.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <memory>
#include <string>

/// Create a minimal tokenizer.json for testing.
/// Vocab: individual bytes (256 entries) + a few merged tokens.
static std::string create_test_tokenizer(const std::string& dir) {
    nlohmann::json j;

    // GPT-2 bytes_to_unicode: build the 256 base byte tokens
    // We replicate the mapping so vocab keys match what byte_encode produces.
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    auto cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    // Helper: codepoint to UTF-8
    auto cp_to_utf8 = [](int cp) -> std::string {
        std::string r;
        if (cp < 0x80) {
            r += static_cast<char>(cp);
        } else if (cp < 0x800) {
            r += static_cast<char>(0xC0 | (cp >> 6));
            r += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            r += static_cast<char>(0xE0 | (cp >> 12));
            r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            r += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return r;
    };

    // Build vocab: 256 byte tokens (id 0-255)
    nlohmann::json vocab;
    for (int i = 0; i < 256; ++i) {
        std::string key = cp_to_utf8(cs[i]);
        vocab[key] = bs[i];  // byte value as id
    }

    // Add merged tokens (id 256+)
    // "H" + "i" -> "Hi" (id 256)
    std::string H_enc = cp_to_utf8('H');
    std::string i_enc = cp_to_utf8('i');
    vocab[H_enc + i_enc] = 256;

    // Ġ (space byte-encoded) + "t" -> "Ġt" (id 257)
    std::string space_enc = cp_to_utf8(cs[std::find(bs.begin(), bs.end(), ' ') - bs.begin()]);
    std::string t_enc = cp_to_utf8('t');
    vocab[space_enc + t_enc] = 257;

    // "Ġt" + "h" -> "Ġth" (id 258)
    std::string h_enc = cp_to_utf8('h');
    vocab[space_enc + t_enc + h_enc] = 258;

    // "Ġth" + "e" -> "Ġthe" (id 259)
    std::string e_enc = cp_to_utf8('e');
    vocab[space_enc + t_enc + h_enc + e_enc] = 259;

    // "r" + "e" -> "re" (id 260)
    std::string r_enc = cp_to_utf8('r');
    vocab[r_enc + e_enc] = 260;

    j["model"]["type"] = "BPE";
    j["model"]["vocab"] = vocab;

    // Merges (order = priority)
    j["model"]["merges"] = nlohmann::json::array({
        H_enc + " " + i_enc,               // rank 0: H+i -> Hi
        space_enc + " " + t_enc,            // rank 1: Ġ+t -> Ġt
        space_enc + t_enc + " " + h_enc,    // rank 2: Ġt+h -> Ġth
        space_enc + t_enc + h_enc + " " + e_enc,  // rank 3: Ġth+e -> Ġthe
        r_enc + " " + e_enc,               // rank 4: r+e -> re
    });

    // Special tokens
    j["added_tokens"] = nlohmann::json::array({
        {{"id", 300}, {"content", "<|bos|>"}, {"special", true}},
        {{"id", 301}, {"content", "<|eos|>"}, {"special", true}},
    });
    vocab["<|bos|>"] = 300;
    vocab["<|eos|>"] = 301;
    j["model"]["vocab"] = vocab;

    std::string path = dir + "/tokenizer.json";
    std::ofstream f(path);
    f << j.dump();
    return path;
}

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_dir_ = "./test_tokenizer_tmp";
        (void)std::system(("mkdir -p " + tmp_dir_).c_str());
        auto path = create_test_tokenizer(tmp_dir_);
        tok_ = std::make_unique<spbitnet::Tokenizer>(
            spbitnet::Tokenizer::load(path));
    }

    void TearDown() override {
        tok_.reset();
        (void)std::system(("rm -rf " + tmp_dir_).c_str());
    }

    std::string tmp_dir_;
    std::unique_ptr<spbitnet::Tokenizer> tok_;
};

TEST_F(TokenizerTest, VocabSize) {
    // 256 bytes + 5 merged + 2 special = 263, but max_id is 301
    EXPECT_EQ(tok_->vocab_size(), 302);
}

TEST_F(TokenizerTest, EncodeSingleWord) {
    // "Hi" should be tokenized as [256] (merged token)
    auto ids = tok_->encode("Hi");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 256);
}

TEST_F(TokenizerTest, EncodeWithSpace) {
    // " the" -> pre-tokenize as " the" -> byte_encode -> "Ġthe" -> merged to id 259
    auto ids = tok_->encode(" the");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 259);
}

TEST_F(TokenizerTest, EncodeMultipleTokens) {
    // "Hi the" -> pre-tokenize as ["Hi", " the"]
    // "Hi" -> bpe -> [256]
    // " the" -> byte_encode -> "Ġthe" -> bpe -> [259]
    auto ids = tok_->encode("Hi the");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 256);
    EXPECT_EQ(ids[1], 259);
}

TEST_F(TokenizerTest, DecodeRoundtrip) {
    std::string text = "Hi there";
    auto ids = tok_->encode(text);
    auto decoded = tok_->decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerTest, DecodeSkipsSpecialTokens) {
    // Special token IDs (300, 301) should produce empty string
    EXPECT_EQ(tok_->decode_token(300), "");
    EXPECT_EQ(tok_->decode_token(301), "");

    // Regular decode should skip them
    auto decoded = tok_->decode({300, 256, 301});  // <bos> Hi <eos>
    EXPECT_EQ(decoded, "Hi");
}

TEST_F(TokenizerTest, DecodeSingleToken) {
    EXPECT_EQ(tok_->decode_token(256), "Hi");
    EXPECT_EQ(tok_->decode_token(259), " the");
}

TEST_F(TokenizerTest, EncodeUnknownCharsUseByteFallback) {
    // Characters not in any merge should fall back to individual byte tokens.
    // "9" -> byte token for '9' (id = ascii value of '9' = 57)
    auto ids = tok_->encode("9");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], '9');
}
