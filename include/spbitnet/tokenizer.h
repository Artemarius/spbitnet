#pragma once

/// \file tokenizer.h
/// \brief Byte-level BPE tokenizer for Llama 3 / GPT-4 style tokenizer.json.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace spbitnet {

/// Loads a HuggingFace tokenizer.json and provides encode/decode.
/// Implements byte-level BPE with a simplified GPT-4 pre-tokenizer.
class Tokenizer {
public:
    /// Load from a HuggingFace tokenizer.json file.
    /// \throws std::runtime_error on parse or I/O errors.
    static Tokenizer load(const std::string& tokenizer_json_path);

    /// Encode text to token IDs (does NOT add BOS/EOS).
    std::vector<int> encode(const std::string& text) const;

    /// Decode a sequence of token IDs to text. Special tokens are skipped.
    std::string decode(const std::vector<int>& ids) const;

    /// Decode a single token ID. Returns "" for special tokens.
    std::string decode_token(int id) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

private:
    Tokenizer() = default;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_set<int> special_ids_;

    // BPE merge rules: "tok_a tok_b" -> priority rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_ranks_;

    // GPT-2 byte-level encoding tables
    char32_t byte_encoder_[256]{};
    std::unordered_map<char32_t, uint8_t> byte_decoder_;
    void init_byte_maps();

    // Split text into pre-token chunks (simplified GPT-4 regex for ASCII)
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Apply BPE merges to a byte-encoded chunk
    std::vector<std::string> bpe(const std::string& chunk) const;

    // Map raw bytes <-> GPT-2 byte-encoded unicode string
    std::string byte_encode(const std::string& raw) const;
    std::string byte_decode_str(const std::string& encoded) const;

    // UTF-8 <-> codepoint helpers
    static std::string cp_to_utf8(char32_t cp);
    static std::vector<char32_t> utf8_to_cps(const std::string& s);
};

}  // namespace spbitnet
