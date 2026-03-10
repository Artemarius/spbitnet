#include "spbitnet/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <climits>
#include <fstream>
#include <stdexcept>

namespace spbitnet {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

std::string Tokenizer::cp_to_utf8(char32_t cp) {
    std::string r;
    if (cp < 0x80) {
        r += static_cast<char>(cp);
    } else if (cp < 0x800) {
        r += static_cast<char>(0xC0 | (cp >> 6));
        r += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        r += static_cast<char>(0xE0 | (cp >> 12));
        r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        r += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        r += static_cast<char>(0xF0 | (cp >> 18));
        r += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        r += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return r;
}

std::vector<char32_t> Tokenizer::utf8_to_cps(const std::string& s) {
    std::vector<char32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        auto c = static_cast<unsigned char>(s[i]);
        char32_t cp;
        int len;
        if (c < 0x80) {
            cp = c; len = 1;
        } else if (c < 0xE0) {
            cp = c & 0x1F; len = 2;
        } else if (c < 0xF0) {
            cp = c & 0x0F; len = 3;
        } else {
            cp = c & 0x07; len = 4;
        }
        for (int j = 1; j < len && i + j < s.size(); ++j)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + j]) & 0x3F);
        out.push_back(cp);
        i += len;
    }
    return out;
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level encoding tables
// ---------------------------------------------------------------------------

void Tokenizer::init_byte_maps() {
    // bytes_to_unicode() from GPT-2.
    // Printable byte values map to themselves; non-printable get shifted to 256+.
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);     // 33-126
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);   // 161-172
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);   // 174-255

    auto cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        byte_encoder_[bs[i]] = static_cast<char32_t>(cs[i]);
        byte_decoder_[static_cast<char32_t>(cs[i])] =
            static_cast<uint8_t>(bs[i]);
    }
}

std::string Tokenizer::byte_encode(const std::string& raw) const {
    std::string out;
    out.reserve(raw.size() * 2);  // worst case: each byte -> 2-byte UTF-8
    for (unsigned char c : raw)
        out += cp_to_utf8(byte_encoder_[c]);
    return out;
}

std::string Tokenizer::byte_decode_str(const std::string& encoded) const {
    auto cps = utf8_to_cps(encoded);
    std::string out;
    out.reserve(cps.size());
    for (auto cp : cps) {
        auto it = byte_decoder_.find(cp);
        if (it != byte_decoder_.end())
            out += static_cast<char>(it->second);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Pre-tokenizer  (simplified GPT-4 / Llama 3 regex for ASCII)
//
// Original pattern:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+
//   |\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
//
// We approximate \p{L} with isalpha and \p{N} with isdigit (ASCII-only).
// Non-ASCII bytes fall through to the punctuation/other matchers and are
// handled correctly by byte-level BPE.
// ---------------------------------------------------------------------------

namespace {

inline bool is_alpha(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }
inline bool is_digit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }
inline bool is_space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

// (?i:'s|'t|'re|'ve|'m|'ll|'d)
int match_contraction(const std::string& t, int p) {
    if (p >= static_cast<int>(t.size()) || t[p] != '\'') return 0;
    int rem = static_cast<int>(t.size()) - p - 1;
    if (rem >= 2) {
        char a = std::tolower(static_cast<unsigned char>(t[p + 1]));
        char b = std::tolower(static_cast<unsigned char>(t[p + 2]));
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l'))
            return 3;
    }
    if (rem >= 1) {
        char a = std::tolower(static_cast<unsigned char>(t[p + 1]));
        if (a == 's' || a == 't' || a == 'm' || a == 'd')
            return 2;
    }
    return 0;
}

// [^\r\n\p{L}\p{N}]?\p{L}+
int match_word(const std::string& t, int p) {
    int i = p;
    int n = static_cast<int>(t.size());
    // optional leading non-letter/digit (not \r\n)
    if (i < n && !is_alpha(t[i]) && !is_digit(t[i]) &&
        t[i] != '\r' && t[i] != '\n')
        ++i;
    // one or more letters
    int start = i;
    while (i < n && is_alpha(t[i])) ++i;
    return (i > start) ? i - p : 0;
}

// \p{N}{1,3}
int match_number(const std::string& t, int p) {
    int i = p, n = static_cast<int>(t.size());
    while (i < n && is_digit(t[i]) && i - p < 3) ++i;
    return i - p;
}

// ' ?'[^\s\p{L}\p{N}]+[\r\n]*
int match_punct(const std::string& t, int p) {
    int i = p, n = static_cast<int>(t.size());
    if (i < n && t[i] == ' ') ++i;
    int start = i;
    while (i < n && !is_space(t[i]) && !is_alpha(t[i]) && !is_digit(t[i]))
        ++i;
    if (i == start) return 0;
    while (i < n && (t[i] == '\r' || t[i] == '\n')) ++i;
    return i - p;
}

// \s*[\r\n]+
int match_newlines(const std::string& t, int p) {
    int i = p, n = static_cast<int>(t.size());
    while (i < n && is_space(t[i]) && t[i] != '\r' && t[i] != '\n') ++i;
    int start = i;
    while (i < n && (t[i] == '\r' || t[i] == '\n')) ++i;
    return (i > start) ? i - p : 0;
}

// \s+(?!\S)  — whitespace NOT followed by non-whitespace
int match_trailing_space(const std::string& t, int p) {
    int i = p, n = static_cast<int>(t.size());
    while (i < n && is_space(t[i])) ++i;
    if (i == p) return 0;
    if (i < n && !is_space(t[i])) return 0;  // negative lookahead
    return i - p;
}

// \s+
int match_whitespace(const std::string& t, int p) {
    int i = p, n = static_cast<int>(t.size());
    while (i < n && is_space(t[i])) ++i;
    return i - p;
}

}  // anonymous namespace

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> chunks;
    int pos = 0;
    int n = static_cast<int>(text.size());
    while (pos < n) {
        int len = match_contraction(text, pos);
        if (!len) len = match_word(text, pos);
        if (!len) len = match_number(text, pos);
        if (!len) len = match_punct(text, pos);
        if (!len) len = match_newlines(text, pos);
        if (!len) len = match_trailing_space(text, pos);
        if (!len) len = match_whitespace(text, pos);
        if (!len) { ++pos; continue; }  // shouldn't happen
        chunks.push_back(text.substr(pos, len));
        pos += len;
    }
    return chunks;
}

// ---------------------------------------------------------------------------
// BPE algorithm
// ---------------------------------------------------------------------------

std::vector<std::string> Tokenizer::bpe(const std::string& chunk) const {
    // Split byte-encoded chunk into individual codepoints (each as a UTF-8 string)
    auto cps = utf8_to_cps(chunk);
    std::vector<std::string> word;
    word.reserve(cps.size());
    for (auto cp : cps)
        word.push_back(cp_to_utf8(cp));

    // Iteratively merge highest-priority (lowest-rank) pair
    while (word.size() >= 2) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i < static_cast<int>(word.size()) - 1; ++i) {
            auto it = merge_ranks_.find(word[i] + " " + word[i + 1]);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i < 0) break;

        // Merge ALL occurrences of the winning pair
        const std::string first = word[best_i];
        const std::string second = word[best_i + 1];
        const std::string merged = first + second;

        std::vector<std::string> new_word;
        new_word.reserve(word.size());
        size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == first &&
                word[i + 1] == second) {
                new_word.push_back(merged);
                i += 2;
            } else {
                new_word.push_back(std::move(word[i]));
                ++i;
            }
        }
        word = std::move(new_word);
    }
    return word;
}

// ---------------------------------------------------------------------------
// Load from tokenizer.json
// ---------------------------------------------------------------------------

Tokenizer Tokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open tokenizer: " + path);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Tokenizer JSON parse error: " +
                                 std::string(e.what()));
    }

    Tokenizer tok;
    tok.init_byte_maps();

    // --- Load model vocab ---
    int max_id = 0;
    if (j.contains("model") && j["model"].contains("vocab")) {
        for (auto& [token, id_val] : j["model"]["vocab"].items()) {
            int id = id_val.get<int>();
            tok.token_to_id_[token] = id;
            if (id > max_id) max_id = id;
        }
    }

    // --- Load added tokens (special tokens override) ---
    if (j.contains("added_tokens")) {
        for (auto& at : j["added_tokens"]) {
            int id = at["id"].get<int>();
            std::string content = at["content"].get<std::string>();
            tok.token_to_id_[content] = id;
            if (id > max_id) max_id = id;
            if (at.contains("special") && at["special"].get<bool>())
                tok.special_ids_.insert(id);
        }
    }

    // --- Build reverse map ---
    tok.id_to_token_.resize(max_id + 1);
    for (auto& [token, id] : tok.token_to_id_)
        tok.id_to_token_[id] = token;

    // --- Load merges ---
    if (j.contains("model") && j["model"].contains("merges")) {
        auto& merges = j["model"]["merges"];
        for (int i = 0; i < static_cast<int>(merges.size()); ++i)
            tok.merge_ranks_[merges[i].get<std::string>()] = i;
    }

    return tok;
}

// ---------------------------------------------------------------------------
// Encode / Decode
// ---------------------------------------------------------------------------

std::vector<int> Tokenizer::encode(const std::string& text) const {
    auto chunks = pre_tokenize(text);
    std::vector<int> ids;
    for (const auto& chunk : chunks) {
        auto encoded = byte_encode(chunk);
        auto tokens = bpe(encoded);
        for (const auto& tok : tokens) {
            auto it = token_to_id_.find(tok);
            if (it != token_to_id_.end())
                ids.push_back(it->second);
            // byte-level BPE guarantees every byte is in vocab,
            // so this should never be skipped
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string byte_str;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        if (special_ids_.count(id)) continue;
        byte_str += id_to_token_[id];
    }
    return byte_decode_str(byte_str);
}

std::string Tokenizer::decode_token(int id) const {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return "";
    if (special_ids_.count(id)) return "";
    return byte_decode_str(id_to_token_[id]);
}

}  // namespace spbitnet
