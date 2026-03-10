#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace spbitnet {

// ---------------------------------------------------------------------------
// Ternary encoding: 2 bits per weight
//   00 = 0, 01 = +1, 10 = -1, 11 = reserved
// Packing: 16 weights per uint32_t, LSB-first
//   weight 0 → bits [1:0], weight 1 → bits [3:2], ..., weight 15 → bits [31:30]
// ---------------------------------------------------------------------------

constexpr int kWeightsPerPack = 16;  // weights per uint32_t

// Quantize float to 2-bit ternary code (simple threshold quantization).
inline uint8_t encode_ternary(float val) {
    if (val > 0.5f) return 0b01;   // +1
    if (val < -0.5f) return 0b10;  // -1
    return 0b00;                    //  0
}

// Decode 2-bit ternary code to int8.
inline int8_t decode_ternary(uint8_t code) {
    if (code == 0b01) return +1;
    if (code == 0b10) return -1;
    return 0;
}

// ---------------------------------------------------------------------------
// TernaryTensor — CPU-side packed ternary weight storage
// ---------------------------------------------------------------------------

class TernaryTensor {
public:
    TernaryTensor(int rows, int cols)
        : rows_(rows), cols_(cols),
          data_(rows * packed_words_per_row(cols), 0u) {}

    // Pack float weights with threshold quantization.
    static TernaryTensor pack_from_float(const float* weights, int rows, int cols) {
        TernaryTensor t(rows, cols);
        const int stride = t.packed_row_stride();

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int word_idx = c / kWeightsPerPack;
                int bit_pos = (c % kWeightsPerPack) * 2;
                uint8_t code = encode_ternary(weights[r * cols + c]);
                t.data_[r * stride + word_idx] |=
                    (static_cast<uint32_t>(code) << bit_pos);
            }
        }
        return t;
    }

    // Pack pre-quantized {-1, 0, +1} int8 weights.
    static TernaryTensor pack_from_int8(const int8_t* weights, int rows, int cols) {
        TernaryTensor t(rows, cols);
        const int stride = t.packed_row_stride();

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int word_idx = c / kWeightsPerPack;
                int bit_pos = (c % kWeightsPerPack) * 2;

                int8_t val = weights[r * cols + c];
                uint8_t code = 0b00;
                if (val > 0) code = 0b01;
                else if (val < 0) code = 0b10;

                t.data_[r * stride + word_idx] |=
                    (static_cast<uint32_t>(code) << bit_pos);
            }
        }
        return t;
    }

    // Unpack all weights to int8 array (row-major, rows*cols elements).
    void unpack_to_int8(int8_t* out) const {
        const int stride = packed_row_stride();

        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                int word_idx = c / kWeightsPerPack;
                int bit_pos = (c % kWeightsPerPack) * 2;

                uint32_t word = data_[r * stride + word_idx];
                uint8_t code = (word >> bit_pos) & 0x3;
                out[r * cols_ + c] = decode_ternary(code);
            }
        }
    }

    // Get single weight value.
    int8_t get(int row, int col) const {
        assert(row >= 0 && row < rows_);
        assert(col >= 0 && col < cols_);

        const int stride = packed_row_stride();
        int word_idx = col / kWeightsPerPack;
        int bit_pos = (col % kWeightsPerPack) * 2;

        uint32_t word = data_[row * stride + word_idx];
        uint8_t code = (word >> bit_pos) & 0x3;
        return decode_ternary(code);
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    const uint32_t* packed_data() const { return data_.data(); }
    size_t packed_size() const { return data_.size(); }
    size_t packed_row_stride() const { return packed_words_per_row(cols_); }

private:
    int rows_;
    int cols_;
    std::vector<uint32_t> data_;

    // Number of uint32_t words needed per row (ceil(cols / 16)).
    static size_t packed_words_per_row(int cols) {
        return (static_cast<size_t>(cols) + kWeightsPerPack - 1) / kWeightsPerPack;
    }
};

} // namespace spbitnet
