#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace spbitnet {

// ---------------------------------------------------------------------------
// Compressed Sparse-Ternary format for 2:4 structured sparsity
//
// Every 4 consecutive elements in a row form a "group". Each group has
// exactly 2 non-zero weights and 2 zeros (2:4 sparsity pattern).
//
// Two separated arrays for GPU coalescing:
//
// 1. Meta array (uint32_t*): position bitmaps
//    - Per group of 4: a 4-bit bitmap with exactly 2 bits set
//    - Bit i (0-indexed from LSB) is set iff position i is non-zero
//    - 8 groups per uint32_t, LSB-first
//    - Group g's bitmap: (meta[g/8] >> ((g%8)*4)) & 0xF
//
// 2. Values array (uint32_t*): sign bits of non-zero weights
//    - Per group of 4: 2 bits [sign_hi, sign_lo]
//    - sign_lo (bit 0) = sign of lower-positioned non-zero  (0=+1, 1=-1)
//    - sign_hi (bit 1) = sign of higher-positioned non-zero (0=+1, 1=-1)
//    - 16 groups per uint32_t, LSB-first
//    - Group g's signs: (values[g/16] >> ((g%16)*2)) & 0x3
//
// Memory: 1.5 bits/weight (vs 2 bits/weight for dense ternary = 75%)
// ---------------------------------------------------------------------------

constexpr int kSparseGroupSize = 4;
constexpr int kSparseNonZeros = 2;
constexpr int kMetaGroupsPerWord = 8;     // 8 x 4-bit bitmaps per uint32
constexpr int kValuesGroupsPerWord = 16;  // 16 x 2-bit sign pairs per uint32

class SparseTernaryTensor {
public:
    SparseTernaryTensor(int rows, int cols)
        : rows_(rows), cols_(cols),
          meta_(rows * meta_words_per_row(cols), 0u),
          values_(rows * values_words_per_row(cols), 0u) {
        assert(cols % kSparseGroupSize == 0 &&
               "cols must be a multiple of 4 for 2:4 sparsity");
        assert(rows > 0 && cols > 0);
    }

    // -----------------------------------------------------------------------
    // Pack from pre-quantized ternary weights {-1,0,+1} with a mask.
    // mask[r*cols+c] != 0 means this weight is kept (non-zero in sparse
    // output). Exactly 2 of every 4 mask entries must be non-zero.
    // Uses uint8_t instead of bool to avoid std::vector<bool> pitfalls.
    // -----------------------------------------------------------------------
    static SparseTernaryTensor pack_from_dense(const int8_t* weights,
                                                const uint8_t* mask,
                                                int rows, int cols) {
        assert(cols % kSparseGroupSize == 0);
        SparseTernaryTensor t(rows, cols);
        const int groups_per_row = cols / kSparseGroupSize;
        const size_t m_stride = t.meta_row_stride();
        const size_t v_stride = t.values_row_stride();

        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < groups_per_row; ++g) {
                int base = r * cols + g * kSparseGroupSize;

                // Validate: exactly 2 of 4 mask entries are true.
                int mask_count = 0;
                for (int i = 0; i < kSparseGroupSize; ++i)
                    mask_count += mask[base + i] ? 1 : 0;
                assert(mask_count == kSparseNonZeros &&
                       "Each group of 4 must have exactly 2 mask entries set");

                // Build 4-bit position bitmap and collect sign bits.
                uint32_t bitmap = 0;
                uint32_t signs = 0;
                int nz_idx = 0;  // 0 for lower-positioned, 1 for higher

                for (int i = 0; i < kSparseGroupSize; ++i) {
                    if (mask[base + i]) {
                        bitmap |= (1u << i);

                        // Encode sign: 0 = +1, 1 = -1
                        // If the masked position has weight 0, treat as +1.
                        int8_t val = weights[base + i];
                        uint32_t sign_bit = (val < 0) ? 1u : 0u;
                        signs |= (sign_bit << nz_idx);
                        ++nz_idx;
                    }
                }

                // Pack bitmap into meta array: 8 groups per uint32, 4 bits each
                int meta_word = g / kMetaGroupsPerWord;
                int meta_shift = (g % kMetaGroupsPerWord) * 4;
                t.meta_[r * m_stride + meta_word] |= (bitmap << meta_shift);

                // Pack signs into values array: 16 groups per uint32, 2 bits each
                int val_word = g / kValuesGroupsPerWord;
                int val_shift = (g % kValuesGroupsPerWord) * 2;
                t.values_[r * v_stride + val_word] |= (signs << val_shift);
            }
        }
        return t;
    }

    // -----------------------------------------------------------------------
    // Pack with automatic 2:4 pruning from latent (pre-quantization) floats.
    //
    // For each group of 4:
    //   1. Find the 2 positions with largest |latent_weights[i]|
    //      (tie-break: prefer lower index)
    //   2. Quantize those to ternary (>0.5 -> +1, <-0.5 -> -1, else 0)
    //   3. The other 2 positions are forced to zero.
    //
    // Note: pruning is done on latent (float) magnitudes, NOT on the
    // quantized ternary values, to avoid tie-breaking issues
    // (Sparse-BitNet paper, Section 3.2).
    // -----------------------------------------------------------------------
    static SparseTernaryTensor pack_with_pruning(const float* latent_weights,
                                                  int rows, int cols) {
        assert(cols % kSparseGroupSize == 0);
        SparseTernaryTensor t(rows, cols);
        const int groups_per_row = cols / kSparseGroupSize;
        const size_t m_stride = t.meta_row_stride();
        const size_t v_stride = t.values_row_stride();

        // Index array for sorting within each group of 4
        int indices[kSparseGroupSize];

        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < groups_per_row; ++g) {
                int base = r * cols + g * kSparseGroupSize;

                // Initialize indices [0, 1, 2, 3]
                for (int i = 0; i < kSparseGroupSize; ++i)
                    indices[i] = i;

                // Sort by descending |latent_weights|, then ascending index
                // for tie-breaking (stable sort preserves original order for
                // equal magnitudes, giving lower-index preference).
                std::stable_sort(indices, indices + kSparseGroupSize,
                    [&](int a, int b) {
                        return std::fabs(latent_weights[base + a]) >
                               std::fabs(latent_weights[base + b]);
                    });

                // The top-2 indices are kept; build a keep set.
                uint8_t keep[kSparseGroupSize] = {};
                keep[indices[0]] = 1;
                keep[indices[1]] = 1;

                // Build 4-bit position bitmap and collect sign bits.
                uint32_t bitmap = 0;
                uint32_t signs = 0;
                int nz_idx = 0;

                for (int i = 0; i < kSparseGroupSize; ++i) {
                    if (keep[i]) {
                        bitmap |= (1u << i);

                        // Quantize: same threshold as encode_ternary
                        float val = latent_weights[base + i];
                        // sign_bit: 0 = +1 (or zero), 1 = -1
                        uint32_t sign_bit;
                        if (val < -0.5f)
                            sign_bit = 1u;  // -1
                        else
                            sign_bit = 0u;  // +1 or 0 (treated as +1)

                        signs |= (sign_bit << nz_idx);
                        ++nz_idx;
                    }
                }

                // Pack bitmap into meta array
                int meta_word = g / kMetaGroupsPerWord;
                int meta_shift = (g % kMetaGroupsPerWord) * 4;
                t.meta_[r * m_stride + meta_word] |= (bitmap << meta_shift);

                // Pack signs into values array
                int val_word = g / kValuesGroupsPerWord;
                int val_shift = (g % kValuesGroupsPerWord) * 2;
                t.values_[r * v_stride + val_word] |= (signs << val_shift);
            }
        }
        return t;
    }

    // -----------------------------------------------------------------------
    // Unpack back to dense int8 array (row-major, rows * cols elements).
    // Zeros are restored at pruned positions.
    // -----------------------------------------------------------------------
    void unpack_to_int8(int8_t* out) const {
        const int groups_per_row = cols_ / kSparseGroupSize;
        const size_t m_stride = meta_row_stride();
        const size_t v_stride = values_row_stride();

        // Zero-initialize output (pruned positions remain zero)
        std::memset(out, 0, static_cast<size_t>(rows_) * cols_ * sizeof(int8_t));

        for (int r = 0; r < rows_; ++r) {
            for (int g = 0; g < groups_per_row; ++g) {
                int base = r * cols_ + g * kSparseGroupSize;

                // Extract 4-bit bitmap for this group
                int meta_word = g / kMetaGroupsPerWord;
                int meta_shift = (g % kMetaGroupsPerWord) * 4;
                uint32_t bitmap = (meta_[r * m_stride + meta_word] >> meta_shift) & 0xF;

                // Extract 2-bit signs for this group
                int val_word = g / kValuesGroupsPerWord;
                int val_shift = (g % kValuesGroupsPerWord) * 2;
                uint32_t signs = (values_[r * v_stride + val_word] >> val_shift) & 0x3;

                // Reconstruct: iterate positions, assign signs to non-zeros
                int nz_idx = 0;
                for (int i = 0; i < kSparseGroupSize; ++i) {
                    if (bitmap & (1u << i)) {
                        uint32_t sign_bit = (signs >> nz_idx) & 1u;
                        out[base + i] = sign_bit ? static_cast<int8_t>(-1)
                                                 : static_cast<int8_t>(+1);
                        ++nz_idx;
                    }
                    // else: already zero from memset
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    const uint32_t* meta_data() const { return meta_.data(); }
    const uint32_t* values_data() const { return values_.data(); }
    size_t meta_row_stride() const { return meta_words_per_row(cols_); }
    size_t values_row_stride() const { return values_words_per_row(cols_); }
    size_t meta_size() const { return meta_.size(); }
    size_t values_size() const { return values_.size(); }

private:
    int rows_;
    int cols_;
    std::vector<uint32_t> meta_;
    std::vector<uint32_t> values_;

    // Number of uint32_t words needed per row in meta array.
    // ceil(groups_per_row / 8), where groups_per_row = cols / 4.
    static size_t meta_words_per_row(int cols) {
        size_t groups = static_cast<size_t>(cols) / kSparseGroupSize;
        return (groups + kMetaGroupsPerWord - 1) / kMetaGroupsPerWord;
    }

    // Number of uint32_t words needed per row in values array.
    // ceil(groups_per_row / 16), where groups_per_row = cols / 4.
    static size_t values_words_per_row(int cols) {
        size_t groups = static_cast<size_t>(cols) / kSparseGroupSize;
        return (groups + kValuesGroupsPerWord - 1) / kValuesGroupsPerWord;
    }
};

} // namespace spbitnet
