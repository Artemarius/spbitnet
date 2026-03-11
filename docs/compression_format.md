# Sparse Ternary Weight Storage Format — Specification

**Project**: spbitnet
**Version**: Format version 1 (`FORMAT_VERSION = 1`)
**Status**: Implemented and in production use for BitNet-2B-4T inference

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dense Ternary Format](#2-dense-ternary-format)
3. [Sparse Ternary Format (2:4)](#3-sparse-ternary-format-24)
4. [On-Disk Binary Format](#4-on-disk-binary-format)
5. [Sparsity Mask Generation](#5-sparsity-mask-generation)
6. [Absmean Quantization](#6-absmean-quantization)

---

## 1. Overview

Weight tensors in neural network inference can be stored at different numeric precisions. Each step in the hierarchy below trades representational fidelity for reduced memory bandwidth and storage.

### 1.1 Representation Hierarchy

| Format | Bits per weight | Relative size | Notes |
|---|---|---|---|
| FP16 | 16 | 1.00x (baseline) | Standard half-precision floating point |
| INT8 | 8 | 0.50x | Uniform 8-bit quantization |
| Dense ternary | 2 | 0.125x | Values in {-1, 0, +1}, 2-bit codes packed 16 per uint32 |
| Sparse ternary (2:4) | 1.5 | 0.094x | Ternary with exactly 2 non-zeros per group of 4; separated meta + values arrays |

The sparse ternary format achieves 1.5 bits per weight as follows: each group of 4 weights requires a 4-bit position bitmap (1 bit/weight) and a 2-bit sign pair (0.5 bits/weight), totaling 1.5 bits/weight. The two zero-valued weights in each group consume no storage.

### 1.2 Compression Ratios Relative to FP16

- Dense ternary: 8x reduction in weight storage
- Sparse ternary (2:4): 10.67x reduction in weight storage

These ratios apply to weight storage only. Activations remain in FP16 or INT8 during inference.

---

## 2. Dense Ternary Format

The dense ternary format is used internally and as an intermediate representation during conversion. It is defined in `include/spbitnet/ternary_tensor.h`.

### 2.1 Value Encoding

Each weight value is encoded as a 2-bit code:

| Code (binary) | Code (hex) | Weight value |
|---|---|---|
| `00` | `0x0` | 0 |
| `01` | `0x1` | +1 |
| `10` | `0x2` | -1 |
| `11` | `0x3` | Reserved — must not appear |

### 2.2 Packing Layout

Sixteen weight codes are packed into each `uint32_t` word in LSB-first order:

- Weight at column index `c` within a row occupies bits `[2c+1 : 2c]` of word `c / 16`.
- Within a word, weight 0 occupies bits `[1:0]`, weight 1 occupies bits `[3:2]`, ..., weight 15 occupies bits `[31:30]`.

```
uint32_t word layout (dense ternary, 16 weights):

bit:  31 30  29 28  27 26  ...  3  2   1  0
      [w15]  [w14]  [w13]  ... [w1]   [w0]

Each [wN] is a 2-bit code: 00=0, 01=+1, 10=-1
```

### 2.3 Row Stride

The number of `uint32_t` words per row is:

```
words_per_row = ceil(cols / 16)
```

For a matrix with `rows` rows and `cols` columns, total storage is:

```
total_bytes = rows * ceil(cols / 16) * 4
```

### 2.4 Threshold Quantization

When encoding from float:

- `val > 0.5f`  → code `01` (+1)
- `val < -0.5f` → code `10` (-1)
- otherwise     → code `00` (0)

---

## 3. Sparse Ternary Format (2:4)

This is the primary inference format. It is defined in `include/spbitnet/sparse_ternary_tensor.h` and produced by `python/convert_model.py`.

### 3.1 The 2:4 Sparsity Constraint

The weight matrix is divided into groups of 4 consecutive elements along each row. Each group must contain exactly 2 non-zero values and exactly 2 zeros. This pattern is called 2:4 structured sparsity.

The constraint is: for every row `r` and every group index `g = 0, 1, ..., (cols/4 - 1)`, the 4 weights at columns `4g, 4g+1, 4g+2, 4g+3` have exactly 2 values from {-1, +1} and exactly 2 zeros.

The 2:4 pattern is the same sparsity structure supported natively by NVIDIA Sparse Tensor Cores (Ampere and later). The spbitnet custom kernel exploits this pattern through the separated arrays described below.

### 3.2 Separated Arrays Design

The format uses two distinct arrays rather than interleaving position and sign data:

- **Meta array**: Stores position information (which 2 of the 4 positions are non-zero) for all groups.
- **Values array**: Stores sign information (+1 or -1) for the non-zero weights of all groups.

The rationale is GPU memory coalescing. During a matrix-vector product, the GPU reads entire rows of the meta and values arrays in contiguous cache lines, rather than alternating between heterogeneous field types within a packed record. This access pattern is suited to bandwidth-bound workloads.

### 3.3 Meta Array

The meta array stores one 4-bit position bitmap per group.

**Bitmap semantics**: For a group at column positions `[4g, 4g+1, 4g+2, 4g+3]`, the bitmap is a 4-bit value where bit `i` (0-indexed from the LSB) is 1 if and only if position `i` within the group is non-zero. Exactly 2 bits are set in every valid bitmap.

**Valid bitmap patterns**: There are C(4,2) = 6 valid patterns out of 16 possible 4-bit values:

| Bitmap (binary) | Hex | Non-zero positions |
|---|---|---|
| `0011` | `0x3` | 0, 1 |
| `0101` | `0x5` | 0, 2 |
| `0110` | `0x6` | 1, 2 |
| `1001` | `0x9` | 0, 3 |
| `1010` | `0xA` | 1, 3 |
| `1100` | `0xC` | 2, 3 |

All other 4-bit values are invalid in a well-formed meta array.

**Packing**: Eight 4-bit bitmaps are packed into each `uint32_t` word in LSB-first order. Group `g` within a row is stored as:

```
word index  = g / 8
bit offset  = (g % 8) * 4

extraction: bitmap = (meta[row * meta_stride + word_index] >> bit_offset) & 0xF
```

**Meta array bit layout (one uint32_t word, 8 groups)**:

```
bit:  31..28  27..24  23..20  19..16  15..12  11..8   7..4    3..0
      [g7]    [g6]    [g5]    [g4]    [g3]    [g2]    [g1]    [g0]

Each [gN] is a 4-bit bitmap with exactly 2 bits set.
```

**Row stride** (in `uint32_t` words):

```
groups_per_row = cols / 4
meta_stride    = ceil(groups_per_row / 8)
```

### 3.4 Values Array

The values array stores one 2-bit sign pair per group.

**Sign pair semantics**: For a group, the two non-zero positions (identified by the meta bitmap) are ordered by ascending column position within the group — that is, the one at the lower column index is "first" and the one at the higher column index is "second". The 2-bit sign pair encodes:

- Bit 0 (`sign_lo`): sign of the first (lower-position) non-zero. `0` = +1, `1` = -1.
- Bit 1 (`sign_hi`): sign of the second (higher-position) non-zero. `0` = +1, `1` = -1.

Sign encoding uses inverted convention: bit value `0` means positive (+1), bit value `1` means negative (-1). Zero cannot appear in the values array at positions marked non-zero by the meta bitmap; see Section 5.3.

**Packing**: Sixteen 2-bit sign pairs are packed into each `uint32_t` word in LSB-first order. Group `g` within a row is stored as:

```
word index  = g / 16
bit offset  = (g % 16) * 2

extraction: signs = (values[row * values_stride + word_index] >> bit_offset) & 0x3
```

**Values array bit layout (one uint32_t word, 16 groups)**:

```
bit:  31 30  29 28  27 26  ...  3  2   1  0
      [g15]  [g14]  [g13]  ... [g1]   [g0]

Each [gN] is a 2-bit field: bit 0 = sign_lo, bit 1 = sign_hi
Sign convention: 0 = +1, 1 = -1
```

**Row stride** (in `uint32_t` words):

```
groups_per_row  = cols / 4
values_stride   = ceil(groups_per_row / 16)
```

### 3.5 Storage Summary and Size Formulas

For a weight matrix of shape `(rows, cols)` where `cols` is a multiple of 4:

```
groups_per_row  = cols / 4
meta_stride     = ceil(groups_per_row / 8)     [uint32_t words per row]
values_stride   = ceil(groups_per_row / 16)    [uint32_t words per row]

meta_bytes      = rows * meta_stride * 4
values_bytes    = rows * values_stride * 4
total_bytes     = meta_bytes + values_bytes
```

**Verification** (that this equals 1.5 bits/weight for cols divisible by 32):

When `cols` is a multiple of 32, `groups_per_row = cols/4` is a multiple of 8, so:

```
meta_stride   = cols / 32    (no rounding)
values_stride = cols / 64    (no rounding)
total_bits    = rows * (cols/32 * 32 + cols/64 * 32)
              = rows * cols * (1 + 0.5)
              = rows * cols * 1.5
```

### 3.6 Full Bit Layout Diagram

The following diagram shows how a single group of 4 weights is represented across both arrays. The example has non-zero weights at positions 1 and 3 (bitmap `1010`), with the weight at position 1 being +1 and the weight at position 3 being -1.

```
Group of 4 weights: [0, +1, 0, -1]
                      ^   ^   ^   ^
                      p0  p1  p2  p3

Step 1 — Bitmap (meta):
  Position 0 is zero  -> bit 0 = 0
  Position 1 is +1    -> bit 1 = 1
  Position 2 is zero  -> bit 2 = 0
  Position 3 is -1    -> bit 3 = 1
  bitmap = 0b1010 = 0xA

Step 2 — Sign pair (values):
  Non-zeros in ascending position order: p1 (+1), p3 (-1)
  sign_lo = sign of p1 = +1  -> bit value 0
  sign_hi = sign of p3 = -1  -> bit value 1
  signs = 0b10 = 0x2

Meta word contribution (if this is group 0):
  bits [3:0] = 0b1010

Values word contribution (if this is group 0):
  bits [1:0] = 0b10

Reconstruction (unpack):
  bitmap = 0xA = 0b1010    -> non-zeros at positions 1 and 3
  signs  = 0x2 = 0b10
    sign_lo (bit 0) = 0    -> position 1 is +1
    sign_hi (bit 1) = 1    -> position 3 is -1
  Result: [0, +1, 0, -1]   (correct)
```

### 3.7 Constraint: Column Count Must Be a Multiple of 4

The `SparseTernaryTensor` constructor asserts that `cols % 4 == 0`. The conversion script pads any weight matrix whose column count is not a multiple of 4 before packing.

---

## 4. On-Disk Binary Format

### 4.1 Directory Structure

A converted model is stored as a directory with the following structure:

```
<model_dir>/
    config.json          # Model architecture parameters (JSON)
    manifest.json        # Weight index: file paths, shapes, gammas (JSON)
    tokenizer.json       # Tokenizer vocabulary (copied from HuggingFace)
    tokenizer_config.json
    weights/
        embed_tokens.bin              # Embedding table (float16, row-major)
        norm.bin                      # Final RMSNorm scale (float32)
        layers.0.input_layernorm.bin  # Per-layer norm scales (float32)
        layers.0.self_attn.q_proj.meta    # Sparse ternary meta (uint32, row-major)
        layers.0.self_attn.q_proj.values  # Sparse ternary values (uint32, row-major)
        layers.0.self_attn.k_proj.meta
        layers.0.self_attn.k_proj.values
        ...
```

The weights directory uses flat filenames with dots as separators for layer hierarchy (e.g., `layers.0.self_attn.q_proj`). Subdirectories are not used for layer hierarchy because Python's `Path.with_suffix()` does not handle names containing dots correctly; the conversion script uses string concatenation for the `.meta` and `.values` suffixes.

### 4.2 Byte Order

All binary files are written in native little-endian byte order. NVIDIA GPUs and x86/x86-64 CPUs are both little-endian; no byte swapping is required on these platforms.

### 4.3 config.json Schema

Contains the model architecture parameters required by the C++ inference engine. All fields are mandatory unless noted.

```json
{
    "model_type":             "bitnet",
    "hidden_size":            2560,
    "num_hidden_layers":      30,
    "num_attention_heads":    20,
    "num_key_value_heads":    5,
    "intermediate_size":      6912,
    "vocab_size":             128256,
    "max_position_embeddings": 4096,
    "rms_norm_eps":           1e-5,
    "rope_theta":             500000.0,
    "head_dim":               128,
    "hidden_act":             "relu2",
    "tie_word_embeddings":    true,
    "bos_token_id":           128000,
    "eos_token_id":           128001
}
```

`head_dim` is derived from `hidden_size / num_attention_heads` at conversion time and stored explicitly to avoid recomputation. When `tie_word_embeddings` is `true`, the language model head shares weights with `embed_tokens` and no separate `lm_head.bin` file is written.

### 4.4 manifest.json Schema

The manifest is a JSON index over all weight tensors. It contains one entry per tensor, keyed by the export name (e.g., `layers.0.self_attn.q_proj`).

**Top-level structure**:

```json
{
    "format_version": 1,
    "weights": {
        "<tensor_name>": { ... }
    }
}
```

**Entry for a sparse ternary linear layer**:

```json
"layers.0.self_attn.q_proj": {
    "meta_file":      "weights/layers.0.self_attn.q_proj.meta",
    "values_file":    "weights/layers.0.self_attn.q_proj.values",
    "dtype":          "sparse_ternary",
    "shape":          [2560, 2560],
    "meta_stride":    20,
    "values_stride":  10,
    "gamma":          0.003842,
    "meta_nbytes":    204800,
    "values_nbytes":  102400,
    "stats": {
        "nz_before_pruning": 4718532,
        "nz_after_pruning":  3276800,
        "density_before":    0.716,
        "density_after":     0.500
    }
}
```

Field definitions:

| Field | Type | Description |
|---|---|---|
| `meta_file` | string | Path to `.meta` file, relative to model root |
| `values_file` | string | Path to `.values` file, relative to model root |
| `dtype` | string | Always `"sparse_ternary"` for linear layers |
| `shape` | int[2] | `[rows, cols]` of the original weight matrix |
| `meta_stride` | int | `uint32_t` words per row in the meta array |
| `values_stride` | int | `uint32_t` words per row in the values array |
| `gamma` | float | Per-tensor absmean scale factor; see Section 6 |
| `meta_nbytes` | int | Byte size of the `.meta` file |
| `values_nbytes` | int | Byte size of the `.values` file |
| `stats` | object | Informational density statistics; not used at inference |

**Entry for a float16 embedding or lm_head**:

```json
"embed_tokens": {
    "file":    "weights/embed_tokens.bin",
    "dtype":   "float16",
    "shape":   [128256, 2560],
    "nbytes":  655687680
}
```

**Entry for a float32 norm scale**:

```json
"layers.0.input_layernorm": {
    "file":    "weights/layers.0.input_layernorm.bin",
    "dtype":   "float32",
    "shape":   [2560],
    "nbytes":  10240
}
```

**Entry for a tied weight** (when `tie_word_embeddings` is true):

```json
"lm_head": {
    "tied_to": "embed_tokens",
    "dtype":   "tied"
}
```

### 4.5 Binary File Layout

**Sparse ternary `.meta` file**:

```
Byte offset 0: row 0, meta word 0      (uint32_t, 4 bytes)
Byte offset 4: row 0, meta word 1      (uint32_t, 4 bytes)
...
Byte offset meta_stride*4*(row): row N, meta word 0
...
Total bytes = rows * meta_stride * 4
```

**Sparse ternary `.values` file**:

```
Byte offset 0: row 0, values word 0    (uint32_t, 4 bytes)
Byte offset 4: row 0, values word 1    (uint32_t, 4 bytes)
...
Byte offset values_stride*4*(row): row N, values word 0
...
Total bytes = rows * values_stride * 4
```

Both files are written by NumPy's `ndarray.tofile()` from a row-major `(rows, stride)` uint32 array. The resulting layout is equivalent to C's row-major (C-order) array serialization.

**Float16 `.bin` files** (embeddings, lm_head):

Raw row-major IEEE 754 half-precision values. For an embedding table of shape `(vocab_size, hidden_size)`, byte offset of element `[i, j]` is `(i * hidden_size + j) * 2`.

**Float32 `.bin` files** (norm scales):

Raw row-major IEEE 754 single-precision values. For a 1-D scale of shape `(hidden_size,)`, byte offset of element `[j]` is `j * 4`.

---

## 5. Sparsity Mask Generation

### 5.1 Procedure

The 2:4 sparsity mask is generated from the latent (pre-quantization) floating-point weights, not from the quantized ternary values. The procedure is:

1. For each row of the weight matrix, partition the columns into groups of 4.
2. Within each group, compute the absolute value of each element.
3. Select the 2 elements with the largest absolute values. These positions are marked non-zero in the mask; the other 2 are marked zero (forced to zero regardless of their ternary value).
4. Ties in magnitude are broken by preferring the lower column index (stable sort).

This is implemented in `generate_24_mask()` in `python/convert_model.py` and in `SparseTernaryTensor::pack_with_pruning()` in `include/spbitnet/sparse_ternary_tensor.h`.

### 5.2 Why Pruning Uses Latent Weights

The mask is generated from latent (float) weights rather than from the quantized ternary values because ternary quantization maps a range of float values to the same output. If two adjacent weights are, for example, 0.49 and 0.51, their ternary codes are 0 and +1 respectively, making the selection appear straightforward. However, after absmean scaling, the actual magnitudes that determine which weight is more significant are the latent values before rounding.

More directly: applying magnitude selection to ternary values can create tie-breaking ambiguities where zero-valued ternary weights appear to have equal magnitude to non-zero ones. The Sparse-BitNet paper (Zhang et al., Section 3.2) prescribes pruning on latent weights to avoid this class of error.

### 5.3 The Zero-at-Mask Bug

**Problem**: After ternary quantization (`RoundClip(W/gamma, -1, 1)`), a weight whose latent value is close to zero may quantize to 0 even though the magnitude-selection step marked its position as non-zero in the bitmap. When this position is packed into the values array, the format has no way to represent a zero at a bitmap-marked (non-zero) position — bit value `0` encodes +1, not 0.

Without a fix, such a weight silently becomes +1 in the packed representation, introducing a quantization error that is not reflected in the gamma-scaled dequantization.

**Fix**: After applying the mask to the ternary weights, identify any positions where the mask is `True` but the ternary value is `0`. For these positions, force the ternary value to +1 or -1 based on the sign of the original latent weight:

```python
zero_at_mask = mask & (sparse_ternary == 0)
sparse_ternary[zero_at_mask & (weight_f32 >= 0)] = +1
sparse_ternary[zero_at_mask & (weight_f32 < 0)]  = -1
```

This is applied in `export_sparse_layer()` in `python/convert_model.py`. The corresponding C++ path in `SparseTernaryTensor::pack_from_dense()` handles this at the point of sign encoding: if `weights[base + i] == 0` at a masked position, the sign bit is recorded as 0 (encoding +1), which matches the Python convention for non-negative latent values.

This fix must be applied before packing and must be consistent between the Python conversion script and any C++ or Python code that validates the packed output against a reference.

---

## 6. Absmean Quantization

### 6.1 Formula

Linear layer weights are quantized using the absmean method from BitNet b1.58 (Eq. (1)):

```
gamma   = mean(|W|)                          [scalar, per-tensor]
W_q     = RoundClip(W / gamma, -1, 1)       [element-wise]
```

where `RoundClip(x, a, b) = clip(round(x), a, b)`.

The result `W_q` is an integer array with values in {-1, 0, +1}.

### 6.2 Gamma Storage

`gamma` is stored as a 64-bit IEEE 754 double-precision float in `manifest.json` (JSON number). During inference, the C++ engine reads `gamma` for each linear layer from the manifest and uses it to dequantize the integer output of the sparse GEMV back to float16:

```
output_float = output_int32 * gamma * activation_scale
```

where `activation_scale` is the per-token absmax quantization factor for the activation vector (the `absmax / 127` factor from INT8 quantization of the input).

### 6.3 Degenerate Case

If `gamma < 1e-10` (a near-zero weight matrix), the conversion script returns an all-zero ternary tensor and stores `gamma = 0.0` in the manifest. The C++ inference engine must handle this case to avoid division by zero during dequantization.

### 6.4 Precision Notes

- `gamma` is computed in float64 (NumPy default for `np.mean` on float32 input).
- The division `W / gamma` is performed in float32.
- The rounded ternary values are stored as int8.
- `gamma` is serialized to manifest.json as a JSON number, which preserves approximately 15-17 significant decimal digits (double precision).
- The C++ loader reads `gamma` from JSON as a double and stores it as a float; the conversion from double to float introduces at most 1 ULP of error relative to the double value.
