# CUDA Kernel Design: spbitnet

This document describes the design decisions behind the custom CUDA kernels in spbitnet, a C++ inference engine for Sparse-BitNet models. The target hardware is an NVIDIA RTX 3060 Laptop GPU: Compute Capability 8.6, 30 SMs, 336 GB/s memory bandwidth, 3 MB L2 cache.

All kernel source is in `src/kernels/sparse_ternary.cu` and `src/kernels/inference_kernels.cu`. Host interfaces are in `include/spbitnet/sparse_ternary_kernels.h` and `include/spbitnet/inference_kernels.h`. Forward pass orchestration is in `src/inference.cu`.

---

## 1. The BitLinear Pipeline

Every linear projection in a BitNet model — Q, K, V, O, gate, up, and down — is a BitLinear layer. A BitLinear layer computes:

```
output = dequantize(W_ternary * quantize(input))
```

where `W_ternary` holds values in `{-1, 0, +1}` and `quantize` maps the float16 input vector to INT8 via absmax scaling.

The pipeline has three stages:

1. **Absmax quantization.** Compute `absmax = max(|x|)`, then `q[i] = round(x[i] * 127 / absmax)`, clamped to `[-128, 127]`. This produces an INT8 vector and a scalar scale factor `absmax` that is written to a device pointer for the dequantize step.

2. **Sparse ternary GEMV.** Multiply the INT8 input vector against the compressed sparse ternary weight matrix, accumulating into INT32. Because weights are in `{-1, 0, +1}`, each multiply reduces to a conditional add or subtract. Because weights have 2:4 structured sparsity, exactly half the elements in every group of 4 are zero and are skipped entirely.

3. **Dequantize.** Convert INT32 accumulators back to float16: `out[i] = acc[i] * gamma * absmax / 127`, where `gamma` is a per-tensor scale stored in the model manifest during conversion.

**Why not cuBLAS?** cuBLAS operates on float16 or float32 matrices. Storing BitNet weights as float16 would expand them from ~1 bit/weight to 16 bits/weight, consuming roughly 16x more memory bandwidth per GEMV. The ternary-specific arithmetic also cannot be exploited in a general-purpose GEMM.

**Why not cuSPARSELt?** cuSPARSELt's SpMMA instruction requires the `n` dimension (output columns) to be at least 16, which reflects the minimum Tensor Core tile width for INT8 SpMMA on Ampere. Autoregressive decode processes one token at a time, so `n = 1` always. cuSPARSELt cannot handle this case. See Section 5 for the full comparison.

The custom path also exploits the ternary constraint specifically: weight values are known to be `{-1, 0, +1}` at kernel compile time, so no integer multiplication is issued — the accumulation is a conditional add/subtract per non-zero element.

---

## 2. Sparse Ternary GEMV Kernel

**Source:** `src/kernels/sparse_ternary.cu`, function `sparse_ternary_gemv_kernel`.

### 2.1 Memory Layout

Weights are stored in two separate arrays per layer: `meta` and `values`. Both are row-major, with one row of data per weight matrix row.

**Meta array** encodes the non-zero positions. For every group of 4 consecutive weights, exactly 2 are non-zero (2:4 sparsity). The positions of those 2 non-zeros within the group are stored as a 4-bit bitmap. Eight groups pack into one `uint32_t`, so the meta stride per row is `ceil(cols / 4 / 8)` words.

**Values array** encodes the signs. Each group contributes 2 bits: bit 0 is the sign of the lower-positioned non-zero, bit 1 is the sign of the higher-positioned non-zero. `0` means `+1`, `1` means `-1`. Sixteen groups pack into one `uint32_t`, so the values stride per row is `ceil(cols / 4 / 16)` words.

This separation keeps meta reads and values reads independent, avoids packing overhead at access time, and allows each warp to read contiguous words from both arrays without interleaving.

### 2.2 LUT-Based Bitmap Decode

The 4-bit bitmap has only 6 valid states (C(4,2) = 6 ways to choose 2 positions from 4). Rather than computing the two non-zero positions via bit-scan instructions at runtime, a 16-entry lookup table maps each bitmap to a pre-decoded `(p0, p1)` pair:

```cpp
struct Pos2 { int8_t p0; int8_t p1; };

__device__ __constant__ Pos2 kBitmapToPos[16] = {
    {0, 0},  //  0 = 0b0000 — invalid
    ...
    {0, 1},  //  3 = 0b0011
    {0, 2},  //  5 = 0b0101
    {1, 2},  //  6 = 0b0110
    {0, 3},  //  9 = 0b1001
    {1, 3},  // 10 = 0b1010
    {2, 3},  // 12 = 0b1100
    ...
};
```

The table lives in `__constant__` memory, which is cached in L1 on Ampere. Because all threads in a warp read the same table at the same time (different bitmaps but the same 16-entry address space), the access pattern is broadcast-friendly. The decode reduces to a single indexed load with no branch.

Invalid entries (fewer or more than 2 bits set) map to `{0, 0}`. A well-formed 2:4 matrix never produces an invalid bitmap during inference.

### 2.3 Warp-Per-Row Parallelism

Each warp (32 threads) is responsible for one output row. Thread `lane` (0–31) processes groups `g = lane, lane+32, lane+64, ...`, covering all `cols/4` groups in the row with a stride-32 loop:

```cpp
for (int g = lane; g < groups_per_row; g += 32) {
    const uint32_t bitmap = (meta_row[g / 8] >> ((g % 8) * 4)) & 0xF;
    const uint32_t signs  = (vals_row[g / 16] >> ((g % 16) * 2)) & 0x3;
    const Pos2 pos = kBitmapToPos[bitmap];
    const int base_col = g * 4;

    const int32_t x0 = static_cast<int32_t>(x[base_col + pos.p0]);
    const int32_t x1 = static_cast<int32_t>(x[base_col + pos.p1]);

    acc += (signs & 1u) ? -x0 : x0;
    acc += (signs & 2u) ? -x1 : x1;
}
```

The ternary sign check `(signs & 1u) ? -x0 : x0` is branchless: the compiler emits a conditional negate, which on Ampere resolves without divergence overhead when the condition varies across lanes (which it will for typical weight distributions).

### 2.4 Warp Reduction

After the group loop, each lane holds a partial sum. A standard `__shfl_down_sync` butterfly reduction produces the full dot product in lane 0:

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

if (lane == 0) y[row] = acc;
```

Five shuffle instructions cover a 32-lane reduction. No shared memory is required, which avoids `__syncthreads` overhead and reduces register pressure.

### 2.5 Launch Configuration

The kernel uses 256 threads per block (8 warps), so 8 rows are computed per block. The grid size is `ceil(rows / 8)` blocks. For BitNet-2B-4T, the largest weight matrices are 6912 × 2560 (gate/up projections), requiring `ceil(6912 / 8) = 864` blocks — well within the SM dispatch capacity of an RTX 3060.

### 2.6 Input Scatter Pattern and Bandwidth

The non-zero positions `pos.p0` and `pos.p1` within each group of 4 are data-dependent: they come from the bitmap, which varies by group and by weight matrix row. Threads in the same warp therefore issue non-coalesced loads from the input vector `x`. At a row width of 2560 (the hidden dimension), each warp reads 2 × 2560/4 = 1280 INT8 values scattered across `[0, 2560)`. These loads are not contiguous, so they do not coalesce into full cache lines and result in partial utilization of L2 bandwidth.

This is the primary reason the BitLinear kernels achieve roughly 18% of peak memory bandwidth (61 GB/s of 336 GB/s measured) rather than approaching the hardware ceiling. The fundamental constraint is the sparse gather pattern, not kernel overhead. See Section 6 for further discussion.

---

## 3. Fused BitLinear Kernel

**Source:** `src/kernels/sparse_ternary.cu`, function `fused_sparse_bitlinear_kernel`; `src/kernels/inference_kernels.cu`, function `absmax_reduce_kernel`.

### 3.1 Motivation

The unfused BitLinear pipeline requires three kernel launches per layer:

1. `absmax_quantize_kernel` — reads float16 input, writes INT8 output and a float scalar.
2. `sparse_ternary_gemv_kernel` — reads INT8 input and weight arrays, writes INT32 output.
3. `dequantize_kernel` — reads INT32 input and the float scalar, writes float16 output.

Between kernels 1 and 2, the full INT8 activation vector (`d_quant_`, up to 6912 bytes for intermediate MLP dimension) must be written to DRAM and read back. Between kernels 2 and 3, the full INT32 accumulator vector (`d_int_out_`) must do the same. These are unnecessary round-trips: all the data a warp needs for quantization is the same data it needs for the GEMV.

### 3.2 Two-Kernel Fused Approach

The absmax value must be known before quantization can begin, and computing it requires a reduction over the entire input vector. That reduction cannot be fused into the per-row GEMV kernel without a global barrier. Two kernels therefore remain:

- `absmax_reduce_gpu` — reduction only, no quantization. Writes `absmax` to a single device float.
- `fused_sparse_bitlinear_gpu` — reads float16 input directly, quantizes each element inline, accumulates into INT32, then dequantizes and writes float16 output.

The `InferenceEngine::bitlinear()` method in `src/inference.cu` executes this sequence:

```cpp
absmax_reduce_gpu(input, d_absmax_, input_size);
fused_sparse_bitlinear_gpu(layer.d_meta, layer.d_values,
                            input, output,
                            d_absmax_, layer.gamma,
                            layer.rows, layer.cols,
                            layer.meta_stride, layer.values_stride);
```

Both kernels execute on the same stream. CUDA stream ordering guarantees that `fused_sparse_bitlinear_gpu` reads `d_absmax_` only after `absmax_reduce_gpu` has written it.

### 3.3 Inline Quantization

Inside `fused_sparse_bitlinear_kernel`, each thread reads float16 values from the input vector at the non-zero positions, converts them to int32 inline, and accumulates:

```cpp
const float absmax = *d_absmax;
const float quant_scale = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;

// ...for each group g:
float f0 = __half2float(x[base_col + pos.p0]) * quant_scale;
float f1 = __half2float(x[base_col + pos.p1]) * quant_scale;
int32_t x0 = max(-128, min(127, __float2int_rn(f0)));
int32_t x1 = max(-128, min(127, __float2int_rn(f1)));
```

Because the GEMV kernel only accesses the two non-zero positions per group, the inline quantization path processes exactly the same elements the unfused path would, with no redundant computation. The INT8 `d_quant_` buffer is not written.

### 3.4 Inline Dequantization

Lane 0 of each warp writes the final dequantized result directly to the float16 output:

```cpp
if (lane == 0) {
    float dequant_scale = gamma * absmax / 127.0f;
    output[row] = __float2half(static_cast<float>(acc) * dequant_scale);
}
```

The INT32 `d_int_out_` buffer is not written (it is repurposed for the float32 MLP intermediate path; see Section 4.4).

### 3.5 Measured Speedup

Profiled on BitNet-2B-4T with a 128-token sequence on the RTX 3060 Laptop. The unfused path produced approximately 50 tokens/second under load. The fused path produces 58.3 tokens/second in clean benchmark runs — a 15.6% improvement. The gain comes from three sources: one fewer kernel launch per BitLinear call (saving roughly 210 launches per token across 30 layers × 7 BitLinear calls per layer), elimination of the `d_quant_` write-back, and elimination of the `d_int_out_` write-back.

---

## 4. Inference Kernels

**Source:** `src/kernels/inference_kernels.cu`.

### 4.1 RMSNorm

RMSNorm computes `y[i] = x[i] * w[i] * rsqrt(mean(x²) + eps)`.

The kernel is single-block (launched as `<<<1, 256>>>`). It processes vectors of up to the hidden dimension (2560 for BitNet-2B-4T) — comfortably within 256 threads with a strided loop.

All arithmetic is in float32 despite float16 input and output. Float16 has insufficient range to accumulate squared values for a 2560-element vector without risking overflow or significant rounding error.

The reduction pattern is two-stage. Each warp reduces its local sum to a single value via `__shfl_down_sync`, storing the warp result in shared memory (`shared_sum[8]`, one entry per warp for a 256-thread block with 8 warps). The first warp then reduces across those 8 values using a second shuffle pass. This avoids a full `__syncthreads` barrier per shuffle step:

```cpp
// Warp reduction
for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

if (lane == 0) shared_sum[warp] = local_sum;
__syncthreads();

// Cross-warp reduction (first warp only)
if (warp == 0) {
    float val = (lane < num_warps) ? shared_sum[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    if (lane == 0) shared_sum[0] = val;
}
__syncthreads();
```

One `__syncthreads` follows the warp-level writes, and one follows the cross-warp reduction. The same two-stage pattern appears in `absmax_quantize_kernel`, `absmax_reduce_kernel`, `softmax_kernel`, and `rms_norm_f32in_kernel`.

BitNet-2B-4T applies two sub-layer normalizations (SubLN) in addition to the standard pre-attention and pre-FFN norms: `attn_sub_norm` after the output projection, and `ffn_sub_norm` after the ReLU² activation. This results in 4 RMSNorm calls per transformer layer, accounting for 8.9% of total forward pass time as measured by the Profiler class.

### 4.2 RoPE

The RoPE kernel applies rotary positional embeddings in-place to a `(num_heads, head_dim)` float16 vector. One thread handles one dimension pair `(2i, 2i+1)`:

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int head = tid / (head_dim / 2);
int pair = tid % (head_dim / 2);

float freq  = 1.0f / powf(theta, 2.0f * pair / head_dim);
float angle = pos * freq;
float cos_a = cosf(angle);
float sin_a = sinf(angle);

int idx = head * head_dim + 2 * pair;
float x0 = __half2float(vec[idx]);
float x1 = __half2float(vec[idx + 1]);
vec[idx]     = __float2half(x0 * cos_a - x1 * sin_a);
vec[idx + 1] = __float2half(x0 * sin_a + x1 * cos_a);
```

For BitNet-2B-4T: 20 Q heads × 64 pairs/head = 1280 threads for Q, and 5 KV heads × 64 pairs = 320 threads for K. Both fit comfortably in a single block dispatch.

RoPE is called twice per layer: once for Q (20 heads × 128 head_dim) and once for K (5 KV heads × 128 head_dim), applied at the current sequence position `pos`.

### 4.3 GQA Attention

BitNet-2B-4T uses Grouped Query Attention with 20 Q heads and 5 KV heads (4 Q heads per KV head). The attention score kernel maps each Q head to its KV head with integer arithmetic:

```cpp
const int kv_head = head * num_kv_heads / num_heads;
```

**Attention scores** (`attention_scores_kernel`): the grid is `(ceil(seq_len / warps_per_block), num_heads)`. Each warp computes the dot product of one Q head against one K position. Lanes stride over head_dim with stride 32, accumulating in float32. After warp reduction, lane 0 writes `score[head][j] = acc * scale` where `scale = 1 / sqrt(head_dim)`.

**Softmax** (`softmax_kernel`): one block per Q head, 256 threads. Three passes over the `seq_len` scores per head: find max (for numerical stability), compute exp and sum, normalize. Uses the same two-stage warp + shared memory reduction as RMSNorm.

**Attention output** (`attention_output_kernel`): one block per Q head, `head_dim` threads. Each thread accumulates `sum_j score[head][j] * V_cache[kv_h, j, d]` over all sequence positions. This is a sequential loop over `seq_len` per thread, which becomes the dominant attention cost for long sequences. For BitNet-2B-4T decode (short sequence context typical), the cost is small relative to the BitLinear GEMVs.

The KV cache layout is `(num_kv_heads, max_seq_len, head_dim)` float16 per layer. Storing keys and values separately (not interleaved) allows the attention score and output kernels to read contiguous strides per head without cross-head interference.

### 4.4 ReLU² Overflow Fix

The BitNet-2B-4T MLP block uses gated activation with ReLU²:

```
hidden = relu²(gate_proj(x)) * up_proj(x)
```

`relu²(x) = max(0, x)²`. For float16, the maximum representable value is 65504. If a gate activation after dequantization is around 256, then `relu²(256) = 65536`, which overflows float16 before the subsequent `ffn_sub_norm` can bring it back into range.

The fix uses a float32 intermediate. `relu2_mul_f32_kernel` computes `relu²(gate) * up` in float32 and writes a float32 output, stored in the `d_int_out_` scratch buffer (reinterpreted as `float*`; it is idle at this point in the forward pass):

```cpp
float* d_mlp_f32 = reinterpret_cast<float*>(d_int_out_);
relu2_mul_f32_gpu(d_gate_, d_up_, d_mlp_f32, inter);
```

`rms_norm_f32in_kernel` then reads float32, normalizes, and writes float16. The kernel is identical in structure to `rms_norm_kernel` but takes `float*` input, avoiding the half-to-float conversion in the inner loop.

The `relu2_mul_kernel` (float16 output) is retained in the codebase for models that do not have SubLN after the gate activation, where overflow does not occur.

### 4.5 LM Head: Vectorized half_gemv

The LM head projects the final hidden state (2560-element float16) to vocabulary logits (128256 outputs). This is a 128256 × 2560 GEMV — by far the largest single GEMV in the model — and accounts for 12.6% of forward pass time.

The kernel uses float4 vectorized loads (16 bytes = 8 float16 values per instruction) to reduce load instruction count eightfold:

```cpp
const float4* row_v = reinterpret_cast<const float4*>(row);
const float4* x_v   = reinterpret_cast<const float4*>(x);

for (int j = lane; j < cols_v; j += 32) {
    float4 w4 = row_v[j];
    float4 x4 = x_v[j];
    const half* wp = reinterpret_cast<const half*>(&w4);
    const half* xp = reinterpret_cast<const half*>(&x4);
    for (int k = 0; k < 8; ++k)
        acc += __half2float(wp[k]) * __half2float(xp[k]);
}
```

At 128256 rows × 2560 cols × 2 bytes = 656 MB of weight data per token, `half_gemv` reaches 92.7% of the L2/DRAM bandwidth available to a sequential stream. The high efficiency is due to near-perfect coalescing: each warp reads 32 contiguous float4 chunks from both the weight row and the input vector on each iteration.

The accumulator `y` is float32 to avoid precision loss when summing 2560 products. A separate pass converts to float16 is not needed — `d_logits_` remains float32 for the argmax/sampling step on the host.

---

## 5. cuSPARSELt Comparison

`CuSparseLtGemm` in `src/cusparselt_backend.cu` wraps the cuSPARSELt API for 2:4 sparse INT8 GEMM. The class manages handle, matrix descriptors, matmul plan, compressed weight storage, and workspace allocation via RAII. The constructor enforces the `n >= 16` constraint with an explicit check:

```cpp
if (n < 16) {
    fprintf(stderr, "CuSparseLtGemm: n=%d but cuSPARSELt INT8 requires n >= 16 "
            "(Tensor Core tile constraint). Use sparse_ternary_gemv_gpu for GEMV.\n", n);
    exit(EXIT_FAILURE);
}
```

This constraint is architectural, not a library limitation. INT8 SpMMA on Ampere Tensor Cores operates on tiles of size 16 × 16 × 32 (M × N × K). The N dimension corresponds to the output column count. A GEMV with N=1 cannot fill an N=16 tile. cuSPARSELt has no fallback path for N < 16.

**When cuSPARSELt applies:** batch inference where multiple tokens are processed simultaneously (prefill with batch size ≥ 16, or batched decode). In that case, the weight matrix is multiplied against a matrix of activations rather than a single vector, and N equals the batch size. On our target hardware and use case (autoregressive decode, one token per forward pass), N is always 1.

**Weight representation difference:** cuSPARSELt operates on INT8 `{-128, ..., +127}`, with the 2:4 pattern applied to INT8 values. It cannot exploit the ternary constraint (`{-1, 0, +1}`) to eliminate multiplications — Tensor Cores execute integer multiply-accumulate regardless. The custom ternary kernel replaces every multiply with a conditional add/subtract at the PTX level.

**What cuSPARSELt provides that the custom kernel does not:** hardware SpMMA via dedicated Sparse Tensor Cores, which can deliver up to 2x throughput relative to dense Tensor Cores for batch GEMMs. For future work on batched inference, the cuSPARSELt path would be the appropriate choice.

The `prepare()` call prunes a dense INT8 weight matrix to 2:4 using cuSPARSELt's magnitude-based strip algorithm and stores the compressed format internally. This is used in the benchmark suite to establish a baseline for hardware-accelerated sparse GEMM performance, but it is not called during inference.

---

## 6. Rejected Optimizations and Bandwidth Analysis

### 6.1 Shared Memory Preload for the Input Vector

The input vector `x` is read multiple times inside `sparse_ternary_gemv_kernel` — once per group assigned to a lane, and at non-contiguous positions determined by the bitmap. An obvious candidate optimization is to load `x` into shared memory once per block (covering all rows assigned to the block) and have all warps read from there.

This was tested and rejected. The `__syncthreads` barrier required to ensure all threads have finished loading before any warp begins its GEMV would stall all warps in the block. For a 256-thread block (8 warps) processing 8 rows, the barrier overhead exceeded the bandwidth saved by eliminating the repeated L2 reads. The input vector at 2560 elements × 1 byte = 2560 bytes fits entirely within the 128 KB shared L1 cache on Ampere, so repeated accesses are already served from L1 after the first pass. Adding a barrier to force explicit preloading into shared memory introduces synchronization cost without a corresponding cache miss reduction.

### 6.2 Bandwidth Utilization of BitLinear

Benchmark results for BitNet-2B-4T decode on the RTX 3060 Laptop:

- Total weight data read per token: approximately 1047 MB.
- Achieved bandwidth during BitLinear kernels: approximately 61 GB/s.
- Peak memory bandwidth: 336 GB/s.
- Effective utilization: approximately 18%.

The low utilization is not primarily a kernel inefficiency — it is a consequence of the sparse access pattern. Each warp reads from non-contiguous positions in `x` (determined by per-row bitmaps), preventing coalescing on the activation loads. The weight reads (meta and values arrays) do coalesce, since each lane reads from a strided but monotonically increasing group index.

For the LM head (`half_gemv`), which uses a dense activation vector and fully coalesced loads, bandwidth utilization is 92.7%. The contrast illustrates that the bottleneck in BitLinear is the sparse gather from `x`, not the streaming of weight data.

Further improvements to BitLinear bandwidth utilization would require either restructuring the sparse format to enable coalesced activation loads (e.g., grouping non-zero column indices by proximity), or increasing arithmetic intensity to hide latency (e.g., processing multiple tokens in parallel). Both directions require changes to the memory layout or the inference mode.

---

## Appendix: Kernel Reference

| Kernel | Launch shape | Purpose |
|---|---|---|
| `sparse_ternary_gemv_kernel` | `(ceil(rows/8), 1)` × 256 threads | INT8 sparse ternary GEMV, warp per row |
| `fused_sparse_bitlinear_kernel` | `(ceil(rows/8), 1)` × 256 threads | Half→INT8 quantize inline + GEMV + dequantize |
| `absmax_reduce_kernel` | `(1, 1)` × 256 threads | Compute max(|x|), write to device float |
| `absmax_quantize_kernel` | `(1, 1)` × 256 threads | Absmax + INT8 quantize (unfused path) |
| `dequantize_kernel` | `(ceil(rows/256), 1)` × 256 threads | INT32 → float16 scale |
| `rms_norm_kernel` | `(1, 1)` × 256 threads | RMSNorm, float16 in/out, float32 compute |
| `rms_norm_f32in_kernel` | `(1, 1)` × 256 threads | RMSNorm, float32 in, float16 out |
| `rope_kernel` | `(ceil(heads×pairs/256), 1)` × 256 | RoPE in-place on float16 head vectors |
| `scatter_kv_kernel` | `(ceil(kv_heads×head_dim/256), 1)` × 256 | Copy KV vector into cache at position |
| `attention_scores_kernel` | `(ceil(seq_len/warps_per_block), num_heads)` × 256 | Q·K dot products, warp per position |
| `softmax_kernel` | `(num_heads, 1)` × 256 | Stable softmax over score vector per head |
| `attention_output_kernel` | `(num_heads, 1)` × `head_dim` | Weighted sum of V over sequence |
| `relu2_mul_f32_kernel` | `(ceil(inter/256), 1)` × 256 | relu²(gate)×up → float32 |
| `half_gemv_kernel` | `(ceil(rows/8), 1)` × 256 threads | Float16 GEMV with float4 vectorized loads |
| `residual_add_kernel` | `(ceil(size/256), 1)` × 256 | Element-wise float16 add |
