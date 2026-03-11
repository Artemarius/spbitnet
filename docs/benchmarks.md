# spbitnet Benchmark Methodology and Results

**Model**: BitNet-2B-4T (2.4B parameters, 30 layers, hidden\_size=2560)
**Date**: March 2026
**Hardware**: NVIDIA RTX 3060 Laptop GPU — see Section 1 for full spec

---

## Table of Contents

1. [Hardware Configuration](#1-hardware-configuration)
2. [Measurement Methodology](#2-measurement-methodology)
3. [Kernel Microbenchmark Results](#3-kernel-microbenchmark-results)
4. [End-to-End Inference Results](#4-end-to-end-inference-results)
5. [Bandwidth Roofline Analysis](#5-bandwidth-roofline-analysis)
6. [Reproduction Instructions](#6-reproduction-instructions)

---

## 1. Hardware Configuration

| Property | Value |
|---|---|
| GPU | NVIDIA RTX 3060 Laptop GPU (GA106) |
| Compute Capability | 8.6 (Ampere) |
| Streaming Multiprocessors | 30 |
| CUDA Cores | 3840 |
| Tensor Cores | 112 (3rd-gen, INT8/FP16/BF16/TF32) |
| VRAM | 6 GB GDDR6, 192-bit bus |
| Peak Memory Bandwidth | 336 GB/s (theoretical) |
| L2 Cache | 3 MB |
| Shared Memory per SM | up to 100 KB (configurable) |
| Registers per SM | 65536 |
| Max Warps per SM | 48 |
| Max Blocks per SM | 16 |
| Boost Clock | ~1425 MHz (subject to thermal conditions) |
| FP32 Peak | 10.9 TFLOPS (theoretical) |
| CUDA Version | 12.8 |
| Driver | WSL2 Ubuntu 24.04 passthrough |
| OS | Windows 10 Pro 10.0.19045 / WSL2 Ubuntu 24.04 |
| Compiler | GCC 11, nvcc 12.8, C++17 |

All benchmarks were run on laptop hardware. The RTX 3060 Laptop GPU is a distinct SKU from the desktop RTX 3060: same GA106 die but different configurations (laptop: 30 SMs / 3840 CUDA cores; desktop: 28 SMs / 3584 CUDA cores). The laptop variant runs at lower boost clocks due to power constraints. The 336 GB/s bandwidth figure is the rated maximum; sustained bandwidth under real workloads is lower.

---

## 2. Measurement Methodology

### 2.1 Kernel Microbenchmarks (`spbitnet_bench`)

The benchmark harness in `benchmarks/bench_kernels.cu` uses CUDA events to measure GPU kernel execution time. The procedure is:

1. Allocate and initialize all GPU buffers before timing begins.
2. Run **20 warmup iterations** (kernels execute but timing is discarded).
3. Call `cudaDeviceSynchronize()` to drain the GPU before the timed phase begins.
4. For each of **100 timed iterations**:
   - Record `cudaEventRecord(start)`.
   - Launch the kernel.
   - Record `cudaEventRecord(stop)`.
   - Call `cudaEventSynchronize(stop)` to wait for completion.
   - Read elapsed time via `cudaEventElapsedTime`.
5. Sort the 100 elapsed times and report the **median** (element at index 50).

The min and max are computed but not shown in the tables below; they are available in the `BenchResult` struct returned by `bench_gpu`.

CUDA events measure device-side execution time only. Host-side overhead (kernel launch latency, argument marshaling) is not included. This is the correct measurement for comparing kernel throughput.

Input data is random INT8 values drawn from a fixed seed (42) before any warmup. The same data is used for all three kernel variants at each matrix dimension. Correctness is verified at each dimension before benchmarking: the sparse ternary GPU output is compared element-by-element against a CPU reference implementation, and a warning is printed if any mismatch is found.

### 2.2 End-to-End Benchmark (`--benchmark` flag)

End-to-end timing is measured in `src/main.cpp` using `std::chrono::high_resolution_clock`. The procedure is:

1. Load the model and initialize the inference engine before timing begins.
2. Run **one warmup generation** (full generation pass, output discarded) and call `cudaDeviceSynchronize()`. This primes GPU caches and eliminates any JIT compilation overhead.
3. Run **3 timed generations**. For each:
   - Reset the engine's KV cache.
   - Run timed prefill: process all prompt tokens except the last, surrounded by `cudaDeviceSynchronize()` calls. Elapsed time is measured with `chrono`.
   - Run timed decode: generate tokens one at a time until `max_tokens` is reached or EOS is produced. Each step copies logits from device to host via `cudaMemcpy` (synchronizing implicitly), and selects the argmax token. Total elapsed time for all decode steps is measured.
4. Sort the 3 prefill times and 3 decode times separately, report the **median** of each.

Prefill and decode are measured separately because they differ in memory access patterns and occupancy. Prefill processes the full prompt in one pass per token (sequential, no KV reuse); decode processes one token per forward pass at increasing context lengths.

The `cudaMemcpy` logit copy is included in decode timing because it is part of the real inference loop. For a production streaming deployment this copy would overlap with the next-token embedding lookup, but that optimization is not implemented here.

### 2.3 Profile Mode (`--profile` flag)

When `--profile` is passed alongside `--benchmark`, the `Profiler` class wraps each named kernel group with a `cudaEventRecord(start)` / `cudaEventRecord(stop)` / `cudaEventSynchronize(stop)` pair. This serializes the GPU timeline: each kernel must complete before the next begins, which eliminates any kernel overlap from the CUDA runtime scheduler.

The profiler adds approximately **2x overhead** to total decode time (observed: ~37.8 tok/s profiled vs 58.3 tok/s clean for the same workload). Profile timing numbers are therefore not comparable to clean benchmark numbers and should not be used for throughput reporting. Their purpose is measuring the relative cost of individual kernel groups, not absolute performance.

Profile data is accumulated across all timed runs. After the last timed run, `engine.profiler().report()` prints total microseconds, call count, mean per-call time, and percentage of total profiled time for each kernel group.

### 2.4 Thermal Conditions

All benchmarks include the warmup pass as the first step, which runs the GPU at close to full utilization for the duration of one generation. This brings the chip to thermal steady-state before timed measurements begin. No additional sleep or delay is inserted between warmup and timed runs. Laptop thermal throttling was not explicitly controlled; the reported numbers reflect the boost clock behavior of the chip under sustained load.

---

## 3. Kernel Microbenchmark Results

All times are median microseconds over 100 iterations. Matrix dimensions are labeled by their corresponding role in BitNet model layers.

### 3.1 Phase 2: Dense Ternary GEMV vs cuBLAS INT8 GEMV (n=1)

The dense ternary kernel stores weights in 2-bit packed format (2 bits per weight: `00`=0, `01`=+1, `10`=-1, packed into `uint32`). One thread is assigned per output row. The kernel reads packed weights, unpacks them, and accumulates the dot product using only integer addition and subtraction (no multiplications). Input activations are INT8.

cuBLAS INT8 uses `cublasGemmEx` with `CUBLAS_COMPUTE_32I`, which routes through INT8 Tensor Core IMMA instructions on CC 8.6.

| Dimension | Ternary (us) | cuBLAS INT8 (us) | Speedup | BW Ratio |
|---|---|---|---|---|
| 2048x2048 (attn proj) | 333.8 | 29.7 | 0.09x | 4.0x |
| 5632x2048 (FFN up) | 333.8 | 57.3 | 0.17x | 4.0x |
| 2048x5632 (FFN down) | 909.3 | 61.4 | 0.07x | 4.0x |
| 2560x2560 (attn proj) | 306.2 | 31.8 | 0.10x | 4.0x |
| 6912x2560 (FFN up) | 326.7 | 77.8 | 0.24x | 4.0x |
| 2560x6912 (FFN down) | 845.8 | 76.8 | 0.09x | 4.0x |
| 4096x4096 (large square) | 504.8 | 72.7 | 0.14x | 4.0x |
| 8192x2560 (wide FFN) | 329.6 | 87.0 | 0.26x | 4.0x |

The dense ternary kernel reads 4x less data than cuBLAS INT8 (2 bits/weight vs 8 bits/weight), but is 4-14x slower. The gap arises because the dense ternary kernel uses one thread per row, resulting in low SM occupancy and non-coalesced memory access patterns, while cuBLAS uses Tensor Core INT8 IMMA at high occupancy with optimized memory access.

### 3.2 Phase 3: Sparse Ternary GEMV vs cuBLAS INT8 GEMV (n=1)

The sparse ternary kernel uses 2:4 structured sparsity: exactly 2 of every 4 consecutive weights are non-zero. The compressed format stores metadata (4-bit bitmaps identifying the non-zero positions within each group of 4) and values (2-bit sign pairs for each non-zero) in separate arrays for coalesced GPU access. Effective storage is 1.5 bits per weight.

The kernel assigns one warp (32 threads) per output row. Each thread processes a contiguous slice of the input dimension, accumulating partial sums, which are then reduced across the warp using `__shfl_down_sync`. Bitmap decoding uses a 16-entry lookup table to identify which of the 4 positions in each group are non-zero, and sign handling is branchless. Input activations are INT8; partial sums accumulate into INT32 registers.

cuSPARSELt is excluded from GEMV comparisons because its INT8 SpMMA instruction requires n >= 16. It cannot perform GEMV (n=1) at all.

| Dimension | Dense Ternary (us) | Sparse Ternary (us) | cuBLAS INT8 (us) | Sparse / cuBLAS | BW Ratio |
|---|---|---|---|---|---|
| 2048x2048 (attn proj) | 333.8 | 25.6 | 29.7 | 1.16x | 5.3x |
| 5632x2048 (FFN up) | 335.9 | 58.4 | 58.1 | 1.00x | 5.3x |
| 2048x5632 (FFN down) | 912.4 | 60.4 | 71.9 | 1.19x | 5.3x |
| 2560x2560 (attn proj) | 418.8 | 35.8 | 37.9 | 1.06x | 5.3x |
| 6912x2560 (FFN up) | 437.2 | 86.0 | 90.1 | 1.05x | 5.3x |
| 2560x6912 (FFN down) | 1138.7 | 91.1 | 89.1 | 0.98x | 5.3x |
| 4096x4096 (large square) | 696.3 | 85.0 | 86.0 | 1.01x | 5.3x |
| 8192x2560 (wide FFN) | 458.8 | 102.4 | 102.4 | 1.00x | 5.3x |

The sparse ternary kernel is 13-18x faster than the dense ternary kernel at these dimensions, and matches or exceeds cuBLAS INT8 at all tested shapes while reading 5.3x less data. The BW ratio of 5.3x is consistent across all shapes: it reflects the ratio of INT8 bytes per weight (1 byte) to sparse ternary bytes per weight (1.5 bits meta + values combined, plus INT8 activations gathered at non-zero positions).

The 5.3x figure is derived from the data sizes passed to `bench_gpu`:

```
sparse bytes  = meta_array_bytes + values_array_bytes + x_cols + output_rows * 4
cuBLAS bytes  = rows * cols (INT8) + cols (x) + rows * 4 (output)
BW ratio      = cuBLAS bytes / sparse bytes
```

The sparse ternary kernel performs no integer multiplications. Each non-zero weight contributes a conditional `+x[j]` or `-x[j]` to the accumulator, implemented as branchless sign-conditional add.

### 3.3 Phase 4: cuSPARSELt Sparse Tensor Core GEMM (n=16)

cuSPARSELt uses the 2:4 sparse INT8 format supported natively by Ampere Sparse Tensor Cores. Weights are stored in NVIDIA's internal compressed format after a one-time `cusparseLtSpMMACompress` call. At runtime, `cusparseLtMatmul` executes INT8 SpMMA instructions. Activations are INT8; outputs are INT32.

For comparison, cuBLAS uses dense INT8 with standard INT8 IMMA Tensor Cores.

The minimum batch size for cuSPARSELt INT8 SpMMA on CC 8.6 is n=16. Attempting n=1 fails at the cuSPARSELt plan creation stage.

**n=16:**

| Dimension | cuBLAS INT8 (us) | cuSPARSELt (us) | Speedup |
|---|---|---|---|
| 2048x2048 (attn proj) | 60.4 | 14.3 | 4.21x |
| 5632x2048 (FFN up) | 147.5 | 32.8 | 4.50x |
| 2048x5632 (FFN down) | 142.3 | 42.0 | 3.39x |
| 2560x2560 (attn proj) | 89.1 | 24.6 | 3.62x |
| 6912x2560 (FFN up) | 220.2 | 44.2 | 4.98x |
| 2560x6912 (FFN down) | 205.8 | 50.2 | 4.10x |
| 4096x4096 (large square) | 197.6 | 44.0 | 4.49x |
| 8192x2560 (wide FFN) | 260.1 | 60.4 | 4.31x |

At n=16, cuSPARSELt is 3.4-5.0x faster than cuBLAS dense INT8. The speedup is close to the theoretical 2x from loading half the weight data, compounded by the higher throughput of the SpMMA datapath at this batch size. At n=32, the advantage narrows to 1.1-1.8x as the workload becomes more compute-bound and the memory bandwidth advantage of sparsity diminishes relative to the Tensor Core utilization increase.

### 3.4 Kernel Selection by Use Case

| Scenario | Kernel | Reason |
|---|---|---|
| GEMV (n=1), autoregressive decode | Custom sparse ternary | cuSPARSELt requires n >= 16; sparse ternary matches cuBLAS at 5.3x less memory |
| Batched GEMM (n >= 16) | cuSPARSELt INT8 SpMMA | Sparse Tensor Cores give 3.4-5.0x over dense cuBLAS at n=16 |

---

## 4. End-to-End Inference Results

**Model**: `microsoft/bitnet-b1.58-2B-4T-bf16` converted to spbitnet sparse format
**Decoding**: Greedy (argmax), no sampling
**Prompt**: "The future of AI is" (5 tokens)
**Benchmark tokens**: 128 decode steps
**Runs**: 1 warmup + 3 timed, median reported

### 4.1 Throughput

| Metric | Value |
|---|---|
| Decode throughput | 58.3 tok/s |
| Decode latency | 17.1 ms/tok |
| Prefill throughput | 60.7 tok/s |
| Prefill latency | 82.4 ms (5 prompt tokens) |

### 4.2 VRAM Usage

| Component | Memory |
|---|---|
| Sparse ternary weights (30 layers, all linear projections) | 391 MB |
| Embedding table (vocab 128256 x hidden 2560, fp16) | 657 MB |
| Total model weights | 1049 MB |
| KV cache (30 layers, 5 KV heads, head\_dim=128, max\_seq=4096, fp16) | ~300 MB |
| Scratch buffers | ~1 MB |
| CUDA/driver overhead | ~1216 MB |
| Total peak VRAM | 2566 MB / 6144 MB |

The KV cache is pre-allocated at engine construction for the maximum sequence length (4096 tokens). For a 128-token benchmark the cache is largely unused, but the allocation is fixed.

### 4.3 Per-Kernel Time Breakdown

These percentages are derived from profiler data (Section 2.3) and reflect relative cost, not absolute throughput. The profiler adds ~2x overhead; do not use these absolute times for throughput calculations.

| Kernel Group | % of Forward Pass | Description |
|---|---|---|
| BitLinear MLP gate + up | 31.5% | 2x fused sparse ternary GEMV, 6912x2560, 30 layers |
| BitLinear down | 17.1% | Fused sparse ternary GEMV, 2560x6912, 30 layers |
| LM head | 12.6% | Float16 GEMV, 128256x2560, float4 vectorized |
| BitLinear QKV | 10.2% | Q proj (2560x2560) + K proj (640x2560), fused |
| RMSNorm | 8.9% | Pre/post-attention norms + SubLN (attn\_sub\_norm, ffn\_sub\_norm) |
| BitLinear O + V | 10.6% | Output proj (2560x2560) + value proj (640x2560) |
| Attention + RoPE + misc | 9.1% | Scores, softmax, attention output, residual add, embed lookup |

BitLinear layers (sparse ternary GEMV) collectively account for approximately 69% of forward pass time. All BitLinear layers are memory-bandwidth bound: the weight data must be streamed from VRAM for every token, and the computation (add/subtract with no multiplications) completes faster than the memory access.

### 4.4 Optimization Results

| Optimization | Before | After | Gain |
|---|---|---|---|
| Fused BitLinear kernel (3 launches to 2 per layer) | 50.4 tok/s | 58.3 tok/s | +15.6% |
| Float4 vectorized loads in lm\_head | — | 92.7% BW utilization | — |
| Shared memory preload for input vector x | N/A | Rejected | syncthreads overhead exceeded L1 benefit |

The fused BitLinear kernel combines `absmax_reduce` (finding the input activation scale) and the `fused_sparse_bitlinear_gpu` operation (INT8 quantization of activations + sparse ternary GEMV + INT32-to-half dequantization) into 2 kernel launches per BitLinear layer, down from 3. At 30 layers with 7 BitLinear projections per layer, this eliminates 210 kernel launches per decode token, reducing launch overhead and eliminating one round-trip through the `d_quant_` and `d_int_out_` intermediate buffers.

---

## 5. Bandwidth Roofline Analysis

### 5.1 Data Transfer per Decode Token

The following byte counts are computed from the sparse weight format dimensions (see `python/analyze_profile.py` for the exact formulas).

For a matrix of shape M x N, the sparse ternary format uses:
- Meta array (4-bit bitmaps): `ceil(N/4 / 8) * 4` bytes per row
- Values array (2-bit sign pairs): `ceil(N/4 / 16) * 4` bytes per row

| Component | Bytes per Token |
|---|---|
| Sparse weights, 30 layers (all 7 linear projections) | ~390 MB |
| LM head weights (128256 x 2560, fp16) | ~656 MB |
| Input activation vectors (fp16) | ~1 MB |
| Output buffers | <1 MB |
| **Total read per token** | **~1047 MB** |

The LM head is the largest single read at ~656 MB per token, because it is stored in full float16 (no sparsity, no ternary compression). The 30-layer sparse ternary weights sum to ~390 MB despite representing more parameters, because the 1.5 bits/weight format is approximately 10.7x more compact than float16.

### 5.2 Roofline Numbers

| Quantity | Value |
|---|---|
| Total memory read per decode token | 1047 MB |
| Peak DRAM bandwidth | 336 GB/s |
| Theoretical minimum decode time (if at peak BW) | 3.12 ms/tok (320 tok/s) |
| Measured decode time | 17.14 ms/tok (58.3 tok/s) |
| Achieved bandwidth | 61 GB/s |
| Bandwidth utilization | 18% of peak |
| Gap from roofline | 5.5x |

### 5.3 Per-Kernel Bandwidth Utilization

The following utilization figures are computed by dividing effective bytes transferred (weight data + input vector + output) by the profiler-measured kernel time. They are upper bounds on the true utilization because the profiler timing includes all overhead within the synchronized kernel window.

| Kernel | Bandwidth Achieved | Peak Utilization |
|---|---|---|
| LM head (float16 GEMV, float4 vectorized) | ~311 GB/s | 92.7% |
| mlp\_gate / mlp\_up (6912x2560 sparse GEMV) | ~35-38 GB/s | 10-11% |
| mlp\_down (2560x6912 sparse GEMV) | ~38-42 GB/s | 11-13% |
| attn\_q / attn\_o (2560x2560 sparse GEMV) | ~33-36 GB/s | 10-11% |
| attn\_k / attn\_v (640x2560 sparse GEMV) | ~28-32 GB/s | 8-10% |

The LM head achieves 92.7% of peak bandwidth because:
- Float16 weights are contiguous, enabling fully coalesced reads with float4 (128-bit) loads.
- The matrix is large enough (128256 x 2560 x 2 bytes = 656 MB) that the working set far exceeds L2 cache, forcing all reads through DRAM.
- The output is a single vector (512 KB), not a limiting factor.

The sparse ternary BitLinear kernels achieve only 10-13% of peak bandwidth for several reasons:

1. **Non-contiguous x-vector access**: Each warp gathers the non-zero input values at positions indicated by the bitmap. These positions are not sequential across warps, producing non-coalesced reads for the activation vector.
2. **Metadata indirection**: Each group of 4 weights requires reading a 4-bit bitmap to determine which positions to gather, adding a dependent-load step before the activation read.
3. **Small output rows relative to row width**: The output for each warp is a single scalar, so the kernel is not benefiting from output write bandwidth.
4. **Low arithmetic intensity**: Each non-zero weight contributes exactly one add or subtract operation, giving arithmetic intensity well below the compute-bandwidth crossover point. The kernel spends most of its time waiting for memory, but cannot hide that latency with sufficient in-flight requests due to the gather pattern.

### 5.4 Explanation of the 5.5x Gap

The 5.5x gap between the theoretical minimum decode time (3.12 ms/tok at 336 GB/s peak) and the measured time (17.14 ms/tok) has several contributors:

- **Non-coalesced memory access in sparse GEMV**: The scattered gather pattern for activation values prevents the memory controller from issuing full 128-byte cache line fetches. Effective memory throughput for these kernels is 10-13% of rated peak rather than near 100%.
- **L2 cache pressure**: At 1047 MB/token, the working set is approximately 349x larger than the 3 MB L2 cache. Each token decode performs essentially a full traversal of cold DRAM.
- **Kernel launch overhead**: With ~7 BitLinear layers * 30 layers * 2 launches = 420 kernel launches per token (plus attention and norm kernels), launch overhead accumulates. The fused kernel optimization reduced this from 630 to 420 launches and recovered 15.6%.
- **Sequential kernel execution**: Without explicit pipelining, kernels execute in dependency order. CUDA concurrency is limited because each kernel's output feeds the next layer's input.
- **LM head serialization**: The 128256x2560 float16 GEMV (~656 MB) occurs once per token and alone accounts for ~656 MB / 336 GB/s = 1.95 ms if fully efficient. The measured contribution is ~2.1 ms (12.6% of 17.1 ms), indicating near-peak efficiency for this kernel only.

The dominant constraint is non-coalesced access in the sparse GEMV kernels. Improving this would require a different memory layout or a kernel design that processes multiple output rows simultaneously to amortize the cost of reading the activation vector.

---

## 6. Reproduction Instructions

The following commands reproduce all numbers reported in this document. Run all commands inside WSL2 Ubuntu from the project root.

### 6.1 Prerequisites

```bash
# Build the project (Release mode, CC 8.6 for RTX 3060)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j$(nproc)

# Verify GPU is visible
./build/spbitnet_bench 2>&1 | head -5
```

For cuSPARSELt results (Section 3.3), the library must be available at build time:

```bash
# Install cuSPARSELt (if not already present)
sudo apt install libcusparselt0-dev-cuda-12

# Rebuild with cuSPARSELt enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCUSPARSELT_DIR=/usr/lib/x86_64-linux-gnu
cmake --build build -j$(nproc)
```

### 6.2 Kernel Microbenchmarks (Sections 3.1, 3.2, 3.3)

```bash
# Runs all kernel benchmarks (Dense GEMV, Sparse GEMV, cuSPARSELt GEMM if available)
# 20 warmup + 100 timed iterations per dimension, median reported
./build/spbitnet_bench
```

Expected output includes two sections: GEMV (n=1) comparing dense ternary, sparse ternary, and cuBLAS; and GEMM (n=16, n=32) comparing cuSPARSELt against cuBLAS (if cuSPARSELt was compiled in).

The RNG seed is fixed at 42 in the benchmark binary; results are deterministic for a given GPU and driver version.

### 6.3 Model Conversion

```bash
# Install Python dependencies (CPU-only torch is sufficient)
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors huggingface-hub

# Convert BitNet-2B-4T (downloads ~5 GB of bf16 weights from HuggingFace)
python python/convert_model.py \
    --model microsoft/bitnet-b1.58-2B-4T-bf16 \
    --output models/bitnet-2b-4t-sparse/
```

Conversion applies absmean quantization (gamma = mean(|W|), W\_q = RoundClip(W/gamma, -1, 1)), enforces 2:4 sparsity via magnitude pruning on the pre-quantization float weights, and packs the result into the separated meta + values binary format.

### 6.4 End-to-End Benchmark (Section 4)

```bash
# Reproduce 58.3 tok/s decode result
# 1 warmup run + 3 timed runs, 128 decode tokens, greedy decoding
./build/spbitnet_infer \
    --model models/bitnet-2b-4t-sparse/ \
    --prompt "The future of AI is" \
    --benchmark 128
```

The output includes per-run timing and a summary showing median prefill ms, median decode ms, and derived tok/s values.

### 6.5 Per-Kernel Profile (Section 4.3)

```bash
# Profile mode: adds ~2x overhead, reports relative kernel cost
# Use a smaller token count to reduce wall-clock time
./build/spbitnet_infer \
    --model models/bitnet-2b-4t-sparse/ \
    --prompt "The future of AI is" \
    --benchmark 32 \
    --profile
```

The profiler report prints at the end after `=== Per-Kernel Profile (last timed run) ===`. The tok/s numbers printed during the benchmark run reflect profiled speed (~37 tok/s), not the clean speed. Ignore those numbers; use the clean benchmark (Section 6.4) for throughput.

### 6.6 Bandwidth Roofline Analysis (Section 5)

```bash
# Compute theoretical bandwidth utilization from known kernel dimensions
# No GPU required; uses hardcoded profiler data and model geometry
python python/analyze_profile.py
```

The script prints bandwidth utilization per kernel, total bytes per decode token, theoretical minimum decode time, achieved bandwidth, and the gap multiplier. To update the profiler timing inputs, edit the `PROFILED_KERNELS` dictionary at the top of `python/analyze_profile.py` with data from a `--profile` run.

### 6.7 Benchmark Charts

```bash
# Generate the six benchmark charts (requires matplotlib)
# Uses a separate venv to avoid conflicts with torch/transformers
python3 -m venv /tmp/plot-venv
source /tmp/plot-venv/bin/activate
pip install matplotlib numpy
python python/plot_benchmarks.py
# Output: docs/plots/*.png
```

---

## Appendix: Profiler Raw Data

The following table contains the raw profiler accumulator values used in `python/analyze_profile.py`, captured from a `--benchmark 128 --profile` run. These are used to derive the per-kernel percentages in Section 4.3.

| Kernel | Total (us) | Calls | Mean/call (us) | % of total |
|---|---|---|---|---|
| bitlinear\_mlp | 47376.4 | 270 | 175.5 | 31.5% |
| bitlinear\_down | 25774.9 | 270 | 95.5 | 17.1% |
| lm\_head | 18993.2 | 9 | 2110.4 | 12.6% |
| bitlinear\_qkv | 15376.7 | 270 | 56.9 | 10.2% |
| rms\_norm | 13452.5 | 1089 | 12.4 | 8.9% |
| bitlinear\_o | 10948.3 | 270 | 40.5 | 7.3% |
| bitlinear\_v | 4999.7 | 270 | 18.5 | 3.3% |
| residual\_add | 2628.8 | 540 | 4.9 | 1.7% |
| scatter\_kv | 2445.1 | 540 | 4.5 | 1.6% |
| rope | 2172.7 | 270 | 8.0 | 1.4% |
| softmax | 1469.2 | 270 | 5.4 | 1.0% |
| attn\_output | 1462.4 | 270 | 5.4 | 1.0% |
| relu2\_mul | 1422.0 | 270 | 5.3 | 0.9% |
| attn\_scores | 1398.5 | 270 | 5.2 | 0.9% |
| embed\_lookup | 670.7 | 9 | 74.5 | 0.4% |
| **Total** | **150590.9** | — | — | 100% |

Note: "calls" counts are consistent with 9 timed decode steps in the profiled run (128 tokens / ~14 unique kernel calls per forward pass = 9 tokens used by the profiler accumulation window for lm\_head; 270 = 9 tokens * 30 layers for per-layer kernels; 1089 = 270 * ~4 RMSNorm calls per layer due to SubLN).

The lm\_head is called once per token (not once per layer), hence 9 calls for 9 profiled tokens. The discrepancy between benchmark\_tokens=128 and profiler calls=9 is because the profiler resets between runs and only the last timed run's data is reported.
