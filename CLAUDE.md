# CLAUDE.md — spbitnet

## What This Project Is

Custom CUDA inference kernels for Sparse-BitNet — combining 1.58-bit ternary quantization with 2:4 structured sparsity on consumer NVIDIA Ampere GPUs. The project implements ideas from the Sparse-BitNet paper (Zhang et al., March 2026) with kernels optimized for the RTX 3060 (6 GB VRAM, Compute Capability 8.6, 112 Tensor Cores).

**Read PROJECT.md for full context on goals, motivation, and career strategy.**

## Developer Background

Artem has deep expertise in: C++ (15+ years, expert), CUDA (expert), GPU optimization, multithreading, real-time systems, memory management, computer vision, 3D processing. He comes from industrial 3D scanning (Artec3D) and Samsung R&D. He is building ML/DL infrastructure expertise for quantitative finance and AI company roles. This project specifically demonstrates understanding of:
- LLM inference at the hardware level
- Quantization and sparsity — two pillars of efficient inference
- Custom CUDA kernel development for novel academic techniques
- Bridging research papers to working implementations

## Development Environment

- **OS**: Windows 10 Pro with WSL2 Ubuntu 24.04
- **GPU**: NVIDIA RTX 3060 12GB, Compute Capability 8.6, 112 3rd-gen Tensor Cores
- **CUDA**: 12.8
- **Compiler**: GCC 11+ (WSL2), targeting C++17
- **Build**: CMake 3.24+
- **IDE**: VS Code with Remote-WSL
- **Python**: 3.9+ (for model conversion scripts only — core engine is C++)

## Architecture Decisions

### Why C++17 (not C++20/23)
- Consistency with CuInfer project
- Broader compiler/toolchain compatibility
- Quant firms typically use C++17 in production

### Why Two Kernel Paths
1. **cuSPARSELt path**: NVIDIA's library for hardware 2:4 sparse GEMM. Gives us the baseline "what does hardware sparsity buy us?" answer. Weights expanded to INT8 {-1, 0, +1} — doesn't exploit ternary structure, but uses Sparse Tensor Cores.
2. **Custom ternary-sparse path**: Our novel contribution. Exploits both ternary arithmetic (no multiplies, only add/sub) and sparsity (skip zeros). Custom memory layout, custom iteration. This is where the portfolio differentiation is.

### Weight Storage Format
- Dense ternary: 2 bits per weight (00=0, 01=+1, 10=-1), packed into uint32
- Sparse ternary (2:4): Only non-zero values stored (2 × 2 bits) + 4-bit position metadata per group of 4. Effective ~1.0 bit/weight
- cuSPARSELt path: INT8 representation with NVIDIA's compressed sparse format

### Target Models (all fit in 6 GB VRAM easily)
- `1bitLLM/bitnet_b1_58-large` — 729M params, ~0.2 GB ternary weights
- `microsoft/BitNet-b1.58-2B-4T` — 2.4B params, ~0.5 GB ternary weights
- `tiiuae/Falcon3-1B-Instruct-1.58bit` — 1B params, ~0.2 GB ternary weights

## Code Style & Conventions

- Namespaces: `spbitnet::` for all project code
- Kernel naming: `sparse_ternary_*` for custom sparsity kernels, `cusparselt_*` for library wrappers, descriptive names for inference ops (e.g. `rms_norm_kernel`, `attention_scores_kernel`, `rope_kernel`)
- Error handling: `CUDA_CHECK()` macro for all CUDA calls, exceptions for host errors
- Memory: explicit `cudaMalloc`/`cudaFree`, no smart pointers for device memory
- Testing: Google Test for host, custom assertion macros for device code
- Comments: reference paper equations by number (e.g., "Eq. (2) from Sparse-BitNet")

## Build Commands

```bash
# Configure (from project root, inside WSL2)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCUSPARSELT_DIR=/path/to/cusparselt

# Build
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure

# Run benchmarks
./build/spbitnet_bench --suite all
```

## Key Dependencies

- **cuSPARSELt** (NVIDIA): For hardware sparse GEMM baseline. Download from developer.nvidia.com/cusparselt
- **hnswlib** or similar: NOT needed here (this is inference, not vector search)
- **Google Test**: For unit testing
- **nlohmann/json**: For model config parsing (header-only)

## Phase Tracking

Current phase tracked in PROJECT.md. Each phase has:
- Specific deliverables
- Validation criteria (correctness and performance)
- Definition of Done

## Important Notes

- The cuSPARSELt library requires separate download — it's not bundled with CUDA Toolkit
- cuSPARSELt minimum CC is 8.0, our RTX 3060 is CC 8.6 — fully supported
- The Sparse-BitNet paper's reference code (github.com/AAzdi/Sparse-BitNet) is PyTorch/training focused — we are building C++/CUDA inference from scratch
- bitnet.cpp (Microsoft) is CPU-focused — our contribution is GPU-focused with sparsity
- The 2:4 sparsity mask must be applied to pre-quantized (latent float) weights using magnitude pruning, NOT to the quantized ternary values (tie-breaking issues)
