#!/usr/bin/env python3
"""Compute theoretical bandwidth utilization and occupancy for spbitnet kernels.

This script computes hardware roofline metrics from known kernel configurations
and profiler timing data, without requiring Nsight Compute GPU counter access.

Usage:
    python python/analyze_profile.py
"""

import sys

# ---------------------------------------------------------------------------
# Hardware specs: RTX 3060 Laptop GPU (GA106, CC 8.6)
# ---------------------------------------------------------------------------

GPU = {
    "name": "RTX 3060 Laptop GPU",
    "sms": 30,
    "memory_bw_gbs": 336.0,       # GB/s theoretical peak
    "l2_cache_kb": 3072,
    "max_threads_per_sm": 1536,
    "max_warps_per_sm": 48,
    "max_blocks_per_sm": 16,
    "regs_per_sm": 65536,
    "smem_per_sm_kb": 100,        # configurable up to 100 KB on Ampere
    "clock_mhz": 1425,            # boost clock (varies with thermals)
    "fp32_tflops": 10.9,          # theoretical peak
}

# ---------------------------------------------------------------------------
# Model: BitNet-2B-4T (2.4B params)
# ---------------------------------------------------------------------------

MODEL = {
    "hidden_size": 2560,
    "num_layers": 30,
    "num_heads": 20,
    "num_kv_heads": 5,
    "head_dim": 128,
    "intermediate_size": 6912,
    "vocab_size": 128256,
}

# ---------------------------------------------------------------------------
# Kernel configurations
# ---------------------------------------------------------------------------

# fused_sparse_bitlinear_kernel: 1 warp per row, 256 threads/block (8 warps)
# Data read per row: meta + values + x (scattered reads)
# For a matrix [M x N]:
#   meta bytes per row = ceil(N/4 / 8) * 4 = ceil(groups_per_row / 8) * 4
#   values bytes per row = ceil(N/4 / 16) * 4 = ceil(groups_per_row / 16) * 4
#   x bytes: 2 * groups_per_row (2 half values per group, gathered)
#   output: 2 bytes (1 half per row)

def sparse_bitlinear_analysis(M, N, time_us, label=""):
    groups = N // 4
    meta_bytes_per_row = ((groups + 7) // 8) * 4
    values_bytes_per_row = ((groups + 15) // 16) * 4
    # x is read by all warps (broadcast via L1/L2)
    x_bytes = N * 2  # float16 input vector

    # Total bytes read for all rows (ignoring x cache reuse)
    weight_bytes = M * (meta_bytes_per_row + values_bytes_per_row)
    total_bytes = weight_bytes + x_bytes + M * 2  # +output writes

    # With x cached in L1/L2 (N * 2 bytes < L2 3 MB), effective reads:
    # Each warp reads x independently but L1 caching is very effective
    # Effective: weight_bytes + x_bytes (read once) + output_bytes
    effective_bytes = weight_bytes + x_bytes + M * 2

    bw_gbs = effective_bytes / (time_us * 1e-6) / 1e9
    bw_util = bw_gbs / GPU["memory_bw_gbs"] * 100

    # Occupancy: 256 threads/block = 8 warps/block
    warps_per_block = 8
    blocks_needed = (M + 7) // 8  # 8 rows per block
    blocks_per_sm = min(GPU["max_blocks_per_sm"],
                       GPU["max_warps_per_sm"] // warps_per_block)
    active_warps = blocks_per_sm * warps_per_block
    occupancy = active_warps / GPU["max_warps_per_sm"] * 100

    bits_per_weight = (meta_bytes_per_row + values_bytes_per_row) * 8 / N

    return {
        "label": label,
        "M": M, "N": N,
        "weight_bytes": weight_bytes,
        "effective_bytes": effective_bytes,
        "time_us": time_us,
        "bw_gbs": bw_gbs,
        "bw_util_pct": bw_util,
        "occupancy_pct": occupancy,
        "blocks_per_sm": blocks_per_sm,
        "bits_per_weight": bits_per_weight,
    }


def half_gemv_analysis(M, N, time_us, label=""):
    """LM head: half precision GEMV with float4 vectorized loads."""
    weight_bytes = M * N * 2  # float16 weights
    x_bytes = N * 2           # float16 input
    output_bytes = M * 4      # float32 output

    effective_bytes = weight_bytes + x_bytes + output_bytes
    bw_gbs = effective_bytes / (time_us * 1e-6) / 1e9
    bw_util = bw_gbs / GPU["memory_bw_gbs"] * 100

    return {
        "label": label,
        "M": M, "N": N,
        "weight_bytes": weight_bytes,
        "effective_bytes": effective_bytes,
        "time_us": time_us,
        "bw_gbs": bw_gbs,
        "bw_util_pct": bw_util,
    }


# ---------------------------------------------------------------------------
# Profiler data (from --profile benchmark, last timed run)
# These are CUDA event timings (accurate, includes kernel execution only)
# ---------------------------------------------------------------------------

# Per-BitLinear call timing (microseconds), from --profile with 128 tokens
# profiler reports: total_us / num_calls = mean_us
PROFILED_KERNELS = {
    # name: (total_us, calls, matrix_dims_MxN)
    "bitlinear_mlp":  (47376.4, 270, [(6912, 2560), (6912, 2560)]),  # gate + up (same dims)
    "bitlinear_down": (25774.9, 270, [(2560, 6912)]),
    "bitlinear_qkv":  (15376.7, 270, [(2560, 2560), (640, 2560)]),   # Q, K projections
    "bitlinear_o":    (10948.3, 270, [(2560, 2560)]),
    "bitlinear_v":    (4999.7,  270, [(640, 2560)]),
    "lm_head":        (18993.2, 9,   [(128256, 2560)]),
    "rms_norm":       (13452.5, 1089, []),
    "residual_add":   (2628.8,  540,  []),
    "scatter_kv":     (2445.1,  540,  []),
    "rope":           (2172.7,  270,  []),
    "softmax":        (1469.2,  270,  []),
    "attn_output":    (1462.4,  270,  []),
    "relu2_mul":      (1422.0,  270,  []),
    "attn_scores":    (1398.5,  270,  []),
    "embed_lookup":   (670.7,   9,    []),
}


def main():
    H = MODEL["hidden_size"]
    I = MODEL["intermediate_size"]
    V = MODEL["vocab_size"]
    KV = MODEL["num_kv_heads"] * MODEL["head_dim"]  # 640

    print("=" * 78)
    print("spbitnet Kernel Bandwidth & Occupancy Analysis")
    print(f"GPU: {GPU['name']} ({GPU['sms']} SMs, {GPU['memory_bw_gbs']} GB/s peak)")
    print(f"Model: BitNet-2B-4T ({MODEL['num_layers']}L, H={H}, I={I}, V={V})")
    print("=" * 78)

    # Analyze each BitLinear kernel
    print("\n--- Fused Sparse BitLinear Kernels (bandwidth-bound GEMV) ---\n")
    print(f"{'Kernel':<22} {'Dims':>14} {'Time/call':>10} "
          f"{'BW (GB/s)':>10} {'BW Util':>8} {'Occupancy':>10} {'b/w':>6}")
    print("-" * 86)

    bitlinear_analyses = [
        # MLP gate+up: 2 projections per layer per token
        ("mlp_gate", 6912, 2560, PROFILED_KERNELS["bitlinear_mlp"][0] / PROFILED_KERNELS["bitlinear_mlp"][1] / 2),
        ("mlp_up", 6912, 2560, PROFILED_KERNELS["bitlinear_mlp"][0] / PROFILED_KERNELS["bitlinear_mlp"][1] / 2),
        ("mlp_down", 2560, 6912, PROFILED_KERNELS["bitlinear_down"][0] / PROFILED_KERNELS["bitlinear_down"][1]),
        ("attn_q", 2560, 2560, PROFILED_KERNELS["bitlinear_qkv"][0] / PROFILED_KERNELS["bitlinear_qkv"][1] / 2),
        ("attn_k", 640, 2560, PROFILED_KERNELS["bitlinear_qkv"][0] / PROFILED_KERNELS["bitlinear_qkv"][1] / 2),
        ("attn_o", 2560, 2560, PROFILED_KERNELS["bitlinear_o"][0] / PROFILED_KERNELS["bitlinear_o"][1]),
        ("attn_v", 640, 2560, PROFILED_KERNELS["bitlinear_v"][0] / PROFILED_KERNELS["bitlinear_v"][1]),
    ]

    for name, M, N, time_us in bitlinear_analyses:
        r = sparse_bitlinear_analysis(M, N, time_us, name)
        print(f"  {name:<20} {M:>6}x{N:<6} {time_us:>8.1f}us "
              f"{r['bw_gbs']:>8.1f}   {r['bw_util_pct']:>6.1f}%  {r['occupancy_pct']:>8.1f}%  "
              f"{r['bits_per_weight']:>4.1f}")

    # LM head
    print("\n--- LM Head (float16 GEMV, float4 vectorized) ---\n")
    lm_time_us = PROFILED_KERNELS["lm_head"][0] / PROFILED_KERNELS["lm_head"][1]
    r = half_gemv_analysis(V, H, lm_time_us, "lm_head")
    print(f"  {'lm_head':<20} {V:>6}x{H:<6} {lm_time_us:>8.1f}us "
          f"{r['bw_gbs']:>8.1f}   {r['bw_util_pct']:>6.1f}%")

    # Summary
    print("\n--- Decode Token Time Budget (128 tokens, median) ---\n")
    total_profiled = sum(v[0] for v in PROFILED_KERNELS.values())
    print(f"{'Kernel':<22} {'Total(us)':>10} {'Mean(us)':>10} {'Calls':>6} {'%':>6}")
    print("-" * 58)
    for name in sorted(PROFILED_KERNELS, key=lambda k: PROFILED_KERNELS[k][0], reverse=True):
        total, calls, _ = PROFILED_KERNELS[name]
        per_token_us = total / 9  # 9 tokens in profiled run
        pct = total / total_profiled * 100
        print(f"  {name:<20} {total:>10.1f} {total/calls:>10.1f} {calls:>6} {pct:>5.1f}%")
    print(f"  {'TOTAL':<20} {total_profiled:>10.1f}")
    print(f"\n  Per-token (profiled): {total_profiled/9:.1f} us = {total_profiled/9/1000:.2f} ms")
    print(f"  Note: --profile adds ~2x overhead from cudaEventSynchronize per kernel")
    print(f"  Clean decode: 17.14 ms/tok (58.3 tok/s)")

    # Bandwidth roofline analysis
    print("\n--- Bandwidth Roofline Summary ---\n")
    # Total weight bytes read per decode token
    layers = MODEL["num_layers"]
    # Per-layer BitLinear weight bytes: sum of all 7 linear projections
    # Each projection: M * (meta_bytes_per_row + values_bytes_per_row) per row
    def weight_bytes_for(M, N):
        groups = N // 4
        meta_per_row = ((groups + 7) // 8) * 4
        vals_per_row = ((groups + 15) // 16) * 4
        return M * (meta_per_row + vals_per_row)

    per_layer_bytes = (
        weight_bytes_for(6912, 2560) * 2 +  # gate + up
        weight_bytes_for(2560, 6912) +        # down
        weight_bytes_for(2560, 2560) * 2 +    # q + o
        weight_bytes_for(640, 2560) * 2        # k + v
    )
    total_weight_bytes = per_layer_bytes * layers
    lm_head_bytes = V * H * 2  # float16

    total_read_bytes = total_weight_bytes + lm_head_bytes
    total_read_mb = total_read_bytes / 1e6

    clean_decode_ms = 17.14
    achieved_bw = total_read_bytes / (clean_decode_ms * 1e-3) / 1e9

    # Theoretical minimum time at peak bandwidth
    min_time_ms = total_read_bytes / (GPU["memory_bw_gbs"] * 1e9) * 1e3

    print(f"  Sparse weight bytes per token:  {total_weight_bytes/1e6:.1f} MB ({layers} layers)")
    print(f"  LM head bytes per token:        {lm_head_bytes/1e6:.1f} MB ({V}x{H} fp16)")
    print(f"  Total memory read per token:    {total_read_mb:.1f} MB")
    print(f"  Peak memory bandwidth:          {GPU['memory_bw_gbs']:.0f} GB/s")
    print(f"  Theoretical min decode time:    {min_time_ms:.2f} ms/tok ({1000/min_time_ms:.0f} tok/s)")
    print(f"  Achieved decode time:           {clean_decode_ms:.2f} ms/tok ({1000/clean_decode_ms:.0f} tok/s)")
    print(f"  Achieved bandwidth:             {achieved_bw:.1f} GB/s ({achieved_bw/GPU['memory_bw_gbs']*100:.0f}% of peak)")
    print(f"  Efficiency gap:                 {clean_decode_ms/min_time_ms:.2f}x (overhead from launch, L2 misses, compute)")


if __name__ == "__main__":
    main()
