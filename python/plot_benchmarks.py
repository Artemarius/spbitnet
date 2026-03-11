#!/usr/bin/env python3
"""Generate benchmark plots for spbitnet README and docs.

Usage:
    source /tmp/spbitnet-plot-venv/bin/activate
    python python/plot_benchmarks.py [--outdir docs/plots]

Generates PNG charts from hardcoded benchmark data (RTX 3060 Laptop GPU).
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Benchmark data (RTX 3060 Laptop GPU, 30 SMs, CUDA 12.8)
# ---------------------------------------------------------------------------

GEMV_DIMS = [
    "2048x2048", "5632x2048", "2048x5632", "2560x2560",
    "6912x2560", "2560x6912", "4096x4096", "8192x2560",
]
GEMV_LABELS = [
    "2048x2048\nattn proj", "5632x2048\nFFN up", "2048x5632\nFFN down",
    "2560x2560\nattn proj", "6912x2560\nFFN up", "2560x6912\nFFN down",
    "4096x4096\nlarge sq", "8192x2560\nwide FFN",
]

# GEMV (n=1) timings in microseconds
GEMV_DENSE   = [94.2, 94.2, 255.0, 123.9, 129.0, 323.6, 198.7, 166.9]
GEMV_SPARSE  = [25.6, 58.4, 60.4, 35.8, 86.0, 91.1, 86.0, 102.4]
GEMV_CUBLAS  = [29.7, 58.4, 71.8, 37.9, 90.1, 89.1, 86.0, 102.4]

# cuSPARSELt GEMM (n=16) in microseconds
GEMM16_CUBLAS  = [60.4, 147.5, 142.3, 89.1, 220.2, 205.8, 197.6, 260.1]
GEMM16_CSPLT   = [14.3, 32.8, 42.0, 24.6, 44.2, 50.2, 44.0, 60.4]

# cuSPARSELt GEMM (n=32) in microseconds
GEMM32_CUBLAS  = [22.4, 48.1, 50.0, 29.7, 69.6, 66.6, 80.9, 90.1]
GEMM32_CSPLT   = [16.3, 34.8, 44.0, 27.6, 48.1, 54.3, 46.1, 64.5]

# Per-kernel breakdown (% of forward pass, decode, 128 tokens)
KERNEL_NAMES = [
    "BitLinear MLP\n(gate+up)", "BitLinear down", "BitLinear QKV",
    "BitLinear O+V", "LM head", "RMSNorm",
    "Attention\n+RoPE+misc",
]
KERNEL_PCT = [36.1, 18.2, 10.5, 10.7, 9.2, 7.8, 7.5]

# Memory breakdown (MB)
MEM_LABELS = ["Embedding\n(fp16)", "Sparse ternary\nweights", "KV cache\n(fp16)",
              "Norms+scratch", "CUDA/driver\noverhead"]
MEM_VALUES = [656.7, 390.8, 300.0, 2.7, 1216.0]

# End-to-end throughput (clean, no profiling overhead)
E2E_LABELS = ["Before fused\nBitLinear", "After fused\nBitLinear"]
E2E_TOKS   = [50.4, 58.3]  # Profiled: 32.7 → 37.8; Clean: ~50.4 → 58.3 (same 15.6% gain)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {
    "sparse": "#2ecc71",     # green
    "cublas": "#3498db",     # blue
    "dense":  "#e74c3c",     # red
    "csplt":  "#9b59b6",     # purple
    "accent": "#f39c12",     # orange
    "bg":     "#fafafa",
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": COLORS["bg"],
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })


def save_fig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Chart 1: GEMV kernel comparison
# ---------------------------------------------------------------------------

def plot_gemv_comparison(outdir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(GEMV_DIMS))
    w = 0.25

    bars_dense  = ax.bar(x - w, GEMV_DENSE,  w, label="Dense ternary", color=COLORS["dense"], alpha=0.85)
    bars_cublas = ax.bar(x,     GEMV_CUBLAS,  w, label="cuBLAS INT8",   color=COLORS["cublas"], alpha=0.85)
    bars_sparse = ax.bar(x + w, GEMV_SPARSE,  w, label="Sparse ternary (ours)", color=COLORS["sparse"], alpha=0.85)

    # Speedup annotations
    for i in range(len(GEMV_DIMS)):
        speedup = GEMV_CUBLAS[i] / GEMV_SPARSE[i]
        if speedup >= 1.01:
            ax.annotate(f"{speedup:.2f}x",
                        xy=(x[i] + w, GEMV_SPARSE[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, fontweight="bold", color=COLORS["sparse"])

    ax.set_ylabel("Latency (\u00b5s, lower is better)")
    ax.set_title("GEMV (n=1) \u2014 Autoregressive Inference Kernel Comparison\n"
                 "RTX 3060 Laptop \u2022 cuSPARSELt excluded (requires n\u226516)")
    ax.set_xticks(x)
    ax.set_xticklabels(GEMV_LABELS, fontsize=8)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(GEMV_DENSE) * 1.15)

    save_fig(fig, outdir, "gemv_comparison.png")


# ---------------------------------------------------------------------------
# Chart 2: cuSPARSELt GEMM comparison
# ---------------------------------------------------------------------------

def plot_gemm_comparison(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    x = np.arange(len(GEMV_DIMS))
    w = 0.35

    for ax, n, cb_data, sp_data in [
        (axes[0], 16, GEMM16_CUBLAS, GEMM16_CSPLT),
        (axes[1], 32, GEMM32_CUBLAS, GEMM32_CSPLT),
    ]:
        ax.bar(x - w/2, cb_data, w, label="cuBLAS INT8 (dense)", color=COLORS["cublas"], alpha=0.85)
        ax.bar(x + w/2, sp_data, w, label="cuSPARSELt (2:4 sparse)", color=COLORS["csplt"], alpha=0.85)

        for i in range(len(GEMV_DIMS)):
            speedup = cb_data[i] / sp_data[i]
            ax.annotate(f"{speedup:.1f}x",
                        xy=(x[i] + w/2, sp_data[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, fontweight="bold", color=COLORS["csplt"])

        ax.set_ylabel("Latency (\u00b5s, lower is better)")
        ax.set_title(f"GEMM n={n} \u2014 Sparse Tensor Cores")
        ax.set_xticks(x)
        ax.set_xticklabels([d.split("\n")[0] for d in GEMV_LABELS], fontsize=7, rotation=30, ha="right")
        ax.legend(fontsize=8)

    fig.suptitle("cuSPARSELt vs cuBLAS \u2014 Batched Inference (Sparse Tensor Cores)\n"
                 "RTX 3060 Laptop \u2022 INT8 2:4 structured sparsity", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, outdir, "gemm_cusparselt.png")


# ---------------------------------------------------------------------------
# Chart 3: Per-kernel breakdown
# ---------------------------------------------------------------------------

def plot_kernel_breakdown(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                     gridspec_kw={"width_ratios": [2, 1]})

    # Horizontal bar chart
    colors = ["#e74c3c", "#c0392b", "#e67e22", "#d35400",
              "#3498db", "#f39c12", "#95a5a6"]
    y = np.arange(len(KERNEL_NAMES))
    bars = ax1.barh(y, KERNEL_PCT, color=colors, alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(KERNEL_NAMES, fontsize=9)
    ax1.set_xlabel("% of forward pass time")
    ax1.set_title("Per-Kernel Time Breakdown (Decode)")
    ax1.invert_yaxis()
    for bar, pct in zip(bars, KERNEL_PCT):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{pct:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, 42)

    # Pie chart — grouped
    bitlinear_total = 36.1 + 18.2 + 10.5 + 10.7
    grouped = [bitlinear_total, 9.2, 7.8, 7.5]
    grouped_labels = [f"BitLinear\n(sparse GEMV)\n{bitlinear_total:.0f}%",
                      f"LM head\n{9.2}%", f"RMSNorm\n{7.8}%",
                      f"Attn+misc\n{7.5}%"]
    grouped_colors = ["#e74c3c", "#3498db", "#f39c12", "#95a5a6"]
    wedges, texts = ax2.pie(grouped, labels=grouped_labels, colors=grouped_colors,
                            startangle=90, textprops={"fontsize": 9})
    ax2.set_title("Grouped Breakdown")

    fig.suptitle("BitNet-2B-4T Decode Kernel Profile \u2014 RTX 3060 Laptop\n"
                 "58.3 tok/s \u2022 17.1 ms/token \u2022 BitLinear = 76% (bandwidth-bound)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, outdir, "kernel_breakdown.png")


# ---------------------------------------------------------------------------
# Chart 4: Memory usage
# ---------------------------------------------------------------------------

def plot_memory(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                     gridspec_kw={"width_ratios": [1.2, 1]})

    # Stacked bar: VRAM usage
    colors = ["#3498db", "#2ecc71", "#e67e22", "#95a5a6", "#bdc3c7"]
    bottom = 0
    for label, val, color in zip(MEM_LABELS, MEM_VALUES, colors):
        ax1.bar(0, val, bottom=bottom, color=color, label=f"{label} ({val:.0f} MB)",
                width=0.5, alpha=0.85)
        if val > 50:
            ax1.text(0, bottom + val/2, f"{val:.0f} MB", ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white")
        bottom += val
    ax1.axhline(y=6144, color="red", linestyle="--", linewidth=1.5, label="RTX 3060 6 GB")
    ax1.set_ylabel("VRAM (MB)")
    ax1.set_title("Peak VRAM Usage")
    ax1.set_xticks([0])
    ax1.set_xticklabels(["BitNet-2B-4T"])
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(0, 7000)

    # Weight compression comparison
    model_params = 2.4  # billion
    fp16_gb = model_params * 2  # 2 bytes per param
    int8_gb = model_params * 1  # 1 byte per param
    ternary_gb = model_params * 2 / 8  # 2 bits per param
    sparse_gb = model_params * 1.5 / 8  # 1.5 bits per param (our format)

    formats = ["FP16\n(16 b/w)", "INT8\n(8 b/w)", "Dense ternary\n(2 b/w)", "Sparse ternary\n(1.5 b/w)"]
    sizes = [fp16_gb * 1024, int8_gb * 1024, ternary_gb * 1024, sparse_gb * 1024]
    fmt_colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    bars = ax2.bar(range(len(formats)), sizes, color=fmt_colors, alpha=0.85)
    for bar, sz in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{sz:.0f} MB", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Weight Memory (MB)")
    ax2.set_title("Weight Compression (2.4B params)")
    ax2.set_xticks(range(len(formats)))
    ax2.set_xticklabels(formats, fontsize=8)
    ax2.set_ylim(0, sizes[0] * 1.15)

    fig.suptitle("Memory Efficiency \u2014 Sparse-BitNet on RTX 3060 (6 GB VRAM)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, outdir, "memory_usage.png")


# ---------------------------------------------------------------------------
# Chart 5: Optimization impact (fused kernel)
# ---------------------------------------------------------------------------

def plot_optimization(outdir):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    colors = [COLORS["cublas"], COLORS["sparse"]]
    bars = ax.bar(E2E_LABELS, E2E_TOKS, color=colors, width=0.5, alpha=0.85)

    for bar, val in zip(bars, E2E_TOKS):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f} tok/s", ha="center", fontsize=11, fontweight="bold")

    # Speedup annotation
    speedup = (E2E_TOKS[1] - E2E_TOKS[0]) / E2E_TOKS[0] * 100
    ax.annotate(f"+{speedup:.1f}%",
                xy=(1, E2E_TOKS[1]),
                xytext=(0.5, E2E_TOKS[1] + 2),
                fontsize=14, fontweight="bold", color=COLORS["sparse"],
                arrowprops=dict(arrowstyle="->", color=COLORS["sparse"]))

    ax.set_ylabel("Decode Throughput (tok/s)")
    ax.set_title("Fused BitLinear Kernel Optimization\n"
                 "BitNet-2B-4T \u2022 RTX 3060 Laptop \u2022 128 tokens",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 45)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    save_fig(fig, outdir, "optimization_impact.png")


# ---------------------------------------------------------------------------
# Chart 6: Bandwidth advantage summary
# ---------------------------------------------------------------------------

def plot_bandwidth_summary(outdir):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    categories = ["Weight bits\nper param", "Memory read\n(2.4B model)", "Bandwidth\nsavings vs FP16"]

    fp16_vals  = [16, 4800, 0]
    int8_vals  = [8, 2400, 50]
    tern_vals  = [2, 600, 87.5]
    spar_vals  = [1.5, 450, 90.6]  # sparse ternary = ~1.0-1.5 effective

    x = np.arange(len(categories))
    w = 0.2

    # Only show for "Weight bits per param" — normalized display
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    formats = ["FP16", "INT8", "Dense\nternary", "Sparse\nternary\n(ours)"]
    bits = [16, 8, 2, 1.5]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    bars = ax2.bar(formats, bits, color=colors, alpha=0.85, width=0.6)
    for bar, b in zip(bars, bits):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{b} bits", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Bits per weight")
    ax2.set_title("Weight Representation Efficiency\n"
                  "Sparse ternary: 10.7x compression vs FP16",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 20)

    save_fig(fig2, outdir, "bits_per_weight.png")
    plt.close(fig)  # close unused first figure


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate spbitnet benchmark plots")
    parser.add_argument("--outdir", default="docs/plots", help="Output directory for PNGs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup_style()

    print("Generating benchmark plots...")
    plot_gemv_comparison(args.outdir)
    plot_gemm_comparison(args.outdir)
    plot_kernel_breakdown(args.outdir)
    plot_memory(args.outdir)
    plot_optimization(args.outdir)
    plot_bandwidth_summary(args.outdir)
    print(f"\nDone! {len(os.listdir(args.outdir))} plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
