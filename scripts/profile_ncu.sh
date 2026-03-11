#!/bin/bash
# Nsight Compute profiling script for spbitnet kernels.
#
# PREREQUISITE: On consumer NVIDIA GPUs (GeForce), GPU performance counter
# access requires a Windows registry change + reboot:
#
#   1. Open regedit as Administrator
#   2. Navigate to: HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm
#   3. Add DWORD: RmProfilingAdminOnly = 0
#   4. Reboot
#
# See: https://developer.nvidia.com/ERR_NVGPUCTRPERM
#
# After the registry change, run this script from WSL2:
#   cd /mnt/e/Repos/spbitnet && bash scripts/profile_ncu.sh

set -euo pipefail

NCU=/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu
MODEL=models/bitnet-2b-4t-sparse
BINARY=./build/spbitnet_infer
OUTDIR=docs/ncu

mkdir -p "$OUTDIR"

echo "=== Profiling fused_sparse_bitlinear_kernel (the main bottleneck) ==="
$NCU \
    --kernel-name-base demangled \
    --kernel-name 'regex:fused_sparse_bitlinear_kernel' \
    --launch-skip 0 --launch-count 3 \
    --set full \
    --export "$OUTDIR/fused_bitlinear" \
    --csv \
    "$BINARY" --model "$MODEL" --prompt 'Hi' --max-tokens 2 \
    2>&1 | tee "$OUTDIR/fused_bitlinear.csv"

echo ""
echo "=== Profiling half_gemv_kernel (lm_head) ==="
$NCU \
    --kernel-name-base demangled \
    --kernel-name 'regex:half_gemv_kernel' \
    --launch-skip 0 --launch-count 1 \
    --set full \
    --export "$OUTDIR/half_gemv" \
    --csv \
    "$BINARY" --model "$MODEL" --prompt 'Hi' --max-tokens 2 \
    2>&1 | tee "$OUTDIR/half_gemv.csv"

echo ""
echo "=== Profiling rms_norm_kernel ==="
$NCU \
    --kernel-name-base demangled \
    --kernel-name 'regex:rms_norm_kernel' \
    --launch-skip 0 --launch-count 3 \
    --set full \
    --export "$OUTDIR/rms_norm" \
    --csv \
    "$BINARY" --model "$MODEL" --prompt 'Hi' --max-tokens 2 \
    2>&1 | tee "$OUTDIR/rms_norm.csv"

echo ""
echo "=== Done! Open .ncu-rep files in Nsight Compute UI for visualization ==="
echo "Reports saved to: $OUTDIR/"
ls -la "$OUTDIR/"
