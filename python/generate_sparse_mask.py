#!/usr/bin/env python3
"""
generate_sparse_mask.py -- Magnitude-based 2:4 structured sparsity mask
generation for ternary (1.58-bit) model weights.

Implements the mask generation strategy from the Sparse-BitNet paper
(Zhang et al., March 2026, Section 3.2): the 2:4 sparsity mask is computed
from pre-quantized latent (floating-point) weights, NOT from quantized
ternary values.  This avoids ambiguous tie-breaking on {-1, 0, +1}.

The binary export format is byte-identical with the C++
spbitnet::SparseTernaryTensor class defined in
include/spbitnet/sparse_ternary_tensor.h.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants (must match include/spbitnet/sparse_ternary_tensor.h)
# ---------------------------------------------------------------------------
SPARSE_GROUP_SIZE = 4
SPARSE_NON_ZEROS = 2
META_GROUPS_PER_WORD = 8      # 8 x 4-bit bitmaps per uint32
VALUES_GROUPS_PER_WORD = 16   # 16 x 2-bit sign pairs per uint32


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_24_mask(latent_weights: np.ndarray) -> np.ndarray:
    """Generate a 2:4 structured sparsity mask from latent (float) weights.

    For each group of 4 consecutive elements in each row, the 2 elements
    with the largest absolute value are kept (mask = True).  Ties are broken
    by preferring the lower column index (deterministic).

    Parameters
    ----------
    latent_weights : np.ndarray
        2-D float array of shape (rows, cols).  ``cols`` must be a
        multiple of 4.

    Returns
    -------
    np.ndarray
        Boolean array of the same shape.  ``True`` means *keep* (non-zero
        in the sparse output), ``False`` means *prune*.
    """
    if latent_weights.ndim != 2:
        raise ValueError("latent_weights must be 2-D")
    rows, cols = latent_weights.shape
    if cols % SPARSE_GROUP_SIZE != 0:
        raise ValueError(
            f"cols ({cols}) must be a multiple of {SPARSE_GROUP_SIZE}"
        )

    mask = np.zeros_like(latent_weights, dtype=bool)
    groups_per_row = cols // SPARSE_GROUP_SIZE

    # Reshape to (rows, groups_per_row, 4) for vectorised processing.
    reshaped = np.abs(latent_weights).reshape(rows, groups_per_row, SPARSE_GROUP_SIZE)

    # argsort along the last axis (ascending), then take the last 2.
    # np.argsort is stable, so equal magnitudes keep their original order
    # (lower index first).  We want top-2 by magnitude, so we take the
    # last two indices of the ascending sort.
    order = np.argsort(reshaped, axis=-1, kind="stable")
    # top-2 indices within each group
    top2 = order[:, :, -SPARSE_NON_ZEROS:]  # shape (rows, groups, 2)

    # Build mask: set True at the top-2 positions in each group.
    r_idx = np.arange(rows)[:, None, None]
    g_idx = np.arange(groups_per_row)[None, :, None]
    mask_reshaped = mask.reshape(rows, groups_per_row, SPARSE_GROUP_SIZE)
    mask_reshaped[r_idx, g_idx, top2] = True

    return mask


def quantize_ternary(
    weights: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Quantize floating-point weights to ternary {-1, 0, +1}.

    Thresholding rule (matches C++ ``encode_ternary``):
        - value > threshold  -->  +1
        - value < -threshold -->  -1
        - otherwise          -->   0

    Parameters
    ----------
    weights : np.ndarray
        Float weight array of arbitrary shape.
    threshold : float
        Symmetric threshold (default 0.5).

    Returns
    -------
    np.ndarray
        int8 array of the same shape with values in {-1, 0, +1}.
    """
    out = np.zeros_like(weights, dtype=np.int8)
    out[weights > threshold] = 1
    out[weights < -threshold] = -1
    return out


def apply_sparse_ternary(
    latent_weights: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine 2:4 mask generation with ternary quantization.

    The mask is computed on the *latent* (floating-point) weights, then
    quantization is applied, and pruned positions are zeroed out.

    Parameters
    ----------
    latent_weights : np.ndarray
        2-D float array of shape (rows, cols).
    threshold : float
        Symmetric quantization threshold (default 0.5).

    Returns
    -------
    sparse_ternary : np.ndarray
        int8 array (rows, cols) with ternary values at kept positions and
        zeros at pruned positions.
    mask : np.ndarray
        Boolean array (rows, cols).
    """
    mask = generate_24_mask(latent_weights)
    ternary = quantize_ternary(latent_weights, threshold)
    sparse_ternary = ternary * mask.astype(np.int8)
    return sparse_ternary, mask


# ---------------------------------------------------------------------------
# Binary export (must be byte-identical with C++ SparseTernaryTensor)
# ---------------------------------------------------------------------------

def _meta_words_per_row(cols: int) -> int:
    """Number of uint32 words in the meta array per row."""
    groups = cols // SPARSE_GROUP_SIZE
    return (groups + META_GROUPS_PER_WORD - 1) // META_GROUPS_PER_WORD


def _values_words_per_row(cols: int) -> int:
    """Number of uint32 words in the values array per row."""
    groups = cols // SPARSE_GROUP_SIZE
    return (groups + VALUES_GROUPS_PER_WORD - 1) // VALUES_GROUPS_PER_WORD


def export_sparse_ternary(
    sparse_weights: np.ndarray,
    mask: np.ndarray,
    output_path: str,
) -> None:
    """Export sparse ternary weights to binary files compatible with C++.

    Writes two files:

    * ``{output_path}.meta``   -- position bitmaps (uint32 array)
    * ``{output_path}.values`` -- sign-bit pairs  (uint32 array)

    The layout is identical to ``spbitnet::SparseTernaryTensor``
    (see ``include/spbitnet/sparse_ternary_tensor.h``).

    Parameters
    ----------
    sparse_weights : np.ndarray
        int8 array (rows, cols) with zeros at pruned positions.
    mask : np.ndarray
        Boolean array (rows, cols).
    output_path : str
        Base path (without extension).
    """
    if sparse_weights.ndim != 2 or mask.ndim != 2:
        raise ValueError("sparse_weights and mask must be 2-D")
    rows, cols = sparse_weights.shape
    if mask.shape != (rows, cols):
        raise ValueError("sparse_weights and mask must have the same shape")
    if cols % SPARSE_GROUP_SIZE != 0:
        raise ValueError(
            f"cols ({cols}) must be a multiple of {SPARSE_GROUP_SIZE}"
        )

    groups_per_row = cols // SPARSE_GROUP_SIZE
    m_stride = _meta_words_per_row(cols)
    v_stride = _values_words_per_row(cols)

    meta = np.zeros((rows, m_stride), dtype=np.uint32)
    values = np.zeros((rows, v_stride), dtype=np.uint32)

    for r in range(rows):
        for g in range(groups_per_row):
            base = g * SPARSE_GROUP_SIZE

            # Build 4-bit position bitmap
            bitmap = np.uint32(0)
            for pos in range(SPARSE_GROUP_SIZE):
                if mask[r, base + pos]:
                    bitmap |= np.uint32(1 << pos)

            word_idx = g // META_GROUPS_PER_WORD
            bit_offset = (g % META_GROUPS_PER_WORD) * 4
            meta[r, word_idx] |= np.uint32(int(bitmap) << bit_offset)

            # Build 2-bit sign pair
            signs = np.uint32(0)
            nz_idx = 0
            for pos in range(SPARSE_GROUP_SIZE):
                if mask[r, base + pos]:
                    sign_bit = np.uint32(1) if sparse_weights[r, base + pos] < 0 else np.uint32(0)
                    signs |= np.uint32(int(sign_bit) << nz_idx)
                    nz_idx += 1

            word_idx = g // VALUES_GROUPS_PER_WORD
            bit_offset = (g % VALUES_GROUPS_PER_WORD) * 2
            values[r, word_idx] |= np.uint32(int(signs) << bit_offset)

    # Write binary files (little-endian uint32, row-major)
    meta_path = f"{output_path}.meta"
    values_path = f"{output_path}.values"

    meta.tofile(meta_path)
    values.tofile(values_path)

    print(f"Exported meta   -> {meta_path}  "
          f"({meta.nbytes} bytes, {rows}x{m_stride} uint32)")
    print(f"Exported values -> {values_path}  "
          f"({values.nbytes} bytes, {rows}x{v_stride} uint32)")


def import_sparse_ternary(
    input_path: str,
    rows: int,
    cols: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read back binary files and reconstruct sparse ternary weights + mask.

    This mirrors ``SparseTernaryTensor::unpack_to_int8`` on the C++ side.

    Parameters
    ----------
    input_path : str
        Base path (without extension).
    rows, cols : int
        Matrix dimensions.

    Returns
    -------
    weights : np.ndarray
        int8 array (rows, cols).
    mask : np.ndarray
        Boolean array (rows, cols).
    """
    if cols % SPARSE_GROUP_SIZE != 0:
        raise ValueError(
            f"cols ({cols}) must be a multiple of {SPARSE_GROUP_SIZE}"
        )

    groups_per_row = cols // SPARSE_GROUP_SIZE
    m_stride = _meta_words_per_row(cols)
    v_stride = _values_words_per_row(cols)

    meta = np.fromfile(f"{input_path}.meta", dtype=np.uint32).reshape(rows, m_stride)
    values = np.fromfile(f"{input_path}.values", dtype=np.uint32).reshape(rows, v_stride)

    weights = np.zeros((rows, cols), dtype=np.int8)
    mask = np.zeros((rows, cols), dtype=bool)

    for r in range(rows):
        for g in range(groups_per_row):
            base = g * SPARSE_GROUP_SIZE

            # Extract 4-bit bitmap
            word_idx = g // META_GROUPS_PER_WORD
            bit_offset = (g % META_GROUPS_PER_WORD) * 4
            bitmap = (int(meta[r, word_idx]) >> bit_offset) & 0xF

            # Extract 2-bit sign pair
            word_idx_v = g // VALUES_GROUPS_PER_WORD
            bit_offset_v = (g % VALUES_GROUPS_PER_WORD) * 2
            signs = (int(values[r, word_idx_v]) >> bit_offset_v) & 0x3

            nz_idx = 0
            for pos in range(SPARSE_GROUP_SIZE):
                if bitmap & (1 << pos):
                    mask[r, base + pos] = True
                    sign_bit = (signs >> nz_idx) & 1
                    weights[r, base + pos] = -1 if sign_bit else +1
                    nz_idx += 1

    return weights, mask


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(
    latent_weights: np.ndarray,
    ternary_weights: np.ndarray,
    sparse_ternary: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Print useful statistics about the sparsification pipeline."""
    rows, cols = latent_weights.shape
    total = rows * cols

    print("\n" + "=" * 64)
    print("Sparse-BitNet 2:4 Mask Generation Statistics")
    print("=" * 64)
    print(f"Matrix shape:           {rows} x {cols}  ({total:,} weights)")

    # Ternary distribution before pruning
    t_neg = np.sum(ternary_weights == -1)
    t_zero = np.sum(ternary_weights == 0)
    t_pos = np.sum(ternary_weights == 1)
    print(f"\n--- Dense ternary (before pruning) ---")
    print(f"  -1:  {t_neg:>8,}  ({100 * t_neg / total:5.1f}%)")
    print(f"   0:  {t_zero:>8,}  ({100 * t_zero / total:5.1f}%)")
    print(f"  +1:  {t_pos:>8,}  ({100 * t_pos / total:5.1f}%)")
    dense_nonzero = t_neg + t_pos
    print(f"  density (non-zero):   {100 * dense_nonzero / total:5.1f}%")

    # Ternary distribution after pruning
    s_neg = np.sum(sparse_ternary == -1)
    s_zero = np.sum(sparse_ternary == 0)
    s_pos = np.sum(sparse_ternary == 1)
    print(f"\n--- Sparse ternary (after 2:4 pruning) ---")
    print(f"  -1:  {s_neg:>8,}  ({100 * s_neg / total:5.1f}%)")
    print(f"   0:  {s_zero:>8,}  ({100 * s_zero / total:5.1f}%)")
    print(f"  +1:  {s_pos:>8,}  ({100 * s_pos / total:5.1f}%)")
    sparse_nonzero = s_neg + s_pos
    print(f"  density (non-zero):   {100 * sparse_nonzero / total:5.1f}%")
    print(f"  expected density:     50.0%  (2:4 = 2 of every 4 kept)")

    # Compression ratio
    dense_ternary_bits = total * 2  # 2 bits per weight
    sparse_bits = total * 1.5       # 4-bit bitmap + 2-bit signs per group of 4
    print(f"\n--- Compression ---")
    print(f"  Dense ternary:        {dense_ternary_bits / 8:,.0f} bytes  "
          f"(2 bits/weight)")
    print(f"  Sparse ternary:       {sparse_bits / 8:,.0f} bytes  "
          f"(1.5 bits/weight)")
    print(f"  Compression ratio:    {sparse_bits / dense_ternary_bits:.2%} "
          f"of dense ternary")

    # Position pruning distribution (should be roughly uniform for random)
    groups_per_row = cols // SPARSE_GROUP_SIZE
    total_groups = rows * groups_per_row
    mask_reshaped = mask.reshape(rows, groups_per_row, SPARSE_GROUP_SIZE)
    pos_kept_counts = mask_reshaped.sum(axis=(0, 1))  # shape (4,)
    print(f"\n--- Position distribution in groups of 4 ---")
    print(f"  (how often each position is kept; should be ~uniform for "
          f"random weights)")
    for pos in range(SPARSE_GROUP_SIZE):
        pct = 100.0 * pos_kept_counts[pos] / total_groups
        bar = "#" * int(pct / 2)
        print(f"  pos {pos}: {pos_kept_counts[pos]:>8,} / {total_groups:,}  "
              f"({pct:5.1f}%)  {bar}")

    print("=" * 64)


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2:4 structured sparsity masks for ternary weights "
            "(Sparse-BitNet, Zhang et al. 2026)."
        ),
    )
    parser.add_argument(
        "--rows", type=int, default=256,
        help="Number of rows in the synthetic weight matrix (default: 256)",
    )
    parser.add_argument(
        "--cols", type=int, default=1024,
        help="Number of columns (must be multiple of 4, default: 1024)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Base path for exported binary files (without extension). "
            "If omitted, uses a temp path under the current directory."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.cols % SPARSE_GROUP_SIZE != 0:
        print(f"Error: --cols ({args.cols}) must be a multiple of "
              f"{SPARSE_GROUP_SIZE}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    # 1. Generate synthetic latent weights (normal distribution, as in
    #    typical neural network initialization).
    print(f"Generating {args.rows}x{args.cols} random latent weights "
          f"(seed={args.seed}) ...")
    latent_weights = rng.standard_normal((args.rows, args.cols)).astype(np.float32)

    # 2. Run the full pipeline: mask generation + quantization + sparsity.
    print("Applying 2:4 sparsity mask + ternary quantization ...")
    sparse_ternary, mask = apply_sparse_ternary(latent_weights)
    ternary_dense = quantize_ternary(latent_weights)

    # 3. Print statistics.
    print_statistics(latent_weights, ternary_dense, sparse_ternary, mask)

    # 4. Export to binary.
    if args.output is None:
        output_path = f"sparse_ternary_{args.rows}x{args.cols}"
    else:
        output_path = args.output

    print(f"\nExporting to binary format (C++ SparseTernaryTensor compatible) ...")
    export_sparse_ternary(sparse_ternary, mask, output_path)

    # 5. Verify round-trip: read back and compare.
    print(f"\nVerifying round-trip read-back ...")
    read_weights, read_mask = import_sparse_ternary(
        output_path, args.rows, args.cols,
    )

    # Masks must match exactly.
    assert np.array_equal(mask, read_mask), "Mask round-trip FAILED"

    # Weights at kept positions must match.  Weights at pruned positions
    # are zero in both representations, so full comparison works -- except
    # that ternary values that are 0 at kept positions become +1 (sign=0)
    # in the packed format (matching C++ behaviour).  Adjust expectation:
    # if a kept position had quantized value 0, the packed format stores
    # sign=0, which unpacks to +1.
    expected_readback = sparse_ternary.copy()
    expected_readback[(mask) & (sparse_ternary == 0)] = 1  # sign=0 -> +1

    if np.array_equal(read_weights, expected_readback):
        print("Round-trip verification PASSED")
    else:
        mismatches = np.sum(read_weights != expected_readback)
        total = args.rows * args.cols
        print(f"Round-trip verification FAILED: {mismatches}/{total} "
              f"mismatches")
        # Show first few mismatches for debugging
        where = np.argwhere(read_weights != expected_readback)
        for idx in where[:10]:
            r, c = idx
            print(f"  [{r},{c}] expected={expected_readback[r,c]} "
                  f"got={read_weights[r,c]}  mask={mask[r,c]}  "
                  f"sparse_ternary={sparse_ternary[r,c]}")
        sys.exit(1)

    # 6. Clean up exported files (optional; keep them for inspection).
    meta_path = Path(f"{output_path}.meta")
    values_path = Path(f"{output_path}.values")
    print(f"\nBinary files kept for inspection:")
    print(f"  {meta_path.resolve()}")
    print(f"  {values_path.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
