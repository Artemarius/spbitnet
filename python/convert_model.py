#!/usr/bin/env python3
"""convert_model.py — Convert HuggingFace BitNet model to spbitnet sparse ternary format.

Downloads the bf16 variant of a BitNet model, applies 2:4 structured sparsity
using magnitude-based pruning on latent weights, quantizes to ternary with
absmean scaling (Eq. (1) from BitNet b1.58), packs into the compressed
sparse ternary format, and exports binary weight files for C++ inference.

The output binary format is byte-identical with the C++ SparseTernaryTensor
class defined in include/spbitnet/sparse_ternary_tensor.h.

Usage:
    # Full conversion (downloads ~5 GB on first run):
    python convert_model.py \\
        --model microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --output models/bitnet-2b-4t-sparse/

    # Inspect model structure without converting:
    python convert_model.py \\
        --model microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --dry-run

    # Convert from local directory:
    python convert_model.py \\
        --model /path/to/local/model \\
        --output models/my-model-sparse/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants (must match include/spbitnet/sparse_ternary_tensor.h)
# ---------------------------------------------------------------------------
SPARSE_GROUP_SIZE = 4
SPARSE_NON_ZEROS = 2
META_GROUPS_PER_WORD = 8       # 8 x 4-bit bitmaps per uint32
VALUES_GROUPS_PER_WORD = 16    # 16 x 2-bit sign pairs per uint32
FORMAT_VERSION = 1

# Weight categories — covers Llama-style and BitNet architectures
EMBED_NAMES = {"model.embed_tokens.weight", "embed_tokens.weight"}
FINAL_NORM_NAMES = {"model.norm.weight", "norm.weight"}
LM_HEAD_NAMES = {"lm_head.weight"}

NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.attn_sub_norm.weight",
    "mlp.ffn_sub_norm.weight",
)

LINEAR_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


# ---------------------------------------------------------------------------
# Sparse ternary packing (vectorized, C++-compatible)
# ---------------------------------------------------------------------------

def meta_words_per_row(cols: int) -> int:
    groups = cols // SPARSE_GROUP_SIZE
    return (groups + META_GROUPS_PER_WORD - 1) // META_GROUPS_PER_WORD


def values_words_per_row(cols: int) -> int:
    groups = cols // SPARSE_GROUP_SIZE
    return (groups + VALUES_GROUPS_PER_WORD - 1) // VALUES_GROUPS_PER_WORD


def generate_24_mask(weights: np.ndarray) -> np.ndarray:
    """Generate 2:4 structured sparsity mask from weight magnitudes.

    For each group of 4 consecutive elements in each row, keeps the 2 with
    the largest absolute value.  Stable argsort gives deterministic
    tie-breaking (lower index preferred).

    Parameters
    ----------
    weights : (rows, cols) float array.  cols must be a multiple of 4.

    Returns
    -------
    Boolean mask (rows, cols).  True = keep (non-zero).
    """
    rows, cols = weights.shape
    assert cols % SPARSE_GROUP_SIZE == 0, f"cols={cols} not multiple of 4"

    num_groups = cols // SPARSE_GROUP_SIZE
    grouped = np.abs(weights).reshape(rows, num_groups, SPARSE_GROUP_SIZE)

    # Ascending sort; top-2 = last 2 indices
    order = np.argsort(grouped, axis=-1, kind="stable")
    top2 = order[:, :, -SPARSE_NON_ZEROS:]

    mask = np.zeros_like(weights, dtype=bool)
    mask_g = mask.reshape(rows, num_groups, SPARSE_GROUP_SIZE)
    r_idx = np.arange(rows)[:, None, None]
    g_idx = np.arange(num_groups)[None, :, None]
    mask_g[r_idx, g_idx, top2] = True

    return mask


def quantize_ternary_absmean(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize using BitNet's absmean method (Eq. (1) from BitNet b1.58).

    W_q = RoundClip(W / gamma, -1, 1)  where  gamma = mean(|W|)

    Returns
    -------
    ternary : int8 array {-1, 0, +1}
    gamma   : per-tensor scale factor (float)
    """
    gamma = float(np.mean(np.abs(weights)))
    if gamma < 1e-10:
        return np.zeros_like(weights, dtype=np.int8), gamma

    scaled = weights / gamma
    rounded = np.clip(np.round(scaled), -1, 1).astype(np.int8)
    return rounded, gamma


def pack_sparse_ternary(
    weights: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Vectorized packing into C++-compatible sparse ternary format.

    Produces the same binary layout as SparseTernaryTensor::pack_from_dense().

    Parameters
    ----------
    weights : (rows, cols) int8 ternary weights {-1, 0, +1}
    mask    : (rows, cols) boolean mask (exactly 2 True per group of 4)

    Returns
    -------
    meta          : uint32 array (rows, meta_stride)  — position bitmaps
    values        : uint32 array (rows, values_stride) — sign pairs
    meta_stride   : words per row in meta
    values_stride : words per row in values
    """
    rows, cols = weights.shape
    num_groups = cols // SPARSE_GROUP_SIZE

    m_stride = meta_words_per_row(cols)
    v_stride = values_words_per_row(cols)

    # Apply mask to weights
    sparse_w = (weights * mask.astype(np.int8)).reshape(rows, num_groups, SPARSE_GROUP_SIZE)
    mask_g = mask.reshape(rows, num_groups, SPARSE_GROUP_SIZE)

    # --- Build 4-bit bitmaps (rows, num_groups) ---
    bitmaps = np.zeros((rows, num_groups), dtype=np.uint32)
    for i in range(SPARSE_GROUP_SIZE):
        bitmaps |= mask_g[:, :, i].astype(np.uint32) << i

    # --- Build 2-bit sign pairs (rows, num_groups) ---
    # Use cumsum to identify 1st vs 2nd non-zero in each group
    cumsum = np.cumsum(mask_g.astype(np.int32), axis=2)  # (R, G, 4)
    is_first_nz = mask_g & (cumsum == 1)
    is_second_nz = mask_g & (cumsum == 2)

    # sign bit = 1 if weight is -1, 0 otherwise
    first_neg = np.any(is_first_nz & (sparse_w == -1), axis=2).astype(np.uint32)
    second_neg = np.any(is_second_nz & (sparse_w == -1), axis=2).astype(np.uint32)
    signs = first_neg | (second_neg << 1)  # (rows, num_groups)

    # --- Pack bitmaps into meta words (8 groups per uint32) ---
    pad_m = m_stride * META_GROUPS_PER_WORD - num_groups
    if pad_m > 0:
        bitmaps = np.pad(bitmaps, ((0, 0), (0, pad_m)))
    bitmaps = bitmaps.reshape(rows, m_stride, META_GROUPS_PER_WORD)

    meta = np.zeros((rows, m_stride), dtype=np.uint32)
    for j in range(META_GROUPS_PER_WORD):
        meta |= bitmaps[:, :, j] << np.uint32(j * 4)

    # --- Pack signs into values words (16 groups per uint32) ---
    pad_v = v_stride * VALUES_GROUPS_PER_WORD - num_groups
    if pad_v > 0:
        signs = np.pad(signs, ((0, 0), (0, pad_v)))
    signs = signs.reshape(rows, v_stride, VALUES_GROUPS_PER_WORD)

    values = np.zeros((rows, v_stride), dtype=np.uint32)
    for j in range(VALUES_GROUPS_PER_WORD):
        values |= signs[:, :, j] << np.uint32(j * 2)

    return meta, values, m_stride, v_stride


# ---------------------------------------------------------------------------
# Weight categorization helpers
# ---------------------------------------------------------------------------

def categorize_weight(name: str) -> str:
    """Categorize a weight tensor by its role in the model."""
    if name in EMBED_NAMES:
        return "embedding"
    if name in FINAL_NORM_NAMES:
        return "norm"
    if name in LM_HEAD_NAMES:
        return "lm_head"
    for suffix in NORM_SUFFIXES:
        if name.endswith(suffix):
            return "norm"
    for suffix in LINEAR_SUFFIXES:
        if name.endswith(suffix):
            return "linear"
    return "unknown"


def export_name(param_name: str) -> str:
    """Convert HF param name to our naming convention.

    model.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj
    model.embed_tokens.weight              -> embed_tokens
    model.norm.weight                      -> norm
    """
    name = param_name
    if name.startswith("model."):
        name = name[len("model."):]
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    return name


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_float_tensor(tensor: np.ndarray, path: Path, dtype=np.float16) -> int:
    """Export tensor as raw binary (little-endian, row-major). Returns nbytes."""
    data = tensor.astype(dtype)
    data.tofile(str(path))
    return data.nbytes


def export_sparse_layer(
    weight_f32: np.ndarray,
    base_path: Path,
) -> Dict[str, Any]:
    """Apply 2:4 sparsity + ternary quantization + pack + export.

    Writes {base_path}.meta and {base_path}.values.

    Returns manifest entry dict (including gamma and stats).
    """
    rows, cols = weight_f32.shape

    # 1. Generate 2:4 mask on float magnitudes
    mask = generate_24_mask(weight_f32)

    # 2. Quantize to ternary with absmean scaling
    ternary, gamma = quantize_ternary_absmean(weight_f32)

    # 3. Apply mask
    sparse_ternary = ternary * mask.astype(np.int8)

    # 4. Fix zeros at masked positions — the packed format can't represent
    #    ternary 0 at a position marked non-zero in the bitmap (it defaults
    #    to +1).  Force these positions to ±1 based on the original float sign.
    zero_at_mask = mask & (sparse_ternary == 0)
    sparse_ternary[zero_at_mask & (weight_f32 >= 0)] = +1
    sparse_ternary[zero_at_mask & (weight_f32 < 0)] = -1

    # 5. Pack
    meta, values, m_stride, v_stride = pack_sparse_ternary(sparse_ternary, mask)

    # 6. Write binary files
    # Note: can't use with_suffix() because export names contain dots
    meta_path = Path(str(base_path) + ".meta")
    values_path = Path(str(base_path) + ".values")
    meta.tofile(str(meta_path))
    values.tofile(str(values_path))

    # 7. Stats
    total = rows * cols
    nz_before = int(np.count_nonzero(ternary))
    nz_after = int(np.count_nonzero(sparse_ternary))

    return {
        "meta_file": meta_path.name,
        "values_file": values_path.name,
        "dtype": "sparse_ternary",
        "shape": [rows, cols],
        "meta_stride": m_stride,
        "values_stride": v_stride,
        "gamma": gamma,
        "meta_nbytes": int(meta.nbytes),
        "values_nbytes": int(values.nbytes),
        "stats": {
            "nz_before_pruning": nz_before,
            "nz_after_pruning": nz_after,
            "density_before": nz_before / total,
            "density_after": nz_after / total,
        },
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_weights(
    model_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load model weights as float32 numpy arrays + config dict.

    Supports both safetensors (preferred) and torch bin formats.
    Handles bfloat16 → float32 conversion automatically.
    """
    import torch
    from huggingface_hub import snapshot_download

    # Download model files (config.json + weights)
    print(f"Downloading model files from {model_path} ...")
    local_dir = snapshot_download(
        model_path,
        allow_patterns=["*.safetensors", "*.bin", "*.json"],
    )
    local_path = Path(local_dir)
    print(f"  Local cache: {local_dir}")

    # Load config directly from JSON (avoids AutoConfig issues with
    # auto_map pointing to missing custom code files)
    config_path = local_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    with open(config_path) as f:
        config_dict = json.load(f)
    print(f"  Config loaded: {config_dict.get('model_type', 'unknown')} "
          f"({config_dict.get('hidden_size', '?')}h, "
          f"{config_dict.get('num_hidden_layers', '?')}L)")

    state_dict: Dict[str, np.ndarray] = {}

    # Try safetensors first
    st_files = sorted(local_path.glob("*.safetensors"))
    if st_files:
        from safetensors.torch import load_file

        for st_file in st_files:
            print(f"  Loading {st_file.name} ...")
            tensors = load_file(str(st_file))
            for key, tensor in tensors.items():
                state_dict[key] = tensor.float().numpy()
        print(f"  Loaded {len(state_dict)} tensors from safetensors")
    else:
        # Fallback: torch .bin files
        bin_files = sorted(local_path.glob("pytorch_model*.bin"))
        if bin_files:
            for bin_file in bin_files:
                print(f"  Loading {bin_file.name} ...")
                checkpoint = torch.load(str(bin_file), map_location="cpu")
                for key, tensor in checkpoint.items():
                    state_dict[key] = tensor.float().numpy()
            print(f"  Loaded {len(state_dict)} tensors from .bin files")
        else:
            raise FileNotFoundError(
                f"No .safetensors or .bin files found in {local_dir}"
            )

    return state_dict, config_dict


# ---------------------------------------------------------------------------
# Dry run: inspect model without converting
# ---------------------------------------------------------------------------

def dry_run(model_path: str) -> None:
    """Print model structure and estimated output sizes."""
    state_dict, config = load_model_weights(model_path)

    print(f"\n{'=' * 72}")
    print(f"Model: {model_path}")
    print(f"{'=' * 72}")

    # Config summary
    print(f"\nArchitecture:")
    for key in [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size",
        "max_position_embeddings", "rms_norm_eps", "rope_theta",
        "hidden_act", "tie_word_embeddings",
    ]:
        val = config.get(key)
        if val is not None:
            print(f"  {key}: {val}")

    head_dim = config["hidden_size"] // config["num_attention_heads"]
    print(f"  head_dim (computed): {head_dim}")

    # Weight table
    print(f"\nWeights ({len(state_dict)} tensors):")
    totals = {"embedding": 0, "norm": 0, "linear": 0, "lm_head": 0, "unknown": 0}

    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        cat = categorize_weight(name)
        totals[cat] += tensor.size

        # Sample to check if already ternary
        sample = tensor.flatten()[:1000]
        unique = np.unique(np.round(sample, 4))
        is_ternary = len(unique) <= 5  # allow some noise around {-1, 0, +1}

        flag = " [~ternary]" if is_ternary else ""
        print(
            f"  {cat:10s} {name:55s} "
            f"{str(tensor.shape):20s} {tensor.dtype}{flag}"
        )

    total_params = sum(totals.values())
    print(f"\nTotal: {total_params:,} parameters")

    # Estimated output size
    sparse_bytes = totals["linear"] * 1.5 / 8
    embed_bytes = totals["embedding"] * 2  # float16
    if config.get("tie_word_embeddings", False):
        lm_head_bytes = 0
    else:
        lm_head_bytes = totals["lm_head"] * 2
    norm_bytes = totals["norm"] * 4  # float32

    total_bytes = sparse_bytes + embed_bytes + lm_head_bytes + norm_bytes
    print(f"\nEstimated output size:")
    print(f"  Sparse ternary (linear layers): {sparse_bytes / 1e6:>8.1f} MB  "
          f"({totals['linear']:>12,} params)")
    print(f"  Embedding (float16):            {embed_bytes / 1e6:>8.1f} MB  "
          f"({totals['embedding']:>12,} params)")
    if lm_head_bytes > 0:
        print(f"  LM head (float16):              {lm_head_bytes / 1e6:>8.1f} MB  "
              f"({totals['lm_head']:>12,} params)")
    else:
        print(f"  LM head:                         tied with embedding")
    print(f"  Norms (float32):                {norm_bytes / 1e6:>8.1f} MB  "
          f"({totals['norm']:>12,} params)")
    print(f"  {'':34s} {'─' * 10}")
    print(f"  Total:                          {total_bytes / 1e6:>8.1f} MB")
    print(f"\n  (vs {total_params * 2 / 1e6:.1f} MB as float16, "
          f"{total_bytes / (total_params * 2) * 100:.1f}% of original)")


# ---------------------------------------------------------------------------
# Full conversion pipeline
# ---------------------------------------------------------------------------

def convert_model(
    model_path: str,
    output_dir: str,
    copy_tokenizer: bool = True,
) -> None:
    """Convert a HuggingFace BitNet model to spbitnet sparse ternary format."""

    output = Path(output_dir)
    weights_dir = output / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    state_dict, config = load_model_weights(model_path)

    # 2. Export inference config
    inference_config = {
        "model_type": config.get("model_type", "bitnet"),
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config.get(
            "num_key_value_heads", config["num_attention_heads"]
        ),
        "intermediate_size": config["intermediate_size"],
        "vocab_size": config["vocab_size"],
        "max_position_embeddings": config["max_position_embeddings"],
        "rms_norm_eps": config.get("rms_norm_eps", 1e-5),
        "rope_theta": config.get("rope_theta", 10000.0),
        "head_dim": config["hidden_size"] // config["num_attention_heads"],
        "hidden_act": config.get("hidden_act", "relu2"),
        "tie_word_embeddings": config.get("tie_word_embeddings", False),
        "bos_token_id": config.get("bos_token_id", 1),
        "eos_token_id": config.get("eos_token_id", 2),
    }

    config_path = output / "config.json"
    with open(config_path, "w") as f:
        json.dump(inference_config, f, indent=2)
    print(f"\nExported config -> {config_path}")

    # 3. Copy tokenizer files
    if copy_tokenizer:
        _copy_tokenizer(model_path, output)

    # 4. Convert all weights
    print(f"\nConverting weights ...")
    manifest: Dict[str, Any] = {"format_version": FORMAT_VERSION, "weights": {}}
    total_sparse = 0
    total_float = 0
    param_names = sorted(state_dict.keys())

    for idx, param_name in enumerate(param_names):
        tensor = state_dict[param_name]
        cat = categorize_weight(param_name)
        ename = export_name(param_name)
        tag = f"[{idx + 1}/{len(param_names)}]"

        if cat == "embedding":
            path = weights_dir / f"{ename}.bin"
            nb = export_float_tensor(tensor, path, np.float16)
            manifest["weights"][ename] = {
                "file": f"weights/{ename}.bin",
                "dtype": "float16",
                "shape": list(tensor.shape),
                "nbytes": nb,
            }
            total_float += nb
            print(f"  {tag} {ename}: embedding {list(tensor.shape)} "
                  f"-> float16 ({nb / 1e6:.1f} MB)")

        elif cat == "norm":
            path = weights_dir / f"{ename}.bin"
            nb = export_float_tensor(tensor, path, np.float32)
            manifest["weights"][ename] = {
                "file": f"weights/{ename}.bin",
                "dtype": "float32",
                "shape": list(tensor.shape),
                "nbytes": nb,
            }
            total_float += nb
            print(f"  {tag} {ename}: norm {list(tensor.shape)} "
                  f"-> float32 ({nb:,} bytes)")

        elif cat == "linear":
            weight_f32 = tensor.astype(np.float32)
            rows, cols = weight_f32.shape

            # Ensure cols divisible by 4 (pad if needed)
            if cols % SPARSE_GROUP_SIZE != 0:
                pad = SPARSE_GROUP_SIZE - (cols % SPARSE_GROUP_SIZE)
                weight_f32 = np.pad(weight_f32, ((0, 0), (0, pad)))
                print(f"    Padded cols {cols} -> {cols + pad}")
                cols += pad

            base_path = weights_dir / ename
            # Create subdirectory if layer name contains dots before final part
            base_path.parent.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            entry = export_sparse_layer(weight_f32, base_path)
            dt = time.time() - t0

            # Fix file paths in manifest to be relative to output root
            entry["meta_file"] = f"weights/{entry['meta_file']}"
            entry["values_file"] = f"weights/{entry['values_file']}"
            manifest["weights"][ename] = entry

            layer_bytes = entry["meta_nbytes"] + entry["values_nbytes"]
            total_sparse += layer_bytes
            density = entry["stats"]["density_after"]
            gamma = entry["gamma"]
            print(
                f"  {tag} {ename}: {list(tensor.shape)} -> sparse ternary "
                f"({layer_bytes / 1e6:.2f} MB, gamma={gamma:.6f}, "
                f"density={density:.1%}) [{dt:.1f}s]"
            )

        elif cat == "lm_head":
            if inference_config["tie_word_embeddings"]:
                manifest["weights"][ename] = {
                    "tied_to": "embed_tokens",
                    "dtype": "tied",
                }
                print(f"  {tag} {ename}: tied to embed_tokens (skipped)")
            else:
                path = weights_dir / f"{ename}.bin"
                nb = export_float_tensor(tensor, path, np.float16)
                manifest["weights"][ename] = {
                    "file": f"weights/{ename}.bin",
                    "dtype": "float16",
                    "shape": list(tensor.shape),
                    "nbytes": nb,
                }
                total_float += nb
                print(f"  {tag} {ename}: lm_head {list(tensor.shape)} "
                      f"-> float16 ({nb / 1e6:.1f} MB)")

        else:
            print(f"  {tag} {ename}: SKIPPED (unknown category, "
                  f"shape={list(tensor.shape)})")

    # 5. Write manifest
    manifest_path = output / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nExported manifest -> {manifest_path}")

    # 6. Summary
    grand_total = total_sparse + total_float
    print(f"\n{'=' * 72}")
    print(f"Conversion complete: {output}")
    print(f"{'=' * 72}")
    print(f"  Sparse ternary weights: {total_sparse / 1e6:>8.1f} MB")
    print(f"  Float tensors:          {total_float / 1e6:>8.1f} MB")
    print(f"  Total on disk:          {grand_total / 1e6:>8.1f} MB")
    print(f"\nReady for C++ inference loader.")


def _copy_tokenizer(model_path: str, output: Path) -> None:
    """Copy tokenizer files from HuggingFace cache to output directory."""
    try:
        from huggingface_hub import snapshot_download

        tok_dir = Path(snapshot_download(
            model_path,
            allow_patterns=["tokenizer*", "special_tokens*"],
        ))
        copied = 0
        for pattern in ["tokenizer*", "special_tokens*"]:
            for tok_file in tok_dir.glob(pattern):
                if tok_file.is_file():
                    shutil.copy2(tok_file, output / tok_file.name)
                    copied += 1
        print(f"Copied {copied} tokenizer files -> {output}")
    except Exception as e:
        print(f"Warning: Could not copy tokenizer files: {e}")


# ---------------------------------------------------------------------------
# Verification: round-trip check against reference packing
# ---------------------------------------------------------------------------

def verify_packing(rows: int = 64, cols: int = 256, seed: int = 42) -> bool:
    """Verify vectorized packing matches the reference (loop-based) packing.

    Uses the reference implementation from generate_sparse_mask.py.
    """
    print(f"\nVerifying packing ({rows}x{cols}, seed={seed}) ...")

    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((rows, cols)).astype(np.float32)

    # Our vectorized path
    mask = generate_24_mask(latent)
    ternary, gamma = quantize_ternary_absmean(latent)
    sparse = ternary * mask.astype(np.int8)
    # Fix zeros at masked positions (same as export_sparse_layer)
    zero_at_mask = mask & (sparse == 0)
    sparse[zero_at_mask & (latent >= 0)] = +1
    sparse[zero_at_mask & (latent < 0)] = -1
    meta_v, values_v, ms, vs = pack_sparse_ternary(sparse, mask)

    # Reference path (from generate_sparse_mask.py)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_sparse_mask import (
            export_sparse_ternary,
            import_sparse_ternary,
        )
    except ImportError:
        print("  Skipped (generate_sparse_mask.py not found)")
        return True

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = str(Path(tmpdir) / "ref")
        export_sparse_ternary(sparse, mask, ref_path)

        # Read back reference
        ref_meta = np.fromfile(f"{ref_path}.meta", dtype=np.uint32).reshape(rows, ms)
        ref_values = np.fromfile(f"{ref_path}.values", dtype=np.uint32).reshape(rows, vs)

    meta_match = np.array_equal(meta_v, ref_meta)
    values_match = np.array_equal(values_v, ref_values)

    if meta_match and values_match:
        print("  Packing verification PASSED (matches reference)")
        return True
    else:
        mismatches_m = np.sum(meta_v != ref_meta)
        mismatches_v = np.sum(values_v != ref_values)
        print(f"  Packing verification FAILED!")
        print(f"    Meta mismatches:   {mismatches_m} / {meta_v.size}")
        print(f"    Values mismatches: {mismatches_v} / {values_v.size}")
        return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert HuggingFace BitNet model to spbitnet sparse ternary "
            "format for C++ inference."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HuggingFace model ID or local path (default: bf16 variant)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bitnet-2b-4t-sparse",
        help="Output directory (default: models/bitnet-2b-4t-sparse/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect model structure without converting",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip copying tokenizer files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run packing verification against reference implementation",
    )
    args = parser.parse_args()

    # --verify only needs numpy, skip torch/transformers check
    if args.verify:
        ok = verify_packing()
        sys.exit(0 if ok else 1)

    # Check dependencies for model loading
    try:
        import torch
        import transformers
        import huggingface_hub

        print(
            f"torch={torch.__version__}, "
            f"transformers={transformers.__version__}, "
            f"huggingface_hub={huggingface_hub.__version__}"
        )
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print(
            "Install with:\n"
            "  pip install torch transformers safetensors huggingface-hub",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        dry_run(args.model)
    else:
        convert_model(
            args.model,
            args.output,
            copy_tokenizer=not args.no_tokenizer,
        )


if __name__ == "__main__":
    main()
