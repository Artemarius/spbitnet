#!/usr/bin/env python3
"""baseline_pytorch.py — PyTorch reference for spbitnet C++ inference validation.

Loads the bf16 BitNet model from HuggingFace, applies identical 2:4 sparsity
and ternary quantization as convert_model.py, runs greedy decoding with the
same BitLinear pipeline as the C++ engine, and compares token-by-token.

The forward pass replicates the exact numerical behaviour of the C++ kernels:
  - RMSNorm: float32 compute, float16 I/O
  - BitLinear: absmax INT8 quantize → integer GEMV → dequantize
  - RoPE: float32 trig, float16 storage
  - GQA attention with KV-cache
  - ReLU²-gated MLP

Usage:
    # Generate reference output:
    python baseline_pytorch.py \\
        --model microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --prompt "Hello" --max-tokens 32

    # Save tokens/logits for comparison:
    python baseline_pytorch.py \\
        --model microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --prompt "Hello" --max-tokens 32 \\
        --output ref_output.json --dump-logits ref_logits.bin

    # Compare against C++ output JSON:
    python baseline_pytorch.py \\
        --model microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --prompt "Hello" --max-tokens 32 \\
        --compare cpp_output.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Import quantization functions from convert_model.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from convert_model import generate_24_mask, quantize_ternary_absmean


# ---------------------------------------------------------------------------
# Sparse BitLinear layer
# ---------------------------------------------------------------------------

class SparseBitLinear:
    """Sparse ternary linear layer matching C++ BitLinear pipeline exactly.

    Three-stage computation (see inference.cu::bitlinear):
      1. Absmax quantize activation: half → int8  (inference_kernels.cu)
      2. Sparse ternary GEMV: int8 × {-1,0,+1} → int32  (sparse_ternary.cu)
      3. Dequantize: int32 → half, scale = gamma * absmax / 127  (inference_kernels.cu)
    """

    def __init__(self, weight_f32: np.ndarray):
        rows, cols = weight_f32.shape

        # Apply same 2:4 sparsity + ternary quantization as convert_model.py
        mask = generate_24_mask(weight_f32)
        ternary, gamma = quantize_ternary_absmean(weight_f32)
        sparse_ternary = ternary * mask.astype(np.int8)

        # Fix zeros at masked positions — the packed format can't represent
        # ternary 0 at a bitmap-marked position, so force ±1 from float sign.
        zero_at_mask = mask & (sparse_ternary == 0)
        sparse_ternary[zero_at_mask & (weight_f32 >= 0)] = +1
        sparse_ternary[zero_at_mask & (weight_f32 < 0)] = -1

        # Store as float32 for matmul — exact for integers up to 2^24.
        # Max accumulator value per row: (cols/2) * 127 ≈ 163K << 2^24.
        self.weight = torch.from_numpy(sparse_ternary.astype(np.float32))
        self.gamma = gamma
        self.rows = rows
        self.cols = cols

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BitLinear forward matching C++ absmax_quantize → GEMV → dequantize.

        Input:  float16 tensor, shape (cols,)
        Output: float16 tensor, shape (rows,)
        """
        # 1. Absmax INT8 quantization (matches absmax_quantize_kernel)
        x_f32 = x.float()
        absmax = x_f32.abs().max().item()
        scale = 127.0 / absmax if absmax > 0 else 0.0
        x_q = (x_f32 * scale).round().clamp(-128, 127)  # float32 (exact ints)

        # 2. Sparse ternary GEMV — integer arithmetic via float32 matmul
        # W values are {-1, 0, +1}, x_q values are int8 range → no precision loss
        y = self.weight @ x_q  # (rows,) float32, exact integer values

        # 3. Dequantize (matches dequantize_kernel)
        dequant_scale = self.gamma * absmax / 127.0
        return (y * dequant_scale).half()


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: y[i] = x[i] * w[i] * rsqrt(mean(x^2) + eps).

    Matches rms_norm_kernel: float32 compute, float16 input/output, float32 weight.
    Input can be float16 or float32 (after relu2_mul_f32). Output is always float16.
    """
    x_f32 = x.float()
    scale = torch.rsqrt(torch.mean(x_f32 * x_f32) + eps)
    return (x_f32 * scale * weight).half()


def rope_inplace(vec: torch.Tensor, num_heads: int, head_dim: int,
                 pos: int, theta: float) -> None:
    """RoPE applied in-place. Matches rope_kernel (float32 trig, float16 I/O)."""
    vec_2d = vec.view(num_heads, head_dim)
    half_dim = head_dim // 2

    # Compute angles in float32 (matching C++ powf/cosf/sinf)
    pair_idx = torch.arange(half_dim, dtype=torch.float32)
    freqs = 1.0 / torch.pow(
        torch.tensor(theta, dtype=torch.float32),
        2.0 * pair_idx / float(head_dim),
    )
    angles = float(pos) * freqs
    cos_a = angles.cos()  # (half_dim,)
    sin_a = angles.sin()

    # Apply rotation across all heads simultaneously
    x0 = vec_2d[:, 0::2].float()  # (num_heads, half_dim)
    x1 = vec_2d[:, 1::2].float()

    vec_2d[:, 0::2] = (x0 * cos_a - x1 * sin_a).half()
    vec_2d[:, 1::2] = (x0 * sin_a + x1 * cos_a).half()


def attention(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    scale: float,
) -> torch.Tensor:
    """Multi-head attention with GQA and KV-cache.

    Matches attention_scores_kernel + softmax_kernel + attention_output_kernel.

    Q:       (num_heads * head_dim,) float16
    K_cache: (num_kv_heads, max_seq, head_dim) float16
    V_cache: (num_kv_heads, max_seq, head_dim) float16
    seq_len: number of valid positions (including current)
    Returns: (num_heads * head_dim,) float16
    """
    Q_heads = Q.view(num_heads, head_dim)
    output = torch.zeros(num_heads, head_dim, dtype=torch.float16)

    for h in range(num_heads):
        # GQA mapping: integer division matches C++ exactly
        kv_h = h * num_kv_heads // num_heads

        # Attention scores: Q[h] · K[kv_h, 0:seq_len]^T * scale
        q_vec = Q_heads[h].float()                    # (head_dim,)
        k_mat = K_cache[kv_h, :seq_len].float()       # (seq_len, head_dim)
        scores = (k_mat @ q_vec) * scale               # (seq_len,) float32

        # Stable softmax (matches softmax_kernel 3-pass algorithm)
        scores = F.softmax(scores, dim=0)

        # Weighted sum of V (matches attention_output_kernel)
        v_mat = V_cache[kv_h, :seq_len].float()       # (seq_len, head_dim)
        out_h = scores @ v_mat                          # (head_dim,) float32
        output[h] = out_h.half()

    return output.view(-1)


def relu2_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ReLU²(gate) * up. Returns float32 to avoid float16 overflow.

    relu²(x) can produce values >> 65504 (float16 max) before sub-normalization.
    Keeping the output in float32 matches the C++ relu2_mul_f32_kernel.
    """
    g = gate.float()
    u = up.float()
    relu = torch.clamp(g, min=0.0)
    return relu * relu * u  # float32 — NOT .half()


def half_gemv(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Float16 GEMV with float32 accumulation. Matches half_gemv_kernel.
    Returns float32 logits."""
    return W.float() @ x.float()


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

class BitNetModel:
    """Holds all quantized/sparse weights for the BitNet model."""

    def __init__(self, state_dict: Dict[str, np.ndarray], config: Dict[str, Any]):
        self.hidden_size: int = config["hidden_size"]
        self.num_layers: int = config["num_hidden_layers"]
        self.num_heads: int = config["num_attention_heads"]
        self.num_kv_heads: int = config.get("num_key_value_heads", self.num_heads)
        self.head_dim: int = self.hidden_size // self.num_heads
        self.intermediate_size: int = config["intermediate_size"]
        self.vocab_size: int = config["vocab_size"]
        self.max_seq: int = config["max_position_embeddings"]
        self.rms_norm_eps: float = config.get("rms_norm_eps", 1e-5)
        self.rope_theta: float = config.get("rope_theta", 10000.0)
        self.tie_embeddings: bool = config.get("tie_word_embeddings", False)
        self.bos_token_id: int = config.get("bos_token_id", 1)
        self.eos_token_id: int = config.get("eos_token_id", 2)

        print(
            f"BitNetModel: {self.num_layers}L, H={self.hidden_size}, "
            f"heads={self.num_heads}/{self.num_kv_heads}, "
            f"inter={self.intermediate_size}, vocab={self.vocab_size}"
        )

        # --- Embedding (float16) ---
        embed_key = _find_key(state_dict, ["model.embed_tokens.weight",
                                            "embed_tokens.weight"])
        self.embed_tokens = torch.from_numpy(
            state_dict[embed_key].astype(np.float16))

        # --- LM head (float16, possibly tied) ---
        if self.tie_embeddings:
            self.lm_head = self.embed_tokens
        else:
            lm_key = _find_key(state_dict, ["lm_head.weight"])
            self.lm_head = torch.from_numpy(
                state_dict[lm_key].astype(np.float16))

        # --- Final norm (float32) ---
        norm_key = _find_key(state_dict, ["model.norm.weight", "norm.weight"])
        self.final_norm = torch.from_numpy(
            state_dict[norm_key].astype(np.float32))

        # --- Transformer layers ---
        print("Quantizing linear layers (2:4 sparsity + ternary) ...")
        self.layers: List[Dict[str, Any]] = []
        t0 = time.time()
        for i in range(self.num_layers):
            prefix = f"model.layers.{i}"
            layer = self._build_layer(state_dict, prefix)
            self.layers.append(layer)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (self.num_layers - i - 1)
            print(f"  Layer {i + 1}/{self.num_layers} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        print(f"Model ready ({time.time() - t0:.0f}s total)")

    def _build_layer(
        self, sd: Dict[str, np.ndarray], prefix: str
    ) -> Dict[str, Any]:
        layer: Dict[str, Any] = {}

        # Norms (float32)
        layer["input_ln"] = torch.from_numpy(
            sd[f"{prefix}.input_layernorm.weight"].astype(np.float32))
        layer["post_attn_ln"] = torch.from_numpy(
            sd[f"{prefix}.post_attention_layernorm.weight"].astype(np.float32))

        # Linear layers → SparseBitLinear
        for name in [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
            "mlp.down_proj",
        ]:
            w = sd[f"{prefix}.{name}.weight"].astype(np.float32)
            layer[name] = SparseBitLinear(w)

        # Sub-normalization weights (SubLN — applied after o_proj / down_proj)
        attn_sub_key = f"{prefix}.self_attn.attn_sub_norm.weight"
        if attn_sub_key in sd:
            layer["attn_sub_norm"] = torch.from_numpy(
                sd[attn_sub_key].astype(np.float32))

        ffn_sub_key = f"{prefix}.mlp.ffn_sub_norm.weight"
        if ffn_sub_key in sd:
            layer["ffn_sub_norm"] = torch.from_numpy(
                sd[ffn_sub_key].astype(np.float32))

        return layer


def _find_key(d: dict, candidates: list) -> str:
    for key in candidates:
        if key in d:
            return key
    raise KeyError(f"None of {candidates} found in state dict")


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Autoregressive inference matching C++ InferenceEngine exactly."""

    def __init__(self, model: BitNetModel):
        self.m = model
        self.seq_len = 0

        # KV caches: (num_kv_heads, max_seq, head_dim) float16 per layer
        self.k_cache = [
            torch.zeros(model.num_kv_heads, model.max_seq, model.head_dim,
                         dtype=torch.float16)
            for _ in range(model.num_layers)
        ]
        self.v_cache = [
            torch.zeros(model.num_kv_heads, model.max_seq, model.head_dim,
                         dtype=torch.float16)
            for _ in range(model.num_layers)
        ]

    def reset(self) -> None:
        self.seq_len = 0
        for l in range(self.m.num_layers):
            self.k_cache[l].zero_()
            self.v_cache[l].zero_()

    def forward(self, token_id: int) -> torch.Tensor:
        """Forward pass for one token. Returns logits (vocab_size,) float32."""
        m = self.m
        pos = self.seq_len
        seq_len = pos + 1

        # ===== 1. Embedding lookup =====
        x = m.embed_tokens[token_id].clone()  # (hidden_size,) float16

        # ===== 2. Transformer layers =====
        for l in range(m.num_layers):
            layer = m.layers[l]

            # -- Save residual --
            residual = x.clone()

            # -- Pre-attention RMSNorm --
            x = rms_norm(residual, layer["input_ln"], m.rms_norm_eps)

            # -- Q, K projections via BitLinear --
            q = layer["self_attn.q_proj"](x)   # (num_heads * head_dim,)
            k = layer["self_attn.k_proj"](x)   # (num_kv_heads * head_dim,)

            # -- RoPE (in-place) --
            rope_inplace(q, m.num_heads, m.head_dim, pos, m.rope_theta)
            rope_inplace(k, m.num_kv_heads, m.head_dim, pos, m.rope_theta)

            # -- Store K in cache --
            self.k_cache[l][:, pos, :] = k.view(m.num_kv_heads, m.head_dim)

            # -- V projection (K already cached, safe to reuse buffer) --
            v = layer["self_attn.v_proj"](x)
            self.v_cache[l][:, pos, :] = v.view(m.num_kv_heads, m.head_dim)

            # -- Attention --
            attn_scale = 1.0 / (m.head_dim ** 0.5)
            attn_out = attention(
                q, self.k_cache[l], self.v_cache[l],
                m.num_heads, m.num_kv_heads, m.head_dim,
                seq_len, attn_scale,
            )

            # -- Output projection --
            x = layer["self_attn.o_proj"](attn_out)

            # -- SubLN: attention sub-normalization --
            if "attn_sub_norm" in layer:
                x = rms_norm(x, layer["attn_sub_norm"], m.rms_norm_eps)

            # -- Residual add (float32 compute, float16 output) --
            x = (x.float() + residual.float()).half()

            # -- Save residual for MLP --
            residual = x.clone()

            # -- Post-attention RMSNorm --
            x = rms_norm(residual, layer["post_attn_ln"], m.rms_norm_eps)

            # -- MLP: gated with ReLU² --
            gate = layer["mlp.gate_proj"](x)
            up = layer["mlp.up_proj"](x)
            mlp_out = relu2_mul(gate, up)  # float32 (avoids float16 overflow)

            # -- SubLN: FFN sub-normalization (float32 → half) --
            if "ffn_sub_norm" in layer:
                mlp_out = rms_norm(mlp_out, layer["ffn_sub_norm"], m.rms_norm_eps)
            else:
                mlp_out = mlp_out.half()

            x = layer["mlp.down_proj"](mlp_out)

            # -- Residual add --
            x = (x.float() + residual.float()).half()

        # ===== 3. Final RMSNorm =====
        x = rms_norm(x, m.final_norm, m.rms_norm_eps)

        # ===== 4. LM head logits =====
        logits = half_gemv(m.lm_head, x)  # (vocab_size,) float32

        self.seq_len += 1
        return logits


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    engine: InferenceEngine,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    bos_id: int,
    eos_id: int,
    dump_logits_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Greedy decoding matching C++ main.cpp logic."""

    # Tokenize (no special tokens — BOS added manually, matching C++)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f'Prompt: "{prompt}" -> {len(prompt_ids)} tokens {prompt_ids}')
    print(f"Max tokens: {max_tokens}\n")

    # Input sequence: BOS + prompt
    input_ids = [bos_id] + list(prompt_ids)

    # Prefill: process all input tokens except the last
    print("Prefilling ...", end="", flush=True)
    t0 = time.time()
    for i in range(len(input_ids) - 1):
        engine.forward(input_ids[i])
    prefill_time = time.time() - t0
    print(f" done ({prefill_time:.1f}s, {len(input_ids)-1} tokens)")

    # Generation starts from the last input token
    token = input_ids[-1]

    generated: List[int] = []
    all_logits: List[np.ndarray] = []
    top_k_per_step: List[Dict[str, Any]] = []

    print(f"\n--- Generation ---")
    print(prompt, end="", flush=True)

    gen_t0 = time.time()
    for t in range(max_tokens):
        step_t0 = time.time()
        logits = engine.forward(token)
        step_dt = time.time() - step_t0

        if dump_logits_path:
            all_logits.append(logits.numpy().copy())

        # Top-5 for debugging
        topk_vals, topk_ids = logits.topk(5)
        top_k_per_step.append({
            "step": t,
            "input_token": int(token),
            "top5_ids": topk_ids.tolist(),
            "top5_logits": [round(v, 4) for v in topk_vals.tolist()],
            "time_ms": round(step_dt * 1000, 1),
        })

        # Greedy argmax
        token = int(logits.argmax())
        generated.append(token)

        # Stream decoded text
        text = tokenizer.decode([token])
        print(text, end="", flush=True)

        if token == eos_id:
            break

    gen_time = time.time() - gen_t0
    tok_per_sec = len(generated) / gen_time if gen_time > 0 else 0

    print(f"\n\n--- {len(generated)} tokens in {gen_time:.1f}s "
          f"({tok_per_sec:.2f} tok/s) ---")

    # Dump logits if requested
    if dump_logits_path and all_logits:
        logits_array = np.stack(all_logits)  # (num_steps, vocab_size)
        logits_array.tofile(dump_logits_path)
        print(f"Logits saved: {dump_logits_path} "
              f"(shape={logits_array.shape}, {logits_array.nbytes / 1e6:.1f} MB)")

    return {
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "input_ids": input_ids,
        "generated_ids": generated,
        "generated_text": tokenizer.decode(generated),
        "top_k": top_k_per_step,
        "prefill_time_s": round(prefill_time, 2),
        "generation_time_s": round(gen_time, 2),
        "tokens_per_second": round(tok_per_sec, 2),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_outputs(ref: Dict[str, Any], cpp_path: str) -> bool:
    """Compare PyTorch reference tokens against C++ output. Returns True if pass.

    Uses greedy comparison: reports agreement until first divergence.
    """

    with open(cpp_path, "r") as f:
        cpp_data = json.load(f)

    ref_ids = ref["generated_ids"]
    cpp_ids = cpp_data["generated_ids"]
    total = max(len(ref_ids), len(cpp_ids))

    if total == 0:
        print("No tokens to compare.")
        return True

    print(f"\n{'=' * 64}")
    print(f"Token-by-token comparison (PyTorch vs C++)")
    print(f"{'=' * 64}")

    matches = 0
    first_mismatch = -1

    for i in range(total):
        ref_tok = ref_ids[i] if i < len(ref_ids) else None
        cpp_tok = cpp_ids[i] if i < len(cpp_ids) else None
        match = ref_tok == cpp_tok

        if match:
            matches += 1
        elif first_mismatch < 0:
            first_mismatch = i

        status = "  OK" if match else "MISS"
        print(f"  [{i:3d}] ref={ref_tok!s:>8}  cpp={cpp_tok!s:>8}  {status}")

    agreement = matches / total * 100
    print(f"\nGreedy agreement: {matches}/{total} ({agreement:.1f}%)")

    if first_mismatch >= 0:
        print(f"First mismatch at step {first_mismatch} "
              f"({first_mismatch} consecutive matches before divergence)")

    passed = agreement >= 95.0
    print(f"\n{'PASS' if passed else 'FAIL'}: "
          f"threshold is 95% top-1 token agreement")
    return passed


def teacher_forced_compare(
    engine: "InferenceEngine",
    cpp_path: str,
    max_tokens: int,
) -> bool:
    """Teacher-forced comparison: feed same tokens, compare top-1 at each step.

    Uses the C++ output token sequence as the forced input at each step.
    Reports whether the Python model would have produced the same top-1 at each
    step given the same input history.
    """
    with open(cpp_path, "r") as f:
        cpp_data = json.load(f)

    cpp_gen = cpp_data["generated_ids"]
    cpp_input = cpp_data["input_ids"]
    total = min(len(cpp_gen), max_tokens)

    if total == 0:
        print("No tokens to compare with teacher forcing.")
        return True

    engine.reset()

    # Prefill: process all input tokens except the last
    for i in range(len(cpp_input) - 1):
        engine.forward(cpp_input[i])

    # Start from last input token
    token = cpp_input[-1]

    print(f"\n{'=' * 64}")
    print(f"Teacher-forced comparison ({total} steps)")
    print(f"{'=' * 64}")

    matches = 0
    top5_matches = 0

    for t in range(total):
        logits = engine.forward(token)
        py_top1 = int(logits.argmax())
        py_top5 = logits.topk(5).indices.tolist()
        cpp_tok = cpp_gen[t]

        match = py_top1 == cpp_tok
        in_top5 = cpp_tok in py_top5

        if match:
            matches += 1
        if in_top5:
            top5_matches += 1

        status = "  OK" if match else ("+TOP5" if in_top5 else " MISS")
        print(f"  [{t:3d}] py_top1={py_top1:>8}  cpp={cpp_tok:>8}  {status}")

        # Teacher forcing: always use the C++ token for next step
        token = cpp_tok

    top1_pct = matches / total * 100
    top5_pct = top5_matches / total * 100
    print(f"\nTeacher-forced top-1 agreement: {matches}/{total} ({top1_pct:.1f}%)")
    print(f"Teacher-forced top-5 agreement: {top5_matches}/{total} ({top5_pct:.1f}%)")

    passed = top1_pct >= 95.0
    print(f"\n{'PASS' if passed else 'FAIL'}: "
          f"threshold is 95% top-1 token agreement (teacher-forced)")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch reference for spbitnet C++ inference validation")
    parser.add_argument(
        "--model", type=str,
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HuggingFace model ID or local path (default: bf16 variant)")
    parser.add_argument(
        "--prompt", type=str, default="Hello",
        help="Prompt text (default: Hello)")
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="Max tokens to generate (default: 32)")
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Path to C++ output JSON for greedy comparison")
    parser.add_argument(
        "--teacher-force", type=str, default=None,
        help="Path to C++ output JSON for teacher-forced comparison "
             "(feeds C++ tokens, compares logit top-1)")
    parser.add_argument(
        "--dump-logits", type=str, default=None,
        help="Dump raw logits to binary file (float32, shape: steps x vocab)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file")
    args = parser.parse_args()

    print("spbitnet — PyTorch Reference Validation")
    print("=" * 40)
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    print()

    # Load model weights (float32 numpy arrays)
    from convert_model import load_model_weights
    state_dict, config = load_model_weights(args.model)

    # Build quantized model (applies 2:4 sparsity + ternary quantization)
    model = BitNetModel(state_dict, config)
    del state_dict  # free ~10 GB

    # Create inference engine
    engine = InferenceEngine(model)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    print(f"Tokenizer: {len(tokenizer)} entries\n")

    # Run generation
    result = generate(
        engine, tokenizer, args.prompt, args.max_tokens,
        model.bos_token_id, model.eos_token_id,
        args.dump_logits,
    )

    # Print top-5 per step
    print(f"\nPer-step details:")
    for info in result["top_k"]:
        ids_str = ", ".join(str(x) for x in info["top5_ids"])
        vals_str = ", ".join(f"{v:.1f}" for v in info["top5_logits"][:3])
        print(f"  step {info['step']:3d}: "
              f"in={info['input_token']:>8} -> [{ids_str}] ({vals_str}...)")

    # Save output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Compare against C++ output
    if args.compare:
        passed = compare_outputs(result, args.compare)
        if not args.teacher_force:
            sys.exit(0 if passed else 1)

    # Teacher-forced comparison (more rigorous — same input at every step)
    if args.teacher_force:
        passed = teacher_forced_compare(engine, args.teacher_force, args.max_tokens)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
