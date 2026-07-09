"""
Phase 3 — Step 1: Download & Inspect Model

Downloads a model from HuggingFace and prints a summary of all weight
matrices (linear layers) that will be quantized in Step 2.

Usage:
  Change MODEL_CHOICE below to switch models, then run:
    python step1_inspect_model.py

  Or pass model name as argument:
    python step1_inspect_model.py mistral
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# ═══════════════════════════════════════════════════════════════════
#  MODEL SELECTION — Change this to switch models
# ═══════════════════════════════════════════════════════════════════
#  Options: "tinyllama", "phi3", "mistral", "llama2_7b"
# ═══════════════════════════════════════════════════════════════════

MODEL_CHOICE = "tinyllama"

# ── Model Registry ──
MODELS = {
    "tinyllama": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama 1.1B",
        "size": "~4.1 GB FP32",
    },
    "phi3": {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "name": "Phi-3 Mini 3.8B",
        "size": "~15 GB FP32",
    },
    "mistral": {
        "id": "mistralai/Mistral-7B-v0.1",
        "name": "Mistral 7B",
        "size": "~26 GB FP32",
    },
    "llama2_7b": {
        "id": "meta-llama/Llama-2-7b-hf",
        "name": "LLaMA-2 7B",
        "size": "~26 GB FP32",
    },
}


def format_size(num_bytes):
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def main():
    # Allow command-line override: python step1_inspect_model.py mistral
    choice = MODEL_CHOICE
    if len(sys.argv) > 1 and sys.argv[1] in MODELS:
        choice = sys.argv[1]

    if choice not in MODELS:
        print(f"Unknown model '{choice}'. Options: {list(MODELS.keys())}")
        sys.exit(1)

    model_info = MODELS[choice]
    MODEL_ID = model_info["id"]

    print("=" * 70)
    print(f"  Phase 3 — Step 1: Model Inspection")
    print(f"  Model: {model_info['name']} ({model_info['size']})")
    print("=" * 70)

    # ── Load config first (lightweight, no download) ──
    print(f"\n[1/3] Loading model config from {MODEL_ID}...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    print(f"  Architecture    : {config.architectures}")
    print(f"  Hidden size     : {config.hidden_size}")
    print(f"  Num layers      : {config.num_hidden_layers}")
    print(f"  Num attn heads  : {config.num_attention_heads}")
    kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    print(f"  Num KV heads    : {kv_heads}")
    inter_size = getattr(config, 'intermediate_size', 'N/A')
    print(f"  Intermediate    : {inter_size}")
    print(f"  Vocab size      : {config.vocab_size}")

    # ── Download and load the full model (CPU, FP32) ──
    print(f"\n[2/3] Downloading model weights (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # ── Enumerate all parameters ──
    print(f"\n[3/3] Scanning model parameters...\n")

    # Categorize layers
    categories = {
        "Embedding (lm_head / embed_tokens)": [],
        "Attention QKV projections": [],
        "Attention O projection": [],
        "FFN gate/up projections": [],
        "FFN down projection": [],
        "LayerNorm / RMSNorm": [],
        "Other": [],
    }

    total_params = 0
    linear_params = 0
    linear_layers = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        shape_str = "×".join(str(d) for d in param.shape)

        # Classify
        if "embed_tokens" in name or "lm_head" in name:
            cat = "Embedding (lm_head / embed_tokens)"
        elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
            cat = "Attention QKV projections"
        elif "o_proj" in name:
            cat = "Attention O projection"
        elif "gate_proj" in name or "up_proj" in name:
            cat = "FFN gate/up projections"
        elif "down_proj" in name:
            cat = "FFN down projection"
        elif "norm" in name.lower() or "layernorm" in name.lower():
            cat = "LayerNorm / RMSNorm"
        else:
            cat = "Other"

        categories[cat].append((name, param.shape, param.numel()))

        # Track linear (2-D) layers — these are what we'll quantize
        if param.dim() == 2:
            linear_params += param.numel()
            linear_layers.append((name, param.shape, param.numel()))

    # ── Print layer table ──
    print(f"{'Layer Name':<55} {'Shape':<20} {'Params':>12}")
    print("─" * 90)

    for cat_name, layers in categories.items():
        if not layers:
            continue
        cat_params = sum(p for _, _, p in layers)
        print(f"\n  ┌─ {cat_name}  ({len(layers)} tensors, {cat_params:,} params)")
        for name, shape, numel in layers:
            shape_str = "×".join(str(d) for d in shape)
            print(f"  │  {name:<52} {shape_str:<20} {numel:>12,}")
        print(f"  └─ Subtotal: {format_size(cat_params * 4)} (FP32)")

    # ── Summary ──
    fp32_bytes = total_params * 4
    linear_fp32_bytes = linear_params * 4

    # MXFP4 estimate: 4-bit data + 1 byte scale per 32 elements
    mxfp4_data_bits = linear_params * 4
    mxfp4_scale_bytes = (linear_params + 31) // 32 * 1
    mxfp4_total = mxfp4_data_bits // 8 + mxfp4_scale_bytes

    # NVFP4 estimate: 4-bit data + 1 byte scale per 16 elements + 4 bytes tensor scale
    nvfp4_data_bits = linear_params * 4
    nvfp4_scale_bytes = (linear_params + 15) // 16 * 1
    nvfp4_tensor_scales = len(linear_layers) * 4
    nvfp4_total = nvfp4_data_bits // 8 + nvfp4_scale_bytes + nvfp4_tensor_scales

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {model_info['name']}")
    print(f"{'=' * 70}")
    print(f"  Total parameters     : {total_params:>14,}")
    print(f"  Linear (2-D) params  : {linear_params:>14,}  ({100*linear_params/total_params:.1f}%)")
    print(f"  Non-linear params    : {total_params - linear_params:>14,}  (norms, kept in FP32)")
    print(f"")
    print(f"  FP32 model size      : {format_size(fp32_bytes):>14}")
    print(f"  MXFP4 estimated size : {format_size(mxfp4_total):>14}  ({mxfp4_total/linear_fp32_bytes*100:.1f}% of FP32 linear)")
    print(f"  NVFP4 estimated size : {format_size(nvfp4_total):>14}  ({nvfp4_total/linear_fp32_bytes*100:.1f}% of FP32 linear)")
    print(f"  Compression ratio    : MXFP4 = {linear_fp32_bytes/mxfp4_total:.1f}×  |  NVFP4 = {linear_fp32_bytes/nvfp4_total:.1f}×")
    print(f"")
    print(f"  Layers to quantize   : {len(linear_layers)}")
    print(f"{'=' * 70}")

    print(f"\n✓ Step 1 complete. Ready for Step 2 (quantize weights).\n")


if __name__ == "__main__":
    main()
