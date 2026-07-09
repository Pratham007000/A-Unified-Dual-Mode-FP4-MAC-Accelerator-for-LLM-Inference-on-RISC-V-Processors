"""
Phase 3 — Step 2: Quantize Model Weights (MXFP4 vs NVFP4)

Loads a model, quantizes every linear layer's weight matrix to both MXFP4
and NVFP4 using the Phase-1 quantizers, measures per-layer quantization
error, and saves quantized weights to disk.

Usage:
  Change MODEL_CHOICE below to switch models, then run:
    python step2_quantize_weights.py

  Or pass model name as argument:
    python step2_quantize_weights.py mistral

Prerequisites:
  - Phase-1 quantizer files (fp_utils.py, mxfp4.py, nvfp4.py) accessible
"""

import sys
import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

# ── Import Phase-1 quantizers ──
# Works both as a standalone script and inside a Kaggle/Jupyter notebook
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PHASE1_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Phase-1")
    sys.path.insert(0, PHASE1_DIR)
except NameError:
    pass  # __file__ not defined in notebooks — Phase-1 files should be in same dir

from mxfp4 import MXFP4Quantizer
from nvfp4 import NVFP4Quantizer

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
    },
    "phi3": {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "name": "Phi-3 Mini 3.8B",
    },
    "mistral": {
        "id": "mistralai/Mistral-7B-v0.1",
        "name": "Mistral 7B",
    },
    "llama2_7b": {
        "id": "meta-llama/Llama-2-7b-hf",
        "name": "LLaMA-2 7B",
    },
}


def format_size(num_bytes):
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def compute_metrics(original, reconstructed):
    """Compute MSE, RMSE, and relative error between original and reconstructed."""
    diff = original - reconstructed
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    rms_orig = np.sqrt(np.mean(original ** 2))
    rel_err = rmse / rms_orig if rms_orig > 0 else float('inf')
    return mse, rmse, rel_err


def classify_layer(name):
    """Classify a layer name into a category."""
    if "embed_tokens" in name or "lm_head" in name:
        return "Embedding"
    elif "q_proj" in name:
        return "Attn Q"
    elif "k_proj" in name:
        return "Attn K"
    elif "v_proj" in name:
        return "Attn V"
    elif "o_proj" in name:
        return "Attn O"
    elif "gate_proj" in name:
        return "FFN gate"
    elif "up_proj" in name:
        return "FFN up"
    elif "down_proj" in name:
        return "FFN down"
    else:
        return "Other"


def main():
    # Allow command-line override: python step2_quantize_weights.py mistral
    choice = MODEL_CHOICE
    if len(sys.argv) > 1 and sys.argv[1] in MODELS:
        choice = sys.argv[1]

    if choice not in MODELS:
        print(f"Unknown model '{choice}'. Options: {list(MODELS.keys())}")
        sys.exit(1)

    model_info = MODELS[choice]
    MODEL_ID = model_info["id"]
    model_short = choice  # used for output folder naming

    print("=" * 80)
    print(f"  Phase 3 — Step 2: Quantize Weights (MXFP4 vs NVFP4)")
    print(f"  Model: {model_info['name']}")
    print("=" * 80)

    # ── Create output directory (per-model) ──
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    OUTPUT_DIR = os.path.join(base_dir, "quantized_weights", model_short)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ──
    print(f"\n[1/4] Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("  ✓ Model loaded successfully")

    # ── Identify linear layers to quantize ──
    linear_layers = []
    for name, param in model.named_parameters():
        if param.dim() == 2:
            linear_layers.append((name, param))

    print(f"  Found {len(linear_layers)} linear layers to quantize\n")

    # ── Quantize each layer ──
    print(f"[2/4] Quantizing all linear layers...\n")

    mx_quant = MXFP4Quantizer()
    nv_quant = NVFP4Quantizer()

    results = []
    total_fp32_bytes = 0
    total_mxfp4_bytes = 0
    total_nvfp4_bytes = 0

    # Header
    print(f"  {'#':<4} {'Layer':<45} {'Shape':<16} {'MX RMSE':>10} {'NV RMSE':>10} {'Winner':>8}")
    print(f"  {'─'*4} {'─'*45} {'─'*16} {'─'*10} {'─'*10} {'─'*8}")

    for idx, (name, param) in enumerate(linear_layers):
        weight = param.detach().numpy().astype(np.float32)
        shape_str = f"{weight.shape[0]}×{weight.shape[1]}"
        numel = weight.size

        fp32_bytes = numel * 4
        total_fp32_bytes += fp32_bytes

        # ── MXFP4 quantize + dequantize ──
        mx_codes, mx_scales, mx_shape = mx_quant.quantize(weight)
        mx_recon = mx_quant.dequantize(mx_codes, mx_scales, mx_shape)
        mx_mse, mx_rmse, mx_rel = compute_metrics(weight, mx_recon)

        mx_packed_bytes = (numel + 1) // 2
        mx_scale_bytes = mx_scales.size
        mx_total_bytes = mx_packed_bytes + mx_scale_bytes
        total_mxfp4_bytes += mx_total_bytes

        # ── NVFP4 quantize + dequantize ──
        nv_codes, nv_bscales, nv_tscale, nv_shape = nv_quant.quantize(weight)
        nv_recon = nv_quant.dequantize(nv_codes, nv_bscales, nv_tscale, nv_shape)
        nv_mse, nv_rmse, nv_rel = compute_metrics(weight, nv_recon)

        nv_packed_bytes = (numel + 1) // 2
        nv_bscale_bytes = nv_bscales.size
        nv_total_bytes = nv_packed_bytes + nv_bscale_bytes + 4
        total_nvfp4_bytes += nv_total_bytes

        # Determine winner
        winner = "NVFP4" if nv_rmse < mx_rmse else "MXFP4"
        if abs(nv_rmse - mx_rmse) / max(mx_rmse, 1e-10) < 0.01:
            winner = "TIE"

        results.append({
            'name': name, 'shape': weight.shape, 'numel': numel,
            'category': classify_layer(name),
            'mx_rmse': mx_rmse, 'mx_rel': mx_rel,
            'nv_rmse': nv_rmse, 'nv_rel': nv_rel,
            'mx_bytes': mx_total_bytes, 'nv_bytes': nv_total_bytes,
            'fp32_bytes': fp32_bytes, 'winner': winner,
        })

        print(f"  {idx+1:<4} {name:<45} {shape_str:<16} {mx_rmse:>10.4e} {nv_rmse:>10.4e} {winner:>8}")

    # ── Save quantized weights ──
    print(f"\n[3/4] Saving quantized weights to {OUTPUT_DIR}/...")

    for idx, (name, param) in enumerate(linear_layers):
        weight = param.detach().numpy().astype(np.float32)
        safe_name = name.replace(".", "_")

        mx_codes, mx_scales, mx_shape = mx_quant.quantize(weight)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f"{safe_name}_mxfp4.npz"),
                            codes=mx_codes, scales=mx_scales,
                            original_shape=np.array(mx_shape))

        nv_codes, nv_bscales, nv_tscale, nv_shape = nv_quant.quantize(weight)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f"{safe_name}_nvfp4.npz"),
                            codes=nv_codes, block_scales=nv_bscales,
                            tensor_scale=np.array(nv_tscale),
                            original_shape=np.array(nv_shape))

    print(f"  ✓ Saved {len(linear_layers)} × 2 files (MXFP4 + NVFP4)")

    # ── Print summary ──
    print(f"\n[4/4] Summary\n")
    print(f"{'=' * 80}")

    # Accuracy summary by category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'mx_rmse': [], 'nv_rmse': [], 'mx_wins': 0, 'nv_wins': 0, 'ties': 0}
        categories[cat]['mx_rmse'].append(r['mx_rmse'])
        categories[cat]['nv_rmse'].append(r['nv_rmse'])
        if r['winner'] == 'MXFP4':
            categories[cat]['mx_wins'] += 1
        elif r['winner'] == 'NVFP4':
            categories[cat]['nv_wins'] += 1
        else:
            categories[cat]['ties'] += 1

    print(f"\n  Accuracy by Layer Type ({model_info['name']}):")
    print(f"  {'Category':<15} {'Avg MX RMSE':>12} {'Avg NV RMSE':>12} {'MX Wins':>8} {'NV Wins':>8}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*8} {'─'*8}")
    for cat in ['Embedding', 'Attn Q', 'Attn K', 'Attn V', 'Attn O',
                'FFN gate', 'FFN up', 'FFN down', 'Other']:
        if cat not in categories:
            continue
        c = categories[cat]
        avg_mx = np.mean(c['mx_rmse'])
        avg_nv = np.mean(c['nv_rmse'])
        print(f"  {cat:<15} {avg_mx:>12.4e} {avg_nv:>12.4e} {c['mx_wins']:>8} {c['nv_wins']:>8}")

    # Overall stats
    total_mx_wins = sum(1 for r in results if r['winner'] == 'MXFP4')
    total_nv_wins = sum(1 for r in results if r['winner'] == 'NVFP4')
    total_ties = sum(1 for r in results if r['winner'] == 'TIE')
    avg_mx_rmse = np.mean([r['mx_rmse'] for r in results])
    avg_nv_rmse = np.mean([r['nv_rmse'] for r in results])
    avg_mx_rel = np.mean([r['mx_rel'] for r in results])
    avg_nv_rel = np.mean([r['nv_rel'] for r in results])

    print(f"\n  Overall Accuracy ({model_info['name']}):")
    print(f"    MXFP4 avg RMSE     : {avg_mx_rmse:.6e}")
    print(f"    NVFP4 avg RMSE     : {avg_nv_rmse:.6e}")
    print(f"    MXFP4 avg rel error: {avg_mx_rel:.4%}")
    print(f"    NVFP4 avg rel error: {avg_nv_rel:.4%}")
    print(f"    NVFP4 wins {total_nv_wins}/{len(results)} layers | MXFP4 wins {total_mx_wins}/{len(results)} | Ties {total_ties}/{len(results)}")

    print(f"\n  Memory Footprint ({model_info['name']}):")
    print(f"    FP32 (baseline)    : {format_size(total_fp32_bytes)}")
    print(f"    MXFP4 total        : {format_size(total_mxfp4_bytes)}  ({total_mxfp4_bytes/total_fp32_bytes*100:.1f}% of FP32)")
    print(f"    NVFP4 total        : {format_size(total_nvfp4_bytes)}  ({total_nvfp4_bytes/total_fp32_bytes*100:.1f}% of FP32)")
    print(f"    Compression ratio  : MXFP4 = {total_fp32_bytes/total_mxfp4_bytes:.1f}×  |  NVFP4 = {total_fp32_bytes/total_nvfp4_bytes:.1f}×")

    print(f"\n  Saved weights        : {OUTPUT_DIR}/")
    print(f"{'=' * 80}")
    print(f"\n✓ Phase 3 complete for {model_info['name']}.\n")


if __name__ == "__main__":
    main()
