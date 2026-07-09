    """
    Phase 5 — Unified MAC Mixed-Mode Perplexity on WikiText-2

    Simulates the unified MAC's per-layer format selection strategy:
    - Attention layers (Q, K, V, O) → NVFP4 (high precision needed)
    - FFN layers (gate, up, down)   → MXFP4 (tolerant to coarser quantization)

    Compares against:
    - FP32 baseline (normal RISC-V FP32 MAC)
    - All-MXFP4 (every layer uses MXFP4)
    - All-NVFP4 (every layer uses NVFP4)
    - Mixed/Unified (attention=NVFP4, FFN=MXFP4)

    This proves the unified MAC's adaptive strategy maintains NVFP4-level accuracy
    while reducing memory bandwidth on FFN layers.

    Run on Kaggle with GPU. Requires Phase-1 quantizer files (mxfp4.py, nvfp4.py, fp_utils.py).
    """
    import torch
    import numpy as np
    import json
    import os
    import gc
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from mxfp4 import MXFP4Quantizer
    from nvfp4 import NVFP4Quantizer

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MAX_LEN  = 512

    def classify_layer(name):
        """Determine if layer is attention or FFN."""
        attn_keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
        ffn_keys  = ["gate_proj", "up_proj", "down_proj"]
        for k in attn_keys:
            if k in name:
                return "attention"
        for k in ffn_keys:
            if k in name:
                return "ffn"
        return "other"


    def load_model_with_mode(mode):
        """
        Load model with weights replaced based on mode:
        'fp32'    - no quantization
        'mxfp4'   - all layers MXFP4
        'nvfp4'   - all layers NVFP4
        'unified' - attention=NVFP4, FFN=MXFP4
        """
        print(f"  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32,
            device_map="cpu", low_cpu_mem_usage=True
        )
        model.eval()

        if mode == "fp32":
            return model, {"attn_nvfp4": 0, "ffn_mxfp4": 0, "attn_mxfp4": 0, "ffn_nvfp4": 0}

        qm = MXFP4Quantizer()
        qn = NVFP4Quantizer()

        stats = {"attn_nvfp4": 0, "ffn_mxfp4": 0, "attn_mxfp4": 0, "ffn_nvfp4": 0, "other_nvfp4": 0}
        n_replaced = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() != 2:
                    continue

                w = param.detach().float().numpy()
                layer_type = classify_layer(name)

                # Decide format based on mode
                if mode == "mxfp4":
                    use_format = "mxfp4"
                elif mode == "nvfp4":
                    use_format = "nvfp4"
                elif mode == "unified":
                    if layer_type == "attention":
                        use_format = "nvfp4"   # attention needs precision
                    elif layer_type == "ffn":
                        use_format = "mxfp4"   # FFN tolerates coarser quant
                    else:
                        use_format = "nvfp4"   # default to safer option
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Quantize + dequantize
                if use_format == "mxfp4":
                    codes, scales, shape = qm.quantize(w)
                    w_rec = qm.dequantize(codes, scales, shape)
                    if layer_type == "ffn":
                        stats["ffn_mxfp4"] += 1
                    elif layer_type == "attention":
                        stats["attn_mxfp4"] += 1
                else:
                    codes, bscales, tscale, shape = qn.quantize(w)
                    w_rec = qn.dequantize(codes, bscales, tscale, shape)
                    if layer_type == "attention":
                        stats["attn_nvfp4"] += 1
                    elif layer_type == "ffn":
                        stats["ffn_nvfp4"] += 1
                    else:
                        stats["other_nvfp4"] += 1

                param.copy_(torch.from_numpy(w_rec))
                n_replaced += 1

        print(f"  Replaced {n_replaced} linear layers")
        if mode == "unified":
            print(f"    Attention layers → NVFP4: {stats['attn_nvfp4']}")
            print(f"    FFN layers       → MXFP4: {stats['ffn_mxfp4']}")
            if stats.get('other_nvfp4', 0) > 0:
                print(f"    Other layers     → NVFP4: {stats['other_nvfp4']}")

        return model, stats


    def compute_perplexity(model, tokenizer, text_corpus):
        """Compute perplexity over WikiText-2 using non-overlapping chunks."""
        model.eval()

        encodings = tokenizer("\n\n".join(text_corpus), return_tensors="pt")
        input_ids = encodings.input_ids
        total_tokens = input_ids.shape[1]
        print(f"  Total tokens: {total_tokens}")

        total_nll = 0.0
        total_count = 0
        n_chunks = (total_tokens - 1) // MAX_LEN

        for i in range(n_chunks):
            start = i * MAX_LEN
            end = min(start + MAX_LEN, total_tokens)
            chunk_ids = input_ids[:, start:end]

            if chunk_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                outputs = model(chunk_ids, labels=chunk_ids)

            n_tokens = chunk_ids.shape[1] - 1
            total_nll += outputs.loss.item() * n_tokens
            total_count += n_tokens

            if (i + 1) % 20 == 0:
                running_ppl = np.exp(total_nll / total_count)
                print(f"    Chunk {i+1}/{n_chunks} | running PPL: {running_ppl:.2f}")

        avg_nll = total_nll / total_count
        ppl = float(np.exp(avg_nll))
        print(f"  Evaluated {total_count} tokens across {n_chunks} chunks")
        return ppl


    def compute_mixed_memory(model, stats_unified):
        """Estimate memory usage of mixed-mode quantization."""
        total_fp32 = 0
        total_mixed = 0

        for name, param in model.named_parameters():
            if param.dim() != 2:
                continue
            numel = param.numel()
            fp32_bytes = numel * 4
            total_fp32 += fp32_bytes

            layer_type = classify_layer(name)
            if layer_type == "ffn":
                # MXFP4: 4 bits per element + 1 byte per 32 elements
                packed = (numel + 1) // 2
                scales = numel // 32
                total_mixed += packed + scales
            else:
                # NVFP4: 4 bits per element + 1 byte per 16 elements + 4 bytes tensor scale
                packed = (numel + 1) // 2
                scales = numel // 16
                total_mixed += packed + scales + 4

        return total_fp32, total_mixed


    def main():
        print("=" * 70)
        print("  Phase 5 — Unified MAC Mixed-Mode Perplexity")
        print("  Strategy: Attention=NVFP4 | FFN=MXFP4")
        print("  Model: TinyLlama 1.1B | Dataset: WikiText-2")
        print("=" * 70)

        # Load dataset
        print("\nLoading WikiText-2 test set...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [row["text"] for row in dataset if len(row["text"].strip()) > 0]
        print(f"  {len(texts)} non-empty lines")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results = {}

        # Run all 4 modes
        for mode in ["fp32", "mxfp4", "nvfp4", "unified"]:
            print(f"\n{'─'*70}")
            label = mode.upper()
            if mode == "unified":
                label = "UNIFIED (Attn=NVFP4, FFN=MXFP4)"
            print(f"  Running: {label}")
            print(f"{'─'*70}")

            t0 = time.time()
            model, stats = load_model_with_mode(mode)
            ppl = compute_perplexity(model, tokenizer, texts)
            elapsed = time.time() - t0

            results[mode] = {
                "perplexity": round(ppl, 3),
                "time_sec": round(elapsed, 1),
                "layer_stats": stats
            }
            print(f"  Perplexity: {ppl:.3f}  (took {elapsed:.1f}s)")

            # Compute memory for unified
            if mode == "unified":
                fp32_bytes, mixed_bytes = compute_mixed_memory(model, stats)
                results[mode]["fp32_bytes"] = fp32_bytes
                results[mode]["mixed_bytes"] = mixed_bytes

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Final Summary ──
        fp32_ppl = results["fp32"]["perplexity"]
        mx_ppl   = results["mxfp4"]["perplexity"]
        nv_ppl   = results["nvfp4"]["perplexity"]
        uni_ppl  = results["unified"]["perplexity"]

        print(f"\n{'='*70}")
        print(f"  RESULTS — WikiText-2 Perplexity (TinyLlama 1.1B)")
        print(f"{'='*70}")
        print(f"")
        print(f"  {'Mode':<35} {'Perplexity':>12} {'Degradation':>14}")
        print(f"  {'─'*35} {'─'*12} {'─'*14}")
        print(f"  {'FP32 (normal RISC-V FP32 MAC)':<35} {fp32_ppl:>12.3f} {'baseline':>14}")
        print(f"  {'All-NVFP4':<35} {nv_ppl:>12.3f} {f'+{(nv_ppl-fp32_ppl)/fp32_ppl*100:.1f}%':>14}")
        print(f"  {'UNIFIED (Attn=NVFP4, FFN=MXFP4)':<35} {uni_ppl:>12.3f} {f'+{(uni_ppl-fp32_ppl)/fp32_ppl*100:.1f}%':>14}")
        print(f"  {'All-MXFP4':<35} {mx_ppl:>12.3f} {f'+{(mx_ppl-fp32_ppl)/fp32_ppl*100:.1f}%':>14}")
        print(f"")
        print(f"  Key Findings:")
        print(f"    • FP32 baseline (RISC-V FPU)    : {fp32_ppl:.3f}")
        print(f"    • Unified MAC degradation        : +{(uni_ppl-fp32_ppl)/fp32_ppl*100:.1f}% vs FP32")
        print(f"    • Unified vs All-NVFP4 difference: {abs(uni_ppl-nv_ppl):.3f} perplexity points")

        if uni_ppl < mx_ppl * 0.5:
            print(f"    • Unified MAC AVOIDS MXFP4's catastrophic collapse!")
        if abs(uni_ppl - nv_ppl) / nv_ppl < 0.05:
            print(f"    • Unified MAC matches NVFP4 accuracy (within 5%)!")

        if "mixed_bytes" in results["unified"]:
            fp32_b = results["unified"]["fp32_bytes"]
            mixed_b = results["unified"]["mixed_bytes"]
            print(f"")
            print(f"  Memory Footprint:")
            print(f"    FP32     : {fp32_b/1024/1024:.1f} MB")
            print(f"    Mixed    : {mixed_b/1024/1024:.1f} MB ({fp32_b/mixed_b:.1f}x compression)")

        print(f"{'='*70}")

        # Save results
        out_path = "/kaggle/working/phase5_unified_perplexity.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out_path}")


    if __name__ == "__main__":
        main()
