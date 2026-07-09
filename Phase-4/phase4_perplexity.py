"""
Phase 4 — Perplexity on WikiText-2
Loads TinyLlama, replaces linear weights with dequantized MXFP4/NVFP4,
measures perplexity on WikiText-2 test set.
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

MODEL_ID  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LEN   = 512   # context window per chunk

def load_quantized_model(mode):
    """
    mode: 'fp32', 'mxfp4', 'nvfp4'
    Returns model with weights replaced accordingly.
    """
    print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()

    if mode == "fp32":
        return model

    qm = MXFP4Quantizer()
    qn = NVFP4Quantizer()

    n_replaced = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() != 2:
                continue
            w = param.detach().float().numpy()

            if mode == "mxfp4":
                codes, scales, shape = qm.quantize(w)
                w_rec = qm.dequantize(codes, scales, shape)
            else:  # nvfp4
                codes, bscales, tscale, shape = qn.quantize(w)
                w_rec = qn.dequantize(codes, bscales, tscale, shape)

            param.copy_(torch.from_numpy(w_rec))
            n_replaced += 1

    print(f"  Replaced {n_replaced} linear layers with {mode.upper()} dequantized weights")
    return model


def compute_perplexity(model, tokenizer, text_corpus):
    """
    Compute perplexity over the full corpus using non-overlapping chunks.
    Concatenates all text, tokenizes, then evaluates in MAX_LEN-sized windows.
    """
    model.eval()

    # Tokenize the entire corpus at once
    encodings = tokenizer("\n\n".join(text_corpus), return_tensors="pt")
    input_ids = encodings.input_ids  # shape: [1, total_tokens]
    total_tokens = input_ids.shape[1]
    print(f"  Total tokens in corpus: {total_tokens}")

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


def main():
    print("=" * 60)
    print("  Phase 4 — WikiText-2 Perplexity (TinyLlama 1.1B)")
    print("=" * 60)

    # Load WikiText-2 test set
    print("\nLoading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Filter out empty/short lines
    texts = [row["text"] for row in dataset if len(row["text"].strip()) > 0]
    print(f"  {len(texts)} non-empty lines loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    for mode in ["fp32", "mxfp4", "nvfp4"]:
        print(f"\n{'─'*60}")
        print(f"  Running: {mode.upper()}")
        print(f"{'─'*60}")

        t0 = time.time()
        model = load_quantized_model(mode)
        ppl = compute_perplexity(model, tokenizer, texts)
        elapsed = time.time() - t0

        results[mode] = {"perplexity": round(ppl, 3), "time_sec": round(elapsed, 1)}
        print(f"  Perplexity: {ppl:.3f}  (took {elapsed:.1f}s)")

        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    fp32_ppl = results["fp32"]["perplexity"]
    mx_ppl = results["mxfp4"]["perplexity"]
    nv_ppl = results["nvfp4"]["perplexity"]

    print(f"\n{'='*60}")
    print(f"  WikiText-2 Perplexity — TinyLlama 1.1B")
    print(f"{'='*60}")
    print(f"  FP32  (baseline) : {fp32_ppl:.3f}")
    print(f"  MXFP4            : {mx_ppl:.3f}  (Δ = +{mx_ppl - fp32_ppl:.3f})")
    print(f"  NVFP4            : {nv_ppl:.3f}  (Δ = +{nv_ppl - fp32_ppl:.3f})")
    print(f"")
    print(f"  MXFP4 degradation: {(mx_ppl - fp32_ppl)/fp32_ppl*100:.2f}%")
    print(f"  NVFP4 degradation: {(nv_ppl - fp32_ppl)/fp32_ppl*100:.2f}%")
    print(f"  Winner           : {'NVFP4' if nv_ppl < mx_ppl else 'MXFP4'} (lower = better)")
    print(f"{'='*60}")

    out_path = "/kaggle/working/phase4_perplexity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
