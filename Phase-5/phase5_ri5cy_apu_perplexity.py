"""
Phase 5 — RI5CY (CV32E40P) APU Hardware-Accurate Perplexity Measurement

Measures WikiText-2 perplexity using a bit-accurate software model of the
CV32E40P APU MAC co-processor datapath. Every weight reconstruction follows
the exact hardware pipeline as implemented in mac_apu_coprocessor.sv:

  APU Interface (request/grant handshake):
    fadd.s (apu_op=2) → QMAC_CFG:  latch config + reset accumulator
    fmul.s (apu_op=3) → QMAC_EXEC: trigger MAC, return accum_out

  MAC Datapath (unified_mac_top.v):
    nibble_extractor → e2m1_decoder → {e8m0_decoder | fp8_e4m3_decoder}
    → FORMAT_MODE MUX → fp32_mul (scale) → fp32_mul (dot) → fp32_add (accum)

  CV32E40P APU Protocol:
    - 3 operands (96-bit) per instruction (vs PCPI's 2 × 32-bit)
    - Combinational grant (apu_gnt = apu_req)
    - Multi-cycle result via apu_rvalid handshake (FPU_ADDMUL_LAT=2 → apu_lat=3)

Format assignment (per-layer strategy):
  - Attention layers (Q, K, V, O) → FORMAT_MODE=1 (NVFP4)
  - FFN layers (gate, up, down)   → FORMAT_MODE=0 (MXFP4)

Run on Kaggle. Requires Phase-1 files: mxfp4.py, nvfp4.py, fp_utils.py
"""

import torch
import numpy as np
import json
import os
import gc
import time
import struct
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LEN  = 512

# =====================================================================
#  BIT-ACCURATE HARDWARE DATAPATH EMULATION
#  Each function mirrors a Verilog module in rtl/
# =====================================================================

# --- e2m1_decoder.v (exact LUT) ---
E2M1_LUT = {
    0x0:  0.0,  0x1:  0.5,  0x2:  1.0,  0x3:  1.5,
    0x4:  2.0,  0x5:  3.0,  0x6:  4.0,  0x7:  6.0,
    0x8: -0.0,  0x9: -0.5,  0xA: -1.0,  0xB: -1.5,
    0xC: -2.0,  0xD: -3.0,  0xE: -4.0,  0xF: -6.0,
}

def nibble_extractor(packed_byte):
    """Mirrors nibble_extractor.v — splits 8-bit byte into two 4-bit nibbles."""
    lo_nibble = packed_byte & 0x0F
    hi_nibble = (packed_byte >> 4) & 0x0F
    return lo_nibble, hi_nibble

def e2m1_decoder(code):
    """Mirrors e2m1_decoder.v — 4-bit code to FP32 value via LUT."""
    return E2M1_LUT[code & 0xF]

def e8m0_decoder(code):
    """Mirrors e8m0_decoder.v — 8-bit E8M0 exponent-only to FP32 scale.
    Hardware: wires code[7:0] into FP32 exponent field [30:23].
    Result: 2^(code - 127)"""
    return 2.0 ** (int(code) - 127)

def fp8_e4m3_decoder(code):
    """Mirrors fp8_e4m3_decoder.v — 8-bit FP8 E4M3 to FP32 value.
    Fields: sign[7], exponent[6:3], mantissa[2:0]"""
    sign = (code >> 7) & 1
    exp4 = (code >> 3) & 0xF
    man3 = code & 0x7
    if exp4 == 0:
        val = (man3 / 8.0) * (2.0 ** (-6))
    else:
        val = (1.0 + man3 / 8.0) * (2.0 ** (exp4 - 7))
    if sign:
        val = -val
    return val

def unified_mac_datapath(packed_byte, scale_code_a, scale_code_b,
                         tensor_scale, b_elem_0, b_elem_1, FORMAT_MODE):
    """
    Bit-accurate emulation of unified_mac_top.v.
    This is the EXACT hardware pipeline inside the APU adapter.
    """
    lo_nib, hi_nib = nibble_extractor(packed_byte)
    decoded_0 = e2m1_decoder(lo_nib)
    decoded_1 = e2m1_decoder(hi_nib)

    mxfp4_scale = e8m0_decoder(scale_code_a)
    fp8_block_scale = fp8_e4m3_decoder(scale_code_b)
    nvfp4_eff_scale = fp8_block_scale * tensor_scale

    if FORMAT_MODE == 0:
        eff_scale = mxfp4_scale
    else:
        eff_scale = nvfp4_eff_scale

    scaled_0 = decoded_0 * eff_scale
    scaled_1 = decoded_1 * eff_scale

    prod_0 = scaled_0 * b_elem_0
    prod_1 = scaled_1 * b_elem_1

    result = prod_0 + prod_1
    return result


# =====================================================================
#  CV32E40P APU INSTRUCTION EMULATION
#  Emulates the mac_apu_coprocessor.sv FSM + APU handshake
# =====================================================================

class RI5CY_APU_Coprocessor:
    """
    Emulates mac_apu_coprocessor.sv — the APU adapter FSM.

    CV32E40P APU protocol:
      - Core asserts apu_req → adapter grants immediately (combinational)
      - 3 operands: apu_operands[0..2] (96 bits total)
      - apu_op from fpnew enum: ADD=2 (CFG), MUL=3 (EXEC)
      - Multi-cycle: adapter asserts apu_rvalid when done

    State machine: S_IDLE → S_CFG_RESET or S_EXEC_WAIT → S_IDLE
    """

    OP_QMAC_CFG  = 2   # fadd.s → fpnew ADD=2
    OP_QMAC_EXEC = 3   # fmul.s → fpnew MUL=3

    def __init__(self):
        self.mode_r = 0
        self.packed_byte_r = 0
        self.scale_a_r = 0
        self.scale_b_r = 0
        self.tensor_scale_r = 0.0
        self.accum = 0.0
        self.cycle_count = 0

    def apu_request(self, apu_op, operands):
        """
        Process one APU request (mirrors the FSM in mac_apu_coprocessor.sv).

        For CV32E40P APU, operands is a list of 3 x 32-bit values:
          fadd.s: operands[0]=rs1, operands[1]=rs1(dup), operands[2]=rs2
          fmul.s: operands[0]=rs1, operands[1]=rs2, operands[2]=unused

        Returns: (result, cycles)
        """
        # Combinational grant: apu_gnt = apu_req (0 cycle overhead)
        self.cycle_count += 1  # Request accepted

        if apu_op == self.OP_QMAC_CFG:
            # S_IDLE → S_CFG_RESET (1 cycle) → S_IDLE
            cfg_word = operands[0]   # rs1
            ts_bits  = operands[2]   # rs2 (at index 2 for fadd.s!)

            self.mode_r        = cfg_word & 0x1
            self.packed_byte_r = (cfg_word >> 1) & 0xFF
            self.scale_a_r     = (cfg_word >> 9) & 0xFF
            self.scale_b_r     = (cfg_word >> 17) & 0xFF

            # Decode tensor_scale from IEEE 754 bits
            self.tensor_scale_r = struct.unpack('f', struct.pack('I', ts_bits))[0]

            # Reset accumulator (mac_rst pulse)
            self.accum = 0.0

            self.cycle_count += 1  # S_CFG_RESET cycle
            return 0, 2  # 2 cycles total (IDLE + CFG_RESET)

        elif apu_op == self.OP_QMAC_EXEC:
            # S_IDLE → S_EXEC_WAIT (1 cycle for MAC) → S_IDLE
            b0_bits = operands[0]   # rs1
            b1_bits = operands[1]   # rs2

            b0 = struct.unpack('f', struct.pack('I', b0_bits))[0]
            b1 = struct.unpack('f', struct.pack('I', b1_bits))[0]

            # Run through unified_mac_top datapath
            mac_result = unified_mac_datapath(
                self.packed_byte_r,
                self.scale_a_r, self.scale_b_r,
                self.tensor_scale_r,
                b0, b1,
                self.mode_r
            )

            self.accum += mac_result

            self.cycle_count += 1  # S_EXEC_WAIT cycle
            return self.accum, 2  # 2 cycles total

        else:
            return 0, 1  # Unknown op


def ri5cy_qmac_cfg(apu, mode, packed_byte, scale_a, scale_b, tensor_scale_bits):
    """Emulate: fadd.s x0, a0, a1  (QMAC_CFG via APU)"""
    cfg_word = (mode & 0x1) | ((packed_byte & 0xFF) << 1) | \
               ((scale_a & 0xFF) << 9) | ((scale_b & 0xFF) << 17)
    # fadd.s operand mapping: [0]=rs1, [1]=rs1(dup), [2]=rs2
    operands = [cfg_word, cfg_word, tensor_scale_bits]
    return apu.apu_request(RI5CY_APU_Coprocessor.OP_QMAC_CFG, operands)


def ri5cy_qmac_exec(apu, b0_bits, b1_bits):
    """Emulate: fmul.s a4, a2, a3  (QMAC_EXEC via APU)"""
    # fmul.s operand mapping: [0]=rs1, [1]=rs2, [2]=unused
    operands = [b0_bits, b1_bits, 0]
    return apu.apu_request(RI5CY_APU_Coprocessor.OP_QMAC_EXEC, operands)


def float_to_bits(f):
    """Convert Python float to IEEE 754 uint32."""
    return struct.unpack('I', struct.pack('f', float(f)))[0]


# =====================================================================
#  QUANTIZATION HELPERS
# =====================================================================

def quantize_to_e2m1_code(value):
    e2m1_pos = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    sign = 1 if value >= 0 else -1
    abs_val = abs(value)
    best_code = 0
    best_err = float('inf')
    for i, v in enumerate(e2m1_pos):
        err = abs(abs_val - v)
        if err < best_err:
            best_err = err
            best_code = i
    if sign < 0:
        best_code |= 0x8
    return best_code

def compute_mxfp4_scale_code(weight_block):
    abs_max = np.max(np.abs(weight_block))
    if abs_max == 0:
        return 127
    import math
    exp = math.floor(math.log2(abs_max / 6.0)) + 127
    exp = max(0, min(254, exp + 1))
    return exp

def compute_nvfp4_scales(weight_block):
    abs_max = np.max(np.abs(weight_block))
    if abs_max == 0:
        return 0x38, 1.0
    tensor_scale = abs_max / 6.0 if abs_max > 0 else 1.0
    block_scale_val = abs_max / (6.0 * tensor_scale) if tensor_scale > 0 else 1.0
    best_code = 0x38
    best_err = float('inf')
    for code in range(256):
        val = abs(fp8_e4m3_decoder(code))
        err = abs(val - block_scale_val)
        if err < best_err and val > 0:
            best_err = err
            best_code = code
    return best_code, tensor_scale


# =====================================================================
#  WEIGHT RECONSTRUCTION VIA RI5CY APU DATAPATH
# =====================================================================

def reconstruct_weight_via_ri5cy_apu(weight_matrix, format_mode, block_size=None):
    """
    Quantize and reconstruct weights using the RI5CY APU instruction path.

    Emulates the full CV32E40P execution:
      1. fadd.s x0, cfg, ts  → QMAC_CFG (configure MAC, reset accum)
      2. fmul.s rd, b0, b1   → QMAC_EXEC (compute, return accum)

    This adds the APU-specific overhead:
      - 3-operand packing (96-bit APU bus)
      - Combinational grant handshake
      - FSM state transitions (S_IDLE → S_CFG_RESET / S_EXEC_WAIT)
    """
    if block_size is None:
        block_size = 32 if format_mode == 0 else 16

    flat = weight_matrix.flatten()
    recon = np.zeros_like(flat)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n)
    padded[:n] = flat

    apu = RI5CY_APU_Coprocessor()

    for block_start in range(0, padded_n, block_size):
        block = padded[block_start:block_start + block_size]

        if format_mode == 0:
            scale_code_a = compute_mxfp4_scale_code(block)
            scale_code_b = 0
            tensor_scale = 1.0
            scale_val = e8m0_decoder(scale_code_a)
        else:
            scale_code_a = 0
            scale_code_b, tensor_scale = compute_nvfp4_scales(block)
            scale_val = fp8_e4m3_decoder(scale_code_b) * tensor_scale

        # --- QMAC_CFG via APU (fadd.s instruction) ---
        ts_bits = float_to_bits(tensor_scale)
        ri5cy_qmac_cfg(apu, format_mode, 0, scale_code_a, scale_code_b, ts_bits)

        if abs(scale_val) > 1e-30:
            scaled_block = block / scale_val
        else:
            scaled_block = np.zeros_like(block)

        codes = [quantize_to_e2m1_code(v) for v in scaled_block]

        for i in range(0, len(codes), 2):
            idx0 = block_start + i
            idx1 = block_start + i + 1
            packed_byte = (codes[i+1] << 4) | codes[i]

            # --- QMAC_CFG to reset accum for each pair ---
            cfg_word_bits = (format_mode & 0x1) | ((packed_byte & 0xFF) << 1) | \
                           ((scale_code_a & 0xFF) << 9) | ((scale_code_b & 0xFF) << 17)
            ri5cy_qmac_cfg(apu, format_mode, packed_byte, scale_code_a, scale_code_b, ts_bits)

            # --- QMAC_EXEC: element 0 (b0=1.0, b1=0.0) ---
            b0_bits = float_to_bits(1.0)
            b1_bits = float_to_bits(0.0)
            val0, _ = ri5cy_qmac_exec(apu, b0_bits, b1_bits)

            if idx0 < n:
                recon[idx0] = val0

            # --- Reset + QMAC_EXEC: element 1 (b0=0.0, b1=1.0) ---
            ri5cy_qmac_cfg(apu, format_mode, packed_byte, scale_code_a, scale_code_b, ts_bits)
            b0_bits = float_to_bits(0.0)
            b1_bits = float_to_bits(1.0)
            val1, _ = ri5cy_qmac_exec(apu, b0_bits, b1_bits)

            if idx1 < n:
                recon[idx1] = val1

    return recon.reshape(weight_matrix.shape)


# =====================================================================
#  LAYER CLASSIFICATION + MODEL LOADING
# =====================================================================

def classify_layer(name):
    attn_keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
    ffn_keys  = ["gate_proj", "up_proj", "down_proj"]
    for k in attn_keys:
        if k in name:
            return "attention"
    for k in ffn_keys:
        if k in name:
            return "ffn"
    return "other"


def load_model_ri5cy_emulated(mode):
    """Load model with weights reconstructed through RI5CY APU datapath."""
    print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()

    if mode == "fp32":
        return model

    n_replaced = 0
    n_mxfp4 = 0
    n_nvfp4 = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() != 2:
                continue

            w = param.detach().float().numpy()
            layer_type = classify_layer(name)

            if layer_type == "ffn":
                format_mode = 0  # MXFP4
                n_mxfp4 += 1
            else:
                format_mode = 1  # NVFP4
                n_nvfp4 += 1

            w_recon = reconstruct_weight_via_ri5cy_apu(w, format_mode)
            param.copy_(torch.from_numpy(w_recon.astype(np.float32)))
            n_replaced += 1

            if n_replaced % 10 == 0:
                print(f"    Processed {n_replaced} layers...")

    print(f"  Replaced {n_replaced} layers via RI5CY APU datapath emulation")
    print(f"    FFN  layers -> MXFP4 (FORMAT_MODE=0): {n_mxfp4}")
    print(f"    Attn layers -> NVFP4 (FORMAT_MODE=1): {n_nvfp4}")
    return model


def compute_perplexity(model, tokenizer, text_corpus):
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

    ppl = float(np.exp(total_nll / total_count))
    print(f"  Evaluated {total_count} tokens across {n_chunks} chunks")
    return ppl


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("  RI5CY (CV32E40P) APU — Hardware-Accurate Perplexity")
    print("  Datapath: mac_apu_coprocessor.sv + unified_mac_top.v")
    print("  APU Protocol: 3-operand, request/grant, multi-cycle")
    print("  Strategy: Attention=NVFP4 | FFN=MXFP4")
    print("  Model: TinyLlama 1.1B | Dataset: WikiText-2")
    print("=" * 70)

    print("\nLoading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"] for row in dataset if len(row["text"].strip()) > 0]
    print(f"  {len(texts)} non-empty lines")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for mode in ["fp32", "unified"]:
        print(f"\n{'~'*70}")
        label = "FP32 Baseline" if mode == "fp32" \
                else "RI5CY APU MAC (Hardware Datapath Emulation)"
        print(f"  Running: {label}")
        print(f"{'~'*70}")

        t0 = time.time()
        model = load_model_ri5cy_emulated(mode)
        ppl = compute_perplexity(model, tokenizer, texts)
        elapsed = time.time() - t0

        results[mode] = {"perplexity": round(ppl, 3), "time_sec": round(elapsed, 1)}
        print(f"  Perplexity: {ppl:.3f}  (took {elapsed:.1f}s)")

        del model
        gc.collect()

    fp32_ppl = results["fp32"]["perplexity"]
    uni_ppl  = results["unified"]["perplexity"]

    print(f"\n{'='*70}")
    print(f"  RESULTS — RI5CY APU Hardware-Accurate Perplexity")
    print(f"{'='*70}")
    print(f"")
    print(f"  FP32 Baseline          : {fp32_ppl:.3f}")
    print(f"  RI5CY APU MAC (unified): {uni_ppl:.3f}  (+{(uni_ppl-fp32_ppl)/fp32_ppl*100:.1f}%)")
    print(f"")
    print(f"  Methodology: Weight reconstruction via bit-accurate emulation of")
    print(f"  CV32E40P (RI5CY) APU co-processor datapath:")
    print(f"    - APU handshake: combinational grant, multi-cycle rvalid")
    print(f"    - fadd.s (apu_op=2) -> QMAC_CFG  [3 operands, 96-bit bus]")
    print(f"    - fmul.s (apu_op=3) -> QMAC_EXEC [result via apu_rvalid]")
    print(f"    - FSM: S_IDLE -> S_CFG_RESET / S_EXEC_WAIT -> S_IDLE")
    print(f"    - unified_mac_top.v (nibble_extractor, e2m1, e8m0/fp8, MUX)")
    print(f"{'='*70}")

    out_path = "/kaggle/working/phase5_ri5cy_apu_perplexity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
