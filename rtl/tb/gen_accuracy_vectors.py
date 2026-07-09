"""
gen_accuracy_vectors.py - Large-scale accuracy test for unified MAC
Generates 100 random test vectors (50 MXFP4 + 50 NVFP4), computes
FP32 golden reference, and reports MSE between quantized and FP32 results.

Run: python gen_accuracy_vectors.py
Output: accuracy_test_vectors.vh (for Vivado testbench)
"""
import struct, os, random, math

random.seed(42)  # reproducible

def float_to_hex(f):
    return format(struct.unpack('>I', struct.pack('>f', f))[0], '08X')

def hex_to_float(h):
    return struct.unpack('>f', struct.pack('>I', h))[0]

# E2M1 lookup table (same as hardware)
E2M1 = {
    0x0: 0.0,  0x1: 0.5,  0x2: 1.0,  0x3: 1.5,
    0x4: 2.0,  0x5: 3.0,  0x6: 4.0,  0x7: 6.0,
    0x8: -0.0, 0x9: -0.5, 0xA: -1.0, 0xB: -1.5,
    0xC: -2.0, 0xD: -3.0, 0xE: -4.0, 0xF: -6.0,
}

# -------------------------------------------------------------------
# FP32 golden reference: what a perfect RISC-V FPU would compute
# This is: (original_weight_0 * b0) + (original_weight_1 * b1)
# where original weights are the TRUE float values BEFORE quantization
# -------------------------------------------------------------------
def fp32_golden(w0, w1, b0, b1):
    """Perfect FP32 dot product — no quantization."""
    return w0 * b0 + w1 * b1

# -------------------------------------------------------------------
# Quantized MAC: what our hardware computes
# Weight is quantized to E2M1 + scale, then reconstructed
# -------------------------------------------------------------------
def quantize_to_e2m1(value):
    """Quantize a float to the nearest E2M1 code (4-bit)."""
    e2m1_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    sign = 1 if value >= 0 else -1
    abs_val = abs(value)
    # Find nearest
    best_code = 0
    best_err = float('inf')
    for i, v in enumerate(e2m1_values):
        err = abs(abs_val - v)
        if err < best_err:
            best_err = err
            best_code = i
    if sign < 0:
        best_code |= 0x8
    return best_code

def compute_mx(packed, scale_code, b0, b1):
    """MXFP4 mode MAC computation."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    d0 = E2M1[lo]
    d1 = E2M1[hi]
    scale = 2.0 ** (scale_code - 127)
    return d0 * scale * b0 + d1 * scale * b1

def decode_fp8_e4m3(code):
    """Decode FP8 E4M3 scale code to float."""
    exp4 = (code >> 3) & 0xF
    man3 = code & 0x7
    sign = (code >> 7) & 1
    if exp4 == 0:
        val = (man3 / 8.0) * (2.0 ** (-6))
    else:
        val = (1.0 + man3 / 8.0) * (2.0 ** (exp4 - 7))
    if sign:
        val = -val
    return val

def compute_nv(packed, bscale_code, tensor_scale, b0, b1):
    """NVFP4 mode MAC computation."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    d0 = E2M1[lo]
    d1 = E2M1[hi]
    bscale_val = decode_fp8_e4m3(bscale_code)
    eff = bscale_val * tensor_scale
    return d0 * eff * b0 + d1 * eff * b1

# -------------------------------------------------------------------
# Generate random test cases
# -------------------------------------------------------------------
tests = []       # (mode, packed, sca, scb, ts, b0, b1, quantized_result)
fp32_refs = []   # FP32 golden reference for each test

NUM_MX = 50
NUM_NV = 50

print("=" * 60)
print("Generating accuracy test vectors")
print("=" * 60)

# --- MXFP4 tests ---
print(f"\nGenerating {NUM_MX} MXFP4 test cases...")
mx_errors = []
for i in range(NUM_MX):
    # Random original weights (what we'd have in FP32 before quantization)
    w0_orig = random.uniform(-6.0, 6.0)
    w1_orig = random.uniform(-6.0, 6.0)

    # Random scale code (exponent 120-134 for reasonable range)
    scale_code = random.randint(120, 134)
    scale_val = 2.0 ** (scale_code - 127)

    # Quantize: scale down, round to E2M1, pack
    w0_scaled = w0_orig / scale_val
    w1_scaled = w1_orig / scale_val
    code0 = quantize_to_e2m1(w0_scaled)
    code1 = quantize_to_e2m1(w1_scaled)
    packed = (code1 << 4) | code0

    # Reconstructed values (what hardware produces)
    w0_recon = E2M1[code0] * scale_val
    w1_recon = E2M1[code1] * scale_val

    # Random B elements
    b0 = random.uniform(-2.0, 2.0)
    b1 = random.uniform(-2.0, 2.0)

    # Quantized MAC result (hardware answer)
    quant_result = compute_mx(packed, scale_code, b0, b1)

    # FP32 golden (perfect answer with original weights)
    golden = fp32_golden(w0_orig, w1_orig, b0, b1)

    # Error
    if abs(golden) > 1e-10:
        rel_err = abs(quant_result - golden) / abs(golden)
    else:
        rel_err = abs(quant_result - golden)
    mx_errors.append((quant_result - golden) ** 2)

    tests.append((0, packed, scale_code, 0x00, 1.0, b0, b1, quant_result))
    fp32_refs.append(golden)

mx_mse = sum(mx_errors) / len(mx_errors)
mx_rmse = math.sqrt(mx_mse)
print(f"  MXFP4 RMSE vs FP32: {mx_rmse:.6f}")

# --- NVFP4 tests ---
print(f"\nGenerating {NUM_NV} NVFP4 test cases...")
nv_errors = []
for i in range(NUM_NV):
    # Random original weights
    w0_orig = random.uniform(-6.0, 6.0)
    w1_orig = random.uniform(-6.0, 6.0)

    # Random FP8 E4M3 block scale (normal range)
    bscale_code = random.choice([0x30, 0x34, 0x38, 0x3C, 0x40, 0x44, 0x48])
    bscale_val = decode_fp8_e4m3(bscale_code)

    # Random tensor scale
    ts = random.uniform(0.001, 2.0)
    eff_scale = bscale_val * ts

    # Quantize
    if abs(eff_scale) > 1e-30:
        w0_scaled = w0_orig / eff_scale
        w1_scaled = w1_orig / eff_scale
    else:
        w0_scaled = 0
        w1_scaled = 0
    code0 = quantize_to_e2m1(w0_scaled)
    code1 = quantize_to_e2m1(w1_scaled)
    packed = (code1 << 4) | code0

    # Random B elements
    b0 = random.uniform(-2.0, 2.0)
    b1 = random.uniform(-2.0, 2.0)

    # Quantized MAC result
    quant_result = compute_nv(packed, bscale_code, ts, b0, b1)

    # FP32 golden
    golden = fp32_golden(w0_orig, w1_orig, b0, b1)

    # Error
    if abs(golden) > 1e-10:
        rel_err = abs(quant_result - golden) / abs(golden)
    else:
        rel_err = abs(quant_result - golden)
    nv_errors.append((quant_result - golden) ** 2)

    tests.append((1, packed, 0x00, bscale_code, ts, b0, b1, quant_result))
    fp32_refs.append(golden)

nv_mse = sum(nv_errors) / len(nv_errors)
nv_rmse = math.sqrt(nv_mse)
print(f"  NVFP4 RMSE vs FP32: {nv_rmse:.6f}")

# --- Summary ---
print(f"\n{'=' * 60}")
print(f"ACCURACY SUMMARY")
print(f"{'=' * 60}")
print(f"  MXFP4: MSE = {mx_mse:.6f}, RMSE = {mx_rmse:.6f}")
print(f"  NVFP4: MSE = {nv_mse:.6f}, RMSE = {nv_rmse:.6f}")
print(f"  NVFP4 is {mx_rmse/nv_rmse:.2f}x more accurate than MXFP4")
print(f"{'=' * 60}")

# --- Write .vh file ---
N = len(tests)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'accuracy_test_vectors.vh')
with open(out_path, 'w') as f:
    f.write("// Auto-generated by gen_accuracy_vectors.py - DO NOT EDIT\n")
    f.write(f"// {NUM_MX} MXFP4 + {NUM_NV} NVFP4 = {N} total test vectors\n")
    f.write(f"// MXFP4 RMSE vs FP32 = {mx_rmse:.6f}\n")
    f.write(f"// NVFP4 RMSE vs FP32 = {nv_rmse:.6f}\n\n")
    f.write(f"localparam ACC_N = {N};\n")
    f.write(f"localparam ACC_NUM_MX = {NUM_MX};\n")
    f.write(f"localparam ACC_NUM_NV = {NUM_NV};\n\n")

    f.write(f"logic        acc_mode    [0:{N-1}] = '{{" + ", ".join(f"1'b{t[0]}" for t in tests) + "};\n")
    f.write(f"logic [7:0]  acc_packed  [0:{N-1}] = '{{" + ", ".join(f"8'h{t[1]:02X}" for t in tests) + "};\n")
    f.write(f"logic [7:0]  acc_sca     [0:{N-1}] = '{{" + ", ".join(f"8'd{t[2]}" for t in tests) + "};\n")
    f.write(f"logic [7:0]  acc_scb     [0:{N-1}] = '{{" + ", ".join(f"8'h{t[3]:02X}" for t in tests) + "};\n")
    f.write(f"logic [31:0] acc_ts      [0:{N-1}] = '{{" + ", ".join(f"32'h{float_to_hex(t[4])}" for t in tests) + "};\n")
    f.write(f"logic [31:0] acc_b0      [0:{N-1}] = '{{" + ", ".join(f"32'h{float_to_hex(t[5])}" for t in tests) + "};\n")
    f.write(f"logic [31:0] acc_b1      [0:{N-1}] = '{{" + ", ".join(f"32'h{float_to_hex(t[6])}" for t in tests) + "};\n")
    f.write(f"logic [31:0] acc_expected [0:{N-1}] = '{{" + ", ".join(f"32'h{float_to_hex(t[7])}" for t in tests) + "};\n\n")

    # Also store FP32 golden references for MSE computation in testbench
    f.write(f"// FP32 golden references (what a perfect FP32 MAC would produce)\n")
    f.write(f"logic [31:0] acc_fp32_golden [0:{N-1}] = '{{" + ", ".join(f"32'h{float_to_hex(g)}" for g in fp32_refs) + "};\n")

print(f"\nGenerated: {out_path}")
print(f"Total test vectors: {N}")
