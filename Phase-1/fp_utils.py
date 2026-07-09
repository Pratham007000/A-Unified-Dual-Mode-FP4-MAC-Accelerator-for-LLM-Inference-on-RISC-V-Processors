"""
fp_utils.py — Shared low-level primitives for 4-bit float formats.

E2M1 layout (4 bits):  [s | e1 e0 | m0]
  - 1 sign bit
  - 2 exponent bits  (biased, bias = 1)
  - 1 mantissa bit

Representable non-zero magnitudes (positive):
  Exponent=00 (subnormal): 0.m  × 2^(1-bias) = {0.0, 0.5}
  Exponent=01:             1.m  × 2^(1-1)    = {1.0, 1.5}
  Exponent=10:             1.m  × 2^(2-1)    = {2.0, 3.0}
  Exponent=11:             1.m  × 2^(3-1)    = {4.0, 6.0}

Full value table (positive half, 8 entries 0..7):
  0 → 0.0
  1 → 0.5
  2 → 1.0
  3 → 1.5
  4 → 2.0
  5 → 3.0
  6 → 4.0
  7 → 6.0
"""

import numpy as np

# ── E2M1 value table (positive half) ──────────────────────────────────────────
# Index is the unsigned 3-bit code (sign=0); sign bit is handled separately.
E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# Max representable magnitude
E2M1_MAX = 6.0


def e2m1_quantize(x: np.ndarray) -> np.ndarray:
    """
    Map float32 array x to E2M1 4-bit integer codes (uint8, range 0-15).

    Encoding:
      bit3 = sign
      bits[2:0] = nearest index into E2M1_VALUES for |x|, clamped.
    """
    x = np.asarray(x, dtype=np.float32)
    sign = (x < 0).astype(np.uint8)
    abs_x = np.abs(x)

    # Clamp to representable range
    abs_x_clamped = np.clip(abs_x, 0.0, E2M1_MAX)

    # Find nearest representable value via table lookup.
    # Tie-break: round-half-away-from-zero (pick larger magnitude on exact ties).
    # np.argmin returns the first (lower) index on ties, which is smaller magnitude —
    # so we nudge ties upward by subtracting a tiny epsilon before argmin.
    diff = np.abs(abs_x_clamped[..., None] - E2M1_VALUES)
    # Subtract tiny relative epsilon so ties favor the higher-index (larger) value
    eps = np.finfo(np.float32).eps * E2M1_VALUES
    idx = np.argmin(diff - eps, axis=-1).astype(np.uint8)  # 0-7

    code = (sign << 3) | idx  # 4-bit code, stored in uint8
    return code


def e2m1_dequantize(codes: np.ndarray) -> np.ndarray:
    """
    Decode E2M1 4-bit integer codes back to float32.

    codes: uint8 array, values 0-15.
    """
    codes = np.asarray(codes, dtype=np.uint8)
    sign = (codes >> 3) & 0x1
    idx = codes & 0x7
    magnitude = E2M1_VALUES[idx]
    sign_multiplier = np.where(sign, -1.0, 1.0).astype(np.float32)
    return magnitude * sign_multiplier


# ── FP8 E4M3 encode/decode (used for NVFP4 block scales) ──────────────────────
# E4M3: 1 sign + 4 exp (bias=7) + 3 mantissa, no inf, NaN = S.1111.111
# Range: ±448.0, smallest normal: 2^(1-7) = 0.015625

FP8_E4M3_MAX = 240.0  # E4M3 standard: (1 + 7/8) * 2^7 = 240. Note: 448 is the fnuz variant max.
FP8_E4M3_MIN_NORMAL = 2 ** (1 - 7)  # 0.015625


def fp8_e4m3_quantize_scalar(x: float) -> int:
    """Encode a single float32 to FP8 E4M3 (returned as uint8 int)."""
    sign = 1 if x < 0 else 0
    x_abs = abs(x)
    if x_abs == 0.0:
        return 0
    x_abs = min(x_abs, FP8_E4M3_MAX)

    import math
    # Find biased exponent; clamp to [0, 14] — 15 is reserved for NaN
    log_val = math.log2(x_abs) if x_abs >= FP8_E4M3_MIN_NORMAL else -6.0
    exp_unbiased = int(math.floor(log_val))
    exp_biased = exp_unbiased + 7
    exp_biased = max(0, min(exp_biased, 14))  # 15 is NaN sentinel

    if exp_biased == 0:
        # Subnormal
        mantissa_f = x_abs / (2 ** (-6))
    else:
        mantissa_f = x_abs / (2 ** (exp_biased - 7)) - 1.0

    mantissa = int(round(mantissa_f * 8))
    mantissa = max(0, min(mantissa, 7))

    return (sign << 7) | (exp_biased << 3) | mantissa


def fp8_e4m3_dequantize_scalar(code: int) -> float:
    """Decode a single FP8 E4M3 byte to float32."""
    sign = (code >> 7) & 1
    exp_biased = (code >> 3) & 0xF
    mantissa = code & 0x7

    if exp_biased == 15 and mantissa == 7:
        return float('nan')

    if exp_biased == 0:
        value = mantissa / 8.0 * (2 ** (-6))
    else:
        value = (1.0 + mantissa / 8.0) * (2 ** (exp_biased - 7))

    return -value if sign else value


def fp8_e4m3_quantize(x: np.ndarray) -> np.ndarray:
    """Vectorized FP8 E4M3 quantization. Returns uint8 array."""
    flat = np.asarray(x, dtype=np.float32).ravel()
    codes = np.array([fp8_e4m3_quantize_scalar(float(v)) for v in flat], dtype=np.uint8)
    return codes.reshape(x.shape)


def fp8_e4m3_dequantize(codes: np.ndarray) -> np.ndarray:
    """Vectorized FP8 E4M3 dequantization. Returns float32 array."""
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    values = np.array([fp8_e4m3_dequantize_scalar(int(c)) for c in flat], dtype=np.float32)
    return values.reshape(codes.shape)
