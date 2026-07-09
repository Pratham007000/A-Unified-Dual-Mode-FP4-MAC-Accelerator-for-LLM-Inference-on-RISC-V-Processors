"""
mxfp4.py — OCP MXFP4 Quantizer / Dequantizer

Format spec (OCP MX Spec v1.0):
  - Block size: 32 elements
  - Element type: E2M1 (4-bit float)
  - Scale type: E8M0 (8-bit, power-of-two only, bias=127)
    · Encodes powers of 2 from 2^(-127) to 2^(127), plus 0 and NaN.
    · Scale = 2^(code - 127)  for code in [0, 254]
    · code = 255 → NaN (reserved)
  - Scale is SHARED across the entire 32-element block (one scale per block)
  - Quantization: round-to-nearest

E8M0 scale covers the max absolute value in the block:
  scale = 2^floor(log2(max_abs))  if max_abs > 0 else 1.0

After scaling, elements are in range [-6, 6] (E2M1 max magnitude),
and are encoded per e2m1_quantize().
"""

import numpy as np
from fp_utils import e2m1_quantize, e2m1_dequantize, E2M1_MAX

BLOCK_SIZE = 32

# ── E8M0 scale helpers ─────────────────────────────────────────────────────────

def e8m0_encode(scale: float) -> int:
    """Encode a positive power-of-2 scale to E8M0 uint8 code."""
    if scale <= 0 or not np.isfinite(scale):
        return 255  # NaN / invalid
    exp = int(np.floor(np.log2(scale + 1e-38)))
    code = exp + 127
    return int(np.clip(code, 0, 254))


def e8m0_decode(code: int) -> float:
    """Decode E8M0 uint8 code to float32 scale."""
    if code == 255:
        return float('nan')
    return float(2.0 ** (int(code) - 127))


# ── Per-block quantization ─────────────────────────────────────────────────────

def _quantize_block(block: np.ndarray):
    """
    Quantize a single 32-element block.

    Returns:
        codes     : uint8 array of 32 E2M1 codes
        scale_code: uint8 E8M0 scale code
    """
    assert block.shape == (BLOCK_SIZE,), f"Expected {BLOCK_SIZE} elements, got {block.shape}"

    max_abs = np.max(np.abs(block))

    if max_abs == 0.0:
        scale = 1.0
    else:
        # Largest power-of-2 ≤ max_abs, then adjust so max maps within E2M1 range
        # E2M1 max magnitude is 6.0, so scale = 2^floor(log2(max_abs / E2M1_MAX))
        # This ensures max_abs / scale ≤ E2M1_MAX
        raw_exp = np.floor(np.log2(max_abs / E2M1_MAX + 1e-38))
        scale = float(2.0 ** raw_exp)
        scale = max(scale, 2.0 ** (-127))  # clamp to E8M0 minimum

    scale_code = e8m0_encode(scale)
    actual_scale = e8m0_decode(scale_code)

    scaled = block / actual_scale
    codes = e2m1_quantize(scaled)
    return codes, np.uint8(scale_code)


def _dequantize_block(codes: np.ndarray, scale_code: int) -> np.ndarray:
    """Dequantize a single 32-element block."""
    scale = e8m0_decode(int(scale_code))
    return e2m1_dequantize(codes) * scale


# ── Public API ─────────────────────────────────────────────────────────────────

class MXFP4Quantizer:
    """
    Quantizes a 1-D or 2-D float32 tensor to MXFP4.

    Usage:
        q = MXFP4Quantizer()
        codes, scales = q.quantize(weight_matrix)   # weight_matrix: (M, N)
        reconstructed = q.dequantize(codes, scales, original_shape)
    """

    def quantize(self, x: np.ndarray):
        """
        Quantize tensor x to MXFP4.

        x is flattened, then split into non-overlapping 32-element blocks.
        If the total number of elements is not divisible by 32, the last
        block is zero-padded.

        Returns:
            codes       : uint8 ndarray, shape (num_blocks, 32)
            scale_codes : uint8 ndarray, shape (num_blocks,)
            original_shape: tuple — needed for dequantize
        """
        x = np.asarray(x, dtype=np.float32)
        original_shape = x.shape
        flat = x.ravel()
        n = flat.shape[0]

        # Pad to multiple of BLOCK_SIZE
        pad = (-n) % BLOCK_SIZE
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        num_blocks = flat.shape[0] // BLOCK_SIZE
        blocks = flat.reshape(num_blocks, BLOCK_SIZE)

        all_codes = np.zeros((num_blocks, BLOCK_SIZE), dtype=np.uint8)
        all_scales = np.zeros(num_blocks, dtype=np.uint8)

        for i, block in enumerate(blocks):
            codes, sc = _quantize_block(block)
            all_codes[i] = codes
            all_scales[i] = sc

        return all_codes, all_scales, original_shape

    def dequantize(self, codes: np.ndarray, scale_codes: np.ndarray, original_shape: tuple) -> np.ndarray:
        """
        Dequantize MXFP4 codes back to float32.

        codes       : uint8 (num_blocks, 32)
        scale_codes : uint8 (num_blocks,)
        original_shape: original tensor shape

        Returns float32 array of original_shape.
        """
        num_blocks = codes.shape[0]
        flat = np.zeros(num_blocks * BLOCK_SIZE, dtype=np.float32)

        for i in range(num_blocks):
            flat[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = _dequantize_block(codes[i], scale_codes[i])

        total_orig = int(np.prod(original_shape))
        return flat[:total_orig].reshape(original_shape)

    def memory_bytes(self, original_shape: tuple) -> dict:
        """Estimate memory footprint for a quantized tensor."""
        n = int(np.prod(original_shape))
        num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        # 4 bits per element → n/2 bytes (packed), 1 byte per block scale
        element_bytes = (n * 4 + 7) // 8  # 4 bits per element, rounded up
        scale_bytes = num_blocks * 1       # 1 byte (E8M0) per block
        fp32_bytes = n * 4
        return {
            "elements_bits": n * 4,
            "element_bytes_packed": element_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": element_bytes + scale_bytes,
            "fp32_bytes": fp32_bytes,
            "compression_ratio": fp32_bytes / (element_bytes + scale_bytes),
        }
