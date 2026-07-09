"""
nvfp4.py — NVIDIA NVFP4 Quantizer / Dequantizer

Format spec (NVIDIA FP4 as used in Blackwell / NV FP4):
  - Block size: 16 elements
  - Element type: E2M1 (4-bit float, same bit layout as MXFP4)
  - DUAL-scale structure:
      · Block scale  : E4M3 FP8 (one per 16-element block)
          - Encodes the local block maximum → finer granularity
      · Tensor scale : FP32 (one per entire tensor)
          - Global scale applied on top of block scales
  - Dequantization: x = e2m1_val * block_scale * tensor_scale

Compared to MXFP4:
  · Smaller blocks (16 vs 32) → more scales, better accuracy
  · Block scale is FP8 E4M3 (8 bits) vs E8M0 (8 bits) — different range
  · Extra per-tensor FP32 scale adds 4 bytes total (negligible overhead)
  · More complex scale bookkeeping in hardware
"""

import numpy as np
from fp_utils import (
    e2m1_quantize, e2m1_dequantize, E2M1_MAX,
    fp8_e4m3_quantize_scalar, fp8_e4m3_dequantize_scalar,
    FP8_E4M3_MAX
)

BLOCK_SIZE = 16


# ── Per-block quantization ─────────────────────────────────────────────────────

def _quantize_block(block: np.ndarray, tensor_scale: float):
    """
    Quantize a single 16-element block given the pre-computed tensor_scale.

    The block is pre-divided by tensor_scale, then a block-level FP8 E4M3
    scale is computed and applied before E2M1 encoding.

    Returns:
        codes      : uint8 array of 16 E2M1 codes
        block_scale_code: uint8 FP8 E4M3 code for this block's scale
    """
    assert block.shape == (BLOCK_SIZE,)

    # Step 1: apply tensor scale (undo global normalization)
    pre_scaled = block / tensor_scale if tensor_scale != 0 else block

    max_abs = np.max(np.abs(pre_scaled))

    if max_abs == 0.0:
        block_scale = 1.0
    else:
        # Choose block scale so max_abs / block_scale ≤ E2M1_MAX
        # block_scale is encoded as FP8 E4M3 (range up to 448)
        block_scale = float(max_abs / E2M1_MAX)
        block_scale = min(block_scale, FP8_E4M3_MAX)
        block_scale = max(block_scale, 2 ** (-9))  # FP8 E4M3 min normal ~0.001953

    block_scale_code = fp8_e4m3_quantize_scalar(block_scale)
    actual_block_scale = fp8_e4m3_dequantize_scalar(block_scale_code)

    if actual_block_scale == 0 or np.isnan(actual_block_scale):
        actual_block_scale = 1.0

    scaled = pre_scaled / actual_block_scale
    codes = e2m1_quantize(scaled)
    return codes, np.uint8(block_scale_code)


def _dequantize_block(codes: np.ndarray, block_scale_code: int, tensor_scale: float) -> np.ndarray:
    """Dequantize a single 16-element block."""
    block_scale = fp8_e4m3_dequantize_scalar(int(block_scale_code))
    if np.isnan(block_scale):
        block_scale = 1.0
    return e2m1_dequantize(codes) * block_scale * tensor_scale


# ── Tensor scale computation ───────────────────────────────────────────────────

def compute_tensor_scale(x: np.ndarray) -> float:
    """
    Compute a single FP32 per-tensor scale.

    Convention: tensor_scale = max_abs_of_tensor / (FP8_E4M3_MAX * E2M1_MAX)
    This leaves headroom for block scales to refine locally.
    If the tensor is zero, return 1.0.
    """
    max_abs = float(np.max(np.abs(x)))
    if max_abs == 0.0:
        return 1.0
    # Global scale normalizes to FP8 range; block scales handle local variation
    return max_abs / (FP8_E4M3_MAX * E2M1_MAX)


# ── Public API ─────────────────────────────────────────────────────────────────

class NVFP4Quantizer:
    """
    Quantizes a float32 tensor to NVFP4 (dual-scale FP4).

    Usage:
        q = NVFP4Quantizer()
        codes, block_scales, tensor_scale = q.quantize(weight_matrix)
        reconstructed = q.dequantize(codes, block_scales, tensor_scale, original_shape)
    """

    def quantize(self, x: np.ndarray):
        """
        Quantize tensor x to NVFP4.

        Returns:
            codes         : uint8 ndarray, shape (num_blocks, 16)
            block_scales  : uint8 ndarray, shape (num_blocks,)  [FP8 E4M3 codes]
            tensor_scale  : float32 scalar
            original_shape: tuple
        """
        x = np.asarray(x, dtype=np.float32)
        original_shape = x.shape
        flat = x.ravel()
        n = flat.shape[0]

        tensor_scale = compute_tensor_scale(flat)

        # Pad to multiple of BLOCK_SIZE
        pad = (-n) % BLOCK_SIZE
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        num_blocks = flat.shape[0] // BLOCK_SIZE
        blocks = flat.reshape(num_blocks, BLOCK_SIZE)

        all_codes = np.zeros((num_blocks, BLOCK_SIZE), dtype=np.uint8)
        all_block_scales = np.zeros(num_blocks, dtype=np.uint8)

        for i, block in enumerate(blocks):
            codes, bsc = _quantize_block(block, tensor_scale)
            all_codes[i] = codes
            all_block_scales[i] = bsc

        return all_codes, all_block_scales, np.float32(tensor_scale), original_shape

    def dequantize(self, codes: np.ndarray, block_scales: np.ndarray,
                   tensor_scale: float, original_shape: tuple) -> np.ndarray:
        """
        Dequantize NVFP4 codes back to float32.

        codes        : uint8 (num_blocks, 16)
        block_scales : uint8 (num_blocks,)  [FP8 E4M3 codes]
        tensor_scale : float32 scalar
        original_shape: original tensor shape
        """
        num_blocks = codes.shape[0]
        flat = np.zeros(num_blocks * BLOCK_SIZE, dtype=np.float32)

        for i in range(num_blocks):
            flat[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = _dequantize_block(
                codes[i], block_scales[i], tensor_scale
            )

        total_orig = int(np.prod(original_shape))
        return flat[:total_orig].reshape(original_shape)

    def memory_bytes(self, original_shape: tuple) -> dict:
        """Estimate memory footprint for a quantized tensor."""
        n = int(np.prod(original_shape))
        num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        element_bytes = (n * 4 + 7) // 8   # 4 bits per element
        block_scale_bytes = num_blocks * 1  # 1 byte FP8 per block
        tensor_scale_bytes = 4             # 1 × FP32
        fp32_bytes = n * 4
        total = element_bytes + block_scale_bytes + tensor_scale_bytes
        return {
            "elements_bits": n * 4,
            "element_bytes_packed": element_bytes,
            "block_scale_bytes": block_scale_bytes,
            "tensor_scale_bytes": tensor_scale_bytes,
            "total_bytes": total,
            "fp32_bytes": fp32_bytes,
            "compression_ratio": fp32_bytes / total,
        }
