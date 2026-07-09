"""
test_quantizers.py — Round-trip accuracy and correctness tests.

Run with:  python test_quantizers.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fp_utils import e2m1_quantize, e2m1_dequantize, fp8_e4m3_quantize_scalar, fp8_e4m3_dequantize_scalar
from mxfp4 import MXFP4Quantizer
from nvfp4 import NVFP4Quantizer

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {detail}")
    return condition


# ══════════════════════════════════════════════════════════════════════════════
# 1. E2M1 encode / decode
# ══════════════════════════════════════════════════════════════════════════════
def test_e2m1():
    print("\n── E2M1 encode/decode ────────────────────────────────────────────")

    # All 8 positive representable values
    expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    ok = True
    for val in expected:
        code = e2m1_quantize(np.array([val]))[0]
        rec  = e2m1_dequantize(np.array([code]))[0]
        ok &= check(f"  encode→decode {val}", np.isclose(rec, val), f"got {rec}")

    # Negative values
    for val in [-1.0, -3.0, -6.0]:
        code = e2m1_quantize(np.array([val]))[0]
        rec  = e2m1_dequantize(np.array([code]))[0]
        ok &= check(f"  negative {val}", np.isclose(rec, val), f"got {rec}")

    # Clamping
    code = e2m1_quantize(np.array([100.0]))[0]
    rec  = e2m1_dequantize(np.array([code]))[0]
    ok &= check("  clamp +100 → 6.0", np.isclose(rec, 6.0), f"got {rec}")

    code = e2m1_quantize(np.array([-100.0]))[0]
    rec  = e2m1_dequantize(np.array([code]))[0]
    ok &= check("  clamp -100 → -6.0", np.isclose(rec, -6.0), f"got {rec}")

    # Round-to-nearest: 0.75 is between 0.5 and 1.0, nearest is 1.0
    code = e2m1_quantize(np.array([0.75]))[0]
    rec  = e2m1_dequantize(np.array([code]))[0]
    ok &= check("  round 0.75 → 1.0", np.isclose(rec, 1.0), f"got {rec}")

    # 0.62 is closer to 0.5
    code = e2m1_quantize(np.array([0.62]))[0]
    rec  = e2m1_dequantize(np.array([code]))[0]
    ok &= check("  round 0.62 → 0.5", np.isclose(rec, 0.5), f"got {rec}")

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 2. FP8 E4M3 encode / decode
# ══════════════════════════════════════════════════════════════════════════════
def test_fp8():
    print("\n── FP8 E4M3 encode/decode ────────────────────────────────────────")
    ok = True
    for val in [0.0, 1.0, -1.0, 2.0, 0.015625, 240.0, -240.0, 0.5]:
        code = fp8_e4m3_quantize_scalar(val)
        rec  = fp8_e4m3_dequantize_scalar(code)
        tol  = max(abs(val) * 0.15, 1e-4)  # FP8 has ~3 bit mantissa, ~12% error
        ok  &= check(f"  fp8 {val}", abs(rec - val) <= tol, f"got {rec}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 3. MXFP4 round-trip
# ══════════════════════════════════════════════════════════════════════════════
def test_mxfp4_round_trip():
    print("\n── MXFP4 round-trip ─────────────────────────────────────────────")
    q = MXFP4Quantizer()
    ok = True

    # Test on random data at various scales
    for seed, scale in [(0, 1.0), (1, 0.001), (2, 1000.0), (3, 1e-5)]:
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(256) * scale).astype(np.float32)
        codes, scales, shape = q.quantize(x)
        x_rec = q.dequantize(codes, scales, shape)
        mse = float(np.mean((x - x_rec) ** 2))
        rel_err = float(np.sqrt(mse) / (np.std(x) + 1e-12))
        ok &= check(f"  scale={scale:.0e} MSE={mse:.3e} rel={rel_err:.3f}",
                    rel_err < 0.4, f"rel_err={rel_err:.3f}")

    # Non-multiple-of-32 length
    x = np.random.default_rng(99).standard_normal(100).astype(np.float32)
    codes, scales, shape = q.quantize(x)
    x_rec = q.dequantize(codes, scales, shape)
    ok &= check("  non-multiple-of-32 (100 elements)", x_rec.shape == (100,))

    # 2D tensor (matrix)
    x = np.random.default_rng(7).standard_normal((64, 64)).astype(np.float32)
    codes, scales, shape = q.quantize(x)
    x_rec = q.dequantize(codes, scales, shape)
    ok &= check("  2D matrix (64×64) shape preserved", x_rec.shape == (64, 64))

    # All zeros
    x = np.zeros(64, dtype=np.float32)
    codes, scales, shape = q.quantize(x)
    x_rec = q.dequantize(codes, scales, shape)
    ok &= check("  all-zeros tensor", np.allclose(x_rec, 0.0))

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 4. NVFP4 round-trip
# ══════════════════════════════════════════════════════════════════════════════
def test_nvfp4_round_trip():
    print("\n── NVFP4 round-trip ─────────────────────────────────────────────")
    q = NVFP4Quantizer()
    ok = True

    for seed, scale in [(0, 1.0), (1, 0.001), (2, 1000.0), (3, 1e-5)]:
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(256) * scale).astype(np.float32)
        codes, bscales, tscale, shape = q.quantize(x)
        x_rec = q.dequantize(codes, bscales, tscale, shape)
        mse = float(np.mean((x - x_rec) ** 2))
        rel_err = float(np.sqrt(mse) / (np.std(x) + 1e-12))
        ok &= check(f"  scale={scale:.0e} MSE={mse:.3e} rel={rel_err:.3f}",
                    rel_err < 0.4, f"rel_err={rel_err:.3f}")

    # Non-multiple-of-16
    x = np.random.default_rng(55).standard_normal(50).astype(np.float32)
    codes, bscales, tscale, shape = q.quantize(x)
    x_rec = q.dequantize(codes, bscales, tscale, shape)
    ok &= check("  non-multiple-of-16 (50 elements)", x_rec.shape == (50,))

    # 2D matrix
    x = np.random.default_rng(8).standard_normal((64, 64)).astype(np.float32)
    codes, bscales, tscale, shape = q.quantize(x)
    x_rec = q.dequantize(codes, bscales, tscale, shape)
    ok &= check("  2D matrix (64×64) shape preserved", x_rec.shape == (64, 64))

    # All zeros
    x = np.zeros(64, dtype=np.float32)
    codes, bscales, tscale, shape = q.quantize(x)
    x_rec = q.dequantize(codes, bscales, tscale, shape)
    ok &= check("  all-zeros tensor", np.allclose(x_rec, 0.0))

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 5. Head-to-head accuracy comparison
# ══════════════════════════════════════════════════════════════════════════════
def test_comparison():
    print("\n── MXFP4 vs NVFP4 accuracy comparison ───────────────────────────")
    qm = MXFP4Quantizer()
    qn = NVFP4Quantizer()
    ok = True

    print(f"  {'Distribution':<28} {'MXFP4 MSE':>12} {'NVFP4 MSE':>12} {'Winner':>8}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*8}")

    distributions = [
        ("Normal N(0,1)",         lambda: np.random.default_rng(0).standard_normal(1024).astype(np.float32)),
        ("Uniform U(-1,1)",       lambda: np.random.default_rng(1).uniform(-1, 1, 1024).astype(np.float32)),
        ("Laplace(0, 0.5)",       lambda: np.random.default_rng(2).laplace(0, 0.5, 1024).astype(np.float32)),
        ("Sparse (90% zeros)",    lambda: _sparse(0.9, 1024)),
        ("Heavy tail (t-dist)",   lambda: _heavy_tail(1024)),
        ("Small scale N(0,1e-3)", lambda: (np.random.default_rng(5).standard_normal(1024)*1e-3).astype(np.float32)),
        ("Large scale N(0,100)",  lambda: (np.random.default_rng(6).standard_normal(1024)*100).astype(np.float32)),
    ]

    mx_wins = 0
    for name, gen in distributions:
        x = gen()
        std = float(np.std(x)) + 1e-12

        cm, sm, sh = qm.quantize(x)
        xm = qm.dequantize(cm, sm, sh)
        mse_m = float(np.mean((x - xm)**2))

        cn, bn, ts, sh = qn.quantize(x)
        xn = qn.dequantize(cn, bn, ts, sh)
        mse_n = float(np.mean((x - xn)**2))

        winner = "MXFP4" if mse_m <= mse_n else "NVFP4"
        if winner == "MXFP4":
            mx_wins += 1
        print(f"  {name:<28} {mse_m:>12.4e} {mse_n:>12.4e} {winner:>8}")

    ok &= check(f"\n  Comparison ran without errors", True)
    return ok


def _sparse(zero_frac, n):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n).astype(np.float32)
    mask = rng.random(n) < zero_frac
    x[mask] = 0.0
    return x

def _heavy_tail(n):
    rng = np.random.default_rng(4)
    return (rng.standard_t(2, n)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Memory footprint
# ══════════════════════════════════════════════════════════════════════════════
def test_memory():
    print("\n── Memory footprint ─────────────────────────────────────════════")
    qm = MXFP4Quantizer()
    qn = NVFP4Quantizer()
    ok = True

    shape = (4096, 4096)  # typical weight matrix
    n = 4096 * 4096

    mm = qm.memory_bytes(shape)
    mn = qn.memory_bytes(shape)

    print(f"  Tensor shape: {shape}  ({n:,} elements)")
    print(f"  FP32 baseline:      {mm['fp32_bytes']:>12,} bytes")
    print(f"  MXFP4 total:        {mm['total_bytes']:>12,} bytes  (ratio {mm['compression_ratio']:.2f}×)")
    print(f"    elements (4-bit): {mm['element_bytes_packed']:>12,} bytes")
    print(f"    scales (E8M0):    {mm['scale_bytes']:>12,} bytes  ({n//32:,} blocks × 1B)")
    print(f"  NVFP4 total:        {mn['total_bytes']:>12,} bytes  (ratio {mn['compression_ratio']:.2f}×)")
    print(f"    elements (4-bit): {mn['element_bytes_packed']:>12,} bytes")
    print(f"    block scales(FP8):{mn['block_scale_bytes']:>12,} bytes  ({n//16:,} blocks × 1B)")
    print(f"    tensor scale(FP32):{mn['tensor_scale_bytes']:>11,} bytes")

    # MXFP4 should be slightly more memory-efficient (fewer scales)
    ok &= check("  MXFP4 uses fewer scale bytes than NVFP4",
                mm['scale_bytes'] < mn['block_scale_bytes'])
    ok &= check("  Both achieve ~8× compression over FP32",
                7.0 < mm['compression_ratio'] and 7.0 < mn['compression_ratio'])
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1 — Quantizer Test Suite")
    print("=" * 60)

    results = [
        test_e2m1(),
        test_fp8(),
        test_mxfp4_round_trip(),
        test_nvfp4_round_trip(),
        test_comparison(),
        test_memory(),
    ]

    total = len(results)
    passed = sum(results)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} test groups passed")
    print(f"{'='*60}\n")
    sys.exit(0 if all(results) else 1)
