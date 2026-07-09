/**
 * quant_types.h — Shared types and decode helpers for MXFP4 and NVFP4 GEMM kernels.
 *
 * E2M1 4-bit float layout (stored packed: two codes per byte, lo nibble first):
 *   bit3 = sign, bits[2:0] = index into value table
 *
 * Value table (positive half):
 *   0→0.0  1→0.5  2→1.0  3→1.5  4→2.0  5→3.0  6→4.0  7→6.0
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>

/* ── E2M1 value table ─────────────────────────────────────────── */
static const float E2M1_TABLE[16] = {
    /* sign=0 */  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    /* sign=1 */ -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
};

/* Decode a 4-bit E2M1 code (0-15) to float32 */
static inline float e2m1_decode(uint8_t code) {
    return E2M1_TABLE[code & 0xF];
}

/* ── E8M0 scale decode (MXFP4) ───────────────────────────────── */
/* scale = 2^(code - 127),  code 255 → NaN (treat as 1.0) */
static inline float e8m0_decode(uint8_t code) {
    if (code == 255) return 1.0f;
    int exp = (int)code - 127;
    /* Use ldexpf to avoid denormal issues */
    return ldexpf(1.0f, exp);
}

/* ── FP8 E4M3 scale decode (NVFP4 block scale) ───────────────── */
/* Standard E4M3: bias=7, max=240, exp_biased=15 → NaN */
static inline float fp8_e4m3_decode(uint8_t code) {
    int sign      = (code >> 7) & 1;
    int exp_b     = (code >> 3) & 0xF;
    int mantissa  = (code)      & 0x7;
    float value;
    if (exp_b == 15 && mantissa == 7) return 0.0f; /* NaN → 0 (safe for scale) */
    if (exp_b == 0) {
        value = (mantissa / 8.0f) * ldexpf(1.0f, -6);  /* subnormal */
    } else {
        value = (1.0f + mantissa / 8.0f) * ldexpf(1.0f, exp_b - 7);
    }
    return sign ? -value : value;
}

/* ── Packed nibble unpacking ─────────────────────────────────── */
/* Each byte stores two E2M1 codes: lo nibble = element[2i], hi nibble = element[2i+1] */
static inline void unpack_nibbles(const uint8_t *packed, float *out, int n) {
    for (int i = 0; i < n / 2; i++) {
        out[2*i]   = e2m1_decode(packed[i] & 0xF);
        out[2*i+1] = e2m1_decode(packed[i] >> 4);
    }
    if (n & 1) out[n-1] = e2m1_decode(packed[n/2] & 0xF);
}

/* ── MXFP4 tensor descriptor ─────────────────────────────────── */
typedef struct {
    const uint8_t *packed;      /* packed nibbles: ceil(rows*cols/2) bytes     */
    const uint8_t *scales;      /* E8M0 scales: ceil(rows*cols/32) bytes        */
    int rows, cols;
    int block_size;             /* always 32 for MXFP4                          */
} mxfp4_tensor_t;

/* ── NVFP4 tensor descriptor ─────────────────────────────────── */
typedef struct {
    const uint8_t *packed;      /* packed nibbles: ceil(rows*cols/2) bytes      */
    const uint8_t *block_scales;/* FP8 E4M3 block scales: ceil(rows*cols/16)    */
    float          tensor_scale;/* per-tensor FP32 scale                        */
    int rows, cols;
    int block_size;             /* always 16 for NVFP4                          */
} nvfp4_tensor_t;
