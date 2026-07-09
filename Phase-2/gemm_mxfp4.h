/**
 * gemm_mxfp4.h — RVV-optimized GEMM for MXFP4
 *
 * Computes: C[M×N] = A[M×K] × B[K×N]  (accumulate into C in FP32)
 *
 * A is MXFP4 (packed E2M1 nibbles + E8M0 block scales, block_size=32, row-major)
 * B is FP32  (weights already dequantized, or can be MXFP4 — see gemm_mxfp4_mxfp4)
 * C is FP32
 *
 * Strategy:
 *   Outer loops: tile over M and N in software.
 *   Inner loop (K): process 32-element blocks aligned to the MXFP4 block size.
 *     For each block:
 *       1. Load E8M0 scale (one scalar per 32 elements).
 *       2. Unpack 16 bytes → 32 nibbles using RVV bit-manipulation.
 *       3. Table-gather the E2M1 float values (using vloxei).
 *       4. Scale and accumulate via vfmacc into the running FP32 sum.
 *
 * RVV registers used: LMUL=2 (vfloat32m2), giving 32 float lanes per op
 * on a 256-bit VLEN machine (8 × 32-bit × 2 = 64 bytes = 512 bits — adjust
 * with vsetvl at runtime for portability).
 */
#pragma once
#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "quant_types.h"

/* ── Decode one 32-element MXFP4 block into float scratch buffer ── */
static inline void mxfp4_decode_block(
        const uint8_t * __restrict__ packed_base,  /* first packed byte of this block */
        uint8_t scale_code,
        float  * __restrict__ out)                 /* 32 floats output */
{
    float scale = e8m0_decode(scale_code);

    /* Unpack 16 bytes → 32 nibble codes */
    uint8_t codes[32];
    for (int b = 0; b < 16; b++) {
        codes[2*b]   = packed_base[b] & 0xF;
        codes[2*b+1] = packed_base[b] >> 4;
    }

    /* RVV: u8m1→u16m2→u32m4 widening, byte-offset gather into f32m4, scale */
    int i = 0;
    while (i < 32) {
        size_t vl = __riscv_vsetvl_e8m1(32 - i);
        vuint8m1_t  vcodes8  = __riscv_vle8_v_u8m1(codes + i, vl);
        vuint16m2_t vcodes16 = __riscv_vzext_vf2_u16m2(vcodes8, vl);
        vuint32m4_t vcodes32 = __riscv_vzext_vf2_u32m4(vcodes16, vl);
        vuint32m4_t voffsets = __riscv_vsll_vx_u32m4(vcodes32, 2, vl);
        vfloat32m4_t vvals   = __riscv_vloxei32_v_f32m4(E2M1_TABLE, voffsets, vl);
        vfloat32m4_t vscaled = __riscv_vfmul_vf_f32m4(vvals, scale, vl);
        __riscv_vse32_v_f32m4(out + i, vscaled, vl);
        i += (int)vl;
    }
}

/* ── Single output element: dot product of one MXFP4 row with FP32 column ── */
static float mxfp4_dot_row_col(
        const uint8_t * __restrict__ a_packed,   /* row packed nibbles */
        const uint8_t * __restrict__ a_scales,   /* row block scales   */
        const float   * __restrict__ b_col,      /* FP32 column vector */
        int K)
{
    float accum = 0.0f;
    float block_buf[32];
    int num_blocks = (K + 31) / 32;

    for (int blk = 0; blk < num_blocks; blk++) {
        int elem_start = blk * 32;
        int elem_count = (elem_start + 32 <= K) ? 32 : (K - elem_start);

        /* Decode MXFP4 block into float buffer */
        mxfp4_decode_block(a_packed + blk * 16, a_scales[blk], block_buf);

        /* RVV dot product with LMUL=4 accumulator */
        const float *bptr = b_col + elem_start;
        vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
        int i = 0;
        size_t vl;
        for (; i < elem_count; ) {
            vl = __riscv_vsetvl_e32m4(elem_count - i);
            vfloat32m4_t va = __riscv_vle32_v_f32m4(block_buf + i, vl);
            vfloat32m4_t vb = __riscv_vle32_v_f32m4(bptr + i, vl);
            vacc = __riscv_vfmacc_vv_f32m4(vacc, va, vb, vl);
            i += (int)vl;
        }
        /* Horizontal reduction via vfredusum into scalar */
        vl = __riscv_vsetvlmax_e32m4();
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
        vfloat32m1_t vred  = __riscv_vfredusum_vs_f32m4_f32m1(vacc, vzero, vl);
        accum += __riscv_vfmv_f_s_f32m1_f32(vred);
    }
    return accum;
}

/**
 * gemm_mxfp4_fp32 — Main GEMM: A (MXFP4) × B (FP32) → C (FP32)
 *
 * A: M×K packed MXFP4 matrix
 * B: K×N FP32 matrix (row-major)
 * C: M×N FP32 output (row-major, accumulated)
 *
 * A row i: packed nibbles start at a->packed + i * ceil(K/2) bytes
 *          scale codes start at   a->scales  + i * ceil(K/32) bytes
 *
 * This kernel is designed for weight-stationary operation:
 * A = quantized weight matrix (pre-loaded once)
 * B = FP32 activation matrix
 */
void gemm_mxfp4_fp32(
        const mxfp4_tensor_t * __restrict__ A,
        const float          * __restrict__ B,
        float                * __restrict__ C,
        int M, int N, int K)
{
    int packed_row_bytes = (K + 1) / 2;       /* bytes per A row (packed nibbles) */
    int scales_per_row   = (K + 31) / 32;     /* scale bytes per A row            */

    float *col_buf = (float*)malloc(K * sizeof(float));

    for (int m = 0; m < M; m++) {
        const uint8_t *a_packed = A->packed + m * packed_row_bytes;
        const uint8_t *a_scales = A->scales + m * scales_per_row;

        for (int n = 0; n < N; n++) {
            /* Build a temporary column of B (stride = N) */
            /* For small K this is cheap; for large K we'd prefer B transposed. */
            /* We access B[k][n] = B[k*N + n] */
            for (int k = 0; k < K; k++) col_buf[k] = B[k * N + n];

            C[m * N + n] = mxfp4_dot_row_col(a_packed, a_scales, col_buf, K);
        }
    }
    free(col_buf);
}

/**
 * gemm_mxfp4_mxfp4 — Both A and B are MXFP4; C is FP32.
 * Dequantizes B rows on the fly into a temporary FP32 buffer.
 */
void gemm_mxfp4_mxfp4(
        const mxfp4_tensor_t * __restrict__ A,
        const mxfp4_tensor_t * __restrict__ B,
        float                * __restrict__ C,
        int M, int N, int K)
{
    /* Dequantize B into FP32 scratch (K×N) */
    float *B_fp32 = (float*)malloc(K * N * sizeof(float));
    int b_packed_row = (N + 1) / 2;
    int b_scales_row = (N + 31) / 32;
    float block_buf[32];

    for (int k = 0; k < K; k++) {
        const uint8_t *bp = B->packed + k * b_packed_row;
        const uint8_t *bs = B->scales + k * b_scales_row;
        int num_blocks = (N + 31) / 32;
        for (int blk = 0; blk < num_blocks; blk++) {
            int elem_start = blk * 32;
            int elem_count = (elem_start + 32 <= N) ? 32 : (N - elem_start);
            mxfp4_decode_block(bp + blk * 16, bs[blk], block_buf);
            memcpy(B_fp32 + k * N + elem_start, block_buf, elem_count * sizeof(float));
        }
    }
    gemm_mxfp4_fp32(A, B_fp32, C, M, N, K);
    free(B_fp32);
}
