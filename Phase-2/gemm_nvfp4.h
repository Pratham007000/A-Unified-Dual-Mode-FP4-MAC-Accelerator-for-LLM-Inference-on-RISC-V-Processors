/**
 * gemm_nvfp4.h — RVV-optimized GEMM for NVFP4
 *
 * Computes: C[M×N] = A[M×K] × B[K×N]  (FP32 accumulation)
 *
 * A is NVFP4 (packed E2M1 nibbles + FP8-E4M3 block scales (block=16) + FP32 tensor scale)
 * B is FP32
 * C is FP32
 *
 * Key difference from MXFP4:
 *   - Block size = 16 (half of MXFP4's 32)
 *   - Per-block scale is FP8 E4M3 (not E8M0 power-of-two)
 *   - Additional per-tensor FP32 scale multiplied in
 *   - Dequant: val = e2m1(code) * fp8_e4m3(block_scale) * tensor_scale
 *
 * RVV strategy: same as MXFP4 but inner block loop processes 16 elements.
 * The extra tensor_scale multiply is hoisted outside the block loop and
 * folded into the block scale at decode time: effective_scale = block_scale * tensor_scale.
 */
#pragma once
#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "quant_types.h"

/* ── Decode one 16-element NVFP4 block into float scratch buffer ── */
static inline void nvfp4_decode_block(
        const uint8_t * __restrict__ packed_base, /* 8 bytes for 16 elements */
        uint8_t block_scale_code,
        float   tensor_scale,
        float  * __restrict__ out)                /* 16 floats output        */
{
    /* Effective scale = fp8_e4m3(block_scale_code) * tensor_scale */
    float eff_scale = fp8_e4m3_decode(block_scale_code) * tensor_scale;

    /* Unpack 8 bytes → 16 nibble codes */
    uint8_t codes[16];
    for (int b = 0; b < 8; b++) {
        codes[2*b]   = packed_base[b] & 0xF;
        codes[2*b+1] = packed_base[b] >> 4;
    }

    /* RVV: u8m1 → u16m2 → u32m4 widening, gather, fmul */
    int i = 0;
    while (i < 16) {
        size_t vl = __riscv_vsetvl_e8m1(16 - i);
        vuint8m1_t  vcodes8  = __riscv_vle8_v_u8m1(codes + i, vl);
        vuint16m2_t vcodes16 = __riscv_vzext_vf2_u16m2(vcodes8, vl);
        vuint32m4_t vcodes32 = __riscv_vzext_vf2_u32m4(vcodes16, vl);
        vuint32m4_t voffsets = __riscv_vsll_vx_u32m4(vcodes32, 2, vl);
        vfloat32m4_t vvals   = __riscv_vloxei32_v_f32m4(E2M1_TABLE, voffsets, vl);
        vfloat32m4_t vscaled = __riscv_vfmul_vf_f32m4(vvals, eff_scale, vl);
        __riscv_vse32_v_f32m4(out + i, vscaled, vl);
        i += (int)vl;
    }
}

/* ── Single output element: dot product of one NVFP4 row with FP32 column ── */
static float nvfp4_dot_row_col(
        const uint8_t * __restrict__ a_packed,
        const uint8_t * __restrict__ a_block_scales,
        float           tensor_scale,
        const float   * __restrict__ b_col,
        int K)
{
    float accum = 0.0f;
    float block_buf[16];
    int num_blocks = (K + 15) / 16;

    for (int blk = 0; blk < num_blocks; blk++) {
        int elem_start = blk * 16;
        int elem_count = (elem_start + 16 <= K) ? 16 : (K - elem_start);

        nvfp4_decode_block(
            a_packed + blk * 8,
            a_block_scales[blk],
            tensor_scale,
            block_buf);

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
        vl = __riscv_vsetvlmax_e32m4();
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
        vfloat32m1_t vred  = __riscv_vfredusum_vs_f32m4_f32m1(vacc, vzero, vl);
        accum += __riscv_vfmv_f_s_f32m1_f32(vred);
    }
    return accum;
}

/**
 * gemm_nvfp4_fp32 — Main GEMM: A (NVFP4) × B (FP32) → C (FP32)
 *
 * A row i: packed nibbles at a->packed + i * ceil(K/2)
 *          block scales at  a->block_scales + i * ceil(K/16)
 * Tensor scale: a->tensor_scale (scalar, applies to all blocks)
 */
void gemm_nvfp4_fp32(
        const nvfp4_tensor_t * __restrict__ A,
        const float          * __restrict__ B,
        float                * __restrict__ C,
        int M, int N, int K)
{
    int packed_row_bytes  = (K + 1) / 2;
    int bscales_per_row   = (K + 15) / 16;
    float ts = A->tensor_scale;

    float *col_buf = (float*)malloc(K * sizeof(float));

    for (int m = 0; m < M; m++) {
        const uint8_t *a_packed = A->packed       + m * packed_row_bytes;
        const uint8_t *a_bscale = A->block_scales + m * bscales_per_row;

        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) col_buf[k] = B[k * N + n];
            C[m * N + n] = nvfp4_dot_row_col(a_packed, a_bscale, ts, col_buf, K);
        }
    }
    free(col_buf);
}

/**
 * gemm_nvfp4_nvfp4 — Both A and B are NVFP4; C is FP32.
 */
void gemm_nvfp4_nvfp4(
        const nvfp4_tensor_t * __restrict__ A,
        const nvfp4_tensor_t * __restrict__ B,
        float                * __restrict__ C,
        int M, int N, int K)
{
    /* Dequantize B into FP32 scratch (K×N) */
    float *B_fp32 = (float*)malloc(K * N * sizeof(float));
    int b_packed_row = (N + 1) / 2;
    int b_bscale_row = (N + 15) / 16;
    float ts = B->tensor_scale;
    float block_buf[16];

    for (int k = 0; k < K; k++) {
        const uint8_t *bp = B->packed       + k * b_packed_row;
        const uint8_t *bs = B->block_scales + k * b_bscale_row;
        int num_blocks = (N + 15) / 16;
        for (int blk = 0; blk < num_blocks; blk++) {
            int elem_start = blk * 16;
            int elem_count = (elem_start + 16 <= N) ? 16 : (N - elem_start);
            nvfp4_decode_block(bp + blk * 8, bs[blk], ts, block_buf);
            memcpy(B_fp32 + k * N + elem_start, block_buf, elem_count * sizeof(float));
        }
    }
    gemm_nvfp4_fp32(A, B_fp32, C, M, N, K);
    free(B_fp32);
}
