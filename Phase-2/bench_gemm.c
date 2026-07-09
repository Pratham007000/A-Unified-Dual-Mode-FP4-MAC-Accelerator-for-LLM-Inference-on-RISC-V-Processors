/**
 * bench_gemm.c — Correctness + performance benchmark for MXFP4 and NVFP4 GEMM kernels.
 *
 * Tests:
 *   1. Correctness: compare quantized GEMM output vs naive FP32 reference.
 *   2. Performance: measure instruction count via rdinstret CSR and wall-clock cycles
 *      via rdcycle CSR (both standard on RISC-V).
 *   3. Head-to-head: side-by-side MXFP4 vs NVFP4 on the same matrix sizes.
 *
 * Compile:
 *   riscv64-linux-gnu-gcc -O2 -march=rv64gcv -static -o bench_gemm bench_gemm.c -lm
 *                          
 * Run (QEMU):
 *   QEMU_CPU=rv64,v=true,vlen=256 qemu-riscv64 ./bench_gemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "quant_types.h"
#include "gemm_mxfp4.h"
#include "gemm_nvfp4.h"

/* ── Timing helper (works on QEMU user-mode) ───────────────────── */
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ── Simple PRNG (xorshift64) ───────────────────────────────────── */
static uint64_t rng_state = 0xdeadbeefcafe1234ULL;
static float rand_float(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    /* [-1, 1] */
    return (float)(int64_t)rng_state * (1.0f / (float)(1ULL << 63));
}

/* ── Reference FP32 GEMM ────────────────────────────────────────── */
static void gemm_fp32_ref(const float *A, const float *B, float *C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float a = A[m * K + k];
            for (int n = 0; n < N; n++)
                C[m * N + n] += a * B[k * N + n];
        }
}

/* ── Quantization helpers (mirror Python logic, scalar C) ──────── */

/* Pack float array into MXFP4 (returns packed nibbles + E8M0 scales) */
static void quantize_mxfp4(const float *src, int n,
                             uint8_t *packed, uint8_t *scales) {
    int num_blocks = (n + 31) / 32;
    for (int blk = 0; blk < num_blocks; blk++) {
        int start = blk * 32;
        int count = (start + 32 <= n) ? 32 : (n - start);

        /* Find max abs in block */
        float max_abs = 0.0f;
        for (int i = 0; i < count; i++) {
            float v = fabsf(src[start + i]);
            if (v > max_abs) max_abs = v;
        }

        /* E8M0 scale: largest power-of-2 such that max_abs/scale <= 6.0 */
        float scale;
        if (max_abs == 0.0f) {
            scale = 1.0f;
            scales[blk] = 127; /* 2^0 = 1 */
        } else {
            float raw = max_abs / 6.0f;
            int exp = (int)floorf(log2f(raw + 1e-38f));
            int code = exp + 127;
            if (code < 0) code = 0;
            if (code > 254) code = 254;
            scales[blk] = (uint8_t)code;
            scale = ldexpf(1.0f, exp);
        }

        /* Encode 32 elements as E2M1 nibbles */
        uint8_t codes[32] = {0};
        static const float vals[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
        for (int i = 0; i < count; i++) {
            float v = src[start + i];
            uint8_t sign = (v < 0) ? 8 : 0;
            float av = (v < 0) ? -v : v;
            av = av / scale;
            if (av > 6.0f) av = 6.0f;
            /* Find nearest value */
            int best = 0;
            float best_d = fabsf(av - vals[0]);
            for (int j = 1; j < 8; j++) {
                float d = fabsf(av - vals[j]);
                if (d < best_d || (d == best_d && j > best)) {
                    best_d = d; best = j;
                }
            }
            codes[i] = sign | (uint8_t)best;
        }
        /* Pack two 4-bit codes per byte */
        for (int b = 0; b < 16; b++) {
            uint8_t lo = (b*2 < count) ? (codes[b*2] & 0xF) : 0;
            uint8_t hi = (b*2+1 < count) ? (codes[b*2+1] & 0xF) : 0;
            packed[blk * 16 + b] = lo | (hi << 4);
        }
    }
}

/* ── FP8 E4M3 quantizer (forward declared, defined below) ──────── */
static int fp8_e4m3_quantize_approx(float x);

/* Pack float array into NVFP4 — split into tensor-scale computation and per-row quantization
 * so the benchmark can quantize per-row (matching GEMM kernel's per-row layout). */

/* Compute per-tensor FP32 scale for NVFP4 */
static float compute_nvfp4_tensor_scale(const float *src, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(src[i]);
        if (v > max_abs) max_abs = v;
    }
    return (max_abs == 0.0f) ? 1.0f : (max_abs / (240.0f * 6.0f));
}

/* Quantize a single row to NVFP4 given a pre-computed tensor scale */
static void quantize_nvfp4_row(const float *src, int n,
                                uint8_t *packed, uint8_t *block_scales,
                                float tensor_scale) {
    int num_blocks = (n + 15) / 16;
    static const float vals[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};

    for (int blk = 0; blk < num_blocks; blk++) {
        int start = blk * 16;
        int count = (start + 16 <= n) ? 16 : (n - start);

        /* Pre-scale by tensor_scale */
        float pre[16] = {0};
        float bmax = 0.0f;
        for (int i = 0; i < count; i++) {
            pre[i] = src[start + i] / tensor_scale;
            float v = fabsf(pre[i]);
            if (v > bmax) bmax = v;
        }

        /* FP8 E4M3 block scale */
        float bscale = (bmax == 0.0f) ? 1.0f : (bmax / 6.0f);
        if (bscale > 240.0f) bscale = 240.0f;
        block_scales[blk] = (uint8_t)fp8_e4m3_quantize_approx(bscale);

        float actual_bscale = fp8_e4m3_decode(block_scales[blk]);
        if (actual_bscale == 0.0f) actual_bscale = 1.0f;

        /* Encode 16 E2M1 nibbles */
        uint8_t codes[16] = {0};
        for (int i = 0; i < count; i++) {
            float v = pre[i] / actual_bscale;
            uint8_t sign = (v < 0) ? 8 : 0;
            float av = (v < 0) ? -v : v;
            if (av > 6.0f) av = 6.0f;
            int best = 0;
            float best_d = fabsf(av - vals[0]);
            for (int j = 1; j < 8; j++) {
                float d = fabsf(av - vals[j]);
                if (d < best_d || (d == best_d && j > best)) {
                    best_d = d; best = j;
                }
            }
            codes[i] = sign | (uint8_t)best;
        }
        for (int b = 0; b < 8; b++) {
            uint8_t lo = (b*2 < count) ? (codes[b*2] & 0xF) : 0;
            uint8_t hi = (b*2+1 < count) ? (codes[b*2+1] & 0xF) : 0;
            packed[blk * 8 + b] = lo | (hi << 4);
        }
    }
}

/* Approximate FP8 E4M3 quantizer (C scalar, mirrors fp_utils.py) */
static int fp8_e4m3_quantize_approx(float x) {
    if (x == 0.0f) return 0;
    int sign = (x < 0) ? 1 : 0;
    float ax = (x < 0) ? -x : x;
    if (ax > 240.0f) ax = 240.0f;
    float min_normal = ldexpf(1.0f, -6);
    float log2_ax = (ax >= min_normal) ? log2f(ax) : -6.0f;
    int exp_unbiased = (int)floorf(log2_ax);
    int exp_biased = exp_unbiased + 7;
    if (exp_biased < 0) exp_biased = 0;
    if (exp_biased > 14) exp_biased = 14;
    float mant_f;
    if (exp_biased == 0) {
        mant_f = ax / ldexpf(1.0f, -6);
    } else {
        mant_f = ax / ldexpf(1.0f, exp_biased - 7) - 1.0f;
    }
    int mant = (int)roundf(mant_f * 8.0f);
    if (mant < 0) mant = 0;
    if (mant > 7) mant = 7;
    return (sign << 7) | (exp_biased << 3) | mant;
}

/* ── RMSE between two FP32 arrays ───────────────────────────────── */
static float rmse(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum / n);
}

/* ── Benchmark one kernel ────────────────────────────────────────── */
typedef void (*gemm_fn)(const void*, const float*, float*, int, int, int);

typedef struct {
    uint64_t instret;
    uint64_t cycles;
    float    rmse_vs_fp32;
} bench_result_t;

/* ── Test case ───────────────────────────────────────────────────── */
static void run_test(const char *name, int M, int K, int N) {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s║\n", name);
    printf("║  M=%-4d K=%-4d N=%-4d   (GEMM: %lld ops)%-14s║\n",
           M, K, N, (long long)2*M*K*N, "");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    /* Generate random FP32 matrices */
    float *A_fp32 = malloc(M * K * sizeof(float));
    float *B_fp32 = malloc(K * N * sizeof(float));
    float *C_ref  = malloc(M * N * sizeof(float));
    float *C_mx   = malloc(M * N * sizeof(float));
    float *C_nv   = malloc(M * N * sizeof(float));

    rng_state = 0x123456789abcdef0ULL + M * 1000 + K;
    for (int i = 0; i < M * K; i++) A_fp32[i] = rand_float();
    rng_state = 0xfedcba9876543210ULL + K * 1000 + N;
    for (int i = 0; i < K * N; i++) B_fp32[i] = rand_float();

    /* FP32 reference */
    gemm_fp32_ref(A_fp32, B_fp32, C_ref, M, N, K);

    /* ── MXFP4 ── */
    int mx_packed_bytes = M * ((K + 1) / 2);
    int mx_scale_bytes  = M * ((K + 31) / 32);
    uint8_t *mx_packed = malloc(mx_packed_bytes);
    uint8_t *mx_scales = malloc(mx_scale_bytes);

    memset(mx_packed, 0, mx_packed_bytes);
    memset(mx_scales, 127, mx_scale_bytes);

    for (int m = 0; m < M; m++) {
        quantize_mxfp4(A_fp32 + m * K, K,
                        mx_packed + m * ((K+1)/2),
                        mx_scales + m * ((K+31)/32));
    }

    mxfp4_tensor_t A_mx = {
        .packed = mx_packed, .scales = mx_scales,
        .rows = M, .cols = K, .block_size = 32
    };

    uint64_t t0, t1;
    t0 = now_ns();
    gemm_mxfp4_fp32(&A_mx, B_fp32, C_mx, M, N, K);
    t1 = now_ns();

    float rmse_mx = rmse(C_ref, C_mx, M * N);
    printf("  MXFP4  │ time: %10llu ns │ RMSE: %.4e\n",
           (unsigned long long)(t1-t0), rmse_mx);

    /* ── NVFP4 ── */
    int nv_packed_bytes = M * ((K + 1) / 2);
    int nv_bscale_bytes = M * ((K + 15) / 16);
    uint8_t *nv_packed  = malloc(nv_packed_bytes);
    uint8_t *nv_bscales = malloc(nv_bscale_bytes);
    float nv_tensor_scale = 1.0f;

    memset(nv_packed,  0, nv_packed_bytes);
    memset(nv_bscales, 0, nv_bscale_bytes);

    /* Compute global tensor scale over entire matrix, then quantize per-row */
    nv_tensor_scale = compute_nvfp4_tensor_scale(A_fp32, M * K);
    for (int m = 0; m < M; m++) {
        quantize_nvfp4_row(A_fp32 + m * K, K,
                           nv_packed  + m * ((K+1)/2),
                           nv_bscales + m * ((K+15)/16),
                           nv_tensor_scale);
    }

    nvfp4_tensor_t A_nv = {
        .packed = nv_packed, .block_scales = nv_bscales,
        .tensor_scale = nv_tensor_scale,
        .rows = M, .cols = K, .block_size = 16
    };

    t0 = now_ns();
    gemm_nvfp4_fp32(&A_nv, B_fp32, C_nv, M, N, K);
    t1 = now_ns();

    float rmse_nv = rmse(C_ref, C_nv, M * N);
    printf("  NVFP4  │ time: %10llu ns │ RMSE: %.4e\n",
           (unsigned long long)(t1-t0), rmse_nv);

    /* ── Memory overhead ── */
    int n_elem = M * K;
    int mx_total = mx_packed_bytes + mx_scale_bytes;
    int nv_total = nv_packed_bytes + nv_bscale_bytes + 4;
    printf("  Memory │ MXFP4: %d B (scales: %d B) │ NVFP4: %d B (bscales: %d B)\n",
           mx_total, mx_scale_bytes, nv_total, nv_bscale_bytes);
    printf("  Winner │ Accuracy: %-6s  Memory: %-6s\n",
           (rmse_nv < rmse_mx) ? "NVFP4" : "MXFP4",
           (mx_total < nv_total) ? "MXFP4" : "NVFP4");

    free(A_fp32); free(B_fp32); free(C_ref); free(C_mx); free(C_nv);
    free(mx_packed); free(mx_scales); free(nv_packed); free(nv_bscales);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║       Phase 2: MXFP4 vs NVFP4 GEMM Benchmark            ║\n");
    printf("║       RVV on RISC-V (QEMU, VLEN=256)                     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    /* Sizes representative of LLM linear layers in TinyLlama:
       hidden=2048, intermediate=5632, vocab=32000 (projected down) */
    run_test("Small: sanity check",          4,   64,   4);
    run_test("Token embed projection",       8,  256,   8);
    run_test("Attention QKV (tiny)",        16,  256,  16);
    run_test("FFN hidden → intermediate",   16,  512,  32);
    run_test("TinyLlama-like: 32×512×128",  32,  512, 128);

    printf("\n✓ Benchmark complete.\n");
    return 0;
}
