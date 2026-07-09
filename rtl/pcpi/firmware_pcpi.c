/* firmware_pcpi.c
 * Bare-metal C program for PicoRV32 (RV32IMC).
 * Uses QMAC_CFG and QMAC_EXEC custom instructions via .insn directive.
 * No GCC modification needed.
 *
 * Compile:
 *   riscv64-unknown-elf-gcc -march=rv32imc -mabi=ilp32 \
 *       -nostdlib -nostartfiles -O2 \
 *       -Wl,-Ttext=0x0 -o firmware_pcpi.elf firmware_pcpi.c
 *
 * Convert to hex:
 *   riscv64-unknown-elf-objcopy -O verilog firmware_pcpi.elf firmware_pcpi.hex
 */

typedef unsigned int   uint32_t;
typedef unsigned char  uint8_t;

/* ---- I/O Ports (memory-mapped) ---- */
#define PRINT_PORT  (*(volatile uint32_t*)0x10000000)
#define FINISH_PORT (*(volatile uint32_t*)0x20000000)

/* ---- QMAC_CFG: configure MAC (funct3=0) ----
 * rs1 = mode[0] | packed_byte[8:1] | scale_a[16:9] | scale_b[24:17]
 * rs2 = tensor_scale (FP32 bits)
 * No result written back.
 */
static inline void qmac_cfg(uint8_t mode, uint8_t packed_byte,
                             uint8_t scale_a, uint8_t scale_b,
                             uint32_t tensor_scale_bits) {
    uint32_t cfg_word = (uint32_t)(mode & 1)
                      | ((uint32_t)packed_byte << 1)
                      | ((uint32_t)scale_a << 9)
                      | ((uint32_t)scale_b << 17);
    uint32_t dummy;
    __asm__ volatile (
        ".insn r 0x0B, 0, 0, %0, %1, %2"
        : "=r"(dummy)
        : "r"(cfg_word), "r"(tensor_scale_bits)
    );
}

/* ---- QMAC_EXEC: execute MAC (funct3=1) ----
 * rs1 = b_elem_0 (FP32 bits)
 * rs2 = b_elem_1 (FP32 bits)
 * rd  = accum_out (FP32 result)
 */
static inline uint32_t qmac_exec(uint32_t b0_bits, uint32_t b1_bits) {
    uint32_t result;
    __asm__ volatile (
        ".insn r 0x0B, 1, 0, %0, %1, %2"
        : "=r"(result)
        : "r"(b0_bits), "r"(b1_bits)
    );
    return result;
}

/* ---- Entry point (no startup code, linker places this at 0x0) ---- */
void _start(void) {
    uint32_t result;

    /* ========================================
     * TEST 1: MXFP4 mode
     * ========================================
     * packed = 0xA3: lo=0x3(+1.5), hi=0xA(-1.0)
     * scale_a = 127 → E8M0 = 2^(127-127) = 1.0
     * b0 = 0.5, b1 = 0.0
     * Expected: 1.5 * 1.0 * 0.5 + (-1.0) * 1.0 * 0.0 = 0.75
     */
    qmac_cfg(
        0,          /* mode = MXFP4 */
        0xA3,       /* packed_byte */
        127,        /* scale_a (E8M0 = 1.0) */
        0,          /* scale_b (unused in MXFP4) */
        0x3F800000  /* tensor_scale = 1.0 (unused in MXFP4) */
    );

    result = qmac_exec(
        0x3F000000,  /* b0 = 0.5 */
        0x00000000   /* b1 = 0.0 */
    );

    PRINT_PORT = result;  /* Should print 0x3F400000 (0.75) */

    /* ========================================
     * TEST 2: NVFP4 mode
     * ========================================
     * packed = 0x32: lo=0x2(+1.0), hi=0x3(+1.5)
     * scale_b = 0x38 → FP8 E4M3 = 1.0
     * tensor_scale = 1.0
     * b0 = 2.0, b1 = 1.0
     * Expected: 1.0 * 1.0 * 2.0 + 1.5 * 1.0 * 1.0 = 3.5
     */
    qmac_cfg(
        1,          /* mode = NVFP4 */
        0x32,       /* packed_byte */
        0,          /* scale_a (unused in NVFP4) */
        0x38,       /* scale_b (FP8 = 1.0) */
        0x3F800000  /* tensor_scale = 1.0 */
    );

    result = qmac_exec(
        0x40000000,  /* b0 = 2.0 */
        0x3F800000   /* b1 = 1.0 */
    );

    PRINT_PORT = result;  /* Should print 0x40600000 (3.5) */

    /* ========================================
     * Signal completion
     * ======================================== */
    FINISH_PORT = 1;

    /* Halt */
    while(1);
}
