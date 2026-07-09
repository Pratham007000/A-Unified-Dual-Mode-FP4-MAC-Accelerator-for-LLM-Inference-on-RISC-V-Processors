// firmware.c — C program that runs on PicoRV32 and uses MAC accelerator
// Compile with: riscv32-unknown-elf-gcc -O2 -march=rv32im -mabi=ilp32 
//               -nostdlib -T linker.ld firmware.c -o firmware.elf
// Convert:      riscv32-unknown-elf-objcopy -O verilog firmware.elf firmware.hex

// ---- MAC Register Addresses ----
#define MAC_BASE    0x40000000
#define MAC_CTRL    (*(volatile unsigned int*)(MAC_BASE + 0x00))
#define MAC_DATA    (*(volatile unsigned int*)(MAC_BASE + 0x04))
#define MAC_SCA     (*(volatile unsigned int*)(MAC_BASE + 0x08))
#define MAC_SCB     (*(volatile unsigned int*)(MAC_BASE + 0x0C))
#define MAC_TS      (*(volatile unsigned int*)(MAC_BASE + 0x10))
#define MAC_B0      (*(volatile unsigned int*)(MAC_BASE + 0x14))
#define MAC_B1      (*(volatile unsigned int*)(MAC_BASE + 0x18))
#define MAC_RESULT  (*(volatile unsigned int*)(MAC_BASE + 0x1C))

// ---- Result Storage (read by testbench) ----
#define DONE_FLAG   (*(volatile unsigned int*)0x0000FF00)
#define RESULT_MX   (*(volatile unsigned int*)0x0000FF04)
#define RESULT_NV   (*(volatile unsigned int*)0x0000FF08)

void main() {
    unsigned int result;
    
    // ============================================
    // TEST 1: MXFP4 mode (FORMAT_MODE = 0)
    // ============================================
    // packed=0xA3: lo=0x3(+1.5), hi=0xA(-1.0)
    // scale_a=127 (E8M0 = 2^0 = 1.0)
    // b0=0.5 (0x3F000000), b1=0.0 (0x00000000)
    // Expected: 1.5*1.0*0.5 + (-1.0)*1.0*0.0 = 0.75
    
    MAC_CTRL = 0x08;              // reset accumulator
    MAC_CTRL = 0x00;              // release reset, MXFP4 mode
    MAC_DATA = 0xA3;              // packed byte
    MAC_SCA  = 127;               // E8M0 scale = 2^(127-127) = 1.0
    MAC_B0   = 0x3F000000;        // 0.5 in FP32
    MAC_B1   = 0x00000000;        // 0.0 in FP32
    MAC_CTRL = 0x01;              // START + MXFP4 mode
    
    // Poll until done
    while (!(MAC_CTRL & 0x02));
    
    // Read and store result
    RESULT_MX = MAC_RESULT;
    
    // ============================================
    // TEST 2: NVFP4 mode (FORMAT_MODE = 1)
    // ============================================
    // packed=0x32: lo=0x2(+1.0), hi=0x3(+1.5)
    // scale_b=0x38 (FP8 E4M3 = 1.0)
    // tensor_scale=1.0 (0x3F800000)
    // b0=2.0 (0x40000000), b1=1.0 (0x3F800000)
    // Expected: 1.0*1.0*2.0 + 1.5*1.0*1.0 = 3.5
    
    MAC_CTRL = 0x0C;              // reset accumulator + NVFP4 mode
    MAC_CTRL = 0x04;              // release reset, NVFP4 mode
    MAC_DATA = 0x32;              // packed byte
    MAC_SCB  = 0x38;              // FP8 block scale = 1.0
    MAC_TS   = 0x3F800000;        // tensor scale = 1.0
    MAC_B0   = 0x40000000;        // 2.0 in FP32
    MAC_B1   = 0x3F800000;        // 1.0 in FP32
    MAC_CTRL = 0x05;              // START + NVFP4 mode
    
    // Poll until done
    while (!(MAC_CTRL & 0x02));
    
    // Read and store result
    RESULT_NV = MAC_RESULT;
    
    // ============================================
    // Signal completion
    // ============================================
    DONE_FLAG = 0xDEADBEEF;
    
    // Infinite loop (CPU halts here)
    while(1);
}
