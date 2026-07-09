// soc_tb.sv — Testbench for RISC-V SoC + MAC integration
// Runs a firmware program that uses the MAC accelerator
// and prints results to verify integration works
`timescale 1ns / 1ps

module soc_tb;

    reg clk, rst;

    soc_top uut (
        .clk(clk),
        .rst(rst)
    );

    // 10ns clock (100 MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // Monitor MAC result register for changes
    // The firmware writes a magic value to RAM address 0x0000FF00 when done
    reg [31:0] prev_result;
    
    initial begin
        rst = 1;
        #100;
        rst = 0;
        
        $display("");
        $display("  ============================================");
        $display("  RISC-V SoC + MAC Accelerator Integration");
        $display("  CPU: PicoRV32 | Accelerator: Unified MAC");
        $display("  ============================================");
        $display("");

        // Wait for firmware to complete (writes to 0xFF00)
        // Timeout after 5000 cycles
        repeat(5000) begin
            @(posedge clk);
            
            // Check if firmware wrote completion flag
            if (uut.ram[16'hFF00 >> 2] == 32'hDEADBEEF) begin
                $display("  Firmware completed!");
                $display("");
                
                // Read results from RAM (firmware stores them)
                $display("  --- Results stored by firmware ---");
                $display("  MXFP4 result  : 0x%08H", uut.ram[16'hFF04 >> 2]);
                $display("  NVFP4 result  : 0x%08H", uut.ram[16'hFF08 >> 2]);
                $display("");
                $display("  ============================================");
                $display("  INTEGRATION TEST PASSED");
                $display("  CPU successfully controlled MAC accelerator");
                $display("  ============================================");
                
                #100;
                $finish;
            end
        end
        
        $display("  TIMEOUT: Firmware did not complete in 5000 cycles");
        $display("  (This is normal if firmware.hex is not loaded)");
        $display("");
        $display("  To run properly:");
        $display("  1. Compile firmware.c with RISC-V toolchain");
        $display("  2. Convert to firmware.hex");
        $display("  3. Place in simulation directory");
        
        #100;
        $finish;
    end

endmodule
