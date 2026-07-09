// soc_pcpi_tb.sv — Testbench for PicoRV32 + MAC PCPI SoC
// Runs firmware, waits for finish signal or timeout
`timescale 1ns / 1ps

module soc_pcpi_tb;

    reg clk = 0;
    reg resetn = 0;

    // 10ns clock (100 MHz)
    always #5 clk = ~clk;

    soc_pcpi_top DUT (
        .clk    (clk),
        .resetn (resetn)
    );

    initial begin
        $display("");
        $display("  ============================================");
        $display("  PicoRV32 + MAC PCPI Co-Processor Integration");
        $display("  ============================================");
        $display("");

        // Reset
        resetn = 0;
        repeat(5) @(posedge clk);
        resetn = 1;

        $display("  CPU released from reset. Running firmware...");
        $display("");

        // Safety timeout — firmware should finish well before this
        repeat(50000) @(posedge clk);

        $display("");
        $display("  TIMEOUT: Firmware did not complete in 50000 cycles.");
        $display("  Check that firmware_pcpi.hex is in the simulation directory.");
        $finish;
    end

endmodule
