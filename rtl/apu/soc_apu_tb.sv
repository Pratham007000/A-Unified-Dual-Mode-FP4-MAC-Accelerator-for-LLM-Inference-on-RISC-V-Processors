// soc_apu_tb.sv — Testbench for CV32E40P (RI5CY) + MAC APU SoC
`timescale 1ns / 1ps

module soc_apu_tb;

    logic clk  = 0;
    logic rstn = 0;

    // 10ns clock (100 MHz)
    always #5 clk = ~clk;

    soc_apu_top DUT (
        .clk_i  (clk),
        .rst_ni (rstn)
    );

    initial begin
        $display("");
        $display("  ============================================");
        $display("  CV32E40P (RI5CY) + MAC APU Co-Processor");
        $display("  ============================================");
        $display("");

        rstn = 0;
        repeat(10) @(posedge clk);
        rstn = 1;

        $display("  CPU released from reset. Running firmware...");
        $display("");

        // Safety timeout
        repeat(100000) @(posedge clk);

        $display("");
        $display("  TIMEOUT: Firmware did not complete.");
        $finish;
    end

endmodule
