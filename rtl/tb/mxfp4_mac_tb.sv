// mxfp4_mac_tb.sv - Testbench for MXFP4 MAC unit
`timescale 1ns / 1ps

module mxfp4_mac_tb;

    reg         clk, rst, valid_in;
    reg  [7:0]  packed_byte, scale_code;
    reg  [31:0] b_elem_0, b_elem_1;
    wire [31:0] accum_out;
    wire        valid_out;

    mxfp4_mac_top uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .packed_byte(packed_byte), .scale_code(scale_code),
        .b_elem_0(b_elem_0), .b_elem_1(b_elem_1),
        .accum_out(accum_out), .valid_out(valid_out)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // Include test vectors
    `include "test_vectors.vh"

    integer i;
    integer pass_count = 0;

    initial begin
        // Initialize
        rst = 1; valid_in = 0;
        packed_byte = 0; scale_code = 0;
        b_elem_0 = 0; b_elem_1 = 0;
        #100;  // Hold reset
        rst = 0;
        #20;

        for (i = 0; i < MX_N; i = i + 1) begin
            // 1. Reset the accumulator
            rst = 1;
            #20;
            rst = 0;
            #20;

            // 2. Setup inputs BEFORE the clock edge
            @(negedge clk);  // Setup at falling edge
            packed_byte = mx_packed[i];
            scale_code  = mx_scale[i];
            b_elem_0    = mx_b0[i];
            b_elem_1    = mx_b1[i];
            valid_in    = 1;

            // 3. Wait for rising edge to latch
            @(posedge clk);

            // 4. Deassert valid at next falling edge
            @(negedge clk);
            valid_in = 0;

            // 5. Wait one more full cycle for output to settle
            @(posedge clk);
            #2;

            // 6. Check result
            if (accum_out == mx_expected[i]) begin
                $display("  PASS [%0d] accum=0x%08H expected=0x%08H", i, accum_out, mx_expected[i]);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL [%0d] accum=0x%08H expected=0x%08H", i, accum_out, mx_expected[i]);
            end
        end

        $display("");
        $display("  MXFP4 MAC: %0d/%0d passed", pass_count, MX_N);
        if (pass_count == MX_N)
            $display("  ALL PASS");
        else
            $display("  SOME FAILED");

        #50;
        $finish;
    end

endmodule
