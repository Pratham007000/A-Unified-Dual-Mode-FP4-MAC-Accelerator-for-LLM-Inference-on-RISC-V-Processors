// unified_mac_tb.sv - Testbench for unified dual-mode MAC unit
`timescale 1ns / 1ps

module unified_mac_tb;

    reg         clk, rst, valid_in;
    reg         FORMAT_MODE;
    reg  [7:0]  packed_byte, scale_code_a, scale_code_b;
    reg  [31:0] tensor_scale;
    reg  [31:0] b_elem_0, b_elem_1;
    wire [31:0] accum_out;
    wire        valid_out, mode_out;

    unified_mac_top uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .FORMAT_MODE(FORMAT_MODE),
        .packed_byte(packed_byte),
        .b_elem_0(b_elem_0), .b_elem_1(b_elem_1),
        .scale_code_a(scale_code_a),
        .scale_code_b(scale_code_b),
        .tensor_scale(tensor_scale),
        .accum_out(accum_out),
        .valid_out(valid_out),
        .mode_out(mode_out)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // Include test vectors
    `include "unified_test_vectors.vh"

    integer i;
    integer pass_count = 0;
    integer mx_pass = 0, nv_pass = 0;
    integer mx_total = 0, nv_total = 0;

    initial begin
        rst = 1; valid_in = 0; FORMAT_MODE = 0;
        packed_byte = 0; scale_code_a = 0; scale_code_b = 0;
        tensor_scale = 0; b_elem_0 = 0; b_elem_1 = 0;
        #100;
        rst = 0;
        #20;

        $display("");
        $display("  ============================================");
        $display("  Unified MXFP4/NVFP4 MAC - Testbench");
        $display("  ============================================");
        $display("");

        for (i = 0; i < U_N; i = i + 1) begin
            // Reset accumulator
            rst = 1;
            #20;
            rst = 0;
            #20;

            // Setup at falling edge
            @(negedge clk);
            FORMAT_MODE  = u_mode[i];
            packed_byte  = u_packed[i];
            scale_code_a = u_sca[i];
            scale_code_b = u_scb[i];
            tensor_scale = u_ts[i];
            b_elem_0     = u_b0[i];
            b_elem_1     = u_b1[i];
            valid_in     = 1;

            // Latch at rising edge
            @(posedge clk);

            // Deassert valid
            @(negedge clk);
            valid_in = 0;

            // Wait for output
            @(posedge clk);
            #2;

            // Check result
            if (u_mode[i] == 0) mx_total = mx_total + 1;
            else                nv_total = nv_total + 1;

            if (accum_out == u_expected[i]) begin
                $display("  PASS [%0d] mode=%s accum=0x%08H expected=0x%08H",
                    i, u_mode[i] ? "NVFP4" : "MXFP4", accum_out, u_expected[i]);
                pass_count = pass_count + 1;
                if (u_mode[i] == 0) mx_pass = mx_pass + 1;
                else                nv_pass = nv_pass + 1;
            end else begin
                $display("  FAIL [%0d] mode=%s accum=0x%08H expected=0x%08H",
                    i, u_mode[i] ? "NVFP4" : "MXFP4", accum_out, u_expected[i]);
            end
        end

        $display("");
        $display("  ============================================");
        $display("  RESULTS");
        $display("  ============================================");
        $display("  MXFP4 mode : %0d/%0d passed", mx_pass, mx_total);
        $display("  NVFP4 mode : %0d/%0d passed", nv_pass, nv_total);
        $display("  Total      : %0d/%0d passed", pass_count, U_N);
        if (pass_count == U_N)
            $display("  ALL PASS");
        else
            $display("  SOME FAILED");
        $display("  ============================================");

        #50;
        $finish;
    end

endmodule
