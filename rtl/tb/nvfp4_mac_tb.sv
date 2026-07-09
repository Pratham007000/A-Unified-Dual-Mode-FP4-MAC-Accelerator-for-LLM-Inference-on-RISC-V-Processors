// nvfp4_mac_tb.sv - Testbench for NVFP4 MAC unit
`timescale 1ns / 1ps

module nvfp4_mac_tb;

    reg         clk, rst, valid_in;
    reg  [7:0]  packed_byte, block_scale_code;
    reg  [31:0] tensor_scale;
    reg  [31:0] b_elem_0, b_elem_1;
    wire [31:0] accum_out;
    wire        valid_out;

    nvfp4_mac_top uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .packed_byte(packed_byte), .block_scale_code(block_scale_code),
        .tensor_scale(tensor_scale),
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
        rst = 1; valid_in = 0;
        packed_byte = 0; block_scale_code = 0;
        tensor_scale = 0; b_elem_0 = 0; b_elem_1 = 0;
        #100;
        rst = 0;
        #20;

        for (i = 0; i < NV_N; i = i + 1) begin
            // 1. Reset
            rst = 1;
            #20;
            rst = 0;
            #20;

            // 2. Setup at falling edge
            @(negedge clk);
            packed_byte      = nv_packed[i];
            block_scale_code = nv_bscale[i];
            tensor_scale     = nv_tscale[i];
            b_elem_0         = nv_b0[i];
            b_elem_1         = nv_b1[i];
            valid_in         = 1;

            // 3. Latch at rising edge
            @(posedge clk);

            // 4. Deassert valid
            @(negedge clk);
            valid_in = 0;

            // 5. Wait for output
            @(posedge clk);
            #2;

            // 6. Check
            if (accum_out == nv_expected[i]) begin
                $display("  PASS [%0d] accum=0x%08H expected=0x%08H", i, accum_out, nv_expected[i]);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL [%0d] accum=0x%08H expected=0x%08H", i, accum_out, nv_expected[i]);
            end
        end

        $display("");
        $display("  NVFP4 MAC: %0d/%0d passed", pass_count, NV_N);
        if (pass_count == NV_N)
            $display("  ALL PASS");
        else
            $display("  SOME FAILED");

        #50;
        $finish;
    end

endmodule
