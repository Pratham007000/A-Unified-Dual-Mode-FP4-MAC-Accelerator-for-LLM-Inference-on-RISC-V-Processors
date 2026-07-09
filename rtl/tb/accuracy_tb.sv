// accuracy_tb.sv - Large-scale accuracy testbench for unified MAC
// Compares RTL output against:
//   1. Quantized golden (Python's quantized result) - should match exactly
//   2. FP32 golden (perfect FP32 result) - measures quantization error
`timescale 1ns / 1ps

module accuracy_tb;

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
    `include "accuracy_test_vectors.vh"

    integer i;
    integer mx_exact_pass = 0, nv_exact_pass = 0;
    integer mx_total = 0, nv_total = 0;
    integer total_exact_pass = 0;

    // For MSE computation (comparing RTL vs FP32 golden)
    real rtl_float, fp32_float, quant_float;
    real mx_sq_err_sum, nv_sq_err_sum;
    real mx_mse, nv_mse, mx_rmse, nv_rmse;
    real err;

    // Convert 32-bit IEEE 754 to real for comparison
    function real bits_to_real;
        input [31:0] bits;
        reg sign;
        reg [7:0] exp_val;
        reg [22:0] mant;
        real result;
        real mant_real;
        integer exp_int;
        begin
            sign = bits[31];
            exp_val = bits[30:23];
            mant = bits[22:0];

            if (exp_val == 0 && mant == 0) begin
                result = 0.0;
            end else if (exp_val == 255) begin
                result = 0.0; // treat inf/nan as 0 for MSE
            end else if (exp_val == 0) begin
                // subnormal
                mant_real = $itor(mant) / 8388608.0;
                result = mant_real * (2.0 ** (-126));
            end else begin
                mant_real = 1.0 + $itor(mant) / 8388608.0;
                exp_int = $signed({1'b0, exp_val}) - 127;
                result = mant_real * (2.0 ** exp_int);
            end

            if (sign) result = -result;
            bits_to_real = result;
        end
    endfunction

    initial begin
        rst = 1; valid_in = 0; FORMAT_MODE = 0;
        packed_byte = 0; scale_code_a = 0; scale_code_b = 0;
        tensor_scale = 0; b_elem_0 = 0; b_elem_1 = 0;
        mx_sq_err_sum = 0.0;
        nv_sq_err_sum = 0.0;
        #100;
        rst = 0;
        #20;

        $display("");
        $display("  ============================================================");
        $display("  Unified MAC - Large-Scale Accuracy Test (100 vectors)");
        $display("  ============================================================");
        $display("  Comparing RTL output vs quantized golden AND vs FP32 golden");
        $display("  ============================================================");
        $display("");

        for (i = 0; i < ACC_N; i = i + 1) begin
            // Reset accumulator before each test
            rst = 1;
            #20;
            rst = 0;
            #20;

            // Setup inputs at falling edge
            @(negedge clk);
            FORMAT_MODE  = acc_mode[i];
            packed_byte  = acc_packed[i];
            scale_code_a = acc_sca[i];
            scale_code_b = acc_scb[i];
            tensor_scale = acc_ts[i];
            b_elem_0     = acc_b0[i];
            b_elem_1     = acc_b1[i];
            valid_in     = 1;

            @(posedge clk);
            @(negedge clk);
            valid_in = 0;
            @(posedge clk);
            #2;

            // Count by mode
            if (acc_mode[i] == 0) mx_total = mx_total + 1;
            else                  nv_total = nv_total + 1;

            // --- Check 1: Exact match vs quantized golden ---
            if (accum_out == acc_expected[i]) begin
                total_exact_pass = total_exact_pass + 1;
                if (acc_mode[i] == 0) mx_exact_pass = mx_exact_pass + 1;
                else                  nv_exact_pass = nv_exact_pass + 1;
            end else begin
                $display("  MISMATCH [%0d] mode=%s RTL=0x%08H expected=0x%08H",
                    i, acc_mode[i] ? "NVFP4" : "MXFP4", accum_out, acc_expected[i]);
            end

            // --- Check 2: MSE vs FP32 golden ---
            rtl_float  = bits_to_real(accum_out);
            fp32_float = bits_to_real(acc_fp32_golden[i]);
            err = rtl_float - fp32_float;

            if (acc_mode[i] == 0)
                mx_sq_err_sum = mx_sq_err_sum + (err * err);
            else
                nv_sq_err_sum = nv_sq_err_sum + (err * err);
        end

        // Compute MSE and RMSE
        mx_mse  = mx_sq_err_sum / $itor(mx_total);
        nv_mse  = nv_sq_err_sum / $itor(nv_total);
        mx_rmse = mx_mse ** 0.5;
        nv_rmse = nv_mse ** 0.5;

        $display("");
        $display("  ============================================================");
        $display("  RESULTS");
        $display("  ============================================================");
        $display("");
        $display("  --- Exact Match (RTL vs Quantized Golden) ---");
        $display("  MXFP4 mode : %0d/%0d exact matches", mx_exact_pass, mx_total);
        $display("  NVFP4 mode : %0d/%0d exact matches", nv_exact_pass, nv_total);
        $display("  Total      : %0d/%0d exact matches", total_exact_pass, ACC_N);
        $display("");
        $display("  --- Accuracy (RTL vs FP32 Golden Reference) ---");
        $display("  MXFP4 MSE  : %e", mx_mse);
        $display("  MXFP4 RMSE : %e", mx_rmse);
        $display("  NVFP4 MSE  : %e", nv_mse);
        $display("  NVFP4 RMSE : %e", nv_rmse);
        $display("");

        if (total_exact_pass == ACC_N)
            $display("  *** ALL %0d VECTORS PASS - Hardware matches quantized model ***", ACC_N);
        else
            $display("  *** %0d MISMATCHES DETECTED ***", ACC_N - total_exact_pass);

        $display("");
        $display("  ============================================================");

        #50;
        $finish;
    end

endmodule
