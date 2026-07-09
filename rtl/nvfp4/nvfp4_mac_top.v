// nvfp4_mac_top.v - NVFP4 Multiply-Accumulate Unit
// Processes 2 elements per cycle (1 packed byte)
// Key difference from MXFP4: uses FP8 E4M3 block scale + FP32 tensor scale
`timescale 1ns / 1ps

module nvfp4_mac_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire [7:0]  packed_byte,       // 2 packed E2M1 elements
    input  wire [7:0]  block_scale_code,  // FP8 E4M3 block scale
    input  wire [31:0] tensor_scale,      // FP32 per-tensor scale
    input  wire [31:0] b_elem_0,          // FP32 B element (even)
    input  wire [31:0] b_elem_1,          // FP32 B element (odd)
    output reg  [31:0] accum_out,
    output reg         valid_out
);

    // ???? Stage 1: Decode (combinational) ????
    wire [3:0]  lo_nib, hi_nib;
    wire [31:0] decoded_0, decoded_1;
    wire [31:0] block_scale_fp32;

    nibble_extractor u_unpack (
        .packed_byte(packed_byte),
        .lo_nibble(lo_nib),
        .hi_nibble(hi_nib)
    );

    e2m1_decoder     u_dec0  (.code(lo_nib), .fp32_val(decoded_0));
    e2m1_decoder     u_dec1  (.code(hi_nib), .fp32_val(decoded_1));
    fp8_e4m3_decoder u_bscale (.code(block_scale_code), .fp32_val(block_scale_fp32));

    // ???? Stage 1b: Effective scale = block_scale × tensor_scale ????
    // (This is the EXTRA multiply that NVFP4 needs vs MXFP4)
    wire [31:0] eff_scale;
    fp32_mul u_eff_scale (.a(block_scale_fp32), .b(tensor_scale), .result(eff_scale));

    // ???? Stage 2: Scale multiply (decoded × effective_scale) ????
    wire [31:0] scaled_0, scaled_1;
    fp32_mul u_smul0 (.a(decoded_0), .b(eff_scale), .result(scaled_0));
    fp32_mul u_smul1 (.a(decoded_1), .b(eff_scale), .result(scaled_1));

    // ???? Stage 3: Dot multiply (scaled × B_element) ????
    wire [31:0] prod_0, prod_1;
    fp32_mul u_dmul0 (.a(scaled_0), .b(b_elem_0), .result(prod_0));
    fp32_mul u_dmul1 (.a(scaled_1), .b(b_elem_1), .result(prod_1));

    // ???? Stage 4: Accumulate ????
    wire [31:0] sum_01;
    wire [31:0] new_accum;
    fp32_add u_add01  (.a(prod_0),    .b(prod_1),  .result(sum_01));
    fp32_add u_accadd (.a(accum_out), .b(sum_01),  .result(new_accum));

    // ???? Register update ????
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accum_out <= 32'd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            accum_out <= new_accum;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule
