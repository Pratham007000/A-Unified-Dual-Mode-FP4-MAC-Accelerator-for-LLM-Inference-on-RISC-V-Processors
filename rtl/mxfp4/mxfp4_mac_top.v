// mxfp4_mac_top.v - MXFP4 Multiply-Accumulate Unit
// Processes 2 elements per cycle (1 packed byte)
// Pipeline: DECODE ? SCALE ? MULTIPLY ? ACCUMULATE
`timescale 1ns / 1ps

module mxfp4_mac_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,      // high when inputs are valid
    input  wire [7:0]  packed_byte,   // 2 packed E2M1 elements
    input  wire [7:0]  scale_code,    // E8M0 block scale (shared per 32 elements)
    input  wire [31:0] b_elem_0,      // FP32 B element for even index
    input  wire [31:0] b_elem_1,      // FP32 B element for odd index
    output reg  [31:0] accum_out,     // FP32 accumulated result
    output reg         valid_out      // high when accum_out has been updated
);

    // ???? Stage 1: Decode (combinational) ????
    wire [3:0]  lo_nib, hi_nib;
    wire [31:0] decoded_0, decoded_1;
    wire [31:0] scale_fp32;

    nibble_extractor u_unpack (
        .packed_byte(packed_byte),
        .lo_nibble(lo_nib),
        .hi_nibble(hi_nib)
    );

    e2m1_decoder u_dec0 (.code(lo_nib), .fp32_val(decoded_0));
    e2m1_decoder u_dec1 (.code(hi_nib), .fp32_val(decoded_1));
    e8m0_decoder u_scale (.code(scale_code), .fp32_scale(scale_fp32));

    // ???? Stage 2: Scale multiply (decoded × scale) ????
    wire [31:0] scaled_0, scaled_1;

    fp32_mul u_smul0 (.a(decoded_0), .b(scale_fp32), .result(scaled_0));
    fp32_mul u_smul1 (.a(decoded_1), .b(scale_fp32), .result(scaled_1));

    // ???? Stage 3: Dot multiply (scaled × B_element) ????
    wire [31:0] prod_0, prod_1;

    fp32_mul u_dmul0 (.a(scaled_0), .b(b_elem_0), .result(prod_0));
    fp32_mul u_dmul1 (.a(scaled_1), .b(b_elem_1), .result(prod_1));

    // ???? Stage 4: Accumulate ????
    wire [31:0] sum_01;         // prod_0 + prod_1
    wire [31:0] new_accum;      // accum + sum_01

    fp32_add u_add01  (.a(prod_0),    .b(prod_1),   .result(sum_01));
    fp32_add u_accadd (.a(accum_out), .b(sum_01),   .result(new_accum));

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
