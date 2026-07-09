// unified_mac_top.v
// Dual-mode MAC unit: supports MXFP4 and NVFP4 via FORMAT_MODE input
// FORMAT_MODE=0: MXFP4 (E8M0 scale, 32-element blocks)
// FORMAT_MODE=1: NVFP4 (FP8 E4M3 scale x tensor_scale, 16-element blocks)
`timescale 1ns / 1ps

module unified_mac_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,

    // Format control: 0 = MXFP4, 1 = NVFP4
    input  wire        FORMAT_MODE,

    // Data inputs (shared)
    input  wire [7:0]  packed_byte,    // 2 packed E2M1 elements
    input  wire [31:0] b_elem_0,       // FP32 B element, even index
    input  wire [31:0] b_elem_1,       // FP32 B element, odd index

    // Scale inputs
    input  wire [7:0]  scale_code_a,   // E8M0 scale    (used when MODE=0)
    input  wire [7:0]  scale_code_b,   // FP8 E4M3 scale (used when MODE=1)
    input  wire [31:0] tensor_scale,   // FP32 tensor scale (used when MODE=1)

    // Outputs
    output reg  [31:0] accum_out,
    output reg         valid_out,
    output reg         mode_out        // echoes which format produced the result
);

    // ---- Stage 1: Decode ----
    // Nibble extract + E2M1 LUT (shared by both formats)
    wire [3:0]  lo_nib, hi_nib;
    wire [31:0] decoded_0, decoded_1;

    nibble_extractor u_nib (
        .packed_byte(packed_byte),
        .lo_nibble(lo_nib),
        .hi_nibble(hi_nib)
    );

    e2m1_decoder u_dec0 (.code(lo_nib), .fp32_val(decoded_0));
    e2m1_decoder u_dec1 (.code(hi_nib), .fp32_val(decoded_1));

    // Both scale decoders run in parallel
    wire [31:0] mxfp4_scale;       // E8M0 path (zero logic - wire routing)
    wire [31:0] fp8_block_scale;   // FP8 E4M3 path (field extraction)

    e8m0_decoder     u_e8m0 (.code(scale_code_a), .fp32_scale(mxfp4_scale));
    fp8_e4m3_decoder u_fp8  (.code(scale_code_b), .fp32_val(fp8_block_scale));

    // NVFP4 needs: effective_scale = block_scale x tensor_scale
    wire [31:0] nvfp4_eff_scale;
    fp32_mul u_eff (.a(fp8_block_scale), .b(tensor_scale), .result(nvfp4_eff_scale));

    // MUX: select scale based on FORMAT_MODE
    wire [31:0] eff_scale = FORMAT_MODE ? nvfp4_eff_scale : mxfp4_scale;

    // ---- Stage 2: Scale multiply (decoded x scale) ----
    wire [31:0] scaled_0, scaled_1;
    fp32_mul u_smul0 (.a(decoded_0), .b(eff_scale), .result(scaled_0));
    fp32_mul u_smul1 (.a(decoded_1), .b(eff_scale), .result(scaled_1));

    // ---- Stage 3: Dot multiply (scaled x B_element) ----
    wire [31:0] prod_0, prod_1;
    fp32_mul u_dmul0 (.a(scaled_0), .b(b_elem_0), .result(prod_0));
    fp32_mul u_dmul1 (.a(scaled_1), .b(b_elem_1), .result(prod_1));

    // ---- Stage 4: Accumulate ----
    wire [31:0] sum_01;
    wire [31:0] new_accum;
    fp32_add u_add01  (.a(prod_0),    .b(prod_1),  .result(sum_01));
    fp32_add u_accadd (.a(accum_out), .b(sum_01),  .result(new_accum));

    // ---- Register update ----
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accum_out <= 32'd0;
            valid_out <= 1'b0;
            mode_out  <= 1'b0;
        end else if (valid_in) begin
            accum_out <= new_accum;
            valid_out <= 1'b1;
            mode_out  <= FORMAT_MODE;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
