// e2m1_decoder.v — Decodes a 4-bit E2M1 code to IEEE 754 FP32
// Used by BOTH MXFP4 and NVFP4 formats
// E2M1: 1 sign + 2 exponent + 1 mantissa = 16 possible values
module e2m1_decoder (
    input  wire [3:0]  code,
    output reg  [31:0] fp32_val
);
    always @(*) begin
        case (code)
            4'h0: fp32_val = 32'h00000000; // +0.0
            4'h1: fp32_val = 32'h3F000000; // +0.5
            4'h2: fp32_val = 32'h3F800000; // +1.0
            4'h3: fp32_val = 32'h3FC00000; // +1.5
            4'h4: fp32_val = 32'h40000000; // +2.0
            4'h5: fp32_val = 32'h40400000; // +3.0
            4'h6: fp32_val = 32'h40800000; // +4.0
            4'h7: fp32_val = 32'h40C00000; // +6.0
            4'h8: fp32_val = 32'h80000000; // -0.0
            4'h9: fp32_val = 32'hBF000000; // -0.5
            4'hA: fp32_val = 32'hBF800000; // -1.0
            4'hB: fp32_val = 32'hBFC00000; // -1.5
            4'hC: fp32_val = 32'hC0000000; // -2.0
            4'hD: fp32_val = 32'hC0400000; // -3.0
            4'hE: fp32_val = 32'hC0800000; // -4.0
            4'hF: fp32_val = 32'hC0C00000; // -6.0
        endcase
    end
endmodule
