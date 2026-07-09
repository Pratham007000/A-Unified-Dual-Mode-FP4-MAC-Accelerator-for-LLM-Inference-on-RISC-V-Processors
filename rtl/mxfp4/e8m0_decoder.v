// e8m0_decoder.v — Decodes E8M0 scale code to IEEE 754 FP32
// E8M0: pure exponent format, scale = 2^(code - 127)
// In hardware: code becomes the FP32 exponent field directly (zero cost!)
module e8m0_decoder (
    input  wire [7:0]  code,
    output wire [31:0] fp32_scale
);
    // code=255 → NaN
    // code=0   → 2^(-127) (smallest positive)
    // code=127 → 2^0 = 1.0
    // code=254 → 2^127
    wire is_nan = (code == 8'hFF);

    // FP32 format: {sign=0, exponent=code, mantissa=0}
    // This works because FP32 value = 2^(exponent - 127) when mantissa=0
    assign fp32_scale = is_nan ? 32'h7FC00000 :   // quiet NaN
                                {1'b0, code, 23'b0};
endmodule
