// fp8_e4m3_decoder.v — Decodes FP8 E4M3 code to IEEE 754 FP32
// FP8 E4M3: 1 sign + 4 exponent (bias=7) + 3 mantissa
// More complex than E8M0 — requires field extraction and exponent rebias
module fp8_e4m3_decoder (
    input  wire [7:0]  code,
    output wire [31:0] fp32_val
);
    wire        sign = code[7];
    wire [3:0]  exp4 = code[6:3];
    wire [2:0]  man3 = code[2:0];

    // Zero detection
    wire is_zero = (exp4 == 4'd0) && (man3 == 3'd0);
    // NaN detection (E4M3 has no Inf, only NaN at max exp + max mantissa)
    wire is_nan  = (exp4 == 4'd15) && (man3 == 3'd7);

    // Normal: value = (-1)^sign × 1.man3 × 2^(exp4 - 7)
    // FP32 exponent = exp4 - 7 + 127 = exp4 + 120
    // FP32 mantissa = man3 << 20 (3 bits → 23 bit field)

    // Subnormal: value = (-1)^sign × 0.man3 × 2^(-6)
    // FP32 exponent = -6 + 127 = 121, but need to normalize
    // For subnormals, we handle the 7 possible non-zero values via LUT

    reg [7:0]  res_exp;
    reg [22:0] res_man;

    always @(*) begin
        if (is_zero || is_nan) begin
            res_exp = is_nan ? 8'hFF : 8'd0;
            res_man = is_nan ? 23'h400000 : 23'd0;
        end else if (exp4 == 4'd0) begin
            // Subnormal E4M3: 0.man3 × 2^(-6)
            // man3=001 → 0.125 × 2^(-6) = 2^(-9)  → exp=118, man=0
            // man3=010 → 0.25  × 2^(-6) = 2^(-8)  → exp=119, man=0
            // man3=011 → 0.375 × 2^(-6) = 3×2^(-9) → exp=118, man=100...
            // man3=100 → 0.5   × 2^(-6) = 2^(-7)  → exp=120, man=0
            // man3=101 → 0.625 × 2^(-6) = 5×2^(-9) → exp=118, man=010...
            // man3=110 → 0.75  × 2^(-6) = 3×2^(-8) → exp=119, man=100...
            // man3=111 → 0.875 × 2^(-6) = 7×2^(-9) → exp=118, man=110...
            case (man3)
                3'd1: begin res_exp = 8'd118; res_man = 23'd0;         end // 2^(-9)
                3'd2: begin res_exp = 8'd119; res_man = 23'd0;         end // 2^(-8)
                3'd3: begin res_exp = 8'd118; res_man = 23'h400000;    end // 1.5×2^(-9)
                3'd4: begin res_exp = 8'd120; res_man = 23'd0;         end // 2^(-7)
                3'd5: begin res_exp = 8'd118; res_man = 23'h200000;    end // 1.25×2^(-9)
                3'd6: begin res_exp = 8'd119; res_man = 23'h400000;    end // 1.5×2^(-8)
                3'd7: begin res_exp = 8'd118; res_man = 23'h600000;    end // 1.75×2^(-9)
                default: begin res_exp = 8'd0; res_man = 23'd0;        end
            endcase
        end else begin
            // Normal: exponent rebias from E4M3 (bias=7) to FP32 (bias=127)
            res_exp = {4'd0, exp4} + 8'd120;  // exp4 + 120
            res_man = {man3, 20'd0};           // place 3 mantissa bits at top
        end
    end

    assign fp32_val = is_zero ? {sign, 31'd0} :
                      {sign, res_exp, res_man};
endmodule
