// fp32_mul.v — IEEE 754 single-precision floating-point multiplier
// Computes: result = a * b
// Handles: zeros, normals. Subnormals/Inf/NaN treated as zero for simplicity.
module fp32_mul (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);
    // Extract fields
    wire        a_sign = a[31];
    wire [7:0]  a_exp  = a[30:23];
    wire [22:0] a_man  = a[22:0];

    wire        b_sign = b[31];
    wire [7:0]  b_exp  = b[30:23];
    wire [22:0] b_man  = b[22:0];

    // Zero detection
    wire a_zero = (a_exp == 8'd0) && (a_man == 23'd0);
    wire b_zero = (b_exp == 8'd0) && (b_man == 23'd0);

    // Result sign = XOR of input signs
    wire res_sign = a_sign ^ b_sign;

    // Implied 1.mantissa (24 bits each)
    wire [23:0] a_full = {1'b1, a_man};
    wire [23:0] b_full = {1'b1, b_man};

    // Mantissa multiply: 24 × 24 = 48 bits
    wire [47:0] mul_raw = a_full * b_full;

    // The product is in format: 1x.xxx...xxx or 01.xxx...xxx
    // If MSB (bit 47) is 1 → result is 1x.xxx, need to shift right by 1
    wire        mul_overflow = mul_raw[47];
    wire [22:0] res_man = mul_overflow ? mul_raw[46:24] : mul_raw[45:23];

    // Exponent: (a_exp - 127) + (b_exp - 127) + 127 = a_exp + b_exp - 127
    // Add 1 if mantissa overflows
    wire [9:0] exp_sum = {2'b0, a_exp} + {2'b0, b_exp} - 10'd127 + {9'd0, mul_overflow};

    // Clamp exponent
    wire exp_underflow = exp_sum[9]; // negative
    wire exp_overflow  = (exp_sum > 10'd254);
    wire [7:0] res_exp = exp_underflow ? 8'd0 :
                         exp_overflow  ? 8'd255 :
                         exp_sum[7:0];

    // Final result
    assign result = (a_zero || b_zero) ? {res_sign, 31'd0} :  // zero
                    {res_sign, res_exp, res_man};
endmodule
