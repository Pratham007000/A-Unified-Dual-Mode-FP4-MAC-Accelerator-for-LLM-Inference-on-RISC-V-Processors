// fp32_add.v - IEEE 754 single-precision floating-point adder
// Computes: result = a + b
// Handles: zeros, normals, sign differences. Simplified subnormal handling.
module fp32_add (
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

    // Determine which operand is larger (by magnitude)
    wire a_gt_b = (a_exp > b_exp) ||
                  (a_exp == b_exp && a_man >= b_man);

    // Swap so 'big' has larger exponent
    wire        big_sign = a_gt_b ? a_sign : b_sign;
    wire [7:0]  big_exp  = a_gt_b ? a_exp  : b_exp;
    wire [23:0] big_full = a_gt_b ? {1'b1, a_man} : {1'b1, b_man};

    wire        sml_sign = a_gt_b ? b_sign : a_sign;
    wire [7:0]  sml_exp  = a_gt_b ? b_exp  : a_exp;
    wire [23:0] sml_full = a_gt_b ? {1'b1, b_man} : {1'b1, a_man};

    // Exponent difference and alignment shift
    wire [7:0] exp_diff = big_exp - sml_exp;
    wire [24:0] sml_shifted = (exp_diff < 8'd25) ?
                              ({1'b0, sml_full} >> exp_diff) : 25'd0;

    // Effective operation: add or subtract based on signs
    wire same_sign = (big_sign == sml_sign);
    wire [25:0] sum_raw = same_sign ?
                          ({2'b0, big_full} + {1'b0, sml_shifted}) :
                          ({2'b0, big_full} - {1'b0, sml_shifted});

    // Result sign
    wire res_sign = big_sign;

    // Normalize: find leading 1
    // Check for overflow (bit 25 or 24 set)
    reg [7:0]  norm_exp;
    reg [22:0] norm_man;

    // Simple normalization (handles common cases)
    always @(*) begin
        if (sum_raw[25]) begin
            norm_exp = big_exp + 8'd2;
            norm_man = sum_raw[24:2];
        end else if (sum_raw[24]) begin
            // Overflow: leading 1 at bit 24, shift right
            norm_exp = big_exp + 8'd1;
            norm_man = sum_raw[23:1];
        end else if (sum_raw[23]) begin
            // Normal: leading 1 at bit 23
            norm_exp = big_exp;
            norm_man = sum_raw[22:0];
        end else if (sum_raw[22]) begin
            norm_exp = big_exp - 8'd1;
            norm_man = {sum_raw[21:0], 1'b0};
        end else if (sum_raw[21]) begin
            norm_exp = big_exp - 8'd2;
            norm_man = {sum_raw[20:0], 2'b0};
        end else begin
            norm_exp = 8'd0;
            norm_man = 23'd0;
        end
    end

    // Handle zero inputs
    assign result = (a_zero && b_zero) ? 32'd0 :
                    a_zero             ? b :
                    b_zero             ? a :
                    (sum_raw == 26'd0) ? 32'd0 :
                    {res_sign, norm_exp, norm_man};
endmodule
