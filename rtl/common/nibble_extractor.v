// nibble_extractor.v — Extracts two 4-bit E2M1 codes from one packed byte
module nibble_extractor (
    input  wire [7:0] packed_byte,
    output wire [3:0] lo_nibble,   // even-index element
    output wire [3:0] hi_nibble    // odd-index element
);
    assign lo_nibble = packed_byte[3:0];
    assign hi_nibble = packed_byte[7:4];
endmodule
