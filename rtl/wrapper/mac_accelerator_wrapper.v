// mac_accelerator_wrapper.v
// Memory-mapped register wrapper for unified MAC unit
// Provides RISC-V bus interface for CPU-controlled MAC operations
//
// Register Map (32-bit aligned):
//   0x00  CONTROL      [W/R]  bit[0]=start, bit[1]=done(RO), bit[2]=FORMAT_MODE, bit[3]=rst_accum
//   0x04  PACKED_BYTE  [W]    bits[7:0] = 2 packed E2M1 elements
//   0x08  SCALE_A      [W]    bits[7:0] = E8M0 scale code (MXFP4 mode)
//   0x0C  SCALE_B      [W]    bits[7:0] = FP8 E4M3 scale code (NVFP4 mode)
//   0x10  TENSOR_SCALE [W]    bits[31:0] = FP32 tensor scale (NVFP4 mode)
//   0x14  B_ELEM_0     [W]    bits[31:0] = FP32 B vector element 0
//   0x18  B_ELEM_1     [W]    bits[31:0] = FP32 B vector element 1
//   0x1C  RESULT       [R]    bits[31:0] = FP32 accumulator output
//
// Usage from RISC-V C code:
//   #define MAC_BASE 0x40000000
//   *(volatile uint32_t*)(MAC_BASE + 0x00) = 0x04;      // FORMAT_MODE=1 (NVFP4)
//   *(volatile uint32_t*)(MAC_BASE + 0x04) = packed;     // write data
//   *(volatile uint32_t*)(MAC_BASE + 0x08) = scale_a;    // write E8M0 scale
//   *(volatile uint32_t*)(MAC_BASE + 0x0C) = scale_b;    // write FP8 scale
//   *(volatile uint32_t*)(MAC_BASE + 0x10) = tensor_sc;  // write tensor scale
//   *(volatile uint32_t*)(MAC_BASE + 0x14) = b0;         // write B elem 0
//   *(volatile uint32_t*)(MAC_BASE + 0x18) = b1;         // write B elem 1
//   *(volatile uint32_t*)(MAC_BASE + 0x00) = 0x05;       // start + NVFP4 mode
//   while (!(*(volatile uint32_t*)(MAC_BASE + 0x00) & 0x02)); // poll done
//   uint32_t result = *(volatile uint32_t*)(MAC_BASE + 0x1C); // read result

`timescale 1ns / 1ps

module mac_accelerator_wrapper (
    input  wire        clk,
    input  wire        rst,

    // ---- Simple bus interface (compatible with AXI-Lite / Wishbone) ----
    input  wire        cs,            // chip select (active high)
    input  wire        we,            // write enable
    input  wire [4:0]  addr,          // byte address [4:2] selects register
    input  wire [31:0] wdata,         // write data from CPU
    output reg  [31:0] rdata          // read data to CPU
);

    // ---- Internal registers ----
    reg        reg_start;
    reg        reg_format_mode;
    reg        reg_rst_accum;
    reg [7:0]  reg_packed_byte;
    reg [7:0]  reg_scale_a;
    reg [7:0]  reg_scale_b;
    reg [31:0] reg_tensor_scale;
    reg [31:0] reg_b0;
    reg [31:0] reg_b1;

    // ---- MAC unit signals ----
    wire [31:0] mac_result;
    wire        mac_valid;
    wire        mac_mode_out;

    // ---- Control state machine ----
    reg        mac_valid_in;
    reg        done_flag;
    reg [2:0]  state;

    localparam S_IDLE    = 3'd0;
    localparam S_COMPUTE = 3'd1;
    localparam S_DONE    = 3'd2;

    // ---- Instantiate unified MAC ----
    unified_mac_top u_mac (
        .clk          (clk),
        .rst          (reg_rst_accum),
        .valid_in     (mac_valid_in),
        .FORMAT_MODE  (reg_format_mode),
        .packed_byte  (reg_packed_byte),
        .b_elem_0     (reg_b0),
        .b_elem_1     (reg_b1),
        .scale_code_a (reg_scale_a),
        .scale_code_b (reg_scale_b),
        .tensor_scale (reg_tensor_scale),
        .accum_out    (mac_result),
        .valid_out    (mac_valid),
        .mode_out     (mac_mode_out)
    );

    // ---- Write logic (CPU writes to registers) ----
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            reg_start       <= 1'b0;
            reg_format_mode <= 1'b0;
            reg_rst_accum   <= 1'b1;
            reg_packed_byte <= 8'd0;
            reg_scale_a     <= 8'd0;
            reg_scale_b     <= 8'd0;
            reg_tensor_scale<= 32'd0;
            reg_b0          <= 32'd0;
            reg_b1          <= 32'd0;
        end else if (cs && we) begin
            case (addr[4:2])
                3'd0: begin  // 0x00 CONTROL
                    reg_start       <= wdata[0];
                    reg_format_mode <= wdata[2];
                    reg_rst_accum   <= wdata[3];
                end
                3'd1: reg_packed_byte  <= wdata[7:0];    // 0x04
                3'd2: reg_scale_a      <= wdata[7:0];    // 0x08
                3'd3: reg_scale_b      <= wdata[7:0];    // 0x0C
                3'd4: reg_tensor_scale <= wdata;          // 0x10
                3'd5: reg_b0           <= wdata;          // 0x14
                3'd6: reg_b1           <= wdata;          // 0x18
                default: ;  // 0x1C is read-only
            endcase
        end else begin
            // Auto-clear start after one cycle
            reg_start <= 1'b0;
        end
    end

    // ---- Read logic (CPU reads from registers) ----
    always @(*) begin
        rdata = 32'd0;
        if (cs && !we) begin
            case (addr[4:2])
                3'd0: rdata = {28'd0, reg_rst_accum, reg_format_mode, done_flag, reg_start};
                3'd1: rdata = {24'd0, reg_packed_byte};
                3'd2: rdata = {24'd0, reg_scale_a};
                3'd3: rdata = {24'd0, reg_scale_b};
                3'd4: rdata = reg_tensor_scale;
                3'd5: rdata = reg_b0;
                3'd6: rdata = reg_b1;
                3'd7: rdata = mac_result;  // 0x1C RESULT
                default: rdata = 32'd0;
            endcase
        end
    end

    // ---- State machine: start → compute → done ----
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state       <= S_IDLE;
            mac_valid_in <= 1'b0;
            done_flag    <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    mac_valid_in <= 1'b0;
                    if (reg_start) begin
                        mac_valid_in <= 1'b1;
                        done_flag    <= 1'b0;
                        state        <= S_COMPUTE;
                    end
                end
                S_COMPUTE: begin
                    mac_valid_in <= 1'b0;
                    if (mac_valid) begin
                        done_flag <= 1'b1;
                        state     <= S_DONE;
                    end
                end
                S_DONE: begin
                    // Stay done until next start
                    if (reg_start) begin
                        mac_valid_in <= 1'b1;
                        done_flag    <= 1'b0;
                        state        <= S_COMPUTE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
