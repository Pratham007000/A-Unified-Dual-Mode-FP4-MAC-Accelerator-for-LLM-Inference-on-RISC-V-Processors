/**
 * mac_pcpi_coprocessor.v
 * PicoRV32 PCPI adapter for unified_mac_top.
 *
 * Decodes 2 custom-0 instructions (opcode 0x0B):
 *   funct3=0  QMAC_CFG   rs1={scale_b[7:0],scale_a[7:0],packed_byte[7:0],mode}
 *                         rs2=tensor_scale
 *   funct3=1  QMAC_EXEC  rs1=b_elem_0  rs2=b_elem_1  ->  rd=accum_out
 */
`timescale 1ns / 1ps

module mac_pcpi_coprocessor (
    input  wire        clk,
    input  wire        resetn,

    // PCPI interface (directly from PicoRV32)
    input  wire        pcpi_valid,
    input  wire [31:0] pcpi_insn,
    input  wire [31:0] pcpi_rs1,
    input  wire [31:0] pcpi_rs2,
    output reg         pcpi_wr,
    output reg  [31:0] pcpi_rd,
    output reg         pcpi_wait,
    output reg         pcpi_ready
);

    // ---- Instruction decode ----
    localparam OPCODE_CUSTOM0 = 7'b0001011; // 0x0B, RISC-V reserved custom-0

    wire        is_custom0 = (pcpi_insn[6:0] == OPCODE_CUSTOM0);
    wire [2:0]  funct3     = pcpi_insn[14:12];
    wire        is_cfg     = pcpi_valid && is_custom0 && (funct3 == 3'b000);
    wire        is_exec    = pcpi_valid && is_custom0 && (funct3 == 3'b001);

    // ---- Latched config (from QMAC_CFG) ----
    // rs1 bit layout: [0]=mode, [8:1]=packed_byte, [16:9]=scale_a, [24:17]=scale_b
    reg        mode_r;
    reg [7:0]  packed_byte_r, scale_a_r, scale_b_r;
    reg [31:0] tensor_scale_r;

    // ---- B element registers (from QMAC_EXEC) ----
    reg [31:0] b0_r, b1_r;

    // ---- Unified MAC instance ----
    reg         mac_valid_in;
    reg         mac_rst;             // pulsed during QMAC_CFG to reset accumulator
    wire        mac_valid_out;
    wire [31:0] mac_accum_out;

    wire mac_rst_combined = (~resetn) | mac_rst;  // reset on global reset OR CFG

    unified_mac_top u_mac (
        .clk         (clk),
        .rst         (mac_rst_combined), // resets on global reset OR QMAC_CFG
        .valid_in    (mac_valid_in),
        .FORMAT_MODE (mode_r),
        .packed_byte (packed_byte_r),
        .b_elem_0    (b0_r),
        .b_elem_1    (b1_r),
        .scale_code_a(scale_a_r),
        .scale_code_b(scale_b_r),
        .tensor_scale(tensor_scale_r),
        .accum_out   (mac_accum_out),
        .valid_out   (mac_valid_out),
        .mode_out    ()                 // unused
    );

    // ---- FSM ----
    localparam S_IDLE      = 2'd0;
    localparam S_EXEC_WAIT = 2'd1;
    localparam S_CFG_RESET = 2'd2;  // 1-cycle reset pulse for accumulator
    reg [1:0] state;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            state          <= S_IDLE;
            pcpi_wr        <= 1'b0;
            pcpi_rd        <= 32'd0;
            pcpi_wait      <= 1'b0;
            pcpi_ready     <= 1'b0;
            mac_valid_in   <= 1'b0;
            mac_rst        <= 1'b0;
            mode_r         <= 1'b0;
            packed_byte_r  <= 8'd0;
            scale_a_r      <= 8'd0;
            scale_b_r      <= 8'd0;
            tensor_scale_r <= 32'd0;
            b0_r           <= 32'd0;
            b1_r           <= 32'd0;
        end else begin
            // Default: deassert handshake signals each cycle
            pcpi_ready   <= 1'b0;
            pcpi_wr      <= 1'b0;
            mac_valid_in <= 1'b0;
            mac_rst      <= 1'b0;

            case (state)
                S_IDLE: begin
                    pcpi_wait <= 1'b0;

                    if (is_cfg) begin
                        // ---- QMAC_CFG: latch config + reset accumulator ----
                        mode_r         <= pcpi_rs1[0];
                        packed_byte_r  <= pcpi_rs1[8:1];
                        scale_a_r      <= pcpi_rs1[16:9];
                        scale_b_r      <= pcpi_rs1[24:17];
                        tensor_scale_r <= pcpi_rs2;
                        mac_rst        <= 1'b1;   // pulse reset to clear accumulator
                        pcpi_wait      <= 1'b1;
                        state          <= S_CFG_RESET;
                    end
                    else if (is_exec) begin
                        // ---- QMAC_EXEC: trigger MAC computation ----
                        b0_r         <= pcpi_rs1;
                        b1_r         <= pcpi_rs2;
                        mac_valid_in <= 1'b1;     // pulse valid to MAC
                        pcpi_wait    <= 1'b1;     // tell CPU to wait
                        state        <= S_EXEC_WAIT;
                    end
                    // else: not our instruction, pcpi_wait=0 + pcpi_ready=0
                    //        tells PicoRV32 we don't handle it
                end

                S_EXEC_WAIT: begin
                    pcpi_wait <= 1'b1;            // keep CPU stalled

                    if (mac_valid_out) begin
                        // MAC finished - return result to CPU
                        pcpi_rd    <= mac_accum_out;
                        pcpi_wr    <= 1'b1;       // write result to rd register
                        pcpi_ready <= 1'b1;       // signal completion
                        pcpi_wait  <= 1'b0;       // release CPU
                        state      <= S_IDLE;
                    end
                end

                S_CFG_RESET: begin
                    // Accumulator reset pulse done, release CPU
                    mac_rst    <= 1'b0;
                    pcpi_ready <= 1'b1;
                    pcpi_wr    <= 1'b0;
                    pcpi_wait  <= 1'b0;
                    state      <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
