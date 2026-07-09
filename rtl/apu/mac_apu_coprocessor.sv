/**
 * mac_apu_coprocessor.sv
 * CV32E40P (RI5CY) APU adapter for unified_mac_top.
 *
 * Replaces the FPU on the APU interface.
 * Decodes 2 operations via apu_op:
 *   apu_op=0 (fadd.s) ? QMAC_CFG:  latches config, resets accumulator
 *   apu_op=2 (fmul.s) ? QMAC_EXEC: triggers MAC, returns accum_out
 *
 * CRITICAL: apu_gnt_o must be COMBINATIONAL (same cycle as req).
 */

`timescale 1ns / 1ps

module mac_apu_coprocessor
  import cv32e40p_apu_core_pkg::*;
(
    input  logic                              clk_i,
    input  logic                              rst_ni,

    // APU interface (directly from cv32e40p_core)
    input  logic                              apu_req_i,
    output logic                              apu_gnt_o,
    input  logic [   APU_NARGS_CPU-1:0][31:0] apu_operands_i,  // 3 x 32-bit
    input  logic [     APU_WOP_CPU-1:0]       apu_op_i,        // 6-bit opcode
    input  logic [APU_NDSFLAGS_CPU-1:0]       apu_flags_i,     // 15-bit flags (unused)
    output logic                              apu_rvalid_o,
    output logic [                31:0]       apu_rdata_o,
    output logic [APU_NUSFLAGS_CPU-1:0]       apu_rflags_o     // 5-bit flags (unused)
);

    // ---- Operation decode ----
    // CV32E40P decoder: apu_op = {fpu_vec_op, fpu_op_mod, fpu_op[3:0]}
    // fpnew enum: FMADD=0, FNMSUB=1, ADD=2, MUL=3
    // fadd.s -> op = {0, 0, ADD} = {0, 0, 4'd2} = 6'd2
    // fmul.s -> op = {0, 0, MUL} = {0, 0, 4'd3} = 6'd3
    localparam OP_QMAC_CFG  = 6'd2;   // fadd.s (ADD=2)
    localparam OP_QMAC_EXEC = 6'd3;   // fmul.s (MUL=3)

    // ---- Latched config (from QMAC_CFG) ----
    logic        mode_r;
    logic [7:0]  packed_byte_r, scale_a_r, scale_b_r;
    logic [31:0] tensor_scale_r;

    // ---- B element registers (from QMAC_EXEC) ----
    logic [31:0] b0_r, b1_r;

    // ---- Unified MAC instance ----
    logic        mac_valid_in;
    logic        mac_rst;
    wire         mac_valid_out;
    wire  [31:0] mac_accum_out;

    wire mac_rst_combined = (~rst_ni) | mac_rst;

    unified_mac_top u_mac (
        .clk         (clk_i),
        .rst         (mac_rst_combined),
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
        .mode_out    ()
    );

    // ---- COMBINATIONAL grant: always accept immediately ----
    assign apu_gnt_o    = apu_req_i;
    assign apu_rflags_o = '0;

    // ---- FSM ----
    typedef enum logic [1:0] {
        S_IDLE,
        S_CFG_RESET,
        S_EXEC_WAIT
    } state_t;

    state_t state;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state          <= S_IDLE;
            apu_rvalid_o   <= 1'b0;
            apu_rdata_o    <= 32'd0;
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
            // Defaults each cycle
            apu_rvalid_o <= 1'b0;
            mac_valid_in <= 1'b0;
            mac_rst      <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (apu_req_i) begin
                        $display("[APU DEBUG] t=%0t req! op=%0d op0=0x%08H op1=0x%08H op2=0x%08H",
                                 $time, apu_op_i, apu_operands_i[0], apu_operands_i[1], apu_operands_i[2]);
                        if (apu_op_i == OP_QMAC_CFG) begin
                            // ---- QMAC_CFG: latch config + reset accumulator ----
                            // fadd.s operand mapping: [0]=rs1, [1]=rs1(dup), [2]=rs2
                            mode_r         <= apu_operands_i[0][0];
                            packed_byte_r  <= apu_operands_i[0][8:1];
                            scale_a_r      <= apu_operands_i[0][16:9];
                            scale_b_r      <= apu_operands_i[0][24:17];
                            tensor_scale_r <= apu_operands_i[2];  // rs2 at index 2
                            mac_rst        <= 1'b1;
                            state          <= S_CFG_RESET;
                        end
                        else if (apu_op_i == OP_QMAC_EXEC) begin
                            // ---- QMAC_EXEC: trigger MAC computation ----
                            // fmul.s operand mapping: [0]=rs1, [1]=rs2
                            b0_r         <= apu_operands_i[0];
                            b1_r         <= apu_operands_i[1];
                            mac_valid_in <= 1'b1;
                            state        <= S_EXEC_WAIT;
                        end
                        else begin
                            // Unknown op - return 0 immediately
                            apu_rvalid_o <= 1'b1;
                            apu_rdata_o  <= 32'd0;
                        end
                    end
                end

                S_CFG_RESET: begin
                    // Accumulator reset done, return dummy result
                    mac_rst      <= 1'b0;
                    apu_rvalid_o <= 1'b1;
                    apu_rdata_o  <= 32'd0;
                    state        <= S_IDLE;
                end

                S_EXEC_WAIT: begin
                    if (mac_valid_out) begin
                        // MAC done - return result
                        $display("[APU DEBUG] t=%0t MAC done! result=0x%08H", $time, mac_accum_out);
                        apu_rdata_o  <= mac_accum_out;
                        apu_rvalid_o <= 1'b1;
                        state        <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
