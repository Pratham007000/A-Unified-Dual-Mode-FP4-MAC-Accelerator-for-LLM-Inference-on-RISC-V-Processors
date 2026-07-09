/**
 * soc_apu_top.sv
 * Minimal SoC: CV32E40P (RI5CY) + instruction/data RAM + MAC APU co-processor.
 *
 * Memory map:
 *   0x00000000 - 0x00003FFF : RAM (16 KB, firmware)
 *   0x10000000              : Print port (write -> $display)
 *   0x20000000              : Finish port (write -> $finish)
 *
 * Note: CV32E40P uses separate instruction and data memory ports (Harvard).
 *       Both are mapped to the same RAM for simplicity.
 */

`timescale 1ns / 1ps

module soc_apu_top
  import cv32e40p_apu_core_pkg::*;
(
    input logic clk_i,
    input logic rst_ni
);

    // ---- Instruction memory interface ----
    logic        instr_req;
    logic        instr_gnt;
    logic        instr_rvalid;
    logic [31:0] instr_addr;
    logic [31:0] instr_rdata;

    // ---- Data memory interface ----
    logic        data_req;
    logic        data_gnt;
    logic        data_rvalid;
    logic        data_we;
    logic [3:0]  data_be;
    logic [31:0] data_addr;
    logic [31:0] data_wdata;
    logic [31:0] data_rdata;

    // ---- APU interface ----
    logic                              apu_req;
    logic                              apu_gnt;
    logic [   APU_NARGS_CPU-1:0][31:0] apu_operands;
    logic [     APU_WOP_CPU-1:0]       apu_op;
    logic [APU_NDSFLAGS_CPU-1:0]       apu_flags;
    logic                              apu_rvalid;
    logic [                31:0]       apu_rdata;
    logic [APU_NUSFLAGS_CPU-1:0]       apu_rflags;
    logic                              apu_busy;

    // ---- CV32E40P Core (FPU=1 to enable APU port) ----
    cv32e40p_core #(
        .COREV_PULP      (0),
        .COREV_CLUSTER   (0),
        .FPU             (1),     // Enable APU interface
        .FPU_ADDMUL_LAT  (2),     // Multi-cycle latency (apu_lat=3, uses rvalid handshake)
        .FPU_OTHERS_LAT  (2),     // Multi-cycle latency
        .ZFINX           (1),    // Use integer registers for FP (no separate FP regfile)
        .NUM_MHPMCOUNTERS(0)
    ) core_i (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),

        .pulp_clock_en_i    (1'b0),
        .scan_cg_en_i       (1'b0),

        .boot_addr_i        (32'h0000_0000),
        .mtvec_addr_i       (32'h0000_0000),
        .dm_halt_addr_i     (32'h0000_0000),
        .hart_id_i          (32'h0000_0000),
        .dm_exception_addr_i(32'h0000_0000),

        // Instruction port
        .instr_req_o        (instr_req),
        .instr_gnt_i        (instr_gnt),
        .instr_rvalid_i     (instr_rvalid),
        .instr_addr_o       (instr_addr),
        .instr_rdata_i      (instr_rdata),

        // Data port
        .data_req_o         (data_req),
        .data_gnt_i         (data_gnt),
        .data_rvalid_i      (data_rvalid),
        .data_we_o          (data_we),
        .data_be_o          (data_be),
        .data_addr_o        (data_addr),
        .data_wdata_o       (data_wdata),
        .data_rdata_i       (data_rdata),

        // APU co-processor
        .apu_busy_o         (apu_busy),
        .apu_req_o          (apu_req),
        .apu_gnt_i          (apu_gnt),
        .apu_operands_o     (apu_operands),
        .apu_op_o           (apu_op),
        .apu_flags_o        (apu_flags),
        .apu_rvalid_i       (apu_rvalid),
        .apu_result_i       (apu_rdata),
        .apu_flags_i        (apu_rflags),

        // Interrupts
        .irq_i              (32'd0),
        .irq_ack_o          (),
        .irq_id_o           (),

        // Debug
        .debug_req_i        (1'b0),
        .debug_havereset_o  (),
        .debug_running_o    (),
        .debug_halted_o     (),

        // Control
        .fetch_enable_i     (1'b1),
        .core_sleep_o       ()
    );

    // ---- MAC APU Co-Processor ----
    mac_apu_coprocessor u_mac_apu (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .apu_req_i      (apu_req),
        .apu_gnt_o      (apu_gnt),
        .apu_operands_i (apu_operands),
        .apu_op_i       (apu_op),
        .apu_flags_i    (apu_flags),
        .apu_rvalid_o   (apu_rvalid),
        .apu_rdata_o    (apu_rdata),
        .apu_rflags_o   (apu_rflags)
    );

    // ---- Shared RAM (16 KB = 4096 words) ----
    logic [31:0] memory [0:4095];

    initial begin
        $readmemh("C:/Users/satya/OneDrive/Desktop/SRIP/rtl/apu/firmware_apu.hex", memory);
    end

    // ---- Instruction port (read-only, 1-cycle latency) ----
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            instr_rvalid <= 1'b0;
            instr_rdata  <= 32'd0;
        end else begin
            instr_rvalid <= instr_req & instr_gnt;
            if (instr_req)
                instr_rdata <= memory[instr_addr[13:2]];
        end
    end
    assign instr_gnt = instr_req;  // always grant immediately

    // ---- Data port (read/write, 1-cycle latency) ----
    wire is_print_addr  = (data_addr == 32'h1000_0000);
    wire is_finish_addr = (data_addr == 32'h2000_0000);

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            data_rvalid <= 1'b0;
            data_rdata  <= 32'd0;
        end else begin
            data_rvalid <= data_req & data_gnt;

            if (data_req && data_gnt) begin
                if (is_print_addr && data_we) begin
                    $display("[FIRMWARE] Output: 0x%08H", data_wdata);
                end
                else if (is_finish_addr && data_we) begin
                    $display("[FIRMWARE] Test complete. Finish signal received.");
                    $finish;
                end
                else if (data_we) begin
                    // RAM write
                    if (data_be[0]) memory[data_addr[13:2]][ 7: 0] <= data_wdata[ 7: 0];
                    if (data_be[1]) memory[data_addr[13:2]][15: 8] <= data_wdata[15: 8];
                    if (data_be[2]) memory[data_addr[13:2]][23:16] <= data_wdata[23:16];
                    if (data_be[3]) memory[data_addr[13:2]][31:24] <= data_wdata[31:24];
                end
                else begin
                    // RAM read
                    data_rdata <= memory[data_addr[13:2]];
                end
            end
        end
    end
    assign data_gnt = data_req;  // always grant immediately

endmodule
