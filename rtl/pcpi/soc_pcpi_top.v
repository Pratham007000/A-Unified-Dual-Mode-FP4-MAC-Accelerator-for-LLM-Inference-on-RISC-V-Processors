/**
 * soc_pcpi_top.v
 * Minimal SoC: PicoRV32 (RV32IMC) + synchronous RAM + MAC PCPI coprocessor.
 *
 * Memory map:
 *   0x00000000 - 0x00001FFF : RAM (8 KB, firmware)
 *   0x10000000              : Print port (write → $display in simulation)
 *   0x20000000              : Finish port (write → $finish)
 */
`timescale 1ns / 1ps

module soc_pcpi_top (
    input wire clk,
    input wire resetn
);

    // ---- CPU memory interface ----
    wire        mem_valid, mem_instr, mem_ready;
    wire [31:0] mem_addr, mem_wdata, mem_rdata;
    wire [3:0]  mem_wstrb;

    // ---- PCPI interface ----
    wire        pcpi_valid;
    wire [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2;
    wire        pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_rd;

    // ---- PicoRV32 CPU ----
    picorv32 #(
        .ENABLE_PCPI    (1),
        .ENABLE_MUL     (1),
        .ENABLE_DIV     (1),
        .PROGADDR_RESET (32'h0000_0000),
        .STACKADDR      (32'h0000_2000)    // top of 8KB RAM
    ) cpu (
        .clk         (clk),
        .resetn      (resetn),

        // Memory interface
        .mem_valid   (mem_valid),
        .mem_instr   (mem_instr),
        .mem_ready   (mem_ready),
        .mem_addr    (mem_addr),
        .mem_wdata   (mem_wdata),
        .mem_wstrb   (mem_wstrb),
        .mem_rdata   (mem_rdata),

        // Look-ahead (unused, left open)
        .mem_la_read (),
        .mem_la_write(),
        .mem_la_addr (),
        .mem_la_wdata(),
        .mem_la_wstrb(),

        // PCPI co-processor interface
        .pcpi_valid  (pcpi_valid),
        .pcpi_insn   (pcpi_insn),
        .pcpi_rs1    (pcpi_rs1),
        .pcpi_rs2    (pcpi_rs2),
        .pcpi_wr     (pcpi_wr),
        .pcpi_rd     (pcpi_rd),
        .pcpi_wait   (pcpi_wait),
        .pcpi_ready  (pcpi_ready),

        // Interrupts (none)
        .irq         (32'd0),
        .eoi         (),

        // Trap & trace (unused)
        .trap        (),
        .trace_valid (),
        .trace_data  ()
    );

    // ---- MAC PCPI Co-Processor ----
    mac_pcpi_coprocessor u_mac_pcpi (
        .clk        (clk),
        .resetn     (resetn),
        .pcpi_valid (pcpi_valid),
        .pcpi_insn  (pcpi_insn),
        .pcpi_rs1   (pcpi_rs1),
        .pcpi_rs2   (pcpi_rs2),
        .pcpi_wr    (pcpi_wr),
        .pcpi_rd    (pcpi_rd),
        .pcpi_wait  (pcpi_wait),
        .pcpi_ready (pcpi_ready)
    );

    // ---- RAM (8 KB = 2048 words) ----
    reg [31:0] memory [0:2047];

    // Load firmware at startup
    initial begin
        $readmemh("C:/Users/satya/OneDrive/Desktop/SRIP/rtl/pcpi/firmware_pcpi.hex", memory);
    end

    // Special addresses
    wire is_ram_addr    = (mem_addr[31:13] == 19'd0);         // 0x00000000-0x00001FFF
    wire is_print_addr  = (mem_addr == 32'h1000_0000);
    wire is_finish_addr = (mem_addr == 32'h2000_0000);

    reg        ram_ready;
    reg [31:0] ram_rdata;

    always @(posedge clk) begin
        ram_ready <= 1'b0;

        if (mem_valid && !ram_ready) begin
            if (is_print_addr && |mem_wstrb) begin
                // Print port — firmware writes values here for debug
                $display("[FIRMWARE] Output: 0x%08H", mem_wdata);
                ram_ready <= 1'b1;
            end
            else if (is_finish_addr && |mem_wstrb) begin
                // Finish port — firmware signals it's done
                $display("[FIRMWARE] Test complete. Finish signal received.");
                $finish;
            end
            else if (|mem_wstrb) begin
                // RAM write (byte-lane aware)
                if (mem_wstrb[0]) memory[mem_addr[12:2]][ 7: 0] <= mem_wdata[ 7: 0];
                if (mem_wstrb[1]) memory[mem_addr[12:2]][15: 8] <= mem_wdata[15: 8];
                if (mem_wstrb[2]) memory[mem_addr[12:2]][23:16] <= mem_wdata[23:16];
                if (mem_wstrb[3]) memory[mem_addr[12:2]][31:24] <= mem_wdata[31:24];
                ram_ready <= 1'b1;
            end
            else begin
                // RAM read
                ram_rdata <= memory[mem_addr[12:2]];
                ram_ready <= 1'b1;
            end
        end
    end

    assign mem_ready = ram_ready;
    assign mem_rdata = ram_rdata;

endmodule
