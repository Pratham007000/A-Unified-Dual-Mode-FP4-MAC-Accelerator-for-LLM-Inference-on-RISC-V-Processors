// soc_top.v — Minimal RISC-V SoC with MAC Accelerator
// Wires PicoRV32 CPU + RAM + your unified MAC wrapper together
//
// Memory Map:
//   0x00000000 - 0x0000FFFF : RAM (64 KB, program + data)
//   0x40000000 - 0x4000001F : MAC Accelerator (8 registers)

`timescale 1ns / 1ps

module soc_top (
    input  wire clk,
    input  wire rst
);

    // ---- PicoRV32 CPU signals ----
    wire        mem_valid;
    wire        mem_ready;
    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [ 3:0] mem_wstrb;
    wire [31:0] mem_rdata;

    // Directly tie off unused PCPI / IRQ / trace
    wire        pcpi_valid;
    wire [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2;

    // ---- PicoRV32 CPU ----
    picorv32 #(
        .ENABLE_MUL(1),
        .ENABLE_DIV(1),
        .STACKADDR(32'h0000FFFC)   // top of 64KB RAM
    ) cpu (
        .clk         (clk),
        .resetn      (~rst),       // PicoRV32 uses active-low reset
        .mem_valid   (mem_valid),
        .mem_ready   (mem_ready),
        .mem_addr    (mem_addr),
        .mem_wdata   (mem_wdata),
        .mem_wstrb   (mem_wstrb),
        .mem_rdata   (mem_rdata),
        // Tie off unused
        .mem_instr   (),
        .mem_la_read (),
        .mem_la_write(),
        .mem_la_addr (),
        .mem_la_wdata(),
        .mem_la_wstrb(),
        .pcpi_valid  (pcpi_valid),
        .pcpi_insn   (pcpi_insn),
        .pcpi_rs1    (pcpi_rs1),
        .pcpi_rs2    (pcpi_rs2),
        .pcpi_wr     (1'b0),
        .pcpi_rd     (32'd0),
        .pcpi_wait   (1'b0),
        .pcpi_ready  (1'b0),
        .irq         (32'd0),
        .eoi         (),
        .trap        (),
        .trace_valid (),
        .trace_data  ()
    );

    // ---- Address Decoder ----
    // RAM:  addr[31:16] == 0x0000  (0x00000000 - 0x0000FFFF)
    // MAC:  addr[31:8]  == 0x400000 (0x40000000 - 0x400000FF)
    wire sel_ram = (mem_addr[31:16] == 16'h0000);
    wire sel_mac = (mem_addr[31:8]  == 24'h400000);

    // ---- RAM (64 KB) ----
    reg [31:0] ram [0:16383];  // 16K words = 64KB
    reg [31:0] ram_rdata;
    reg        ram_ready;

    // Load firmware at startup
    initial begin
        $readmemh("firmware.hex", ram);
    end

    always @(posedge clk) begin
        ram_ready <= 1'b0;
        if (mem_valid && sel_ram && !ram_ready) begin
            ram_ready <= 1'b1;
            if (mem_wstrb == 4'b0000) begin
                // Read
                ram_rdata <= ram[mem_addr[15:2]];
            end else begin
                // Write (byte-lane aware)
                if (mem_wstrb[0]) ram[mem_addr[15:2]][ 7: 0] <= mem_wdata[ 7: 0];
                if (mem_wstrb[1]) ram[mem_addr[15:2]][15: 8] <= mem_wdata[15: 8];
                if (mem_wstrb[2]) ram[mem_addr[15:2]][23:16] <= mem_wdata[23:16];
                if (mem_wstrb[3]) ram[mem_addr[15:2]][31:24] <= mem_wdata[31:24];
            end
        end
    end

    // ---- MAC Accelerator ----
    wire [31:0] mac_rdata;
    wire        mac_cs    = mem_valid && sel_mac;
    wire        mac_we    = (mem_wstrb != 4'b0000);
    wire        mac_ready_w;

    mac_accelerator_wrapper u_mac (
        .clk   (clk),
        .rst   (rst),
        .cs    (mac_cs),
        .we    (mac_we),
        .addr  (mem_addr[4:0]),
        .wdata (mem_wdata),
        .rdata (mac_rdata)
    );

    // MAC always responds in 1 cycle
    reg mac_ready;
    always @(posedge clk) begin
        mac_ready <= 1'b0;
        if (mac_cs && !mac_ready)
            mac_ready <= 1'b1;
    end

    // ---- Bus Multiplexer ----
    assign mem_rdata = sel_mac ? mac_rdata : ram_rdata;
    assign mem_ready = sel_mac ? mac_ready : ram_ready;

endmodule
