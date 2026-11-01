// mgpu_host_regs: host register map of ISA 13.1 (docs/uarch/top.md 2).
//
// CSR port: one transaction per cycle; reads return data one cycle later. Byte offsets,
// bits [1:0] ignored; unmapped offsets read 0 and ignore writes.
//
// ENTRY, GRID_DIM, BLOCK_DIM and ARG0-7 are writable only while no launch is running
// ([HOST-1]). A write of GO with bit 0 set while idle clears STATUS and ERR_* ([HOST-3]),
// sets `running` and raises `launch` for one cycle (the dispatcher's validation cycle
// follows). GO reads `running`. The dispatcher ends the launch with `end`, which loads
// STATUS, ERR_* and CYCLES, clears `running` and pulses `irq_done`. While a launch runs,
// CYCLES reads the dispatcher's live cycle count.
module mgpu_host_regs #(
    parameter int          W        = 8,
    parameter logic [32:0] MEM_SIZE = 33'h1_0000_0000
) (
    input  logic             clk,
    input  logic             rst_n,
    // CSR port (interfaces.md 1)
    input  logic             csr_valid,
    output logic             csr_ready,
    input  logic             csr_write,
    input  logic [7:0]       csr_addr,
    input  logic [31:0]      csr_wdata,
    output logic             csr_rvalid,
    output logic [31:0]      csr_rdata,
    output logic             irq_done,
    // launch parameters, to the dispatcher
    output logic [31:0]      entry,
    output logic [31:0]      grid_dim,
    output logic [31:0]      block_dim,
    output logic [7:0][31:0] args,
    output logic             launch,
    // end of launch, from the dispatcher
    input  logic             end_valid,
    input  logic [5:0][31:0] end_status,   // STATUS, ERR_PC, ERR_BLOCK, ERR_WARP, ERR_LANE, ERR_INFO
    input  logic [31:0]      end_cycles,
    input  logic [31:0]      live_cycles
);
  localparam logic [31:0] ID_VALUE = 32'h4D47_5055;       // "MGPU"
  localparam logic [31:0] VERSION_VALUE = 32'h0001_0000;  // 1.0
  localparam logic [5:0] A_ID = 6'h00, A_VERSION = 6'h01, A_MAX_BLOCK_DIM = 6'h02,
                         A_MEM_SIZE = 6'h03, A_ENTRY = 6'h04, A_GRID_DIM = 6'h05,
                         A_BLOCK_DIM = 6'h06, A_GO = 6'h07, A_STATUS = 6'h08, A_ERR_INFO = 6'h0D,
                         A_CYCLES = 6'h0E, A_ARG0 = 6'h10, A_ARG7 = 6'h17;

  logic             running;
  logic [5:0][31:0] status;
  logic [31:0]      cycles;
  logic [5:0]       wa;                 // word address
  assign wa = csr_addr[7:2];
  assign csr_ready = 1'b1;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      running <= 1'b0;
      launch <= 1'b0;
      irq_done <= 1'b0;
      csr_rvalid <= 1'b0;
      entry <= '0;
      grid_dim <= '0;
      block_dim <= '0;
      args <= '0;
      status <= '0;
      cycles <= '0;
    end else begin
      launch <= 1'b0;
      irq_done <= 1'b0;
      // writes
      if (csr_valid && csr_write && !running) begin
        unique case (wa) inside
          A_ENTRY:          entry <= csr_wdata;
          A_GRID_DIM:       grid_dim <= csr_wdata;
          A_BLOCK_DIM:      block_dim <= csr_wdata;
          [A_ARG0:A_ARG7]:  args[3'(wa - A_ARG0)] <= csr_wdata;
          A_GO: if (csr_wdata[0]) begin
            running <= 1'b1;
            launch <= 1'b1;
            status <= '0;
            cycles <= '0;
          end
          default: ;
        endcase
      end
      // end of launch
      if (end_valid) begin
        running <= 1'b0;
        irq_done <= 1'b1;
        status <= end_status;
        cycles <= end_cycles;
      end
      // reads (one-cycle latency)
      csr_rvalid <= csr_valid && !csr_write;
      if (csr_valid && !csr_write) begin
        unique case (wa) inside
          A_ID:                csr_rdata <= ID_VALUE;
          A_VERSION:           csr_rdata <= VERSION_VALUE;
          A_MAX_BLOCK_DIM:     csr_rdata <= 32'(32 * W);
          A_MEM_SIZE:          csr_rdata <= MEM_SIZE[31:0];    // 0 means 2^32
          A_ENTRY:             csr_rdata <= entry;
          A_GRID_DIM:          csr_rdata <= grid_dim;
          A_BLOCK_DIM:         csr_rdata <= block_dim;
          A_GO:                csr_rdata <= 32'(running);
          [A_STATUS:A_ERR_INFO]: csr_rdata <= status[3'(wa - A_STATUS)];
          A_CYCLES:            csr_rdata <= running ? live_cycles : cycles;
          [A_ARG0:A_ARG7]:     csr_rdata <= args[3'(wa - A_ARG0)];
          default:             csr_rdata <= '0;
        endcase
      end
    end
  end

  /* verilator lint_off UNUSEDPARAM */
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{MEM_SIZE[32], csr_addr[1:0]};   // MEM_SIZE = 2^32 reads 0; byte offset bits ignored
  /* verilator lint_on UNUSEDSIGNAL */
  /* verilator lint_on UNUSEDPARAM */
endmodule
