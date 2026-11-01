// Test wrapper: mgpu_axil_csr in front of mgpu_host_regs, with the dispatcher side on ports.
module mgpu_axil_probe (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             s_awvalid,
    output logic             s_awready,
    input  logic [7:0]       s_awaddr,
    input  logic             s_wvalid,
    output logic             s_wready,
    input  logic [31:0]      s_wdata,
    input  logic [3:0]       s_wstrb,
    output logic             s_bvalid,
    input  logic             s_bready,
    output logic [1:0]       s_bresp,
    input  logic             s_arvalid,
    output logic             s_arready,
    input  logic [7:0]       s_araddr,
    output logic             s_rvalid,
    input  logic             s_rready,
    output logic [31:0]      s_rdata,
    output logic [1:0]       s_rresp,
    output logic             launch,
    output logic             irq_done,
    input  logic             end_valid,
    input  logic [5:0][31:0] end_status,
    input  logic [31:0]      end_cycles
);
  logic             csr_valid, csr_ready, csr_write, csr_rvalid;
  logic [7:0]       csr_addr;
  logic [31:0]      csr_wdata, csr_rdata, entry, grid_dim, block_dim;
  logic [7:0][31:0] args;
  logic [31:0]      live_cycles;
  assign live_cycles = 32'd0;
  mgpu_axil_csr u_axil (.*);
  mgpu_host_regs #(.W(8), .MEM_SIZE(33'h0010_0000)) u_regs (.*);
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{entry, grid_dim, block_dim, args};
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
