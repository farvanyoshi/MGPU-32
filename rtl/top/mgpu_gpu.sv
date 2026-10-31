// mgpu_gpu: the deliverable GPU for integration and synthesis: mgpu_top behind an AXI4-Lite
// host port (mgpu_axil_csr), with the 128-bit memory port. The verification outputs of
// mgpu_top (retire port, queue overflow) are left unconnected, so synthesis removes the
// logic that only serves them.
module mgpu_gpu #(
    parameter int          C        = 4,
    parameter int          W        = 8,
    parameter int          L        = 8,
    parameter int          S        = 2,
    parameter logic [32:0] MEM_SIZE = 33'h1_0000_0000,
    parameter int          QL       = 64
) (
    input  logic         clk,
    input  logic         rst_n,
    // AXI4-Lite host port
    input  logic         s_awvalid,
    output logic         s_awready,
    input  logic [7:0]   s_awaddr,
    input  logic         s_wvalid,
    output logic         s_wready,
    input  logic [31:0]  s_wdata,
    input  logic [3:0]   s_wstrb,
    output logic         s_bvalid,
    input  logic         s_bready,
    output logic [1:0]   s_bresp,
    input  logic         s_arvalid,
    output logic         s_arready,
    input  logic [7:0]   s_araddr,
    output logic         s_rvalid,
    input  logic         s_rready,
    output logic [31:0]  s_rdata,
    output logic [1:0]   s_rresp,
    output logic         irq_done,
    // memory port
    output logic         mreq_valid,
    input  logic         mreq_ready,
    output logic [1:0]   mreq_op,
    output logic [31:0]  mreq_addr,
    output logic [15:0]  mreq_tag,
    output logic [31:0]  mreq_mask,
    output logic         mwd_valid,
    output logic [127:0] mwd_data,
    output logic         mwd_last,
    input  logic         mrd_valid,
    input  logic [15:0]  mrd_tag,
    input  logic [127:0] mrd_data,
    input  logic         mrd_last,
    input  logic         mack_valid,
    input  logic [15:0]  mack_tag
);
  logic        csr_valid, csr_ready, csr_write, csr_rvalid;
  logic [7:0]  csr_addr;
  logic [31:0] csr_wdata, csr_rdata;
  mgpu_axil_csr u_axil (.*);
  /* verilator lint_off PINCONNECTEMPTY */
  mgpu_top #(.C(C), .W(W), .L(L), .S(S), .MEM_SIZE(MEM_SIZE), .QL(QL)) u_top (
      .clk, .rst_n, .csr_valid, .csr_ready, .csr_write, .csr_addr, .csr_wdata, .csr_rvalid,
      .csr_rdata, .irq_done, .mreq_valid, .mreq_ready, .mreq_op, .mreq_addr, .mreq_tag,
      .mreq_mask, .mwd_valid, .mwd_data, .mwd_last, .mrd_valid, .mrd_tag, .mrd_data, .mrd_last,
      .mack_valid, .mack_tag, .ret_valid(), .ret(), .overflow());
  /* verilator lint_on PINCONNECTEMPTY */
endmodule
