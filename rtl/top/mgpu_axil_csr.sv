// mgpu_axil_csr: AXI4-Lite slave in front of the host CSR port (docs/uarch/top.md 1).
//
// One transaction at a time. A write needs both its address (AW) and data (W); it becomes
// one CSR write and is answered on B. A read (AR) becomes one CSR read; the data returns on
// R. Writes take precedence when both are offered. Responses are always OKAY. Byte strobes
// are ignored: every MGPU-32 register is a full 32-bit word.
module mgpu_axil_csr (
    input  logic        clk,
    input  logic        rst_n,
    // AXI4-Lite slave
    input  logic        s_awvalid,
    output logic        s_awready,
    input  logic [7:0]  s_awaddr,
    input  logic        s_wvalid,
    output logic        s_wready,
    input  logic [31:0] s_wdata,
    input  logic [3:0]  s_wstrb,
    output logic        s_bvalid,
    input  logic        s_bready,
    output logic [1:0]  s_bresp,
    input  logic        s_arvalid,
    output logic        s_arready,
    input  logic [7:0]  s_araddr,
    output logic        s_rvalid,
    input  logic        s_rready,
    output logic [31:0] s_rdata,
    output logic [1:0]  s_rresp,
    // CSR master (interfaces.md 1)
    output logic        csr_valid,
    input  logic        csr_ready,
    output logic        csr_write,
    output logic [7:0]  csr_addr,
    output logic [31:0] csr_wdata,
    input  logic        csr_rvalid,
    input  logic [31:0] csr_rdata
);
  typedef enum logic [1:0] { S_IDLE, S_BRESP, S_RWAIT, S_RRESP } state_e;
  state_e state;

  logic do_write, do_read;
  assign do_write = state == S_IDLE && s_awvalid && s_wvalid && csr_ready;
  assign do_read = state == S_IDLE && !(s_awvalid && s_wvalid) && s_arvalid && csr_ready;

  assign s_awready = do_write;
  assign s_wready = do_write;
  assign s_arready = do_read;
  assign csr_valid = do_write || do_read;
  assign csr_write = do_write;
  assign csr_addr = do_write ? s_awaddr : s_araddr;
  assign csr_wdata = s_wdata;
  assign s_bvalid = state == S_BRESP;
  assign s_bresp = 2'b00;
  assign s_rvalid = state == S_RRESP;
  assign s_rresp = 2'b00;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_IDLE;
    end else begin
      unique case (state)
        S_IDLE:  if (do_write) state <= S_BRESP;
                 else if (do_read) state <= S_RWAIT;
        S_BRESP: if (s_bready) state <= S_IDLE;
        S_RWAIT: if (csr_rvalid) begin
          s_rdata <= csr_rdata;
          state <= S_RRESP;
        end
        S_RRESP: if (s_rready) state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{s_wstrb};   // registers are whole words
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
