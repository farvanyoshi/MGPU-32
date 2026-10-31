// mgpu_srf: scalar register file, W warps x 32 x 32 bits, 3 read / 1 write ports.
// s0 reads as zero and ignores writes (docs/uarch/core.md section 6).
module mgpu_srf #(
    parameter int W = 8,
    localparam int WID = $clog2(W)
) (
    input  logic                clk,
    input  logic [2:0][WID-1:0] rwarp,
    input  logic [2:0][4:0]     raddr,
    output logic [2:0][31:0]    rdata,
    input  logic                we,
    input  logic [WID-1:0]      wwarp,
    input  logic [4:0]          waddr,
    input  logic [31:0]         wdata
);
  logic [31:0] regs [W][32];
  always_ff @(posedge clk) if (we && waddr != 0) regs[wwarp][waddr] <= wdata;
  for (genvar p = 0; p < 3; p++) begin : g_rd
    assign rdata[p] = (raddr[p] == 0) ? 32'b0 : regs[rwarp[p]][raddr[p]];
  end
endmodule
