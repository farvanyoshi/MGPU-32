// mgpu_vrf: vector register file. Three identical copies (one per source operand),
// each W x 32 registers x P passes entries of L x 32 bits; one write port with a lane
// mask written to all copies. v0 reads as zero (docs/uarch/core.md section 6).
// Reads are synchronous (block-RAM style): data appears the cycle after the address.
module mgpu_vrf #(
    parameter int W = 8,
    parameter int L = 8,
    localparam int P = 32 / L,
    localparam int WID = $clog2(W),
    localparam int PB = (P > 1) ? $clog2(P) : 1,
    localparam int AW = WID + 5 + PB
) (
    input  logic                    clk,
    input  logic [2:0][AW-1:0]      raddr,   // {warp, reg, pass}
    output logic [2:0][L-1:0][31:0] rdata,
    input  logic                    we,
    input  logic [AW-1:0]           waddr,
    input  logic [L-1:0]            wmask,
    input  logic [L-1:0][31:0]      wdata
);
  for (genvar c = 0; c < 3; c++) begin : g_copy
    logic [L-1:0][31:0] mem [2**AW];
    logic zero_q;
    logic [L-1:0][31:0] q;
    always_ff @(posedge clk) begin
      if (we && waddr[PB +: 5] != 0)
        for (int i = 0; i < L; i++) if (wmask[i]) mem[waddr][i] <= wdata[i];
      q <= mem[raddr[c]];
      zero_q <= raddr[c][PB +: 5] == 0;
    end
    assign rdata[c] = zero_q ? '0 : q;
  end
endmodule
