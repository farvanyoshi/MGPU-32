// mgpu_vrf: vector register file, W warps x 32 registers x 32 lanes x 32 bits.
// Full-register (32-lane) combinational reads on NR ports and full-register writes on NW
// ports (one per writing unit; the executors merge inactive lanes before writing). v0
// reads as zero. `init` clears every register (ISA 13.3).
//
// This is the behavioural form used for simulation. The FPGA mapping (block-RAM banks per
// pass and operand, a live-value table for the multiple writers) is a synthesis-time
// refinement; see docs/uarch/core.md section 6.
module mgpu_vrf #(
    parameter int W  = 8,
    parameter int NR = 4,
    parameter int NW = 5,
    localparam int WID = $clog2(W)
) (
    input  logic                       clk,
    input  logic [NR-1:0][WID-1:0]     rwarp,
    input  logic [NR-1:0][4:0]         raddr,
    output logic [NR-1:0][31:0][31:0]  rdata,
    input  logic [NW-1:0]              we,
    input  logic [NW-1:0][WID-1:0]     wwarp,
    input  logic [NW-1:0][4:0]         waddr,
    input  logic [NW-1:0][31:0][31:0]  wdata,
    input  logic                       init
);
  logic [31:0][31:0] regs [W][32];
  always_ff @(posedge clk) begin
    if (init) begin
      for (int w = 0; w < W; w++) for (int r = 0; r < 32; r++) regs[w][r] <= '0;
    end else begin
      for (int p = 0; p < NW; p++)
        if (we[p] && waddr[p] != 0) regs[wwarp[p]][waddr[p]] <= wdata[p];
    end
  end
  for (genvar p = 0; p < NR; p++) begin : g_rd
    assign rdata[p] = (raddr[p] == 0) ? '0 : regs[rwarp[p]][raddr[p]];
  end
endmodule
