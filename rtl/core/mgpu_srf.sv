// mgpu_srf: scalar register file, W warps x 32 x 32 bits (docs/uarch/core.md section 6).
// NR combinational read ports, NW write ports (one per writing unit; a warp has one
// instruction in flight, so two ports never write the same warp in a cycle). s0 reads as
// zero. `init` loads the block-start state: s1-s8 = args, s31 = launch EXEC, others 0.
module mgpu_srf #(
    parameter int W  = 8,
    parameter int NR = 5,
    parameter int NW = 3,
    localparam int WID = $clog2(W)
) (
    input  logic                  clk,
    input  logic [NR-1:0][WID-1:0] rwarp,
    input  logic [NR-1:0][4:0]     raddr,
    output logic [NR-1:0][31:0]    rdata,
    input  logic [NW-1:0]          we,
    input  logic [NW-1:0][WID-1:0] wwarp,
    input  logic [NW-1:0][4:0]     waddr,
    input  logic [NW-1:0][31:0]    wdata,
    input  logic                  init,
    input  logic [7:0][31:0]      init_args,
    input  logic [W-1:0][31:0]    init_exec
);
  logic [31:0] regs [W][32];
  always_ff @(posedge clk) begin
    if (init) begin
      for (int w = 0; w < W; w++)
        for (int r = 0; r < 32; r++)
          regs[w][r] <= (r >= 1 && r <= 8) ? init_args[r-1] : (r == 31) ? init_exec[w] : 32'b0;
    end else begin
      for (int p = 0; p < NW; p++)
        if (we[p] && waddr[p] != 0) regs[wwarp[p]][waddr[p]] <= wdata[p];
    end
  end
  for (genvar p = 0; p < NR; p++) begin : g_rd
    assign rdata[p] = (raddr[p] == 0) ? 32'b0 : regs[rwarp[p]][raddr[p]];
  end
endmodule
