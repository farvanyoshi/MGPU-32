// mgpu_vrf: vector register file, W warps x 32 registers x 32 lanes x 32 bits, in RAMs
// (docs/uarch/core.md 6).
//
// Organization: one bank per warp slot; in each bank NC identical read copies (one per
// operand stream), and in each copy one RAM per pass (32 registers x L lanes). So:
//   - a warp's instruction reads one pass per cycle on each copy (read in cycle n, data in
//     cycle n + 1), and instructions of different warps read different banks;
//   - a write may cover any lanes of a register, so a full-register write (a load, V_SHFL)
//     takes one cycle, writing every pass RAM of the bank;
//   - at most one writer targets a warp in a cycle (a warp has one instruction in flight),
//     so each bank has one write port, muxed among the writers.
// Never-written lanes read 0: a valid bit per (warp, register, lane) is cleared by `init`
// (block start, ISA 13.3) and set by writes. v0 is never written, so it reads 0.
module mgpu_vrf #(
    parameter int W  = 8,
    parameter int L  = 8,
    parameter int NC = 4,    // read copies (operand streams)
    parameter int NW = 6,    // write ports
    localparam int P  = 32 / L,
    localparam int PB = (P > 1) ? $clog2(P) : 1,
    localparam int WB = (W > 1) ? $clog2(W) : 1
) (
    input  logic                                 clk,
    input  logic                                 init,
    // reads, per bank: pass rpass of register rreg on each copy; data next cycle
    input  logic [W-1:0]                         re,
    input  logic [W-1:0][PB-1:0]                 rpass,
    input  logic [W-1:0][NC-1:0][4:0]            rreg,
    output logic [W-1:0][NC-1:0][L-1:0][31:0]    rdata,
    // writes: any lanes of one register
    input  logic [NW-1:0]                        we,
    input  logic [NW-1:0][WB-1:0]                wwarp,
    input  logic [NW-1:0][4:0]                   wreg,
    input  logic [NW-1:0][31:0]                  wmask,
    input  logic [NW-1:0][31:0][31:0]            wdata
);
  logic [W-1:0][31:0][31:0] vld;               // [warp][register][lane]

  for (genvar w = 0; w < W; w++) begin : g_bank
    // the writer of this bank, if any
    logic              bw;
    logic [4:0]        breg;
    logic [31:0]       bmask;
    logic [31:0][31:0] bdata;
    always_comb begin
      bw = 1'b0;
      breg = '0;
      bmask = '0;
      bdata = '0;
      for (int k = NW - 1; k >= 0; k--) begin
        if (we[k] && int'(wwarp[k]) == w) begin
          bw = 1'b1;
          breg = wreg[k];
          bmask = wmask[k];
          bdata = wdata[k];
        end
      end
    end

    // valid bits of the lanes read, aligned with the RAM output
    logic [NC-1:0][L-1:0] rv;
    logic [PB-1:0]        rpass_q;
    always_ff @(posedge clk) begin
      if (re[w]) begin
        rpass_q <= rpass[w];
        for (int c = 0; c < NC; c++)
          for (int l = 0; l < L; l++)
            rv[c][l] <= vld[w][rreg[w][c]][int'(rpass[w]) * L + l];
      end
    end

    for (genvar c = 0; c < NC; c++) begin : g_copy
      logic [P-1:0][L-1:0][31:0] out;
      for (genvar p = 0; p < P; p++) begin : g_pass
        logic [L-1:0] lwe;
        always_comb for (int l = 0; l < L; l++) lwe[l] = bw && bmask[p * L + l];
        mgpu_ram #(.DEPTH(32), .LANES(L)) u_ram (
            .clk, .we(lwe), .waddr(breg), .wdata(bdata[p * L +: L]),
            .re(re[w] && int'(rpass[w]) == p), .raddr(rreg[w][c]), .rdata(out[p]));
      end
      always_comb
        for (int l = 0; l < L; l++) rdata[w][c][l] = rv[c][l] ? out[rpass_q][l] : 32'd0;
    end

    always_ff @(posedge clk) begin
      if (init) vld[w] <= '0;
      else if (bw) vld[w][breg] <= vld[w][breg] | bmask;
    end
  end
endmodule
