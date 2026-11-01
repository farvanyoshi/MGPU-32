// Test wrapper: decodes `word` into an issue packet for mgpu_vexec, plays the register file
// (slice k of va, vb, vc and vd in the k-th cycle after entry, from the full vectors given
// at entry), and flattens the retire record and the register writes. The testbench forms
// B per lane itself (S[rb], the immediate or V[rb]).
module mgpu_vexec_probe
  import mgpu_core_pkg::*;
#(
    parameter int KIND  = 0,
    parameter int LANES = 8,
    parameter int LAT   = 2,
    parameter int L     = 8,
    localparam int P    = 32 / L
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              in_valid,
    input  logic [31:0]       pc,
    input  logic [31:0]       word,
    input  logic [31:0]       exec,
    input  logic [31:0]       sa, sb, sc, sd,
    input  logic [31:0][31:0] va, vb, vc, vd,
    output logic              out_valid,
    output logic [31:0]       out_pc,
    output logic              sw_valid,
    output logic [4:0]        sw_reg,
    output logic [31:0]       sw_val,
    output logic              vw_valid,
    output logic [4:0]        vw_reg,
    output logic [31:0][31:0] vw_val,
    output logic [31:0]       exec_after,
    output logic              vwr_valid,
    output logic [4:0]        vwr_rd,
    output logic [31:0]       vwr_mask,
    output logic [31:0][31:0] vwr_data
);
  ipkt_t   p;
  retire_t r;
  vwr_t    w;
  ctrl_t   c;
  mgpu_decode u_dec (.word(word), .ctrl(c));
  always_comb begin
    p = '0;
    p.valid = in_valid;
    p.pc = pc;
    p.word = word;
    p.ctrl = c;
    p.exec = exec;
    p.sa = sa;
    p.sb = sb;
    p.sc = sc;
    p.sd = sd;
  end

  // the register file stand-in
  logic [31:0][31:0] ha, hb, hc, hd;
  logic [7:0]        k;
  logic [L-1:0][31:0] sl_a, sl_b, sl_c, sl_d;
  always_ff @(posedge clk) begin
    if (!rst_n) k <= 8'(P);
    else if (in_valid) begin
      ha <= va;
      hb <= vb;
      hc <= vc;
      hd <= vd;
      k <= 8'd1;
    end else if (k < 8'(P)) k <= k + 8'd1;
  end
  always_comb begin
    for (int l = 0; l < L; l++) begin
      sl_a[l] = in_valid ? va[l] : ha[int'(k) * L + l];
      sl_b[l] = in_valid ? vb[l] : hb[int'(k) * L + l];
      sl_c[l] = in_valid ? vc[l] : hc[int'(k) * L + l];
      sl_d[l] = in_valid ? vd[l] : hd[int'(k) * L + l];
      if (!in_valid && k >= 8'(P)) begin
        sl_a[l] = 32'hBAD0_0000;     // nothing should read beyond the last slice
        sl_b[l] = 32'hBAD0_0000;
        sl_c[l] = 32'hBAD0_0000;
        sl_d[l] = 32'hBAD0_0000;
      end
    end
  end

  mgpu_vexec #(.KIND(KIND), .LANES(LANES), .LAT(LAT), .L(L)) u (
      .clk, .rst_n, .in(p), .sl_a, .sl_b, .sl_c, .sl_d, .vwr(w), .out(r));
  assign out_valid = r.valid;
  assign out_pc = r.pc;
  assign sw_valid = r.sw_valid;
  assign sw_reg = r.sw_reg;
  assign sw_val = r.sw_val;
  assign vw_valid = r.vw_valid;
  assign vw_reg = r.vw_reg;
  assign vw_val = r.vw_val;
  assign exec_after = r.exec_after;
  assign vwr_valid = w.valid;
  assign vwr_rd = w.rd;
  assign vwr_mask = w.mask;
  assign vwr_data = w.data;
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{r, w.warp};
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
