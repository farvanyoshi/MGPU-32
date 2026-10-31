// Test wrapper: decodes `word` into an issue packet for mgpu_vexec and flattens its retire
// record. The testbench forms the operands as the issue stage does (B per lane: S[rb], the
// immediate or V[rb]).
module mgpu_vexec_probe
  import mgpu_core_pkg::*;
#(
    parameter int KIND  = 0,
    parameter int LANES = 8,
    parameter int LAT   = 2
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
    output logic [31:0]       exec_after
);
  ipkt_t   p;
  retire_t r;
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
    p.va = va;
    p.vb = vb;
    p.vc = vc;
    p.vd = vd;
  end
  mgpu_vexec #(.KIND(KIND), .LANES(LANES), .LAT(LAT)) u (.clk, .rst_n, .in(p), .out(r));
  assign out_valid = r.valid;
  assign out_pc = r.pc;
  assign sw_valid = r.sw_valid;
  assign sw_reg = r.sw_reg;
  assign sw_val = r.sw_val;
  assign vw_valid = r.vw_valid;
  assign vw_reg = r.vw_reg;
  assign vw_val = r.vw_val;
  assign exec_after = r.exec_after;
endmodule
