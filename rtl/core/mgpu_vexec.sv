// mgpu_vexec: a vector execution unit with its pass sequencer (docs/uarch/core.md 7, 11.2).
//
// KIND 0 = VALU (mgpu_valu), 1 = FPU (mgpu_fpu), 2 = SFU (mgpu_sfu). An instruction enters
// as a full-operand packet. Its 32 lanes are fed LANES per cycle (PASSES = 32 / LANES),
// starting in the entry cycle; results are collected in order, and the merged 32-lane
// result is presented as a retire record in the cycle the last pass leaves the unit:
// entry + PASSES + LAT - 1, the completion cycle of the timing contract. A new
// instruction may enter as soon as the previous one has been fed (the scheduler's unit
// occupancy guarantees this).
//
// VALU only: V_SHFL and V_READLANE bypass the lanes. They are computed at entry and
// presented 2 * PASSES + 1 cycles later (occupancy 2 * PASSES, latency 2).
module mgpu_vexec
  import mgpu_isa_pkg::*;
  import mgpu_core_pkg::*;
#(
    parameter int KIND  = 0,
    parameter int LANES = 8,
    parameter int LAT   = 2,
    localparam int PASSES = 32 / LANES,
    localparam int PB = (PASSES > 1) ? $clog2(PASSES) : 1
) (
    input  logic    clk,
    input  logic    rst_n,
    input  ipkt_t   in,
    output retire_t out
);
  logic xlane;   // V_SHFL / V_READLANE
  assign xlane = KIND == 0 && in.valid && (in.ctrl.op == OP_V_SHFL || in.ctrl.op == OP_V_READLANE);

  // ---- feeding: pass 0 comes straight from `in`, later passes from `cur`
  ipkt_t         cur;
  logic          feeding;
  logic [PB-1:0] pass;
  logic          f_first, f_valid;
  /* verilator lint_off UNUSEDSIGNAL */
  ipkt_t         feed_pkt;   // only the operands and op are read
  /* verilator lint_on UNUSEDSIGNAL */
  logic [PB-1:0] feed_pass;

  assign f_first   = in.valid && !xlane;
  assign f_valid   = f_first || feeding;
  assign feed_pkt  = f_first ? in : cur;
  assign feed_pass = f_first ? '0 : pass;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      feeding <= 1'b0;
      pass <= '0;
    end else if (f_first) begin
      cur <= in;
      feeding <= PASSES > 1;
      pass <= PB'(1);
    end else if (feeding) begin
      pass <= pass + 1'b1;
      if (pass == PB'(PASSES - 1)) feeding <= 1'b0;
    end
  end

  logic [LANES-1:0][31:0] fa, fb, fc;
  logic [LANES-1:0]       fsel;
  for (genvar i = 0; i < LANES; i++) begin : g_slice
    assign fa[i]   = feed_pkt.va[int'(feed_pass) * LANES + i];
    assign fb[i]   = feed_pkt.vb[int'(feed_pass) * LANES + i];
    assign fc[i]   = feed_pkt.vc[int'(feed_pass) * LANES + i];
    assign fsel[i] = feed_pkt.sc[int'(feed_pass) * LANES + i];
  end

  // ---- the unit, and a metadata pipe of the same latency
  logic                   u_valid;
  logic [LANES-1:0][31:0] u_res;
  logic [LANES-1:0]       u_cond;
  if (KIND == 0) begin : g_valu
    mgpu_valu #(.L(LANES), .LAT(LAT)) u (
        .clk, .rst_n, .in_valid(f_valid), .op(feed_pkt.ctrl.op), .a(fa), .b(fb), .c(fc),
        .sel(fsel), .out_valid(u_valid), .result(u_res), .cond(u_cond));
  end else if (KIND == 1) begin : g_fpu
    mgpu_fpu #(.L(LANES), .LAT(LAT)) u (
        .clk, .rst_n, .in_valid(f_valid), .op(feed_pkt.ctrl.op), .a(fa), .b(fb), .c(fc),
        .out_valid(u_valid), .result(u_res));
    assign u_cond = '0;
  end else begin : g_sfu
    mgpu_sfu #(.S(LANES), .LAT(LAT)) u (
        .clk, .rst_n, .in_valid(f_valid), .op(feed_pkt.ctrl.op), .a(fa),
        .out_valid(u_valid), .result(u_res));
    assign u_cond = '0;
  end

  logic        m_valid;
  logic [PB:0] m_meta, f_meta;   // {last pass, pass}
  assign f_meta = {feed_pass == PB'(PASSES - 1), feed_pass};
  mgpu_pipe #(.DEPTH(LAT), .WIDTH(PB + 1)) u_meta (
      .clk, .rst_n, .in_valid(f_valid), .in_data(f_meta), .out_valid(m_valid), .out_data(m_meta));

  // Instructions between feed and output (at most two are in the pipe at once).
  ipkt_t q [4];
  logic [1:0] q_wr, q_rd;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      q_wr <= '0;
      q_rd <= '0;
    end else begin
      if (f_first) begin
        q[q_wr] <= in;
        q_wr <= q_wr + 1'b1;
      end
      if (m_valid && m_meta[PB]) q_rd <= q_rd + 1'b1;
    end
  end

  // ---- collect the passes of the instruction at the output
  logic [31:0][31:0] acc, res_full;
  logic [31:0]       cacc, cond_full;
  always_ff @(posedge clk) begin
    if (m_valid) begin
      for (int i = 0; i < LANES; i++) begin
        acc[int'(m_meta[PB-1:0]) * LANES + i]  <= u_res[i];
        cacc[int'(m_meta[PB-1:0]) * LANES + i] <= u_cond[i];
      end
    end
  end
  always_comb begin
    res_full = acc;
    cond_full = cacc;
    for (int i = 0; i < LANES; i++) begin
      res_full[int'(m_meta[PB-1:0]) * LANES + i]  = u_res[i];
      cond_full[int'(m_meta[PB-1:0]) * LANES + i] = u_cond[i];
    end
  end

  // ---- cross-lane path (VALU)
  retire_t x_rec, x_out;
  logic    x_valid;
  always_comb begin
    logic [31:0][31:0] merged;
    x_rec = ret_from(in);
    merged = in.vd;
    if (in.ctrl.op == OP_V_READLANE) begin
      x_rec = ret_scalar(x_rec, in.ctrl.rd, in.va[in.sb[4:0]]);   // ignores EXEC
    end else begin
      for (int i = 0; i < 32; i++) if (in.exec[i]) merged[i] = in.va[in.vb[i][4:0]];
      x_rec = ret_vector(x_rec, in.ctrl.rd, merged);
    end
  end
  if (KIND == 0) begin : g_cross
    mgpu_pipe #(.DEPTH(2 * PASSES + 1), .WIDTH($bits(retire_t))) u_x (
        .clk, .rst_n, .in_valid(xlane), .in_data(x_rec), .out_valid(x_valid), .out_data(x_out));
  end else begin : g_nocross
    assign x_valid = 1'b0;
    assign x_out = '0;
  end

  // ---- the retire record
  always_comb begin
    ipkt_t p;
    logic [31:0][31:0] merged;
    p = q[q_rd];
    merged = p.vd;
    for (int i = 0; i < 32; i++) if (p.exec[i]) merged[i] = res_full[i];
    out = '0;
    if (x_valid) begin
      out = x_out;
    end else if (m_valid && m_meta[PB]) begin
      out = ret_from(p);
      if (KIND == 0 && unit_of(p.ctrl.op) == UNIT_VALU && p.ctrl.scl_dst)   // V_CMP
        out = ret_scalar(out, p.ctrl.rd, cond_full & p.exec);
      else
        out = ret_vector(out, p.ctrl.rd, merged);
    end
  end

  // Fields of the packet a given unit kind does not use.
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{u_valid, fsel, fb, fc};
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
