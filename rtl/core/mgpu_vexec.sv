// mgpu_vexec: a vector execution unit with its pass sequencer (docs/uarch/core.md 7, 11.2).
//
// KIND 0 = VALU (mgpu_valu), 1 = FPU (mgpu_fpu), 2 = SFU (mgpu_sfu). An instruction enters
// in cycle e (= issue + 2) as a scalar packet; its vector operands arrive from the register
// file one slice of L lanes per cycle, slice k in cycle e + k (k < P = 32 / L): va, vb, vc
// and the old vd (vd only for the retire record). The unit takes LANES lanes per cycle
// (PASSES = 32 / LANES >= P), unit pass j in cycle e + j, from the arriving slice or from
// the slices already buffered. Each pass's result is written back (lanes under EXEC) as it
// leaves the unit; the retire record, with all 32 lanes after the write, is presented with
// the last pass, in cycle e + PASSES + LAT - 1, the completion cycle of the contract.
//
// VALU only: V_SHFL and V_READLANE bypass the lanes. Once all slices of va and vb are in
// (cycle e + P - 1) the result is formed and presented P + 2 cycles later, in cycle
// e + 2P + 1 (occupancy 2P, latency 2).
module mgpu_vexec
  import mgpu_isa_pkg::*;
  import mgpu_core_pkg::*;
#(
    parameter int KIND  = 0,
    parameter int LANES = 8,
    parameter int LAT   = 2,
    parameter int L     = 8,      // lanes per register-file slice
    localparam int PASSES = 32 / LANES,
    localparam int P      = 32 / L,
    localparam int PB = (PASSES > 1) ? $clog2(PASSES) : 1
) (
    input  logic               clk,
    input  logic               rst_n,
    input  ipkt_t              in,
    input  logic [L-1:0][31:0] sl_a, sl_b, sl_c, sl_d,
    output vwr_t               vwr,
    output retire_t            out
);
  logic xlane;   // V_SHFL / V_READLANE
  assign xlane = KIND == 0 && in.valid && (in.ctrl.op == OP_V_SHFL || in.ctrl.op == OP_V_READLANE);

  // ---------------------------------------------------------------- operand slices
  // k = cycles since entry while slices arrive; the arriving slice is used directly, earlier
  // ones from the buffers
  ipkt_t             cur;              // instruction whose slices are arriving / being fed
  logic [7:0]        k;
  logic              arriving;
  logic [31:0][31:0] ba, bb, bc, bd;   // buffered lanes
  logic [31:0][31:0] va, vb, vc, vd;   // all lanes, as far as they have arrived
  ipkt_t             x;                // the instruction of this cycle's slice
  logic [7:0]        ks;               // index of the arriving slice
  assign x = in.valid ? in : cur;
  assign ks = in.valid ? 8'd0 : k;
  assign arriving = in.valid || k < 8'(P);
  always_comb begin
    va = ba;
    vb = bb;
    vc = bc;
    vd = bd;
    if (arriving) begin
      for (int l = 0; l < L; l++) begin
        va[int'(ks) * L + l] = sl_a[l];
        vb[int'(ks) * L + l] = sl_b[l];
        vc[int'(ks) * L + l] = sl_c[l];
        vd[int'(ks) * L + l] = sl_d[l];
      end
    end
    for (int i = 0; i < 32; i++) begin
      if (x.gsr) va[i] = x.gsr_base + (x.gsr_lin ? 32'(i) : 32'd0);
      if (x.bcast) vb[i] = x.sb;
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      k <= 8'(P);
    end else begin
      if (in.valid) begin
        cur <= in;
        k <= 8'd1;
      end else if (k < 8'(P)) begin
        k <= k + 8'd1;
      end
      if (arriving) begin
        ba <= va;
        bb <= vb;
        bc <= vc;
        bd <= vd;
      end
    end
  end
  logic last_slice;                     // all slices are in (this cycle)
  assign last_slice = arriving && (in.valid ? P == 1 : k == 8'(P - 1));

  // ---------------------------------------------------------------- feeding the unit
  logic          feeding, f_first, f_valid;
  logic [PB-1:0] pass, feed_pass;
  assign f_first   = in.valid && !xlane;
  assign f_valid   = f_first || feeding;
  assign feed_pass = f_first ? '0 : pass;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      feeding <= 1'b0;
      pass <= '0;
    end else if (f_first) begin
      feeding <= PASSES > 1;
      pass <= PB'(1);
    end else if (feeding) begin
      pass <= pass + 1'b1;
      if (pass == PB'(PASSES - 1)) feeding <= 1'b0;
    end
  end

  logic [LANES-1:0][31:0] fa, fb, fc;
  logic [LANES-1:0]       fsel;
  always_comb begin
    for (int i = 0; i < LANES; i++) begin
      fa[i]   = va[int'(feed_pass) * LANES + i];
      fb[i]   = vb[int'(feed_pass) * LANES + i];
      fc[i]   = vc[int'(feed_pass) * LANES + i];
      fsel[i] = x.sc[int'(feed_pass) * LANES + i];
    end
  end

  logic                   u_valid;
  logic [LANES-1:0][31:0] u_res;
  logic [LANES-1:0]       u_cond;
  if (KIND == 0) begin : g_valu
    mgpu_valu #(.L(LANES), .LAT(LAT)) u (
        .clk, .rst_n, .in_valid(f_valid), .op(x.ctrl.op), .a(fa), .b(fb), .c(fc),
        .sel(fsel), .out_valid(u_valid), .result(u_res), .cond(u_cond));
  end else if (KIND == 1) begin : g_fpu
    mgpu_fpu #(.L(LANES), .LAT(LAT)) u (
        .clk, .rst_n, .in_valid(f_valid), .op(x.ctrl.op), .a(fa), .b(fb), .c(fc),
        .out_valid(u_valid), .result(u_res));
    assign u_cond = '0;
  end else begin : g_sfu
    mgpu_sfu #(.S(LANES), .LAT(LAT)) u (
        .clk, .rst_n, .in_valid(f_valid), .op(x.ctrl.op), .a(fa),
        .out_valid(u_valid), .result(u_res));
    assign u_cond = '0;
  end

  logic        m_valid;
  logic [PB:0] m_meta, f_meta;   // {last pass, pass}
  assign f_meta = {feed_pass == PB'(PASSES - 1), feed_pass};
  mgpu_pipe #(.DEPTH(LAT), .WIDTH(PB + 1)) u_meta (
      .clk, .rst_n, .in_valid(f_valid), .in_data(f_meta), .out_valid(m_valid), .out_data(m_meta));

  // Instructions between feeding and output (at most two are in the unit at once); the old
  // vd, needed by the record only, is added once its last slice is in, before the last pass
  // leaves the unit.
  typedef struct packed {
    ipkt_t             p;
    logic [31:0][31:0] vd;
  } qent_t;
  qent_t      q [4];
  logic [1:0] q_wr, q_rd, slot;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      q_wr <= '0;
      q_rd <= '0;
    end else begin
      if (f_first) begin
        q[q_wr].p <= in;
        slot <= q_wr;
        q_wr <= q_wr + 1'b1;
      end
      if (last_slice && !(KIND == 0 && (x.ctrl.op == OP_V_SHFL || x.ctrl.op == OP_V_READLANE)))
        q[in.valid ? q_wr : slot].vd <= vd;
      if (m_valid && m_meta[PB]) q_rd <= q_rd + 1'b1;
    end
  end

  // ---------------------------------------------------------------- results
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

  // the instruction at the output: the queue head, or the one whose last slice is arriving
  // now (P = PASSES = 1 and LAT = 1 never happen, but keep it general)
  qent_t o;
  assign o = q[q_rd];

  // per-pass write-back
  vwr_t p_vwr;
  always_comb begin
    p_vwr = '0;
    if (m_valid && o.p.ctrl.vec_dst && o.p.ctrl.rd != 5'd0) begin
      p_vwr.valid = 1'b1;
      p_vwr.warp = o.p.warp;
      p_vwr.rd = o.p.ctrl.rd;
      for (int i = 0; i < LANES; i++) begin
        p_vwr.mask[int'(m_meta[PB-1:0]) * LANES + i] = o.p.exec[int'(m_meta[PB-1:0]) * LANES + i];
        p_vwr.data[int'(m_meta[PB-1:0]) * LANES + i] = u_res[i];
      end
    end
  end

  // ---------------------------------------------------------------- cross-lane path (VALU)
  typedef struct packed {
    retire_t r;
    vwr_t    w;
  } xres_t;
  xres_t x_res, x_out;
  logic  x_go, x_valid;
  assign x_go = KIND == 0 && last_slice && (x.ctrl.op == OP_V_SHFL || x.ctrl.op == OP_V_READLANE);
  always_comb begin
    logic [31:0][31:0] merged;
    x_res = '0;
    x_res.r = ret_from(x);
    merged = vd;
    for (int i = 0; i < 32; i++) if (x.exec[i]) merged[i] = va[vb[i][4:0]];   // V_SHFL
    if (x.ctrl.op == OP_V_READLANE) begin
      x_res.r = ret_scalar(x_res.r, x.ctrl.rd, va[x.sb[4:0]]);   // ignores EXEC
    end else begin
      x_res.r = ret_vector(x_res.r, x.ctrl.rd, merged);
      x_res.w.valid = x.ctrl.rd != 5'd0;
      x_res.w.warp = x.warp;
      x_res.w.rd = x.ctrl.rd;
      x_res.w.mask = x.exec;
      x_res.w.data = merged;
    end
  end
  if (KIND == 0) begin : g_cross
    mgpu_pipe #(.DEPTH(P + 2), .WIDTH($bits(xres_t))) u_x (
        .clk, .rst_n, .in_valid(x_go), .in_data(x_res), .out_valid(x_valid), .out_data(x_out));
  end else begin : g_nocross
    assign x_valid = 1'b0;
    assign x_out = '0;
  end

  // ---------------------------------------------------------------- outputs
  always_comb begin
    logic [31:0][31:0] merged;
    merged = o.vd;
    for (int i = 0; i < 32; i++) if (o.p.exec[i]) merged[i] = res_full[i];
    out = '0;
    vwr = p_vwr;
    if (x_valid) begin
      out = x_out.r;
      vwr = x_out.w;
    end else if (m_valid && m_meta[PB]) begin
      out = ret_from(o.p);
      if (KIND == 0 && unit_of(o.p.ctrl.op) == UNIT_VALU && o.p.ctrl.scl_dst)   // V_CMP
        out = ret_scalar(out, o.p.ctrl.rd, cond_full & o.p.exec);
      else
        out = ret_vector(out, o.p.ctrl.rd, merged);
    end
  end

  // Fields a given unit kind does not use.
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{u_valid, fsel, fb, fc, x_go, x_res};
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
