// mgpu_lsu: load/store unit (docs/uarch/core.md 8, 11.4).
//
// An instruction enters at e = t + 2 as a scalar packet; its va (addresses), vb (atomic
// operands) and vd (store data, and the old values for the records) arrive from the
// register file one slice per cycle, slice k at e + k. With i = cycle - e:
//   i in [0, P)            address passes; at i = P - 1, with the last slice, mgpu_agen
//                          analyses the access: the precise fault ([MEM-3]; the instruction
//                          ends here, before any request), and the segment count, conflict
//                          degree or active lanes, reported to the scheduler (occ_*)
//   global, i = P + k      segment request k (k < n_seg), one per cycle
//     store                record at i = P - 1 + n_seg (with its last request; LSU_S)
//     load / atomic        record one cycle after the last response (LSU_B / LSU_C), from a
//                          per-warp slot, so the unit can move on meanwhile; with no active
//                          lane, at i = P + 1 (LSU_A)
//   shared, i = P + j      bank round j (j < d): every lane whose word is the j-th distinct
//                          word of its bank; record at i = P + d (LSU_A)
//   shared atomic          one active lane per cycle in ascending order; record at
//                          i = P + n_act (LSU_A)
// The unit holds one instruction at a time (occupancy P + n_seg, P + d or P + n_act). The
// LSU_A record register assumes P >= 2.
//
// Shared memory is 4096 words, poisoned with 0xDEADBEEF at block start. It is a flop array
// here; the rounds guarantee one word per bank per cycle, so it maps to 32 single-port
// banks.
module mgpu_lsu
  import mgpu_isa_pkg::*;
  import mgpu_core_pkg::*;
#(
    parameter int W = 8,
    parameter int L = 8,
    localparam int P = 32 / L,
    localparam int WB = (W > 1) ? $clog2(W) : 1
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              clr,          // block start
    input  logic [31:0]       mem_last,
    input  ipkt_t             in,
    input  logic [L-1:0][31:0] sl_a, sl_b, sl_d,
    output logic              occ_valid,    // analysis done: occupancy = P + occ_n
    output logic [7:0]        occ_n,
    output logic              fault,        // the access faults (with the analysis)
    output logic [7:0]        fault_code,
    output logic [4:0]        fault_lane,
    output logic [31:0]       fault_info,
    output ipkt_t             fault_pkt,
    output vwr_t              vwr_a,        // register writes: shared accesses, global loads,
    output vwr_t              vwr_b,        // global atomics (with rec_a, rec_b, rec_c)
    output vwr_t              vwr_c,
    output mreq_t             req,
    input  logic              rd_valid,     // READ response: the segment
    /* verilator lint_off UNUSEDSIGNAL */   // tags carry 5-bit warp numbers; W slots use the low bits
    input  logic [4:0]        rd_warp,
    input  logic [4:0]        rd_seg,
    input  logic [31:0][31:0] rd_data,
    input  logic              at_valid,     // ATOMIC response: old value of each lane
    input  logic [4:0]        at_warp,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [4:0]        at_seg,
    input  logic [31:0][31:0] at_old,
    output retire_t           rec_s,
    output retire_t           rec_a,
    output retire_t           rec_b,
    output retire_t           rec_c
);
  localparam logic [31:0] POISON = 32'hDEAD_BEEF;

  // ---------------------------------------------------------------- current instruction
  ipkt_t     cur, x;
  lsu_info_t ci, xi, ai;
  logic [7:0] cnt, xc, occ;
  logic       busy, xv, xs, xl, xst, xa;
  assign x   = in.valid ? in : cur;
  assign xc  = in.valid ? 8'd0 : cnt;
  assign xv  = in.valid || busy;
  assign xs  = x.ctrl.mem_shared;
  assign xl  = is_load(x.ctrl.op);
  assign xst = is_store(x.ctrl.op);
  assign xa  = !xl && !xst;

  // ---------------------------------------------------------------- slices and analysis
  logic [31:0][31:0] ba, bb, bd, va, vb, vd;
  logic              arriving, ana, af;
  logic [7:0]        af_code;
  logic [4:0]        af_lane;
  logic [31:0]       af_info;
  assign arriving = xv && xc < 8'(P);
  assign ana = xv && xc == 8'(P - 1);           // the last slice arrives
  always_comb begin
    va = ba;
    vb = bb;
    vd = bd;
    if (arriving) begin
      for (int l = 0; l < L; l++) begin
        va[int'(xc) * L + l] = sl_a[l];
        vb[int'(xc) * L + l] = sl_b[l];
        vd[int'(xc) * L + l] = sl_d[l];
      end
    end
    if (x.bcast) for (int i = 0; i < 32; i++) vb[i] = x.sb;
  end
  mgpu_agen u_agen (.ctrl(x.ctrl), .exec(x.exec), .sa(x.sa), .sd(x.sd), .va(va), .vb(vb), .vd(vd),
                    .mem_last, .info(ai), .fault(af), .fcode(af_code), .flane(af_lane), .finfo(af_info));
  assign xi = ana ? ai : ci;
  assign occ = 8'(P) + (!xs ? 8'(xi.n_seg) : xa ? 8'(xi.n_act) : 8'(xi.d));
  // reported only for the accesses whose occupancy is unknown at issue: vector accesses
  // with some active lane (scalar ones and EXEC = 0 are settled at issue)
  assign occ_valid = ana && !af && !is_scalar_mem(x.ctrl.op) && x.exec != 32'd0;
  assign occ_n = occ - 8'(P);
  assign fault = ana && af;
  assign fault_code = af_code;
  assign fault_lane = af_lane;
  assign fault_info = af_info;
  assign fault_pkt = x;
  logic ok;                                     // past the analysis without a fault
  assign ok = !(ana && af);

  // ---------------------------------------------------------------- global requests
  logic       g_req;
  logic [4:0] k;
  assign g_req = xv && !xs && xc >= 8'(P) && xc < 8'(P) + 8'(xi.n_seg);
  assign k = 5'(xc - 8'(P));
  always_comb begin
    logic [31:0] lk;
    logic        found;
    req = '0;
    found = 1'b0;
    for (int j = 0; j < 32; j++) lk[j] = xi.lanes[j] && xi.seg[j] == k;
    req.valid = g_req;
    req.kind = xl ? MK_READ : xst ? MK_WRITE : MK_ATOMIC;
    req.warp = x.warp;
    req.seg = k;
    req.lanes = lk;
    req.aop = atom_kind(x.ctrl.op);
    for (int j = 0; j < 32; j++) begin
      req.off[j] = xi.addr[j][6:2];
      req.opnd[j] = xi.data[j];
      if (lk[j]) begin
        if (!found) req.addr = {xi.addr[j][31:7], 7'b0};
        found = 1'b1;
        if (xst) begin                          // ascending: the highest lane wins
          req.wmask[xi.addr[j][6:2]] = 1'b1;
          req.wdata[xi.addr[j][6:2]] = xi.data[j];
        end
      end
    end
  end

  // store record, with the last request
  always_comb begin
    rec_s = '0;
    if (xv && ok && xc >= 8'(P - 1) && !xs && xst && xc == 8'(P - 1) + 8'(xi.n_seg)) begin
      rec_s = ret_from(x);
      rec_s.mw_lanes = xi.lanes;
      rec_s.mw_addr = xi.addr;
      rec_s.mw_val = xi.data;
    end
  end

  // ---------------------------------------------------------------- shared memory
  // 32 banks of 128 words (word address a: bank a[4:0], row a[11:5]), each a RAM with one
  // read and one write per cycle and a registered read. The reads of round j (or atomic step
  // j) are issued in cycle i = P - 1 + j and used in i = P + j; writes happen in i = P + j.
  // A word never written since block start reads as the poison value.
  logic [31:0][31:0]   sacc, snew;
  logic                s_round, s_last, a_step, a_last, s_rd, a_rd;
  logic [4:0]          j_round, j_rd;
  logic [31:0]         lr, lrd;       // lanes of this round / of the round being read
  logic [31:0][31:0]   rdl;           // read data of each lane (this round)
  int                  a_lane, a_lane_rd;
  logic [31:0]         a_old, a_new;
  logic [31:0]         b_re, b_vq;
  logic [31:0][6:0]    b_raddr, b_waddr;
  logic [31:0]         b_we;
  logic [31:0][31:0]   b_wdata, b_out;
  logic [4095:0]       sv;            // word written since block start
  logic                a_prev;        // the previous cycle was an atomic step: forward its result
  logic [11:0]         a_prev_word;
  logic [31:0]         a_prev_new;
  assign s_round = xv && xs && !xa && xc >= 8'(P) && xc < 8'(P) + 8'(xi.d);
  assign s_last  = s_round && xc == 8'(P) + 8'(xi.d) - 8'd1;
  assign j_round = 5'(xc - 8'(P));
  assign a_step  = xv && xs && xa && xc >= 8'(P) && xc < 8'(P) + 8'(xi.n_act);
  assign a_last  = a_step && xc == 8'(P) + 8'(xi.n_act) - 8'd1;
  assign j_rd    = 5'(xc - 8'(P - 1));
  assign s_rd    = xv && xs && !xa && xc >= 8'(P - 1) && xc < 8'(P - 1) + 8'(xi.d);
  assign a_rd    = xv && xs && xa && xc >= 8'(P - 1) && xc < 8'(P - 1) + 8'(xi.n_act);
  always_comb begin
    int n;
    a_lane = 0;
    a_lane_rd = 0;
    n = 0;
    for (int l = 0; l < 32; l++) begin
      if (xi.lanes[l]) begin
        if (8'(n) == xc - 8'(P)) a_lane = l;
        if (8'(n) == xc - 8'(P - 1)) a_lane_rd = l;
        n++;
      end
    end
    // bank reads
    b_re = '0;
    b_raddr = '0;
    for (int l = 0; l < 32; l++) begin
      lrd[l] = xi.lanes[l] && xi.round[l] == j_rd;
      if (s_rd && lrd[l]) begin
        b_re[xi.addr[l][6:2]] = 1'b1;
        b_raddr[xi.addr[l][6:2]] = xi.addr[l][13:7];
      end
    end
    if (a_rd) begin
      b_re[xi.addr[a_lane_rd][6:2]] = 1'b1;
      b_raddr[xi.addr[a_lane_rd][6:2]] = xi.addr[a_lane_rd][13:7];
    end
    // this round's data
    for (int l = 0; l < 32; l++) begin
      lr[l] = xi.lanes[l] && xi.round[l] == j_round;
      rdl[l] = b_vq[xi.addr[l][6:2]] ? b_out[xi.addr[l][6:2]] : POISON;
    end
    a_old = b_vq[xi.addr[a_lane][6:2]] ? b_out[xi.addr[a_lane][6:2]] : POISON;
    if (a_prev && a_prev_word == xi.addr[a_lane][13:2]) a_old = a_prev_new;
    a_new = atom_apply(atom_kind(x.ctrl.op), a_old, xi.data[a_lane]);
    // bank writes: a store's round (ascending lanes: the highest lane wins), an atomic step
    b_we = '0;
    b_waddr = '0;
    b_wdata = '0;
    if (s_round && xst) begin
      for (int l = 0; l < 32; l++) begin
        if (lr[l]) begin
          b_we[xi.addr[l][6:2]] = 1'b1;
          b_waddr[xi.addr[l][6:2]] = xi.addr[l][13:7];
          b_wdata[xi.addr[l][6:2]] = xi.data[l];
        end
      end
    end
    if (a_step) begin
      b_we[xi.addr[a_lane][6:2]] = 1'b1;
      b_waddr[xi.addr[a_lane][6:2]] = xi.addr[a_lane][13:7];
      b_wdata[xi.addr[a_lane][6:2]] = a_new;
    end
  end
  for (genvar b = 0; b < 32; b++) begin : g_bank
    mgpu_ram #(.DEPTH(128), .LANES(1)) u_ram (
        .clk, .we(b_we[b]), .waddr(b_waddr[b]), .wdata(b_wdata[b]),
        .re(b_re[b]), .raddr(b_raddr[b]), .rdata(b_out[b]));
  end

  // ---------------------------------------------------------------- post-occupancy records (LSU_A)
  retire_t    pw_rec, pw_next;
  vwr_t       pw_vwr, pw_wnext;
  logic [1:0] pw_cnt, pw_set;
  always_comb begin
    logic [31:0][31:0] m;
    pw_next = ret_from(x);
    pw_wnext = '0;
    pw_wnext.warp = x.warp;
    pw_wnext.rd = x.ctrl.rd;
    pw_set = 2'd0;
    m = vd;
    if (s_last) begin                        // shared load / store
      pw_set = 2'd1;
      if (xl) begin
        for (int l = 0; l < 32; l++) if (xi.lanes[l]) m[l] = lr[l] ? rdl[l] : sacc[l];
        pw_next = ret_vector(pw_next, x.ctrl.rd, m);
        pw_wnext.valid = x.ctrl.rd != 5'd0;
        pw_wnext.mask = xi.lanes;
        pw_wnext.data = m;
      end else begin
        pw_next.mw_lanes = xi.lanes;
        pw_next.mw_shared = 1'b1;
        pw_next.mw_addr = xi.addr;
        pw_next.mw_val = xi.data;
      end
    end else if (ana && ok && xs && xa && xi.n_act == 6'd0) begin
      pw_set = 2'd1;                         // shared atomic, no active lane
      pw_next = ret_vector(pw_next, x.ctrl.rd, m);
    end else if (a_last) begin               // shared atomic, last lane
      pw_set = 2'd1;
      for (int l = 0; l < 32; l++) begin
        if (xi.lanes[l]) begin
          m[l] = l == a_lane ? a_old : sacc[l];
          pw_next.mw_val[l] = l == a_lane ? a_new : snew[l];
        end
      end
      pw_next = ret_vector(pw_next, x.ctrl.rd, m);
      pw_next.mw_lanes = xi.lanes;
      pw_next.mw_shared = 1'b1;
      pw_next.mw_addr = xi.addr;
      pw_wnext.valid = x.ctrl.rd != 5'd0;
      pw_wnext.mask = xi.lanes;
      pw_wnext.data = m;
    end else if (ana && ok && !xs && !xst && xi.n_seg == 6'd0) begin
      pw_set = 2'd2;                         // global load / atomic, no active lane
      pw_next = ret_vector(pw_next, x.ctrl.rd, m);
    end
  end
  assign rec_a = pw_cnt == 2'd1 ? pw_rec : '0;
  assign vwr_a = pw_cnt == 2'd1 ? pw_vwr : '0;

  // ---------------------------------------------------------------- per-warp load / atomic slots
  logic [W-1:0]             done_b, done_c;
  ipkt_t [W-1:0]            ow_pkt;
  logic [W-1:0][31:0][31:0] ow_vd;
  logic [W-1:0][31:0]       ow_lanes;
  logic [W-1:0][31:0][4:0]  ow_seg;
  logic [W-1:0][31:0][31:0] ow_addr, ow_data, ow_acc;
  logic [W-1:0][5:0]        ow_pend;

  always_comb begin
    rec_b = '0;
    rec_c = '0;
    vwr_b = '0;
    vwr_c = '0;
    for (int w = 0; w < W; w++) begin
      logic [31:0][31:0] m;
      retire_t r;
      vwr_t    vw;
      m = ow_vd[w];
      vw = '0;
      vw.valid = !is_scalar_mem(ow_pkt[w].ctrl.op) && ow_pkt[w].ctrl.rd != 5'd0;
      vw.warp = ow_pkt[w].warp;
      vw.rd = ow_pkt[w].ctrl.rd;
      vw.mask = ow_lanes[w];
      vw.data = ow_acc[w];
      for (int l = 0; l < 32; l++) if (ow_lanes[w][l]) m[l] = ow_acc[w][l];
      r = ret_from(ow_pkt[w]);
      if (done_b[w]) begin
        rec_b = is_scalar_mem(ow_pkt[w].ctrl.op) ? ret_scalar(r, ow_pkt[w].ctrl.rd, ow_acc[w][0])
                                                 : ret_vector(r, ow_pkt[w].ctrl.rd, m);
        vwr_b = vw;
      end
      if (done_c[w]) begin
        vwr_c = vw;
        rec_c = ret_vector(r, ow_pkt[w].ctrl.rd, m);
        rec_c.mw_lanes = ow_lanes[w];
        rec_c.mw_addr = ow_addr[w];
        for (int l = 0; l < 32; l++)
          rec_c.mw_val[l] = ow_lanes[w][l]
              ? atom_apply(atom_kind(ow_pkt[w].ctrl.op), ow_acc[w][l], ow_data[w][l]) : 32'd0;
      end
    end
  end

  // ---------------------------------------------------------------- state
  always_ff @(posedge clk) begin
    if (!rst_n || clr) begin
      busy <= 1'b0;
      pw_cnt <= '0;
      done_b <= '0;
      done_c <= '0;
      if (clr) sv <= '0;
    end else begin
      // sequencing
      if (in.valid) begin
        cur <= in;
        cnt <= 8'd1;
        busy <= 1'b1;                        // at least the P address passes
      end else if (busy) begin
        cnt <= cnt + 8'd1;
        if (cnt >= 8'(P - 1) && (cnt == occ - 8'd1 || (ana && af))) busy <= 1'b0;
      end
      if (arriving) begin
        ba <= va;
        bb <= vb;
        bd <= vd;
      end
      if (ana) ci <= ai;
      // shared memory: valid bits of the words read and written, the lanes' results
      for (int b = 0; b < 32; b++) begin
        if (b_re[b]) b_vq[b] <= sv[{b_raddr[b], 5'(b)}];
        if (b_we[b]) sv[{b_waddr[b], 5'(b)}] <= 1'b1;
      end
      if (s_round && xl) for (int l = 0; l < 32; l++) if (lr[l]) sacc[l] <= rdl[l];
      a_prev <= a_step;
      if (a_step) begin
        a_prev_word <= xi.addr[a_lane][13:2];
        a_prev_new <= a_new;
        sacc[a_lane] <= a_old;
        snew[a_lane] <= a_new;
      end
      // post-occupancy record
      if (pw_cnt != 2'd0) pw_cnt <= pw_cnt - 2'd1;
      if (pw_set != 2'd0) begin
        pw_rec <= pw_next;
        pw_vwr <= pw_wnext;
        pw_cnt <= pw_set;
      end
      // per-warp slots: allocate with the analysis, fill from responses, complete one cycle
      // after the last response
      done_b <= '0;
      done_c <= '0;
      if (ana && ok && !xs && !xst && ai.n_seg != 6'd0) begin
        ow_pkt[WB'(x.warp)] <= x;
        ow_vd[WB'(x.warp)] <= vd;
        ow_lanes[WB'(x.warp)] <= ai.lanes;
        ow_seg[WB'(x.warp)] <= ai.seg;
        ow_addr[WB'(x.warp)] <= ai.addr;
        ow_data[WB'(x.warp)] <= ai.data;
        ow_pend[WB'(x.warp)] <= ai.n_seg;
      end
      if (rd_valid) begin
        for (int l = 0; l < 32; l++)
          if (ow_lanes[WB'(rd_warp)][l] && ow_seg[WB'(rd_warp)][l] == rd_seg)
            ow_acc[WB'(rd_warp)][l] <= rd_data[ow_addr[WB'(rd_warp)][l][6:2]];
        ow_pend[WB'(rd_warp)] <= ow_pend[WB'(rd_warp)] - 6'd1;
        if (ow_pend[WB'(rd_warp)] == 6'd1) done_b[WB'(rd_warp)] <= 1'b1;
      end
      if (at_valid) begin
        for (int l = 0; l < 32; l++)
          if (ow_lanes[WB'(at_warp)][l] && ow_seg[WB'(at_warp)][l] == at_seg)
            ow_acc[WB'(at_warp)][l] <= at_old[l];
        ow_pend[WB'(at_warp)] <= ow_pend[WB'(at_warp)] - 6'd1;
        if (ow_pend[WB'(at_warp)] == 6'd1) done_c[WB'(at_warp)] <= 1'b1;
      end
    end
  end
endmodule
