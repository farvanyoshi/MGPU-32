// mgpu_lsu: load/store unit (docs/uarch/core.md 8, 11.4).
//
// An instruction enters at e = t + 2 together with its access analysis (mgpu_agen, done at
// issue; faulting accesses never get here). With i = cycle - e:
//   i in [0, P)            address passes
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
    parameter int P = 4,
    localparam int WB = (W > 1) ? $clog2(W) : 1
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              clr,          // block start
    input  ipkt_t             in,
    input  lsu_info_t         in_info,
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
  lsu_info_t ci, xi;
  logic [7:0] cnt, xc, occ;
  logic       busy, xv, xs, xl, xst, xa;
  assign x   = in.valid ? in : cur;
  assign xi  = in.valid ? in_info : ci;
  assign xc  = in.valid ? 8'd0 : cnt;
  assign xv  = in.valid || busy;
  assign xs  = x.ctrl.mem_shared;
  assign xl  = is_load(x.ctrl.op);
  assign xst = is_store(x.ctrl.op);
  assign xa  = !xl && !xst;
  assign occ = 8'(P) + (!xs ? 8'(xi.n_seg) : xa ? 8'(xi.n_act) : 8'(xi.d));

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
    if (xv && !xs && xst && xc == 8'(P - 1) + 8'(xi.n_seg)) begin
      rec_s = ret_from(x);
      rec_s.mw_lanes = xi.lanes;
      rec_s.mw_addr = xi.addr;
      rec_s.mw_val = xi.data;
    end
  end

  // ---------------------------------------------------------------- shared memory
  logic [4095:0][31:0] sh;
  logic [31:0][31:0]   sacc, snew;
  logic                s_round, s_last, a_step, a_last;
  logic [4:0]          j_round;
  logic [31:0]         lr;            // lanes of this round
  logic [31:0][31:0]   rdl;           // their read data
  int                  a_lane;        // shared atomic: lane of this step
  logic [31:0]         a_old, a_new;
  assign s_round = xv && xs && !xa && xc >= 8'(P) && xc < 8'(P) + 8'(xi.d);
  assign s_last  = s_round && xc == 8'(P) + 8'(xi.d) - 8'd1;
  assign j_round = 5'(xc - 8'(P));
  assign a_step  = xv && xs && xa && xc >= 8'(P) && xc < 8'(P) + 8'(xi.n_act);
  assign a_last  = a_step && xc == 8'(P) + 8'(xi.n_act) - 8'd1;
  always_comb begin
    int n;
    for (int l = 0; l < 32; l++) begin
      lr[l] = xi.lanes[l] && xi.round[l] == j_round;
      rdl[l] = sh[xi.addr[l][13:2]];
    end
    a_lane = 0;
    n = 0;
    for (int l = 0; l < 32; l++) begin
      if (xi.lanes[l]) begin
        if (8'(n) == xc - 8'(P)) a_lane = l;
        n++;
      end
    end
    a_old = sh[xi.addr[a_lane][13:2]];
    a_new = atom_apply(atom_kind(x.ctrl.op), a_old, xi.data[a_lane]);
  end

  // ---------------------------------------------------------------- post-occupancy records (LSU_A)
  retire_t    pw_rec, pw_next;
  logic [1:0] pw_cnt, pw_set;
  always_comb begin
    logic [31:0][31:0] m;
    pw_next = ret_from(x);
    pw_set = 2'd0;
    m = x.vd;
    if (s_last) begin                        // shared load / store
      pw_set = 2'd1;
      if (xl) begin
        for (int l = 0; l < 32; l++) if (xi.lanes[l]) m[l] = lr[l] ? rdl[l] : sacc[l];
        pw_next = ret_vector(pw_next, x.ctrl.rd, m);
      end else begin
        pw_next.mw_lanes = xi.lanes;
        pw_next.mw_shared = 1'b1;
        pw_next.mw_addr = xi.addr;
        pw_next.mw_val = xi.data;
      end
    end else if (xv && xs && xa && xi.n_act == 6'd0 && xc == 8'(P - 1)) begin
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
    end else if (xv && !xs && !xst && xi.n_seg == 6'd0 && xc == 8'(P - 1)) begin
      pw_set = 2'd2;                         // global load / atomic, no active lane
      pw_next = ret_vector(pw_next, x.ctrl.rd, m);
    end
  end
  assign rec_a = pw_cnt == 2'd1 ? pw_rec : '0;

  // ---------------------------------------------------------------- per-warp load / atomic slots
  logic [W-1:0]             done_b, done_c;
  ipkt_t [W-1:0]            ow_pkt;
  logic [W-1:0][31:0]       ow_lanes;
  logic [W-1:0][31:0][4:0]  ow_seg;
  logic [W-1:0][31:0][31:0] ow_addr, ow_data, ow_acc;
  logic [W-1:0][5:0]        ow_pend;

  always_comb begin
    rec_b = '0;
    rec_c = '0;
    for (int w = 0; w < W; w++) begin
      logic [31:0][31:0] m;
      retire_t r;
      m = ow_pkt[w].vd;
      for (int l = 0; l < 32; l++) if (ow_lanes[w][l]) m[l] = ow_acc[w][l];
      r = ret_from(ow_pkt[w]);
      if (done_b[w]) begin
        rec_b = is_scalar_mem(ow_pkt[w].ctrl.op) ? ret_scalar(r, ow_pkt[w].ctrl.rd, ow_acc[w][0])
                                                 : ret_vector(r, ow_pkt[w].ctrl.rd, m);
      end
      if (done_c[w]) begin
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
      if (clr) for (int a = 0; a < 4096; a++) sh[a] <= POISON;
    end else begin
      // sequencing
      if (in.valid) begin
        cur <= in;
        ci <= in_info;
        cnt <= 8'd1;
        busy <= occ > 8'd1;
      end else if (busy) begin
        cnt <= cnt + 8'd1;
        if (cnt == occ - 8'd1) busy <= 1'b0;
      end
      // shared rounds (ascending lanes: the highest lane wins a store)
      if (s_round) begin
        for (int l = 0; l < 32; l++) begin
          if (lr[l]) begin
            if (xl) sacc[l] <= rdl[l];
            else sh[xi.addr[l][13:2]] <= xi.data[l];
          end
        end
      end
      if (a_step) begin
        sh[xi.addr[a_lane][13:2]] <= a_new;
        sacc[a_lane] <= a_old;
        snew[a_lane] <= a_new;
      end
      // post-occupancy record
      if (pw_cnt != 2'd0) pw_cnt <= pw_cnt - 2'd1;
      if (pw_set != 2'd0) begin
        pw_rec <= pw_next;
        pw_cnt <= pw_set;
      end
      // per-warp slots: allocate at entry, fill from responses, complete one cycle after
      // the last response
      done_b <= '0;
      done_c <= '0;
      if (in.valid && !in.ctrl.mem_shared && !is_store(in.ctrl.op) && in_info.n_seg != 6'd0) begin
        ow_pkt[WB'(in.warp)] <= in;
        ow_lanes[WB'(in.warp)] <= in_info.lanes;
        ow_seg[WB'(in.warp)] <= in_info.seg;
        ow_addr[WB'(in.warp)] <= in_info.addr;
        ow_data[WB'(in.warp)] <= in_info.data;
        ow_pend[WB'(in.warp)] <= in_info.n_seg;
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
