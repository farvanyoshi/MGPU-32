// mgpu_mem_arbiter: memory request ordering and response routing (docs/uarch/top.md 4).
//
// Every core has two request ports, LSU and I-fetch, which issue at most one segment
// request per cycle each and never wait. Each port has a queue; every entry is stamped with
// the cycle its request was made. Whenever the memory side is ready, the arbiter sends the
// request that comes first in (request cycle, core, LSU before I-fetch) order, which is the
// order of the timing contract (core.md 11.4). A request made in a cycle in which its queue
// is empty and memory is ready goes through in that same cycle, so it is accepted at
// a = max(r, free) with no added latency.
//
// Responses come back at segment level from mgpu_atomic_unit and are routed by tag to the
// core's I-fetch response, LSU READ data, ATOMIC old values or WRITE acknowledgement inputs.
//
// The queues must never overflow (the cores cannot be held back without changing the
// timing contract); `overflow` is sticky and checked by the testbenches.
module mgpu_mem_arbiter
  import mgpu_core_pkg::*;
#(
    parameter int C  = 4,
    parameter int QL = 64,   // LSU queue depth per core
    parameter int QI = 8,    // I-fetch queue depth per core (one fill per warp slot at most)
    localparam int LB = $clog2(QL),
    localparam int IB = $clog2(QI)
) (
    input  logic                       clk,
    input  logic                       rst_n,
    // core side
    input  logic [C-1:0]               if_req_valid,
    input  logic [C-1:0][31:0]         if_req_addr,
    input  logic [C-1:0][4:0]          if_req_id,
    output logic [C-1:0]               if_resp_valid,
    output logic [C-1:0][4:0]          if_resp_id,
    output logic [C-1:0][31:0][31:0]   if_resp_data,
    input  mreq_t [C-1:0]              lsu_req,
    output logic [C-1:0]               rd_valid,
    output logic [C-1:0][4:0]          rd_warp,
    output logic [C-1:0][4:0]          rd_seg,
    output logic [C-1:0][31:0][31:0]   rd_data,
    output logic [C-1:0]               at_valid,
    output logic [C-1:0][4:0]          at_warp,
    output logic [C-1:0][4:0]          at_seg,
    output logic [C-1:0][31:0][31:0]   at_old,
    output logic [C-1:0]               ack_valid,
    output logic [C-1:0][4:0]          ack_warp,
    // memory side (segment level)
    output logic                       sreq_valid,
    input  logic                       sreq_ready,
    output sreq_t                      sreq,
    input  logic                       srd_valid,
    input  logic [15:0]                srd_tag,
    input  logic [31:0][31:0]          srd_data,
    input  logic                       sat_valid,
    input  logic [15:0]                sat_tag,
    input  logic [31:0][31:0]          sat_old,
    input  logic                       sack_valid,
    input  logic [15:0]                sack_tag,
    output logic                       overflow
);
  logic [31:0] now;

  // ---------------------------------------------------------------- queues
  typedef struct packed {
    logic [31:0] r;
    sreq_t       q;
  } entry_t;

  entry_t [C-1:0][QL-1:0] lq;
  entry_t [C-1:0][QI-1:0] iq;
  logic [C-1:0][LB:0]     lcnt;
  logic [C-1:0][IB:0]     icnt;
  logic [C-1:0][LB-1:0]   lhead;
  logic [C-1:0][IB-1:0]   ihead;

  /* verilator lint_off UNUSEDSIGNAL */  // `valid` is the candidate's, not part of the request
  function automatic sreq_t from_lsu(input mreq_t m, input int c);
    sreq_t s;
    s.kind = m.kind;
    s.tag = mem_tag(4'(c), 1'b0, m.warp, m.seg);
    s.addr = m.addr;
    s.wmask = m.wmask;
    s.wdata = m.wdata;
    s.lanes = m.lanes;
    s.off = m.off;
    s.opnd = m.opnd;
    s.aop = m.aop;
    return s;
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  function automatic sreq_t from_ifetch(input logic [31:0] addr, input logic [4:0] id,
                                        input logic [3:0] c);
    sreq_t s;
    s = '0;
    s.kind = MK_READ;
    s.tag = mem_tag(c, 1'b1, id, 5'd0);
    s.addr = addr;
    return s;
  endfunction

  // candidates: the head of each queue, or this cycle's request if the queue is empty
  logic [2*C-1:0]        cv;
  entry_t [2*C-1:0]      ce;
  always_comb begin
    for (int c = 0; c < C; c++) begin
      // index 2c: LSU, 2c + 1: I-fetch (LSU first within a core)
      if (lcnt[c] != '0) begin
        cv[2 * c] = 1'b1;
        ce[2 * c] = lq[c][lhead[c]];
      end else begin
        cv[2 * c] = lsu_req[c].valid;
        ce[2 * c] = '{r: now, q: from_lsu(lsu_req[c], c)};
      end
      if (icnt[c] != '0) begin
        cv[2 * c + 1] = 1'b1;
        ce[2 * c + 1] = iq[c][ihead[c]];
      end else begin
        cv[2 * c + 1] = if_req_valid[c];
        ce[2 * c + 1] = '{r: now, q: from_ifetch(if_req_addr[c], if_req_id[c], 4'(c))};
      end
    end
  end

  // the first candidate in (request cycle, core, port) order
  logic g_any;
  int   g;
  always_comb begin
    g_any = 1'b0;
    g = 0;
    for (int k = 0; k < 2 * C; k++) begin
      if (cv[k] && (!g_any || ce[k].r < ce[g].r)) begin
        g_any = 1'b1;
        g = k;
      end
    end
  end
  assign sreq_valid = g_any;
  assign sreq = ce[g].q;

  logic [2*C-1:0] taken;
  always_comb begin
    taken = '0;
    if (g_any && sreq_ready) taken[g] = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      now <= '0;
      lcnt <= '0;
      icnt <= '0;
      lhead <= '0;
      ihead <= '0;
      overflow <= 1'b0;
    end else begin
      now <= now + 32'd1;
      for (int c = 0; c < C; c++) begin
        // LSU queue: pop the head if taken; push this cycle's request unless it went through
        logic lpop, lpush, ipop, ipush;
        lpop = taken[2 * c] && lcnt[c] != '0;
        lpush = lsu_req[c].valid && !(taken[2 * c] && lcnt[c] == '0);
        ipop = taken[2 * c + 1] && icnt[c] != '0;
        ipush = if_req_valid[c] && !(taken[2 * c + 1] && icnt[c] == '0);
        if (lpush) begin
          lq[c][LB'(int'(lhead[c]) + int'(lcnt[c]))] <= '{r: now, q: from_lsu(lsu_req[c], c)};
          if (int'(lcnt[c]) - int'(lpop) == QL) overflow <= 1'b1;
        end
        if (ipush) begin
          iq[c][IB'(int'(ihead[c]) + int'(icnt[c]))] <= '{r: now, q: from_ifetch(if_req_addr[c], if_req_id[c], 4'(c))};
          if (int'(icnt[c]) - int'(ipop) == QI) overflow <= 1'b1;
        end
        if (lpop) lhead[c] <= lhead[c] + 1'b1;
        if (ipop) ihead[c] <= ihead[c] + 1'b1;
        lcnt[c] <= lcnt[c] + (LB + 1)'(lpush) - (LB + 1)'(lpop);
        icnt[c] <= icnt[c] + (IB + 1)'(ipush) - (IB + 1)'(ipop);
      end
    end
  end

  // ---------------------------------------------------------------- response routing
  always_comb begin
    for (int c = 0; c < C; c++) begin
      if_resp_valid[c] = srd_valid && int'(srd_tag[15:12]) == c && srd_tag[11];
      if_resp_id[c] = srd_tag[10:6];
      if_resp_data[c] = srd_data;
      rd_valid[c] = srd_valid && int'(srd_tag[15:12]) == c && !srd_tag[11];
      rd_warp[c] = srd_tag[10:6];
      rd_seg[c] = srd_tag[5:1];
      rd_data[c] = srd_data;
      at_valid[c] = sat_valid && int'(sat_tag[15:12]) == c;
      at_warp[c] = sat_tag[10:6];
      at_seg[c] = sat_tag[5:1];
      at_old[c] = sat_old;
      ack_valid[c] = sack_valid && int'(sack_tag[15:12]) == c;
      ack_warp[c] = sack_tag[10:6];
    end
  end

  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{srd_tag[0], sat_tag[11], sat_tag[0], sack_tag[11], sack_tag[5:0]};
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
