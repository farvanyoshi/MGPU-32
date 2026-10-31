// mgpu_core: one MGPU-32 core (docs/uarch/core.md). Runs one block of up to W warps.
//
// Pipeline, for an instruction issued at cycle t (core.md 11.2):
//   t      schedule: round-robin choice among ready warps whose unit is free at t + 2;
//          operands are read (all 32 lanes) and the issue packet is formed
//   t + 1  packet register
//   t + 2  the packet enters its unit (SALU, VALU, FPU, SFU; LSU in a later stage)
//   c      the unit presents the retire record; registers are written at the end of c
//          and the warp may issue again at c + 1
//
// Faults are detected at issue. The core stops issuing, lets every instruction already in
// flight retire, and then reports the fault record (no earlier than t + 3).
//
// The retire port carries one record per completed instruction on NRET channels (one per
// writer, mgpu_core_pkg). A testbench consumes a cycle's records in channel order.
module mgpu_core
  import mgpu_isa_pkg::*;
  import mgpu_core_pkg::*;
#(
    parameter int W     = 8,
    parameter int L     = 8,
    parameter int S     = 2,
    parameter int LINES = 16,
    localparam int P    = 32 / L,
    localparam int WB   = (W > 1) ? $clog2(W) : 1
) (
    input  logic                    clk,
    input  logic                    rst_n,
    // dispatch (interfaces.md 3)
    input  logic                    start,        // accepted when idle or block_done
    input  logic                    new_launch,   // with start: invalidate the I-cache
    input  logic [31:0]             entry,
    input  logic [31:0]             block_id,
    input  logic [31:0]             block_dim,
    input  logic [31:0]             grid_dim,
    input  logic [5:0]              num_warps,
    input  logic [7:0][31:0]        args,
    input  logic [31:0]             mem_last,     // MEM_SIZE - 4
    output logic                    idle,
    output logic                    block_done,   // first idle cycle after the block completes
    // I-fetch memory port
    output logic                    if_req_valid,
    output logic [31:0]             if_req_addr,
    output logic [4:0]              if_req_id,
    input  logic                    if_resp_valid,
    input  logic [4:0]              if_resp_id,
    input  logic [31:0][31:0]       if_resp_data,
    // retire port
    output logic [NRET-1:0]         ret_valid,
    output retire_t [NRET-1:0]      ret
);
  localparam int U_SALU = 0, U_VALU = 1, U_FPU = 2, U_SFU = 3, U_LSU = 4;

  // ------------------------------------------------------------------ block state
  logic              active, halt;
  logic [31:0]       bclk;              // CLOCK: cycles since the block started
  logic [5:0]        boot;              // warp w starts fetching when boot > w
  logic [31:0]       r_block_id, r_block_dim, r_grid_dim;
  logic [5:0]        r_num_warps;
  logic              accept;
  assign accept = start && (!active || block_done);

  // ------------------------------------------------------------------ warp table
  logic [W-1:0]       live, exited, in_flight, fetch_pend;
  logic [W-1:0][31:0] pc;

  // ------------------------------------------------------------------ fetch
  logic [W-1:0]       f_req, f_grant, consume, ibuf_valid, ibuf_ff;
  logic [W-1:0][31:0] f_req_pc, ibuf_word;
  ctrl_t [W-1:0]      ibuf_ctrl;
  logic               br_valid;
  logic [31:0]        br_pc;
  logic [W-1:0]       br_now;
  logic               ev_hit, ev_miss;

  always_comb begin
    for (int w = 0; w < W; w++) begin
      br_now[w] = br_valid && ret[R_SALU].warp == 5'(w);
      f_req[w] = active && live[w] && !exited[w]
                 && ((fetch_pend[w] && int'(boot) > w) || br_now[w]);
      f_req_pc[w] = br_now[w] ? br_pc : pc[w];
    end
  end

  mgpu_fetch #(.W(W), .LINES(LINES)) u_fetch (
      .clk, .rst_n, .inv(accept && new_launch), .clr(accept), .mem_last,
      .req(f_req), .req_pc(f_req_pc), .grant(f_grant),
      .consume, .ibuf_valid, .ibuf_word, .ibuf_ctrl, .ibuf_ff,
      .mreq_valid(if_req_valid), .mreq_addr(if_req_addr), .mreq_id(if_req_id),
      .mresp_valid(if_resp_valid), .mresp_id(if_resp_id), .mresp_data(if_resp_data),
      .ev_hit, .ev_miss);

  // ------------------------------------------------------------------ scheduler
  logic [4:0][7:0] rem;                 // cycles until free_at(U), saturating at 0
  logic [W-1:0]    ready, issuable;
  logic [W-1:0][2:0] wunit;
  logic            g_any;
  int              g;
  logic [WB-1:0]   rr;

  function automatic logic [2:0] unit_slot(input unit_e unit, input logic bad);
    if (bad) return 3'(U_SALU);
    unique case (unit)
      UNIT_VALU: return 3'(U_VALU);
      UNIT_FPU:  return 3'(U_FPU);
      UNIT_SFU:  return 3'(U_SFU);
      UNIT_LSU:  return 3'(U_LSU);
      default:   return 3'(U_SALU);
    endcase
  endfunction

  always_comb begin
    for (int w = 0; w < W; w++) begin
      wunit[w] = unit_slot(ibuf_ctrl[w].unit, ibuf_ff[w] || ibuf_ctrl[w].illegal);
      ready[w] = active && !halt && live[w] && !exited[w] && !in_flight[w] && ibuf_valid[w];
      issuable[w] = ready[w] && rem[wunit[w]] <= 8'd2;
    end
    g_any = 1'b0;
    g = 0;
    for (int k = 0; k < W; k++) begin
      int i;
      i = (int'(rr) + k) % W;
      if (!g_any && issuable[i]) begin
        g_any = 1'b1;
        g = i;
      end
    end
  end

  // ------------------------------------------------------------------ operand read (issue)
  ctrl_t           c0;
  logic [4:0][WB-1:0] s_rwarp;
  logic [4:0][4:0]    s_raddr;
  logic [4:0][31:0]   s_rdata;
  logic [3:0][WB-1:0] v_rwarp;
  logic [3:0][4:0]    v_raddr;
  logic [3:0][31:0][31:0] v_rdata;
  assign c0 = ibuf_ctrl[g];
  assign s_rwarp = {5{WB'(g)}};
  assign s_raddr = {5'd31, c0.rd, c0.rc, c0.rb, c0.ra};
  assign v_rwarp = {4{WB'(g)}};
  assign v_raddr = {c0.rd, c0.rc, c0.rb, c0.ra};

  // special registers ([SR-3]: CLOCK at the issue cycle)
  logic [31:0]       sr_uniform;
  logic [31:0][31:0] sr_vec;
  always_comb begin
    unique case (c0.ra)
      SR_BLOCK_ID:  sr_uniform = r_block_id;
      SR_BLOCK_DIM: sr_uniform = r_block_dim;
      SR_GRID_DIM:  sr_uniform = r_grid_dim;
      SR_WARP_ID:   sr_uniform = 32'(g);
      SR_NUM_WARPS: sr_uniform = 32'(r_num_warps);
      default:      sr_uniform = bclk;           // SR_CLOCK
    endcase
    for (int i = 0; i < 32; i++) begin
      unique case (c0.ra)
        SR_TID:     sr_vec[i] = 32'(32 * g + i);
        SR_LANE_ID: sr_vec[i] = 32'(i);
        SR_GTID:    sr_vec[i] = r_block_id * r_block_dim + 32'(32 * g + i);
        default:    sr_vec[i] = sr_uniform;
      endcase
    end
  end

  ipkt_t       p0;
  logic        f0;               // faults at issue
  logic [7:0]  f0_code;
  logic [4:0]  f0_lane;
  logic [31:0] f0_info;
  logic [2:0]  u0;
  always_comb begin
    logic [31:0] bs;
    bs = c0.b_imm ? c0.imm : s_rdata[1];
    p0 = '0;
    p0.valid = g_any;
    p0.warp = 5'(g);
    p0.pc = pc[g];
    p0.word = ibuf_word[g];
    p0.clock = bclk;
    p0.ctrl = c0;
    p0.exec = s_rdata[4];
    p0.sa = s_rdata[0];
    p0.sb = bs;
    p0.sc = s_rdata[2];
    p0.sd = s_rdata[3];
    p0.va = v_rdata[0];
    p0.vb = (c0.sb || c0.b_imm) ? {32{bs}} : v_rdata[1];
    p0.vc = v_rdata[2];
    p0.vd = v_rdata[3];
    if (c0.op == OP_S_GETSR) begin       // executes as S_ADD sd, sr, 0
      p0.ctrl.op = OP_S_ADD;
      p0.sa = sr_uniform;
      p0.sb = '0;
    end
    if (c0.op == OP_V_GETSR) begin       // executes as V_ADD vd, sr, 0 (VALU timing)
      p0.ctrl.op = OP_V_ADD;
      p0.va = sr_vec;
      p0.vb = '0;
    end
    u0 = wunit[g];
    f0 = ibuf_ff[g] || c0.illegal;
    f0_code = ibuf_ff[g] ? E_RANGE : E_ILLEGAL;
    f0_lane = '0;
    f0_info = ibuf_ff[g] ? pc[g] : ibuf_word[g];
  end

  // occupancy of the issued instruction (core.md 7)
  logic [7:0] occ0;
  always_comb begin
    unique case (u0)
      3'(U_VALU): occ0 = (c0.op == OP_V_SHFL || c0.op == OP_V_READLANE) ? 8'(2 * P) : 8'(P);
      3'(U_FPU):  occ0 = 8'(P);
      3'(U_SFU):  occ0 = 8'(32 / S);
      default:    occ0 = 8'd1;
    endcase
  end

  // ------------------------------------------------------------------ packet pipeline
  typedef struct packed {
    ipkt_t       p;
    logic [2:0]  unit;
    logic        fault;
    logic [7:0]  fcode;
    logic [4:0]  flane;
    logic [31:0] finfo;
  } stage_t;
  stage_t st1, st2;
  always_ff @(posedge clk) begin
    if (!rst_n || accept) begin
      st1 <= '0;
      st2 <= '0;
    end else begin
      st1 <= '{p: p0, unit: u0, fault: f0, fcode: f0_code, flane: f0_lane, finfo: f0_info};
      st2 <= st1;
    end
  end

  /* verilator lint_off UNUSEDSIGNAL */  // the fault fields are not needed here
  function automatic ipkt_t to_unit(input stage_t s, input int u);
    ipkt_t q;
    q = s.p;
    q.valid = s.p.valid && !s.fault && int'(s.unit) == u;
    return q;
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  // ------------------------------------------------------------------ execution units
  retire_t r_salu, r_valu, r_fpu, r_sfu;
  mgpu_salu u_salu (.clk, .rst_n, .in(to_unit(st2, U_SALU)), .out(r_salu), .br_valid, .br_pc);
  mgpu_vexec #(.KIND(0), .LANES(L), .LAT(2)) u_valu (.clk, .rst_n, .in(to_unit(st2, U_VALU)), .out(r_valu));
  mgpu_vexec #(.KIND(1), .LANES(L), .LAT(4)) u_fpu (.clk, .rst_n, .in(to_unit(st2, U_FPU)), .out(r_fpu));
  mgpu_vexec #(.KIND(2), .LANES(S), .LAT(8)) u_sfu (.clk, .rst_n, .in(to_unit(st2, U_SFU)), .out(r_sfu));

  // ------------------------------------------------------------------ faults
  retire_t    fq;                 // the fault waiting for in-flight instructions to drain
  logic [4:0] fq_warp;
  logic       fq_emit;
  assign fq_emit = fq.valid && (in_flight & ~(W'(1) << fq_warp)) == '0;

  always_ff @(posedge clk) begin
    if (!rst_n || accept) begin
      fq <= '0;
    end else if (st2.p.valid && st2.fault) begin
      fq <= ret_from(st2.p);
      fq.event_ <= EV_FAULT;
      fq.fault_code <= st2.fcode;
      fq.fault_lane <= st2.flane;
      fq.fault_info <= st2.finfo;
      fq_warp <= st2.p.warp;
    end else if (fq_emit) begin
      fq <= '0;
    end
  end

  // ------------------------------------------------------------------ retire port
  always_comb begin
    for (int i = 0; i < NRET; i++) ret[i] = '0;
    ret[R_SALU] = r_salu;
    ret[R_VALU] = r_valu;
    ret[R_FPU] = r_fpu;
    ret[R_SFU] = r_sfu;
    if (fq_emit) ret[R_FAULT] = fq;
    for (int i = 0; i < NRET; i++) ret_valid[i] = ret[i].valid;
  end

  // ------------------------------------------------------------------ writeback
  logic [2:0]            s_we;
  logic [2:0][WB-1:0]    s_wwarp;
  logic [2:0][4:0]       s_waddr;
  logic [2:0][31:0]      s_wdata;
  logic [4:0]            v_we;
  logic [4:0][WB-1:0]    v_wwarp;
  logic [4:0][4:0]       v_waddr;
  logic [4:0][31:0][31:0] v_wdata;
  always_comb begin
    s_we = '0;
    s_wwarp = '0;
    s_waddr = '0;
    s_wdata = '0;
    v_we = '0;
    v_wwarp = '0;
    v_waddr = '0;
    v_wdata = '0;
    // scalar ports: SALU, VALU (V_CMP, V_READLANE), LSU response path (S_LW)
    s_we[0] = r_salu.valid && r_salu.sw_valid;
    s_wwarp[0] = WB'(r_salu.warp);
    s_waddr[0] = r_salu.sw_reg;
    s_wdata[0] = r_salu.sw_val;
    s_we[1] = r_valu.valid && r_valu.sw_valid;
    s_wwarp[1] = WB'(r_valu.warp);
    s_waddr[1] = r_valu.sw_reg;
    s_wdata[1] = r_valu.sw_val;
    // vector ports: VALU, FPU, SFU, LSU issue path, LSU response path
    v_we[0] = r_valu.valid && r_valu.vw_valid;
    v_wwarp[0] = WB'(r_valu.warp);
    v_waddr[0] = r_valu.vw_reg;
    v_wdata[0] = r_valu.vw_val;
    v_we[1] = r_fpu.valid && r_fpu.vw_valid;
    v_wwarp[1] = WB'(r_fpu.warp);
    v_waddr[1] = r_fpu.vw_reg;
    v_wdata[1] = r_fpu.vw_val;
    v_we[2] = r_sfu.valid && r_sfu.vw_valid;
    v_wwarp[2] = WB'(r_sfu.warp);
    v_waddr[2] = r_sfu.vw_reg;
    v_wdata[2] = r_sfu.vw_val;
  end

  logic [W-1:0][31:0] init_exec;
  always_comb begin
    for (int w = 0; w < W; w++) begin
      int n;
      n = int'(block_dim) - 32 * w;
      init_exec[w] = n >= 32 ? 32'hFFFF_FFFF : n <= 0 ? 32'd0 : (32'd1 << n) - 32'd1;
    end
  end

  mgpu_srf #(.W(W), .NR(5), .NW(3)) u_srf (
      .clk, .rwarp(s_rwarp), .raddr(s_raddr), .rdata(s_rdata),
      .we(s_we), .wwarp(s_wwarp), .waddr(s_waddr), .wdata(s_wdata),
      .init(accept), .init_args(args), .init_exec);
  mgpu_vrf #(.W(W), .NR(4), .NW(5)) u_vrf (
      .clk, .rwarp(v_rwarp), .raddr(v_raddr), .rdata(v_rdata),
      .we(v_we), .wwarp(v_wwarp), .waddr(v_waddr), .wdata(v_wdata), .init(accept));

  // ------------------------------------------------------------------ control state
  logic all_exited;
  assign all_exited = &(exited | ~live);
  assign block_done = active && all_exited;
  assign idle = !active;

  always_comb begin
    consume = '0;
    if (g_any) consume[g] = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      active <= 1'b0;
      halt <= 1'b0;
      live <= '0;
      exited <= '0;
      in_flight <= '0;
      fetch_pend <= '0;
      rem <= '0;
      rr <= '0;
      boot <= '0;
    end else if (accept) begin
      active <= 1'b1;
      halt <= 1'b0;
      bclk <= 32'd1;
      boot <= 6'd1;
      r_block_id <= block_id;
      r_block_dim <= block_dim;
      r_grid_dim <= grid_dim;
      r_num_warps <= num_warps;
      for (int w = 0; w < W; w++) begin
        live[w] <= w < int'(num_warps);
        pc[w] <= entry;
      end
      exited <= '0;
      in_flight <= '0;
      fetch_pend <= '1;
      rem <= '0;
    end else begin
      bclk <= bclk + 32'd1;
      if (int'(boot) <= W) boot <= boot + 6'd1;
      if (block_done) active <= 1'b0;
      for (int u = 0; u < 5; u++) if (rem[u] != 0) rem[u] <= rem[u] - 8'd1;

      // fetch grants and branch fetch requests
      for (int w = 0; w < W; w++) begin
        if (f_grant[w]) fetch_pend[w] <= 1'b0;
        else if (br_now[w]) fetch_pend[w] <= 1'b1;
      end
      if (br_valid) pc[WB'(ret[R_SALU].warp)] <= br_pc;

      // retire
      for (int i = 0; i < NRET; i++) begin
        if (ret[i].valid && i != R_FAULT) begin
          in_flight[WB'(ret[i].warp)] <= 1'b0;
          if (ret[i].event_ == EV_EXIT) exited[WB'(ret[i].warp)] <= 1'b1;
        end
      end
      if (fq_emit) begin
        active <= 1'b0;
        halt <= 1'b1;
      end

      // issue
      if (g_any) begin
        rr <= WB'((g + 1) % W);
        in_flight[g] <= 1'b1;
        rem[u0] <= occ0 + 8'd1;
        if (f0) begin
          halt <= 1'b1;
        end else if (!c0.branch && c0.op != OP_EXIT) begin
          pc[g] <= pc[g] + 32'd4;
          fetch_pend[g] <= 1'b1;
        end
      end
    end
  end

  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{ev_hit, ev_miss};
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
