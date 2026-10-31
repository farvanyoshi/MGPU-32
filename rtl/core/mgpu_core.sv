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
    input  logic                    kill,         // abort the block at once (another core faulted)
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
    // faults across cores (top.md 6)
    input  logic                    halt_in,      // issue nothing this cycle (another core faulted)
    input  logic                    fault_go,     // every core has drained: the fault may be reported
    output logic                    fault_issue,  // a faulting instruction issues this cycle
    output logic                    busy_ops,     // instructions in flight (other than a fault)
    output logic                    mem_quiet,    // no memory request outstanding
    // I-fetch memory port
    output logic                    if_req_valid,
    output logic [31:0]             if_req_addr,
    output logic [4:0]              if_req_id,
    input  logic                    if_resp_valid,
    input  logic [4:0]              if_resp_id,
    input  logic [31:0][31:0]       if_resp_data,
    // LSU memory port (segment requests; READ data, ATOMIC old values and WRITE acks)
    output mreq_t                   lsu_req,
    input  logic                    rd_valid,
    input  logic [4:0]              rd_warp,
    input  logic [4:0]              rd_seg,
    input  logic [31:0][31:0]       rd_data,
    input  logic                    at_valid,
    input  logic [4:0]              at_warp,
    input  logic [4:0]              at_seg,
    input  logic [31:0][31:0]       at_old,
    input  logic                    ack_valid,
    input  logic [4:0]              ack_warp,
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
  assign accept = start && !kill && (!active || block_done);

  // ------------------------------------------------------------------ warp table
  logic [W-1:0]       live, exited, in_flight, fetch_pend;
  logic [W-1:0][31:0] pc;
  logic [W-1:0][5:0]  stores_out;      // global store segments not yet acknowledged
  logic [W-1:0]       wait_bar;        // BAR completed, waiting for release
  logic [W-1:0]       wait_fence;      // FENCE completed, waiting for its stores

  // ------------------------------------------------------------------ barrier (core.md 9)
  // A warp arrives once its BAR has completed and its stores are acknowledged. The barrier
  // releases in the cycle in which every live, non-exited warp has arrived (the last BAR
  // or an EXIT completed it); released warps may issue in that same cycle. The barrier PC
  // is recorded when the first BAR issues, and a BAR at another PC faults at issue.
  logic        bar_active;
  logic [31:0] bar_pc;
  logic        release_now;
  always_comb begin
    logic all_arrived;
    all_arrived = 1'b1;
    for (int w = 0; w < W; w++)
      if (live[w] && !exited[w] && !(wait_bar[w] && stores_out[w] == 6'd0)) all_arrived = 1'b0;
    release_now = bar_active && |wait_bar && all_arrived;
  end

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
      .ev_hit, .ev_miss, .fills_pending);

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
      ready[w] = active && !halt && !halt_in && live[w] && !exited[w] && !in_flight[w] && ibuf_valid[w]
                 && !(wait_bar[w] && !release_now) && !(wait_fence[w] && stores_out[w] != 6'd0);
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

  lsu_info_t   li0;
  logic        af;               // LSU address fault
  logic [7:0]  af_code;
  logic [4:0]  af_lane;
  logic [31:0] af_info;
  mgpu_agen u_agen (.ctrl(c0), .exec(s_rdata[4]), .sa(s_rdata[0]), .sd(s_rdata[3]),
                    .va(v_rdata[0]), .vb(p0.vb), .vd(v_rdata[3]), .mem_last,
                    .info(li0), .fault(af), .fcode(af_code), .flane(af_lane), .finfo(af_info));

  ipkt_t       p0;
  logic        bf;               // barrier PC mismatch ([BAR-2])
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
    bf = !ibuf_ff[g] && !c0.illegal && c0.op == OP_BAR && bar_active && !release_now
         && pc[g] != bar_pc;
    f0 = ibuf_ff[g] || c0.illegal || (u0 == 3'(U_LSU) && af) || bf;
    f0_code = ibuf_ff[g] ? E_RANGE : c0.illegal ? E_ILLEGAL : bf ? E_BARRIER : af_code;
    f0_lane = (ibuf_ff[g] || c0.illegal || bf) ? 5'd0 : af_lane;
    f0_info = ibuf_ff[g] ? pc[g] : c0.illegal ? ibuf_word[g] : bf ? bar_pc : af_info;
  end

  // occupancy of the issued instruction (core.md 7)
  logic [7:0] occ0;
  always_comb begin
    unique case (u0)
      3'(U_VALU): occ0 = (c0.op == OP_V_SHFL || c0.op == OP_V_READLANE) ? 8'(2 * P) : 8'(P);
      3'(U_FPU):  occ0 = 8'(P);
      3'(U_SFU):  occ0 = 8'(32 / S);
      3'(U_LSU):  occ0 = 8'(P) + (!c0.mem_shared ? 8'(li0.n_seg)
                                 : !is_load(c0.op) && !is_store(c0.op) ? 8'(li0.n_act) : 8'(li0.d));
      default:    occ0 = 8'd1;
    endcase
  end

  // ------------------------------------------------------------------ packet pipeline
  typedef struct packed {
    ipkt_t       p;
    lsu_info_t   li;
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
      st1 <= '{p: p0, li: li0, unit: u0, fault: f0, fcode: f0_code, flane: f0_lane, finfo: f0_info};
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
  retire_t r_salu, r_valu, r_fpu, r_sfu, r_lsu_s, r_lsu_a, r_lsu_b, r_lsu_c;
  logic    salu_fence;
  mgpu_salu u_salu (.clk, .rst_n, .in(to_unit(st2, U_SALU)), .out(r_salu), .br_valid, .br_pc,
                    .fence(salu_fence));
  mgpu_vexec #(.KIND(0), .LANES(L), .LAT(2)) u_valu (.clk, .rst_n, .in(to_unit(st2, U_VALU)), .out(r_valu));
  mgpu_vexec #(.KIND(1), .LANES(L), .LAT(4)) u_fpu (.clk, .rst_n, .in(to_unit(st2, U_FPU)), .out(r_fpu));
  mgpu_vexec #(.KIND(2), .LANES(S), .LAT(8)) u_sfu (.clk, .rst_n, .in(to_unit(st2, U_SFU)), .out(r_sfu));
  mgpu_lsu #(.W(W), .P(P)) u_lsu (
      .clk, .rst_n, .clr(accept), .in(to_unit(st2, U_LSU)), .in_info(st2.li), .req(lsu_req),
      .rd_valid, .rd_warp, .rd_seg, .rd_data, .at_valid, .at_warp, .at_seg, .at_old,
      .rec_s(r_lsu_s), .rec_a(r_lsu_a), .rec_b(r_lsu_b), .rec_c(r_lsu_c));

  // ------------------------------------------------------------------ faults
  retire_t    fq;                 // the fault waiting for in-flight instructions to drain
  logic [4:0] fq_warp;
  logic       fq_emit;
  logic [W-1:0] others;
  assign others = fq.valid ? in_flight & ~(W'(1) << fq_warp) : in_flight;
  assign busy_ops = |others;
  assign fq_emit = fq.valid && !busy_ops && fault_go;
  assign fault_issue = g_any && f0;

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
    ret[R_LSU_S] = r_lsu_s;
    ret[R_LSU_A] = r_lsu_a;
    ret[R_LSU_B] = r_lsu_b;
    ret[R_LSU_C] = r_lsu_c;
    if (fq_emit) ret[R_FAULT] = fq;
    for (int i = 0; i < NRET; i++) ret_valid[i] = ret[i].valid;
  end

  // ------------------------------------------------------------------ writeback
  logic [2:0]            s_we;
  logic [2:0][WB-1:0]    s_wwarp;
  logic [2:0][4:0]       s_waddr;
  logic [2:0][31:0]      s_wdata;
  logic [5:0]            v_we;
  logic [5:0][WB-1:0]    v_wwarp;
  logic [5:0][4:0]       v_waddr;
  logic [5:0][31:0][31:0] v_wdata;
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
    s_we[2] = r_lsu_b.valid && r_lsu_b.sw_valid;
    s_wwarp[2] = WB'(r_lsu_b.warp);
    s_waddr[2] = r_lsu_b.sw_reg;
    s_wdata[2] = r_lsu_b.sw_val;
    // vector ports: VALU, FPU, SFU, LSU (shared, loads, atomics)
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
    v_we[3] = r_lsu_a.valid && r_lsu_a.vw_valid;
    v_wwarp[3] = WB'(r_lsu_a.warp);
    v_waddr[3] = r_lsu_a.vw_reg;
    v_wdata[3] = r_lsu_a.vw_val;
    v_we[4] = r_lsu_b.valid && r_lsu_b.vw_valid;
    v_wwarp[4] = WB'(r_lsu_b.warp);
    v_waddr[4] = r_lsu_b.vw_reg;
    v_wdata[4] = r_lsu_b.vw_val;
    v_we[5] = r_lsu_c.valid && r_lsu_c.vw_valid;
    v_wwarp[5] = WB'(r_lsu_c.warp);
    v_waddr[5] = r_lsu_c.vw_reg;
    v_wdata[5] = r_lsu_c.vw_val;
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
  mgpu_vrf #(.W(W), .NR(4), .NW(6)) u_vrf (
      .clk, .rwarp(v_rwarp), .raddr(v_raddr), .rdata(v_rdata),
      .we(v_we), .wwarp(v_wwarp), .waddr(v_waddr), .wdata(v_wdata), .init(accept));

  // ------------------------------------------------------------------ control state
  logic all_exited;
  assign all_exited = &(exited | ~live);
  logic no_stores;
  always_comb begin
    no_stores = 1'b1;
    for (int w = 0; w < W; w++) if (stores_out[w] != 6'd0) no_stores = 1'b0;
  end
  assign block_done = active && all_exited && no_stores;
  assign idle = !active;
  logic fills_pending;
  always_comb begin
    mem_quiet = !fills_pending && !if_req_valid && !busy_ops;
    for (int w = 0; w < W; w++) if (stores_out[w] != 6'd0) mem_quiet = 1'b0;
  end

  always_comb begin
    consume = '0;
    if (g_any) consume[g] = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (!rst_n || kill) begin
      active <= 1'b0;
      halt <= 1'b0;
      live <= '0;
      wait_bar <= '0;
      wait_fence <= '0;
      bar_active <= 1'b0;
      exited <= '0;
      in_flight <= '0;
      fetch_pend <= '0;
      stores_out <= '0;
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
      if (new_launch) rr <= '0;          // every launch starts from fresh cores
      for (int w = 0; w < W; w++) begin
        live[w] <= w < int'(num_warps);
        pc[w] <= entry;
      end
      exited <= '0;
      in_flight <= '0;
      fetch_pend <= '1;
      wait_bar <= '0;
      wait_fence <= '0;
      bar_active <= 1'b0;
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
          if (ret[i].event_ == EV_BAR) wait_bar[WB'(ret[i].warp)] <= 1'b1;
        end
      end
      for (int w = 0; w < W; w++) if (wait_fence[w] && stores_out[w] == 6'd0) wait_fence[w] <= 1'b0;
      if (salu_fence) wait_fence[WB'(r_salu.warp)] <= 1'b1;
      if (release_now) begin
        wait_bar <= '0;
        bar_active <= 1'b0;
      end
      // outstanding global stores
      for (int w = 0; w < W; w++) begin
        stores_out[w] <= stores_out[w]
            + 6'(lsu_req.valid && lsu_req.kind == MK_WRITE && lsu_req.warp == 5'(w))
            - 6'(ack_valid && ack_warp == 5'(w));
      end
      if (fq_emit) begin                 // the faulting warp is finished too
        active <= 1'b0;
        halt <= 1'b1;
        in_flight[WB'(fq_warp)] <= 1'b0;
      end

      // issue
      if (g_any) begin
        rr <= WB'((g + 1) % W);
        in_flight[g] <= 1'b1;
        rem[u0] <= occ0 + 8'd1;
        if (!f0 && c0.op == OP_BAR) begin
          bar_active <= 1'b1;
          bar_pc <= pc[g];
        end
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
