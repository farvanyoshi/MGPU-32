// mgpu_agen: LSU access analysis at issue (docs/uarch/core.md 8). Combinational.
//
// Lane addresses (S[ra] + imm for S_LW/S_SW on lane 0 only; V[ra] + imm for vector
// accesses; V[ra] for atomics), the precise fault check ([MEM-3]: per active lane in
// ascending order, misaligned before out of range, lowest lane wins), and the numbers the
// timing contract needs at issue: distinct segments with each lane's segment index (global),
// the bank-conflict degree with each lane's bank round (shared), and the active lane count.
module mgpu_agen
  import mgpu_isa_pkg::*;
  import mgpu_core_pkg::*;
(
    /* verilator lint_off UNUSEDSIGNAL */
    input  ctrl_t             ctrl,      // op, imm and the memory space are used
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [31:0]       exec,
    input  logic [31:0]       sa,
    input  logic [31:0]       sd,
    input  logic [31:0][31:0] va,
    input  logic [31:0][31:0] vb,    // B per lane (atomic operand)
    input  logic [31:0][31:0] vd,    // store data
    input  logic [31:0]       mem_last,
    output lsu_info_t         info,
    output logic              fault,
    output logic [7:0]        fcode,
    output logic [4:0]        flane,
    output logic [31:0]       finfo
);
  localparam logic [31:0] SHARED_LAST = 32'd16380;   // 16 KiB - 4

  logic        scalar, atom;
  logic [31:0] limit;
  assign scalar = is_scalar_mem(ctrl.op);
  assign atom = !is_load(ctrl.op) && !is_store(ctrl.op);
  assign limit = ctrl.mem_shared ? SHARED_LAST : mem_last;

  always_comb begin
    logic [31:0] lead, wlead;
    logic [31:0][4:0] first, wfirst;
    logic [5:0] prefix [33];
    logic [31:0][5:0] bank_cnt;
    info = '0;
    // ---- addresses, lanes, data
    for (int i = 0; i < 32; i++) begin
      info.addr[i] = scalar ? (i == 0 ? sa + ctrl.imm : 32'd0) : va[i] + (atom ? 32'd0 : ctrl.imm);
      info.data[i] = scalar ? sd : atom ? vb[i] : vd[i];
    end
    info.lanes = scalar ? 32'd1 : exec;
    info.n_act = 6'($countones(info.lanes));
    // ---- precise fault
    fault = 1'b0;
    fcode = '0;
    flane = '0;
    finfo = '0;
    for (int i = 31; i >= 0; i--) begin
      if (info.lanes[i] && (info.addr[i][1:0] != 2'b0 || info.addr[i] > limit)) begin
        fault = 1'b1;
        fcode = info.addr[i][1:0] != 2'b0 ? E_MISALIGNED : E_RANGE;
        flane = 5'(i);
        finfo = info.addr[i];
      end
    end
    // ---- segments, in order of their lowest lane
    for (int i = 0; i < 32; i++) begin
      lead[i] = info.lanes[i];
      first[i] = 5'(i);
      for (int j = i - 1; j >= 0; j--) begin
        if (info.lanes[j] && info.addr[j][31:7] == info.addr[i][31:7]) begin
          lead[i] = 1'b0;
          first[i] = 5'(j);
        end
      end
    end
    prefix[0] = '0;
    for (int i = 0; i < 32; i++) prefix[i + 1] = prefix[i] + 6'(lead[i]);
    info.n_seg = prefix[32];
    for (int i = 0; i < 32; i++) info.seg[i] = prefix[int'(first[i])][4:0];
    // ---- shared banks: word = addr[13:2], bank = word mod 32
    bank_cnt = '0;
    for (int i = 0; i < 32; i++) begin
      wlead[i] = info.lanes[i];
      wfirst[i] = 5'(i);
      for (int j = i - 1; j >= 0; j--) begin
        if (info.lanes[j] && info.addr[j][13:2] == info.addr[i][13:2]) begin
          wlead[i] = 1'b0;
          wfirst[i] = 5'(j);
        end
      end
    end
    for (int i = 0; i < 32; i++) begin
      // round of a lane = distinct words of its bank whose lowest lane comes first
      logic [4:0] r;
      r = '0;
      for (int j = 0; j < 32; j++)
        if (wlead[j] && j < int'(wfirst[i]) && info.addr[j][6:2] == info.addr[i][6:2]) r = r + 5'd1;
      info.round[i] = r;
      if (wlead[i]) bank_cnt[info.addr[i][6:2]] = bank_cnt[info.addr[i][6:2]] + 6'd1;
    end
    info.d = 6'd1;
    for (int b = 0; b < 32; b++) if (bank_cnt[b] > info.d) info.d = bank_cnt[b];
  end
endmodule
