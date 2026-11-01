// mgpu_top: the MGPU-32 GPU (docs/uarch/top.md).
//
// Host registers, block dispatcher, C cores, memory arbiter and atomic unit, behind two
// ports: the host CSR port (interfaces.md 1; mgpu_axil_csr adapts it to AXI4-Lite) and the
// 128-bit memory port (interfaces.md 2). The cores' retire ports and the arbiter's queue
// overflow flag are brought out for verification only.
module mgpu_top
  import mgpu_core_pkg::*;
#(
    parameter int          C        = 4,
    parameter int          W        = 8,
    parameter int          L        = 8,
    parameter int          S        = 2,
    parameter logic [32:0] MEM_SIZE = 33'h1_0000_0000,
    parameter int          QL       = 64
) (
    input  logic                         clk,
    input  logic                         rst_n,
    // host CSR port
    input  logic                         csr_valid,
    output logic                         csr_ready,
    input  logic                         csr_write,
    input  logic [7:0]                   csr_addr,
    input  logic [31:0]                  csr_wdata,
    output logic                         csr_rvalid,
    output logic [31:0]                  csr_rdata,
    output logic                         irq_done,
    // memory port
    output logic                         mreq_valid,
    input  logic                         mreq_ready,
    output logic [1:0]                   mreq_op,
    output logic [31:0]                  mreq_addr,
    output logic [15:0]                  mreq_tag,
    output logic [31:0]                  mreq_mask,
    output logic                         mwd_valid,
    output logic [127:0]                 mwd_data,
    output logic                         mwd_last,
    input  logic                         mrd_valid,
    input  logic [15:0]                  mrd_tag,
    input  logic [127:0]                 mrd_data,
    input  logic                         mrd_last,
    input  logic                         mack_valid,
    input  logic [15:0]                  mack_tag,
    // verification
    output logic [C-1:0][NRET-1:0]       ret_valid,
    output retire_t [C-1:0][NRET-1:0]    ret,
    output logic                         overflow
);
  localparam logic [32:0] MEM_LAST = MEM_SIZE - 33'd4;

  // ---------------------------------------------------------------- host registers + dispatcher
  logic             launch, end_valid;
  logic [31:0]      entry, grid_dim, block_dim, end_cycles, live_cycles;
  logic [7:0][31:0] args;
  logic [5:0][31:0] end_status;
  mgpu_host_regs #(.W(W), .MEM_SIZE(MEM_SIZE)) u_regs (.*);

  logic [C-1:0]       start, new_launch, idle, block_done, busy_ops, mem_quiet;
  // halt[c] depends on fault_issue of lower cores only; split per bit so that the simulator
  // sees there is no loop
  logic [C-1:0]       halt /* verilator split_var */;
  logic [C-1:0]       fault_issue /* verilator split_var */;
  logic [C-1:0]       fault_valid, fault_det, fault_go;
  logic [C-1:0][31:0] fault_clk;
  logic [C-1:0][31:0] block_id, fault_pc, fault_info;
  logic [C-1:0][7:0]  fault_code;
  logic [C-1:0][4:0]  fault_warp, fault_lane;
  logic [5:0]         num_warps;
  logic               kill, busy;
  mgpu_dispatcher #(.C(C), .W(W)) u_disp (.*);

  always_comb begin
    for (int c = 0; c < C; c++) begin
      fault_valid[c] = ret_valid[c][R_FAULT];
      fault_code[c] = ret[c][R_FAULT].fault_code;
      fault_pc[c] = ret[c][R_FAULT].pc;
      fault_warp[c] = ret[c][R_FAULT].warp;
      fault_lane[c] = ret[c][R_FAULT].fault_lane;
      fault_info[c] = ret[c][R_FAULT].fault_info;
    end
  end

  // ---------------------------------------------------------------- cores
  logic [C-1:0]             if_req_valid, if_resp_valid, rd_valid, at_valid, ack_valid;
  logic [C-1:0][31:0]       if_req_addr;
  logic [C-1:0][4:0]        if_req_id, if_resp_id, rd_warp, rd_seg, at_warp, at_seg, ack_warp;
  logic [C-1:0][31:0][31:0] if_resp_data, rd_data, at_old;
  mreq_t [C-1:0]            lsu_req;

  for (genvar c = 0; c < C; c++) begin : g_core
    mgpu_core #(.W(W), .L(L), .S(S)) u_core (
        .clk, .rst_n,
        .start(start[c]), .kill, .new_launch(new_launch[c]), .entry, .block_id(block_id[c]),
        .block_dim, .grid_dim, .num_warps, .args, .mem_last(MEM_LAST[31:0]),
        .idle(idle[c]), .block_done(block_done[c]),
        .halt_in(halt[c]), .fault_go(fault_go[c]), .fault_issue(fault_issue[c]),
        .fault_det(fault_det[c]), .fault_clk(fault_clk[c]), .busy_ops(busy_ops[c]),
        .mem_quiet(mem_quiet[c]),
        .if_req_valid(if_req_valid[c]), .if_req_addr(if_req_addr[c]), .if_req_id(if_req_id[c]),
        .if_resp_valid(if_resp_valid[c]), .if_resp_id(if_resp_id[c]), .if_resp_data(if_resp_data[c]),
        .lsu_req(lsu_req[c]),
        .rd_valid(rd_valid[c]), .rd_warp(rd_warp[c]), .rd_seg(rd_seg[c]), .rd_data(rd_data[c]),
        .at_valid(at_valid[c]), .at_warp(at_warp[c]), .at_seg(at_seg[c]), .at_old(at_old[c]),
        .ack_valid(ack_valid[c]), .ack_warp(ack_warp[c]),
        .ret_valid(ret_valid[c]), .ret(ret[c]));
  end

  // ---------------------------------------------------------------- memory system
  logic              sreq_valid, sreq_ready, srd_valid, sat_valid, sack_valid;
  sreq_t             sreq;
  logic [15:0]       srd_tag, sat_tag, sack_tag;
  logic [31:0][31:0] srd_data, sat_old;
  mgpu_mem_arbiter #(.C(C), .QL(QL), .QI(W)) u_arb (.*);
  mgpu_atomic_unit u_atom (.*);

  /* verilator lint_off UNUSEDSIGNAL */
  logic unused;
  assign unused = ^{busy, MEM_LAST[32]};
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
