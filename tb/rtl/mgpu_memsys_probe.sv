// Test wrapper: mgpu_mem_arbiter + mgpu_atomic_unit, i.e. everything between the cores'
// memory ports and the 128-bit memory port.
module mgpu_memsys_probe
  import mgpu_core_pkg::*;
#(
    parameter int C = 4
) (
    input  logic                       clk,
    input  logic                       rst_n,
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
    output logic                       overflow,
    output logic                       mreq_valid,
    input  logic                       mreq_ready,
    output logic [1:0]                 mreq_op,
    output logic [31:0]                mreq_addr,
    output logic [15:0]                mreq_tag,
    output logic [31:0]                mreq_mask,
    output logic                       mwd_valid,
    output logic [127:0]               mwd_data,
    output logic                       mwd_last,
    input  logic                       mrd_valid,
    input  logic [15:0]                mrd_tag,
    input  logic [127:0]               mrd_data,
    input  logic                       mrd_last,
    input  logic                       mack_valid,
    input  logic [15:0]                mack_tag
);
  logic              sreq_valid, sreq_ready, srd_valid, sat_valid, sack_valid;
  sreq_t             sreq;
  logic [15:0]       srd_tag, sat_tag, sack_tag;
  logic [31:0][31:0] srd_data, sat_old;
  mgpu_mem_arbiter #(.C(C)) u_arb (.*);
  mgpu_atomic_unit u_atom (.*);
endmodule
