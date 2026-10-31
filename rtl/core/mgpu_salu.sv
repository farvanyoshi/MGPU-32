// mgpu_salu: scalar unit (docs/uarch/core.md 7). Occupancy 1, latency 1: an instruction
// entering at t + 2 presents its retire record at t + 3.
//
// Executes S_ALU and the scalar immediates (through mgpu_int_lane), branches (the next PC
// goes out on br_*), NOP, FENCE, EXIT and BAR (events). S_GETSR arrives already turned
// into S_ADD of the register value and 0 by the issue stage.
module mgpu_salu
  import mgpu_isa_pkg::*;
  import mgpu_core_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,
    input  ipkt_t       in,
    output retire_t     out,
    output logic        br_valid,   // branch completing with `out` (same warp)
    output logic [31:0] br_pc,      // its next PC
    output logic        fence       // a FENCE completing with `out`
);
  logic [31:0] res, target;
  logic        taken;
  /* verilator lint_off UNUSEDSIGNAL */
  logic        cond_unused;
  /* verilator lint_on UNUSEDSIGNAL */
  mgpu_int_lane u_alu (.op(in.ctrl.op), .a(in.sa), .b(in.sb), .c(32'd0), .sel(1'b0),
                       .result(res), .cond(cond_unused));

  assign target = in.pc + 32'd4 + in.ctrl.imm;
  always_comb begin
    unique case (in.ctrl.op)
      OP_S_BZ:  taken = in.sd == 32'd0;
      OP_S_BNZ: taken = in.sd != 32'd0;
      default:  taken = 1'b1;      // S_J
    endcase
  end

  retire_t r;
  always_comb begin
    r = ret_from(in);
    unique case (in.ctrl.op)
      OP_NOP, OP_FENCE, OP_S_BZ, OP_S_BNZ, OP_S_J: ;
      OP_EXIT: r.event_ = EV_EXIT;
      OP_BAR:  r.event_ = EV_BAR;
      default: r = ret_scalar(r, in.ctrl.rd, res);
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      out.valid <= 1'b0;
      br_valid <= 1'b0;
      fence <= 1'b0;
    end else begin
      fence <= in.valid && in.ctrl.op == OP_FENCE;
      out <= in.valid ? r : '0;
      br_valid <= in.valid && in.ctrl.branch;
      br_pc <= taken ? target : in.pc + 32'd4;
    end
  end
endmodule
