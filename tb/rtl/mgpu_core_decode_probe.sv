// Test wrapper: mgpu_decode's ctrl_t fields on separate ports.
module mgpu_core_decode_probe
  import mgpu_core_pkg::*;
(
    input  logic [31:0] word,
    output logic        illegal,
    output logic [6:0]  op,
    output logic [2:0]  unit,
    output logic [31:0] imm,
    output logic        sb,
    output logic        vec_dst,
    output logic        scl_dst,
    output logic        branch,
    output logic        b_imm,
    output logic        mem_shared
);
  ctrl_t c;
  mgpu_decode u_dec (.word(word), .ctrl(c));
  assign illegal = c.illegal;
  assign op = c.op;
  assign unit = c.unit;
  assign imm = c.imm;
  assign sb = c.sb;
  assign vec_dst = c.vec_dst;
  assign scl_dst = c.scl_dst;
  assign branch = c.branch;
  assign b_imm = c.b_imm;
  assign mem_shared = c.mem_shared;
endmodule
