// Test wrapper: exposes mgpu_isa_pkg::decode_op and unit_of on ports.
module mgpu_decode_probe
  import mgpu_isa_pkg::*;
(
    input  logic [31:0] word,
    output logic        valid,
    output logic [6:0]  op,
    output logic [2:0]  unit
);
  dec_t d;
  assign d     = decode_op(word);
  assign valid = d.valid;
  assign op    = d.op;
  assign unit  = unit_of(d.op);
endmodule
