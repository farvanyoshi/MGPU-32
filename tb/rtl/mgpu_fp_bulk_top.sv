// Bulk-test wrapper: the FPU (L = 8) and the SFU (S = 2) side by side.
module mgpu_fp_bulk_top
  import mgpu_isa_pkg::*;
(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             fpu_valid,
    input  logic [6:0]       fpu_op,
    input  logic [7:0][31:0] fpu_a,
    input  logic [7:0][31:0] fpu_b,
    input  logic [7:0][31:0] fpu_c,
    output logic             fpu_out_valid,
    output logic [7:0][31:0] fpu_result,
    input  logic             sfu_valid,
    input  logic [6:0]       sfu_op,
    input  logic [1:0][31:0] sfu_a,
    output logic             sfu_out_valid,
    output logic [1:0][31:0] sfu_result
);
  mgpu_fpu #(.L(8), .LAT(4)) u_fpu (
      .clk, .rst_n, .in_valid(fpu_valid), .op(op_e'(fpu_op)), .a(fpu_a), .b(fpu_b), .c(fpu_c),
      .out_valid(fpu_out_valid), .result(fpu_result)
  );
  mgpu_sfu #(.S(2), .LAT(8)) u_sfu (
      .clk, .rst_n, .in_valid(sfu_valid), .op(op_e'(sfu_op)), .a(sfu_a),
      .out_valid(sfu_out_valid), .result(sfu_result)
  );
endmodule
