// mgpu_valu: L integer lanes behind a LAT-cycle pipeline (docs/uarch/core.md section 7).
// Accepts one pass (L lanes) per cycle; out_valid/result follow in_valid by LAT cycles.
module mgpu_valu
  import mgpu_isa_pkg::*;
#(
    parameter int L   = 8,
    parameter int LAT = 2
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  op_e                op,
    input  logic [L-1:0][31:0] a,
    input  logic [L-1:0][31:0] b,
    input  logic [L-1:0][31:0] c,
    input  logic [L-1:0]       sel,
    output logic               out_valid,
    output logic [L-1:0][31:0] result,
    output logic [L-1:0]       cond
);
  logic [L-1:0][31:0] res_c;
  logic [L-1:0]       cond_c;

  for (genvar i = 0; i < L; i++) begin : g_lane
    mgpu_int_lane u_lane (
        .op(op), .a(a[i]), .b(b[i]), .c(c[i]), .sel(sel[i]),
        .result(res_c[i]), .cond(cond_c[i])
    );
  end

  mgpu_pipe #(.DEPTH(LAT), .WIDTH(L * 33)) u_pipe (
      .clk, .rst_n, .in_valid,
      .in_data({cond_c, res_c}),
      .out_valid,
      .out_data({cond, result})
  );
endmodule
