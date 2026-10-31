// mgpu_fpu: L floating-point lanes behind a LAT-cycle pipeline (docs/uarch/core.md section 7).
module mgpu_fpu
  import mgpu_isa_pkg::*;
#(
    parameter int L   = 8,
    parameter int LAT = 4
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  op_e                op,
    input  logic [L-1:0][31:0] a,
    input  logic [L-1:0][31:0] b,
    input  logic [L-1:0][31:0] c,
    output logic               out_valid,
    output logic [L-1:0][31:0] result
);
  logic [L-1:0][31:0] res_c;

  for (genvar i = 0; i < L; i++) begin : g_lane
    mgpu_fpu_lane u_lane (.op(op), .a(a[i]), .b(b[i]), .c(c[i]), .result(res_c[i]));
  end

  mgpu_pipe #(.DEPTH(LAT), .WIDTH(L * 32)) u_pipe (
      .clk, .rst_n, .in_valid, .in_data(res_c), .out_valid, .out_data(result)
  );
endmodule
