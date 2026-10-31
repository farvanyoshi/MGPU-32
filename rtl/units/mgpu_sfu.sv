// mgpu_sfu: S special-function lanes behind a LAT-cycle pipeline (docs/uarch/core.md
// section 7). A 32-lane instruction takes 32 / S passes.
module mgpu_sfu
  import mgpu_isa_pkg::*;
#(
    parameter int S   = 2,
    parameter int LAT = 8
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  op_e                op,
    input  logic [S-1:0][31:0] a,
    output logic               out_valid,
    output logic [S-1:0][31:0] result
);
  logic [S-1:0][31:0] res_c;

  for (genvar i = 0; i < S; i++) begin : g_lane
    mgpu_sfu_lane u_lane (.op(op), .a(a[i]), .result(res_c[i]));
  end

  mgpu_pipe #(.DEPTH(LAT), .WIDTH(S * 32)) u_pipe (
      .clk, .rst_n, .in_valid, .in_data(res_c), .out_valid, .out_data(result)
  );
endmodule
