// mgpu_pipe: a valid/data delay line of DEPTH register stages (DEPTH = 0 is a wire).
// Execution units compute combinationally and use this to present results with the
// latency given in docs/uarch/core.md section 7.
module mgpu_pipe #(
    parameter int DEPTH = 1,
    parameter int WIDTH = 32
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             in_valid,
    input  logic [WIDTH-1:0] in_data,
    output logic             out_valid,
    output logic [WIDTH-1:0] out_data
);
  if (DEPTH == 0) begin : g_wire
    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_clk_rst;
    assign unused_clk_rst = clk ^ rst_n;  // a zero-latency pipe needs no clock
    /* verilator lint_on UNUSEDSIGNAL */
    assign out_valid = in_valid;
    assign out_data  = in_data;
  end else begin : g_regs
    logic [DEPTH-1:0]            valid_q;
    logic [DEPTH-1:0][WIDTH-1:0] data_q;
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        valid_q <= '0;
      end else begin
        valid_q[0] <= in_valid;
        for (int i = 1; i < DEPTH; i++) valid_q[i] <= valid_q[i-1];
      end
      data_q[0] <= in_data;
      for (int i = 1; i < DEPTH; i++) data_q[i] <= data_q[i-1];
    end
    assign out_valid = valid_q[DEPTH-1];
    assign out_data  = data_q[DEPTH-1];
  end
endmodule
