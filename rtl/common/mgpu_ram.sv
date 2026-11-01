// mgpu_ram: DEPTH x (LANES x 32 bits) RAM with one write port (a write enable per 32-bit
// lane) and one read port with a registered output: data addressed in cycle n appears in
// cycle n + 1. Written in the form FPGA synthesis tools infer as block or distributed RAM.
module mgpu_ram #(
    parameter int DEPTH = 32,
    parameter int LANES = 8,
    localparam int AB = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  logic                   clk,
    input  logic [LANES-1:0]       we,
    input  logic [AB-1:0]          waddr,
    input  logic [LANES-1:0][31:0] wdata,
    input  logic                   re,
    input  logic [AB-1:0]          raddr,
    output logic [LANES-1:0][31:0] rdata
);
  logic [LANES*32-1:0] mem [DEPTH];
  always_ff @(posedge clk) begin
    for (int l = 0; l < LANES; l++)
      if (we[l]) mem[waddr][32*l +: 32] <= wdata[l];
    if (re) rdata <= mem[raddr];
  end
endmodule
