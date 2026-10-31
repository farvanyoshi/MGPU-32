// mgpu_round: round (-1)^sign * m * 2^scale to binary32 ([FP-3]).
//
// Nearest-even at 24 significant bits with an unbounded exponent, then overflow to +-Inf
// and flush to +-0 when the rounded exponent is below -126 (tininess after rounding).
// m = 0 gives a signed zero. Combinational; MW is the width of m.
module mgpu_round #(
    parameter int MW = 128
) (
    input  logic                 sign,
    input  logic [MW-1:0]        m,
    input  logic signed [15:0]   scale,
    output logic [31:0]          result
);
  int p;              // index of the leading one
  logic [24:0] shifted;    // significand plus the guard bit
  logic [MW-1:0] low;
  logic [23:0] q;
  logic guard, sticky;
  logic [24:0] sig;   // 24-bit significand plus a carry bit
  int e;

  always_comb begin
    p = 0;
    for (int i = 0; i < MW; i++) if (m[i]) p = i;
    guard = 1'b0;
    sticky = 1'b0;
    shifted = '0;
    q = '0;
    low = '0;
    if (p > 23) begin
      shifted = 25'(m >> (p - 24));
      guard = shifted[0];
      q = shifted[24:1];
      low = m << (MW - (p - 24));        // bits below the guard bit (shift by MW -> 0)
      sticky = (p - 24 > 0) && (low != '0);
      sig = {1'b0, q} + {24'b0, guard && (sticky || q[0])};
    end else begin
      sig = {1'b0, 24'(m << (23 - p))};
    end
    e = p + int'(scale);
    if (sig[24]) begin                   // rounding carried into a new binade
      sig = sig >> 1;
      e = e + 1;
    end
    if (m == '0) result = {sign, 31'b0};
    else if (e > 127) result = {sign, 8'hFF, 23'b0};
    else if (e < -126) result = {sign, 31'b0};
    else result = {sign, 8'(e + 127), sig[22:0]};
  end
endmodule
