// Floating-point helpers shared by the VALU (float compares), FPU and SFU.
// Bit-level definitions of docs/isa/MGPU-32-ISA.md section 8.
// Predicates take whole binary32 words and ignore the fields they do not need, and not
// every constant is used by every design: both are expected in a helper package.
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */
package mgpu_fp_pkg;

  localparam logic [31:0] CANON_NAN = 32'h7FC0_0000;
  localparam logic [31:0] POS_INF   = 32'h7F80_0000;
  localparam logic [31:0] NEG_INF   = 32'hFF80_0000;
  localparam logic [31:0] ONE       = 32'h3F80_0000;

  function automatic logic is_nan(input logic [31:0] x);
    return x[30:23] == 8'hFF && x[22:0] != '0;
  endfunction

  function automatic logic is_inf(input logic [31:0] x);
    return x[30:23] == 8'hFF && x[22:0] == '0;
  endfunction

  // True for +-0 after flushing, i.e. also for denormals.
  function automatic logic is_zero(input logic [31:0] x);
    return x[30:23] == '0;
  endfunction

  // [FP-1]: denormal inputs become a zero of the same sign.
  function automatic logic [31:0] ftz(input logic [31:0] x);
    return (x[30:23] == '0) ? {x[31], 31'b0} : x;
  endfunction

  function automatic logic [31:0] signed_inf(input logic s);
    return s ? NEG_INF : POS_INF;
  endfunction

  // Total order on non-NaN values with -0 < +0 (FMIN/FMAX), as a signed 33-bit key.
  function automatic logic signed [32:0] order_key(input logic [31:0] x);
    logic signed [32:0] mag;
    mag = {2'b00, x[30:0]};
    return x[31] ? -mag - 33'sd1 : mag;
  endfunction

  // Order for compares ([FP-8]): like order_key, but -0 == +0 (x already flushed).
  function automatic logic signed [32:0] cmp_key(input logic [31:0] x);
    return is_zero(x) ? 33'sd0 : order_key(x);
  endfunction

endpackage
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on UNUSEDPARAM */
