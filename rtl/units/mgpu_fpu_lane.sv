// mgpu_fpu_lane: one lane of FADD, FSUB, FMUL, FFMA, FMIN, FMAX, I2F, F2I (combinational).
//
// Bit-exact with python/mgpu/fp/ops.py and rust/mgpu-core/src/fp.rs ([FP-1]..[FP-10]).
// Additions use the same structure as the Rust add_round: both addends are normalized to
// 64-bit mantissas, the smaller one is aligned (shifted out bits beyond 63 positions are
// jammed into a sticky bit), the 128-bit sum or difference is rounded once.
module mgpu_fpu_lane
  import mgpu_isa_pkg::*;
  import mgpu_fp_pkg::*;
(
    input  op_e         op,
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [31:0] c,
    output logic [31:0] result
);
  // ---- operands after [FP-1]; FSUB flips b's sign (after the NaN check, ops.py)
  logic [31:0] fa, fb, fc;
  assign fa = ftz(a);
  assign fb = (op == OP_V_FSUB) ? ftz(b) ^ 32'h8000_0000 : ftz(b);
  assign fc = ftz(c);

  // A normal x is M * 2^E with M = mant(x[22:0]) (24 bits) and E = expo(x[30:23]).
  function automatic logic [23:0] mant(input logic [22:0] frac);
    return {1'b1, frac};
  endfunction

  function automatic int expo(input logic [7:0] biased);
    return int'(biased) - 150;
  endfunction

  function automatic int lzc64(input logic [63:0] x);
    int n;
    n = 64;
    for (int i = 0; i < 64; i++) if (x[i]) n = 63 - i;
    return n;
  endfunction

  // ---- exact addition of two non-zero values (sign, M, E), as (sign, 128-bit, exponent)
  logic        add_sa, add_sb;
  logic [63:0] add_ma, add_mb;
  int          add_ea, add_eb;
  logic        sum_sign;
  logic [127:0] sum_mag;
  logic signed [15:0] sum_exp;  // exponents stay within +-1024

  always_comb begin
    logic        s1, s2;
    logic [63:0] m1, m2, shifted;
    int          e1, e2, d, sh, l1, l2;
    logic [127:0] larger, smaller;
    logic        lost;
    shifted = '0;
    lost = 1'b0;
    sh = 0;
    l1 = lzc64(add_ma);
    l2 = lzc64(add_mb);
    s1 = add_sa; m1 = add_ma << l1; e1 = add_ea - l1;
    s2 = add_sb; m2 = add_mb << l2; e2 = add_eb - l2;
    if (e2 > e1 || (e2 == e1 && m2 > m1)) begin
      {s1, s2} = {s2, s1};
      {m1, m2} = {m2, m1};
      d = e1; e1 = e2; e2 = d;
    end
    d = e1 - e2;
    if (d <= 63) begin
      larger = {64'b0, m1} << d;
      smaller = {64'b0, m2};
      sum_exp = 16'(e2);
    end else begin
      sh = d - 63;
      shifted = (sh >= 64) ? 64'b0 : m2 >> sh;
      lost = (sh >= 64) ? (m2 != 0) : ((m2 << (64 - sh)) != 0);
      larger = {64'b0, m1} << 63;
      smaller = {64'b0, shifted | {63'b0, lost}};
      sum_exp = 16'(e1 - 63);
    end
    sum_sign = s1;
    sum_mag = (s1 == s2) ? larger + smaller : larger - smaller;   // |larger| >= |smaller|; equal -> 0
  end

  // ---- the shared rounder
  logic        rnd_sign;
  logic [127:0] rnd_m;
  logic signed [15:0] rnd_scale;
  logic [31:0] rounded;
  mgpu_round #(.MW(128)) u_round (
      .sign(rnd_sign), .m(rnd_m), .scale(rnd_scale), .result(rounded)
  );

  // ---- operation select, specials first (in the order of ops.py)
  logic sp;            // product sign
  logic p_inf, p_zero;
  logic [47:0] prod;
  logic signed [32:0] ka, kb;
  logic signed [63:0] f2i_mag, f2i_val;
  logic [31:0] i2f_mag;

  always_comb begin
    sp = fa[31] ^ fb[31];
    p_inf = is_inf(fa) || is_inf(fb);
    p_zero = is_zero(fa) || is_zero(fb);
    prod = mant(fa[22:0]) * mant(fb[22:0]);
    ka = order_key(fa);
    kb = order_key(fb);
    add_sa = fa[31]; add_ma = {40'b0, mant(fa[22:0])}; add_ea = expo(fa[30:23]);
    add_sb = fb[31]; add_mb = {40'b0, mant(fb[22:0])}; add_eb = expo(fb[30:23]);
    rnd_sign = 1'b0; rnd_m = '0; rnd_scale = '0;
    f2i_mag = '0; f2i_val = '0; i2f_mag = '0;
    result = CANON_NAN;
    unique case (op)
      OP_V_FADD, OP_V_FSUB: begin
        if (is_nan(fa) || is_nan(fb)) result = CANON_NAN;
        else if (is_inf(fa) && is_inf(fb)) result = (fa[31] == fb[31]) ? fa : CANON_NAN;
        else if (is_inf(fa)) result = fa;
        else if (is_inf(fb)) result = fb;
        else if (is_zero(fa) && is_zero(fb)) result = (fa[31] == fb[31]) ? fa : 32'h0;
        else if (is_zero(fa)) result = fb;
        else if (is_zero(fb)) result = fa;
        else begin
          rnd_sign = sum_sign; rnd_m = sum_mag; rnd_scale = sum_exp;
          result = (sum_mag == '0) ? 32'h0 : rounded;
        end
      end
      OP_V_FMUL: begin
        if (is_nan(fa) || is_nan(fb)) result = CANON_NAN;
        else if (p_inf && p_zero) result = CANON_NAN;
        else if (p_inf) result = signed_inf(sp);
        else if (p_zero) result = {sp, 31'b0};
        else begin
          rnd_sign = sp; rnd_m = {80'b0, prod}; rnd_scale = 16'(expo(fa[30:23]) + expo(fb[30:23]));
          result = rounded;
        end
      end
      OP_V_FFMA: begin
        add_sa = sp; add_ma = {16'b0, prod}; add_ea = expo(fa[30:23]) + expo(fb[30:23]);
        add_sb = fc[31]; add_mb = {40'b0, mant(fc[22:0])}; add_eb = expo(fc[30:23]);
        if (is_nan(fa) || is_nan(fb) || is_nan(fc)) result = CANON_NAN;
        else if (p_inf && p_zero) result = CANON_NAN;
        else if (p_inf) result = (is_inf(fc) && fc[31] != sp) ? CANON_NAN : signed_inf(sp);
        else if (is_inf(fc)) result = fc;
        else if (p_zero) result = is_zero(fc) ? ((fc[31] == sp) ? fc : 32'h0) : fc;
        else if (is_zero(fc)) begin
          rnd_sign = sp; rnd_m = {80'b0, prod}; rnd_scale = 16'(expo(fa[30:23]) + expo(fb[30:23]));
          result = rounded;
        end else begin
          rnd_sign = sum_sign; rnd_m = sum_mag; rnd_scale = sum_exp;
          result = (sum_mag == '0) ? 32'h0 : rounded;
        end
      end
      OP_V_FMIN, OP_V_FMAX: begin
        if (is_nan(fa) && is_nan(fb)) result = CANON_NAN;
        else if (is_nan(fa)) result = fb;
        else if (is_nan(fb)) result = fa;
        else if (op == OP_V_FMIN) result = (ka <= kb) ? fa : fb;
        else result = (ka >= kb) ? fa : fb;
      end
      OP_V_I2F: begin
        i2f_mag = a[31] ? -a : a;
        rnd_sign = a[31]; rnd_m = {96'b0, i2f_mag}; rnd_scale = '0;
        result = rounded;
      end
      OP_V_F2I: begin
        if (is_nan(fa) || is_zero(fa)) result = 32'h0;
        else if (is_inf(fa)) result = fa[31] ? 32'h8000_0000 : 32'h7FFF_FFFF;
        else begin
          if (expo(fa[30:23]) >= 40) f2i_mag = 64'sh4000_0000_0000_0000;
          else if (expo(fa[30:23]) >= 0) f2i_mag = 64'(mant(fa[22:0])) << expo(fa[30:23]);
          else if (expo(fa[30:23]) > -64) f2i_mag = 64'(mant(fa[22:0])) >> (-expo(fa[30:23]));
          else f2i_mag = '0;
          f2i_val = fa[31] ? -f2i_mag : f2i_mag;
          if (f2i_val > 64'sh7FFF_FFFF) result = 32'h7FFF_FFFF;
          else if (f2i_val < -64'sh8000_0000) result = 32'h8000_0000;
          else result = f2i_val[31:0];
        end
      end
      default: result = CANON_NAN;
    endcase
  end
endmodule
