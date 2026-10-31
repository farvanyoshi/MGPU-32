// mgpu_sfu_lane: one lane of FRCP, FRSQ, FEX2, FLG2 (combinational).
//
// Bit-exact implementation of docs/isa/sfu.md (python/mgpu/fp/sfu.py): special cases,
// table selection, the shared quadratic datapath on the coefficient ROM, and one final
// rounding with mgpu_round.
module mgpu_sfu_lane
  import mgpu_isa_pkg::*;
  import mgpu_fp_pkg::*;
(
    input  op_e         op,
    input  logic [31:0] a,
    output logic [31:0] result
);
  localparam logic [2:0] T_RCP = 3'd0, T_RSQ_EVEN = 3'd1, T_RSQ_ODD = 3'd2, T_EXP2 = 3'd3,
                         T_LOG2 = 3'd4, T_LOG2_HI = 3'd5, T_LOG2_LO = 3'd6;

  logic [31:0] x;
  logic        s;
  int          e;              // unbiased exponent
  logic [22:0] f;
  assign x = ftz(a);
  assign s = x[31];
  assign e = int'(x[30:23]) - 127;
  assign f = x[22:0];

  // ---- FEX2 fixed-point input: X = floor(|x| * 2^30), negated for x < 0
  logic signed [47:0] xf;
  logic signed [47:0] n_ex2;     // floor(x)
  logic [29:0]        t_ex2;     // fractional part, 30 bits
  always_comb begin
    logic [47:0] mag;
    int sh;
    sh = e + 7;
    if (sh >= 0) mag = 48'({1'b1, f}) << sh;
    else if (sh > -48) mag = 48'({1'b1, f}) >> (-sh);
    else mag = '0;
    xf = s ? -$signed(mag) : $signed(mag);
    n_ex2 = xf >>> 30;
    t_ex2 = xf[29:0];
  end

  // ---- table selection and the input T (W bits) of the table datapath
  logic [2:0]  tbl;
  logic [29:0] t;              // T, left as an integer (not aligned)
  logic        wide;           // W = 30 (exp2) instead of 23
  always_comb begin
    tbl = T_RCP;
    t = {7'b0, f};
    wide = 1'b0;
    unique case (op)
      OP_V_FRCP: tbl = T_RCP;
      OP_V_FRSQ: tbl = e[0] ? T_RSQ_ODD : T_RSQ_EVEN;
      OP_V_FEX2: begin
        tbl = T_EXP2;
        t = t_ex2;
        wide = 1'b1;
      end
      OP_V_FLG2: tbl = (e == 0) ? T_LOG2_HI : (e == -1) ? T_LOG2_LO : T_LOG2;
      default: tbl = T_RCP;
    endcase
  end

  // ---- Y = C0 + ((C1 * d) >> (D - 1)) + ((C2 * sq) >> 16), sq = (d * d) >> (2D - 16)
  logic [6:0]  seg;
  logic [22:0] d;              // D = 16 or 23 bits
  logic [31:0] c0;
  logic signed [23:0] c1;
  logic signed [17:0] c2;
  logic signed [63:0] y;
  assign seg = wide ? t[29:23] : t[22:16];
  assign d = wide ? t[22:0] : {7'b0, t[15:0]};
  mgpu_sfu_rom u_rom (.tbl(tbl), .seg(seg), .c0(c0), .c1(c1), .c2(c2));
  always_comb begin
    logic [45:0] dd;
    logic signed [63:0] sq, p1, p2;
    dd = 46'(d) * 46'(d);
    sq = wide ? $signed({18'b0, dd >> 30}) : $signed({18'b0, dd >> 16});
    p1 = $signed(64'(c1)) * $signed({41'b0, d});
    p2 = $signed(64'(c2)) * sq;
    y = $signed({32'b0, c0}) + (p1 >>> (wide ? 22 : 15)) + (p2 >>> 16);
  end

  // ---- per-operation assembly and the final rounding
  logic        r_sign;
  logic [63:0] r_m;
  logic signed [15:0] r_scale;
  logic [31:0] rounded;
  mgpu_round #(.MW(64)) u_round (.sign(r_sign), .m(r_m), .scale(r_scale), .result(rounded));

  always_comb begin
    int half;
    logic signed [63:0] v;
    half = e >>> 1;              // floor(e / 2)
    v = (64'(signed'(e)) <<< 30) + y;
    r_sign = 1'b0;
    r_m = 64'(y);
    r_scale = '0;
    result = CANON_NAN;
    unique case (op)
      OP_V_FRCP: begin
        r_sign = s;
        if (f == '0) begin r_m = 64'd1; r_scale = 16'(-e); end
        else r_scale = 16'(-30 - e);
        if (is_nan(x)) result = CANON_NAN;
        else if (is_inf(x)) result = {s, 31'b0};
        else if (is_zero(x)) result = signed_inf(s);
        else result = rounded;
      end
      OP_V_FRSQ: begin
        if (!e[0] && f == '0) begin r_m = 64'd1; r_scale = 16'(-half); end
        else r_scale = 16'(-30 - half);
        if (is_nan(x)) result = CANON_NAN;
        else if (is_zero(x)) result = signed_inf(s);
        else if (s) result = CANON_NAN;
        else if (is_inf(x)) result = 32'h0;
        else result = rounded;
      end
      OP_V_FEX2: begin
        if (t_ex2 == '0) begin r_m = 64'd1; r_scale = 16'(n_ex2); end
        else r_scale = 16'(n_ex2 - 30);
        if (is_nan(x)) result = CANON_NAN;
        else if (is_inf(x)) result = s ? 32'h0 : POS_INF;
        else if (is_zero(x)) result = ONE;
        else if (e >= 8) result = s ? 32'h0 : POS_INF;
        else result = rounded;
      end
      OP_V_FLG2: begin
        if (f == '0) begin
          r_sign = e < 0;
          r_m = {32'b0, (e < 0) ? 32'(-e) : 32'(e)};
          r_scale = '0;
        end else if (e == 0) begin
          r_m = 64'(f) * 64'(y);
          r_scale = -16'sd53;
        end else if (e == -1) begin
          r_sign = 1'b1;
          r_m = (64'd1 << 23) - 64'(f);
          r_m = r_m * 64'(y);
          r_scale = -16'sd54;
        end else begin
          r_sign = v < 0;
          r_m = v < 0 ? 64'(-v) : 64'(v);
          r_scale = -16'sd30;
        end
        if (is_nan(x)) result = CANON_NAN;
        else if (is_zero(x)) result = NEG_INF;
        else if (s) result = CANON_NAN;
        else if (is_inf(x)) result = POS_INF;
        else result = rounded;
      end
      default: result = CANON_NAN;
    endcase
  end
endmodule
