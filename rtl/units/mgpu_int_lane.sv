// mgpu_int_lane: one lane of integer work (combinational).
//
// Covers every S_ALU / V_ALU funct, the immediate forms (the operand stage supplies the
// extended immediate or the shift amount as b), all 16 V_CMP conditions (cond output),
// V_IMAD and V_SEL. ISA sections 5.1, 5.3, 6.2-6.5. The SALU of the core reuses it.
module mgpu_int_lane
  import mgpu_isa_pkg::*;
  import mgpu_fp_pkg::*;
(
    input  op_e         op,
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [31:0] c,      // V_IMAD addend
    input  logic        sel,    // V_SEL: this lane's bit of s[sc]
    output logic [31:0] result,
    output logic        cond    // V_CMP: condition true
);
  logic signed [31:0] sa, sb;
  logic [4:0] sh;
  logic [31:0] fa, fb;
  logic fnan;
  logic signed [32:0] ka, kb;
  logic [5:0] ones;
  logic [31:0] lowest;

  assign sa = a;
  assign sb = b;
  assign sh = b[4:0];
  assign fa = ftz(a);
  assign fb = ftz(b);
  assign fnan = is_nan(fa) || is_nan(fb);
  assign ka = cmp_key(fa);
  assign kb = cmp_key(fb);

  always_comb begin
    ones = '0;
    for (int i = 0; i < 32; i++) ones += {5'b0, a[i]};
    lowest = 32'hFFFF_FFFF;
    for (int i = 31; i >= 0; i--) if (a[i]) lowest = 32'(i);
  end

  always_comb begin
    result = '0;
    cond   = 1'b0;
    unique case (op)
      OP_S_ADD, OP_V_ADD, OP_S_ADDI, OP_V_ADDI:   result = a + b;
      OP_S_SUB, OP_V_SUB:                         result = a - b;
      OP_S_MUL, OP_V_MUL:                         result = a * b;
      OP_S_AND, OP_V_AND, OP_S_ANDI, OP_V_ANDI:   result = a & b;
      OP_S_OR, OP_V_OR, OP_S_ORI:                 result = a | b;
      OP_S_XOR, OP_V_XOR:                         result = a ^ b;
      OP_S_ANDN, OP_V_ANDN:                       result = a & ~b;
      OP_S_SHL, OP_V_SHL, OP_S_SHLI, OP_V_SHLI:   result = a << sh;
      OP_S_SHR, OP_V_SHR, OP_S_SHRI, OP_V_SHRI:   result = a >> sh;
      OP_S_SRA, OP_V_SRA, OP_S_SRAI, OP_V_SRAI:   result = 32'(sa >>> sh);
      OP_S_SLT, OP_V_SLT:                         result = {31'b0, sa < sb};
      OP_S_SLTU, OP_V_SLTU:                       result = {31'b0, a < b};
      OP_S_MIN, OP_V_MIN:                         result = (sa <= sb) ? a : b;
      OP_S_MAX, OP_V_MAX:                         result = (sa >= sb) ? a : b;
      OP_S_POPC:                                  result = {26'b0, ones};
      OP_S_FF1:                                   result = lowest;
      OP_S_LUI:                                   result = {b[15:0], 16'b0};
      OP_V_IMAD:                                  result = a * b + c;
      OP_V_SEL:                                   result = sel ? a : b;
      OP_V_CMP_EQ:  cond = a == b;
      OP_V_CMP_NE:  cond = a != b;
      OP_V_CMP_LT:  cond = sa < sb;
      OP_V_CMP_LE:  cond = sa <= sb;
      OP_V_CMP_GT:  cond = sa > sb;
      OP_V_CMP_GE:  cond = sa >= sb;
      OP_V_CMP_LTU: cond = a < b;
      OP_V_CMP_LEU: cond = a <= b;
      OP_V_CMP_GTU: cond = a > b;
      OP_V_CMP_GEU: cond = a >= b;
      OP_V_CMP_FEQ: cond = !fnan && ka == kb;
      OP_V_CMP_FNE: cond = fnan || ka != kb;
      OP_V_CMP_FLT: cond = !fnan && ka < kb;
      OP_V_CMP_FLE: cond = !fnan && ka <= kb;
      OP_V_CMP_FGT: cond = !fnan && ka > kb;
      OP_V_CMP_FGE: cond = !fnan && ka >= kb;
      default:      result = a;   // pass-through (moves, GETSR values supplied by the core)
    endcase
  end
endmodule
