// mgpu_decode: instruction word -> ctrl_t (combinational). docs/uarch/core.md section 5.
module mgpu_decode
  import mgpu_isa_pkg::*;
  import mgpu_core_pkg::*;
(
    input  logic [31:0] word,
    output ctrl_t       ctrl
);
  dec_t d;
  logic [31:0] simm, uimm, shamt, boff, joff;
  assign d     = decode_op(word);
  assign simm  = {{16{word[15]}}, word[15:0]};
  assign uimm  = {16'b0, word[15:0]};
  assign shamt = {27'b0, word[15:11]};
  assign boff  = {{14{word[15]}}, word[15:0], 2'b00};
  assign joff  = {{4{word[25]}}, word[25:0], 2'b00};

  always_comb begin
    ctrl = '0;
    ctrl.illegal = !d.valid;
    ctrl.op = d.op;
    ctrl.unit = unit_of(d.op);
    ctrl.rd = word[25:21];
    ctrl.ra = word[20:16];
    ctrl.rb = word[15:11];
    ctrl.rc = word[10:6];
    ctrl.imm = simm;
    unique case (d.op)
      OP_S_ANDI, OP_S_ORI, OP_S_LUI, OP_V_ANDI: ctrl.imm = uimm;
      OP_S_SHLI, OP_S_SHRI, OP_S_SRAI, OP_V_SHLI, OP_V_SHRI, OP_V_SRAI: ctrl.imm = shamt;
      OP_S_BZ, OP_S_BNZ: ctrl.imm = boff;
      OP_S_J: ctrl.imm = joff;
      default: ;
    endcase
    // sb only for vb|sb operands: vector ALU/FPU binary ops, V_CMP, R4 (bit 5), SHFL, ATOM
    unique case (d.op)
      OP_V_FFMA, OP_V_IMAD, OP_V_SEL: ctrl.sb = word[5];
      OP_V_SHLI, OP_V_SHRI, OP_V_SRAI, OP_V_FRCP, OP_V_FRSQ, OP_V_FEX2, OP_V_FLG2,
      OP_V_I2F, OP_V_F2I, OP_V_READLANE, OP_V_GETSR, OP_V_ADDI, OP_V_ANDI,
      OP_V_LW, OP_V_SW, OP_V_LDS, OP_V_STS: ctrl.sb = 1'b0;
      default: ctrl.sb = (word[31:26] inside {6'h02, 6'h03, 6'h04, 6'h06, 6'h16}) && word[10];
    endcase
    ctrl.branch = d.op inside {OP_S_BZ, OP_S_BNZ, OP_S_J};
    ctrl.mem_shared = d.op inside {OP_V_LDS, OP_V_STS, OP_V_ATOM_ADD_S, OP_V_ATOM_MIN_S,
                                   OP_V_ATOM_MAX_S, OP_V_ATOM_XCHG_S, OP_V_ATOM_AND_S,
                                   OP_V_ATOM_OR_S};
    ctrl.vec_dst = (d.op inside {
        OP_V_ADD, OP_V_SUB, OP_V_MUL, OP_V_AND, OP_V_OR, OP_V_XOR, OP_V_ANDN, OP_V_SHL, OP_V_SHR,
        OP_V_SRA, OP_V_SLT, OP_V_SLTU, OP_V_MIN, OP_V_MAX, OP_V_SHLI, OP_V_SHRI, OP_V_SRAI,
        OP_V_FADD, OP_V_FSUB, OP_V_FMUL, OP_V_FMIN, OP_V_FMAX, OP_V_FRCP, OP_V_FRSQ, OP_V_FEX2,
        OP_V_FLG2, OP_V_I2F, OP_V_F2I, OP_V_FFMA, OP_V_IMAD, OP_V_SEL, OP_V_SHFL, OP_V_GETSR,
        OP_V_ADDI, OP_V_ANDI, OP_V_LW, OP_V_LDS, OP_V_ATOM_ADD, OP_V_ATOM_MIN, OP_V_ATOM_MAX,
        OP_V_ATOM_XCHG, OP_V_ATOM_AND, OP_V_ATOM_OR, OP_V_ATOM_ADD_S, OP_V_ATOM_MIN_S,
        OP_V_ATOM_MAX_S, OP_V_ATOM_XCHG_S, OP_V_ATOM_AND_S, OP_V_ATOM_OR_S});
    ctrl.scl_dst = d.valid && !ctrl.vec_dst && !(d.op inside {
        OP_NOP, OP_EXIT, OP_BAR, OP_FENCE, OP_S_SW, OP_V_SW, OP_V_STS, OP_S_BZ, OP_S_BNZ, OP_S_J});
    // GETSR legality ([SR-1], [SR-2]); the SR id is in ra
    if (d.valid && d.op == OP_S_GETSR && !SR_UNIFORM[word[20:16]]) ctrl.illegal = 1'b1;
    if (d.valid && d.op == OP_V_GETSR && !(SR_UNIFORM[word[20:16]] || SR_LANE[word[20:16]])) ctrl.illegal = 1'b1;
  end
endmodule
