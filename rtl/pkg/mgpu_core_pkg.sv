// Core-internal types (docs/uarch/core.md section 5).
/* verilator lint_off UNUSEDPARAM */
package mgpu_core_pkg;
  import mgpu_isa_pkg::*;

  // Decoded instruction (the instruction buffer holds this).
  typedef struct packed {
    logic        illegal;     // [DEC-1] / [DEC-3]
    op_e         op;
    unit_e       unit;
    logic [4:0]  rd, ra, rb, rc;
    logic        sb;          // rb is a broadcast scalar ([ENC-1]); 0 when rb is not vb|sb
    logic        b_imm;       // operand B is `imm` (immediate or shift-amount forms)
    logic [31:0] imm;         // extended immediate, shift amount, or byte branch offset
    logic        vec_dst;     // writes a vector register
    logic        scl_dst;     // writes a scalar register
    logic        branch;      // S_BZ / S_BNZ / S_J
    logic        mem_shared;  // LSU access targets shared memory
  } ctrl_t;

  // Instruction issued to an execution unit: operands read in full (core.md section 6).
  typedef struct packed {
    logic              valid;
    logic [4:0]        warp;
    logic [31:0]       pc;
    logic [31:0]       word;
    logic [31:0]       clock;           // CLOCK at issue (cycles since block start)
    ctrl_t             ctrl;
    logic [31:0]       exec;            // EXEC before the instruction ([EX-3])
    logic [31:0]       sa, sb, sc, sd;  // S[ra], scalar B (S[rb] or imm), S[rc], S[rd]
    logic [31:0][31:0] va, vb, vc, vd;  // V[ra], B per lane, V[rc], old V[rd]
  } ipkt_t;

  // One retired instruction (the fields of mgpu_record / CommitRecord).
  typedef struct packed {
    logic              valid;
    logic [4:0]        warp;
    logic [31:0]       pc;
    logic [31:0]       word;
    logic [31:0]       clock;
    logic              sw_valid;
    logic [4:0]        sw_reg;
    logic [31:0]       sw_val;
    logic              vw_valid;
    logic [4:0]        vw_reg;
    logic [31:0][31:0] vw_val;          // all 32 lanes after the write
    logic [31:0]       exec_after;
    logic [1:0]        event_;          // 0 none, 1 EXIT, 2 BAR, 3 fault
    logic [7:0]        fault_code;
    logic [4:0]        fault_lane;
    logic [31:0]       fault_info;
    logic [31:0]       mw_lanes;        // lanes that wrote memory (reported in ascending order)
    logic              mw_shared;
    logic [31:0][31:0] mw_addr;
    logic [31:0][31:0] mw_val;
  } retire_t;

  // Fault codes (ISA 12, STATUS / ERR_CODE).
  localparam logic [7:0] E_ILLEGAL = 8'd1, E_MISALIGNED = 8'd2, E_RANGE = 8'd3, E_BARRIER = 8'd4;
  // Record events.
  localparam logic [1:0] EV_NONE = 2'd0, EV_EXIT = 2'd1, EV_BAR = 2'd2, EV_FAULT = 2'd3;

  // Retire channels, in the order the testbench consumes a cycle's records.
  localparam int NRET = 7;
  localparam int R_SALU = 0, R_VALU = 1, R_FPU = 2, R_SFU = 3, R_LSU_A = 4, R_LSU_B = 5, R_FAULT = 6;

  /* verilator lint_off UNUSEDSIGNAL */  // only the identity fields of the packet are used
  function automatic retire_t ret_from(input ipkt_t p);
    retire_t r;
    r = '0;
    r.valid = 1'b1;
    r.warp = p.warp;
    r.pc = p.pc;
    r.word = p.word;
    r.clock = p.clock;
    r.exec_after = p.exec;
    return r;
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  // Record a scalar destination write (none for s0); writing s31 changes exec_after.
  function automatic retire_t ret_scalar(input retire_t r, input logic [4:0] rd,
                                         input logic [31:0] v);
    retire_t o;
    o = r;
    if (rd != 0) begin
      o.sw_valid = 1'b1;
      o.sw_reg = rd;
      o.sw_val = v;
      if (rd == 5'd31) o.exec_after = v;
    end
    return o;
  endfunction

  // Record a vector destination write (none for v0); `merged` holds all 32 lanes.
  function automatic retire_t ret_vector(input retire_t r, input logic [4:0] rd,
                                         input logic [31:0][31:0] merged);
    retire_t o;
    o = r;
    if (rd != 0) begin
      o.vw_valid = 1'b1;
      o.vw_reg = rd;
      o.vw_val = merged;
    end
    return o;
  endfunction

endpackage
/* verilator lint_on UNUSEDPARAM */
