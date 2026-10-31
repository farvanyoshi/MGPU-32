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

  // Retire channels, in the order the testbench consumes a cycle's records. The LSU has
  // four, so that no channel ever carries two records in a cycle: stores (LSU_S, at their
  // last request), shared accesses and empty global loads/atomics (LSU_A, after the unit's
  // occupancy), global loads (LSU_B) and global atomics (LSU_C) at their last response + 1.
  localparam int NRET = 9;
  localparam int R_SALU = 0, R_VALU = 1, R_FPU = 2, R_SFU = 3, R_LSU_S = 4, R_LSU_A = 5,
                 R_LSU_B = 6, R_LSU_C = 7, R_FAULT = 8;

  // LSU access analysis, computed at issue (mgpu_agen).
  typedef struct packed {
    logic [31:0]       lanes;     // lanes that access memory
    logic [31:0][31:0] addr;      // byte address of each lane
    logic [31:0][31:0] data;      // store data or atomic operand of each lane
    logic [5:0]        n_seg;     // global: distinct 128-byte segments
    logic [31:0][4:0]  seg;       // global: segment index of each lane (order of lowest lane)
    logic [5:0]        d;         // shared: bank-conflict degree (at least 1)
    logic [31:0][4:0]  round;     // shared: bank round of each lane
    logic [5:0]        n_act;     // number of accessing lanes
  } lsu_info_t;

  // A segment request on the LSU memory port (interfaces.md 2).
  localparam logic [1:0] MK_READ = 2'd0, MK_WRITE = 2'd1, MK_ATOMIC = 2'd2;
  typedef struct packed {
    logic              valid;
    logic [1:0]        kind;
    logic [4:0]        warp;      // tag: warp and segment index
    logic [4:0]        seg;
    logic [31:0]       addr;      // segment base address
    logic [31:0]       wmask;     // WRITE: words written
    logic [31:0][31:0] wdata;     // WRITE: segment data (highest lane wins)
    logic [31:0]       lanes;     // ATOMIC: lanes of this segment, applied in ascending order
    logic [31:0][4:0]  off;       // ATOMIC: word offset of each lane
    logic [31:0][31:0] opnd;      // ATOMIC: operand of each lane
    logic [2:0]        aop;       // ATOMIC: operation (atom_kind)
  } mreq_t;

  // A segment request on its way to memory (mgpu_mem_arbiter -> mgpu_atomic_unit). The tag
  // routes the response: {core[3:0], port (1 = I-fetch), warp[4:0], seg[4:0], internal}.
  typedef struct packed {
    logic [1:0]        kind;      // MK_READ, MK_WRITE, MK_ATOMIC
    logic [15:0]       tag;
    logic [31:0]       addr;
    logic [31:0]       wmask;
    logic [31:0][31:0] wdata;
    logic [31:0]       lanes;
    logic [31:0][4:0]  off;
    logic [31:0][31:0] opnd;
    logic [2:0]        aop;
  } sreq_t;

  function automatic logic [15:0] mem_tag(input logic [3:0] core, input logic ifetch,
                                          input logic [4:0] warp, input logic [4:0] seg);
    return {core, ifetch, warp, seg, 1'b0};
  endfunction

  function automatic logic is_load(input op_e op);
    return op inside {OP_S_LW, OP_V_LW, OP_V_LDS};
  endfunction
  function automatic logic is_store(input op_e op);
    return op inside {OP_S_SW, OP_V_SW, OP_V_STS};
  endfunction
  function automatic logic is_scalar_mem(input op_e op);
    return op inside {OP_S_LW, OP_S_SW};
  endfunction
  // V_ATOM operations (ISA 6.9): 0 add, 1 min, 2 max, 3 xchg, 4 and, 5 or.
  function automatic logic [2:0] atom_kind(input op_e op);
    unique case (op)
      OP_V_ATOM_MIN, OP_V_ATOM_MIN_S:   return 3'd1;
      OP_V_ATOM_MAX, OP_V_ATOM_MAX_S:   return 3'd2;
      OP_V_ATOM_XCHG, OP_V_ATOM_XCHG_S: return 3'd3;
      OP_V_ATOM_AND, OP_V_ATOM_AND_S:   return 3'd4;
      OP_V_ATOM_OR, OP_V_ATOM_OR_S:     return 3'd5;
      default:                          return 3'd0;
    endcase
  endfunction
  function automatic logic [31:0] atom_apply(input logic [2:0] k, input logic [31:0] old,
                                             input logic [31:0] x);
    unique case (k)
      3'd1:    return $signed(old) <= $signed(x) ? old : x;
      3'd2:    return $signed(old) >= $signed(x) ? old : x;
      3'd3:    return x;
      3'd4:    return old & x;
      3'd5:    return old | x;
      default: return old + x;
    endcase
  endfunction

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
