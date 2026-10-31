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
    logic [31:0] imm;         // extended immediate, shift amount, or byte branch offset
    logic        vec_dst;     // writes a vector register
    logic        scl_dst;     // writes a scalar register
    logic        branch;      // S_BZ / S_BNZ / S_J
    logic        mem_shared;  // LSU access targets shared memory
  } ctrl_t;

endpackage
/* verilator lint_on UNUSEDPARAM */
