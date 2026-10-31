// mgpu_dispatcher: launch validation, block dispatch and fault handling (docs/uarch/top.md 3, 6).
//
// Launch cycles are numbered from the `launch` pulse (cycle 0), as in core.md 11.1:
//   cycle 0  validation ([HOST-4]): BLOCK_DIM = 0 or > 32 W -> code 5; ENTRY & 3 -> code 2;
//            GRID_DIM = 0 -> done. Any of these ends the launch here with CYCLES = 1.
//   cycle 1+ every idle core (also one in its block_done cycle) receives the next block, in
//            ascending core order, all in the same cycle. When every block is done, the
//            launch ends with CYCLES = the cycle in which the last block completed.
//
// Faults: a core reports a faulting instruction in the cycle it issues (fault_issue). In
// that cycle every higher-numbered core is held (halt), and from the next cycle every core
// is. Since all faults are detected at issue, exactly one instruction faults. Once every
// core has drained (fault_go) the faulting core emits its fault record; once all memory
// traffic has settled, the launch ends with CYCLES = issue cycle + 1 (core.md 11.6) and
// every core is killed.
module mgpu_dispatcher #(
    parameter int C = 4,
    parameter int W = 8
) (
    input  logic                  clk,
    input  logic                  rst_n,
    // from the host registers
    input  logic                  launch,
    input  logic [31:0]           entry,
    input  logic [31:0]           grid_dim,
    input  logic [31:0]           block_dim,
    output logic                  busy,
    output logic                  end_valid,
    output logic [5:0][31:0]      end_status,
    output logic [31:0]           end_cycles,
    output logic [31:0]           live_cycles,
    // to and from the cores
    output logic [C-1:0]          start,
    output logic [C-1:0]          new_launch,
    output logic [C-1:0][31:0]    block_id,
    output logic [5:0]            num_warps,
    output logic                  kill,
    output logic [C-1:0]          halt,
    output logic                  fault_go,
    input  logic [C-1:0]          idle,
    input  logic [C-1:0]          block_done,
    input  logic [C-1:0]          fault_issue,
    input  logic [C-1:0]          busy_ops,
    input  logic [C-1:0]          mem_quiet,
    input  logic [C-1:0]          fault_valid,
    input  logic [C-1:0][7:0]     fault_code,
    input  logic [C-1:0][31:0]    fault_pc,
    input  logic [C-1:0][4:0]     fault_warp,
    input  logic [C-1:0][4:0]     fault_lane,
    input  logic [C-1:0][31:0]    fault_info
);
  typedef enum logic [1:0] { S_IDLE, S_RUN, S_QUIET } state_e;
  state_e              state;
  logic [31:0]         cyc, next_block, blocks_done, t_fault;
  logic                fault_seen;
  logic [C-1:0]        fresh;
  logic [C-1:0][31:0]  cur_block;
  logic [5:0][31:0]    fstatus;

  assign busy = state != S_IDLE;
  assign live_cycles = cyc;
  assign num_warps = 6'((block_dim + 32'd31) >> 5);
  assign fault_go = !(|busy_ops);

  // ---- validation (cycle 0)
  logic v_dim, v_entry, v_grid;
  assign v_dim = block_dim == 32'd0 || block_dim > 32'(32 * W);
  assign v_entry = entry[1:0] != 2'b0;
  assign v_grid = grid_dim == 32'd0;

  // ---- dispatch and completion (combinational for this cycle)
  logic [31:0] n_start, n_done;
  logic        all_done;
  always_comb begin
    start = '0;
    new_launch = '0;
    block_id = '0;
    n_start = '0;
    n_done = 32'($countones(block_done));
    if (state == S_RUN && !fault_seen) begin
      for (int c = 0; c < C; c++) begin
        if ((idle[c] || block_done[c]) && next_block + n_start < grid_dim) begin
          start[c] = 1'b1;
          new_launch[c] = fresh[c];
          block_id[c] = next_block + n_start;
          n_start = n_start + 32'd1;
        end
      end
    end
    all_done = state == S_RUN && !fault_seen && blocks_done + n_done == grid_dim;
  end

  // same-cycle priority of a fault at issue, then a hold on every core (one assign per core,
  // so that halt[c] visibly depends on the lower cores only)
  for (genvar c = 0; c < C; c++) begin : g_halt
    if (c == 0) begin : g_first
      assign halt[c] = fault_seen;
    end else begin : g_rest
      assign halt[c] = fault_seen || |fault_issue[c-1:0];
    end
  end

  // ---- end of launch
  always_comb begin
    end_valid = 1'b0;
    end_status = '0;
    end_cycles = '0;
    kill = 1'b0;
    if (state == S_IDLE && launch && (v_dim || v_entry || v_grid)) begin
      end_valid = 1'b1;
      end_cycles = 32'd1;
      if (v_dim) end_status[0] = 32'd5;
      else if (v_entry) begin
        end_status[0] = 32'd2;
        end_status[1] = entry;
        end_status[5] = entry;
      end
    end else if (all_done) begin
      end_valid = 1'b1;
      end_cycles = cyc - 32'd1;
    end else if (state == S_QUIET && &mem_quiet) begin
      end_valid = 1'b1;
      end_status = fstatus;
      end_cycles = t_fault + 32'd1;
      kill = 1'b1;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_IDLE;
      cyc <= '0;
      fault_seen <= 1'b0;
    end else begin
      cyc <= state == S_IDLE ? 32'd0 : cyc + 32'd1;
      unique case (state)
        S_IDLE: if (launch) begin
          cyc <= 32'd1;
          if (!(v_dim || v_entry || v_grid)) begin
            state <= S_RUN;
            next_block <= '0;
            blocks_done <= '0;
            fresh <= '1;
            fault_seen <= 1'b0;
          end
        end
        S_RUN: begin
          next_block <= next_block + n_start;
          blocks_done <= blocks_done + n_done;
          for (int c = 0; c < C; c++) begin
            if (start[c]) begin
              fresh[c] <= 1'b0;
              cur_block[c] <= block_id[c];
            end
          end
          if (|fault_issue && !fault_seen) begin
            fault_seen <= 1'b1;
            t_fault <= cyc;
          end
          for (int c = C - 1; c >= 0; c--) begin
            if (fault_valid[c]) begin
              fstatus <= '{fault_info[c], 32'(fault_lane[c]), 32'(fault_warp[c]), cur_block[c],
                           fault_pc[c], 32'(fault_code[c])};
              state <= S_QUIET;
            end
          end
          if (all_done) state <= S_IDLE;
        end
        S_QUIET: if (&mem_quiet) begin
          state <= S_IDLE;
          fault_seen <= 1'b0;
        end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
