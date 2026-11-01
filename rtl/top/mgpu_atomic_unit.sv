// mgpu_atomic_unit: memory front end (docs/uarch/top.md 5, interfaces.md 2).
//
// Turns segment requests into transactions on the 128-bit memory port and responses back
// into segments:
//   READ    request; the 8 data beats are assembled and the segment is returned in the
//           cycle of the last beat (done = a + T_mem + 8).
//   WRITE   request with its 8 data beats, beat k in cycle a + k (beat 0 with the request);
//           the acknowledgement is passed back.
//   ATOMIC  READ of the segment; as each beat arrives, every lane addressing a word of that
//           beat is applied in ascending lane order to a local copy (colliding lanes see each
//           other's results) and its old value kept. In the cycle of the last beat the old
//           values go back to the core and the WRITE of the touched words is requested; the
//           memory is free by then. No other request is accepted from the READ until that
//           WRITE is, which makes the atomic indivisible ([MEM-5]).
// Only one WRITE's data beats are in flight at a time: memory accepts a request at most
// every 8 cycles (T_seg = 8, one beat per cycle).
module mgpu_atomic_unit
  import mgpu_core_pkg::*;
(
    input  logic               clk,
    input  logic               rst_n,
    // segment side (mgpu_mem_arbiter)
    input  logic               sreq_valid,
    output logic               sreq_ready,
    input  sreq_t              sreq,
    output logic               srd_valid,
    output logic [15:0]        srd_tag,
    output logic [31:0][31:0]  srd_data,
    output logic               sat_valid,
    output logic [15:0]        sat_tag,
    output logic [31:0][31:0]  sat_old,
    output logic               sack_valid,
    output logic [15:0]        sack_tag,
    // memory port
    output logic               mreq_valid,
    input  logic               mreq_ready,
    output logic [1:0]         mreq_op,      // 0 READ, 1 WRITE
    output logic [31:0]        mreq_addr,
    output logic [15:0]        mreq_tag,
    output logic [31:0]        mreq_mask,
    output logic               mwd_valid,
    output logic [127:0]       mwd_data,
    output logic               mwd_last,
    input  logic               mrd_valid,
    input  logic [15:0]        mrd_tag,
    input  logic [127:0]       mrd_data,
    input  logic               mrd_last,
    input  logic               mack_valid,
    input  logic [15:0]        mack_tag
);
  localparam logic [15:0] T_ATOM = 16'h0001;   // internal: the atomic's READ and write-back

  // ---------------------------------------------------------------- atomic in progress
  logic              a_busy, a_wb;          // READ outstanding / write-back being requested
  /* verilator lint_off UNUSEDSIGNAL */   // an atomic needs no WRITE mask or data
  sreq_t             a_req;
  /* verilator lint_on UNUSEDSIGNAL */
  logic [31:0][31:0] a_word, a_old;
  logic [2:0]        rbeat;                 // READ beats received (all reads)
  logic [6:0][127:0] rbuf;

  // ---------------------------------------------------------------- write data beats
  logic              w_active;
  logic [2:0]        w_beat;
  logic [31:0][31:0] w_data;

  // ---------------------------------------------------------------- the atomic's last beat
  // apply the lanes of the arriving beat to the local copy, in ascending lane order
  logic [31:0][31:0] b_word, b_old;
  always_comb begin
    b_word = a_word;
    b_old = a_old;
    if (mrd_valid && mrd_tag == T_ATOM) begin
      for (int i = 0; i < 4; i++) b_word[4 * int'(rbeat) + i] = mrd_data[32 * i +: 32];
      for (int l = 0; l < 32; l++) begin
        if (a_req.lanes[l] && a_req.off[l][4:2] == rbeat) begin
          b_old[l] = b_word[a_req.off[l]];
          b_word[a_req.off[l]] = atom_apply(a_req.aop, b_word[a_req.off[l]], a_req.opnd[l]);
        end
      end
    end
  end
  logic atom_done;
  assign atom_done = mrd_valid && mrd_tag == T_ATOM && mrd_last;

  // ---------------------------------------------------------------- requests
  logic [31:0] a_mask;
  always_comb begin
    a_mask = '0;
    for (int l = 0; l < 32; l++) if (a_req.lanes[l]) a_mask[a_req.off[l]] = 1'b1;
  end
  always_comb begin
    mreq_valid = 1'b0;
    mreq_op = 2'd0;
    mreq_addr = sreq.addr;
    mreq_tag = sreq.tag;
    mreq_mask = sreq.wmask;
    sreq_ready = 1'b0;
    if (a_wb || atom_done) begin                // the atomic's write-back
      mreq_valid = 1'b1;
      mreq_op = 2'd1;
      mreq_addr = a_req.addr;
      mreq_tag = T_ATOM;
      mreq_mask = a_mask;
    end else if (!a_busy && sreq_valid) begin
      mreq_valid = 1'b1;
      mreq_op = sreq.kind == MK_WRITE ? 2'd1 : 2'd0;
      if (sreq.kind == MK_ATOMIC) mreq_tag = T_ATOM;
      sreq_ready = mreq_ready;
    end
  end
  logic wb_accept, wr_accept;
  assign wb_accept = (a_wb || atom_done) && mreq_ready;
  assign wr_accept = sreq_ready && sreq_valid && sreq.kind == MK_WRITE;

  // write data: beat 0 together with the accepted request, then one beat per cycle
  always_comb begin
    mwd_valid = 1'b0;
    mwd_data = '0;
    mwd_last = 1'b0;
    if (w_active) begin
      mwd_valid = 1'b1;
      mwd_data = w_data[4 * int'(w_beat) +: 4];
      mwd_last = w_beat == 3'd7;
    end else if (wr_accept) begin
      mwd_valid = 1'b1;
      mwd_data = sreq.wdata[3:0];
    end else if (wb_accept) begin
      mwd_valid = 1'b1;
      mwd_data = b_word[3:0];
    end
  end

  // ---------------------------------------------------------------- responses
  always_comb begin
    srd_valid = mrd_valid && mrd_last && mrd_tag != T_ATOM;
    srd_tag = mrd_tag;
    srd_data = {mrd_data, rbuf[6], rbuf[5], rbuf[4], rbuf[3], rbuf[2], rbuf[1], rbuf[0]};
    sat_valid = atom_done;
    sat_tag = a_req.tag;
    sat_old = b_old;
    sack_valid = mack_valid && mack_tag != T_ATOM;
    sack_tag = mack_tag;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      a_busy <= 1'b0;
      a_wb <= 1'b0;
      rbeat <= '0;
      w_active <= 1'b0;
    end else begin
      // READ beats (one READ's beats at a time)
      if (mrd_valid) begin
        rbuf[rbeat] <= mrd_data;
        rbeat <= mrd_last ? 3'd0 : rbeat + 3'd1;
      end
      // an atomic starts with its READ, collects its beats, then writes back
      if (sreq_ready && sreq_valid && sreq.kind == MK_ATOMIC) begin
        a_busy <= 1'b1;
        a_req <= sreq;
      end
      if (mrd_valid && mrd_tag == T_ATOM) begin
        a_word <= b_word;
        a_old <= b_old;
      end
      if (atom_done && !mreq_ready) a_wb <= 1'b1;   // memory not ready yet: keep requesting
      if (wb_accept) begin
        a_wb <= 1'b0;
        a_busy <= 1'b0;
      end
      // write data beats 1..7
      if (wr_accept || wb_accept) begin
        w_active <= 1'b1;
        w_beat <= 3'd1;
        w_data <= wr_accept ? sreq.wdata : b_word;
      end else if (w_active) begin
        w_beat <= w_beat + 3'd1;
        if (w_beat == 3'd7) w_active <= 1'b0;
      end
    end
  end
endmodule
