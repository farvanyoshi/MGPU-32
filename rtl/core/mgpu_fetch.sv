// mgpu_fetch: fetch arbiter, I-cache and per-warp instruction buffers (docs/uarch/core.md 4, 11.3).
//
// Cycle f: one warp, chosen round robin among those requesting with an empty buffer, looks
// up its PC.
//   - PC > mem_last: a fetch-fault marker, no cache access.
//   - Hit on a valid line: the word goes to the warp's decode stage.
//   - The line is being filled: the warp waits for that fill. If the fill's response
//     arrives in this very cycle, the word is taken from it directly.
//   - Miss: the line becomes "pending" for a new fill, tagged with the requesting warp's
//     slot (unique, since that warp cannot fetch again before its word arrives), and a
//     READ goes out at f + 1.
// Cycle f + 1: decode. Cycle f + 2: the instruction buffer is valid.
//
// A fill response (cycle `done`) delivers the word to every warp waiting for that fill
// (buffer valid at done + 2), and installs the line only if the entry is still pending for
// that same fill (a later miss may have replaced it). Fills may overlap; the memory model
// serializes them.
module mgpu_fetch
  import mgpu_core_pkg::*;
#(
    parameter int W     = 8,
    parameter int LINES = 16,
    localparam int IB = $clog2(LINES)
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                inv,          // launch: invalidate the cache, clear all state
    input  logic [31:0]         mem_last,     // MEM_SIZE - 4
    // requests from the warp table
    input  logic [W-1:0]        req,
    input  logic [W-1:0][31:0]  req_pc,
    output logic [W-1:0]        grant,        // this cycle's lookup (one-hot)
    // instruction buffers
    input  logic [W-1:0]        consume,      // issued: empty the buffer
    output logic [W-1:0]        ibuf_valid,
    output logic [W-1:0][31:0]  ibuf_word,
    output ctrl_t [W-1:0]       ibuf_ctrl,
    output logic [W-1:0]        ibuf_ff,      // fetch fault ([MEM-7])
    // I-fetch memory port (segment wide)
    output logic                mreq_valid,
    output logic [31:0]         mreq_addr,
    output logic [4:0]          mreq_id,
    input  logic                mresp_valid,
    input  logic [4:0]          mresp_id,
    input  logic [31:0][31:0]   mresp_data,
    // events for the performance counters
    output logic                ev_hit,
    output logic                ev_miss
);
  // ---- cache arrays
  logic [LINES-1:0]            valid, pend;
  logic [LINES-1:0][24:0]      tag;
  logic [LINES-1:0][4:0]       pend_id;
  logic [LINES-1:0][31:0][31:0] data;
  logic [W-1:0][24:0]          fill_line;   // line of the fill started by each warp

  // ---- per-warp state
  logic [W-1:0]       wait_v, d_valid, d_ff;
  logic [W-1:0][4:0]  wait_id, wait_off;
  logic [W-1:0][31:0] d_word;
  logic [W-1:0]       busy;
  assign busy = ibuf_valid | d_valid | wait_v;

  // ---- round-robin choice
  logic [W-1:0] elig;
  logic         g_any;
  int           g_idx;
  logic [$clog2(W > 1 ? W : 2)-1:0] rr;
  assign elig = req & ~busy;
  always_comb begin
    g_any = 1'b0;
    g_idx = 0;
    for (int k = 0; k < W; k++) begin
      int i;
      i = (int'(rr) + k) % W;
      if (!g_any && elig[i]) begin
        g_any = 1'b1;
        g_idx = i;
      end
    end
    grant = '0;
    if (g_any) grant[g_idx] = 1'b1;
  end

  // ---- lookup
  logic [31:0]   l_pc;
  logic [24:0]   l_line;
  logic [IB-1:0] l_idx;
  logic [4:0]    l_off;
  logic          l_ff, l_tag_eq, l_hit, l_bypass, l_wait, l_miss;
  assign l_pc     = req_pc[g_idx];
  assign l_line   = l_pc[31:7];
  assign l_idx    = l_pc[7+IB-1:7];
  assign l_off    = l_pc[6:2];
  assign l_ff     = g_any && l_pc > mem_last;
  assign l_tag_eq = (valid[l_idx] || pend[l_idx]) && tag[l_idx] == l_line;
  assign l_hit    = g_any && !l_ff && l_tag_eq && valid[l_idx];
  assign l_bypass = g_any && !l_ff && l_tag_eq && pend[l_idx] && mresp_valid
                    && mresp_id == pend_id[l_idx];
  assign l_wait   = g_any && !l_ff && l_tag_eq && pend[l_idx] && !l_bypass;
  assign l_miss   = g_any && !l_ff && !l_tag_eq;
  assign ev_hit   = l_hit || l_bypass || l_wait;
  assign ev_miss  = l_miss;

  // ---- response install
  logic [24:0]   r_line;
  logic [IB-1:0] r_idx;
  logic          r_install;
  assign r_line    = fill_line[mresp_id];
  assign r_idx     = r_line[IB-1:0];
  assign r_install = mresp_valid && pend[r_idx] && pend_id[r_idx] == mresp_id
                     && tag[r_idx] == r_line;

  always_ff @(posedge clk) begin
    if (!rst_n || inv) begin
      valid <= '0;
      pend <= '0;
      wait_v <= '0;
      d_valid <= '0;
      ibuf_valid <= '0;
      mreq_valid <= 1'b0;
      rr <= '0;
    end else begin
      // install a returning line (a miss in this cycle to the same entry overrides it)
      if (r_install) begin
        valid[r_idx] <= 1'b1;
        pend[r_idx] <= 1'b0;
        data[r_idx] <= mresp_data;
      end
      // deliver the response to every waiting warp
      for (int w = 0; w < W; w++) begin
        if (wait_v[w] && mresp_valid && wait_id[w] == mresp_id) begin
          wait_v[w] <= 1'b0;
          d_valid[w] <= 1'b1;
          d_ff[w] <= 1'b0;
          d_word[w] <= mresp_data[wait_off[w]];
        end
      end
      // lookup
      mreq_valid <= l_miss;
      if (g_any) rr <= ($bits(rr))'((g_idx + 1) % W);
      if (l_ff || l_hit || l_bypass) begin
        d_valid[g_idx] <= 1'b1;
        d_ff[g_idx] <= l_ff;
        d_word[g_idx] <= l_ff ? 32'd0 : l_hit ? data[l_idx][l_off] : mresp_data[l_off];
      end
      if (l_wait) begin
        wait_v[g_idx] <= 1'b1;
        wait_id[g_idx] <= pend_id[l_idx];
        wait_off[g_idx] <= l_off;
      end
      if (l_miss) begin
        valid[l_idx] <= 1'b0;
        pend[l_idx] <= 1'b1;
        tag[l_idx] <= l_line;
        pend_id[l_idx] <= 5'(g_idx);
        fill_line[g_idx] <= l_line;
        wait_v[g_idx] <= 1'b1;
        wait_id[g_idx] <= 5'(g_idx);
        wait_off[g_idx] <= l_off;
        mreq_addr <= {l_line, 7'b0};
        mreq_id <= 5'(g_idx);
      end
      // decode stage -> instruction buffer; issue empties it
      for (int w = 0; w < W; w++) begin
        if (d_valid[w]) begin
          d_valid[w] <= 1'b0;
          ibuf_valid[w] <= 1'b1;
          ibuf_word[w] <= d_word[w];
          ibuf_ff[w] <= d_ff[w];
          ibuf_ctrl[w] <= d_ctrl[w];
        end else if (consume[w]) begin
          ibuf_valid[w] <= 1'b0;
        end
      end
    end
  end

  ctrl_t [W-1:0] d_ctrl;
  for (genvar w = 0; w < W; w++) begin : g_dec
    mgpu_decode u_dec (.word(d_word[w]), .ctrl(d_ctrl[w]));
  end
endmodule
