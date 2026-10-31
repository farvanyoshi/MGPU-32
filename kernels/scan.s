; scan.s - inclusive prefix sum of uint32 values (wrapping), in three launches:
;
;   1. scan_block  in  -> out, sums   each block scans its slice; sums[b] = block total
;   2. scan_block  sums -> sums       one block scans the block totals (GRID_DIM <= BLOCK_DIM)
;   3. scan_add    out, sums          out[i] += sums[BLOCK_ID - 1] for every block after the first
;
; scan_block args: s1 = &in, s2 = &out, s3 = &sums (may be 0 if unused), s4 = n
; scan_add   args: s1 = &out, s2 = &sums (scanned), s3 = n
; launch: the same BLOCK_DIM (a power of two) for every step; GRID_DIM = ceil(n / BLOCK_DIM)
;
; scan_block is a Hillis-Steele scan in shared memory: log2(BLOCK_DIM) rounds, each with a
; read phase and a write phase separated by BAR.

        .kernel scan_block
        .kernel scan_add

scan_block:
        V_GETSR   v1, GTID
        V_GETSR   v2, TID
        S_MOV     s10, exec         ; launch EXEC
        V_MOV     v3, v0            ; value = 0 beyond n
        V_CMP.LTU s11, v1, s4
        S_MOV     exec, s11
        V_SHLI    v4, v1, 2
        V_ADD     v5, v4, s1
        V_LW      v3, (v5)
        S_MOV     exec, s10
        V_SHLI    v6, v2, 2         ; shared slot: tid * 4
        V_STS     v3, (v6)
        BAR
        S_GETSR   s12, BLOCK_DIM
        S_ADDI    s13, s0, 1        ; offset = 1, 2, 4, ...
round:  S_SLTU    s14, s13, s12
        S_BZ      s14, scanned
        V_CMP.GEU s15, v2, s13      ; tid >= offset
        S_MOV     exec, s15
        S_SHLI    s16, s13, 2
        V_SUB     v7, v6, s16
        V_LDS     v8, (v7)          ; sh[tid - offset]
        V_ADD     v3, v3, v8        ; running value (lanes below offset keep theirs)
        S_MOV     exec, s10
        BAR                         ; everyone has read before anyone writes
        V_STS     v3, (v6)
        BAR
        S_SHLI    s13, s13, 1
        S_J       round
scanned:
        S_MOV     exec, s11         ; store in-range results
        V_ADD     v5, v4, s2
        V_SW      v3, (v5)
        S_BZ      s3, done          ; no block-sum array
        S_ADDI    s17, s12, -1
        S_MOV     exec, s10
        V_CMP.EQ  s18, v2, s17      ; the last thread of the block holds the block total
        S_MOV     exec, s18
        S_GETSR   s19, BLOCK_ID
        S_SHLI    s19, s19, 2
        S_ADD     s19, s19, s3
        V_BCAST   v9, s19
        V_SW      v3, (v9)
done:   EXIT

scan_add:
        S_GETSR   s10, BLOCK_ID
        S_BZ      s10, add_done     ; block 0 needs no offset
        S_SHLI    s11, s10, 2
        S_ADD     s11, s11, s2
        S_LW      s12, -4(s11)      ; sums[BLOCK_ID - 1]
        V_GETSR   v1, GTID
        V_CMP.LTU s13, v1, s3
        S_MOV     exec, s13
        V_SHLI    v2, v1, 2
        V_ADD     v2, v2, s1
        V_LW      v3, (v2)
        V_ADD     v3, v3, s12
        V_SW      v3, (v2)
add_done:
        EXIT
