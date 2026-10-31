; warp_reduce.s - integer sum of in[0..n) into *out, using warp shuffles
;
; args:   s1 = &in (int32[n]), s2 = &out (one int32, initialized by the host), s3 = n
; launch: any BLOCK_DIM and GRID_DIM (threads beyond n contribute 0)
;
; Each warp reduces its 32 values with a 5-step XOR butterfly of V_SHFL (no shared memory,
; no barriers), then lane 0 adds the warp total to *out with one global atomic.

        .kernel warp_reduce
warp_reduce:
        V_GETSR   v1, GTID
        V_GETSR   v9, LANE_ID
        S_MOV     s10, exec         ; launch EXEC
        V_CMP.LTU s11, v1, s3       ; in-range lanes load, the rest keep v2 = 0
        S_MOV     exec, s11
        V_SHLI    v3, v1, 2
        V_ADD     v3, v3, s1
        V_LW      v2, (v3)
        ; The butterfly runs on all 32 lanes: a lane that is not a thread still holds 0 and
        ; must take part, because it is an intermediate node of other lanes' sums.
        S_ADDI    exec, s0, -1
        S_ADDI    s12, s0, 16       ; k = 16, 8, 4, 2, 1
tree:   V_XOR     v4, v9, s12       ; partner = lane ^ k
        V_SHFL    v5, v2, v4
        V_ADD     v2, v2, v5
        S_SHRI    s12, s12, 1
        S_BNZ     s12, tree
        V_CMP.EQ  s13, v9, v0       ; lane 0 holds the warp total
        S_AND     exec, s13, s10
        V_BCAST   v6, s2
        V_ATOM.ADD v0, v6, v2       ; result discarded (vd = v0)
        EXIT
