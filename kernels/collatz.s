; collatz.s - out[i] = number of Collatz steps to reach 1 from in[i]
;
; args:   s1 = &in (uint32[n], values >= 1), s2 = &out (uint32[n]), s3 = n
; launch: any BLOCK_DIM, GRID_DIM = ceil(n / BLOCK_DIM)
;
; A divergent loop (ISA section 9.3): each lane iterates a different number of times, lanes
; that finish drop out of EXEC, and the warp runs as long as its slowest lane.

        .kernel collatz
collatz:
        V_GETSR   v1, GTID
        V_CMP.LTU s10, v1, s3
        S_MOV     exec, s10
        S_BZ      exec, done
        V_SHLI    v2, v1, 2
        V_ADD     v3, v2, s1
        V_LW      v4, (v3)          ; x = in[i]
        V_MOV     v5, v0            ; steps = 0
        V_ADDI    v10, v0, 1        ; vector of 1s
        S_ADDI    s13, s0, 1
        S_ADDI    s15, s0, 3
        S_MOV     s12, exec         ; save EXEC at loop entry
loop:   V_CMP.GTU s11, v4, s13      ; lanes still above 1 (ANDed with EXEC)
        S_MOV     exec, s11
        S_BZ      exec, finish      ; every lane reached 1
        V_ANDI    v7, v4, 1
        V_CMP.NE  s14, v7, v0       ; odd lanes
        V_SHRI    v8, v4, 1         ; even: x / 2
        V_IMAD    v9, v4, s15, v10  ; odd:  3x + 1
        V_SEL     v4, v9, v8, s14   ; pick per lane, no branch
        V_ADDI    v5, v5, 1
        S_J       loop
finish: S_MOV     exec, s12
        V_ADD     v3, v2, s2
        V_SW      v5, (v3)          ; out[i] = steps
done:   EXIT
