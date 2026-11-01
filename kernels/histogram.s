; histogram.s - hist[v & (BINS-1)] += 1 for every v in in[0..n)
;
; args:   s1 = &in (uint32[n]), s2 = &hist (uint32[BINS], zeroed by the host), s3 = n
; launch: any BLOCK_DIM and GRID_DIM (grid-stride loop over the input)
;
; Each block counts into a private histogram in shared memory with V_ATOM.ADD.S, then
; merges it into the global histogram with one V_ATOM.ADD per bin.

BINS = 64
        .kernel histogram
histogram:
        V_GETSR   v1, TID
        S_GETSR   s10, BLOCK_DIM
        S_SHLI    s11, s10, 2       ; block stride in bytes
        S_LI      s14, BINS * 4
        S_ADDI    s16, s0, 1
        S_MOV     s20, exec         ; launch EXEC
        V_SHLI    v2, v1, 2         ; this thread's first bin (bytes)

        V_MOV     v3, v2            ; clear: for (b = tid; b < BINS; b += BLOCK_DIM) sh[b] = 0
clear:  V_CMP.LTU s12, v3, s14
        S_MOV     exec, s12
        S_BZ      exec, cleared
        V_STS     v0, (v3)
        V_ADD     v3, v3, s11
        S_J       clear
cleared:
        S_MOV     exec, s20
        BAR

        V_GETSR   v4, GTID          ; count: grid-stride loop over the input
        S_GETSR   s15, GRID_DIM
        S_MUL     s15, s15, s10     ; total threads
count:  V_CMP.LTU s12, v4, s3
        S_MOV     exec, s12
        S_BZ      exec, counted
        V_SHLI    v5, v4, 2
        V_ADD     v5, v5, s1
        V_LW      v6, (v5)
        V_ANDI    v6, v6, BINS - 1
        V_SHLI    v6, v6, 2
        V_ATOM.ADD.S v0, v6, s16    ; sh[bin] += 1 (colliding lanes all count)
        V_ADD     v4, v4, s15
        S_J       count
counted:
        S_MOV     exec, s20
        BAR

        V_MOV     v3, v2            ; merge: hist[b] += sh[b]
merge:  V_CMP.LTU s12, v3, s14
        S_MOV     exec, s12
        S_BZ      exec, merged
        V_LDS     v7, (v3)
        V_ADD     v8, v3, s2
        V_ATOM.ADD v0, v8, v7
        V_ADD     v3, v3, s11
        S_J       merge
merged: EXIT
