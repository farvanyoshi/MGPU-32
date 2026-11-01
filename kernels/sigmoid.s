; sigmoid.s - y[i] = 1 / (1 + e^(-x[i]))
;
; args:   s1 = n, s2 = &x (float32[n]), s3 = &y (float32[n])
; launch: any BLOCK_DIM, GRID_DIM = ceil(n / BLOCK_DIM)
;
; Uses the special function unit: e^-x = 2^(-x * log2(e)) with V_FEX2, and the division
; with the V_FDIV pseudo-instruction (V_FRCP + V_FMUL). V_FNEG flips the sign bit.

LOG2E = f32(1.4426950408889634)
ONE   = f32(1.0)
        .kernel sigmoid
sigmoid:
        V_GETSR   v1, GTID
        V_CMP.LTU s10, v1, s1
        S_MOV     exec, s10
        S_BZ      exec, done
        V_SHLI    v2, v1, 2
        V_ADD     v3, v2, s2
        V_LW      v4, (v3)          ; x
        V_FNEG    v5, v4, s11       ; -x            (s11 is scratch)
        S_LI      s12, LOG2E
        V_FMUL    v5, v5, s12       ; -x * log2(e)
        V_FEX2    v5, v5            ; e^-x
        S_LI      s13, ONE
        V_FADD    v5, v5, s13       ; 1 + e^-x
        V_BCAST   v6, s13           ; 1.0 in every lane
        V_FDIV    v7, v6, v5, v8    ; 1 / (1 + e^-x)  (v8 is scratch)
        V_ADD     v3, v2, s3
        V_SW      v7, (v3)
done:   EXIT
