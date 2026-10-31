; saxpy.s - y[i] = a * x[i] + y[i]                      (ISA section 15.1)
;
; args:   s1 = n, s2 = a (float bits), s3 = &x (float32[n]), s4 = &y (float32[n])
; launch: any BLOCK_DIM, GRID_DIM = ceil(n / BLOCK_DIM)
;
; Out-of-range lanes are masked off with EXEC instead of branching per thread.

        .kernel saxpy
saxpy:  V_GETSR   v1, GTID          ; i = global thread index
        V_CMP.LTU s10, v1, s1       ; lanes with i < n
        S_MOV     exec, s10         ; turn off out-of-range lanes
        S_BZ      exec, done        ; whole warp out of range
        V_SHLI    v2, v1, 2         ; byte offset = i * 4
        V_ADD     v3, v2, s3        ; &x[i]   (s3 broadcast)
        V_ADD     v4, v2, s4        ; &y[i]
        V_LW      v5, (v3)          ; x[i]    (gather)
        V_LW      v6, (v4)          ; y[i]
        V_FFMA    v6, v5, s2, v6    ; a * x[i] + y[i], one rounding
        V_SW      v6, (v4)          ; y[i] = result
done:   EXIT
