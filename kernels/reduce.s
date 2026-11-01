; reduce.s - per-block float sum: out[BLOCK_ID] = sum of this block's slice of in
;                                                        (ISA section 15.2)
; args:   s1 = &in (float32[n]), s2 = &out (float32[GRID_DIM]), s3 = n
; launch: BLOCK_DIM a power of two (<= 4096 so the tree fits in shared memory),
;         GRID_DIM = ceil(n / BLOCK_DIM)
;
; Tree reduction in shared memory. The loop counter is scalar, so every warp runs the
; same number of iterations and reaches every BAR at the same PC.

        .kernel reduce
reduce: V_GETSR   v1, GTID
        V_GETSR   v2, TID
        S_MOV     s11, exec         ; save launch EXEC
        V_MOV     v4, v0            ; default 0.0 for out-of-range lanes
        V_CMP.LTU s10, v1, s3       ; in-range lanes
        S_MOV     exec, s10
        V_SHLI    v3, v1, 2
        V_ADD     v3, v3, s1        ; &in[i]
        V_LW      v4, (v3)          ; only in-range lanes load
        S_MOV     exec, s11         ; all threads again
        V_SHLI    v5, v2, 2         ; shared address = tid * 4
        V_STS     v4, (v5)          ; sh[tid] = value
        BAR                         ; whole array in shared memory

        S_GETSR   s12, BLOCK_DIM
        S_SHRI    s12, s12, 1       ; stride = BLOCK_DIM / 2   (uniform)
loop:   S_BZ      s12, last
        V_CMP.LTU s10, v2, s12      ; tid < stride
        S_MOV     exec, s10
        S_SHLI    s13, s12, 2       ; stride in bytes
        V_ADD     v6, v5, s13       ; &sh[tid + stride]
        V_LDS     v7, (v6)
        V_LDS     v8, (v5)
        V_FADD    v8, v8, v7
        V_STS     v8, (v5)          ; sh[tid] += sh[tid + stride]
        S_MOV     exec, s11
        BAR                         ; every warp reaches this: the loop count is uniform
        S_SHRI    s12, s12, 1
        S_J       loop

last:   V_CMP.EQ  s10, v2, v0       ; tid == 0
        S_MOV     exec, s10
        S_BZ      exec, done        ; only warp 0 continues
        V_LDS     v8, (v0)          ; sh[0] = block total
        S_GETSR   s14, BLOCK_ID
        S_SHLI    s14, s14, 2
        S_ADD     s14, s14, s2      ; &out[BLOCK_ID]
        V_BCAST   v9, s14
        V_SW      v8, (v9)
done:   EXIT
