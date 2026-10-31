; conv3x3.s - 3x3 image filter with clamped edges
;   out[y][x] = sum_{dy,dx in -1..1} w[dy+1][dx+1] * in[clamp(y+dy)][clamp(x+dx)]
;
; args:   s1 = &in (float32[H][W]), s2 = &out (float32[H][W]), s3 = &w (float32[3][3]),
;         s4 = W, s5 = H
; launch: one block per row: GRID_DIM = H, BLOCK_DIM = W (so W <= MAX_BLOCK_DIM)
;
; 2-D indexing without a divide: BLOCK_ID is y and TID is x. The nine weights are uniform,
; so they are read once per warp with scalar loads into s10..s18.

        .kernel conv3x3
conv3x3:
        S_LW      s10, 0(s3)        ; weights w00 .. w22
        S_LW      s11, 4(s3)
        S_LW      s12, 8(s3)
        S_LW      s13, 12(s3)
        S_LW      s14, 16(s3)
        S_LW      s15, 20(s3)
        S_LW      s16, 24(s3)
        S_LW      s17, 28(s3)
        S_LW      s18, 32(s3)
        ; rows: ym = max(y-1, 0), y, yp = min(y+1, H-1), as byte offsets of the row start
        S_GETSR   s6, BLOCK_ID      ; y
        S_ADDI    s7, s5, -1        ; H - 1
        S_ADDI    s19, s6, -1
        S_MAX     s19, s19, s0
        S_ADDI    s21, s6, 1
        S_MIN     s21, s21, s7
        S_SHLI    s8, s4, 2         ; row pitch in bytes
        S_MUL     s19, s19, s8
        S_ADD     s19, s19, s1      ; &in[ym][0]
        S_MUL     s20, s6, s8
        S_ADD     s20, s20, s1      ; &in[y][0]
        S_MUL     s21, s21, s8
        S_ADD     s21, s21, s1      ; &in[yp][0]
        ; columns: xm = max(x-1, 0), x, xp = min(x+1, W-1), as byte offsets
        V_GETSR   v1, TID           ; x
        S_ADDI    s9, s4, -1
        V_ADDI    v2, v1, -1
        V_MAX     v2, v2, v0
        V_ADDI    v3, v1, 1
        V_MIN     v3, v3, s9
        V_SHLI    v2, v2, 2
        V_SHLI    v4, v1, 2
        V_SHLI    v3, v3, 2
        V_MOV     v10, v0           ; acc = 0.0
        ; row ym
        V_ADD     v5, v2, s19
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s10, v10
        V_ADD     v5, v4, s19
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s11, v10
        V_ADD     v5, v3, s19
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s12, v10
        ; row y
        V_ADD     v5, v2, s20
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s13, v10
        V_ADD     v5, v4, s20
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s14, v10
        V_ADD     v5, v3, s20
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s15, v10
        ; row yp
        V_ADD     v5, v2, s21
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s16, v10
        V_ADD     v5, v4, s21
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s17, v10
        V_ADD     v5, v3, s21
        V_LW      v6, (v5)
        V_FFMA    v10, v6, s18, v10
        ; out[y][x]
        S_MUL     s22, s6, s8
        S_ADD     s22, s22, s2
        V_ADD     v5, v4, s22
        V_SW      v10, (v5)
        EXIT
