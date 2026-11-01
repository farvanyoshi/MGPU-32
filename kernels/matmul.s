; matmul.s - C = A * B for row-major float32 matrices A[M][K], B[K][N], C[M][N]
;
; args:   s1 = &A, s2 = &B, s3 = &C, s4 = M, s5 = N, s6 = K
; launch: BLOCK_DIM = 256 (a 16x16 thread tile), GRID_DIM = ceil(M/16) * ceil(N/16)
;
; Classic shared-memory tiling: every block computes one 16x16 tile of C. For each 16-wide
; slice of K it stages a tile of A and a tile of B in shared memory (BAR), then each thread
; accumulates 16 products with V_FFMA. Edges are handled by loading 0 outside the matrices
; and masking the final store. MGPU-32 has no integer divide, so the tile coordinates
; are derived from BLOCK_ID with a short scalar loop.

TILE = 16
AS   = 0                            ; shared: As[16][16] at 0, Bs[16][16] at 1024
BS   = TILE * TILE * 4
        .kernel matmul
matmul: S_MOV     s20, exec         ; launch EXEC (all lanes: BLOCK_DIM = 256)
        ; --- tile coordinates: by = BLOCK_ID / tilesN, bx = BLOCK_ID % tilesN
        S_ADDI    s7, s5, TILE - 1
        S_SHRI    s7, s7, 4         ; tilesN = ceil(N / 16)
        S_GETSR   s8, BLOCK_ID      ; s8 = bx (remainder), s9 = by
        S_ADDI    s9, s0, 0
divide: S_SLTU    s10, s8, s7
        S_BNZ     s10, divided
        S_SUB     s8, s8, s7
        S_ADDI    s9, s9, 1
        S_J       divide
divided:
        ; --- thread coordinates
        V_GETSR   v1, TID
        V_ANDI    v2, v1, TILE - 1  ; tx
        V_SHRI    v3, v1, 4         ; ty
        S_SHLI    s10, s9, 4
        V_ADD     v4, v3, s10       ; row = by*16 + ty
        S_SHLI    s10, s8, 4
        V_ADD     v5, v2, s10       ; col = bx*16 + tx
        V_SHLI    v6, v1, 2         ; my slot in a tile: tid * 4
        V_SHLI    v7, v3, 6         ; &As[ty][0] = ty * 64
        V_SHLI    v8, v2, 2
        V_ADDI    v8, v8, BS        ; &Bs[0][tx] = 1024 + tx * 4
        V_CMP.LTU s21, v4, s4       ; row < M
        V_CMP.LTU s22, v5, s5       ; col < N
        V_MOV     v14, v0           ; acc = 0.0
        S_ADDI    s11, s0, 0        ; t0 = 0: first K index of this slice

slice:  S_SLTU    s10, s11, s6      ; while t0 < K
        S_BZ      s10, store
        ; As[ty][tx] = (row < M && t0+tx < K) ? A[row][t0+tx] : 0
        V_MOV     v15, v0
        V_ADD     v9, v2, s11       ; a_col = t0 + tx
        V_CMP.LTU s12, v9, s6
        S_AND     exec, s12, s21
        V_MUL     v10, v4, s6       ; row * K
        V_ADD     v10, v10, v9
        V_SHLI    v10, v10, 2
        V_ADD     v10, v10, s1
        V_LW      v15, (v10)
        S_MOV     exec, s20
        V_STS     v15, AS(v6)
        ; Bs[ty][tx] = (t0+ty < K && col < N) ? B[t0+ty][col] : 0
        V_MOV     v15, v0
        V_ADD     v9, v3, s11       ; b_row = t0 + ty
        V_CMP.LTU s12, v9, s6
        S_AND     exec, s12, s22
        V_MUL     v10, v9, s5       ; b_row * N
        V_ADD     v10, v10, v5
        V_SHLI    v10, v10, 2
        V_ADD     v10, v10, s2
        V_LW      v15, (v10)
        S_MOV     exec, s20
        V_STS     v15, BS(v6)
        BAR                         ; both tiles staged
        V_MOV     v12, v7
        V_MOV     v13, v8
        S_ADDI    s13, s0, TILE
inner:  V_LDS     v10, (v12)        ; As[ty][k]
        V_LDS     v11, (v13)        ; Bs[k][tx]
        V_FFMA    v14, v10, v11, v14
        V_ADDI    v12, v12, 4
        V_ADDI    v13, v13, TILE * 4
        S_ADDI    s13, s13, -1
        S_BNZ     s13, inner
        BAR                         ; done with the tiles before they are overwritten
        S_ADDI    s11, s11, TILE
        S_J       slice

store:  S_AND     exec, s21, s22    ; row < M && col < N
        V_MUL     v10, v4, s5
        V_ADD     v10, v10, v5
        V_SHLI    v10, v10, 2
        V_ADD     v10, v10, s3
        V_SW      v14, (v10)
        EXIT
